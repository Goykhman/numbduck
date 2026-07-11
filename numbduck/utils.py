import ctypes
import duckdb
import os
import platform
import re
import sys
import tempfile
from inspect import getfile
from numba.experimental.structref import register
from numba.core.types import StructRef
from numbox.core.bindings.utils import load_lib_path


@register
class DuckdbResultTypeClass(StructRef):
    pass


_LIBDUCKDB_CACHE_BASE = os.path.join(
    os.path.expanduser("~"), ".numbduck", "lib"
)
_LIBDUCKDB_CACHE_DIR = os.path.join(
    _LIBDUCKDB_CACHE_BASE, duckdb.__version__
)

# Basename of the standalone libduckdb, shared by the locator (the path it
# looks for), the downloader (the path it writes to), and the ZIP member it
# extracts, so the discover/write contract can never silently drift apart.
_LIBDUCKDB_DYLIB_NAME = "libduckdb.dylib"

# Filename the standalone libduckdb is cached under.
_LIBDUCKDB_CACHED_DYLIB = os.path.join(_LIBDUCKDB_CACHE_DIR, _LIBDUCKDB_DYLIB_NAME)

# Bounded so a hung or trickling endpoint fails fast instead of wedging import.
_LIBDUCKDB_DOWNLOAD_TIMEOUT = 60

_MACOS_LIBDUCKDB_SEARCH_PATHS = [
    "/opt/homebrew/lib/libduckdb.dylib",
    "/usr/local/lib/libduckdb.dylib",
]


# The libduckdb whose C API backs numbduck's JIT bindings, retained so the
# coordination check can probe its reported version. On the wheel path this is
# the same shared object the Python ``duckdb`` module uses; on the standalone
# fallback it is a second, independently loaded libduckdb (see load_duckdb).
_loaded_libduckdb = None


def _has_capi_symbols(lib):
    return hasattr(lib, "duckdb_open")


def _normalize_version(version):
    # duckdb_library_version() and PRAGMA version report e.g. "v1.5.4" while the
    # bare package version is "1.5.4"; drop the single leading "v" so library
    # versions from either source compare directly (removeprefix, not lstrip, so
    # it strips one prefix rather than a run of leading "v"s).
    return version.strip().removeprefix("v")


def _library_version(lib):
    """Version string reported by *lib*'s ``duckdb_library_version()``.

    Returns the normalized version (leading ``v`` stripped) or ``None`` when the
    symbol is absent or reports nothing.
    """
    if not hasattr(lib, "duckdb_library_version"):
        return None
    fn = lib.duckdb_library_version
    fn.restype = ctypes.c_char_p
    raw = fn()
    if raw is None:
        return None
    return _normalize_version(raw.decode())


_wheel_library_version_cache = None


def _wheel_library_version():
    """Library version the Python ``duckdb`` wheel reports for its own core.

    Read via ``PRAGMA version`` through the Python API — which works even on the
    macOS wheels that strip the C-API symbols — so it is in the same
    ``vX.Y.Z[-devN]`` scheme as a standalone's ``duckdb_library_version()``. That
    is the right identity to coordinate on: ``duckdb.__version__`` is the Python
    *package* version, which coincides with the library version only for releases
    and diverges for pre-release (``.dev``/``rc``) builds, so comparing against it
    refuses a genuinely-matched dev build. Cached because the running wheel's core
    version cannot change in-process. Returns the normalized version, or ``None``
    when it cannot be read.
    """
    global _wheel_library_version_cache
    if _wheel_library_version_cache is None:
        try:
            row = duckdb.connect(":memory:").execute("PRAGMA version").fetchone()
        except Exception:
            # Fail closed: an unreadable wheel version makes the caller refuse
            # rather than proceed with an unverified pairing.
            return None
        if not row or row[0] is None:
            return None
        _wheel_library_version_cache = _normalize_version(row[0])
    return _wheel_library_version_cache


def _require_coordinated_standalone(lib, source):
    """Refuse a standalone libduckdb whose build disagrees with the wheel.

    pybridge extracts a raw ``Connection*`` minted by the Python ``duckdb``
    wheel's runtime and hands it to this standalone library's C API, whose
    ``duckdb_query`` casts it back to the standalone's own internal Connection
    layout and dereferences it. That is only sound when both are the same DuckDB
    build: a different build has a candidate-different struct layout, so the
    dereference is undefined behavior. Coordinate on the library version, which
    both sides report in the same ``vX.Y.Z[-devN]`` scheme — the wheel's via
    :func:`_wheel_library_version`, the standalone's via :func:`_library_version`.
    Detect a mismatch and refuse rather than corrupt memory at query time.
    """
    lib_version = _library_version(lib)
    wheel_version = _wheel_library_version()
    if wheel_version is None:
        raise RuntimeError(
            f"numbduck could not read the duckdb wheel's own library version "
            f"(via PRAGMA version) to coordinate it with the standalone libduckdb "
            f"loaded from {source!r}; refusing to hand a wheel-minted connection "
            f"to an unverifiable pairing."
        )
    if lib_version is None:
        raise RuntimeError(
            f"numbduck loaded a standalone libduckdb from {source!r} but could "
            f"not read its library version (duckdb_library_version is absent or "
            f"returned nothing). A verifiable version is required to confirm it "
            f"matches the duckdb wheel (library version {wheel_version!r}) before "
            f"a wheel-minted connection is handed to it; refusing to load an "
            f"unverifiable libduckdb build. Set NUMBDUCK_LIBDUCKDB to a libduckdb "
            f"whose version equals {wheel_version!r}."
        )
    if lib_version != wheel_version:
        raise RuntimeError(
            f"numbduck loaded a standalone libduckdb (version {lib_version!r}) "
            f"from {source!r}, but the duckdb wheel's library version is "
            f"{wheel_version!r}. Passing a connection minted by the duckdb wheel "
            f"into a different libduckdb build is undefined behavior. Install a "
            f"matching libduckdb or set NUMBDUCK_LIBDUCKDB to a libduckdb whose "
            f"version equals {wheel_version!r}."
        )


def _migrate_legacy_cache():
    """Best-effort removal of the pre-versioned unversioned cache file.

    Releases before the per-version cache layout wrote the standalone dylib
    directly under _LIBDUCKDB_CACHE_BASE. This cleanup is kept out of the
    discovery function and guarded so a read-only home directory or a process
    racing the same unlink can never abort import over a stale migration
    artifact; a failed unlink simply leaves the harmless legacy file in place.
    """
    old_cached = os.path.join(_LIBDUCKDB_CACHE_BASE, _LIBDUCKDB_DYLIB_NAME)
    try:
        os.remove(old_cached)
    except OSError:
        pass


def _find_standalone_libduckdb():
    """Locate a standalone libduckdb, or return ``None``, without side effects.

    Consulted in a fixed, documented order so the pick is deterministic: an
    explicit ``NUMBDUCK_LIBDUCKDB`` override first, then numbduck's own
    per-version cache, then the known Homebrew install paths. Pure lookup — it
    never mutates the filesystem, so it is safe on the import-time load path and
    testable as a plain discovery function.
    """
    env_path = os.environ.get("NUMBDUCK_LIBDUCKDB")
    if env_path and os.path.isfile(env_path):
        return env_path
    if os.path.isfile(_LIBDUCKDB_CACHED_DYLIB):
        return _LIBDUCKDB_CACHED_DYLIB
    for path in _MACOS_LIBDUCKDB_SEARCH_PATHS:
        if os.path.isfile(path):
            return path
    return None


def _consent_to_download(version):
    """Decide whether the standalone-libduckdb download is authorized.

    Pure policy plus, at most, the consent prompt itself: no network, extract,
    or cache I/O happens here. ``NUMBDUCK_LIBDUCKDB_DOWNLOAD=1`` authorizes the
    download unattended; otherwise an interactive TTY is prompted. A closed or
    redirected stdin (a headless import, CI, a subprocess) is treated as "no" —
    returning ``False`` lets the caller surface the branded install guidance
    instead of letting ``input`` raise ``EOFError`` out of a bare import.
    """
    if os.environ.get("NUMBDUCK_LIBDUCKDB_DOWNLOAD") == "1":
        return True
    stdin = sys.stdin
    if stdin is None or not stdin.isatty():
        return False
    print(
        f"\nnumbduck: The installed duckdb Python "
        f"package (v{version}) does not export "
        f"C API symbols on macOS.\n"
        f"numbduck can download the standalone "
        f"libduckdb.dylib and cache it in "
        f"{_LIBDUCKDB_CACHE_DIR}\n"
    )
    try:
        answer = input("Download now? [y/N] ").strip().lower()
    except EOFError:
        # isatty lied (or stdin reached EOF mid-prompt); treat as no consent.
        return False
    return answer in ("y", "yes")


def _fetch_and_cache(url, dest):
    """Download the libduckdb ZIP at *url*, extract the dylib, and cache it at *dest*.

    Downloaded over HTTPS with a bounded timeout; the extracted dylib is staged to
    a unique temp file and atomically ``os.replace``'d into place so two processes
    downloading at once can't clobber a file another may already have mapped. There
    is deliberately no cryptographic check here: DuckDB publishes no signed
    checksum, so this trusts the official release fetched over TLS exactly as
    ``pip`` trusts the duckdb wheel it installs the same way. The caller re-checks
    that the loaded library actually exports the C API. Returns *dest*.
    """
    import io
    import zipfile
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen
    try:
        with urlopen(url, timeout=_LIBDUCKDB_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()
    except (HTTPError, URLError, OSError) as exc:
        # OSError also covers socket.timeout from the bounded urlopen above.
        raise RuntimeError(
            f"numbduck failed to download libduckdb from {url}: {exc}. Install "
            f"it via brew install duckdb, or set NUMBDUCK_LIBDUCKDB="
            f"/path/to/libduckdb.dylib."
        ) from exc
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            member = zf.read(_LIBDUCKDB_DYLIB_NAME)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise RuntimeError(
            f"numbduck could not extract {_LIBDUCKDB_DYLIB_NAME} from {url}: "
            f"{exc}. Install it via brew install duckdb, or set "
            f"NUMBDUCK_LIBDUCKDB=/path/to/libduckdb.dylib."
        ) from exc
    cache_dir = os.path.dirname(dest)
    os.makedirs(cache_dir, exist_ok=True)
    # Stage into a unique temp file and atomically rename into place: unique names
    # keep concurrent downloads from clobbering each other's in-flight write, and
    # the same-dir os.replace never rewrites a libduckdb.dylib another process may
    # already have mapped.
    fd_dest, tmp_dest = tempfile.mkstemp(dir=cache_dir, prefix="libduckdb.", suffix=".tmp")
    try:
        with os.fdopen(fd_dest, "wb") as dst:
            dst.write(member)
        os.replace(tmp_dest, dest)
    except OSError as exc:
        try:
            os.remove(tmp_dest)
        except OSError:
            pass
        raise RuntimeError(
            f"numbduck could not write the downloaded libduckdb to its cache at "
            f"{dest!r} ({exc}). Check permissions on {os.path.dirname(dest)!r}, "
            f"or set NUMBDUCK_LIBDUCKDB=/path/to/libduckdb.dylib to skip the "
            f"download."
        ) from exc
    return dest


def _download_libduckdb():
    version = duckdb.__version__
    url = (
        "https://github.com/duckdb/duckdb/releases"
        f"/download/v{version}"
        "/libduckdb-osx-universal.zip"
    )
    if not _consent_to_download(version):
        raise RuntimeError(
            "numbduck requires the DuckDB C API but consent to download the "
            "standalone libduckdb was not given. Re-run in a terminal and "
            "answer yes, or set NUMBDUCK_LIBDUCKDB_DOWNLOAD=1 to authorize the "
            "download unattended. Alternatively install it via 'brew install "
            "duckdb' and set NUMBDUCK_LIBDUCKDB=/path/to/libduckdb.dylib."
        )
    print(f"numbduck: Downloading libduckdb "
          f"v{version}...")
    dest = _fetch_and_cache(url, _LIBDUCKDB_CACHED_DYLIB)
    print(f"numbduck: Saved to {dest}")
    return dest


def find_duckdb_shared_lib():
    # Consulted in a fixed order: the duckdb 1.3.x layout (shared lib inside the
    # duckdb/ package dir) first, then the 1.4+ layout (shared lib in the parent
    # site-packages/ as _duckdb.*). The extension alternation is end-anchored so
    # a name that merely contains a valid suffix (e.g. a ".so.debug" sidecar or
    # a versioned ".so.1" soname) cannot ride along and trip the len==1 guard.
    duckdb_dir = os.path.dirname(getfile(duckdb))
    duckdb_dir_files = next(iter(os.walk(duckdb_dir)))[2]
    pkg_libs = [file_ for file_ in duckdb_dir_files if re.fullmatch(r"duckdb[\w.-]*\.(so|dll|dylib|pyd)", file_)]
    if len(pkg_libs) == 1:
        return os.path.join(duckdb_dir, pkg_libs[0])
    parent_dir = os.path.dirname(duckdb_dir)
    parent_files = next(iter(os.walk(parent_dir)))[2]
    site_libs = [file_ for file_ in parent_files if re.fullmatch(r"_duckdb[\w.-]*\.(so|dll|dylib|pyd)", file_)]
    if len(site_libs) == 1:
        return os.path.join(parent_dir, site_libs[0])
    raise RuntimeError(
        f"could not find unambiguous duckdb shared library: "
        f"duckdb/ candidates = {pkg_libs}, site-packages/ candidates = {site_libs}"
    )


def _non_darwin_capi_error():
    """Message for the non-macOS branch when the wheel lacks the C API.

    Distinguishes an unset ``NUMBDUCK_LIBDUCKDB`` from one set to a path that
    was not found (echoing the rejected path so the user is not told to do what
    they just did), and states the platform asymmetry: the automatic standalone
    download is macOS-only, so off macOS the env var is the recourse.
    """
    env_path = os.environ.get("NUMBDUCK_LIBDUCKDB")
    if env_path:
        detail = (
            f"NUMBDUCK_LIBDUCKDB is set to {env_path!r}, but no readable "
            f"libduckdb was found there; point it at an existing libduckdb "
            f"shared library that exports the DuckDB C API."
        )
    else:
        lib_name = (
            "libduckdb.dll" if platform.system() == "Windows"
            else "libduckdb.so"
        )
        detail = (
            f"Set NUMBDUCK_LIBDUCKDB=/path/to/{lib_name} to a libduckdb "
            f"shared library that exports the DuckDB C API."
        )
    return (
        "numbduck could not find the DuckDB C API: the installed duckdb wheel "
        f"does not export it. {detail} The automatic standalone-libduckdb "
        "download is macOS-only, so on this platform NUMBDUCK_LIBDUCKDB (or a "
        "system libduckdb already on the loader path) is the way to supply it."
    )


def load_duckdb():
    global _loaded_libduckdb
    lib_path = find_duckdb_shared_lib()
    lib = load_lib_path(lib_path)
    if _has_capi_symbols(lib):
        # Single-runtime invariant: the wheel's own shared object backs both the
        # Python ``duckdb`` module and numbduck's JIT bindings, so any handle is
        # allocated and consumed by one libduckdb — coordinated by construction.
        _loaded_libduckdb = lib
        return lib
    # Python wheel missing C API symbols (seen on the macOS duckdb wheels that
    # strip them; detected dynamically by _has_capi_symbols above rather than
    # pinned to a version window that would rot as the duckdb pin advances).
    # numbduck must load a second, standalone libduckdb RTLD_GLOBAL to supply the
    # C API, leaving two DuckDB runtimes resident: the wheel's (used by the
    # Python ``duckdb`` module) and this standalone (bound by numbduck's JIT
    # code). The single-runtime invariant no longer holds automatically. pybridge
    # hands a Connection* minted by the wheel to this standalone's C API, which is
    # sound only when the two are the same DuckDB build; refuse any version
    # mismatch rather than dereference a cross-runtime pointer. Every handle used
    # from JIT must originate from this standalone runtime, never from the Python
    # ``duckdb`` module directly.
    _migrate_legacy_cache()
    standalone = _find_standalone_libduckdb()
    if standalone:
        lib = load_lib_path(standalone)
        if _has_capi_symbols(lib):
            _require_coordinated_standalone(lib, standalone)
            _loaded_libduckdb = lib
            return lib
    if platform.system() != "Darwin":
        raise RuntimeError(_non_darwin_capi_error())
    downloaded = _download_libduckdb()
    lib = load_lib_path(downloaded)
    if not _has_capi_symbols(lib):
        # Mirror the standalone branch's check so a corrupt or wrong-arch
        # download fails here with a pointer back to the cache, not later as an
        # opaque JIT link error at the first @njit call.
        raise RuntimeError(
            f"numbduck downloaded libduckdb to {downloaded!r} but it does not "
            f"export the DuckDB C API symbols numbduck needs. Delete "
            f"{_LIBDUCKDB_CACHE_DIR!r} and retry, or set NUMBDUCK_LIBDUCKDB to a "
            f"libduckdb that provides the C API."
        )
    _require_coordinated_standalone(lib, downloaded)
    _loaded_libduckdb = lib
    return lib


def loaded_library_version():
    """Normalized version of the libduckdb backing numbduck's JIT bindings.

    Returns ``None`` before :func:`load_duckdb` has run or when the version
    cannot be read.
    """
    if _loaded_libduckdb is None:
        return None
    return _library_version(_loaded_libduckdb)


def libraries_coordinated():
    """True when numbduck's JIT libduckdb matches the Python ``duckdb`` wheel.

    Compares the library version reported by the libduckdb :func:`load_duckdb`
    bound into numbduck's bindings against the wheel's own library version
    (:func:`_wheel_library_version`, read via ``PRAGMA version`` so it is the same
    ``vX.Y.Z[-devN]`` scheme, not the ``duckdb.__version__`` package version).
    Only an exact match makes it safe to hand a wheel-minted ``Connection*``
    across the two, mirroring numbox's ``libraries_coordinated`` guard. When it
    returns ``False``, pybridge refuses the extraction rather than dereference a
    pointer under a possibly-different internal layout.
    """
    jit_version = loaded_library_version()
    if jit_version is None:
        return False
    wheel_version = _wheel_library_version()
    if wheel_version is None:
        return False
    return jit_version == wheel_version
