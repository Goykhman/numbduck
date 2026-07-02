import ctypes
import duckdb
import hashlib
import os
import platform
import re
import sys
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

# Filename the standalone libduckdb is cached under, and the sidecar recording
# the SHA-256 of exactly those bytes so a cache poisoned between runs is caught
# before the dylib is dlopen'd.
_LIBDUCKDB_CACHED_DYLIB = os.path.join(_LIBDUCKDB_CACHE_DIR, _LIBDUCKDB_DYLIB_NAME)

# Bounded so a hung or trickling endpoint fails fast instead of wedging import.
_LIBDUCKDB_DOWNLOAD_TIMEOUT = 60

# SHA-256 of each supported ``libduckdb-osx-universal.zip`` release asset, pinned
# in source. The downloaded ZIP is checked against this before anything is
# extracted, persisted, or dlopen'd, so neither a compromised GitHub release
# asset nor a poisoned local cache can substitute attacker-controlled native
# code. DuckDB publishes no per-release checksum asset, so these were computed
# from the released zips. A version absent here needs an explicit
# NUMBDUCK_LIBDUCKDB_SHA256 override or the download is refused (fail-closed).
_PINNED_LIBDUCKDB_SHA256 = {
    "1.3.2": "6e4a9bfd4f15c83f43b9585b4f62c6f9851de3410cd28275de37d6d74b138415",
    "1.4.0": "b81db597d72bea1beb20edff78f69961b0bec4a16841996b1d95bd72ee89514c",
    "1.5.1": "d9dd723c59f8571202b468f6bf71d4555238544553dd1445e6c9ecb39f54c0f3",
}

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
    # duckdb_library_version() reports e.g. "v1.5.4" while duckdb.__version__ is
    # "1.5.4"; strip the leading "v" so the two compare directly.
    return version.strip().lstrip("v")


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


def _require_coordinated_standalone(lib, source):
    """Refuse a standalone libduckdb whose version disagrees with the wheel.

    pybridge extracts a raw ``Connection*`` minted by the Python ``duckdb``
    wheel's runtime and hands it to the C API resolved against this standalone
    library. That is only sound when both are the same DuckDB build: a
    different build casts the pointer back to its own internal Connection
    layout, so dereferencing it is undefined behavior. Detect the mismatch and
    refuse rather than corrupt memory at query time.
    """
    lib_version = _library_version(lib)
    wheel_version = _normalize_version(duckdb.__version__)
    if lib_version is not None and lib_version != wheel_version:
        raise RuntimeError(
            f"numbduck loaded a standalone libduckdb (version {lib_version!r}) "
            f"from {source!r}, but the installed Python duckdb package is "
            f"version {wheel_version!r}. Passing a connection minted by the "
            f"duckdb wheel into a different libduckdb build is undefined "
            f"behavior. Install a matching libduckdb or set NUMBDUCK_LIBDUCKDB "
            f"to a libduckdb whose version equals {wheel_version!r}."
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


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _expected_zip_sha256(version):
    """Expected SHA-256 of *version*'s libduckdb ZIP, or ``None`` if unknown.

    An explicit ``NUMBDUCK_LIBDUCKDB_SHA256`` override is authoritative (the user
    takes responsibility, as they already can with ``NUMBDUCK_LIBDUCKDB``);
    otherwise the in-source pin is used. ``None`` means neither is available and
    the download must be refused.
    """
    override = os.environ.get("NUMBDUCK_LIBDUCKDB_SHA256")
    if override:
        return override.strip().lower()
    return _PINNED_LIBDUCKDB_SHA256.get(version)


def _verify_cached_dylib(path):
    """Re-hash a cached standalone libduckdb against its download-time sidecar.

    The sidecar (written 0o600 inside the 0o700 cache dir when the download's ZIP
    digest was verified) records the SHA-256 of exactly the bytes that were
    persisted. Recompute it on every load so a cache poisoned between runs is
    refused before the file is dlopen'd (dlopen runs the Mach-O's constructors
    immediately). This bounds the local-poisoning window only: an attacker who
    can rewrite both the dylib and its sidecar together defeats it, but that
    already requires same-user write access to the 0o700/0o600 per-user cache, so
    it is out of scope for this threat model. The download-time pinned-ZIP digest
    is the layer that resists a compromised or corrupted upstream asset.
    """
    sidecar = path + ".sha256"
    try:
        with open(sidecar) as fh:
            expected = fh.read().strip()
    except OSError:
        expected = None
    if not expected:
        raise RuntimeError(
            f"numbduck's cached libduckdb at {path!r} has no integrity sidecar "
            f"({sidecar!r}); refusing to load it. Delete "
            f"{_LIBDUCKDB_CACHE_DIR!r} to re-download, or set NUMBDUCK_LIBDUCKDB "
            f"to a trusted libduckdb.dylib."
        )
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"numbduck's cached libduckdb at {path!r} failed its integrity "
            f"check (expected sha256 {expected}, got {actual}); the cache is "
            f"corrupted or tampered with. Delete {_LIBDUCKDB_CACHE_DIR!r} to "
            f"re-download, or set NUMBDUCK_LIBDUCKDB to a trusted "
            f"libduckdb.dylib."
        )


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


def _fetch_and_cache(url, dest, expected_sha):
    """Download the libduckdb ZIP at *url*, verify it, and cache the dylib.

    The expected digest is injected rather than read from module globals so
    this fetch/verify/extract/persist seam can be driven end-to-end in a unit
    test against a locally built ``file://`` archive with a computable digest.
    Nothing is extracted, persisted, or dlopen'd until the downloaded bytes
    match *expected_sha*, so neither a substituted release asset nor a corrupted
    transfer can plant native code. The extracted dylib is written to *dest*
    (0o600) inside a 0o700 cache dir, with a re-verification sidecar recording
    the SHA-256 of exactly the persisted bytes. Returns *dest*.
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
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != expected_sha:
        # Reject before extract/persist/load: mismatched bytes mean a corrupted
        # download or a substituted (compromised) release asset.
        raise RuntimeError(
            f"numbduck downloaded libduckdb from {url} but its SHA-256 "
            f"{actual_sha} does not match the expected {expected_sha}; refusing "
            f"to extract or load it. Install libduckdb via brew install duckdb "
            f"and set NUMBDUCK_LIBDUCKDB instead."
        )
    try:
        member = zipfile.ZipFile(io.BytesIO(data)).read(_LIBDUCKDB_DYLIB_NAME)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise RuntimeError(
            f"numbduck could not extract {_LIBDUCKDB_DYLIB_NAME} from {url}: "
            f"{exc}. Install it via brew install duckdb, or set "
            f"NUMBDUCK_LIBDUCKDB=/path/to/libduckdb.dylib."
        ) from exc
    cache_dir = os.path.dirname(dest)
    os.makedirs(cache_dir, mode=0o700, exist_ok=True)
    # exist_ok skips the mode above when the dir already exists under a looser
    # umask, so tighten it explicitly to close the local-poisoning window.
    os.chmod(cache_dir, 0o700)
    # Stage into temp files in the same dir and os.replace() into place so we
    # never rewrite a libduckdb.dylib another process may already have mapped
    # (an in-place rewrite can corrupt that live mapping); a same-dir rename is
    # atomic. Put the sidecar in place first so that once the dylib appears, its
    # integrity sidecar is already paired with it.
    digest = hashlib.sha256(member).hexdigest()
    sidecar = dest + ".sha256"
    tmp_dest = dest + ".tmp"
    tmp_sidecar = sidecar + ".tmp"
    try:
        with open(tmp_dest, "wb") as dst:
            dst.write(member)
        os.chmod(tmp_dest, 0o600)
        with open(tmp_sidecar, "w") as fh:
            fh.write(digest)
        os.chmod(tmp_sidecar, 0o600)
        os.replace(tmp_sidecar, sidecar)
        os.replace(tmp_dest, dest)
    except OSError:
        for tmp in (tmp_dest, tmp_sidecar):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    return dest


def _download_libduckdb():
    version = duckdb.__version__
    url = (
        "https://github.com/duckdb/duckdb/releases"
        f"/download/v{version}"
        "/libduckdb-osx-universal.zip"
    )
    expected_sha = _expected_zip_sha256(version)
    if expected_sha is None:
        # Fail closed: without a pinned digest or an explicit override there is
        # no way to tell a genuine release asset from a substituted one, so
        # never download-and-hope.
        raise RuntimeError(
            f"numbduck has no pinned SHA-256 for the libduckdb '{version}' "
            f"release asset, so it will not download and execute unverified "
            f"native code from {url}. Install libduckdb yourself (brew install "
            f"duckdb) and set NUMBDUCK_LIBDUCKDB=/path/to/libduckdb.dylib, or "
            f"set NUMBDUCK_LIBDUCKDB_SHA256 to the expected SHA-256 of that ZIP "
            f"to authorize the download."
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
    dest = _fetch_and_cache(url, _LIBDUCKDB_CACHED_DYLIB, expected_sha)
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
        detail = (
            "Set NUMBDUCK_LIBDUCKDB=/path/to/libduckdb.so to a libduckdb "
            "shared library that exports the DuckDB C API."
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
        if standalone == _LIBDUCKDB_CACHED_DYLIB:
            # Re-verify our own cache on every hit before dlopen'ing it; an
            # env-var or Homebrew path is user-supplied and trusted as-is.
            _verify_cached_dylib(standalone)
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
    """True when numbduck's JIT libduckdb matches the Python ``duckdb`` module.

    Compares the version reported by the libduckdb :func:`load_duckdb` bound
    into numbduck's bindings against ``duckdb.__version__`` (the wheel that mints
    the ``Connection*`` pybridge extracts). Only an exact match makes it safe to
    hand that pointer across the two, mirroring numbox's ``libraries_coordinated``
    guard. When it returns ``False``, pybridge refuses the extraction rather than
    dereference a pointer under a possibly-different internal layout.
    """
    jit_version = loaded_library_version()
    if jit_version is None:
        return False
    return jit_version == _normalize_version(duckdb.__version__)
