"""Find and load the libduckdb whose C API numbduck's bindings call, and check that it matches the duckdb wheel.

:mod:`numbduck.ducklib` calls :func:`load_duckdb` at import. :func:`find_duckdb_shared_lib` locates the shared library
inside the installed duckdb wheel, and the loader uses it if it exports DuckDB's C API. Some macOS wheels do not, as
:func:`load_duckdb` lists. The loader then falls back to a standalone libduckdb: the file that ``NUMBDUCK_LIBDUCKDB``
names, numbduck's cache in ``~/.numbduck/lib``, or a Homebrew install. On macOS, it can also download one from DuckDB's
GitHub release after the user consents. A standalone library is loaded only if it reports the same library version as
the wheel, and :func:`libraries_coordinated` repeats that comparison for :mod:`numbduck.pybridge`.

The module also holds ``duckdb_version`` and ``duckdb_major_version``, read from the installed duckdb package. Its
:func:`complete_open_result` finishes the open result of a Python connection, which duckdb 2.0.0 requires before the
C API can run a query on that connection.
"""
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


# numbduck caches a downloaded libduckdb under ~/.numbduck/lib, in a directory for each version of the duckdb package.
_LIBDUCKDB_CACHE_BASE = os.path.join(
    os.path.expanduser("~"), ".numbduck", "lib"
)
_LIBDUCKDB_CACHE_DIR = os.path.join(
    _LIBDUCKDB_CACHE_BASE, duckdb.__version__
)

# The file name of the standalone libduckdb. The cache path that the search looks at, the path that the download
# writes, the member that the download extracts from DuckDB's archive and the old cache file all use this one name, so
# their file names cannot differ.
_LIBDUCKDB_DYLIB_NAME = "libduckdb.dylib"

# The path of the cached standalone libduckdb for the installed duckdb version.
_LIBDUCKDB_CACHED_DYLIB = os.path.join(_LIBDUCKDB_CACHE_DIR, _LIBDUCKDB_DYLIB_NAME)

# The timeout of the download, in seconds. urlopen applies it to each blocking socket operation, so a server that stops
# responding fails the import within a minute instead of hanging it. A download that keeps receiving data is not cut
# off, however slow it is.
_LIBDUCKDB_DOWNLOAD_TIMEOUT = 60

# Where Homebrew's duckdb puts libduckdb.dylib: under /opt/homebrew on Apple silicon and under /usr/local on Intel.
_MACOS_LIBDUCKDB_SEARCH_PATHS = [
    "/opt/homebrew/lib/libduckdb.dylib",
    "/usr/local/lib/libduckdb.dylib",
]


# The ctypes handle of the libduckdb whose C API the bindings call. load_duckdb sets it, and loaded_library_version
# reads its version. If the wheel's library exports the C API, this is the library that the Python duckdb module uses.
# Otherwise it is a second libduckdb, loaded from a standalone file.
_loaded_libduckdb = None


def _has_capi_symbols(lib):
    return hasattr(lib, "duckdb_open")


def _normalize_version(version):
    # duckdb_library_version() and PRAGMA version report a version with a leading "v", such as "v1.5.4". The function
    # strips the surrounding whitespace and that one "v", so that the version reads as "1.5.4", the form of the package
    # version. removeprefix removes one "v", where lstrip would remove a run of them.
    return version.strip().removeprefix("v")


def _library_version(lib):
    """Return the version that ``lib``'s ``duckdb_library_version()`` reports, without its leading ``v``.

    Return None if ``lib`` does not export the function, or if the function returns NULL.
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
    """Return the library version that the duckdb wheel reports for its own core, without its leading ``v``, or None.

    The function runs ``PRAGMA version`` on an in-memory connection of the Python ``duckdb`` module, which works for a
    wheel that does not export the C API too. The result has the ``vX.Y.Z[-devN]`` form of a standalone library's
    ``duckdb_library_version()``, so the two compare directly. The package version, ``duckdb.__version__``, equals
    the library version only for a release. It differs for a development or release-candidate build, and a
    comparison with it would refuse a library that matches the wheel.

    The version is kept after the first successful read, because the loaded wheel cannot change within a process. If
    the query fails or returns no version, the function returns None, and the next call tries again.
    """
    global _wheel_library_version_cache
    if _wheel_library_version_cache is None:
        try:
            row = duckdb.connect(":memory:").execute("PRAGMA version").fetchone()
        except Exception:
            # Without the wheel's version, the callers refuse the standalone library and the connection pointer, rather
            # than pair two libraries that might be different builds.
            return None
        if not row or row[0] is None:
            return None
        _wheel_library_version_cache = _normalize_version(row[0])
    return _wheel_library_version_cache


def _require_coordinated_standalone(lib, source):
    """Raise ``RuntimeError`` unless the standalone libduckdb ``lib`` reports the same library version as the wheel.

    :mod:`numbduck.pybridge` reads the ``Connection*`` that the duckdb wheel created for a Python connection, and the
    bindings pass it to this library's C API. ``duckdb_query`` there treats the pointer as a ``Connection`` of its own
    build and dereferences it. That can work only if both libraries are the same DuckDB build, because another build
    may lay the struct out differently, and the dereference is then undefined behavior. Both libraries report their
    version in the ``vX.Y.Z[-devN]`` scheme, the wheel through :func:`_wheel_library_version` and ``lib`` through
    :func:`_library_version`. The function also raises if either version cannot be read. ``source`` is the path that
    the messages name.
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


def _open_private(path):
    """Open the library at ``path`` without adding its symbols to the process-wide scope, and return its handle.

    The library is opened with ``RTLD_LOCAL`` where ``os`` defines it, as on Linux and macOS, and with the default mode
    of ``ctypes.CDLL`` elsewhere.
    """
    mode = getattr(os, "RTLD_LOCAL", None)
    return ctypes.CDLL(path) if mode is None else ctypes.CDLL(path, mode=mode)


def _load_coordinated_standalone(path):
    """Load the standalone libduckdb at ``path`` for the bindings if it exports the C API and matches the wheel.

    Return the handle that numbox's ``load_lib_path`` opens, or None if the library does not export the C API. Raise
    ``RuntimeError`` if its version differs from the wheel's or if either version cannot be read, and ``OSError`` if
    ``path`` cannot be opened as a library.
    """
    # The bindings call C functions by name, and the JIT linker resolves each name in the process-wide symbol scope,
    # where the first library that defines the name takes precedence. ctypes never unloads a library. If a library that
    # failed the checks entered that scope, the bindings would therefore call it for the life of the process, even after
    # another library passed the checks. The library is opened privately first, and it enters that scope only after
    # both checks pass.
    probe = _open_private(path)
    if not _has_capi_symbols(probe):
        return None
    _require_coordinated_standalone(probe, path)
    return load_lib_path(path)


def _migrate_legacy_cache():
    """Remove the file in which earlier releases cached the standalone libduckdb, if it exists, and ignore a failure.

    Earlier numbduck releases cached the library as ``~/.numbduck/lib/libduckdb.dylib``, outside the directory of a
    duckdb version. A removal that fails, as in a read-only home directory or when another process has removed the file
    first, does not fail the import.
    """
    old_cached = os.path.join(_LIBDUCKDB_CACHE_BASE, _LIBDUCKDB_DYLIB_NAME)
    try:
        os.remove(old_cached)
    except OSError:
        pass


def _find_standalone_libduckdb():
    """Return the path of the first standalone libduckdb found, or None; the search changes nothing on disk.

    The search looks, in this order, at the file that ``NUMBDUCK_LIBDUCKDB`` names, at numbduck's cache for the
    installed duckdb version, ``~/.numbduck/lib/<version>/libduckdb.dylib``, and at Homebrew's ``libduckdb.dylib`` in
    ``/opt/homebrew/lib`` and ``/usr/local/lib``. A ``NUMBDUCK_LIBDUCKDB`` that names no existing file is skipped. The
    cache and Homebrew paths are macOS ``.dylib`` files, so off macOS only ``NUMBDUCK_LIBDUCKDB`` supplies a library.
    The function does not check that the file it returns is a library.
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
    """Return whether numbduck may download the standalone libduckdb, asking the user only at a terminal.

    ``NUMBDUCK_LIBDUCKDB_DOWNLOAD=1`` allows the download without a question. Otherwise, if stdin is a terminal, the
    function prints what it would download and where it would cache it, and asks ``Download now? [y/N]``; ``y`` or
    ``yes`` allows the download. Without a terminal, as in a headless import, CI or a subprocess, the answer is no.
    ``input`` would raise ``EOFError`` there, and the error would propagate out of the import. A return value of False
    instead lets the caller raise its own error, whose message names the ways to install libduckdb. The function
    downloads nothing and writes no file.
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
        # stdin reported a terminal but reached its end before an answer, so the answer is no.
        return False
    return answer in ("y", "yes")


def _fetch_and_cache(url, dest):
    """Download the ZIP archive at ``url``, extract ``libduckdb.dylib``, write it to ``dest``, and return ``dest``.

    The download fails if the server does not respond for ``_LIBDUCKDB_DOWNLOAD_TIMEOUT`` seconds. The library replaces
    ``dest`` in one step, so two processes that download at once do not corrupt it, and a process that has already
    loaded the old file keeps it. ``RuntimeError`` is raised if the download raises urllib's ``HTTPError`` or
    ``URLError`` or an ``OSError``, such as a timeout, if the archive lacks the library, or if writing or moving the
    library fails. An ``OSError`` from creating the cache directory or the temporary file propagates unchanged. No
    checksum is checked, because DuckDB publishes no signed checksum: the function trusts the official release over
    HTTPS, as pip trusts the duckdb wheel that it installs the same way. The caller checks that the library exports the
    C API.
    """
    import io
    import zipfile
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen
    try:
        with urlopen(url, timeout=_LIBDUCKDB_DOWNLOAD_TIMEOUT) as resp:
            data = resp.read()
    except (HTTPError, URLError, OSError) as exc:
        # The error that the timeout raises, socket.timeout, is an OSError.
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
    # The function writes the library to a temporary file with a unique name in the cache directory, and os.replace
    # then moves that file over dest in one step. Two processes that download at once each write their own temporary
    # file. os.replace gives dest a new file instead of writing into the old one, which another process may have loaded.
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


duckdb_version = duckdb.__version__
duckdb_major_version = int(duckdb_version.split(".")[0])


def _download_libduckdb():
    """Download the standalone libduckdb of the installed duckdb version to numbduck's cache, and return its path.

    The function asks for consent first, and raises ``RuntimeError`` if the answer is no. It downloads
    ``libduckdb-osx-universal.zip`` from DuckDB's GitHub release whose tag is ``v`` followed by ``duckdb.__version__``.
    """
    # Every DuckDB release from v1.2.2 to v1.5.6 has libduckdb-osx-universal.zip. The v2.0.0 release publishes the
    # library as duckdb-shared-libs-osx-universal.tar.gz instead, so for duckdb 2.0.0 this URL returns HTTP 404 and the
    # download raises RuntimeError. The macOS wheels of duckdb 2.0.0 export the C API, so the import never reaches this
    # download for 2.0.0.
    url = (
        "https://github.com/duckdb/duckdb/releases"
        f"/download/v{duckdb_version}"
        "/libduckdb-osx-universal.zip"
    )
    if not _consent_to_download(duckdb_version):
        raise RuntimeError(
            "numbduck requires the DuckDB C API but consent to download the "
            "standalone libduckdb was not given. Re-run in a terminal and "
            "answer yes, or set NUMBDUCK_LIBDUCKDB_DOWNLOAD=1 to authorize the "
            "download unattended. Alternatively install it via 'brew install "
            "duckdb' and set NUMBDUCK_LIBDUCKDB=/path/to/libduckdb.dylib."
        )
    print(f"numbduck: Downloading libduckdb "
          f"v{duckdb_version}...")
    dest = _fetch_and_cache(url, _LIBDUCKDB_CACHED_DYLIB)
    print(f"numbduck: Saved to {dest}")
    return dest


def find_duckdb_shared_lib():
    """Return the path of the shared library inside the installed duckdb wheel.

    duckdb 1.3 and earlier install the library in the ``duckdb`` package directory, as a file whose name starts with
    ``duckdb``. duckdb 1.4.0 and later install it in ``site-packages`` as ``_duckdb`` with the platform's extension,
    such as ``_duckdb.abi3.so`` for duckdb 2.0.0 on Linux. The function looks in the package directory first. It raises
    ``RuntimeError`` if neither directory holds exactly one candidate.
    """
    # Each pattern must match a whole file name. A file whose name only contains a library extension, such as a
    # ".so.debug" file beside the library or a versioned ".so.1", is therefore not a candidate, so it cannot become a
    # second candidate beside the library.
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
    """Return the message of the ``RuntimeError`` that :func:`load_duckdb` raises off macOS when it finds no C API.

    The message distinguishes an unset ``NUMBDUCK_LIBDUCKDB`` from a set one. For a set one, it quotes the path, which
    names no file or a library without the C API, so that it does not ask the user to set a variable that is already
    set. For an unset one, it names ``libduckdb.dll`` on Windows and ``libduckdb.so`` elsewhere. Either way, it says
    that the automatic download of a standalone libduckdb is macOS-only, so that off macOS ``NUMBDUCK_LIBDUCKDB`` is
    the way to supply the C API.
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
    """Load the libduckdb whose C API the bindings call, and return its ``ctypes`` handle.

    :mod:`numbduck.ducklib` calls this function when it is imported. The function loads the shared library of the
    installed duckdb wheel with numbox's ``load_lib_path``, which makes the library's symbols visible to the JIT linker.
    If that library exports the C API, the bindings and the Python ``duckdb`` module share one DuckDB runtime. The
    Linux and Windows wheels export it, and so do the macOS wheels of duckdb 1.2.2 to 1.4.0 and 1.5.3 to 2.0.0. The
    macOS wheels of duckdb 1.4.1 to 1.5.2 do not.

    If the wheel's library does not export the C API, the function first deletes ``~/.numbduck/lib/libduckdb.dylib``,
    where earlier numbduck releases cached the standalone library. It deletes that file even if ``NUMBDUCK_LIBDUCKDB``
    names it. The function then tries only the first standalone libduckdb that :func:`_find_standalone_libduckdb`
    finds, and raises ``OSError`` if that file is not a loadable library. It ignores a library that does not export the
    C API, as if it had found no file. A library that exports the C API must report the wheel's library version,
    and the function raises ``RuntimeError`` if the two versions differ or if either cannot be read.

    If no standalone library is loaded, the function raises ``RuntimeError`` off macOS. On macOS, it downloads the
    library for the installed duckdb version after the user consents, as :func:`_download_libduckdb` describes. The
    download must pass the same checks: the function raises ``OSError`` if it cannot be loaded, and ``RuntimeError`` if
    it does not export the C API or reports another version.
    """
    global _loaded_libduckdb
    lib_path = find_duckdb_shared_lib()
    lib = load_lib_path(lib_path)
    if _has_capi_symbols(lib):
        # The Python duckdb module and the bindings both use the wheel's own library, so one libduckdb creates and uses
        # every handle, and no handle passes from one DuckDB build to another.
        _loaded_libduckdb = lib
        return lib
    # The macOS wheels that lack the C API export only the Python module's two entry points
    # (https://github.com/duckdb/duckdb-python/pull/81), and duckdb 1.5.3 exports the C API again
    # (https://github.com/duckdb/duckdb-python/pull/445). The versions in the docstring come from the export tables of
    # the cp312 arm64 wheel of every release from 1.2.2 to 2.0.0, and of the x86_64 and universal2 wheels of 1.4.0,
    # 1.4.1, 1.5.2 and 1.5.3. The test above looks for a symbol rather than at the duckdb version, so it needs no list
    # of versions.
    #
    # numbduck then loads a second, standalone libduckdb with RTLD_GLOBAL, and the process holds two DuckDB runtimes.
    # The Python duckdb module uses the wheel's, and the bindings call the standalone. pybridge passes a Connection*
    # that the wheel created to the standalone's C API. That can work only if the two are the same DuckDB build, so a
    # standalone library of another version is refused. Every other handle that jitted code uses must come from the
    # standalone runtime, never from the Python duckdb module.
    _migrate_legacy_cache()
    standalone = _find_standalone_libduckdb()
    if standalone:
        lib = _load_coordinated_standalone(standalone)
        if lib is not None:
            _loaded_libduckdb = lib
            return lib
    if platform.system() != "Darwin":
        raise RuntimeError(_non_darwin_capi_error())
    downloaded = _download_libduckdb()
    lib = _load_coordinated_standalone(downloaded)
    if lib is None:
        # A download that loads but does not export the C API fails here, with a message that names the cache. A file
        # that cannot be loaded, such as a corrupt download or a library for another architecture, has already raised
        # OSError in _load_coordinated_standalone. Without this check, the failure would come later, when numba
        # compiles the first binding at import. Its error would be about a missing C symbol and would name neither the
        # download nor the cache.
        raise RuntimeError(
            f"numbduck downloaded libduckdb to {downloaded!r} but it does not "
            f"export the DuckDB C API symbols numbduck needs. Delete "
            f"{_LIBDUCKDB_CACHE_DIR!r} and retry, or set NUMBDUCK_LIBDUCKDB to a "
            f"libduckdb that provides the C API."
        )
    _loaded_libduckdb = lib
    return lib


def loaded_library_version():
    """Return the version of the libduckdb that the bindings call, without its leading ``v``.

    Return None before :func:`load_duckdb` has run, or if the library does not report a version.
    """
    if _loaded_libduckdb is None:
        return None
    return _library_version(_loaded_libduckdb)


def libraries_coordinated():
    """Return whether the libduckdb that the bindings call reports the same library version as the duckdb wheel.

    Both are library versions in the ``vX.Y.Z[-devN]`` scheme: ``duckdb_library_version()`` for the bindings' library,
    and ``PRAGMA version`` for the wheel, as :func:`_wheel_library_version` reads it. The function does not use the
    package version, ``duckdb.__version__``. It returns False if either version cannot be read.

    :func:`numbduck.pybridge.extract_connection_ptr` returns a ``Connection*`` that the wheel created only if this
    function returns True, because a library of another build may lay the connection out differently. numbox's
    ``numbox.utils.pysqlite_bridge.libraries_coordinated`` makes the same check for SQLite.
    """
    jit_version = loaded_library_version()
    if jit_version is None:
        return False
    wheel_version = _wheel_library_version()
    if wheel_version is None:
        return False
    return jit_version == wheel_version


def complete_open_result(conn):
    """Finish the open result of the last query on the connection ``conn``, so that the C API can run a query on it.

    From duckdb 2.0.0, an open result holds its connection until the result is done, and the C API refuses a new query
    on that connection: ``duckdb_query`` and ``duckdb_prepare`` return ``DuckDBError`` with a
    ``Resource In Use Error``. The function calls ``conn.complete()``, a method that duckdb 2.0.0 added. ``complete()``
    runs the result to its end without reading its rows (https://github.com/duckdb/duckdb-python/releases/tag/v2.0.0),
    so every row that the caller has not fetched is discarded: afterwards the result returns no more rows, and no error
    or warning reports the loss.

    On an earlier duckdb, the function does nothing, and a C API query on a connection whose Python result is still
    open succeeds. The duckdb module reads a result in chunks of up to 2,048 rows. After such a query, the open result
    returns only the rest of the chunk that its last fetch read, and no error reports the rows that it loses. A result
    that was never fetched returns no row. On every duckdb version, fetch the rows still needed before this call and
    before any C API query on the connection.
    """
    if duckdb_major_version >= 2:
        assert hasattr(conn, "complete")
        conn.complete()
