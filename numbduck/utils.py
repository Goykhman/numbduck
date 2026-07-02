import ctypes
import duckdb
import os
import platform
import re
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


def _find_standalone_libduckdb():
    env_path = os.environ.get("NUMBDUCK_LIBDUCKDB")
    if env_path and os.path.isfile(env_path):
        return env_path
    # Remove old unversioned cache if present
    old_cached = os.path.join(_LIBDUCKDB_CACHE_BASE, "libduckdb.dylib")
    if os.path.isfile(old_cached):
        os.remove(old_cached)
    cached = os.path.join(_LIBDUCKDB_CACHE_DIR, "libduckdb.dylib")
    if os.path.isfile(cached):
        return cached
    for path in _MACOS_LIBDUCKDB_SEARCH_PATHS:
        if os.path.isfile(path):
            return path
    return None


def _download_libduckdb():
    version = duckdb.__version__
    url = (
        "https://github.com/duckdb/duckdb/releases"
        f"/download/v{version}"
        "/libduckdb-osx-universal.zip"
    )
    auto = os.environ.get("NUMBDUCK_LIBDUCKDB_DOWNLOAD")
    if auto != "1":
        print(
            f"\nnumbduck: The installed duckdb Python "
            f"package (v{version}) does not export "
            f"C API symbols on macOS.\n"
            f"numbduck can download the standalone "
            f"libduckdb.dylib and cache it in "
            f"{_LIBDUCKDB_CACHE_DIR}\n"
        )
        answer = input(
            "Download now? [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            raise RuntimeError(
                "numbduck requires the DuckDB C API."
                " Install it via:\n"
                "  brew install duckdb\n"
                "or set NUMBDUCK_LIBDUCKDB="
                "/path/to/libduckdb.dylib"
            )
    import io
    import zipfile
    from urllib.request import urlopen
    print(f"numbduck: Downloading libduckdb "
          f"v{version}...")
    with urlopen(url) as resp:
        data = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(data))
    os.makedirs(_LIBDUCKDB_CACHE_DIR, exist_ok=True)
    dest = os.path.join(_LIBDUCKDB_CACHE_DIR, "libduckdb.dylib")
    with zf.open("libduckdb.dylib") as src, open(dest, "wb") as dst:
        dst.write(src.read())
    print(f"numbduck: Saved to {dest}")
    return dest


def find_duckdb_shared_lib():
    duckdb_dir = os.path.dirname(getfile(duckdb))
    duckdb_dir_files = next(iter(os.walk(duckdb_dir)))[2]
    # duckdb 1.3.x: shared lib inside duckdb/ package dir
    pkg_libs = [file_ for file_ in duckdb_dir_files if re.match(r"duckdb[\w.-]*\.(so|dll|dylib|pyd)", file_)]
    if len(pkg_libs) == 1:
        return os.path.join(duckdb_dir, pkg_libs[0])
    # duckdb 1.4+: shared lib in parent site-packages/ as _duckdb.*
    parent_dir = os.path.dirname(duckdb_dir)
    parent_files = next(iter(os.walk(parent_dir)))[2]
    site_libs = [file_ for file_ in parent_files if re.match(r"_duckdb[\w.-]*\.(so|dll|dylib|pyd)", file_)]
    if len(site_libs) == 1:
        return os.path.join(parent_dir, site_libs[0])
    raise RuntimeError(
        f"could not find unambiguous duckdb shared library: "
        f"duckdb/ candidates = {pkg_libs}, site-packages/ candidates = {site_libs}"
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
    # Python wheel missing C API symbols (macOS with 1.5.2 >= duckdb >= 1.4.1).
    # numbduck must load a second, standalone libduckdb RTLD_GLOBAL to supply the
    # C API, leaving two DuckDB runtimes resident: the wheel's (used by the
    # Python ``duckdb`` module) and this standalone (bound by numbduck's JIT
    # code). The single-runtime invariant no longer holds automatically. pybridge
    # hands a Connection* minted by the wheel to this standalone's C API, which is
    # sound only when the two are the same DuckDB build; refuse any version
    # mismatch rather than dereference a cross-runtime pointer. Every handle used
    # from JIT must originate from this standalone runtime, never from the Python
    # ``duckdb`` module directly.
    standalone = _find_standalone_libduckdb()
    if standalone:
        lib = load_lib_path(standalone)
        if _has_capi_symbols(lib):
            _require_coordinated_standalone(lib, standalone)
            _loaded_libduckdb = lib
            return lib
    if platform.system() != "Darwin":
        raise RuntimeError(
            "DuckDB C API symbols not found in the Python wheel. "
            "Set NUMBDUCK_LIBDUCKDB=/path/to/libduckdb.so"
        )
    downloaded = _download_libduckdb()
    lib = load_lib_path(downloaded)
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
