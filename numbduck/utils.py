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


def _has_capi_symbols(lib):
    return hasattr(lib, "duckdb_open")


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
    lib_path = find_duckdb_shared_lib()
    lib = load_lib_path(lib_path)
    if _has_capi_symbols(lib):
        return lib
    # Python wheel missing C API symbols (macOS with duckdb >= 1.4.1)
    standalone = _find_standalone_libduckdb()
    if standalone:
        lib = load_lib_path(standalone)
        if _has_capi_symbols(lib):
            return lib
    if platform.system() != "Darwin":
        raise RuntimeError(
            "DuckDB C API symbols not found in the Python wheel. "
            "Set NUMBDUCK_LIBDUCKDB=/path/to/libduckdb.so"
        )
    downloaded = _download_libduckdb()
    return load_lib_path(downloaded)
