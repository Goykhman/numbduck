import duckdb
import os
import platform
import re

from ctypes import CDLL
from inspect import getfile
from numba.experimental.structref import register
from numba.core.types import StructRef


@register
class DuckdbResultTypeClass(StructRef):
    pass


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
    duckdb_shared_lib_path = find_duckdb_shared_lib()
    platform_ = platform.system()
    if platform_ in ("Darwin", "Linux"):
        from ctypes import RTLD_GLOBAL
        return CDLL(duckdb_shared_lib_path, mode=RTLD_GLOBAL)
    elif platform_ == "Windows":
        return CDLL(duckdb_shared_lib_path, winmode=0)
    else:
        raise RuntimeError(f"Platform {platform_} is not supported, yet.")
