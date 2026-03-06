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
    shared_lib = [file_ for file_ in duckdb_dir_files if re.match(r"duckdb[\w.-]*\.(so|dll|dylib|pyd)", file_)]
    if len(shared_lib) != 1:
        raise RuntimeError(f"shared_lib = {shared_lib}, could not find unambiguous duckdb shared library")
    return os.path.join(duckdb_dir, shared_lib[0])


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
