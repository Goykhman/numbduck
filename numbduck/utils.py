import duckdb
import os
import re

from ctypes import CDLL, RTLD_GLOBAL
from inspect import getfile
from numba.experimental.structref import register
from numba.core.types import intp, StructRef
from numbox.utils.highlevel import make_structref
from numbox.utils.meminfo import structref_meminfo
from numpy import ndarray


@register
class DuckdbResultTypeClass(StructRef):
    pass


def find_duckdb_shared_lib():
    duckdb_dir = os.path.dirname(getfile(duckdb))
    duckdb_dir_files = next(iter(os.walk(duckdb_dir)))[2]
    shared_lib = [file_ for file_ in duckdb_dir_files if re.match(r"duckdb[\w\-.]*(.so|.dll|.dylib)", file_)]
    if len(shared_lib) != 1:
        raise RuntimeError(f"shared_lib = {shared_lib}, could not find unambiguous duck db shared library")
    return os.path.join(duckdb_dir, shared_lib[0])


def load_duckdb():
    duckdb_shared_lib_path = find_duckdb_shared_lib()
    return CDLL(duckdb_shared_lib_path, mode=RTLD_GLOBAL)


def make_duckdb_result(out_result: ndarray, duckdb_result_type_class=DuckdbResultTypeClass):
    from numbduck.ducklib import struct_by_pointer
    if not struct_by_pointer:
        return tuple(out_result), 0
    duckdb_result_type = make_structref("DuckdbResultType", {
        "f1": intp, "f2": intp, "f3": intp, "f4": intp, "f5": intp, "f6": intp
    }, duckdb_result_type_class)
    duckdb_result = duckdb_result_type(*out_result)
    _, data_p = structref_meminfo(duckdb_result)
    return duckdb_result, data_p
