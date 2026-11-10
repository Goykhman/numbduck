"""Zeroed numpy buffers for the handles and results that DuckDB's C API writes through out-parameters.

A C function such as ``duckdb_open`` takes the address at which it writes the handle that it creates. A destroy
function such as ``duckdb_close`` takes the address of the handle that it destroys, and sets the handle to NULL. Each
function here returns a new ``int64`` array of zeros: one element for a handle, and six for the 48-byte
``duckdb_result`` struct. Pass the array's address to the binding, as ``buffer.ctypes.data`` in Python or as numbox's
``array_data_p(buffer)`` in Python or jitted code, and read the handle back as ``buffer[0]``. Inside ``@njit``, numba
frees an array after its last use, and an address does not keep the array alive, so use the buffer again after the
call that writes or destroys through it. The functions compile at their first call, under numbduck's ``jit_options``.
"""
import numpy
from numba import njit

from numbduck.configurations import jit_options


@njit(**jit_options)
def allocate_buffer(sz: int):
    """Return a new ``int64`` array of ``sz`` zeros."""
    return numpy.zeros(sz, dtype=numpy.int64)


@njit(**jit_options)
def create_duckdb_connection():
    """Return a zeroed buffer for one ``duckdb_connection`` handle, which ``duckdb_connect`` writes.

    https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L469
    """
    return allocate_buffer(1)


@njit(**jit_options)
def create_duckdb_database():
    """Return a zeroed buffer for one ``duckdb_database`` handle, which ``duckdb_open`` writes.

    https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L464
    """
    return allocate_buffer(1)


@njit(**jit_options)
def create_duckdb_prepared_statement():
    """Return a zeroed buffer for one ``duckdb_prepared_statement`` handle, which ``duckdb_prepare`` writes.

    https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L480
    """
    return allocate_buffer(1)


@njit(**jit_options)
def create_duckdb_data_chunk():
    """Return a zeroed buffer for one ``duckdb_data_chunk`` handle, whose address ``duckdb_destroy_data_chunk`` takes.

    https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L527
    """
    return allocate_buffer(1)


@njit(**jit_options)
def create_duckdb_result():
    """Return a zeroed buffer for one ``duckdb_result`` of six 8-byte fields, as ``duckdb_query`` writes it.

    https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L454
    """
    return allocate_buffer(6)


@njit(**jit_options)
def create_duckdb_value():
    """Return a zeroed buffer for one ``duckdb_value`` handle, whose address ``duckdb_destroy_value`` takes.

    https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L531
    """
    return allocate_buffer(1)


@njit(**jit_options)
def create_duckdb_vector():
    """Return a zeroed buffer for one ``duckdb_vector`` handle.

    https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L395
    """
    return allocate_buffer(1)
