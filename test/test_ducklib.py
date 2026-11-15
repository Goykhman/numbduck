"""Tests of numbduck's bindings of the DuckDB C API, of its libduckdb loader, and of the UDFs and UDAFs in examples/.

The tests call the bindings from Python and from @njit code. They open databases, run queries and prepared statements,
bind parameters of every type, create values and logical types, and register scalar and aggregate functions whose
callbacks numba compiles. A counting harness checks, in the tests it wraps, that every handle created is destroyed.
Other tests keep an aggregate's state in a numba structref or array and compare numba's counts of allocations and
frees; test_ducklib.md explains that design. Further tests cover numbduck's jit options and the connection pointer it
borrows from the duckdb Python package. A test runs in a child process when it needs other jit options, an allocator
that overwrites freed memory, a guard against a deadlock on the GIL, or a clean process-wide symbol scope. A test
whose behavior depends on the duckdb version or the platform checks the installed version or skips itself.
test_no_cache_location.py covers the imports where numba cannot write a cache for numbduck.
"""
import ast
import ctypes
import functools
import math
import os
import shutil
import subprocess
import sys

import duckdb
import numpy
import pytest
from numba import njit, cfunc, carray
from numba import types as nb_types
from numba.core.caching import FunctionCache, NullCache
from numba.core.errors import TypingError
from numba.experimental import structref
from numbox.core.bindings.call import _call_lib_func
from numbox.core.proxy.proxy import proxy_if_available
from numbox.core.vector.vector import vector_push
from numbox.utils.highlevel import make_structref
from numbox.utils.lowlevel import get_str_from_p_as_int, get_unicode_data_p
from numbox.utils.meminfo import (
    borrow_structref, export_meminfo, _incref_meminfo,
    release_meminfo, structref_meminfo
)

from numbduck import ducklib
from numbduck.configurations import get_jit_options
from numbduck.duckdb_utils import (
    create_duckdb_connection, create_duckdb_data_chunk,
    create_duckdb_database, create_duckdb_prepared_statement,
    create_duckdb_result
)
from numbduck.utils import complete_open_result, duckdb_major_version
from numbox.utils.lowlevel import _cast_int_to_void_p
from numbox.utils.lowlevel import array_data_p


class _HandleFamily:
    """A kind of DuckDB handle, named by the bindings that create one and the bindings that destroy one.

    ``acquires`` maps the name of each creating binding to a predicate. The predicate receives the call's arguments and
    result, and returns True if that call created a handle. ``releases`` names the destroying bindings.
    """

    def __init__(self, name, acquires, releases):
        self.name = name
        self.acquires = acquires
        self.releases = releases


def _acquired_by_return(args, result):
    return result != 0


def _acquired_by_last_arg(args, result):
    # duckdb_query fills a result only when the caller passes a non-null out pointer.
    return args[-1] != 0


RESULT_FAMILY = _HandleFamily(
    "result",
    {"duckdb_query": _acquired_by_last_arg},
    ("duckdb_destroy_result",),
)
CHUNK_FAMILY = _HandleFamily(
    "data_chunk",
    {"duckdb_fetch_chunk": _acquired_by_return},
    ("duckdb_destroy_data_chunk",),
)
VALUE_FAMILY = _HandleFamily(
    "value",
    {name: _acquired_by_return for name in (
        "duckdb_create_int64", "duckdb_create_bool", "duckdb_create_double"
    )},
    ("duckdb_destroy_value",),
)
LOGICAL_TYPE_FAMILY = _HandleFamily(
    "logical_type",
    {"duckdb_create_logical_type": _acquired_by_return},
    ("duckdb_destroy_logical_type",),
)
SCALAR_FUNCTION_FAMILY = _HandleFamily(
    "scalar_function",
    {"duckdb_create_scalar_function": _acquired_by_return},
    ("duckdb_destroy_scalar_function",),
)


class handle_balance:
    """A context manager that asserts on exit that each family destroyed as many handles as it created.

    DuckDB's C API has no function that reports its allocations, so the manager counts calls of the bindings instead.
    If a handle is never destroyed, or is destroyed twice, the two counts of its family differ. The manager replaces
    each named binding in ``ducklib`` with a counting wrapper, so it counts only calls made from Python through
    ``ducklib``. It restores every binding before it checks the counts, so a failed assertion never leaves ``ducklib``
    patched. If the body raises, the exception propagates, and the manager does not check the counts.
    """

    def __init__(self, *families):
        self._families = families
        self._saved = {}
        self._counts = {}

    def __enter__(self):
        for family in self._families:
            self._counts[family.name] = [0, 0]
            for name, predicate in family.acquires.items():
                self._wrap(name, family.name, 0, predicate)
            for name in family.releases:
                self._wrap(name, family.name, 1, None)
        return self

    def _wrap(self, name, family_name, slot, predicate):
        original = getattr(ducklib, name)
        self._saved[name] = original
        counts = self._counts[family_name]

        def wrapper(*args, **kwargs):
            result = original(*args, **kwargs)
            if predicate is None or predicate(args, result):
                counts[slot] += 1
            return result

        setattr(ducklib, name, wrapper)

    def __exit__(self, exc_type, exc, tb):
        for name, original in self._saved.items():
            setattr(ducklib, name, original)
        if exc_type is not None:
            return False
        for name, (acquired, released) in self._counts.items():
            assert acquired == released, \
                f"{name} handle imbalance: {acquired} acquired, {released} released"
        return False


def balanced(*families):
    """Return a decorator that runs a test inside ``handle_balance`` over the given families."""
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with handle_balance(*families):
                return func(*args, **kwargs)
        return wrapper
    return decorate


def aux_open_database(db_name_p_):
    duckdb_database = create_duckdb_database()
    duckdb_database_pp = duckdb_database.ctypes.data
    duckdb_open_rc = ducklib.duckdb_open(db_name_p_, duckdb_database_pp)
    assert duckdb_open_rc == ducklib.DuckDBSuccess, f"Failed to open duckdb, rc = {duckdb_open_rc}"
    return duckdb_database


def test_open_close_database():
    db_name_bytes = ctypes.c_char_p(":memory:".encode())
    db_name_p = ctypes.c_void_p.from_buffer(db_name_bytes).value
    duckdb_database = aux_open_database(db_name_p)
    duckdb_database_p = duckdb_database[0]
    assert duckdb_database_p != 0, f"Expected pointer to DB, got {duckdb_database_p}"
    duckdb_database_pp = duckdb_database.ctypes.data
    ducklib.duckdb_close(duckdb_database_pp)


def test_open_invalid_path():
    db_name_bytes = ctypes.c_char_p("/no/such/dir/db.duckdb".encode())
    db_name_p = ctypes.c_void_p.from_buffer(db_name_bytes).value
    duckdb_database = create_duckdb_database()
    duckdb_database_pp = duckdb_database.ctypes.data
    rc = ducklib.duckdb_open(db_name_p, duckdb_database_pp)
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError for invalid path, got {rc}"
    ducklib.duckdb_close(duckdb_database_pp)


def aux_connect_db():
    duckdb_database = aux_open_database(0)
    duckdb_database_p = duckdb_database[0]

    duckdb_connection = create_duckdb_connection()
    duckdb_connection_pp = duckdb_connection.ctypes.data
    duckdb_connect_rc = ducklib.duckdb_connect(duckdb_database_p, duckdb_connection_pp)
    assert duckdb_connect_rc == ducklib.DuckDBSuccess, duckdb_connect_rc
    return duckdb_database, duckdb_connection


def aux_close_db(duckdb_database, duckdb_connection):
    ducklib.duckdb_disconnect(duckdb_connection.ctypes.data)
    ducklib.duckdb_close(duckdb_database.ctypes.data)


def aux_destroy_data_chunk(chunk_p):
    """Destroy the data chunk ``chunk_p`` that ``duckdb_fetch_chunk`` returned.

    ``duckdb_destroy_data_chunk`` takes a pointer to the handle, so the helper stores the handle in a one-element buffer
    and passes the buffer's address.
    """
    buf = numpy.zeros(1, dtype=numpy.intp)
    buf[0] = chunk_p
    ducklib.duckdb_destroy_data_chunk(buf.ctypes.data)


def test_connect():
    duckdb_database, duckdb_connection = aux_connect_db()
    duckdb_database_p = duckdb_database[0]
    duckdb_connection_p = duckdb_connection[0]
    assert duckdb_database_p != 0, f"Expected pointer to DB, got {duckdb_database_p}"
    assert duckdb_connection_p != 0, f"Expected pointer to connection, got {duckdb_connection_p}"
    aux_close_db(duckdb_database, duckdb_connection)


def test_connect_invalid_database():
    duckdb_connection = create_duckdb_connection()
    duckdb_connection_pp = duckdb_connection.ctypes.data
    rc = ducklib.duckdb_connect(0, duckdb_connection_pp)
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError for null database, got {rc}"


def test_disconnect():
    duckdb_database, duckdb_connection = aux_connect_db()
    duckdb_connection_p = duckdb_connection[0]
    assert duckdb_connection_p != 0, f"Expected pointer to connection, got {duckdb_connection_p}"
    query_txt = "SELECT 1;"
    query_p = get_unicode_data_p(query_txt)
    rc = ducklib.duckdb_query(duckdb_connection_p, query_p, 0)
    assert rc == ducklib.DuckDBSuccess, f"Query before disconnect failed, rc = {rc}"
    duckdb_connection_pp = duckdb_connection.ctypes.data
    ducklib.duckdb_disconnect(duckdb_connection_pp)
    assert duckdb_connection[0] == 0, f"Expected null pointer after disconnect, got {duckdb_connection[0]}"
    ducklib.duckdb_close(duckdb_database.ctypes.data)


i_col = [3, 5, 7]
j_col = [4, 6, "NULL"]
arr_ty = ctypes.c_int32 * 3


def aux_query_1():
    """Create and select the table ``integers`` of ``i_col`` and ``j_col``; return the result, database and connection.

    It returns them in the order ``(out_result, duckdb_database, duckdb_connection)``. The caller destroys the result
    and closes the connection and the database. The steps follow DuckDB's example at
    https://duckdb.org/docs/2.0/clients/c/query#duckdb_fetch_chunk.
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    duckdb_connection_p = duckdb_connection[0]

    query_txt = "CREATE TABLE integers (i INTEGER, j INTEGER);"
    query_p = get_unicode_data_p(query_txt)
    duckdb_query_rc = ducklib.duckdb_query(duckdb_connection_p, query_p, 0)
    assert duckdb_query_rc == ducklib.DuckDBSuccess, duckdb_query_rc

    query_txt = f"INSERT INTO integers VALUES ({i_col[0]}, {j_col[0]}), ({i_col[1]}, {j_col[1]}), ({i_col[2]}, {j_col[2]});"  # noqa: E501
    query_p = get_unicode_data_p(query_txt)
    duckdb_query_rc = ducklib.duckdb_query(duckdb_connection_p, query_p, 0)
    assert duckdb_query_rc == ducklib.DuckDBSuccess, duckdb_query_rc

    query_txt = "SELECT * FROM integers;"
    query_p = get_unicode_data_p(query_txt)
    out_result = create_duckdb_result()
    out_result_p = out_result.ctypes.data
    duckdb_query_rc = ducklib.duckdb_query(duckdb_connection_p, query_p, out_result_p)
    assert duckdb_query_rc == ducklib.DuckDBSuccess, duckdb_query_rc

    return out_result, duckdb_database, duckdb_connection


@balanced(RESULT_FAMILY)
def test_query():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    print(f"out_result = {out_result}")
    num_of_columns = out_result[0]
    assert num_of_columns == 2, f"expected 'i', 'j', got {num_of_columns} columns"
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


def test_query_invalid_sql():
    duckdb_database, duckdb_connection = aux_connect_db()
    duckdb_connection_p = duckdb_connection[0]
    query_txt = "NOT VALID SQL;"
    query_p = get_unicode_data_p(query_txt)
    out_result = create_duckdb_result()
    out_result_p = out_result.ctypes.data
    rc = ducklib.duckdb_query(duckdb_connection_p, query_p, out_result_p)
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError for invalid SQL, got {rc}"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


@balanced(RESULT_FAMILY)
def test_duckdb_column_count_and_duckdb_row_count():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    out_result_p = out_result.ctypes.data
    num_of_cols = ducklib.duckdb_column_count(out_result_p)
    num_of_rows = ducklib.duckdb_row_count(out_result_p)
    assert num_of_cols == 2, num_of_cols
    assert num_of_rows == 3, num_of_rows
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


@balanced(RESULT_FAMILY)
def test_duckdb_destroy_result():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    out_result_p = out_result.ctypes.data
    num_of_cols = ducklib.duckdb_column_count(out_result_p)
    assert num_of_cols == 2, f"Expected valid result before destroy, got {num_of_cols} columns"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


def aux_get_data_vector():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    duckdb_result = tuple(out_result)
    data_chunk_p = ducklib.duckdb_fetch_chunk(duckdb_result)
    assert data_chunk_p, f"Expected pointer to data chunk, got {data_chunk_p}"

    i_vec_p = ducklib.duckdb_data_chunk_get_vector(data_chunk_p, 0)
    i_vec_data_p = ducklib.duckdb_vector_get_data(i_vec_p)
    i_arr = arr_ty.from_address(i_vec_data_p)
    assert all([i_arr_ == i_col_ for i_arr_, i_col_ in zip(i_arr, i_col)])
    return out_result, data_chunk_p, duckdb_database, duckdb_connection


@balanced(RESULT_FAMILY, CHUNK_FAMILY)
def test_duckdb_data_chunk_get_column_count():
    out_result, data_chunk_p, duckdb_database, duckdb_connection = aux_get_data_vector()
    col_count = ducklib.duckdb_data_chunk_get_column_count(data_chunk_p)
    assert col_count == 2, f"Expected 2 columns, got {col_count}"
    aux_destroy_data_chunk(data_chunk_p)
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


@balanced(RESULT_FAMILY, CHUNK_FAMILY)
def test_duckdb_data_chunk_get_size():
    out_result, data_chunk_p, duckdb_database, duckdb_connection = aux_get_data_vector()
    chunk_size = ducklib.duckdb_data_chunk_get_size(data_chunk_p)
    assert chunk_size == 3, f"Expected 3 rows in chunk, got {chunk_size}"
    aux_destroy_data_chunk(data_chunk_p)
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


@balanced(RESULT_FAMILY, CHUNK_FAMILY)
def test_duckdb_destroy_data_chunk():
    out_result, data_chunk_p, duckdb_database, duckdb_connection = aux_get_data_vector()
    assert data_chunk_p != 0, f"Expected valid chunk pointer, got {data_chunk_p}"
    chunk_size = ducklib.duckdb_data_chunk_get_size(data_chunk_p)
    assert chunk_size > 0, f"Expected rows in chunk before destroy, got {chunk_size}"
    data_chunk = create_duckdb_data_chunk()
    data_chunk[0] = data_chunk_p
    data_chunk_pp = data_chunk.ctypes.data
    ducklib.duckdb_destroy_data_chunk(data_chunk_pp)
    assert data_chunk[0] == 0, f"Expected null after destroy, got {data_chunk[0]}"
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


@balanced(RESULT_FAMILY, CHUNK_FAMILY)
def test_duckdb_fetch_chunk_exhausted():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    duckdb_result = tuple(out_result)
    chunk_p = ducklib.duckdb_fetch_chunk(duckdb_result)
    assert chunk_p != 0, "Expected first chunk, got null"
    aux_destroy_data_chunk(chunk_p)
    chunk_p = ducklib.duckdb_fetch_chunk(duckdb_result)
    assert chunk_p == 0, f"Expected null for exhausted result, got {chunk_p}"
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


@balanced(RESULT_FAMILY, CHUNK_FAMILY)
def test_duckdb_fetch_chunk_data_chunk_get_vector_get_data_vector():
    out_result, data_chunk_p, duckdb_database, duckdb_connection = aux_get_data_vector()
    assert data_chunk_p
    j_vec_p = ducklib.duckdb_data_chunk_get_vector(data_chunk_p, 1)
    j_vec_data_p = ducklib.duckdb_vector_get_data(j_vec_p)
    j_validity_p = ducklib.duckdb_vector_get_validity(j_vec_p)
    j_arr = arr_ty.from_address(j_vec_data_p)
    j_val = [ducklib.duckdb_validity_row_is_valid(j_validity_p, ind_) for ind_ in range(3)]
    assert j_val == [1, 1, 0]
    assert all([
        (j_arr_ == j_col_) if j_val_ else True
        for j_arr_, j_col_, j_val_ in zip(j_arr, j_col, j_val)
    ])
    aux_destroy_data_chunk(data_chunk_p)
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


def test_handle_balance_value_family():
    with handle_balance(VALUE_FAMILY):
        v_int = ducklib.duckdb_create_int64(7)
        v_bool = ducklib.duckdb_create_bool(1)
        v_double = ducklib.duckdb_create_double(1.5)
        assert v_int != 0 and v_bool != 0 and v_double != 0
        aux_destroy_value(v_int)
        aux_destroy_value(v_bool)
        aux_destroy_value(v_double)


def test_handle_balance_logical_type_family():
    with handle_balance(LOGICAL_TYPE_FAMILY):
        lt_int = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
        lt_big = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_BIGINT)
        assert lt_int != 0 and lt_big != 0
        for lt_p in (lt_int, lt_big):
            lt_buf = numpy.zeros(1, dtype=numpy.intp)
            lt_buf[0] = lt_p
            ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)


def test_handle_balance_detects_missing_destroy():
    leaked_p = 0
    with pytest.raises(AssertionError, match="value handle imbalance"):
        with handle_balance(VALUE_FAMILY):
            leaked_p = ducklib.duckdb_create_int64(11)
            assert leaked_p != 0
    aux_destroy_value(leaked_p)


def test_handle_balance_scalar_function_family():
    """Check that 50 scalar functions, each created and then destroyed, balance under ``handle_balance``.

    A function handle that is never destroyed would make the family's two counts differ.
    """
    with handle_balance(SCALAR_FUNCTION_FAMILY):
        for _ in range(50):
            fn_p = ducklib.duckdb_create_scalar_function()
            assert fn_p != 0
            fn_buf = numpy.zeros(1, dtype=numpy.intp)
            fn_buf[0] = fn_p
            ducklib.duckdb_destroy_scalar_function(fn_buf.ctypes.data)
            assert fn_buf[0] == 0


def test_destroy_zeroes_handle_slots():
    """Check that the destroy bindings of values, logical types, scalar functions and aggregate functions zero the slot.

    DuckDB zeroes the caller's slot for these handles, as it does for a database, a connection, a prepared statement and
    a data chunk. ``handle_balance`` counts the calls of a destroy binding but cannot detect whether DuckDB destroyed
    anything. A zeroed slot is direct evidence that DuckDB's destroy function ran.
    """
    buf = numpy.zeros(1, dtype=numpy.intp)

    buf[0] = ducklib.duckdb_create_int64(7)
    assert buf[0] != 0
    ducklib.duckdb_destroy_value(buf.ctypes.data)
    assert buf[0] == 0, "duckdb_destroy_value did not null the handle slot"

    buf[0] = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    assert buf[0] != 0
    ducklib.duckdb_destroy_logical_type(buf.ctypes.data)
    assert buf[0] == 0, "duckdb_destroy_logical_type did not null the slot"

    buf[0] = ducklib.duckdb_create_scalar_function()
    assert buf[0] != 0
    ducklib.duckdb_destroy_scalar_function(buf.ctypes.data)
    assert buf[0] == 0, "duckdb_destroy_scalar_function did not null the slot"

    buf[0] = ducklib.duckdb_create_aggregate_function()
    assert buf[0] != 0
    ducklib.duckdb_destroy_aggregate_function(buf.ctypes.data)
    assert buf[0] == 0, "duckdb_destroy_aggregate_function did not null the slot"


# --- Prepared Statements ---

def aux_prepare(connection_p, sql):
    """Prepare ``sql`` on the connection and return the statement's buffer and the state code.

    The caller destroys the statement with ``aux_destroy_prepared`` whether the prepare succeeded or not, as DuckDB
    requires.
    """
    query_p = get_unicode_data_p(sql)
    stmt = create_duckdb_prepared_statement()
    stmt_pp = stmt.ctypes.data
    rc = ducklib.duckdb_prepare(connection_p, query_p, stmt_pp)
    return stmt, rc


def aux_execute_prepared(stmt_p):
    """Execute the prepared statement ``stmt_p`` and return the result's buffer and its first data chunk.

    The helper asserts that the execution succeeds and that the result has a first chunk.
    """
    out_result = create_duckdb_result()
    out_result_p = out_result.ctypes.data
    rc = ducklib.duckdb_execute_prepared(stmt_p, out_result_p)
    assert rc == ducklib.DuckDBSuccess, f"Execute failed, rc = {rc}"
    duckdb_result = tuple(out_result)
    chunk_p = ducklib.duckdb_fetch_chunk(duckdb_result)
    assert chunk_p != 0, "Expected chunk"
    return out_result, chunk_p


def aux_read_column_data(chunk_p, col_idx):
    """Return the address of the data of column ``col_idx`` in the data chunk ``chunk_p``."""
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, col_idx)
    return ducklib.duckdb_vector_get_data(vec_p)


def aux_read_inline_string(data_p):
    """Return the ``duckdb_string_t`` entry at ``data_p`` as text: the bytes of ``aux_read_string_t``, decoded as UTF-8.

    Despite its name, the helper reads a string stored out of line as well as one stored inline. The layout is in
    https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L365.
    """
    return aux_read_string_t(data_p).decode()


def aux_read_string_t(data_p):
    """Return the bytes of the ``duckdb_string_t`` entry at ``data_p``.

    The entry starts with a 4-byte length. DuckDB stores a string of up to 12 bytes inline, in the 12 bytes that follow
    the length. For a longer string, it keeps the first 4 bytes after the length and a pointer to the whole string at
    ``data_p + 8``. The layout is in https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L365.
    """
    length = ctypes.c_uint32.from_address(data_p).value
    if length <= 12:
        raw = (ctypes.c_char * length).from_address(data_p + 4)
    else:
        out_p = ctypes.c_void_p.from_address(data_p + 8).value
        raw = (ctypes.c_char * length).from_address(out_p)
    return raw[:]


def aux_destroy_prepared(stmt):
    """Destroy the prepared statement whose handle the buffer ``stmt`` holds; DuckDB zeroes the slot."""
    ducklib.duckdb_destroy_prepare(stmt.ctypes.data)


def test_prepare_and_destroy():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT 1;")
    assert rc == ducklib.DuckDBSuccess, f"Expected DuckDBSuccess, got {rc}"
    assert stmt[0] != 0, f"Expected valid prepared statement, got {stmt[0]}"
    aux_destroy_prepared(stmt)
    assert stmt[0] == 0, f"Expected null after destroy, got {stmt[0]}"
    aux_close_db(duckdb_database, duckdb_connection)


def test_prepare_invalid_sql():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "NOT VALID SQL;")
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError, got {rc}"
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_nparams():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1, $2;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"
    nparams = ducklib.duckdb_nparams(stmt[0])
    assert nparams == 2, f"Expected 2 params, got {nparams}"
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_nparams_no_params():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT 1;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"
    nparams = ducklib.duckdb_nparams(stmt[0])
    assert nparams == 0, f"Expected 0 params, got {nparams}"
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_execute_prepared():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT 42 AS val;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    out_result_p = out_result.ctypes.data
    row_count = ducklib.duckdb_row_count(out_result_p)
    assert row_count == 1, f"Expected 1 row, got {row_count}"
    col_count = ducklib.duckdb_column_count(out_result_p)
    assert col_count == 1, f"Expected 1 column, got {col_count}"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_all_types():
    """Check that one statement binds an int32, an int64, a double, a varchar and a NULL, and returns each of them."""
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    sql = "SELECT $1::INTEGER, $2::BIGINT, $3::DOUBLE, $4::VARCHAR, $5::INTEGER;"  # noqa: E501
    stmt, rc = aux_prepare(connection_p, sql)
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"
    assert ducklib.duckdb_nparams(stmt[0]) == 5

    rc = ducklib.duckdb_bind_int32(stmt[0], 1, 99)
    assert rc == ducklib.DuckDBSuccess, f"Bind int32 failed, rc = {rc}"
    rc = ducklib.duckdb_bind_int64(stmt[0], 2, 2**40)
    assert rc == ducklib.DuckDBSuccess, f"Bind int64 failed, rc = {rc}"
    rc = ducklib.duckdb_bind_double(stmt[0], 3, 3.14)
    assert rc == ducklib.DuckDBSuccess, f"Bind double failed, rc = {rc}"
    val_bytes = ctypes.c_char_p(b"hello")
    val_p = ctypes.c_void_p.from_buffer(val_bytes).value
    rc = ducklib.duckdb_bind_varchar(stmt[0], 4, val_p)
    assert rc == ducklib.DuckDBSuccess, f"Bind varchar failed, rc = {rc}"
    rc = ducklib.duckdb_bind_null(stmt[0], 5)
    assert rc == ducklib.DuckDBSuccess, f"Bind null failed, rc = {rc}"

    out_result, chunk_p = aux_execute_prepared(stmt[0])

    # col 0: int32
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_int32 * 1).from_address(data_p)[0] == 99

    # col 1: int64
    data_p = aux_read_column_data(chunk_p, 1)
    assert (ctypes.c_int64 * 1).from_address(data_p)[0] == 2**40

    # col 2: double
    data_p = aux_read_column_data(chunk_p, 2)
    assert abs((ctypes.c_double * 1).from_address(data_p)[0] - 3.14) < 1e-10

    # col 3: varchar
    data_p = aux_read_column_data(chunk_p, 3)
    assert aux_read_inline_string(data_p) == "hello"

    # col 4: null
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 4)
    validity_p = ducklib.duckdb_vector_get_validity(vec_p)
    assert ducklib.duckdb_validity_row_is_valid(validity_p, 0) == 0

    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_execute_prepared_unbound_params():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::INTEGER;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"
    out_result = create_duckdb_result()
    out_result_p = out_result.ctypes.data
    rc = ducklib.duckdb_execute_prepared(stmt[0], out_result_p)
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError for unbound params, got {rc}"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


# --- Error Messages ---

def test_prepare_error_on_invalid_sql():
    """Check that ``duckdb_prepare_error`` returns a non-empty message after a failed prepare.

    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_prepare_error
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "NOT VALID SQL;")
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError, got {rc}"
    err_p = ducklib.duckdb_prepare_error(stmt[0])
    assert err_p != 0, "Expected non-null error pointer"
    err_str = ctypes.c_char_p(err_p).value.decode()
    assert len(err_str) > 0, "Expected non-empty error message"
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_prepare_error_on_valid_sql():
    """Check that ``duckdb_prepare_error`` returns a null pointer after a successful prepare.

    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_prepare_error
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT 1;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"
    err_p = ducklib.duckdb_prepare_error(stmt[0])
    assert err_p == 0, f"Expected null error pointer for valid SQL, got {err_p}"
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_result_error_on_invalid_query():
    """Check that ``duckdb_result_error`` returns a non-empty message after a failed query.

    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_result_error
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    query_p = get_unicode_data_p("NOT VALID SQL;")
    out_result = create_duckdb_result()
    out_result_p = out_result.ctypes.data
    rc = ducklib.duckdb_query(connection_p, query_p, out_result_p)
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError, got {rc}"
    err_p = ducklib.duckdb_result_error(out_result_p)
    assert err_p != 0, "Expected non-null error pointer"
    err_str = ctypes.c_char_p(err_p).value.decode()
    assert len(err_str) > 0, "Expected non-empty error message"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


def test_result_error_on_valid_query():
    """Check that ``duckdb_result_error`` returns a null pointer after a successful query.

    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_result_error
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    query_p = get_unicode_data_p("SELECT 1;")
    out_result = create_duckdb_result()
    out_result_p = out_result.ctypes.data
    rc = ducklib.duckdb_query(connection_p, query_p, out_result_p)
    assert rc == ducklib.DuckDBSuccess, f"Query failed, rc = {rc}"
    err_p = ducklib.duckdb_result_error(out_result_p)
    assert err_p == 0, f"Expected null error pointer for valid query, got {err_p}"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


# --- Binding booleans, floats, dates and timestamps ---

def test_bind_boolean():
    """Check that true and false, bound with ``duckdb_bind_boolean``, read back as the bytes 1 and 0.

    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_bind_boolean
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::BOOLEAN, $2::BOOLEAN;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"

    rc = ducklib.duckdb_bind_boolean(stmt[0], 1, 1)
    assert rc == ducklib.DuckDBSuccess, f"Bind boolean true failed, rc = {rc}"
    rc = ducklib.duckdb_bind_boolean(stmt[0], 2, 0)
    assert rc == ducklib.DuckDBSuccess, f"Bind boolean false failed, rc = {rc}"

    out_result, chunk_p = aux_execute_prepared(stmt[0])

    # col 0: true
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_int8 * 1).from_address(data_p)[0] == 1

    # col 1: false
    data_p = aux_read_column_data(chunk_p, 1)
    assert (ctypes.c_int8 * 1).from_address(data_p)[0] == 0

    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_float():
    """Check that a float bound with ``duckdb_bind_float`` reads back as the same 32-bit float.

    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_bind_float
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::FLOAT;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"

    rc = ducklib.duckdb_bind_float(stmt[0], 1, 2.5)
    assert rc == ducklib.DuckDBSuccess, f"Bind float failed, rc = {rc}"

    out_result, chunk_p = aux_execute_prepared(stmt[0])

    data_p = aux_read_column_data(chunk_p, 0)
    assert abs((ctypes.c_float * 1).from_address(data_p)[0] - 2.5) < 1e-6

    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_date():
    """Check that a date bound with ``duckdb_bind_date``, as days since 1970-01-01, reads back as the same day count.

    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_bind_date
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::DATE;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"

    # 20089 days after 1970-01-01 is 2025-01-01.
    days = 20089
    rc = ducklib.duckdb_bind_date(stmt[0], 1, days)
    assert rc == ducklib.DuckDBSuccess, f"Bind date failed, rc = {rc}"

    out_result, chunk_p = aux_execute_prepared(stmt[0])

    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_int32 * 1).from_address(data_p)[0] == days

    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_timestamp():
    """Check that a timestamp bound with ``duckdb_bind_timestamp``, in microseconds since 1970-01-01, reads back intact.

    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_bind_timestamp
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::TIMESTAMP;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"

    # 2025-01-01 00:00:00 = 1735689600 seconds = 1735689600000000 microseconds
    micros = 1735689600000000
    rc = ducklib.duckdb_bind_timestamp(stmt[0], 1, micros)
    assert rc == ducklib.DuckDBSuccess, f"Bind timestamp failed, rc = {rc}"

    out_result, chunk_p = aux_execute_prepared(stmt[0])

    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_int64 * 1).from_address(data_p)[0] == micros

    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


# --- Binding the other scalar types, strings and blobs ---

def test_bind_int8():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::TINYINT;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_int8(stmt[0], 1, 42)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_int8 * 1).from_address(data_p)[0] == 42
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_int16():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::SMALLINT;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_int16(stmt[0], 1, 1234)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_int16 * 1).from_address(data_p)[0] == 1234
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_uint8():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::UTINYINT;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_uint8(stmt[0], 1, 200)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_uint8 * 1).from_address(data_p)[0] == 200
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_uint16():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::USMALLINT;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_uint16(stmt[0], 1, 50000)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_uint16 * 1).from_address(data_p)[0] == 50000
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_uint32():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::UINTEGER;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_uint32(stmt[0], 1, 3000000000)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_uint32 * 1).from_address(data_p)[0] == 3000000000
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_uint64():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::UBIGINT;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_uint64(stmt[0], 1, 2**50)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_uint64 * 1).from_address(data_p)[0] == 2**50
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_time():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::TIME;")
    assert rc == ducklib.DuckDBSuccess
    micros = 45000000000  # 12:30:00
    rc = ducklib.duckdb_bind_time(stmt[0], 1, micros)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_int64 * 1).from_address(data_p)[0] == micros
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_timestamp_tz():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::TIMESTAMPTZ;")
    assert rc == ducklib.DuckDBSuccess
    micros = 1735689600000000
    rc = ducklib.duckdb_bind_timestamp_tz(stmt[0], 1, micros)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_int64 * 1).from_address(data_p)[0] == micros
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_varchar_length():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::VARCHAR;")
    assert rc == ducklib.DuckDBSuccess
    val_bytes = ctypes.c_char_p(b"hello world")
    val_p = ctypes.c_void_p.from_buffer(val_bytes).value
    rc = ducklib.duckdb_bind_varchar_length(stmt[0], 1, val_p, 5)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert aux_read_inline_string(data_p) == "hello"
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_blob():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::BLOB;")
    assert rc == ducklib.DuckDBSuccess
    blob_data = ctypes.create_string_buffer(b"\x00\x01\x02\x03", 4)
    blob_p = ctypes.cast(blob_data, ctypes.c_void_p).value
    rc = ducklib.duckdb_bind_blob(stmt[0], 1, blob_p, 4)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    # A BLOB has the duckdb_string_t layout too: a uint32 length, then up to 12 bytes stored inline.
    blob_len = ctypes.c_uint32.from_address(data_p).value
    assert blob_len == 4
    raw = (ctypes.c_char * blob_len).from_address(data_p + 4)
    assert raw[:] == b"\x00\x01\x02\x03"
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_varchar_long_out_of_line():
    """Check that a VARCHAR longer than 12 bytes binds and reads back through the out-of-line string layout.

    That layout holds a 4-byte length, a 4-byte prefix and an 8-byte pointer to the characters. The other string tests
    use strings of 12 bytes or fewer, which DuckDB stores inline.
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::VARCHAR;")
    assert rc == ducklib.DuckDBSuccess
    text = "abcdefghijklmnopqrstuvwxyz0123456789"
    assert len(text) > 12
    val_bytes = ctypes.c_char_p(text.encode())
    val_p = ctypes.c_void_p.from_buffer(val_bytes).value
    rc = ducklib.duckdb_bind_varchar(stmt[0], 1, val_p)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert ctypes.c_uint32.from_address(data_p).value == len(text)
    assert aux_read_string_t(data_p).decode() == text
    aux_destroy_data_chunk(chunk_p)
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_blob_long_out_of_line():
    """Check that a BLOB longer than 12 bytes reads back through the same out-of-line layout.

    ``test_bind_blob`` binds a BLOB of 4 bytes, which DuckDB stores inline.
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::BLOB;")
    assert rc == ducklib.DuckDBSuccess
    payload = bytes(range(20))
    assert len(payload) > 12
    blob_data = ctypes.create_string_buffer(payload, len(payload))
    blob_p = ctypes.cast(blob_data, ctypes.c_void_p).value
    rc = ducklib.duckdb_bind_blob(stmt[0], 1, blob_p, len(payload))
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert ctypes.c_uint32.from_address(data_p).value == len(payload)
    assert aux_read_string_t(data_p) == payload
    aux_destroy_data_chunk(chunk_p)
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_int8_negative():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::TINYINT;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_int8(stmt[0], 1, -42)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_int8 * 1).from_address(data_p)[0] == -42
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_int16_negative():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::SMALLINT;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_int16(stmt[0], 1, -1234)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    assert (ctypes.c_int16 * 1).from_address(data_p)[0] == -1234
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_parameter_index():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::INTEGER, $2::VARCHAR;")
    assert rc == ducklib.DuckDBSuccess
    idx_buf = numpy.zeros(1, dtype=numpy.uint64)
    name_bytes = ctypes.c_char_p(b"1")
    name_p = ctypes.c_void_p.from_buffer(name_bytes).value
    rc = ducklib.duckdb_bind_parameter_index(stmt[0], idx_buf.ctypes.data, name_p)
    assert rc == ducklib.DuckDBSuccess
    assert idx_buf[0] == 1
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


# --- Binding structs passed by value, and value handles ---

def test_bind_hugeint():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::HUGEINT;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_hugeint(stmt[0], 1, (42, 0))
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    lower = (ctypes.c_uint64 * 1).from_address(data_p)[0]
    upper = (ctypes.c_int64 * 1).from_address(data_p + 8)[0]
    assert lower == 42
    assert upper == 0
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_uhugeint():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::UHUGEINT;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_uhugeint(stmt[0], 1, (100, 0))
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    lower = (ctypes.c_uint64 * 1).from_address(data_p)[0]
    upper = (ctypes.c_uint64 * 1).from_address(data_p + 8)[0]
    assert lower == 100
    assert upper == 0
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_interval():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::INTERVAL;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_interval(stmt[0], 1, (1, 2, 3000000))
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    months = (ctypes.c_int32 * 1).from_address(data_p)[0]
    days = (ctypes.c_int32 * 1).from_address(data_p + 4)[0]
    micros = (ctypes.c_int64 * 1).from_address(data_p + 8)[0]
    assert months == 1
    assert days == 2
    assert micros == 3000000
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_decimal():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::DECIMAL(10, 2);")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_decimal(stmt[0], 1, (10, 2, 12345, 0))
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    # The bound duckdb_decimal is (width, scale, lower 64 bits, upper 64 bits). DuckDB stores a DECIMAL of width 10 to
    # 18 as an int64 that holds the unscaled value, so 12345 represents 123.45.
    stored = (ctypes.c_int64 * 1).from_address(data_p)[0]
    assert stored == 12345
    aux_destroy_data_chunk(chunk_p)
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_decimal_wide_upper64():
    """Check that a DECIMAL(38, 0) value bound with ``duckdb_bind_decimal`` reads back with both 64-bit halves intact.

    DuckDB stores a DECIMAL wider than 18 digits as an int128, and the bound value needs its upper half.
    ``test_bind_decimal`` binds a DECIMAL(10, 2), which DuckDB stores as an int64, so its upper half is always zero.
    """
    lower = 0xFEDCBA9876543210
    upper = 0x0123456789ABCDEF
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::DECIMAL(38, 0);")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_decimal(stmt[0], 1, (38, 0, lower, upper))
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    halves = (ctypes.c_uint64 * 2).from_address(data_p)
    assert halves[0] == lower, halves[0]
    assert halves[1] == upper, halves[1]
    aux_destroy_data_chunk(chunk_p)
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_value_success_roundtrip():
    """Check that a value handle bound with ``duckdb_bind_value`` at a valid index reads back as the value it holds.

    The invalid-index sweep is the only other test of this binding, and a binding that never bound a value would also
    return ``DuckDBError`` there. This test also checks that the binding passes the handle to DuckDB intact.
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::BIGINT;")
    assert rc == ducklib.DuckDBSuccess
    val_p = ducklib.duckdb_create_int64(-654321)
    assert val_p != 0
    rc = ducklib.duckdb_bind_value(stmt[0], 1, val_p)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    stored = (ctypes.c_int64 * 1).from_address(data_p)[0]
    assert stored == -654321
    aux_destroy_data_chunk(chunk_p)
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_value(val_p)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


_INVALID_INDEX_VARCHAR = ctypes.c_char_p(b"x")
_INVALID_INDEX_VARCHAR_P = ctypes.c_void_p.from_buffer(_INVALID_INDEX_VARCHAR).value
_INVALID_INDEX_BLOB = ctypes.create_string_buffer(b"\x00\x01\x02\x03", 4)
_INVALID_INDEX_BLOB_P = ctypes.cast(_INVALID_INDEX_BLOB, ctypes.c_void_p).value

# Every duckdb_bind_* binding that takes a parameter index, with the arguments that follow the index. The sweep below
# calls each one at the out-of-range index 999. It looks the binding up by the name in the first column, so each row
# names its binding once and calls the binding it names.
#
# The arguments are real values, not null pointers. DuckDB builds the bound value before it checks the index, at least
# in duckdb_bind_blob, and there a null data pointer aborts the process with std::logic_error ("construction from null
# is not valid") instead of returning DuckDBError.
#
# duckdb_bind_parameter_index is left out: its second argument is an out pointer that receives the index of a named
# parameter, not an index to bind at.
_INVALID_INDEX_BINDS = [
    ("duckdb_bind_boolean", (1,)),
    ("duckdb_bind_int8", (1,)),
    ("duckdb_bind_int16", (1,)),
    ("duckdb_bind_int32", (1,)),
    ("duckdb_bind_int64", (1,)),
    ("duckdb_bind_uint8", (1,)),
    ("duckdb_bind_uint16", (1,)),
    ("duckdb_bind_uint32", (1,)),
    ("duckdb_bind_uint64", (1,)),
    ("duckdb_bind_float", (1.0,)),
    ("duckdb_bind_double", (1.0,)),
    ("duckdb_bind_date", (0,)),
    ("duckdb_bind_time", (0,)),
    ("duckdb_bind_timestamp", (0,)),
    ("duckdb_bind_timestamp_tz", (0,)),
    ("duckdb_bind_null", ()),
    ("duckdb_bind_varchar", (_INVALID_INDEX_VARCHAR_P,)),
    ("duckdb_bind_varchar_length", (_INVALID_INDEX_VARCHAR_P, 1)),
    ("duckdb_bind_blob", (_INVALID_INDEX_BLOB_P, 4)),
    ("duckdb_bind_hugeint", ((42, 0),)),
    ("duckdb_bind_uhugeint", ((100, 0),)),
    ("duckdb_bind_interval", ((1, 2, 3),)),
    ("duckdb_bind_decimal", ((10, 2, 12345, 0),)),
    # duckdb_bind_value takes a value handle, which aux_bind_value_out_of_range creates and destroys around the call.
    # None sends the row to that helper.
    ("duckdb_bind_value", None),
]


def aux_bind_value_out_of_range(stmt_p):
    """Call ``duckdb_bind_value`` at the out-of-range index 999 with a real value handle, and return the state code.

    The helper creates an int64 value for the call and destroys it afterward. A null handle would also return
    ``DuckDBError`` at an out-of-range index, but it crashes the process at a valid one, so the sweep does not rely on
    it.
    """
    val_p = ducklib.duckdb_create_int64(1)
    assert val_p != 0
    try:
        return ducklib.duckdb_bind_value(stmt_p, 999, val_p)
    finally:
        aux_destroy_value(val_p)


@pytest.mark.parametrize("name,args", _INVALID_INDEX_BINDS,
                         ids=[entry[0] for entry in _INVALID_INDEX_BINDS])
def test_bind_family_invalid_param_index(name, args):
    """Check that each ``duckdb_bind_*`` binding that takes an index returns ``DuckDBError`` for an index out of range.

    None of them crashes, including the four that pass a struct by value (hugeint, uhugeint, interval and decimal), so
    the table omits none of them.
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::INTEGER;")
    assert rc == ducklib.DuckDBSuccess
    if args is None:
        rc = aux_bind_value_out_of_range(stmt[0])
    else:
        rc = getattr(ducklib, name)(stmt[0], 999, *args)
    assert rc == ducklib.DuckDBError, f"{name}: expected DuckDBError, got {rc}"
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_invalid_param_index_sweep_covers_every_indexed_bind():
    """Check that ``_INVALID_INDEX_BINDS`` names exactly the bindings in ``ducklib`` that bind a parameter at an index.

    The table is written by hand, so without this test the sweep would not call a binding added to ``ducklib`` later.
    The sweep calls each binding by the name in the table, so equal sets of names mean that the sweep calls every such
    binding, not only that the table lists it. ``duckdb_bind_parameter_index`` is left out, because it looks up the
    index of a named parameter instead of binding a value at an index.

    The test does not filter out the bindings whose C symbol the loaded libduckdb lacks. Every ``duckdb_bind_*`` binding
    is a plain ``@proxy`` binding, and a missing symbol makes the import of ``ducklib`` fail instead of producing a stub
    that can be called. numbox's ``proxy_if_available`` turns a version-gated binding into such a stub when the symbol
    is missing. A filter would remove that binding from both sets, and the sets would still be equal. If one is added,
    this test fails until the table lists it with a check of its own for the symbol.
    """
    swept = {name for name, _ in _INVALID_INDEX_BINDS}
    expected = {
        name for name in dir(ducklib)
        if name.startswith("duckdb_bind_")
        and name in ducklib.signatures
        and name != "duckdb_bind_parameter_index"
    }
    assert swept == expected, (
        f"uncovered: {sorted(expected - swept)}, stale: {sorted(swept - expected)}")


# --- Result Metadata ---

def test_column_name():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    out_result_p = out_result.ctypes.data
    name_p = ducklib.duckdb_column_name(out_result_p, 0)
    assert name_p != 0
    name = ctypes.c_char_p(name_p).value.decode()
    assert name == "i"
    name_p = ducklib.duckdb_column_name(out_result_p, 1)
    name = ctypes.c_char_p(name_p).value.decode()
    assert name == "j"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


def test_column_type():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    out_result_p = out_result.ctypes.data
    col_type = ducklib.duckdb_column_type(out_result_p, 0)
    assert col_type == ducklib.DUCKDB_TYPE_INTEGER
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


def test_column_logical_type():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    out_result_p = out_result.ctypes.data
    logical_type_p = ducklib.duckdb_column_logical_type(out_result_p, 0)
    assert logical_type_p != 0, "Expected valid logical type pointer"
    assert ducklib.duckdb_get_type_id(logical_type_p) == ducklib.DUCKDB_TYPE_INTEGER
    lt_buf = numpy.zeros(1, dtype=numpy.intp)
    lt_buf[0] = logical_type_p
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


def test_rows_changed():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    query_p = get_unicode_data_p("CREATE TABLE rc_test (x INTEGER);")
    rc = ducklib.duckdb_query(connection_p, query_p, 0)
    assert rc == ducklib.DuckDBSuccess, f"CREATE TABLE failed, rc={rc}"
    query_p = get_unicode_data_p("INSERT INTO rc_test VALUES (1), (2), (3);")
    out_result = create_duckdb_result()
    out_result_p = out_result.ctypes.data
    rc = ducklib.duckdb_query(connection_p, query_p, out_result_p)
    assert rc == ducklib.DuckDBSuccess
    changed = ducklib.duckdb_rows_changed(out_result_p)
    assert changed == 3, f"Expected 3 rows changed, got {changed}"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


def test_result_error_type_on_success():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    out_result_p = out_result.ctypes.data
    # DUCKDB_ERROR_INVALID = 0 (no error)
    error_type = ducklib.duckdb_result_error_type(out_result_p)
    assert error_type == 0, f"Expected DUCKDB_ERROR_INVALID (0), got {error_type}"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


def test_result_return_type():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    result_tup = tuple(out_result)
    # DUCKDB_RESULT_TYPE_QUERY_RESULT = 3
    ret_type = ducklib.duckdb_result_return_type(result_tup)
    assert ret_type == 3, f"Expected DUCKDB_RESULT_TYPE_QUERY_RESULT (3), got {ret_type}"
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


def test_result_statement_type():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    result_tup = tuple(out_result)
    # DUCKDB_STATEMENT_TYPE_SELECT = 1
    stmt_type = ducklib.duckdb_result_statement_type(result_tup)
    assert stmt_type == 1, f"Expected SELECT (1), got {stmt_type}"
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


_BYVAL_ROUNDTRIP_SCRIPT = """
import ctypes

import numbduck.ducklib as ducklib
from numbduck.duckdb_utils import (
    create_duckdb_connection, create_duckdb_database, create_duckdb_result,
)
from numbox.utils.lowlevel import get_unicode_data_p

db = create_duckdb_database()
assert ducklib.duckdb_open(0, db.ctypes.data) == ducklib.DuckDBSuccess
conn = create_duckdb_connection()
assert ducklib.duckdb_connect(db[0], conn.ctypes.data) == ducklib.DuckDBSuccess
conn_p = conn[0]


def run(sql):
    return ducklib.duckdb_query(conn_p, get_unicode_data_p(sql), 0)


assert run("CREATE TABLE t (i INTEGER);") == ducklib.DuckDBSuccess
assert run("INSERT INTO t VALUES (3), (5), (7);") == ducklib.DuckDBSuccess

result = create_duckdb_result()
rc = ducklib.duckdb_query(
    conn_p, get_unicode_data_p("SELECT i FROM t ORDER BY i;"), result.ctypes.data)
assert rc == ducklib.DuckDBSuccess
result_tup = tuple(result)

assert ducklib.duckdb_result_return_type(result_tup) == 3
assert ducklib.duckdb_result_statement_type(result_tup) == 1

chunk_p = ducklib.duckdb_fetch_chunk(result_tup)
assert chunk_p != 0
assert ducklib.duckdb_data_chunk_get_size(chunk_p) == 3
vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
data_p = ducklib.duckdb_vector_get_data(vec_p)
arr = (ctypes.c_int32 * 3).from_address(data_p)
assert list(arr) == [3, 5, 7], list(arr)

chunk_buf = (ctypes.c_int64 * 1)(chunk_p)
ducklib.duckdb_destroy_data_chunk(ctypes.addressof(chunk_buf))
ducklib.duckdb_destroy_result(result.ctypes.data)
ducklib.duckdb_disconnect(conn.ctypes.data)
ducklib.duckdb_close(db.ctypes.data)
print("BYVAL_ROUNDTRIP_OK")
"""


def test_fetch_chunk_byval_roundtrip_low_opt():
    """Check that the bindings that take a ``duckdb_result`` by value work when numba compiles them unoptimized.

    ``duckdb_fetch_chunk``, ``duckdb_result_return_type`` and ``duckdb_result_statement_type`` take the 48-byte struct
    by value. On SysV x86-64, numbox passes a struct over 16 bytes as a copy on the stack, marked ``byval`` as the ABI
    requires. With optimization on, a wrong lowering could still pass if the optimized code happened to leave the value
    where C reads it. ``_dbg_optnone`` turns the optimizer off, so the test passes only if the lowering itself is right.

    The test runs in a child process, because numbduck reads ``NUMBDUCK_JIT_OPTIONS`` once, when it is imported, and
    compiles every binding with those options.
    """
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["NUMBDUCK_JIT_OPTIONS"] = '{"cache": false, "_dbg_optnone": true}'
    proc = subprocess.run(
        [sys.executable, "-c", _BYVAL_ROUNDTRIP_SCRIPT],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "BYVAL_ROUNDTRIP_OK" in proc.stdout, proc.stdout


def test_get_jit_options_rejects_non_object(monkeypatch):
    monkeypatch.setenv("NUMBDUCK_JIT_OPTIONS", '["cache"]')
    with pytest.raises(ValueError, match="NUMBDUCK_JIT_OPTIONS"):
        get_jit_options()


def test_get_jit_options_rejects_unknown_key(monkeypatch):
    monkeypatch.setenv("NUMBDUCK_JIT_OPTIONS", '{"cache": true, "foobar": 1}')
    with pytest.raises(ValueError, match="NUMBDUCK_JIT_OPTIONS") as exc_info:
        get_jit_options()
    assert "foobar" in str(exc_info.value)


def test_get_jit_options_accepts_valid_object(monkeypatch):
    monkeypatch.setenv("NUMBDUCK_JIT_OPTIONS", '{"cache": false, "_dbg_optnone": true}')
    assert get_jit_options() == {"cache": False, "nogil": True, "_dbg_optnone": True}


def test_get_jit_options_defaults_release_the_gil(monkeypatch):
    """Check that ``nogil`` is on by default and stays on when ``NUMBDUCK_JIT_OPTIONS`` sets other keys.

    The variable overrides single defaults instead of replacing them all.
    """
    monkeypatch.delenv("NUMBDUCK_JIT_OPTIONS", raising=False)
    assert get_jit_options() == {"cache": True, "nogil": True}
    monkeypatch.setenv("NUMBDUCK_JIT_OPTIONS", '{"cache": false}')
    assert get_jit_options() == {"cache": False, "nogil": True}
    monkeypatch.setenv("NUMBDUCK_JIT_OPTIONS", '{"nogil": true}')
    assert get_jit_options() == {"cache": True, "nogil": True}


def test_get_jit_options_caches_only_the_default_compile_options(monkeypatch):
    """Check that only the default compile options use numba's disk cache.

    numba's cache key does not include the compile options, so code cached under one set of options would load in a
    later process that runs under another. Any other option turns the cache off. If ``NUMBDUCK_JIT_OPTIONS`` also sets
    ``"cache": true``, a ``RuntimeWarning`` says why the cache is off.
    """
    monkeypatch.setenv("NUMBDUCK_JIT_OPTIONS", '{"nogil": false}')
    assert get_jit_options() == {"cache": False, "nogil": False}
    monkeypatch.setenv("NUMBDUCK_JIT_OPTIONS", '{"fastmath": true}')
    assert get_jit_options() == {"cache": False, "nogil": True, "fastmath": True}
    monkeypatch.setenv("NUMBDUCK_JIT_OPTIONS", '{"cache": true, "fastmath": true}')
    with pytest.warns(RuntimeWarning, match="NUMBDUCK_JIT_OPTIONS"):
        assert get_jit_options() == {"cache": False, "nogil": True, "fastmath": True}


_NON_DEFAULT_WRITER_SCRIPT = r"""
from numbduck.duckdb_utils import create_duckdb_result

create_duckdb_result()
"""


def test_non_default_compile_options_leave_the_disk_cache_untouched(tmp_path):
    """Check that a process compiling under non-default options writes nothing to numba's disk cache.

    A later process would load those entries regardless of its own options. The child process has a cache directory of
    its own, so the test can list what the child wrote.
    """
    cache_dir = tmp_path / "numba-cache"
    cache_dir.mkdir()
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env["NUMBA_CACHE_DIR"] = str(cache_dir)
    env["NUMBDUCK_JIT_OPTIONS"] = '{"cache": true, "fastmath": true}'
    proc = subprocess.run(
        [sys.executable, "-c", _NON_DEFAULT_WRITER_SCRIPT],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    written = [os.path.join(root, name) for root, _, names in os.walk(cache_dir) for name in names]
    assert written == [], written


# The child sets two threads. With one thread, DuckDB runs the whole query on the calling thread, which already holds
# the GIL, so the UDF runs and nothing deadlocks, whether or not the binding releases the GIL. With two threads, DuckDB
# also uses a worker thread on every host, regardless of its number of cores. The 250000 rows fill three row groups,
# so the workers scan some of them. The filter passes 1 row in 64 to the UDF, because a Python UDF costs tens of
# microseconds per row on duckdb 1.x without pandas.
_GIL_WORKER_SCRIPT = r"""
import duckdb
from numbox.utils.lowlevel import get_unicode_data_p

from numbduck import ducklib
from numbduck.duckdb_utils import create_duckdb_result
from numbduck.pybridge import extract_connection_ptr

conn = duckdb.connect()
conn.execute("SET threads = 2")
conn.execute("SET enable_progress_bar = false")
conn.execute("CREATE TABLE t AS SELECT range::DOUBLE AS x FROM range(250000)")
conn.create_function("plus_one", lambda x: x + 1.0, ["DOUBLE"], "DOUBLE")
conn_p = extract_connection_ptr(conn)
result = create_duckdb_result()
sql = "SELECT sum(plus_one(x)) FROM t WHERE x % 64 = 0"
rc = ducklib.duckdb_query(conn_p, get_unicode_data_p(sql), result.ctypes.data)
ducklib.duckdb_destroy_result(result.ctypes.data)
print("QUERY_RC", rc)
"""


def test_binding_called_from_python_lets_duckdb_workers_take_the_gil():
    """Check that ``duckdb_query``, called from Python, releases the GIL while DuckDB runs the query.

    DuckDB runs a query on its worker threads. In this test, a worker needs the GIL to run a Python UDF registered on
    the same connection. Such a worker cannot run while the calling thread holds the GIL and waits for the workers, and
    the query can hang. The query runs in a child process, under the default jit options, so a deadlock shows as a
    timeout instead of a hung test session. On duckdb 1.2.2, the query sometimes finishes even when the binding keeps
    the GIL, so on that version the test can pass without detecting that defect.
    """
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.pop("NUMBDUCK_JIT_OPTIONS", None)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _GIL_WORKER_SCRIPT],
            env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("duckdb_query called from Python never returned: the binding kept the GIL")
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "QUERY_RC 0" in proc.stdout, proc.stdout


# --- Bindings called from @njit code ---
# The jitted functions below pass string literals to numbox's get_unicode_data_p. From numbox 0.5.6 on, it reads the
# data pointer from numba's string structure itself, not through the string's NRT meminfo, which a literal lacks.

def test_jit_create_duckdb_database():
    @njit
    def _jit_create():
        db = create_duckdb_database()
        return db.shape[0], db[0]
    size, val = _jit_create()
    assert size == 1
    assert val == 0


def test_array_data_p():
    for dtype in [numpy.int64, numpy.float64, numpy.uint8]:
        arr = numpy.zeros(1, dtype=dtype)
        assert array_data_p(arr) == arr.ctypes.data


# In the jitted lifecycle functions below, each buffer is read once more after the call that destroys its handle.
# numba frees an array immediately after its last use, and the address that array_data_p returns does not keep the
# array alive. If array_data_p(buf) were the last use of a buffer, numba would free the buffer before the destroy call
# read the handle from it and wrote NULL back. The read afterward keeps the buffer alive across the call and checks
# that the destroy wrote NULL.

@njit
def jit_open_close():
    db = create_duckdb_database()
    rc = ducklib.duckdb_open(
        get_unicode_data_p(':memory:'), array_data_p(db))
    db_p = db[0]
    ducklib.duckdb_close(array_data_p(db))
    return rc, db_p, db[0]


def test_jit_open_close_database():
    rc, db_p, db_after = jit_open_close()
    assert rc == ducklib.DuckDBSuccess, f"open failed, rc={rc}"
    assert db_p != 0, f"expected valid pointer, got {db_p}"
    assert db_after == 0, f"expected null after close, got {db_after}"


@njit
def jit_connect_query_disconnect():
    db = create_duckdb_database()
    conn = create_duckdb_connection()

    open_rc = ducklib.duckdb_open(0, array_data_p(db))
    db_p = db[0]

    connect_rc = ducklib.duckdb_connect(db_p, array_data_p(conn))
    conn_p = conn[0]

    query_rc = ducklib.duckdb_query(
        conn_p, get_unicode_data_p('SELECT 42;'), 0)

    ducklib.duckdb_disconnect(array_data_p(conn))
    conn_after = conn[0]
    ducklib.duckdb_close(array_data_p(db))
    return open_rc, db_p, connect_rc, conn_p, query_rc, conn_after, db[0]


def test_jit_connect_query_disconnect():
    open_rc, db_p, connect_rc, conn_p, query_rc, conn_after, db_after = jit_connect_query_disconnect()
    assert open_rc == ducklib.DuckDBSuccess, f"open failed, rc={open_rc}"
    assert db_p != 0, f"expected valid db pointer, got {db_p}"
    assert connect_rc == ducklib.DuckDBSuccess, f"connect failed, rc={connect_rc}"
    assert conn_p != 0, f"expected valid connection pointer, got {conn_p}"
    assert query_rc == ducklib.DuckDBSuccess, f"query failed, rc={query_rc}"
    assert conn_after == 0, f"expected null after disconnect, got {conn_after}"
    assert db_after == 0, f"expected null after close, got {db_after}"


@njit
def jit_query_invalid_sql():
    db = create_duckdb_database()
    conn = create_duckdb_connection()
    open_rc = ducklib.duckdb_open(0, array_data_p(db))
    connect_rc = ducklib.duckdb_connect(db[0], array_data_p(conn))
    conn_p = conn[0]
    out = create_duckdb_result()
    rc = ducklib.duckdb_query(
        conn_p, get_unicode_data_p('NOT VALID SQL;'), array_data_p(out))
    detected = 0
    if rc == ducklib.DuckDBError:
        detected = 1
    ducklib.duckdb_destroy_result(array_data_p(out))
    ducklib.duckdb_disconnect(array_data_p(conn))
    ducklib.duckdb_close(array_data_p(db))
    left = numpy.count_nonzero(out) + numpy.count_nonzero(conn) + numpy.count_nonzero(db)
    return open_rc, connect_rc, conn_p, rc, detected, left


def test_jit_query_invalid_sql():
    """Check that jitted code receives ``DuckDBError`` from ``duckdb_query`` and can branch on it.

    The other error tests call the bindings from Python, through numba's dispatcher; this one makes the call and the
    comparison inside compiled code. It also checks that the destroy calls zero every handle.

    The error comes from invalid SQL on an open connection, and the test asserts that the open and the connect
    succeeded. The error cannot come from a null connection: ``duckdb_query`` checks neither of its arguments for NULL,
    so it crashes on a null connection instead of returning ``DuckDBError``. A crash inside compiled
    code ends the test session without a traceback.
    """
    open_rc, connect_rc, conn_p, rc, detected, left = jit_query_invalid_sql()
    assert open_rc == ducklib.DuckDBSuccess, f"open failed, rc={open_rc}"
    assert connect_rc == ducklib.DuckDBSuccess, f"connect failed, rc={connect_rc}"
    assert conn_p != 0, "expected a valid connection pointer"
    assert rc == ducklib.DuckDBError, f"expected DuckDBError, got {rc}"
    assert detected == 1, "in-JIT DuckDBError comparison did not observe the error"
    assert left == 0, f"{left} handle words not nulled by the destroy calls"


# --- JIT: Prepared Statements ---

@njit
def jit_prepare_bind_execute():
    db = create_duckdb_database()
    conn = create_duckdb_connection()
    stmt = create_duckdb_prepared_statement()

    open_rc = ducklib.duckdb_open(0, array_data_p(db))
    connect_rc = ducklib.duckdb_connect(db[0], array_data_p(conn))
    conn_p = conn[0]

    # prepare a statement with four parameters
    prepare_rc = ducklib.duckdb_prepare(
        conn_p,
        get_unicode_data_p(
            'SELECT $1::INTEGER, $2::BIGINT, $3::DOUBLE, $4::INTEGER;'),
        array_data_p(stmt))
    stmt_p = stmt[0]
    nparams = ducklib.duckdb_nparams(stmt_p)

    # bind values
    bind1_rc = ducklib.duckdb_bind_int32(stmt_p, numpy.uint64(1), numpy.int32(99))
    bind2_rc = ducklib.duckdb_bind_int64(stmt_p, numpy.uint64(2), numpy.int64(2**40))
    bind3_rc = ducklib.duckdb_bind_double(stmt_p, numpy.uint64(3), numpy.float64(3.14))
    bind4_rc = ducklib.duckdb_bind_null(stmt_p, numpy.uint64(4))

    # execute
    result = create_duckdb_result()
    exec_rc = ducklib.duckdb_execute_prepared(stmt_p, array_data_p(result))

    # fetch chunk and read back values
    # duckdb_fetch_chunk takes the duckdb_result struct by value, as a tuple of its six words.
    result_tup = (result[0], result[1], result[2],
                  result[3], result[4], result[5])
    chunk_p = ducklib.duckdb_fetch_chunk(result_tup)
    chunk_size = ducklib.duckdb_data_chunk_get_size(chunk_p)

    # col 0: int32
    v0_p = ducklib.duckdb_vector_get_data(
        ducklib.duckdb_data_chunk_get_vector(chunk_p, 0))
    vp0 = _cast_int_to_void_p(v0_p)
    col0 = carray(vp0, (chunk_size,), dtype=numpy.int32)[0]

    # col 1: int64
    v1_p = ducklib.duckdb_vector_get_data(
        ducklib.duckdb_data_chunk_get_vector(chunk_p, 1))
    vp1 = _cast_int_to_void_p(v1_p)
    col1 = carray(vp1, (chunk_size,), dtype=numpy.int64)[0]

    # col 2: double
    v2_p = ducklib.duckdb_vector_get_data(
        ducklib.duckdb_data_chunk_get_vector(chunk_p, 2))
    vp2 = _cast_int_to_void_p(v2_p)
    col2 = carray(vp2, (chunk_size,), dtype=numpy.float64)[0]

    # col 3: null check
    v3_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 3)
    v3_validity_p = ducklib.duckdb_vector_get_validity(v3_p)
    col3_valid = ducklib.duckdb_validity_row_is_valid(v3_validity_p, 0)

    # cleanup (reverse order)
    chunk_buf = create_duckdb_data_chunk()
    chunk_buf[0] = chunk_p
    ducklib.duckdb_destroy_data_chunk(array_data_p(chunk_buf))
    ducklib.duckdb_destroy_result(array_data_p(result))
    ducklib.duckdb_destroy_prepare(array_data_p(stmt))
    ducklib.duckdb_disconnect(array_data_p(conn))
    ducklib.duckdb_close(array_data_p(db))
    left = (numpy.count_nonzero(chunk_buf) + numpy.count_nonzero(result) + numpy.count_nonzero(stmt)
            + numpy.count_nonzero(conn) + numpy.count_nonzero(db))

    return (open_rc, connect_rc, prepare_rc, nparams,
            bind1_rc, bind2_rc, bind3_rc, bind4_rc, exec_rc,
            chunk_size, col0, col1, col2, col3_valid, left)


def test_jit_prepare_bind_execute():
    """Check that jitted code can prepare a statement, bind four parameters, execute it and read the values back.

    The test also checks that the destroy calls zero every handle.

    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_prepare
    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_nparams
    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_bind_int32
    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_bind_int64
    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_bind_double
    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_bind_null
    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_execute_prepared
    https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_destroy_prepare
    """
    (open_rc, connect_rc, prepare_rc, nparams,
     bind1_rc, bind2_rc, bind3_rc, bind4_rc, exec_rc,
     chunk_size, col0, col1, col2, col3_valid, left) = jit_prepare_bind_execute()
    assert open_rc == ducklib.DuckDBSuccess, f"open failed, rc={open_rc}"
    assert connect_rc == ducklib.DuckDBSuccess, f"connect failed, rc={connect_rc}"
    assert prepare_rc == ducklib.DuckDBSuccess, f"prepare failed, rc={prepare_rc}"
    assert nparams == 4, f"expected 4 params, got {nparams}"
    assert bind1_rc == ducklib.DuckDBSuccess, f"bind int32 failed, rc={bind1_rc}"
    assert bind2_rc == ducklib.DuckDBSuccess, f"bind int64 failed, rc={bind2_rc}"
    assert bind3_rc == ducklib.DuckDBSuccess, f"bind double failed, rc={bind3_rc}"
    assert bind4_rc == ducklib.DuckDBSuccess, f"bind null failed, rc={bind4_rc}"
    assert exec_rc == ducklib.DuckDBSuccess, f"execute failed, rc={exec_rc}"
    assert chunk_size == 1, f"expected 1 row, got {chunk_size}"
    assert col0 == 99, f"col0: expected 99, got {col0}"
    assert col1 == 2**40, f"col1: expected 2^40, got {col1}"
    assert abs(col2 - 3.14) < 1e-10, f"col2: expected 3.14, got {col2}"
    assert col3_valid == 0, f"col3: expected NULL, validity={col3_valid}"
    assert left == 0, f"{left} handle words not nulled by the destroy calls"


_POISON_FREE_C = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <malloc.h>
#include <string.h>

static void (*real_free)(void *) = 0;

void free(void *p) {
    if (!real_free) real_free = (void (*)(void *))dlsym(RTLD_NEXT, "free");
    if (p) {
        size_t n = malloc_usable_size(p);
        if (n > 0 && n < 4096) memset(p, 0xDE, n);
    }
    real_free(p);
}
"""

_FREED_BYTES_SCRIPT = r"""
import ctypes
import numpy
from numba import njit

@njit
def fill_and_drop():
    a = numpy.empty(8, numpy.int64)
    for i in range(8):
        a[i] = 0x1111111111111111
    return a.ctypes.data

addr = fill_and_drop()
print("FREED-BYTES", bytes((ctypes.c_ubyte * 16).from_address(addr + 32)).hex())
"""

_BUFFER_LIFETIME_TESTS = [
    "test_jit_open_close_database",
    "test_jit_connect_query_disconnect",
    "test_jit_query_invalid_sql",
    "test_jit_prepare_bind_execute",
    "test_online_scoring_missing_key_raises",
    "test_online_scoring_null_feature_raises",
]


def _poison_on_free(tmp_path):
    """Return the environment variables under which the C allocator overwrites memory as it frees it.

    On macOS, ``MallocScribble`` makes libmalloc write 0x55 over every block it frees, and ``MallocZeroOnFree=0`` stops
    it from zeroing the block, which a recent macOS does by default. On Linux, the helper compiles ``_POISON_FREE_C``
    with gcc into ``tmp_path`` and preloads it with ``LD_PRELOAD``. That ``free`` writes 0xDE over a block of fewer
    than 4096 usable bytes before it frees the block.
    """
    if sys.platform == "darwin":
        return {"MallocScribble": "1", "MallocZeroOnFree": "0"}
    src = tmp_path / "poison_free.c"
    src.write_text(_POISON_FREE_C)
    lib = tmp_path / "poison_free.so"
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-O2", "-o", str(lib), str(src), "-ldl"],
        check=True, capture_output=True)
    return {"LD_PRELOAD": str(lib)}


@pytest.mark.skipif(
    not (sys.platform == "darwin" or (sys.platform == "linux" and shutil.which("gcc"))),
    reason="needs macOS, or Linux with gcc, for an allocator that poisons freed memory")
def test_jit_destroy_calls_read_no_freed_buffer(tmp_path):
    """Run the JIT lifecycle tests and online_scoring's raise branches under an allocator that overwrites freed memory.

    These tests destroy each handle through a buffer that holds it: the destroy call reads the handle from the buffer
    and writes NULL back. A freed buffer usually keeps its bytes until the memory is reused, so a destroy that read
    a buffer numba had already freed would still pass. An allocator that overwrites memory as it frees it makes the
    same read return garbage, and the process crashes. The tests run in a child process, because the allocator reads
    its variables when the process starts.

    A first child fills a numba array, lets numba free it and prints the freed bytes. If the freed memory keeps its
    bytes, or the allocator zeroed them, the test skips instead of passing. macOS ignores the variables for a restricted
    Python, and a destroy that checks the handle for NULL does not crash on a zeroed buffer.
    """
    here = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(here))
    env = dict(os.environ)
    env.update(_poison_on_free(tmp_path))
    check = subprocess.run(
        [sys.executable, "-c", _FREED_BYTES_SCRIPT],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=600,
    )
    assert check.returncode == 0, f"stdout={check.stdout!r} stderr={check.stderr[-3000:]!r}"
    freed = check.stdout.split("FREED-BYTES", 1)[1].split()[0]
    if "1111" in freed:
        pytest.skip(f"freed memory keeps its bytes under this allocator: {freed}")
    if freed.strip("0") == "":
        pytest.skip(f"this allocator zeroes freed memory, which a destroy checking for NULL survives: {freed}")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        + [f"{here}::{name}" for name in _BUFFER_LIFETIME_TESTS],
        cwd=repo_root, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout[-3000:]!r} stderr={proc.stderr[-3000:]!r}"


# --- Value Interface ---

def aux_destroy_value(val_p):
    """Destroy the value handle ``val_p``.

    ``duckdb_destroy_value`` takes a pointer to the handle, so the helper stores the handle in a one-element buffer and
    passes the buffer's address.
    """
    buf = numpy.zeros(1, dtype=numpy.intp)
    buf[0] = val_p
    ducklib.duckdb_destroy_value(buf.ctypes.data)


def test_create_get_date():
    days = numpy.int32(19000)
    val_p = ducklib.duckdb_create_date(days)
    assert val_p != 0
    result = ducklib.duckdb_get_date(val_p)
    assert result == days
    aux_destroy_value(val_p)


def test_create_get_time():
    micros = numpy.int64(45000000000)
    val_p = ducklib.duckdb_create_time(micros)
    assert val_p != 0
    result = ducklib.duckdb_get_time(val_p)
    assert result == micros
    aux_destroy_value(val_p)


def test_create_get_time_tz():
    micros = numpy.int64(45000000000)
    offset = numpy.int32(3600)
    time_tz = ducklib.duckdb_create_time_tz(micros, offset)
    assert time_tz != 0
    val_p = ducklib.duckdb_create_time_tz_value(time_tz)
    assert val_p != 0
    result = ducklib.duckdb_get_time_tz(val_p)
    assert result == time_tz
    aux_destroy_value(val_p)


def test_create_get_timestamp():
    micros = numpy.int64(1735689600000000)
    val_p = ducklib.duckdb_create_timestamp(micros)
    assert val_p != 0
    result = ducklib.duckdb_get_timestamp(val_p)
    assert result == micros
    aux_destroy_value(val_p)


def test_create_get_timestamp_ms():
    ms = numpy.int64(1735689600000)
    val_p = ducklib.duckdb_create_timestamp_ms(ms)
    assert val_p != 0
    result = ducklib.duckdb_get_timestamp_ms(val_p)
    assert result == ms
    aux_destroy_value(val_p)


def test_create_get_timestamp_ns():
    ns = numpy.int64(1735689600000000000)
    val_p = ducklib.duckdb_create_timestamp_ns(ns)
    assert val_p != 0
    result = ducklib.duckdb_get_timestamp_ns(val_p)
    assert result == ns
    aux_destroy_value(val_p)


def test_create_get_timestamp_s():
    secs = numpy.int64(1735689600)
    val_p = ducklib.duckdb_create_timestamp_s(secs)
    assert val_p != 0
    result = ducklib.duckdb_get_timestamp_s(val_p)
    assert result == secs
    aux_destroy_value(val_p)


def test_create_get_timestamp_tz():
    micros = numpy.int64(1735689600000000)
    val_p = ducklib.duckdb_create_timestamp_tz(micros)
    assert val_p != 0
    result = ducklib.duckdb_get_timestamp_tz(val_p)
    assert result == micros
    aux_destroy_value(val_p)


def test_create_get_blob():
    blob_data = ctypes.create_string_buffer(b"\x00\x01\x02\x03", 4)
    blob_p = ctypes.cast(blob_data, ctypes.c_void_p).value
    val_p = ducklib.duckdb_create_blob(blob_p, 4)
    assert val_p != 0
    result = ducklib.duckdb_get_blob(val_p)
    data_p = result[0]
    size = result[1]
    assert size == 4
    raw = (ctypes.c_char * size).from_address(data_p)
    assert raw[:] == b"\x00\x01\x02\x03"
    ducklib.duckdb_free(data_p)
    aux_destroy_value(val_p)


def test_create_get_hugeint():
    val_p = ducklib.duckdb_create_hugeint((42, 0))
    assert val_p != 0
    result = ducklib.duckdb_get_hugeint(val_p)
    assert result[0] == 42
    assert result[1] == 0
    aux_destroy_value(val_p)


def test_create_get_hugeint_negative():
    val_p = ducklib.duckdb_create_hugeint((0, -1))
    assert val_p != 0
    result = ducklib.duckdb_get_hugeint(val_p)
    assert result[0] == 0
    assert result[1] == -1
    aux_destroy_value(val_p)


def test_create_get_uhugeint():
    val_p = ducklib.duckdb_create_uhugeint((100, 0))
    assert val_p != 0
    result = ducklib.duckdb_get_uhugeint(val_p)
    assert result[0] == 100
    assert result[1] == 0
    aux_destroy_value(val_p)


def test_create_get_uhugeint_large():
    val_p = ducklib.duckdb_create_uhugeint((2**63, 1))
    assert val_p != 0
    result = ducklib.duckdb_get_uhugeint(val_p)
    assert result[0] == 2**63
    assert result[1] == 1
    aux_destroy_value(val_p)


def test_create_get_interval():
    val_p = ducklib.duckdb_create_interval((1, 2, 3000000))
    assert val_p != 0
    result = ducklib.duckdb_get_interval(val_p)
    assert result[0] == 1
    assert result[1] == 2
    assert result[2] == 3000000
    aux_destroy_value(val_p)


def test_create_get_interval_zero():
    val_p = ducklib.duckdb_create_interval((0, 0, 0))
    assert val_p != 0
    result = ducklib.duckdb_get_interval(val_p)
    assert result[0] == 0
    assert result[1] == 0
    assert result[2] == 0
    aux_destroy_value(val_p)


def test_create_get_decimal():
    val_p = ducklib.duckdb_create_decimal((10, 2, 12345, 0))
    assert val_p != 0
    result = ducklib.duckdb_get_decimal(val_p)
    assert result[0] == 10
    assert result[1] == 2
    assert result[2] == 12345
    assert result[3] == 0
    aux_destroy_value(val_p)


def test_create_get_uuid():
    lower = numpy.uint64(0x0123456789ABCDEF)
    upper = numpy.uint64(0xFEDCBA9876543210)
    val_p = ducklib.duckdb_create_uuid((lower, upper))
    assert val_p != 0
    result = ducklib.duckdb_get_uuid(val_p)
    assert result[0] == lower
    assert result[1] == upper
    aux_destroy_value(val_p)


def test_create_get_bignum():
    """Check that a BIGNUM keeps its bytes and its sign through ``duckdb_create_bignum`` and ``duckdb_get_bignum``.

    The value has a magnitude of five bytes, and the test runs it with the negative flag clear and set. duckdb 1.4.0
    renamed ``duckdb_create_varint`` and ``duckdb_get_varint`` to the bignum spellings and kept the enum value and the
    struct. The bindings call whichever spelling the loaded libduckdb exports, so the test runs, without a skip, on
    every duckdb that numbduck supports.
    """
    payload = b"\x01\x02\x03\x04\x05"
    data_bytes = ctypes.create_string_buffer(payload, len(payload))
    data_p = ctypes.cast(data_bytes, ctypes.c_void_p).value
    for is_negative in (0, 1):
        val_p = ducklib.duckdb_create_bignum((data_p, len(payload), is_negative))
        assert val_p != 0, f"create failed for is_negative={is_negative}"
        out_p, size, out_negative = ducklib.duckdb_get_bignum(val_p)
        assert size == len(payload), f"expected size {len(payload)}, got {size}"
        assert out_negative == is_negative, f"expected {is_negative}, got {out_negative}"
        raw = (ctypes.c_char * size).from_address(out_p)
        assert raw[:] == payload, f"expected {payload!r}, got {raw[:]!r}"
        ducklib.duckdb_free(out_p)
        aux_destroy_value(val_p)


def test_bignum_bindings_are_not_disk_cached():
    """Check that numba does not cache the two bignum bindings on disk, but caches ``duckdb_create_uuid`` beside them.

    numba's cache key hashes the bytecode and the closure cells only. ducklib chooses the C symbol that the bignum
    bindings call when it is imported, and the bindings read it from a module global, so neither the bytecode nor the
    closure cells contain the symbol. Code cached under one spelling would load in a process whose libduckdb exports
    the other, and the import of ducklib would then abort with ``LLVM ERROR: Symbol not found``.

    The test reads each dispatcher's cache object, not the cache directory. The bindings compile when ducklib is
    imported, so a listing of the directory taken here could not tell a new file from one that an earlier run left. It
    would also find nothing when ``NUMBA_CACHE_DIR`` points elsewhere.
    """
    assert "cache" not in ducklib._bignum_jit_options
    for name in ("duckdb_create_bignum", "duckdb_get_bignum"):
        assert isinstance(getattr(ducklib, name)._cache, NullCache), name
    assert isinstance(ducklib.duckdb_create_uuid._cache, FunctionCache)


def test_bignum_binds_the_spelling_this_libduckdb_exports():
    """Check that the bignum bindings call the bignum symbols on duckdb 1.4 and later, and the varint symbols before it.

    They never call a symbol that the library lacks, and ducklib has no ``duckdb_create_varint`` or
    ``duckdb_get_varint`` of its own.
    """
    resolved = ducklib._BIGNUM_CREATE, ducklib._BIGNUM_GET
    expected = (
        ("duckdb_create_bignum", "duckdb_get_bignum")
        if hasattr(ducklib.duckdb_lib, "duckdb_create_bignum")
        else ("duckdb_create_varint", "duckdb_get_varint")
    )
    assert resolved == expected
    for name in resolved:
        assert hasattr(ducklib.duckdb_lib, name)
    assert not hasattr(ducklib, "duckdb_create_varint")
    assert not hasattr(ducklib, "duckdb_get_varint")


def test_gated_binding_missing_symbol_raises_clear_njit_error():
    """Check that a version-gated binding whose C symbol the library lacks fails clearly when jitted code calls it.

    numbox's ``proxy_if_available`` makes a version-gated binding. When jitted code that calls it is compiled, numba
    raises a ``TypingError`` whose message names the binding and says that its C symbol is missing, instead of an
    opaque typing error. The test passes a library object without the attribute in place of a libduckdb without the
    symbol, so it runs whether or not the loaded libduckdb exports the symbol.
    """

    class _NoSymLib:
        pass

    @proxy_if_available(
        _NoSymLib(), ducklib.signatures.get("duckdb_create_bignum"),
        jit_options=ducklib.jit_options)
    def duckdb_create_bignum(val):
        return _call_lib_func("duckdb_create_bignum", (val,))

    @njit
    def use_gated(x):
        return duckdb_create_bignum(x)

    with pytest.raises(TypingError) as excinfo:
        use_gated(0)
    message = str(excinfo.value)
    assert "duckdb_create_bignum" in message
    assert "not available" in message
    assert "C symbol missing" in message


@pytest.mark.skipif(
    hasattr(ducklib.duckdb_lib, 'duckdb_scalar_function_set_init'),
    reason="duckdb_scalar_function_set_init present in this libduckdb; absent-symbol path not exercised",
)
def test_gated_binding_unavailable_real_name_njit_error():
    """Check that ducklib's ``duckdb_scalar_function_set_init`` fails clearly in jitted code if its symbol is missing.

    The ``TypingError`` names the binding and says that it is not available. The test above builds a gated binding of
    its own; this one checks the binding as ducklib builds it. It runs only with a libduckdb that lacks the symbol, such
    as duckdb 1.4 and earlier.
    """

    @njit
    def use_set_init(scalar_function_p, init_p):
        return ducklib.duckdb_scalar_function_set_init(scalar_function_p, init_p)

    with pytest.raises(TypingError) as excinfo:
        use_set_init(0, 0)
    message = str(excinfo.value)
    assert "duckdb_scalar_function_set_init" in message
    assert "not available" in message


def test_create_get_bit():
    data_bytes = ctypes.create_string_buffer(b"\x05\xA0", 2)
    data_p = ctypes.cast(data_bytes, ctypes.c_void_p).value
    val_p = ducklib.duckdb_create_bit((data_p, 2))
    assert val_p != 0
    result = ducklib.duckdb_get_bit(val_p)
    assert result[1] == 2
    raw = (ctypes.c_char * result[1]).from_address(result[0])
    assert raw[:] == b"\x05\xA0"
    ducklib.duckdb_free(result[0])
    aux_destroy_value(val_p)


# --- Scalar Value Creators/Getters ---


def test_create_get_bool_true():
    val_p = ducklib.duckdb_create_bool(1)
    assert val_p != 0
    result = ducklib.duckdb_get_bool(val_p)
    assert result == 1
    aux_destroy_value(val_p)


def test_create_get_bool_false():
    val_p = ducklib.duckdb_create_bool(0)
    assert val_p != 0
    result = ducklib.duckdb_get_bool(val_p)
    assert result == 0
    aux_destroy_value(val_p)


def test_create_get_int8():
    val_p = ducklib.duckdb_create_int8(-42)
    assert val_p != 0
    result = ducklib.duckdb_get_int8(val_p)
    assert result == -42
    aux_destroy_value(val_p)


def test_create_get_int16():
    val_p = ducklib.duckdb_create_int16(-1234)
    assert val_p != 0
    result = ducklib.duckdb_get_int16(val_p)
    assert result == -1234
    aux_destroy_value(val_p)


def test_create_get_int32():
    val_p = ducklib.duckdb_create_int32(-100000)
    assert val_p != 0
    result = ducklib.duckdb_get_int32(val_p)
    assert result == -100000
    aux_destroy_value(val_p)


def test_create_get_int64():
    val_p = ducklib.duckdb_create_int64(-9999999999)
    assert val_p != 0
    result = ducklib.duckdb_get_int64(val_p)
    assert result == -9999999999
    aux_destroy_value(val_p)


def test_create_get_uint8():
    val_p = ducklib.duckdb_create_uint8(200)
    assert val_p != 0
    result = ducklib.duckdb_get_uint8(val_p)
    assert result == 200
    aux_destroy_value(val_p)


def test_create_get_uint16():
    val_p = ducklib.duckdb_create_uint16(60000)
    assert val_p != 0
    result = ducklib.duckdb_get_uint16(val_p)
    assert result == 60000
    aux_destroy_value(val_p)


def test_create_get_uint32():
    val_p = ducklib.duckdb_create_uint32(3000000000)
    assert val_p != 0
    result = ducklib.duckdb_get_uint32(val_p)
    assert result == 3000000000
    aux_destroy_value(val_p)


def test_create_get_uint64():
    val_p = ducklib.duckdb_create_uint64(10000000000000000000)
    assert val_p != 0
    result = ducklib.duckdb_get_uint64(val_p)
    assert result == 10000000000000000000
    aux_destroy_value(val_p)


def test_create_get_float():
    val_p = ducklib.duckdb_create_float(numpy.float32(3.14))
    assert val_p != 0
    result = ducklib.duckdb_get_float(val_p)
    assert abs(result - 3.14) < 0.01
    aux_destroy_value(val_p)


def test_create_get_double():
    val_p = ducklib.duckdb_create_double(2.718281828)
    assert val_p != 0
    result = ducklib.duckdb_get_double(val_p)
    assert abs(result - 2.718281828) < 1e-9
    aux_destroy_value(val_p)


# --- String Value Functions ---


def test_create_get_varchar():
    text = ctypes.c_char_p(b"hello duckdb")
    val_p = ducklib.duckdb_create_varchar(
        ctypes.cast(text, ctypes.c_void_p).value
    )
    assert val_p != 0
    str_p = ducklib.duckdb_get_varchar(val_p)
    assert str_p != 0
    result = ctypes.string_at(str_p)
    assert result == b"hello duckdb"
    ducklib.duckdb_free(str_p)
    aux_destroy_value(val_p)


def test_create_varchar_length():
    text = ctypes.c_char_p(b"hello\x00world")
    val_p = ducklib.duckdb_create_varchar_length(
        ctypes.cast(text, ctypes.c_void_p).value, 5
    )
    assert val_p != 0
    str_p = ducklib.duckdb_get_varchar(val_p)
    assert str_p != 0
    result = ctypes.string_at(str_p)
    assert result == b"hello"
    ducklib.duckdb_free(str_p)
    aux_destroy_value(val_p)


@pytest.mark.skipif(
    not hasattr(ducklib.duckdb_lib, 'duckdb_value_to_string'),
    reason="duckdb_value_to_string not available",
)
def test_value_to_string():
    val_p = ducklib.duckdb_create_int32(42)
    assert val_p != 0
    str_p = ducklib.duckdb_value_to_string(val_p)
    assert str_p != 0
    result = ctypes.string_at(str_p)
    assert result == b"42"
    ducklib.duckdb_free(str_p)
    aux_destroy_value(val_p)


# --- Null Value Functions ---


def test_create_null_value():
    val_p = ducklib.duckdb_create_null_value()
    assert val_p != 0
    assert ducklib.duckdb_is_null_value(val_p) == 1
    aux_destroy_value(val_p)


def test_is_null_value_false():
    val_p = ducklib.duckdb_create_int32(7)
    assert val_p != 0
    assert ducklib.duckdb_is_null_value(val_p) == 0
    aux_destroy_value(val_p)


# --- Container Value Creators/Getters ---


def test_create_get_list_value():
    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    v1 = ducklib.duckdb_create_int32(10)
    v2 = ducklib.duckdb_create_int32(20)
    vals = numpy.array([v1, v2], dtype=numpy.intp)
    lv = ducklib.duckdb_create_list_value(int_type_p, vals.ctypes.data, 2)
    assert lv != 0
    assert ducklib.duckdb_get_list_size(lv) == 2
    c0 = ducklib.duckdb_get_list_child(lv, 0)
    assert ducklib.duckdb_get_int32(c0) == 10
    aux_destroy_value(c0)
    c1 = ducklib.duckdb_get_list_child(lv, 1)
    assert ducklib.duckdb_get_int32(c1) == 20
    aux_destroy_value(c1)
    aux_destroy_value(lv)
    aux_destroy_value(v1)
    aux_destroy_value(v2)
    lt_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)


def test_create_get_list_value_empty():
    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    vals = numpy.zeros(0, dtype=numpy.intp)
    lv = ducklib.duckdb_create_list_value(int_type_p, vals.ctypes.data, 0)
    assert lv != 0
    assert ducklib.duckdb_get_list_size(lv) == 0
    aux_destroy_value(lv)
    lt_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)


@pytest.mark.skipif(
    not hasattr(ducklib.duckdb_lib, 'duckdb_create_map_value'),
    reason="duckdb_create_map_value not available",
)
def test_create_get_map_value():
    key_type = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    val_type = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    map_type = ducklib.duckdb_create_map_type(key_type, val_type)
    k1 = ducklib.duckdb_create_int32(1)
    k2 = ducklib.duckdb_create_int32(2)
    v1 = ducklib.duckdb_create_int32(100)
    v2 = ducklib.duckdb_create_int32(200)
    keys = numpy.array([k1, k2], dtype=numpy.intp)
    vals = numpy.array([v1, v2], dtype=numpy.intp)
    mv = ducklib.duckdb_create_map_value(
        map_type, keys.ctypes.data, vals.ctypes.data, 2)
    assert mv != 0
    assert ducklib.duckdb_get_map_size(mv) == 2
    mk0 = ducklib.duckdb_get_map_key(mv, 0)
    assert ducklib.duckdb_get_int32(mk0) == 1
    aux_destroy_value(mk0)
    mgv0 = ducklib.duckdb_get_map_value(mv, 0)
    assert ducklib.duckdb_get_int32(mgv0) == 100
    aux_destroy_value(mgv0)
    mk1 = ducklib.duckdb_get_map_key(mv, 1)
    assert ducklib.duckdb_get_int32(mk1) == 2
    aux_destroy_value(mk1)
    mgv1 = ducklib.duckdb_get_map_value(mv, 1)
    assert ducklib.duckdb_get_int32(mgv1) == 200
    aux_destroy_value(mgv1)
    aux_destroy_value(mv)
    aux_destroy_value(k1)
    aux_destroy_value(k2)
    aux_destroy_value(v1)
    aux_destroy_value(v2)
    lt_buf = numpy.array([map_type], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)
    lt_buf[0] = key_type
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)
    lt_buf[0] = val_type
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)


def test_create_get_struct_value():
    t1 = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    t2 = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    types_arr = numpy.array([t1, t2], dtype=numpy.intp)
    n1_p = get_unicode_data_p("a")
    n2_p = get_unicode_data_p("b")
    names_arr = numpy.array([n1_p, n2_p], dtype=numpy.intp)
    struct_type = ducklib.duckdb_create_struct_type(
        types_arr.ctypes.data, names_arr.ctypes.data, 2)
    assert struct_type != 0
    sv1 = ducklib.duckdb_create_int32(42)
    sv2 = ducklib.duckdb_create_int32(99)
    svals = numpy.array([sv1, sv2], dtype=numpy.intp)
    stv = ducklib.duckdb_create_struct_value(struct_type, svals.ctypes.data)
    assert stv != 0
    sc0 = ducklib.duckdb_get_struct_child(stv, 0)
    assert ducklib.duckdb_get_int32(sc0) == 42
    aux_destroy_value(sc0)
    sc1 = ducklib.duckdb_get_struct_child(stv, 1)
    assert ducklib.duckdb_get_int32(sc1) == 99
    aux_destroy_value(sc1)
    aux_destroy_value(stv)
    aux_destroy_value(sv1)
    aux_destroy_value(sv2)
    lt_buf = numpy.array([struct_type], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)
    lt_buf[0] = t1
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)
    lt_buf[0] = t2
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)


def test_create_array_value():
    int_type = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    v1 = ducklib.duckdb_create_int32(1)
    v2 = ducklib.duckdb_create_int32(2)
    v3 = ducklib.duckdb_create_int32(3)
    vals = numpy.array([v1, v2, v3], dtype=numpy.intp)
    av = ducklib.duckdb_create_array_value(int_type, vals.ctypes.data, 3)
    assert av != 0
    # duckdb_get_list_size and duckdb_get_list_child do not read a value of ARRAY type: on duckdb 1.5.4 they return a
    # size of 0 and a null child. The test checks the array through its logical type instead: the type id, the fixed
    # size and the element type. duckdb_get_value_type returns the value handle itself, so the test does not destroy
    # array_type_p separately.
    array_type_p = ducklib.duckdb_get_value_type(av)
    assert array_type_p == av
    assert ducklib.duckdb_get_type_id(array_type_p) == ducklib.DUCKDB_TYPE_ARRAY
    assert ducklib.duckdb_array_type_array_size(array_type_p) == 3
    child_type_p = ducklib.duckdb_array_type_child_type(array_type_p)
    assert ducklib.duckdb_get_type_id(child_type_p) == ducklib.DUCKDB_TYPE_INTEGER
    child_buf = numpy.array([child_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(child_buf.ctypes.data)
    aux_destroy_value(av)
    aux_destroy_value(v1)
    aux_destroy_value(v2)
    aux_destroy_value(v3)
    lt_buf = numpy.array([int_type], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)


def test_list_value_from_column_logical_type():
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]
    query_p = get_unicode_data_p("SELECT [1, 2, 3] AS lst")
    result = create_duckdb_result()
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    col_type = ducklib.duckdb_column_logical_type(result.ctypes.data, 0)
    assert col_type != 0
    child_type = ducklib.duckdb_list_type_child_type(col_type)
    assert child_type != 0
    v1 = ducklib.duckdb_create_int32(77)
    v2 = ducklib.duckdb_create_int32(88)
    vals = numpy.array([v1, v2], dtype=numpy.intp)
    lv = ducklib.duckdb_create_list_value(child_type, vals.ctypes.data, 2)
    assert lv != 0
    assert ducklib.duckdb_get_list_size(lv) == 2
    c0 = ducklib.duckdb_get_list_child(lv, 0)
    assert ducklib.duckdb_get_int32(c0) == 77
    aux_destroy_value(c0)
    c1 = ducklib.duckdb_get_list_child(lv, 1)
    assert ducklib.duckdb_get_int32(c1) == 88
    aux_destroy_value(c1)
    aux_destroy_value(lv)
    aux_destroy_value(v1)
    aux_destroy_value(v2)
    lt_buf = numpy.array([child_type], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)
    lt_buf[0] = col_type
    ducklib.duckdb_destroy_logical_type(lt_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


# --- Value Type and Destroy ---


def test_get_value_type():
    val_p = ducklib.duckdb_create_int32(99)
    assert val_p != 0
    type_p = ducklib.duckdb_get_value_type(val_p)
    assert type_p != 0
    assert ducklib.duckdb_get_type_id(type_p) == ducklib.DUCKDB_TYPE_INTEGER
    # duckdb_get_value_type returns the type that the value owns, at the address of the value handle itself, so the
    # test destroys only the value: destroying the type as well would free the same memory twice. The assertion checks
    # the aliasing, so the rule not to destroy the type depends on a check that fails if the call returns a type at
    # another address.
    assert type_p == val_p
    aux_destroy_value(val_p)


def test_destroy_value():
    val_p = ducklib.duckdb_create_int32(1)
    assert val_p != 0
    aux_destroy_value(val_p)


# --- Struct-by-value ABI round-trip ---


def test_struct_by_value_abi_roundtrip():
    """Check that a 16-byte and a 24-byte struct keep every bit through ducklib's create and get bindings.

    On SysV x86-64 and AAPCS64, numbox's ``_call_lib_func`` passes a struct of 16 bytes or less by value. It passes a
    larger one as a copy on the stack on SysV x86-64, and as a pointer to a copy on AAPCS64. The 16-byte struct and the
    24-byte struct cover both cases. Both 64-bit halves of each struct's 128-bit value are nonzero. The test therefore
    catches an upper half that a binding drops or packs wrongly.
    """
    lower = 0xFEDCBA9876543210
    upper = 0x0123456789ABCDEF

    # duckdb_hugeint = {uint64 lower; int64 upper} is a 16-byte struct passed and
    # returned by value on SysV x86-64 and AAPCS64.
    hugeint_p = ducklib.duckdb_create_hugeint((lower, upper))
    assert hugeint_p != 0
    hugeint = ducklib.duckdb_get_hugeint(hugeint_p)
    assert hugeint[0] == lower
    assert hugeint[1] == upper
    aux_destroy_value(hugeint_p)

    # duckdb_decimal is padded to 24 bytes, so it tests how numbox passes a struct over 16 bytes. A width over 18
    # makes DuckDB keep the value as an int128, so the upper half carries data.
    decimal_p = ducklib.duckdb_create_decimal((38, 0, lower, upper))
    assert decimal_p != 0
    decimal = ducklib.duckdb_get_decimal(decimal_p)
    assert decimal[0] == 38
    assert decimal[1] == 0
    assert decimal[2] == lower
    assert decimal[3] == upper
    aux_destroy_value(decimal_p)


def test_create_logical_type_integer():
    type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    assert type_p != 0
    type_id = ducklib.duckdb_get_type_id(type_p)
    assert type_id == ducklib.DUCKDB_TYPE_INTEGER
    buf = numpy.array([type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_create_logical_type_varchar():
    type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_VARCHAR)
    assert type_p != 0
    type_id = ducklib.duckdb_get_type_id(type_p)
    assert type_id == ducklib.DUCKDB_TYPE_VARCHAR
    buf = numpy.array([type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_create_decimal_type():
    type_p = ducklib.duckdb_create_decimal_type(10, 2)
    assert type_p != 0
    type_id = ducklib.duckdb_get_type_id(type_p)
    assert type_id == ducklib.DUCKDB_TYPE_DECIMAL
    assert ducklib.duckdb_decimal_width(type_p) == 10
    assert ducklib.duckdb_decimal_scale(type_p) == 2
    internal_type = ducklib.duckdb_decimal_internal_type(type_p)
    assert internal_type != 0
    buf = numpy.array([type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_logical_type_alias():
    type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    assert type_p != 0
    alias_p = ducklib.duckdb_logical_type_get_alias(type_p)
    assert alias_p == 0  # no alias set yet
    alias_bytes = ctypes.c_char_p(b"my_int")
    alias_c_p = ctypes.c_void_p.from_buffer(alias_bytes).value
    ducklib.duckdb_logical_type_set_alias(type_p, alias_c_p)
    alias_p = ducklib.duckdb_logical_type_get_alias(type_p)
    assert alias_p != 0
    alias_str = ctypes.c_char_p(alias_p).value.decode()
    assert alias_str == "my_int"
    ducklib.duckdb_free(alias_p)
    buf = numpy.array([type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_create_list_type():
    child_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    assert child_p != 0
    list_p = ducklib.duckdb_create_list_type(child_p)
    assert list_p != 0
    type_id = ducklib.duckdb_get_type_id(list_p)
    assert type_id == ducklib.DUCKDB_TYPE_LIST
    child_back_p = ducklib.duckdb_list_type_child_type(list_p)
    assert child_back_p != 0
    child_type_id = ducklib.duckdb_get_type_id(child_back_p)
    assert child_type_id == ducklib.DUCKDB_TYPE_INTEGER
    for p in [child_back_p, list_p, child_p]:
        buf = numpy.array([p], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_create_array_type():
    child_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    assert child_p != 0
    array_p = ducklib.duckdb_create_array_type(child_p, 5)
    assert array_p != 0
    type_id = ducklib.duckdb_get_type_id(array_p)
    assert type_id == ducklib.DUCKDB_TYPE_ARRAY
    size = ducklib.duckdb_array_type_array_size(array_p)
    assert size == 5
    child_back_p = ducklib.duckdb_array_type_child_type(array_p)
    assert child_back_p != 0
    child_type_id = ducklib.duckdb_get_type_id(child_back_p)
    assert child_type_id == ducklib.DUCKDB_TYPE_INTEGER
    for p in [child_back_p, array_p, child_p]:
        buf = numpy.array([p], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_create_map_type():
    key_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    val_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_VARCHAR)
    assert key_p != 0 and val_p != 0
    map_p = ducklib.duckdb_create_map_type(key_p, val_p)
    assert map_p != 0
    type_id = ducklib.duckdb_get_type_id(map_p)
    assert type_id == ducklib.DUCKDB_TYPE_MAP
    key_back_p = ducklib.duckdb_map_type_key_type(map_p)
    val_back_p = ducklib.duckdb_map_type_value_type(map_p)
    assert ducklib.duckdb_get_type_id(key_back_p) == ducklib.DUCKDB_TYPE_INTEGER
    assert ducklib.duckdb_get_type_id(val_back_p) == ducklib.DUCKDB_TYPE_VARCHAR
    for p in [val_back_p, key_back_p, map_p, val_p, key_p]:
        buf = numpy.array([p], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_create_struct_type():
    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    varchar_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_VARCHAR)
    types_arr = numpy.array([int_type_p, varchar_type_p], dtype=numpy.intp)
    name1 = ctypes.c_char_p(b"id")
    name2 = ctypes.c_char_p(b"name")
    names_arr = numpy.array(
        [ctypes.c_void_p.from_buffer(name1).value, ctypes.c_void_p.from_buffer(name2).value],
        dtype=numpy.intp
    )
    struct_p = ducklib.duckdb_create_struct_type(types_arr.ctypes.data, names_arr.ctypes.data, 2)
    assert struct_p != 0
    type_id = ducklib.duckdb_get_type_id(struct_p)
    assert type_id == ducklib.DUCKDB_TYPE_STRUCT
    count = ducklib.duckdb_struct_type_child_count(struct_p)
    assert count == 2
    child_name_p = ducklib.duckdb_struct_type_child_name(struct_p, 0)
    assert child_name_p != 0
    child_name = ctypes.c_char_p(child_name_p).value.decode()
    assert child_name == "id"
    ducklib.duckdb_free(child_name_p)
    child_type_p = ducklib.duckdb_struct_type_child_type(struct_p, 0)
    assert ducklib.duckdb_get_type_id(child_type_p) == ducklib.DUCKDB_TYPE_INTEGER
    child_type_buf = numpy.array([child_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(child_type_buf.ctypes.data)
    struct_buf = numpy.array([struct_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(struct_buf.ctypes.data)
    int_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(int_buf.ctypes.data)
    varchar_buf = numpy.array([varchar_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(varchar_buf.ctypes.data)


def test_create_union_type():
    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    varchar_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_VARCHAR)
    types_arr = numpy.array([int_type_p, varchar_type_p], dtype=numpy.intp)
    name1 = ctypes.c_char_p(b"num")
    name2 = ctypes.c_char_p(b"str")
    names_arr = numpy.array(
        [ctypes.c_void_p.from_buffer(name1).value, ctypes.c_void_p.from_buffer(name2).value],
        dtype=numpy.intp
    )
    union_p = ducklib.duckdb_create_union_type(types_arr.ctypes.data, names_arr.ctypes.data, 2)
    assert union_p != 0
    type_id = ducklib.duckdb_get_type_id(union_p)
    assert type_id == ducklib.DUCKDB_TYPE_UNION
    count = ducklib.duckdb_union_type_member_count(union_p)
    assert count == 2
    member_name_p = ducklib.duckdb_union_type_member_name(union_p, 0)
    assert member_name_p != 0
    member_name = ctypes.c_char_p(member_name_p).value.decode()
    assert member_name == "num"
    ducklib.duckdb_free(member_name_p)
    member_type_p = ducklib.duckdb_union_type_member_type(union_p, 0)
    assert ducklib.duckdb_get_type_id(member_type_p) == ducklib.DUCKDB_TYPE_INTEGER
    member_type_buf = numpy.array([member_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(member_type_buf.ctypes.data)
    union_buf = numpy.array([union_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(union_buf.ctypes.data)
    int_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(int_buf.ctypes.data)
    varchar_buf = numpy.array([varchar_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(varchar_buf.ctypes.data)


def test_create_enum_type():
    name1 = ctypes.c_char_p(b"small")
    name2 = ctypes.c_char_p(b"medium")
    name3 = ctypes.c_char_p(b"large")
    names_arr = numpy.array(
        [ctypes.c_void_p.from_buffer(n).value for n in [name1, name2, name3]],
        dtype=numpy.intp
    )
    enum_p = ducklib.duckdb_create_enum_type(names_arr.ctypes.data, 3)
    assert enum_p != 0
    type_id = ducklib.duckdb_get_type_id(enum_p)
    assert type_id == ducklib.DUCKDB_TYPE_ENUM
    dict_size = ducklib.duckdb_enum_dictionary_size(enum_p)
    assert dict_size == 3
    val_p = ducklib.duckdb_enum_dictionary_value(enum_p, 0)
    assert val_p != 0
    val_str = ctypes.c_char_p(val_p).value.decode()
    assert val_str == "small"
    ducklib.duckdb_free(val_p)
    val_p = ducklib.duckdb_enum_dictionary_value(enum_p, 2)
    val_str = ctypes.c_char_p(val_p).value.decode()
    assert val_str == "large"
    ducklib.duckdb_free(val_p)
    internal_type = ducklib.duckdb_enum_internal_type(enum_p)
    assert internal_type != 0
    enum_buf = numpy.array([enum_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(enum_buf.ctypes.data)





@njit
def _add_one_impl(info, chunk, output):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    input_vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(input_vec)
    out_data = ducklib.duckdb_vector_get_data(output)
    in_arr = carray(_cast_int_to_void_p(in_data), (n,), dtype=numpy.int32)
    out_arr = carray(_cast_int_to_void_p(out_data), (n,), dtype=numpy.int32)
    for i in range(n):
        out_arr[i] = in_arr[i] + 1


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _add_one_cb(info, chunk, output):
    _add_one_impl(info, chunk, output)


def test_scalar_function_round_trip():
    """Check that SQL calls a scalar UDF registered through the C API: ``add_one(42)`` returns 43."""
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    func_p = ducklib.duckdb_create_scalar_function()
    assert func_p != 0

    name_p = get_unicode_data_p("add_one")
    ducklib.duckdb_scalar_function_set_name(func_p, name_p)

    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    ducklib.duckdb_scalar_function_add_parameter(func_p, int_type_p)
    ducklib.duckdb_scalar_function_set_return_type(func_p, int_type_p)
    type_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)

    ducklib.duckdb_scalar_function_set_function(func_p, _add_one_cb.address)

    rc = ducklib.duckdb_register_scalar_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess

    # The connection keeps the registered function, so the handle can be destroyed before the query runs.
    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)

    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT add_one(42)")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess, f"Query failed, rc={rc}"

    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    data_p = ducklib.duckdb_vector_get_data(vec_p)
    val = (ctypes.c_int32 * 1).from_address(data_p)[0]
    assert val == 43, f"Expected 43, got {val}"

    chunk_buf = numpy.array([chunk_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_data_chunk(chunk_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


EXTRA_INFO_MAGIC = 0xDEADBEEF


@njit
def _extra_info_impl(info, chunk, output):
    extra = ducklib.duckdb_scalar_function_get_extra_info(info)
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    out_data = ducklib.duckdb_vector_get_data(output)
    out_arr = carray(_cast_int_to_void_p(out_data), (n,), dtype=numpy.int64)
    for i in range(n):
        out_arr[i] = extra


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _extra_info_cb(info, chunk, output):
    _extra_info_impl(info, chunk, output)


def test_scalar_function_extra_info():
    """Check that the callback reads back, with ``duckdb_scalar_function_get_extra_info``, the extra info set on it."""
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    func_p = ducklib.duckdb_create_scalar_function()
    ducklib.duckdb_scalar_function_set_name(
        func_p, get_unicode_data_p("get_extra"))

    bigint_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_BIGINT)
    ducklib.duckdb_scalar_function_set_return_type(func_p, bigint_type_p)
    type_buf = numpy.array([bigint_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)

    ducklib.duckdb_scalar_function_set_function(func_p, _extra_info_cb.address)
    ducklib.duckdb_scalar_function_set_extra_info(func_p, EXTRA_INFO_MAGIC, 0)

    rc = ducklib.duckdb_register_scalar_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess

    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)

    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT get_extra()")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess

    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    data_p = ducklib.duckdb_vector_get_data(vec_p)
    val = (ctypes.c_int64 * 1).from_address(data_p)[0]
    assert val == EXTRA_INFO_MAGIC, f"Expected {EXTRA_INFO_MAGIC}, got {val}"

    chunk_buf = numpy.array([chunk_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_data_chunk(chunk_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


@njit
def _error_impl(info, chunk, output):
    ducklib.duckdb_scalar_function_set_error(
        info, get_unicode_data_p("test error from callback"))


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _error_cb(info, chunk, output):
    _error_impl(info, chunk, output)


def test_scalar_function_set_error():
    """Check that a callback that calls ``duckdb_scalar_function_set_error`` makes the query return ``DuckDBError``."""
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    func_p = ducklib.duckdb_create_scalar_function()
    ducklib.duckdb_scalar_function_set_name(
        func_p, get_unicode_data_p("will_fail"))

    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    ducklib.duckdb_scalar_function_add_parameter(func_p, int_type_p)
    ducklib.duckdb_scalar_function_set_return_type(func_p, int_type_p)
    type_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)

    ducklib.duckdb_scalar_function_set_function(func_p, _error_cb.address)
    rc = ducklib.duckdb_register_scalar_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess

    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)

    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT will_fail(42)")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError, got {rc}"

    ducklib.duckdb_destroy_result(result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


@njit
def _double_it_int_impl(info, chunk, output):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    in_data = ducklib.duckdb_vector_get_data(
        ducklib.duckdb_data_chunk_get_vector(chunk, 0))
    out_data = ducklib.duckdb_vector_get_data(output)
    in_arr = carray(_cast_int_to_void_p(in_data), (n,), dtype=numpy.int32)
    out_arr = carray(_cast_int_to_void_p(out_data), (n,), dtype=numpy.int32)
    for i in range(n):
        out_arr[i] = in_arr[i] * 2


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _double_it_int_cb(info, chunk, output):
    _double_it_int_impl(info, chunk, output)


@njit
def _double_it_dbl_impl(info, chunk, output):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    in_data = ducklib.duckdb_vector_get_data(
        ducklib.duckdb_data_chunk_get_vector(chunk, 0))
    out_data = ducklib.duckdb_vector_get_data(output)
    in_arr = carray(_cast_int_to_void_p(in_data), (n,), dtype=numpy.float64)
    out_arr = carray(_cast_int_to_void_p(out_data), (n,), dtype=numpy.float64)
    for i in range(n):
        out_arr[i] = in_arr[i] * 2.0


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _double_it_dbl_cb(info, chunk, output):
    _double_it_dbl_impl(info, chunk, output)


def test_scalar_function_set_overloads():
    """Check that a scalar function set registers ``double_it`` for INTEGER and DOUBLE.

    DuckDB calls the INTEGER variant for an INTEGER argument and the DOUBLE variant for a DOUBLE argument.
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]


    # Integer variant
    int_func_p = ducklib.duckdb_create_scalar_function()
    ducklib.duckdb_scalar_function_set_name(
        int_func_p, get_unicode_data_p("double_it"))
    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    ducklib.duckdb_scalar_function_add_parameter(int_func_p, int_type_p)
    ducklib.duckdb_scalar_function_set_return_type(
        int_func_p, int_type_p)
    ducklib.duckdb_scalar_function_set_function(
        int_func_p, _double_it_int_cb.address)
    type_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)

    # Double variant
    dbl_func_p = ducklib.duckdb_create_scalar_function()
    ducklib.duckdb_scalar_function_set_name(
        dbl_func_p, get_unicode_data_p("double_it"))
    dbl_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
    ducklib.duckdb_scalar_function_add_parameter(
        dbl_func_p, dbl_type_p)
    ducklib.duckdb_scalar_function_set_return_type(
        dbl_func_p, dbl_type_p)
    ducklib.duckdb_scalar_function_set_function(
        dbl_func_p, _double_it_dbl_cb.address)
    type_buf = numpy.array([dbl_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)

    # Create function set and register
    set_p = ducklib.duckdb_create_scalar_function_set(
        get_unicode_data_p("double_it"))
    rc = ducklib.duckdb_add_scalar_function_to_set(set_p, int_func_p)
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_add_scalar_function_to_set(set_p, dbl_func_p)
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_register_scalar_function_set(conn_p, set_p)
    assert rc == ducklib.DuckDBSuccess

    # Cleanup handles
    for p in [int_func_p, dbl_func_p]:
        buf = numpy.array([p], dtype=numpy.intp)
        ducklib.duckdb_destroy_scalar_function(buf.ctypes.data)
    set_buf = numpy.array([set_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function_set(set_buf.ctypes.data)

    # Test integer variant
    result = create_duckdb_result()
    rc = ducklib.duckdb_query(
        conn_p, get_unicode_data_p("SELECT double_it(21::INTEGER)"),
        result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    val = (ctypes.c_int32 * 1).from_address(
        ducklib.duckdb_vector_get_data(vec_p))[0]
    assert val == 42, f"Expected 42, got {val}"
    chunk_buf = numpy.array([chunk_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_data_chunk(chunk_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)

    # Test double variant
    result = create_duckdb_result()
    rc = ducklib.duckdb_query(
        conn_p, get_unicode_data_p("SELECT double_it(1.5::DOUBLE)"),
        result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    val = (ctypes.c_double * 1).from_address(
        ducklib.duckdb_vector_get_data(vec_p))[0]
    assert val == 3.0, f"Expected 3.0, got {val}"
    chunk_buf = numpy.array([chunk_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_data_chunk(chunk_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)

    aux_close_db(duckdb_database, duckdb_connection)


# Simple registered aggregate that stores all the required data directly in the state buffer.
AGG_STATE_SIZE = 8  # int64 accumulator


@njit
def _agg_state_size_impl(info):
    return AGG_STATE_SIZE


@cfunc(nb_types.uint64(nb_types.intp))
def _agg_state_size_cb(info):
    return _agg_state_size_impl(info)


@njit
def _agg_init_impl(info, state):
    carray(_cast_int_to_void_p(state), (1,), dtype=numpy.int64)[0] = 0


@cfunc(nb_types.void(nb_types.intp, nb_types.intp))
def _agg_init_cb(info, state):
    _agg_init_impl(info, state)


@njit
def _agg_update_impl(info, chunk, states):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    input_vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(input_vec)
    state_ptrs = carray(_cast_int_to_void_p(states), (n,), dtype=numpy.intp)
    in_vals = carray(_cast_int_to_void_p(in_data), (n,), dtype=numpy.int32)
    for i in range(n):
        acc = carray(_cast_int_to_void_p(state_ptrs[i]), (1,), dtype=numpy.int64)
        acc[0] += in_vals[i]


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _agg_update_cb(info, chunk, states):
    _agg_update_impl(info, chunk, states)


@njit
def _agg_update_i64_impl(info, chunk, states):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    input_vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(input_vec)
    state_ptrs = carray(
        _cast_int_to_void_p(states), (n,), dtype=numpy.intp)
    in_vals = carray(
        _cast_int_to_void_p(in_data), (n,), dtype=numpy.int64)
    for i in range(n):
        acc = carray(
            _cast_int_to_void_p(state_ptrs[i]),
            (1,), dtype=numpy.int64)
        acc[0] += in_vals[i]


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _agg_update_i64_cb(info, chunk, states):
    _agg_update_i64_impl(info, chunk, states)


@njit
def _agg_combine_impl(info, source, target, count):
    src_ptrs = carray(_cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    tgt_ptrs = carray(_cast_int_to_void_p(target), (count,), dtype=numpy.intp)
    for i in range(count):
        src_acc = carray(_cast_int_to_void_p(src_ptrs[i]), (1,), dtype=numpy.int64)[0]
        tgt_acc = carray(_cast_int_to_void_p(tgt_ptrs[i]), (1,), dtype=numpy.int64)
        tgt_acc[0] += src_acc


@cfunc(nb_types.void(nb_types.intp, nb_types.intp,
                     nb_types.intp, nb_types.uint64))
def _agg_combine_cb(info, source, target, count):
    _agg_combine_impl(info, source, target, count)


@njit
def _agg_finalize_impl(info, source, result, count, offset):
    out_data = ducklib.duckdb_vector_get_data(result)
    src_ptrs = carray(
        _cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    out_vals = carray(
        _cast_int_to_void_p(out_data), (offset + count,),
        dtype=numpy.int64)
    for i in range(count):
        acc = carray(_cast_int_to_void_p(src_ptrs[i]), (1,), dtype=numpy.int64)[0]
        out_vals[offset + i] = acc


@cfunc(nb_types.void(nb_types.intp, nb_types.intp,
                     nb_types.intp, nb_types.uint64, nb_types.uint64))
def _agg_finalize_cb(info, source, result, count, offset):
    _agg_finalize_impl(info, source, result, count, offset)


def test_aggregate_function_round_trip():
    """Check that SQL calls an aggregate UDF registered through the C API: ``my_sum`` of 1, 2 and 3 returns 6."""
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    # Create test table
    result = create_duckdb_result()
    query_p = get_unicode_data_p(
        "CREATE TABLE t AS SELECT * FROM (VALUES (1), (2), (3)) AS t(v)")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_result(result.ctypes.data)

    # Create and configure aggregate function
    func_p = ducklib.duckdb_create_aggregate_function()
    assert func_p != 0

    name_p = get_unicode_data_p("my_sum")
    ducklib.duckdb_aggregate_function_set_name(func_p, name_p)

    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    bigint_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_BIGINT)
    ducklib.duckdb_aggregate_function_add_parameter(func_p, int_type_p)
    ducklib.duckdb_aggregate_function_set_return_type(
        func_p, bigint_type_p)
    for tp in [int_type_p, bigint_type_p]:
        buf = numpy.array([tp], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)

    ducklib.duckdb_aggregate_function_set_functions(
        func_p, _agg_state_size_cb.address, _agg_init_cb.address,
        _agg_update_cb.address, _agg_combine_cb.address,
        _agg_finalize_cb.address
    )

    rc = ducklib.duckdb_register_aggregate_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess

    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_aggregate_function(func_buf.ctypes.data)

    # Query using the UDAF
    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT my_sum(v) FROM t")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess, f"Query failed, rc={rc}"

    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    data_p = ducklib.duckdb_vector_get_data(vec_p)
    val = (ctypes.c_int64 * 1).from_address(data_p)[0]
    assert val == 6, f"Expected 6, got {val}"

    chunk_buf = numpy.array([chunk_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_data_chunk(chunk_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


# numba compiles the callback below when the module is imported and freezes _INIT_CB_FIRED_ADDR into it as a
# constant. A reset must therefore write into the existing buffer. If _INIT_CB_FIRED were bound to a new array, the
# old one would be freed, and the callback would write into freed memory from a DuckDB worker thread.
_INIT_CB_FIRED = numpy.zeros(1, dtype=numpy.int64)
_INIT_CB_FIRED_ADDR = _INIT_CB_FIRED.ctypes.data


@njit
def _init_cb_impl(info):
    marker = carray(
        _cast_int_to_void_p(_INIT_CB_FIRED_ADDR), (1,), dtype=numpy.int64)
    marker[0] += 1


@cfunc(nb_types.void(nb_types.intp))
def _init_cb(info):
    _init_cb_impl(info)


@pytest.mark.skipif(
    not hasattr(ducklib.duckdb_lib, 'duckdb_scalar_function_set_init'),
    reason="duckdb_scalar_function_set_init not available",
)
def test_scalar_function_set_init():
    """Check that DuckDB calls the init callback set with ``duckdb_scalar_function_set_init``, which duckdb 1.5 added.

    The callback increments a counter that the test reads after the query. A binding that dropped the init pointer
    would leave the counter at 0.

    The test asserts a count of at least one, not exactly one. duckdb.h documents that the init callback runs once for
    each worker thread that executes the function, so the count depends on the query plan, not on the binding. The
    one-row constant projection here runs it once, but another plan may run it more often.
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]
    _INIT_CB_FIRED[0] = 0  # reset in place; see the note above _INIT_CB_FIRED

    func_p = ducklib.duckdb_create_scalar_function()
    ducklib.duckdb_scalar_function_set_name(
        func_p, get_unicode_data_p("with_init"))

    int_type_p = ducklib.duckdb_create_logical_type(
        ducklib.DUCKDB_TYPE_INTEGER)
    ducklib.duckdb_scalar_function_add_parameter(
        func_p, int_type_p)
    ducklib.duckdb_scalar_function_set_return_type(
        func_p, int_type_p)
    type_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)

    ducklib.duckdb_scalar_function_set_function(
        func_p, _add_one_cb.address)
    ducklib.duckdb_scalar_function_set_init(
        func_p, _init_cb.address)

    rc = ducklib.duckdb_register_scalar_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess

    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)

    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT with_init(42)")
    rc = ducklib.duckdb_query(
        conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess

    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    data_p = ducklib.duckdb_vector_get_data(vec_p)
    val = (ctypes.c_int32 * 1).from_address(data_p)[0]
    assert val == 43, f"Expected 43, got {val}"
    assert _INIT_CB_FIRED[0] >= 1, \
        f"init callback never fired, marker={_INIT_CB_FIRED[0]}"

    chunk_buf = numpy.array([chunk_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_data_chunk(chunk_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


def test_aggregate_function_set_overloads():
    """Check that an aggregate function set registers ``my_sum2`` for INTEGER and BIGINT.

    DuckDB calls the INTEGER variant for an INTEGER argument and the BIGINT variant for a BIGINT argument.
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    result = create_duckdb_result()
    query_p = get_unicode_data_p(
        "CREATE TABLE t2 AS SELECT * FROM"
        " (VALUES (1), (2), (3)) AS t(v)")
    rc = ducklib.duckdb_query(
        conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_result(result.ctypes.data)


    # Integer input variant, with the callbacks of test_aggregate_function_round_trip
    int_func_p = ducklib.duckdb_create_aggregate_function()
    ducklib.duckdb_aggregate_function_set_name(
        int_func_p, get_unicode_data_p("my_sum2"))
    int_type_p = ducklib.duckdb_create_logical_type(
        ducklib.DUCKDB_TYPE_INTEGER)
    bigint_type_p = ducklib.duckdb_create_logical_type(
        ducklib.DUCKDB_TYPE_BIGINT)
    ducklib.duckdb_aggregate_function_add_parameter(
        int_func_p, int_type_p)
    ducklib.duckdb_aggregate_function_set_return_type(
        int_func_p, bigint_type_p)
    ducklib.duckdb_aggregate_function_set_functions(
        int_func_p, _agg_state_size_cb.address,
        _agg_init_cb.address, _agg_update_cb.address,
        _agg_combine_cb.address, _agg_finalize_cb.address)

    # Bigint input variant
    big_func_p = ducklib.duckdb_create_aggregate_function()
    ducklib.duckdb_aggregate_function_set_name(
        big_func_p, get_unicode_data_p("my_sum2"))
    ducklib.duckdb_aggregate_function_add_parameter(
        big_func_p, bigint_type_p)
    ducklib.duckdb_aggregate_function_set_return_type(
        big_func_p, bigint_type_p)
    ducklib.duckdb_aggregate_function_set_functions(
        big_func_p, _agg_state_size_cb.address,
        _agg_init_cb.address, _agg_update_i64_cb.address,
        _agg_combine_cb.address, _agg_finalize_cb.address)

    for tp in [int_type_p, bigint_type_p]:
        buf = numpy.array([tp], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)

    # Create set, add both, register
    set_p = ducklib.duckdb_create_aggregate_function_set(
        get_unicode_data_p("my_sum2"))
    rc = ducklib.duckdb_add_aggregate_function_to_set(
        set_p, int_func_p)
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_add_aggregate_function_to_set(
        set_p, big_func_p)
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_register_aggregate_function_set(
        conn_p, set_p)
    assert rc == ducklib.DuckDBSuccess

    for p in [int_func_p, big_func_p]:
        buf = numpy.array([p], dtype=numpy.intp)
        ducklib.duckdb_destroy_aggregate_function(
            buf.ctypes.data)
    set_buf = numpy.array([set_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_aggregate_function_set(
        set_buf.ctypes.data)

    # Test integer input variant
    result = create_duckdb_result()
    rc = ducklib.duckdb_query(
        conn_p,
        get_unicode_data_p(
            "SELECT my_sum2(v::INTEGER) FROM t2"),
        result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    data_p = ducklib.duckdb_vector_get_data(vec_p)
    val = (ctypes.c_int64 * 1).from_address(data_p)[0]
    assert val == 6, f"Expected 6, got {val}"
    chunk_buf = numpy.array([chunk_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_data_chunk(chunk_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)

    # Test bigint input variant
    result = create_duckdb_result()
    rc = ducklib.duckdb_query(
        conn_p,
        get_unicode_data_p(
            "SELECT my_sum2(v::BIGINT) FROM t2"),
        result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    data_p = ducklib.duckdb_vector_get_data(vec_p)
    val = (ctypes.c_int64 * 1).from_address(data_p)[0]
    assert val == 6, f"Expected 6, got {val}"
    chunk_buf = numpy.array([chunk_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_data_chunk(chunk_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)

    aux_close_db(duckdb_database, duckdb_connection)


@njit
def _isqrt_impl(info, chunk, output):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    input_vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(input_vec)
    out_data = ducklib.duckdb_vector_get_data(output)
    in_arr = carray(_cast_int_to_void_p(in_data), (n,), dtype=numpy.int32)
    out_arr = carray(_cast_int_to_void_p(out_data), (n,), dtype=numpy.int32)
    for i in range(n):
        x = numpy.int32(in_arr[i])
        guess = x
        for _ in range(16):
            next_g = (guess + x // max(guess, numpy.int32(1))) // numpy.int32(2)
            if next_g >= guess:
                break
            guess = next_g
        out_arr[i] = guess


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _isqrt_cb(info, chunk, output):
    _isqrt_impl(info, chunk, output)


def test_hybrid_jit_udf_on_python_connection():
    """Check that SQL on a connection of the duckdb Python package calls a jitted UDF that numbduck registered on it."""
    from numbduck.pybridge import extract_connection_ptr

    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE nums AS SELECT range::INTEGER + 1 AS x FROM range(10)")
    conn_ptr = extract_connection_ptr(conn)

    func_p = ducklib.duckdb_create_scalar_function()
    assert func_p != 0

    ducklib.duckdb_scalar_function_set_name(
        func_p, get_unicode_data_p("jit_isqrt"))

    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    ducklib.duckdb_scalar_function_add_parameter(func_p, int_type_p)
    ducklib.duckdb_scalar_function_set_return_type(func_p, int_type_p)
    type_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)

    ducklib.duckdb_scalar_function_set_function(func_p, _isqrt_cb.address)

    rc = ducklib.duckdb_register_scalar_function(conn_ptr, func_p)
    assert rc == ducklib.DuckDBSuccess

    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)

    rows = conn.execute(
        "SELECT x, jit_isqrt(x) FROM nums").fetchall()

    for x, got in rows:
        expected = int(x ** 0.5)
        assert got == expected, (
            f"jit_isqrt({x}): expected {expected}, got {got}")

    conn.close()


@njit
def _triple_impl(info, chunk, output):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    input_vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(input_vec)
    out_data = ducklib.duckdb_vector_get_data(output)
    in_arr = carray(_cast_int_to_void_p(in_data), (n,), dtype=numpy.int32)
    out_arr = carray(_cast_int_to_void_p(out_data), (n,), dtype=numpy.int32)
    for i in range(n):
        out_arr[i] = in_arr[i] * numpy.int32(3)


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _triple_cb(info, chunk, output):
    _triple_impl(info, chunk, output)


def test_jit_udf_vs_python_udf():
    """Check that a jitted UDF and a Python UDF that triple an integer return the same values for 100 rows."""
    from numbduck.pybridge import extract_connection_ptr

    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE vals AS SELECT range::INTEGER + 1 AS x FROM range(100)")
    conn_ptr = extract_connection_ptr(conn)

    conn.create_function(
        "py_triple",
        lambda x: x * 3,
        ["INTEGER"],
        "INTEGER",
    )

    func_p = ducklib.duckdb_create_scalar_function()
    assert func_p != 0

    ducklib.duckdb_scalar_function_set_name(
        func_p, get_unicode_data_p("jit_triple"))

    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    ducklib.duckdb_scalar_function_add_parameter(func_p, int_type_p)
    ducklib.duckdb_scalar_function_set_return_type(func_p, int_type_p)
    type_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)

    ducklib.duckdb_scalar_function_set_function(func_p, _triple_cb.address)

    rc = ducklib.duckdb_register_scalar_function(conn_ptr, func_p)
    assert rc == ducklib.DuckDBSuccess

    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)

    rows = conn.execute(
        "SELECT py_triple(x), jit_triple(x) FROM vals").fetchall()

    assert len(rows) == 100
    for py_val, jit_val in rows:
        assert py_val == jit_val, (
            f"Mismatch: py_triple={py_val}, jit_triple={jit_val}")

    conn.remove_function("py_triple")
    conn.close()


@njit
def _bench_square_impl(info, chunk, output):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    input_vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(input_vec)
    out_data = ducklib.duckdb_vector_get_data(output)
    in_arr = carray(_cast_int_to_void_p(in_data), (n,), dtype=numpy.int64)
    out_arr = carray(_cast_int_to_void_p(out_data), (n,), dtype=numpy.int64)
    for i in range(n):
        out_arr[i] = in_arr[i] * in_arr[i]


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _bench_square_cb(info, chunk, output):
    _bench_square_impl(info, chunk, output)


@pytest.mark.benchmark
def test_udf_benchmark(capsys):
    """Time x*x with a Python scalar UDF, a Python Arrow UDF and a jitted numbduck UDF, and check that the sums agree.

    The test runs over 10,000, 100,000 and 1,000,000 rows. DuckDB calls the jitted UDF through a C function pointer,
    with no Python code per call. The Python UDFs go through the interpreter: the scalar UDF once per row, and the Arrow
    UDF once per batch, through pyarrow. conftest.py skips the test unless ``-m`` names ``benchmark``.

    Under duckdb 1.x, the Python column depends on pandas. On duckdb 1.5.6 without pandas installed, the scalar UDF
    took 113 s for a million rows on the Linux machine below. With pandas 3.1.1 installed, the scalar UDF took 0.10 s
    for a million rows.

    One run gave these times (Linux x86_64, Python 3.12.3, duckdb 2.0.0, numba 0.66.0, pyarrow 26.0.1, pandas 3.1.1)::

             Rows      Python       Arrow         JIT    Py/JIT   Arr/JIT
           10,000     0.0011s     0.0005s     0.0006s        2x        1x
          100,000     0.0082s     0.0039s     0.0008s       10x        5x
        1,000,000     0.0892s     0.0420s     0.0010s       89x       42x

    MacBook Pro M1
    env: python 3.12.13, duckdb 1.5.6, numba 0.66.0, numpy 2.4.6, arm64
    (requires pandas to avoid suboptimal type checking; installed pandas==3.1.1)

             Rows      Python       Arrow         JIT    Py/JIT   Arr/JIT
           10,000     0.0020s     0.0013s     0.0001s       15x       10x
          100,000     0.0190s     0.0070s     0.0004s       48x       18x
        1,000,000     0.1468s     0.2146s     0.0014s      104x      152x

    MacBook Pro M1
    env: python 3.12.13, duckdb 2.0.0, numba 0.66.0, numpy 2.4.6, arm64

             Rows      Python       Arrow         JIT    Py/JIT   Arr/JIT
           10,000     0.0012s     0.0005s     0.0002s        6x        2x
          100,000     0.0102s     0.0015s     0.0003s       33x        5x
        1,000,000     0.1005s     0.0598s     0.0008s      130x       78x
    """
    import sys
    import time
    from numbduck.pybridge import extract_connection_ptr
    from numbduck.utils import complete_open_result

    ROW_COUNTS = [10_000, 100_000, 1_000_000]

    conn = duckdb.connect()

    # Register Python scalar UDF
    conn.create_function(
        "py_square", lambda x: x * x, ["BIGINT"], "BIGINT")

    # Register Python Arrow UDF
    has_arrow = True
    try:
        import pyarrow.compute as pc
        conn.create_function(
            "arrow_square",
            lambda x: pc.multiply(x, x),
            ["BIGINT"], "BIGINT",
            type="arrow")
    except ImportError:
        has_arrow = False

    # Register numbduck JIT UDF
    conn_ptr = extract_connection_ptr(conn)
    func_p = ducklib.duckdb_create_scalar_function()
    ducklib.duckdb_scalar_function_set_name(
        func_p, get_unicode_data_p("jit_square"))
    bigint_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_BIGINT)
    ducklib.duckdb_scalar_function_add_parameter(func_p, bigint_type_p)
    ducklib.duckdb_scalar_function_set_return_type(func_p, bigint_type_p)
    type_buf = numpy.array([bigint_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)
    ducklib.duckdb_scalar_function_set_function(
        func_p, _bench_square_cb.address)
    rc = ducklib.duckdb_register_scalar_function(conn_ptr, func_p)
    assert rc == ducklib.DuckDBSuccess
    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)

    # Warm up each UDF with a small query before timing
    conn.execute("CREATE TABLE warmup AS SELECT 1::BIGINT AS x")
    conn.execute("SELECT py_square(x) FROM warmup").fetchone()
    if has_arrow:
        conn.execute("SELECT arrow_square(x) FROM warmup").fetchone()
    conn.execute("SELECT jit_square(x) FROM warmup").fetchone()
    conn.execute("DROP TABLE warmup")

    results = []
    for N in ROW_COUNTS:
        conn.execute(
            f"CREATE OR REPLACE TABLE bench "
            f"AS SELECT range::BIGINT + 1 AS x FROM range({N})")

        t0 = time.perf_counter()
        py_result = conn.execute(
            "SELECT SUM(py_square(x)) FROM bench").fetchone()
        t_python = time.perf_counter() - t0

        if has_arrow:
            t0 = time.perf_counter()
            arrow_result = conn.execute(
                "SELECT SUM(arrow_square(x)) FROM bench").fetchone()
            t_arrow = time.perf_counter() - t0
            assert arrow_result == py_result
        else:
            t_arrow = None

        t0 = time.perf_counter()
        jit_result = conn.execute(
            "SELECT SUM(jit_square(x)) FROM bench").fetchone()
        t_jit = time.perf_counter() - t0

        assert jit_result == py_result
        results.append((N, t_python, t_arrow, t_jit))

    complete_open_result(conn)
    conn.remove_function("py_square")
    if has_arrow:
        conn.remove_function("arrow_square")
    conn.close()

    with capsys.disabled():
        pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(f"\n  UDF benchmark (x*x, Python {pyver}, duckdb {duckdb.__version__}):")
        print(f"    {'Rows':>10s}  {'Python':>10s}  {'Arrow':>10s}"
              f"  {'JIT':>10s}  {'Py/JIT':>8s}  {'Arr/JIT':>8s}")
        for N, t_python, t_arrow, t_jit in results:
            arrow_str = f"{t_arrow:.4f}s" if t_arrow is not None else "n/a"
            arr_ratio = f"{t_arrow / t_jit:.0f}x" if t_arrow else "n/a"
            print(f"    {N:>10,d}  {t_python:>9.4f}s  {arrow_str:>10s}"
                  f"  {t_jit:>9.4f}s  {t_python / t_jit:>7.0f}x"
                  f"  {arr_ratio:>8s}")


# ---- Structrefs kept in DuckDB's state slots as raw MemInfo pointers ----
#
# test_ducklib.md explains the DuckDB aggregate lifecycle, how numba's removerefctpass bears on numbox's MemInfo
# helpers, and why release_meminfo calls NRT_MemInfo_release while the other helpers do not.
#
# DuckDB owns each group's state slot, 8 bytes of memory here. The init callback writes a MemInfo pointer into the
# slot. A MemInfo is the header through which numba's runtime, NRT, manages the memory of a structref or an array with
# a reference count. The init callback takes one more reference, so NRT does not free the memory when the callback
# returns and its local variables go out of scope.
#
# How numbox's helpers change the count:
#   export_meminfo(s):   +1; the returned intp keeps the allocation alive.
#   borrow_structref():  +1 on entry, which numba's release of the local structref after its last use balances, so
#                        the count held for the slot does not change.
#   release_meminfo(p):  -1; the destructor runs when the count reaches 0.


# The intrinsic commented out below reads a MemInfo's reference count from jitted code, and it cannot check the counts
# step by step. It reads 1 inside the @njit function that exports a structref, never the 2 that the local structref
# and the exported pointer hold together. numba releases a local immediately after its last use, so a structref whose
# last use is the export has already dropped its reference when the count is read. If the structref stays in use,
# numba's reference-count pruning pass removes the export's increment together with the local's later decrement. With
# that pass off (NUMBA_LLVM_REFPRUNE_PASS=0), the read returns 2 on numba 0.65.1. Either way the net change is zero,
# and the object stays alive. numba's removerefctpass has no effect here: it does not run in a function that calls a
# structref constructor, because numba counts the references of the constructor's result.
#
# The two tests below whose names end in refcount_ladder therefore read the count from Python, with _read_refcount,
# between separate @njit calls. The intrinsic stays here, commented out, as an example of a check that cannot work.
# @intrinsic
# def _refcount_of_meminfo(typingctx, p_ty):
#     """Read MemInfo refcount from JIT code. MemInfo.refct is the first field."""
#     sig = nb_types.intp(p_ty)
#
#     def codegen(context, builder, signature, args):
#         ptr_ty = llir.IntType(8).as_pointer()
#         meminfo = builder.inttoptr(args[0], ptr_ty)
#         rc_ptr = builder.bitcast(meminfo, cgutils.intp_t.as_pointer())
#         return builder.load(rc_ptr)
#     return sig, codegen
#
#
# @njit
# def refcount_of_meminfo(p):
#     return _refcount_of_meminfo(p)


# ---- Welford state structref ----

@structref.register
class WelfordStateType(nb_types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


class WelfordState(structref.StructRefProxy):
    def __new__(cls, mean, count, m2):
        return structref.StructRefProxy.__new__(cls, mean, count, m2)

    @property
    def mean(self):
        return _welford_get_mean(self)

    @property
    def count(self):
        return _welford_get_count(self)

    @property
    def m2(self):
        return _welford_get_m2(self)


@njit
def _welford_get_mean(s):
    return s.mean


@njit
def _welford_get_count(s):
    return s.count


@njit
def _welford_get_m2(s):
    return s.m2


structref.define_proxy(
    WelfordState, WelfordStateType, ["mean", "count", "m2"])

welford_type = WelfordStateType(
    [("mean", nb_types.float64),
     ("count", nb_types.int64),
     ("m2", nb_types.float64)])


def _read_refcount(meminfo_intp):
    """Return the reference count of the MemInfo at the address ``meminfo_intp``, read from Python with ctypes.

    ``refct``, a ``size_t``, is the first field of numba's MemInfo structure, and has been since numba 0.50. The
    structure is defined in numba/core/runtime/nrt.cpp.
    """
    return int(ctypes.c_size_t.from_address(meminfo_intp).value)


# ---- A structref that owns heap memory, for the test of its destructor ----

@structref.register
class _HeapStateType(nb_types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


class _HeapState(structref.StructRefProxy):
    def __new__(cls, values):
        return structref.StructRefProxy.__new__(cls, values)


structref.define_proxy(_HeapState, _HeapStateType, ["values"])

_heap_state_type = _HeapStateType(
    [("values", nb_types.ListType(nb_types.float64))])


def test_structref_meminfo_bridge_refcount_ladder():
    """Check, step by step, the reference count that export, borrow and release leave on a structref's MemInfo.

    Each step runs in an @njit function of its own. numba has released the function's local references when it
    returns, so Python reads the count between the calls.
    """

    @njit
    def _allocate_and_export():
        s = WelfordState(1.5, 2, 3.25)
        p = export_meminfo(s)
        return p

    @njit
    def _borrow_and_read(p):
        s = borrow_structref(welford_type, p)
        return s.mean, s.count, s.m2

    @njit
    def _do_release(p):
        release_meminfo(p)

    # The comment above the commented-out refcount_of_meminfo explains why jitted code cannot read the count of 2
    # inside a step, so Python reads the count between the steps.

    # Step 1: allocate and export. After the return, the count is 1, the reference that the export took.
    p = _allocate_and_export()
    assert p != 0, "export_meminfo returned null"
    rc = _read_refcount(p)
    assert rc == 1, f"expected refcount 1 after export+local-drop, got {rc}"

    # Step 2: borrow and read the fields. The borrow's reference ends with the function, so the count is still 1.
    mean, count, m2 = _borrow_and_read(p)
    assert mean == 1.5, f"mean={mean}"
    assert count == 2, f"count={count}"
    assert m2 == 3.25, f"m2={m2}"
    rc = _read_refcount(p)
    assert rc == 1, f"expected refcount 1 after borrow-drop, got {rc}"

    # Step 3: release the exported reference. The count drops to 0, and the destructor runs.
    _do_release(p)


def test_array_meminfo_bridge_refcount_ladder():
    """Check that an array exported by its allocating function stays alive with a count of 1 and can be read back.

    The allocating function takes the extra reference with numbox's ``_incref_meminfo`` intrinsic, and a second
    function reads the data back through ``carray``. numba's removerefctpass does not run in the allocating function,
    because the array's allocation calls an NRT allocation function there, so the pass leaves the increment in place.
    The increment and the release of the local array when the function returns cancel out, which leaves the
    allocation's count of 1. ``export_meminfo(arr)`` would take the reference as well: with numbox 0.7.8 and later,
    ``_incref_meminfo`` keeps removerefctpass from removing the increment inside ``export_meminfo`` too.
    """

    @njit
    def _allocate_and_export():
        arr = numpy.zeros(5, dtype=numpy.float64)
        for i in range(5):
            arr[i] = float(i * 10)
        meminfo_p, data_p = structref_meminfo(arr)
        _incref_meminfo(meminfo_p)
        return meminfo_p, data_p

    @njit
    def _read_via_carray(data_p):
        view = carray(_cast_int_to_void_p(data_p), (5,), dtype=numpy.float64)
        return view[0], view[2], view[4]

    @njit
    def _do_release(p):
        release_meminfo(p)

    meminfo_p, data_p = _allocate_and_export()
    assert meminfo_p != 0, "meminfo_p is null"
    rc = _read_refcount(meminfo_p)
    assert rc == 1, f"expected refcount 1 after export+local-drop, got {rc}"

    v0, v2, v4 = _read_via_carray(data_p)
    assert v0 == 0.0, f"v0={v0}"
    assert v2 == 20.0, f"v2={v2}"
    assert v4 == 40.0, f"v4={v4}"
    rc = _read_refcount(meminfo_p)
    assert rc == 1, f"expected refcount 1 after carray read, got {rc}"

    _do_release(meminfo_p)


# ---- Callbacks of the array-backed variance UDAF ----
#
# This UDAF keeps its state in a numpy array instead of a structref: three float64 values, the sum, the sum of squares
# and the count. Each group's 16-byte state slot holds two intp values. The first is the address of the array's
# MemInfo, for the destroy callback to release. The second is the address of the array's data, for the other callbacks
# to read and write through carray. The init callback takes the reference with numbox's _incref_meminfo intrinsic,
# which numba inlines into the function that allocates the array. numba's removerefctpass does not run in that
# function, because numpy.zeros calls an NRT allocation function there.

_ARR_STATE_SLOTS = 2  # meminfo_p, data_p
_ARR_STATE_ELEMS = 3  # sum, sum_sq, count
_ARR_IDX_SUM = 0
_ARR_IDX_SUM_SQ = 1
_ARR_IDX_COUNT = 2


@njit
def _arr_state_size_impl(info):
    return numpy.uint64(16)


@cfunc(nb_types.uint64(nb_types.intp))
def _arr_state_size_cb(info):
    return _arr_state_size_impl(info)


@njit
def _arr_init_impl(info, state):
    arr = numpy.zeros(_ARR_STATE_ELEMS, dtype=numpy.float64)
    meminfo_p, data_p = structref_meminfo(arr)
    _incref_meminfo(meminfo_p)
    slot = carray(
        _cast_int_to_void_p(state), (_ARR_STATE_SLOTS,), dtype=numpy.intp)
    slot[0] = meminfo_p
    slot[1] = data_p


@cfunc(nb_types.void(nb_types.intp, nb_types.intp))
def _arr_init_cb(info, state):
    _arr_init_impl(info, state)


@njit
def _arr_update_impl(info, chunk, states):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(vec)
    state_slots = carray(
        _cast_int_to_void_p(states), (n,), dtype=numpy.intp)
    in_vals = carray(
        _cast_int_to_void_p(in_data), (n,), dtype=numpy.float64)
    for i in range(n):
        slot = carray(
            _cast_int_to_void_p(state_slots[i]),
            (_ARR_STATE_SLOTS,), dtype=numpy.intp)
        s = carray(
            _cast_int_to_void_p(slot[1]),
            (_ARR_STATE_ELEMS,), dtype=numpy.float64)
        s[_ARR_IDX_SUM] += in_vals[i]
        s[_ARR_IDX_SUM_SQ] += in_vals[i] * in_vals[i]
        s[_ARR_IDX_COUNT] += 1.0


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _arr_update_cb(info, chunk, states):
    _arr_update_impl(info, chunk, states)


@njit
def _arr_combine_impl(info, source, target, count):
    src_slots = carray(
        _cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    tgt_slots = carray(
        _cast_int_to_void_p(target), (count,), dtype=numpy.intp)
    for i in range(count):
        src_slot = carray(
            _cast_int_to_void_p(src_slots[i]),
            (_ARR_STATE_SLOTS,), dtype=numpy.intp)
        tgt_slot = carray(
            _cast_int_to_void_p(tgt_slots[i]),
            (_ARR_STATE_SLOTS,), dtype=numpy.intp)
        src = carray(
            _cast_int_to_void_p(src_slot[1]),
            (_ARR_STATE_ELEMS,), dtype=numpy.float64)
        tgt = carray(
            _cast_int_to_void_p(tgt_slot[1]),
            (_ARR_STATE_ELEMS,), dtype=numpy.float64)
        tgt[_ARR_IDX_SUM] += src[_ARR_IDX_SUM]
        tgt[_ARR_IDX_SUM_SQ] += src[_ARR_IDX_SUM_SQ]
        tgt[_ARR_IDX_COUNT] += src[_ARR_IDX_COUNT]


@cfunc(nb_types.void(nb_types.intp, nb_types.intp,
                     nb_types.intp, nb_types.uint64))
def _arr_combine_cb(info, source, target, count):
    _arr_combine_impl(info, source, target, count)


@njit
def _arr_finalize_impl(info, source, result, count, offset):
    out_data = ducklib.duckdb_vector_get_data(result)
    src_slots = carray(
        _cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    out_vals = carray(
        _cast_int_to_void_p(out_data), (offset + count,),
        dtype=numpy.float64)
    for i in range(count):
        src_slot = carray(
            _cast_int_to_void_p(src_slots[i]),
            (_ARR_STATE_SLOTS,), dtype=numpy.intp)
        s = carray(
            _cast_int_to_void_p(src_slot[1]),
            (_ARR_STATE_ELEMS,), dtype=numpy.float64)
        n = s[_ARR_IDX_COUNT]
        if n < 2.0:
            out_vals[offset + i] = math.nan
        else:
            mean = s[_ARR_IDX_SUM] / n
            var = (s[_ARR_IDX_SUM_SQ] / n) - mean * mean
            out_vals[offset + i] = var


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp,
                     nb_types.uint64, nb_types.uint64))
def _arr_finalize_cb(info, source, result, count, offset):
    _arr_finalize_impl(info, source, result, count, offset)


@njit
def _arr_destroy_impl(states, count):
    state_slots = carray(
        _cast_int_to_void_p(states), (count,), dtype=numpy.intp)
    for i in range(count):
        slot = carray(
            _cast_int_to_void_p(state_slots[i]),
            (_ARR_STATE_SLOTS,), dtype=numpy.intp)
        release_meminfo(slot[0])


@cfunc(nb_types.void(nb_types.intp, nb_types.uint64))
def _arr_destroy_cb(states, count):
    _arr_destroy_impl(states, count)


@njit
def welford_update(s, x):
    s.count += 1
    delta = x - s.mean
    s.mean += delta / s.count
    delta2 = x - s.mean
    s.m2 += delta * delta2


@njit
def welford_combine(src, tgt):
    """Merge the Welford state ``src`` into ``tgt`` in place, with the pairwise formula of Chan et al."""
    if src.count == 0:
        return
    if tgt.count == 0:
        tgt.mean = src.mean
        tgt.count = src.count
        tgt.m2 = src.m2
        return
    new_count = src.count + tgt.count
    delta = src.mean - tgt.mean
    new_mean = tgt.mean + delta * src.count / new_count
    new_m2 = (tgt.m2 + src.m2
              + delta * delta * tgt.count * src.count / new_count)
    tgt.mean = new_mean
    tgt.count = new_count
    tgt.m2 = new_m2


@njit
def welford_finalize(s):
    if s.count < 2:
        return math.nan
    return math.sqrt(s.m2 / (s.count - 1))


# ---- Welford stddev UDAF callbacks for DuckDB ----

@njit
def _welford_state_size_impl(info):
    return numpy.uint64(8)


@cfunc(nb_types.uint64(nb_types.intp))
def _welford_state_size_cb(info):
    return _welford_state_size_impl(info)


@njit
def _welford_init_impl(info, state):
    s = WelfordState(0.0, 0, 0.0)
    p = export_meminfo(s)
    slot = carray(_cast_int_to_void_p(state), (1,), dtype=numpy.intp)
    slot[0] = p


@cfunc(nb_types.void(nb_types.intp, nb_types.intp))
def _welford_init_cb(info, state):
    _welford_init_impl(info, state)


# The update reads every row as valid, without the validity mask, because the test data holds no NULL. A UDAF for
# real data must read the mask with duckdb_vector_get_validity and skip the NULL rows.
@njit
def _welford_update_impl(info, chunk, states):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(vec)
    state_slots = carray(
        _cast_int_to_void_p(states), (n,), dtype=numpy.intp)
    in_vals = carray(
        _cast_int_to_void_p(in_data), (n,), dtype=numpy.float64)
    for i in range(n):
        slot = carray(
            _cast_int_to_void_p(state_slots[i]), (1,), dtype=numpy.intp)
        s = borrow_structref(welford_type, slot[0])
        welford_update(s, in_vals[i])


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _welford_update_cb(info, chunk, states):
    _welford_update_impl(info, chunk, states)


@njit
def _welford_combine_impl(info, source, target, count):
    src_slots = carray(
        _cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    tgt_slots = carray(
        _cast_int_to_void_p(target), (count,), dtype=numpy.intp)
    for i in range(count):
        src_slot = carray(
            _cast_int_to_void_p(src_slots[i]), (1,), dtype=numpy.intp)
        tgt_slot = carray(
            _cast_int_to_void_p(tgt_slots[i]), (1,), dtype=numpy.intp)
        src = borrow_structref(welford_type, src_slot[0])
        tgt = borrow_structref(welford_type, tgt_slot[0])
        welford_combine(src, tgt)


@cfunc(nb_types.void(nb_types.intp, nb_types.intp,
                     nb_types.intp, nb_types.uint64))
def _welford_combine_cb(info, source, target, count):
    _welford_combine_impl(info, source, target, count)


@njit
def _welford_finalize_impl(info, source, result, count, offset):
    out_data = ducklib.duckdb_vector_get_data(result)
    src_slots = carray(
        _cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    out_vals = carray(
        _cast_int_to_void_p(out_data), (offset + count,),
        dtype=numpy.float64)
    for i in range(count):
        src_slot = carray(
            _cast_int_to_void_p(src_slots[i]), (1,), dtype=numpy.intp)
        s = borrow_structref(welford_type, src_slot[0])
        out_vals[offset + i] = welford_finalize(s)


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp,
                     nb_types.uint64, nb_types.uint64))
def _welford_finalize_cb(info, source, result, count, offset):
    _welford_finalize_impl(info, source, result, count, offset)


@njit
def _welford_destroy_impl(states, count):
    state_slots = carray(
        _cast_int_to_void_p(states), (count,), dtype=numpy.intp)
    for i in range(count):
        slot = carray(
            _cast_int_to_void_p(state_slots[i]), (1,), dtype=numpy.intp)
        release_meminfo(slot[0])


@cfunc(nb_types.void(nb_types.intp, nb_types.uint64))
def _welford_destroy_cb(states, count):
    _welford_destroy_impl(states, count)


def test_welford_numba_only():
    """Check, without DuckDB, that the Welford update, combine and finalize match ``numpy.std`` with ``ddof=1``.

    The test also checks the combined mean, and that a single value gives NaN.
    """

    @njit
    def _compute_serial(xs):
        s = WelfordState(0.0, 0, 0.0)
        for x in xs:
            welford_update(s, x)
        return welford_finalize(s)

    @njit
    def _compute_combined(xs_a, xs_b):
        sa = WelfordState(0.0, 0, 0.0)
        sb = WelfordState(0.0, 0, 0.0)
        for x in xs_a:
            welford_update(sa, x)
        for x in xs_b:
            welford_update(sb, x)
        welford_combine(sa, sb)
        return welford_finalize(sb), sb.mean

    xs = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    expected = float(numpy.std(xs, ddof=1))

    serial = _compute_serial(xs)
    assert abs(serial - expected) < 1e-10, (
        f"serial: got {serial}, expected {expected}")

    combined, combined_mean = _compute_combined(xs[:3], xs[3:])
    assert abs(combined - expected) < 1e-10, (
        f"combined: got {combined}, expected {expected}")
    expected_mean = float(numpy.mean(xs))
    assert abs(combined_mean - expected_mean) < 1e-10, (
        f"combined mean: got {combined_mean}, expected {expected_mean}")

    single = _compute_serial(numpy.array([42.0]))
    assert math.isnan(single), f"expected NaN for count<2, got {single}"


def test_structref_meminfo_bridge_nested_heap():
    """Check that release_meminfo runs the destructor of a structref that owns heap memory, freeing its typed List too.

    numba's runtime counts its allocations and frees. One warm-up cycle of export and release measures the allocations
    of a cycle, and 100 more cycles must allocate and free exactly 100 times as much.
    """
    from numba.core.runtime import _nrt_python as _nrt
    _nrt.memsys_enable_stats()
    from numba.core.runtime.nrt import rtsys
    from numba.typed import List

    @njit
    def _alloc_and_export():
        lst = List.empty_list(nb_types.float64)
        for i in range(10):
            lst.append(float(i))
        s = _HeapState(lst)
        return export_meminfo(s)

    @njit
    def _release(p):
        release_meminfo(p)

    # One warm-up cycle measures the allocations of a cycle.
    stats_before_warmup = rtsys.get_allocation_stats()
    p = _alloc_and_export()
    _release(p)
    stats_after_warmup = rtsys.get_allocation_stats()

    warmup_alloc = stats_after_warmup.alloc - stats_before_warmup.alloc
    warmup_free = stats_after_warmup.free - stats_before_warmup.free
    assert warmup_alloc > 0, (
        f"warmup must allocate; got alloc delta {warmup_alloc}")
    assert warmup_alloc == warmup_free, (
        f"warmup leak: alloc={warmup_alloc}, free={warmup_free}")

    # 100 more cycles must allocate and free exactly 100 times as much as the warm-up.
    N = 100
    stats_before = rtsys.get_allocation_stats()
    for _ in range(N):
        p = _alloc_and_export()
        _release(p)
    stats_after = rtsys.get_allocation_stats()

    alloc_delta = stats_after.alloc - stats_before.alloc
    free_delta = stats_after.free - stats_before.free
    assert alloc_delta == N * warmup_alloc, (
        f"alloc count: got {alloc_delta}, expected {N * warmup_alloc}")
    assert free_delta == N * warmup_free, (
        f"free count: got {free_delta}, expected {N * warmup_free}")


def test_aggregate_function_structref_stddev():
    """Check that the structref-backed Welford UDAF computes the sample standard deviation in DuckDB.

    The test also checks that numba's runtime frees as many blocks as it allocates, from the query to the close of the
    database.
    """
    from numba.core.runtime import _nrt_python as _nrt
    _nrt.memsys_enable_stats()
    from numba.core.runtime.nrt import rtsys

    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    result = create_duckdb_result()
    query_p = get_unicode_data_p(
        "CREATE TABLE t_stddev AS SELECT * FROM "
        "(VALUES (1.0),(2.0),(3.0),(4.0),(5.0),(6.0),(7.0)) AS t(v)")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_result(result.ctypes.data)

    func_p = ducklib.duckdb_create_aggregate_function()
    assert func_p != 0

    name_p = get_unicode_data_p("welford_stddev")
    ducklib.duckdb_aggregate_function_set_name(func_p, name_p)

    dbl_type_p = ducklib.duckdb_create_logical_type(
        ducklib.DUCKDB_TYPE_DOUBLE)
    ducklib.duckdb_aggregate_function_add_parameter(func_p, dbl_type_p)
    ducklib.duckdb_aggregate_function_set_return_type(func_p, dbl_type_p)
    tp_buf = numpy.array([dbl_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(tp_buf.ctypes.data)

    ducklib.duckdb_aggregate_function_set_functions(
        func_p,
        _welford_state_size_cb.address,
        _welford_init_cb.address,
        _welford_update_cb.address,
        _welford_combine_cb.address,
        _welford_finalize_cb.address,
    )
    ducklib.duckdb_aggregate_function_set_destructor(
        func_p, _welford_destroy_cb.address)

    rc = ducklib.duckdb_register_aggregate_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess

    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_aggregate_function(func_buf.ctypes.data)

    stats_before = rtsys.get_allocation_stats()

    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT welford_stddev(v) FROM t_stddev")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess, f"Query failed, rc={rc}"

    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    data_p = ducklib.duckdb_vector_get_data(vec_p)
    val = (ctypes.c_double * 1).from_address(data_p)[0]

    xs = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    expected = float(numpy.std(xs, ddof=1))
    assert abs(val - expected) < 1e-10, (
        f"got {val}, expected {expected}")

    chunk_buf = numpy.array([chunk_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_data_chunk(chunk_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)

    aux_close_db(duckdb_database, duckdb_connection)

    stats_after = rtsys.get_allocation_stats()
    alloc_delta = stats_after.alloc - stats_before.alloc
    free_delta = stats_after.free - stats_before.free
    assert alloc_delta == free_delta, (
        f"leak: alloc={alloc_delta}, free={free_delta}")


def test_aggregate_function_array_variance():
    """Check that the array-backed UDAF computes the population variance in DuckDB, with no NRT allocation left over.

    The test repeats the steps of test_aggregate_function_structref_stddev, with a numpy array as the state instead of a
    structref. Each 16-byte state slot holds the address of the array's MemInfo and the address of its data, as the
    comment above ``_ARR_STATE_SLOTS`` describes.
    """
    from numba.core.runtime import _nrt_python as _nrt
    _nrt.memsys_enable_stats()
    from numba.core.runtime.nrt import rtsys

    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    result = create_duckdb_result()
    query_p = get_unicode_data_p(
        "CREATE TABLE t_var AS SELECT * FROM "
        "(VALUES (1.0),(2.0),(3.0),(4.0),(5.0),(6.0),(7.0)) AS t(v)")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_result(result.ctypes.data)

    func_p = ducklib.duckdb_create_aggregate_function()
    assert func_p != 0

    name_p = get_unicode_data_p("arr_variance")
    ducklib.duckdb_aggregate_function_set_name(func_p, name_p)

    dbl_type_p = ducklib.duckdb_create_logical_type(
        ducklib.DUCKDB_TYPE_DOUBLE)
    ducklib.duckdb_aggregate_function_add_parameter(func_p, dbl_type_p)
    ducklib.duckdb_aggregate_function_set_return_type(func_p, dbl_type_p)
    tp_buf = numpy.array([dbl_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(tp_buf.ctypes.data)

    ducklib.duckdb_aggregate_function_set_functions(
        func_p,
        _arr_state_size_cb.address,
        _arr_init_cb.address,
        _arr_update_cb.address,
        _arr_combine_cb.address,
        _arr_finalize_cb.address,
    )
    ducklib.duckdb_aggregate_function_set_destructor(
        func_p, _arr_destroy_cb.address)

    rc = ducklib.duckdb_register_aggregate_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess

    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_aggregate_function(func_buf.ctypes.data)

    stats_before = rtsys.get_allocation_stats()

    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT arr_variance(v) FROM t_var")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess, f"Query failed, rc={rc}"

    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    data_p = ducklib.duckdb_vector_get_data(vec_p)
    val = (ctypes.c_double * 1).from_address(data_p)[0]

    xs = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    expected = float(numpy.var(xs))
    assert abs(val - expected) < 1e-10, (
        f"got {val}, expected {expected}")

    chunk_buf = numpy.array([chunk_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_data_chunk(chunk_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)

    aux_close_db(duckdb_database, duckdb_connection)

    stats_after = rtsys.get_allocation_stats()
    alloc_delta = stats_after.alloc - stats_before.alloc
    free_delta = stats_after.free - stats_before.free
    assert alloc_delta == free_delta, (
        f"leak: alloc={alloc_delta}, free={free_delta}")


# --- UDFs over more than one chunk, and over NULL input ---

STANDARD_VECTOR_SIZE = 2048
NULL_SENTINEL = -999


@njit
def _null_to_sentinel_impl(info, chunk, output):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(vec)
    validity_p = ducklib.duckdb_vector_get_validity(vec)
    out_data = ducklib.duckdb_vector_get_data(output)
    in_arr = carray(_cast_int_to_void_p(in_data), (n,), dtype=numpy.int32)
    out_arr = carray(_cast_int_to_void_p(out_data), (n,), dtype=numpy.int32)
    for i in range(n):
        # A validity pointer of 0 means that every row of the vector is valid; otherwise row i is valid if bit i of the
        # mask is set.
        valid = validity_p == 0 or ducklib.duckdb_validity_row_is_valid(
            validity_p, i) != 0
        out_arr[i] = in_arr[i] if valid else NULL_SENTINEL


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _null_to_sentinel_cb(info, chunk, output):
    _null_to_sentinel_impl(info, chunk, output)


@njit
def _null_count_state_size_impl(info):
    return numpy.uint64(8)


@cfunc(nb_types.uint64(nb_types.intp))
def _null_count_state_size_cb(info):
    return _null_count_state_size_impl(info)


@njit
def _null_count_init_impl(info, state):
    carray(_cast_int_to_void_p(state), (1,), dtype=numpy.int64)[0] = 0


@cfunc(nb_types.void(nb_types.intp, nb_types.intp))
def _null_count_init_cb(info, state):
    _null_count_init_impl(info, state)


@njit
def _null_count_update_impl(info, chunk, states):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    validity_p = ducklib.duckdb_vector_get_validity(vec)
    state_ptrs = carray(_cast_int_to_void_p(states), (n,), dtype=numpy.intp)
    for i in range(n):
        valid = validity_p == 0 or ducklib.duckdb_validity_row_is_valid(
            validity_p, i) != 0
        if valid:
            acc = carray(_cast_int_to_void_p(state_ptrs[i]), (1,), dtype=numpy.int64)
            acc[0] += 1


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _null_count_update_cb(info, chunk, states):
    _null_count_update_impl(info, chunk, states)


@njit
def _null_count_combine_impl(info, source, target, count):
    src_ptrs = carray(_cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    tgt_ptrs = carray(_cast_int_to_void_p(target), (count,), dtype=numpy.intp)
    for i in range(count):
        src_acc = carray(_cast_int_to_void_p(src_ptrs[i]), (1,), dtype=numpy.int64)[0]
        tgt_acc = carray(_cast_int_to_void_p(tgt_ptrs[i]), (1,), dtype=numpy.int64)
        tgt_acc[0] += src_acc


@cfunc(nb_types.void(nb_types.intp, nb_types.intp,
                     nb_types.intp, nb_types.uint64))
def _null_count_combine_cb(info, source, target, count):
    _null_count_combine_impl(info, source, target, count)


@njit
def _null_count_finalize_impl(info, source, result, count, offset):
    out_data = ducklib.duckdb_vector_get_data(result)
    src_ptrs = carray(_cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    out_vals = carray(
        _cast_int_to_void_p(out_data), (offset + count,), dtype=numpy.int64)
    for i in range(count):
        acc = carray(_cast_int_to_void_p(src_ptrs[i]), (1,), dtype=numpy.int64)[0]
        out_vals[offset + i] = acc


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp,
                     nb_types.uint64, nb_types.uint64))
def _null_count_finalize_cb(info, source, result, count, offset):
    _null_count_finalize_impl(info, source, result, count, offset)


@balanced(RESULT_FAMILY, CHUNK_FAMILY)
def test_scalar_function_multi_chunk():
    """Check that a scalar UDF returns the right values over 5000 rows, which take more than one chunk of 2048 rows.

    DuckDB calls the callback once for each chunk. The test fetches the result chunk by chunk and checks every value,
    including the two on each side of the first chunk boundary.
    """
    n_rows = 5000
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    result = create_duckdb_result()
    # get_unicode_data_p returns a pointer into the string without a copy. A string literal lives as long as the code
    # that holds it, but a string built at run time needs a name that keeps it alive until duckdb_query has read it.
    create_sql = f"CREATE TABLE t_mc AS SELECT range::INTEGER AS v FROM range({n_rows})"
    query_p = get_unicode_data_p(create_sql)
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_result(result.ctypes.data)

    func_p = ducklib.duckdb_create_scalar_function()
    ducklib.duckdb_scalar_function_set_name(func_p, get_unicode_data_p("add_one"))
    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    ducklib.duckdb_scalar_function_add_parameter(func_p, int_type_p)
    ducklib.duckdb_scalar_function_set_return_type(func_p, int_type_p)
    type_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)
    ducklib.duckdb_scalar_function_set_function(func_p, _add_one_cb.address)
    rc = ducklib.duckdb_register_scalar_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess
    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)

    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT add_one(v) FROM t_mc ORDER BY v")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess, f"Query failed, rc={rc}"

    outputs = []
    chunk_sizes = []
    while True:
        chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
        if chunk_p == 0:
            break
        size = ducklib.duckdb_data_chunk_get_size(chunk_p)
        vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
        data_p = ducklib.duckdb_vector_get_data(vec_p)
        outputs.extend((ctypes.c_int32 * size).from_address(data_p))
        chunk_sizes.append(size)
        aux_destroy_data_chunk(chunk_p)

    assert sum(chunk_sizes) == n_rows, chunk_sizes
    assert len(chunk_sizes) >= 2, "expected a multi-chunk result"
    assert chunk_sizes[0] == STANDARD_VECTOR_SIZE, chunk_sizes[0]
    assert outputs == list(range(1, n_rows + 1))
    # values straddling the first chunk boundary
    assert outputs[STANDARD_VECTOR_SIZE - 1] == STANDARD_VECTOR_SIZE
    assert outputs[STANDARD_VECTOR_SIZE] == STANDARD_VECTOR_SIZE + 1

    ducklib.duckdb_destroy_result(result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


@balanced(RESULT_FAMILY, CHUNK_FAMILY)
def test_aggregate_function_multi_chunk():
    """Check that a UDAF returns the right sum for each of 5000 groups, whose results take several output chunks.

    DuckDB therefore calls finalize for more than one output chunk. The test fetches the grouped result chunk by chunk
    and checks every sum, including those of the groups around row 2048. It does not expect full chunks: DuckDB need
    not fill a chunk of grouped output, and on duckdb 1.3.2 the sizes of the chunks change with the number of threads.
    """
    n_groups = 5000
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    result = create_duckdb_result()
    # A name keeps the SQL built at run time alive until duckdb_query has read it, because get_unicode_data_p does not
    # copy the string.
    create_sql = (
        "CREATE TABLE t_grp AS SELECT range::INTEGER AS g, "
        f"range::INTEGER AS v FROM range({n_groups})")
    query_p = get_unicode_data_p(create_sql)
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_result(result.ctypes.data)

    func_p = ducklib.duckdb_create_aggregate_function()
    ducklib.duckdb_aggregate_function_set_name(func_p, get_unicode_data_p("mc_sum"))
    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    bigint_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_BIGINT)
    ducklib.duckdb_aggregate_function_add_parameter(func_p, int_type_p)
    ducklib.duckdb_aggregate_function_set_return_type(func_p, bigint_type_p)
    for tp in (int_type_p, bigint_type_p):
        buf = numpy.array([tp], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)
    ducklib.duckdb_aggregate_function_set_functions(
        func_p, _agg_state_size_cb.address, _agg_init_cb.address,
        _agg_update_cb.address, _agg_combine_cb.address, _agg_finalize_cb.address)
    rc = ducklib.duckdb_register_aggregate_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess
    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_aggregate_function(func_buf.ctypes.data)

    result = create_duckdb_result()
    query_p = get_unicode_data_p(
        "SELECT mc_sum(v) FROM t_grp GROUP BY g ORDER BY g")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess, f"Query failed, rc={rc}"

    sums = []
    chunk_sizes = []
    while True:
        chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
        if chunk_p == 0:
            break
        size = ducklib.duckdb_data_chunk_get_size(chunk_p)
        vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
        data_p = ducklib.duckdb_vector_get_data(vec_p)
        sums.extend((ctypes.c_int64 * size).from_address(data_p))
        chunk_sizes.append(size)
        aux_destroy_data_chunk(chunk_p)

    assert sum(chunk_sizes) == n_groups, chunk_sizes
    assert len(chunk_sizes) >= 2, "expected a multi-chunk result"
    assert all(s <= STANDARD_VECTOR_SIZE for s in chunk_sizes), chunk_sizes
    # Each group g holds the one row v = g, so mc_sum(v) is g.
    assert sums == list(range(n_groups))
    assert sums[STANDARD_VECTOR_SIZE - 1] == STANDARD_VECTOR_SIZE - 1
    assert sums[STANDARD_VECTOR_SIZE] == STANDARD_VECTOR_SIZE

    ducklib.duckdb_destroy_result(result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


@balanced(RESULT_FAMILY, CHUNK_FAMILY)
def test_scalar_function_null_aware():
    """Check that a scalar UDF that reads the validity mask maps a column's NULL rows to a sentinel.

    DuckDB passes a column's NULL rows to the callback with or without special handling; special handling also makes
    DuckDB call the function for a constant NULL argument. The callback reads the mask with
    ``duckdb_vector_get_validity``. A callback that ignored the mask would return whatever the NULL rows' storage holds.
    """
    inputs = [1, None, 3, None, 5]
    expected = sorted(NULL_SENTINEL if x is None else x for x in inputs)
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    result = create_duckdb_result()
    query_p = get_unicode_data_p(
        "CREATE TABLE t_null AS SELECT * FROM "
        "(VALUES (1),(NULL),(3),(NULL),(5)) AS t(v)")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_result(result.ctypes.data)

    func_p = ducklib.duckdb_create_scalar_function()
    ducklib.duckdb_scalar_function_set_name(
        func_p, get_unicode_data_p("null_to_sentinel"))
    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    ducklib.duckdb_scalar_function_add_parameter(func_p, int_type_p)
    ducklib.duckdb_scalar_function_set_return_type(func_p, int_type_p)
    ducklib.duckdb_scalar_function_set_special_handling(func_p)
    type_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)
    ducklib.duckdb_scalar_function_set_function(func_p, _null_to_sentinel_cb.address)
    rc = ducklib.duckdb_register_scalar_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess
    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)

    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT null_to_sentinel(v) FROM t_null")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess, f"Query failed, rc={rc}"

    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    size = ducklib.duckdb_data_chunk_get_size(chunk_p)
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    data_p = ducklib.duckdb_vector_get_data(vec_p)
    outputs = sorted((ctypes.c_int32 * size).from_address(data_p))
    assert outputs == expected, f"got {outputs}, expected {expected}"

    aux_destroy_data_chunk(chunk_p)
    ducklib.duckdb_destroy_result(result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


@balanced(RESULT_FAMILY, CHUNK_FAMILY)
def test_aggregate_function_null_aware():
    """Check that a UDAF's update skips a column's NULL rows by the validity mask.

    DuckDB passes a column's NULL rows to update with or without special handling. The UDAF counts the valid rows, so
    its result must equal DuckDB's ``count(v)``, which ignores NULL. An update that ignored the mask would count every
    row, as ``count(*)`` does.
    """
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    result = create_duckdb_result()
    query_p = get_unicode_data_p(
        "CREATE TABLE t_nullagg AS SELECT * FROM "
        "(VALUES (1),(NULL),(3),(NULL),(5)) AS t(v)")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_result(result.ctypes.data)

    func_p = ducklib.duckdb_create_aggregate_function()
    ducklib.duckdb_aggregate_function_set_name(
        func_p, get_unicode_data_p("null_count"))
    int_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    bigint_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_BIGINT)
    ducklib.duckdb_aggregate_function_add_parameter(func_p, int_type_p)
    ducklib.duckdb_aggregate_function_set_return_type(func_p, bigint_type_p)
    ducklib.duckdb_aggregate_function_set_special_handling(func_p)
    for tp in (int_type_p, bigint_type_p):
        buf = numpy.array([tp], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)
    ducklib.duckdb_aggregate_function_set_functions(
        func_p, _null_count_state_size_cb.address, _null_count_init_cb.address,
        _null_count_update_cb.address, _null_count_combine_cb.address,
        _null_count_finalize_cb.address)
    rc = ducklib.duckdb_register_aggregate_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess
    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_aggregate_function(func_buf.ctypes.data)

    result = create_duckdb_result()
    query_p = get_unicode_data_p(
        "SELECT null_count(v), count(v), count(*) FROM t_nullagg")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess, f"Query failed, rc={rc}"

    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    d0 = ducklib.duckdb_vector_get_data(
        ducklib.duckdb_data_chunk_get_vector(chunk_p, 0))
    d1 = ducklib.duckdb_vector_get_data(
        ducklib.duckdb_data_chunk_get_vector(chunk_p, 1))
    d2 = ducklib.duckdb_vector_get_data(
        ducklib.duckdb_data_chunk_get_vector(chunk_p, 2))
    null_count = (ctypes.c_int64 * 1).from_address(d0)[0]
    builtin_count_v = (ctypes.c_int64 * 1).from_address(d1)[0]
    count_star = (ctypes.c_int64 * 1).from_address(d2)[0]
    assert null_count == 3, null_count
    assert null_count == builtin_count_v
    assert count_star == 5
    assert null_count != count_star

    aux_destroy_data_chunk(chunk_p)
    ducklib.duckdb_destroy_result(result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


def test_load_duckdb_refuses_version_mismatch(monkeypatch):
    """Check that load_duckdb refuses a standalone libduckdb whose version differs from the duckdb wheel's.

    load_duckdb loads a standalone libduckdb when the wheel's library does not export the C API, as on macOS, and the
    process then holds two DuckDB runtimes. The test fakes that case on any platform: the wheel's library seems to lack
    the C API, and the standalone library reports the version 9.9.9.
    """
    from numbduck import utils

    wheel_lib = object()
    standalone_lib = object()

    def fake_load_lib_path(path):
        return wheel_lib if path == "/fake/wheel.so" else standalone_lib

    monkeypatch.setattr(utils, "find_duckdb_shared_lib", lambda: "/fake/wheel.so")
    monkeypatch.setattr(utils, "load_lib_path", fake_load_lib_path)
    monkeypatch.setattr(utils, "_open_private", fake_load_lib_path)
    monkeypatch.setattr(utils, "_has_capi_symbols", lambda lib: lib is standalone_lib)
    monkeypatch.setattr(
        utils, "_find_standalone_libduckdb", lambda: "/fake/libduckdb.dylib")
    monkeypatch.setattr(utils, "_library_version", lambda lib: "9.9.9")

    with pytest.raises(RuntimeError, match="NUMBDUCK_LIBDUCKDB"):
        utils.load_duckdb()


_REFUSED_STANDALONE_SCRIPT = r"""
import ctypes
import ctypes.util
import os

from numbduck import utils

print("BEFORE", hasattr(ctypes.CDLL(None), "duckdb_open"))
os.environ["NUMBDUCK_LIBDUCKDB"] = utils.find_duckdb_shared_lib()
utils.find_duckdb_shared_lib = lambda: ctypes.util.find_library("m")
utils._wheel_library_version = lambda: "0.0.0"
try:
    utils.load_duckdb()
except RuntimeError as exc:
    print("REFUSED", "0.0.0" in str(exc))
print("GLOBAL", hasattr(ctypes.CDLL(None), "duckdb_open"))
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads the process-wide symbol scope through dlopen(NULL)")
def test_refused_standalone_libduckdb_stays_out_of_the_global_symbol_scope(tmp_path):
    """Check that a standalone libduckdb that the version check refuses never enters the process-wide symbol scope.

    numbox resolves every call from jitted code by name in that scope, where the first definition of a name takes
    precedence, and a loaded library is never unloaded. A refused library left there would receive the jitted calls
    even after a later candidate passed the check.

    The test runs in a child process. The child uses libm, which lacks the C API, as the wheel's library, and offers the
    wheel's own libduckdb as the standalone library; the check refuses it because the child fakes the wheel's version.
    The child then checks, through dlopen(NULL), whether duckdb_open is visible.
    """
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        [repo_root, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    # With HOME in a temporary directory, the child's loader sees an empty numbduck cache instead of the user's.
    env["HOME"] = str(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", _REFUSED_STANDALONE_SCRIPT],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    lines = proc.stdout.split("\n")
    assert "BEFORE False" in lines, proc.stdout
    assert "REFUSED True" in lines, proc.stdout
    assert "GLOBAL False" in lines, proc.stdout


def test_pybridge_refuses_uncoordinated_runtime(monkeypatch):
    """Check that extract_connection_ptr raises RuntimeError when numbduck's libduckdb and the duckdb wheel's differ."""
    from numbduck import pybridge

    monkeypatch.setattr(pybridge, "libraries_coordinated", lambda: False)
    monkeypatch.setattr(pybridge, "loaded_library_version", lambda: "9.9.9")

    conn = duckdb.connect()
    try:
        with pytest.raises(RuntimeError, match="libduckdb"):
            pybridge.extract_connection_ptr(conn)
    finally:
        conn.close()


def test_non_ascii_text_reaches_duckdb_as_utf8(tmp_path):
    """Check that a path, an identifier and a bound value that are not ASCII reach DuckDB intact through ``c_string``.

    The C API reads every ``char*`` as UTF-8 that ends with a NUL. numbox's ``get_unicode_data_p`` passes CPython's own
    storage of the string instead, in Latin-1, UCS-2 or UCS-4, which is UTF-8 only for ASCII text. numbox's
    ``c_string`` makes a UTF-8 copy, and the examples pass that.
    """
    from numbox.utils.cstrings import c_string

    path = str(tmp_path / "data_€.duckdb")
    db = create_duckdb_database()
    with c_string(path) as path_p:
        assert ducklib.duckdb_open(path_p, db.ctypes.data) == ducklib.DuckDBSuccess
    conn = create_duckdb_connection()
    assert ducklib.duckdb_connect(db[0], conn.ctypes.data) == ducklib.DuckDBSuccess
    result = create_duckdb_result()
    with c_string('CREATE TABLE "café" (city VARCHAR)') as sql_p:
        assert ducklib.duckdb_query(conn[0], sql_p, result.ctypes.data) == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_result(result.ctypes.data)
    stmt = create_duckdb_prepared_statement()
    with c_string('INSERT INTO "café" VALUES ($1)') as sql_p:
        assert ducklib.duckdb_prepare(conn[0], sql_p, stmt.ctypes.data) == ducklib.DuckDBSuccess
    with c_string("Tokyo 東京") as city_p:
        assert ducklib.duckdb_bind_varchar(stmt[0], 1, city_p) == ducklib.DuckDBSuccess
    result = create_duckdb_result()
    assert ducklib.duckdb_execute_prepared(stmt[0], result.ctypes.data) == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_result(result.ctypes.data)
    ducklib.duckdb_destroy_prepare(stmt.ctypes.data)
    ducklib.duckdb_disconnect(conn.ctypes.data)
    ducklib.duckdb_close(db.ctypes.data)

    assert "data_€.duckdb" in os.listdir(tmp_path)
    with duckdb.connect(path) as check:
        assert check.execute('SELECT city FROM "café"').fetchall() == [("Tokyo 東京",)]


def test_pybridge_closed_connection_raises_runtime_error():
    """Check that extract_connection_ptr raises RuntimeError for a closed connection instead of returning a null.

    Closing a connection resets its ``unique_ptr<Connection>`` to null, so the pointer at the offset that pybridge reads
    is null. If that pointer were passed on to ``duckdb_query``, it would cause an opaque numba TypeError or crash the
    process on the null connection.
    """
    from numbduck.pybridge import extract_connection_ptr

    conn = duckdb.connect()
    conn.close()
    with pytest.raises(RuntimeError, match="null"):
        extract_connection_ptr(conn)


def test_extract_connection_ptr_leaves_pending_result_intact():
    """Check that extracting the pointer runs nothing on the connection, so a result it is streaming stays complete.

    A query issued on the connection would close that result, and every later fetch of it would return too few rows
    without an error.
    """
    from numbduck.pybridge import extract_connection_ptr

    conn = duckdb.connect()
    try:
        pending = conn.execute("SELECT * FROM range(5000)")
        extract_connection_ptr(conn)
        assert len(pending.fetchall()) == 5000
    finally:
        conn.close()


def test_extract_connection_ptr_inside_an_aborted_transaction():
    """Check that the extraction works on a connection whose transaction has failed, and leaves the failure visible.

    Such a connection still holds a valid ``Connection*``. The next query on it still raises the aborted transaction's
    error, for the caller to handle.
    """
    from numbduck.pybridge import extract_connection_ptr

    conn = duckdb.connect()
    try:
        conn.execute("BEGIN")
        with pytest.raises(duckdb.Error):
            conn.execute("SELECT error('boom')")
        assert extract_connection_ptr(conn) != 0
        with pytest.raises(duckdb.Error, match="aborted"):
            conn.execute("SELECT 42")
    finally:
        conn.close()


def test_load_duckdb_refuses_unverifiable_standalone_version(monkeypatch):
    """Check that load_duckdb refuses a standalone libduckdb whose version it cannot read, instead of loading it.

    A library loaded without a version would fail later, with a coordination error that names libduckdb None. The test
    uses no network.
    """
    from numbduck import utils

    wheel_lib = object()
    standalone_lib = object()

    def fake_load_lib_path(path):
        return wheel_lib if path == "/fake/wheel.so" else standalone_lib

    monkeypatch.setattr(utils, "find_duckdb_shared_lib", lambda: "/fake/wheel.so")
    monkeypatch.setattr(utils, "load_lib_path", fake_load_lib_path)
    monkeypatch.setattr(utils, "_open_private", fake_load_lib_path)
    monkeypatch.setattr(utils, "_has_capi_symbols", lambda lib: lib is standalone_lib)
    monkeypatch.setattr(
        utils, "_find_standalone_libduckdb", lambda: "/fake/libduckdb.dylib")
    monkeypatch.setattr(utils, "_library_version", lambda lib: None)

    with pytest.raises(RuntimeError, match="could not read its library version"):
        utils.load_duckdb()


def test_load_duckdb_refuses_dev_suffix_same_base(monkeypatch):
    """Check that load_duckdb refuses a standalone dev build whose version shares the release's ``X.Y.Z``.

    The check compares the full library version, suffix included, and never strips it to ``X.Y.Z``. A dev or rc build
    comes from another commit, whose Connection layout may differ, so the check must not accept it as the release.
    """
    from numbduck import utils

    wheel_lib = object()
    standalone_lib = object()

    def fake_load_lib_path(path):
        return wheel_lib if path == "/fake/wheel.so" else standalone_lib

    monkeypatch.setattr(utils, "find_duckdb_shared_lib", lambda: "/fake/wheel.so")
    monkeypatch.setattr(utils, "load_lib_path", fake_load_lib_path)
    monkeypatch.setattr(utils, "_open_private", fake_load_lib_path)
    monkeypatch.setattr(utils, "_has_capi_symbols", lambda lib: lib is standalone_lib)
    monkeypatch.setattr(
        utils, "_find_standalone_libduckdb", lambda: "/fake/libduckdb.dylib")
    monkeypatch.setattr(utils, "_wheel_library_version", lambda: "1.5.4")
    monkeypatch.setattr(utils, "_library_version", lambda lib: "1.5.4-dev12")

    with pytest.raises(RuntimeError, match="undefined behavior"):
        utils.load_duckdb()


def test_libraries_coordinated_compares_library_version_not_package(monkeypatch):
    """Check that libraries_coordinated compares library versions, as PRAGMA version reports them, not the package's.

    A dev build whose library version matches the standalone library's counts as coordinated, although the Python
    package writes its pre-release version in another scheme. A comparison with ``duckdb.__version__`` would return
    False for that pair. A library version that differs still counts as not coordinated.
    """
    from numbduck import utils

    monkeypatch.setattr(utils, "loaded_library_version", lambda: "1.6.0-dev10121")
    monkeypatch.setattr(utils, "_wheel_library_version", lambda: "1.6.0-dev10121")
    assert utils.libraries_coordinated() is True

    monkeypatch.setattr(utils, "_wheel_library_version", lambda: "1.6.0-dev9999")
    assert utils.libraries_coordinated() is False


def test_wheel_library_version_reads_pragma_version():
    """Check that _wheel_library_version returns the version that PRAGMA version reports, without its leading ``v``.

    A standalone library's ``duckdb_library_version()`` reports its version in the same scheme, so the two compare
    directly.
    """
    from numbduck import utils

    row = duckdb.connect(":memory:").execute("PRAGMA version").fetchone()
    assert utils._wheel_library_version() == utils._normalize_version(row[0])


def test_non_darwin_capi_error_uses_platform_extension(monkeypatch):
    """Check that the error for a wheel without the C API, off macOS, names the platform's library file.

    The message names libduckdb.dll on Windows and libduckdb.so on other platforms.
    """
    from numbduck import utils

    monkeypatch.delenv("NUMBDUCK_LIBDUCKDB", raising=False)
    monkeypatch.setattr(utils.platform, "system", lambda: "Windows")
    assert "libduckdb.dll" in utils._non_darwin_capi_error()
    monkeypatch.setattr(utils.platform, "system", lambda: "Linux")
    assert "libduckdb.so" in utils._non_darwin_capi_error()


def _build_fake_libduckdb_zip(tmp_path):
    """Write a ZIP archive holding a fake libduckdb.dylib into ``tmp_path``; return its path and the dylib's bytes.

    The tests pass the archive to ``_fetch_and_cache`` as a file:// URL.
    """
    import io
    import zipfile

    dylib_bytes = b"\x00fake-libduckdb-native-code\x7fELF"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("libduckdb.dylib", dylib_bytes)
    zip_path = tmp_path / "libduckdb-osx-universal.zip"
    zip_path.write_bytes(buf.getvalue())
    return zip_path, dylib_bytes


def test_fetch_and_cache_local_zip_roundtrip(tmp_path):
    """Check that ``_fetch_and_cache`` extracts the dylib from a local file:// archive and writes it to the cache path.

    ``_fetch_and_cache`` writes the file atomically, under a temporary name that it then renames. The test uses no
    network.
    """
    from numbduck import utils

    zip_path, dylib_bytes = _build_fake_libduckdb_zip(tmp_path)
    dest = tmp_path / "cache" / "libduckdb.dylib"

    returned = utils._fetch_and_cache(zip_path.as_uri(), str(dest))

    assert returned == str(dest)
    assert dest.read_bytes() == dylib_bytes


@pytest.mark.parametrize("headless_mode", ["non_tty", "eof"])
def test_download_consent_headless_raises_branded(monkeypatch, headless_mode):
    """Check that a headless import raises numbduck's RuntimeError with instructions, instead of an EOFError.

    An import is headless when stdin is not a terminal, or when the consent prompt reaches the end of stdin. The message
    names ``NUMBDUCK_LIBDUCKDB_DOWNLOAD=1``, ``NUMBDUCK_LIBDUCKDB`` and brew. The test fails if any code calls
    ``urllib.request.urlopen``.
    """
    from numbduck import utils

    monkeypatch.delenv("NUMBDUCK_LIBDUCKDB_DOWNLOAD", raising=False)

    class _NonTtyStdin:
        def isatty(self):
            return False

    class _TtyStdin:
        def isatty(self):
            return True

    def _raise_eof(_prompt):
        raise EOFError

    if headless_mode == "non_tty":
        monkeypatch.setattr(sys, "stdin", _NonTtyStdin())
    else:
        monkeypatch.setattr(sys, "stdin", _TtyStdin())
        monkeypatch.setattr("builtins.input", _raise_eof)

    def _no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr("urllib.request.urlopen", _no_network)

    with pytest.raises(RuntimeError) as exc_info:
        utils._download_libduckdb()
    message = str(exc_info.value)
    assert "NUMBDUCK_LIBDUCKDB_DOWNLOAD=1" in message
    assert "NUMBDUCK_LIBDUCKDB=" in message
    assert "brew" in message


@pytest.mark.parametrize("env_configured", [False, True])
def test_load_duckdb_non_darwin_message(monkeypatch, env_configured):
    """Check the error that load_duckdb raises off macOS for a wheel without the C API and no standalone library.

    The message says that the automatic download exists only on macOS. It tells an unset ``NUMBDUCK_LIBDUCKDB`` apart
    from one set to a missing path, and repeats that path. The test uses no network.
    """
    from numbduck import utils

    monkeypatch.setattr(utils.platform, "system", lambda: "Linux")
    monkeypatch.setattr(utils, "find_duckdb_shared_lib", lambda: "/fake/wheel.so")
    monkeypatch.setattr(utils, "load_lib_path", lambda path: object())
    monkeypatch.setattr(utils, "_has_capi_symbols", lambda lib: False)
    monkeypatch.setattr(utils, "_find_standalone_libduckdb", lambda: None)
    monkeypatch.setattr(utils, "_migrate_legacy_cache", lambda: None)

    bad_path = "/no/such/dir/libduckdb.so"
    if env_configured:
        monkeypatch.setenv("NUMBDUCK_LIBDUCKDB", bad_path)
    else:
        monkeypatch.delenv("NUMBDUCK_LIBDUCKDB", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        utils.load_duckdb()
    message = str(exc_info.value)
    assert "macOS" in message
    if env_configured:
        assert bad_path in message
    else:
        assert "NUMBDUCK_LIBDUCKDB=" in message


def _import_irr_example():
    """Import examples/irr.py, the IRR UDAF example, as the module ``irr`` and return it.

    ``irr`` is the name that examples/run_irr.py uses, so ``IRRStateType.__module__`` stays the same and numba's type
    inference matches. The import compiles every @cfunc callback of the example and the @njit functions they call, so
    it also checks that finalize, with the irr_bisect it calls, still compiles.
    """
    examples_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    import irr  # type: ignore  # noqa: E402
    return irr


def test_irr_bisect_large_magnitude_converges():
    """Check that scaling a case by 1000 leaves its monthly IRR finite and unchanged.

    IRR does not depend on the scale of the amounts. An investment of 10,000,000 with 13,000,000 back at period 12 must
    give the rate of 10,000 with 13,000 back. The solver stops when the bracket of rates is narrower than a fixed width,
    which works at every scale. A stop on a fixed NPV tolerance could fail for large amounts, because the NPV left at
    the true root grows with the amounts; such a solver would return NaN after it had converged.
    """
    irr = _import_irr_example()
    periods = numpy.array([12.0], dtype=numpy.float64)
    expected = (13000.0 / 10000.0) ** (1.0 / 12.0) - 1.0

    small = irr.irr_bisect(
        numpy.array([13000.0], dtype=numpy.float64), periods, 1, 10000.0, 0.0)
    large = irr.irr_bisect(
        numpy.array([13000000.0], dtype=numpy.float64), periods, 1,
        10000000.0, 0.0)

    assert math.isfinite(small), small
    assert math.isfinite(large), large
    assert abs(small - expected) < 1e-9, (small, expected)
    assert abs(large - expected) < 1e-9, (large, expected)
    assert abs(large - small) < 1e-12, (large, small)


def test_irr_bisect_out_of_bracket_sentinel():
    """Check that a root outside the bracket [-0.99, 10.0] gives the IRR_NO_BRACKET sentinel, +inf, and not NaN.

    NPV then has the same sign at both ends of the bracket. +inf keeps the result apart from the NaN that finalize
    returns for an empty group.
    """
    irr = _import_irr_example()

    # Root near r=999 (monthly return far above the 10.0 bracket top):
    # NPV(r) = -1 + 1000 / (1 + r) is positive at both r=-0.99 and r=10.
    high = irr.irr_bisect(
        numpy.array([1000.0], dtype=numpy.float64),
        numpy.array([1.0], dtype=numpy.float64), 1, 1.0, 0.0)
    assert high == irr.IRR_NO_BRACKET
    assert not math.isnan(high)

    # Near-total loss, root below -0.99: NPV is negative at both ends of the bracket.
    low = irr.irr_bisect(
        numpy.array([1.0], dtype=numpy.float64),
        numpy.array([1.0], dtype=numpy.float64), 1, 1000.0, 0.0)
    assert low == irr.IRR_NO_BRACKET
    assert not math.isnan(low)


def test_irr_bisect_financing_direction_converges():
    """Check that the solver finds the root of a financing stream, in which cash arrives first and is repaid later.

    NPV then rises with r instead of falling: it is negative at -0.99 and positive at 10.0, and it still changes sign
    once inside the bracket. A solver that assumed a falling NPV would move toward an end of the bracket and return a
    rate near 10.0.
    """
    irr = _import_irr_example()
    # Receive 1000 now (investment = -1000) and repay 1300 at month 12. NPV(r) = 1000 - 1300 / (1 + r)^12 is negative
    # at r=-0.99 and positive at r=10, with its single root at (1300/1000)^(1/12) - 1.
    rate = irr.irr_bisect(
        numpy.array([-1300.0], dtype=numpy.float64),
        numpy.array([12.0], dtype=numpy.float64), 1, -1000.0, 0.0)
    expected = (1300.0 / 1000.0) ** (1.0 / 12.0) - 1.0
    assert math.isfinite(rate), rate
    assert rate != irr.IRR_NO_BRACKET
    assert abs(rate - expected) < 1e-9, (rate, expected)


def test_irr_bisect_large_period_converges():
    """Check that the solver converges on the true rate for a cashflow at period 200.

    At r = -0.99, the lower end of the bracket, the discount factor (1 + r)**period underflows to 0.0 for a period of
    162 or more. An NPV computed directly would divide by that 0.0, which raises ZeroDivisionError under numba's default
    error model. The finalize callback would then report the valid group as NaN, the result of an empty group. The
    solver evaluates only the sign of NPV, in log space, and never divides by the discount factor.
    """
    irr = _import_irr_example()
    rate = irr.irr_bisect(
        numpy.array([13000.0], dtype=numpy.float64),
        numpy.array([200.0], dtype=numpy.float64), 1, 10000.0, 0.0)
    expected = (13000.0 / 10000.0) ** (1.0 / 200.0) - 1.0
    assert math.isfinite(rate), rate
    assert abs(rate - expected) < 1e-9, (rate, expected)


def test_irr_bisect_nan_input_returns_nan():
    """Check that a NaN cashflow gives NaN, the result of an empty group, and not the +inf sentinel.

    A NaN is a legal DOUBLE, not a SQL NULL, so it reaches the solver and makes the sign at an end of the bracket NaN.
    A result of +inf would mean that the bracket holds no single root.
    """
    irr = _import_irr_example()
    rate = irr.irr_bisect(
        numpy.array([math.nan], dtype=numpy.float64),
        numpy.array([12.0], dtype=numpy.float64), 1, 10000.0, 0.0)
    assert math.isnan(rate), rate


def test_irr_bisect_zero_cashflow_far_period_matches():
    """Check that a zero cashflow at period 200 leaves the root unchanged.

    A zero cashflow adds nothing to NPV at any rate, so the group with it must return the same root as the group
    without it. At r = -0.99, (1 + r)**-200 overflows to inf, and an NPV computed directly would multiply the zero by
    inf and get NaN, the result of an empty group. The solver skips zero cashflows when it evaluates the sign.
    """
    irr = _import_irr_example()
    with_zero = irr.irr_bisect(
        numpy.array([13000.0, 0.0], dtype=numpy.float64),
        numpy.array([12.0, 200.0], dtype=numpy.float64), 2, 10000.0, 0.0)
    without_zero = irr.irr_bisect(
        numpy.array([13000.0], dtype=numpy.float64),
        numpy.array([12.0], dtype=numpy.float64), 1, 10000.0, 0.0)
    expected = (13000.0 / 10000.0) ** (1.0 / 12.0) - 1.0
    assert math.isfinite(with_zero), with_zero
    assert with_zero == without_zero, (with_zero, without_zero)
    assert abs(with_zero - expected) < 1e-9, (with_zero, expected)


def test_irr_bisect_mixed_sign_far_period_converges():
    """Check that the solver finds the root when cashflows of opposite sign both lie at periods of 162 or more.

    At r = -0.99, an NPV computed directly would add inf and -inf there and get NaN, the result of an empty group. The
    solver evaluates the sign without overflow, so the sign of NPV at each end of the bracket is correct, and the
    solver finds the single root inside the bracket. The test compares the rate with a bisection in exact rational
    arithmetic over the same bracket. Fraction arithmetic never underflows, so the reference does not depend on the
    solver's floating-point method.
    """
    from fractions import Fraction
    irr = _import_irr_example()
    cashflows = [150.0, -1.0, 1.0]
    periods = [1.0, 200.0, 201.0]
    rate = irr.irr_bisect(
        numpy.array(cashflows, dtype=numpy.float64),
        numpy.array(periods, dtype=numpy.float64), 3, 100.0, 0.0)

    def npv_sign(r):
        total = Fraction(-100)
        for c, p in zip(cashflows, periods):
            total += Fraction(c) / (1 + r) ** int(p)
        return (total > 0) - (total < 0)

    lo, hi = Fraction(-99, 100), Fraction(10)
    lo_pos = npv_sign(lo) > 0
    for _ in range(200):
        if hi - lo < Fraction(1, 10 ** 15):
            break
        mid = (lo + hi) / 2
        if (npv_sign(mid) > 0) == lo_pos:
            lo = mid
        else:
            hi = mid
    ref = float((lo + hi) / 2)
    assert math.isfinite(rate), rate
    assert abs(rate - ref) < 1e-9, (rate, ref)


# --- Exception guards in the callbacks of a structref-backed UDAF ---
#
# numba does not let a Python exception from an @njit function cross a @cfunc into DuckDB: the @cfunc prints the
# exception and returns zero, or nothing for a void callback. If the exception comes from a nested call while a
# borrowed structref is alive, numba also skips the decref that ends the borrow, and the structref's MemInfo leaks. A
# bare try/except inside the @njit function catches the exception in its own frame, so the decref runs. The tests
# below check both cases: an unguarded raise leaks, and a guarded one does not.


@njit
def _cfunc_guard_raise(x):
    """Raise ValueError for an input over 0.5, and return the input otherwise.

    The callers call it while they hold a borrowed structref, so the raise crosses a call boundary, as a failed
    allocation does when numbox's vector_push calls numpy.empty. On an unguarded path, numba skips the borrow's decref
    when the raise crosses a call boundary, and still runs it when the raise is in the caller's own frame.
    """
    if x > 0.5:
        raise ValueError("induced failure inside a UDAF callback")
    return x


@njit
def _welford_update_raise_unguarded_impl(info, chunk, states):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(vec)
    state_slots = carray(
        _cast_int_to_void_p(states), (n,), dtype=numpy.intp)
    in_vals = carray(
        _cast_int_to_void_p(in_data), (n,), dtype=numpy.float64)
    for i in range(n):
        slot = carray(
            _cast_int_to_void_p(state_slots[i]), (1,), dtype=numpy.intp)
        s = borrow_structref(welford_type, slot[0])
        welford_update(s, _cfunc_guard_raise(in_vals[i]))


@njit
def _welford_update_raise_guarded_impl(info, chunk, states):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(vec)
    state_slots = carray(
        _cast_int_to_void_p(states), (n,), dtype=numpy.intp)
    in_vals = carray(
        _cast_int_to_void_p(in_data), (n,), dtype=numpy.float64)
    for i in range(n):
        slot = carray(
            _cast_int_to_void_p(state_slots[i]), (1,), dtype=numpy.intp)
        # The borrow is inside the guard, so numba releases it when the nested call raises.
        try:
            s = borrow_structref(welford_type, slot[0])
            welford_update(s, _cfunc_guard_raise(in_vals[i]))
        except Exception:
            pass


@functools.cache
def _welford_raise_callbacks():
    """Compile the unguarded and the guarded raising update callbacks on the first call, and return them.

    A @cfunc compiles when it is defined. These are defined inside the function, so importing the module does not
    compile them, and test collection stays fast.
    """

    @cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
    def unguarded_cb(info, chunk, states):
        _welford_update_raise_unguarded_impl(info, chunk, states)

    @cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
    def guarded_cb(info, chunk, states):
        _welford_update_raise_guarded_impl(info, chunk, states)

    return unguarded_cb, guarded_cb


def _register_raising_welford(conn, name, update_cb):
    """Register on ``conn`` a Welford UDAF named ``name`` whose update callback is ``update_cb``.

    The other callbacks, init, combine, finalize and destroy, are the Welford UDAF's own.
    """
    from numbduck.pybridge import extract_connection_ptr

    conn_p = extract_connection_ptr(conn)
    func_p = ducklib.duckdb_create_aggregate_function()
    ducklib.duckdb_aggregate_function_set_name(func_p, get_unicode_data_p(name))
    dbl = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
    ducklib.duckdb_aggregate_function_add_parameter(func_p, dbl)
    ducklib.duckdb_aggregate_function_set_return_type(func_p, dbl)
    tb = numpy.array([dbl], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(tb.ctypes.data)
    ducklib.duckdb_aggregate_function_set_functions(
        func_p, _welford_state_size_cb.address, _welford_init_cb.address,
        update_cb.address, _welford_combine_cb.address,
        _welford_finalize_cb.address)
    ducklib.duckdb_aggregate_function_set_destructor(
        func_p, _welford_destroy_cb.address)
    rc = ducklib.duckdb_register_aggregate_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess
    fb = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_aggregate_function(fb.ctypes.data)


def _run_raising_udaf_nrt_delta(update_cb):
    """Run a Welford UDAF whose update callback is ``update_cb`` over eight rows, and return the leak and the result.

    The leak is the number of NRT allocations minus the number of frees during the query. The @cfunc catches the
    exception, so the query succeeds whether or not memory leaked.
    """
    from numba.core.runtime import _nrt_python as _nrt
    _nrt.memsys_enable_stats()
    from numba.core.runtime.nrt import rtsys

    conn = duckdb.connect()
    _register_raising_welford(conn, "wraise", update_cb)
    conn.execute("CREATE TABLE t_raise AS SELECT 1.0 AS v FROM range(8)")
    before = rtsys.get_allocation_stats()
    row = conn.execute("SELECT wraise(v) FROM t_raise").fetchone()
    conn.close()
    after = rtsys.get_allocation_stats()
    delta = (after.alloc - before.alloc) - (after.free - before.free)
    return delta, row[0]


@pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning")
def test_udaf_cfunc_unguarded_raise_leaks_borrowed_state():
    """Check that, without the guard, a raise from a nested call leaks the reference of a borrowed structref.

    The @cfunc catches the exception, and numba skips the decref that ends the borrow. This control shows that the
    raise happens and has an effect, so a leak of zero in the guarded test below comes from the guard.

    numba prints the exception that the @cfunc caught, and pytest reports it as an unraisable-exception warning, which
    the test's ``filterwarnings`` mark ignores.
    """
    unguarded_cb, _ = _welford_raise_callbacks()
    leak, _ = _run_raising_udaf_nrt_delta(unguarded_cb)
    assert leak > 0, f"expected an NRT leak without the guard, got delta {leak}"


def test_udaf_cfunc_guarded_raise_no_leak():
    """Check that a bare try/except around the borrow and the raising call keeps the borrow from leaking.

    numba runs the borrow's decref when the exception is caught in the same frame, so NRT allocates and frees the same
    number of blocks although every row's update raises. The example UDAFs guard their callbacks the same way.
    """
    _, guarded_cb = _welford_raise_callbacks()
    leak, value = _run_raising_udaf_nrt_delta(guarded_cb)
    assert leak == 0, f"guarded raise must not leak, got NRT delta {leak}"
    # Every row's update raised, so the guard skipped every row. finalize sees an empty state, and the result is NULL
    # or NaN. Without the raise, the result would be a standard deviation.
    assert value is None or math.isnan(value), value


# ---- A merge-counting UDAF, which shows whether DuckDB calls the combine callback ----

@structref.register
class _MergeStateType(nb_types.StructRef):
    pass


_merge_state_fields = [("total", nb_types.float64), ("merges", nb_types.int64)]
# cache=False keeps the structref constructor that numbox generates out of numba's disk cache. The cache keys the
# constructor by the struct's name and fields alone, and it resolves _MergeStateType through the module name under
# which it first compiled. Once the cache is warm, an import of this file under another qualified name fails type
# inference. IRRState in examples/irr.py is cached, so that module must always be imported under the name irr.
# Without the cache, this module imports under any name, at no measurable cost.
_MergeState = make_structref(
    "_MergeState", dict(_merge_state_fields), _MergeStateType,
    jit_options={"cache": False})
merge_state_type = _MergeStateType(_merge_state_fields)


@njit
def _merge_state_size_impl(info):
    return numpy.uint64(8)


@njit
def _merge_init_impl(info, state):
    s = _MergeState(0.0, 0)
    slot = carray(_cast_int_to_void_p(state), (1,), dtype=numpy.intp)
    slot[0] = export_meminfo(s)


@njit
def _merge_update_impl(info, chunk, states):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(vec)
    state_slots = carray(
        _cast_int_to_void_p(states), (n,), dtype=numpy.intp)
    in_vals = carray(
        _cast_int_to_void_p(in_data), (n,), dtype=numpy.float64)
    for i in range(n):
        slot = carray(
            _cast_int_to_void_p(state_slots[i]), (1,), dtype=numpy.intp)
        s = borrow_structref(merge_state_type, slot[0])
        s.total += in_vals[i]


@njit
def _merge_combine_impl(info, source, target, count):
    src_slots = carray(
        _cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    tgt_slots = carray(
        _cast_int_to_void_p(target), (count,), dtype=numpy.intp)
    for i in range(count):
        iu = numpy.uint64(i)
        src_slot = carray(
            _cast_int_to_void_p(src_slots[iu]), (1,), dtype=numpy.intp)
        tgt_slot = carray(
            _cast_int_to_void_p(tgt_slots[iu]), (1,), dtype=numpy.intp)
        src = borrow_structref(merge_state_type, src_slot[0])
        tgt = borrow_structref(merge_state_type, tgt_slot[0])
        tgt.total += src.total
        # Count each merged partial state, so that finalize can report whether combine ran.
        tgt.merges += src.merges + 1


@njit
def _merge_finalize_impl(info, source, result, count, offset):
    out_data = ducklib.duckdb_vector_get_data(result)
    src_slots = carray(
        _cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    out_vals = carray(
        _cast_int_to_void_p(out_data), (offset + count,),
        dtype=numpy.float64)
    for i in range(count):
        iu = numpy.uint64(i)
        src_slot = carray(
            _cast_int_to_void_p(src_slots[iu]), (1,), dtype=numpy.intp)
        s = borrow_structref(merge_state_type, src_slot[0])
        out_vals[numpy.uint64(offset + iu)] = numpy.float64(s.merges)


@njit
def _merge_destroy_impl(states, count):
    state_slots = carray(
        _cast_int_to_void_p(states), (count,), dtype=numpy.intp)
    for i in range(count):
        iu = numpy.uint64(i)
        slot = carray(
            _cast_int_to_void_p(state_slots[iu]), (1,), dtype=numpy.intp)
        release_meminfo(slot[0])


@functools.cache
def _merge_probe_callbacks():
    """Compile the six callbacks of the merge-counting UDAF on the first call, and return them in DuckDB's order.

    The order is state size, init, update, combine, finalize and destroy. The callbacks are defined inside the
    function, so importing the module does not compile them.
    """

    @cfunc(nb_types.uint64(nb_types.intp))
    def state_size_cb(info):
        return _merge_state_size_impl(info)

    @cfunc(nb_types.void(nb_types.intp, nb_types.intp))
    def init_cb(info, state):
        _merge_init_impl(info, state)

    @cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
    def update_cb(info, chunk, states):
        _merge_update_impl(info, chunk, states)

    @cfunc(nb_types.void(nb_types.intp, nb_types.intp,
                         nb_types.intp, nb_types.uint64))
    def combine_cb(info, source, target, count):
        _merge_combine_impl(info, source, target, count)

    @cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp,
                         nb_types.uint64, nb_types.uint64))
    def finalize_cb(info, source, result, count, offset):
        _merge_finalize_impl(info, source, result, count, offset)

    @cfunc(nb_types.void(nb_types.intp, nb_types.uint64))
    def destroy_cb(states, count):
        _merge_destroy_impl(states, count)

    return state_size_cb, init_cb, update_cb, combine_cb, finalize_cb, destroy_cb


def _register_merge_probe(conn):
    from numbduck.pybridge import extract_connection_ptr

    (state_size_cb, init_cb, update_cb, combine_cb, finalize_cb,
     destroy_cb) = _merge_probe_callbacks()
    conn_p = extract_connection_ptr(conn)
    func_p = ducklib.duckdb_create_aggregate_function()
    ducklib.duckdb_aggregate_function_set_name(
        func_p, get_unicode_data_p("merge_probe"))
    dbl = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
    ducklib.duckdb_aggregate_function_add_parameter(func_p, dbl)
    ducklib.duckdb_aggregate_function_set_return_type(func_p, dbl)
    tb = numpy.array([dbl], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(tb.ctypes.data)
    ducklib.duckdb_aggregate_function_set_functions(
        func_p, state_size_cb.address, init_cb.address,
        update_cb.address, combine_cb.address,
        finalize_cb.address)
    ducklib.duckdb_aggregate_function_set_destructor(
        func_p, destroy_cb.address)
    rc = ducklib.duckdb_register_aggregate_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess
    fb = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_aggregate_function(fb.ctypes.data)


def test_irr_example_combine_path_nrt_balanced():
    """Check that the IRR UDAF returns the right rate, without an NRT leak, when DuckDB merges partial states.

    PRAGMA verify_parallelism makes DuckDB build several partial states and merge them with the combine callback. A
    correct rate alone does not show that combine ran, because a plan with one partition would be correct too. A
    merge-counting UDAF therefore first checks that the installed DuckDB calls combine under the pragma. Both queries
    must free every NRT block they allocate.
    """
    from numba.core.runtime import _nrt_python as _nrt
    _nrt.memsys_enable_stats()
    from numba.core.runtime.nrt import rtsys

    irr = _import_irr_example()

    conn = duckdb.connect()
    # On duckdb 1.5.4, PRAGMA verify_parallelism makes DuckDB take the parallel aggregate path even for very small
    # inputs, so combine runs every time. SET verify_parallelism=true does not work: DuckDB has no setting of that
    # name, only the PRAGMA.
    conn.execute("PRAGMA verify_parallelism")

    _register_merge_probe(conn)
    conn.execute(
        "CREATE TABLE t_probe AS SELECT (range % 3)::INTEGER AS g, 1.0 AS v "
        "FROM range(96)")
    before = rtsys.get_allocation_stats()
    probe_rows = conn.execute(
        "SELECT g, merge_probe(v) FROM t_probe GROUP BY g ORDER BY g"
    ).fetchall()
    after = rtsys.get_allocation_stats()
    max_merges = max(int(r[1]) for r in probe_rows)
    assert max_merges > 0, (
        f"combine never ran under verify_parallelism: {probe_rows}")
    assert after.alloc - before.alloc == after.free - before.free, (
        "combine borrow/release imbalance in the merge probe")

    irr.register_irr(conn)
    conn.execute(
        "CREATE TABLE t_irr AS SELECT (range + 1)::DOUBLE AS period, "
        "1000.0 AS cashflow, 10000.0 AS investment, 0.0 AS target_npv "
        "FROM range(12)")
    before = rtsys.get_allocation_stats()
    irr_val = conn.execute(
        "SELECT irr(cashflow, period, investment, target_npv) "
        "FROM t_irr").fetchone()[0]
    conn.close()
    after = rtsys.get_allocation_stats()

    npv = -10000.0
    for t in range(1, 13):
        npv += 1000.0 / (1.0 + irr_val) ** t
    assert abs(npv) < 1e-6, f"irr wrong under forced parallelism: npv={npv}"
    assert after.alloc - before.alloc == after.free - before.free, (
        "irr example combine-path NRT leak under forced parallelism")


def _load_examples_common():
    """Import examples/_common.py, the helpers that the examples share, and return it.

    Like _import_irr_example, it puts examples/ on sys.path first.
    """
    examples_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    import _common  # type: ignore  # noqa: E402
    return _common


def test_assert_results_match_both_nan_equal():
    """Check that assert_results_match accepts variants that all return NaN, although NaN != NaN in IEEE 754."""
    common = _load_examples_common()
    common.assert_results_match(math.nan, math.nan, label="both nan")


def test_assert_results_match_nan_vs_number_fails():
    """Check that assert_results_match still fails for a NaN against a number."""
    common = _load_examples_common()
    with pytest.raises(AssertionError):
        common.assert_results_match(math.nan, 1.0, label="nan vs number")


def test_format_table_wrong_width_row_raises():
    """Check that format_table raises ValueError for a row whose number of cells differs from the number of headers.

    A longer row would otherwise raise IndexError, and a shorter one would be truncated.
    """
    common = _load_examples_common()
    headers = ["A", "B"]
    alignments = ["<", "<"]
    with pytest.raises(ValueError):
        common.format_table(headers, [["1", "2", "3"]], alignments)
    with pytest.raises(ValueError):
        common.format_table(headers, [["1"]], alignments)


def test_format_table_renders():
    """Check that format_table renders one line for the header and one for each data row."""
    common = _load_examples_common()
    out = common.format_table(
        ["Variant", "Time"],
        [["Python", "1.000s"], ["JIT", "0.010s"]],
        ["<", ">"],
    )
    lines = out.splitlines()
    assert len(lines) == 3
    assert "Variant" in lines[0] and "Time" in lines[0]
    assert "Python" in lines[1]
    assert "JIT" in lines[2]


# ---- The examples' UDFs and UDAFs in DuckDB: NULL input, sentinels and errors ----

def _import_example(name):
    """Return the module ``name`` of examples/, imported with examples/ on sys.path as the example launchers do."""
    examples_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    return __import__(name)


def test_haversine_udf_nulls_and_antipodal():
    """Check how the haversine UDFs handle a NULL input, an antipodal pair and a NaN coordinate.

    hv_jit reads the validity masks and writes the NaN sentinel for a row with a NULL input. It clamps ``a`` for an
    antipodal pair whose ``a`` overshoots 1.0 by about 2 ULP, and returns half the circumference, about pi * R km,
    instead of the NaN that asin would return. The Python, Arrow and jitted variants agree on the antipodal rows,
    because they compute ``a`` in the same s*s form. For a NaN coordinate, hv_jit and hv_arrow return NaN. hv_py
    returns NaN too, but DuckDB's bridge for Python scalar UDFs turns a returned NaN into SQL NULL.
    """
    hv = _import_example("haversine")
    conn = duckdb.connect()
    conn.create_function("hv_py", hv.haversine_py, ["DOUBLE"] * 4, "DOUBLE")
    variants = ["hv_py", "hv_jit"]
    try:
        # haversine_arrow imports pyarrow only when DuckDB calls it, so the test imports pyarrow here first. Otherwise
        # the registration would succeed on a host without pyarrow, and the ImportError would appear in the middle of
        # the query as a _duckdb.Error.
        import pyarrow.compute  # noqa: F401
        conn.create_function(
            "hv_arrow", hv.haversine_arrow, ["DOUBLE"] * 4, "DOUBLE",
            type="arrow")
        variants.insert(1, "hv_arrow")
    except ImportError:
        pass
    hv.register_jit_udf(conn)
    # The row with rid 2 is a finite pair, nearly antipodal, whose float64 `a` overshoots 1.0 by about 2 ULP. Without
    # the clamp, math.asin(sqrt(a)) raises in hv_py and returns NaN in hv_jit.
    #
    # The UDFs read directly from the VALUES relation, not from a stored table. A NULL DOUBLE stored in an in-memory
    # table reads back as NaN bytes, but the VALUES vector leaves zero bytes, a finite value, in the NULL row's data
    # slot. A read that ignored the validity mask would therefore compute a real distance from them, and the test would
    # fail. With a stored NaN, the test would pass whether or not hv_jit reads the validity mask.
    pts = (
        "(VALUES "
        "(0, 0.0, 0.0, 0.0, 0.0), "
        "(1, 0.0, 0.0, 0.0, 180.0), "
        "(2, 58.1876182035592, 101.86801103432617, "
        "-58.187618203558735, -78.13198896566404), "
        "(3, CAST(NULL AS DOUBLE), 0.0, 10.0, 20.0), "
        "(4, CAST('nan' AS DOUBLE), 0.0, 10.0, 20.0)) "
        "AS t(rid, lat1, lon1, lat2, lon2)")
    cols = ", ".join(f"{v}(lat1, lon1, lat2, lon2)" for v in variants)
    # ORDER BY the row id makes the positional unpack below independent of the order in which DuckDB scans VALUES, as
    # in the fraud_score tests below.
    same, antipode, overshoot, null_row, nan_row = conn.execute(
        f"SELECT {cols} FROM {pts} ORDER BY rid").fetchall()
    conn.close()
    jit = variants.index("hv_jit")
    half = math.pi * 6371.0
    for v in same:
        assert v is not None and abs(v) < 1e-6, same
    # The exact antipode and the overshoot both clamp to half the circumference, where the variants agree. Near the
    # singularity of asin, `a` would differ between the variants if they did not compute it in the same form.
    for label, row in (("antipode", antipode), ("overshoot", overshoot)):
        assert all(v is not None and math.isfinite(v) for v in row), (label, row)
        assert abs(row[jit] - half) < 1.0, (label, row)
        assert max(row) - min(row) < 1e-6, (label, row)
    # NULL-input row: hv_jit writes the NaN sentinel, and hv_py and hv_arrow return NULL.
    assert null_row[jit] is not None and math.isnan(null_row[jit]), null_row
    for i, v in enumerate(variants):
        if v != "hv_jit":
            assert null_row[i] is None, (v, null_row)
    # NaN-coordinate row: every variant computes with the NaN. hv_jit and hv_arrow return NaN, and DuckDB's bridge for
    # Python scalar UDFs turns hv_py's NaN into SQL NULL. hv_arrow's NaN checks the if_else form of its clamp: a clamp
    # built on a minimum would turn the NaN `a` into 1.0 and return the antipodal distance.
    assert nan_row[jit] is not None and math.isnan(nan_row[jit]), nan_row
    assert nan_row[variants.index("hv_py")] is None, nan_row
    if "hv_arrow" in variants:
        arrow = variants.index("hv_arrow")
        assert nan_row[arrow] is not None and math.isnan(nan_row[arrow]), nan_row


def test_haversine_udf_null_aggregation_divergence():
    """Check where hv_jit's NaN for a NULL input agrees with the SQL NULL of the reference hv_py, and where it does not.

    A filtering count, as in the demo's ``WHERE dist < 50``, gives the same answer for both, because ``NaN < 50`` and
    ``NULL < 50`` both exclude the row. A sum does not: the NaN makes the jitted sum NaN, while SQL's sum skips hv_py's
    NULL. The NaN behaves like NULL only under a predicate that excludes the row. DuckDB orders NaN above every other
    DOUBLE, so ``dist >= X`` keeps the row, and so does the negation of any such predicate.
    """
    hv = _import_example("haversine")
    conn = duckdb.connect()
    conn.create_function("hv_py", hv.haversine_py, ["DOUBLE"] * 4, "DOUBLE")
    hv.register_jit_udf(conn)
    # The query reads directly from VALUES, not from a stored table. A NULL DOUBLE stored in an in-memory table reads
    # back as NaN bytes, so the sum would be NaN with or without a validity check, and the isnan assertion below would
    # pass either way. The VALUES vector leaves zero bytes in the NULL row's data slot instead: without the check,
    # hv_jit would add a real distance, and the assertion would fail.
    pts = (
        "(VALUES "
        "(0.0, 0.0, 0.0, 0.0), "
        "(0.0, 0.0, 89.0, 179.0), "
        "(CAST(NULL AS DOUBLE), 0.0, 10.0, 20.0)) AS t(lat1, lon1, lat2, lon2)")
    args = "lat1, lon1, lat2, lon2"
    c_py = conn.execute(
        f"SELECT count(*) FROM {pts} WHERE hv_py({args}) < 50").fetchone()[0]
    c_jit = conn.execute(
        f"SELECT count(*) FROM {pts} WHERE hv_jit({args}) < 50").fetchone()[0]
    assert c_py == c_jit == 1, (c_py, c_jit)
    s_py = conn.execute(f"SELECT sum(hv_py({args})) FROM {pts}").fetchone()[0]
    s_jit = conn.execute(f"SELECT sum(hv_jit({args})) FROM {pts}").fetchone()[0]
    conn.close()
    assert s_py is not None and math.isfinite(s_py), s_py
    assert s_jit is not None and math.isnan(s_jit), s_jit


def test_fraud_score_udf_null_sentinel():
    """Check that fr_jit gives a valid row its positive rule score, and a row with a NULL input the sentinel -1.

    The UDF writes the sentinel in place of NULL, because ducklib binds no function that would let it write SQL NULL.
    """
    fr = _import_example("fraud_score")
    conn = duckdb.connect()
    fr.register_jit_udf(conn)
    conn.execute(
        "CREATE TABLE tx AS SELECT * FROM (VALUES "
        "(5000.0, CAST(1 AS TINYINT), CAST(2 AS TINYINT), CAST(3 AS TINYINT), "
        "CAST(4 AS TINYINT), CAST(20 AS INTEGER)), "
        "(CAST(NULL AS DOUBLE), CAST(1 AS TINYINT), CAST(1 AS TINYINT), "
        "CAST(12 AS TINYINT), CAST(0 AS TINYINT), CAST(0 AS INTEGER))) "
        "AS t(amount, country, home_country, hour, risk_tier, recent_txns)")
    rows = conn.execute(
        "SELECT fr_jit(amount, country, home_country, hour, risk_tier, "
        "recent_txns) FROM tx ORDER BY amount NULLS LAST").fetchall()
    conn.close()
    assert rows[0][0] > 0, rows[0]              # valid row (amount = 5000)
    assert rows[1][0] == fr.NULL_SCORE, rows[1]  # NULL-amount row sorts last


def test_fraud_score_udf_null_divergence():
    """Check that fr_jit's sentinel makes its sum differ from fr_py's, and that filtering it out removes the difference.

    fr_jit cannot write SQL NULL, so it writes -1 for a row with a NULL input, where the reference fr_py returns SQL
    NULL. SQL's sum skips fr_py's NULL but adds fr_jit's -1, so a caller must filter out ``NULL_SCORE`` before it
    aggregates fr_jit's scores.
    """
    fr = _import_example("fraud_score")
    types = ["DOUBLE", "TINYINT", "TINYINT", "TINYINT", "TINYINT", "INTEGER"]
    conn = duckdb.connect()
    conn.create_function("fr_py", fr.fraud_score_py, types, "INTEGER")
    fr.register_jit_udf(conn)
    conn.execute(
        "CREATE TABLE tx AS SELECT * FROM (VALUES "
        "(1500.0, CAST(1 AS TINYINT), CAST(2 AS TINYINT), CAST(3 AS TINYINT), "
        "CAST(4 AS TINYINT), CAST(20 AS INTEGER)), "
        "(CAST(NULL AS DOUBLE), CAST(1 AS TINYINT), CAST(1 AS TINYINT), "
        "CAST(12 AS TINYINT), CAST(0 AS TINYINT), CAST(0 AS INTEGER))) "
        "AS t(amount, country, home_country, hour, risk_tier, recent_txns)")
    cols = "amount, country, home_country, hour, risk_tier, recent_txns"
    per_row = conn.execute(
        f"SELECT fr_py({cols}), fr_jit({cols}) FROM tx ORDER BY amount NULLS LAST"
    ).fetchall()
    assert per_row[0] == (17, 17), per_row[0]
    assert per_row[1][0] is None, per_row[1]           # fr_py returns SQL NULL
    assert per_row[1][1] == fr.NULL_SCORE, per_row[1]  # fr_jit returns the -1 sentinel
    s_py, s_jit = conn.execute(
        f"SELECT sum(fr_py({cols})), sum(fr_jit({cols})) FROM tx").fetchone()
    assert s_py == 17, s_py            # SQL sum skips the NULL row
    assert s_jit == 16, s_jit          # SQL sum adds the -1 sentinel
    s_jit_filtered = conn.execute(
        f"SELECT sum(s) FROM (SELECT fr_jit({cols}) AS s FROM tx) q "
        f"WHERE s <> {fr.NULL_SCORE}").fetchone()[0]
    conn.close()
    assert s_jit_filtered == s_py, s_jit_filtered


def test_fraud_score_udf_all_valid_fast_path():
    """Check that fr_jit's scores match fr_py's exactly, row by row, on data without NULL.

    DuckDB then returns a validity pointer of 0 for every input, so fr_jit runs its loop without validity checks.
    """
    fr = _import_example("fraud_score")
    types = ["DOUBLE", "TINYINT", "TINYINT", "TINYINT", "TINYINT", "INTEGER"]
    conn = duckdb.connect()
    conn.create_function("fr_py", fr.fraud_score_py, types, "INTEGER")
    fr.register_jit_udf(conn)
    fr.setup_data(conn, 5000)
    cols = "amount, country, home_country, hour, risk_tier, recent_txns"
    mism = conn.execute(
        f"SELECT count(*) FROM transactions "
        f"WHERE fr_py({cols}) IS DISTINCT FROM fr_jit({cols})").fetchone()[0]
    conn.close()
    assert mism == 0, mism


def test_online_scoring_missing_key_raises():
    """Check that score_jit raises RuntimeError for an id that has no feature row.

    The point lookup for a missing id returns no rows, so duckdb_fetch_chunk returns a null chunk (``chunk_p == 0``),
    and the loop raises instead of dereferencing it. The same guard also checks for an empty chunk (``size == 0``), a
    case that this path does not reach.
    """
    osc = _import_example("online_scoring")
    conn = duckdb.connect()
    osc.setup_features(conn, 8)
    ids = numpy.array([0, 999999], dtype=numpy.int64)
    x = numpy.zeros((2, 4), dtype=numpy.float64)
    with pytest.raises(RuntimeError):
        osc.score_jit(conn, ids, x)
    conn.close()


def test_online_scoring_null_feature_raises():
    """Check that both scorers raise for an id whose feature is NULL.

    score_jit reads the validity mask and raises RuntimeError. The Python reference, score_python, raises TypeError when
    it multiplies None by a float. Without the mask, score_jit would read whatever the NULL slot holds and return a
    wrong score without an error.
    """
    osc = _import_example("online_scoring")
    conn = duckdb.connect()
    osc.setup_features(conn, 8)
    conn.execute("UPDATE features SET w2 = NULL WHERE id = 3")
    ids = numpy.array([3], dtype=numpy.int64)
    x = numpy.ones((1, 4), dtype=numpy.float64)
    with pytest.raises(TypeError):
        osc.score_python(conn, ids, x)
    with pytest.raises(RuntimeError):
        osc.score_jit(conn, ids, x)
    conn.close()


def test_online_scoring_big_knob_scales_workload():
    """Check that NUMBDUCK_BENCH_BIG enlarges the workload that the example scores, not only the data it allocates.

    sample_sizes derives the latency and scaling sample sizes from the dataset size, so a dataset ten times larger
    scores proportionally more events, while the default sizes stay the same. The scaling sample is always smaller than
    the latency sample.
    """
    osc = _import_example("online_scoring")
    assert osc.sample_sizes(50_000) == (5_000, 2_000)
    lat_big, scale_big = osc.sample_sizes(500_000)
    assert lat_big > 5_000 and scale_big > 2_000
    assert scale_big < lat_big


def test_irr_udaf_group_sentinels():
    """Check the three kinds of result that the irr aggregate gives under GROUP BY.

    A normal group gives a finite rate, a group whose root lies outside the bracket gives +inf, and a group whose rows
    are all NULL gives NaN, the result for a group without data.
    """
    irr = _import_example("irr")
    conn = duckdb.connect()
    irr.register_irr(conn)
    conn.execute(
        "CREATE TABLE cf AS SELECT * FROM (VALUES "
        "(1, -10000.0, 0.0, 0.0, 0.0), "
        "(1, 13000.0, 12.0, 0.0, 0.0), "
        "(2, 1000.0, 1.0, 1.0, 0.0), "
        "(3, CAST(NULL AS DOUBLE), 12.0, 10000.0, 0.0)) "
        "AS t(g, cashflow, period, investment, target_npv)")
    res = dict(conn.execute(
        "SELECT g, irr(cashflow, period, investment, target_npv) "
        "FROM cf GROUP BY g ORDER BY g").fetchall())
    conn.close()
    expected = (13000.0 / 10000.0) ** (1.0 / 12.0) - 1.0
    assert math.isfinite(res[1]) and abs(res[1] - expected) < 1e-9, res[1]
    assert res[2] == math.inf, res[2]
    assert math.isnan(res[3]), res[3]


# A harness for an irr update callback that raises. It checks that update rolls a push that fails in the middle of a
# row back to the start of the row. The two vectors, cashflows and periods, then keep the same length and stay paired.
# The harness is built on first use, not when this module is imported, so the first test that needs it imports the irr
# example and compiles the probe.
@functools.cache
def _irr_rollback_probe():
    irr = _import_example("irr")
    irr_state_type = irr.irr_state_type

    @njit
    def probe_impl(info, chunk, states):
        n = ducklib.duckdb_data_chunk_get_size(chunk)
        cf = carray(_cast_int_to_void_p(ducklib.duckdb_vector_get_data(
            ducklib.duckdb_data_chunk_get_vector(chunk, 0))), (n,), dtype=numpy.float64)
        pd = carray(_cast_int_to_void_p(ducklib.duckdb_vector_get_data(
            ducklib.duckdb_data_chunk_get_vector(chunk, 1))), (n,), dtype=numpy.float64)
        inv = carray(_cast_int_to_void_p(ducklib.duckdb_vector_get_data(
            ducklib.duckdb_data_chunk_get_vector(chunk, 2))), (n,), dtype=numpy.float64)
        tgt = carray(_cast_int_to_void_p(ducklib.duckdb_vector_get_data(
            ducklib.duckdb_data_chunk_get_vector(chunk, 3))), (n,), dtype=numpy.float64)
        slots = carray(_cast_int_to_void_p(states), (n,), dtype=numpy.intp)
        for i in range(n):
            slot = carray(_cast_int_to_void_p(slots[i]), (1,), dtype=numpy.intp)
            if slot[0] == 0:
                continue
            s = borrow_structref(irr_state_type, slot[0])
            n0 = s.cashflows.size
            try:
                vector_push(s.cashflows, cf[i])
                vector_push(s.periods, pd[i])
                # A negative period marks the poison row. The probe raises through a nested call after both pushes
                # have grown the vectors, so the except branch must reset both lengths to the start of the row. Both
                # rollback lines then have an effect; if the probe raised before the second push, the reset of
                # periods would change nothing.
                if pd[i] < 0.0:
                    _cfunc_guard_raise(1.0)
                if s.initialized == 0:
                    s.investment = inv[i]
                    s.target_npv = tgt[i]
                    s.initialized = 1
            except Exception:
                s.cashflows.size = n0
                s.periods.size = n0

    @cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
    def probe_cb(info, chunk, states):
        probe_impl(info, chunk, states)

    return irr, probe_cb


def _register_irr_with_update(conn, name, update_cb):
    from numbduck.pybridge import extract_connection_ptr

    irr = _import_example("irr")
    conn_p = extract_connection_ptr(conn)
    func_p = ducklib.duckdb_create_aggregate_function()
    ducklib.duckdb_aggregate_function_set_name(func_p, get_unicode_data_p(name))
    dbl = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
    for _ in range(4):
        ducklib.duckdb_aggregate_function_add_parameter(func_p, dbl)
    ducklib.duckdb_aggregate_function_set_return_type(func_p, dbl)
    tb = numpy.array([dbl], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(tb.ctypes.data)
    ducklib.duckdb_aggregate_function_set_functions(
        func_p, irr._irr_state_size_cb.address,
        irr._irr_init_cb.address, update_cb.address,
        irr._irr_combine_cb.address,
        irr._irr_finalize_cb.address)
    ducklib.duckdb_aggregate_function_set_destructor(
        func_p, irr._irr_destroy_cb.address)
    rc = ducklib.duckdb_register_aggregate_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess
    fb = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_aggregate_function(fb.ctypes.data)


@pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning")
def test_irr_update_rollback_drops_poison_row_cleanly():
    """Check the rollback in the IRR update callback through a copy of it that can be made to fail.

    The real ``_irr_update_cb`` in examples/irr.py rolls back only when ``vector_push`` fails to allocate in the middle
    of a row, and SQL input cannot cause that. ``_irr_rollback_probe`` copies the callback's rollback logic and raises
    on a poison row, one with a negative period, after both pushes have grown the vectors, so the rollback must reset
    both lengths. The rollback must restore the start of the row, so that the group's cashflows and periods stay
    paired: the result equals that of the group without the poison row, and NRT frees all that it allocates.

    The probe copies the rollback's source; it does not register the example's callback. This test therefore does not
    catch a regression in the example itself. No test exercises the except branches of the example's finalize and
    combine either: SQL input cannot make finalize raise, and combine raises only when a vector fails to allocate.
    Those branches are checked by reading the code.
    """
    from numba.core.runtime import _nrt_python as _nrt
    _nrt.memsys_enable_stats()
    from numba.core.runtime.nrt import rtsys

    _, probe_cb = _irr_rollback_probe()
    conn = duckdb.connect()
    _register_irr_with_update(conn, "irr_probe", probe_cb)
    conn.execute(
        "CREATE TABLE cfp AS SELECT * FROM (VALUES "
        "(-10000.0, 0.0, 0.0, 0.0), "
        "(99999.0, -5.0, 0.0, 0.0), "
        "(13000.0, 12.0, 0.0, 0.0)) "
        "AS t(cashflow, period, investment, target_npv)")
    before = rtsys.get_allocation_stats()
    rate = conn.execute(
        "SELECT irr_probe(cashflow, period, investment, target_npv) "
        "FROM cfp").fetchone()[0]
    conn.close()
    after = rtsys.get_allocation_stats()
    leak = (after.alloc - before.alloc) - (after.free - before.free)
    expected = (13000.0 / 10000.0) ** (1.0 / 12.0) - 1.0
    assert leak == 0, f"rollback path leaked NRT delta {leak}"
    assert rate is not None and abs(rate - expected) < 1e-9, rate


# --- The names in the source of ducklib's bindings ---

def _called_c_symbol(arg):
    """Return the C symbol that the first argument of a ``_call_lib_func`` call names.

    The symbol is the string itself, or the value that the named variable has in ``ducklib``. For any other expression,
    the function returns None.
    """
    if isinstance(arg, ast.Constant):
        return arg.value
    if isinstance(arg, ast.Name):
        return getattr(ducklib, arg.id, None)
    return None


def _decorator_signature_key(node):
    """Return the key of the signatures table that a binding's decorator names, or None if it names none.

    The key is the string passed to ``signatures.get``, or the bignum name that ``_bignum_proxy`` receives as its
    second argument.
    """
    key = None
    for dec in node.decorator_list:
        if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name)
                and dec.func.id == "_bignum_proxy"):
            key = dec.args[1].value
        args = list(getattr(dec, "args", []))
        args += [kw.value for kw in getattr(dec, "keywords", [])]
        for arg in args:
            if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute)
                    and arg.func.attr == "get"
                    and isinstance(arg.func.value, ast.Name)
                    and arg.func.value.id == "signatures"
                    and arg.args and isinstance(arg.args[0], ast.Constant)):
                key = arg.args[0].value
    return key


def _wrapper_name_triples(source):
    """Yield a tuple for each module-level def in ``source`` whose last statement returns a ``_call_lib_func`` call.

    Each tuple is ``(def_name, decorator_key, body_literal, param_names, call_arg_names)``. ``decorator_key`` is the
    signatures key that the decorator names, and ``body_literal`` is the C symbol that the ``_call_lib_func`` call
    names. ``param_names`` lists the function's parameters, and ``call_arg_names`` the names in the tuple of arguments
    passed to ``_call_lib_func``. ``call_arg_names`` is None when that argument is not a plain tuple of names, so the
    caller's comparison fails instead of skipping the binding.
    """
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        ret = node.body[-1]
        if not (isinstance(ret, ast.Return) and isinstance(ret.value, ast.Call)
                and isinstance(ret.value.func, ast.Name)
                and ret.value.func.id == "_call_lib_func"):
            continue
        body_literal = _called_c_symbol(ret.value.args[0])
        call_arg_names = None
        if len(ret.value.args) > 1 and isinstance(ret.value.args[1], ast.Tuple):
            elts = ret.value.args[1].elts
            if all(isinstance(e, ast.Name) for e in elts):
                call_arg_names = [e.id for e in elts]
        param_names = [a.arg for a in node.args.args]
        yield node.name, _decorator_signature_key(node), body_literal, param_names, call_arg_names


def _proxy_decorated_names(source):
    """Return the names of the module-level functions in ``source`` that carry a proxy decorator.

    The decorators are ``proxy``, ``proxy_if_available`` and ``_bignum_proxy``, and every binding carries one.
    test_wrapper_name_integrity compares this set with the functions that ``_wrapper_name_triples`` yields. A binding
    whose shape the walk misses then fails the test, instead of making the checked set smaller without notice.
    """
    names = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            func = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(func, ast.Name) and func.id in ("proxy", "proxy_if_available", "_bignum_proxy"):
                names.add(node.name)
    return names


# The bignum bindings call whichever spelling the loaded libduckdb exports, so before duckdb 1.4.0 their body names a
# C symbol other than the function's own name.
_ALIASED_BINDINGS = {
    "duckdb_create_bignum": {"duckdb_create_bignum", "duckdb_create_varint"},
    "duckdb_get_bignum": {"duckdb_get_bignum", "duckdb_get_varint"},
}


def test_wrapper_name_integrity():
    """Check that each binding names its C function consistently and passes its parameters through unchanged.

    A binding spells its C function's name three times: as the key that its decorator passes to ``signatures.get``, as
    the name of the def, and as the string passed to ``_call_lib_func``. A typo in one of them that matched another key
    would compile against the wrong signature without an error. The test checks that the three agree and that each is
    a key of the signatures table.

    numbox's ``_call_lib_func`` takes only the result type from the signatures table, so a binding that dropped or
    swapped an argument would still compile and pass wrong values at run time. The test therefore also checks that the
    tuple passed to ``_call_lib_func`` holds exactly the def's parameters, in order, and that their number matches the
    signature's.
    """
    with open(ducklib.__file__, encoding="utf-8") as f:
        source = f.read()
    triples = list(_wrapper_name_triples(source))
    # The walk recognizes only a module-level def whose last statement returns a _call_lib_func call. It would skip,
    # without notice, a binding that moved under an if or a try, gained a statement after its return, or switched to
    # another call helper, such as numbox's _call_lib_func_byval. Comparing the walked set with the proxy-decorated
    # set, instead of with a minimum count, turns that into a failure that names the binding.
    seen = {triple[0] for triple in triples}
    assert seen == _proxy_decorated_names(source), {
        "unchecked": sorted(_proxy_decorated_names(source) - seen),
        "unexpected": sorted(seen - _proxy_decorated_names(source)),
    }
    for name, decorator_key, body_literal, param_names, call_arg_names in triples:
        assert body_literal in _ALIASED_BINDINGS.get(name, {name}), (name, body_literal)
        assert decorator_key == name, (name, decorator_key)
        assert name in ducklib.signatures, name
        assert body_literal in ducklib.signatures, body_literal
        assert call_arg_names == param_names, (name, param_names, call_arg_names)
        # The number of parameters is checked against the signature of the symbol that the body calls, which for a
        # bignum binding is the spelling chosen at import.
        assert len(param_names) == len(ducklib.signatures[body_literal].args), (
            name, param_names, str(ducklib.signatures[body_literal]))


@pytest.mark.skipif(duckdb_major_version < 2, reason="Testing duckdb>=2.0.0 behavior.")
def test_open_result_error():
    """Check that, from duckdb 2.0.0 on, a prepare fails on a connection with an open result until the result completes.

    duckdb 2.0.0 requires ``conn.complete()``, which ``complete_open_result`` calls, to mark that the last query has
    finished streaming from the connection. Until then, ``duckdb_prepare`` on the connection fails with a Resource In
    Use error. https://github.com/duckdb/duckdb-python/releases/tag/v2.0.0
    """
    from numbox.utils.cstrings import c_string
    from numbduck.pybridge import extract_connection_ptr
    conn = duckdb.connect()
    conn_ptr = extract_connection_ptr(conn)
    _ = conn.execute("SELECT 137;")
    stmt = create_duckdb_prepared_statement()
    sql = "SELECT 101;"
    try:
        with c_string(sql) as sql_p:
            rc = ducklib.duckdb_prepare(conn_ptr, sql_p, stmt.ctypes.data)
        assert rc == 1, rc
        error_ptr = ducklib.duckdb_prepare_error(stmt[0])
        error = get_str_from_p_as_int(error_ptr)
        assert error == "Resource In Use Error: connection has an open result; drain or destroy it before starting a new query"  # noqa: E501
        complete_open_result(conn)
        with c_string(sql) as sql_p:
            rc = ducklib.duckdb_prepare(conn_ptr, sql_p, stmt.ctypes.data)
        assert rc == 0, rc
    finally:
        ducklib.duckdb_destroy_prepare(stmt.ctypes.data)


@pytest.mark.skipif(duckdb_major_version >= 2, reason="Testing duckdb<2.0.0 behavior.")
def test_api_call_completes_open_query():
    """Check that, before duckdb 2.0.0, a prepare succeeds on a connection whose result is still open.

    The open result still returns its remaining rows afterward, because its four rows lie in the chunk that Python
    has already fetched. A larger result loses the rows past that chunk. This is the counterpart of
    test_open_result_error for duckdb 1.x.
    """
    from numbox.utils.cstrings import c_string
    from numbduck.pybridge import extract_connection_ptr
    conn = duckdb.connect()
    conn_ptr = extract_connection_ptr(conn)
    cur = conn.execute("SELECT * FROM range(15, 19);")
    assert cur.fetchone() == (15,)
    assert cur.fetchone() == (16,)
    stmt = create_duckdb_prepared_statement()
    sql = "SELECT 101;"
    with c_string(sql) as sql_p:
        rc = ducklib.duckdb_prepare(conn_ptr, sql_p, stmt.ctypes.data)
    assert rc == 0, rc
    assert cur.fetchall() == [(17,), (18,)]


@pytest.mark.skipif(duckdb_major_version < 2, reason="Testing duckdb>=2.0.0 behavior.")
def test_python_open_result():
    """Check the behavior of test_open_result_error through the duckdb Python API alone.

    From duckdb 2.0.0 on, ``conn.sql`` on a connection with an open result raises ``ResourceInUseException`` until
    ``complete_open_result`` completes the result. A second ``conn.execute`` succeeds.
    """
    conn = duckdb.connect()
    conn.execute("SELECT 1")
    with pytest.raises(duckdb.ResourceInUseException):
        conn.sql("SELECT 2").fetchall()
    complete_open_result(conn)
    assert conn.sql("SELECT 2").fetchall() == [(2,)]
