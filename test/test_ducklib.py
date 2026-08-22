import ast
import ctypes
import functools
import math
import os
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
from numbox.utils.lowlevel import get_unicode_data_p
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
from numbox.utils.lowlevel import _cast_int_to_void_p
from numbox.utils.lowlevel import array_data_p


class _HandleFamily:
    """One DuckDB C-handle lifetime family: the binding calls that acquire a
    handle (each paired with a predicate that says whether a given call actually
    produced one) and the calls that release it."""

    def __init__(self, name, acquires, releases):
        self.name = name
        self.acquires = acquires
        self.releases = releases


def _acquired_by_return(args, result):
    return result != 0


def _acquired_by_last_arg(args, result):
    # duckdb_query only allocates a result when a non-null out pointer is passed.
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
    """Count paired acquire/release calls on the DuckDB bindings and assert the
    two match on exit. DuckDB's C API exposes no allocator-stats accessor, so a
    missing destroy or a double free surfaces here as a per-family imbalance in
    the wrapped call counts. Every patched binding is restored before the balance
    is checked, so a failing assertion never leaves the module monkeypatched."""

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
    """Run the decorated test under a handle_balance harness over the given families."""
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
    """Destroy a data chunk fetched via duckdb_fetch_chunk by passing the buffer
    address (pointer-to-handle) to duckdb_destroy_data_chunk."""
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
    """ https://duckdb.org/docs/stable/clients/c/query#duckdb_fetch_chunk """
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
    """Create and destroy scalar-function handles in a long loop under the
    balance harness, so a leaked (never-destroyed) function handle surfaces as
    an acquire/release imbalance."""
    with handle_balance(SCALAR_FUNCTION_FAMILY):
        for _ in range(50):
            fn_p = ducklib.duckdb_create_scalar_function()
            assert fn_p != 0
            fn_buf = numpy.zeros(1, dtype=numpy.intp)
            fn_buf[0] = fn_p
            ducklib.duckdb_destroy_scalar_function(fn_buf.ctypes.data)
            assert fn_buf[0] == 0


def test_destroy_zeroes_handle_slots():
    """duckdb_destroy_* nulls the caller's handle slot for the value,
    logical-type, and scalar/aggregate-function families, exactly as it does
    for the database/connection/prepared/data-chunk handles. This is the only
    direct proof that destroy actually ran for these largest handle families."""
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
    """Prepare a statement and return (prepared_statement buffer, rc)."""
    query_p = get_unicode_data_p(sql)
    stmt = create_duckdb_prepared_statement()
    stmt_pp = stmt.ctypes.data
    rc = ducklib.duckdb_prepare(connection_p, query_p, stmt_pp)
    return stmt, rc


def aux_execute_prepared(stmt_p):
    """Execute a prepared statement, fetch the first chunk. Returns (result, chunk_p)."""
    out_result = create_duckdb_result()
    out_result_p = out_result.ctypes.data
    rc = ducklib.duckdb_execute_prepared(stmt_p, out_result_p)
    assert rc == ducklib.DuckDBSuccess, f"Execute failed, rc = {rc}"
    duckdb_result = tuple(out_result)
    chunk_p = ducklib.duckdb_fetch_chunk(duckdb_result)
    assert chunk_p != 0, "Expected chunk"
    return out_result, chunk_p


def aux_read_column_data(chunk_p, col_idx):
    """Return the raw data pointer for a column in a chunk."""
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, col_idx)
    return ducklib.duckdb_vector_get_data(vec_p)


def aux_read_inline_string(data_p):
    """Read a DuckDB string_t vector entry as text, the text-returning
    counterpart of aux_read_string_t below, which handles both layouts.
    https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L365 """
    return aux_read_string_t(data_p).decode()


def aux_read_string_t(data_p):
    """Read a DuckDB string_t vector entry as raw bytes, handling both layouts:
    inline (length <= 12, chars at data_p + 4) and out-of-line (length > 12,
    4-byte prefix at data_p + 4 and a pointer at data_p + 8).
    https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L365 """
    length = ctypes.c_uint32.from_address(data_p).value
    if length <= 12:
        raw = (ctypes.c_char * length).from_address(data_p + 4)
    else:
        out_p = ctypes.c_void_p.from_address(data_p + 8).value
        raw = (ctypes.c_char * length).from_address(out_p)
    return raw[:]


def aux_destroy_prepared(stmt):
    """Destroy a prepared statement via its buffer."""
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
    """Bind int32, int64, double, varchar, and null in a single statement."""
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
    """duckdb_prepare_error returns a non-null pointer with error text after failed prepare.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_prepare_error """
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
    """duckdb_prepare_error returns null pointer after successful prepare.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_prepare_error """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT 1;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"
    err_p = ducklib.duckdb_prepare_error(stmt[0])
    assert err_p == 0, f"Expected null error pointer for valid SQL, got {err_p}"
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_result_error_on_invalid_query():
    """duckdb_result_error returns a non-null pointer with error text after failed query.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_result_error """
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
    """duckdb_result_error returns null pointer after successful query.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_result_error """
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


# --- Additional Bind Types ---

def test_bind_boolean():
    """Bind boolean values and verify readback.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_boolean """
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
    """Bind float value and verify readback.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_float """
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
    """Bind a date value (days since 1970-01-01) and verify readback.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_date """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::DATE;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"

    # 2025-01-02 = 20089 days since 1970-01-01
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
    """Bind a timestamp value (microseconds since epoch) and verify readback.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_timestamp """
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


# --- Scalar Bind Types ---

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
    # BLOB uses same string_t layout: uint32 length + inline data
    blob_len = ctypes.c_uint32.from_address(data_p).value
    assert blob_len == 4
    raw = (ctypes.c_char * blob_len).from_address(data_p + 4)
    assert raw[:] == b"\x00\x01\x02\x03"
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_varchar_long_out_of_line():
    """A > 12-character VARCHAR uses the out-of-line string_t layout (4-byte
    length, 4-byte prefix, 8-byte data pointer) rather than inlined chars. The
    other string tests only ever exercise the <= 12-byte inline layout."""
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
    """A > 12-byte BLOB uses the same out-of-line string_t layout; the existing
    blob test only covers the 4-byte inline case."""
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


# --- Struct Bind Types ---

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
    # DECIMAL(10,2) uses INT64 physical storage (width <= 18)
    # The stored value is the unscaled integer: 12345 represents 123.45
    stored = (ctypes.c_int64 * 1).from_address(data_p)[0]
    assert stored == 12345
    aux_destroy_data_chunk(chunk_p)
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


def test_bind_decimal_wide_upper64():
    """DECIMAL(38,0) uses INT128 physical storage. Bind a value whose magnitude
    genuinely needs the upper 64 bits and assert both halves of the stored
    int128 round-trip (test_bind_decimal only exercises width <= 18, INT64
    physical, with the upper word always zero)."""
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
    """duckdb_bind_value's only other coverage is the invalid-index sweep, whose
    DuckDBError assertion a completely broken binding would also satisfy. Bind a
    real value handle at a valid index and read the result back so the success
    path, handle marshaling included, is pinned."""
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

# Every duckdb_bind_* wrapper that takes a parameter index, paired with the
# trailing arguments to call it with after an out-of-range index (999). The
# wrapper itself is resolved from the name in the first column, so each binding
# is named exactly once here and a row cannot drift into calling something other
# than what it is labelled. Pass real values, not null: duckdb builds the bound
# value before it checks the index for at least duckdb_bind_blob, where a null
# data pointer aborts the process (std::logic_error, "construction from null is
# not valid") rather than returning DuckDBError. duckdb_bind_parameter_index is
# excluded deliberately: its second argument is an out-pointer for a looked-up
# index, not an index to bind at.
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
    # duckdb_bind_value needs a duckdb_value handle created and destroyed around
    # the call, so it carries None and is dispatched to its own helper below.
    ("duckdb_bind_value", None),
]


def aux_bind_value_out_of_range(stmt_p):
    """duckdb_bind_value takes a duckdb_value handle rather than a plain scalar,
    so the sweep's dummy has to be created and destroyed around the call. A null
    handle happens to return DuckDBError at an out-of-range index, but it
    segfaults at a valid one, so it is not worth relying on."""
    val_p = ducklib.duckdb_create_int64(1)
    assert val_p != 0
    try:
        return ducklib.duckdb_bind_value(stmt_p, 999, val_p)
    finally:
        aux_destroy_value(val_p)


@pytest.mark.parametrize("name,args", _INVALID_INDEX_BINDS,
                         ids=[entry[0] for entry in _INVALID_INDEX_BINDS])
def test_bind_family_invalid_param_index(name, args):
    """Every duckdb_bind_* wrapper that takes a parameter index returns
    DuckDBError (not a crash) for an out-of-range one. Verified on duckdb 1.5.4:
    all of them, including the struct-by-value binds
    (hugeint/uhugeint/interval/decimal), return DuckDBError; none segfault, so
    none are excluded."""
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
    """The sweep table above is hand-written, so a binding added later would
    silently go uncovered. Assert the table names exactly the prepared-statement
    duckdb_bind_* wrappers that take a parameter index, reading the set off
    ducklib itself. Because the sweep resolves each wrapper from the same name
    this test compares, matching name sets certify that every one of those
    wrappers was actually called, not merely listed.

    One exclusion. duckdb_bind_parameter_index resolves a name to an index
    rather than binding at one. There is deliberately no symbol-presence filter:
    every bind wrapper is a plain @proxy binding, so a missing C symbol aborts
    the ducklib import outright rather than producing a callable stub, and a
    presence filter would let a future version-gated bind wrapper drop out of
    both sides of this comparison silently. If one ever appears, this test
    should fail and force it into the table with its own gating."""
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
    """Exercise the by-value duckdb_result path (duckdb_fetch_chunk,
    duckdb_result_return_type, duckdb_result_statement_type) with the numba
    optimizer disabled, so the >16-byte struct argument is marshalled the
    ABI-correct way (byval stack copy on SysV x86-64) rather than relying on
    the optimized stack layout happening to place the value where the callee
    reads it. Runs in a subprocess because NUMBDUCK_JIT_OPTIONS is read once,
    at ducklib import time, to build every wrapper's jit_options.
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
    assert get_jit_options() == {"cache": False, "_dbg_optnone": True}


# --- JIT Tests ---
# get_unicode_data_p is safe inside @njit with numbox >= 0.5.6, which
# extracts the data pointer directly instead of going through NRT meminfo.

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


@njit
def jit_open_close():
    db = create_duckdb_database()
    rc = ducklib.duckdb_open(
        get_unicode_data_p(':memory:'), array_data_p(db))
    db_p = db[0]
    ducklib.duckdb_close(array_data_p(db))
    return rc, db_p


def test_jit_open_close_database():
    rc, db_p = jit_open_close()
    assert rc == ducklib.DuckDBSuccess, f"open failed, rc={rc}"
    assert db_p != 0, f"expected valid pointer, got {db_p}"


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
    return open_rc, db_p, connect_rc, conn_p, query_rc, conn_after


def test_jit_connect_query_disconnect():
    open_rc, db_p, connect_rc, conn_p, query_rc, conn_after = jit_connect_query_disconnect()
    assert open_rc == ducklib.DuckDBSuccess, f"open failed, rc={open_rc}"
    assert db_p != 0, f"expected valid db pointer, got {db_p}"
    assert connect_rc == ducklib.DuckDBSuccess, f"connect failed, rc={connect_rc}"
    assert conn_p != 0, f"expected valid connection pointer, got {conn_p}"
    assert query_rc == ducklib.DuckDBSuccess, f"query failed, rc={query_rc}"
    assert conn_after == 0, f"expected null after disconnect, got {conn_after}"


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
    return open_rc, connect_rc, conn_p, rc, detected


def test_jit_query_invalid_sql():
    """A DuckDBError return code is produced and branched on entirely inside
    compiled @njit code (the native LLVM call path), not just via the
    Python-called dispatcher path every other error test uses.

    Open and connect are asserted first. duckdb_query checks the query string
    before the connection, so on a null connection it segfaults rather than
    returning DuckDBError, and a crash inside compiled code takes the session
    down with no traceback."""
    open_rc, connect_rc, conn_p, rc, detected = jit_query_invalid_sql()
    assert open_rc == ducklib.DuckDBSuccess, f"open failed, rc={open_rc}"
    assert connect_rc == ducklib.DuckDBSuccess, f"connect failed, rc={connect_rc}"
    assert conn_p != 0, "expected a valid connection pointer"
    assert rc == ducklib.DuckDBError, f"expected DuckDBError, got {rc}"
    assert detected == 1, "in-JIT DuckDBError comparison did not observe the error"


# --- JIT: Prepared Statements ---

@njit
def jit_prepare_bind_execute():
    db = create_duckdb_database()
    conn = create_duckdb_connection()
    stmt = create_duckdb_prepared_statement()

    open_rc = ducklib.duckdb_open(0, array_data_p(db))
    connect_rc = ducklib.duckdb_connect(db[0], array_data_p(conn))
    conn_p = conn[0]

    # prepare: SELECT $1::INTEGER, $2::BIGINT, $3::DOUBLE, $4::INTEGER
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

    return (open_rc, connect_rc, prepare_rc, nparams,
            bind1_rc, bind2_rc, bind3_rc, bind4_rc, exec_rc,
            chunk_size, col0, col1, col2, col3_valid)


def test_jit_prepare_bind_execute():
    """Prepared statement with parameter binding from JIT context.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_prepare
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_nparams
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_int32
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_int64
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_double
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_null
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_execute_prepared
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_prepare """
    (open_rc, connect_rc, prepare_rc, nparams,
     bind1_rc, bind2_rc, bind3_rc, bind4_rc, exec_rc,
     chunk_size, col0, col1, col2, col3_valid) = jit_prepare_bind_execute()
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


# --- Value Interface ---

def aux_destroy_value(val_p):
    """Destroy a duckdb_value by storing the handle in a buffer and passing
    the buffer address (pointer-to-handle) to duckdb_destroy_value."""
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
    """duckdb renamed duckdb_create_varint/duckdb_get_varint to the bignum
    spellings in 1.4.0. Same enum value, same struct, different C symbols.

    The binding calls whichever spelling the loaded libduckdb exports, so this
    runs unskipped across the whole supported duckdb range. Covers a multi-byte
    magnitude and both settings of the negative flag."""
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
    """numba's cache key hashes bytecode and closure cells only, so the symbol
    the bignum pair resolves at import reaches neither. A cached object built
    against one spelling is served to a process needing the other, and importing
    ducklib aborts with LLVM ERROR: Symbol not found.

    The check reads each dispatcher's cache object rather than the cache
    directory: the pair compiles at import, so a directory listing taken here
    cannot tell a fresh write from files left behind by an earlier build, and it
    sees nothing at all when NUMBA_CACHE_DIR points elsewhere. duckdb_create_uuid
    is the control, a sibling that does cache."""
    assert "cache" not in ducklib._bignum_jit_options
    for name in ("duckdb_create_bignum", "duckdb_get_bignum"):
        assert isinstance(getattr(ducklib, name)._cache, NullCache), name
    assert isinstance(ducklib.duckdb_create_uuid._cache, FunctionCache)


def test_bignum_binds_the_spelling_this_libduckdb_exports():
    """The binding resolves to the bignum spelling on duckdb >=1.4 and to the
    varint spelling below it, and never to a symbol the library lacks."""
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
    """A version-gated binding whose C symbol is absent from libduckdb raises a
    clear, named error at @njit typing time, not an opaque numba TypingError.

    The absent-symbol path is simulated with a lib object that lacks the
    attribute, so this holds whether or not the loaded libduckdb has it."""

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
    """The actual version-gated ducklib binding, when absent from the loaded
    libduckdb, raises a clear named error at @njit typing time rather than an
    opaque numba TypingError. Complements the simulated case above by covering
    the binding as ducklib actually built it."""

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
    # duckdb_get_list_* reject an ARRAY-typed value, so check the array through
    # its logical type instead: id, fixed size, and element type. Do not destroy
    # array_type_p; duckdb_get_value_type returns the value handle itself.
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
    # duckdb_get_value_type returns the value handle itself; assert the
    # aliasing so the do-not-destroy rule stays pinned by a check the call
    # can actually fail (destroying both would double-free).
    assert type_p == val_p
    aux_destroy_value(val_p)


def test_destroy_value():
    val_p = ducklib.duckdb_create_int32(1)
    assert val_p != 0
    aux_destroy_value(val_p)


# --- Struct-by-value ABI round-trip ---


def test_struct_by_value_abi_roundtrip():
    """Round-trip a 16-byte and a 24-byte struct through the real create/get
    wrappers, exercising numbox's <=16-byte by-value vs >16-byte by-pointer
    struct-ABI classification. Both eightbytes of each struct's 128-bit payload
    are fully populated, so a dropped or mis-packed upper eightbyte is caught.
    (The arithmetic-only predecessor asserted sum(bitwidth)/8 over numba type
    metadata and never called into ducklib, so it could not detect an ABI
    regression at all.)"""
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

    # duckdb_decimal is padded to 24 bytes and therefore passed by pointer; a
    # width > 18 keeps INT128 physical storage so the upper eightbyte is live.
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
    """Register a scalar UDF that adds 1 to an integer, call it from SQL."""
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
    """Verify extra_info pointer round-trips through set/get."""
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
    """Verify set_error in callback causes query failure."""
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
    """Register overloaded scalar function with integer and double variants."""
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
    """Register an aggregate UDF that sums int32 values, call it from SQL."""
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


# The compiled callback below captures _INIT_CB_FIRED_ADDR at import time, so
# any reset must write through the existing buffer in place; rebinding
# _INIT_CB_FIRED to a fresh array would leave the callback writing freed
# memory from a DuckDB worker thread.
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
    """Verify the set_init callback (v1.5+) is not just accepted but actually
    invoked: the init impl increments an observable marker, asserted after the
    query. A binding that silently dropped the init pointer would leave it 0.

    The assertion is >= 1, not == 1. duckdb.h documents the scalar init callback
    as running once per worker thread that executes the function, so the count is
    a property of the query plan rather than of the binding; a single-row
    constant projection happens to give exactly one today."""
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
    """Register overloaded aggregate with int32 and int64 variants."""
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


    # Integer input variant (reuses existing agg callbacks)
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
    """Register a JIT-compiled UDF on a Python duckdb connection via numbduck."""
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
    """Compare JIT UDF registered via numbduck C API with a pure Python UDF."""
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
    """Benchmark three UDF approaches: Python scalar, Python Arrow, numbduck JIT.

    Computes x*x at 10K, 100K, and 1M rows. The JIT UDF runs as a raw C
    function pointer with no Python overhead per invocation, while the Python
    UDFs round-trip through the interpreter (scalar: per-row, Arrow: per-batch
    via pyarrow).

    Results (WSL2, Python 3.12, duckdb 1.5.1, numba 0.60):
        Rows      Python       Arrow         JIT    Py/JIT   Arr/JIT
      10,000     0.749s      0.003s      0.000s     2,095x        7x
     100,000     7.382s      0.019s      0.001s    13,879x       36x
   1,000,000   118.835s      0.242s      0.001s   130,568x      266x
    """
    import sys
    import time
    from numbduck.pybridge import extract_connection_ptr

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


# ---- structref <-> raw MemInfo pointer bridge intrinsics ----
#
# See test_ducklib.md for a full explanation of the DuckDB aggregate
# lifecycle, the removerefctpass interaction, and why release_meminfo
# uses NRT_MemInfo_release while the others don't.
#
# The state slot is an 8-byte stretch of memory owned by DuckDB. The
# initializer writes a MemInfo pointer into it. MemInfo is NRT's handle
# for memory allocated to store a data structure (structref, array, etc.).
# NRT manages that memory via a reference count. The initializer bumps
# the refcount by one so NRT won't free the memory when the callback
# returns and its local variables go out of scope.
#
# Reference counting contract:
#   export_meminfo(s):   +1 incref; returned intp keeps the allocation alive.
#   borrow_structref():  +1 incref on entry; local decref on scope exit
#                        balances it. Net zero for the external owner.
#   release_meminfo(p):  -1 decref; triggers dtor at refcount 0.


# WARNING: This intrinsic does not work as intended. It was written to
# observe intermediate refcount states (e.g. refcount==2 while both a
# local structref and an exported MemInfo pointer are live in the same
# @njit scope). The idea was that reading refcount from JIT code would
# let the test assert the full 1→2→1→2→1→0 ladder at each step.
#
# It fails because of removerefctpass (numba/core/lowering.py L207-210).
# That pass checks whether the function's args and return type involve
# NRT-tracked types. If they don't — and our bridge functions return
# plain intp, not structrefs — it strips ALL NRT_incref/NRT_decref calls
# from the function. So by the time this intrinsic reads the refcount,
# the incref from export_meminfo has already been removed by the pass.
# The refcount reads 1 (initial allocation), not 2.
#
# The test still works because the stripping is symmetric: the export's
# incref AND the local variable's scope-exit decref are both removed,
# so the net refcount change is zero and the object survives correctly.
# But the intermediate refcount==2 state never actually exists at runtime.
#
# Python-side _read_refcount() between separate @njit calls is the only
# way to verify post-scope-exit refcounts, which is what the ladder test
# does. This intrinsic is preserved as a cautionary example.
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
    """Read NRT MemInfo refcount from Python via ctypes.

    MemInfo.refct is the first field (size_t) — stable since numba 0.50.
    See numba/core/runtime/nrt.cpp.
    """
    return int(ctypes.c_size_t.from_address(meminfo_intp).value)


# ---- Heap-owning structref (for nested-dtor test) ----

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
    """Prove export/borrow/release maintain refcount invariants step by step.

    Each njit function is its own scope so numba's decref fires on return,
    giving us clean refcount readings from Python between calls.
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

    # See refcount_of_meminfo above for why in-scope refcount==2 checks
    # don't work. We verify via Python-side _read_refcount between calls.

    # Step 1: allocate + export. After return, refcount == 1.
    p = _allocate_and_export()
    assert p != 0, "export_meminfo returned null"
    rc = _read_refcount(p)
    assert rc == 1, f"expected refcount 1 after export+local-drop, got {rc}"

    # Step 2: borrow + read. After return, refcount still 1.
    mean, count, m2 = _borrow_and_read(p)
    assert mean == 1.5, f"mean={mean}"
    assert count == 2, f"count={count}"
    assert m2 == 3.25, f"m2={m2}"
    rc = _read_refcount(p)
    assert rc == 1, f"expected refcount 1 after borrow-drop, got {rc}"

    # Step 3: release the slot's reference. refcount → 0, dtor runs.
    _do_release(p)


def test_array_meminfo_bridge_refcount_ladder():
    """Array bridge using inline incref + carray access.

    Unlike structrefs, arrays allocate via NRT_MemInfo_alloc in the same
    function body. This disables removerefctpass, so both the manual incref
    AND the scope-exit decref survive. Net: +1 -1 = 0 on top of the initial
    refcount of 1 from allocation. The memory stays alive.

    The incref MUST be done via the @intrinsic _incref_meminfo (which inlines)
    rather than the @njit export_meminfo wrapper (separate compilation unit
    where removerefctpass WOULD strip it).
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


# ---- Array-backed variance UDAF callbacks for DuckDB ----
#
# Demonstrates the array bridge pattern: state is a numpy array instead
# of a structref. state_size=16 stores two intp values: meminfo_p (for
# lifecycle) and data_p (for carray access). The incref is done via the
# @intrinsic _incref_meminfo directly in the allocating function so it
# inlines — removerefctpass is already disabled there because numpy.zeros
# generates NRT_MemInfo_alloc*.
#
# State layout: float64 array of 3 elements [sum, sum_sq, count].

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
    """Merge src into tgt in place. Chan et al. pairwise formula."""
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


# NOTE: skips NULL validity checks — test data has no NULLs.
# Production use must call duckdb_vector_get_validity() and check bits.
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
    """Verify Welford ops match numpy.std(ddof=1) without DuckDB."""

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
    """Prove release_meminfo cascades dtor into heap-owning fields."""
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

    # Warmup: single cycle to measure per-cycle alloc delta
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

    # 100 cycles: exact multiple of warmup
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
    """End-to-end: structref-backed Welford stddev UDAF in DuckDB."""
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
    """End-to-end: array-backed population variance UDAF in DuckDB.

    Mirrors test_aggregate_function_structref_stddev but uses a numpy array
    for state instead of a structref. state_size=16 stores meminfo_p + data_p.
    Demonstrates the inline incref + carray access pattern.
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


# --- Multi-chunk and NULL-aware coverage ---

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
        # A null validity pointer means the vector is all-valid; otherwise bit i
        # of the mask is the row's validity.
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
    """Drive a scalar UDF over more than STANDARD_VECTOR_SIZE rows so its
    callback runs on chunks past the first, then loop duckdb_fetch_chunk to
    reassemble the multi-chunk result and assert the values straddling the
    2048-row chunk boundary."""
    n_rows = 5000
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    result = create_duckdb_result()
    # Bind the SQL to a name: get_unicode_data_p borrows the str's data pointer,
    # so a runtime-built string must outlive the duckdb_query call.
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
    """Drive a UDAF that produces more than STANDARD_VECTOR_SIZE groups so
    finalize is called across multiple output chunks, then loop
    duckdb_fetch_chunk over the grouped result and assert the group sums
    across the chunk boundaries (grouped output is not guaranteed to fill
    chunks to STANDARD_VECTOR_SIZE; duckdb 1.3.2 emits 1295-row chunks)."""
    n_groups = 5000
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    result = create_duckdb_result()
    # get_unicode_data_p borrows the str's data pointer, so hold the runtime-built
    # SQL in a name until duckdb_query has read it.
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
    # each singleton group g has my_sum(v) == g
    assert sums == list(range(n_groups))
    assert sums[STANDARD_VECTOR_SIZE - 1] == STANDARD_VECTOR_SIZE - 1
    assert sums[STANDARD_VECTOR_SIZE] == STANDARD_VECTOR_SIZE

    ducklib.duckdb_destroy_result(result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


@balanced(RESULT_FAMILY, CHUNK_FAMILY)
def test_scalar_function_null_aware():
    """A scalar UDF over a NULL-containing column. Special handling passes the
    NULL rows to the callback, which reads duckdb_vector_get_validity and maps
    them to a sentinel; a validity-blind callback would return the NULL rows'
    (arbitrary) storage instead."""
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
    """A UDAF over a NULL-containing column. Special handling passes the NULL
    rows to update, which reads duckdb_vector_get_validity and counts only the
    valid rows; the result matches DuckDB's NULL-ignoring count(v), whereas a
    validity-blind update would instead count every row (count(*))."""
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
    """The standalone-fallback loader refuses a libduckdb whose version differs.

    Simulates the macOS dual-runtime seam on any platform: force the wheel to
    look symbol-less, hand back a standalone libduckdb whose reported version
    disagrees with the installed duckdb wheel, and assert load_duckdb refuses.
    """
    from numbduck import utils

    wheel_lib = object()
    standalone_lib = object()

    def fake_load_lib_path(path):
        return wheel_lib if path == "/fake/wheel.so" else standalone_lib

    monkeypatch.setattr(utils, "find_duckdb_shared_lib", lambda: "/fake/wheel.so")
    monkeypatch.setattr(utils, "load_lib_path", fake_load_lib_path)
    monkeypatch.setattr(utils, "_has_capi_symbols", lambda lib: lib is standalone_lib)
    monkeypatch.setattr(
        utils, "_find_standalone_libduckdb", lambda: "/fake/libduckdb.dylib")
    monkeypatch.setattr(utils, "_library_version", lambda lib: "9.9.9")

    with pytest.raises(RuntimeError, match="NUMBDUCK_LIBDUCKDB"):
        utils.load_duckdb()


def test_pybridge_refuses_uncoordinated_runtime(monkeypatch):
    """pybridge refuses to extract a Connection* when runtimes are uncoordinated."""
    from numbduck import pybridge

    monkeypatch.setattr(pybridge, "libraries_coordinated", lambda: False)
    monkeypatch.setattr(pybridge, "loaded_library_version", lambda: "9.9.9")

    conn = duckdb.connect()
    try:
        with pytest.raises(RuntimeError, match="libduckdb"):
            pybridge.extract_connection_ptr(conn)
    finally:
        conn.close()


def test_pybridge_closed_connection_raises_runtime_error():
    """A closed connection resets its unique_ptr<Connection> to null, so the
    documented offset yields a null pointer. extract_connection_ptr must raise a
    clean RuntimeError there rather than pass the null on to duckdb_query (an
    opaque numba TypeError or a NULL-connection dereference).
    """
    from numbduck.pybridge import extract_connection_ptr

    conn = duckdb.connect()
    conn.close()
    with pytest.raises(RuntimeError, match="null"):
        extract_connection_ptr(conn)


def test_load_duckdb_refuses_unverifiable_standalone_version(monkeypatch):
    """A standalone libduckdb whose version cannot be read is refused rather than
    loaded unverified — otherwise it surfaces later as a confusing 'libduckdb
    None' coordination error. No network.
    """
    from numbduck import utils

    wheel_lib = object()
    standalone_lib = object()

    def fake_load_lib_path(path):
        return wheel_lib if path == "/fake/wheel.so" else standalone_lib

    monkeypatch.setattr(utils, "find_duckdb_shared_lib", lambda: "/fake/wheel.so")
    monkeypatch.setattr(utils, "load_lib_path", fake_load_lib_path)
    monkeypatch.setattr(utils, "_has_capi_symbols", lambda lib: lib is standalone_lib)
    monkeypatch.setattr(
        utils, "_find_standalone_libduckdb", lambda: "/fake/libduckdb.dylib")
    monkeypatch.setattr(utils, "_library_version", lambda lib: None)

    with pytest.raises(RuntimeError, match="could not read its library version"):
        utils.load_duckdb()


def test_load_duckdb_refuses_dev_suffix_same_base(monkeypatch):
    """A standalone dev build is refused even when it shares the release's base.

    The coordination check compares the full library version, never a stripped
    ``X.Y.Z``, so a ``.dev``/``rc`` build — a different commit with a possibly
    different Connection layout — never passes as the release that shares its
    base. Pins that we do not "normalize out" the suffix.
    """
    from numbduck import utils

    wheel_lib = object()
    standalone_lib = object()

    def fake_load_lib_path(path):
        return wheel_lib if path == "/fake/wheel.so" else standalone_lib

    monkeypatch.setattr(utils, "find_duckdb_shared_lib", lambda: "/fake/wheel.so")
    monkeypatch.setattr(utils, "load_lib_path", fake_load_lib_path)
    monkeypatch.setattr(utils, "_has_capi_symbols", lambda lib: lib is standalone_lib)
    monkeypatch.setattr(
        utils, "_find_standalone_libduckdb", lambda: "/fake/libduckdb.dylib")
    monkeypatch.setattr(utils, "_wheel_library_version", lambda: "1.5.4")
    monkeypatch.setattr(utils, "_library_version", lambda lib: "1.5.4-dev12")

    with pytest.raises(RuntimeError, match="undefined behavior"):
        utils.load_duckdb()


def test_libraries_coordinated_compares_library_version_not_package(monkeypatch):
    """Coordination compares the wheel's own library version (PRAGMA version),
    not ``duckdb.__version__``. A dev build whose library version matches the
    standalone's is coordinated even though the Python package version uses a
    different pre-release scheme — the pairing the old package-version compare
    wrongly refused. A differing library version still fails closed.
    """
    from numbduck import utils

    monkeypatch.setattr(utils, "loaded_library_version", lambda: "1.6.0-dev10121")
    monkeypatch.setattr(utils, "_wheel_library_version", lambda: "1.6.0-dev10121")
    assert utils.libraries_coordinated() is True

    monkeypatch.setattr(utils, "_wheel_library_version", lambda: "1.6.0-dev9999")
    assert utils.libraries_coordinated() is False


def test_wheel_library_version_reads_pragma_version():
    """_wheel_library_version reports the wheel's own library version from
    PRAGMA version (normalized) — the like-for-like counterpart to a standalone's
    duckdb_library_version().
    """
    from numbduck import utils

    row = duckdb.connect(":memory:").execute("PRAGMA version").fetchone()
    assert utils._wheel_library_version() == utils._normalize_version(row[0])


def test_non_darwin_capi_error_uses_platform_extension(monkeypatch):
    """The non-macOS C-API-missing guidance names the platform's own library
    extension (.dll on Windows, .so elsewhere), not a hard-coded .so.
    """
    from numbduck import utils

    monkeypatch.delenv("NUMBDUCK_LIBDUCKDB", raising=False)
    monkeypatch.setattr(utils.platform, "system", lambda: "Windows")
    assert "libduckdb.dll" in utils._non_darwin_capi_error()
    monkeypatch.setattr(utils.platform, "system", lambda: "Linux")
    assert "libduckdb.so" in utils._non_darwin_capi_error()


def _build_fake_libduckdb_zip(tmp_path):
    """Write a ZIP holding a fake libduckdb.dylib under a tmp dir.

    Returns (zip_path, dylib_bytes) for driving _fetch_and_cache against a local
    file:// archive.
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
    """Drive _fetch_and_cache end-to-end against a local file:// archive: it must
    extract the dylib and atomically write it to the cache path. No network.
    """
    from numbduck import utils

    zip_path, dylib_bytes = _build_fake_libduckdb_zip(tmp_path)
    dest = tmp_path / "cache" / "libduckdb.dylib"

    returned = utils._fetch_and_cache(zip_path.as_uri(), str(dest))

    assert returned == str(dest)
    assert dest.read_bytes() == dylib_bytes


@pytest.mark.parametrize("headless_mode", ["non_tty", "eof"])
def test_download_consent_headless_raises_branded(monkeypatch, headless_mode):
    """A headless import (stdin not a TTY, or input hitting EOF) gets the branded
    instructional RuntimeError naming NUMBDUCK_LIBDUCKDB_DOWNLOAD=1,
    NUMBDUCK_LIBDUCKDB, and brew, instead of a raw EOFError. No network.
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
    """On a non-macOS platform with a symbol-less wheel and no standalone lib,
    load_duckdb names the macOS-only download asymmetry and distinguishes an
    unset NUMBDUCK_LIBDUCKDB from one set to a missing path (echoing it). No
    network.
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
    """Import the IRR UDAF example module (examples/irr.py).

    The module is imported as ``irr`` (the name examples/run_irr.py also uses),
    keeping ``IRRStateType.__module__`` stable so numba type inference matches.
    Importing it also compiles every @cfunc/@njit callback, exercising that the
    finalize -> irr_bisect chain still builds under numba.
    """
    examples_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    import irr
    return irr


def test_irr_bisect_large_magnitude_converges():
    """IRR is scale-invariant, so scaling the sparse single-cashflow case by
    1000 (10,000,000 investment / 13,000,000 cashflow at period 12) must yield
    the same monthly rate as the 10,000 / 13,000 case, and a finite value.

    The pre-fix solver only exited on a fixed absolute NPV tolerance, which the
    residual at the true root cannot reach at large magnitude, so it returned
    NaN despite full convergence. The bracket-width exit returns the converged
    rate at every scale.
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
    """A root outside [-0.99, 10.0] leaves NPV the same sign at both bracket
    ends; the solver returns the IRR_NO_BRACKET sentinel (+inf), which is
    distinct from the NaN the finalize step emits for an empty group.
    """
    irr = _import_irr_example()

    # Root near r=999 (monthly return far above the 10.0 bracket top):
    # NPV(r) = -1 + 1000 / (1 + r) is positive at both r=-0.99 and r=10.
    high = irr.irr_bisect(
        numpy.array([1000.0], dtype=numpy.float64),
        numpy.array([1.0], dtype=numpy.float64), 1, 1.0, 0.0)
    assert high == irr.IRR_NO_BRACKET
    assert not math.isnan(high)

    # Near-total loss, root below -0.99: NPV negative at both bracket ends.
    low = irr.irr_bisect(
        numpy.array([1.0], dtype=numpy.float64),
        numpy.array([1.0], dtype=numpy.float64), 1, 1000.0, 0.0)
    assert low == irr.IRR_NO_BRACKET
    assert not math.isnan(low)


def test_irr_bisect_financing_direction_converges():
    """A financing-direction stream -- cash received up front, repaid later --
    makes NPV increase in r (npv_lo < 0 < npv_hi) instead of decrease, yet it
    still changes sign exactly once inside the bracket. The solver must follow
    the sign change to that root rather than walk the bracket the wrong way to
    an endpoint (which would fabricate a ~10.0 rate).
    """
    irr = _import_irr_example()
    # Receive 1000 now (investment = -1000), repay 1300 at month 12:
    # NPV(r) = 1000 - 1300 / (1 + r)^12 -- negative at r=-0.99, positive at
    # r=10, with its single root at (1300/1000)^(1/12) - 1.
    rate = irr.irr_bisect(
        numpy.array([-1300.0], dtype=numpy.float64),
        numpy.array([12.0], dtype=numpy.float64), 1, -1000.0, 0.0)
    expected = (1300.0 / 1000.0) ** (1.0 / 12.0) - 1.0
    assert math.isfinite(rate), rate
    assert rate != irr.IRR_NO_BRACKET
    assert abs(rate - expected) < 1e-9, (rate, expected)


def test_irr_bisect_large_period_converges():
    """At r=-0.99 the discount factor (1+r)**period underflows to 0.0 for
    period >= ~162; numba's default 'python' error model would raise
    ZeroDivisionError on the divide, which finalize turns into the empty-group
    NaN -- misreporting a valid group as "no data". error_model="numpy" yields
    +inf there instead, so the solver still converges on the true rate.
    """
    irr = _import_irr_example()
    rate = irr.irr_bisect(
        numpy.array([13000.0], dtype=numpy.float64),
        numpy.array([200.0], dtype=numpy.float64), 1, 10000.0, 0.0)
    expected = (13000.0 / 10000.0) ** (1.0 / 200.0) - 1.0
    assert math.isfinite(rate), rate
    assert abs(rate - expected) < 1e-9, (rate, expected)


def test_irr_bisect_nan_input_returns_nan():
    """A NaN cashflow is a legal DOUBLE (not SQL NULL), so it reaches the solver
    and makes an endpoint NaN. It must return the empty-group NaN, not the +inf
    out-of-bracket sentinel a caller reads as "root above the bracket".
    """
    irr = _import_irr_example()
    rate = irr.irr_bisect(
        numpy.array([math.nan], dtype=numpy.float64),
        numpy.array([12.0], dtype=numpy.float64), 1, 10000.0, 0.0)
    assert math.isnan(rate), rate


def test_irr_bisect_zero_cashflow_far_period_matches():
    """A zero cashflow at a far period (>= 162) contributes nothing to NPV at
    any rate, so a fully valid group plus one such row must return the SAME root
    as the group without it. A raw NPV at r=-0.99 instead evaluates 0.0 * inf =
    NaN there (the discount factor underflows to 0.0 for period >= ~162), so the
    old solver misreported the valid group as the empty-group NaN.
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
    """A mixed-sign stream whose opposite-sign cashflows both sit at far periods
    (>= 162) makes a raw NPV at r=-0.99 evaluate inf + -inf = NaN, which the old
    solver misreported as the empty-group NaN. The overflow-free sign evaluation
    keeps both endpoint signs correct, so the single in-bracket root is found.
    Cross-checked against an exact-rational bisection over the same bracket --
    Fraction arithmetic never underflows, so it is independent of the solver's
    floating-point sign trick.
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


# --- @cfunc exception-guard coverage for structref-backed UDAF callbacks ---
#
# A Python exception raised from an @njit impl invoked through a @cfunc is
# swallowed at the C boundary: numba prints it and returns the zero/void default
# without unwinding into DuckDB. When the raise originates in a nested call while
# a borrowed structref is still live, numba also skips that borrow's scope-exit
# decref -- an NRT meminfo leak. A bare try/except inside the impl catches the
# exception in-frame so the decref runs. These tests pin both halves: an
# unguarded raise leaks, the guarded form does not.


@njit
def _cfunc_guard_raise(x):
    """Raise from a nested njit call while the caller holds a live borrow.

    Models vector_push -> numpy.empty raising on allocation failure: the raise
    crosses a call boundary, which is what skips the borrow's decref on an
    unguarded path (a direct in-frame raise still runs the decref).
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
        # Borrow inside the guard so a raise in the nested call releases it.
        try:
            s = borrow_structref(welford_type, slot[0])
            welford_update(s, _cfunc_guard_raise(in_vals[i]))
        except Exception:
            pass


@functools.cache
def _welford_raise_callbacks():
    """Compile the raising-update @cfunc probes on first use, keeping their eager
    compilation out of module import so collection cost stays near base."""

    @cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
    def unguarded_cb(info, chunk, states):
        _welford_update_raise_unguarded_impl(info, chunk, states)

    @cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
    def guarded_cb(info, chunk, states):
        _welford_update_raise_guarded_impl(info, chunk, states)

    return unguarded_cb, guarded_cb


def _register_raising_welford(conn, name, update_cb):
    """Register a Welford UDAF that reuses the shared init/combine/finalize/
    destroy callbacks but swaps in a raising update callback."""
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
    """Register a Welford UDAF whose update raises via a nested call, drive it
    over a non-empty table, and return the NRT (alloc - free) delta across the
    query. The raise is swallowed at the @cfunc boundary, so the query reports
    success regardless of whether a leak occurred."""
    from numba.core.runtime import _nrt_python as _nrt
    _nrt.memsys_enable_stats()
    from numba.core.runtime.nrt import rtsys

    conn = duckdb.connect()
    _register_raising_welford(conn, "wraise", update_cb)
    conn.execute("CREATE TABLE t_raise AS SELECT 1.0 AS v FROM range(8)")
    before = rtsys.get_allocation_stats()
    row = conn.execute("SELECT wraise(v) FROM t_raise").fetchone()
    after = rtsys.get_allocation_stats()
    conn.close()
    delta = (after.alloc - before.alloc) - (after.free - before.free)
    return delta, row[0]


@pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning")
def test_udaf_cfunc_unguarded_raise_leaks_borrowed_state():
    """Control: without the guard, a raise from a nested call while a borrowed
    structref is live leaks the borrow's incref -- the C boundary swallows the
    exception and skips the scope-exit decref. This establishes that the raise
    is real and consequential, so the guarded assertion below is meaningful.

    The swallow makes numba print the ignored exception, which pytest surfaces
    as an unraisable-exception warning; it is expected here and filtered."""
    unguarded_cb, _ = _welford_raise_callbacks()
    leak, _ = _run_raising_udaf_nrt_delta(unguarded_cb)
    assert leak > 0, f"expected an NRT leak without the guard, got delta {leak}"


def test_udaf_cfunc_guarded_raise_no_leak():
    """A bare try/except around the borrow and its nested raising call catches
    the exception in-frame, so numba runs the borrow's decref: NRT alloc == free
    even though every row's update raises. This is the guard the example
    aggregate callbacks use."""
    _, guarded_cb = _welford_raise_callbacks()
    leak, value = _run_raising_udaf_nrt_delta(guarded_cb)
    assert leak == 0, f"guarded raise must not leak, got NRT delta {leak}"
    # Every row's update raised, so the guard dropped them all: the Welford
    # finalize sees an empty state and yields no real number. This pins that the
    # induced raise actually fired -- otherwise the result would be a stddev.
    assert value is None or math.isnan(value), value


# ---- Merge-counter UDAF: proves DuckDB invokes the C-API combine callback ----

@structref.register
class _MergeStateType(nb_types.StructRef):
    pass


_merge_state_fields = [("total", nb_types.float64), ("merges", nb_types.int64)]
# cache=False keeps numbox's generated structref constructor out of numba's
# on-disk cache. That cache is keyed by struct name + fields only and resolves
# _MergeStateType through the module name it was first compiled under, so once it
# is warm, importing this file under any other qualified name fails type
# inference (examples/irr.py documents the same trap for IRRState). Disabling the
# cache has no measurable cost here and keeps the module importable under any name.
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
        # Count each merged partial so finalize can surface whether combine ran.
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
    """Compile the merge-counter UDAF @cfunc callbacks on first use, keeping their
    eager compilation out of module import so collection cost stays near base.
    Returns the six callbacks in DuckDB's callback order."""

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
    """Drive the irr example UDAF under forced parallelism so DuckDB produces
    multiple partial states and invokes the C-API combine callback, inside an
    NRT alloc==free region, and assert the aggregate is still correct. A sibling
    merge-counter UDAF proves that this configuration actually triggers combine
    on this DuckDB build -- correctness alone cannot, since a single-partition
    plan would be correct too."""
    from numba.core.runtime import _nrt_python as _nrt
    _nrt.memsys_enable_stats()
    from numba.core.runtime.nrt import rtsys

    irr = _import_irr_example()

    conn = duckdb.connect()
    # PRAGMA verify_parallelism forces the parallel aggregate path even for tiny
    # inputs on duckdb 1.5.4, deterministically exercising combine. (SET
    # verify_parallelism=true is not a recognized parameter -- only the PRAGMA.)
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
    after = rtsys.get_allocation_stats()
    conn.close()

    npv = -10000.0
    for t in range(1, 13):
        npv += 1000.0 / (1.0 + irr_val) ** t
    assert abs(npv) < 1e-6, f"irr wrong under forced parallelism: npv={npv}"
    assert after.alloc - before.alloc == after.free - before.free, (
        "irr example combine-path NRT leak under forced parallelism")


def _load_examples_common():
    """Import examples/_common.py (the shared example utilities), using the same
    examples/ sys.path insertion as _import_irr_example."""
    examples_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    import _common
    return _common


def test_assert_results_match_both_nan_equal():
    """Variants that all legitimately produce NaN agree; NaN != NaN under
    IEEE-754 must not be reported as a spurious mismatch."""
    common = _load_examples_common()
    common.assert_results_match(math.nan, math.nan, label="both nan")


def test_assert_results_match_nan_vs_number_fails():
    """A NaN against a real value is a genuine disagreement and still fails."""
    common = _load_examples_common()
    with pytest.raises(AssertionError):
        common.assert_results_match(math.nan, 1.0, label="nan vs number")


def test_format_table_wrong_width_row_raises():
    """A row whose cell count differs from the header count is rejected up front,
    both when it is too long (would IndexError) and too short (would truncate)."""
    common = _load_examples_common()
    headers = ["A", "B"]
    alignments = ["<", "<"]
    with pytest.raises(ValueError):
        common.format_table(headers, [["1", "2", "3"]], alignments)
    with pytest.raises(ValueError):
        common.format_table(headers, [["1"]], alignments)


def test_format_table_renders():
    """A well-formed table renders one line per header row and data row."""
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


# ---- Example UDF/UDAF integration coverage (NULL handling, sentinels, errors) ----

def _import_example(name):
    """Import an examples/ module by name via the same examples/ sys.path
    insertion the example launchers use."""
    examples_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)
    return __import__(name)


def test_haversine_udf_nulls_and_antipodal():
    """hv_jit emits the NaN sentinel for a NULL-input row (validity gating) and
    clamps a genuine ~2-ULP antipodal overshoot to the half-circumference
    (~pi*R km) instead of returning NaN from an asin domain overshoot. The
    per-row Python/Arrow/JIT variants agree on the antipodal and overshoot rows
    (aligned s*s form); a NaN coordinate surfaces as NaN from hv_jit/hv_arrow
    and as SQL NULL from hv_py (the scalar-UDF bridge maps its NaN return)."""
    hv = _import_example("haversine")
    conn = duckdb.connect()
    conn.create_function("hv_py", hv.haversine_py, ["DOUBLE"] * 4, "DOUBLE")
    variants = ["hv_py", "hv_jit"]
    try:
        # haversine_arrow imports pyarrow lazily at call time, so probe it
        # eagerly here: without this, registration succeeds on a pyarrow-less
        # host and the ImportError surfaces mid-query as a _duckdb.Error.
        import pyarrow.compute  # noqa: F401
        conn.create_function(
            "hv_arrow", hv.haversine_arrow, ["DOUBLE"] * 4, "DOUBLE",
            type="arrow")
        variants.insert(1, "hv_arrow")
    except ImportError:
        pass
    hv.register_jit_udf(conn)
    # Row 3 is a real finite near-antipodal pair whose float64 `a` overshoots 1.0
    # by ~2 ULP: unclamped, math.asin(sqrt(a)) raises (hv_py) or yields NaN
    # (hv_jit). Read the UDFs directly off the VALUES relation (no intervening
    # TABLE): a materialized NULL DOUBLE decodes to NaN, but the VALUES vector
    # leaves the NULL row's data slot as finite zero-page bytes, so an ungated
    # read would fold them into a real distance -- that is what makes the
    # validity gate observable here rather than coinciding with a stored NaN.
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
    # ORDER BY an explicit row id so the positional unpack below is independent
    # of VALUES scan order (mirrors the sibling divergence/sentinel tests).
    same, antipode, overshoot, null_row, nan_row = conn.execute(
        f"SELECT {cols} FROM {pts} ORDER BY rid").fetchall()
    conn.close()
    jit = variants.index("hv_jit")
    half = math.pi * 6371.0
    for v in same:
        assert v is not None and abs(v) < 1e-6, same
    # Exact antipode and the 2-ULP overshoot both clamp to the half-circumference
    # and the variants agree there (a would otherwise diverge near the asin
    # singularity if the formula forms were not aligned).
    for label, row in (("antipode", antipode), ("overshoot", overshoot)):
        assert all(v is not None and math.isfinite(v) for v in row), (label, row)
        assert abs(row[jit] - half) < 1.0, (label, row)
        assert max(row) - min(row) < 1e-6, (label, row)
    # NULL-input row: hv_jit writes the NaN sentinel; hv_py/hv_arrow yield NULL.
    assert null_row[jit] is not None and math.isnan(null_row[jit]), null_row
    for i, v in enumerate(variants):
        if v != "hv_jit":
            assert null_row[i] is None, (v, null_row)
    # NaN-coordinate row: every variant computes through the NaN. hv_jit and
    # hv_arrow surface it as NaN -- pinning the Arrow clamp's if_else form,
    # since a min-based clamp would collapse the NaN `a` to 1.0 and fabricate
    # the antipodal distance -- while DuckDB's scalar-UDF bridge maps hv_py's
    # NaN return to SQL NULL.
    assert nan_row[jit] is not None and math.isnan(nan_row[jit]), nan_row
    assert nan_row[variants.index("hv_py")] is None, nan_row
    if "hv_arrow" in variants:
        arrow = variants.index("hv_arrow")
        assert nan_row[arrow] is not None and math.isnan(nan_row[arrow]), nan_row


def test_haversine_udf_null_aggregation_divergence():
    """hv_jit emits NaN for a NULL-input row while the reference hv_py yields SQL
    NULL. Under the demo's filtering count (WHERE dist < 50) the two agree (NaN
    < 50 and NULL < 50 both exclude the row), but under sum() the NaN poisons the
    JIT aggregate while hv_py skips the NULL -- the NaN sentinel round-trips as
    NULL only under predicates that exclude the row (DuckDB orders NaN as the
    largest DOUBLE, so dist >= X and negations retain it)."""
    hv = _import_example("haversine")
    conn = duckdb.connect()
    conn.create_function("hv_py", hv.haversine_py, ["DOUBLE"] * 4, "DOUBLE")
    hv.register_jit_udf(conn)
    # Query straight off VALUES (not a stored TABLE): a materialized NULL DOUBLE
    # decodes to NaN bytes, which would mask the validity gate, whereas the
    # VALUES vector leaves the NULL row's data slot as finite zero-page bytes --
    # so without the gate hv_jit folds a real distance in and the sum assertion
    # below (isnan) would fail instead of passing vacuously.
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
    """fr_jit scores a normal row with its positive rule sum and any NULL-input
    row with the -1 sentinel (a scalar UDF cannot write SQL NULL here)."""
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
    """The JIT variant cannot emit SQL NULL, so on a NULL-input row it writes the
    -1 sentinel while the reference fr_py yields SQL NULL. A naive sum() therefore
    diverges (fr_py skips NULLs, fr_jit folds in -1); filtering the sentinel
    restores agreement -- the contract the NULL_SCORE comment documents."""
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
    assert per_row[1][0] is None, per_row[1]           # fr_py -> SQL NULL
    assert per_row[1][1] == fr.NULL_SCORE, per_row[1]  # fr_jit -> -1 sentinel
    s_py, s_jit = conn.execute(
        f"SELECT sum(fr_py({cols})), sum(fr_jit({cols})) FROM tx").fetchone()
    assert s_py == 17, s_py            # SQL sum skips the NULL row
    assert s_jit == 16, s_jit          # -1 sentinel folded into the total
    s_jit_filtered = conn.execute(
        f"SELECT sum(s) FROM (SELECT fr_jit({cols}) AS s FROM tx) q "
        f"WHERE s <> {fr.NULL_SCORE}").fetchone()[0]
    conn.close()
    assert s_jit_filtered == s_py, s_jit_filtered


def test_fraud_score_udf_all_valid_fast_path():
    """With no NULL inputs every validity pointer is zero, so fr_jit runs the
    check-free fast loop; its per-row scores must still match fr_py exactly."""
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
    """score_jit's point lookup matches exactly one feature row; a missing id
    makes duckdb_fetch_chunk return a NULL chunk (chunk_p == 0), which the loop's
    null-pointer guard turns into a RuntimeError instead of dereferencing it. The
    companion ``size == 0`` half of that guard is defensive: this missing-key path
    reaches the null-pointer half, not the empty-chunk half."""
    osc = _import_example("online_scoring")
    conn = duckdb.connect()
    osc.setup_features(conn, 8)
    ids = numpy.array([0, 999999], dtype=numpy.int64)
    x = numpy.zeros((2, 4), dtype=numpy.float64)
    with pytest.raises(RuntimeError):
        osc.score_jit(conn, ids, x)
    conn.close()


def test_online_scoring_null_feature_raises():
    """A NULL feature value for a matched id would be read as stale storage and
    silently mis-scored; the JIT loop reads the validity mask and raises, matching
    the pure-Python reference (which raises on None*float)."""
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
    """NUMBDUCK_BENCH_BIG must enlarge the actually-scored workload, not just the
    allocated dataset: the latency/scaling sample sizes are derived from the
    dataset size, so a 10x-bigger dataset scores proportionally more events while
    the default sizes stay put (and scaling always scores fewer than latency)."""
    osc = _import_example("online_scoring")
    assert osc.sample_sizes(50_000) == (5_000, 2_000)
    lat_big, scale_big = osc.sample_sizes(500_000)
    assert lat_big > 5_000 and scale_big > 2_000
    assert scale_big < lat_big


def test_irr_udaf_group_sentinels():
    """Through the real irr aggregate under GROUP BY: an all-NULL group yields
    NaN (no data), an out-of-bracket group yields +inf, and a normal group
    yields a finite rate -- pinning the finalize sentinel distinctions."""
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


# irr raising-update harness: pins that update rolls a mid-row push failure back
# to the row boundary so cashflows/periods stay paired (not just truncated).
# Built lazily so importing the irr example and compiling this probe stay out of
# module import; the first test that needs it pays the cost.
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
                # A negative period marks the poison row: raise via a nested call
                # AFTER both pushes have grown the vectors, so the except path
                # must reset BOTH lengths to the row boundary -- exercising both
                # rollback lines rather than leaving the periods reset a no-op.
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
    """Validate the update-callback rollback LOGIC via a structural stand-in.

    The real _irr_update_cb rollback (examples/irr.py) fires only on an actual
    vector_push allocation failure mid-row, which SQL input cannot trigger, so
    it is not directly reachable from a test. _irr_rollback_probe builds a copy
    of that rollback logic with an injected raise on a poison row (period < 0)
    placed AFTER both pushes have grown the vectors, so the except path must
    reset both lengths -- both rollback lines are required here. The failure
    is rolled back to the row boundary so the group's cashflows/periods stay
    paired, the aggregate equals the same group without the poison row, and NRT
    stays balanced. This pins the rollback logic but does NOT by itself guard the
    example's real callback against regression -- they share source, not this
    registration. The example's real finalize and combine except guards share
    this coverage gap: they defend against faults SQL input cannot trigger
    (finalize confirmed undrivable across an adversarial input battery; combine
    can fault only on a vector-allocation failure), so no test drives their
    except branch -- they are pinned by inspection, not execution."""
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
    after = rtsys.get_allocation_stats()
    conn.close()
    leak = (after.alloc - before.alloc) - (after.free - before.free)
    expected = (13000.0 / 10000.0) ** (1.0 / 12.0) - 1.0
    assert leak == 0, f"rollback path leaked NRT delta {leak}"
    assert rate is not None and abs(rate - expected) < 1e-9, rate


# --- Binding source integrity ---

def _called_c_symbol(arg):
    """The C symbol named by ``_call_lib_func``'s first argument: the string
    itself, or the value that name has in ``ducklib``."""
    if isinstance(arg, ast.Constant):
        return arg.value
    if isinstance(arg, ast.Name):
        return getattr(ducklib, arg.id, None)
    return None


def _decorator_signature_key(node):
    """The registered signatures key named in a wrapper's decorator: the string
    passed to ``signatures.get(...)``, or the canonical name passed to
    ``_bignum_proxy(<resolved symbol>, "<canonical bignum name>")``."""
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
    """For every ``_call_lib_func`` wrapper defined at module level in `source`,
    yield ``(def_name, decorator_key, body_literal, param_names, call_arg_names)``
    where ``decorator_key`` is the string passed to ``signatures.get(...)`` in the
    wrapper's decorator, ``body_literal`` is the string passed to
    ``_call_lib_func(...)`` in the body, ``param_names`` is the def's parameter
    list, and ``call_arg_names`` is the names inside the ``_call_lib_func``
    argument tuple (None when that argument is not a plain tuple of names, so the
    caller fails loudly instead of skipping the wrapper).
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
    """Names of the module-level functions carrying a proxy decorator, which is
    every binding wrapper. Used to pin how many wrappers the triple walk above is
    expected to see, so one dropping out of its shape is a failure rather than a
    silently smaller set."""
    names = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            func = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(func, ast.Name) and func.id in ("proxy", "proxy_if_available", "_bignum_proxy"):
                names.add(node.name)
    return names


# The bignum bindings call whichever spelling the loaded libduckdb exports, so
# their body names a C symbol that differs from the def's own name below 1.4.0.
_ALIASED_BINDINGS = {
    "duckdb_create_bignum": {"duckdb_create_bignum", "duckdb_create_varint"},
    "duckdb_get_bignum": {"duckdb_get_bignum", "duckdb_get_varint"},
}


def test_wrapper_name_integrity():
    """Every binding names its C function three times by hand (the decorator's
    ``signatures.get`` key, the ``def`` name, and the ``_call_lib_func`` literal);
    a typo in any one that happens to match another registered key would compile
    against the wrong signature silently. Assert all three agree and resolve to a
    registered signatures key. Names alone are not enough: numbox uses the
    registered signature only for the return type, so a wrapper that forwards the
    wrong arguments (one dropped, two swapped) still emits a native call and
    passes garbage at runtime. Assert arity and order too: the ``_call_lib_func``
    tuple must forward exactly the def's parameters, and their count must match
    the registered signature's.
    """
    with open(ducklib.__file__, encoding="utf-8") as f:
        source = f.read()
    triples = list(_wrapper_name_triples(source))
    # Pin the count against the proxy-decorated set rather than a floor. The walk
    # only recognises a module-level def whose last statement returns
    # _call_lib_func, so a wrapper that moves under an if/try, gains a trailing
    # statement, or switches to one of the struct-by-value call helpers drops out
    # of the integrity check silently; comparing the two sets makes that a
    # failure that names the wrapper.
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
        # Check arity against the signature of the symbol actually called, which
        # is the aliased spelling for a binding resolved at import.
        assert len(param_names) == len(ducklib.signatures[body_literal].args), (
            name, param_names, str(ducklib.signatures[body_literal]))
