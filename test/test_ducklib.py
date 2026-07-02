import ctypes
import hashlib
import math
import os
import subprocess
import sys

import duckdb
import numpy
import pytest
from numba import njit, cfunc, carray
from numba import types as nb_types
from numba.experimental import structref
from numbox.utils.lowlevel import get_unicode_data_p
from numbox.utils.meminfo import (
    borrow_structref, export_meminfo, _incref_meminfo,
    release_meminfo, structref_meminfo
)

from numba.core.types import intp

from numbduck import ducklib
from numbduck.configurations import get_jit_options
from numbduck.duckdb_utils import (
    create_duckdb_connection, create_duckdb_data_chunk,
    create_duckdb_database, create_duckdb_prepared_statement,
    create_duckdb_result
)
from numbox.utils.lowlevel import _cast_int_to_void_p
from numbox.utils.lowlevel import array_data_p


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


def test_query():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    print(f"out_result = {out_result}")
    num_of_columns = out_result[0]
    assert num_of_columns == 2, f"expected 'i', 'j', got {num_of_columns} columns"
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


def test_duckdb_column_count_and_duckdb_row_count():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    out_result_p = out_result.ctypes.data
    num_of_cols = ducklib.duckdb_column_count(out_result_p)
    num_of_rows = ducklib.duckdb_row_count(out_result_p)
    assert num_of_cols == 2, num_of_cols
    assert num_of_rows == 3, num_of_rows
    aux_close_db(duckdb_database, duckdb_connection)


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
    return duckdb_result, data_chunk_p, duckdb_database, duckdb_connection


def test_duckdb_data_chunk_get_column_count():
    duckdb_result, data_chunk_p, duckdb_database, duckdb_connection = aux_get_data_vector()
    col_count = ducklib.duckdb_data_chunk_get_column_count(data_chunk_p)
    assert col_count == 2, f"Expected 2 columns, got {col_count}"
    aux_close_db(duckdb_database, duckdb_connection)


def test_duckdb_data_chunk_get_size():
    duckdb_result, data_chunk_p, duckdb_database, duckdb_connection = aux_get_data_vector()
    chunk_size = ducklib.duckdb_data_chunk_get_size(data_chunk_p)
    assert chunk_size == 3, f"Expected 3 rows in chunk, got {chunk_size}"
    aux_close_db(duckdb_database, duckdb_connection)


def test_duckdb_destroy_data_chunk():
    duckdb_result, data_chunk_p, duckdb_database, duckdb_connection = aux_get_data_vector()
    assert data_chunk_p != 0, f"Expected valid chunk pointer, got {data_chunk_p}"
    chunk_size = ducklib.duckdb_data_chunk_get_size(data_chunk_p)
    assert chunk_size > 0, f"Expected rows in chunk before destroy, got {chunk_size}"
    data_chunk = create_duckdb_data_chunk()
    data_chunk[0] = data_chunk_p
    data_chunk_pp = data_chunk.ctypes.data
    ducklib.duckdb_destroy_data_chunk(data_chunk_pp)
    assert data_chunk[0] == 0, f"Expected null after destroy, got {data_chunk[0]}"
    aux_close_db(duckdb_database, duckdb_connection)


def test_duckdb_fetch_chunk_exhausted():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    duckdb_result = tuple(out_result)
    chunk_p = ducklib.duckdb_fetch_chunk(duckdb_result)
    assert chunk_p != 0, "Expected first chunk, got null"
    chunk_p = ducklib.duckdb_fetch_chunk(duckdb_result)
    assert chunk_p == 0, f"Expected null for exhausted result, got {chunk_p}"
    aux_close_db(duckdb_database, duckdb_connection)


def test_duckdb_fetch_chunk_data_chunk_get_vector_get_data_vector():
    duckdb_result, data_chunk_p, duckdb_database, duckdb_connection = aux_get_data_vector()
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
    aux_close_db(duckdb_database, duckdb_connection)


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
    """Read a DuckDB inline string (4-byte uint32 length + char data).
    https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L365 """
    str_len = ctypes.c_uint32.from_address(data_p).value
    raw = (ctypes.c_char * str_len).from_address(data_p + 4)
    return raw[:].decode()


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


def test_bind_invalid_param_index():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::INTEGER;")
    assert rc == ducklib.DuckDBSuccess, f"Prepare failed, rc = {rc}"
    rc = ducklib.duckdb_bind_int32(stmt[0], 999, 42)
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError for invalid param index, got {rc}"
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


def test_bind_boolean_invalid_param_index():
    """duckdb_bind_boolean returns DuckDBError for out-of-range param index.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_boolean """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::BOOLEAN;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_boolean(stmt[0], 999, 1)
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError, got {rc}"
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


def test_bind_float_invalid_param_index():
    """duckdb_bind_float returns DuckDBError for out-of-range param index.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_float """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::FLOAT;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_float(stmt[0], 999, 1.0)
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError, got {rc}"
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


def test_bind_date_invalid_param_index():
    """duckdb_bind_date returns DuckDBError for out-of-range param index.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_date """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::DATE;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_date(stmt[0], 999, 0)
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError, got {rc}"
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


def test_bind_timestamp_invalid_param_index():
    """duckdb_bind_timestamp returns DuckDBError for out-of-range param index.
    https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_timestamp """
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::TIMESTAMP;")
    assert rc == ducklib.DuckDBSuccess
    rc = ducklib.duckdb_bind_timestamp(stmt[0], 999, 0)
    assert rc == ducklib.DuckDBError, f"Expected DuckDBError, got {rc}"
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
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)


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
    col3_valid = ducklib.duckdb_validity_row_is_valid(
        intp(v3_validity_p), intp(0))

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


@pytest.mark.skipif(
    not (hasattr(ducklib.duckdb_lib, 'duckdb_create_varint') and hasattr(ducklib.duckdb_lib, 'duckdb_get_varint')),
    reason="duckdb_create_varint or duckdb_get_varint not available in this duckdb version",
)
def test_create_get_varint():
    data_bytes = ctypes.create_string_buffer(b"\x01\x00", 2)
    data_p = ctypes.cast(data_bytes, ctypes.c_void_p).value
    val_p = ducklib.duckdb_create_varint((data_p, 2, 0))
    assert val_p != 0
    result = ducklib.duckdb_get_varint(val_p)
    size = result[1]
    assert size == 2
    raw = (ctypes.c_char * size).from_address(result[0])
    assert raw[:] == b"\x01\x00"
    ducklib.duckdb_free(result[0])
    aux_destroy_value(val_p)


def test_create_get_bit():
    data_bytes = ctypes.create_string_buffer(b"\x05\xA0", 2)
    data_p = ctypes.cast(data_bytes, ctypes.c_void_p).value
    val_p = ducklib.duckdb_create_bit((data_p, 2))
    assert val_p != 0
    result = ducklib.duckdb_get_bit(val_p)
    assert result[1] == 2
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
    # duckdb_get_value_type returns the same handle as the value for
    # scalar types — do NOT destroy both (double-free)
    aux_destroy_value(val_p)


def test_destroy_value():
    val_p = ducklib.duckdb_create_int32(1)
    assert val_p != 0
    aux_destroy_value(val_p)


# --- Struct size guard ---


def test_struct_size_guard():
    """Verify the size computation used by _call_lib_func_struct_in/out."""
    from numba.core.types import UniTuple, Tuple, int32, int64, uint64, uint8

    # 16-byte structs (should pass the ≤16 byte guard)
    assert sum(t.bitwidth for t in UniTuple(int64, 2)) / 8 == 16
    assert sum(t.bitwidth for t in UniTuple(uint64, 2)) / 8 == 16
    assert sum(t.bitwidth for t in Tuple((uint64, int64))) / 8 == 16

    # 8-byte struct
    assert sum(t.bitwidth for t in UniTuple(int32, 2)) / 8 == 8

    # Mixed-width tuples
    assert sum(t.bitwidth for t in Tuple((uint8, uint8, uint64, int64))) / 8 == 18
    assert sum(t.bitwidth for t in Tuple((uint8, uint8, uint8, uint64, int64))) / 8 == 19
    assert sum(t.bitwidth for t in Tuple((uint8, uint8, uint8, uint64, int64))) / 8 > 16

    # 24-byte struct (should fail the guard)
    assert sum(t.bitwidth for t in UniTuple(int64, 3)) / 8 == 24


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






@njit
def _init_cb_impl(info):
    pass


@cfunc(nb_types.void(nb_types.intp))
def _init_cb(info):
    _init_cb_impl(info)


@pytest.mark.skipif(
    not hasattr(ducklib.duckdb_lib, 'duckdb_scalar_function_set_init'),
    reason="duckdb_scalar_function_set_init not available",
)
def test_scalar_function_set_init():
    """Verify set_init callback is accepted (v1.5+ only)."""
    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

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
        return welford_finalize(sb)

    xs = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    expected = float(numpy.std(xs, ddof=1))

    serial = _compute_serial(xs)
    assert abs(serial - expected) < 1e-10, (
        f"serial: got {serial}, expected {expected}")

    combined = _compute_combined(xs[:3], xs[3:])
    assert abs(combined - expected) < 1e-10, (
        f"combined: got {combined}, expected {expected}")

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


def test_verify_cached_dylib_roundtrip(monkeypatch, tmp_path):
    """A cached dylib matching its sidecar passes; a missing sidecar is refused."""
    from numbduck import utils

    cached = tmp_path / "libduckdb.dylib"
    sidecar = tmp_path / "libduckdb.dylib.sha256"
    cached.write_bytes(b"trusted-bytes")
    sidecar.write_text(hashlib.sha256(b"trusted-bytes").hexdigest())

    monkeypatch.setattr(utils, "_LIBDUCKDB_SIDECAR", str(sidecar))
    monkeypatch.setattr(utils, "_LIBDUCKDB_CACHE_DIR", str(tmp_path))

    utils._verify_cached_dylib(str(cached))  # matches -> no raise

    sidecar.unlink()
    with pytest.raises(RuntimeError, match="integrity sidecar"):
        utils._verify_cached_dylib(str(cached))


def test_load_duckdb_rejects_tampered_cache(monkeypatch, tmp_path):
    """A cached standalone libduckdb whose bytes no longer match its sidecar is
    refused before it is dlopen'd, simulating local cache poisoning. No network.
    """
    from numbduck import utils

    cached = tmp_path / "libduckdb.dylib"
    sidecar = tmp_path / "libduckdb.dylib.sha256"
    cached.write_bytes(b"trusted-bytes")
    sidecar.write_text(hashlib.sha256(b"trusted-bytes").hexdigest())
    # Poison the cached file without updating the sidecar.
    cached.write_bytes(b"attacker-controlled-native-code")

    monkeypatch.setattr(utils, "_LIBDUCKDB_CACHED_DYLIB", str(cached))
    monkeypatch.setattr(utils, "_LIBDUCKDB_SIDECAR", str(sidecar))
    monkeypatch.setattr(utils, "_LIBDUCKDB_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(utils, "find_duckdb_shared_lib", lambda: "/fake/wheel.so")
    monkeypatch.setattr(utils, "load_lib_path", lambda path: object())
    # Force the standalone fallback and steer it at the poisoned cache path.
    monkeypatch.setattr(utils, "_has_capi_symbols", lambda lib: False)
    monkeypatch.setattr(utils, "_find_standalone_libduckdb", lambda: str(cached))

    with pytest.raises(RuntimeError, match="integrity check"):
        utils.load_duckdb()


def test_download_refuses_unpinned_version_without_override(monkeypatch):
    """A download for a version with neither a pinned digest nor an explicit
    NUMBDUCK_LIBDUCKDB_SHA256 override is refused before any network access.
    """
    from numbduck import utils

    monkeypatch.setattr(duckdb, "__version__", "9.9.9")
    monkeypatch.delenv("NUMBDUCK_LIBDUCKDB_SHA256", raising=False)
    monkeypatch.setenv("NUMBDUCK_LIBDUCKDB_DOWNLOAD", "1")

    def _no_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr("urllib.request.urlopen", _no_network)

    with pytest.raises(RuntimeError, match="NUMBDUCK_LIBDUCKDB_SHA256"):
        utils._download_libduckdb()


def _build_fake_libduckdb_zip(tmp_path):
    """Write a ZIP holding a fake libduckdb.dylib under a tmp dir.

    Returns (zip_path, dylib_bytes, zip_sha256) so a test can inject the real
    digest of the archive it built and check the extracted bytes and sidecar.
    """
    import io
    import zipfile

    dylib_bytes = b"\x00fake-libduckdb-native-code\x7fELF"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("libduckdb.dylib", dylib_bytes)
    zip_path = tmp_path / "libduckdb-osx-universal.zip"
    zip_path.write_bytes(buf.getvalue())
    return zip_path, dylib_bytes, hashlib.sha256(buf.getvalue()).hexdigest()


def test_fetch_and_cache_local_zip_roundtrip(tmp_path):
    """Drive _fetch_and_cache end-to-end against a local file:// archive with the
    expected digest injected: verify extraction, cache-file and sidecar contents,
    and (on POSIX) the 0o600 file / 0o700 dir permissions. No network.
    """
    from numbduck import utils

    zip_path, dylib_bytes, zip_sha = _build_fake_libduckdb_zip(tmp_path)
    dest = tmp_path / "cache" / "libduckdb.dylib"

    returned = utils._fetch_and_cache(zip_path.as_uri(), str(dest), zip_sha)

    assert returned == str(dest)
    assert dest.read_bytes() == dylib_bytes
    sidecar = dest.parent / "libduckdb.dylib.sha256"
    assert sidecar.read_text() == hashlib.sha256(dylib_bytes).hexdigest()
    if sys.platform != "win32":
        assert oct(os.stat(dest).st_mode & 0o777) == "0o600"
        assert oct(os.stat(dest.parent).st_mode & 0o777) == "0o700"


def test_fetch_and_cache_wrong_digest_refuses(tmp_path):
    """A downloaded ZIP whose SHA-256 does not match the injected expectation is
    refused before anything is extracted or persisted. No network.
    """
    from numbduck import utils

    zip_path, _dylib_bytes, _zip_sha = _build_fake_libduckdb_zip(tmp_path)
    dest = tmp_path / "cache" / "libduckdb.dylib"

    with pytest.raises(RuntimeError, match="does not match"):
        utils._fetch_and_cache(zip_path.as_uri(), str(dest), "0" * 64)
    assert not dest.exists()
    assert not dest.parent.exists()


@pytest.mark.parametrize("headless_mode", ["non_tty", "eof"])
def test_download_consent_headless_raises_branded(monkeypatch, headless_mode):
    """A headless import (stdin not a TTY, or input hitting EOF) gets the branded
    instructional RuntimeError naming NUMBDUCK_LIBDUCKDB_DOWNLOAD=1,
    NUMBDUCK_LIBDUCKDB, and brew, instead of a raw EOFError. No network.
    """
    from numbduck import utils

    # An override digest so the flow reaches the consent gate rather than the
    # unpinned-refusal gate that sits ahead of it.
    monkeypatch.setenv("NUMBDUCK_LIBDUCKDB_SHA256", "0" * 64)
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
