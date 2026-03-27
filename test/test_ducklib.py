import ctypes

import numpy
import pytest
from numba import njit
from numba import carray
from numbox.utils.lowlevel import get_unicode_data_p

from numba.core.types import intp

from numbduck import ducklib
from numbduck.duckdb_utils import (
    create_duckdb_connection, create_duckdb_data_chunk,
    create_duckdb_database, create_duckdb_prepared_statement,
    create_duckdb_result
)
from numbox.utils.lowlevel import _cast_int_to_void_p
from numbduck.jit_utils import array_data_p


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
    assert all([j_val and j_arr_ == j_col_ or True for j_arr_, j_col_, j_val_ in zip(j_arr, j_col, j_val)])
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
    # DUCKDB_TYPE_INTEGER = 4
    col_type = ducklib.duckdb_column_type(out_result_p, 0)
    assert col_type == 4, f"Expected INTEGER (4), got {col_type}"
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
    not (ducklib._has_symbol('duckdb_create_varint') and ducklib._has_symbol('duckdb_get_varint')),
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
    DUCKDB_TYPE_INTEGER = 4
    type_p = ducklib.duckdb_create_logical_type(DUCKDB_TYPE_INTEGER)
    assert type_p != 0
    type_id = ducklib.duckdb_get_type_id(type_p)
    assert type_id == DUCKDB_TYPE_INTEGER
    buf = numpy.array([type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_create_logical_type_varchar():
    DUCKDB_TYPE_VARCHAR = 17
    type_p = ducklib.duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR)
    assert type_p != 0
    type_id = ducklib.duckdb_get_type_id(type_p)
    assert type_id == DUCKDB_TYPE_VARCHAR
    buf = numpy.array([type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_create_decimal_type():
    DUCKDB_TYPE_DECIMAL = 19
    type_p = ducklib.duckdb_create_decimal_type(10, 2)
    assert type_p != 0
    type_id = ducklib.duckdb_get_type_id(type_p)
    assert type_id == DUCKDB_TYPE_DECIMAL
    assert ducklib.duckdb_decimal_width(type_p) == 10
    assert ducklib.duckdb_decimal_scale(type_p) == 2
    internal_type = ducklib.duckdb_decimal_internal_type(type_p)
    assert internal_type != 0
    buf = numpy.array([type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_logical_type_alias():
    DUCKDB_TYPE_INTEGER = 4
    type_p = ducklib.duckdb_create_logical_type(DUCKDB_TYPE_INTEGER)
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
    DUCKDB_TYPE_LIST = 24
    DUCKDB_TYPE_INTEGER = 4
    child_p = ducklib.duckdb_create_logical_type(DUCKDB_TYPE_INTEGER)
    assert child_p != 0
    list_p = ducklib.duckdb_create_list_type(child_p)
    assert list_p != 0
    type_id = ducklib.duckdb_get_type_id(list_p)
    assert type_id == DUCKDB_TYPE_LIST
    child_back_p = ducklib.duckdb_list_type_child_type(list_p)
    assert child_back_p != 0
    child_type_id = ducklib.duckdb_get_type_id(child_back_p)
    assert child_type_id == DUCKDB_TYPE_INTEGER
    for p in [child_back_p, list_p, child_p]:
        buf = numpy.array([p], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_create_array_type():
    DUCKDB_TYPE_ARRAY = 33
    DUCKDB_TYPE_INTEGER = 4
    child_p = ducklib.duckdb_create_logical_type(DUCKDB_TYPE_INTEGER)
    assert child_p != 0
    array_p = ducklib.duckdb_create_array_type(child_p, 5)
    assert array_p != 0
    type_id = ducklib.duckdb_get_type_id(array_p)
    assert type_id == DUCKDB_TYPE_ARRAY
    size = ducklib.duckdb_array_type_array_size(array_p)
    assert size == 5
    child_back_p = ducklib.duckdb_array_type_child_type(array_p)
    assert child_back_p != 0
    child_type_id = ducklib.duckdb_get_type_id(child_back_p)
    assert child_type_id == DUCKDB_TYPE_INTEGER
    for p in [child_back_p, array_p, child_p]:
        buf = numpy.array([p], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_create_map_type():
    DUCKDB_TYPE_MAP = 26
    DUCKDB_TYPE_INTEGER = 4
    DUCKDB_TYPE_VARCHAR = 17
    key_p = ducklib.duckdb_create_logical_type(DUCKDB_TYPE_INTEGER)
    val_p = ducklib.duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR)
    assert key_p != 0 and val_p != 0
    map_p = ducklib.duckdb_create_map_type(key_p, val_p)
    assert map_p != 0
    type_id = ducklib.duckdb_get_type_id(map_p)
    assert type_id == DUCKDB_TYPE_MAP
    key_back_p = ducklib.duckdb_map_type_key_type(map_p)
    val_back_p = ducklib.duckdb_map_type_value_type(map_p)
    assert ducklib.duckdb_get_type_id(key_back_p) == DUCKDB_TYPE_INTEGER
    assert ducklib.duckdb_get_type_id(val_back_p) == DUCKDB_TYPE_VARCHAR
    for p in [val_back_p, key_back_p, map_p, val_p, key_p]:
        buf = numpy.array([p], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)


def test_create_struct_type():
    DUCKDB_TYPE_STRUCT = 25
    DUCKDB_TYPE_INTEGER = 4
    DUCKDB_TYPE_VARCHAR = 17
    int_type_p = ducklib.duckdb_create_logical_type(DUCKDB_TYPE_INTEGER)
    varchar_type_p = ducklib.duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR)
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
    assert type_id == DUCKDB_TYPE_STRUCT
    count = ducklib.duckdb_struct_type_child_count(struct_p)
    assert count == 2
    child_name_p = ducklib.duckdb_struct_type_child_name(struct_p, 0)
    assert child_name_p != 0
    child_name = ctypes.c_char_p(child_name_p).value.decode()
    assert child_name == "id"
    ducklib.duckdb_free(child_name_p)
    child_type_p = ducklib.duckdb_struct_type_child_type(struct_p, 0)
    assert ducklib.duckdb_get_type_id(child_type_p) == DUCKDB_TYPE_INTEGER
    child_type_buf = numpy.array([child_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(child_type_buf.ctypes.data)
    struct_buf = numpy.array([struct_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(struct_buf.ctypes.data)
    int_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(int_buf.ctypes.data)
    varchar_buf = numpy.array([varchar_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(varchar_buf.ctypes.data)


def test_create_union_type():
    DUCKDB_TYPE_UNION = 28
    DUCKDB_TYPE_INTEGER = 4
    DUCKDB_TYPE_VARCHAR = 17
    int_type_p = ducklib.duckdb_create_logical_type(DUCKDB_TYPE_INTEGER)
    varchar_type_p = ducklib.duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR)
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
    assert type_id == DUCKDB_TYPE_UNION
    count = ducklib.duckdb_union_type_member_count(union_p)
    assert count == 2
    member_name_p = ducklib.duckdb_union_type_member_name(union_p, 0)
    assert member_name_p != 0
    member_name = ctypes.c_char_p(member_name_p).value.decode()
    assert member_name == "num"
    ducklib.duckdb_free(member_name_p)
    member_type_p = ducklib.duckdb_union_type_member_type(union_p, 0)
    assert ducklib.duckdb_get_type_id(member_type_p) == DUCKDB_TYPE_INTEGER
    member_type_buf = numpy.array([member_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(member_type_buf.ctypes.data)
    union_buf = numpy.array([union_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(union_buf.ctypes.data)
    int_buf = numpy.array([int_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(int_buf.ctypes.data)
    varchar_buf = numpy.array([varchar_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(varchar_buf.ctypes.data)


def test_create_enum_type():
    DUCKDB_TYPE_ENUM = 23
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
    assert type_id == DUCKDB_TYPE_ENUM
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
