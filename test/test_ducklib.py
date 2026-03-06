import ctypes

from numbox.utils.lowlevel import get_unicode_data_p

from numbduck import ducklib
from numbduck.duckdb_utils import (
    create_duckdb_connection, create_duckdb_database, create_duckdb_result
)


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


def aux_connect_db():
    duckdb_database = aux_open_database(0)
    duckdb_database_p = duckdb_database[0]

    duckdb_connection = create_duckdb_connection()
    duckdb_connection_pp = duckdb_connection.ctypes.data
    duckdb_connect_rc = ducklib.duckdb_connect(duckdb_database_p, duckdb_connection_pp)
    assert duckdb_connect_rc == ducklib.DuckDBSuccess, duckdb_connect_rc
    return duckdb_database, duckdb_connection


def test_connect():
    duckdb_database, duckdb_connection = aux_connect_db()
    duckdb_database_p = duckdb_database[0]
    duckdb_connection_p = duckdb_connection[0]
    assert duckdb_database_p != 0, f"Expected pointer to DB, got {duckdb_database_p}"
    assert duckdb_connection_p != 0, f"Expected pointer to connection, got {duckdb_connection_p}"


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

    return out_result


def test_query():
    out_result = aux_query_1()
    print(f"out_result = {out_result}")
    num_of_columns = out_result[0]
    assert num_of_columns == 2, f"expected 'i', 'j', got {num_of_columns} columns"


def test_duckdb_column_count_and_duckdb_row_count():
    out_result = aux_query_1()
    out_result_p = out_result.ctypes.data
    num_of_cols = ducklib.duckdb_column_count(out_result_p)
    num_of_rows = ducklib.duckdb_row_count(out_result_p)
    assert num_of_cols == 2, num_of_cols
    assert num_of_rows == 3, num_of_rows


def test_duckdb_destroy_result():
    out_result = aux_query_1()
    out_result_p = out_result.ctypes.data
    num_of_cols = ducklib.duckdb_column_count(out_result_p)
    assert num_of_cols == 2, f"Expected valid result before destroy, got {num_of_cols} columns"
    ducklib.duckdb_destroy_result(out_result_p)


def aux_get_data_vector():
    out_result = aux_query_1()
    duckdb_result = tuple(out_result)
    data_chunk_p = ducklib.duckdb_fetch_chunk(duckdb_result)
    assert data_chunk_p, f"Expected pointer to data chunk, got {data_chunk_p}"

    i_vec_p = ducklib.duckdb_data_chunk_get_vector(data_chunk_p, 0)
    i_vec_data_p = ducklib.duckdb_vector_get_data(i_vec_p)
    i_arr = arr_ty.from_address(i_vec_data_p)
    assert all([i_arr_ == i_col_ for i_arr_, i_col_ in zip(i_arr, i_col)])
    return duckdb_result, data_chunk_p


def test_duckdb_data_chunk_get_column_count():
    duckdb_result, data_chunk_p = aux_get_data_vector()
    col_count = ducklib.duckdb_data_chunk_get_column_count(data_chunk_p)
    assert col_count == 2, f"Expected 2 columns, got {col_count}"


def test_duckdb_data_chunk_get_size():
    duckdb_result, data_chunk_p = aux_get_data_vector()
    chunk_size = ducklib.duckdb_data_chunk_get_size(data_chunk_p)
    assert chunk_size == 3, f"Expected 3 rows in chunk, got {chunk_size}"


def test_duckdb_fetch_chunk_data_chunk_get_vector_get_data_vector():
    duckdb_result, data_chunk_p = aux_get_data_vector()
    assert data_chunk_p
    j_vec_p = ducklib.duckdb_data_chunk_get_vector(data_chunk_p, 1)
    j_vec_data_p = ducklib.duckdb_vector_get_data(j_vec_p)
    j_validity_p = ducklib.duckdb_vector_get_validity(j_vec_p)
    j_arr = arr_ty.from_address(j_vec_data_p)
    j_val = [ducklib.duckdb_validity_row_is_valid(j_validity_p, ind_) for ind_ in range(3)]
    assert j_val == [1, 1, 0]
    assert all([j_val and j_arr_ == j_col_ or True for j_arr_, j_col_, j_val_ in zip(j_arr, j_col, j_val)])
