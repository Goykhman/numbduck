import numpy


def allocate_buffer(sz: int):
    return numpy.zeros(sz, dtype=numpy.int64)


def create_duckdb_connection():
    """ https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L469 """
    return allocate_buffer(1)


def create_duckdb_database():
    """ https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L464 """
    return allocate_buffer(1)


def create_duckdb_data_chunk():
    """ https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L527 """
    return allocate_buffer(1)


def create_duckdb_result():
    """ https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L454 """
    return allocate_buffer(6)


def create_duckdb_vector():
    """ https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L395 """
    return allocate_buffer(1)
