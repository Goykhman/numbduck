import ctypes

import duckdb

from numbduck import ducklib
from numbduck.duckdb_utils import create_duckdb_result
from numbox.utils.lowlevel import get_unicode_data_p


def extract_connection_ptr(conn):
    """Extract the raw C API ``Connection*`` from a Python duckdb connection.

    Uses ctypes to read the pybind11 instance layout of
    ``DuckDBPyConnection``:

    1. ``id(conn) + 16`` — reads a ``c_void_p``; this is the
       ``DuckDBPyConnection*`` (the C++ object managed by pybind11).
    2. ``DuckDBPyConnection* + 32`` — reads a ``c_void_p``; this is the
       ``Connection*`` (``duckdb_connection`` in the C API).

    The +32 offset inside ``DuckDBPyConnection`` is derived from the struct
    layout::

        [0]  enable_shared_from_this weak_ptr  — 16 bytes
        [16] shared_ptr<DuckDB> database       — 16 bytes
        [32] unique_ptr<Connection> connection — pointer we want

    The extracted pointer is validated by running ``SELECT 1`` via the C API
    before it is returned.

    Validated on duckdb 1.3.2 / Linux x86-64 / libstdc++. The pybind11
    instance layout and the ``DuckDBPyConnection`` struct layout are
    implementation details of the duckdb Python package and may change with
    major duckdb releases. Re-verify offsets when upgrading duckdb.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection

    Returns
    -------
    int
        The ``Connection*`` as a Python int (``intp``-compatible).

    Raises
    ------
    TypeError
        If *conn* is not a ``duckdb.DuckDBPyConnection``.
    RuntimeError
        If the extracted pointer fails the validation query.
    """
    if not isinstance(conn, duckdb.DuckDBPyConnection):
        raise TypeError(
            f"expected duckdb.DuckDBPyConnection, got {type(conn).__name__}"
        )

    # Step 1: read the C++ object pointer from the pybind11 instance header.
    py_obj_addr = id(conn)
    cpp_obj_p = ctypes.c_void_p.from_address(py_obj_addr + 16).value

    # Step 2: read the Connection* from offset 32 inside DuckDBPyConnection.
    conn_ptr = ctypes.c_void_p.from_address(cpp_obj_p + 32).value

    # Validate by running a trivial query through the C API.
    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT 1")
    rc = ducklib.duckdb_query(conn_ptr, query_p, result.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)
    if rc != ducklib.DuckDBSuccess:
        raise RuntimeError(
            "extracted connection pointer failed validation"
        )

    return conn_ptr
