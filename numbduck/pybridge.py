import ctypes

import duckdb

from numbduck import ducklib
from numbduck.duckdb_utils import create_duckdb_result
from numbduck.utils import libraries_coordinated, loaded_library_version
from numbox.utils.lowlevel import get_unicode_data_p


# Byte offset from the Python object address (``id(conn)``) to the pybind11
# instance's held C++ object pointer. pybind11 lays its ``instance`` struct out
# as ``PyObject_HEAD`` followed by the holder; on 64-bit CPython ``PyObject_HEAD``
# is ``ob_refcnt`` (8 bytes) + ``ob_type`` (8 bytes) = 16 bytes, so the
# ``DuckDBPyConnection*`` sits immediately after it at +16.
PYBIND11_HELD_OBJECT_OFFSET = 16

# Byte offset from a ``DuckDBPyConnection*`` to its ``unique_ptr<Connection>``.
# The C++ object begins with an ``enable_shared_from_this`` weak_ptr (16 bytes)
# and a ``shared_ptr<DuckDB> database`` (16 bytes); the ``unique_ptr<Connection>``
# we want follows at +32.
DUCKDBPY_CONNECTION_OFFSET = 32


def extract_connection_ptr(conn):
    """Return the raw C API ``Connection*`` borrowed from a Python duckdb connection.

    Reads the pointer out of *conn*'s pybind11 instance layout using the
    ``PYBIND11_HELD_OBJECT_OFFSET`` / ``DUCKDBPY_CONNECTION_OFFSET`` constants
    (which document the offset derivations). Those offsets are an implementation
    detail of the duckdb Python package, validated on duckdb 1.3.2 / Linux x86-64
    / libstdc++; re-verify them when upgrading duckdb, since stale offsets yield a
    wild pointer whose dereference inside ``duckdb_query`` is undefined behavior
    the ``SELECT 1`` check below cannot catch.

    The pointer is handed out only after four guards: exact-type identity (not
    ``isinstance``: a subclass or an object spoofing ``__class__`` would steer
    the raw pointer walk through arbitrary memory); runtime coordination (refuse
    when numbduck's JIT libduckdb and the wheel that minted *conn* are different
    builds, i.e. the macOS dual-runtime seam); a null check on each intermediate
    pointer (a closed connection nulls its ``unique_ptr<Connection>``); and a
    best-effort ``SELECT 1`` liveness smoke-test.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection

    Returns
    -------
    int
        The ``Connection*`` as a Python int (``intp``-compatible). The pointer is
        **borrowed** from *conn*: it is owned by the ``DuckDBPyConnection`` (which
        holds the ``unique_ptr<Connection>``) and stays valid only while *conn* is
        alive and open. A Python ``int`` cannot keep *conn* referenced, so the
        keep-alive is the caller's responsibility.

    Warning
    -------
    The caller must retain *conn*, alive and open, for the entire lifetime of any
    ``@njit`` use of the returned pointer. Using the pointer after
    ``conn.close()`` or after *conn* is garbage-collected dereferences a dangling
    ``Connection*``: a use-after-free.

    Raises
    ------
    TypeError
        If *conn* is not exactly a ``duckdb.DuckDBPyConnection``.
    RuntimeError
        If numbduck's JIT bindings and the Python ``duckdb`` module resolve
        different libduckdb versions, if the extracted pointer is null (a closed
        or uninitialized connection), or if the pointer fails the ``SELECT 1``
        liveness smoke-test.
    """
    if type(conn) is not duckdb.DuckDBPyConnection:
        raise TypeError(
            f"expected duckdb.DuckDBPyConnection, got {type(conn).__name__}"
        )

    if not libraries_coordinated():
        jit_version = loaded_library_version()
        raise RuntimeError(
            f"numbduck's JIT bindings resolve libduckdb {jit_version!r}, but the "
            f"Python duckdb module is {duckdb.__version__!r}. Refusing to hand a "
            f"Connection* minted by the duckdb wheel to a different libduckdb "
            f"runtime. Dereferencing it under a mismatched internal layout is "
            f"undefined behavior. Load a matching libduckdb (set "
            f"NUMBDUCK_LIBDUCKDB to a libduckdb whose version equals "
            f"{duckdb.__version__!r})."
        )

    # Step 1: read the C++ object pointer from the pybind11 instance header.
    py_obj_addr = id(conn)
    cpp_obj_p = ctypes.c_void_p.from_address(
        py_obj_addr + PYBIND11_HELD_OBJECT_OFFSET).value
    if not cpp_obj_p:
        raise RuntimeError(
            "extracted null DuckDBPyConnection* from duckdb.DuckDBPyConnection"
        )

    # Step 2: read the Connection* from the unique_ptr<Connection> field.
    conn_ptr = ctypes.c_void_p.from_address(
        cpp_obj_p + DUCKDBPY_CONNECTION_OFFSET).value
    if not conn_ptr:
        raise RuntimeError(
            "extracted null Connection* from duckdb.DuckDBPyConnection "
            "(is the connection closed?)"
        )

    # Liveness smoke-test: run a trivial query through the C API.
    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT 1")
    rc = ducklib.duckdb_query(conn_ptr, query_p, result.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)
    if rc != ducklib.DuckDBSuccess:
        raise RuntimeError(
            "extracted connection pointer failed validation"
        )

    return conn_ptr
