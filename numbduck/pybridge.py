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
    """Extract the raw C API ``Connection*`` from a Python duckdb connection.

    Uses ctypes to read the pybind11 instance layout of
    ``DuckDBPyConnection``:

    1. ``id(conn) + PYBIND11_HELD_OBJECT_OFFSET`` — reads a ``c_void_p``; this is
       the ``DuckDBPyConnection*`` (the C++ object managed by pybind11).
    2. ``DuckDBPyConnection* + DUCKDBPY_CONNECTION_OFFSET`` — reads a
       ``c_void_p``; this is the ``Connection*`` (``duckdb_connection`` in the C
       API).

    See the module-level ``PYBIND11_HELD_OBJECT_OFFSET`` /
    ``DUCKDBPY_CONNECTION_OFFSET`` constants for the derivation of each offset.
    The ``DUCKDBPY_CONNECTION_OFFSET`` (32) inside ``DuckDBPyConnection`` is
    derived from the struct layout::

        [0]  enable_shared_from_this weak_ptr  — 16 bytes
        [16] shared_ptr<DuckDB> database       — 16 bytes
        [32] unique_ptr<Connection> connection — pointer we want

    Before the pointer is walked, :func:`~numbduck.utils.libraries_coordinated`
    checks that the libduckdb backing numbduck's JIT bindings is the same version
    as the Python ``duckdb`` module that minted *conn*. When they differ — the
    macOS dual-runtime seam where numbduck loads a standalone libduckdb while the
    wheel keeps its own — this raises rather than hand a wheel-minted
    ``Connection*`` to a possibly-different libduckdb build (a cross-runtime
    dereference under a mismatched internal layout is undefined behavior that the
    ``SELECT 1`` validation cannot catch).

    The extracted pointer is then validated by running ``SELECT 1`` via the C API
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
        If numbduck's JIT bindings and the Python ``duckdb`` module resolve
        different libduckdb versions, or if the extracted pointer fails the
        validation query.
    """
    if not isinstance(conn, duckdb.DuckDBPyConnection):
        raise TypeError(
            f"expected duckdb.DuckDBPyConnection, got {type(conn).__name__}"
        )

    if not libraries_coordinated():
        jit_version = loaded_library_version()
        raise RuntimeError(
            f"numbduck's JIT bindings resolve libduckdb {jit_version!r}, but the "
            f"Python duckdb module is {duckdb.__version__!r}. Refusing to hand a "
            f"Connection* minted by the duckdb wheel to a different libduckdb "
            f"runtime — dereferencing it under a mismatched internal layout is "
            f"undefined behavior. Load a matching libduckdb (set "
            f"NUMBDUCK_LIBDUCKDB to a libduckdb whose version equals "
            f"{duckdb.__version__!r})."
        )

    # Step 1: read the C++ object pointer from the pybind11 instance header.
    py_obj_addr = id(conn)
    cpp_obj_p = ctypes.c_void_p.from_address(
        py_obj_addr + PYBIND11_HELD_OBJECT_OFFSET).value

    # Step 2: read the Connection* from the unique_ptr<Connection> field.
    conn_ptr = ctypes.c_void_p.from_address(
        cpp_obj_p + DUCKDBPY_CONNECTION_OFFSET).value

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
