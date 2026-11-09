"""Read the ``Connection*`` of a Python duckdb connection, so that the bindings can use that connection.

:func:`extract_connection_ptr` returns the address of the C++ ``Connection`` that a ``duckdb.DuckDBPyConnection``
holds. DuckDB's C API represents a ``duckdb_connection`` handle as that same address, so the bindings in
:mod:`numbduck.ducklib` can run queries and register functions on a connection that Python opened. The function reads
the address from the memory layout of the duckdb Python package's objects, which pybind11 defines before duckdb 2.0.0
and nanobind from duckdb 2.0.0. Importing this module imports :mod:`numbduck.ducklib`, which loads libduckdb.
"""
import ctypes

import duckdb

# Importing ducklib loads libduckdb, and libraries_coordinated reads that library's version.
from numbduck import ducklib  # noqa: F401
from numbduck.utils import libraries_coordinated, loaded_library_version


# The byte offset from a Python object's address, id(conn), to the pointer to its C++ object in pybind11's layout.
# pybind11 defines its instance struct in detail/common.h:
# https://github.com/pybind/pybind11/blob/v2.13.6/include/pybind11/detail/common.h#L602
# The struct begins with PyObject_HEAD. In its simple layout, which a DuckDBPyConnection has, the pointer to the C++
# object comes next, and then the holder that owns the object. In 64-bit CPython with the GIL, PyObject_HEAD is
# ob_refcnt and ob_type, 8 bytes each, so the DuckDBPyConnection* is at +16.
PYBIND11_HELD_OBJECT_OFFSET = 16


def extract_duckdb_conn_ptr(x: duckdb.DuckDBPyConnection):
    """Return the address of the C++ ``DuckDBPyConnection`` object behind the Python connection ``x``.

    ``x`` must be exactly a ``duckdb.DuckDBPyConnection``; any other object raises ``TypeError``. Before duckdb 2.0.0,
    the function reads the address from pybind11's instance. From duckdb 2.0.0, it reads nanobind's instance header,
    and raises ``RuntimeError`` if nanobind does not mark the C++ object as ready. The address is valid only while
    ``x`` is alive. :func:`extract_connection_ptr` reads the ``Connection*`` from this object.
    """
    if type(x) is not duckdb.DuckDBPyConnection:
        raise TypeError("Expected duckdb.DuckDBPyConnection")

    obj_addr = id(x)
    if int(duckdb.__version__.split(".")[0]) < 2:
        return ctypes.c_void_p.from_address(obj_addr + PYBIND11_HELD_OBJECT_OFFSET).value
    # From duckdb 2.0.0, the duckdb Python package binds its classes with nanobind instead of pybind11. nanobind's
    # instance struct, nb_inst, follows PyObject_HEAD with the int32 offset at byte 16 and the state at byte 20.
    offset = ctypes.c_int32.from_address(obj_addr + 16).value

    # The first byte of the state holds its bit fields. The compilers that build duckdb's wheels give the first bit
    # field the lowest bits.
    state_byte = ctypes.c_uint8.from_address(obj_addr + 20).value

    # Bits 0 and 1 are the state of the C++ object:
    # https://github.com/wjakob/nanobind/blob/v3.1.0/src/nb_internals.h#L365
    state = state_byte & 0b11
    # Bit 2, direct, is set if the C++ object itself is at the offset, and clear if a pointer to the object is there:
    # https://github.com/wjakob/nanobind/blob/v3.1.0/src/nb_internals.h#L373
    is_direct = bool(state_byte & 0b100)

    # State 2, state_ready, means that the C++ object is constructed and usable:
    # https://github.com/wjakob/nanobind/blob/v3.1.0/src/nb_internals.h#L361
    if state != 2:
        raise RuntimeError(f"Nanobind instance is not ready: state={state}")

    # nanobind counts the offset from the address of the Python object.
    instance_addr = obj_addr + offset

    # nanobind's inst_ptr resolves the C++ object's address the same way:
    # https://github.com/wjakob/nanobind/blob/v3.1.0/src/nb_internals.h#L1062
    if is_direct:
        cpp_address = instance_addr
    else:
        cpp_address = ctypes.c_void_p.from_address(instance_addr).value or 0
    return cpp_address


# The byte offset from a DuckDBPyConnection* to the Connection* that the object owns. A DuckDBPyConnection derives
# from enable_shared_from_this, whose weak_ptr takes its first 16 bytes. Its first member, the ConnectionGuard con,
# begins with shared_ptr<DuckDB> database, 16 bytes, and then unique_ptr<Connection> connection, at +32. A unique_ptr
# holds only the raw pointer. The layout is that of duckdb-python's pyconnection.hpp, at v1.5.4 and at v2.0.0:
# https://github.com/duckdb/duckdb-python/blob/v2.0.0/src/include/duckdb_python/pyconnection/pyconnection.hpp
DUCKDBPY_CONNECTION_OFFSET = 32


def extract_connection_ptr(conn):
    """Return the ``Connection*`` of the Python duckdb connection ``conn`` as an ``int``, for the bindings to use.

    The bindings take the result wherever DuckDB's C API takes a ``duckdb_connection``. The pointer is borrowed:
    ``conn`` owns the ``Connection``, and the pointer is valid only while ``conn`` is alive and open. An ``int`` keeps
    nothing alive, so the caller keeps ``conn`` referenced and open for as long as any jitted code uses the pointer. A
    use after ``conn.close()``, or after ``conn`` is garbage-collected, reads freed memory.

    ``conn`` must be exactly a ``duckdb.DuckDBPyConnection``, not a subclass, and any other object raises
    ``TypeError``. The function reads raw memory at fixed offsets, and for a subclass, or for an object that overrides
    ``__class__``, it would read arbitrary memory. ``RuntimeError`` is raised if the libduckdb that the bindings call is
    not the build of the duckdb wheel that created ``conn``, as :func:`~numbduck.utils.libraries_coordinated` checks.
    Two libduckdb libraries are loaded in the process only when the loader falls back to a standalone library, as for a
    macOS wheel without the C API. ``RuntimeError`` is also raised if a pointer that the function reads is null, as for
    a closed connection.

    The function runs no query on the connection, so a result that the connection is still streaming loses no rows.
    :func:`~numbduck.utils.complete_open_result` says what a C API query on the connection does to such a result.

    The offsets are details of the duckdb Python package, not a public interface. ``PYBIND11_HELD_OBJECT_OFFSET``
    applies before duckdb 2.0.0, and the layout of nanobind's instance header from duckdb 2.0.0 on.
    ``DUCKDBPY_CONNECTION_OFFSET`` applies to both, and the comments at these constants say where each comes from.
    Check them again for a new duckdb release: a wrong offset gives a wild pointer, and its use inside the C API is
    undefined behavior that nothing here can detect.
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

    # Read the address of the C++ DuckDBPyConnection from the pybind11 or nanobind instance.
    cpp_obj_p = extract_duckdb_conn_ptr(conn)
    if not cpp_obj_p:
        raise RuntimeError(
            "extracted null DuckDBPyConnection* from duckdb.DuckDBPyConnection"
        )

    # Read the Connection* that the unique_ptr<Connection> holds. Closing the connection resets the unique_ptr to null.
    conn_ptr = ctypes.c_void_p.from_address(
        cpp_obj_p + DUCKDBPY_CONNECTION_OFFSET).value
    if not conn_ptr:
        raise RuntimeError(
            "extracted null Connection* from duckdb.DuckDBPyConnection "
            "(is the connection closed?)"
        )

    return conn_ptr
