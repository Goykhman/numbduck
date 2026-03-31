# Hybrid JIT UDF Demo — Design Spec

## Goal

Demonstrate numbduck's value in a hybrid Python+JIT workflow: open a connection in Python duckdb, register a JIT-compiled UDF via numbduck's C API, and query from Python. This addresses Goykhman's review comment on Goykhman/numbduck#19 asking for tests that show where numbduck proxies add value over pure Python duckdb capabilities.

## Components

### 1. `numbduck/pybridge.py` — connection pointer extraction

A utility function that extracts the raw `Connection*` from a Python `duckdb.DuckDBPyConnection` object.

**Function:** `extract_connection_ptr(conn) -> int`

**How it works:**
- `DuckDBPyConnection` is a pybind11 class with `shared_ptr` holder
- pybind11's `instance` struct stores `simple_value_holder[0]` (the C++ object pointer) at offset +16 from `id(obj)` (after `PyObject_HEAD`)
- The `DuckDBPyConnection` C++ struct inherits `enable_shared_from_this` (16 bytes for `weak_ptr`), then holds `ConnectionGuard con` whose first member is `shared_ptr<DuckDB> database` (16 bytes), followed by `unique_ptr<Connection> connection` (8 bytes)
- Full chain: `id(conn) + 16` -> `DuckDBPyConnection*`, then `+ 32` -> `Connection*`
- The `Connection*` is the same type as `duckdb_connection` in the C API (via `reinterpret_cast`)

**Runtime validation:**
- After extraction, runs `duckdb_query(conn_ptr, "SELECT 1", ...)` and checks for `DuckDBSuccess`
- Raises `RuntimeError` if validation fails (wrong offsets due to version change)
- Destroys the validation result before returning

**Version coupling:**
- Offsets depend on: pybind11 instance layout, `sizeof(weak_ptr)`, `sizeof(shared_ptr)`, member ordering of `DuckDBPyConnection` and `ConnectionGuard`
- Validated against duckdb 1.3.2 on Linux x86-64 (libstdc++)
- These are stable across duckdb 1.x but could change in a major release
- The runtime validation catches breakage rather than silently returning a bad pointer

### 2. Tests

#### `test_hybrid_jit_udf_on_python_connection`

The primary demo. Shows the full hybrid workflow:
1. Open connection via `duckdb.connect()`
2. Create a table with test data via Python
3. Extract `Connection*` via `extract_connection_ptr()`
4. Define a `@njit` implementation + `@cfunc` wrapper for a scalar UDF
5. Register the UDF via numbduck's C API (`duckdb_create_scalar_function`, etc.)
6. Query from Python using the JIT UDF: `conn.execute("SELECT jit_func(x) FROM t")`
7. Assert correct results

The UDF should do something non-trivial to illustrate the JIT advantage — e.g., element-wise Newton's method sqrt approximation (iterative, branch-heavy, benefits from native code).

#### `test_jit_udf_vs_python_udf`

Side-by-side comparison on the same connection:
1. Open connection, create table
2. Register a Python UDF via `conn.create_function()` — pure Python callback
3. Register an equivalent JIT UDF via numbduck C API — `@cfunc` callback
4. Run both on the same data, assert identical results
5. Docstring explains the performance difference: Python UDF round-trips through the GIL per batch; JIT UDF is called as a raw function pointer with zero Python overhead

No timing assertions — those are flaky in CI.

## Not in scope

- Making `extract_connection_ptr` work inside `@njit` (it's a Python-side bridge)
- Aggregate UDF demos (scalar is sufficient for the hybrid story)
- Benchmarking infrastructure
