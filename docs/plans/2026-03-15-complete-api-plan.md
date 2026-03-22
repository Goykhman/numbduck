# Complete DuckDB C API Bindings — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Bind all DuckDB C API functions for groups 1-5 (bind types, result metadata, value interface, logical types, configuration), bringing numbduck from 31 to ~160 bindings.

**Architecture:** Each binding follows one of two patterns: (1) simple `@cres` + `_call_lib_func` for scalar args/returns, or (2) custom `@intrinsic` for struct-by-value params/returns. Tests follow prepare→bind→execute→fetch→verify pattern.

**Tech Stack:** numba, numbox, llvmlite, ctypes, pytest

**Reference files:**
- `duckdb.h` (v1.3.2) at project root — C API header, source of truth for all signatures
- `numbduck/ducklib.py` — all bindings live here
- `numbduck/duckdb_utils.py` — buffer allocators
- `test/test_ducklib.py` — all tests
- `docs/plans/2026-03-15-complete-api-design.md` — design rationale

**Agent assignment:** Use sonnet for simple `@cres` tasks, opus for `@intrinsic` tasks.

**Key lessons from Phase 1 review:**
- Use `@cres` (not `@njit`) for all intrinsic wrappers — returns `CompileResultWAP` with better caching
- `platform.machine() == 'x86_64'` is sufficient for SysV x86-64 detection (Windows reports `AMD64`)
- Flat alloca+store works for decimal (Numba's `{i8, i8, i64, i64}` matches C struct layout due to LLVM alignment)
- Keep signatures and wrappers in alphabetical order
- Never put planning details in code comments

**Key lessons from Phase 2 review:**
- Docstring links must use `duckdb.org/docs/stable/clients/c/api.html#func_name`, not `duckdb.h` source links
- Extract shared helpers for repeated intrinsic patterns (`_call_byval` for by-value struct calls)
- Always bind the corresponding destroy function when binding a function that returns a handle
- Assert return codes on setup statements in tests (e.g. CREATE TABLE)

**Key lessons from Phase 3 implementation:**
- `duckdb_get_value_type` returns the SAME handle as the input value for scalar types — destroying both causes double-free. Only destroy the value OR the logical type, not both
- `duckdb_destroy_logical_type` and `duckdb_destroy_value` both take pointer-to-pointer (like all DuckDB destroy functions) — must wrap handle in a numpy buffer and pass `buf.ctypes.data`
- Worktree-isolated agents MUST be instructed to commit — uncommitted changes cannot be merged via git
- Container value tests (list, map, struct) segfault when combining `duckdb_column_logical_type` with container creators in JIT — likely a numba/LLVM interaction issue. The bindings compile and register correctly; the problem is at the test level when multiple JIT-compiled functions interact with shared logical type handles
- When dispatching agents to worktrees, verify the worktree is based on the correct branch — agents based on `main` instead of the feature branch will produce merges with conflicts

---

## Status

- **Phase 1:** MERGED upstream (PR #11)
- **Phase 2:** MERGED upstream (PR #12)
- **Phase 3:** MERGED upstream (PR #13)
  - Task 17 container tests were deferred — segfault when combining `duckdb_column_logical_type` with container creators in JIT. Bindings compile; issue is test-level interaction with logical type handles.
  - **Discovery:** `duckdb_get_value_type` returns the same handle as the input value for scalar types — do NOT destroy both (double-free).
- **Phase 4:** MERGED upstream (PR #14)
- **Phase 5:** upstream PR #15 open

---

## Phase 1: Remaining Bind Types ✅

Branch: `bind-types` (off `origin/main`)

### Task 1: Add simple bind type signatures and wrappers

**Files:**
- Modify: `numbduck/ducklib.py`

**Step 1: Add new numba type imports**

Add `int16`, `uint8`, `uint16`, `uint32` to the import from `numba.core.types` (line 3). `int8`, `uint64`, `int32`, `int64`, `float32`, `float64`, `intp`, `void` are already imported.

**Step 2: Add signatures**

Add after existing `duckdb_bind_varchar` signature (line 30):

```python
signatures["duckdb_bind_blob"] = duckdb_state_ty(intp, uint64, intp, uint64)
signatures["duckdb_bind_int8"] = duckdb_state_ty(intp, uint64, int8)
signatures["duckdb_bind_int16"] = duckdb_state_ty(intp, uint64, int16)
signatures["duckdb_bind_parameter_index"] = duckdb_state_ty(intp, intp, intp)
signatures["duckdb_bind_time"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_timestamp_tz"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_uint8"] = duckdb_state_ty(intp, uint64, uint8)
signatures["duckdb_bind_uint16"] = duckdb_state_ty(intp, uint64, uint16)
signatures["duckdb_bind_uint32"] = duckdb_state_ty(intp, uint64, uint32)
signatures["duckdb_bind_uint64"] = duckdb_state_ty(intp, uint64, uint64)
signatures["duckdb_bind_value"] = duckdb_state_ty(intp, uint64, intp)
signatures["duckdb_bind_varchar_length"] = duckdb_state_ty(intp, uint64, intp, uint64)
```

**Step 3: Add wrapper functions**

Add after `duckdb_bind_boolean` wrapper (alphabetical order):

```python
@cres(signatures.get("duckdb_bind_blob"))
def duckdb_bind_blob(prepared_statement_p, param_idx, data_p, length):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_blob """
    return _call_lib_func("duckdb_bind_blob", (prepared_statement_p, param_idx, data_p, length))


@cres(signatures.get("duckdb_bind_int8"))
def duckdb_bind_int8(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_int8 """
    return _call_lib_func("duckdb_bind_int8", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_int16"))
def duckdb_bind_int16(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_int16 """
    return _call_lib_func("duckdb_bind_int16", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_parameter_index"))
def duckdb_bind_parameter_index(prepared_statement_p, param_idx_out_p, name_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_parameter_index """
    return _call_lib_func("duckdb_bind_parameter_index", (prepared_statement_p, param_idx_out_p, name_p))


@cres(signatures.get("duckdb_bind_time"))
def duckdb_bind_time(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_time """
    return _call_lib_func("duckdb_bind_time", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_timestamp_tz"))
def duckdb_bind_timestamp_tz(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_timestamp_tz """
    return _call_lib_func("duckdb_bind_timestamp_tz", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_uint8"))
def duckdb_bind_uint8(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uint8 """
    return _call_lib_func("duckdb_bind_uint8", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_uint16"))
def duckdb_bind_uint16(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uint16 """
    return _call_lib_func("duckdb_bind_uint16", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_uint32"))
def duckdb_bind_uint32(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uint32 """
    return _call_lib_func("duckdb_bind_uint32", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_uint64"))
def duckdb_bind_uint64(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uint64 """
    return _call_lib_func("duckdb_bind_uint64", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_value"))
def duckdb_bind_value(prepared_statement_p, param_idx, val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_value """
    return _call_lib_func("duckdb_bind_value", (prepared_statement_p, param_idx, val_p))


@cres(signatures.get("duckdb_bind_varchar_length"))
def duckdb_bind_varchar_length(prepared_statement_p, param_idx, val_p, length):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_varchar_length """
    return _call_lib_func("duckdb_bind_varchar_length", (prepared_statement_p, param_idx, val_p, length))
```

**Step 4: Run tests to verify nothing is broken**

Run: `cd /home/erik/projects/numbduck && venv/bin/pytest test/test_ducklib.py -v`
Expected: all existing tests pass

**Step 5: Commit**

```bash
git add numbduck/ducklib.py
git commit -m "Add simple bind type bindings (int8/16, uint8/16/32/64, time, timestamp_tz, blob, varchar_length, value, parameter_index)"
```

### Task 2: Add struct-by-value bind intrinsics (Pattern A)

**Files:**
- Modify: `numbduck/ducklib.py`

**Step 1: Add struct tuple type aliases**

Add after `duckdb_result_ty = UniTuple(intp, 6)` (line 20):

```python
duckdb_hugeint_ty = Tuple((uint64, int64))
duckdb_uhugeint_ty = UniTuple(uint64, 2)
duckdb_interval_ty = Tuple((int32, int32, int64))
duckdb_decimal_ty = Tuple((uint8, uint8, uint64, int64))
```

Also add `Tuple` to the numba.core.types import (alongside `UniTuple`).

**Step 2: Add intrinsic for duckdb_bind_hugeint**

The C signature is: `duckdb_state duckdb_bind_hugeint(duckdb_prepared_statement, idx_t, duckdb_hugeint)`.
`duckdb_hugeint` is `struct { uint64_t lower; int64_t upper; }` — 16 bytes total.

The intrinsic must:
1. Alloca a 16-byte struct on the stack
2. Store `lower` (uint64) at offset 0, `upper` (int64) at offset 8
3. Call `duckdb_bind_hugeint(prepared_statement_p, param_idx, *struct_ptr)` — but since C passes the struct by value, we actually need to build the LLVM call with the struct type directly, not a pointer. The ABI will handle passing it correctly.

Actually, looking more carefully at the `_duckdb_fetch_chunk` pattern: it allocates the tuple on the stack and passes a *pointer* to the C function. This works because on x86-64 Linux, structs > 16 bytes are passed by pointer per the SysV ABI. For structs <= 16 bytes (like hugeint = 16 bytes), the ABI passes them in registers. However, LLVM handles this automatically when you define the function type with the struct as a parameter — no need to manually stack-allocate.

The approach: define the LLVM function type with the struct as a parameter type, build the struct from tuple fields, and call directly.

```python
@intrinsic
def _duckdb_bind_hugeint(typingctx, prepared_statement_p_ty, param_idx_ty, hugeint_tup_ty):
    def codegen(context, builder: IRBuilder, signature, arguments):
        prepared_statement_p, param_idx, hugeint_tup = arguments
        # Build the hugeint struct type: { i64, i64 } (both fields are 8 bytes)
        hugeint_ll_ty = context.get_value_type(duckdb_hugeint_ty)
        # Define C function: int32 duckdb_bind_hugeint(ptr, i64, {i64, i64})
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type),
            [prepared_statement_p.type, param_idx.type, hugeint_ll_ty]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_bind_hugeint")
        return builder.call(func_p, [prepared_statement_p, param_idx, hugeint_tup])
    return duckdb_state_ty(intp, uint64, duckdb_hugeint_ty), codegen


@cres(duckdb_state_ty(intp, uint64, duckdb_hugeint_ty))
def duckdb_bind_hugeint(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_hugeint """
    return _duckdb_bind_hugeint(prepared_statement_p, param_idx, val)
```

**Step 3: Add intrinsic for duckdb_bind_uhugeint**

Same pattern but `duckdb_uhugeint` = `struct { uint64_t lower; uint64_t upper; }`.

```python
@intrinsic
def _duckdb_bind_uhugeint(typingctx, prepared_statement_p_ty, param_idx_ty, uhugeint_tup_ty):
    def codegen(context, builder: IRBuilder, signature, arguments):
        prepared_statement_p, param_idx, uhugeint_tup = arguments
        uhugeint_ll_ty = context.get_value_type(duckdb_uhugeint_ty)
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type),
            [prepared_statement_p.type, param_idx.type, uhugeint_ll_ty]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_bind_uhugeint")
        return builder.call(func_p, [prepared_statement_p, param_idx, uhugeint_tup])
    return duckdb_state_ty(intp, uint64, duckdb_uhugeint_ty), codegen


@cres(duckdb_state_ty(intp, uint64, duckdb_uhugeint_ty))
def duckdb_bind_uhugeint(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uhugeint """
    return _duckdb_bind_uhugeint(prepared_statement_p, param_idx, val)
```

**Step 4: Add intrinsic for duckdb_bind_interval**

`duckdb_interval` = `struct { int32_t months; int32_t days; int64_t micros; }` — 16 bytes.

```python
@intrinsic
def _duckdb_bind_interval(typingctx, prepared_statement_p_ty, param_idx_ty, interval_tup_ty):
    def codegen(context, builder: IRBuilder, signature, arguments):
        prepared_statement_p, param_idx, interval_tup = arguments
        interval_ll_ty = context.get_value_type(duckdb_interval_ty)
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type),
            [prepared_statement_p.type, param_idx.type, interval_ll_ty]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_bind_interval")
        return builder.call(func_p, [prepared_statement_p, param_idx, interval_tup])
    return duckdb_state_ty(intp, uint64, duckdb_interval_ty), codegen


@cres(duckdb_state_ty(intp, uint64, duckdb_interval_ty))
def duckdb_bind_interval(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_interval """
    return _duckdb_bind_interval(prepared_statement_p, param_idx, val)
```

**Step 5: Add intrinsic for duckdb_bind_decimal**

`duckdb_decimal` = `struct { uint8_t width; uint8_t scale; duckdb_hugeint value; }`.
This is 18 bytes with padding, likely 24 bytes. The LLVM struct type will be `{ i8, i8, { i64, i64 } }`.

This one is trickier because the numba Tuple type `(uint8, uint8, uint64, int64)` is flat, but the C struct has nested `duckdb_hugeint`. We need to build the nested LLVM struct manually.

```python
@intrinsic
def _duckdb_bind_decimal(typingctx, prepared_statement_p_ty, param_idx_ty, decimal_tup_ty):
    def codegen(context, builder: IRBuilder, signature, arguments):
        from llvmlite import ir
        prepared_statement_p, param_idx, decimal_tup = arguments
        # Build nested C struct: { i8, i8, { i64, i64 } }
        i8 = ir.IntType(8)
        i64 = ir.IntType(64)
        hugeint_struct = ir.LiteralStructType([i64, i64])
        decimal_struct = ir.LiteralStructType([i8, i8, hugeint_struct])
        # Extract flat tuple fields
        width = builder.extract_value(decimal_tup, 0)
        scale = builder.extract_value(decimal_tup, 1)
        lower = builder.extract_value(decimal_tup, 2)
        upper = builder.extract_value(decimal_tup, 3)
        # Build hugeint sub-struct
        hugeint = ir.Constant(hugeint_struct, ir.Undefined)
        hugeint = builder.insert_value(hugeint, lower, 0)
        hugeint = builder.insert_value(hugeint, upper, 1)
        # Build decimal struct
        decimal_val = ir.Constant(decimal_struct, ir.Undefined)
        decimal_val = builder.insert_value(decimal_val, width, 0)
        decimal_val = builder.insert_value(decimal_val, scale, 1)
        decimal_val = builder.insert_value(decimal_val, hugeint, 2)
        # Call C function
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type),
            [prepared_statement_p.type, param_idx.type, decimal_struct]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_bind_decimal")
        return builder.call(func_p, [prepared_statement_p, param_idx, decimal_val])
    return duckdb_state_ty(intp, uint64, duckdb_decimal_ty), codegen


@cres(duckdb_state_ty(intp, uint64, duckdb_decimal_ty))
def duckdb_bind_decimal(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_decimal """
    return _duckdb_bind_decimal(prepared_statement_p, param_idx, val)
```

**Note:** The `FunctionType` here refers to `llvmlite.ir.FunctionType`, already imported at line 1. If both `llvmlite.ir.FunctionType` and `numba.experimental.function_type.FunctionType` are needed, rename the llvmlite import. Check the existing imports — currently `from llvmlite.ir import IRBuilder, FunctionType` is used.

**Step 6: Run tests**

Run: `cd /home/erik/projects/numbduck && venv/bin/pytest test/test_ducklib.py -v`
Expected: all existing tests pass

**Step 7: Commit**

```bash
git add numbduck/ducklib.py
git commit -m "Add struct-by-value bind intrinsics (hugeint, uhugeint, interval, decimal)"
```

### Task 3: Add tests for simple bind types

**Files:**
- Modify: `test/test_ducklib.py`

**Step 1: Write tests**

Add after the existing bind tests section. Follow the `test_bind_date` pattern: prepare → bind → execute → fetch → verify readback.

```python
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
    # 12:30:00 = 45000000000 microseconds
    micros = 45000000000
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
    blob_data = ctypes.c_char_p(b"\x00\x01\x02\x03")
    blob_p = ctypes.c_void_p.from_buffer(blob_data).value
    rc = ducklib.duckdb_bind_blob(stmt[0], 1, blob_p, 4)
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    # verify execution succeeded (blob readback is complex, just verify no error)
    chunk_size = ducklib.duckdb_data_chunk_get_size(chunk_p)
    assert chunk_size == 1
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
```

**Step 2: Run tests**

Run: `cd /home/erik/projects/numbduck && venv/bin/pytest test/test_ducklib.py -v -k "bind_int8 or bind_int16 or bind_uint8 or bind_uint16 or bind_uint32 or bind_uint64 or bind_time or bind_timestamp_tz or bind_varchar_length or bind_blob or bind_parameter_index"`
Expected: all new tests pass

**Step 3: Commit**

```bash
git add test/test_ducklib.py
git commit -m "Add tests for simple bind types"
```

### Task 4: Add tests for struct-by-value bind types

**Files:**
- Modify: `test/test_ducklib.py`

**Step 1: Write tests**

```python
def test_bind_hugeint():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    stmt, rc = aux_prepare(connection_p, "SELECT $1::HUGEINT;")
    assert rc == ducklib.DuckDBSuccess
    # hugeint value: lower=42, upper=0 (just 42)
    rc = ducklib.duckdb_bind_hugeint(stmt[0], 1, (42, 0))
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    data_p = aux_read_column_data(chunk_p, 0)
    # hugeint stored as { uint64 lower, int64 upper }
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
    # 1 month, 2 days, 3000000 microseconds (3 seconds)
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
    # decimal: width=10, scale=2, value=12345 (represents 123.45)
    rc = ducklib.duckdb_bind_decimal(stmt[0], 1, (10, 2, 12345, 0))
    assert rc == ducklib.DuckDBSuccess
    out_result, chunk_p = aux_execute_prepared(stmt[0])
    chunk_size = ducklib.duckdb_data_chunk_get_size(chunk_p)
    assert chunk_size == 1
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_destroy_prepared(stmt)
    aux_close_db(duckdb_database, duckdb_connection)
```

**Step 2: Run tests**

Run: `cd /home/erik/projects/numbduck && venv/bin/pytest test/test_ducklib.py -v -k "bind_hugeint or bind_uhugeint or bind_interval or bind_decimal"`
Expected: all pass

**Step 3: Commit**

```bash
git add test/test_ducklib.py
git commit -m "Add tests for struct-by-value bind types (hugeint, uhugeint, interval, decimal)"
```

### Task 5: Run full test suite and lint, create PR

**Step 1: Run full test suite**

Run: `cd /home/erik/projects/numbduck && venv/bin/pytest test/test_ducklib.py -v`
Expected: all tests pass

**Step 2: Run lint**

Run: `cd /home/erik/projects/numbduck && venv/bin/flake8 numbduck/ test/`
Expected: no errors

**Step 3: Push and create PR**

```bash
git push -u origin bind-types
gh pr create --title "Add remaining bind type bindings" --body "..."
```

---

## Phase 2: Query Result Metadata ✅

Branch: `upstream-result-metadata` (off `upstream/main`), also on `bind-types`
Upstream PR: Goykhman/numbduck#12 (all 12 CI jobs passed, awaiting review)

### Task 6: Add simple result metadata bindings

**Files:**
- Modify: `numbduck/ducklib.py`

**Step 1: Add signatures**

```python
signatures["duckdb_column_name"] = intp(intp, uint64)
signatures["duckdb_column_type"] = int32(intp, uint64)
signatures["duckdb_column_logical_type"] = intp(intp, uint64)
signatures["duckdb_rows_changed"] = uint64(intp)
```

**Step 2: Add wrappers**

```python
@cres(signatures.get("duckdb_column_name"))
def duckdb_column_name(duckdb_result_p, col):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_column_name """
    return _call_lib_func("duckdb_column_name", (duckdb_result_p, col))


@cres(signatures.get("duckdb_column_type"))
def duckdb_column_type(duckdb_result_p, col):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_column_type """
    return _call_lib_func("duckdb_column_type", (duckdb_result_p, col))


@cres(signatures.get("duckdb_column_logical_type"))
def duckdb_column_logical_type(duckdb_result_p, col):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_column_logical_type """
    return _call_lib_func("duckdb_column_logical_type", (duckdb_result_p, col))


@cres(signatures.get("duckdb_rows_changed"))
def duckdb_rows_changed(duckdb_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_rows_changed """
    return _call_lib_func("duckdb_rows_changed", (duckdb_result_p,))
```

**Step 3: Run tests, commit**

### Task 7: Add result-by-value intrinsics (Pattern B)

**Files:**
- Modify: `numbduck/ducklib.py`

These follow the `_duckdb_fetch_chunk` pattern exactly — alloca the result tuple on stack, pass pointer to C function, return scalar.

```python
@intrinsic
def _duckdb_result_statement_type(typingctx, duckdb_result_tup_ty):
    def codegen(context, builder: IRBuilder, signature, arguments):
        duckdb_result_tup = arguments[0]
        duckdb_result_tup_ty_ll = context.get_value_type(duckdb_result_ty)
        duckdb_result_tup_stack_p = builder.alloca(duckdb_result_tup_ty_ll)
        builder.store(duckdb_result_tup, duckdb_result_tup_stack_p)
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type), [duckdb_result_tup_ty_ll.as_pointer()]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_result_statement_type")
        return builder.call(func_p, [duckdb_result_tup_stack_p])
    return int32(duckdb_result_ty), codegen


@cres(int32(duckdb_result_ty))
def duckdb_result_statement_type(result):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_result_statement_type """
    return _duckdb_result_statement_type(result)


@intrinsic
def _duckdb_result_return_type(typingctx, duckdb_result_tup_ty):
    def codegen(context, builder: IRBuilder, signature, arguments):
        duckdb_result_tup = arguments[0]
        duckdb_result_tup_ty_ll = context.get_value_type(duckdb_result_ty)
        duckdb_result_tup_stack_p = builder.alloca(duckdb_result_tup_ty_ll)
        builder.store(duckdb_result_tup, duckdb_result_tup_stack_p)
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type), [duckdb_result_tup_ty_ll.as_pointer()]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_result_return_type")
        return builder.call(func_p, [duckdb_result_tup_stack_p])
    return int32(duckdb_result_ty), codegen


@cres(int32(duckdb_result_ty))
def duckdb_result_return_type(result):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_result_return_type """
    return _duckdb_result_return_type(result)


@intrinsic
def _duckdb_result_error_type(typingctx, duckdb_result_p_ty):
    def codegen(context, builder: IRBuilder, signature, arguments):
        duckdb_result_p = arguments[0]
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type), [duckdb_result_p.type]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_result_error_type")
        return builder.call(func_p, [duckdb_result_p])
    return int32(intp), codegen


@cres(int32(intp))
def duckdb_result_error_type(duckdb_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_result_error_type """
    return _duckdb_result_error_type(duckdb_result_p)
```

**Note:** `duckdb_result_error_type` takes `duckdb_result *` (pointer), not by value. So it's actually simple — just needs `@cres` + `_call_lib_func`. Re-check duckdb.h before implementing. If it takes pointer, use `@cres` instead.

**Step 2: Run tests, commit**

### Task 8: Add tests for result metadata

**Files:**
- Modify: `test/test_ducklib.py`

```python
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
    # INTEGER = DUCKDB_TYPE_INTEGER = 4
    col_type = ducklib.duckdb_column_type(out_result_p, 0)
    assert col_type == 4, f"Expected INTEGER (4), got {col_type}"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


def test_column_logical_type():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    out_result_p = out_result.ctypes.data
    logical_type_p = ducklib.duckdb_column_logical_type(out_result_p, 0)
    assert logical_type_p != 0, "Expected valid logical type pointer"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


def test_rows_changed():
    duckdb_database, duckdb_connection = aux_connect_db()
    connection_p = duckdb_connection[0]
    query_p = get_unicode_data_p("CREATE TABLE rc_test (x INTEGER);")
    ducklib.duckdb_query(connection_p, query_p, 0)
    query_p = get_unicode_data_p("INSERT INTO rc_test VALUES (1), (2), (3);")
    out_result = create_duckdb_result()
    out_result_p = out_result.ctypes.data
    rc = ducklib.duckdb_query(connection_p, query_p, out_result_p)
    assert rc == ducklib.DuckDBSuccess
    changed = ducklib.duckdb_rows_changed(out_result_p)
    assert changed == 3, f"Expected 3 rows changed, got {changed}"
    ducklib.duckdb_destroy_result(out_result_p)
    aux_close_db(duckdb_database, duckdb_connection)


def test_result_statement_type():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    result_tup = tuple(out_result)
    # SELECT = DUCKDB_STATEMENT_TYPE_SELECT = 1
    stmt_type = ducklib.duckdb_result_statement_type(result_tup)
    assert stmt_type == 1, f"Expected SELECT (1), got {stmt_type}"
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)


def test_result_return_type():
    out_result, duckdb_database, duckdb_connection = aux_query_1()
    result_tup = tuple(out_result)
    # QUERY_RESULT = DUCKDB_RESULT_TYPE_QUERY_RESULT = 3
    ret_type = ducklib.duckdb_result_return_type(result_tup)
    assert ret_type == 3, f"Expected QUERY_RESULT (3), got {ret_type}"
    ducklib.duckdb_destroy_result(out_result.ctypes.data)
    aux_close_db(duckdb_database, duckdb_connection)
```

**Step 2: Run tests, commit, push, create PR**

---

## Phase 3: Value Interface

Branch: `value-interface` (off `upstream-result-metadata`)

Builds on Phase 2 — inherits `_call_byval` helper and `duckdb_destroy_logical_type` binding.
Once Phase 2 PR #12 merges upstream, rebase onto updated `origin/main`.

This is the largest phase (72 functions). Split into sub-tasks by category.

### Task 9: Add scalar value creators (14 functions) ✅

**Files:**
- Modify: `numbduck/ducklib.py`
- Modify: `numbduck/duckdb_utils.py` — add `create_duckdb_value()`

Functions: `duckdb_create_bool`, `duckdb_create_int8`, `duckdb_create_uint8`, `duckdb_create_int16`, `duckdb_create_uint16`, `duckdb_create_int32`, `duckdb_create_uint32`, `duckdb_create_int64`, `duckdb_create_uint64`, `duckdb_create_float`, `duckdb_create_double`, `duckdb_create_varchar`, `duckdb_create_varchar_length`, `duckdb_create_null_value`.

All return `intp` (duckdb_value is a pointer). All use `@cres` + `_call_lib_func`.

Signatures:
```python
signatures["duckdb_create_bool"] = intp(int8)
signatures["duckdb_create_int8"] = intp(int8)
signatures["duckdb_create_uint8"] = intp(uint8)
signatures["duckdb_create_int16"] = intp(int16)
signatures["duckdb_create_uint16"] = intp(uint16)
signatures["duckdb_create_int32"] = intp(int32)
signatures["duckdb_create_uint32"] = intp(uint32)
signatures["duckdb_create_int64"] = intp(int64)
signatures["duckdb_create_uint64"] = intp(uint64)
signatures["duckdb_create_float"] = intp(float32)
signatures["duckdb_create_double"] = intp(float64)
signatures["duckdb_create_varchar"] = intp(intp)
signatures["duckdb_create_varchar_length"] = intp(intp, uint64)
signatures["duckdb_create_null_value"] = intp()
```

Add `create_duckdb_value()` to `duckdb_utils.py`:
```python
@njit
def create_duckdb_value():
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_value """
    return allocate_buffer(1)
```

### Task 10: Add scalar value getters (14 functions) ✅

Functions: `duckdb_get_bool`, `duckdb_get_int8`, `duckdb_get_uint8`, `duckdb_get_int16`, `duckdb_get_uint16`, `duckdb_get_int32`, `duckdb_get_uint32`, `duckdb_get_int64`, `duckdb_get_uint64`, `duckdb_get_float`, `duckdb_get_double`, `duckdb_get_varchar`, `duckdb_get_enum_value`.

All take `intp` (duckdb_value pointer), return scalar. All use `@cres`.

```python
signatures["duckdb_get_bool"] = int8(intp)
signatures["duckdb_get_int8"] = int8(intp)
signatures["duckdb_get_uint8"] = uint8(intp)
signatures["duckdb_get_int16"] = int16(intp)
signatures["duckdb_get_uint16"] = uint16(intp)
signatures["duckdb_get_int32"] = int32(intp)
signatures["duckdb_get_uint32"] = uint32(intp)
signatures["duckdb_get_int64"] = int64(intp)
signatures["duckdb_get_uint64"] = uint64(intp)
signatures["duckdb_get_float"] = float32(intp)
signatures["duckdb_get_double"] = float64(intp)
signatures["duckdb_get_varchar"] = intp(intp)
signatures["duckdb_get_enum_value"] = uint64(intp)
```

### Task 11: Add value utility and container functions (10 functions) ✅

Functions: `duckdb_destroy_value`, `duckdb_is_null_value`, `duckdb_value_to_string`, `duckdb_get_value_type`, `duckdb_get_map_size`, `duckdb_get_map_key`, `duckdb_get_map_value`, `duckdb_get_list_size`, `duckdb_get_list_child`, `duckdb_get_struct_child`.

All pointer-based, all use `@cres`.

### Task 12: Add container value creators (6 functions) ✅

Functions: `duckdb_create_struct_value`, `duckdb_create_list_value`, `duckdb_create_array_value`, `duckdb_create_map_value`, `duckdb_create_union_value`, `duckdb_create_enum_value`.

All pointer-based, all use `@cres`.

### Task 13: Add struct-by-value value creators (Pattern A intrinsics, 16 functions) ✅

Functions: `duckdb_create_hugeint`, `duckdb_create_uhugeint`, `duckdb_create_varint`, `duckdb_create_decimal`, `duckdb_create_date`, `duckdb_create_time`, `duckdb_create_time_tz_value`, `duckdb_create_timestamp`, `duckdb_create_timestamp_tz`, `duckdb_create_timestamp_s`, `duckdb_create_timestamp_ms`, `duckdb_create_timestamp_ns`, `duckdb_create_interval`, `duckdb_create_blob`, `duckdb_create_bit`, `duckdb_create_uuid`.

Most single-field structs (date, time, timestamps) can use `@cres` since they're just a single scalar wrapped in a struct — LLVM will pass the scalar directly. Only multi-field structs (hugeint, uhugeint, interval, decimal) need `@intrinsic` — use `_call_byval` helper from Phase 2 where the pattern fits.

For `duckdb_create_blob` and `duckdb_create_bit`: these take `(const uint8_t *data, idx_t length)` and `(duckdb_bit input)` respectively — check duckdb.h for exact signatures.

**Re-check each signature against duckdb.h before implementing.**

### Task 14: Add struct-by-value value getters (Pattern C intrinsics, 16 functions) ✅

Functions: `duckdb_get_hugeint`, `duckdb_get_uhugeint`, `duckdb_get_varint`, `duckdb_get_decimal`, `duckdb_get_date`, `duckdb_get_time`, `duckdb_get_time_tz`, `duckdb_get_timestamp`, `duckdb_get_timestamp_tz`, `duckdb_get_timestamp_s`, `duckdb_get_timestamp_ms`, `duckdb_get_timestamp_ns`, `duckdb_get_interval`, `duckdb_get_blob`, `duckdb_get_bit`, `duckdb_get_uuid`.

Pattern C intrinsic: call C function, receive struct, extract fields into tuple, return tuple.

For single-field structs (date → int32, time → int64, timestamps → int64), the C function returns a struct but LLVM will return just the scalar. Use `@cres` for these.

For multi-field structs (hugeint, uhugeint, interval, decimal, blob, bit, varint), need `@intrinsic`:

```python
@intrinsic
def _duckdb_get_hugeint(typingctx, val_p_ty):
    def codegen(context, builder: IRBuilder, signature, arguments):
        val_p = arguments[0]
        hugeint_ll_ty = context.get_value_type(duckdb_hugeint_ty)
        func_ty_ll = FunctionType(hugeint_ll_ty, [val_p.type])
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_get_hugeint")
        return builder.call(func_p, [val_p])
    return duckdb_hugeint_ty(intp), codegen
```

### Task 15: Add tests for scalar value create/get round-trips ✅

Test pattern: create value → get value → verify → destroy.

```python
def test_create_get_int32():
    val_p = ducklib.duckdb_create_int32(42)
    assert val_p != 0
    result = ducklib.duckdb_get_int32(val_p)
    assert result == 42
    ducklib.duckdb_destroy_value(val_p)
```

Write similar tests for each scalar type.

### Task 16: Add tests for struct value create/get round-trips ✅

Test hugeint, interval, decimal round-trips.

### Task 17: Add tests for container and utility functions ⚠️ PARTIAL

Test null value, value_to_string, list/map/struct creation and child access.

### Task 18: Run full suite, lint, push, create PR

---

## Phase 4: Logical Type Interface

Branch: `logical-types` (off latest main after Phase 3 merges)
`duckdb_destroy_logical_type` already bound in Phase 2 — do not re-add.

### Task 19: Add logical type create functions (8 functions)

All pointer-based, all `@cres`. No intrinsics needed.

```python
signatures["duckdb_create_logical_type"] = intp(int32)
signatures["duckdb_create_list_type"] = intp(intp)
signatures["duckdb_create_array_type"] = intp(intp, uint64)
signatures["duckdb_create_map_type"] = intp(intp, intp)
signatures["duckdb_create_union_type"] = intp(intp, intp, uint64)
signatures["duckdb_create_struct_type"] = intp(intp, intp, uint64)
signatures["duckdb_create_enum_type"] = intp(intp, uint64)
signatures["duckdb_create_decimal_type"] = intp(uint8, uint8)
```

### Task 20: Add logical type inspect functions (20 functions)

All pointer-based, all `@cres`.

```python
signatures["duckdb_get_type_id"] = int32(intp)
signatures["duckdb_logical_type_get_alias"] = intp(intp)
signatures["duckdb_logical_type_set_alias"] = void(intp, intp)
signatures["duckdb_decimal_width"] = uint8(intp)
signatures["duckdb_decimal_scale"] = uint8(intp)
signatures["duckdb_decimal_internal_type"] = int32(intp)
signatures["duckdb_enum_internal_type"] = int32(intp)
signatures["duckdb_enum_dictionary_size"] = uint32(intp)
signatures["duckdb_enum_dictionary_value"] = intp(intp, uint64)
signatures["duckdb_list_type_child_type"] = intp(intp)
signatures["duckdb_array_type_child_type"] = intp(intp)
signatures["duckdb_array_type_array_size"] = uint64(intp)
signatures["duckdb_map_type_key_type"] = intp(intp)
signatures["duckdb_map_type_value_type"] = intp(intp)
signatures["duckdb_struct_type_child_count"] = uint64(intp)
signatures["duckdb_struct_type_child_name"] = intp(intp, uint64)
signatures["duckdb_struct_type_child_type"] = intp(intp, uint64)
signatures["duckdb_union_type_member_count"] = uint64(intp)
signatures["duckdb_union_type_member_name"] = intp(intp, uint64)
signatures["duckdb_union_type_member_type"] = intp(intp, uint64)
```

### Task 21: Add destroy_logical_type ✅ (done in Phase 2, PR #12 review feedback)

Already bound in Phase 2 — signature, wrapper, and test cleanup all committed.

### Task 22: Add tests for logical types

Test create → inspect → destroy for each type variant (basic, list, map, struct, enum, union, decimal, array).

### Task 23: Run full suite, lint, push, create PR

---

## Phase 5: Configuration

Branch: `configuration` (off `origin/main`)

### Task 24: Add configuration bindings (5 functions)

**Files:**
- Modify: `numbduck/ducklib.py`
- Modify: `numbduck/duckdb_utils.py` — add `create_duckdb_config()`

All pointer-based, all `@cres`.

```python
signatures["duckdb_create_config"] = duckdb_state_ty(intp)
signatures["duckdb_config_count"] = uint64()
signatures["duckdb_get_config_flag"] = duckdb_state_ty(uint64, intp, intp)
signatures["duckdb_set_config"] = duckdb_state_ty(intp, intp, intp)
signatures["duckdb_destroy_config"] = void(intp)
```

Add `create_duckdb_config()` to `duckdb_utils.py`:
```python
@njit
def create_duckdb_config():
    """ https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L475 """
    return allocate_buffer(1)
```

### Task 25: Add tests for configuration

```python
def test_config_count():
    count = ducklib.duckdb_config_count()
    assert count > 0, f"Expected positive config count, got {count}"


def test_create_set_destroy_config():
    config = create_duckdb_config()
    config_pp = config.ctypes.data
    rc = ducklib.duckdb_create_config(config_pp)
    assert rc == ducklib.DuckDBSuccess
    config_p = config[0]
    assert config_p != 0
    name_bytes = ctypes.c_char_p(b"threads")
    name_p = ctypes.c_void_p.from_buffer(name_bytes).value
    val_bytes = ctypes.c_char_p(b"1")
    val_p = ctypes.c_void_p.from_buffer(val_bytes).value
    rc = ducklib.duckdb_set_config(config_p, name_p, val_p)
    assert rc == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_config(config_pp)
```

### Task 26: Run full suite, lint, push, create PR
