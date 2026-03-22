# Logical Type Interface Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Add logical type create and inspect bindings to numbduck, enabling JIT code to create, inspect, and destroy logical types for all DuckDB type variants.

**Architecture:** All 28 functions use `@cres` + `_call_lib_func` (pointer-based, no struct-by-value). `duckdb_destroy_logical_type` already bound in Phase 2. Signatures verified against `duckdb.h` v1.3.2.

**Tech Stack:** numba, numbox, llvmlite, ctypes, pytest

**Reference files:**
- `duckdb.h` (v1.3.2) at project root — C API header, source of truth for all signatures
- `numbduck/ducklib.py` — all bindings live here
- `test/test_ducklib.py` — all tests

**Key type mappings (from `duckdb.h`):**
- `duckdb_logical_type` → pointer (`intp`) — `typedef struct { void *internal_ptr; } *duckdb_logical_type`
- `duckdb_type` → enum (`int32`) — `DUCKDB_TYPE_BOOLEAN=1`, `DUCKDB_TYPE_INTEGER=4`, `DUCKDB_TYPE_VARCHAR=17`, `DUCKDB_TYPE_LIST=24`, `DUCKDB_TYPE_STRUCT=25`, `DUCKDB_TYPE_MAP=26`, `DUCKDB_TYPE_ENUM=23`, `DUCKDB_TYPE_ARRAY=33`, `DUCKDB_TYPE_UNION=28`, `DUCKDB_TYPE_DECIMAL=19`
- `idx_t` → `uint64`
- `char *` → `intp`

**Agent assignment:** Use sonnet for all tasks (all `@cres`, no intrinsics).

**Lessons from prior phases:**
- Keep signatures and wrappers in alphabetical order
- Docstring links must use `duckdb.org/docs/stable/clients/c/api.html#func_name`
- Always bind the corresponding destroy function when binding a function that returns a handle (already done: `duckdb_destroy_logical_type`)
- Assert return codes on setup statements in tests
- Functions returning `char *` (like `duckdb_enum_dictionary_value`, `duckdb_struct_type_child_name`, `duckdb_union_type_member_name`, `duckdb_logical_type_get_alias`) return allocated strings that must be freed with `duckdb_free`
- Never put planning details in code comments
- Clean `__pycache__` and `~/.cache/numba` before every pytest run

---

## Task 1: Create feature branch

**Step 1: Create and push branch**

```bash
cd /home/erik/projects/numbduck
git checkout main
git pull origin main
git checkout -b logical-types
git push -u origin logical-types
```

**Step 2: Verify CLAUDE.md is present**

```bash
test -f CLAUDE.md && echo "OK"
```

---

## Task 2: Add logical type create signatures and wrappers (8 functions)

**Files:**
- Modify: `numbduck/ducklib.py`

**Step 1: Add signatures**

Insert in alphabetical order among existing signatures (after `duckdb_create_int64` / before `duckdb_create_list_value`):

```python
signatures["duckdb_create_array_type"] = intp(intp, uint64)
signatures["duckdb_create_decimal_type"] = intp(uint8, uint8)
signatures["duckdb_create_enum_type"] = intp(intp, uint64)
signatures["duckdb_create_list_type"] = intp(intp)
signatures["duckdb_create_logical_type"] = intp(int32)
signatures["duckdb_create_map_type"] = intp(intp, intp)
signatures["duckdb_create_struct_type"] = intp(intp, intp, uint64)
signatures["duckdb_create_union_type"] = intp(intp, intp, uint64)
```

Note: these must be interleaved with existing `duckdb_create_*` signatures to maintain alphabetical order.

**Step 2: Add wrapper functions**

Insert in alphabetical order among existing wrappers. Each follows the standard `@cres` + `_call_lib_func` pattern:

```python
@cres(signatures.get("duckdb_create_array_type"))
def duckdb_create_array_type(type_p, array_size):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_array_type """
    return _call_lib_func("duckdb_create_array_type", (type_p, array_size))


@cres(signatures.get("duckdb_create_decimal_type"))
def duckdb_create_decimal_type(width, scale):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_decimal_type """
    return _call_lib_func("duckdb_create_decimal_type", (width, scale))


@cres(signatures.get("duckdb_create_enum_type"))
def duckdb_create_enum_type(member_names_p, member_count):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_enum_type """
    return _call_lib_func("duckdb_create_enum_type", (member_names_p, member_count))


@cres(signatures.get("duckdb_create_list_type"))
def duckdb_create_list_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_list_type """
    return _call_lib_func("duckdb_create_list_type", (type_p,))


@cres(signatures.get("duckdb_create_logical_type"))
def duckdb_create_logical_type(type_id):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_logical_type """
    return _call_lib_func("duckdb_create_logical_type", (type_id,))


@cres(signatures.get("duckdb_create_map_type"))
def duckdb_create_map_type(key_type_p, value_type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_map_type """
    return _call_lib_func("duckdb_create_map_type", (key_type_p, value_type_p))


@cres(signatures.get("duckdb_create_struct_type"))
def duckdb_create_struct_type(member_types_p, member_names_p, member_count):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_struct_type """
    return _call_lib_func("duckdb_create_struct_type", (member_types_p, member_names_p, member_count))


@cres(signatures.get("duckdb_create_union_type"))
def duckdb_create_union_type(member_types_p, member_names_p, member_count):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_union_type """
    return _call_lib_func("duckdb_create_union_type", (member_types_p, member_names_p, member_count))
```

**Step 3: Clean caches and run tests**

```bash
find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba
cd /home/erik/projects/numbduck && venv/bin/pytest test/test_ducklib.py -v
```

Expected: all existing tests pass (no new tests yet).

**Step 4: Lint**

```bash
cd /home/erik/projects/numbduck && venv/bin/flake8 numbduck/ducklib.py
```

**Step 5: Commit**

```bash
git add numbduck/ducklib.py
git commit -m "Add logical type create signatures and wrappers (8 functions)"
```

---

## Task 3: Add logical type inspect signatures and wrappers (20 functions)

**Files:**
- Modify: `numbduck/ducklib.py`

**Step 1: Add signatures**

Insert in alphabetical order among existing signatures:

```python
signatures["duckdb_array_type_array_size"] = uint64(intp)
signatures["duckdb_array_type_child_type"] = intp(intp)
signatures["duckdb_decimal_internal_type"] = int32(intp)
signatures["duckdb_decimal_scale"] = uint8(intp)
signatures["duckdb_decimal_width"] = uint8(intp)
signatures["duckdb_enum_dictionary_size"] = uint32(intp)
signatures["duckdb_enum_dictionary_value"] = intp(intp, uint64)
signatures["duckdb_enum_internal_type"] = int32(intp)
signatures["duckdb_get_type_id"] = int32(intp)
signatures["duckdb_list_type_child_type"] = intp(intp)
signatures["duckdb_logical_type_get_alias"] = intp(intp)
signatures["duckdb_logical_type_set_alias"] = void(intp, intp)
signatures["duckdb_map_type_key_type"] = intp(intp)
signatures["duckdb_map_type_value_type"] = intp(intp)
signatures["duckdb_struct_type_child_count"] = uint64(intp)
signatures["duckdb_struct_type_child_name"] = intp(intp, uint64)
signatures["duckdb_struct_type_child_type"] = intp(intp, uint64)
signatures["duckdb_union_type_member_count"] = uint64(intp)
signatures["duckdb_union_type_member_name"] = intp(intp, uint64)
signatures["duckdb_union_type_member_type"] = intp(intp, uint64)
```

**Step 2: Add wrapper functions**

Insert in alphabetical order among existing wrappers:

```python
@cres(signatures.get("duckdb_array_type_array_size"))
def duckdb_array_type_array_size(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_array_type_array_size """
    return _call_lib_func("duckdb_array_type_array_size", (type_p,))


@cres(signatures.get("duckdb_array_type_child_type"))
def duckdb_array_type_child_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_array_type_child_type """
    return _call_lib_func("duckdb_array_type_child_type", (type_p,))


@cres(signatures.get("duckdb_decimal_internal_type"))
def duckdb_decimal_internal_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_decimal_internal_type """
    return _call_lib_func("duckdb_decimal_internal_type", (type_p,))


@cres(signatures.get("duckdb_decimal_scale"))
def duckdb_decimal_scale(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_decimal_scale """
    return _call_lib_func("duckdb_decimal_scale", (type_p,))


@cres(signatures.get("duckdb_decimal_width"))
def duckdb_decimal_width(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_decimal_width """
    return _call_lib_func("duckdb_decimal_width", (type_p,))


@cres(signatures.get("duckdb_enum_dictionary_size"))
def duckdb_enum_dictionary_size(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_enum_dictionary_size """
    return _call_lib_func("duckdb_enum_dictionary_size", (type_p,))


@cres(signatures.get("duckdb_enum_dictionary_value"))
def duckdb_enum_dictionary_value(type_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_enum_dictionary_value """
    return _call_lib_func("duckdb_enum_dictionary_value", (type_p, index))


@cres(signatures.get("duckdb_enum_internal_type"))
def duckdb_enum_internal_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_enum_internal_type """
    return _call_lib_func("duckdb_enum_internal_type", (type_p,))


@cres(signatures.get("duckdb_get_type_id"))
def duckdb_get_type_id(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_type_id """
    return _call_lib_func("duckdb_get_type_id", (type_p,))


@cres(signatures.get("duckdb_list_type_child_type"))
def duckdb_list_type_child_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_list_type_child_type """
    return _call_lib_func("duckdb_list_type_child_type", (type_p,))


@cres(signatures.get("duckdb_logical_type_get_alias"))
def duckdb_logical_type_get_alias(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_logical_type_get_alias """
    return _call_lib_func("duckdb_logical_type_get_alias", (type_p,))


@cres(signatures.get("duckdb_logical_type_set_alias"))
def duckdb_logical_type_set_alias(type_p, alias_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_logical_type_set_alias """
    return _call_lib_func("duckdb_logical_type_set_alias", (type_p, alias_p))


@cres(signatures.get("duckdb_map_type_key_type"))
def duckdb_map_type_key_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_map_type_key_type """
    return _call_lib_func("duckdb_map_type_key_type", (type_p,))


@cres(signatures.get("duckdb_map_type_value_type"))
def duckdb_map_type_value_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_map_type_value_type """
    return _call_lib_func("duckdb_map_type_value_type", (type_p,))


@cres(signatures.get("duckdb_struct_type_child_count"))
def duckdb_struct_type_child_count(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_struct_type_child_count """
    return _call_lib_func("duckdb_struct_type_child_count", (type_p,))


@cres(signatures.get("duckdb_struct_type_child_name"))
def duckdb_struct_type_child_name(type_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_struct_type_child_name """
    return _call_lib_func("duckdb_struct_type_child_name", (type_p, index))


@cres(signatures.get("duckdb_struct_type_child_type"))
def duckdb_struct_type_child_type(type_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_struct_type_child_type """
    return _call_lib_func("duckdb_struct_type_child_type", (type_p, index))


@cres(signatures.get("duckdb_union_type_member_count"))
def duckdb_union_type_member_count(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_union_type_member_count """
    return _call_lib_func("duckdb_union_type_member_count", (type_p,))


@cres(signatures.get("duckdb_union_type_member_name"))
def duckdb_union_type_member_name(type_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_union_type_member_name """
    return _call_lib_func("duckdb_union_type_member_name", (type_p, index))


@cres(signatures.get("duckdb_union_type_member_type"))
def duckdb_union_type_member_type(type_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_union_type_member_type """
    return _call_lib_func("duckdb_union_type_member_type", (type_p, index))
```

**Step 3: Clean caches and run tests**

```bash
find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba
cd /home/erik/projects/numbduck && venv/bin/pytest test/test_ducklib.py -v
```

Expected: all existing tests pass.

**Step 4: Lint**

```bash
cd /home/erik/projects/numbduck && venv/bin/flake8 numbduck/ducklib.py
```

**Step 5: Commit**

```bash
git add numbduck/ducklib.py
git commit -m "Add logical type inspect signatures and wrappers (20 functions)"
```

---

## Task 4: Add tests for basic and decimal logical types

**Files:**
- Modify: `test/test_ducklib.py`

**Step 1: Add helper for logical type buffer**

Add near the top of the test file, after existing imports/helpers:

```python
def aux_create_logical_type_buf():
    """Allocate a numpy buffer to hold a duckdb_logical_type pointer."""
    return numpy.zeros(1, dtype=numpy.intp)
```

**Step 2: Write tests**

```python
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
```

**Step 3: Clean caches and run tests**

```bash
find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba
cd /home/erik/projects/numbduck && venv/bin/pytest test/test_ducklib.py -v -k "logical_type or decimal_type"
```

Expected: all 4 new tests pass.

**Step 4: Commit**

```bash
git add test/test_ducklib.py
git commit -m "Add tests for basic and decimal logical types"
```

---

## Task 5: Add tests for list, array, and map logical types

**Files:**
- Modify: `test/test_ducklib.py`

**Step 1: Write tests**

```python
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
```

**Step 2: Clean caches and run tests**

```bash
find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba
cd /home/erik/projects/numbduck && venv/bin/pytest test/test_ducklib.py -v -k "list_type or array_type or map_type"
```

Expected: all 3 new tests pass.

**Step 3: Commit**

```bash
git add test/test_ducklib.py
git commit -m "Add tests for list, array, and map logical types"
```

---

## Task 6: Add tests for struct, union, and enum logical types

**Files:**
- Modify: `test/test_ducklib.py`

**Step 1: Write tests**

```python
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
```

**Step 2: Clean caches and run tests**

```bash
find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba
cd /home/erik/projects/numbduck && venv/bin/pytest test/test_ducklib.py -v -k "struct_type or union_type or enum_type"
```

Expected: all 3 new tests pass.

**Step 3: Commit**

```bash
git add test/test_ducklib.py
git commit -m "Add tests for struct, union, and enum logical types"
```

---

## Task 7: Run full suite, lint, push

**Step 1: Clean caches and run full test suite**

```bash
find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; rm -rf ~/.cache/numba
cd /home/erik/projects/numbduck && venv/bin/pytest test/test_ducklib.py -v
```

Expected: all tests pass (existing + 10 new).

**Step 2: Lint**

```bash
cd /home/erik/projects/numbduck && venv/bin/flake8 numbduck/ducklib.py test/test_ducklib.py
```

Expected: no errors.

**Step 3: Push**

```bash
git push origin logical-types
```

**Step 4: Wait for CI to pass, then create PR**

Check CI status and create PR against upstream when ready.

---

## Task 8: Create upstream PR branch and PR

**Step 1: Create upstream PR branch**

```bash
git checkout -b upstream-logical-types upstream/main
```

**Step 2: Cherry-pick commits (excluding CLAUDE.md changes)**

Cherry-pick only the commits from `logical-types` that contain binding and test changes. Exclude any commits that only touch CLAUDE.md, docs/plans/, or .github/workflows/numbduck_ci.yml.

**Step 3: Verify no fork-only files**

```bash
git diff upstream/main --name-only | grep -E 'CLAUDE.md|numbduck_ci.yml|docs/plans/' && echo "ERROR: fork-only files present" || echo "OK: clean"
```

**Step 4: Push and create PR**

```bash
git push -u origin upstream-logical-types
```

Create PR against `Goykhman/numbduck:main` with title: "Add logical type interface bindings and tests"
