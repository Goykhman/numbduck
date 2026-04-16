# Structref-backed UDAF stddev — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:subagent-driven-development (recommended) or superpowers-extended-cc:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three tests to `test/test_ducklib.py` that prove a numba structref can back a DuckDB UDAF via the indirect-state pattern, with refcount correctness verified at the intrinsic level before any DuckDB wiring.

**Architecture:** DuckDB's 8-byte `internal_ptr` slot holds a MemInfo pointer to an NRT-allocated structref. Four bridge intrinsics (`export_meminfo`, `borrow_structref`, `release_meminfo`, `refcount_of_meminfo_intp`) handle the engine ↔ numba boundary. Welford's algorithm computes sample stddev with a 3-field state.

**Tech Stack:** numba 0.60+, llvmlite (for intrinsic codegen), numbox utility helpers, DuckDB 1.3.2 aggregate C API.

**Spec:** `docs/plans/2026-04-15-structref-udaf-stddev-design.md`

**Branch:** `feat/structref-udaf-stddev`

**Test file:** all work lands in `test/test_ducklib.py`. No production-package changes; promotion to `numbduck/structref_bridge.py` is deferred until a second use case exists.

**Ordering guard:** Task 1 MUST pass before starting Task 3 or Task 4. Task 2 is independent and can run in parallel with Task 1.

---

## Task 1: Bridge intrinsics + refcount ladder test

**Goal:** Implement the four bridge intrinsics and a test that asserts the full refcount ladder (1 → 2 → 1 → 2 → 1 → 0) using numbox's `get_nrt_refcount`. Until this passes, the approach is unverified.

**Files:**
- Modify: `test/test_ducklib.py` — add bridge helpers at module scope near existing `_cast_int_to_void_p` import, add test at the end of the file

**Acceptance Criteria:**
- [ ] `export_meminfo(s)` returns a nonzero `intp` and leaves `s` with refcount == 2
- [ ] `borrow_structref(ty, p)` returns a structref whose field reads match what was written, refcount == 2 during borrow
- [ ] `release_meminfo(p)` decrements refcount; from 1 → 0 the structref is freed (NRT allocation stats show matching free)
- [ ] `refcount_of_meminfo_intp(p)` returns the correct refcount at each step of the ladder
- [ ] Test passes: `pytest test/test_ducklib.py::test_structref_meminfo_bridge_refcount_ladder -v`

**Verify:**
```bash
rm -rf ~/.cache/numba && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
cd /home/erik/projects/numbduck && venv/bin/pytest \
    test/test_ducklib.py::test_structref_meminfo_bridge_refcount_ladder -v
```
Expected: `1 passed`.

**Steps:**

- [ ] **Step 1: Add imports and bridge intrinsics at module scope**

Add to imports section of `test/test_ducklib.py`:

```python
from numba import njit
from numba.core import cgutils, types
from numba.experimental import structref
from numba.extending import intrinsic
from numbox.utils.lowlevel import _cast_int_to_void_p
from numbox.utils.meminfo import get_nrt_refcount
```

Add the four intrinsics as module-level helpers:

```python
# ---- structref <-> raw pointer bridge intrinsics ----
#
# DuckDB's aggregate state slot is a single `void *internal_ptr` owned by the
# engine. These helpers let an NRT-managed structref round-trip through that
# slot without breaking reference counts.
#
# Reference counting contract:
#   export_meminfo(s):
#       Increfs s's MemInfo. Returned intp represents a new owning reference
#       held outside numba's reach (typically: stored into a C struct slot).
#       The local structref variable `s` goes out of scope normally; its
#       decref-on-exit balances the original +1 from allocation, leaving the
#       refcount at exactly 1 after the caller's scope exits.
#
#   borrow_structref(ty, p):
#       Reconstructs a typed structref value whose .meminfo = p, with an
#       incref-on-entry. The local variable's decref-on-exit cancels this
#       incref. Net effect: zero change to the MemInfo's refcount across the
#       borrow's lifetime. The external owner's reference is undisturbed.
#
#   release_meminfo(p):
#       Drops one NRT reference on the MemInfo at p. When refcount hits 0,
#       the dtor (registered at alloc time via imp_dtor) runs.
#
#   refcount_of_meminfo_intp(p):
#       Reads the MemInfo's refcount for test assertions.

@intrinsic
def _export_meminfo(typingctx, struct_ty):
    """ Incref MemInfo and return as intp. Input: any StructRef. """
    assert isinstance(struct_ty, types.StructRef)
    sig = types.intp(struct_ty)

    def codegen(context, builder, signature, args):
        struct_val = args[0]
        meminfo = context.nrt.get_meminfos(
            builder, struct_ty, struct_val)[0][1]
        context.nrt.incref(
            builder, types.MemInfoPointer(types.voidptr), meminfo)
        return builder.ptrtoint(meminfo, cgutils.intp_t)
    return sig, codegen


@njit
def export_meminfo(s):
    return _export_meminfo(s)


@intrinsic
def _borrow_structref(typingctx, struct_type_ref, p_ty):
    """ Reconstruct structref from MemInfo intp; increfs on entry. """
    inst_type = struct_type_ref.instance_type
    sig = inst_type(struct_type_ref, p_ty)

    def codegen(context, builder, signature, args):
        _, p_val = args
        mi_ll_ty = context.get_value_type(
            types.MemInfoPointer(types.voidptr))
        meminfo = builder.inttoptr(p_val, mi_ll_ty)
        context.nrt.incref(
            builder, types.MemInfoPointer(types.voidptr), meminfo)
        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo
        return st._getvalue()
    return sig, codegen


@njit
def borrow_structref(struct_type, p):
    return _borrow_structref(struct_type, p)


@intrinsic
def _release_meminfo(typingctx, p_ty):
    """ Decref MemInfo at intp. Triggers dtor at refcount 0. """
    sig = types.void(p_ty)

    def codegen(context, builder, signature, args):
        mi_ll_ty = context.get_value_type(
            types.MemInfoPointer(types.voidptr))
        meminfo = builder.inttoptr(args[0], mi_ll_ty)
        context.nrt.decref(
            builder, types.MemInfoPointer(types.voidptr), meminfo)
    return sig, codegen


@njit
def release_meminfo(p):
    _release_meminfo(p)


@intrinsic
def _refcount_of_meminfo_intp(typingctx, p_ty):
    """ Read refcount field from MemInfo at intp. """
    sig = types.intp(p_ty)

    def codegen(context, builder, signature, args):
        mi_ll_ty = context.get_value_type(
            types.MemInfoPointer(types.voidptr))
        meminfo = builder.inttoptr(args[0], mi_ll_ty)
        # MemInfo layout: { refct, dtor, dtor_info, data, size }
        # refct is the first field (i64 / size_t)
        refct_ptr = builder.gep(
            meminfo, [cgutils.int32_t(0), cgutils.int32_t(0)])
        return builder.load(refct_ptr)
    return sig, codegen


@njit
def refcount_of_meminfo_intp(p):
    return _refcount_of_meminfo_intp(p)
```

Note on the MemInfo layout: numba's MemInfo struct has `refct` as its first field. If the test below fails with a weird refcount value, verify against `numba.core.runtime.nrtdynmod` (search for `_define_nrt_meminfo_data`) — the field order has been stable since numba 0.50 but is internal API.

- [ ] **Step 2: Define the Welford structref type (used by test 1 and later tasks)**

Also at module scope:

```python
# ---- Welford state structref (used by stddev UDAF + bridge tests) ----

@structref.register
class WelfordStateType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)


class WelfordState(structref.StructRefProxy):
    def __new__(cls, mean, count, m2):
        return structref.StructRefProxy.__new__(cls, mean, count, m2)

    @property
    def mean(self):
        return _welford_get_mean(self)

    @property
    def count(self):
        return _welford_get_count(self)

    @property
    def m2(self):
        return _welford_get_m2(self)


@njit
def _welford_get_mean(s):
    return s.mean


@njit
def _welford_get_count(s):
    return s.count


@njit
def _welford_get_m2(s):
    return s.m2


structref.define_proxy(
    WelfordState, WelfordStateType, ["mean", "count", "m2"])

welford_type = WelfordStateType(
    [("mean", nb_types.float64),
     ("count", nb_types.int64),
     ("m2", nb_types.float64)])
```

- [ ] **Step 3: Write the refcount ladder test**

Add at end of `test/test_ducklib.py`:

```python
def test_structref_meminfo_bridge_refcount_ladder():
    """Prove export/borrow/release maintain refcount invariants step by step.

    Each njit step is its own function so numba's liveness analysis cannot
    extend a local's lifetime past its last use — we want the decref to
    happen when the function returns, not at some ambiguous later point.
    """

    @njit
    def _allocate_and_export():
        s = WelfordState(1.5, 2, 3.25)
        p = export_meminfo(s)
        # refcount here should be 2: one from `s`, one from export.
        rc_before_return = refcount_of_meminfo_intp(p)
        return p, rc_before_return

    @njit
    def _check_refcount(p):
        return refcount_of_meminfo_intp(p)

    @njit
    def _borrow_check_and_release(p):
        # borrow increfs (rc = 2), verify fields, then scope exit decrefs
        # (rc = 1), then we release explicitly (rc = 0, dtor runs).
        s = borrow_structref(welford_type, p)
        rc_during_borrow = refcount_of_meminfo_intp(p)
        mean, count, m2 = s.mean, s.count, s.m2
        # end of scope: borrow's decref fires when this function returns
        return rc_during_borrow, mean, count, m2

    p, rc_before_return = _allocate_and_export()
    assert rc_before_return == 2, (
        f"expected refcount 2 (local+export), got {rc_before_return}")

    # Local `s` inside _allocate_and_export went out of scope on return.
    # Only the export-owned reference remains.
    rc_after_export_return = _check_refcount(p)
    assert rc_after_export_return == 1, (
        f"expected refcount 1 after local dropped, got {rc_after_export_return}")

    rc_during_borrow, mean, count, m2 = _borrow_check_and_release(p)
    assert rc_during_borrow == 2, (
        f"expected refcount 2 during borrow, got {rc_during_borrow}")
    assert mean == 1.5, f"mean={mean}"
    assert count == 2, f"count={count}"
    assert m2 == 3.25, f"m2={m2}"

    rc_after_borrow_return = _check_refcount(p)
    assert rc_after_borrow_return == 1, (
        f"expected refcount 1 after borrow dropped, got {rc_after_borrow_return}")

    # Release the slot's reference. refcount → 0, dtor runs.
    @njit
    def _release(p):
        release_meminfo(p)

    _release(p)

    # Can't read refcount after free — memory may be reused. Instead verify
    # via NRT alloc stats that we ended up with matched alloc/free counts.
    # That check happens in test_structref_meminfo_bridge_nested_heap.
```

- [ ] **Step 4: Run the test**

```bash
rm -rf ~/.cache/numba && find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
cd /home/erik/projects/numbduck && venv/bin/pytest \
    test/test_ducklib.py::test_structref_meminfo_bridge_refcount_ladder -v
```

Expected: PASS.

**If it fails**, common causes in decreasing order of likelihood:

1. `preprocess_fields` signature mismatch between numba versions — check `numba.experimental.structref.StructRef.preprocess_fields` for the current signature.
2. MemInfo refcount field offset wrong — add a print of `meminfo` as a raw `intp` at Step 1 intrinsic entry and compare vs numbox's `get_nrt_refcount` on the same structref.
3. `define_proxy` called too late (must be before `welford_type` is constructed).
4. The `structref.StructRefProxy.__new__` call needs the proxy attribute accessors registered via `@overload_method(WelfordStateType, ...)` — follow
   [numba structref docs](https://numba.readthedocs.io/en/stable/extending/high-level.html#structref).
   The `_welford_get_*` helpers above may not be enough; may need `@overload_attribute` for the `.mean`/`.count`/`.m2` reads used inside `@njit`.

Iterate on these until the test passes. Do not proceed to later tasks until it does.

- [ ] **Step 5: Commit**

```bash
git -c safe.directory=/home/erik/projects/numbduck add test/test_ducklib.py
git -c safe.directory=/home/erik/projects/numbduck commit -m "Add structref-MemInfo bridge intrinsics with refcount ladder test"
```

---

## Task 2: Welford algorithm + pure-numba stddev test

**Goal:** Implement `welford_update`, `welford_combine`, `welford_finalize` as `@njit` functions operating on `WelfordState`, and verify correctness against `numpy.std(ddof=1)` without any DuckDB involvement.

**Files:**
- Modify: `test/test_ducklib.py` — add Welford ops and a correctness test

**Acceptance Criteria:**
- [ ] `welford_update(s, x)` maintains Welford invariants
- [ ] `welford_combine(src, tgt)` correctly merges two states (tested against serial update on concatenated data)
- [ ] `welford_finalize(s)` returns `sqrt(m2/(count-1))` for count≥2, NaN otherwise
- [ ] Result matches `numpy.std([1..7], ddof=1)` to 1e-10

**Verify:**
```bash
cd /home/erik/projects/numbduck && venv/bin/pytest \
    test/test_ducklib.py::test_welford_numba_only -v
```
Expected: `1 passed`.

**Steps:**

- [ ] **Step 1: Add Welford operations near the WelfordState definition**

```python
@njit
def welford_update(s, x):
    s.count += 1
    delta = x - s.mean
    s.mean += delta / s.count
    delta2 = x - s.mean
    s.m2 += delta * delta2


@njit
def welford_combine(src, tgt):
    """Merge src into tgt in place. Chan et al. pairwise formula."""
    if src.count == 0:
        return
    if tgt.count == 0:
        tgt.mean = src.mean
        tgt.count = src.count
        tgt.m2 = src.m2
        return
    new_count = src.count + tgt.count
    delta = src.mean - tgt.mean
    new_mean = tgt.mean + delta * src.count / new_count
    new_m2 = (tgt.m2 + src.m2
              + delta * delta * tgt.count * src.count / new_count)
    tgt.mean = new_mean
    tgt.count = new_count
    tgt.m2 = new_m2


@njit
def welford_finalize(s):
    if s.count < 2:
        return math.nan
    return math.sqrt(s.m2 / (s.count - 1))
```

Requires `import math` at module top (check whether already imported).

- [ ] **Step 2: Write the pure-numba test**

```python
def test_welford_numba_only():
    """Verify Welford ops match numpy.std(ddof=1) without DuckDB."""

    @njit
    def _compute_serial(xs):
        s = WelfordState(0.0, 0, 0.0)
        for x in xs:
            welford_update(s, x)
        return welford_finalize(s)

    @njit
    def _compute_combined(xs_a, xs_b):
        sa = WelfordState(0.0, 0, 0.0)
        sb = WelfordState(0.0, 0, 0.0)
        for x in xs_a:
            welford_update(sa, x)
        for x in xs_b:
            welford_update(sb, x)
        welford_combine(sa, sb)
        return welford_finalize(sb)

    xs = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    expected = float(numpy.std(xs, ddof=1))

    serial = _compute_serial(xs)
    assert abs(serial - expected) < 1e-10, (
        f"serial: got {serial}, expected {expected}")

    # Split and combine: verify combine correctness
    combined = _compute_combined(xs[:3], xs[3:])
    assert abs(combined - expected) < 1e-10, (
        f"combined: got {combined}, expected {expected}")
```

- [ ] **Step 3: Run and commit**

```bash
cd /home/erik/projects/numbduck && venv/bin/pytest \
    test/test_ducklib.py::test_welford_numba_only -v
```

```bash
git -c safe.directory=/home/erik/projects/numbduck add test/test_ducklib.py
git -c safe.directory=/home/erik/projects/numbduck commit -m "Add Welford algorithm with pure-numba stddev correctness test"
```

---

## Task 3: Nested-heap dtor chain test

**Goal:** Prove `release_meminfo` correctly cascades decref into a nested heap-owning field (a `typed.List[float64]`). Uses a dedicated structref with a list field — separate from Welford.

**Depends on:** Task 1.

**Files:**
- Modify: `test/test_ducklib.py` — add second structref type and test

**Acceptance Criteria:**
- [ ] After 100 alloc+release cycles, NRT allocation stats show equal alloc and free counts (no leak)
- [ ] The delta from one cycle is measurable (warmup confirms cycle actually allocates)

**Verify:**
```bash
rm -rf ~/.cache/numba && find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
cd /home/erik/projects/numbduck && NUMBA_NRT_STATS=1 venv/bin/pytest \
    test/test_ducklib.py::test_structref_meminfo_bridge_nested_heap -v
```
Expected: `1 passed`.

**Steps:**

- [ ] **Step 1: Add a list-bearing structref type**

```python
@structref.register
class _HeapStateType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)


class _HeapState(structref.StructRefProxy):
    def __new__(cls, values):
        return structref.StructRefProxy.__new__(cls, values)


structref.define_proxy(_HeapState, _HeapStateType, ["values"])

_heap_state_type = _HeapStateType(
    [("values", nb_types.ListType(nb_types.float64))])
```

- [ ] **Step 2: Write the test**

```python
def test_structref_meminfo_bridge_nested_heap():
    """Prove release_meminfo cascades dtor into heap-owning fields."""
    from numba.core.runtime.nrt import rtsys
    from numba.typed import List

    @njit
    def _alloc_and_export():
        lst = List.empty_list(nb_types.float64)
        for i in range(10):
            lst.append(float(i))
        s = _HeapState(lst)
        return export_meminfo(s)

    @njit
    def _release(p):
        release_meminfo(p)

    # Warmup: single cycle to measure per-cycle alloc delta
    stats_before_warmup = rtsys.get_allocation_stats()
    p = _alloc_and_export()
    _release(p)
    stats_after_warmup = rtsys.get_allocation_stats()

    warmup_alloc = stats_after_warmup.alloc - stats_before_warmup.alloc
    warmup_free = stats_after_warmup.free - stats_before_warmup.free
    assert warmup_alloc > 0, (
        f"warmup must allocate; got alloc delta {warmup_alloc}")
    assert warmup_alloc == warmup_free, (
        f"warmup leak: alloc={warmup_alloc}, free={warmup_free}")

    # 100 cycles: exact multiple of warmup
    N = 100
    stats_before = rtsys.get_allocation_stats()
    for _ in range(N):
        p = _alloc_and_export()
        _release(p)
    stats_after = rtsys.get_allocation_stats()

    alloc_delta = stats_after.alloc - stats_before.alloc
    free_delta = stats_after.free - stats_before.free
    assert alloc_delta == N * warmup_alloc, (
        f"alloc count: got {alloc_delta}, expected {N * warmup_alloc}")
    assert free_delta == N * warmup_free, (
        f"free count: got {free_delta}, expected {N * warmup_free}")
```

- [ ] **Step 3: Run and commit**

```bash
rm -rf ~/.cache/numba && find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
cd /home/erik/projects/numbduck && NUMBA_NRT_STATS=1 venv/bin/pytest \
    test/test_ducklib.py::test_structref_meminfo_bridge_nested_heap -v
```

```bash
git -c safe.directory=/home/erik/projects/numbduck add test/test_ducklib.py
git -c safe.directory=/home/erik/projects/numbduck commit -m "Add nested-heap dtor chain verification for structref bridge"
```

**If it fails:**
- If warmup_alloc == 0 → `get_allocation_stats` requires `NUMBA_NRT_STATS=1`; verify the env var is set when pytest runs. If needed, set via `os.environ["NUMBA_NRT_STATS"] = "1"` at top of test file (but before any numba import).
- If alloc != free → the typed.List dtor isn't running under raw `nrt.decref`. Investigate: compare with releasing via `borrow_structref` + scope exit instead (which uses the typed decref path). If the typed path frees but raw doesn't, change `release_meminfo` to accept the struct type and call `nrt.decref` with full type info.

---

## Task 4: DuckDB stddev UDAF integration test

**Goal:** End-to-end test — register `welford_stddev(DOUBLE) -> DOUBLE` with DuckDB using the structref bridge for all callbacks + destructor, run a SQL query, verify result, assert no leak after connection close.

**Depends on:** Task 1 and Task 2.

**Files:**
- Modify: `test/test_ducklib.py` — add five `@cfunc` wrappers and the integration test

**Acceptance Criteria:**
- [ ] `SELECT welford_stddev(v) FROM t` on values `[1..7]` returns `numpy.std(xs, ddof=1)` to 1e-10
- [ ] After `duckdb_destroy_aggregate_function` + connection close, NRT alloc/free deltas are balanced

**Verify:**
```bash
rm -rf ~/.cache/numba && find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
cd /home/erik/projects/numbduck && NUMBA_NRT_STATS=1 venv/bin/pytest \
    test/test_ducklib.py::test_aggregate_function_structref_stddev -v
```
Expected: `1 passed`.

**Steps:**

- [ ] **Step 1: Add the five callbacks + destructor**

```python
@njit
def _welford_state_size_impl(info):
    return numpy.uint64(8)  # sizeof(void*)


@cfunc(nb_types.uint64(nb_types.intp))
def _welford_state_size_cb(info):
    return _welford_state_size_impl(info)


@njit
def _welford_init_impl(info, state):
    s = WelfordState(0.0, 0, 0.0)
    p = export_meminfo(s)
    slot = carray(_cast_int_to_void_p(state), (1,), dtype=numpy.intp)
    slot[0] = p


@cfunc(nb_types.void(nb_types.intp, nb_types.intp))
def _welford_init_cb(info, state):
    _welford_init_impl(info, state)


@njit
def _welford_update_impl(info, chunk, states):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(vec)
    state_slots = carray(
        _cast_int_to_void_p(states), (n,), dtype=numpy.intp)
    in_vals = carray(
        _cast_int_to_void_p(in_data), (n,), dtype=numpy.float64)
    for i in range(n):
        slot = carray(
            _cast_int_to_void_p(state_slots[i]), (1,), dtype=numpy.intp)
        s = borrow_structref(welford_type, slot[0])
        welford_update(s, in_vals[i])


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _welford_update_cb(info, chunk, states):
    _welford_update_impl(info, chunk, states)


@njit
def _welford_combine_impl(info, source, target, count):
    src_slots = carray(
        _cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    tgt_slots = carray(
        _cast_int_to_void_p(target), (count,), dtype=numpy.intp)
    for i in range(count):
        src_slot = carray(
            _cast_int_to_void_p(src_slots[i]), (1,), dtype=numpy.intp)
        tgt_slot = carray(
            _cast_int_to_void_p(tgt_slots[i]), (1,), dtype=numpy.intp)
        src = borrow_structref(welford_type, src_slot[0])
        tgt = borrow_structref(welford_type, tgt_slot[0])
        welford_combine(src, tgt)


@cfunc(nb_types.void(nb_types.intp, nb_types.intp,
                     nb_types.intp, nb_types.uint64))
def _welford_combine_cb(info, source, target, count):
    _welford_combine_impl(info, source, target, count)


@njit
def _welford_finalize_impl(info, source, result, count, offset):
    out_data = ducklib.duckdb_vector_get_data(result)
    src_slots = carray(
        _cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    out_vals = carray(
        _cast_int_to_void_p(out_data), (offset + count,),
        dtype=numpy.float64)
    for i in range(count):
        src_slot = carray(
            _cast_int_to_void_p(src_slots[i]), (1,), dtype=numpy.intp)
        s = borrow_structref(welford_type, src_slot[0])
        out_vals[offset + i] = welford_finalize(s)


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp,
                     nb_types.uint64, nb_types.uint64))
def _welford_finalize_cb(info, source, result, count, offset):
    _welford_finalize_impl(info, source, result, count, offset)


@njit
def _welford_destroy_impl(states, count):
    state_slots = carray(
        _cast_int_to_void_p(states), (count,), dtype=numpy.intp)
    for i in range(count):
        slot = carray(
            _cast_int_to_void_p(state_slots[i]), (1,), dtype=numpy.intp)
        release_meminfo(slot[0])


@cfunc(nb_types.void(nb_types.intp, nb_types.uint64))
def _welford_destroy_cb(states, count):
    _welford_destroy_impl(states, count)
```

- [ ] **Step 2: Add the integration test**

Model after `test_aggregate_function_round_trip` but with DOUBLE instead of INTEGER and with the destructor registered.

```python
def test_aggregate_function_structref_stddev():
    """End-to-end: structref-backed Welford stddev UDAF in DuckDB."""
    from numba.core.runtime.nrt import rtsys

    duckdb_database, duckdb_connection = aux_connect_db()
    conn_p = duckdb_connection[0]

    # Create table with doubles
    result = create_duckdb_result()
    query_p = get_unicode_data_p(
        "CREATE TABLE t AS SELECT * FROM "
        "(VALUES (1.0),(2.0),(3.0),(4.0),(5.0),(6.0),(7.0)) AS t(v)")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess
    ducklib.duckdb_destroy_result(result.ctypes.data)

    # Register aggregate
    func_p = ducklib.duckdb_create_aggregate_function()
    assert func_p != 0

    name_p = get_unicode_data_p("welford_stddev")
    ducklib.duckdb_aggregate_function_set_name(func_p, name_p)

    dbl_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
    ducklib.duckdb_aggregate_function_add_parameter(func_p, dbl_type_p)
    ducklib.duckdb_aggregate_function_set_return_type(func_p, dbl_type_p)
    tp_buf = numpy.array([dbl_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(tp_buf.ctypes.data)

    ducklib.duckdb_aggregate_function_set_functions(
        func_p,
        _welford_state_size_cb.address,
        _welford_init_cb.address,
        _welford_update_cb.address,
        _welford_combine_cb.address,
        _welford_finalize_cb.address,
    )
    ducklib.duckdb_aggregate_function_set_destructor(
        func_p, _welford_destroy_cb.address)

    rc = ducklib.duckdb_register_aggregate_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess

    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_aggregate_function(func_buf.ctypes.data)

    # Run the query
    stats_before = rtsys.get_allocation_stats()

    result = create_duckdb_result()
    query_p = get_unicode_data_p("SELECT welford_stddev(v) FROM t")
    rc = ducklib.duckdb_query(conn_p, query_p, result.ctypes.data)
    assert rc == ducklib.DuckDBSuccess, f"Query failed, rc={rc}"

    chunk_p = ducklib.duckdb_fetch_chunk(tuple(result))
    vec_p = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
    data_p = ducklib.duckdb_vector_get_data(vec_p)
    val = (ctypes.c_double * 1).from_address(data_p)[0]

    xs = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    expected = float(numpy.std(xs, ddof=1))
    assert abs(val - expected) < 1e-10, (
        f"got {val}, expected {expected}")

    chunk_buf = numpy.array([chunk_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_data_chunk(chunk_buf.ctypes.data)
    ducklib.duckdb_destroy_result(result.ctypes.data)

    aux_close_db(duckdb_database, duckdb_connection)

    stats_after = rtsys.get_allocation_stats()
    alloc_delta = stats_after.alloc - stats_before.alloc
    free_delta = stats_after.free - stats_before.free
    assert alloc_delta == free_delta, (
        f"leak: alloc={alloc_delta}, free={free_delta}")
```

- [ ] **Step 3: Run and commit**

```bash
rm -rf ~/.cache/numba && find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
cd /home/erik/projects/numbduck && NUMBA_NRT_STATS=1 venv/bin/pytest \
    test/test_ducklib.py::test_aggregate_function_structref_stddev -v
```

```bash
git -c safe.directory=/home/erik/projects/numbduck add test/test_ducklib.py
git -c safe.directory=/home/erik/projects/numbduck commit -m "Add DuckDB structref-backed stddev UDAF integration test"
```

**If it fails:**
- Wrong query result → add logging in `_welford_update_impl` (printing `n` and first in_val) to verify the chunk ABI. Compare slot indirection with working `test_aggregate_function_round_trip`.
- Refcount epilogue fails (`alloc != free`) → the destructor isn't being called for every init. Check whether DuckDB's aggregate executor guarantees 1:1 init/destroy by reading `src/execution/aggregate_hashtable.cpp` in the DuckDB source. If not 1:1, this is a known limitation; document in spec's "Risks" section (already noted as a risk).
- Segfault in combine → DuckDB may not call combine at all for small inputs (only splits to multiple threads above some threshold). This isn't a failure; if combine isn't exercised, consider a larger input (e.g., 100_000 values generated numerically) in a second assertion to force threading.

---

## Full test suite verification

After all four tasks complete, run the entire test suite to confirm no regressions:

```bash
rm -rf ~/.cache/numba && find /home/erik/projects/numbduck -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
cd /home/erik/projects/numbduck && NUMBA_NRT_STATS=1 venv/bin/pytest test/ -v --durations=20
```

Expected: all prior tests still pass + 4 new tests pass.

Push and verify CI on the fork:
```bash
git -c safe.directory=/home/erik/projects/numbduck push -u origin feat/structref-udaf-stddev
```

Check fork CI matrix passes before considering the branch ready for PR.
