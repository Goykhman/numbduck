# Structref-backed UDAF — design (2026-04-15)

## Goal

Add a unit test demonstrating that a DuckDB User-Defined Aggregate Function can
be backed by a numba `structref` whose lifetime is NRT-managed, using DuckDB's
aggregate-state slot as a single `void*` handoff. The test subject is **sample
standard deviation** via Welford's algorithm (3-field state: `mean`, `count`,
`m2`). Secondary goal: produce a small, carefully-verified bridge
(`export_meminfo` / `borrow_structref` / `release_meminfo`) that could later be
promoted from test-local helpers to a public numbduck API for structref-backed
aggregates.

Non-goals: window aggregates (`value`/`inverse`), variable-length logical
types, scalar-UDF integration, production-quality error handling on DuckDB's
side.

## Architecture

### The indirect-state ABI mapping

DuckDB's aggregate state is a `_duckdb_aggregate_state` struct containing a
single `void *internal_ptr` field
([duckdb.h@v1.3.2 L599](https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h#L599)).
DuckDB allocates `state_size()` bytes per aggregation group; we set
`state_size()` to `sizeof(void*) == 8` so the slot holds exactly one pointer.

- `init(info, state)` allocates a numba-managed structref, extracts its
  MemInfo pointer as an `intp`, and stores it into `state->internal_ptr`.
  The slot takes ownership of one NRT reference.
- `update`/`combine`/`finalize` reconstruct a structref value from the stashed
  pointer as a *borrow* (transient incref balanced by scope-exit decref), call
  the relevant Welford operation, and return. The slot's owning reference is
  undisturbed.
- `destroy(states, count)` drops the slot's reference on each state, which
  triggers the structref's `imp_dtor` when refcount reaches zero.

### The bridge intrinsics

All are module-level helpers in `test/test_ducklib.py` (promotion to
`numbduck/structref_bridge.py` deferred until we have a second use case).
Four functions: the three listed below plus `refcount_of_meminfo_intp(p)`
used by tests (see Test 1).

```
export_meminfo(s: <StructRef>) -> intp
    # Increfs s's MemInfo, returns the MemInfo pointer as intp.
    # Net effect: caller now owns two references — the local structref `s`
    # (which will decref on scope exit) and the returned intp (which is
    # numba-invisible and will NOT decref automatically).
    # After scope exit, the intp represents the single surviving reference.

borrow_structref(struct_type, p: intp) -> <StructRef>
    # Reconstructs a structref value whose .meminfo is the given MemInfo
    # pointer. Increfs on entry so that the local variable's automatic
    # decref-on-exit is balanced. The original reference held by whoever
    # owns `p` is undisturbed.
    # Per numba discourse thread
    # https://numba.discourse.group/t/any-numba-equivalent-for-casting-a-raw-pointer-to-a-structref-dict-list-etc/351/5

release_meminfo(p: intp) -> None
    # Directly releases one NRT reference on the MemInfo at `p`.
    # If refcount reaches zero, the dtor registered at allocation time
    # (imp_dtor) runs, cascading decref into any nested heap-owning fields.
```

### Welford state and operations

```python
@structref.register
class WelfordStateType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, types.unliteral(t)) for n, t in fields)

class WelfordState(structref.StructRefProxy):
    def __new__(cls, mean, count, m2):
        return structref.StructRefProxy.__new__(cls, mean, count, m2)

structref.define_proxy(WelfordState, WelfordStateType,
                      ["mean", "count", "m2"])
welford_type = WelfordStateType(
    [("mean", float64), ("count", int64), ("m2", float64)])

@njit
def welford_update(s, x):
    s.count += 1
    delta = x - s.mean
    s.mean += delta / s.count
    delta2 = x - s.mean
    s.m2 += delta * delta2

@njit
def welford_combine(src, tgt):
    # Merges `src` into `tgt` in place. Chan et al. pairwise formula.
    if src.count == 0: return
    if tgt.count == 0:
        tgt.mean, tgt.count, tgt.m2 = src.mean, src.count, src.m2
        return
    new_count = src.count + tgt.count
    delta = src.mean - tgt.mean
    tgt.mean = tgt.mean + delta * src.count / new_count
    tgt.m2 = tgt.m2 + src.m2 + delta * delta * tgt.count * src.count / new_count
    tgt.count = new_count

@njit
def welford_finalize(s) -> float64:
    if s.count < 2:
        return math.nan
    return math.sqrt(s.m2 / (s.count - 1))
```

## Tests

### 1. `test_structref_meminfo_bridge_refcount_ladder`

Purpose: prove that a single `export → borrow → release` cycle maintains
refcount invariants at every step.

All refcount reads use `numbox.utils.meminfo.get_nrt_refcount(meminfo_p)`.
The test body is a single `@njit` function so references are numba-managed;
assertions use `assert` statements inside njit (which raise `AssertionError`
visible to pytest).

```
Step                                          Expected refcount
----                                          -----------------
s = WelfordState(0.0, 0, 0.0)                 1
p = export_meminfo(s)                         2
# s still in scope
<decrement by letting s go out of scope>      1
s2 = borrow_structref(welford_type, p)        2
<use s2, then let it go out of scope>         1
release_meminfo(p)                            0 (freed; dtor ran)
```

Because numba scoping inside a single function can be subtle (liveness
analysis may extend or truncate lifetimes), we write the test as a
sequence of small njit functions that each return an `intp` representing
the slot-owned reference, isolating local scopes explicitly:

```python
@njit
def _step_allocate_and_export():
    s = WelfordState(0.0, 0, 0.0)
    p = export_meminfo(s)
    # after return, s is out of scope; p survives as the single reference
    return p

@njit
def _step_borrow_and_verify(p, expected_count):
    s = borrow_structref(welford_type, p)
    assert s.count == expected_count
    rc = refcount_of_meminfo_intp(p)  # expected: 2 (slot + borrow)
    # s goes out of scope on return; borrow was balanced
    return rc

@njit
def _step_check_refcount(p):
    return refcount_of_meminfo_intp(p)  # expected: 1 after borrow returns

@njit
def _step_release(p):
    release_meminfo(p)
```

`refcount_of_meminfo_intp` is a small `@njit` helper: cast `intp` →
`voidptr` → `MemInfoPointer(voidptr)`, then read the refcount field.
Written as a fourth bridge intrinsic (promoted from numbox's
`get_nrt_refcount` which takes a MemInfo value directly; we need the
`intp`-taking variant because DuckDB gives us raw pointers).

### 2. `test_structref_meminfo_bridge_nested_heap`

Purpose: prove the full dtor chain — outer structref and any heap-owning
field inside it are both freed on `release_meminfo`.

State type: `{values: ListType(float64)}`. Allocate 100 instances, append
10 floats to each list, export all pointers into a numpy array, then
`release_meminfo` each one. Use
`numba.core.runtime.nrt.rtsys.get_allocation_stats()` to assert:

- `alloc` count after == `alloc` count before + (100 outer + 100 list) = before + 200 (approximately — exact count depends on internal typed-List buffers; we'll measure baseline first and assert equality against the measured value, not a hardcoded number).
- `free` count after matches `alloc` count from this test.
- `mi_alloc` - `mi_free` delta is zero.

To avoid brittleness across numba versions, the test's first phase runs
one alloc/release cycle as a warmup and records the per-cycle delta;
phase two runs 100 cycles and asserts the delta is exactly 100× the
warmup.

Skip this test on numba < 0.60 (already asserted in numbox) and when
`NUMBA_DISABLE_JIT=1`.

### 3. `test_aggregate_function_structref_stddev`

Purpose: end-to-end DuckDB test.

- Create table `t(v DOUBLE)` with values `[1.0, 2.0, ..., 7.0]`.
- Register `welford_stddev(v DOUBLE) -> DOUBLE` UDAF using the structref
  bridge for all five callbacks + destructor.
- Query `SELECT welford_stddev(v) FROM t`.
- Assert result ≈ `numpy.std([1..7], ddof=1)` to 1e-10.
- Refcount epilogue: before closing the connection, snapshot
  `rtsys.get_allocation_stats()`; after closing + destroying aggregate
  function, snapshot again; assert net alloc/free delta is zero.

Callback shapes (all `@cfunc` wrapping `@njit` impls to match existing
numbduck patterns):

```python
@njit
def _welford_state_size_impl(info):
    return 8  # sizeof(void*)

@njit
def _welford_init_impl(info, state):
    s = WelfordState(0.0, 0, 0.0)
    p = export_meminfo(s)
    # state is a duckdb_aggregate_state* → void**; write p into *state.
    slot = carray(_cast_int_to_void_p(state), (1,), dtype=numpy.intp)
    slot[0] = p

@njit
def _welford_update_impl(info, chunk, states):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    vec = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    in_data = ducklib.duckdb_vector_get_data(vec)
    state_slots = carray(_cast_int_to_void_p(states), (n,), dtype=numpy.intp)
    in_vals = carray(_cast_int_to_void_p(in_data), (n,), dtype=numpy.float64)
    for i in range(n):
        slot = carray(
            _cast_int_to_void_p(state_slots[i]), (1,), dtype=numpy.intp)
        s = borrow_structref(welford_type, slot[0])
        welford_update(s, in_vals[i])

@njit
def _welford_combine_impl(info, source, target, count):
    src_slots = carray(_cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    tgt_slots = carray(_cast_int_to_void_p(target), (count,), dtype=numpy.intp)
    for i in range(count):
        src_slot = carray(_cast_int_to_void_p(src_slots[i]), (1,), dtype=numpy.intp)
        tgt_slot = carray(_cast_int_to_void_p(tgt_slots[i]), (1,), dtype=numpy.intp)
        src = borrow_structref(welford_type, src_slot[0])
        tgt = borrow_structref(welford_type, tgt_slot[0])
        welford_combine(src, tgt)

@njit
def _welford_finalize_impl(info, source, result, count, offset):
    out_data = ducklib.duckdb_vector_get_data(result)
    src_slots = carray(_cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    out_vals = carray(
        _cast_int_to_void_p(out_data), (offset + count,), dtype=numpy.float64)
    for i in range(count):
        src_slot = carray(_cast_int_to_void_p(src_slots[i]), (1,), dtype=numpy.intp)
        s = borrow_structref(welford_type, src_slot[0])
        out_vals[offset + i] = welford_finalize(s)

@njit
def _welford_destroy_impl(states, count):
    slots = carray(_cast_int_to_void_p(states), (count,), dtype=numpy.intp)
    for i in range(count):
        slot = carray(_cast_int_to_void_p(slots[i]), (1,), dtype=numpy.intp)
        release_meminfo(slot[0])
```

## Risks and mitigations

| Risk | Detection | Mitigation |
|------|-----------|------------|
| Refcount ladder wrong | Test 1 fails | Fix intrinsic; do not proceed to test 4. |
| Dtor doesn't cascade into nested heap fields | Test 3 alloc/free delta nonzero | Investigate numba's dtor chain; may need to define a custom dtor; if unsolvable, document limitation — Welford test still works but future `list_agg`-style UDAFs blocked. |
| DuckDB calls `init` without matching `destroy` | Test 4 leak epilogue fails | Read DuckDB's aggregate executor source; if 1:1 guarantee is absent, document failure mode and gate a production API on DuckDB fix. |
| DuckDB state slot ABI differs from my reading (e.g., `state` arg is already `void**` not `void**[]`) | Query returns wrong answer or segfaults | Start with the simplest possible case; if broken, instrument with printf-style logging from the `@cfunc` and re-derive the pointer indirection by inspection. |
| `imp_dtor` doesn't run on raw-MemInfo decref | Test 3 fails | Use `nrt.decref` with full type info instead of raw `nrt.release`; requires the release intrinsic to take the structref type as a second argument. |
| numba version drift | Test 1 or 3 fails after a numba upgrade | Isolated tests catch this before any DuckDB breakage; re-derive intrinsics if internals change. |

## Out of scope (documented for future work)

- Promoting the bridge helpers to `numbduck/structref_bridge.py` — wait for a
  second use case.
- Supporting `init`-without-`destroy` safely (e.g., via a ref-tracking
  side-table indexed by state-slot address).
- `list_agg`-style aggregates with growing internal state.
- Window aggregates (`value` + `inverse`).
- Parallel-safe combine semantics under DuckDB's partitioned aggregation.

## Testing strategy

Tests are added to `test/test_ducklib.py` alongside existing aggregate tests.
Run order matters only for test 1 (must pass before considering 4). In CI,
all four tests run in whatever order pytest picks; tests 1–3 don't depend on
each other. Set `NUMBA_NRT_STATS=1` at the top of the test module via an
environment import guard so the allocation counters are available.

Clean numba cache and `__pycache__` before pytest per project convention
(MEMORY: "Clean cache before tests").
