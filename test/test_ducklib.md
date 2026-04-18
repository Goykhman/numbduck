# Structref-backed UDAFs: bridging Numba NRT and DuckDB

This document explains how [`test_ducklib.py`](test_ducklib.py)
implements a DuckDB aggregate function (UDAF) whose per-group state is a
Numba [structref](https://numba.readthedocs.io/en/stable/developer/extending.html#primitives-and-automatic-reference-counting).
It covers the DuckDB aggregate lifecycle, the problem of passing
NRT-managed objects through a raw `void*` slot, and the
[`removerefctpass`](https://github.com/numba/numba/blob/0.61.2/numba/core/removerefctpass.py)
interaction that makes the bridge intrinsics work.


## Background: two runtimes, one pointer

**Numba's NRT** (Numba Runtime) manages structref lifetimes with automatic
reference counting.  When a structref is created, NRT allocates a `MemInfo`
header (refcount + destructor pointer + data pointer) and sets refcount to 1.
Scope-exit decrefs and destructor calls are inserted by the compiler.
See [numba/core/runtime/nrt.cpp](https://github.com/numba/numba/blob/0.61.2/numba/core/runtime/nrt.cpp).

**DuckDB's aggregate API** gives each aggregate group a fixed-size buffer
(`state_size` bytes) of opaque memory.  DuckDB manages that buffer's
lifetime, but knows nothing about what's inside it — it's just bytes.
See [duckdb.h — Aggregate Functions](https://duckdb.org/docs/stable/clients/c/api#aggregate-functions).

The challenge: a structref needs NRT to track its refcount, but DuckDB's
state slot is a raw `void*` invisible to NRT.  We need to transfer
ownership of the structref from NRT-managed local variables into that slot,
keep it alive across multiple C callbacks, and eventually release it.


## DuckDB aggregate lifecycle

DuckDB calls six callbacks during aggregate execution.  Each receives a raw
pointer to the state buffer.  Our UDAF sets `state_size = 8` so the buffer
holds exactly one `intp`-sized pointer:

```
┌──────────────┐
│  state_size  │──▶  DuckDB calls this first to learn how many bytes to
│   (8 bytes)  │     allocate per group.
└──────────────┘
       │
       ▼
┌──────────────┐     Allocate a WelfordState structref (refcount=1).
│     init     │──▶  Store the MemInfo pointer in the 8-byte state slot.
└──────────────┘     The slot now "owns" that reference.
       │
       ▼
┌──────────────┐     For each input row in this group:
│    update    │──▶  Read the MemInfo pointer from the slot, reconstruct
└──────────────┘     the structref temporarily, run welford_update(s, x).
       │
       ▼
┌──────────────┐     When DuckDB merges partial aggregates (e.g. parallel):
│   combine    │──▶  Reconstruct both source and target structrefs from
└──────────────┘     their state slots, run welford_combine(src, tgt).
       │
       ▼
┌──────────────┐     Produce the final result for each group:
│   finalize   │──▶  Reconstruct structref, compute stddev, write to
└──────────────┘     output vector.
       │
       ▼
┌──────────────┐     Drop the NRT reference.  If refcount reaches 0,
│   destroy    │──▶  NRT calls the structref's destructor (imp_dtor),
└──────────────┘     freeing nested allocations like typed.List.
```

Source: callbacks defined at
[`test_ducklib.py` L3075–3176](test_ducklib.py#L3075-L3176);
registered at
[`test_ducklib.py` L3280–3304](test_ducklib.py#L3280-L3304).


## The @cfunc + @njit callback pattern

DuckDB calls C function pointers.  Numba's `@cfunc` compiles a Python
function to a C-callable entry point, but `@cfunc` bodies can't use
`import` statements or complex numba features.  The pattern:

```python
@njit
def _welford_init_impl(info, state):
    # all the real work happens here, in full @njit
    s = WelfordState(0.0, 0, 0.0)
    p = export_meminfo(s)
    slot = carray(_cast_int_to_void_p(state), (1,), dtype=numpy.intp)
    slot[0] = p

@cfunc(nb_types.void(nb_types.intp, nb_types.intp))
def _welford_init_cb(info, state):
    # thin wrapper — just forwards to the @njit impl
    _welford_init_impl(info, state)
```

All pointer arguments use `nb_types.intp` (not `voidptr`) because
numbduck's C API bindings return `intp`.  Inside `@njit`, `carray()`
requires `voidptr`, so we bridge with numbox's
[`_cast_int_to_void_p`](https://github.com/Goykhman/numbox/blob/main/numbox/utils/lowlevel.py).

Source: [`test_ducklib.py` L3075–3176](test_ducklib.py#L3075-L3176).


## Bridge intrinsics

Three functions transfer ownership between NRT and the DuckDB state slot.
Defined at [`test_ducklib.py` L2796–2874](test_ducklib.py#L2796-L2874).

### `export_meminfo(s)` — init

Extracts the `MemInfo*` from a structref and returns it as an `intp`.
The codegen emits `context.nrt.incref` to account for the new external
reference.

**Used in:** `_welford_init_impl` — stores the returned pointer in the
DuckDB state slot.

### `borrow_structref(type, p)` — update / combine / finalize

Reconstructs a structref from a raw `MemInfo*` pointer.  The codegen emits
`context.nrt.incref` on entry; the compiler inserts a scope-exit decref
for the local variable.  Net effect: zero change to the external owner's
refcount.

**Used in:** `_welford_update_impl`, `_welford_combine_impl`,
`_welford_finalize_impl` — reads the pointer from the state slot,
operates on the structref, lets the local reference expire at scope exit.

### `release_meminfo(p)` — destroy

Decrefs the `MemInfo*` via a direct call to
[`NRT_MemInfo_release`](https://github.com/numba/numba/blob/0.61.2/numba/core/runtime/nrt.cpp#L220).
When refcount reaches 0, NRT invokes the structref's `imp_dtor`,
freeing nested allocations (e.g. `typed.List` contents).

**Used in:** `_welford_destroy_impl` — called once per group when DuckDB
discards aggregate state.


## Why `release_meminfo` is different from the other two

`release_meminfo` calls `NRT_MemInfo_release` instead of using
`context.nrt.decref()`.  This is not arbitrary — it's forced by
numba's [`removerefctpass`](https://github.com/numba/numba/blob/0.61.2/numba/core/removerefctpass.py).

### What `removerefctpass` does

After lowering a function to LLVM IR, numba checks whether the function's
argument types and return type involve any NRT-tracked types (structrefs,
arrays, etc.).  If they don't, it strips **all** `NRT_incref` and
`NRT_decref` calls from the function.  The rationale: if no NRT-managed
objects enter or leave the function, any refcount operations inside it are
redundant.

The check is implemented in
[`_legalize()`](https://github.com/numba/numba/blob/0.61.2/numba/core/removerefctpass.py#L37-L94),
and the stripping in
[`_rewrite_function()`](https://github.com/numba/numba/blob/0.61.2/numba/core/removerefctpass.py#L24-L31).
Only functions named `NRT_incref` or `NRT_decref` are stripped
([`_accepted_nrtfns`](https://github.com/numba/numba/blob/0.61.2/numba/core/removerefctpass.py#L34)).

### How it affects the callbacks

All our UDAF callbacks have signatures like `void(intp, intp)` — no
NRT-tracked types.  So `removerefctpass` strips every `NRT_incref` and
`NRT_decref` in them:

| Callback | Emitted NRT ops | After `removerefctpass` | Net refcount |
|----------|----------------|------------------------|-------------|
| `init` | incref (export) + decref (scope exit of local `s`) | both stripped | 0 (object keeps refcount=1 from allocation) |
| `update` | incref (borrow) + decref (scope exit of local `s`) | both stripped | 0 (slot's reference undisturbed) |
| `destroy` | `NRT_MemInfo_release` | **survives** | −1 (frees the object) |

The stripping is **symmetric** — for every incref that gets removed, the
corresponding scope-exit decref is also removed.  The net refcount change
at each stage is exactly what we need.

### Why `NRT_MemInfo_release` survives

`_legalize()` scans all functions in the LLVM module.  If it finds any
function whose name starts with `NRT_` but is NOT in `_accepted_nrtfns`
(`NRT_incref`, `NRT_decref`), it returns `False` and the pass is skipped
entirely for that function.  `NRT_MemInfo_release` triggers this bail-out,
protecting the entire `_welford_destroy_impl` function from the rewrite.

This is why `release_meminfo` must use a direct call to `NRT_MemInfo_release`
rather than `context.nrt.decref()` — the latter emits `NRT_decref`, which
would be stripped.

### Why we don't use direct calls for incref too

If `export_meminfo` used a direct `NRT_MemInfo_acquire` call (analogous to
`NRT_MemInfo_release`), that call's name would disable `removerefctpass`
for the entire function.  Then the scope-exit decref of the local structref
would **also** survive:

```
allocation:     +1  (refcount = 1)
our incref:     +1  (refcount = 2)  ← now survives
scope-exit:     −1  (refcount = 1)  ← also survives
                                     init returns with refcount = 1
destroy:        −1  (refcount = 0)  ← freed
```

In this specific case the math still works out — but only because there's
exactly one scope-exit decref.  If the function body were more complex
(multiple local refs, temporary borrows), extra decrefs would survive too,
potentially freeing the object prematurely.  The current approach — letting
the pass strip everything symmetrically — is simpler and correct regardless
of function complexity.

### Stability across numba versions

`removerefctpass` has used the same `_accepted_nrtfns` set and `_legalize`
logic since
[numba 0.43 (2019)](https://github.com/numba/numba/blob/0.43.0/numba/core/removerefctpass.py).
The symmetric stripping is a structural property of the pass, not an
implementation accident.  If a future numba version changes the pass to
strip increfs but not decrefs (or vice versa), both the current approach
and a direct-call approach would break — the whole structref-via-raw-pointer
pattern depends on the pass being all-or-nothing.


## Registration: telling DuckDB about the UDAF

The aggregate is registered from Python (not JIT) code.  DuckDB receives
raw function pointers and calls them at the right lifecycle points:

```python
func_p = ducklib.duckdb_create_aggregate_function()
ducklib.duckdb_aggregate_function_set_name(func_p, name_p)
ducklib.duckdb_aggregate_function_add_parameter(func_p, dbl_type_p)
ducklib.duckdb_aggregate_function_set_return_type(func_p, dbl_type_p)

ducklib.duckdb_aggregate_function_set_functions(
    func_p,
    _welford_state_size_cb.address,  # how many bytes per group?
    _welford_init_cb.address,        # allocate structref
    _welford_update_cb.address,      # process input rows
    _welford_combine_cb.address,     # merge parallel partials
    _welford_finalize_cb.address,    # produce output
)
ducklib.duckdb_aggregate_function_set_destructor(
    func_p, _welford_destroy_cb.address)  # free structref

ducklib.duckdb_register_aggregate_function(conn_p, func_p)
```

After registration, `SELECT welford_stddev(v) FROM t` triggers the full
callback lifecycle above.  DuckDB allocates 8 bytes per group, calls our
callbacks with pointers to those bytes, and calls destroy when done.

Source: [`test_ducklib.py` L3280–3304](test_ducklib.py#L3280-L3304).


## The `refcount_of_meminfo` cautionary tale

The commented-out intrinsic at
[`test_ducklib.py` L2877–2914](test_ducklib.py#L2877-L2914)
attempted to read the NRT refcount from inside JIT code, expecting to
observe refcount=2 while both a local structref and an exported MemInfo
pointer were live.  It always read 1, because `removerefctpass` had already
stripped the incref that would have bumped it to 2.  The intrinsic is
preserved as an example of why in-scope refcount observation doesn't work
when the function signature has no NRT-tracked types.

Python-side `_read_refcount()` between separate `@njit` calls is the only
reliable way to verify refcounts.  That's what
[`test_structref_meminfo_bridge_refcount_ladder`](test_ducklib.py#L3179)
does.
