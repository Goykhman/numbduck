# Structref state in a DuckDB aggregate

The aggregates in [test_ducklib.py](test_ducklib.py) keep each group's state in a numba [structref][structref], a
mutable struct whose memory numba allocates and reference-counts. This note explains how a structref survives from one
of DuckDB's callbacks to the next, and why its reference count stays balanced on every numba version that numbduck
supports. It ends with a checklist of the parts that a new aggregate needs and that a change to the existing ones must
keep. The README's section on [aggregate functions](../README.md#aggregate-functions) has a complete, shorter example,
and [examples/irr.py](../examples/irr.py) a longer one.

[structref]: https://numba.readthedocs.io/en/stable/extending/high-level.html#defining-a-structref

## Two owners of one object

NRT, numba's runtime, allocates a structref's data in memory headed by a MemInfo: a reference count, a destructor, and
the address of the data ([numba's `MemInfo` struct][nrt-meminfo]). numba increments the count when it copies a
reference to the structref, and decrements it when a reference is no longer used. When the count reaches zero, NRT
runs the destructor, which releases what the struct's fields hold, such as a typed list, and frees the memory.

DuckDB gives each group of an aggregate a state: `state_size` bytes that DuckDB allocates and frees, and whose content
it never reads. The aggregate stores the MemInfo's address in those bytes. The state thereby holds one reference of
its own, which keeps the structref alive from `init` to `destroy`, across any number of callbacks in between. numba
cannot see that reference, so the code takes and drops it explicitly. It does so through the helpers of
[numbox][numbox], the toolbox of numba utilities that numbduck depends on.

[nrt-meminfo]: https://github.com/numba/numba/blob/0.65.1/numba/core/runtime/nrt.cpp#L17
[numbox]: https://github.com/Goykhman/numbox

## The callbacks

DuckDB runs an aggregate as six callbacks. `duckdb_aggregate_function_set_functions` registers the first five, and
[`duckdb_aggregate_function_set_destructor`][set-destructor] registers `destroy`. The Welford aggregate of the tests,
`welford_stddev`, computes a sample standard deviation:

| Callback | `@cfunc` signature | What the Welford aggregate does |
|---|---|---|
| `state_size` | `uint64(intp)` | returns 8, the size of one address |
| `init` | `void(intp, intp)` | creates a `WelfordState` and stores its MemInfo's address in the state |
| `update` | `void(intp, intp, intp)` | borrows the structref of each input row's group and adds the row's value |
| `combine` | `void(intp, intp, intp, uint64)` | borrows the source and target structrefs of each pair and merges them |
| `finalize` | `void(intp, intp, intp, uint64, uint64)` | borrows each structref and writes its result at `offset + i` |
| `destroy` | `void(intp, uint64)` | releases the reference that each state holds |

DuckDB's typedefs in `duckdb.h` give the C signatures, and each pointer becomes an `intp`, the type that numbduck's
bindings use for a pointer. The callbacks after `init` receive arrays of state addresses. `update` receives one address
for each input row, the state of the row's group, and `combine`, `finalize` and `destroy` receive one for each group.
Each state holds a MemInfo's address. numba's `carray` views the memory at a pointer as a numpy array, and numbox's
`_cast_int_to_void_p` turns each `intp` address into the pointer that `carray` takes.

An aggregate whose state is a few numbers needs no structref, because the numbers fit in the state's bytes. The tests'
`my_sum` aggregate keeps an `int64` sum there, and its `init` writes 0.

[set-destructor]: https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_aggregate_function_set_destructor

## The bridge

Three helpers of [`numbox.utils.meminfo`][numbox-meminfo] move the reference between numba and the state:

| Helper | Called in | Effect on the reference count |
|---|---|---|
| `export_meminfo(s)` | `init` | +1, a reference that the state holds; returns the MemInfo's address |
| `borrow_structref(struct_type, address)` | `update`, `combine`, `finalize` | +1, and -1 when it is no longer used |
| `release_meminfo(address)` | `destroy` | -1; at zero, NRT frees the struct and what its fields hold |

In `init`, a new structref starts with one reference, the one that its local variable holds. `export_meminfo` adds
the state's reference, and numba drops the local one when `init` returns, which leaves a count of 1. Each borrow adds
a reference and drops it again, so the count is 1 between callbacks, and `destroy` brings it to 0.

The helpers check only that an address is an `intp`. `borrow_structref` does not check that its `struct_type` is the
type of the exported structref. Each address must be released exactly once, and after that one release, NRT may free
the memory at it.

The block below applies the three steps to one structref and reads its count from Python between them:

```python
import ctypes

from numba import njit
from numba import types as nb_types
from numba.experimental import structref
from numbox.utils.meminfo import borrow_structref, export_meminfo, release_meminfo


@structref.register
class PointType(nb_types.StructRef):
    pass


class Point(structref.StructRefProxy):
    def __new__(cls, x, y):
        return structref.StructRefProxy.__new__(cls, x, y)


structref.define_proxy(Point, PointType, ["x", "y"])
point_type = PointType([("x", nb_types.float64), ("y", nb_types.float64)])


def refcount(address):
    """Read the reference count, the first field of numba's MemInfo, at the address."""
    return ctypes.c_size_t.from_address(address).value


@njit
def init():
    return export_meminfo(Point(1.0, 2.0))


@njit
def update(address):
    point = borrow_structref(point_type, address)
    point.x += 1.0
    return point.x


@njit
def destroy(address):
    release_meminfo(address)


address = init()
print(refcount(address))  # 1: the reference that export_meminfo took
print(update(address), refcount(address))  # 2.0 1: the borrowed reference is gone again
destroy(address)  # the count reaches 0, and NRT frees the struct
```

Three tests check the rest:

- `test_structref_meminfo_bridge_refcount_ladder` asserts the same counts for a `WelfordState`;
- `test_structref_meminfo_bridge_nested_heap` checks that `release_meminfo` also frees a typed list that a field holds;
- `test_aggregate_function_structref_stddev` runs the aggregate in a query, and checks that NRT freed as many memory
  blocks as it allocated.

[numbox-meminfo]: https://goykhman.github.io/numbox/numbox.utils.html#numbox.utils.meminfo.export_meminfo

## The `@cfunc` and `@njit` split

DuckDB calls a C function, and numba's `@cfunc` compiles one with a fixed signature. The tests write the work of each
callback as an `@njit` function, such as `_welford_update_impl`, and wrap it in a `@cfunc` of one line, such as
`_welford_update_cb`. Python code can then call the `@njit` body: `test_welford_numba_only` checks `welford_update`,
`welford_combine` and `welford_finalize` against numpy without DuckDB. The split is a convention, not a requirement. A
`@cfunc` can hold the body itself and call numbduck's bindings, `carray` and numbox's helpers, as the README's example
does.

## numba's removerefctpass

numba 0.65 and earlier run a pass, `removerefctpass`, on each function as numba lowers it to LLVM. The pass deletes
every call to `NRT_incref` and `NRT_decref` in a function that seems not to need reference counting. If the pass deleted
such a call in the bridge, a structref would leak, or NRT would free it while a state still held its address. numba's
[`_legalize`][legalize] lets the pass delete the calls of a function only if all of these hold:

1. the function's module has no function marked with the metadata `numba_args_may_always_need_nrt`;
2. no argument holds a MemInfo, unless the argument is an array;
3. its return type holds no MemInfo;
4. no function that it calls returns a type that holds a MemInfo;
5. its module declares no function named `NRT_...` other than `NRT_incref` and `NRT_decref`.

numba 0.66 removed the pass ([numba PR 10511][pr-10511]), so numba 0.66 to 0.68 run none. On numba 0.63.1, 0.64.0 and
0.65.1, every function here that changes a count fails at least one condition:

| Function | The conditions it fails |
|---|---|
| `export_meminfo`, called with a structref | its argument is a structref (2), and its module carries the marker (1) |
| `export_meminfo`, called with an array | numbox's marker (1), and nothing else |
| `borrow_structref` | it and a function it calls return a structref (3, 4); its module has numbox's marker (1) |
| `release_meminfo` | its module declares `NRT_MemInfo_release` (5) |
| the callbacks' `@njit` bodies, such as `_welford_update_impl` | each calls `carray`, which returns an array (4) |

The `@cfunc` wrappers meet all five conditions but hold no call to delete, so the counts stay as
[The bridge](#the-bridge) describes.

Two of those failures are deliberate, and numbox's intrinsics cause them:

- `release_meminfo` fails condition 5, because numbox's `_release_meminfo` intrinsic decrements the count with
  `NRT_MemInfo_release`, not `NRT_decref`. With an `NRT_decref`, `release_meminfo`, which takes an `intp` and returns
  nothing, would meet every condition, and the pass would delete the decrement.
- `export_meminfo` called with an array fails condition 1, because numbox's `_incref_meminfo` intrinsic marks the
  function that calls it with `numba_args_may_always_need_nrt` ([meminfo.py at numbox 0.7.9][incref-marker]). NRT has no
  function for an increment other than `NRT_incref`, whose calls the pass deletes, so only the marker can keep the
  increment. Without it, the pass would delete the increment, and NRT would free the array while a state still held its
  address. numbox 0.7.6 added the marker, and numbduck requires numbox 0.7.8 or later.

A change to the bridge, or a new aggregate, must therefore make every reference count change through
`export_meminfo`, `borrow_structref` and `release_meminfo`. On numba 0.65 and earlier, the pass deletes a call to
`NRT_incref` or `NRT_decref` that an intrinsic of your own emits, in any function that meets the five conditions.

[incref-marker]: https://github.com/Goykhman/numbox/blob/0.7.9/numbox/utils/meminfo.py#L88-L117
[legalize]: https://github.com/numba/numba/blob/0.65.1/numba/core/removerefctpass.py#L37-L96
[pr-10511]: https://github.com/numba/numba/pull/10511

## Reading a reference count

Read a count from Python, between separate `@njit` calls, as the block above and
`test_structref_meminfo_bridge_refcount_ladder` do. The count is the first field of the MemInfo, a `size_t` at the
address that `export_meminfo` returns.

A read inside the jitted function that exported the structref does not give the count that the code suggests.
Immediately after `export_meminfo(s)`, while `s` is still in use, such a read gives 1, not 2. numba's LLVM pass for
reference-count pruning, a pass other than `removerefctpass`, deletes the export's increment together with a later
decrement of the same structref, and the count after the function returns is the same as without the pass.
[`NUMBA_LLVM_REFPRUNE_PASS=0`][refprune] turns that pass off, and on numba 0.65.1 the same read then gives 2. A
[question on numba's forum][forum] describes the same effect.

[forum]: https://numba.discourse.group/t/re-assigning-structrefs-member-doesnt-decrease-ref-count/3018/6
[refprune]: https://numba.readthedocs.io/en/stable/reference/envvars.html#envvar-NUMBA_LLVM_REFPRUNE_PASS

## An array as the state

`test_aggregate_function_array_variance` keeps a numpy array in the state instead of a structref. Its `state_size` is 16
bytes, for two addresses. `destroy` releases the first, the array's MemInfo address, and the other callbacks read and
write through the second, the address of the array's data. `_arr_init_impl` creates the array in jitted code. It reads
both addresses with numbox's `structref_meminfo`, which returns them for a structref or an array without taking a
reference, and it takes the state's reference with numbox's `_incref_meminfo`. `update`, `combine` and `finalize` read
and write the data through `carray` on the data address, so they need no borrow. `destroy` passes the MemInfo address to
`release_meminfo`.

`test_array_meminfo_bridge_refcount_ladder` checks the counts of this form, with `_incref_meminfo` called directly.
`export_meminfo(arr)` takes the same reference, because numbox's marker keeps `removerefctpass` from deleting it
([numba's removerefctpass](#numbas-removerefctpass)).

## Exceptions in a callback

An exception that escapes a `@cfunc` does not reach DuckDB: numba prints it, and the `@cfunc` returns zero, or nothing
if it returns `void`. If the exception comes from a nested call while the callback holds a borrowed structref, numba
also skips the release of that borrow, and the structref leaks. `test_udaf_cfunc_unguarded_raise_leaks_borrowed_state`
demonstrates the leak. A `try` block with an `except Exception` clause, in the function that holds the borrow, prevents
it. The exception then stops in that function's frame, and numba releases the borrow.
`test_udaf_cfunc_guarded_raise_no_leak` checks this with `_welford_update_raise_guarded_impl`, whose `try` block
encloses the borrow and the work. examples/irr.py guards the work of `update`, `combine` and `finalize` with such
blocks. Its `init` stores 0 in the state if the structref cannot be created, and its other callbacks skip a state that
holds 0.

## A new aggregate

An aggregate with a structref state needs these parts:

- a `state_size` that returns 8, the size of a MemInfo address, or 16 for an array and the address of its data;
- an `init` that creates the structref and stores the address that numbox's `export_meminfo` returns in the state;
- an `update`, a `combine` and a `finalize` that get each group's structref from numbox's `borrow_structref`;
- in `update`, a test of each input row with `duckdb_vector_get_validity` and `duckdb_validity_row_is_valid` that
  skips the NULL rows, which DuckDB passes to `update` with or without special handling;
- in each callback that borrows, a `try` block with an `except Exception` clause that encloses the borrow and the work;
- a `destroy` that calls numbox's `release_meminfo` for each state, registered with
  `duckdb_aggregate_function_set_destructor`, because every group's structref leaks without it;
- a structref whose proxy class numbox's `make_structref` generates, defined with its numba type in a module that the
  program imports, not in the script that runs. numba caches the code that `make_structref` generates under a key
  that does not include the importing module, whether or not any function sets `cache=True`. A type defined in the
  running script therefore does not match the cached one in the next run.

Its tests need two checks. One reads the count from Python after each step, as
`test_structref_meminfo_bridge_refcount_ladder` does. The other runs a query and checks that NRT freed as many blocks
as it allocated, as `test_aggregate_function_structref_stddev` does with `rtsys.get_allocation_stats()` from
`numba.core.runtime.nrt`. Run them on numba 0.63.1 to 0.65.x as well as on 0.66 or later, because only the earlier
versions run `removerefctpass`.
