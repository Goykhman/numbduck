# Vector Container — Design Spec

## Overview

A generic, growable, contiguous-storage container for use inside numba `@njit` code. Inspired by C++ `std::vector`: backed by a numpy array with geometric doubling on overflow. Lives in `numbduck/vector.py` as a reusable utility; first consumer is the IRR UDAF example (`examples/irr.py`), replacing `numba.typed.List`.

## Factory

```python
from numbduck.vector import make_vector

Float64Vector, float64_vec_type = make_vector(nb_types.float64)
```

`make_vector(elem_type)` returns a `(create, type_instance)` pair:

- **`create`** — an `@njit` factory function that takes a single `capacity` argument and returns a freshly-allocated vector of that element type. The factory embeds the correct numpy dtype via closure, so callers don't pass dtypes at call time. Usage: `v = Float64Vector(8)` — allocate `buf=numpy.empty(8, dtype=float64)` and `size=0`.
- **`type_instance`** — the concrete `VectorType` instance with resolved field types, used as a field type in other structrefs and as the first arg to `borrow_structref`.

The factory shape (not a `StructRefProxy` subclass) means `Float64Vector` is callable from `@njit` code directly and the dtype is locked by the type — callers can't pass a mismatched buffer dtype.

Internally, `make_vector` does:

1. Create a concrete subclass of `VectorType` (the shared base) via `type()`.
2. Register it with `structref.register`.
3. Call numbox's `make_structref` to produce the underlying proxy class (used only by `create`).
4. Build the `@njit` factory `create(capacity)` that allocates the buffer with the correct dtype and invokes the proxy constructor.
5. Instantiate the type class with resolved fields to produce the type instance.

Multiple calls with the same `elem_type` return the cached `(create, type_instance)` tuple (cache key: `elem_type.key`).

## Structref Fields

| Field  | Type                        | Meaning                          |
|--------|-----------------------------|----------------------------------|
| `buf`  | `Array(elem_type, 1, 'C')` | Contiguous storage               |
| `size` | `int64`                     | Number of elements in use        |

Capacity is `buf.shape[0]`. No separate capacity field — it's always derivable.

## Construction

Inside `@njit`:

```python
v = Float64Vector(8)
```

Single `capacity` argument — the dtype is locked by the type. Callers choose the initial capacity (convention: 8).

## API

All operations are generic across vector types via numba's per-type specialization.

### `len(v)` — element count

Registered via `@overload(len)` on the `VectorType` base. Returns `v.size`.

### `v[i]` — element access

Registered via `@overload(operator.getitem)` on the `VectorType` base. Returns `v.buf[i]`. No bounds checking (matches numpy behavior in `@njit`).

### `v[i] = x` — element mutation

Registered via `@overload(operator.setitem)` on the `VectorType` base. Sets `v.buf[i] = x`. No bounds checking.

### `vector_push(v, val)` — append one element

Free `@njit` function. If `v.size == v.buf.shape[0]`, doubles the buffer (allocate new array, copy, reassign `v.buf`). Then sets `v.buf[v.size] = val` and increments `v.size`.

```python
@njit
def vector_push(v, val):
    if v.size == v.buf.shape[0]:
        new_buf = numpy.empty(v.buf.shape[0] * 2, v.buf.dtype)
        new_buf[:v.size] = v.buf[:v.size]
        v.buf = new_buf
    v.buf[v.size] = val
    v.size += 1
```

### `vector_extend(dst, src)` — bulk append from another vector

Free `@njit` function. Grows `dst` if needed (repeated doubling until `dst.buf.shape[0] >= dst.size + src.size`), then bulk-copies `src.buf[:src.size]` into `dst.buf[dst.size:]`.

```python
@njit
def vector_extend(dst, src):
    needed = dst.size + src.size
    cap = dst.buf.shape[0]
    if needed > cap:
        while cap < needed:
            cap *= 2
        new_buf = numpy.empty(cap, dst.buf.dtype)
        new_buf[:dst.size] = dst.buf[:dst.size]
        dst.buf = new_buf
    dst.buf[dst.size:dst.size + src.size] = src.buf[:src.size]
    dst.size += src.size
```

## Growth Strategy

Geometric doubling (2x) on overflow, same as `std::vector` in libstdc++/libc++. Initial capacity chosen by the caller at construction time (convention: 8). No shrink-to-fit — not needed for the accumulate-then-read UDAF lifecycle.

## Base Type

```python
class VectorType(nb_types.StructRef):
    """Base type class for all Vector[T] structrefs."""
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)
```

All concrete vector types are subclasses of `VectorType`. The `@overload` registrations check `isinstance(v, VectorType)` so they apply to all vector dtypes automatically.

## `removerefctpass` Safety

The vector is used inside DuckDB aggregate callbacks, which have `intp`-only signatures. The same symmetric-stripping analysis from the structref UDAF PR applies:

- **Callback functions** (`_irr_update_impl` etc.): `removerefctpass` strips NRT incref/decref symmetrically. Safe because the DuckDB state slot holds the root reference to the IRRState, which in turn holds references to its vector fields.
- **`vector_push` / `vector_extend`**: these take a structref argument (the vector), so `removerefctpass` sees NRT-tracked types in the signature and preserves all NRT operations. Buffer reallocation (decref old array, incref new) stays intact.

No new risks compared to the current `typed.List` approach.

## IRR Integration

### State definition changes

```python
Float64Vector, float64_vec_type = make_vector(nb_types.float64)

_irr_state_fields = [
    ("cashflows", float64_vec_type),
    ("periods", float64_vec_type),
    ("investment", nb_types.float64),
    ("target_npv", nb_types.float64),
    ("initialized", nb_types.int64),
]
IRRState = make_structref("IRRState", dict(_irr_state_fields), IRRStateType)
irr_state_type = IRRStateType(_irr_state_fields)
```

The field list is named once and reused for both the proxy (via `dict(...)` for `make_structref`) and the type instance (list-of-tuples form). Keeping a single source of truth avoids the cryptic type-identity mismatch when the two field lists drift.

### Callback changes

| Callback   | Before                                        | After                                              |
|-----------|-----------------------------------------------|----------------------------------------------------|
| init      | `typed_list.empty_list(nb_types.float64)`     | `Float64Vector(64)`                                |
| update    | `s.cashflows.append(val)`                     | `vector_push(s.cashflows, val)`                    |
| combine   | `tgt.cashflows.extend(src.cashflows)`         | `vector_extend(tgt.cashflows, src.cashflows)`      |
| finalize  | `s.cashflows[j]`, `len(s.cashflows)`          | unchanged (operator overloads)                     |
| destroy   | unchanged                                     | unchanged                                          |

### Removed imports

`from numba.typed import List as typed_list` — no longer needed.

## Testing

### `test/test_vector.py` — unit tests for the container

- **Construction**: create a vector, verify `len(v) == 0`, `v.buf.shape[0] == initial_capacity`.
- **Push + read**: push N elements, verify `len(v) == N`, verify `v[i]` returns correct values.
- **Growth**: push beyond initial capacity, verify capacity doubled and all elements preserved.
- **Extend**: create two vectors, extend one into the other, verify combined contents and size.
- **Multiple dtypes**: repeat core tests with `int64` and `float32` to exercise the generic factory.

### `examples/irr.py` — integration test

The existing IRR `main()` with its NPV cross-checks and NRT leak check serves as the integration test. The multi-group test exercises combine (and therefore `vector_extend`).

## File Layout

```
numbduck/
  vector.py          # make_vector factory, VectorType base, overloads, vector_push, vector_extend
examples/
  irr.py             # updated to use vector instead of typed.List
test/
  test_vector.py     # unit tests for the vector container
```

## Not In Scope

- Shrink-to-fit / `reserve` / `pop` / `clear` — not needed for the UDAF accumulation pattern
- Method syntax (`v.push(x)`) — free functions are more reliable in numba than `@overload_method`
- Upstreaming to numbox — future work after the pattern proves out
