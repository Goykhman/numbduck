# IRR UDAF Example — Design Spec

## Overview

A self-contained example script (`examples/irr.py`) that demonstrates how to build a DuckDB aggregate function (UDAF) using numbduck and numbox's `make_structref`. This is the first aggregate example in the project — the existing examples are all scalar UDFs.

The UDAF computes the Internal Rate of Return (IRR): the monthly discount rate `r` such that the net present value of a series of cashflows equals a user-supplied target.

## SQL Interface

```sql
SELECT project, irr(cashflow, period, investment, target_npv)
FROM monthly_data
GROUP BY project;
```

Four parameters, all registered as `DOUBLE` in the DuckDB C API:

| Parameter    | Meaning                                      |
|-------------|----------------------------------------------|
| `cashflow`  | Monthly cashflow amount (one per row)         |
| `period`    | 1-based month index                            |
| `investment`| Upfront cost at t=0 (constant across group)   |
| `target_npv`| Desired NPV to solve for (constant across group) |

Return type: `DOUBLE` (the monthly rate `r`, or `NaN` if bisection doesn't converge).

## Formula

Find `r` such that:

```
f(r) = -investment - target_npv + sum(cashflows[i] / (1 + r) ^ periods[i]) = 0
```

Solved via bisection on `r` in `[-0.99, 10.0]`, tolerance `1e-9`, max 100 iterations.

## State Structref

Defined via numbox's `make_structref` (replaces ~45 lines of manual boilerplate):

```python
@structref.register
class IRRStateType(nb_types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)

IRRState = make_structref(
    "IRRState",
    {
        "cashflows": nb_types.ListType(nb_types.float64),
        "periods": nb_types.ListType(nb_types.float64),
        "investment": nb_types.float64,
        "target_npv": nb_types.float64,
        "initialized": nb_types.int64,
    },
    IRRStateType,
)
```

The `IRRStateType` class must be defined at module level (not inside `make_structref`) for numba's type system to work correctly. See `make_structref` docstring for rationale.

## Aggregate Callbacks

Six callbacks following the pattern established in `test/test_ducklib.py`:

### state_size
Returns 8 (one `intp`-sized pointer per group).

### init
Create an `IRRState` with empty `typed.List`s, `investment=0.0`, `target_npv=0.0`, `initialized=0`. Store via `export_meminfo`.

### update
For each row in the input chunk:
1. Read the state pointer from the DuckDB state slot via `borrow_structref`.
2. Append `cashflow` and `period` to the lists.
3. If `initialized == 0`, capture `investment` and `target_npv` from this row, set `initialized = 1`.

All four columns are read from the `duckdb_data_chunk` using the existing vector/data helpers.

### combine
Concatenate the source's `cashflows` and `periods` lists into the target's lists. If the target is uninitialized, copy `investment` and `target_npv` from the source.

### finalize
For each group:
1. Sort `(periods, cashflows)` pairs by period.
2. Run bisection to find `r`.
3. Write result to the output vector (or `NaN` if no convergence / empty input).

### destroy
Call `release_meminfo` to decref the NRT-managed structref.

## Bridge Intrinsics

The example defines its own copies of `export_meminfo`, `borrow_structref`, and `release_meminfo` locally — the example must be self-contained, not import from the test suite.

## Bisection Solver

A standalone `@njit` function:

```python
@njit
def irr_bisect(cashflows, periods, n, investment, target_npv):
    r_lo = -0.99
    r_hi = 10.0
    for _ in range(100):
        r_mid = (r_lo + r_hi) / 2.0
        npv = -investment - target_npv
        for i in range(n):
            npv += cashflows[i] / (1.0 + r_mid) ** periods[i]
        if abs(npv) < 1e-9:
            return r_mid
        if npv > 0.0:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return math.nan
```

## Registration

Plain single-function registration (no function set needed since all inputs are `DOUBLE` — no overloads).

```python
func_p = ducklib.duckdb_create_aggregate_function()
ducklib.duckdb_aggregate_function_set_name(func_p, name_p)
# add four DOUBLE parameters
for _ in range(4):
    ducklib.duckdb_aggregate_function_add_parameter(func_p, dbl_type_p)
ducklib.duckdb_aggregate_function_set_return_type(func_p, dbl_type_p)
ducklib.duckdb_aggregate_function_set_functions(
    func_p,
    state_size_cb.address,
    init_cb.address,
    update_cb.address,
    combine_cb.address,
    finalize_cb.address,
)
ducklib.duckdb_aggregate_function_set_destructor(func_p, destroy_cb.address)
ducklib.duckdb_register_aggregate_function(conn_p, func_p)
```

## Verification

The example constructs a known scenario and verifies the result:

```sql
-- 10,000 investment, 12 months of 1,000 cashflow each, target NPV = 0
-- Expected IRR ≈ 0.02921 (monthly)
CREATE TABLE test_irr AS
SELECT
    range + 1 AS period,
    1000.0 AS cashflow,
    10000.0 AS investment,
    0.0 AS target_npv
FROM range(12);

SELECT irr(cashflow, period, investment, target_npv) FROM test_irr;
```

Cross-check via a hand calculation with an assertion using `math.isclose(result, expected, rel_tol=1e-6)`.

## Narrative Structure

The script is written as a pedagogical walkthrough with print statements explaining each step:

1. **Intro** — what IRR is, why it's a good UDAF candidate (per-group variable-length accumulation + numerical solve)
2. **State definition** — `make_structref` usage, contrast with manual boilerplate
3. **Bridge intrinsics** — brief explanation of the NRT↔DuckDB round-trip
4. **Bisection solver** — the `@njit` function
5. **Callbacks** — the six DuckDB aggregate lifecycle functions
6. **Registration** — wiring it up via the C API
7. **Query and verify** — run the UDAF, check the answer

## File Layout

```
examples/
  irr.py          # the example script (self-contained)
```

No changes to `examples/_common.py` (no benchmarking needed). Update `examples/README.md` to add the IRR entry.

## Dependencies

Same as existing examples: `numbduck`, `numbox`, `duckdb`, `numba`, `numpy`.

## Not In Scope

- Multiple input type overloads (all inputs are DOUBLE)
- Benchmarking / timing tables (no stock-DuckDB equivalent to compare)
- Moving the bridge intrinsics into a shared module (future work, not this PR)
