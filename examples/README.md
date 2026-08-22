# Numbduck examples

Runnable narrative-style scripts that compare numbduck against the closest
stock-DuckDB-Python equivalents. Each script generates its own data, runs
all variants under timing, and prints the results — including the cases
where numbduck wins by a lot, where it wins moderately, and (eventually)
where it doesn't.

Ratios below were measured as of July 2026 on the reference dev machine
(x86_64, 4 cores); the absolute timings are small and vary run to run, so
treat the numbers as orders of magnitude, not fixed constants.

## Scripts

- **[haversine.py](haversine.py)**: *throughput axis.* Per-row great-circle
  distance computation over synthetic customer points. Measured on this
  machine: the JIT chunk callback is **~850×** faster than the per-row Python
  scalar UDF (10K rows) and **~36×** faster than the [PyArrow expression UDF](https://duckdb.org/docs/stable/clients/python/function.html)
  at 1M rows.

- **[online_scoring.py](online_scoring.py)**: *latency + GIL-free axis.*
  Per-event feature lookup and dot-product score inside a single
  [`@njit(nogil=True)`](https://numba.readthedocs.io/en/stable/user/jit.html#nogil) loop, with timestamps captured via a cross-platform
  monotonic clock bound inside the JIT loop ([`numbox.utils.clock.monotonic_ns`](https://github.com/Goykhman/numbox/blob/0.6.2/numbox/utils/clock.py)).
  Measured: **~1.8× lower median latency** vs a pure-Python [`conn.execute`](https://duckdb.org/docs/stable/clients/python/dbapi.html)
  loop, and **monotonic parallel scaling to ~2.7× on 8 threads** while the
  Python loop stays near ~1.5× under GIL contention.

- **[fraud_score.py](fraud_score.py)**: *branchy logic axis.* Per-row
  business rules with several `if/else` branches over six columns. Arrow's
  [`pc.if_else`](https://arrow.apache.org/docs/python/generated/pyarrow.compute.if_else.html) chain beats the per-row Python scalar UDF by **~85×** at 10K rows
  (full credit: Arrow is the right stock-DuckDB tool for branchy work). The
  JIT chunk callback then beats Arrow by **~25×** at 10K and **~1000×** at 1M
  rows; the growing gap is partly Arrow's per-chunk Python boundary plus
  intermediate-array allocation per [`pc.*`](https://arrow.apache.org/docs/python/api/compute.html) step.

- **[irr.py](irr.py)** (run via **[run_irr.py](run_irr.py)**): *aggregate
  (UDAF) tutorial.* How to build a DuckDB aggregate function from scratch:
  define state as a numba structref (via numbox's
  [`make_structref`](https://github.com/Goykhman/numbox/blob/0.6.2/numbox/utils/highlevel.py)),
  write the six aggregate lifecycle callbacks, register with the C API, and
  verify against a known answer. Computes the Internal Rate of Return via
  bisection over accumulated `(cashflow, period)` pairs. Unlike the other
  scripts above, this one has no stock-DuckDB comparison. It's a worked
  example of the UDAF pattern. `irr.py` defines the UDAF and is meant to be
  imported, not run directly, so `run_irr.py` is the launcher that runs it.

## Requirements

These scripts require [`pyarrow`](https://arrow.apache.org/docs/python/install.html) in addition to numbduck's normal dependencies
(it is used for the Arrow-based baselines in [`haversine.py`](haversine.py) and [`fraud_score.py`](fraud_score.py)
and is registered via DuckDB's [`create_function(..., type="arrow")`](https://duckdb.org/docs/stable/clients/python/function.html)). Install it
with `pip install pyarrow` if it is not already present in your venv.

## Running

```bash
python examples/haversine.py
python examples/online_scoring.py
python examples/fraud_score.py
python examples/run_irr.py

# Larger row counts (~30s+ each):
NUMBDUCK_BENCH_BIG=1 python examples/haversine.py

# Tighter medians:
NUMBDUCK_BENCH_REPEATS=5 python examples/haversine.py
```
