# Numbduck examples

Runnable narrative-style scripts that compare numbduck against the closest
stock-DuckDB-Python equivalents. Each script generates its own data, runs
all variants under timing, and prints honest results — including the cases
where numbduck wins by a lot, where it wins moderately, and (eventually)
where it doesn't.

## Scripts

- **[haversine.py](haversine.py)** — *throughput axis.* Per-row great-circle
  distance computation over synthetic customer points. Measured on this
  machine: the JIT chunk callback is **~620×** faster than the per-row Python
  scalar UDF (10K rows) and **~80×** faster than the PyArrow expression UDF
  at 1M rows.

- **[online_scoring.py](online_scoring.py)** — *latency + GIL-free axis.*
  Per-event feature lookup and dot-product score inside a single
  `@njit(nogil=True)` loop, with timestamps captured by calling libc
  `clock_gettime` from inside the JIT loop via numbox `_call_lib_func`.
  Measured: **~2.2× lower median latency** vs a pure-Python `conn.execute`
  loop, and **monotonic parallel scaling to ~2.4× on 8 threads** while the
  Python loop plateaus around 1× under GIL contention.

- **[fraud_score.py](fraud_score.py)** — *branchy logic axis.* Per-row
  business rules with several `if/else` branches over six columns. Arrow's
  `pc.if_else` chain beats the per-row Python scalar UDF by ~40× at 10K rows
  (full credit — Arrow is the right stock-DuckDB tool for branchy work). The
  JIT chunk callback then beats Arrow by **~20×** at 10K and **~1800×** at 1M
  rows; the growing gap is partly Arrow's per-chunk Python boundary plus
  intermediate-array allocation per `pc.*` step.

## Requirements

These scripts require `pyarrow` in addition to numbduck's normal dependencies
(it is used for the Arrow-based baselines in `haversine.py` and `fraud_score.py`
and is registered via DuckDB's `create_function(..., type="arrow")`). Install it
with `pip install pyarrow` if it is not already present in your venv.

## Running

```bash
python examples/haversine.py
python examples/online_scoring.py
python examples/fraud_score.py

# Larger row counts (~30s+ each):
NUMBDUCK_BENCH_BIG=1 python examples/haversine.py

# Tighter medians:
NUMBDUCK_BENCH_REPEATS=5 python examples/haversine.py
```
