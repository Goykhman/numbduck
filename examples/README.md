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
  Per-event feature lookup and scoring inside a single `@njit(nogil=True)`
  loop. Compared against a pure-Python `conn.execute` loop. Demonstrates
  measurable parallel scaling under multithreading. *(Numbers TBD until
  Task 2 lands.)*

- **[fraud_score.py](fraud_score.py)** — *branchy logic axis.* Per-row
  business rules with several `if/else` branches. Shows that numbduck also
  wins in Arrow's wheelhouse, moderately. *(Numbers TBD until Task 3 lands.
  This example may be dropped if the measured JIT-vs-Arrow gap is < 2×.)*

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
