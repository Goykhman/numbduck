# NumbDuck
Adapting DuckDB database API to the Numba JIT context.

Leverages [bindings](https://github.com/Goykhman/numbox/tree/main/numbox/core/bindings) toolkit in the [NumbOx](https://github.com/Goykhman/numbox) project.

Inspired by the [NumbSQL](https://github.com/cpcloud/numbsql) project.

## Caching

Every binding and buffer allocator compiles under the options in `NUMBDUCK_JIT_OPTIONS`, a JSON object read
once at import. If it is not set, the options are `{"cache": true, "nogil": true}`, so compiled code goes into
numba's on-disk cache, and `export NUMBDUCK_JIT_OPTIONS='{"cache": false}'` turns the cache off. A compile
option that differs from the defaults turns it off too, since numba's cache key holds no compile options. The
two bignum bindings never use the cache, since the name of their C symbol depends on the duckdb version and is
not in any cache key.

numbduck checks once at import whether numba can write a cache for it. If numba cannot, numbduck compiles
without a cache and warns once, naming the remedy. That happens on a read-only install, and under an import from
a `.zip`, when the user cache directory is read-only too. It also happens under an import from an `.egg`, `.whl`
or `.pyz` archive, and on numba 0.68 and later under an import from a `.zip` that holds only `.pyc` files. numbox's
[Where the cache lands](https://goykhman.github.io/numbox/numbox.core.configurations.html#where-the-cache-lands)
describes each placement and its remedy.

## Examples

Runnable narrative-style examples comparing numbduck against the closest stock
DuckDB Python+Arrow approaches live in [`examples/`](examples/). Each script is
self-contained, generates its own data, and prints the measured numbers.

Highlights:

- **Throughput** ([haversine.py](examples/haversine.py)): JIT chunk callback is
  ~400× faster than a per-row Python scalar UDF (10K rows) and ~100× faster than
  a PyArrow expression UDF at 1M rows. The win comes from no Python crossings per
  chunk and LLVM-fused math with no intermediate arrays.

- **Latency + GIL-free** ([online_scoring.py](examples/online_scoring.py)): ~2.2×
  lower per-event latency vs pure-Python `conn.execute`, and monotonic parallel
  scaling to ~2.4× on 8 threads while the Python loop plateaus under GIL
  contention.

- **Branchy logic** ([fraud_score.py](examples/fraud_score.py)): Arrow's
  `pc.if_else` chain beats the Python scalar UDF by ~60× (Arrow is the right
  stock-DuckDB tool here). The JIT chunk callback then beats Arrow by ~16× at 10K
  and ~1750× at 1M rows. The gap grows because each Arrow UDF chunk crosses the
  Python boundary and allocates intermediates, while the JIT computes in registers.
