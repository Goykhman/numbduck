"""Online event scoring loop — latency + GIL-free axis.

Story: a real-time scoring service. Events arrive one at a time. For each
event we look up the entity's stored features in a DuckDB table and compute
a 4-element dot-product score. The metric the operator cares about is
per-event latency, not throughput — batching events to amortize Python
overhead is exactly what the SLA forbids.

Two variants:
  1. Pure-Python loop calling conn.execute(...).fetchone() per event.
  2. numbduck @njit(nogil=True) loop calling duckdb_execute_prepared and
     reading the result chunk via the bound C API, with no Python crossings
     between iterations. Per-event latency is captured via a cross-platform
     monotonic clock bound inside the JIT loop (numbox.utils.clock.monotonic_ns —
     clock_gettime on POSIX, QueryPerformanceCounter on Windows).

This example also measures parallel scaling on 1/2/4/8 worker threads. The
expected (and dramatic) shape: the Python loop is GIL-bound and shows zero
or negative scaling; the JIT loop scales roughly linearly with cores.

Run:
    python examples/online_scoring.py
    NUMBDUCK_BENCH_BIG=1 python examples/online_scoring.py    # 500K-event dataset

Last measured on: 2026-04-10, x86_64 (WSL2, 8 cores), python 3.12.3,
duckdb 1.5.1, numba 0.64.0:

  Per-event latency (5,000 events):
  Variant   Total  Events/s  p50 µs  p95 µs  p99 µs   max µs
  Python   1.913s     2,614   347.3   582.3   884.0   2290.4
  JIT      0.914s     5,469   144.3   323.6   582.6  12448.8

  Parallel scaling (2,000 events split across T workers):
  T  Python  Py speedup     JIT  JIT speedup
  1  0.830s       1.00x  0.380s        1.00x
  2  0.497s       1.67x  0.292s        1.30x
  4  0.732s       1.13x  0.199s        1.91x
  8  0.859s       0.97x  0.159s        2.40x
"""
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import duckdb
import numpy
from numba import carray, njit
from numba.core.types import intp
from numbox.utils.clock import monotonic_ns
from numbox.utils.lowlevel import _cast_int_to_void_p, get_unicode_data_p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import assert_results_match, format_table, print_env  # noqa: E402

from numbduck import ducklib  # noqa: E402
from numbduck.duckdb_utils import create_duckdb_prepared_statement  # noqa: E402
from numbduck.pybridge import extract_connection_ptr  # noqa: E402


N_EVENTS = 50_000
N_FEATURES = 10_000
if os.environ.get("NUMBDUCK_BENCH_BIG") == "1":
    N_EVENTS = 500_000
# Smoke-test mode: a tiny end-to-end pass so CI can confirm the benchmark
# runs on every platform without burning real minutes.
if os.environ.get("NUMBDUCK_BENCH_TINY") == "1":
    N_EVENTS = 200
    N_FEATURES = 100


def setup_features(conn, k):
    conn.execute("SELECT setseed(0.42)")
    conn.execute(
        f"CREATE OR REPLACE TABLE features AS "
        f"SELECT range AS id, "
        f"  random() AS w0, random() AS w1, random() AS w2, random() AS w3 "
        f"FROM range({k})"
    )


def setup_events(n_events, k_features, seed=43):
    rng = numpy.random.default_rng(seed)
    ids = rng.integers(0, k_features, size=n_events, dtype=numpy.int64)
    x = rng.random((n_events, 4), dtype=numpy.float64)
    return ids, x


def sample_sizes(n_events):
    """(latency_events, scaling_events) actually scored for a dataset of
    n_events. Sized off the dataset so NUMBDUCK_BENCH_BIG enlarges the measured
    workload, not just the allocation; scaling scores fewer than latency so all
    thread counts fit the budget."""
    return n_events // 10, n_events // 25


def score_python(conn, ids, x):
    """Pure-Python loop. Returns (scores, latencies_ns)."""
    n = len(ids)
    scores = numpy.empty(n, dtype=numpy.float64)
    latencies = numpy.empty(n, dtype=numpy.int64)
    sql = "SELECT w0, w1, w2, w3 FROM features WHERE id = ?"
    for i in range(n):
        t0 = time.monotonic_ns()
        row = conn.execute(sql, [int(ids[i])]).fetchone()
        scores[i] = (
            row[0] * x[i, 0] + row[1] * x[i, 1]
            + row[2] * x[i, 2] + row[3] * x[i, 3]
        )
        latencies[i] = time.monotonic_ns() - t0
    return scores, latencies


@njit(nogil=True)
def _score_jit_loop(stmt_p, ids, x, scores_out, latencies_out):
    n = len(ids)
    result_buf = numpy.zeros(6, dtype=numpy.int64)
    chunk_buf = numpy.zeros(1, dtype=numpy.int64)
    result_p = intp(result_buf.ctypes.data)
    chunk_pp = intp(chunk_buf.ctypes.data)

    for i in range(n):
        t0 = monotonic_ns()

        # The bind/execute return-code checks below are defensive: this demo's
        # fixed point-lookup with a valid parameter index never fails them.
        bind_rc = ducklib.duckdb_bind_int64(stmt_p, numpy.uint64(1), ids[i])
        if bind_rc != ducklib.DuckDBSuccess:
            # Bind failed before any result was produced -- nothing to destroy.
            raise RuntimeError("bind failed in scoring loop")
        exec_rc = ducklib.duckdb_execute_prepared(stmt_p, result_p)
        if exec_rc != ducklib.DuckDBSuccess:
            # execute_prepared fills an error result even on failure; destroy it.
            ducklib.duckdb_destroy_result(result_p)
            raise RuntimeError("execute failed in scoring loop")
        result_tup = (
            result_buf[0], result_buf[1], result_buf[2],
            result_buf[3], result_buf[4], result_buf[5],
        )
        chunk_p = ducklib.duckdb_fetch_chunk(result_tup)
        # This point lookup matches exactly one row; a NULL (0) or empty chunk
        # means a missed key. Guard before dereferencing so a bad key raises
        # instead of segfaulting on a NULL chunk inside the nogil loop.
        if chunk_p == 0 or ducklib.duckdb_data_chunk_get_size(chunk_p) == 0:
            chunk_buf[0] = chunk_p
            ducklib.duckdb_destroy_data_chunk(chunk_pp)
            ducklib.duckdb_destroy_result(result_p)
            raise RuntimeError("no matching feature row in scoring loop")

        v0 = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
        v1 = ducklib.duckdb_data_chunk_get_vector(chunk_p, 1)
        v2 = ducklib.duckdb_data_chunk_get_vector(chunk_p, 2)
        v3 = ducklib.duckdb_data_chunk_get_vector(chunk_p, 3)
        # A NULL feature cell would be read as stale storage and silently score
        # wrong; the Python reference raises on None*float, so match that here.
        # Free the chunk+result before raising inside the nogil loop.
        val0 = ducklib.duckdb_vector_get_validity(v0)
        val1 = ducklib.duckdb_vector_get_validity(v1)
        val2 = ducklib.duckdb_vector_get_validity(v2)
        val3 = ducklib.duckdb_vector_get_validity(v3)
        if (
            (val0 != 0 and not ducklib.duckdb_validity_row_is_valid(val0, 0))
            or (val1 != 0 and not ducklib.duckdb_validity_row_is_valid(val1, 0))
            or (val2 != 0 and not ducklib.duckdb_validity_row_is_valid(val2, 0))
            or (val3 != 0 and not ducklib.duckdb_validity_row_is_valid(val3, 0))
        ):
            chunk_buf[0] = chunk_p
            ducklib.duckdb_destroy_data_chunk(chunk_pp)
            ducklib.duckdb_destroy_result(result_p)
            raise RuntimeError("NULL feature value in scoring loop")
        d0 = ducklib.duckdb_vector_get_data(v0)
        d1 = ducklib.duckdb_vector_get_data(v1)
        d2 = ducklib.duckdb_vector_get_data(v2)
        d3 = ducklib.duckdb_vector_get_data(v3)
        a0 = carray(_cast_int_to_void_p(d0), (1,), dtype=numpy.float64)
        a1 = carray(_cast_int_to_void_p(d1), (1,), dtype=numpy.float64)
        a2 = carray(_cast_int_to_void_p(d2), (1,), dtype=numpy.float64)
        a3 = carray(_cast_int_to_void_p(d3), (1,), dtype=numpy.float64)
        scores_out[i] = (
            a0[0] * x[i, 0] + a1[0] * x[i, 1]
            + a2[0] * x[i, 2] + a3[0] * x[i, 3]
        )

        chunk_buf[0] = chunk_p
        ducklib.duckdb_destroy_data_chunk(chunk_pp)
        ducklib.duckdb_destroy_result(result_p)

        t1 = monotonic_ns()
        latencies_out[i] = t1 - t0


def score_jit(conn, ids, x):
    """JIT loop. Prepares the statement once, then runs the @njit core."""
    conn_ptr = extract_connection_ptr(conn)
    stmt = create_duckdb_prepared_statement()
    sql = "SELECT w0, w1, w2, w3 FROM features WHERE id = $1"
    n = len(ids)
    scores = numpy.empty(n, dtype=numpy.float64)
    latencies = numpy.empty(n, dtype=numpy.int64)
    # duckdb_prepare allocates a statement object even when the prepare fails
    # (it owns the error message), so it must always be destroyed — hence the
    # try/finally, which also covers any exception raised inside the loop.
    try:
        rc = ducklib.duckdb_prepare(conn_ptr, get_unicode_data_p(sql), stmt.ctypes.data)
        assert rc == ducklib.DuckDBSuccess
        _score_jit_loop(int(stmt[0]), ids, x, scores, latencies)
    finally:
        ducklib.duckdb_destroy_prepare(stmt.ctypes.data)
    return scores, latencies


def percentiles_us(latencies_ns):
    a = numpy.asarray(latencies_ns) / 1000.0  # ns → µs
    return (
        float(numpy.percentile(a, 50)),
        float(numpy.percentile(a, 95)),
        float(numpy.percentile(a, 99)),
        float(a.max()),
    )


def run_latency_block(conn, ids, x):
    # Warm up
    _ws, _wl = score_python(conn, ids[:200], x[:200])
    _ws, _wl = score_jit(conn, ids[:200], x[:200])

    py_scores, py_lats = score_python(conn, ids, x)
    jit_scores, jit_lats = score_jit(conn, ids, x)

    # Cross-check
    assert_results_match(
        round(float(py_scores.sum()), 6),
        round(float(jit_scores.sum()), 6),
        label="online_scoring sums",
    )
    return (py_lats, jit_lats)


def run_scaling_block(conn_factory, ids, x):
    """For each T in the thread counts, launch T workers that each score n/T of
    the events; each worker uses its own connection. Returns dict
    {variant: {T: total_wall}}."""
    n = len(ids)
    out = {"python": {}, "jit": {}}
    if os.environ.get("NUMBDUCK_BENCH_TINY") == "1":
        thread_counts = [1, 2]
    else:
        thread_counts = [1, 2, 4, 8]

    def python_worker(my_ids, my_x):
        c = conn_factory()
        score_python(c, my_ids, my_x)
        c.close()

    def jit_worker(my_ids, my_x):
        c = conn_factory()
        score_jit(c, my_ids, my_x)
        c.close()

    for T in thread_counts:
        base, extra = divmod(n, T)
        slices = []
        start = 0
        for i in range(T):
            stop = start + base + (1 if i < extra else 0)
            slices.append((ids[start:stop], x[start:stop]))
            start = stop
        for variant, fn in [("python", python_worker), ("jit", jit_worker)]:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=T) as ex:
                list(ex.map(lambda s: fn(*s), slices))
            out[variant][T] = time.perf_counter() - t0
    return out


def main():
    print_env()
    print(f"  Online scoring — {N_EVENTS:,d}-event dataset, {N_FEATURES:,d} features")
    print()

    db = duckdb.connect()
    setup_features(db, N_FEATURES)

    ids, x = setup_events(N_EVENTS, N_FEATURES)

    # Both blocks score a prefix sample of the dataset (the events are i.i.d., so
    # a prefix is a valid random sample) because the Python reference is too slow
    # per call to score the whole pool. Sizing the samples off N_EVENTS is what
    # makes NUMBDUCK_BENCH_BIG enlarge the measured workload, not just the array.
    LAT_N, SCALE_N = sample_sizes(N_EVENTS)
    py_lats, jit_lats = run_latency_block(db, ids[:LAT_N], x[:LAT_N])

    py_total_us = py_lats.sum() / 1000.0
    jit_total_us = jit_lats.sum() / 1000.0
    py_eps = LAT_N / (py_total_us / 1e6)
    jit_eps = LAT_N / (jit_total_us / 1e6)
    py_p50, py_p95, py_p99, py_max = percentiles_us(py_lats)
    jit_p50, jit_p95, jit_p99, jit_max = percentiles_us(jit_lats)

    lat_table = format_table(
        headers=["Variant", "Total", "Events/s", "p50 µs", "p95 µs", "p99 µs", "max µs"],
        rows=[
            ["Python", f"{py_total_us / 1e6:.3f}s", f"{py_eps:,.0f}",
             f"{py_p50:.1f}", f"{py_p95:.1f}", f"{py_p99:.1f}", f"{py_max:.1f}"],
            ["JIT", f"{jit_total_us / 1e6:.3f}s", f"{jit_eps:,.0f}",
             f"{jit_p50:.1f}", f"{jit_p95:.1f}", f"{jit_p99:.1f}", f"{jit_max:.1f}"],
        ],
        alignments=["<", ">", ">", ">", ">", ">", ">"],
    )
    print(f"  Per-event latency ({LAT_N:,d} events):")
    print(lat_table)
    print()

    # Parallel scaling — each worker opens its own connection on the same
    # on-disk db.
    scale_fd, scale_db_path = tempfile.mkstemp(suffix=".duckdb", prefix="numbduck_online_scoring_")
    os.close(scale_fd)
    os.remove(scale_db_path)  # duckdb.connect wants to create the file itself
    try:
        scale_db = duckdb.connect(scale_db_path)
        setup_features(scale_db, N_FEATURES)
        scale_db.close()

        def conn_factory():
            return duckdb.connect(scale_db_path)

        scaling = run_scaling_block(conn_factory, ids[:SCALE_N], x[:SCALE_N])
    finally:
        if os.path.exists(scale_db_path):
            os.remove(scale_db_path)

    scale_rows = []
    base_py = scaling["python"][1]
    base_jit = scaling["jit"][1]
    for T in sorted(scaling["python"].keys()):
        py_t = scaling["python"][T]
        jit_t = scaling["jit"][T]
        scale_rows.append([
            f"{T}",
            f"{py_t:.3f}s",
            f"{base_py / py_t:.2f}x",
            f"{jit_t:.3f}s",
            f"{base_jit / jit_t:.2f}x",
        ])
    print(f"  Parallel scaling ({SCALE_N:,d} events split across T workers):")
    print(format_table(
        headers=["T", "Python", "Py speedup", "JIT", "JIT speedup"],
        rows=scale_rows,
        alignments=[">", ">", ">", ">", ">"],
    ))
    print()
    # Interpolate every number in the Discussion straight from this run's own
    # measurements so the prose can never contradict the tables above (only the
    # qualitative story is fixed) and never cites a thread count this mode did
    # not run.
    lat_ratio = py_p50 / jit_p50
    Ts = sorted(scaling["python"].keys())
    max_T = Ts[-1]
    best_py_T = max(Ts, key=lambda t: base_py / scaling["python"][t])
    best_py_speedup = base_py / scaling["python"][best_py_T]
    jit_max_speedup = base_jit / scaling["jit"][max_T]
    print(
        "  Discussion:\n"
        f"    Per-event latency: the JIT loop runs at ~{jit_eps:,.0f} events/sec\n"
        f"    versus ~{py_eps:,.0f} events/sec for the pure-Python loop — about\n"
        f"    {lat_ratio:.1f}x lower median latency ({jit_p50:.0f}µs vs {py_p50:.0f}µs at p50).\n"
        "    The win is modest because every iteration still pays for a real\n"
        "    DuckDB execute + chunk fetch + chunk destroy + result destroy; the\n"
        "    JIT only removes Python's per-iteration interpreter and ctypes\n"
        "    overhead.\n\n"
        "    Parallel scaling is the dramatic axis. The pure-Python loop is\n"
        f"    GIL-bound: across T=1..{max_T} its speedup never exceeds {best_py_speedup:.2f}x\n"
        f"    (best at T={best_py_T}) because the interpreter serializes every\n"
        "    iteration. The JIT loop, with nogil=True and a cross-platform\n"
        "    monotonic clock bound from libc (POSIX) or kernel32.dll (Windows)\n"
        f"    instead of time.monotonic_ns, scales to ~{jit_max_speedup:.2f}x at T={max_T}.\n"
        "    The remaining gap to ideal is from DuckDB-internal locking and\n"
        "    shared-memory contention, not GIL contention — that is the point of\n"
        "    the example."
    )

    db.close()


if __name__ == "__main__":
    main()
