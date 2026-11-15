"""Score events one at a time against a DuckDB table, from a Python loop and from a numba loop without the GIL.

A scoring service receives events one at a time. For each event, it looks up the entity's four stored features in a
DuckDB table and scores the event as the dot product of those features with the event's own four values. What matters
is the latency of each event, so the service cannot batch events to spread Python's overhead over many of them. The
script compares two loops that do this work:

1. A Python loop calls ``conn.execute(sql, [id]).fetchone()`` for each event.
2. An ``@njit(nogil=True)`` loop binds the id to a prepared statement, runs it with ``duckdb_execute_prepared`` and
   reads the result chunk through numbduck's bindings. It never returns to Python between events. It times each event
   with ``numbox.utils.clock.monotonic_ns``, a monotonic clock that jitted code can call: ``clock_gettime`` on Linux
   and macOS, ``QueryPerformanceCounter`` on Windows.

Each lookup is a real DuckDB query, so the JIT loop saves only the time that Python adds to each event: the time spent
in the interpreter and in the work that the duckdb module does around each call. The script also splits a sample of
the events among 1, 2, 4 and 8 threads, each with its own connection, and prints each loop's speedup over one thread.
The JIT loop never holds the GIL, so its threads run in parallel, while the Python loop holds the GIL whenever it runs
Python code. Under duckdb 2.0, the Python loop scales about as well as the JIT loop up to four threads in the runs
below. Under duckdb 1.x, the duckdb module tries to import pandas for every parameter of a query, and numbduck does
not install pandas. Without pandas, every attempt fails. The failed attempts slow the Python loop, and in a run under
duckdb 1.5.6 it was no faster with more than two threads.

Run it with::

    python examples/online_scoring.py
    NUMBDUCK_BENCH_BIG=1 python examples/online_scoring.py     # a dataset of 500,000 events
    NUMBDUCK_BENCH_TINY=1 python examples/online_scoring.py    # 200 events and 2 thread counts, a quick check

The dataset holds 50,000 events, and the table holds the features of 10,000 entities. The script times a tenth of the
events one by one, for the latencies, and splits a twenty-fifth of them among the threads, so ``NUMBDUCK_BENCH_BIG``
enlarges both measurements.
It checks that the two loops compute the same total score, then prints the latency percentiles of each loop and the
scaling table. Three runs, with "us" for microseconds::

    Linux x86_64, duckdb 2.0.0, with pandas 3.1.1 installed:

    env: python 3.12.3, duckdb 2.0.0, numba 0.66.0, numpy 2.4.6, x86_64, 8 cores
    Online scoring: 50,000-event dataset, 10,000 features

      Per-event latency (5,000 events):
      Variant   Total  Events/s  p50 us  p95 us  p99 us  max us
      Python   1.479s     3,380   291.1   325.2   350.8   692.8
      JIT      1.144s     4,370   227.7   238.8   249.4   510.2

      Parallel scaling (2,000 events split across T workers):
      T  Python  Py speedup     JIT  JIT speedup
      1  0.655s       1.00x  0.486s        1.00x
      2  0.340s       1.93x  0.267s        1.82x
      4  0.206s       3.17x  0.157s        3.10x
      8  0.177s       3.70x  0.137s        3.54x

    MacBook Pro M1, duckdb 1.5.6:

    env: python 3.12.13, duckdb 1.5.6, numba 0.66.0, numpy 2.4.6, arm64, 8 cores
    Online scoring: 50,000-event dataset, 10,000 features

      Per-event latency (5,000 events):
      Variant   Total  Events/s  p50 us  p95 us  p99 us  max us
      Python   0.982s     5,090   178.9   280.3   382.3   997.6
      JIT      0.429s    11,649    82.0   108.0   151.0   664.0

      Parallel scaling (2,000 events split across T workers):
      T  Python  Py speedup     JIT  JIT speedup
      1  0.418s       1.00x  0.205s        1.00x
      2  0.343s       1.22x  0.158s        1.30x
      4  0.423s       0.99x  0.101s        2.03x
      8  0.449s       0.93x  0.083s        2.48x

    MacBook Pro M1, duckdb 2.0.0:

    env: python 3.12.13, duckdb 2.0.0, numba 0.66.0, numpy 2.4.6, arm64, 8 cores
    Online scoring: 50,000-event dataset, 10,000 features

      Per-event latency (5,000 events):
      Variant   Total  Events/s  p50 us  p95 us  p99 us  max us
      Python   1.005s     4,978   191.5   252.5   325.7   785.0
      JIT      0.770s     6,494   150.0   181.0   214.0   499.0

      Parallel scaling (2,000 events split across T workers):
      T  Python  Py speedup     JIT  JIT speedup
      1  0.418s       1.00x  0.334s        1.00x
      2  0.244s       1.71x  0.208s        1.60x
      4  0.174s       2.40x  0.138s        2.43x
      8  0.180s       2.32x  0.106s        3.14x
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
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p, array_data_p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import assert_results_match, format_table, print_env  # noqa: E402

from numbduck import ducklib  # noqa: E402
from numbduck.duckdb_utils import create_duckdb_prepared_statement  # noqa: E402
from numbduck.pybridge import extract_connection_ptr  # noqa: E402
from numbduck.utils import complete_open_result  # noqa: E402


N_EVENTS = 50_000
N_FEATURES = 10_000
if os.environ.get("NUMBDUCK_BENCH_BIG") == "1":
    N_EVENTS = 500_000
# NUMBDUCK_BENCH_TINY=1 shrinks the run to a quick check that the script works: 200 events over 100 entities, and
# two thread counts instead of four.
if os.environ.get("NUMBDUCK_BENCH_TINY") == "1":
    N_EVENTS = 200
    N_FEATURES = 100


def setup_features(conn, k):
    """Replace the table ``features`` with ``k`` rows: an ``id`` from 0 to k - 1 and four random weights, w0 to w3."""
    conn.execute("SELECT setseed(0.42)")
    conn.execute(
        f"CREATE OR REPLACE TABLE features AS "
        f"SELECT range AS id, "
        f"  random() AS w0, random() AS w1, random() AS w2, random() AS w3 "
        f"FROM range({k})"
    )


def setup_events(n_events, k_features, seed=43):
    """Return ``(ids, x)`` for ``n_events`` random events: entity ids below ``k_features``, and 4 values per event."""
    rng = numpy.random.default_rng(seed)
    ids = rng.integers(0, k_features, size=n_events, dtype=numpy.int64)
    x = rng.random((n_events, 4), dtype=numpy.float64)
    return ids, x


def sample_sizes(n_events):
    """Return ``(latency_events, scaling_events)``, the numbers of events that the two measurements score.

    Both are fractions of the dataset, a tenth and a twenty-fifth, so ``NUMBDUCK_BENCH_BIG`` enlarges the measured work
    and not only the dataset. The scaling measurement scores fewer events because it runs once for every thread count.
    """
    return n_events // 10, n_events // 25


def score_python(conn, ids, x):
    """Score each event with the Python loop and return ``(scores, latencies_ns)``.

    An id without a row, or a NULL feature, raises ``TypeError`` from the arithmetic on ``None``.
    """
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
def _release(chunk_buf, result_buf):
    """Destroy the data chunk and the result whose handles ``chunk_buf`` and ``result_buf`` hold.

    The function takes the buffers, not their addresses, because numba frees an array right after its last use, and an
    address does not keep the array alive. On a branch that raises, the caller uses the buffers no further, so a destroy
    call that received only an address would read a freed buffer. A buffer passed in stays alive until the function
    returns.
    """
    ducklib.duckdb_destroy_data_chunk(array_data_p(chunk_buf))
    ducklib.duckdb_destroy_result(array_data_p(result_buf))


@njit(nogil=True)
def _score_jit_loop(stmt_p, ids, x, scores_out, latencies_out):
    """Score each event through the prepared statement ``stmt_p``, writing ``scores_out[i]`` and ``latencies_out[i]``.

    ``stmt_p`` is the address of a statement prepared with one parameter, the entity id, and the loop runs without the
    GIL. Each latency is in nanoseconds and covers the bind, the query, the fetch, the score and the destroy calls.
    ``RuntimeError`` is raised if a bind or a query fails, if an id matches no row, or if a feature is NULL. Before it
    raises, the loop destroys the result and the chunk that the event created.
    """
    n = len(ids)
    # duckdb_execute_prepared writes a duckdb_result, a struct of six 8-byte fields, into result_buf. chunk_buf holds
    # the handle of the current data chunk for _release.
    result_buf = numpy.zeros(6, dtype=numpy.int64)
    chunk_buf = numpy.zeros(1, dtype=numpy.int64)
    result_p = intp(result_buf.ctypes.data)

    for i in range(n):
        t0 = monotonic_ns()

        # The checks of the bind and the query are defensive: this fixed lookup, with its one valid parameter index,
        # never fails them.
        bind_rc = ducklib.duckdb_bind_int64(stmt_p, numpy.uint64(1), ids[i])
        if bind_rc != ducklib.DuckDBSuccess:
            # A failed bind creates no result, so there is nothing to destroy.
            raise RuntimeError("bind failed in scoring loop")
        exec_rc = ducklib.duckdb_execute_prepared(stmt_p, result_p)
        if exec_rc != ducklib.DuckDBSuccess:
            # duckdb_execute_prepared fills result_buf even when the query fails, and that result must be destroyed.
            _release(chunk_buf, result_buf)
            raise RuntimeError("execute failed in scoring loop")
        # duckdb_fetch_chunk takes the duckdb_result struct by value, which numbduck's binding accepts as a tuple of
        # its six fields.
        result_tup = (
            result_buf[0], result_buf[1], result_buf[2],
            result_buf[3], result_buf[4], result_buf[5],
        )
        chunk_p = ducklib.duckdb_fetch_chunk(result_tup)
        # The lookup matches exactly one row, so a chunk address of 0, or an empty chunk, means that the id matched
        # none. The check raises before the loop reads the chunk, because a read at address 0 would crash the process.
        if chunk_p == 0 or ducklib.duckdb_data_chunk_get_size(chunk_p) == 0:
            chunk_buf[0] = chunk_p
            _release(chunk_buf, result_buf)
            raise RuntimeError("no matching feature row in scoring loop")

        v0 = ducklib.duckdb_data_chunk_get_vector(chunk_p, 0)
        v1 = ducklib.duckdb_data_chunk_get_vector(chunk_p, 1)
        v2 = ducklib.duckdb_data_chunk_get_vector(chunk_p, 2)
        v3 = ducklib.duckdb_data_chunk_get_vector(chunk_p, 3)
        # The data of a NULL feature is undefined and would give a wrong score. score_python raises for a NULL
        # feature, because it multiplies None by a float, so this loop raises too, after it frees the chunk and the
        # result.
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
            _release(chunk_buf, result_buf)
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
        _release(chunk_buf, result_buf)

        t1 = monotonic_ns()
        latencies_out[i] = t1 - t0


def score_jit(conn, ids, x):
    """Score each event with the JIT loop and return ``(scores, latencies_ns)``.

    The function prepares the lookup once on ``conn`` and runs ``_score_jit_loop`` over all the events. It destroys the
    statement afterward, also when the loop raises. ``AssertionError`` is raised if the statement cannot be prepared.
    """
    conn_ptr = extract_connection_ptr(conn)
    stmt = create_duckdb_prepared_statement()
    sql = "SELECT w0, w1, w2, w3 FROM features WHERE id = $1"
    n = len(ids)
    scores = numpy.empty(n, dtype=numpy.float64)
    latencies = numpy.empty(n, dtype=numpy.int64)
    # duckdb 2.0 refuses a query through the C API on a connection that still has an unfinished result from
    # execute(), with "Resource In Use Error: connection has an open result". score_python leaves such a result,
    # because fetchone() reads only part of it. complete_open_result runs the result to its end under duckdb 2.0 and
    # does nothing under duckdb 1.x.
    complete_open_result(conn)
    # duckdb_prepare creates a statement even when it fails, to hold the error message, so the finally block always
    # destroys it. The block also runs when the loop raises.
    try:
        with c_string(sql) as sql_p:
            rc = ducklib.duckdb_prepare(conn_ptr, sql_p, stmt.ctypes.data)
        assert rc == ducklib.DuckDBSuccess
        _score_jit_loop(int(stmt[0]), ids, x, scores, latencies)
    finally:
        ducklib.duckdb_destroy_prepare(stmt.ctypes.data)
    return scores, latencies


def percentiles_us(latencies_ns):
    """Return the 50th, 95th and 99th percentiles and the maximum of ``latencies_ns``, converted to microseconds."""
    a = numpy.asarray(latencies_ns) / 1000.0
    return (
        float(numpy.percentile(a, 50)),
        float(numpy.percentile(a, 95)),
        float(numpy.percentile(a, 99)),
        float(a.max()),
    )


def run_latency_block(conn, ids, x):
    """Score the events with both loops on ``conn`` and return their latencies, ``(py_lats, jit_lats)``.

    ``AssertionError`` is raised if the two loops' total scores differ after rounding to six decimals.
    """
    # The first call of _score_jit_loop compiles it. A warm-up over the first 200 events keeps that, and DuckDB's
    # first-query costs, out of the latencies.
    _ws, _wl = score_python(conn, ids[:200], x[:200])
    _ws, _wl = score_jit(conn, ids[:200], x[:200])

    py_scores, py_lats = score_python(conn, ids, x)
    jit_scores, jit_lats = score_jit(conn, ids, x)

    assert_results_match(
        round(float(py_scores.sum()), 6),
        round(float(jit_scores.sum()), 6),
        label="online_scoring sums",
    )
    return (py_lats, jit_lats)


def run_scaling_block(conn_factory, ids, x):
    """Return ``{variant: {T: seconds}}``, the wall time of each loop with the events split among ``T`` threads.

    ``T`` takes the values 1, 2, 4 and 8, or 1 and 2 under ``NUMBDUCK_BENCH_TINY``. Each of the ``T`` workers scores an
    equal share of the events, to within one event. A worker opens its own connection with ``conn_factory`` and closes
    it when it is done, and the wall time includes both.
    """
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
    """Build the data, measure the latency and the scaling of both loops, and print the two tables and a discussion."""
    print_env()
    print(f"  Online scoring: {N_EVENTS:,d}-event dataset, {N_FEATURES:,d} features")
    print()

    db = duckdb.connect()
    setup_features(db, N_FEATURES)

    ids, x = setup_events(N_EVENTS, N_FEATURES)

    # Both measurements score the first events of the dataset, because the Python loop is too slow to score them all.
    # The events are independent draws from one distribution, so the first ones are a random sample. The sample sizes
    # are fractions of N_EVENTS, so NUMBDUCK_BENCH_BIG enlarges the measured work and not only the dataset.
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

    # Each worker of the scaling measurement needs a connection of its own to the same table. duckdb.connect() without
    # a path opens a new, empty in-memory database every time, so the script creates the table in a temporary database
    # file. mkstemp reserves a unique name but leaves an empty file, which DuckDB refuses as "not a valid DuckDB
    # database file", so the script removes it and lets DuckDB create the file.
    scale_fd, scale_db_path = tempfile.mkstemp(suffix=".duckdb", prefix="numbduck_online_scoring_")
    os.close(scale_fd)
    os.remove(scale_db_path)
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
    # The discussion takes every number from this run's measurements, so it cannot contradict the tables above or cite
    # a thread count that this run skipped. Only its wording is fixed.
    lat_ratio = py_p50 / jit_p50
    Ts = sorted(scaling["python"].keys())
    max_T = Ts[-1]
    best_py_T = max(Ts, key=lambda t: base_py / scaling["python"][t])
    best_py_speedup = base_py / scaling["python"][best_py_T]
    jit_max_speedup = base_jit / scaling["jit"][max_T]
    print(
        "  Discussion:\n"
        f"    Per-event latency: the JIT loop runs at ~{jit_eps:,.0f} events/sec\n"
        f"    versus ~{py_eps:,.0f} events/sec for the pure-Python loop, or about\n"
        f"    {lat_ratio:.1f}x lower median latency ({jit_p50:.0f}µs vs {py_p50:.0f}µs at p50).\n"
        "    Both loops run a real DuckDB query for every event. The JIT loop\n"
        "    calls the C API from compiled code, so it skips the Python\n"
        "    interpreter and the duckdb module's Python API between queries.\n\n"
        f"    Parallel scaling across T=1..{max_T}: the pure-Python loop's best\n"
        f"    speedup is {best_py_speedup:.2f}x (at T={best_py_T}), and the JIT loop reaches\n"
        f"    {jit_max_speedup:.2f}x at T={max_T}. The JIT loop runs with nogil=True and\n"
        "    times itself with numbox's monotonic_ns, so its threads hold no GIL\n"
        "    while they work. The Python loop holds the GIL whenever it runs\n"
        "    Python code between DuckDB calls, so its threads take turns for that\n"
        "    part of every event. How far it scales differs between duckdb\n"
        "    versions."
    )

    db.close()


if __name__ == "__main__":
    main()
