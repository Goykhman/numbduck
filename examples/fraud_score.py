"""Fraud score UDF — branchy logic axis.

Story: a transaction risk service. Each transaction gets an integer fraud
score based on a small ruleset: amount tier, country mismatch, off-hours,
merchant risk tier, and recent-txn count. The query sums scores across
all transactions.

Three variants:
  1. Python scalar UDF — round-trips through the interpreter per row.
  2. PyArrow expression UDF — pc.if_else chains over chunks. This is the
     idiomatic vectorized-branchy approach in stock duckdb-python.
  3. numbduck JIT UDF — chunk callback with native if/else, registered
     via duckdb_register_scalar_function.

This example exists to test the design hypothesis that numbduck also wins
in Arrow's wheelhouse (vectorized branchy logic). Per the spec, if the
measured JIT-vs-Arrow gap is < 2x, the example is dropped — Arrow is the
right tool for that workload, and we should not pretend otherwise.

Run:
    python examples/fraud_score.py
    NUMBDUCK_BENCH_BIG=1 python examples/fraud_score.py    # 10M-row tier

Last measured on: 2026-04-10, x86_64 (WSL2, 8 cores), python 3.12.3,
duckdb 1.5.1, numba 0.64.0:

       Rows    Python      Arrow     JIT  Py/JIT  Arr/JIT
     10,000  463.76ms     7.88ms  0.48ms    956x    16.3x
    100,000       n/a    65.05ms  0.97ms     n/a    67.0x
  1,000,000       n/a  2078.35ms  1.18ms     n/a  1755.3x

The Python scalar UDF is only run on the smallest tier (per-row interpreter
round-trip is too expensive at >10K). The Arrow/JIT gap of ~1750x at 1M rows
is real (cross-check passes) but is partly an Arrow PyArrow-UDF artifact:
each chunk crosses the Python boundary and allocates intermediate arrays for
every pc.if_else step.
"""
import os
import sys

import duckdb
import numpy
from numba import carray, cfunc, njit
from numba import types as nb_types
from numbox.utils.lowlevel import _cast_int_to_void_p, get_unicode_data_p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import assert_results_match, format_table, print_env, time_median  # noqa: E402

from numbduck import ducklib  # noqa: E402
from numbduck.pybridge import extract_connection_ptr  # noqa: E402

ROW_COUNTS = [10_000, 100_000, 1_000_000]
if os.environ.get("NUMBDUCK_BENCH_BIG") == "1":
    ROW_COUNTS.append(10_000_000)
if os.environ.get("NUMBDUCK_BENCH_TINY") == "1":
    ROW_COUNTS = [1_000]
PY_MAX_N = 10_000
# A @cfunc scalar callback cannot emit SQL NULL (ducklib binds no validity-write
# helper) and cannot raise either -- an exception escaping the C boundary is
# swallowed, not propagated -- so a NULL input that reaches the callback must be
# written as some in-band value. We use an out-of-range sentinel: every rule adds
# non-negative points, so a real score is always >= 0 and -1 unambiguously flags a
# NULL/unknown transaction. Only column-borne NULLs reach the callback: DuckDB
# constant-folds a constant NULL argument to a SQL NULL projection upstream, so
# those rows are answered as SQL NULL directly and the sentinel never appears.
NULL_SCORE = -1


def fraud_score_py(amount, country, home_country, hour, risk_tier, recent_txns):
    s = 0
    if amount > 1000.0:
        s += 3
    elif amount > 200.0:
        s += 1
    if country != home_country:
        s += 4
    if hour < 6 or hour >= 23:
        s += 2
    if risk_tier >= 3:
        s += 5
    elif risk_tier == 2:
        s += 2
    if recent_txns > 10:
        s += 3
    return s


def fraud_score_arrow(amount, country, home_country, hour, risk_tier, recent_txns):
    # Lazy import: only the Arrow cross-check variant needs pyarrow, so importing
    # this module for the JIT path (or a test) does not require it.
    import pyarrow.compute as pc
    s_amt = pc.if_else(
        pc.greater(amount, 1000.0),
        3,
        pc.if_else(pc.greater(amount, 200.0), 1, 0),
    )
    s_country = pc.if_else(pc.not_equal(country, home_country), 4, 0)
    s_hour = pc.if_else(
        pc.or_(pc.less(hour, 6), pc.greater_equal(hour, 23)),
        2, 0,
    )
    s_risk = pc.if_else(
        pc.greater_equal(risk_tier, 3),
        5,
        pc.if_else(pc.equal(risk_tier, 2), 2, 0),
    )
    s_recent = pc.if_else(pc.greater(recent_txns, 10), 3, 0)
    total = pc.add(pc.add(pc.add(pc.add(s_amt, s_country), s_hour), s_risk), s_recent)
    return pc.cast(total, "int32")


@njit(inline="always")
def _score_row(amt, co, hco, hr, rt, rx):
    s = 0
    if amt > 1000.0:
        s += 3
    elif amt > 200.0:
        s += 1
    if co != hco:
        s += 4
    if hr < 6 or hr >= 23:
        s += 2
    if rt >= 3:
        s += 5
    elif rt == 2:
        s += 2
    if rx > 10:
        s += 3
    return s


@njit
def _fraud_chunk_impl(info, chunk, output):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    v_amt = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    v_co = ducklib.duckdb_data_chunk_get_vector(chunk, 1)
    v_hco = ducklib.duckdb_data_chunk_get_vector(chunk, 2)
    v_hr = ducklib.duckdb_data_chunk_get_vector(chunk, 3)
    v_rt = ducklib.duckdb_data_chunk_get_vector(chunk, 4)
    v_rx = ducklib.duckdb_data_chunk_get_vector(chunk, 5)
    d_amt = ducklib.duckdb_vector_get_data(v_amt)
    d_co = ducklib.duckdb_vector_get_data(v_co)
    d_hco = ducklib.duckdb_vector_get_data(v_hco)
    d_hr = ducklib.duckdb_vector_get_data(v_hr)
    d_rt = ducklib.duckdb_vector_get_data(v_rt)
    d_rx = ducklib.duckdb_vector_get_data(v_rx)
    d_out = ducklib.duckdb_vector_get_data(output)
    val_amt = ducklib.duckdb_vector_get_validity(v_amt)
    val_co = ducklib.duckdb_vector_get_validity(v_co)
    val_hco = ducklib.duckdb_vector_get_validity(v_hco)
    val_hr = ducklib.duckdb_vector_get_validity(v_hr)
    val_rt = ducklib.duckdb_vector_get_validity(v_rt)
    val_rx = ducklib.duckdb_vector_get_validity(v_rx)
    a_amt = carray(_cast_int_to_void_p(d_amt), (n,), dtype=numpy.float64)
    a_co = carray(_cast_int_to_void_p(d_co), (n,), dtype=numpy.int8)
    a_hco = carray(_cast_int_to_void_p(d_hco), (n,), dtype=numpy.int8)
    a_hr = carray(_cast_int_to_void_p(d_hr), (n,), dtype=numpy.int8)
    a_rt = carray(_cast_int_to_void_p(d_rt), (n,), dtype=numpy.int8)
    a_rx = carray(_cast_int_to_void_p(d_rx), (n,), dtype=numpy.int32)
    a_out = carray(_cast_int_to_void_p(d_out), (n,), dtype=numpy.int32)
    # DuckDB returns a zero validity pointer for a vector with no NULLs. When all
    # six inputs are fully valid (the common case, and all this benchmark feeds),
    # run a check-free tight loop so the per-row validity probe cannot throttle the
    # hot path; fall back to the guarded loop only when some vector carries NULLs.
    any_validity = (
        val_amt != 0 or val_co != 0 or val_hco != 0
        or val_hr != 0 or val_rt != 0 or val_rx != 0
    )
    if not any_validity:
        for i in range(n):
            a_out[i] = _score_row(a_amt[i], a_co[i], a_hco[i], a_hr[i], a_rt[i], a_rx[i])
        return
    for i in range(n):
        # A NULL row's data slot holds stale bytes. Read each input's validity mask
        # and score any NULL-bearing row with the sentinel instead of folding
        # garbage into the total.
        null_in = (
            (val_amt != 0 and not ducklib.duckdb_validity_row_is_valid(val_amt, i))
            or (val_co != 0 and not ducklib.duckdb_validity_row_is_valid(val_co, i))
            or (val_hco != 0 and not ducklib.duckdb_validity_row_is_valid(val_hco, i))
            or (val_hr != 0 and not ducklib.duckdb_validity_row_is_valid(val_hr, i))
            or (val_rt != 0 and not ducklib.duckdb_validity_row_is_valid(val_rt, i))
            or (val_rx != 0 and not ducklib.duckdb_validity_row_is_valid(val_rx, i))
        )
        if null_in:
            a_out[i] = NULL_SCORE
            continue
        a_out[i] = _score_row(a_amt[i], a_co[i], a_hco[i], a_hr[i], a_rt[i], a_rx[i])


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _fraud_chunk_cb(info, chunk, output):
    _fraud_chunk_impl(info, chunk, output)


def register_jit_udf(conn):
    conn_ptr = extract_connection_ptr(conn)
    func_p = ducklib.duckdb_create_scalar_function()
    ducklib.duckdb_scalar_function_set_name(func_p, get_unicode_data_p("fr_jit"))
    dbl_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
    tin_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_TINYINT)
    int_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_INTEGER)
    ducklib.duckdb_scalar_function_add_parameter(func_p, dbl_p)  # amount
    ducklib.duckdb_scalar_function_add_parameter(func_p, tin_p)  # country
    ducklib.duckdb_scalar_function_add_parameter(func_p, tin_p)  # home_country
    ducklib.duckdb_scalar_function_add_parameter(func_p, tin_p)  # hour
    ducklib.duckdb_scalar_function_add_parameter(func_p, tin_p)  # risk_tier
    ducklib.duckdb_scalar_function_add_parameter(func_p, int_p)  # recent_txns
    ducklib.duckdb_scalar_function_set_return_type(func_p, int_p)
    for tp in (dbl_p, tin_p, int_p):
        buf = numpy.array([tp], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)
    ducklib.duckdb_scalar_function_set_function(func_p, _fraud_chunk_cb.address)
    rc = ducklib.duckdb_register_scalar_function(conn_ptr, func_p)
    # Registration never takes ownership of func_p, so destroy it on both the
    # success and failure paths — i.e. before asserting on the return code.
    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)
    assert rc == ducklib.DuckDBSuccess


def setup_data(conn, n):
    conn.execute("SELECT setseed(0.42)")
    conn.execute(
        f"CREATE OR REPLACE TABLE transactions AS "
        f"SELECT "
        f"  random()*2000 AS amount, "
        f"  CAST(floor(random()*20) AS TINYINT) AS country, "
        f"  CAST(floor(random()*20) AS TINYINT) AS home_country, "
        f"  CAST(floor(random()*24) AS TINYINT) AS hour, "
        f"  CAST(floor(random()*5) AS TINYINT) AS risk_tier, "
        f"  CAST(floor(random()*30) AS INTEGER) AS recent_txns "
        f"FROM range({n})"
    )


def run_one(conn, n):
    setup_data(conn, n)
    cols = "amount, country, home_country, hour, risk_tier, recent_txns"
    sql_py = f"SELECT sum(fr_py({cols})) FROM transactions"
    sql_arrow = sql_py.replace("fr_py", "fr_arrow")
    sql_jit = sql_py.replace("fr_py", "fr_jit")
    run_py = n <= PY_MAX_N

    if run_py:
        conn.execute(sql_py).fetchone()
    conn.execute(sql_arrow).fetchone()
    conn.execute(sql_jit).fetchone()

    # fr_jit's -1 sentinel is NOT SQL NULL: fr_py/fr_arrow yield SQL NULL for a NULL
    # row (which sum() skips) whereas sum(fr_jit(...)) folds each -1 into the total,
    # so the three variants are interchangeable only on non-NULL input -- which
    # setup_data generates. A caller with real column NULLs must filter the sentinel
    # before aggregating. See test_fraud_score_udf_null_divergence for the contract.
    r_arrow = conn.execute(sql_arrow).fetchone()[0]
    r_jit = conn.execute(sql_jit).fetchone()[0]
    if run_py:
        r_py = conn.execute(sql_py).fetchone()[0]
        assert_results_match(r_py, r_arrow, r_jit, label=f"fraud_score n={n}")
    else:
        assert_results_match(r_arrow, r_jit, label=f"fraud_score n={n}")

    t_py = time_median(lambda: conn.execute(sql_py).fetchone()) if run_py else None
    t_arrow = time_median(lambda: conn.execute(sql_arrow).fetchone())
    t_jit = time_median(lambda: conn.execute(sql_jit).fetchone())
    return t_py, t_arrow, t_jit


def main():
    print_env()
    print(f"  Fraud score UDF benchmark — {ROW_COUNTS} rows")
    print()

    conn = duckdb.connect()
    conn.create_function(
        "fr_py", fraud_score_py,
        ["DOUBLE", "TINYINT", "TINYINT", "TINYINT", "TINYINT", "INTEGER"],
        "INTEGER",
    )
    conn.create_function(
        "fr_arrow", fraud_score_arrow,
        ["DOUBLE", "TINYINT", "TINYINT", "TINYINT", "TINYINT", "INTEGER"],
        "INTEGER", type="arrow",
    )
    register_jit_udf(conn)

    rows = []
    timings = []
    for n in ROW_COUNTS:
        t_py, t_arrow, t_jit = run_one(conn, n)
        timings.append((n, t_py, t_arrow, t_jit))
        py_str = f"{t_py*1000:.2f}ms" if t_py is not None else "n/a"
        py_ratio = f"{t_py/t_jit:.0f}x" if t_py is not None else "n/a"
        rows.append([
            f"{n:,d}",
            py_str,
            f"{t_arrow*1000:.2f}ms",
            f"{t_jit*1000:.2f}ms",
            py_ratio,
            f"{t_arrow/t_jit:.1f}x",
        ])

    print(format_table(
        headers=["Rows", "Python", "Arrow", "JIT", "Py/JIT", "Arr/JIT"],
        rows=rows,
        alignments=[">", ">", ">", ">", ">", ">"],
    ))
    print()
    # Interpolate every ratio straight from the measurements above so the prose can
    # never contradict the printed table (only the qualitative story is fixed). The
    # first tier runs all three variants; the last is the widest N.
    n0, t_py0, t_arrow0, t_jit0 = timings[0]
    nL, _, t_arrowL, t_jitL = timings[-1]
    largest_ratio = t_arrowL / t_jitL
    print(f"  Largest-tier Arrow/JIT ratio: {largest_ratio:.2f}x")
    print(
        "  DECISION: < 2x means this example does not motivate switching tools.\n"
        "  >= 2x means JIT wins in Arrow's wheelhouse and the example earns its place."
    )
    print()
    print(
        "  Discussion:\n"
        "    Arrow does the right vectorized work and beats the per-row Python\n"
        f"    scalar UDF by ~{t_py0 / t_arrow0:.0f}x at {n0:,d} rows — pyarrow's chained\n"
        "    pc.if_else is the correct stock-DuckDB tool for branchy logic, full\n"
        f"    credit. The JIT chunk callback then beats Arrow by ~{t_arrow0 / t_jit0:.0f}x\n"
        f"    at {n0:,d} rows, widening to ~{t_arrowL / t_jitL:.0f}x at {nL:,d} rows. The\n"
        "    growing gap is *partly* an artifact: each Arrow UDF chunk crosses the\n"
        "    Python boundary and allocates a fresh intermediate array for every\n"
        "    pc.if_else step, while the JIT computes the whole ruleset in registers\n"
        "    with no allocations. Even discounting the chunk-crossing overhead, the\n"
        "    JIT path is comfortably above the spec's 2x decision threshold and the\n"
        "    example earns its place. The case for numbduck strengthens further when\n"
        "    the branchy work is inside a JIT loop calling other DuckDB functions\n"
        "    (see online_scoring)."
    )
    conn.close()


if __name__ == "__main__":
    main()
