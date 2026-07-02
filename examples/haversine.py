"""Haversine distance UDF — throughput axis.

Story: a retail analytics question. For each of N synthetic customer
locations, how far is the customer from a fixed store at (37.7749, -122.4194)?
The query runs `SELECT count(*) FROM customers WHERE haversine(...) < 50`.
The bottleneck is the per-row distance computation.

Three variants:
  1. Python scalar UDF — round-trips through the interpreter per row.
  2. PyArrow expression UDF — chained pc.sin/pc.cos/pc.atan2/pc.sqrt over chunks.
  3. numbduck JIT UDF — chunk callback with math.sin/cos/asin/sqrt, registered
     via duckdb_register_scalar_function.

Run:
    python examples/haversine.py
    NUMBDUCK_BENCH_BIG=1 python examples/haversine.py    # adds 10M-row tier

Last measured on: 2026-04-10, x86_64 (WSL2, 8 cores), python 3.12.3,
duckdb 1.5.1, numba 0.64.0:

       Rows  Python   Arrow     JIT  Py/JIT  Arr/JIT
     10,000  0.466s  0.009s  0.001s    400x       8x
    100,000     n/a  0.070s  0.005s     n/a      14x
  1,000,000     n/a  1.368s  0.014s     n/a     101x

The Python scalar UDF is only run on the smallest tier — at 100K it would
take ~10s and at 1M it would take minutes, which violates the example's
"finishes in under 30s" budget. Arrow and JIT scale to all three tiers.
Numbers vary by ±30% across runs due to small absolute timings; the
order-of-magnitude story (Python << Arrow << JIT) is stable.
"""
import math
import os
import sys

import duckdb
import numpy
import pyarrow.compute as pc
from numba import cfunc, njit, carray
from numba import types as nb_types
from numbox.utils.lowlevel import _cast_int_to_void_p, get_unicode_data_p

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import assert_results_match, format_table, print_env, time_median  # noqa: E402

from numbduck import ducklib  # noqa: E402
from numbduck.pybridge import extract_connection_ptr  # noqa: E402

STORE_LAT = 37.7749
STORE_LON = -122.4194
RADIUS_KM = 50.0
ROW_COUNTS = [10_000, 100_000, 1_000_000]
if os.environ.get("NUMBDUCK_BENCH_BIG") == "1":
    ROW_COUNTS.append(10_000_000)
# Python scalar UDF round-trips through the interpreter per row, so it
# becomes prohibitively slow at large N. We only run it on the smallest tier.
PY_MAX_N = 10_000


def haversine_py(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def haversine_arrow(lat1, lon1, lat2, lon2):
    R = 6371.0
    pi_180 = 3.141592653589793 / 180.0
    p1 = pc.multiply(lat1, pi_180)
    p2 = pc.multiply(lat2, pi_180)
    dp = pc.multiply(pc.subtract(lat2, lat1), pi_180)
    dl = pc.multiply(pc.subtract(lon2, lon1), pi_180)
    a = pc.add(
        pc.power(pc.sin(pc.divide(dp, 2.0)), 2),
        pc.multiply(
            pc.multiply(pc.cos(p1), pc.cos(p2)),
            pc.power(pc.sin(pc.divide(dl, 2.0)), 2),
        ),
    )
    return pc.multiply(2.0 * R, pc.asin(pc.sqrt(a)))


@njit
def _haversine_chunk_impl(info, chunk, output):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    v_lat1 = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    v_lon1 = ducklib.duckdb_data_chunk_get_vector(chunk, 1)
    v_lat2 = ducklib.duckdb_data_chunk_get_vector(chunk, 2)
    v_lon2 = ducklib.duckdb_data_chunk_get_vector(chunk, 3)
    d_lat1 = ducklib.duckdb_vector_get_data(v_lat1)
    d_lon1 = ducklib.duckdb_vector_get_data(v_lon1)
    d_lat2 = ducklib.duckdb_vector_get_data(v_lat2)
    d_lon2 = ducklib.duckdb_vector_get_data(v_lon2)
    d_out = ducklib.duckdb_vector_get_data(output)
    val_lat1 = ducklib.duckdb_vector_get_validity(v_lat1)
    val_lon1 = ducklib.duckdb_vector_get_validity(v_lon1)
    val_lat2 = ducklib.duckdb_vector_get_validity(v_lat2)
    val_lon2 = ducklib.duckdb_vector_get_validity(v_lon2)
    a_lat1 = carray(_cast_int_to_void_p(d_lat1), (n,), dtype=numpy.float64)
    a_lon1 = carray(_cast_int_to_void_p(d_lon1), (n,), dtype=numpy.float64)
    a_lat2 = carray(_cast_int_to_void_p(d_lat2), (n,), dtype=numpy.float64)
    a_lon2 = carray(_cast_int_to_void_p(d_lon2), (n,), dtype=numpy.float64)
    a_out = carray(_cast_int_to_void_p(d_out), (n,), dtype=numpy.float64)
    R = 6371.0
    # A raise inside this impl is swallowed at the @cfunc boundary (numba prints
    # it and returns the void default without unwinding into DuckDB), which would
    # hand DuckDB the partially-written output vector as a silent success. The
    # guard wraps the whole compute -- outside the hot loop so throughput is
    # unaffected -- and on failure fills the chunk with a NaN sentinel instead of
    # leaving garbage in the un-reached rows. No structref is borrowed here, so
    # there is nothing to leak; the sole hazard is the silent corrupt output.
    try:
        for i in range(n):
            # DuckDB can hand this callback rows whose inputs are NULL, and the
            # data slot of a NULL row holds stale bytes. Consult each input's
            # validity mask and emit the same NaN sentinel rather than folding a
            # garbage coordinate into the distance. A zero validity pointer means
            # the vector carries no NULLs, so the per-row check is skipped.
            null_in = (
                (val_lat1 != 0 and not ducklib.duckdb_validity_row_is_valid(val_lat1, i))
                or (val_lon1 != 0 and not ducklib.duckdb_validity_row_is_valid(val_lon1, i))
                or (val_lat2 != 0 and not ducklib.duckdb_validity_row_is_valid(val_lat2, i))
                or (val_lon2 != 0 and not ducklib.duckdb_validity_row_is_valid(val_lon2, i))
            )
            if null_in:
                a_out[i] = math.nan
                continue
            p1 = math.radians(a_lat1[i])
            p2 = math.radians(a_lat2[i])
            dp = math.radians(a_lat2[i] - a_lat1[i])
            dl = math.radians(a_lon2[i] - a_lon1[i])
            s_dp = math.sin(dp / 2.0)
            s_dl = math.sin(dl / 2.0)
            a = s_dp * s_dp + math.cos(p1) * math.cos(p2) * s_dl * s_dl
            a_out[i] = 2.0 * R * math.asin(math.sqrt(a))
    except Exception:
        for i in range(n):
            a_out[i] = math.nan


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _haversine_chunk_cb(info, chunk, output):
    _haversine_chunk_impl(info, chunk, output)


def register_jit_udf(conn):
    """Register hv_jit as a DuckDB scalar function. See test_udf_benchmark
    in test/test_ducklib.py for the canonical pattern."""
    conn_ptr = extract_connection_ptr(conn)
    func_p = ducklib.duckdb_create_scalar_function()
    ducklib.duckdb_scalar_function_set_name(func_p, get_unicode_data_p("hv_jit"))
    dbl_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
    for _ in range(4):
        ducklib.duckdb_scalar_function_add_parameter(func_p, dbl_p)
    ducklib.duckdb_scalar_function_set_return_type(func_p, dbl_p)
    type_buf = numpy.array([dbl_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)
    ducklib.duckdb_scalar_function_set_function(func_p, _haversine_chunk_cb.address)
    rc = ducklib.duckdb_register_scalar_function(conn_ptr, func_p)
    # Registration never takes ownership of func_p, so destroy it on both the
    # success and failure paths — i.e. before asserting on the return code.
    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)
    assert rc == ducklib.DuckDBSuccess


def setup_data(conn, n):
    conn.execute("SELECT setseed(0.42)")
    conn.execute(
        f"CREATE OR REPLACE TABLE customers AS "
        f"SELECT random()*180-90 AS lat, random()*360-180 AS lon FROM range({n})"
    )


def run_one(conn, n):
    setup_data(conn, n)
    sql_py = (
        f"SELECT count(*) FROM customers "
        f"WHERE hv_py(lat, lon, {STORE_LAT}, {STORE_LON}) < {RADIUS_KM}"
    )
    sql_arrow = sql_py.replace("hv_py", "hv_arrow")
    sql_jit = sql_py.replace("hv_py", "hv_jit")
    run_py = n <= PY_MAX_N

    # Warm up
    if run_py:
        conn.execute(sql_py).fetchone()
    conn.execute(sql_arrow).fetchone()
    conn.execute(sql_jit).fetchone()

    # Cross-check (Arrow and JIT always; Python included when feasible)
    r_arrow = conn.execute(sql_arrow).fetchone()[0]
    r_jit = conn.execute(sql_jit).fetchone()[0]
    if run_py:
        r_py = conn.execute(sql_py).fetchone()[0]
        assert_results_match(r_py, r_arrow, r_jit, label=f"haversine n={n}")
    else:
        assert_results_match(r_arrow, r_jit, label=f"haversine n={n}")

    t_py = time_median(lambda: conn.execute(sql_py).fetchone()) if run_py else None
    t_arrow = time_median(lambda: conn.execute(sql_arrow).fetchone())
    t_jit = time_median(lambda: conn.execute(sql_jit).fetchone())
    return t_py, t_arrow, t_jit


def main():
    print_env()
    print(f"  Haversine UDF benchmark — {ROW_COUNTS} rows, radius {RADIUS_KM} km")
    print()

    conn = duckdb.connect()
    conn.create_function("hv_py", haversine_py, ["DOUBLE"] * 4, "DOUBLE")
    conn.create_function(
        "hv_arrow", haversine_arrow, ["DOUBLE"] * 4, "DOUBLE", type="arrow"
    )
    register_jit_udf(conn)

    rows = []
    for n in ROW_COUNTS:
        t_py, t_arrow, t_jit = run_one(conn, n)
        py_str = f"{t_py:.3f}s" if t_py is not None else "n/a"
        py_ratio = f"{t_py/t_jit:.0f}x" if t_py is not None else "n/a"
        rows.append([
            f"{n:,d}",
            py_str,
            f"{t_arrow:.3f}s",
            f"{t_jit:.3f}s",
            py_ratio,
            f"{t_arrow/t_jit:.0f}x",
        ])

    print(format_table(
        headers=["Rows", "Python", "Arrow", "JIT", "Py/JIT", "Arr/JIT"],
        rows=rows,
        alignments=[">", ">", ">", ">", ">", ">"],
    ))
    print()
    print(
        "  Discussion:\n"
        "    At 10K rows the JIT chunk callback is ~400x faster than the per-row\n"
        "    Python scalar UDF and ~8x faster than the PyArrow expression UDF.\n"
        "    The gap to Arrow widens with N: at 1M rows the JIT runs in ~14ms\n"
        "    while the Arrow chain takes ~1.4s — a ~100x gap. The win comes from\n"
        "    no Python crossings per chunk, LLVM-fused math (sin/cos/asin/sqrt\n"
        "    inlined into one tight loop), and no intermediate Arrow arrays for\n"
        "    each pc.* step. Python's per-row interpreter round-trip is so\n"
        "    expensive at 100K+ that we omit it from the table — extrapolating\n"
        "    from the 10K timing, it would dominate the runtime budget by far."
    )
    conn.close()


if __name__ == "__main__":
    main()
