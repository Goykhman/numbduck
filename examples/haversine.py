"""Haversine distance UDF: throughput axis.

Story: a retail analytics question. For each of N synthetic customer
locations, how far is the customer from a fixed store at (37.7749, -122.4194)?
The query runs `SELECT count(*) FROM customers WHERE haversine(...) < 50`.
The bottleneck is the per-row distance computation.

Three variants:
  1. Python scalar UDF: round-trips through the interpreter per row.
  2. PyArrow expression UDF: chained pc.sin/pc.cos/pc.atan2/pc.sqrt over chunks.
  3. numbduck JIT UDF: chunk callback with math.sin/cos/asin/sqrt, registered
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

The Python scalar UDF is only run on the smallest tier: at 100K it would
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
    # Reference variant; assumes finite, in-domain coordinates (the demo's
    # generated data). An infinite coordinate makes CPython math.cos/asin raise
    # and abort the query, whereas the @njit and Arrow variants return NaN; a
    # NaN coordinate computes through to NaN in all three, though DuckDB's
    # scalar-UDF bridge surfaces this function's NaN as SQL NULL while
    # hv_arrow/hv_jit surface NaN. The a > 1.0 clamp is
    # required, not decorative: on real finite near-antipodal coordinates
    # the float64 `a` rounds up to ~2 ULP over 1.0, and sqrt does not
    # pull that back (sqrt(1 + 2*2**-52) == 1 + 2**-52 > 1.0; only a 1-ULP
    # overshoot rounds back through sqrt), so an unclamped math.asin(math.sqrt(a))
    # would raise ValueError and abort the whole query. The overshoot is pure
    # float error and the true value is asin(1) -- the antipodal distance pi*R --
    # so clamp to it. All three variants share this s*s form and this clamp, so
    # `a` rounds identically and clamped rows agree exactly (pi*R). Just below
    # the clamp, pyarrow's sin/cos kernels differ from libm by ULPs, which asin
    # amplifies near its singularity: the Arrow variant can differ from the
    # bit-identical py/JIT results by ~0.2 m there -- the same order as this
    # formulation's inherent near-antipodal error.
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    s_dp = math.sin(dp / 2.0)
    s_dl = math.sin(dl / 2.0)
    a = s_dp * s_dp + math.cos(p1) * math.cos(p2) * s_dl * s_dl
    if a > 1.0:
        a = 1.0
    return 2.0 * R * math.asin(math.sqrt(a))


def haversine_arrow(lat1, lon1, lat2, lon2):
    # Lazy import: only the Arrow cross-check variant needs pyarrow, so importing
    # this module for the JIT path (or a test) does not require it.
    import pyarrow.compute as pc
    R = 6371.0
    pi_180 = 3.141592653589793 / 180.0
    p1 = pc.multiply(lat1, pi_180)
    p2 = pc.multiply(lat2, pi_180)
    dp = pc.multiply(pc.subtract(lat2, lat1), pi_180)
    dl = pc.multiply(pc.subtract(lon2, lon1), pi_180)
    s_dp = pc.sin(pc.divide(dp, 2.0))
    s_dl = pc.sin(pc.divide(dl, 2.0))
    a = pc.add(
        pc.multiply(s_dp, s_dp),
        pc.multiply(pc.multiply(pc.cos(p1), pc.cos(p2)), pc.multiply(s_dl, s_dl)),
    )
    # Same required clamp as the other two variants (matched s*s form, so `a`
    # rounds identically): a ~2-ULP antipodal overshoot makes pc.asin silently
    # return NaN, so pin `a` to 1.0 and return asin(1) = the antipodal distance.
    # if_else rather than min_element_wise: greater() is false for a NaN `a` and
    # null for a NULL row, so a non-finite coordinate's NaN propagates to the
    # result (matching the JIT variant) and a NULL-input row stays NULL -- a min
    # would collapse both to 1.0 and fabricate the antipodal distance.
    a = pc.if_else(pc.greater(a, 1.0), 1.0, a)
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
    # A NULL row's data slot holds stale bytes, so consult each input's validity
    # mask and emit the NaN sentinel rather than folding a garbage coordinate
    # into the distance. A zero validity pointer means the vector has no NULLs.
    # NaN is not SQL NULL; see test_haversine_udf_null_aggregation_divergence.
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
    # No structref is borrowed and nothing here raises as written (math.* return
    # NaN out-of-domain), so this scalar callback needs no exception guard.
    for i in range(n):
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
        # Rounding can nudge `a` a hair past 1.0 (or below 0.0); inside @njit
        # asin/sqrt don't raise on an out-of-domain value, they return NaN, so
        # a near-1.0 row would silently come back NaN. Clamp with comparisons:
        # builtin min/max NaN handling is argument-order-dependent (`min(1.0, a)`
        # would pin a NaN `a` to 1.0 and a bogus max distance, while `min(a, 1.0)`
        # propagates it), whereas `a > 1.0`/`a < 0.0` are both False for NaN, so a
        # genuine NaN `a` (an Inf/NaN input coordinate) unambiguously falls through
        # to the NaN sentinel this callback emits for invalid rows.
        if a > 1.0:
            a = 1.0
        elif a < 0.0:
            a = 0.0
        a_out[i] = 2.0 * R * math.asin(math.sqrt(a))


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
    # success and failure paths, i.e. before asserting on the return code.
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
    print(f"  Haversine UDF benchmark: {ROW_COUNTS} rows, radius {RADIUS_KM} km")
    print()

    conn = duckdb.connect()
    conn.create_function("hv_py", haversine_py, ["DOUBLE"] * 4, "DOUBLE")
    conn.create_function(
        "hv_arrow", haversine_arrow, ["DOUBLE"] * 4, "DOUBLE", type="arrow"
    )
    register_jit_udf(conn)

    rows = []
    timings = []
    for n in ROW_COUNTS:
        t_py, t_arrow, t_jit = run_one(conn, n)
        timings.append((n, t_py, t_arrow, t_jit))
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
    # Interpolate every ratio/timing straight from the measurements above so the
    # prose can never contradict the printed table (only the qualitative story is
    # fixed). The first tier runs all three variants; the last is the widest N.
    n0, t_py0, t_arrow0, t_jit0 = timings[0]
    nL, _, t_arrowL, t_jitL = timings[-1]
    print(
        "  Discussion:\n"
        f"    At {n0:,d} rows the JIT chunk callback is ~{t_py0 / t_jit0:.0f}x\n"
        f"    faster than the per-row Python scalar UDF and ~{t_arrow0 / t_jit0:.0f}x\n"
        f"    faster than the PyArrow expression UDF. The gap to Arrow widens with\n"
        f"    N: at {nL:,d} rows the JIT runs in ~{t_jitL * 1000:.0f}ms while the\n"
        f"    Arrow chain takes ~{t_arrowL:.2f}s, a ~{t_arrowL / t_jitL:.0f}x gap.\n"
        "    The win comes from no Python crossings per chunk, LLVM-fused math\n"
        "    (sin/cos/asin/sqrt inlined into one tight loop), and no intermediate\n"
        "    Arrow arrays for each pc.* step. Python's per-row interpreter\n"
        "    round-trip is so expensive at 100K+ that we omit it from the table:\n"
        "    extrapolating from the 10K timing, it would dominate the budget."
    )
    conn.close()


if __name__ == "__main__":
    main()
