"""Count the customers within 50 km of a store, with three implementations of the distance as a DuckDB scalar UDF.

The script fills a table with random customer locations and runs one query::

    SELECT count(*) FROM customers WHERE haversine(lat, lon, 37.7749, -122.4194) < 50

``haversine`` returns the great-circle distance in kilometers. Computing it for every row is most of the query's work,
so the query's time measures the UDF. The script registers the UDF three ways:

1. ``hv_py`` is a Python scalar UDF. DuckDB calls the Python function once per row, through the interpreter.
2. ``hv_arrow`` is a PyArrow UDF, registered with ``type="arrow"``. DuckDB calls it once per chunk of rows, and each
   ``pyarrow.compute`` step allocates an array for the whole chunk.
3. ``hv_jit`` is a numbduck UDF, registered through DuckDB's C API with ``duckdb_register_scalar_function``. DuckDB
   calls its compiled code once per chunk, and one loop computes every row with no Python call and no intermediate
   array.

The other variants' ratios to the JIT time grow with the row count, in part because DuckDB runs a large query on
several threads. The compiled callback never takes the GIL and runs on all of them at once. The Python and Arrow UDFs
hold the GIL, so more threads make them no faster.

Run it with::

    python examples/haversine.py
    NUMBDUCK_BENCH_BIG=1 python examples/haversine.py    # adds a run at 10,000,000 rows

The Arrow variant needs pyarrow, which numbduck does not install. At 10,000, 100,000 and 1,000,000 rows, the script
checks that every variant it runs at that size counts the same customers. It then prints the median time of each and its
ratio to the JIT time. ``NUMBDUCK_BENCH_REPEATS`` sets the number of timed runs, 3 by default; more runs make the
medians of these short timings more stable. The script is meant to finish within 30 seconds, so the Python UDF runs only
up to ``PY_MAX_N`` rows, 10,000. Under duckdb 1.x without pandas, the Python UDF would need more than a minute for a
million rows. With pandas installed, or under duckdb 2.0, raise ``PY_MAX_N`` to time it at every row count. The runs
below were made with ``PY_MAX_N`` raised.

The ratios depend on the machine, on the versions of duckdb and pyarrow, and on two more packages that numbduck does
not install: pandas, and python-dateutil, which pandas requires. Under duckdb 1.x, the duckdb module tries to import
pandas for every value that a Python UDF returns. If pandas is not installed, every attempt fails. Each failed import
takes far longer than the UDF itself. duckdb 2.0 does not try the import. pyarrow tries to import python-dateutil
each time a UDF passes a Python number to a ``pyarrow.compute`` function. ``hv_arrow`` passes Python numbers to
``pyarrow.compute`` functions. If python-dateutil is not installed, every attempt fails. On the Linux machine below,
the Arrow UDF then took about three times as long at a million rows. That factor differs between machines and between
Python environments. The runs below were made on two machines::

    Linux x86_64, with pyarrow 26.0.1 and pandas 3.1.1 installed, and PY_MAX_N raised to cover every row count:

    env: python 3.12.3, duckdb 2.0.0, numba 0.66.0, numpy 2.4.6, x86_64, 8 cores
    Haversine UDF benchmark: [10000, 100000, 1000000] rows, radius 50.0 km

           Rows  Python   Arrow     JIT  Py/JIT  Arr/JIT
         10,000  0.005s  0.002s  0.001s      6x       2x
        100,000  0.041s  0.011s  0.005s      8x       2x
      1,000,000  0.422s  0.194s  0.009s     47x      22x

    MacBook Pro M1, with PY_MAX_N raised to cover every row count:

     env: python 3.12.13, duckdb 1.5.6, numba 0.66.0, numpy 2.4.6, arm64, 8 cores
     (requires pandas to avoid suboptimal type checking; installed pandas==3.1.1)
     Haversine UDF benchmark: [10000, 100000, 1000000] rows, radius 50.0 km

           Rows  Python   Arrow     JIT  Py/JIT  Arr/JIT
         10,000  0.005s  0.002s  0.001s      8x       3x
        100,000  0.050s  0.018s  0.004s     12x       4x
      1,000,000  0.504s  0.403s  0.010s     49x      39x

    env: python 3.12.13, duckdb 2.0.0, numba 0.66.0, numpy 2.4.6, arm64, 8 cores
    Haversine UDF benchmark: [10000, 100000, 1000000] rows, radius 50.0 km

           Rows  Python   Arrow     JIT  Py/JIT  Arr/JIT
         10,000  0.005s  0.004s  0.001s      7x       4x
        100,000  0.048s  0.028s  0.004s     13x       7x
      1,000,000  0.467s  0.478s  0.008s     56x      57x
"""
import math
import os
import sys

import duckdb
import numpy
from numba import cfunc, njit, carray
from numba import types as nb_types
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p

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
# The largest row count at which the script runs the Python UDF. Under duckdb 1.x without pandas, that UDF would need
# more than a minute for a million rows.
PY_MAX_N = 10_000


def haversine_py(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in kilometers between two points given in degrees: the Python scalar UDF.

    The function expects finite coordinates with latitudes in [-90, 90], such as the script generates. An infinite
    coordinate makes a ``math`` function raise ``ValueError``, which aborts the query, while the Arrow and JIT variants
    return NaN. A NaN coordinate gives NaN in all three variants. DuckDB turns the NaN that this function returns into
    SQL NULL, though, while ``hv_arrow`` and ``hv_jit`` return NaN.
    """
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    s_dp = math.sin(dp / 2.0)
    s_dl = math.sin(dl / 2.0)
    a = s_dp * s_dp + math.cos(p1) * math.cos(p2) * s_dl * s_dl
    # The clamp is needed. Rounding puts `a` 2 ULP above 1.0 for some real pairs of nearly antipodal points, and sqrt
    # keeps it above 1.0: sqrt(1 + 2 * 2**-52) == 1 + 2**-52. Only an excess of 1 ULP rounds back to 1.0 through sqrt.
    # Unclamped, math.asin(math.sqrt(a)) would raise ValueError and abort the query. The excess is rounding error, and
    # the true value is asin(1), which gives the antipodal distance pi * R.
    #
    # All three variants square the sines as s * s and clamp `a` at 1.0, so a clamped row gives exactly pi * R in each.
    # This function and the JIT callback also group cos(p1) * cos(p2) * s_dl * s_dl left to right, and they agree bit
    # for bit. The Arrow variant multiplies (cos(p1) * cos(p2)) by (s_dl * s_dl), which can round `a` 1 ULP
    # differently. Just below the clamp, asin magnifies that ULP to about 0.2 m, the size of this formula's own
    # rounding error near the antipode.
    if a > 1.0:
        a = 1.0
    return 2.0 * R * math.asin(math.sqrt(a))


def haversine_arrow(lat1, lon1, lat2, lon2):
    """Return the great-circle distances in kilometers for arrays of points given in degrees: the PyArrow UDF.

    DuckDB calls the function once per chunk of up to 2,048 rows and passes each argument as a ``pyarrow.ChunkedArray``.
    The function returns the distances of the chunk's rows. A NULL row stays NULL, and a NaN or infinite coordinate
    gives NaN.
    """
    # pyarrow is imported here, not at the top of the module, so that a program that imports this module for the JIT
    # variant, such as a test, does not need pyarrow.
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
    # The clamp of haversine_py: without it, an `a` 2 ULP above 1.0 would make pc.asin return NaN, with no error. The
    # clamp uses if_else, because greater() is false for a NaN `a` and null for a NULL row. The NaN of a non-finite
    # coordinate therefore passes through, as in the JIT variant, and a NULL row stays NULL. min_element_wise would
    # turn both into 1.0 and return the antipodal distance.
    a = pc.if_else(pc.greater(a, 1.0), 1.0, a)
    return pc.multiply(2.0 * R, pc.asin(pc.sqrt(a)))


@njit
def _haversine_chunk_impl(info, chunk, output):
    """Compute ``hv_jit`` for one chunk: the distance in kilometers of each row of ``chunk``, written to ``output``.

    ``info``, ``chunk`` and ``output`` are the addresses of DuckDB's function info, input data chunk and output vector,
    as ``_haversine_chunk_cb`` receives them. The four inputs and the output are DOUBLE vectors. A row with a NULL input
    gets NaN, not SQL NULL, because numbduck binds no function that marks a row of the output invalid.
    """
    # Each input column is a vector of the chunk, and its data is a plain array of n float64 values. carray views that
    # memory in place. _cast_int_to_void_p turns the intp address that numbduck's bindings return into the void
    # pointer that carray takes.
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
    # DuckDB does not define the data of a NULL row, so the loop reads each input's validity mask and writes NaN for a
    # row with a NULL input. A validity pointer of 0 means that the vector has no NULLs. NaN is not SQL NULL, though.
    # The script's filter, distance < 50, drops a NaN row as it drops a NULL one. But sum() over hv_jit is NaN where
    # sum() over hv_py skips the NULL, as test_haversine_udf_null_aggregation_divergence shows.
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
    # The loop borrows no structref, and nothing in it raises, because inside @njit the math functions return NaN
    # outside their domain. The callback therefore needs no try/except guard, unlike the callbacks of irr.py.
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
        # Rounding can put `a` just above 1.0, as haversine_py explains. Inside @njit, asin and sqrt return NaN outside
        # their domain instead of raising, so such a row's distance would be NaN. With a latitude outside [-90, 90], a
        # cosine is negative, and rounding can also put `a` just below 0.0. The clamp uses comparisons, which are false
        # for NaN, so the NaN of an infinite or NaN coordinate passes through to the result. The builtins min and max
        # depend on the order of their arguments: min(1.0, a) turns a NaN `a` into 1.0, the antipodal distance, while
        # min(a, 1.0) keeps the NaN.
        if a > 1.0:
            a = 1.0
        elif a < 0.0:
            a = 0.0
        a_out[i] = 2.0 * R * math.asin(math.sqrt(a))


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _haversine_chunk_cb(info, chunk, output):
    """The ``duckdb_scalar_function_t`` that DuckDB calls for each chunk, with the three handles as intp addresses."""
    _haversine_chunk_impl(info, chunk, output)


def register_jit_udf(conn):
    """Register ``hv_jit``, a function of four DOUBLE arguments that returns a DOUBLE, on ``conn`` through the C API.

    ``_haversine_chunk_cb`` computes it, and ``AssertionError`` is raised if DuckDB refuses the registration.
    ``test_udf_benchmark`` in ``test/test_ducklib.py`` registers its JIT UDF in the same steps.
    """
    conn_ptr = extract_connection_ptr(conn)
    func_p = ducklib.duckdb_create_scalar_function()
    # c_string passes the name as NUL-terminated UTF-8, in a buffer that lives until the with block ends. DuckDB copies
    # the name, so the buffer is not needed after the call.
    with c_string("hv_jit") as name_p:
        ducklib.duckdb_scalar_function_set_name(func_p, name_p)
    dbl_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
    for _ in range(4):
        ducklib.duckdb_scalar_function_add_parameter(func_p, dbl_p)
    ducklib.duckdb_scalar_function_set_return_type(func_p, dbl_p)
    # The parameters and the return type hold copies of the logical type, so the type can be destroyed now. A destroy
    # function takes the address of the handle and sets the handle to NULL, so register_jit_udf first stores the
    # handle in an array.
    type_buf = numpy.array([dbl_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(type_buf.ctypes.data)
    ducklib.duckdb_scalar_function_set_function(func_p, _haversine_chunk_cb.address)
    rc = ducklib.duckdb_register_scalar_function(conn_ptr, func_p)
    # Registration does not take ownership of func_p, so the function handle is destroyed before the return code is
    # checked, after a success and after a failure alike.
    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)
    assert rc == ducklib.DuckDBSuccess


def setup_data(conn, n):
    """Replace the table ``customers`` with ``n`` random points, ``lat`` in [-90, 90) and ``lon`` in [-180, 180)."""
    conn.execute("SELECT setseed(0.42)")
    conn.execute(
        f"CREATE OR REPLACE TABLE customers AS "
        f"SELECT random()*180-90 AS lat, random()*360-180 AS lon FROM range({n})"
    )


def run_one(conn, n):
    """Time the three variants over ``n`` rows and return their median times, ``(t_py, t_arrow, t_jit)``.

    ``t_py`` is None above ``PY_MAX_N`` rows. ``AssertionError`` is raised if the variants count different rows.
    """
    setup_data(conn, n)
    sql_py = (
        f"SELECT count(*) FROM customers "
        f"WHERE hv_py(lat, lon, {STORE_LAT}, {STORE_LON}) < {RADIUS_KM}"
    )
    sql_arrow = sql_py.replace("hv_py", "hv_arrow")
    sql_jit = sql_py.replace("hv_py", "hv_jit")
    run_py = n <= PY_MAX_N

    # One untimed run of each query keeps first-call costs, such as the import of pyarrow, out of the timings.
    if run_py:
        conn.execute(sql_py).fetchone()
    conn.execute(sql_arrow).fetchone()
    conn.execute(sql_jit).fetchone()

    # The variants must count the same rows. The check includes the Python UDF at the row counts where it runs.
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
    """Register the three variants, time them at each row count, and print the table and a discussion of it."""
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
    # The discussion takes every number from this run's measurements, so it cannot contradict the table above; only
    # its wording is fixed. The first row count runs all three variants, and the last is the largest.
    n0, t_py0, t_arrow0, t_jit0 = timings[0]
    nL, _, t_arrowL, t_jitL = timings[-1]
    print(
        "  Discussion:\n"
        f"    At {n0:,d} rows the JIT chunk callback takes {t_jit0:.3f}s, the per-row\n"
        f"    Python scalar UDF {t_py0:.3f}s and the PyArrow expression UDF\n"
        f"    {t_arrow0:.3f}s. At {nL:,d} rows the JIT callback takes {t_jitL:.3f}s and\n"
        f"    the Arrow chain {t_arrowL:.3f}s. The JIT callback makes no Python call\n"
        "    per chunk, builds no intermediate arrays, and runs on all of DuckDB's\n"
        "    threads without the GIL. Each Arrow UDF chunk is a Python call that\n"
        "    builds a new array for every pc.* step. The per-row Python UDF runs\n"
        f"    only up to PY_MAX_N = {PY_MAX_N:,d} rows, to keep the run short; raise\n"
        "    PY_MAX_N to time it at more rows."
    )
    conn.close()


if __name__ == "__main__":
    main()
