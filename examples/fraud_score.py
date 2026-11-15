"""Score transactions for fraud with branching rules, in three implementations of a DuckDB scalar UDF.

Each transaction gets an integer score from five rules: the size of the amount, a country other than the home
country, a time before 6:00 or from 23:00 on, the merchant's risk tier, and more than ten recent transactions. The
query sums the scores of all the transactions. The script registers the scoring function three ways:

1. ``fr_py`` is a Python scalar UDF. DuckDB calls it once per row, through the interpreter.
2. ``fr_arrow`` is a PyArrow UDF that chains ``pc.if_else`` calls over each chunk of rows, the usual way to vectorize
   branching logic with duckdb-python alone. Each step allocates an intermediate array for the chunk.
3. ``fr_jit`` is a numbduck UDF registered through DuckDB's C API. Its compiled loop scores each row with plain ``if``
   statements, with no Python call and no intermediate array.

Arrow is most efficient on branching logic over whole vectors, so the script tests whether the JIT UDF is faster even
on this workload. It prints the Arrow/JIT ratio at its largest row count with a decision rule. A ratio below 2 would
mean that Arrow is the appropriate choice for this workload, and that the example would not belong among numbduck's
examples.

Run it with::

    python examples/fraud_score.py
    NUMBDUCK_BENCH_BIG=1 python examples/fraud_score.py     # adds a run at 10,000,000 rows
    NUMBDUCK_BENCH_TINY=1 python examples/fraud_score.py    # 1,000 rows only, a quick check that the script runs

The Arrow variant needs pyarrow, which numbduck does not install. At 10,000, 100,000 and 1,000,000 rows, the script
checks that the variants return the same sum. It then prints the median time of each and its ratio to the JIT time.
As in ``haversine.py``, the Python UDF runs only up to ``PY_MAX_N`` rows, 10,000. ``fr_jit`` cannot return SQL NULL,
so it scores a row with a NULL input as -1, ``NULL_SCORE``; the generated data holds no NULLs.

The JIT UDF is the fastest of the three in every run below, and the Arrow/JIT ratio grows with the row count. DuckDB
calls the Arrow UDF, a Python function, for every chunk, and each step of the UDF allocates a new array, while the JIT
UDF computes each row's score in registers. DuckDB also runs a large query on several threads, and the threads speed up
only the JIT UDF, because it never takes the GIL.

The times of the Python UDF and the Arrow UDF also depend on pandas and python-dateutil. The docstring of
``haversine.py`` gives the reasons. Under duckdb 1.x without pandas, the Python UDF would need more than a minute for
a million rows. ``fr_arrow`` passes about twenty Python numbers to ``pyarrow.compute`` functions for every chunk.
Without python-dateutil, the Arrow UDF therefore took about four times as long at a million rows on the Linux
machine below. That factor differs between machines and between Python environments. The runs below were made on two
machines::

    Linux x86_64, with pyarrow 26.0.1 and pandas 3.1.1 installed, and PY_MAX_N raised to cover every row count:

    env: python 3.12.3, duckdb 2.0.0, numba 0.66.0, numpy 2.4.6, x86_64, 8 cores
    Fraud score UDF benchmark: [10000, 100000, 1000000] rows

           Rows    Python     Arrow     JIT  Py/JIT  Arr/JIT
         10,000    2.88ms    1.58ms  0.38ms      8x     4.2x
        100,000   22.55ms   13.79ms  0.54ms     42x    25.6x
      1,000,000  244.43ms  257.64ms  0.91ms    270x   284.4x

    MacBook Pro M1, with PY_MAX_N raised to cover every row count:

    env: python 3.12.13, duckdb 1.5.6, numba 0.66.0, numpy 2.4.6, arm64, 8 cores
    (requires pandas to avoid suboptimal type checking; installed pandas==3.1.1)
    Fraud score UDF benchmark: [10000, 100000, 1000000] rows

           Rows    Python     Arrow     JIT  Py/JIT  Arr/JIT
         10,000    3.39ms    2.47ms  0.18ms     19x    14.1x
        100,000   30.79ms   19.74ms  0.33ms     93x    59.4x
      1,000,000  306.11ms  478.87ms  0.94ms    327x   511.5x

    env: python 3.12.13, duckdb 2.0.0, numba 0.66.0, numpy 2.4.6, arm64, 8 cores
    Fraud score UDF benchmark: [10000, 100000, 1000000] rows

           Rows    Python     Arrow     JIT  Py/JIT  Arr/JIT
         10,000    3.64ms    8.60ms  0.37ms     10x    23.2x
        100,000   30.63ms   47.82ms  0.69ms     44x    69.2x
      1,000,000  309.46ms  797.50ms  1.35ms    229x   589.6x
"""
import os
import sys

import duckdb
import numpy
from numba import carray, cfunc, njit
from numba import types as nb_types
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p

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
# The score that fr_jit gives a row with a NULL input. The callback cannot write SQL NULL, because ducklib binds no
# function that writes a validity mask. Nor can it raise: numba prints an exception that reaches the @cfunc and returns
# to DuckDB without an error. It could fail the whole query through duckdb_scalar_function_set_error, but the example
# marks the row instead. Every rule adds points, so a real score is at least 0, and -1 cannot be mistaken for one. Only
# a NULL in a column reaches the callback, because DuckDB folds a constant NULL argument into a NULL result.
NULL_SCORE = -1


def fraud_score_py(amount, country, home_country, hour, risk_tier, recent_txns):
    """Return the fraud score of one transaction, from 0 to 17: the Python scalar UDF.

    For a row with a NULL input, DuckDB returns SQL NULL.
    """
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
    """Return the fraud scores of a chunk of transactions as an int32 array: the PyArrow UDF.

    DuckDB calls the function once per chunk and passes each argument as a ``pyarrow.ChunkedArray``. A row with a NULL
    input gets NULL.
    """
    # pyarrow is imported here, not at the top of the module, so that a program that imports this module for the JIT
    # variant, such as a test, does not need pyarrow.
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
    """Return the fraud score of one row, by the rules of ``fraud_score_py``."""
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
    """Compute ``fr_jit`` for one chunk: the fraud score of each row of ``chunk``, written to ``output``.

    ``info``, ``chunk`` and ``output`` are the addresses of DuckDB's function info, input data chunk and output vector,
    as ``_fraud_chunk_cb`` receives them. The inputs are a DOUBLE vector, four TINYINT vectors and an INTEGER vector,
    and the output is an INTEGER vector. A row with a NULL input gets ``NULL_SCORE``.
    """
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
    # duckdb_vector_get_validity may return 0 for a vector without NULLs, and 0 always means that the vector has none.
    # When all six pointers are 0, as they are for the data that this script generates, a loop without validity
    # checks scores every row, so no per-row check slows the common case. Otherwise the guarded loop runs, even if
    # the vectors hold no NULLs.
    any_validity = (
        val_amt != 0 or val_co != 0 or val_hco != 0
        or val_hr != 0 or val_rt != 0 or val_rx != 0
    )
    if not any_validity:
        for i in range(n):
            a_out[i] = _score_row(a_amt[i], a_co[i], a_hco[i], a_hr[i], a_rt[i], a_rx[i])
        return
    for i in range(n):
        # DuckDB does not define the data of a NULL row, so a row with a NULL input gets NULL_SCORE instead of a
        # score computed from undefined bytes.
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
    """The ``duckdb_scalar_function_t`` that DuckDB calls for each chunk, with the three handles as intp addresses."""
    _fraud_chunk_impl(info, chunk, output)


def register_jit_udf(conn):
    """Register ``fr_jit``, a function of a DOUBLE, four TINYINTs and an INTEGER that returns an INTEGER, on ``conn``.

    The registration goes through DuckDB's C API, and ``_fraud_chunk_cb`` computes the function. ``AssertionError`` is
    raised if DuckDB refuses the registration.
    """
    conn_ptr = extract_connection_ptr(conn)
    func_p = ducklib.duckdb_create_scalar_function()
    with c_string("fr_jit") as name_p:
        ducklib.duckdb_scalar_function_set_name(func_p, name_p)
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
    # The parameters and the return type hold copies of the logical types, so the types can be destroyed now.
    for tp in (dbl_p, tin_p, int_p):
        buf = numpy.array([tp], dtype=numpy.intp)
        ducklib.duckdb_destroy_logical_type(buf.ctypes.data)
    ducklib.duckdb_scalar_function_set_function(func_p, _fraud_chunk_cb.address)
    rc = ducklib.duckdb_register_scalar_function(conn_ptr, func_p)
    # Registration does not take ownership of func_p, so the function handle is destroyed before the return code is
    # checked, after a success and after a failure alike.
    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_scalar_function(func_buf.ctypes.data)
    assert rc == ducklib.DuckDBSuccess


def setup_data(conn, n):
    """Replace the table ``transactions`` with ``n`` random transactions, none of them with a NULL."""
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
    """Time the three variants over ``n`` rows and return their median times, ``(t_py, t_arrow, t_jit)``.

    ``t_py`` is None above ``PY_MAX_N`` rows. ``AssertionError`` is raised if the variants return different sums.
    """
    setup_data(conn, n)
    cols = "amount, country, home_country, hour, risk_tier, recent_txns"
    sql_py = f"SELECT sum(fr_py({cols})) FROM transactions"
    sql_arrow = sql_py.replace("fr_py", "fr_arrow")
    sql_jit = sql_py.replace("fr_py", "fr_jit")
    run_py = n <= PY_MAX_N

    # One untimed run of each query keeps first-call costs, such as the import of pyarrow, out of the timings.
    if run_py:
        conn.execute(sql_py).fetchone()
    conn.execute(sql_arrow).fetchone()
    conn.execute(sql_jit).fetchone()

    # The sums agree because setup_data generates no NULLs. For a row with a NULL input, fr_py and fr_arrow return SQL
    # NULL, and sum() skips it. fr_jit returns -1 instead, and sum() adds it. A caller whose columns hold NULLs filters
    # out NULL_SCORE before aggregating, as test_fraud_score_udf_null_divergence does.
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
    """Register the three variants, time them at each row count, and print the table, the decision and a discussion."""
    print_env()
    print(f"  Fraud score UDF benchmark: {ROW_COUNTS} rows")
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
    # The discussion takes every number from this run's measurements, so it cannot contradict the table above; only
    # its wording is fixed. The first row count runs all three variants, and the last is the largest.
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
        f"    At {n0:,d} rows the per-row Python scalar UDF takes {t_py0 * 1000:.2f}ms,\n"
        f"    the Arrow UDF {t_arrow0 * 1000:.2f}ms and the JIT chunk callback\n"
        f"    {t_jit0 * 1000:.2f}ms. At {nL:,d} rows the Arrow UDF takes {t_arrowL * 1000:.2f}ms\n"
        f"    and the JIT callback {t_jitL * 1000:.2f}ms. pyarrow's chained pc.if_else is\n"
        "    the stock DuckDB tool for branchy logic like this. Each Arrow UDF\n"
        "    chunk is a Python call that builds a new array for every pc.if_else\n"
        "    step; the JIT callback computes the whole ruleset in one pass over\n"
        "    each chunk, on all of DuckDB's threads, without the GIL. The\n"
        f"    largest-tier Arrow/JIT ratio, {largest_ratio:.2f}x, is "
        f"{'at or above' if largest_ratio >= 2 else 'below'} the 2x\n"
        "    threshold of the decision above. online_scoring shows a JIT loop that\n"
        "    runs DuckDB queries itself."
    )
    conn.close()


if __name__ == "__main__":
    main()
