# numbduck examples

haversine.py and fraud_score.py each compare three versions of a DuckDB scalar function: a Python UDF and a PyArrow
UDF, which the duckdb module registers without numbduck, and a JIT UDF that numba compiles and numbduck registers.
online_scoring.py compares two loops that score events one at a time: a Python loop over `conn.execute`, and a jitted
loop that calls DuckDB's C API. Each of these three scripts generates its own data, checks that its versions give the
same answer, and then prints how long each one took. The fourth, run_irr.py, runs an aggregate function written with
numbduck and checks its answers. The [README](../README.md) explains the API that the scripts use.

## What each script does

[haversine.py](haversine.py) counts the customers within 50 km of a store, using the great-circle distance between two
points. It runs `SELECT count(*) FROM customers WHERE <distance> < 50` with three versions of the distance function:

- a Python function registered with `conn.create_function`, a [native UDF][native-udf], which DuckDB calls once per
  row;
- a function built from [PyArrow's compute functions][pyarrow-compute] and registered with `type="arrow"`, an
  [Arrow UDF][arrow-udf], which DuckDB calls once per chunk of up to 2,048 rows;
- a JIT UDF: a `@cfunc` callback that numba compiles and numbduck registers through DuckDB's C API.

The script runs the query on tables of 10,000, 100,000 and 1,000,000 rows. The Python UDF runs only on the table of
10,000 rows, because it would need more than a minute for a million rows under duckdb 1.x without pandas. `PY_MAX_N`
in the script sets that limit.

[fraud_score.py](fraud_score.py) scores each transaction with a few branching rules over six columns, and sums the
scores. It has the same three versions, the same table sizes and the same limit on the Python UDF as haversine.py. The
Arrow version chains [`pyarrow.compute.if_else`][if-else], the vectorized way to write branches in a UDF without
numbduck, and the JIT version runs the rules in one compiled loop. A JIT UDF cannot return NULL, so the JIT version
scores a row with a NULL input as -1. The generated data holds no NULLs, so the three versions agree.

[online_scoring.py](online_scoring.py) scores events one at a time, as a service would. For each event, it looks up a
row of features in a table and computes a dot product. One version calls [`conn.execute`][dbapi] for each event from a
Python loop. The other is an `@njit(nogil=True)` loop that binds and executes a prepared statement through the C API
and reads the one row that the statement returns. The script generates 50,000 events and a table of 10,000 rows of
features. It measures the latency of the first tenth of the events, 5,000, one at a time. It then splits the first
twenty-fifth, 2,000 events, across 1, 2, 4 and 8 threads, each with its own connection, and measures how each version
scales. The jitted loop runs without the GIL, and the Python loop holds the GIL whenever it runs Python code.

[run_irr.py](run_irr.py) runs [irr.py](irr.py), an aggregate function that computes the internal rate of return of
each group's cash flows. It has no version without numbduck to compare with. Its state is a numba structref, a mutable
struct whose memory numba reference-counts, and the structref holds the group's cash flows and their periods. The
aggregate follows the pattern of the README's section on [aggregate functions](../README.md#aggregate-functions).
Run it through run_irr.py, which imports irr.py as a module. irr.py refuses to run as the main script, because numba
caches the aggregate's code with its structref type, and a type that the main script defines does not match the
cached one in the next run.

[arrow-udf]: https://duckdb.org/docs/2.0/clients/python/function.html#arrow
[dbapi]: https://duckdb.org/docs/2.0/clients/python/dbapi.html
[if-else]: https://arrow.apache.org/docs/python/generated/pyarrow.compute.if_else.html
[native-udf]: https://duckdb.org/docs/2.0/clients/python/function.html#native
[pyarrow-compute]: https://arrow.apache.org/docs/python/api/compute.html

## What the scripts need

The scripts are not part of the package, so run them from a clone of the repository, with numbduck installed in the
clone by `pip install -e .`.

haversine.py and fraud_score.py also need [pyarrow][pyarrow-install] for their Arrow version, and numbduck does not
install it:

```bash
pip install pyarrow
```

Without pyarrow, the two scripts fail at their first Arrow query. The duckdb module raises its `Error` with the message
`ModuleNotFoundError: No module named 'pyarrow'`; the class prints as `_duckdb.Error` on duckdb 1.5.4 and 2.0.0 and
as `duckdb.duckdb.Error` on 1.2.2. online_scoring.py and run_irr.py run without pyarrow.

Two more packages change the timings, although every script runs without them: pandas, and python-dateutil, which
pandas requires.

- Under duckdb 1.x, the duckdb module tries to import pandas for every value that a Python UDF returns and for every
  parameter of a query. If pandas is not installed, every attempt fails. The Python UDF of haversine.py and
  fraud_score.py would then need more than a minute for a million rows. The Python loop of online_scoring.py is also
  slower without pandas, and in a run under duckdb 1.5.6 it was no faster with more than two threads. duckdb 2.0 does
  not try the import.
- pyarrow tries to import python-dateutil each time a UDF passes a Python number to a `pyarrow.compute` function. If
  python-dateutil is not installed, every attempt fails. The failed attempts slow the Arrow UDF of haversine.py and
  fraud_score.py. The docstrings of the two scripts give the factors measured on one machine.

Install pandas before timing the scripts:

```bash
pip install pandas
```

[pyarrow-install]: https://arrow.apache.org/docs/python/install.html

## Running

From the root of the clone:

```bash
python examples/haversine.py
python examples/fraud_score.py
python examples/online_scoring.py
python examples/run_irr.py
```

Three environment variables change the amount of work:

- `NUMBDUCK_BENCH_BIG=1` adds a table of 10,000,000 rows to haversine.py and fraud_score.py. It makes
  online_scoring.py generate 500,000 events instead of 50,000, so the two samples grow tenfold too, to 50,000 and
  20,000 events.
- `NUMBDUCK_BENCH_TINY=1` makes a run small enough for a quick check that the script works. fraud_score.py then uses one
  table of 1,000 rows. online_scoring.py generates 200 events and 100 rows of features, and runs on 1 and 2 threads.
  haversine.py does not read the variable.
- `NUMBDUCK_BENCH_REPEATS=n` makes haversine.py and fraud_score.py time each query `n` times instead of 3, and report
  the median.

For example:

```bash
NUMBDUCK_BENCH_BIG=1 python examples/haversine.py
NUMBDUCK_BENCH_REPEATS=5 python examples/haversine.py
NUMBDUCK_BENCH_TINY=1 python examples/online_scoring.py
```

## Reading the output

Every script except run_irr.py starts with a line that names the versions of Python, duckdb, numba and numpy, the
processor architecture and the number of cores.

haversine.py and fraud_score.py then print a table with one row per table size, under this header:

```text
       Rows  Python   Arrow     JIT  Py/JIT  Arr/JIT
```

The Python, Arrow and JIT columns give the median time of the whole query, in seconds for haversine.py and in
milliseconds for fraud_score.py. The time therefore covers DuckDB's scan and aggregation as well as the function.
`Py/JIT` and `Arr/JIT` show how many times longer the Python and Arrow versions took than the JIT version, and `n/a`
marks a size where the Python UDF did not run. Each script runs every query once to warm up, and once more to check
that the versions agree, before it times them. A paragraph headed `Discussion` restates figures of the same run in
words. fraud_score.py also prints the Arrow/JIT ratio of its largest table, and compares it with a threshold of 2x:
below it, the speedup would not justify a switch from the Arrow UDF to the JIT UDF.

online_scoring.py prints two tables. The first gives, for each version, the total time of the measured events, the
events per second, and the distribution of the single-event latencies, in microseconds: the median (p50), the 95th and
99th percentiles and the maximum. The second gives, for each number of threads `T`, the time that each version took
to score its sample split across `T` threads, and the version's speedup over its own time on one thread:

```text
  T  Python  Py speedup     JIT  JIT speedup
```

A `Discussion` paragraph again restates the numbers of the run.

run_irr.py runs three queries with known answers. The first computes the rate of one project, and the second the rates
of two projects in one grouped query. For each of those rates, the script checks that the net present value of the
cash flows at that rate is zero to within 1e-6. The third query holds a single cash flow, so the script compares its
rate with the exact rate.
The script prints each rate, and `All checks passed.` at the end. A wrong rate raises `AssertionError`. The script also
counts the memory blocks that numba's runtime, NRT, allocates and frees during the run. If the counts differ, it
prints `WARNING: NRT leak` and exits with status 1.

The ratios depend on the machine, on the versions of duckdb and pyarrow, on whether pandas and python-dateutil are
installed, and on what else runs at the time. Timings this short vary from run to run, and a larger
`NUMBDUCK_BENCH_REPEATS` takes each median over more runs.

## Reading the code

The scripts follow the [README](../README.md), and they import helpers from [numbox][numbox], the toolbox of numba
utilities that numbduck builds on. Each JIT UDF checks its input for NULL rows ([NULL input](../README.md#null-input)),
and irr.py guards each callback that holds the group's structref with `try` and `except`
([Aggregate functions](../README.md#aggregate-functions)).

[numbox]: https://github.com/Goykhman/numbox
