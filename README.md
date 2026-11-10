# numbduck

numbduck lets code that numba compiles call [DuckDB][duckdb] through DuckDB's [C API][c-api]. A function decorated
with [`@njit`][njit] can open a database, run a query and read the result, and no Python code runs between those
calls. numba can also compile the body of a DuckDB scalar function (a UDF) or aggregate function (a UDAF). DuckDB then
calls the compiled code once for each chunk, a batch of up to 2,048 rows.

Compiled code saves the cost of Python's own calls. A Python loop that calls `conn.execute` for each item of its input
runs the interpreter between the queries, and it holds the GIL while it does. A jitted loop runs the same queries
through the C API with no Python code in between, and it can release the GIL. A UDF shows the same difference. DuckDB
calls a Python UDF under the GIL, so its worker threads run the UDF one at a time, while they all run a compiled UDF at
once. The scripts in [Examples](#examples) measure both cases.

The module `numbduck.ducklib` holds the bindings, one numba function for each C function that numbduck supports.
Python code and jitted code can both call them. The bindings keep DuckDB's names and conventions. A handle is an
integer. A function that creates a database, a connection or a result writes the handle, or the result itself, to an
address that the caller passes. A failure is a return code, not an exception.

numbduck builds its bindings with the [bindings toolkit][numbox-bindings] of [numbox][numbox], a toolbox of low-level
utilities for numba. numbduck's idea comes from the [NumbSQL][numbsql] project, which runs numba-compiled functions in
SQLite.

[c-api]: https://duckdb.org/docs/2.0/clients/c/overview.html
[duckdb]: https://duckdb.org/
[njit]: https://numba.readthedocs.io/en/stable/user/jit.html
[numbox]: https://github.com/Goykhman/numbox
[numbox-bindings]: https://github.com/Goykhman/numbox/tree/main/numbox/core/bindings
[numbsql]: https://github.com/cpcloud/numbsql

## Installation

```bash
pip install numbduck
```

This installs duckdb and numbox, and numbox installs numba, which brings llvmlite and numpy.
[Compatibility](#compatibility) lists the versions of Python, duckdb and numba that numbduck supports.

numbduck calls the C API in the shared library of the duckdb wheel, the library that also runs `duckdb.connect()`. On
Linux and Windows, that library exports the C API, and nothing else is needed. Some macOS wheels do not export it, and
[macOS, NUMBDUCK_LIBDUCKDB and import errors](#macos-numbduck_libduckdb-and-import-errors) says what numbduck does
then.

numbduck does not use pyarrow. Two of the example scripts need it for the Arrow UDFs that they compare with numbduck,
as [Examples](#examples) explains.

The tests are not part of the package. To run them, clone the repository, run `pip install -e .` and
`pip install pytest` in the clone, and run `pytest` there.

## Quick start

The function below runs a whole query in compiled code. It opens an in-memory database, counts the multiples of seven
below 1000, and returns the count. The database, the connection and the result exist only inside the function. Each C
function that creates one of them writes the new handle, or the result, into a buffer: a numpy array that the caller
allocates and passes by its address.

```python
import numpy
from numba import carray, njit
from numbox.utils.lowlevel import _cast_int_to_void_p, array_data_p, get_unicode_data_p

from numbduck import ducklib
from numbduck.duckdb_utils import (
    create_duckdb_connection, create_duckdb_data_chunk, create_duckdb_database, create_duckdb_result
)


@njit(nogil=True)
def count_multiples_of_seven():
    db = create_duckdb_database()
    conn = create_duckdb_connection()
    result = create_duckdb_result()
    chunk = create_duckdb_data_chunk()
    sql = get_unicode_data_p("SELECT count(*) FROM range(1000) WHERE range % 7 = 0")
    count = -1
    if (ducklib.duckdb_open(0, array_data_p(db)) == ducklib.DuckDBSuccess
            and ducklib.duckdb_connect(db[0], array_data_p(conn)) == ducklib.DuckDBSuccess
            and ducklib.duckdb_query(conn[0], sql, array_data_p(result)) == ducklib.DuckDBSuccess):
        chunk[0] = ducklib.duckdb_fetch_chunk((result[0], result[1], result[2], result[3], result[4], result[5]))
        values_p = ducklib.duckdb_vector_get_data(ducklib.duckdb_data_chunk_get_vector(chunk[0], 0))
        count = carray(_cast_int_to_void_p(values_p), (1,), dtype=numpy.int64)[0]
        ducklib.duckdb_destroy_data_chunk(array_data_p(chunk))
    ducklib.duckdb_destroy_result(array_data_p(result))
    ducklib.duckdb_disconnect(array_data_p(conn))
    ducklib.duckdb_close(array_data_p(db))
    assert chunk[0] == 0 and result[0] == 0 and conn[0] == 0 and db[0] == 0
    return count


print(count_multiples_of_seven())  # 143
```

The four `create_duckdb_*` functions return the buffers, zeroed numpy arrays of `int64`. Three of them hold one handle
each. The fourth holds a `duckdb_result`, a C struct of six 64-bit fields. numbox's [`array_data_p`][array-data-p]
returns the address of a buffer, and the binding passes that address to DuckDB. Every address and every handle is an
`intp`, numba's integer as wide as a pointer, and a name that ends in `_p`, such as `values_p`, holds one.

`duckdb_open` with the path `0`, a NULL pointer, opens an in-memory database and writes its handle to `db[0]`.
`duckdb_connect` then writes the connection's handle to `conn[0]`, and `duckdb_query` writes the result into `result`.
`duckdb_query` reads the SQL text at an address, which numbox's [`get_unicode_data_p`][get-unicode-data-p] returns
for an ASCII string; [Text](#text) explains why only ASCII. Each of these bindings returns `DuckDBSuccess` or
`DuckDBError`, and the function checks the code before it makes the next call.

DuckDB returns the rows of a result in data chunks of up to 2,048 rows. A chunk holds a vector, an array of values,
for each column. `duckdb_fetch_chunk` returns the first chunk. It takes the `duckdb_result` struct itself, not its
address, and the binding takes the struct as a tuple of its six fields. `duckdb_vector_get_data` returns the address of
the values in column 0. `count(*)` returns a `BIGINT`, a 64-bit integer, so numba's [`carray`][carray] views the
values as an `int64` array. `carray` takes a pointer, not an integer, and numbox's
[`_cast_int_to_void_p`][cast-int-to-void-p] converts the address.

The last four calls free what DuckDB allocated, and each writes NULL into its buffer. The `assert` reads the buffers
back. It checks that DuckDB cleared them, and it keeps numba from freeing a buffer before DuckDB has read and cleared
it, as [Keeping a buffer alive](#keeping-a-buffer-alive) explains. `nogil=True` lets the function release the GIL, as
every jitted function that runs a query should ([Threads and the GIL](#threads-and-the-gil)).

The first run takes several seconds, because the first import of `numbduck.ducklib` compiles the bindings. Later
processes load them from numba's cache, as [Compilation and the cache](#compilation-and-the-cache) describes.

[array-data-p]: https://goykhman.github.io/numbox/numbox.utils.html#numbox.utils.lowlevel.array_data_p
[carray]: https://numba.readthedocs.io/en/stable/reference/utils.html#numba.carray
[cast-int-to-void-p]: https://github.com/Goykhman/numbox/blob/0.7.9/numbox/utils/lowlevel.py#L61-L67
[get-unicode-data-p]: https://goykhman.github.io/numbox/numbox.utils.html#numbox.utils.lowlevel.get_unicode_data_p

## Calling the C API

### The bindings

[`numbduck/ducklib.py`][ducklib] defines every binding, and DuckDB's [C API reference][c-api-reference] documents the
C function behind each one. A binding has the name of its C function, and its docstring links to that function's
entry in the reference. numbduck binds part of the C API, and a C function that `ducklib` does not define cannot be
called. `ducklib` also defines DuckDB's state codes, `DuckDBSuccess` and `DuckDBError`, and its type ids, such as
`DUCKDB_TYPE_INTEGER`.

A binding takes and returns numbers. Every pointer and every handle is an `intp`, and Python code passes an `int`. The
names of the parameters follow a convention. A name that ends in `_p` holds a pointer: a handle, or the address of
data such as text. A name that ends in `_pp` holds the address of a handle, where the function writes a new handle, or
NULL when it destroys the handle. In `duckdb_connect(duckdb_database_p, duckdb_connection_pp)`, for example, DuckDB
reads the database handle and writes the new connection's handle at the address in the second argument.

A C struct that DuckDB passes or returns by value is a tuple of its fields, in C order. A struct of one field, such as
`duckdb_date`, is that field alone: `duckdb_create_date` takes the days as an integer. `duckdb_fetch_chunk`,
`duckdb_result_return_type` and `duckdb_result_statement_type` take the whole `duckdb_result` struct, six 64-bit
fields. Python code passes `tuple(result)`, and jitted code writes out the six elements, as the quick start does.
`ducklib` declares the other struct types, such as `duckdb_hugeint_ty`, whose fields are the lower and upper 64 bits
of a 128-bit integer.

Five bindings call C functions that older versions of DuckDB lack. `duckdb_scalar_function_set_init` needs duckdb 1.5
or later. `duckdb_scalar_function_set_bind`, `duckdb_create_map_value`, `duckdb_create_union_value` and
`duckdb_value_to_string` need duckdb 1.3 or later. If the loaded library lacks the C function, a call from Python
raises `NotImplementedError`, and a jitted function that calls the binding fails to compile with a `TypingError` that
names it. `duckdb_create_bignum` and `duckdb_get_bignum` call `duckdb_create_varint` and `duckdb_get_varint` on duckdb
1.3 and earlier, where DuckDB used those names.

[c-api-reference]: https://duckdb.org/docs/2.0/clients/c/api.html
[ducklib]: https://github.com/Goykhman/numbduck/blob/main/numbduck/ducklib.py

### Buffers for handles

`duckdb_open`, `duckdb_connect` and `duckdb_prepare` return their handle through an out-parameter, an address where
DuckDB writes the new handle. `duckdb_query` and `duckdb_execute_prepared` write their result to an address in the same
way. The functions of `numbduck.duckdb_utils` allocate that memory as zeroed numpy arrays of `int64`, and Python code
and jitted code can call them. `create_duckdb_database`, `create_duckdb_connection`,
`create_duckdb_prepared_statement`, `create_duckdb_data_chunk`, `create_duckdb_value` and `create_duckdb_vector` each
hold one handle, and `create_duckdb_result` holds the six fields of a `duckdb_result`. Pass a buffer's address,
`buf.ctypes.data` in Python code or `array_data_p(buf)` in jitted code, and read the handle back as `buf[0]`.

Every function that destroys something takes the address of what it destroys and clears it there.
`duckdb_destroy_result` writes zeros over the six fields of a result, and `duckdb_disconnect`, `duckdb_close` and the
other `duckdb_destroy_*` functions write NULL over a handle. Most other functions that create something return their
handle, as `duckdb_fetch_chunk` and `duckdb_create_logical_type` do. Such a handle therefore needs a buffer before it
can be destroyed. The quick start stores its chunk in one for that reason, and Python code makes one with
`numpy.array([handle], dtype=numpy.intp)`.

### Keeping a buffer alive

An address does not keep its array alive. In Python code, bind each buffer to a name before you pass its address.
`f(numpy.array([handle], dtype=numpy.intp).ctypes.data)` frees the array before `f` runs, so `f` reads freed memory.

In jitted code, numba frees an array immediately after its last use in the function. If a buffer's address is that last
use, as in `ducklib.duckdb_close(array_data_p(db))` at the end of a function, numba may free `db` before `duckdb_close`
reads the handle and writes NULL back. Use the buffer again after the call, as the quick start's `assert` does. Or pass
the buffer itself to a jitted function that takes its address and makes the call. numba keeps an argument alive until
the function returns:

```python
from numba import njit
from numbox.utils.lowlevel import array_data_p

from numbduck import ducklib


@njit(nogil=True)
def close(db):
    """Close the database whose handle the buffer db holds."""
    ducklib.duckdb_close(array_data_p(db))
```

### Return codes and error messages

A binding reports DuckDB's errors as its C function does, never with an exception. It returns a `duckdb_state`,
`DuckDBSuccess` (0) or `DuckDBError` (1), or a handle that is NULL (0) on failure. A failed query still fills its
result, and `duckdb_result_error` returns the message from it. The example below runs in Python, and numbox's
`c_string`, which [Text](#text) describes, passes the SQL text:

```python
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import get_str_from_p_as_int

from numbduck import ducklib
from numbduck.duckdb_utils import create_duckdb_connection, create_duckdb_database, create_duckdb_result

db, conn, result = create_duckdb_database(), create_duckdb_connection(), create_duckdb_result()
ducklib.duckdb_open(0, db.ctypes.data)
ducklib.duckdb_connect(db[0], conn.ctypes.data)
with c_string("SELECT * FROM no_such_table") as sql_p:
    rc = ducklib.duckdb_query(conn[0], sql_p, result.ctypes.data)
if rc == ducklib.DuckDBError:
    message = get_str_from_p_as_int(ducklib.duckdb_result_error(result.ctypes.data))
    print(message.splitlines()[0])  # Catalog Error: Table with name no_such_table does not exist!
ducklib.duckdb_destroy_result(result.ctypes.data)
ducklib.duckdb_disconnect(conn.ctypes.data)
ducklib.duckdb_close(db.ctypes.data)
```

`duckdb_prepare_error` returns the message of a prepared statement that failed to prepare. DuckDB frees both messages
with their result or statement, so the caller does not free them.

### Using a connection from duckdb.connect()

`numbduck.pybridge.extract_connection_ptr(conn)` returns the C API handle of a connection that `duckdb.connect()`
opened. The bindings can then run queries on the database that Python code uses, and register functions on it, as the
examples below do. The handle is borrowed: `conn` owns the connection, and the handle is valid only while `conn` stays
open and referenced. After `conn.close()`, a binding that gets the handle reads freed memory.

`extract_connection_ptr` raises `TypeError` unless `conn` is exactly a `duckdb.DuckDBPyConnection`. It raises
`RuntimeError` if the connection is closed, or if numbduck cannot confirm that the bindings and the duckdb module run
the same DuckDB version.

Fetch the rows that Python still needs from a connection's open result before a binding runs a query or prepares a
statement on that connection. What happens otherwise depends on the duckdb version. On duckdb 1.x, the binding's call
succeeds, and the open result loses rows without an error or a warning. The duckdb module reads a result in chunks of up
to 2,048 rows, and afterwards the open result returns only the rest of the chunk that its last fetch read. After
`fetchmany(10)` on a 100,000-row result, for example, `fetchall()` returns 2,038 rows. A result that Python never
fetched returns no row. On duckdb 2.0, DuckDB refuses the call: the binding returns `DuckDBError`, and the open result
keeps every row. The error message, from `duckdb_result_error` or `duckdb_prepare_error`, then reads:

```text
Resource In Use Error: connection has an open result; drain or destroy it before starting a new query
```

If the rest of the open result is not needed, `complete_open_result(conn)`, from `numbduck.utils`, ends it. On duckdb
2.0, the function calls `conn.complete()`, which runs the result to its end and discards every row that Python has not
fetched. On duckdb 1.x, the function does nothing, and the binding's query ends the result as described above.

`extract_connection_ptr` reads the handle from the memory layout of duckdb's Python connection object. The duckdb
package builds that object with pybind11 before duckdb 2.0 and with nanobind from 2.0 on, and a new duckdb release can
change the layout. The tests check it on the duckdb versions that [Compatibility](#compatibility) lists.

### Reading a result

`duckdb_fetch_chunk` returns the next chunk of a result, or NULL (0) when no rows are left, and the caller destroys
each chunk with `duckdb_destroy_data_chunk`. `duckdb_data_chunk_get_size` gives the number of rows in a chunk,
`duckdb_data_chunk_get_vector` returns the vector of one column, and `duckdb_vector_get_data` the address of its
values. The function below reads both columns of a 5,000-row result, chunk by chunk:

```python
import duckdb
import numpy
from numba import carray, njit
from numbox.utils.lowlevel import _cast_int_to_void_p, array_data_p, get_unicode_data_p

from numbduck import ducklib
from numbduck.duckdb_utils import create_duckdb_data_chunk, create_duckdb_result
from numbduck.pybridge import extract_connection_ptr


@njit
def column(chunk_p, i, dtype):
    """Return column i of a data chunk as a numpy array of dtype that views DuckDB's memory."""
    n = ducklib.duckdb_data_chunk_get_size(chunk_p)
    data_p = ducklib.duckdb_vector_get_data(ducklib.duckdb_data_chunk_get_vector(chunk_p, i))
    return carray(_cast_int_to_void_p(data_p), (n,), dtype=dtype)


@njit(nogil=True)
def sum_columns(conn_p):
    result = create_duckdb_result()
    chunk = create_duckdb_data_chunk()
    sql = get_unicode_data_p("SELECT range, range / 2 FROM range(5000)")
    rows, chunks, total_i, total_x = 0, 0, 0, 0.0
    if ducklib.duckdb_query(conn_p, sql, array_data_p(result)) == ducklib.DuckDBSuccess:
        while True:
            chunk[0] = ducklib.duckdb_fetch_chunk((result[0], result[1], result[2], result[3], result[4], result[5]))
            if chunk[0] == 0:
                break
            i = column(chunk[0], 0, numpy.int64)
            x = column(chunk[0], 1, numpy.float64)
            rows += len(i)
            chunks += 1
            total_i += i.sum()
            total_x += x.sum()
            ducklib.duckdb_destroy_data_chunk(array_data_p(chunk))
    ducklib.duckdb_destroy_result(array_data_p(result))
    assert chunk[0] == 0 and result[0] == 0
    return rows, chunks, total_i, total_x


conn = duckdb.connect()
print(sum_columns(extract_connection_ptr(conn)))  # (5000, 3, 12497500, 6248750.0)
```

`range` is a `BIGINT`, and `range / 2` is a `DOUBLE`, because DuckDB's `/` returns a `DOUBLE` for two integers. A
vector's data is an array of the C type that DuckDB's [table of native types][c-native-types] gives for the column's
type, such as `int64` for a `BIGINT`, `int32` for an `INTEGER` and `float64` for a `DOUBLE`. A `VARCHAR` value is a
16-byte `duckdb_string_t`, which holds a short string inline and a longer one by pointer, as DuckDB's
[description of strings][c-strings] shows. A row that is NULL holds an undefined value, and the vector's validity mask
marks such rows ([NULL input](#null-input)).

[c-native-types]: https://duckdb.org/docs/2.0/clients/c/vector.html#primitive-types
[c-strings]: https://duckdb.org/docs/2.0/clients/c/vector.html#strings

### Prepared statements

A prepared statement runs one query many times with different parameter values. `duckdb_prepare` creates the
statement, a `duckdb_bind_*` function binds a value to each parameter, and `duckdb_execute_prepared` runs the
statement and writes its result, as `duckdb_query` does. Parameters are numbered from 1, as `$1` is in the SQL text.
`duckdb_bind_int64` binds an `int64`, `duckdb_bind_double` a `float64`, and `duckdb_bind_varchar` the address of text
([Text](#text)); the [C API reference][c-api-reference] lists the others. A value stays bound until it is bound again.
The function below prepares one statement and runs it for three divisors:

```python
import duckdb
import numpy
from numba import carray, njit
from numbox.utils.lowlevel import _cast_int_to_void_p, array_data_p, get_unicode_data_p

from numbduck import ducklib
from numbduck.duckdb_utils import create_duckdb_data_chunk, create_duckdb_prepared_statement, create_duckdb_result
from numbduck.pybridge import extract_connection_ptr


@njit(nogil=True)
def count_multiples(conn_p, divisors):
    counts = numpy.full(len(divisors), -1)
    statement = create_duckdb_prepared_statement()
    result = create_duckdb_result()
    chunk = create_duckdb_data_chunk()
    sql = get_unicode_data_p("SELECT count(*) FROM range(1000) WHERE range % $1 = 0")
    if ducklib.duckdb_prepare(conn_p, sql, array_data_p(statement)) == ducklib.DuckDBSuccess:
        for k in range(len(divisors)):
            if (ducklib.duckdb_bind_int64(statement[0], 1, divisors[k]) == ducklib.DuckDBSuccess
                    and ducklib.duckdb_execute_prepared(statement[0], array_data_p(result)) == ducklib.DuckDBSuccess):
                fields = (result[0], result[1], result[2], result[3], result[4], result[5])
                chunk[0] = ducklib.duckdb_fetch_chunk(fields)
                data_p = ducklib.duckdb_vector_get_data(ducklib.duckdb_data_chunk_get_vector(chunk[0], 0))
                counts[k] = carray(_cast_int_to_void_p(data_p), (1,), dtype=numpy.int64)[0]
                ducklib.duckdb_destroy_data_chunk(array_data_p(chunk))
            ducklib.duckdb_destroy_result(array_data_p(result))
    ducklib.duckdb_destroy_prepare(array_data_p(statement))
    assert chunk[0] == 0 and result[0] == 0 and statement[0] == 0
    return counts


conn = duckdb.connect()
print(count_multiples(extract_connection_ptr(conn), numpy.array([2, 3, 7])))  # [500 334 143]
```

`duckdb_prepare` creates a statement even when it fails, to hold the error message, so the function destroys the
statement on every path. Destroying a zeroed buffer does nothing, so the loop can destroy the result after a failed
bind too. [online_scoring.py][online-scoring] runs a prepared lookup for each event in the
same way.

### Text

The C API reads a `char *` argument as UTF-8 text that ends with a NUL byte, unless the function also takes the text's
length, as `duckdb_bind_varchar_length` and `duckdb_create_varchar_length` do. `get_unicode_data_p(s)` returns the
address of a string's own data, where Python and numba store one, two or four bytes per character, depending on the
widest character in the string. Those bytes are UTF-8 only for ASCII text. With one byte per character, the
non-ASCII characters are Latin-1 bytes, not UTF-8, and DuckDB rejects them or reads other characters. With two or four
bytes per character, the extra bytes of an ASCII character are zero, and the first zero byte ends the text for DuckDB.
Either way, DuckDB receives the wrong text, whether it is a file name, a table or function name, SQL text or a bound
value.

In Python code, pass text through numbox's [`c_string`][c-string], which encodes it as UTF-8 and keeps the bytes alive
for the `with` block. Jitted code cannot use `c_string`. Encode the text in Python into a numpy `uint8` array that ends
with a NUL byte, pass the array in, and take its address with `array_data_p`:

```python
import duckdb
import numpy
from numba import njit
from numbox.utils.lowlevel import array_data_p

from numbduck import ducklib
from numbduck.pybridge import extract_connection_ptr


def utf8(text):
    """Return the UTF-8 bytes of text, followed by a NUL byte, as a numpy uint8 array."""
    return numpy.frombuffer(text.encode("utf-8") + b"\x00", dtype=numpy.uint8)


@njit(nogil=True)
def run(conn_p, sql):
    return ducklib.duckdb_query(conn_p, array_data_p(sql), 0) == ducklib.DuckDBSuccess


conn = duckdb.connect()
print(run(extract_connection_ptr(conn), utf8("CREATE TABLE cities AS SELECT 'Z\u00fcrich' AS name")))  # True
print(conn.execute("SELECT name = 'Z\u00fcrich', length(name) FROM cities").fetchall())  # [(True, 6)]
```

`duckdb_query` accepts `0` for the result when the caller needs only the return code. For ASCII text in jitted code,
a string literal passed through `get_unicode_data_p` is enough, as in the quick start.

A string that DuckDB returns, such as an error message, is a `char *` too. numbox's
[`get_str_from_p_as_int(p)`][get-str-from-p] copies it into a `str` with one character per byte. The function therefore
returns the UTF-8 bytes of a non-ASCII text undecoded, and `s.encode("latin-1").decode("utf-8")` recovers the text from
such a result `s`.

[c-string]: https://goykhman.github.io/numbox/numbox.utils.html#numbox.utils.cstrings.c_string
[get-str-from-p]: https://goykhman.github.io/numbox/numbox.utils.html#numbox.utils.lowlevel.get_str_from_p_as_int
[online-scoring]: https://github.com/Goykhman/numbduck/blob/main/examples/online_scoring.py

### Destroying what you create

DuckDB allocates memory for each handle and result, and only the matching C function frees it. The bindings free
nothing on their own. DuckDB's [C API reference][c-api-reference] names the function for each case:

- `duckdb_destroy_result` for every result that `duckdb_query` or `duckdb_execute_prepared` filled, even when the query
  failed;
- `duckdb_destroy_data_chunk` for every chunk that `duckdb_fetch_chunk` returned;
- `duckdb_destroy_prepare` for every statement that `duckdb_prepare` created, even when it failed to prepare;
- `duckdb_destroy_logical_type`, `duckdb_destroy_value`, `duckdb_destroy_scalar_function` and the other
  `duckdb_destroy_*` functions for what the matching `duckdb_create_*` function returned;
- `duckdb_free` for a string that DuckDB allocated for the caller, as `duckdb_get_varchar` and
  `duckdb_value_to_string` do;
- `duckdb_disconnect` for a connection, and then `duckdb_close` for its database.

## Scalar functions

DuckDB runs a scalar function as a C callback with three arguments: a `duckdb_function_info`, a `duckdb_data_chunk`
of input rows, and the `duckdb_vector` for the output. The callback writes one value per row into the output vector.
numba's [`@cfunc`][cfunc] compiles a Python function into such a callback, and its `.address` is the C function
pointer that DuckDB takes. Every pointer argument is an `intp`, as for the bindings.

A callback cannot mark a row of its output NULL. numbduck does not bind the C functions that do it,
`duckdb_vector_ensure_validity_writable` and `duckdb_validity_set_row_invalid`, so DuckDB returns whatever value the
callback writes, for every row. [NULL input](#null-input) describes what a callback can write instead.

The function below computes the length of the hypotenuse of a right triangle from its two sides:

```python
import math

import duckdb
import numpy
from numba import carray, cfunc, njit
from numba import types as nb_types
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p

from numbduck import ducklib
from numbduck.pybridge import extract_connection_ptr


@njit
def doubles(vector, n):
    """Return the n float64 values of a DuckDB vector, as a numpy array that views them."""
    return carray(_cast_int_to_void_p(ducklib.duckdb_vector_get_data(vector)), (n,), dtype=numpy.float64)


@njit
def is_valid(validity, i):
    """Return whether row i holds a value, given its vector's validity mask; a mask of 0 means no NULLs."""
    return validity == 0 or ducklib.duckdb_validity_row_is_valid(validity, i) != 0


@njit
def hypotenuse(info, chunk, output):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    x_vector = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    y_vector = ducklib.duckdb_data_chunk_get_vector(chunk, 1)
    x_validity = ducklib.duckdb_vector_get_validity(x_vector)
    y_validity = ducklib.duckdb_vector_get_validity(y_vector)
    x, y, out = doubles(x_vector, n), doubles(y_vector, n), doubles(output, n)
    for i in range(n):
        if is_valid(x_validity, i) and is_valid(y_validity, i):
            out[i] = math.sqrt(x[i] * x[i] + y[i] * y[i])
        else:
            out[i] = math.nan


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def hypotenuse_cb(info, chunk, output):
    hypotenuse(info, chunk, output)


conn = duckdb.connect()
function = ducklib.duckdb_create_scalar_function()
with c_string("hypotenuse") as name_p:
    ducklib.duckdb_scalar_function_set_name(function, name_p)
double = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
ducklib.duckdb_scalar_function_add_parameter(function, double)
ducklib.duckdb_scalar_function_add_parameter(function, double)
ducklib.duckdb_scalar_function_set_return_type(function, double)
ducklib.duckdb_scalar_function_set_function(function, hypotenuse_cb.address)
rc = ducklib.duckdb_register_scalar_function(extract_connection_ptr(conn), function)
double_buf = numpy.array([double], dtype=numpy.intp)
function_buf = numpy.array([function], dtype=numpy.intp)
ducklib.duckdb_destroy_logical_type(double_buf.ctypes.data)
ducklib.duckdb_destroy_scalar_function(function_buf.ctypes.data)
assert rc == ducklib.DuckDBSuccess
print(conn.execute("SELECT x, hypotenuse(x, 4) FROM (VALUES (3), (NULL)) t(x)").fetchall())
```

The script prints `[(3, 5.0), (None, nan)]`. `hypotenuse`, an ordinary `@njit` function, computes the output, and the
`@cfunc` only wraps it in a C function with the signature that DuckDB expects. numbduck's tests and examples split every
callback this way, because Python code can then call the body, as a test does. A `@cfunc` can also hold the body itself.
The first argument, `info`, is the `duckdb_function_info` of the call. This example does not use it, but
`duckdb_scalar_function_get_extra_info` and `duckdb_scalar_function_set_error`, below, take it.

The registration names the function, adds a type for each parameter, sets the return type and the callback, and
registers the function on a connection. `duckdb_create_logical_type` creates a logical type, DuckDB's description of a
SQL type such as `DOUBLE`. DuckDB keeps copies of the types and the function that it registers. The script then
destroys both through buffers, as [Buffers for handles](#buffers-for-handles) describes, so that neither leaks.

`duckdb_scalar_function_set_extra_info` attaches an address to the function, such as a numpy array's.
The callback reads it back with `duckdb_scalar_function_get_extra_info(info)`.

A function set registers several functions under one SQL name, one for each list of parameter types.
`duckdb_create_scalar_function_set` takes the set's name, and `duckdb_add_scalar_function_to_set` adds each function.
Each function in the set also needs a name of its own, set with `duckdb_scalar_function_set_name`. SQL calls every
function of the set by the set's name, but `duckdb_register_scalar_function_set` returns `DuckDBError` if a function
in the set has no name.

[cfunc]: https://numba.readthedocs.io/en/stable/user/cfunc.html

### NULL input

DuckDB passes the NULL rows of a column to the callback, and the data array holds an undefined value in each of them.
`duckdb_vector_get_validity` returns the vector's validity mask, a bitmap with one bit per row, or 0 when the vector
has no NULLs. `duckdb_validity_row_is_valid` reads the bit of one row, as `is_valid` does above. DuckDB's
[description of NULL values][c-null-values] gives the layout of the mask.

The output cannot be NULL, so a callback writes a value for a NULL row too. The example writes NaN, which DuckDB treats
as a value: `count(hypotenuse(x, 4))` counts it. A constant NULL argument is different. For `hypotenuse(NULL, 4)`,
DuckDB returns NULL without calling the callback, unless the function was registered with
`duckdb_scalar_function_set_special_handling`.

[c-null-values]: https://duckdb.org/docs/2.0/clients/c/vector.html#null-values

### Errors in a callback

An exception that escapes a callback does not reach DuckDB. numba prints it to standard error, after the line
`Exception ignored in:`, and the query continues with the output that the callback wrote before the exception. To fail
the query, call `duckdb_scalar_function_set_error(info, message_p)` with the address of a NUL-terminated message, and
return. Python code then receives a `duckdb.InvalidInputException` whose text is `Invalid Input Error:` followed by the
message. numba needs the GIL to print an exception, which matters for a query that jitted code runs, as
[Threads and the GIL](#threads-and-the-gil) explains.

## Aggregate functions

An aggregate function computes one value for each group of rows. DuckDB runs it as six callbacks:

| Callback | `@cfunc` signature | DuckDB calls it to |
|---|---|---|
| `state_size` | `uint64(intp)` | learn how many bytes of state a group needs |
| `init` | `void(intp, intp)` | initialize one group's state |
| `update` | `void(intp, intp, intp)` | add a chunk of input rows to the states of their groups |
| `combine` | `void(intp, intp, intp, uint64)` | merge each state of one array into a state of another |
| `finalize` | `void(intp, intp, intp, uint64, uint64)` | write the result of each state into the output vector |
| `destroy` | `void(intp, uint64)` | free an array of states; optional |

`duckdb_aggregate_function_set_functions` registers the first five, and `duckdb_aggregate_function_set_destructor`
registers `destroy`. DuckDB allocates each group's state, as many bytes as `state_size` returns, and passes `init` the
address of one new state. The other callbacks receive arrays of state addresses. `update` receives one address for
each input row, the state of the row's group, and `combine`, `finalize` and `destroy` receive one for each group. A
state of a few numbers fits in its bytes, and the `my_sum` aggregate of [test_ducklib.py][tests] keeps its sum there.

State that grows, such as all the values of a group, needs memory that numba allocates. numbduck's tests and examples
keep it in a [structref][structref], a mutable struct whose memory numba's runtime allocates and reference-counts.
The state then holds the structref's address, and numbox's [meminfo helpers][numbox-meminfo] keep the structref alive
from one callback to the next:

- `init` creates the structref, and `export_meminfo` returns its address. That call also takes a reference to the
  structref, which the state holds from then on.
- `update`, `combine` and `finalize` call `borrow_structref(struct_type, address)`, which returns the structref with
  a reference of its own. numba releases that reference when the callback no longer uses the structref.
- `destroy` calls `release_meminfo(address)` for each state, which drops the state's reference, and numba frees the
  structref and what it holds. Without a `destroy`, every group's structref leaks.

The aggregate below computes the median of each group. Its structref holds a numba typed list of the group's values.
`word(p, i)` reads the i-th 64-bit word at an address, so `word(word(states, i), 0)` reads the address of the i-th
state, and then the structref's address that the state holds:

```python
import duckdb
import numpy
from numba import carray, cfunc, njit
from numba import types as nb_types
from numba.experimental import structref
from numba.typed import List
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p
from numbox.utils.meminfo import borrow_structref, export_meminfo, release_meminfo

from numbduck import ducklib
from numbduck.pybridge import extract_connection_ptr


@structref.register
class ValuesType(nb_types.StructRef):
    pass


class Values(structref.StructRefProxy):
    def __new__(cls, items):
        return structref.StructRefProxy.__new__(cls, items)


structref.define_proxy(Values, ValuesType, ["items"])
values_type = ValuesType([("items", nb_types.ListType(nb_types.float64))])


@njit
def word(p, i):
    """Return the i-th 64-bit word at the address p."""
    return carray(_cast_int_to_void_p(p), (i + 1,), dtype=numpy.intp)[i]


@njit
def held(states, i):
    """Return the structref that the i-th state of the array states holds."""
    return borrow_structref(values_type, word(word(states, i), 0))


@cfunc(nb_types.uint64(nb_types.intp))
def state_size(info):
    return 8


@cfunc(nb_types.void(nb_types.intp, nb_types.intp))
def init(info, state):
    carray(_cast_int_to_void_p(state), (1,), dtype=numpy.intp)[0] = export_meminfo(
        Values(List.empty_list(nb_types.float64)))


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def update(info, chunk, states):
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    vector = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    x = carray(_cast_int_to_void_p(ducklib.duckdb_vector_get_data(vector)), (n,), dtype=numpy.float64)
    validity = ducklib.duckdb_vector_get_validity(vector)
    for i in range(n):
        if validity == 0 or ducklib.duckdb_validity_row_is_valid(validity, i):
            try:
                held(states, i).items.append(x[i])
            except Exception:
                pass


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp, nb_types.uint64))
def combine(info, source, target, count):
    for i in range(count):
        try:
            held(target, i).items.extend(held(source, i).items)
        except Exception:
            pass


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp, nb_types.uint64, nb_types.uint64))
def finalize(info, source, result, count, offset):
    out = carray(_cast_int_to_void_p(ducklib.duckdb_vector_get_data(result)), (offset + count,), dtype=numpy.float64)
    for i in range(count):
        try:
            out[offset + i] = numpy.median(numpy.asarray(held(source, i).items))
        except Exception:
            out[offset + i] = numpy.nan


@cfunc(nb_types.void(nb_types.intp, nb_types.uint64))
def destroy(states, count):
    for i in range(count):
        release_meminfo(word(word(states, i), 0))


conn = duckdb.connect()
function = ducklib.duckdb_create_aggregate_function()
with c_string("jit_median") as name_p:
    ducklib.duckdb_aggregate_function_set_name(function, name_p)
double = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
ducklib.duckdb_aggregate_function_add_parameter(function, double)
ducklib.duckdb_aggregate_function_set_return_type(function, double)
ducklib.duckdb_aggregate_function_set_functions(
    function, state_size.address, init.address, update.address, combine.address, finalize.address)
ducklib.duckdb_aggregate_function_set_destructor(function, destroy.address)
rc = ducklib.duckdb_register_aggregate_function(extract_connection_ptr(conn), function)
double_buf = numpy.array([double], dtype=numpy.intp)
function_buf = numpy.array([function], dtype=numpy.intp)
ducklib.duckdb_destroy_logical_type(double_buf.ctypes.data)
ducklib.duckdb_destroy_aggregate_function(function_buf.ctypes.data)
assert rc == ducklib.DuckDBSuccess
conn.execute("CREATE TABLE t AS SELECT range % 3 AS g, "
             "CASE WHEN range % 10 = 0 THEN NULL ELSE range * 7 % 100 END::DOUBLE AS x FROM range(1000)")
print(conn.execute("SELECT g, jit_median(x), median(x) FROM t GROUP BY g ORDER BY g").fetchall())
```

The script prints each group's median twice, first from the aggregate and then from DuckDB's own `median`, and the two
agree. The example writes each callback directly as a `@cfunc`. `examples/irr.py` and the tests give each one an `@njit`
body, as for a scalar function. Three of the example's choices apply to any aggregate with a structref state:

- DuckDB passes the NULL rows of a column to `update`, with or without
  `duckdb_aggregate_function_set_special_handling`. `update` skips them by their validity mask.
- If a function that a callback calls raises while the callback holds a borrowed structref, and the exception
  escapes the callback, numba does not release that reference, and the structref leaks. A raise in the callback's
  own body still releases it. Each callback therefore wraps each borrow and its work in `try` and `except`.
- A structref whose proxy class numbox's `make_structref` generates must be defined, with its numba type, in a
  module that the program imports, not in the script that runs. numba caches the code that `make_structref`
  generates, whether or not any function of the program sets `cache=True`, under a key that does not include the
  importing module. A type defined in the running script therefore does not match the cached one in the next run.
  [run_irr.py][run-irr] imports [irr.py][irr] for this reason. The example writes its proxy class itself, with
  numba's `structref.StructRefProxy`, which has no such limit: with `cache=True` on its callbacks, a second run of
  the script loads them from the cache.

[test/test_ducklib.md][design-note] explains why the reference counts stay balanced across DuckDB's callbacks, on every
numba version that numbduck supports.

[design-note]: https://github.com/Goykhman/numbduck/blob/main/test/test_ducklib.md
[irr]: https://github.com/Goykhman/numbduck/blob/main/examples/irr.py
[numbox-meminfo]: https://goykhman.github.io/numbox/numbox.utils.html#numbox.utils.meminfo.export_meminfo
[run-irr]: https://github.com/Goykhman/numbduck/blob/main/examples/run_irr.py
[structref]: https://numba.readthedocs.io/en/stable/extending/high-level.html#defining-a-structref
[tests]: https://github.com/Goykhman/numbduck/blob/main/test/test_ducklib.py

## Threads and the GIL

Compile every jitted function that runs a query with `@njit(nogil=True)`, as the examples above do. A query can hang
if the function that runs it holds the GIL.

DuckDB runs a query on its worker threads, and some work there needs the GIL. A Python UDF that `conn.create_function`
registered on the connection needs it. So does a `@cfunc` callback that raises, because numba takes the GIL to print
the exception. If the thread that waits for the query holds the GIL, such a worker cannot run until that thread
releases the GIL, and the query can hang. A jitted function compiled without `nogil=True` holds the GIL for as long
as it runs, and a binding called inside it runs under that function's `nogil` setting.

numbduck compiles every binding and buffer allocator with numba's [`nogil=True`][nogil], unless `NUMBDUCK_JIT_OPTIONS`
turns `nogil` off ([Compilation and the cache](#compilation-and-the-cache)). A binding called from Python therefore
releases the GIL while its C function runs. Other Python threads run meanwhile, and a query that needs the GIL on a
worker completes.

Releasing the GIL does not make the bindings thread-safe. A DuckDB connection is locked while it runs a query, so
threads that share a connection run their queries one at a time. DuckDB's [connection documentation][c-connect] advises
one connection per thread. `online_scoring.py` gives each thread its own connection.

DuckDB can also call a callback from several worker threads at once. Its C API reference says, for example, that it
calls the [init callback of a scalar function][set-init] once for each worker thread that executes the function. A
callback that writes memory other than its own output or its own group's state must therefore synchronize.

[c-connect]: https://duckdb.org/docs/2.0/clients/c/connect.html
[nogil]: https://numba.readthedocs.io/en/stable/user/jit.html#nogil
[set-init]: https://duckdb.org/docs/2.0/clients/c/api.html#duckdb_scalar_function_set_init

## Compilation and the cache

numba compiles numbduck's bindings when the program first imports `numbduck.ducklib`. It compiles each buffer allocator
of `numbduck.duckdb_utils` when the program first calls it. By default, numba saves the compiled code in its on-disk
[cache][numba-cache], and a later process loads the code instead of compiling it again. The first import compiles
every binding whose C function the loaded libduckdb exports, all 193 on duckdb 1.5 or later. It took about 8.5
seconds with an empty cache and under a second with a full one, measured on an AMD Ryzen 5 7640HS with Python 3.12
and numba 0.65.1.

The environment variable `NUMBDUCK_JIT_OPTIONS` sets the [`@njit` options][jit-options] that numbduck compiles with,
as a JSON object. If it is unset, the options are `{"cache": true, "nogil": true}`. Each key of the object replaces
the default of that name or adds another option, so this setting turns the cache off and keeps `nogil`:

```bash
export NUMBDUCK_JIT_OPTIONS='{"cache": false}'
```

numbduck caches only under the default compile options, and any other option turns the cache off. numba's cache does
not record the options that code was compiled with, so code cached under one set of options would load in a process
that asks for another. `{"fastmath": true}` or `{"nogil": false}`, for example, turns the cache off. If the object also
sets `"cache": true`, numbduck issues a `RuntimeWarning` that it compiles without the cache.

numbduck reads `NUMBDUCK_JIT_OPTIONS` once, when the program first imports `numbduck.ducklib` or
`numbduck.duckdb_utils`. It raises `ValueError` at that import if the value is empty or not a JSON object, or if the
object has a key that is not a numba compile option. A compile option that `@njit` does not take, such as
`writable_args` of `@guvectorize`, passes that check. numba then refuses it with a `KeyError` when it compiles the
first binding or buffer allocator: at the import of `numbduck.ducklib`, or at the first call of an allocator of
`numbduck.duckdb_utils`. Unset the variable to get the defaults.

The two bignum bindings, `duckdb_create_bignum` and `duckdb_get_bignum`, never use the cache. Each calls the C
function that the loaded DuckDB exports, the bignum name or the older varint name, and numba's cache key does not
record that choice.

numba writes the cache beside numbduck's modules if it can, and otherwise in the user's cache directory.
[`NUMBA_CACHE_DIR`][numba-cache-dir] overrides both. numba reads `NUMBA_CACHE_DIR` when numba is imported, so set it,
and `NUMBDUCK_JIT_OPTIONS`, before the process imports either numba or numbduck. numbox, which numbduck imports,
compiles its own helpers under its own variable, [`NUMBOX_JIT_OPTIONS`][numbox-jit-options].

[jit-options]: https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#numba.jit
[numba-cache]: https://numba.readthedocs.io/en/stable/user/jit.html#cache
[numba-cache-dir]: https://numba.readthedocs.io/en/stable/reference/envvars.html#envvar-NUMBA_CACHE_DIR
[numbox-jit-options]: https://goykhman.github.io/numbox/numbox.core.configurations.html#setting-the-options

### If numbduck warns that it compiles without a cache

When the program first imports `numbduck.ducklib` or `numbduck.duckdb_utils`, numbduck checks whether numba can write
a cache for its modules. If numba cannot, numbduck compiles without a cache and issues one `RuntimeWarning`. The
warning begins with `numba cannot cache numbduck here` and ends with the remedy for its case. In every case,
`NUMBDUCK_JIT_OPTIONS='{"cache": false}'` turns the cache off and silences the warning.

These are the common cases, and numbox's guide to [reading the warning][numbox-warning] explains each one:

- numbduck is installed in a read-only location, and the user's cache directory cannot be written either. Set
  `NUMBA_CACHE_DIR` to a writable directory.
- The path of the cache directory is so long that numba's cache files, whose names are long, would exceed the file
  system's limit on a path. Set a shorter `NUMBA_CACHE_DIR`.
- numbduck is imported from an `.egg`, a `.whl` or a `.pyz` archive. numba caches no module imported from such an
  archive, and `NUMBA_CACHE_DIR` has no effect on it. Install numbduck with its source files on disk.
- numbduck is imported from a `.zip`, and the user's cache directory cannot be written. numba caches a module from a
  `.zip` in that directory only, so it must be writable.
- numbduck is imported from a `.zip` that holds its modules as `.pyc` files alone, under numba 0.68 or later. From
  0.68, numba reads each module's source from the archive, and such an archive has none.
- numbduck is imported from any `.zip` on Windows under numba 0.68.0 ([numba issue 10889][numba-10889], fixed in
  0.68.1).

[numba-10889]: https://github.com/numba/numba/issues/10889
[numbox-warning]: https://goykhman.github.io/numbox/numbox.core.configurations.html#reading-the-warning

## macOS, NUMBDUCK_LIBDUCKDB and import errors

numbduck looks for the C API when `numbduck.ducklib` is first imported. It first loads the shared library of the
duckdb wheel. The wheels for Linux and Windows export the C API, and numbduck uses that library. The macOS wheels of
some duckdb versions export only the entry points of the Python module ([duckdb-python PR 81][pr81]):

| duckdb | The macOS wheels export the C API |
|---|---|
| 1.2.2 to 1.4.0 | Yes |
| 1.4.1 to 1.5.2 | No |
| 1.5.3 to 2.0.0 | Yes |

### A standalone libduckdb

If the wheel does not export the C API, numbduck loads a standalone libduckdb instead. Before it searches, it deletes
the file `~/.numbduck/lib/libduckdb.dylib` if it exists, even when `NUMBDUCK_LIBDUCKDB` names that file, so keep a
library of your own elsewhere. numbduck then tries the first of these files that exists, and only that one:

1. the file that the environment variable `NUMBDUCK_LIBDUCKDB` names;
2. `~/.numbduck/lib/<version>/libduckdb.dylib`, where numbduck saves a library that it downloads, with `<version>` the
   version of the duckdb package;
3. `/opt/homebrew/lib/libduckdb.dylib` or `/usr/local/lib/libduckdb.dylib`, where Homebrew installs DuckDB.

The file must be a library that exports the C API and reports the same DuckDB version as the wheel, through
`duckdb_library_version`. A file that is not a loadable library raises `OSError`. numbduck treats a library that loads
but does not export the C API as if it had found no file. A library of another DuckDB version raises `RuntimeError`,
because numbduck passes it connections that the duckdb module opened
([Using a connection from duckdb.connect()](#using-a-connection-from-duckdbconnect)), and another version may lay a
connection out differently.

With a standalone library, two copies of DuckDB run in the process. The duckdb module uses the wheel's copy, and the
bindings use the standalone one. `extract_connection_ptr` then returns a handle that the wheel's copy created, the
bindings pass it to the standalone copy, and numbduck checks only that the two report the same version. Of the macOS
wheels, only those of duckdb 1.4.1 to 1.5.2 need the second copy.

### The download on macOS

On macOS, if no file provides the C API, numbduck offers to download the standalone library of the installed duckdb
version, `libduckdb-osx-universal.zip` from [DuckDB's releases][duckdb-releases]. It saves the library at the second
path of the list above. numbduck first asks at the terminal whether to download it, and `NUMBDUCK_LIBDUCKDB_DOWNLOAD=1`
lets it download without asking. Without that setting, the import fails with a `RuntimeError` if the user declines, or
if standard input is not a terminal, as in a CI job. The message names `NUMBDUCK_LIBDUCKDB_DOWNLOAD` and
`NUMBDUCK_LIBDUCKDB`.

The download comes over HTTPS, and numbduck checks the library's exports and version, but not a checksum. DuckDB's
2.0.0 release has no `libduckdb-osx-universal.zip`, so the download fails for duckdb 2.0.0. Its macOS wheels export
the C API, so numbduck does not need the download there.

### Off macOS

Off macOS, numbduck downloads nothing, and `NUMBDUCK_LIBDUCKDB` is the only way to supply a library. If no file provides
the C API, the import fails with a `RuntimeError` whose message begins `numbduck could not find the DuckDB C API` and
also names the variable.

[duckdb-releases]: https://github.com/duckdb/duckdb/releases
[pr81]: https://github.com/duckdb/duckdb-python/pull/81

## Examples

The scripts in [`examples/`][examples] each generate their own data. [examples/README.md][examples-readme] says how to
run them and how to read what they print.

- [haversine.py][haversine] counts the customers within 50 km of a store. It computes each distance three ways, with a
  Python scalar UDF, a PyArrow UDF and a JIT UDF, and checks that they agree. It shows the cost of a Python call for
  each row, and of an intermediate Arrow array for each step of the formula.
- [fraud_score.py][fraud-score] scores transactions with branching rules, in the same three ways. A chain of
  PyArrow's `if_else` is the vectorized way to write the rules without numbduck, and the JIT UDF runs them in one
  compiled loop.
- [online_scoring.py][online-scoring] looks up one row for each event and scores it, two ways: in a Python loop over
  `conn.execute`, and in an `@njit(nogil=True)` loop over a prepared statement. It reports the latency per event, and
  how each loop scales on 1, 2, 4 and 8 threads.
- [irr.py][irr], run through [run_irr.py][run-irr], is an aggregate that computes an internal rate of return. Its
  structref state holds each group's cash flows. It is a complete [aggregate](#aggregate-functions) with a structref
  state, and it has no version without numbduck to compare with.

The scripts print the ratios that their own run measured. Those depend on the machine and on the duckdb version, and
this guide quotes none.

[examples]: https://github.com/Goykhman/numbduck/tree/main/examples
[examples-readme]: https://github.com/Goykhman/numbduck/blob/main/examples/README.md
[fraud-score]: https://github.com/Goykhman/numbduck/blob/main/examples/fraud_score.py
[haversine]: https://github.com/Goykhman/numbduck/blob/main/examples/haversine.py

## Compatibility

| Dependency | Versions |
|---|---|
| Python | 3.10 or later |
| duckdb | 1.2.2 to 2.0.x |
| numbox | 0.7.8 or later |
| numba | 0.63.1 to 0.68.x, the range that numbox declares |

[pyproject.toml][pyproject] declares the ranges of Python, duckdb and numbox, and numbox declares the range of numba.
CI runs the test suite on Linux x86-64, Linux ARM and Windows, with the newest numbox and numba that the ranges allow.
It runs duckdb 1.2.2 on Python 3.10 to 3.12, duckdb 1.5.6 on Python 3.10 to 3.14, and duckdb 2.0.0 on Python 3.12 to
3.14. CI does not run on macOS.

duckdb 2.0.0 requires Python 3.11 or later, so pip installs duckdb 1.5.6 on Python 3.10. duckdb 1.2.2 has no wheel for
Python 3.14.

[pyproject]: https://github.com/Goykhman/numbduck/blob/main/pyproject.toml

## License

MIT, with a citation clause: see [LICENSE][license].

[license]: https://github.com/Goykhman/numbduck/blob/main/LICENSE
