# NumbDuck
Adapting DuckDB database API to the Numba JIT context.

Leverages [bindings](https://github.com/Goykhman/numbox/tree/main/numbox/core/bindings) toolkit in the [NumbOx](https://github.com/Goykhman/numbox) project.

Inspired by the [NumbSQL](https://github.com/cpcloud/numbsql) project.

## Examples

Runnable narrative-style examples comparing numbduck against the closest stock
DuckDB Python+Arrow approaches live in [`examples/`](examples/). Each script is
self-contained, generates its own data, and prints honest measured numbers.
