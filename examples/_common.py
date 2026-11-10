"""Helpers that the three benchmark scripts share.

``print_env`` prints the line that opens each script's output, ``time_median`` times a query, ``format_table`` lays
out the results, and ``assert_results_match`` checks that the variants of a benchmark agree. The module holds only
helpers that more than one script uses. Code that works with DuckDB itself stays in the script that needs it.
"""
import math
import os
import platform
import statistics
import sys
import time

import duckdb
import numba
import numpy


def print_env() -> None:
    """Print the Python, duckdb, numba and numpy versions, the machine type and the number of logical CPUs, on one line.

    Each benchmark script prints this line first, so that a saved output records where its numbers were measured.
    """
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(
        f"  env: python {pyver}, duckdb {duckdb.__version__}, "
        f"numba {numba.__version__}, numpy {numpy.__version__}, "
        f"{platform.machine()}, {os.cpu_count()} cores"
    )


def time_median(fn, repeats: int = 3) -> float:
    """Call ``fn()`` ``repeats`` times and return the median time of one call, in seconds.

    The clock is ``time.perf_counter``, which measures wall time. ``NUMBDUCK_BENCH_REPEATS``, if set, replaces
    ``repeats``. A value that is not an integer, or a count below 1, raises ``ValueError``. The function makes no
    warm-up call, so a caller that needs one makes it first. An occasional slow call has little effect on the median.
    Unlike the minimum, the median does not report the best case alone.
    """
    repeats = int(os.environ.get("NUMBDUCK_BENCH_REPEATS", repeats))
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    return statistics.median(timings)


def format_table(headers: list[str], rows: list[list[str]], alignments: list[str]) -> str:
    """Return ``headers`` and ``rows`` as lines of aligned text columns, ready to print.

    Every cell is a string. ``alignments`` holds one format alignment per column: ``"<"``, ``">"`` or ``"^"``. A column
    is as wide as its widest cell, columns are two spaces apart, and every line starts with two spaces. ``ValueError``
    is raised if ``alignments`` or a row has a length other than that of ``headers``.
    """
    if len(headers) != len(alignments):
        raise ValueError("headers and alignments must be the same length")
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("each row must have the same length as headers")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        return "  " + "  ".join(
            f"{c:{a}{w}s}" for c, a, w in zip(cells, alignments, widths)
        )
    out = [fmt_row(headers)]
    for row in rows:
        out.append(fmt_row(row))
    return "\n".join(out)


def _both_nan(a, b) -> bool:
    """Return True if ``a`` and ``b`` are both NaN.

    A value that ``math.isnan`` rejects, such as ``None``, counts as not NaN.
    """
    try:
        return math.isnan(a) and math.isnan(b)
    except (TypeError, OverflowError):
        return False


def assert_results_match(*results, label: str) -> None:
    """Raise ``AssertionError``, with ``label`` in the message, unless every result equals the first.

    The check catches a variant that is fast only because it returns a wrong result. It compares with ``==`` at full
    precision, so a caller whose variants may differ in the last bits of a float rounds them first, as
    ``online_scoring.py`` does. Two NaN values count as a match: under IEEE 754 ``NaN != NaN``, yet variants that all
    return NaN agree. A NaN compared with a value that is not NaN raises ``AssertionError``. The function returns
    without a check if it receives fewer than two results.
    """
    if len(results) < 2:
        return
    first = results[0]
    for i, other in enumerate(results[1:], start=1):
        if first != other and not _both_nan(first, other):
            raise AssertionError(
                f"{label}: variant 0 produced {first!r} but variant {i} produced {other!r}"
            )
