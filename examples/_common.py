"""Shared utilities for numbduck example scripts.

Intentionally tiny. If this file grows past ~80 lines or gains
DuckDB-specific knowledge, shrink it or inline its bits back into
the example files.
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
    """Print a one-line environment block. Every example output starts with this."""
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(
        f"  env: python {pyver}, duckdb {duckdb.__version__}, "
        f"numba {numba.__version__}, numpy {numpy.__version__}, "
        f"{platform.machine()}, {os.cpu_count()} cores"
    )


def time_median(fn, repeats: int = 3) -> float:
    """Run fn() `repeats` times under perf_counter; return median wall time in seconds.

    No auto-warmup — caller is responsible. We pick the median, not the min,
    to dampen the occasional outlier without hiding real variance.
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
    """Format a small table for stdout. alignments: list of '<', '>', '^'."""
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
    """True only when a and b are both floating-point NaN. Non-numbers are not NaN."""
    try:
        return math.isnan(a) and math.isnan(b)
    except (TypeError, ValueError):
        return False


def assert_results_match(*results, label: str) -> None:
    """Cross-check that all variants produced the same answer.

    Catches 'your fast variant is fast because it's wrong'. Allows numpy floats
    to compare equal at full precision; if you need tolerance, do it before calling.
    Two NaN scalars are treated as a match: under IEEE-754 NaN != NaN, so variants
    that legitimately all produce NaN would otherwise be flagged as a spurious
    mismatch. A NaN against any non-NaN value still fails.
    """
    if len(results) < 2:
        return
    first = results[0]
    for i, other in enumerate(results[1:], start=1):
        if first != other and not _both_nan(first, other):
            raise AssertionError(
                f"{label}: variant 0 produced {first!r} but variant {i} produced {other!r}"
            )


if __name__ == "__main__":
    print_env()
    demo = format_table(
        headers=["Variant", "Time", "Speedup"],
        rows=[
            ["Python", "1.000s", "1.0x"],
            ["JIT", "0.010s", "100.0x"],
        ],
        alignments=["<", ">", ">"],
    )
    print(demo)
    assert_results_match(42, 42, label="demo")
    print("  _common.py self-test OK")
