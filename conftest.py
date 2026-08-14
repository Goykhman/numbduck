import re

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip benchmark-marked tests unless the mark expression names ``benchmark``.

    Plain ``pytest`` skips the benchmarks; ``pytest -m benchmark`` runs only the
    benchmarks. An expression that does not name ``benchmark``, such as
    ``-m 'not slow'``, still skips them even though pytest would otherwise
    select them.

    ``addopts = -m 'not benchmark'`` cannot express that. ``-m`` is a store
    option, so an expression given on the command line replaces the default
    outright instead of combining with it, and ``pytest -m 'not slow'`` would
    then run the benchmarks.

    The expression is matched on a word boundary so that an unrelated marker
    whose name merely contains ``benchmark`` does not suppress the skip.
    """
    if re.search(r"\bbenchmark\b", config.getoption("markexpr")):
        return
    skip_benchmark = pytest.mark.skip(reason="benchmark; run with -m benchmark")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
