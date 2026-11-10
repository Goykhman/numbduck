import re

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip the tests marked ``benchmark`` unless the ``-m`` expression names ``benchmark``.

    A ``pytest`` run without ``-m`` skips the benchmarks, and ``pytest -m benchmark`` runs only them. An expression
    that does not name ``benchmark``, such as ``-m 'not slow'``, still skips them, although pytest would select them.

    A default of ``addopts = -m 'not benchmark'`` cannot do this. pytest keeps only the last ``-m`` that it is given,
    so an expression on the command line would replace the default instead of combining with it, and
    ``pytest -m 'not slow'`` would run the benchmarks.
    """
    # The expression must name benchmark as a whole word, so that a marker whose name only contains it does not prevent
    # the skip.
    if re.search(r"\bbenchmark\b", config.getoption("markexpr")):
        return
    skip_benchmark = pytest.mark.skip(reason="benchmark; run with -m benchmark")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
