import pytest


def pytest_collection_modifyitems(config, items):
    """Skip benchmark-marked tests unless the mark expression selects them.

    Plain ``pytest`` skips the benchmarks; ``pytest -m benchmark`` runs only the
    benchmarks. Doing this here rather than with ``addopts = -m 'not benchmark'``
    keeps ``pytest -m benchmark`` working: pytest ANDs multiple ``-m`` values, so
    a default ``-m 'not benchmark'`` turned an explicit ``-m benchmark`` into
    ``not benchmark and benchmark`` and silently collected zero tests.
    """
    if "benchmark" in config.getoption("markexpr"):
        return
    skip_benchmark = pytest.mark.skip(reason="benchmark; run with -m benchmark")
    for item in items:
        if "benchmark" in item.keywords:
            item.add_marker(skip_benchmark)
