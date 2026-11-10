"""Test that the numbduck package imports.

The package's ``__init__`` sets only ``__version__`` and imports none of numbduck's modules, so this test loads no
libduckdb and compiles nothing. ``test_ducklib.py`` imports the bindings, and ``test_no_cache_location.py`` imports
numbduck from archives and read-only installs.
"""
import numbduck


def test():
    assert numbduck
