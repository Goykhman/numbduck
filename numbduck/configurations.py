import inspect
import json
import os
import warnings

from numba.core.options import DefaultOptions
from numbox.core.configurations import cache_remedy, check_cache_location, is_a_cache_error


# njit's accepted compilation options. Take the target-level options straight
# from numba (``DefaultOptions`` carries one attribute per njit option) so this
# stays in sync automatically, and add the dispatcher-level ``cache`` flag
# handled by numba.core.decorators.jit. Any other key reaches njit as an
# unrecognized kwarg and aborts ducklib import with a numba error that never
# names the env var, so unknown keys are rejected here instead.
_ALLOWED_JIT_OPTIONS = frozenset(
    name for name in vars(DefaultOptions) if not name.startswith("__")
) | {"cache"}

# nogil lets a binding called from Python release the GIL for the length of
# the C call. DuckDB runs a query on its worker threads, and a worker that needs
# the GIL (a Python UDF on the same connection, a numba @cfunc reporting an
# exception it swallowed) would otherwise wait for it forever while the calling
# thread waits for the workers.
_DEFAULT_JIT_OPTIONS = {"cache": True, "nogil": True}


def _changes_compile_options(options):
    return {k: v for k, v in options.items() if k != "cache"} != {
        k: v for k, v in _DEFAULT_JIT_OPTIONS.items() if k != "cache"}


def get_jit_options():
    """The jit options every binding and buffer allocator is compiled with.

    The defaults are ``{"cache": true, "nogil": true}``. NUMBDUCK_JIT_OPTIONS, a
    JSON object, overrides individual defaults and adds other njit options,
    e.g. export NUMBDUCK_JIT_OPTIONS='{"cache": false}' turns the disk cache
    off and keeps nogil.

    Only the default compile options use numba's disk cache. Its key holds no
    compile options, so an object compiled under one set would be served to a
    process running another, and one process under non-default options would
    poison the cache for every later default one. Any other option turns the
    cache off, with a RuntimeWarning when the cache was asked for explicitly.
    """
    as_str = os.environ.get("NUMBDUCK_JIT_OPTIONS")
    if as_str is None:
        return dict(_DEFAULT_JIT_OPTIONS)
    try:
        as_json = json.loads(as_str)
    except json.JSONDecodeError:
        raise ValueError("NUMBDUCK_JIT_OPTIONS must be valid JSON")
    if not isinstance(as_json, dict):
        raise ValueError('NUMBDUCK_JIT_OPTIONS must be a JSON object, e.g. {"cache": false}')
    unknown = set(as_json) - _ALLOWED_JIT_OPTIONS
    if unknown:
        raise ValueError(
            f"NUMBDUCK_JIT_OPTIONS contains unknown jit option(s) {sorted(unknown)}; "
            f"allowed keys are {sorted(_ALLOWED_JIT_OPTIONS)}"
        )
    options = {**_DEFAULT_JIT_OPTIONS, **as_json}
    if options.get("cache") and _changes_compile_options(options):
        if as_json.get("cache"):
            warnings.warn(
                "NUMBDUCK_JIT_OPTIONS asks for the disk cache together with non-default compile options; "
                "numbduck compiles without the cache, because numba's cache key holds no compile options and "
                "an object compiled under one set would be served to a process running another.",
                RuntimeWarning,
                stacklevel=2,
            )
        options["cache"] = False
    return options


def _cache_probe():
    """Give the check this module's file; numba looks the package up by the file that holds this function's code."""


# numbox's bound on the length of a name numba writes for a function's cache file. numbduck's longest name is 96
# bytes, and a test holds every function's name under the bound. A cache directory within 32 bytes of the path
# limit therefore turns caching off though its names would fit; the margin keeps a longer binding name from
# overflowing a directory that passed the check.
LONGEST_CACHE_FILE_NAME = 128


def uncached_where_no_cache_can_be_written(options):
    """Return ``options``, with ``cache`` turned off after one warning if numba cannot write a cache for numbduck.

    numba has no cache location in three placements: a read-only install whose user cache directory is read-only
    too, an import from an ``.egg``, ``.whl`` or ``.pyz`` archive, and an import from a ``.zip`` whose user cache
    directory is read-only. From numba 0.68 on, an import from a ``.zip`` that holds numbduck as ``.pyc`` files
    alone is a fourth placement. The warning names numbox's remedy for the placement. A source install needs
    ``NUMBA_CACHE_DIR`` set. A ``.zip`` whose user cache directory is read-only needs that directory made writable.
    A ``.zip`` of ``.pyc`` files alone, or any other archive, needs the source files on disk.
    ``NUMBDUCK_JIT_OPTIONS='{"cache": false}'`` turns caching off without the warning. An error that numbox's
    ``is_a_cache_error`` does not classify as a cache error is raised unchanged.
    """
    if not options.get("cache"):
        return options
    # numba sets a cached function up at decoration, so the check runs before the first binding is decorated. Every
    # binding and buffer allocator is in this module's directory, so one check of this module's file covers all of
    # them. check_cache_location runs numba's cache set-up and the writability check of its first save without
    # compiling anything.
    py_file = inspect.getfile(_cache_probe)
    try:
        check_cache_location(py_file, LONGEST_CACHE_FILE_NAME)
    except (RuntimeError, OSError, KeyError, ValueError) as error:
        if not is_a_cache_error(error):
            raise
        failure = error
    else:
        return options
    silence = "NUMBDUCK_JIT_OPTIONS='{\"cache\": false}' to turn caching off and silence this warning"
    remedy = cache_remedy(py_file, failure, silence, package="numbduck")
    warnings.warn(
        f"numba cannot cache numbduck here ({failure}); it compiles without a cache. {remedy}",
        RuntimeWarning, stacklevel=2,
    )
    return {**options, "cache": False}


jit_options = uncached_where_no_cache_can_be_written(get_jit_options())
