"""numbduck's njit options, and the check that turns caching off where numba cannot cache numbduck.

``jit_options`` holds the keyword arguments that numbduck passes to numba's ``njit`` for its bindings in
:mod:`numbduck.ducklib` and its buffer allocators in :mod:`numbduck.duckdb_utils`; the two bignum bindings take them
without ``"cache"``. When this module is first imported, :func:`get_jit_options` reads the options from the
``NUMBDUCK_JIT_OPTIONS`` environment variable. If caching is on, :func:`uncached_where_no_cache_can_be_written` then
checks whether numba can cache numbduck's functions, and turns caching off, with one ``RuntimeWarning``, if numba
cannot.
"""
import inspect
import json
import os
import warnings

from numba.core.options import DefaultOptions
from numbox.core.configurations import cache_remedy, check_cache_location, is_a_cache_error


# The keys that NUMBDUCK_JIT_OPTIONS may hold: the attribute names of numba's DefaultOptions, and the dispatcher's
# cache flag, which numba.core.decorators.jit handles. DefaultOptions has an attribute for each option of njit, so the
# set follows the installed numba. It also has writable_args, an option of @guvectorize that njit refuses, so that key
# passes this check and fails at ducklib's import. njit raises a KeyError for an unknown key when it compiles a
# function, which for a binding happens at ducklib's import, and the error does not name the variable.
# get_jit_options therefore refuses any other key itself, with a message that names the variable.
_ALLOWED_JIT_OPTIONS = frozenset(
    name for name in vars(DefaultOptions) if not name.startswith("__")
) | {"cache"}

# With nogil, a binding that Python calls releases the GIL while the C function runs. DuckDB runs a query on worker
# threads, and some of them need the GIL. A Python UDF on the same connection needs it. So does a numba @cfunc whose
# body raises, because numba takes the GIL to print the exception. If the calling thread kept the GIL while it waited
# for the workers, such a worker could not run until the calling thread released the GIL, and the query could hang.
# The option covers a binding that Python calls. A binding called inside a jitted function runs under that function's
# own setting, so a jitted function that starts such a query needs nogil=True itself.
_DEFAULT_JIT_OPTIONS = {"cache": True, "nogil": True}


def _changes_compile_options(options):
    """Return whether ``options`` differ from the defaults in any option other than ``"cache"``."""
    return {k: v for k, v in options.items() if k != "cache"} != {
        k: v for k, v in _DEFAULT_JIT_OPTIONS.items() if k != "cache"}


def get_jit_options():
    """Return the njit options of numbduck's compiled functions: the defaults, updated from ``NUMBDUCK_JIT_OPTIONS``.

    The defaults are ``{"cache": True, "nogil": True}``. The variable holds a JSON object. Each entry replaces the
    default of the same name or adds another njit option, so this setting turns numba's on-disk cache off and keeps
    ``nogil``::

        export NUMBDUCK_JIT_OPTIONS='{"cache": false}'

    The variable is read on every call. ``ValueError`` is raised if the value is not valid JSON, as an empty value is
    not, if it is not a JSON object, or if it has a key that is not one of numba's compile options or ``"cache"``.
    numbduck and numba read ``"cache"`` by its truth value, so the string ``"false"`` keeps the cache on.

    Any option other than ``"cache"`` turns the cache off, unless it is ``"nogil": true``, the default. That includes
    an option set to numba's own default, such as ``"parallel": false``. numba's cache key does not record the
    options that a function was compiled with, so a process could otherwise load code compiled under other options.
    If the object also sets ``"cache"`` to true, a ``RuntimeWarning`` says that numbduck compiles without the cache.
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
    # numbox adds the compile options to the cache key of each binding, but numba caches each buffer allocator in
    # duckdb_utils under its own key alone, which holds no compile options. If the cache stayed on, a process under
    # other options would load allocators that an earlier process compiled under the defaults, and a later process
    # under the defaults would load the allocators that this one compiled.
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
    """Never called: the check reads the file name that this function's code records, as numba does for each binding.

    numba finds a function's cache by the file name that the function's code records. That name can differ from
    ``__file__``, as for a module shipped as a ``.pyc`` alone.
    """


# The bound, in bytes, on the name of a cache file that numba writes for a function of numbduck, numba's temporary
# suffix included. The check creates a file with a name of this length in the cache location. numbduck's longest name
# is 96 bytes, and a test checks that the cache file name of every function of the package fits within the bound. The
# check therefore turns caching off for a location with room for a 96-byte name but not for a 128-byte one, although
# the names of all of numbduck's functions would fit there. Because of the margin of 32 bytes, a location that passed
# the check also has room for a longer name of a binding added later.
LONGEST_CACHE_FILE_NAME = 128


def uncached_where_no_cache_can_be_written(options):
    """Return ``options``, or a copy with ``"cache"`` set to ``False`` after one warning if numba cannot cache numbduck.

    numba sets up a function's cache when it decorates the function, and raises there, or at the first save, if it
    has no location that it can write in. Without this check, numba's error would end the import of numbduck's modules
    or fail the first call of a buffer allocator, and it would not say what to change. The check runs numba's cache
    set-up for this module's file, without compiling anything. numba cannot cache numbduck in these cases, among others:

    - a read-only install whose user cache directory cannot be written either;
    - an import from an ``.egg``, ``.whl`` or ``.pyz`` archive, which numba does not cache;
    - an import from a ``.zip`` whose user cache directory cannot be written;
    - from numba 0.68, an import from a ``.zip`` that holds numbduck's modules as ``.pyc`` files alone;
    - on Windows under numba 0.68.0, an import from any ``.zip`` (https://github.com/numba/numba/issues/10889, fixed in
      numba 0.68.1);
    - a cache location in which a file name of ``LONGEST_CACHE_FILE_NAME`` bytes does not fit, as near the path limit.

    The ``RuntimeWarning`` gives numba's error and the remedy that numbox's ``cache_remedy`` composes for it. For a
    read-only install, the remedy is ``NUMBA_CACHE_DIR`` set to a writable directory. For a ``.zip``, it is a writable
    user cache directory, and for a ``.zip`` of ``.pyc`` files or another archive, the source files on disk. For a
    location near the path limit, it is a shorter path. The warning ends with
    ``NUMBDUCK_JIT_OPTIONS='{"cache": false}'``, which turns caching off and silences the warning.

    Options with caching off are returned as they are, and the check does not run. An error that numbox's
    ``is_a_cache_error`` does not count as a cache error is raised unchanged.
    """
    if not options.get("cache"):
        return options
    # The check runs when this module is imported, before ducklib decorates its first binding, because ducklib and
    # duckdb_utils import jit_options from here. The bindings and the buffer allocators are all in this module's
    # directory, so one check of this module's file covers them all. check_cache_location runs numba's cache set-up
    # and the test that numba runs before its first save, and creates and removes a file with a name of
    # LONGEST_CACHE_FILE_NAME bytes.
    py_file = inspect.getfile(_cache_probe)
    try:
        check_cache_location(py_file, LONGEST_CACHE_FILE_NAME)
    # numba's cache errors belong to these four classes; a UnicodeEncodeError is a ValueError.
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


# The keyword arguments that numbduck passes to numba's njit: the options that get_jit_options returned when this
# module was first imported, with "cache" set to False if numba cannot cache numbduck.
jit_options = uncached_where_no_cache_can_be_written(get_jit_options())
