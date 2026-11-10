"""Tests of numbduck's import where numba cannot write a cache for it, and of the check that turns caching off there.

Each test imports numbduck in a child process, from one placement and with a cache directory of its own. The
placements are an ``.egg`` or ``.whl`` archive, a ``.zip``, a ``.zip`` of ``.pyc`` files alone, a read-only install,
and a ``NUMBA_CACHE_DIR`` near Linux's path limit. The tests check that the import succeeds, that it warns once with the
remedy where numba cannot cache, and where numba writes cache files. The results depend on the numba version, because
numba 0.68 reads a ``.zip`` member's source. They depend on the platform too. On Windows, numba 0.68.0 caches no
``.zip``, and the read-only tests are skipped on Windows and for root, where chmod cannot remove write access. One
test checks that an error unrelated to the cache propagates, and one holds every function's cache file name within the
bound that the check reserves. ``test_ducklib.py`` tests the jit options themselves and the bindings.
"""
import ast
import compileall
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# On Windows, and for root, chmod does not remove write access from a directory.
needs_a_directory_it_cannot_write = pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason="needs a directory this user cannot write to")

# The child calls a buffer allocator and two bindings. numba compiles the allocator at its first call, and numbox
# compiles the bindings when ducklib is imported. The child prints the file of numbduck.configurations, which shows
# the placement that it imported.
IMPORT_AND_USE = (
    "import numbduck.configurations as configurations\n"
    "from numbduck import ducklib\n"
    "from numbduck.duckdb_utils import create_duckdb_database\n"
    "database = create_duckdb_database()\n"
    "assert ducklib.duckdb_open(0, database.ctypes.data) == ducklib.DuckDBSuccess\n"
    "ducklib.duckdb_close(database.ctypes.data)\n"
    "print(configurations.__file__)\n"
)

WARNING = "numba cannot cache numbduck here"


def _archive(path):
    """Zip numbduck's source files into ``path`` and return ``path``, which a test puts on PYTHONPATH as it is."""
    with zipfile.ZipFile(path, "w") as zipped:
        for source in sorted((REPO / "numbduck").rglob("*.py")):
            zipped.write(source, str(source.relative_to(REPO)))
    return path


def _pyc_archive(path):
    """Zip numbduck's modules into ``path`` as ``.pyc`` files alone, and return ``path``.

    Each file is compiled to record its source as a member of the archive, ``numbduck/<module>.py``, which the archive
    does not hold.
    """
    source = path.parent / "pyc-source"
    shutil.copytree(REPO / "numbduck", source / "numbduck", ignore=shutil.ignore_patterns("__pycache__"))
    compileall.compile_dir(str(source / "numbduck"), ddir=str(path / "numbduck"), legacy=True, quiet=1)
    with zipfile.ZipFile(path, "w") as zipped:
        for compiled in sorted((source / "numbduck").glob("*.pyc")):
            zipped.write(compiled, f"numbduck/{compiled.name}")
    return path


def _run(env, cwd, warnings="always"):
    # The child runs in a directory outside the checkout, so it imports numbduck from the placement that PYTHONPATH
    # names, not from the source tree that the test runs from.
    return subprocess.run([sys.executable, "-W", warnings, "-c", IMPORT_AND_USE],
                          capture_output=True, text=True, env=env, cwd=str(cwd))


def _index_files(cache_dir):
    """Return the names of numbduck's index files under ``cache_dir``; numbox's index files there are left out."""
    return sorted(path.name for path in Path(cache_dir).rglob("*.nbi")
                  if path.name.startswith(("ducklib.", "duckdb_utils.")))


def _env(**changes):
    """Return this environment with ``changes`` and without ``NUMBDUCK_JIT_OPTIONS``, so the child uses the defaults."""
    env = dict(os.environ, **changes)
    env.pop("NUMBDUCK_JIT_OPTIONS", None)
    return env


@pytest.mark.parametrize("name", ["numbduck-0.0.0-py3.12.egg", "numbduck-0.0.0-py3-none-any.whl"])
def test_an_import_from_an_archive_compiles_uncached_with_one_warning_naming_the_remedy(tmp_path, name):
    """Check that an import from an ``.egg`` or ``.whl`` compiles uncached and warns once with the right remedy.

    numba has no cache location for a module in an archive other than a ``.zip``, and Spark's ``--py-files`` ships
    such archives. ``NUMBA_CACHE_DIR`` is set to a writable directory here, and numba still writes none of numbduck's
    files there. The warning therefore says that the variable has no effect, and names an install with the source files
    on disk. ``NUMBDUCK_JIT_OPTIONS='{"cache": false}'`` silences it.
    """
    archive = _archive(tmp_path / name)
    env = _env(PYTHONPATH=str(archive), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    assert run.stderr.count(WARNING) == 1, run.stderr
    assert "NUMBA_CACHE_DIR has no effect here" in run.stderr and "Set NUMBA_CACHE_DIR" not in run.stderr
    assert "install numbduck with its source files on disk" in run.stderr, run.stderr
    assert _index_files(tmp_path / "cache") == []
    quiet = _run(dict(env, NUMBDUCK_JIT_OPTIONS='{"cache": false}'), tmp_path, warnings="error::RuntimeWarning")
    assert quiet.returncode == 0, quiet.stderr


def test_a_zip_import_is_cached_by_numba(tmp_path):
    """Check that an import from a ``.zip`` issues no warning and caches in the user's cache directory.

    numba caches a ``.zip`` member in the user's cache directory, whatever ``NUMBA_CACHE_DIR`` names. That is why the
    warning for another archive offers a ``.zip``.
    """
    import numba
    archive = _archive(tmp_path / "numbduck.zip")
    home = tmp_path / "home"
    env = _env(PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    if os.name == "nt" and numba.__version__.startswith("0.68.0"):
        # On Windows, numba 0.68.0 and its release candidate read a .zip member's source under a name with the
        # platform's separator, and the archive holds no member of that name
        # (https://github.com/numba/numba/issues/10889, fixed in 0.68.1). numbduck therefore compiles without a cache
        # there, and the warning names the issue.
        assert run.stderr.count(WARNING) == 1 and "numba/issues/10889" in run.stderr, run.stderr
    else:
        assert WARNING not in run.stderr, run.stderr
    assert _index_files(tmp_path / "cache") == []
    if os.name != "nt":
        # On Windows, numba asks the system for the user's cache directory, and no variable set here changes it.
        assert _index_files(home)


def test_a_zip_of_pyc_files_alone_compiles_uncached_from_numba_0_68_and_is_cached_before(tmp_path):
    """Check that a ``.zip`` of ``.pyc`` files alone is cached before numba 0.68, and from 0.68 warns once instead.

    From numba 0.68, numba reads a ``.zip`` member's source from the archive for the member's stamp, and zipfile raises
    ``KeyError`` for a member shipped as ``.pyc`` alone. The check counts that error as a cache error. Before 0.68,
    numba does not read the source, and it caches the member as it caches any other.
    """
    import numba
    archive = _pyc_archive(tmp_path / "numbduck.zip")
    home = tmp_path / "home"
    env = _env(PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    reads_the_source = tuple(int(part) for part in numba.__version__.split(".")[:2]) >= (0, 68)
    assert (run.stderr.count(WARNING) == 1) == reads_the_source, run.stderr
    if reads_the_source:
        assert "the module shipped as .pyc alone" in run.stderr, run.stderr
        assert "install numbduck with its source files on disk" in run.stderr, run.stderr
    assert _index_files(tmp_path / "cache") == []
    if os.name != "nt":
        assert bool(_index_files(home)) == (not reads_the_source)


@needs_a_directory_it_cannot_write
def test_a_zip_import_with_no_writable_user_cache_directory_compiles_uncached_with_a_warning(tmp_path):
    """Check that a ``.zip`` import with a read-only user cache directory compiles uncached and warns once.

    numba takes the user's cache directory for a ``.zip`` without testing that it can write there
    (https://github.com/numba/numba/issues/10888). Without the check, numba's first save would fail: at import for a
    binding, and at the first call for a buffer allocator. The check detects the read-only directory before ducklib
    decorates a binding, and turns caching off for both.
    """
    archive = _archive(tmp_path / "numbduck.zip")
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o555)
    try:
        env = _env(PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
                   NUMBA_CACHE_DIR=str(tmp_path / "cache"))
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
        assert run.stderr.count(WARNING) == 1, run.stderr
        assert "NUMBA_CACHE_DIR has no effect here" in run.stderr, run.stderr
    finally:
        home.chmod(0o755)


@needs_a_directory_it_cannot_write
def test_a_read_only_install_warns_naming_numba_cache_dir_and_setting_it_caches(tmp_path):
    """Check that a read-only install warns once and names ``NUMBA_CACHE_DIR``, and that setting the variable caches.

    A source file on disk has no cache location if numba can write neither its ``__pycache__`` nor the user's cache
    directory. With ``NUMBA_CACHE_DIR`` set, the import caches there and issues no warning.
    """
    site = tmp_path / "site"
    shutil.copytree(REPO / "numbduck", site / "numbduck", ignore=shutil.ignore_patterns("__pycache__"))
    home = tmp_path / "home"
    home.mkdir()
    read_only = [home, *(path for path in site.rglob("*") if path.is_dir())]
    for path in read_only:
        path.chmod(0o555)
    try:
        env = _env(PYTHONPATH=str(site), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"))
        env.pop("NUMBA_CACHE_DIR", None)
        run = _run(env, tmp_path)
        assert run.returncode == 0 and str(site) in run.stdout, run.stderr
        assert run.stderr.count(WARNING) == 1 and "Set NUMBA_CACHE_DIR" in run.stderr, run.stderr
        cured = _run(dict(env, NUMBA_CACHE_DIR=str(tmp_path / "cache")), tmp_path, warnings="error::RuntimeWarning")
        assert cured.returncode == 0, cured.stderr
        assert _index_files(tmp_path / "cache")
    finally:
        for path in read_only:
            path.chmod(0o755)


def _directory_of_length(root, length):
    """Make and return a directory under ``root`` whose path is ``length`` bytes, whatever the length of ``root``.

    The function appends components of 200 bytes to ``root`` for as long as one more leaves room for a last component,
    and then appends a last component of the remaining length. Each component is at most 201 bytes, within the 255
    bytes that Linux file systems allow in a name, so the directory can always be made. Lengths are counted in the file
    system's encoding, because Linux limits a path's length in bytes. In a multi-byte encoding such as UTF-8, a root
    with non-ASCII characters has more bytes than characters.
    """
    deep = root
    while len(os.fsencode(deep)) + 201 < length - 1:
        deep = deep / ("d" * 200)
    deep = deep / ("d" * (length - 1 - len(os.fsencode(deep))))
    assert len(os.fsencode(deep)) == length
    deep.mkdir(parents=True)
    return deep


def _component_wider_in_bytes():
    """Return a one-character path component that has more bytes than characters in the file system's encoding.

    If no candidate is wider, as in a single-byte encoding, where every character takes one byte, the function returns
    an ASCII component.
    """
    for candidate in ("ü", "中"):
        try:
            if len(os.fsencode(candidate)) > 1:
                return candidate
        except UnicodeEncodeError:
            pass
    return "u"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the path limit of 4096 bytes is Linux's")
def test_a_location_too_close_to_the_path_limit_for_numbas_files_compiles_uncached_and_is_told_so(tmp_path):
    """Check that a cache location with no room for numba's file names compiles uncached and names a shorter path.

    On Linux, numba tests a location with a temporary file that has no name, so a location just within the path limit
    passes numba's own test. numba's file names for a binding can exceed seventy bytes, and its first save there would
    fail. The check reserves ``LONGEST_CACHE_FILE_NAME`` bytes for those names, and turns caching off instead.
    """
    # numba's location for numbduck is a directory under NUMBA_CACHE_DIR, named after numbduck's directory and a hash
    # of its path. The test makes that location's path 20 bytes shorter than the limit, whatever the checkout's path.
    # The directory begins with a character that takes more than one byte in the file system's encoding, if the
    # encoding has one, so that a length counted in characters instead of bytes would fail the test.
    from numba.core.caching import _CacheLocator
    location = _CacheLocator.get_suitable_cache_subpath(str(REPO / "numbduck" / "configurations.py"))
    cache_dir = _directory_of_length(tmp_path / _component_wider_in_bytes(), 4096 - 20 - 1 - len(os.fsencode(location)))
    assert len(os.fsencode(os.path.join(cache_dir, location))) == 4096 - 20
    env = _env(PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(cache_dir))
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(REPO / "numbduck") in run.stdout, run.stderr
    assert run.stderr.count(WARNING) == 1, run.stderr
    assert "the path is too long for the file system: a shorter NUMBA_CACHE_DIR; or NUMBDUCK_JIT_OPTIONS" in (
        run.stderr), run.stderr
    assert not _index_files(cache_dir)


# The child makes numba's cache set-up raise a RuntimeError that is not a cache error, and then imports
# numbduck.configurations, whose check runs the set-up.
ANOTHER_ERROR_AT_THE_CACHE_SET_UP = (
    "import numbox.core.configurations\n"
    "import numba.core.caching as caching\n"
    "def refuse(self, py_func):\n"
    "    raise RuntimeError('a locator of another kind refused')\n"
    "caching.CompileResultCacheImpl.__init__ = refuse\n"
    "import numbduck.configurations\n"
)


def test_an_error_that_is_not_the_caches_is_raised_as_it_was(tmp_path):
    """Check that an error from numba's cache set-up that is not a cache error propagates, with no warning."""
    # The child imports numbox first, so that numbox's own check runs before the set-up is made to fail, and the error
    # comes from numbduck's check.
    env = _env(PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    run = subprocess.run([sys.executable, "-c", ANOTHER_ERROR_AT_THE_CACHE_SET_UP],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode != 0
    assert "RuntimeError: a locator of another kind refused" in run.stderr, run.stderr
    assert "numbduck/configurations.py" in run.stderr.replace(os.sep, "/"), run.stderr
    assert WARNING not in run.stderr


def _function_names(tree):
    """Yield each function's qualified name and line, with the name as numba writes it in a cache file's name.

    numba drops the angle brackets from the name there, as from ``<locals>``.
    """
    def walk(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield ".".join(scope + [child.name]), child.lineno
                yield from walk(child, scope + [child.name, "locals"])
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, scope + [child.name])
            else:
                yield from walk(child, scope)
    return walk(tree, [])


def test_numbas_longest_file_name_for_the_package_is_within_the_probes_bound():
    """Check that the cache file name of every function of the package fits in ``LONGEST_CACHE_FILE_NAME`` bytes."""
    from numbduck.configurations import LONGEST_CACHE_FILE_NAME
    # A cache file's name holds the module's stem, the function's qualified name, a line number, the interpreter tag
    # with its abiflags, and an index number. numba first writes the file under a temporary name 21 bytes longer. The
    # tag is built for the widest case: the abiflag "t", a two-digit index and the temporary suffix.
    tag = f".py{sys.version_info[0]}{sys.version_info[1]}t.99.nbc.tmp.0123456789abcdef"
    # numbox's proxy gives the body of a binding a qualified name with two leading underscores. Every function of the
    # package is measured with them, whether it is a binding or not.
    names = [f"{path.stem}.__{qualname}-{lineno}{tag}"
             for path in sorted((REPO / "numbduck").rglob("*.py"))
             for qualname, lineno in _function_names(ast.parse(path.read_text(encoding="utf-8")))]
    longest = max(names, key=lambda name: len(os.fsencode(name)))
    assert len(names) > 200
    assert len(os.fsencode(longest)) <= LONGEST_CACHE_FILE_NAME, longest
