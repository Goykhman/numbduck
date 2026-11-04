"""These tests import numbduck where numba cannot write a cache for it.

Each test runs the import in a child process with its own source tree and cache directory, so the child process
imports only from the placement under test. The placements are an archive, a ``.zip``, a ``.zip`` of ``.pyc`` files
alone, a read-only install, a cache directory at the path limit, and an error unrelated to the cache. One test holds
every function's cache file name under the bound the check reserves.
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

# The child imports and uses a buffer allocator, compiled lazily at its first call, and two bindings, each
# decorated at import under the package's options.
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
    """Zip numbduck's modules into ``path`` and return it; the archive is put on PYTHONPATH unchanged."""
    with zipfile.ZipFile(path, "w") as zipped:
        for source in sorted((REPO / "numbduck").rglob("*.py")):
            zipped.write(source, str(source.relative_to(REPO)))
    return path


def _pyc_archive(path):
    """Zip numbduck's modules into ``path`` as ``.pyc`` files alone, compiled with the archive as their directory."""
    source = path.parent / "pyc-source"
    shutil.copytree(REPO / "numbduck", source / "numbduck", ignore=shutil.ignore_patterns("__pycache__"))
    compileall.compile_dir(str(source / "numbduck"), ddir=str(path / "numbduck"), legacy=True, quiet=1)
    with zipfile.ZipFile(path, "w") as zipped:
        for compiled in sorted((source / "numbduck").glob("*.pyc")):
            zipped.write(compiled, f"numbduck/{compiled.name}")
    return path


def _run(env, cwd, warnings="always"):
    # From a neutral cwd, the child process imports the placement that PYTHONPATH names, not the source tree the
    # test runs from.
    return subprocess.run([sys.executable, "-W", warnings, "-c", IMPORT_AND_USE],
                          capture_output=True, text=True, env=env, cwd=str(cwd))


def _index_files(cache_dir):
    """Return numbduck's index file names under ``cache_dir``; numbox's own, cached there too, are not counted."""
    return sorted(path.name for path in Path(cache_dir).rglob("*.nbi")
                  if path.name.startswith(("ducklib.", "duckdb_utils.")))


def _env(**changes):
    env = dict(os.environ, **changes)
    env.pop("NUMBDUCK_JIT_OPTIONS", None)
    return env


@pytest.mark.parametrize("name", ["numbduck-0.0.0-py3.12.egg", "numbduck-0.0.0-py3-none-any.whl"])
def test_an_import_from_an_archive_compiles_uncached_with_one_warning_naming_the_remedy(tmp_path, name):
    # numba's cache locators need the source file on disk, so an archive gives numba no cache location. Spark's
    # --py-files option ships such archives. NUMBA_CACHE_DIR has no effect: it is set and writable here, and numba
    # still writes none of numbduck's files there. So the warning states that the variable has no effect and tells
    # the user to install numbduck with its source files on disk.
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
    # numba caches a .zip member in the user's cache directory regardless of NUMBA_CACHE_DIR, so the warning directs
    # an archive's user to a .zip.
    import numba
    archive = _archive(tmp_path / "numbduck.zip")
    home = tmp_path / "home"
    env = _env(PYTHONPATH=str(archive), HOME=str(home), XDG_CACHE_HOME=str(home / "cache"),
               NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(archive) in run.stdout, run.stderr
    if os.name == "nt" and numba.__version__.startswith("0.68.0"):
        # numba 0.68.0 and its release candidate on Windows read a .zip member's source by a name with Windows's
        # separator, and no archive holds that name (numba issue 10889, fixed in 0.68.1). So numbduck compiles
        # uncached there, and the warning names the issue.
        assert run.stderr.count(WARNING) == 1 and "numba/issues/10889" in run.stderr, run.stderr
    else:
        assert WARNING not in run.stderr, run.stderr
    assert _index_files(tmp_path / "cache") == []
    if os.name != "nt":
        # On Windows, numba obtains the user's cache directory from the system, and no variable set here changes it.
        assert _index_files(home)


def test_a_zip_of_pyc_files_alone_compiles_uncached_from_numba_0_68_and_is_cached_before(tmp_path):
    # From numba 0.68 on, numba reads a .zip member's source from the archive for its stamp, and zipfile raises
    # KeyError when the archive holds the member as .pyc alone. The check turns caching off for that error as it
    # does for the others. Before numba 0.68, numba does not read the source and caches the member as it caches any
    # other member.
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
    # numba takes the user's cache directory for a .zip without checking that it can be written. So an executor
    # with a read-only home receives numba's error at the first save: at import for a binding, at the first call
    # for a buffer allocator. The check catches both cases before any binding is decorated.
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
    # A source on disk has no cache location when neither its directory nor the user's cache directory can be
    # written. NUMBA_CACHE_DIR is the remedy, and setting it enables caching.
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
    """Make a directory under ``root`` whose path is ``length`` bytes, whatever the length of ``root``.

    The path takes components of 200 bytes while one more of them leaves room for a last component, and then a last
    component of the remaining length, so the directory itself can always be made. Lengths are counted in the file
    system's encoding, because Linux limits a path's bytes, and a root with non-ASCII characters has more bytes than
    characters.
    """
    deep = root
    while len(os.fsencode(deep)) + 201 < length - 1:
        deep = deep / ("d" * 200)
    deep = deep / ("d" * (length - 1 - len(os.fsencode(deep))))
    assert len(os.fsencode(deep)) == length
    deep.mkdir(parents=True)
    return deep


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the path limit of 4096 bytes is Linux's")
def test_a_location_too_close_to_the_path_limit_for_numbas_files_compiles_uncached_and_is_told_so(tmp_path):
    # numba checks a location with a temporary file that has no name. So on Linux a NUMBA_CACHE_DIR within the path
    # limit for numbduck's location can still leave no room for numba's file names, which run to seventy bytes and
    # more for a binding. The check reserves LONGEST_CACHE_FILE_NAME bytes for them. numba's location for numbduck
    # is the directory's name and a hash of its path under NUMBA_CACHE_DIR, and the test places the end of that
    # location 20 bytes short of the limit, regardless of the checkout's own path. The directory begins with a
    # non-ASCII component, which has more bytes than characters, so the lengths are counted in bytes.
    from numba.core.caching import _CacheLocator
    location = _CacheLocator.get_suitable_cache_subpath(str(REPO / "numbduck" / "configurations.py"))
    cache_dir = _directory_of_length(tmp_path / "ü", 4096 - 20 - 1 - len(os.fsencode(location)))
    assert len(os.fsencode(os.path.join(cache_dir, location))) == 4096 - 20
    env = _env(PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(cache_dir))
    run = _run(env, tmp_path)
    assert run.returncode == 0 and str(REPO / "numbduck") in run.stdout, run.stderr
    assert run.stderr.count(WARNING) == 1, run.stderr
    assert "the path is too long for the file system: a shorter NUMBA_CACHE_DIR; or NUMBDUCK_JIT_OPTIONS" in (
        run.stderr), run.stderr
    assert not _index_files(cache_dir)


ANOTHER_ERROR_AT_THE_CACHE_SET_UP = (
    "import numbox.core.configurations\n"
    "import numba.core.caching as caching\n"
    "def refuse(self, py_func):\n"
    "    raise RuntimeError('a locator of another kind refused')\n"
    "caching.CompileResultCacheImpl.__init__ = refuse\n"
    "import numbduck.configurations\n"
)


def test_an_error_that_is_not_the_caches_is_raised_as_it_was(tmp_path):
    # numbox is imported first, so its own check has run before the cache set-up is made to refuse, and the
    # error comes from numbduck's check.
    env = _env(PYTHONPATH=str(REPO), NUMBA_CACHE_DIR=str(tmp_path / "cache"))
    run = subprocess.run([sys.executable, "-c", ANOTHER_ERROR_AT_THE_CACHE_SET_UP],
                         capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert run.returncode != 0
    assert "RuntimeError: a locator of another kind refused" in run.stderr, run.stderr
    assert "numbduck/configurations.py" in run.stderr.replace(os.sep, "/"), run.stderr
    assert WARNING not in run.stderr


def _function_names(tree):
    """Yield each function's qualified name as numba names its cache files, angle brackets dropped, with its line."""
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
    # The check reserves LONGEST_CACHE_FILE_NAME bytes in a location for the files numba writes there. A file's
    # name holds the module's stem, the function's qualified name, a line number, the interpreter tag with its
    # abiflags and an index number, and numba writes the file under a 21-byte temporary name. numbox's proxy
    # gives a binding's body a qualified name with two leading underscores. Every function of the package must
    # stay within the bound, whether decorated or not, as a binding's body.
    from numbduck.configurations import LONGEST_CACHE_FILE_NAME
    tag = f".py{sys.version_info[0]}{sys.version_info[1]}t.99.nbc.tmp.0123456789abcdef"
    names = [f"{path.stem}.__{qualname}-{lineno}{tag}"
             for path in sorted((REPO / "numbduck").rglob("*.py"))
             for qualname, lineno in _function_names(ast.parse(path.read_text(encoding="utf-8")))]
    longest = max(names, key=lambda name: len(os.fsencode(name)))
    assert len(names) > 200
    assert len(os.fsencode(longest)) <= LONGEST_CACHE_FILE_NAME, longest
