import os
import subprocess
import sys
import textwrap

import numpy
from numba import njit, types as nb_types

from numbduck.vector import make_vector
from numbduck.vector import _vector_cache
from numbduck.vector import vector_push, vector_extend  # noqa: E402


def test_construction_and_len():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        v = Float64Vec(8)
        return len(v), v.buf.shape[0]

    size, cap = go()
    assert size == 0
    assert cap == 8


def test_getitem_setitem():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        v = Float64Vec(4)
        v.buf[0] = 10.0
        v.buf[1] = 20.0
        v.size = 2
        v[1] = 99.0
        return v[0], v[1], len(v)

    a, b, n = go()
    assert a == 10.0
    assert b == 99.0
    assert n == 2


def test_factory_caching():
    r1 = make_vector(nb_types.float64)
    r2 = make_vector(nb_types.float64)
    assert r1[0] is r2[0]
    assert r1[1] is r2[1]
    assert nb_types.float64.key in _vector_cache


def test_vector_push():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        v = Float64Vec(4)
        vector_push(v, 1.0)
        vector_push(v, 2.0)
        vector_push(v, 3.0)
        return v[0], v[1], v[2], len(v), v.buf.shape[0]

    a, b, c, n, cap = go()
    assert (a, b, c) == (1.0, 2.0, 3.0)
    assert n == 3
    assert cap == 4


def test_vector_push_growth():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        v = Float64Vec(2)
        for i in range(5):
            vector_push(v, float(i * 10))
        return v[0], v[1], v[2], v[3], v[4], len(v), v.buf.shape[0]

    vals = go()
    assert vals[:5] == (0.0, 10.0, 20.0, 30.0, 40.0)
    assert vals[5] == 5
    assert vals[6] == 8


def test_vector_extend():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(4)
        b = Float64Vec(4)
        vector_push(a, 1.0)
        vector_push(a, 2.0)
        vector_push(b, 3.0)
        vector_push(b, 4.0)
        vector_push(b, 5.0)
        vector_extend(a, b)
        return a[0], a[1], a[2], a[3], a[4], len(a), a.buf.shape[0]

    vals = go()
    assert vals[:5] == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert vals[5] == 5
    assert vals[6] == 8


def test_vector_extend_no_growth():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(8)
        b = Float64Vec(4)
        vector_push(a, 1.0)
        vector_push(b, 2.0)
        vector_push(b, 3.0)
        vector_extend(a, b)
        return a[0], a[1], a[2], len(a), a.buf.shape[0]

    vals = go()
    assert vals[:3] == (1.0, 2.0, 3.0)
    assert vals[3] == 3
    assert vals[4] == 8


def test_vector_push_exact_capacity():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        v = Float64Vec(4)
        vector_push(v, 1.0)
        vector_push(v, 2.0)
        vector_push(v, 3.0)
        vector_push(v, 4.0)
        return v[0], v[1], v[2], v[3], len(v), v.buf.shape[0]

    vals = go()
    assert vals[:4] == (1.0, 2.0, 3.0, 4.0)
    assert vals[4] == 4
    assert vals[5] == 4


def test_vector_extend_empty_src():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(4)
        b = Float64Vec(4)
        vector_push(a, 1.0)
        vector_push(a, 2.0)
        vector_extend(a, b)
        return a[0], a[1], len(a), a.buf.shape[0]

    vals = go()
    assert vals[:2] == (1.0, 2.0)
    assert vals[2] == 2
    assert vals[3] == 4


def test_vector_extend_empty_dst():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(4)
        b = Float64Vec(4)
        vector_push(b, 1.0)
        vector_push(b, 2.0)
        vector_push(b, 3.0)
        vector_extend(a, b)
        return a[0], a[1], a[2], len(a), a.buf.shape[0]

    vals = go()
    assert vals[:3] == (1.0, 2.0, 3.0)
    assert vals[3] == 3
    assert vals[4] == 4


def test_vector_extend_both_empty():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(4)
        b = Float64Vec(4)
        vector_extend(a, b)
        return len(a), a.buf.shape[0]

    n, cap = go()
    assert n == 0
    assert cap == 4


def test_vector_extend_multiple_doublings():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        a = Float64Vec(2)
        b = Float64Vec(16)
        vector_push(a, 0.0)
        for i in range(10):
            vector_push(b, float(i + 1))
        vector_extend(a, b)
        return a[0], a[1], a[10], len(a), a.buf.shape[0]

    first, second, last, n, cap = go()
    assert first == 0.0
    assert second == 1.0
    assert last == 10.0
    assert n == 11
    assert cap == 16


def test_vector_push_special_floats():
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        v = Float64Vec(4)
        vector_push(v, numpy.nan)
        vector_push(v, numpy.inf)
        vector_push(v, -numpy.inf)
        return v[0], v[1], v[2], len(v)

    a, b, c, n = go()
    assert numpy.isnan(a)
    assert b == numpy.inf
    assert c == -numpy.inf
    assert n == 3


def test_multi_dtype_int64():
    Int64Vec, _ = make_vector(nb_types.int64)

    @njit
    def go():
        v = Int64Vec(2)
        vector_push(v, 100)
        vector_push(v, 200)
        vector_push(v, 300)
        return v[0], v[1], v[2], len(v), v.buf.shape[0]

    vals = go()
    assert vals[:3] == (100, 200, 300)
    assert vals[3] == 3
    assert vals[4] == 4


def test_zero_capacity_rejected():
    import pytest
    Float64Vec, _ = make_vector(nb_types.float64)

    @njit
    def go():
        return Float64Vec(0)

    with pytest.raises(AssertionError):
        go()


def test_cache_survives_across_processes(tmp_path):
    # numbox's make_structref emits @njit(cache=True) proxy accessors. If the
    # StructRef type class for a vector is created per-process (e.g. via
    # type(...) inside a function body), run 2 loads cached code that expects
    # the run-1 class object and fails with "No conversion from X to X".
    probe = textwrap.dedent("""
        from numba import types as nb_types
        from numbduck.vector import make_vector
        create, _ = make_vector(nb_types.float64)
        v = create(8)
        print(v.size)
    """)
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path)}

    r1 = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True,
    )
    assert r1.returncode == 0, f"run1 (cold) failed:\n{r1.stderr}"

    r2 = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True,
    )
    assert r2.returncode == 0, f"run2 (warm) failed:\n{r2.stderr}"
