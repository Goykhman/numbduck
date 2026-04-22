import operator

import numpy
from numba import njit, types as nb_types
from numba.experimental import structref
from numba.experimental.structref import StructRefProxy, define_boxing, new
from numba.extending import overload


@structref.register
class VectorType(nb_types.StructRef):
    # Single module-level class, parameterized per elem_type via field-tuple
    # instances (same pattern as numbox WorkTypeClass / AnyTypeClass). Dynamic
    # subclasses lack stable identity across processes and break numba's disk
    # cache of any code that references the type.
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


class Vector(StructRefProxy):
    def __new__(cls, *args, **kwargs):
        raise NotImplementedError(
            "Use the factory returned by make_vector(elem_type)"
        )

    @property
    @njit(cache=True)
    def size(self):
        return self.size

    @property
    @njit(cache=True)
    def buf(self):
        return self.buf


define_boxing(VectorType, Vector)


_vector_cache = {}


def make_vector(elem_type):
    """Return ``(create, type_instance)`` for ``Vector[elem_type]``.

    - ``create``: an ``@njit`` factory taking a single ``capacity`` argument.
      Dtype is locked by the type — callers cannot pass a mismatched buffer.
    - ``type_instance``: the ``VectorType`` instance with resolved field
      types. Use as a field type in other structrefs and as the first arg
      to ``borrow_structref``.

    Results are memoized in ``_vector_cache`` keyed by ``elem_type.key``.

    The numpy dtype is derived from ``str(elem_type)`` at build time. Works
    for standard scalar numba types (float64, int64, etc.). Exotic types
    where ``str()`` does not match a numpy dtype name are unsupported.

    Initial ``capacity`` must be ``>= 1``. The ``create`` factory asserts
    this. Zero-capacity construction is rejected because the geometric
    growth in ``vector_push`` / ``vector_extend`` would produce ``0 * 2 = 0``
    and either OOB or infinite-loop. Vectors only grow, never shrink, so a
    positive initial capacity guarantees a positive capacity for all time.
    """
    key = elem_type.key
    if key in _vector_cache:
        return _vector_cache[key]

    type_inst = VectorType([
        ("buf", nb_types.Array(elem_type, 1, 'C')),
        ("size", nb_types.int64),
    ])

    np_dtype = numpy.dtype(str(elem_type))

    @njit
    def create(capacity):
        assert capacity >= 1
        v = new(type_inst)
        v.buf = numpy.empty(capacity, dtype=np_dtype)
        v.size = 0
        return v

    result = (create, type_inst)
    _vector_cache[key] = result
    return result


@overload(len)
def _vector_len(v):
    if isinstance(v, VectorType):
        def impl(v):
            return v.size
        return impl


@overload(operator.getitem)
def _vector_getitem(v, i):
    if isinstance(v, VectorType):
        def impl(v, i):
            return v.buf[i]
        return impl


@overload(operator.setitem)
def _vector_setitem(v, i, val):
    if isinstance(v, VectorType):
        def impl(v, i, val):
            v.buf[i] = val
        return impl


@njit
def vector_push(v, val):
    if v.size == v.buf.shape[0]:
        new_buf = numpy.empty(v.buf.shape[0] * 2, v.buf.dtype)
        new_buf[:v.size] = v.buf[:v.size]
        v.buf = new_buf
    v.buf[v.size] = val
    v.size += 1


@njit
def vector_extend(dst, src):
    needed = dst.size + src.size
    cap = dst.buf.shape[0]
    if needed > cap:
        while cap < needed:
            cap *= 2
        new_buf = numpy.empty(cap, dst.buf.dtype)
        new_buf[:dst.size] = dst.buf[:dst.size]
        dst.buf = new_buf
    dst.buf[dst.size:dst.size + src.size] = src.buf[:src.size]
    dst.size += src.size
