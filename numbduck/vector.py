import operator

import numpy
from numba import njit, types as nb_types
from numba.experimental import structref
from numba.extending import overload
from numbox.utils.highlevel import make_structref


class VectorType(nb_types.StructRef):
    # Abstract base for all Vector[T] structrefs. Must NOT be registered with
    # structref.register — only the dynamic concrete subclasses are.
    # `preprocess_fields` defends against literal-type callers; resolved-type
    # callers are unaffected. Placement on the base (inherited by concrete
    # subclasses) is intentional — moving the override into the concrete
    # subclass directly works the same.
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


_vector_cache = {}


def make_vector(elem_type):
    """Return ``(create, type_instance)`` for ``Vector[elem_type]``.

    - ``create``: an ``@njit`` factory taking a single ``capacity`` argument.
      Dtype is locked by the type — callers cannot pass a mismatched buffer.
    - ``type_instance``: the concrete ``VectorType`` instance with resolved
      field types. Use as a field type in other structrefs and as the first
      arg to ``borrow_structref``.

    Results are memoized in ``_vector_cache`` keyed by ``elem_type.key``.

    Single-shot per element type: once registered with numba, the concrete
    ``Vector_{name}_Type`` class cannot be re-created. Popping the cache and
    re-running ``make_vector(t)`` registers a second class with the same name
    and produces "No conversion from X to X" type-identity errors at compile
    time. Tests must not flush the cache.

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

    type_cls = type(
        f"Vector_{elem_type.name}_Type",
        (VectorType,),
        {},
    )
    structref.register(type_cls)

    fields = {
        "buf": nb_types.Array(elem_type, 1, 'C'),
        "size": nb_types.int64,
    }

    proxy_cls = make_structref(
        f"Vector_{elem_type.name}",
        fields,
        type_cls,
    )

    np_dtype = numpy.dtype(str(elem_type))

    @njit
    def create(capacity):
        assert capacity >= 1
        return proxy_cls(numpy.empty(capacity, dtype=np_dtype), 0)

    type_inst = type_cls([
        ("buf", nb_types.Array(elem_type, 1, 'C')),
        ("size", nb_types.int64),
    ])

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
