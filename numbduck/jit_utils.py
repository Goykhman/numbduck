from numba import njit
from numba.core.types import intp
from numba.extending import intrinsic
from numba.np.arrayobj import make_array


@intrinsic
def _array_data_p(typingctx, arr_ty):
    """Extract the data pointer from a numpy array as an intp."""
    def codegen(context, builder, signature, arguments):
        arr_struct = make_array(arr_ty)(context, builder, arguments[0])
        return builder.ptrtoint(arr_struct.data, context.get_data_type(intp))
    return intp(arr_ty), codegen


@njit
def array_data_p(arr):
    """Return the data pointer of a numpy array. Callable from Python and @njit."""
    return _array_data_p(arr)
