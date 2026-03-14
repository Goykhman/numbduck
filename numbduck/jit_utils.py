# These utilities are not DuckDB-specific and may be better moved to numbox
# in the future.
from llvmlite import ir as llvmir
from numba import njit
from numba.core import errors
from numba.core.types import int32, intp, CPointer
from numba.extending import intrinsic


@njit
def array_data_p(arr):
    """Return the data pointer of a numpy array as intp.

    Wraps arr.ctypes.data (which returns uint64 in numba) with an intp cast
    so the result matches the signed-integer pointer convention used by numbox
    binding signatures.

    Callable from both Python and @njit context.
    """
    return intp(arr.ctypes.data)


@intrinsic
def i32_ptr(typingctx, ptr_ty):
    """Cast an intp to CPointer(int32) for use with numba.carray."""
    if ptr_ty != intp:
        raise errors.TypingError(f"i32_ptr expects intp, got {ptr_ty}")
    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], llvmir.IntType(32).as_pointer())
    return CPointer(int32)(intp,), codegen
