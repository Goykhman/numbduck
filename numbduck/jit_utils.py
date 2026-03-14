# These utilities are not DuckDB-specific and may be better moved to numbox
# in the future.
from numba import njit
from numba.core.types import intp


@njit
def array_data_p(arr):
    """Return the data pointer of a numpy array as intp.

    Wraps arr.ctypes.data (which returns uint64 in numba) with an intp cast
    so the result matches the signed-integer pointer convention used by numbox
    binding signatures.

    Callable from both Python and @njit context.
    """
    return intp(arr.ctypes.data)
