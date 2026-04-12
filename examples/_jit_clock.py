"""Cross-platform monotonic nanosecond clock callable from @njit code.

Exposes a single function::

    monotonic_ns() -> int64   # nanoseconds since an unspecified epoch

Implemented as a Numba ``@intrinsic`` that emits an LLVM ``alloca`` for
the platform's time struct, calls the OS clock function into it, and
returns the result — all on the stack, with zero heap allocation and no
NRT liveness concerns.

Platform implementations
------------------------
Linux / macOS:
    Calls libc ``clock_gettime(CLOCK_MONOTONIC, &ts)`` where ``ts`` is a
    stack-allocated ``struct timespec {int64 tv_sec; int64 tv_nsec}``.
    ``clock_gettime`` is resolved via ``address_of_symbol`` from
    RTLD_DEFAULT — libc is always globally loaded.

Windows:
    Calls ``QueryPerformanceCounter`` from ``kernel32.dll``.  The
    performance-counter frequency is read once at module import via
    ctypes and baked into the IR as a compile-time constant.
    ``kernel32`` is registered with LLVM's symbol search via
    ``load_library_permanently``.
"""
import ctypes
import platform
import time

import llvmlite.binding as ll
from llvmlite import ir
from numba.core.cgutils import get_or_insert_function
from numba.core.types import int64
from numba.extending import intrinsic

_SYSTEM = platform.system()

_i32 = ir.IntType(32)
_i64 = ir.IntType(64)
_i32_0 = ir.Constant(_i32, 0)
_i32_1 = ir.Constant(_i32, 1)
_BILLION = ir.Constant(_i64, 1_000_000_000)


if _SYSTEM == "Windows":
    ll.load_library_permanently("kernel32.dll")

    _freq_buf = ctypes.c_int64(0)
    ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(_freq_buf))
    _QPC_FREQ = int(_freq_buf.value)
    if _QPC_FREQ <= 0:
        raise RuntimeError(
            "QueryPerformanceFrequency returned a non-positive value")
    _QPC_FREQ_CONST = ir.Constant(_i64, _QPC_FREQ)

    @intrinsic
    def monotonic_ns(typingctx):
        """Stack-only monotonic clock via QueryPerformanceCounter."""
        def codegen(context, builder, signature, arguments):
            counter_ptr = builder.alloca(_i64)
            fn_ty = ir.FunctionType(_i32, [_i64.as_pointer()])
            fn = get_or_insert_function(
                builder.module, fn_ty, "QueryPerformanceCounter")
            builder.call(fn, [counter_ptr])
            ticks = builder.load(counter_ptr)
            numer = builder.mul(ticks, _BILLION)
            return builder.sdiv(numer, _QPC_FREQ_CONST)
        return int64(), codegen

else:
    _CLOCK_MONOTONIC = getattr(time, "CLOCK_MONOTONIC", 1)
    _CLK_ID = ir.Constant(_i32, _CLOCK_MONOTONIC)
    _timespec_ty = ir.LiteralStructType([_i64, _i64])

    @intrinsic
    def monotonic_ns(typingctx):
        """Stack-only monotonic clock via clock_gettime."""
        def codegen(context, builder, signature, arguments):
            ts_ptr = builder.alloca(_timespec_ty)
            fn_ty = ir.FunctionType(_i32, [_i32, _timespec_ty.as_pointer()])
            fn = get_or_insert_function(
                builder.module, fn_ty, "clock_gettime")
            builder.call(fn, [_CLK_ID, ts_ptr])
            sec = builder.load(builder.gep(ts_ptr, [_i32_0, _i32_0]))
            nsec = builder.load(builder.gep(ts_ptr, [_i32_0, _i32_1]))
            return builder.add(builder.mul(sec, _BILLION), nsec)
        return int64(), codegen
