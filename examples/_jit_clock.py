"""Cross-platform monotonic nanosecond clock callable from @njit code.

Exposes a single @cres function::

    monotonic_ns(scratch_p) -> int64   # nanoseconds since an unspecified epoch

where ``scratch_p`` is the address of a caller-owned 16-byte scratch buffer
(e.g. ``intp(numpy.zeros(2, dtype=numpy.int64).ctypes.data)``). The caller
allocates the buffer once and passes it on every call so the clock helper
itself does zero allocations — a prerequisite for using it to measure
nanosecond-scale latencies inside a hot JIT loop.

Platform implementations
------------------------
Linux / macOS:
    Binds libc ``clock_gettime(CLOCK_MONOTONIC, &ts)``. The scratch buffer is
    read back as a 2-element int64 ``carray`` holding ``tv_sec`` and
    ``tv_nsec``. Works because ``dlsym(RTLD_DEFAULT)`` — which is what
    llvmlite's ``address_of_symbol`` uses on POSIX — finds symbols in any
    globally-loaded shared library, and libc is always globally loaded.

Windows:
    Binds ``QueryPerformanceCounter`` and ``QueryPerformanceFrequency`` from
    ``kernel32.dll``. The frequency is read once at module import via ctypes
    and baked into the ``monotonic_ns`` body as a compile-time constant.
    ``kernel32`` is loaded explicitly into LLVM's symbol search via
    ``ll.load_library_permanently('kernel32.dll')`` so that
    ``address_of_symbol`` can resolve ``QueryPerformance*``.

The single ``monotonic_ns(scratch_p)`` callable has the same signature on
every platform; the implementation is selected at module import time based
on ``platform.system()``.
"""
import ctypes
import platform
import time

import llvmlite.binding as ll
import numpy
from numba import carray
from numba.core.types import int32, int64, intp
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numbox.utils.highlevel import cres
from numbox.utils.lowlevel import _cast_int_to_void_p

_SYSTEM = platform.system()


if _SYSTEM == "Windows":
    # Make kernel32 visible to llvmlite's global symbol search. It is always
    # loaded in every Windows process, but load_library_permanently also
    # registers it with LLVM's internal dynamic-library table so that
    # address_of_symbol can find QueryPerformance* by name.
    ll.load_library_permanently("kernel32.dll")

    signatures["QueryPerformanceCounter"] = int32(intp)

    @cres(int32(intp))
    def _qpc(counter_p):
        return _call_lib_func("QueryPerformanceCounter", (counter_p,))

    # Read the performance-counter frequency (ticks per second) exactly once
    # at module import via ctypes and bake it into monotonic_ns() as a
    # compile-time constant. The frequency is invariant for the lifetime of
    # the OS boot, so this is safe and avoids a per-call library call.
    _freq_buf = ctypes.c_int64(0)
    ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(_freq_buf))
    _QPC_FREQ = int(_freq_buf.value)
    if _QPC_FREQ <= 0:
        raise RuntimeError("QueryPerformanceFrequency returned a non-positive value")
    _QPC_FREQ_CONST = numpy.int64(_QPC_FREQ)

    @cres(int64(intp))
    def monotonic_ns(scratch_p):
        _qpc(scratch_p)
        scratch = carray(_cast_int_to_void_p(scratch_p), (1,), dtype=numpy.int64)
        ticks = scratch[0]
        return (ticks * numpy.int64(1_000_000_000)) // _QPC_FREQ_CONST

else:
    # Linux / Darwin: clock_gettime(CLOCK_MONOTONIC, &ts). The scratch buffer
    # is a struct timespec laid out as { int64 tv_sec; int64 tv_nsec } on all
    # 64-bit POSIX systems we care about.
    signatures["clock_gettime"] = int32(int32, intp)
    _CLOCK_MONOTONIC = getattr(time, "CLOCK_MONOTONIC", 1)

    @cres(int32(int32, intp))
    def _clock_gettime(clk_id, ts_p):
        return _call_lib_func("clock_gettime", (clk_id, ts_p))

    _CLOCK_MONOTONIC_CONST = numpy.int32(_CLOCK_MONOTONIC)

    @cres(int64(intp))
    def monotonic_ns(scratch_p):
        _clock_gettime(_CLOCK_MONOTONIC_CONST, scratch_p)
        scratch = carray(_cast_int_to_void_p(scratch_p), (2,), dtype=numpy.int64)
        return scratch[0] * numpy.int64(1_000_000_000) + scratch[1]
