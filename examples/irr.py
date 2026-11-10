"""An aggregate function for DuckDB, built with numbduck, that computes the internal rate of return of each group.

``irr(cashflow, period, investment, target_npv)`` returns the monthly rate ``r`` that solves this equation over the
rows of a group::

    -investment - target_npv + sum(cashflow[i] / (1 + r) ** period[i]) = 0

For example, ``SELECT irr(cashflow, period, investment, target_npv) FROM monthly_data`` computes the rate of a whole
table, and a ``GROUP BY`` computes one rate per group. ``irr_bisect`` solves the equation by bisection.

The module shows how to build an aggregate function from scratch. The state of each group is a numba structref,
defined with numbox's ``make_structref``, that collects the group's cashflows and periods. Six callbacks give DuckDB
the size of a state and create, fill, merge, finalize and free the states. ``register_irr`` registers them through
DuckDB's C API. Unlike the other examples, this one compares nothing: ``main`` checks its results against known
answers.

A row in which any of the four arguments is NULL is skipped. The result is NaN for a group without a usable row, for a
group whose state could not be created, and for a NaN among the values that the solver uses. A NaN is a legal DOUBLE,
so the NULL check keeps its row. NaN therefore means that the group has no usable rate. The result is +inf,
``IRR_NO_BRACKET``, when NPV has the same sign at the monthly rates -0.99 and 10.0, as it has for a root outside that
range. A caller can thus distinguish "no root in range" from "no data".

``investment`` and ``target_npv`` are constants of the group. The aggregate takes them from the first usable row that
a state receives, or, when DuckDB merges partial states, from the first state that has them. It ignores the values of
later rows, so a caller passes the same two values in every row of a group.

Run the example with ``python examples/run_irr.py``, whose docstring explains why ``irr.py`` must not run as a script.
Import the module under the name ``irr`` only, as ``run_irr.py`` and the tests do. numba caches the code that
``make_structref`` generates under a key that does not include the importing module, and the cached entries name the
class ``irr.IRRStateType``. Once a run has cached them, an import as ``examples.irr`` fails with
``ModuleNotFoundError: No module named 'irr'``.
"""
import math
import sys

# A run as a script stops here, before anything is compiled; run_irr.py explains why it must not go further.
if __name__ == "__main__":
    sys.stderr.write(
        "examples/irr.py is the importable example module; run examples/run_irr.py instead.\n"
        "Running irr.py as __main__ gives IRRStateType a fresh class identity each process,\n"
        "which fails type inference on a warm numba cache.\n"
    )
    sys.exit(1)

import duckdb
import numpy
from numba import cfunc, carray, njit
from numba import types as nb_types
from numba.experimental import structref
from numbox.core.vector.vector import make_vector, vector_push, vector_extend
from numbox.utils.highlevel import make_structref
from numbox.utils.cstrings import c_string
from numbox.utils.lowlevel import _cast_int_to_void_p
from numbox.utils.meminfo import borrow_structref, export_meminfo, release_meminfo

from numbduck import ducklib
from numbduck.pybridge import extract_connection_ptr


Float64Vector, float64_vec_type = make_vector(nb_types.float64)


# ---- IRR state structref ----
#
# Each group's state is an IRRState: two growable float64 vectors from numbox's make_vector, which hold the group's
# cashflows and periods in pairs, the group's investment and target NPV, and a flag that is 1 once those two are set.
# make_structref generates the class. Its numba type, IRRStateType, must be defined in a module that the program
# imports, for the reason that run_irr.py gives.

@structref.register
class IRRStateType(nb_types.StructRef):
    pass


_irr_state_fields = [
    ("cashflows", float64_vec_type),
    ("periods", float64_vec_type),
    ("investment", nb_types.float64),
    ("target_npv", nb_types.float64),
    ("initialized", nb_types.int64),
]
IRRState = make_structref("IRRState", dict(_irr_state_fields), IRRStateType)
irr_state_type = IRRStateType(_irr_state_fields)


# ---- Bisection solver ----
#
# The monthly IRR is the rate r at which a group's NPV is zero. The solver assumes that NPV(r) changes sign exactly
# once between -0.99 and 10.0. A stream whose cashflows change sign once, such as an investment followed by returns,
# has at most one such root, and its NPV falls as r rises. The loop stops when the rate bracket is narrow enough, not
# when NPV is small enough. At the root, the smallest NPV that floating point can reach grows with the cashflows, as
# about |dNPV/dr| * ulp(r). A fixed tolerance on NPV therefore cannot be met for large amounts, even once r is exact
# to machine precision. The width test does not depend on the amounts, and it always ends with the converged rate.

#: What irr_bisect returns when NPV has the same sign at both ends of the bracket: +inf, which differs from the NaN of
#: a group without data.
IRR_NO_BRACKET = math.inf


@njit
def _irr_npv_sign(cashflows, periods, n, investment, target_npv, r):
    """Return a number with the sign of NPV(r) over the first ``n`` cashflows, without a division by zero.

    Bisection needs only the sign of NPV at each probe rate, not its value. The result is NaN if a value that the
    function uses is NaN, and exactly 0.0, which has no sign, if the constant and every cashflow are zero. The sum
    that gives the sign overflows only if the magnitudes of the constant and the cashflows add up to more than the
    largest float64, about 1.8e308.
    """
    # NPV(r) = -investment - target_npv + sum_i cashflows[i] * (1 + r)**(-periods[i])
    #        = -investment - target_npv + sum_i cashflows[i] * exp(-periods[i] * L),  with L = log1p(r).
    # Let M be the largest exponent -periods[i] * L among the nonzero terms. The constant c = -investment - target_npv
    # counts as a term of exponent 0 if it is nonzero. Because exp(M) > 0, NPV(r) has the sign of
    #     c * exp(-M) + sum_i cashflows[i] * exp(-periods[i] * L - M).
    # No scaled term exceeds its cashflow, or c, in magnitude, so the sum overflows only if the magnitudes of c and the
    # cashflows add up to more than the largest float64. An unscaled NPV, which divides each cashflow by (1 + r)**p,
    # fails at r = -0.99. There (1 + r)**p underflows to 0.0 once p reaches about 162, and the division gives an
    # infinite term. A zero cashflow at such a period gives 0 / 0 = NaN, and two such cashflows of opposite signs give
    # inf - inf = NaN.
    #
    # The loops skip zero cashflows. A zero cashflow adds nothing at any rate, but at a far period it would raise M and
    # make the real terms underflow to a false 0.0.
    log1p_r = math.log1p(r)
    const = -investment - target_npv
    have_term = const != 0.0
    max_exp = 0.0
    for i in range(n):
        if cashflows[i] == 0.0:
            continue
        e = -periods[i] * log1p_r
        if not have_term or e > max_exp:
            max_exp = e
            have_term = True
    if not have_term:
        return 0.0
    scaled = 0.0
    if const != 0.0:
        scaled += const * math.exp(-max_exp)
    for i in range(n):
        if cashflows[i] == 0.0:
            continue
        scaled += cashflows[i] * math.exp(-periods[i] * log1p_r - max_exp)
    return scaled


@njit
def irr_bisect(cashflows, periods, n, investment, target_npv):
    """Return the monthly IRR of the first ``n`` cashflows, found by bisection between the rates -0.99 and 10.0.

    The loop halves the bracket until it is narrower than 1e-12 and returns its midpoint, which then lies within 5e-13
    of the root, however large the cashflows are. The solver assumes that NPV changes sign once in the bracket, as it
    does for an investment followed by returns, or for a loan received and then repaid. It checks only the two ends:

    - If NPV is NaN at either end, because a value that the solver uses is NaN, the result is NaN, as for a group
      without data.
    - If NPV has the same sign at both ends, the result is ``IRR_NO_BRACKET``, +inf. That covers a root below -0.99 or
      above 10.0, which means a monthly loss of more than 99% or a gain of more than 1000%. It also covers an even
      number of roots in the bracket. A caller can distinguish this +inf, "no single root in the bracket", from the NaN
      of "no data".

    A stream that changes sign more than once can have several roots. The two ends have opposite signs when the bracket
    holds an odd number of them, so the solver converges to one of the roots and gives no warning that there are
    others. Pass only streams with one sign change where a single IRR matters.
    """
    r_lo = -0.99
    r_hi = 10.0
    sign_lo = _irr_npv_sign(cashflows, periods, n, investment, target_npv, r_lo)
    sign_hi = _irr_npv_sign(cashflows, periods, n, investment, target_npv, r_hi)
    if math.isnan(sign_lo) or math.isnan(sign_hi):
        # A NaN is a legal DOUBLE, not SQL NULL, so update keeps its row, and the NaN reaches the sign sum. The solver
        # returns NaN, not the +inf that a caller would read as no single root in the bracket. The scaled sum does not
        # turn a far period into NaN, so a NaN here comes from a NaN input.
        return math.nan
    if (sign_lo > 0.0) == (sign_hi > 0.0):
        # The ends have the same sign, so no single sign change lies between them. The test counts a sign sum of
        # exactly 0.0 as negative, so a group whose constant and cashflows are all zero also gets IRR_NO_BRACKET.
        return IRR_NO_BRACKET
    # NPV falls across the bracket for an investment followed by returns, and it rises for a loan received and then
    # repaid. Each step therefore compares the sign at the midpoint with the sign at r_lo. If they match, the sign
    # change lies above the midpoint and r_lo moves up; otherwise r_hi moves down.
    lo_positive = sign_lo > 0.0
    rate_tol = 1e-12
    for _ in range(100):
        if (r_hi - r_lo) < rate_tol:
            break
        r_mid = (r_lo + r_hi) / 2.0
        sign_mid = _irr_npv_sign(cashflows, periods, n, investment, target_npv, r_mid)
        if (sign_mid > 0.0) == lo_positive:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return (r_lo + r_hi) / 2.0


# ---- DuckDB aggregate callbacks ----
#
# DuckDB calls the callbacks through the function pointers that register_irr passes. state_size gives the size of a
# state, init creates each new state, and update adds each chunk of input rows. combine merges the partial states of
# parallel threads, finalize writes the results, and destroy frees the states. Each callback is a @cfunc that receives
# the addresses as intp integers and forwards them to an @njit implementation. A group's state is an 8-byte slot that
# holds the address of the group's IRRState. init stores the address with numbox's export_meminfo. update, combine
# and finalize borrow the structref with borrow_structref, and destroy releases it with release_meminfo.
#
# If an exception escapes an @njit function called from a @cfunc, numba prints the exception and returns to DuckDB
# without an error, so the query continues with a wrong result. Suppose the exception comes from a nested call, such as
# vector_push, vector_extend or irr_bisect, while a borrowed structref is alive. numba then also skips the decref that
# ends the borrow, and the state's memory leaks. update, combine and finalize therefore run their work inside a bare
# try/except. Catching the exception in the same frame runs the decref and lets the callback write a defined result
# instead. A try/finally is not enough: the exception escapes after the finally block, on numba 0.65.1 as an
# AssertionError that numba itself raises. DuckDB again receives no error, although the decref of the borrow runs. init
# guards its allocations the same way. state_size and destroy borrow nothing and cannot raise, so they need no guard.

@njit
def _irr_state_size_impl(info):
    """Return the size of a group's state in bytes: 8, for the address of the group's IRRState."""
    return numpy.uint64(8)


@cfunc(nb_types.uint64(nb_types.intp))
def _irr_state_size_cb(info):
    """The ``duckdb_aggregate_state_size`` callback that DuckDB calls."""
    return _irr_state_size_impl(info)


@njit
def _irr_init_impl(info, state):
    """Create an empty IRRState and store its address in the state slot at ``state``, or 0 if the creation fails.

    Every later callback skips a slot that holds 0, and finalize reports its group as NaN.
    """
    slot = carray(_cast_int_to_void_p(state), (1,), dtype=numpy.intp)
    # An allocation that fails raises, and the guard turns the failure into the 0 that later callbacks skip.
    try:
        cfs = Float64Vector(64)
        pds = Float64Vector(64)
        s = IRRState(cfs, pds, 0.0, 0.0, 0)
        slot[0] = export_meminfo(s)
    except Exception:
        slot[0] = 0


@cfunc(nb_types.void(nb_types.intp, nb_types.intp))
def _irr_init_cb(info, state):
    """The ``duckdb_aggregate_init_t`` callback that DuckDB calls."""
    _irr_init_impl(info, state)


@njit
def _irr_update_impl(info, chunk, states):
    """Append the cashflow and period of each row of ``chunk`` without a NULL to the state of the row's group.

    ``states`` is the address of an array that holds one state address per row of the chunk. The first row that a
    state receives also sets the state's investment and target NPV.
    """
    n = ducklib.duckdb_data_chunk_get_size(chunk)
    vec_cf = ducklib.duckdb_data_chunk_get_vector(chunk, 0)
    vec_pd = ducklib.duckdb_data_chunk_get_vector(chunk, 1)
    vec_inv = ducklib.duckdb_data_chunk_get_vector(chunk, 2)
    vec_npv = ducklib.duckdb_data_chunk_get_vector(chunk, 3)
    cf_data = carray(_cast_int_to_void_p(ducklib.duckdb_vector_get_data(vec_cf)), (n,), dtype=numpy.float64)
    pd_data = carray(_cast_int_to_void_p(ducklib.duckdb_vector_get_data(vec_pd)), (n,), dtype=numpy.float64)
    inv_data = carray(_cast_int_to_void_p(ducklib.duckdb_vector_get_data(vec_inv)), (n,), dtype=numpy.float64)
    npv_data = carray(_cast_int_to_void_p(ducklib.duckdb_vector_get_data(vec_npv)), (n,), dtype=numpy.float64)
    val_cf = ducklib.duckdb_vector_get_validity(vec_cf)
    val_pd = ducklib.duckdb_vector_get_validity(vec_pd)
    val_inv = ducklib.duckdb_vector_get_validity(vec_inv)
    val_npv = ducklib.duckdb_vector_get_validity(vec_npv)
    state_slots = carray(_cast_int_to_void_p(states), (n,), dtype=numpy.intp)
    for i in range(n):
        if val_cf != 0 and not ducklib.duckdb_validity_row_is_valid(val_cf, i):
            continue
        if val_pd != 0 and not ducklib.duckdb_validity_row_is_valid(val_pd, i):
            continue
        if val_inv != 0 and not ducklib.duckdb_validity_row_is_valid(val_inv, i):
            continue
        if val_npv != 0 and not ducklib.duckdb_validity_row_is_valid(val_npv, i):
            continue
        slot = carray(_cast_int_to_void_p(state_slots[i]), (1,), dtype=numpy.intp)
        # A slot of 0 marks a state that init failed to create. A borrow of it would read address 0 and crash the
        # process, which no try/except can catch, so the row is skipped.
        if slot[0] == 0:
            continue
        s = borrow_structref(irr_state_type, slot[0])
        n0 = s.cashflows.size
        # The guard keeps an exception in this frame, as the section comment above explains. A push fails if the
        # vector's buffer cannot grow. The except branch then restores both vectors to their length before the row,
        # so cashflows and periods stay paired: the row is dropped instead of shifting every later pair.
        try:
            vector_push(s.cashflows, cf_data[i])
            vector_push(s.periods, pd_data[i])
            if s.initialized == 0:
                s.investment = inv_data[i]
                s.target_npv = npv_data[i]
                s.initialized = 1
        except Exception:
            s.cashflows.size = n0
            s.periods.size = n0


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _irr_update_cb(info, chunk, states):
    """The ``duckdb_aggregate_update_t`` callback that DuckDB calls."""
    _irr_update_impl(info, chunk, states)


@njit
def _irr_combine_impl(info, source, target, count):
    """Append the rows of each of the ``count`` source states to the target state at the same position.

    ``source`` and ``target`` are the addresses of two arrays of ``count`` state addresses. A target state without an
    investment and target NPV takes them from its source state.
    """
    src_slots = carray(_cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    tgt_slots = carray(_cast_int_to_void_p(target), (count,), dtype=numpy.intp)
    for i in range(count):
        src_slot = carray(_cast_int_to_void_p(src_slots[i]), (1,), dtype=numpy.intp)
        tgt_slot = carray(_cast_int_to_void_p(tgt_slots[i]), (1,), dtype=numpy.intp)
        # A slot of 0 on either side marks a state that init failed to create, and a borrow of it would crash the
        # process.
        if src_slot[0] == 0 or tgt_slot[0] == 0:
            continue
        src = borrow_structref(irr_state_type, src_slot[0])
        tgt = borrow_structref(irr_state_type, tgt_slot[0])
        n0 = tgt.cashflows.size
        # The guard keeps an exception in this frame, so both borrows end. If the second extend fails after the first
        # has grown tgt.cashflows, the except branch restores both of the target's vectors to the length n0, so they
        # stay paired.
        try:
            vector_extend(tgt.cashflows, src.cashflows)
            vector_extend(tgt.periods, src.periods)
            if tgt.initialized == 0:
                tgt.investment = src.investment
                tgt.target_npv = src.target_npv
                tgt.initialized = src.initialized
        except Exception:
            tgt.cashflows.size = n0
            tgt.periods.size = n0


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp, nb_types.uint64))
def _irr_combine_cb(info, source, target, count):
    """The ``duckdb_aggregate_combine_t`` callback that DuckDB calls."""
    _irr_combine_impl(info, source, target, count)


@njit
def _irr_finalize_impl(info, source, result, count, offset):
    """Write the IRR of each of the ``count`` source states to rows ``offset`` onward of the vector ``result``.

    ``source`` is the address of an array of ``count`` state addresses, and ``result`` is the DOUBLE output vector. A
    state without rows, a state that init failed to create, and a failure inside the callback all give NaN.
    """
    out_data = ducklib.duckdb_vector_get_data(result)
    src_slots = carray(_cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    out_vals = carray(_cast_int_to_void_p(out_data), (offset + count,), dtype=numpy.float64)
    for i in range(count):
        # On a failure, the row gets NaN, as for an empty group. The borrow is inside the guard, so an exception ends
        # the borrow instead of leaking it.
        try:
            src_slot = carray(_cast_int_to_void_p(src_slots[i]), (1,), dtype=numpy.intp)
            # A slot of 0 marks a state that init failed to create. The row gets NaN, because a borrow of the slot
            # would crash the process.
            if src_slot[0] == 0:
                out_vals[offset + i] = math.nan
                continue
            s = borrow_structref(irr_state_type, src_slot[0])
            # update and combine keep the two vectors the same length. As a safeguard, the shorter length bounds the
            # reads, so _irr_npv_sign cannot index past the end of either vector.
            n = min(len(s.cashflows), len(s.periods))
            if n == 0:
                out_vals[offset + i] = math.nan
            else:
                out_vals[offset + i] = irr_bisect(
                    s.cashflows, s.periods, n, s.investment, s.target_npv)
        except Exception:
            out_vals[offset + i] = math.nan


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp, nb_types.uint64, nb_types.uint64))
def _irr_finalize_cb(info, source, result, count, offset):
    """The ``duckdb_aggregate_finalize_t`` callback that DuckDB calls."""
    _irr_finalize_impl(info, source, result, count, offset)


@njit
def _irr_destroy_impl(states, count):
    """Release the IRRState of each of the ``count`` states in the array of state addresses at ``states``."""
    state_slots = carray(_cast_int_to_void_p(states), (count,), dtype=numpy.intp)
    for i in range(count):
        slot = carray(_cast_int_to_void_p(state_slots[i]), (1,), dtype=numpy.intp)
        # A slot of 0, which a failed init leaves, holds nothing to release. release_meminfo decrements the reference
        # count directly and cannot raise, so destroy needs no guard.
        if slot[0] != 0:
            release_meminfo(slot[0])


@cfunc(nb_types.void(nb_types.intp, nb_types.uint64))
def _irr_destroy_cb(states, count):
    """The ``duckdb_aggregate_destroy_t`` callback that DuckDB calls."""
    _irr_destroy_impl(states, count)


# ---- Registration and query ----

def register_irr(conn):
    """Register the aggregate ``irr`` of four DOUBLE arguments, which returns a DOUBLE, on ``conn`` through the C API.

    ``AssertionError`` is raised if DuckDB refuses the registration.
    """
    conn_p = extract_connection_ptr(conn)

    func_p = ducklib.duckdb_create_aggregate_function()
    # c_string passes the name as NUL-terminated UTF-8, in a buffer that lives until the with block ends. DuckDB copies
    # the name, so the buffer is not needed after the call.
    with c_string("irr") as name_p:
        ducklib.duckdb_aggregate_function_set_name(func_p, name_p)

    dbl_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
    for _ in range(4):
        ducklib.duckdb_aggregate_function_add_parameter(func_p, dbl_type_p)
    ducklib.duckdb_aggregate_function_set_return_type(func_p, dbl_type_p)
    # The parameters and the return type hold copies of the logical type, so the type can be destroyed now.
    tp_buf = numpy.array([dbl_type_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_logical_type(tp_buf.ctypes.data)

    ducklib.duckdb_aggregate_function_set_functions(
        func_p,
        _irr_state_size_cb.address,
        _irr_init_cb.address,
        _irr_update_cb.address,
        _irr_combine_cb.address,
        _irr_finalize_cb.address,
    )
    ducklib.duckdb_aggregate_function_set_destructor(func_p, _irr_destroy_cb.address)

    rc = ducklib.duckdb_register_aggregate_function(conn_p, func_p)
    assert rc == ducklib.DuckDBSuccess, f"Registration failed, rc={rc}"

    func_buf = numpy.array([func_p], dtype=numpy.intp)
    ducklib.duckdb_destroy_aggregate_function(func_buf.ctypes.data)


def main():
    """Run three queries of ``irr`` with known answers, check each answer, and check that no state memory leaked.

    A wrong answer raises ``AssertionError``. If numba's runtime allocated more than it freed during the run, the
    function prints a warning and exits with status 1.
    """
    print("IRR UDAF example")
    print("=" * 40)

    # numba's runtime counts its allocations and frees once its statistics are on. Both counts must grow by the same
    # amount over the run, or the callbacks leaked a state.
    from numba.core.runtime import nrt
    nrt._nrt.memsys_enable_stats()
    stats_before = nrt.rtsys.get_allocation_stats()

    conn = duckdb.connect()
    register_irr(conn)

    # Test 1: uniform cashflows, target NPV = 0
    #   10,000 investment, 12 months of 1,000 each
    #   Expected: monthly rate where NPV = 0
    conn.execute("""
        CREATE TABLE test_irr AS
        SELECT
            (range + 1)::DOUBLE AS period,
            1000.0 AS cashflow,
            10000.0 AS investment,
            0.0 AS target_npv
        FROM range(12)
    """)

    result = conn.execute("SELECT irr(cashflow, period, investment, target_npv) FROM test_irr").fetchone()
    irr_val = result[0]

    # At the rate found, NPV must be zero to within 1e-6.
    npv_check = -10000.0
    for t in range(1, 13):
        npv_check += 1000.0 / (1.0 + irr_val) ** t
    assert abs(npv_check) < 1e-6, f"NPV check failed: {npv_check}"

    print("\nTest 1: uniform cashflows")
    print("  Investment: 10,000 | Cashflows: 12 x 1,000 | Target NPV: 0")
    print(f"  IRR (monthly): {irr_val:.6f}")
    print(f"  IRR (annual):  {(1 + irr_val)**12 - 1:.4f}")
    print(f"  NPV check:     {npv_check:.2e}")

    # Test 2: multi-group (two projects with different patterns)
    conn.execute("DROP TABLE test_irr")
    conn.execute("""
        CREATE TABLE test_irr AS
        SELECT * FROM (VALUES
            ('A', 1.0,  500.0, 5000.0, 0.0),
            ('A', 2.0,  500.0, 5000.0, 0.0),
            ('A', 3.0,  500.0, 5000.0, 0.0),
            ('A', 4.0,  500.0, 5000.0, 0.0),
            ('A', 5.0,  500.0, 5000.0, 0.0),
            ('A', 6.0,  500.0, 5000.0, 0.0),
            ('A', 7.0,  500.0, 5000.0, 0.0),
            ('A', 8.0,  500.0, 5000.0, 0.0),
            ('A', 9.0,  500.0, 5000.0, 0.0),
            ('A', 10.0, 500.0, 5000.0, 0.0),
            ('A', 11.0, 500.0, 5000.0, 0.0),
            ('A', 12.0, 500.0, 5000.0, 0.0),
            ('B', 1.0,  200.0, 1000.0, 0.0),
            ('B', 2.0,  200.0, 1000.0, 0.0),
            ('B', 3.0,  200.0, 1000.0, 0.0),
            ('B', 4.0,  200.0, 1000.0, 0.0),
            ('B', 5.0,  200.0, 1000.0, 0.0),
            ('B', 6.0,  200.0, 1000.0, 0.0)
        ) AS t(project, period, cashflow, investment, target_npv)
    """)

    rows = conn.execute("""
        SELECT project, irr(cashflow, period, investment, target_npv)
        FROM test_irr
        GROUP BY project
        ORDER BY project
    """).fetchall()

    print("\nTest 2: multi-group")
    for project, irr_val in rows:
        print(f"  Project {project}: IRR (monthly) = {irr_val:.6f}, IRR (annual) = {(1 + irr_val)**12 - 1:.4f}")

    # Verify each group's NPV
    for project, irr_val in rows:
        if project == "A":
            inv, cf, n = 5000.0, 500.0, 12
        else:
            inv, cf, n = 1000.0, 200.0, 6
        npv_check = -inv
        for t in range(1, n + 1):
            npv_check += cf / (1.0 + irr_val) ** t
        assert abs(npv_check) < 1e-6, f"Project {project} NPV check failed: {npv_check}"

    # Test 3: one cashflow of 13,000 at period 12, after an investment of 10,000. The exact IRR is
    # (13000 / 10000)^(1/12) - 1. A solver that discounted each row by its position, i + 1, instead of its period would
    # pass tests 1 and 2, whose periods are consecutive integers from 1, but fail this one.
    conn.execute("DROP TABLE test_irr")
    conn.execute("""
        CREATE TABLE test_irr AS
        SELECT
            12.0::DOUBLE AS period,
            13000.0 AS cashflow,
            10000.0 AS investment,
            0.0 AS target_npv
    """)
    result = conn.execute(
        "SELECT irr(cashflow, period, investment, target_npv) FROM test_irr"
    ).fetchone()
    irr_val = result[0]
    expected = (13000.0 / 10000.0) ** (1.0 / 12.0) - 1.0
    assert abs(irr_val - expected) < 1e-6, f"Sparse-period IRR mismatch: {irr_val} vs {expected}"
    print("\nTest 3: sparse periods (single cashflow at period 12)")
    print(f"  IRR (monthly): {irr_val:.6f} (expected {expected:.6f})")

    conn.execute("DROP TABLE test_irr")
    conn.close()

    stats_after = nrt.rtsys.get_allocation_stats()
    alloc_delta = stats_after.alloc - stats_before.alloc
    free_delta = stats_after.free - stats_before.free
    if alloc_delta != free_delta:
        print(f"  WARNING: NRT leak: alloc={alloc_delta}, free={free_delta}")
        sys.exit(1)

    print("\nAll checks passed.")
