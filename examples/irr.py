"""IRR (Internal Rate of Return) UDAF — DuckDB aggregate function example.

Demonstrates the full DuckDB aggregate lifecycle using numbduck:
  1. Define aggregate state as a numba structref (via numbox make_structref)
  2. Write @njit callbacks for init/update/combine/finalize/destroy
  3. Register the aggregate with DuckDB's C API
  4. Run a SQL query that invokes the UDAF

The IRR UDAF finds the monthly discount rate r such that:
    -investment - target_npv + sum(cashflows[i] / (1 + r) ^ periods[i]) = 0

SQL usage:
    SELECT irr(cashflow, period, investment, target_npv) FROM monthly_data;

NULL handling: rows where any of the four input columns is NULL are
skipped. If all rows are skipped, the result is NaN. If a group has data
but its IRR lies outside the solver's [-0.99, 10.0] monthly-rate bracket,
the result is +inf (see ``irr_bisect``), kept distinct from the
empty-group NaN so callers can tell "no root in range" from "no data".

Input contract: ``investment`` and ``target_npv`` are treated as
per-group constants. The aggregate captures the value from the first
non-NULL row of each group (update) or partial state (combine) and
ignores subsequent values; callers are expected to pass the same
investment / target_npv for every row of a given GROUP BY key.

Run via ``python examples/run_irr.py``. This module must be imported
(not executed as ``__main__``) so ``IRRStateType`` has a stable
``__module__`` across processes; otherwise numba's warm cache fails
type inference with ``No conversion from numba.IRRStateType(...) to
numba.IRRStateType(...)``.
"""
import math
import sys

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
from numbox.utils.lowlevel import _cast_int_to_void_p, get_unicode_data_p
from numbox.utils.meminfo import borrow_structref, export_meminfo, release_meminfo

from numbduck import ducklib
from numbduck.pybridge import extract_connection_ptr


Float64Vector, float64_vec_type = make_vector(nb_types.float64)


# ---- IRR state structref ----

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
# Monthly IRR is the rate r that zeroes the group's NPV. The solver assumes a
# single sign change of NPV(r) inside the fixed bracket [-0.99, 10.0] — true
# when the cashflow stream changes sign once (an up-front investment followed
# by positive cashflows), in which case NPV is monotonically decreasing in r.
# Convergence is on the width of the rate bracket, not on the residual NPV:
# the achievable NPV residual at the true root scales with the cashflow
# magnitude (|dNPV/dr| * ulp(r)), so a fixed absolute NPV tolerance cannot be
# met for large amounts even after r has resolved to machine precision. A
# rate-width test is scale-invariant and always terminates with the converged
# rate.

IRR_NO_BRACKET = math.inf


@njit
def _irr_npv(cashflows, periods, n, investment, target_npv, r):
    npv = -investment - target_npv
    for i in range(n):
        iu = numpy.uint64(i)
        npv += cashflows[iu] / (1.0 + r) ** periods[iu]
    return npv


@njit
def irr_bisect(cashflows, periods, n, investment, target_npv):
    """Solve for the monthly IRR by bisection on the rate bracket [-0.99, 10.0].

    Returns the converged rate (the final bracket midpoint) once the bracket
    width in r falls below ``rate_tol``; the midpoint is then within
    ``rate_tol / 2`` of the true root, regardless of cashflow magnitude.

    Assumes NPV(r) changes sign exactly once inside the bracket, which holds
    for a cashflow stream with a single sign change. When NPV has the same
    sign at both bracket ends the root lies outside [-0.99, 10.0] (a monthly
    return above 1000% or a near-total loss below -99%), or the stream lacks a
    single sign change; the solver then returns the ``IRR_NO_BRACKET`` sentinel
    (+inf). That sentinel is deliberately distinct from the NaN the finalize
    step emits for an empty group, so callers can distinguish "no root in the
    bracket" from "no data".
    """
    r_lo = -0.99
    r_hi = 10.0
    npv_lo = _irr_npv(cashflows, periods, n, investment, target_npv, r_lo)
    npv_hi = _irr_npv(cashflows, periods, n, investment, target_npv, r_hi)
    if npv_lo == 0.0:
        return r_lo
    if npv_hi == 0.0:
        return r_hi
    if (npv_lo > 0.0) == (npv_hi > 0.0):
        return IRR_NO_BRACKET
    rate_tol = 1e-12
    for _ in range(100):
        if (r_hi - r_lo) < rate_tol:
            break
        r_mid = (r_lo + r_hi) / 2.0
        npv = _irr_npv(cashflows, periods, n, investment, target_npv, r_mid)
        if npv > 0.0:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return (r_lo + r_hi) / 2.0


# ---- DuckDB aggregate callbacks ----
#
# DuckDB calls these in order: state_size -> init -> update (per chunk) ->
# combine (parallel merge) -> finalize -> destroy.
# Each receives raw pointers; we use the bridge intrinsics to
# reconstruct the structref from the state slot.
#
# Every impl body runs under a bare try/except. A Python exception escaping an
# @njit impl invoked from a @cfunc is swallowed at the C boundary: numba prints
# it, returns the zero/void default WITHOUT unwinding into DuckDB (a silent
# wrong result), and skips the scope-exit decref of any borrowed structref that
# is still live at the raise point (an NRT meminfo leak). Catching the exception
# in-frame runs that decref and lets each callback write a defined sentinel
# instead. The guard must be try/except, not try/finally -- try/finally re-raises
# on numba 0.65.1, which reintroduces both the swallow and the leak.

@njit
def _irr_state_size_impl(info):
    # A swallowed raise here would return the 0 default -- a 0-byte state slot
    # that corrupts every later callback -- so the guard returns the true size.
    try:
        return numpy.uint64(8)
    except Exception:
        return numpy.uint64(8)


@cfunc(nb_types.uint64(nb_types.intp))
def _irr_state_size_cb(info):
    return _irr_state_size_impl(info)


@njit
def _irr_init_impl(info, state):
    slot = carray(_cast_int_to_void_p(state), (1,), dtype=numpy.intp)
    # Sentinel on failure: a null state pointer, which destroy skips.
    try:
        cfs = Float64Vector(64)
        pds = Float64Vector(64)
        s = IRRState(cfs, pds, 0.0, 0.0, 0)
        slot[0] = export_meminfo(s)
    except Exception:
        slot[0] = 0


@cfunc(nb_types.void(nb_types.intp, nb_types.intp))
def _irr_init_cb(info, state):
    _irr_init_impl(info, state)


@njit
def _irr_update_impl(info, chunk, states):
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
        iu = numpy.uint64(i)
        slot = carray(_cast_int_to_void_p(state_slots[iu]), (1,), dtype=numpy.intp)
        # Borrow inside the guard: if vector_push raises (e.g. OOM growing the
        # buffer), catching in-frame runs the borrow's decref; without a
        # per-group error channel the offending row is dropped rather than
        # leaked, and finalize reflects the reduced data.
        try:
            s = borrow_structref(irr_state_type, slot[0])
            vector_push(s.cashflows, cf_data[iu])
            vector_push(s.periods, pd_data[iu])
            if s.initialized == 0:
                s.investment = inv_data[iu]
                s.target_npv = npv_data[iu]
                s.initialized = 1
        except Exception:
            pass


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _irr_update_cb(info, chunk, states):
    _irr_update_impl(info, chunk, states)


@njit
def _irr_combine_impl(info, source, target, count):
    src_slots = carray(_cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    tgt_slots = carray(_cast_int_to_void_p(target), (count,), dtype=numpy.intp)
    for i in range(count):
        iu = numpy.uint64(i)
        # Both borrows live inside the guard: a raise in vector_extend releases
        # source and target in-frame instead of leaking their meminfos.
        try:
            src_slot = carray(_cast_int_to_void_p(src_slots[iu]), (1,), dtype=numpy.intp)
            tgt_slot = carray(_cast_int_to_void_p(tgt_slots[iu]), (1,), dtype=numpy.intp)
            src = borrow_structref(irr_state_type, src_slot[0])
            tgt = borrow_structref(irr_state_type, tgt_slot[0])
            vector_extend(tgt.cashflows, src.cashflows)
            vector_extend(tgt.periods, src.periods)
            if tgt.initialized == 0:
                tgt.investment = src.investment
                tgt.target_npv = src.target_npv
                tgt.initialized = src.initialized
        except Exception:
            pass


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp, nb_types.uint64))
def _irr_combine_cb(info, source, target, count):
    _irr_combine_impl(info, source, target, count)


@njit
def _irr_finalize_impl(info, source, result, count, offset):
    out_data = ducklib.duckdb_vector_get_data(result)
    src_slots = carray(_cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    out_vals = carray(_cast_int_to_void_p(out_data), (offset + count,), dtype=numpy.float64)
    for i in range(count):
        iu = numpy.uint64(i)
        # Sentinel on failure: NaN, matching the empty-group output below. The
        # borrow lives inside the guard so a raise in irr_bisect releases it.
        try:
            src_slot = carray(_cast_int_to_void_p(src_slots[iu]), (1,), dtype=numpy.intp)
            s = borrow_structref(irr_state_type, src_slot[0])
            n = len(s.cashflows)
            if n == 0:
                out_vals[numpy.uint64(offset + iu)] = math.nan
            else:
                out_vals[numpy.uint64(offset + iu)] = irr_bisect(
                    s.cashflows, s.periods, n, s.investment, s.target_npv)
        except Exception:
            out_vals[numpy.uint64(offset + iu)] = math.nan


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp, nb_types.uint64, nb_types.uint64))
def _irr_finalize_cb(info, source, result, count, offset):
    _irr_finalize_impl(info, source, result, count, offset)


@njit
def _irr_destroy_impl(states, count):
    state_slots = carray(_cast_int_to_void_p(states), (count,), dtype=numpy.intp)
    for i in range(count):
        iu = numpy.uint64(i)
        slot = carray(_cast_int_to_void_p(state_slots[iu]), (1,), dtype=numpy.intp)
        # Skip the null sentinel a failed init leaves behind, and never let a
        # release raise back across the @cfunc boundary.
        try:
            if slot[0] != 0:
                release_meminfo(slot[0])
        except Exception:
            pass


@cfunc(nb_types.void(nb_types.intp, nb_types.uint64))
def _irr_destroy_cb(states, count):
    _irr_destroy_impl(states, count)


# ---- Registration and query ----

def register_irr(conn):
    conn_p = extract_connection_ptr(conn)

    func_p = ducklib.duckdb_create_aggregate_function()
    name_p = get_unicode_data_p("irr")
    ducklib.duckdb_aggregate_function_set_name(func_p, name_p)

    dbl_type_p = ducklib.duckdb_create_logical_type(ducklib.DUCKDB_TYPE_DOUBLE)
    for _ in range(4):
        ducklib.duckdb_aggregate_function_add_parameter(func_p, dbl_type_p)
    ducklib.duckdb_aggregate_function_set_return_type(func_p, dbl_type_p)
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
    print("IRR UDAF example")
    print("=" * 40)

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

    # Verify: at the found rate, NPV should be ~0
    npv_check = -10000.0
    for t in range(1, 13):
        npv_check += 1000.0 / (1.0 + irr_val) ** t
    assert abs(npv_check) < 1e-6, f"NPV check failed: {npv_check}"

    print("\nTest 1: uniform cashflows")
    print("  Investment: 10,000 | Cashflows: 12 x 1,000 | Target NPV: 0")
    print(f"  IRR (monthly): {irr_val:.6f}")
    print(f"  IRR (annual):  {(1 + irr_val)**12 - 1:.4f}")
    print(f"  NPV check:     {npv_check:.2e}")

    # Test 2: multi-group — two projects with different patterns
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

    # Test 3: sparse periods — catches the "exponent = i+1" trap
    #   One cashflow at period 12, investment 10,000
    #   Analytic IRR: (13000/10000)^(1/12) - 1
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
