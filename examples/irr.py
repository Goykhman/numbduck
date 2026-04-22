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
skipped. If all rows are skipped, the result is NaN.

Input contract: ``investment`` and ``target_npv`` are treated as
per-group constants. The aggregate captures the value from the first
non-NULL row of each group (update) or partial state (combine) and
ignores subsequent values; callers are expected to pass the same
investment / target_npv for every row of a given GROUP BY key.
"""
import math
import sys

import duckdb
import numpy
from numba import cfunc, carray, njit
from numba import types as nb_types
from numba.core import cgutils
from numba.experimental import structref
from numba.extending import intrinsic
import llvmlite.ir as llir
from numbox.utils.highlevel import make_structref
from numbox.utils.lowlevel import _cast_int_to_void_p, get_unicode_data_p
from numbox.utils.meminfo import structref_meminfo

from numbduck import ducklib
from numbduck.pybridge import extract_connection_ptr
from numbduck.vector import make_vector, vector_push, vector_extend

Float64Vector, float64_vec_type = make_vector(nb_types.float64)


# ---- structref <-> raw MemInfo pointer bridge intrinsics ----
#
# Full lifecycle explanation, removerefctpass interaction, and rationale
# for using NRT_MemInfo_release (instead of context.nrt.decref) live in
# test/test_ducklib.md. Reference impl: test/test_ducklib.py.
#
# Reference counting contract:
#   export_meminfo(s):   +1 incref; returned intp keeps the allocation alive.
#   borrow_structref():  +1 incref on entry; local decref on scope exit
#                        balances it. Net zero for the external owner.
#   release_meminfo(p):  -1 decref; triggers dtor at refcount 0.

_MI_TY = nb_types.MemInfoPointer(nb_types.voidptr)


@intrinsic
def _incref_meminfo(typingctx, p_ty):
    sig = nb_types.void(p_ty)

    def codegen(context, builder, signature, args):
        mi_ll_ty = context.get_value_type(_MI_TY)
        meminfo = builder.inttoptr(args[0], mi_ll_ty)
        context.nrt.incref(builder, _MI_TY, meminfo)
    return sig, codegen


@njit
def export_meminfo(s):
    meminfo_p, _ = structref_meminfo(s)
    _incref_meminfo(meminfo_p)
    return meminfo_p


@intrinsic
def _deref_structref_raw_ptr(typingctx, struct_type_ref, p_ty):
    inst_type = struct_type_ref.instance_type
    sig = inst_type(struct_type_ref, p_ty)

    def codegen(context, builder, signature, args):
        p_val = args[1]
        mi_ll_ty = context.get_value_type(_MI_TY)
        meminfo = builder.inttoptr(p_val, mi_ll_ty)
        st = cgutils.create_struct_proxy(inst_type)(context, builder)
        st.meminfo = meminfo
        return st._getvalue()
    return sig, codegen


@njit
def borrow_structref(struct_type, p):
    _incref_meminfo(p)
    return _deref_structref_raw_ptr(struct_type, p)


@intrinsic
def _release_meminfo(typingctx, p_ty):
    """Decref MemInfo at intp via NRT_MemInfo_release (C runtime).

    Can't use context.nrt.decref() here — removerefctpass strips
    NRT_decref when the function signature has no NRT-tracked types.
    NRT_MemInfo_release also makes _legalize() bail out, protecting
    the whole function from the rewrite.
    """
    sig = nb_types.void(p_ty)

    def codegen(context, builder, signature, args):
        ptr_ty = llir.IntType(8).as_pointer()
        fnty = llir.FunctionType(llir.VoidType(), [ptr_ty])
        fn = cgutils.get_or_insert_function(builder.module, fnty, "NRT_MemInfo_release")
        meminfo = builder.inttoptr(args[0], ptr_ty)
        builder.call(fn, [meminfo])
    return sig, codegen


@njit
def release_meminfo(p):
    _release_meminfo(p)


# ---- IRR state structref ----

@structref.register
class IRRStateType(nb_types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((n, nb_types.unliteral(t)) for n, t in fields)


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

@njit
def irr_bisect(cashflows, periods, n, investment, target_npv):
    r_lo = -0.99
    r_hi = 10.0
    for _ in range(100):
        r_mid = (r_lo + r_hi) / 2.0
        npv = -investment - target_npv
        for i in range(n):
            iu = numpy.uint64(i)
            npv += cashflows[iu] / (1.0 + r_mid) ** periods[iu]
        if abs(npv) < 1e-9:
            return r_mid
        if npv > 0.0:
            r_lo = r_mid
        else:
            r_hi = r_mid
    return math.nan


# ---- DuckDB aggregate callbacks ----
#
# DuckDB calls these in order: state_size -> init -> update (per chunk) ->
# combine (parallel merge) -> finalize -> destroy.
# Each receives raw pointers; we use the bridge intrinsics to
# reconstruct the structref from the state slot.

@njit
def _irr_state_size_impl(info):
    return numpy.uint64(8)


@cfunc(nb_types.uint64(nb_types.intp))
def _irr_state_size_cb(info):
    return _irr_state_size_impl(info)


@njit
def _irr_init_impl(info, state):
    cfs = Float64Vector(64)
    pds = Float64Vector(64)
    s = IRRState(cfs, pds, 0.0, 0.0, 0)
    p = export_meminfo(s)
    slot = carray(_cast_int_to_void_p(state), (1,), dtype=numpy.intp)
    slot[0] = p


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
    val_cf = numpy.intp(ducklib.duckdb_vector_get_validity(vec_cf))
    val_pd = numpy.intp(ducklib.duckdb_vector_get_validity(vec_pd))
    val_inv = numpy.intp(ducklib.duckdb_vector_get_validity(vec_inv))
    val_npv = numpy.intp(ducklib.duckdb_vector_get_validity(vec_npv))
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
        s = borrow_structref(irr_state_type, slot[0])
        vector_push(s.cashflows, cf_data[iu])
        vector_push(s.periods, pd_data[iu])
        if s.initialized == 0:
            s.investment = inv_data[iu]
            s.target_npv = npv_data[iu]
            s.initialized = 1


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp))
def _irr_update_cb(info, chunk, states):
    _irr_update_impl(info, chunk, states)


@njit
def _irr_combine_impl(info, source, target, count):
    src_slots = carray(_cast_int_to_void_p(source), (count,), dtype=numpy.intp)
    tgt_slots = carray(_cast_int_to_void_p(target), (count,), dtype=numpy.intp)
    for i in range(count):
        iu = numpy.uint64(i)
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
        src_slot = carray(_cast_int_to_void_p(src_slots[iu]), (1,), dtype=numpy.intp)
        s = borrow_structref(irr_state_type, src_slot[0])
        n = len(s.cashflows)
        if n == 0:
            out_vals[numpy.uint64(offset + iu)] = math.nan
            continue
        out_vals[numpy.uint64(offset + iu)] = irr_bisect(s.cashflows, s.periods, n, s.investment, s.target_npv)


@cfunc(nb_types.void(nb_types.intp, nb_types.intp, nb_types.intp, nb_types.uint64, nb_types.uint64))
def _irr_finalize_cb(info, source, result, count, offset):
    _irr_finalize_impl(info, source, result, count, offset)


@njit
def _irr_destroy_impl(states, count):
    state_slots = carray(_cast_int_to_void_p(states), (count,), dtype=numpy.intp)
    for i in range(count):
        iu = numpy.uint64(i)
        slot = carray(_cast_int_to_void_p(state_slots[iu]), (1,), dtype=numpy.intp)
        release_meminfo(slot[0])


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


if __name__ == "__main__":
    main()
