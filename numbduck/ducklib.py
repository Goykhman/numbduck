from llvmlite import ir
from llvmlite.ir import IRBuilder, FunctionType
from numba.core.types import float32, float64, int8, int16, int32, int64, intp, Tuple, uint8, uint16, uint32, uint64, UniTuple, void
from numba.extending import intrinsic
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numba.core.cgutils import get_or_insert_function
from numbox.utils.highlevel import cres

import platform
import sys

from numbduck.utils import load_duckdb


duckdb_lib = load_duckdb()

_is_win = sys.platform == 'win32'
_is_sysv_x86_64 = platform.machine() == 'x86_64'

duckdb_state_ty = int32

DuckDBSuccess = 0
DuckDBError = 1

duckdb_result_ty = UniTuple(intp, 6)
duckdb_hugeint_ty = Tuple((uint64, int64))
duckdb_uhugeint_ty = UniTuple(uint64, 2)
duckdb_interval_ty = Tuple((int32, int32, int64))
duckdb_decimal_ty = Tuple((uint8, uint8, uint64, int64))
duckdb_blob_ty = Tuple((intp, uint64))
duckdb_bit_ty = Tuple((intp, uint64))
duckdb_varint_ty = Tuple((intp, uint64, int8))

_i64 = ir.IntType(64)
_i32 = ir.IntType(32)


def _resolve_sig(func_name):
    """Look up a function's signature in the signatures dict."""
    func_sig = signatures.get(func_name, None)
    if func_sig is None:
        raise ValueError(f"Undefined signature for {func_name}")
    return func_sig


def _build_packed_interval(builder, interval_tup):
    """Pack (months:i32, days:i32, micros:i64) into {i64, i64}.

    C struct: { int32 months, int32 days, int64 micros } = 16 bytes.
    Pack two i32 fields into one i64 to match the C struct layout.
    Using {i32, i32, i64} directly fails — LLVM's SysV x86-64 ABI
    lowering drops the second i32 field when coercing to registers.
    See: https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    """
    interval_struct = ir.LiteralStructType([_i64, _i64])
    months = builder.extract_value(interval_tup, 0)
    days = builder.extract_value(interval_tup, 1)
    micros = builder.extract_value(interval_tup, 2)
    months_zext = builder.zext(months, _i64)
    days_zext = builder.zext(days, _i64)
    days_shifted = builder.shl(days_zext, ir.Constant(_i64, 32))
    packed = builder.or_(months_zext, days_shifted)
    val = ir.Constant(interval_struct, ir.Undefined)
    val = builder.insert_value(val, packed, 0)
    val = builder.insert_value(val, micros, 1)
    return val


def _emit_byval_call(builder, context, arg, arg_ll_ty, ret_type, func_name):
    """Emit IR to pass a struct by pointer: alloca, store, call via pointer."""
    stack_p = builder.alloca(arg_ll_ty)
    builder.store(arg, stack_p)
    func_ty_ll = FunctionType(ret_type, [arg_ll_ty.as_pointer()])
    func_p = get_or_insert_function(builder.module, func_ty_ll, func_name)
    return builder.call(func_p, [stack_p])


@intrinsic(prefer_literal=True)
def _call_lib_func_byval(typingctx, func_name_ty, arg_ty):
    """Like _call_lib_func, but allocates the arg on the stack
    and passes a pointer to that stack slot.

    Used for C functions whose parameter is a pointer to a struct
    (e.g. ``duckdb_result *``), when the caller has the struct as
    a value.
    """
    func_name = func_name_ty.literal_value
    func_sig = _resolve_sig(func_name)

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        arg_ll_ty = context.get_value_type(arg_ty)
        ret_type = context.get_value_type(signature.return_type)
        return _emit_byval_call(
            builder, context, arg, arg_ll_ty, ret_type, func_name)

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen


@intrinsic(prefer_literal=True)
def _call_lib_func_struct_in(typingctx, func_name_ty, arg_ty):
    """Like _call_lib_func, but the arg is a small struct.

    Struct must be ≤16 bytes for System V x86-64 by-value passing.
    On Windows: passes via stack pointer (degrades to byval).
    On other platforms: passes the struct directly by value.

    LLVM's JIT treats ABI lowering as a frontend responsibility — it
    won't insert the right calling convention for struct args/returns.
    See: https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
         https://github.com/llvm/llvm-project/issues/85417
    """
    func_name = func_name_ty.literal_value
    func_sig = _resolve_sig(func_name)
    struct_bytes = sum(t.bitwidth for t in arg_ty) / 8
    assert struct_bytes <= 16, (
        f"struct too large for by-value passing ({struct_bytes} bytes > 16)"
    )

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        arg_ll_ty = context.get_value_type(arg_ty)
        ret_type = context.get_value_type(signature.return_type)
        if _is_win:
            return _emit_byval_call(
                builder, context, arg, arg_ll_ty, ret_type, func_name)
        func_ty_ll = FunctionType(ret_type, [arg_ll_ty])
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, func_name)
        return builder.call(func_p, [arg])

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen


@intrinsic(prefer_literal=True)
def _call_lib_func_struct_out(typingctx, func_name_ty, arg_ty):
    """Like _call_lib_func, but the return value is a struct.

    Return struct must be ≤16 bytes for System V x86-64 by-value return.
    On Windows: uses sret (hidden first pointer arg, void return).
    On other platforms: returns the struct directly by value.

    See _call_lib_func_struct_in docstring for ABI references.
    """
    func_name = func_name_ty.literal_value
    func_sig = _resolve_sig(func_name)
    ret_ty = func_sig.return_type
    struct_bytes = sum(t.bitwidth for t in ret_ty) / 8
    assert struct_bytes <= 16, (
        f"return struct too large for by-value return ({struct_bytes} bytes > 16)"
    )

    def codegen(context, builder, signature, arguments):
        _, arg = arguments
        ret_ll_ty = context.get_value_type(signature.return_type)
        if _is_win:
            sret_p = builder.alloca(ret_ll_ty)
            func_ty_ll = FunctionType(
                ir.VoidType(),
                [ret_ll_ty.as_pointer(), arg.type]
            )
            func_p = get_or_insert_function(
                builder.module, func_ty_ll, func_name)
            func_p.args[0].add_attribute('sret')
            builder.call(func_p, [sret_p, arg])
            return builder.load(sret_p)
        func_ty_ll = FunctionType(ret_ll_ty, [arg.type])
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, func_name)
        return builder.call(func_p, [arg])

    sig = func_sig.return_type(func_name_ty, arg_ty)
    return sig, codegen


signatures["duckdb_bind_blob"] = duckdb_state_ty(intp, uint64, intp, uint64)
signatures["duckdb_bind_boolean"] = duckdb_state_ty(intp, uint64, int8)
signatures["duckdb_bind_date"] = duckdb_state_ty(intp, uint64, int32)
signatures["duckdb_bind_double"] = duckdb_state_ty(intp, uint64, float64)
signatures["duckdb_bind_float"] = duckdb_state_ty(intp, uint64, float32)
signatures["duckdb_bind_int8"] = duckdb_state_ty(intp, uint64, int8)
signatures["duckdb_bind_int16"] = duckdb_state_ty(intp, uint64, int16)
signatures["duckdb_bind_int32"] = duckdb_state_ty(intp, uint64, int32)
signatures["duckdb_bind_int64"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_null"] = duckdb_state_ty(intp, uint64)
signatures["duckdb_bind_parameter_index"] = duckdb_state_ty(intp, intp, intp)
signatures["duckdb_bind_time"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_timestamp"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_timestamp_tz"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_uint8"] = duckdb_state_ty(intp, uint64, uint8)
signatures["duckdb_bind_uint16"] = duckdb_state_ty(intp, uint64, uint16)
signatures["duckdb_bind_uint32"] = duckdb_state_ty(intp, uint64, uint32)
signatures["duckdb_bind_uint64"] = duckdb_state_ty(intp, uint64, uint64)
signatures["duckdb_bind_value"] = duckdb_state_ty(intp, uint64, intp)
signatures["duckdb_bind_varchar"] = duckdb_state_ty(intp, uint64, intp)
signatures["duckdb_bind_varchar_length"] = duckdb_state_ty(intp, uint64, intp, uint64)
signatures["duckdb_close"] = void(intp)
signatures["duckdb_column_count"] = intp(intp)
signatures["duckdb_column_logical_type"] = intp(intp, uint64)
signatures["duckdb_column_name"] = intp(intp, uint64)
signatures["duckdb_column_type"] = int32(intp, uint64)
signatures["duckdb_create_bit"] = intp(duckdb_bit_ty)
signatures["duckdb_create_blob"] = intp(intp, uint64)
signatures["duckdb_create_date"] = intp(int32)
signatures["duckdb_create_time"] = intp(int64)
signatures["duckdb_create_time_tz"] = uint64(int64, int32)
signatures["duckdb_create_time_tz_value"] = intp(uint64)
signatures["duckdb_create_timestamp"] = intp(int64)
signatures["duckdb_create_timestamp_ms"] = intp(int64)
signatures["duckdb_create_timestamp_ns"] = intp(int64)
signatures["duckdb_create_timestamp_s"] = intp(int64)
signatures["duckdb_create_timestamp_tz"] = intp(int64)
signatures["duckdb_connect"] = duckdb_state_ty(intp, intp)
signatures["duckdb_create_array_type"] = intp(intp, uint64)
signatures["duckdb_create_array_value"] = intp(intp, intp, uint64)
signatures["duckdb_create_bool"] = intp(int8)
signatures["duckdb_create_decimal_type"] = intp(uint8, uint8)
signatures["duckdb_create_double"] = intp(float64)
signatures["duckdb_create_enum_type"] = intp(intp, uint64)
signatures["duckdb_create_enum_value"] = intp(intp, uint64)
signatures["duckdb_create_hugeint"] = intp(duckdb_hugeint_ty)
signatures["duckdb_create_float"] = intp(float32)
signatures["duckdb_create_int8"] = intp(int8)
signatures["duckdb_create_int16"] = intp(int16)
signatures["duckdb_create_int32"] = intp(int32)
signatures["duckdb_create_int64"] = intp(int64)
signatures["duckdb_create_list_type"] = intp(intp)
signatures["duckdb_create_list_value"] = intp(intp, intp, uint64)
signatures["duckdb_create_logical_type"] = intp(int32)
signatures["duckdb_create_map_type"] = intp(intp, intp)
signatures["duckdb_create_map_value"] = intp(intp, intp, intp, uint64)
signatures["duckdb_create_null_value"] = intp()
signatures["duckdb_create_struct_type"] = intp(intp, intp, uint64)
signatures["duckdb_create_struct_value"] = intp(intp, intp)
signatures["duckdb_create_uhugeint"] = intp(duckdb_uhugeint_ty)
signatures["duckdb_create_uint8"] = intp(uint8)
signatures["duckdb_create_uint16"] = intp(uint16)
signatures["duckdb_create_uint32"] = intp(uint32)
signatures["duckdb_create_uint64"] = intp(uint64)
signatures["duckdb_create_union_type"] = intp(intp, intp, uint64)
signatures["duckdb_create_union_value"] = intp(intp, uint64, intp)
signatures["duckdb_create_uuid"] = intp(duckdb_uhugeint_ty)
signatures["duckdb_create_varchar"] = intp(intp)
signatures["duckdb_create_varchar_length"] = intp(intp, uint64)
signatures["duckdb_data_chunk_get_column_count"] = intp(intp)
signatures["duckdb_data_chunk_get_size"] = intp(intp)
signatures["duckdb_data_chunk_get_vector"] = intp(intp, intp)
signatures["duckdb_destroy_data_chunk"] = void(intp)
signatures["duckdb_destroy_logical_type"] = void(intp)
signatures["duckdb_destroy_prepare"] = void(intp)
signatures["duckdb_destroy_result"] = void(intp)
signatures["duckdb_destroy_value"] = void(intp)
signatures["duckdb_disconnect"] = void(intp)
signatures["duckdb_execute_prepared"] = duckdb_state_ty(intp, intp)
signatures["duckdb_fetch_chunk"] = intp(duckdb_result_ty)
signatures["duckdb_free"] = void(intp)
signatures["duckdb_get_bit"] = duckdb_bit_ty(intp)
signatures["duckdb_get_blob"] = duckdb_blob_ty(intp)
signatures["duckdb_get_bool"] = int8(intp)
signatures["duckdb_get_date"] = int32(intp)
signatures["duckdb_get_double"] = float64(intp)
signatures["duckdb_get_enum_value"] = uint64(intp)
signatures["duckdb_get_float"] = float32(intp)
signatures["duckdb_get_hugeint"] = duckdb_hugeint_ty(intp)
signatures["duckdb_get_int8"] = int8(intp)
signatures["duckdb_get_int16"] = int16(intp)
signatures["duckdb_get_int32"] = int32(intp)
signatures["duckdb_get_int64"] = int64(intp)
signatures["duckdb_get_list_child"] = intp(intp, uint64)
signatures["duckdb_get_list_size"] = uint64(intp)
signatures["duckdb_get_map_key"] = intp(intp, uint64)
signatures["duckdb_get_map_size"] = uint64(intp)
signatures["duckdb_get_map_value"] = intp(intp, uint64)
signatures["duckdb_get_struct_child"] = intp(intp, uint64)
signatures["duckdb_get_time"] = int64(intp)
signatures["duckdb_get_time_tz"] = uint64(intp)
signatures["duckdb_get_timestamp"] = int64(intp)
signatures["duckdb_get_timestamp_ms"] = int64(intp)
signatures["duckdb_get_timestamp_ns"] = int64(intp)
signatures["duckdb_get_timestamp_s"] = int64(intp)
signatures["duckdb_get_timestamp_tz"] = int64(intp)
signatures["duckdb_get_uhugeint"] = duckdb_uhugeint_ty(intp)
signatures["duckdb_get_uint8"] = uint8(intp)
signatures["duckdb_get_uint16"] = uint16(intp)
signatures["duckdb_get_uint32"] = uint32(intp)
signatures["duckdb_get_uint64"] = uint64(intp)
signatures["duckdb_get_uuid"] = duckdb_uhugeint_ty(intp)
signatures["duckdb_get_value_type"] = intp(intp)
signatures["duckdb_get_varchar"] = intp(intp)
signatures["duckdb_is_null_value"] = int8(intp)
signatures["duckdb_nparams"] = uint64(intp)
signatures["duckdb_open"] = duckdb_state_ty(intp, intp)
signatures["duckdb_prepare"] = duckdb_state_ty(intp, intp, intp)
signatures["duckdb_prepare_error"] = intp(intp)
signatures["duckdb_query"] = duckdb_state_ty(intp, intp, intp)
signatures["duckdb_result_error"] = intp(intp)
signatures["duckdb_result_error_type"] = int32(intp)
signatures["duckdb_result_return_type"] = int32(duckdb_result_ty)
signatures["duckdb_result_statement_type"] = int32(duckdb_result_ty)
signatures["duckdb_row_count"] = intp(intp)
signatures["duckdb_rows_changed"] = uint64(intp)
signatures["duckdb_validity_row_is_valid"] = int8(intp, intp)
signatures["duckdb_value_to_string"] = intp(intp)
signatures["duckdb_vector_get_data"] = intp(intp)
signatures["duckdb_vector_get_validity"] = uint64(intp)


@cres(signatures.get("duckdb_bind_boolean"))
def duckdb_bind_boolean(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_boolean """
    return _call_lib_func("duckdb_bind_boolean", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_blob"))
def duckdb_bind_blob(prepared_statement_p, param_idx, data_p, length):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_blob """
    return _call_lib_func("duckdb_bind_blob", (prepared_statement_p, param_idx, data_p, length))


@cres(signatures.get("duckdb_bind_date"))
def duckdb_bind_date(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_date """
    return _call_lib_func("duckdb_bind_date", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_int8"))
def duckdb_bind_int8(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_int8 """
    return _call_lib_func("duckdb_bind_int8", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_int16"))
def duckdb_bind_int16(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_int16 """
    return _call_lib_func("duckdb_bind_int16", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_double"))
def duckdb_bind_double(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_double """
    return _call_lib_func("duckdb_bind_double", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_float"))
def duckdb_bind_float(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_float """
    return _call_lib_func("duckdb_bind_float", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_int32"))
def duckdb_bind_int32(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_int32 """
    return _call_lib_func("duckdb_bind_int32", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_int64"))
def duckdb_bind_int64(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_int64 """
    return _call_lib_func("duckdb_bind_int64", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_parameter_index"))
def duckdb_bind_parameter_index(prepared_statement_p, param_idx_out_p, name_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_parameter_index """
    return _call_lib_func("duckdb_bind_parameter_index", (prepared_statement_p, param_idx_out_p, name_p))


@cres(signatures.get("duckdb_bind_null"))
def duckdb_bind_null(prepared_statement_p, param_idx):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_null """
    return _call_lib_func("duckdb_bind_null", (prepared_statement_p, param_idx))


@cres(signatures.get("duckdb_bind_time"))
def duckdb_bind_time(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_time """
    return _call_lib_func("duckdb_bind_time", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_timestamp"))
def duckdb_bind_timestamp(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_timestamp """
    return _call_lib_func("duckdb_bind_timestamp", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_timestamp_tz"))
def duckdb_bind_timestamp_tz(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_timestamp_tz """
    return _call_lib_func("duckdb_bind_timestamp_tz", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_uint8"))
def duckdb_bind_uint8(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uint8 """
    return _call_lib_func("duckdb_bind_uint8", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_uint16"))
def duckdb_bind_uint16(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uint16 """
    return _call_lib_func("duckdb_bind_uint16", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_uint32"))
def duckdb_bind_uint32(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uint32 """
    return _call_lib_func("duckdb_bind_uint32", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_uint64"))
def duckdb_bind_uint64(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uint64 """
    return _call_lib_func("duckdb_bind_uint64", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_value"))
def duckdb_bind_value(prepared_statement_p, param_idx, val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_value """
    return _call_lib_func("duckdb_bind_value", (prepared_statement_p, param_idx, val_p))


@cres(signatures.get("duckdb_bind_varchar"))
def duckdb_bind_varchar(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_varchar """
    return _call_lib_func("duckdb_bind_varchar", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_varchar_length"))
def duckdb_bind_varchar_length(prepared_statement_p, param_idx, val_p, length):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_varchar_length """
    return _call_lib_func("duckdb_bind_varchar_length", (prepared_statement_p, param_idx, val_p, length))


@cres(signatures.get("duckdb_close"))
def duckdb_close(duckdb_database_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_close """
    return _call_lib_func("duckdb_close", (duckdb_database_pp,))


@cres(signatures.get("duckdb_column_count"))
def duckdb_column_count(duckdb_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_column_count """
    return _call_lib_func("duckdb_column_count", (duckdb_result_p,))


@cres(signatures.get("duckdb_column_logical_type"))
def duckdb_column_logical_type(duckdb_result_p, col):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_column_logical_type """
    return _call_lib_func("duckdb_column_logical_type", (duckdb_result_p, col))


@cres(signatures.get("duckdb_column_name"))
def duckdb_column_name(duckdb_result_p, col):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_column_name """
    return _call_lib_func("duckdb_column_name", (duckdb_result_p, col))


@cres(signatures.get("duckdb_column_type"))
def duckdb_column_type(duckdb_result_p, col):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_column_type """
    return _call_lib_func("duckdb_column_type", (duckdb_result_p, col))


@cres(signatures.get("duckdb_create_array_type"))
def duckdb_create_array_type(type_p, array_size):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_array_type """
    return _call_lib_func("duckdb_create_array_type", (type_p, array_size))


@cres(signatures.get("duckdb_create_array_value"))
def duckdb_create_array_value(type_p, values_p, value_count):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_array_value """
    return _call_lib_func("duckdb_create_array_value", (type_p, values_p, value_count))


@cres(signatures.get("duckdb_create_bool"))
def duckdb_create_bool(input):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_bool """
    return _call_lib_func("duckdb_create_bool", (input,))


@cres(signatures.get("duckdb_create_decimal_type"))
def duckdb_create_decimal_type(width, scale):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_decimal_type """
    return _call_lib_func("duckdb_create_decimal_type", (width, scale))


@cres(signatures.get("duckdb_create_double"))
def duckdb_create_double(input):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_double """
    return _call_lib_func("duckdb_create_double", (input,))


@cres(signatures.get("duckdb_create_enum_type"))
def duckdb_create_enum_type(member_names_p, member_count):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_enum_type """
    return _call_lib_func("duckdb_create_enum_type", (member_names_p, member_count))


@cres(signatures.get("duckdb_create_enum_value"))
def duckdb_create_enum_value(type_p, value):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_enum_value """
    return _call_lib_func("duckdb_create_enum_value", (type_p, value))


@cres(signatures.get("duckdb_create_float"))
def duckdb_create_float(input):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_float """
    return _call_lib_func("duckdb_create_float", (input,))


@cres(signatures.get("duckdb_create_int8"))
def duckdb_create_int8(input):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_int8 """
    return _call_lib_func("duckdb_create_int8", (input,))


@cres(signatures.get("duckdb_create_int16"))
def duckdb_create_int16(input):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_int16 """
    return _call_lib_func("duckdb_create_int16", (input,))


@cres(signatures.get("duckdb_create_int32"))
def duckdb_create_int32(input):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_int32 """
    return _call_lib_func("duckdb_create_int32", (input,))


@cres(signatures.get("duckdb_create_int64"))
def duckdb_create_int64(input):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_int64 """
    return _call_lib_func("duckdb_create_int64", (input,))


@cres(signatures.get("duckdb_create_list_type"))
def duckdb_create_list_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_list_type """
    return _call_lib_func("duckdb_create_list_type", (type_p,))


@cres(signatures.get("duckdb_create_list_value"))
def duckdb_create_list_value(type_p, values_p, value_count):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_list_value """
    return _call_lib_func("duckdb_create_list_value", (type_p, values_p, value_count))


@cres(signatures.get("duckdb_create_logical_type"))
def duckdb_create_logical_type(type_id):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_logical_type """
    return _call_lib_func("duckdb_create_logical_type", (type_id,))


@cres(signatures.get("duckdb_create_map_type"))
def duckdb_create_map_type(key_type_p, value_type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_map_type """
    return _call_lib_func("duckdb_create_map_type", (key_type_p, value_type_p))


@cres(signatures.get("duckdb_create_map_value"))
def duckdb_create_map_value(map_type_p, keys_p, values_p, entry_count):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_map_value """
    return _call_lib_func("duckdb_create_map_value", (map_type_p, keys_p, values_p, entry_count))


@cres(signatures.get("duckdb_create_null_value"))
def duckdb_create_null_value():
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_null_value """
    return _call_lib_func("duckdb_create_null_value", ())


@cres(signatures.get("duckdb_create_struct_type"))
def duckdb_create_struct_type(member_types_p, member_names_p, member_count):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_struct_type """
    return _call_lib_func("duckdb_create_struct_type", (member_types_p, member_names_p, member_count))


@cres(signatures.get("duckdb_create_struct_value"))
def duckdb_create_struct_value(type_p, values_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_struct_value """
    return _call_lib_func("duckdb_create_struct_value", (type_p, values_p))


@cres(signatures.get("duckdb_create_uint8"))
def duckdb_create_uint8(input):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_uint8 """
    return _call_lib_func("duckdb_create_uint8", (input,))


@cres(signatures.get("duckdb_create_uint16"))
def duckdb_create_uint16(input):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_uint16 """
    return _call_lib_func("duckdb_create_uint16", (input,))


@cres(signatures.get("duckdb_create_uint32"))
def duckdb_create_uint32(input):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_uint32 """
    return _call_lib_func("duckdb_create_uint32", (input,))


@cres(signatures.get("duckdb_create_uint64"))
def duckdb_create_uint64(input):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_uint64 """
    return _call_lib_func("duckdb_create_uint64", (input,))


@cres(signatures.get("duckdb_create_union_type"))
def duckdb_create_union_type(member_types_p, member_names_p, member_count):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_union_type """
    return _call_lib_func("duckdb_create_union_type", (member_types_p, member_names_p, member_count))


@cres(signatures.get("duckdb_create_union_value"))
def duckdb_create_union_value(union_type_p, tag_index, value_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_union_value """
    return _call_lib_func("duckdb_create_union_value", (union_type_p, tag_index, value_p))


@cres(signatures.get("duckdb_create_varchar"))
def duckdb_create_varchar(text_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_varchar """
    return _call_lib_func("duckdb_create_varchar", (text_p,))


@cres(signatures.get("duckdb_create_varchar_length"))
def duckdb_create_varchar_length(text_p, length):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_varchar_length """
    return _call_lib_func("duckdb_create_varchar_length", (text_p, length))


@cres(signatures.get("duckdb_data_chunk_get_column_count"))
def duckdb_data_chunk_get_column_count(data_chunk_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_data_chunk_get_column_count """
    return _call_lib_func("duckdb_data_chunk_get_column_count", (data_chunk_p,))


@cres(signatures.get("duckdb_data_chunk_get_size"))
def duckdb_data_chunk_get_size(data_chunk_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_data_chunk_get_size """
    return _call_lib_func("duckdb_data_chunk_get_size", (data_chunk_p,))


@cres(signatures.get("duckdb_connect"))
def duckdb_connect(duckdb_database_p, duckdb_connection_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_connect """
    return _call_lib_func("duckdb_connect", (duckdb_database_p, duckdb_connection_pp))


@cres(signatures.get("duckdb_create_bit"))
def duckdb_create_bit(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_bit """
    return _call_lib_func_struct_in("duckdb_create_bit", val)


@cres(signatures.get("duckdb_create_blob"))
def duckdb_create_blob(data_p, length):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_blob """
    return _call_lib_func("duckdb_create_blob", (data_p, length))


@cres(signatures.get("duckdb_create_date"))
def duckdb_create_date(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_date """
    return _call_lib_func("duckdb_create_date", (val,))


@intrinsic
def _duckdb_create_decimal(typingctx, decimal_tup_ty):
    """Custom intrinsic for duckdb_create_decimal.

    duckdb_decimal is 24 bytes ({uint8, uint8, uint64, int64}) — too large
    for register passing on any platform. Always passed by pointer.
    On SysV x86-64, byval + optnone are needed to prevent LLVM from
    optimizing away the stack copy before the callee reads it.
    See: https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    """
    def codegen(context, builder, signature, arguments):
        decimal_tup = arguments[0]
        decimal_tup_ll_ty = decimal_tup.type
        stack_p = builder.alloca(decimal_tup_ll_ty)
        builder.store(decimal_tup, stack_p)
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type),
            [decimal_tup_ll_ty.as_pointer()]
        )
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, "duckdb_create_decimal")
        if _is_sysv_x86_64:
            func_p.args[0].add_attribute('byval')
            builder.function.attributes.add('optnone')
            builder.function.attributes.add('noinline')
        return builder.call(func_p, [stack_p])
    return intp(duckdb_decimal_ty), codegen


@cres(intp(duckdb_decimal_ty))
def duckdb_create_decimal(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_decimal """
    return _duckdb_create_decimal(val)


@cres(signatures.get("duckdb_create_hugeint"))
def duckdb_create_hugeint(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_hugeint """
    return _call_lib_func_struct_in("duckdb_create_hugeint", val)


@intrinsic
def _duckdb_create_interval(typingctx, interval_tup_ty):
    """Custom intrinsic for duckdb_create_interval.

    Uses _build_packed_interval to repack the 3-field interval into {i64, i64}.
    On Windows: passes by pointer (byval). On SysV x86-64: passes by value
    (≤16 bytes fits in two registers).
    """
    def codegen(context, builder, signature, arguments):
        val = _build_packed_interval(builder, arguments[0])
        interval_struct = val.type
        ret_type = context.get_value_type(signature.return_type)
        if _is_win:
            return _emit_byval_call(
                builder, context, val, interval_struct, ret_type,
                "duckdb_create_interval")
        func_ty_ll = FunctionType(ret_type, [interval_struct])
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, "duckdb_create_interval")
        return builder.call(func_p, [val])
    return intp(duckdb_interval_ty), codegen


@cres(intp(duckdb_interval_ty))
def duckdb_create_interval(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_interval """
    return _duckdb_create_interval(val)


@cres(signatures.get("duckdb_create_time"))
def duckdb_create_time(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_time """
    return _call_lib_func("duckdb_create_time", (val,))


@cres(signatures.get("duckdb_create_time_tz"))
def duckdb_create_time_tz(micros, offset):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_time_tz """
    return _call_lib_func("duckdb_create_time_tz", (micros, offset))


@cres(signatures.get("duckdb_create_time_tz_value"))
def duckdb_create_time_tz_value(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_time_tz_value """
    return _call_lib_func("duckdb_create_time_tz_value", (val,))


@cres(signatures.get("duckdb_create_timestamp"))
def duckdb_create_timestamp(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_timestamp """
    return _call_lib_func("duckdb_create_timestamp", (val,))


@cres(signatures.get("duckdb_create_timestamp_ms"))
def duckdb_create_timestamp_ms(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_timestamp_ms """
    return _call_lib_func("duckdb_create_timestamp_ms", (val,))


@cres(signatures.get("duckdb_create_timestamp_ns"))
def duckdb_create_timestamp_ns(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_timestamp_ns """
    return _call_lib_func("duckdb_create_timestamp_ns", (val,))


@cres(signatures.get("duckdb_create_timestamp_s"))
def duckdb_create_timestamp_s(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_timestamp_s """
    return _call_lib_func("duckdb_create_timestamp_s", (val,))


@cres(signatures.get("duckdb_create_timestamp_tz"))
def duckdb_create_timestamp_tz(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_timestamp_tz """
    return _call_lib_func("duckdb_create_timestamp_tz", (val,))


@cres(signatures.get("duckdb_create_uhugeint"))
def duckdb_create_uhugeint(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_uhugeint """
    return _call_lib_func_struct_in("duckdb_create_uhugeint", val)


@cres(signatures.get("duckdb_create_uuid"))
def duckdb_create_uuid(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_uuid """
    return _call_lib_func_struct_in("duckdb_create_uuid", val)


@intrinsic
def _duckdb_create_varint(typingctx, varint_tup_ty):
    """Custom intrinsic for duckdb_create_varint.

    duckdb_varint is 24 bytes ({intp, uint64, int8}) — same >16-byte
    case as decimal. See _duckdb_create_decimal docstring for details.
    """
    def codegen(context, builder, signature, arguments):
        varint_tup = arguments[0]
        varint_ll_ty = context.get_value_type(duckdb_varint_ty)
        stack_p = builder.alloca(varint_ll_ty)
        builder.store(varint_tup, stack_p)
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type),
            [varint_ll_ty.as_pointer()]
        )
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, "duckdb_create_varint")
        if _is_sysv_x86_64:
            func_p.args[0].add_attribute('byval')
            builder.function.attributes.add('optnone')
            builder.function.attributes.add('noinline')
        return builder.call(func_p, [stack_p])
    return intp(duckdb_varint_ty), codegen


@cres(intp(duckdb_varint_ty))
def duckdb_create_varint(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_varint """
    return _duckdb_create_varint(val)


@cres(signatures.get("duckdb_data_chunk_get_vector"))
def duckdb_data_chunk_get_vector(chunk_p, idx):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_data_chunk_get_vector """
    return _call_lib_func("duckdb_data_chunk_get_vector", (chunk_p, idx))


@cres(signatures.get("duckdb_destroy_data_chunk"))
def duckdb_destroy_data_chunk(data_chunk_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_data_chunk
    todo: need to access pp """
    return _call_lib_func("duckdb_destroy_data_chunk", (data_chunk_pp,))


@cres(signatures.get("duckdb_destroy_logical_type"))
def duckdb_destroy_logical_type(logical_type_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_logical_type """
    return _call_lib_func("duckdb_destroy_logical_type", (logical_type_pp,))


@cres(signatures.get("duckdb_destroy_prepare"))
def duckdb_destroy_prepare(prepared_statement_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_prepare """
    return _call_lib_func("duckdb_destroy_prepare", (prepared_statement_pp,))


@cres(signatures.get("duckdb_destroy_result"))
def duckdb_destroy_result(duckdb_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_result """
    return _call_lib_func("duckdb_destroy_result", (duckdb_result_p,))


@cres(signatures.get("duckdb_destroy_value"))
def duckdb_destroy_value(value_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_value """
    return _call_lib_func("duckdb_destroy_value", (value_pp,))


@cres(signatures.get("duckdb_disconnect"))
def duckdb_disconnect(duckdb_connection_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_disconnect """
    return _call_lib_func("duckdb_disconnect", (duckdb_connection_pp,))


@cres(signatures.get("duckdb_execute_prepared"))
def duckdb_execute_prepared(prepared_statement_p, out_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_execute_prepared """
    return _call_lib_func("duckdb_execute_prepared", (prepared_statement_p, out_result_p))


@cres(signatures.get("duckdb_get_bool"))
def duckdb_get_bool(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_bool """
    return _call_lib_func("duckdb_get_bool", (val_p,))


@cres(signatures.get("duckdb_get_double"))
def duckdb_get_double(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_double """
    return _call_lib_func("duckdb_get_double", (val_p,))


@cres(signatures.get("duckdb_get_enum_value"))
def duckdb_get_enum_value(value_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_enum_value """
    return _call_lib_func("duckdb_get_enum_value", (value_p,))


@cres(signatures.get("duckdb_get_float"))
def duckdb_get_float(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_float """
    return _call_lib_func("duckdb_get_float", (val_p,))


@cres(signatures.get("duckdb_get_int8"))
def duckdb_get_int8(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_int8 """
    return _call_lib_func("duckdb_get_int8", (val_p,))


@cres(signatures.get("duckdb_get_int16"))
def duckdb_get_int16(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_int16 """
    return _call_lib_func("duckdb_get_int16", (val_p,))


@cres(signatures.get("duckdb_get_int32"))
def duckdb_get_int32(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_int32 """
    return _call_lib_func("duckdb_get_int32", (val_p,))


@cres(signatures.get("duckdb_get_int64"))
def duckdb_get_int64(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_int64 """
    return _call_lib_func("duckdb_get_int64", (val_p,))


@cres(signatures.get("duckdb_get_list_child"))
def duckdb_get_list_child(value_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_list_child """
    return _call_lib_func("duckdb_get_list_child", (value_p, index))


@cres(signatures.get("duckdb_get_list_size"))
def duckdb_get_list_size(value_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_list_size """
    return _call_lib_func("duckdb_get_list_size", (value_p,))


@cres(signatures.get("duckdb_get_map_key"))
def duckdb_get_map_key(value_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_map_key """
    return _call_lib_func("duckdb_get_map_key", (value_p, index))


@cres(signatures.get("duckdb_get_map_size"))
def duckdb_get_map_size(value_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_map_size """
    return _call_lib_func("duckdb_get_map_size", (value_p,))


@cres(signatures.get("duckdb_get_map_value"))
def duckdb_get_map_value(value_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_map_value """
    return _call_lib_func("duckdb_get_map_value", (value_p, index))


@cres(signatures.get("duckdb_get_struct_child"))
def duckdb_get_struct_child(value_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_struct_child """
    return _call_lib_func("duckdb_get_struct_child", (value_p, index))


@cres(signatures.get("duckdb_get_uint8"))
def duckdb_get_uint8(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_uint8 """
    return _call_lib_func("duckdb_get_uint8", (val_p,))


@cres(signatures.get("duckdb_get_uint16"))
def duckdb_get_uint16(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_uint16 """
    return _call_lib_func("duckdb_get_uint16", (val_p,))


@cres(signatures.get("duckdb_get_uint32"))
def duckdb_get_uint32(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_uint32 """
    return _call_lib_func("duckdb_get_uint32", (val_p,))


@cres(signatures.get("duckdb_get_uint64"))
def duckdb_get_uint64(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_uint64 """
    return _call_lib_func("duckdb_get_uint64", (val_p,))


@cres(signatures.get("duckdb_get_value_type"))
def duckdb_get_value_type(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_value_type """
    return _call_lib_func("duckdb_get_value_type", (val_p,))


@cres(signatures.get("duckdb_get_varchar"))
def duckdb_get_varchar(value_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_varchar """
    return _call_lib_func("duckdb_get_varchar", (value_p,))


@cres(signatures.get("duckdb_is_null_value"))
def duckdb_is_null_value(value_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_is_null_value """
    return _call_lib_func("duckdb_is_null_value", (value_p,))


@cres(signatures.get("duckdb_fetch_chunk"))
def duckdb_fetch_chunk(duckdb_result):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_fetch_chunk """
    return _call_lib_func_byval("duckdb_fetch_chunk", duckdb_result)


@cres(signatures.get("duckdb_free"))
def duckdb_free(ptr):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_free """
    return _call_lib_func("duckdb_free", (ptr,))


@cres(signatures.get("duckdb_get_bit"))
def duckdb_get_bit(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_bit """
    return _call_lib_func_struct_out("duckdb_get_bit", val_p)


@cres(signatures.get("duckdb_get_blob"))
def duckdb_get_blob(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_blob """
    return _call_lib_func_struct_out("duckdb_get_blob", val_p)


@cres(signatures.get("duckdb_get_date"))
def duckdb_get_date(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_date """
    return _call_lib_func("duckdb_get_date", (val_p,))


@intrinsic
def _duckdb_get_decimal(typingctx, val_p_ty):
    def codegen(context, builder, signature, arguments):
        val_p = arguments[0]
        decimal_ll_ty = context.get_value_type(duckdb_decimal_ty)
        sret_p = builder.alloca(decimal_ll_ty)
        func_ty_ll = FunctionType(
            ir.VoidType(),
            [decimal_ll_ty.as_pointer(), val_p.type]
        )
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, "duckdb_get_decimal")
        func_p.args[0].add_attribute('sret')
        builder.call(func_p, [sret_p, val_p])
        return builder.load(sret_p)
    return duckdb_decimal_ty(intp), codegen


@cres(duckdb_decimal_ty(intp))
def duckdb_get_decimal(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_decimal """
    return _duckdb_get_decimal(val_p)


@cres(signatures.get("duckdb_get_hugeint"))
def duckdb_get_hugeint(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_hugeint """
    return _call_lib_func_struct_out("duckdb_get_hugeint", val_p)


@intrinsic
def _duckdb_get_interval(typingctx, val_p_ty):
    def codegen(context, builder, signature, arguments):
        val_p = arguments[0]
        interval_ll_ty = context.get_value_type(duckdb_interval_ty)
        packed_struct = ir.LiteralStructType([_i64, _i64])
        if _is_win:
            sret_p = builder.alloca(packed_struct)
            func_ty_ll = FunctionType(
                ir.VoidType(),
                [packed_struct.as_pointer(), val_p.type]
            )
            func_p = get_or_insert_function(
                builder.module, func_ty_ll, "duckdb_get_interval")
            func_p.args[0].add_attribute('sret')
            builder.call(func_p, [sret_p, val_p])
            packed_val = builder.load(sret_p)
        else:
            func_ty_ll = FunctionType(packed_struct, [val_p.type])
            func_p = get_or_insert_function(
                builder.module, func_ty_ll, "duckdb_get_interval")
            packed_val = builder.call(func_p, [val_p])
        packed_lo = builder.extract_value(packed_val, 0)
        micros = builder.extract_value(packed_val, 1)
        months = builder.trunc(packed_lo, _i32)
        days_shifted = builder.lshr(packed_lo, ir.Constant(_i64, 32))
        days = builder.trunc(days_shifted, _i32)
        result = ir.Constant(interval_ll_ty, ir.Undefined)
        result = builder.insert_value(result, months, 0)
        result = builder.insert_value(result, days, 1)
        result = builder.insert_value(result, micros, 2)
        return result
    return duckdb_interval_ty(intp), codegen


@cres(duckdb_interval_ty(intp))
def duckdb_get_interval(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_interval """
    return _duckdb_get_interval(val_p)


@cres(signatures.get("duckdb_get_time"))
def duckdb_get_time(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_time """
    return _call_lib_func("duckdb_get_time", (val_p,))


@cres(signatures.get("duckdb_get_time_tz"))
def duckdb_get_time_tz(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_time_tz """
    return _call_lib_func("duckdb_get_time_tz", (val_p,))


@cres(signatures.get("duckdb_get_timestamp"))
def duckdb_get_timestamp(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_timestamp """
    return _call_lib_func("duckdb_get_timestamp", (val_p,))


@cres(signatures.get("duckdb_get_timestamp_ms"))
def duckdb_get_timestamp_ms(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_timestamp_ms """
    return _call_lib_func("duckdb_get_timestamp_ms", (val_p,))


@cres(signatures.get("duckdb_get_timestamp_ns"))
def duckdb_get_timestamp_ns(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_timestamp_ns """
    return _call_lib_func("duckdb_get_timestamp_ns", (val_p,))


@cres(signatures.get("duckdb_get_timestamp_s"))
def duckdb_get_timestamp_s(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_timestamp_s """
    return _call_lib_func("duckdb_get_timestamp_s", (val_p,))


@cres(signatures.get("duckdb_get_timestamp_tz"))
def duckdb_get_timestamp_tz(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_timestamp_tz """
    return _call_lib_func("duckdb_get_timestamp_tz", (val_p,))


@cres(signatures.get("duckdb_get_uhugeint"))
def duckdb_get_uhugeint(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_uhugeint """
    return _call_lib_func_struct_out("duckdb_get_uhugeint", val_p)


@cres(signatures.get("duckdb_get_uuid"))
def duckdb_get_uuid(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_uuid """
    return _call_lib_func_struct_out("duckdb_get_uuid", val_p)


@intrinsic
def _duckdb_get_varint(typingctx, val_p_ty):
    def codegen(context, builder, signature, arguments):
        val_p = arguments[0]
        varint_ll_ty = context.get_value_type(duckdb_varint_ty)
        sret_p = builder.alloca(varint_ll_ty)
        func_ty_ll = FunctionType(
            ir.VoidType(),
            [varint_ll_ty.as_pointer(), val_p.type]
        )
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, "duckdb_get_varint")
        func_p.args[0].add_attribute('sret')
        builder.call(func_p, [sret_p, val_p])
        return builder.load(sret_p)
    return duckdb_varint_ty(intp), codegen


@cres(duckdb_varint_ty(intp))
def duckdb_get_varint(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_varint """
    return _duckdb_get_varint(val_p)


@cres(signatures.get("duckdb_nparams"))
def duckdb_nparams(prepared_statement_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_nparams """
    return _call_lib_func("duckdb_nparams", (prepared_statement_p,))


@cres(signatures.get("duckdb_open"))
def duckdb_open(path_p, duckdb_database_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_open """
    return _call_lib_func("duckdb_open", (path_p, duckdb_database_pp))


@cres(signatures.get("duckdb_prepare_error"))
def duckdb_prepare_error(prepared_statement_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_prepare_error """
    return _call_lib_func("duckdb_prepare_error", (prepared_statement_p,))


@cres(signatures.get("duckdb_prepare"))
def duckdb_prepare(connection_p, query_p, out_prepared_statement_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_prepare """
    return _call_lib_func("duckdb_prepare", (connection_p, query_p, out_prepared_statement_pp))


@cres(signatures.get("duckdb_query"))
def duckdb_query(duckdb_connection_p, query_p, out_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_query """
    return _call_lib_func("duckdb_query", (duckdb_connection_p, query_p, out_result_p))


@cres(signatures.get("duckdb_result_error"))
def duckdb_result_error(duckdb_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_result_error """
    return _call_lib_func("duckdb_result_error", (duckdb_result_p,))


@cres(signatures.get("duckdb_result_error_type"))
def duckdb_result_error_type(duckdb_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_result_error_type """
    return _call_lib_func("duckdb_result_error_type", (duckdb_result_p,))


@cres(signatures.get("duckdb_result_return_type"))
def duckdb_result_return_type(result):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_result_return_type """
    return _call_lib_func_byval("duckdb_result_return_type", result)


@cres(signatures.get("duckdb_result_statement_type"))
def duckdb_result_statement_type(result):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_result_statement_type """
    return _call_lib_func_byval("duckdb_result_statement_type", result)


@cres(signatures.get("duckdb_row_count"))
def duckdb_row_count(duckdb_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_row_count """
    return _call_lib_func("duckdb_row_count", (duckdb_result_p,))


@cres(signatures.get("duckdb_rows_changed"))
def duckdb_rows_changed(duckdb_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_rows_changed """
    return _call_lib_func("duckdb_rows_changed", (duckdb_result_p,))


@cres(signatures.get("duckdb_validity_row_is_valid"))
def duckdb_validity_row_is_valid(validity_p, row):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_validity_row_is_valid """
    return _call_lib_func("duckdb_validity_row_is_valid", (validity_p, row))


@cres(signatures.get("duckdb_value_to_string"))
def duckdb_value_to_string(value_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_value_to_string """
    return _call_lib_func("duckdb_value_to_string", (value_p,))


@cres(signatures.get("duckdb_vector_get_data"))
def duckdb_vector_get_data(duckdb_vector_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_vector_get_data """
    return _call_lib_func("duckdb_vector_get_data", (duckdb_vector_p,))


@cres(signatures.get("duckdb_vector_get_validity"))
def duckdb_vector_get_validity(duckdb_vector_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_vector_get_validity """
    return _call_lib_func("duckdb_vector_get_validity", (duckdb_vector_p,))


@intrinsic
def _duckdb_bind_hugeint(typingctx, prepared_statement_p_ty, param_idx_ty, hugeint_tup_ty):
    def codegen(context, builder: IRBuilder, signature, arguments):
        prepared_statement_p, param_idx, hugeint_tup = arguments
        # Numba lowers Tuple((uint64, int64)) to LLVM {i64, i64}, which
        # matches the C struct layout directly — no repacking needed
        # (unlike interval, which has mixed-width i32/i64 fields).
        hugeint_ll_ty = context.get_value_type(duckdb_hugeint_ty)
        if _is_win:
            # Windows x64: structs >8 bytes passed by pointer
            stack_p = builder.alloca(hugeint_ll_ty)
            builder.store(hugeint_tup, stack_p)
            func_ty_ll = FunctionType(
                context.get_value_type(signature.return_type),
                [prepared_statement_p.type, param_idx.type,
                 hugeint_ll_ty.as_pointer()]
            )
            func_p = get_or_insert_function(
                builder.module, func_ty_ll, "duckdb_bind_hugeint")
            return builder.call(
                func_p, [prepared_statement_p, param_idx, stack_p])
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type),
            [prepared_statement_p.type, param_idx.type, hugeint_ll_ty]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_bind_hugeint")
        return builder.call(func_p, [prepared_statement_p, param_idx, hugeint_tup])
    return duckdb_state_ty(intp, uint64, duckdb_hugeint_ty), codegen


@cres(duckdb_state_ty(intp, uint64, duckdb_hugeint_ty))
def duckdb_bind_hugeint(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_hugeint """
    return _duckdb_bind_hugeint(prepared_statement_p, param_idx, val)


@intrinsic
def _duckdb_bind_uhugeint(typingctx, prepared_statement_p_ty, param_idx_ty, uhugeint_tup_ty):
    def codegen(context, builder: IRBuilder, signature, arguments):
        prepared_statement_p, param_idx, uhugeint_tup = arguments
        # Numba lowers UniTuple(uint64, 2) to LLVM {i64, i64}, which
        # matches the C struct layout directly — no repacking needed.
        uhugeint_ll_ty = context.get_value_type(duckdb_uhugeint_ty)
        if _is_win:
            # Windows x64: structs >8 bytes passed by pointer
            stack_p = builder.alloca(uhugeint_ll_ty)
            builder.store(uhugeint_tup, stack_p)
            func_ty_ll = FunctionType(
                context.get_value_type(signature.return_type),
                [prepared_statement_p.type, param_idx.type,
                 uhugeint_ll_ty.as_pointer()]
            )
            func_p = get_or_insert_function(
                builder.module, func_ty_ll, "duckdb_bind_uhugeint")
            return builder.call(
                func_p, [prepared_statement_p, param_idx, stack_p])
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type),
            [prepared_statement_p.type, param_idx.type, uhugeint_ll_ty]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_bind_uhugeint")
        return builder.call(func_p, [prepared_statement_p, param_idx, uhugeint_tup])
    return duckdb_state_ty(intp, uint64, duckdb_uhugeint_ty), codegen


@cres(duckdb_state_ty(intp, uint64, duckdb_uhugeint_ty))
def duckdb_bind_uhugeint(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uhugeint """
    return _duckdb_bind_uhugeint(prepared_statement_p, param_idx, val)


@intrinsic
def _duckdb_bind_interval(typingctx, prepared_statement_p_ty, param_idx_ty, interval_tup_ty):
    """Custom intrinsic for duckdb_bind_interval.

    Same interval packing as _duckdb_create_interval.
    See _build_packed_interval docstring for details.
    """
    def codegen(context, builder: IRBuilder, signature, arguments):
        prepared_statement_p, param_idx, interval_tup = arguments
        val = _build_packed_interval(builder, interval_tup)
        interval_struct = val.type
        if _is_win:
            stack_p = builder.alloca(interval_struct)
            builder.store(val, stack_p)
            func_ty_ll = FunctionType(
                context.get_value_type(signature.return_type),
                [prepared_statement_p.type, param_idx.type,
                 interval_struct.as_pointer()]
            )
            func_p = get_or_insert_function(
                builder.module, func_ty_ll, "duckdb_bind_interval")
            return builder.call(
                func_p, [prepared_statement_p, param_idx, stack_p])
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type),
            [prepared_statement_p.type, param_idx.type, interval_struct]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_bind_interval")
        return builder.call(func_p, [prepared_statement_p, param_idx, val])
    return duckdb_state_ty(intp, uint64, duckdb_interval_ty), codegen


@cres(duckdb_state_ty(intp, uint64, duckdb_interval_ty))
def duckdb_bind_interval(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_interval """
    return _duckdb_bind_interval(prepared_statement_p, param_idx, val)


@intrinsic
def _duckdb_bind_decimal(typingctx, prepared_statement_p_ty, param_idx_ty, decimal_tup_ty):
    """Custom intrinsic for duckdb_bind_decimal.

    duckdb_decimal is 24 bytes — always passed by pointer. On SysV x86-64,
    byval + optnone prevent LLVM from optimizing away the stack copy.
    See: https://github.com/numba/llvmlite/issues/300#issuecomment-327235846
    """
    def codegen(context, builder: IRBuilder, signature, arguments):
        prepared_statement_p, param_idx, decimal_tup = arguments
        # C struct: { uint8 width, uint8 scale, pad[6], {uint64 lower, int64 upper} }
        # 24 bytes total. Numba's {i8, i8, i64, i64} has the same layout
        # because LLVM aligns the i64 fields to 8-byte boundaries.
        decimal_tup_ty = decimal_tup.type
        decimal_stack_p = builder.alloca(decimal_tup_ty)
        builder.store(decimal_tup, decimal_stack_p)
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type),
            [prepared_statement_p.type, param_idx.type,
             decimal_tup_ty.as_pointer()]
        )
        func_p = get_or_insert_function(
            builder.module, func_ty_ll, "duckdb_bind_decimal")
        if _is_sysv_x86_64:
            # On x86-64 SysV, byval tells LLVM to copy the struct to the
            # stack for the callee. optnone prevents the optimizer from
            # eliminating the store to the alloca.
            # TODO(llvmlite): replace optnone with volatile stores when
            # llvmlite exposes builder.store(..., volatile=True).
            func_p.args[2].add_attribute('byval')
            builder.function.attributes.add('optnone')
            builder.function.attributes.add('noinline')
        # On arm64 and Windows x64, the C ABI passes >16/>8-byte structs
        # by pointer, so a plain pointer parameter matches the ABI.
        return builder.call(
            func_p, [prepared_statement_p, param_idx, decimal_stack_p])
    return duckdb_state_ty(intp, uint64, duckdb_decimal_ty), codegen


@cres(duckdb_state_ty(intp, uint64, duckdb_decimal_ty))
def duckdb_bind_decimal(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_decimal """
    return _duckdb_bind_decimal(prepared_statement_p, param_idx, val)
