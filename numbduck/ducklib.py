from llvmlite.ir import IRBuilder, FunctionType
from numba import njit
from numba.core.types import float32, float64, int8, int16, int32, int64, intp, Tuple, uint8, uint16, uint32, uint64, UniTuple, void
from numba.extending import intrinsic
from numbox.core.bindings.call import _call_lib_func
from numbox.core.bindings.signatures import signatures
from numba.core.cgutils import get_or_insert_function
from numbox.utils.highlevel import cres

from numbduck.utils import load_duckdb


duckdb_lib = load_duckdb()

duckdb_state_ty = int32

DuckDBSuccess = 0
DuckDBError = 1

duckdb_result_ty = UniTuple(intp, 6)
duckdb_hugeint_ty = Tuple((uint64, int64))
duckdb_uhugeint_ty = UniTuple(uint64, 2)
duckdb_interval_ty = Tuple((int32, int32, int64))
duckdb_decimal_ty = Tuple((uint8, uint8, uint64, int64))

signatures["duckdb_bind_boolean"] = duckdb_state_ty(intp, uint64, int8)
signatures["duckdb_bind_date"] = duckdb_state_ty(intp, uint64, int32)
signatures["duckdb_bind_double"] = duckdb_state_ty(intp, uint64, float64)
signatures["duckdb_bind_float"] = duckdb_state_ty(intp, uint64, float32)
signatures["duckdb_bind_int32"] = duckdb_state_ty(intp, uint64, int32)
signatures["duckdb_bind_int64"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_null"] = duckdb_state_ty(intp, uint64)
signatures["duckdb_bind_timestamp"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_varchar"] = duckdb_state_ty(intp, uint64, intp)
signatures["duckdb_bind_blob"] = duckdb_state_ty(intp, uint64, intp, uint64)
signatures["duckdb_bind_int8"] = duckdb_state_ty(intp, uint64, int8)
signatures["duckdb_bind_int16"] = duckdb_state_ty(intp, uint64, int16)
signatures["duckdb_bind_parameter_index"] = duckdb_state_ty(intp, intp, intp)
signatures["duckdb_bind_time"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_timestamp_tz"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_uint8"] = duckdb_state_ty(intp, uint64, uint8)
signatures["duckdb_bind_uint16"] = duckdb_state_ty(intp, uint64, uint16)
signatures["duckdb_bind_uint32"] = duckdb_state_ty(intp, uint64, uint32)
signatures["duckdb_bind_uint64"] = duckdb_state_ty(intp, uint64, uint64)
signatures["duckdb_bind_value"] = duckdb_state_ty(intp, uint64, intp)
signatures["duckdb_bind_varchar_length"] = duckdb_state_ty(intp, uint64, intp, uint64)
signatures["duckdb_close"] = void(intp)
signatures["duckdb_data_chunk_get_column_count"] = intp(intp)
signatures["duckdb_data_chunk_get_size"] = intp(intp)
signatures["duckdb_disconnect"] = void(intp)
signatures["duckdb_execute_prepared"] = duckdb_state_ty(intp, intp)
signatures["duckdb_column_count"] = intp(intp)
signatures["duckdb_connect"] = duckdb_state_ty(intp, intp)
signatures["duckdb_data_chunk_get_vector"] = intp(intp, intp)
signatures["duckdb_destroy_data_chunk"] = void(intp)
signatures["duckdb_destroy_prepare"] = void(intp)
signatures["duckdb_destroy_result"] = void(intp)
signatures["duckdb_fetch_chunk"] = intp(duckdb_result_ty)
signatures["duckdb_nparams"] = uint64(intp)
signatures["duckdb_open"] = duckdb_state_ty(intp, intp)
signatures["duckdb_prepare"] = duckdb_state_ty(intp, intp, intp)
signatures["duckdb_prepare_error"] = intp(intp)
signatures["duckdb_query"] = duckdb_state_ty(intp, intp, intp)
signatures["duckdb_result_error"] = intp(intp)
signatures["duckdb_row_count"] = intp(intp)
signatures["duckdb_validity_row_is_valid"] = int8(intp, intp)
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


@cres(signatures.get("duckdb_data_chunk_get_vector"))
def duckdb_data_chunk_get_vector(chunk_p, idx):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_data_chunk_get_vector """
    return _call_lib_func("duckdb_data_chunk_get_vector", (chunk_p, idx))


@cres(signatures.get("duckdb_disconnect"))
def duckdb_disconnect(duckdb_connection_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_disconnect """
    return _call_lib_func("duckdb_disconnect", (duckdb_connection_pp,))


@cres(signatures.get("duckdb_destroy_data_chunk"))
def duckdb_destroy_data_chunk(data_chunk_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_data_chunk
    todo: need to access pp """
    return _call_lib_func("duckdb_destroy_data_chunk", (data_chunk_pp,))


@cres(signatures.get("duckdb_destroy_prepare"))
def duckdb_destroy_prepare(prepared_statement_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_prepare """
    return _call_lib_func("duckdb_destroy_prepare", (prepared_statement_pp,))


@cres(signatures.get("duckdb_destroy_result"))
def duckdb_destroy_result(duckdb_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_result """
    return _call_lib_func("duckdb_destroy_result", (duckdb_result_p,))


@cres(signatures.get("duckdb_execute_prepared"))
def duckdb_execute_prepared(prepared_statement_p, out_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_execute_prepared """
    return _call_lib_func("duckdb_execute_prepared", (prepared_statement_p, out_result_p))


@intrinsic
def _duckdb_fetch_chunk(typingctx, duckdb_result_tup_ty):
    def codegen(context, builder: IRBuilder, signature, arguments):
        duckdb_result_tup = arguments[0]
        duckdb_result_tup_ty_ll = context.get_value_type(duckdb_result_tup_ty)
        duckdb_result_tup_stack_p = builder.alloca(duckdb_result_tup_ty_ll)
        builder.store(duckdb_result_tup, duckdb_result_tup_stack_p)
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type), [duckdb_result_tup_ty_ll.as_pointer()]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_fetch_chunk")
        return builder.call(func_p, [duckdb_result_tup_stack_p])
    return intp(duckdb_result_ty), codegen


@njit(signatures.get("duckdb_fetch_chunk"))
def duckdb_fetch_chunk(args):
    """ https://duckdb.org/docs/stable/clients/c/query#duckdb_fetch_chunk """
    return _duckdb_fetch_chunk(args)


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


@cres(signatures.get("duckdb_row_count"))
def duckdb_row_count(duckdb_result_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_row_count """
    return _call_lib_func("duckdb_row_count", (duckdb_result_p,))


@cres(signatures.get("duckdb_validity_row_is_valid"))
def duckdb_validity_row_is_valid(validity_p, row):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_validity_row_is_valid """
    return _call_lib_func("duckdb_validity_row_is_valid", (validity_p, row))


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
    import sys
    _is_win = sys.platform == 'win32'

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


@njit(duckdb_state_ty(intp, uint64, duckdb_hugeint_ty))
def duckdb_bind_hugeint(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_hugeint """
    return _duckdb_bind_hugeint(prepared_statement_p, param_idx, val)


@intrinsic
def _duckdb_bind_uhugeint(typingctx, prepared_statement_p_ty, param_idx_ty, uhugeint_tup_ty):
    import sys
    _is_win = sys.platform == 'win32'

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


@njit(duckdb_state_ty(intp, uint64, duckdb_uhugeint_ty))
def duckdb_bind_uhugeint(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uhugeint """
    return _duckdb_bind_uhugeint(prepared_statement_p, param_idx, val)


@intrinsic
def _duckdb_bind_interval(typingctx, prepared_statement_p_ty, param_idx_ty, interval_tup_ty):
    import sys
    _is_win = sys.platform == 'win32'

    def codegen(context, builder: IRBuilder, signature, arguments):
        from llvmlite import ir
        prepared_statement_p, param_idx, interval_tup = arguments
        i64 = ir.IntType(64)
        # C struct: { int32 months, int32 days, int64 micros } = 16 bytes
        # Pack two i32 fields into one i64 to match the C struct layout
        interval_struct = ir.LiteralStructType([i64, i64])
        months = builder.extract_value(interval_tup, 0)
        days = builder.extract_value(interval_tup, 1)
        micros = builder.extract_value(interval_tup, 2)
        # Pack months (low 32) | days (high 32) into first i64
        months_zext = builder.zext(months, i64)
        days_zext = builder.zext(days, i64)
        days_shifted = builder.shl(days_zext, ir.Constant(i64, 32))
        packed = builder.or_(months_zext, days_shifted)
        val = ir.Constant(interval_struct, ir.Undefined)
        val = builder.insert_value(val, packed, 0)
        val = builder.insert_value(val, micros, 1)
        if _is_win:
            # Windows x64: structs >8 bytes passed by pointer
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


@njit(duckdb_state_ty(intp, uint64, duckdb_interval_ty))
def duckdb_bind_interval(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_interval """
    return _duckdb_bind_interval(prepared_statement_p, param_idx, val)


@intrinsic
def _duckdb_bind_decimal(typingctx, prepared_statement_p_ty, param_idx_ty, decimal_tup_ty):
    import platform
    import sys
    # Only SysV x86-64 (Linux/macOS) needs byval — Windows AMD64 passes
    # >8-byte structs by pointer like arm64.  Windows reports 'AMD64' for
    # platform.machine(), but we exclude it explicitly for clarity.
    _is_sysv_x86_64 = platform.machine() == 'x86_64' and sys.platform != 'win32'

    def codegen(context, builder: IRBuilder, signature, arguments):
        from llvmlite import ir
        prepared_statement_p, param_idx, decimal_tup = arguments
        i8 = ir.IntType(8)
        i32 = ir.IntType(32)
        i64 = ir.IntType(64)
        # C struct: { uint8 width, uint8 scale, pad[6], {uint64 lower, int64 upper} }
        # 24 bytes — ABI handling varies by platform:
        # - x86-64 SysV: MEMORY class (>16 bytes), passed on stack via byval
        # - arm64/Windows: passed by implicit pointer in register
        hugeint_struct = ir.LiteralStructType([i64, i64])
        decimal_struct = ir.LiteralStructType([i8, i8, hugeint_struct])
        width = builder.extract_value(decimal_tup, 0)
        scale = builder.extract_value(decimal_tup, 1)
        lower = builder.extract_value(decimal_tup, 2)
        upper = builder.extract_value(decimal_tup, 3)
        zero = ir.Constant(i32, 0)
        decimal_stack_p = builder.alloca(decimal_struct)
        width_p = builder.gep(decimal_stack_p, [zero, ir.Constant(i32, 0)])
        builder.store(width, width_p)
        scale_p = builder.gep(decimal_stack_p, [zero, ir.Constant(i32, 1)])
        builder.store(scale, scale_p)
        lower_p = builder.gep(decimal_stack_p,
                              [zero, ir.Constant(i32, 2), ir.Constant(i32, 0)])
        builder.store(lower, lower_p)
        upper_p = builder.gep(decimal_stack_p,
                              [zero, ir.Constant(i32, 2), ir.Constant(i32, 1)])
        builder.store(upper, upper_p)
        func_ty_ll = FunctionType(
            context.get_value_type(signature.return_type),
            [prepared_statement_p.type, param_idx.type, decimal_struct.as_pointer()]
        )
        func_p = get_or_insert_function(builder.module, func_ty_ll, "duckdb_bind_decimal")
        if _is_sysv_x86_64:
            # On x86-64 SysV, byval tells LLVM to copy the struct to the
            # stack for the callee. optnone prevents the optimizer from
            # eliminating the GEP stores to the alloca. Without optnone,
            # the optimizer drops stores and the C function reads garbage.
            # NOTE: optnone applies to the entire enclosing @njit wrapper.
            # This is safe because the wrapper is trivial, but revisit with
            # volatile stores if llvmlite adds support.
            # TODO(llvmlite): replace optnone with volatile stores when
            # llvmlite exposes builder.store(..., volatile=True).
            func_p.args[2].add_attribute('byval')
            builder.function.attributes.add('optnone')
            builder.function.attributes.add('noinline')
        # On arm64 and Windows x64, the C ABI passes >16/>8-byte structs
        # by pointer, so a plain pointer parameter matches the ABI.
        return builder.call(func_p, [prepared_statement_p, param_idx, decimal_stack_p])
    return duckdb_state_ty(intp, uint64, duckdb_decimal_ty), codegen


@njit(duckdb_state_ty(intp, uint64, duckdb_decimal_ty))
def duckdb_bind_decimal(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_decimal """
    return _duckdb_bind_decimal(prepared_statement_p, param_idx, val)
