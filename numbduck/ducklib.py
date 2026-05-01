from numba.core.types import float32, float64, int8, int16, int32, int64, intp, Tuple, uint8, uint16, uint32, uint64, UniTuple, void
from numbox.core.bindings.call import _call_lib_func, _call_lib_func_byval
from numbox.core.bindings.signatures import signatures
from numbox.utils.highlevel import cres, cres_if_available

import sys

from numbduck.utils import load_duckdb


duckdb_lib = load_duckdb()

_is_win = sys.platform == 'win32'


duckdb_state_ty = int32

DuckDBSuccess = 0
DuckDBError = 1

# duckdb_type enum (from duckdb.h)
DUCKDB_TYPE_INVALID = 0
DUCKDB_TYPE_BOOLEAN = 1
DUCKDB_TYPE_TINYINT = 2
DUCKDB_TYPE_SMALLINT = 3
DUCKDB_TYPE_INTEGER = 4
DUCKDB_TYPE_BIGINT = 5
DUCKDB_TYPE_UTINYINT = 6
DUCKDB_TYPE_USMALLINT = 7
DUCKDB_TYPE_UINTEGER = 8
DUCKDB_TYPE_UBIGINT = 9
DUCKDB_TYPE_FLOAT = 10
DUCKDB_TYPE_DOUBLE = 11
DUCKDB_TYPE_TIMESTAMP = 12
DUCKDB_TYPE_DATE = 13
DUCKDB_TYPE_TIME = 14
DUCKDB_TYPE_INTERVAL = 15
DUCKDB_TYPE_HUGEINT = 16
DUCKDB_TYPE_VARCHAR = 17
DUCKDB_TYPE_BLOB = 18
DUCKDB_TYPE_DECIMAL = 19
DUCKDB_TYPE_TIMESTAMP_S = 20
DUCKDB_TYPE_TIMESTAMP_MS = 21
DUCKDB_TYPE_TIMESTAMP_NS = 22
DUCKDB_TYPE_ENUM = 23
DUCKDB_TYPE_LIST = 24
DUCKDB_TYPE_STRUCT = 25
DUCKDB_TYPE_MAP = 26
DUCKDB_TYPE_UUID = 27
DUCKDB_TYPE_UNION = 28
DUCKDB_TYPE_BIT = 29
DUCKDB_TYPE_TIME_TZ = 30
DUCKDB_TYPE_TIMESTAMP_TZ = 31
DUCKDB_TYPE_UHUGEINT = 32
DUCKDB_TYPE_ARRAY = 33
DUCKDB_TYPE_ANY = 34
# VARINT (duckdb ≤1.4) and BIGNUM (duckdb ≥1.5) are the same type —
# same enum value, same struct layout ({byte_ptr, size, is_negative}).
# Both names are defined for cross-version portability.
DUCKDB_TYPE_VARINT = 35
DUCKDB_TYPE_BIGNUM = 35
DUCKDB_TYPE_SQLNULL = 36
DUCKDB_TYPE_STRING_LITERAL = 37
DUCKDB_TYPE_INTEGER_LITERAL = 38
DUCKDB_TYPE_TIME_NS = 39  # duckdb 1.5+

duckdb_result_ty = UniTuple(intp, 6)
duckdb_hugeint_ty = Tuple((uint64, int64))
duckdb_uhugeint_ty = UniTuple(uint64, 2)
duckdb_interval_ty = Tuple((int32, int32, int64))
duckdb_decimal_ty = Tuple((uint8, uint8, uint64, int64))
duckdb_blob_ty = Tuple((intp, uint64))
duckdb_bit_ty = Tuple((intp, uint64))
duckdb_varint_ty = Tuple((intp, uint64, int8))


signatures["duckdb_array_type_array_size"] = uint64(intp)
signatures["duckdb_array_type_child_type"] = intp(intp)
signatures["duckdb_bind_blob"] = duckdb_state_ty(intp, uint64, intp, uint64)
signatures["duckdb_bind_boolean"] = duckdb_state_ty(intp, uint64, int8)
signatures["duckdb_bind_date"] = duckdb_state_ty(intp, uint64, int32)
signatures["duckdb_bind_decimal"] = duckdb_state_ty(intp, uint64, duckdb_decimal_ty)
signatures["duckdb_bind_double"] = duckdb_state_ty(intp, uint64, float64)
signatures["duckdb_bind_float"] = duckdb_state_ty(intp, uint64, float32)
signatures["duckdb_bind_hugeint"] = duckdb_state_ty(intp, uint64, duckdb_hugeint_ty)
signatures["duckdb_bind_int8"] = duckdb_state_ty(intp, uint64, int8)
signatures["duckdb_bind_int16"] = duckdb_state_ty(intp, uint64, int16)
signatures["duckdb_bind_int32"] = duckdb_state_ty(intp, uint64, int32)
signatures["duckdb_bind_int64"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_interval"] = duckdb_state_ty(intp, uint64, duckdb_interval_ty)
signatures["duckdb_bind_null"] = duckdb_state_ty(intp, uint64)
signatures["duckdb_bind_parameter_index"] = duckdb_state_ty(intp, intp, intp)
signatures["duckdb_bind_time"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_timestamp"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_timestamp_tz"] = duckdb_state_ty(intp, uint64, int64)
signatures["duckdb_bind_uhugeint"] = duckdb_state_ty(intp, uint64, duckdb_uhugeint_ty)
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
signatures["duckdb_create_interval"] = intp(duckdb_interval_ty)
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
signatures["duckdb_create_decimal"] = intp(duckdb_decimal_ty)
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
signatures["duckdb_create_varint"] = intp(duckdb_varint_ty)
signatures["duckdb_create_varchar"] = intp(intp)
signatures["duckdb_create_varchar_length"] = intp(intp, uint64)
signatures["duckdb_data_chunk_get_column_count"] = intp(intp)
signatures["duckdb_data_chunk_get_size"] = intp(intp)
signatures["duckdb_data_chunk_get_vector"] = intp(intp, intp)
signatures["duckdb_decimal_internal_type"] = int32(intp)
signatures["duckdb_decimal_scale"] = uint8(intp)
signatures["duckdb_decimal_width"] = uint8(intp)
signatures["duckdb_destroy_data_chunk"] = void(intp)
signatures["duckdb_destroy_logical_type"] = void(intp)
signatures["duckdb_destroy_prepare"] = void(intp)
signatures["duckdb_destroy_result"] = void(intp)
signatures["duckdb_destroy_value"] = void(intp)
signatures["duckdb_disconnect"] = void(intp)
signatures["duckdb_enum_dictionary_size"] = uint32(intp)
signatures["duckdb_enum_dictionary_value"] = intp(intp, uint64)
signatures["duckdb_enum_internal_type"] = int32(intp)
signatures["duckdb_execute_prepared"] = duckdb_state_ty(intp, intp)
signatures["duckdb_fetch_chunk"] = intp(duckdb_result_ty)
signatures["duckdb_free"] = void(intp)
signatures["duckdb_get_bit"] = duckdb_bit_ty(intp)
signatures["duckdb_get_blob"] = duckdb_blob_ty(intp)
signatures["duckdb_get_bool"] = int8(intp)
signatures["duckdb_get_decimal"] = duckdb_decimal_ty(intp)
signatures["duckdb_get_date"] = int32(intp)
signatures["duckdb_get_double"] = float64(intp)
signatures["duckdb_get_enum_value"] = uint64(intp)
signatures["duckdb_get_float"] = float32(intp)
signatures["duckdb_get_hugeint"] = duckdb_hugeint_ty(intp)
signatures["duckdb_get_int8"] = int8(intp)
signatures["duckdb_get_int16"] = int16(intp)
signatures["duckdb_get_int32"] = int32(intp)
signatures["duckdb_get_int64"] = int64(intp)
signatures["duckdb_get_interval"] = duckdb_interval_ty(intp)
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
signatures["duckdb_get_type_id"] = int32(intp)
signatures["duckdb_get_uhugeint"] = duckdb_uhugeint_ty(intp)
signatures["duckdb_get_uint8"] = uint8(intp)
signatures["duckdb_get_uint16"] = uint16(intp)
signatures["duckdb_get_uint32"] = uint32(intp)
signatures["duckdb_get_uint64"] = uint64(intp)
signatures["duckdb_get_uuid"] = duckdb_uhugeint_ty(intp)
signatures["duckdb_get_value_type"] = intp(intp)
signatures["duckdb_get_varchar"] = intp(intp)
signatures["duckdb_get_varint"] = duckdb_varint_ty(intp)
signatures["duckdb_is_null_value"] = int8(intp)
signatures["duckdb_list_type_child_type"] = intp(intp)
signatures["duckdb_logical_type_get_alias"] = intp(intp)
signatures["duckdb_logical_type_set_alias"] = void(intp, intp)
signatures["duckdb_map_type_key_type"] = intp(intp)
signatures["duckdb_map_type_value_type"] = intp(intp)
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
signatures["duckdb_struct_type_child_count"] = uint64(intp)
signatures["duckdb_struct_type_child_name"] = intp(intp, uint64)
signatures["duckdb_struct_type_child_type"] = intp(intp, uint64)
signatures["duckdb_union_type_member_count"] = uint64(intp)
signatures["duckdb_union_type_member_name"] = intp(intp, uint64)
signatures["duckdb_union_type_member_type"] = intp(intp, uint64)
signatures["duckdb_validity_row_is_valid"] = int8(intp, intp)
signatures["duckdb_value_to_string"] = intp(intp)
signatures["duckdb_vector_get_data"] = intp(intp)
signatures["duckdb_vector_get_validity"] = uint64(intp)

signatures["duckdb_create_scalar_function"] = intp()
signatures["duckdb_destroy_scalar_function"] = void(intp)
signatures["duckdb_register_scalar_function"] = duckdb_state_ty(intp, intp)
signatures["duckdb_scalar_function_set_name"] = void(intp, intp)
signatures["duckdb_scalar_function_add_parameter"] = void(intp, intp)
signatures["duckdb_scalar_function_set_return_type"] = void(intp, intp)
signatures["duckdb_scalar_function_set_function"] = void(intp, intp)
signatures["duckdb_scalar_function_set_bind"] = void(intp, intp)
signatures["duckdb_scalar_function_set_extra_info"] = void(intp, intp, intp)
signatures["duckdb_scalar_function_set_varargs"] = void(intp, intp)
signatures["duckdb_scalar_function_set_volatile"] = void(intp)
signatures["duckdb_scalar_function_set_special_handling"] = void(intp)
signatures["duckdb_scalar_function_set_init"] = void(intp, intp)
signatures["duckdb_create_scalar_function_set"] = intp(intp)
signatures["duckdb_destroy_scalar_function_set"] = void(intp)
signatures["duckdb_add_scalar_function_to_set"] = duckdb_state_ty(intp, intp)
signatures["duckdb_register_scalar_function_set"] = duckdb_state_ty(intp, intp)

signatures["duckdb_create_aggregate_function"] = intp()
signatures["duckdb_destroy_aggregate_function"] = void(intp)
signatures["duckdb_register_aggregate_function"] = duckdb_state_ty(intp, intp)
signatures["duckdb_aggregate_function_set_name"] = void(intp, intp)
signatures["duckdb_aggregate_function_add_parameter"] = void(intp, intp)
signatures["duckdb_aggregate_function_set_return_type"] = void(intp, intp)
signatures["duckdb_aggregate_function_set_functions"] = void(intp, intp, intp, intp, intp, intp)
signatures["duckdb_aggregate_function_set_destructor"] = void(intp, intp)
signatures["duckdb_aggregate_function_set_extra_info"] = void(intp, intp, intp)
signatures["duckdb_aggregate_function_set_special_handling"] = void(intp)
signatures["duckdb_create_aggregate_function_set"] = intp(intp)
signatures["duckdb_destroy_aggregate_function_set"] = void(intp)
signatures["duckdb_add_aggregate_function_to_set"] = duckdb_state_ty(intp, intp)
signatures["duckdb_register_aggregate_function_set"] = duckdb_state_ty(intp, intp)

signatures["duckdb_scalar_function_get_extra_info"] = intp(intp)
signatures["duckdb_scalar_function_set_error"] = void(intp, intp)


@cres(signatures.get("duckdb_array_type_array_size"))
def duckdb_array_type_array_size(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_array_type_array_size """
    return _call_lib_func("duckdb_array_type_array_size", (type_p,))


@cres(signatures.get("duckdb_array_type_child_type"))
def duckdb_array_type_child_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_array_type_child_type """
    return _call_lib_func("duckdb_array_type_child_type", (type_p,))


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
    return _call_lib_func("duckdb_create_bit", (val,))


@cres(signatures.get("duckdb_create_blob"))
def duckdb_create_blob(data_p, length):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_blob """
    return _call_lib_func("duckdb_create_blob", (data_p, length))


@cres(signatures.get("duckdb_create_date"))
def duckdb_create_date(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_date """
    return _call_lib_func("duckdb_create_date", (val,))


@cres(signatures.get("duckdb_create_decimal"))
def duckdb_create_decimal(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_decimal """
    return _call_lib_func("duckdb_create_decimal", (val,))


@cres(signatures.get("duckdb_create_hugeint"))
def duckdb_create_hugeint(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_hugeint """
    return _call_lib_func("duckdb_create_hugeint", (val,))


@cres(signatures.get("duckdb_create_interval"))
def duckdb_create_interval(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_interval """
    return _call_lib_func("duckdb_create_interval", (val,))


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
    return _call_lib_func("duckdb_create_uhugeint", (val,))


@cres(signatures.get("duckdb_create_uuid"))
def duckdb_create_uuid(val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_uuid """
    return _call_lib_func("duckdb_create_uuid", (val,))


@cres_if_available(duckdb_lib, signatures.get("duckdb_create_varint"))
def duckdb_create_varint(val):
    """ https://duckdb.org/docs/1.3/clients/c/api#duckdb_create_varint """
    return _call_lib_func("duckdb_create_varint", (val,))


@cres(signatures.get("duckdb_data_chunk_get_vector"))
def duckdb_data_chunk_get_vector(chunk_p, idx):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_data_chunk_get_vector """
    return _call_lib_func("duckdb_data_chunk_get_vector", (chunk_p, idx))


@cres(signatures.get("duckdb_decimal_internal_type"))
def duckdb_decimal_internal_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_decimal_internal_type """
    return _call_lib_func("duckdb_decimal_internal_type", (type_p,))


@cres(signatures.get("duckdb_decimal_scale"))
def duckdb_decimal_scale(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_decimal_scale """
    return _call_lib_func("duckdb_decimal_scale", (type_p,))


@cres(signatures.get("duckdb_decimal_width"))
def duckdb_decimal_width(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_decimal_width """
    return _call_lib_func("duckdb_decimal_width", (type_p,))


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


@cres(signatures.get("duckdb_enum_dictionary_size"))
def duckdb_enum_dictionary_size(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_enum_dictionary_size """
    return _call_lib_func("duckdb_enum_dictionary_size", (type_p,))


@cres(signatures.get("duckdb_enum_dictionary_value"))
def duckdb_enum_dictionary_value(type_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_enum_dictionary_value """
    return _call_lib_func("duckdb_enum_dictionary_value", (type_p, index))


@cres(signatures.get("duckdb_enum_internal_type"))
def duckdb_enum_internal_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_enum_internal_type """
    return _call_lib_func("duckdb_enum_internal_type", (type_p,))


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
    return _call_lib_func("duckdb_get_bit", (val_p,))


@cres(signatures.get("duckdb_get_blob"))
def duckdb_get_blob(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_blob """
    return _call_lib_func("duckdb_get_blob", (val_p,))


@cres(signatures.get("duckdb_get_date"))
def duckdb_get_date(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_date """
    return _call_lib_func("duckdb_get_date", (val_p,))


@cres(signatures.get("duckdb_get_decimal"))
def duckdb_get_decimal(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_decimal """
    return _call_lib_func("duckdb_get_decimal", (val_p,))


@cres(signatures.get("duckdb_get_hugeint"))
def duckdb_get_hugeint(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_hugeint """
    return _call_lib_func("duckdb_get_hugeint", (val_p,))


@cres(signatures.get("duckdb_get_interval"))
def duckdb_get_interval(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_interval """
    return _call_lib_func("duckdb_get_interval", (val_p,))


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


@cres(signatures.get("duckdb_get_type_id"))
def duckdb_get_type_id(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_type_id """
    return _call_lib_func("duckdb_get_type_id", (type_p,))


@cres(signatures.get("duckdb_get_uhugeint"))
def duckdb_get_uhugeint(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_uhugeint """
    return _call_lib_func("duckdb_get_uhugeint", (val_p,))


@cres(signatures.get("duckdb_get_uuid"))
def duckdb_get_uuid(val_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_get_uuid """
    return _call_lib_func("duckdb_get_uuid", (val_p,))


@cres_if_available(duckdb_lib, signatures.get("duckdb_get_varint"))
def duckdb_get_varint(val_p):
    """ https://duckdb.org/docs/1.3/clients/c/api#duckdb_get_varint """
    return _call_lib_func("duckdb_get_varint", (val_p,))


@cres(signatures.get("duckdb_list_type_child_type"))
def duckdb_list_type_child_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_list_type_child_type """
    return _call_lib_func("duckdb_list_type_child_type", (type_p,))


@cres(signatures.get("duckdb_logical_type_get_alias"))
def duckdb_logical_type_get_alias(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_logical_type_get_alias """
    return _call_lib_func("duckdb_logical_type_get_alias", (type_p,))


@cres(signatures.get("duckdb_logical_type_set_alias"))
def duckdb_logical_type_set_alias(type_p, alias_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_logical_type_set_alias """
    return _call_lib_func("duckdb_logical_type_set_alias", (type_p, alias_p))


@cres(signatures.get("duckdb_map_type_key_type"))
def duckdb_map_type_key_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_map_type_key_type """
    return _call_lib_func("duckdb_map_type_key_type", (type_p,))


@cres(signatures.get("duckdb_map_type_value_type"))
def duckdb_map_type_value_type(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_map_type_value_type """
    return _call_lib_func("duckdb_map_type_value_type", (type_p,))


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


@cres(signatures.get("duckdb_struct_type_child_count"))
def duckdb_struct_type_child_count(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_struct_type_child_count """
    return _call_lib_func("duckdb_struct_type_child_count", (type_p,))


@cres(signatures.get("duckdb_struct_type_child_name"))
def duckdb_struct_type_child_name(type_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_struct_type_child_name """
    return _call_lib_func("duckdb_struct_type_child_name", (type_p, index))


@cres(signatures.get("duckdb_struct_type_child_type"))
def duckdb_struct_type_child_type(type_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_struct_type_child_type """
    return _call_lib_func("duckdb_struct_type_child_type", (type_p, index))


@cres(signatures.get("duckdb_union_type_member_count"))
def duckdb_union_type_member_count(type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_union_type_member_count """
    return _call_lib_func("duckdb_union_type_member_count", (type_p,))


@cres(signatures.get("duckdb_union_type_member_name"))
def duckdb_union_type_member_name(type_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_union_type_member_name """
    return _call_lib_func("duckdb_union_type_member_name", (type_p, index))


@cres(signatures.get("duckdb_union_type_member_type"))
def duckdb_union_type_member_type(type_p, index):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_union_type_member_type """
    return _call_lib_func("duckdb_union_type_member_type", (type_p, index))


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


@cres(signatures.get("duckdb_create_scalar_function"))
def duckdb_create_scalar_function():
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_scalar_function """
    return _call_lib_func("duckdb_create_scalar_function", ())


@cres(signatures.get("duckdb_destroy_scalar_function"))
def duckdb_destroy_scalar_function(scalar_function_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_scalar_function """
    return _call_lib_func("duckdb_destroy_scalar_function", (scalar_function_pp,))


@cres(signatures.get("duckdb_register_scalar_function"))
def duckdb_register_scalar_function(connection_p, scalar_function_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_register_scalar_function """
    return _call_lib_func("duckdb_register_scalar_function", (connection_p, scalar_function_p))


@cres(signatures.get("duckdb_scalar_function_set_name"))
def duckdb_scalar_function_set_name(scalar_function_p, name_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_set_name """
    return _call_lib_func("duckdb_scalar_function_set_name", (scalar_function_p, name_p))


@cres(signatures.get("duckdb_scalar_function_add_parameter"))
def duckdb_scalar_function_add_parameter(scalar_function_p, type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_add_parameter """
    return _call_lib_func("duckdb_scalar_function_add_parameter", (scalar_function_p, type_p))


@cres(signatures.get("duckdb_scalar_function_set_return_type"))
def duckdb_scalar_function_set_return_type(scalar_function_p, type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_set_return_type """
    return _call_lib_func("duckdb_scalar_function_set_return_type", (scalar_function_p, type_p))


@cres(signatures.get("duckdb_scalar_function_set_function"))
def duckdb_scalar_function_set_function(scalar_function_p, function_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_set_function """
    return _call_lib_func("duckdb_scalar_function_set_function", (scalar_function_p, function_p))


@cres(signatures.get("duckdb_scalar_function_set_bind"))
def duckdb_scalar_function_set_bind(scalar_function_p, bind_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_set_bind """
    return _call_lib_func("duckdb_scalar_function_set_bind", (scalar_function_p, bind_p))


@cres(signatures.get("duckdb_scalar_function_set_extra_info"))
def duckdb_scalar_function_set_extra_info(scalar_function_p, extra_info_p, destroy_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_set_extra_info """
    return _call_lib_func("duckdb_scalar_function_set_extra_info", (scalar_function_p, extra_info_p, destroy_p))


@cres(signatures.get("duckdb_scalar_function_set_varargs"))
def duckdb_scalar_function_set_varargs(scalar_function_p, type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_set_varargs """
    return _call_lib_func("duckdb_scalar_function_set_varargs", (scalar_function_p, type_p))


@cres(signatures.get("duckdb_scalar_function_set_volatile"))
def duckdb_scalar_function_set_volatile(scalar_function_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_set_volatile """
    return _call_lib_func("duckdb_scalar_function_set_volatile", (scalar_function_p,))


@cres(signatures.get("duckdb_scalar_function_set_special_handling"))
def duckdb_scalar_function_set_special_handling(scalar_function_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_set_special_handling """
    return _call_lib_func("duckdb_scalar_function_set_special_handling", (scalar_function_p,))


@cres_if_available(duckdb_lib, signatures.get("duckdb_scalar_function_set_init"))
def duckdb_scalar_function_set_init(scalar_function_p, init_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_set_init """
    return _call_lib_func("duckdb_scalar_function_set_init", (scalar_function_p, init_p))


@cres(signatures.get("duckdb_create_scalar_function_set"))
def duckdb_create_scalar_function_set(name_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_scalar_function_set """
    return _call_lib_func("duckdb_create_scalar_function_set", (name_p,))


@cres(signatures.get("duckdb_destroy_scalar_function_set"))
def duckdb_destroy_scalar_function_set(set_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_scalar_function_set """
    return _call_lib_func("duckdb_destroy_scalar_function_set", (set_pp,))


@cres(signatures.get("duckdb_add_scalar_function_to_set"))
def duckdb_add_scalar_function_to_set(set_p, scalar_function_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_add_scalar_function_to_set """
    return _call_lib_func("duckdb_add_scalar_function_to_set", (set_p, scalar_function_p))


@cres(signatures.get("duckdb_register_scalar_function_set"))
def duckdb_register_scalar_function_set(connection_p, set_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_register_scalar_function_set """
    return _call_lib_func("duckdb_register_scalar_function_set", (connection_p, set_p))


@cres(signatures.get("duckdb_create_aggregate_function"))
def duckdb_create_aggregate_function():
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_aggregate_function """
    return _call_lib_func("duckdb_create_aggregate_function", ())


@cres(signatures.get("duckdb_destroy_aggregate_function"))
def duckdb_destroy_aggregate_function(aggregate_function_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_aggregate_function """
    return _call_lib_func("duckdb_destroy_aggregate_function", (aggregate_function_pp,))


@cres(signatures.get("duckdb_register_aggregate_function"))
def duckdb_register_aggregate_function(connection_p, aggregate_function_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_register_aggregate_function """
    return _call_lib_func("duckdb_register_aggregate_function", (connection_p, aggregate_function_p))


@cres(signatures.get("duckdb_aggregate_function_set_name"))
def duckdb_aggregate_function_set_name(aggregate_function_p, name_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_aggregate_function_set_name """
    return _call_lib_func("duckdb_aggregate_function_set_name", (aggregate_function_p, name_p))


@cres(signatures.get("duckdb_aggregate_function_add_parameter"))
def duckdb_aggregate_function_add_parameter(aggregate_function_p, type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_aggregate_function_add_parameter """
    return _call_lib_func("duckdb_aggregate_function_add_parameter", (aggregate_function_p, type_p))


@cres(signatures.get("duckdb_aggregate_function_set_return_type"))
def duckdb_aggregate_function_set_return_type(aggregate_function_p, type_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_aggregate_function_set_return_type """
    return _call_lib_func("duckdb_aggregate_function_set_return_type", (aggregate_function_p, type_p))


@cres(signatures.get("duckdb_aggregate_function_set_functions"))
def duckdb_aggregate_function_set_functions(aggregate_function_p, state_size_p, init_p, update_p, combine_p, finalize_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_aggregate_function_set_functions """
    return _call_lib_func("duckdb_aggregate_function_set_functions", (aggregate_function_p, state_size_p, init_p, update_p, combine_p, finalize_p))


@cres(signatures.get("duckdb_aggregate_function_set_destructor"))
def duckdb_aggregate_function_set_destructor(aggregate_function_p, destroy_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_aggregate_function_set_destructor """
    return _call_lib_func("duckdb_aggregate_function_set_destructor", (aggregate_function_p, destroy_p))


@cres(signatures.get("duckdb_aggregate_function_set_extra_info"))
def duckdb_aggregate_function_set_extra_info(aggregate_function_p, extra_info_p, destroy_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_aggregate_function_set_extra_info """
    return _call_lib_func("duckdb_aggregate_function_set_extra_info", (aggregate_function_p, extra_info_p, destroy_p))


@cres(signatures.get("duckdb_aggregate_function_set_special_handling"))
def duckdb_aggregate_function_set_special_handling(aggregate_function_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_aggregate_function_set_special_handling """
    return _call_lib_func("duckdb_aggregate_function_set_special_handling", (aggregate_function_p,))


@cres(signatures.get("duckdb_create_aggregate_function_set"))
def duckdb_create_aggregate_function_set(name_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_create_aggregate_function_set """
    return _call_lib_func("duckdb_create_aggregate_function_set", (name_p,))


@cres(signatures.get("duckdb_destroy_aggregate_function_set"))
def duckdb_destroy_aggregate_function_set(set_pp):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_destroy_aggregate_function_set """
    return _call_lib_func("duckdb_destroy_aggregate_function_set", (set_pp,))


@cres(signatures.get("duckdb_add_aggregate_function_to_set"))
def duckdb_add_aggregate_function_to_set(set_p, aggregate_function_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_add_aggregate_function_to_set """
    return _call_lib_func("duckdb_add_aggregate_function_to_set", (set_p, aggregate_function_p))


@cres(signatures.get("duckdb_register_aggregate_function_set"))
def duckdb_register_aggregate_function_set(connection_p, set_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_register_aggregate_function_set """
    return _call_lib_func("duckdb_register_aggregate_function_set", (connection_p, set_p))


@cres(signatures.get("duckdb_scalar_function_get_extra_info"))
def duckdb_scalar_function_get_extra_info(info_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_get_extra_info """
    return _call_lib_func("duckdb_scalar_function_get_extra_info", (info_p,))


@cres(signatures.get("duckdb_scalar_function_set_error"))
def duckdb_scalar_function_set_error(info_p, error_p):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_scalar_function_set_error """
    return _call_lib_func("duckdb_scalar_function_set_error", (info_p, error_p))


@cres(signatures.get("duckdb_bind_hugeint"))
def duckdb_bind_hugeint(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_hugeint """
    return _call_lib_func("duckdb_bind_hugeint", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_uhugeint"))
def duckdb_bind_uhugeint(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_uhugeint """
    return _call_lib_func("duckdb_bind_uhugeint", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_interval"))
def duckdb_bind_interval(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_interval """
    return _call_lib_func("duckdb_bind_interval", (prepared_statement_p, param_idx, val))


@cres(signatures.get("duckdb_bind_decimal"))
def duckdb_bind_decimal(prepared_statement_p, param_idx, val):
    """ https://duckdb.org/docs/stable/clients/c/api.html#duckdb_bind_decimal """
    return _call_lib_func("duckdb_bind_decimal", (prepared_statement_p, param_idx, val))
