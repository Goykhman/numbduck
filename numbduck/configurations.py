import json
import os

from numba.core.options import DefaultOptions


# njit's accepted compilation options. Take the target-level options straight
# from numba (``DefaultOptions`` carries one attribute per njit option) so this
# stays in sync automatically, and add the dispatcher-level ``cache`` flag
# handled by numba.core.decorators.jit. Any other key reaches njit as an
# unrecognized kwarg and aborts ducklib import with a numba error that never
# names the env var, so unknown keys are rejected here instead.
_ALLOWED_JIT_OPTIONS = frozenset(
    name for name in vars(DefaultOptions) if not name.startswith("__")
) | {"cache"}


def get_jit_options():
    """
    E.g., export NUMBDUCK_JIT_OPTIONS='{"cache": false}'
    """
    as_str = os.environ.get("NUMBDUCK_JIT_OPTIONS")
    if as_str is None:
        return {"cache": True}
    try:
        as_json = json.loads(as_str)
    except json.JSONDecodeError:
        raise ValueError("NUMBDUCK_JIT_OPTIONS must be valid JSON")
    if not isinstance(as_json, dict):
        raise ValueError('NUMBDUCK_JIT_OPTIONS must be a JSON object, e.g. {"cache": false}')
    unknown = set(as_json) - _ALLOWED_JIT_OPTIONS
    if unknown:
        raise ValueError(
            f"NUMBDUCK_JIT_OPTIONS contains unknown jit option(s) {sorted(unknown)}; "
            f"allowed keys are {sorted(_ALLOWED_JIT_OPTIONS)}"
        )
    return as_json


jit_options = get_jit_options()
