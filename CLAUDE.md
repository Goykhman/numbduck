# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

numbduck — adapts DuckDB's C API for use inside numba `@njit` code. Built on the [numbox](https://github.com/Goykhman/numbox) bindings toolkit.

## Build & Dev

- Venv: `python3.10 -m venv venv && venv/bin/pip install -e . flake8 pytest`
- Install: `pip install -e .`
- Test: `pytest`
- Lint: `flake8`
- Python: >=3.10
- Key dependencies: `duckdb~=1.3.2`, `numbox~=0.2.13`

## Architecture

### Bindings (ducklib.py)

Wraps DuckDB C API functions for use in numba JIT code. Same pattern as numbox bindings:

1. **`utils.py`** — finds and loads the DuckDB shared library via `ctypes.CDLL` with `RTLD_GLOBAL`
2. **`ducklib.py`** — registers signatures in numbox's `signatures` dict, then wraps each function with `@cres` + `_call_lib_func`
3. **`duckdb_utils.py`** — allocates numpy buffers for DuckDB C structs (database, connection, result, chunk, vector)

### Adding a New Binding

1. **Check `duckdb.h` first.** Look up the function signature in [`src/include/duckdb.h`](https://github.com/duckdb/duckdb/blob/v1.3.2/src/include/duckdb.h). Trace every typedef to its underlying type (e.g. `idx_t` → `uint64_t` → `uint64`, `duckdb_prepared_statement` is a pointer → use `_p` suffix). Verify all docstring links point to the correct line in `duckdb.h` at implementation time.
2. Add signature to `ducklib.py`: `signatures["duckdb_func"] = return_type(arg_types...)`
3. Add wrapper following this pattern:
```python
@cres(signatures.get("duckdb_func"))
def duckdb_func(arg):
    return _call_lib_func("duckdb_func", (arg,))
```
4. Function names must match the DuckDB C API names exactly
5. Docstring links must use `https://duckdb.org/docs/stable/clients/c/api.html#func_name`, not links to `duckdb.h` source
6. If a function returns a handle (e.g. `duckdb_logical_type`), also bind the corresponding destroy function (e.g. `duckdb_destroy_logical_type`)
7. **Before submitting an upstream PR**, re-verify all signatures, parameter types, naming conventions (`_p`/`_pp`), and docstring links against `duckdb.h` — this is a second check; step 1 is the first

### Special Case: duckdb_fetch_chunk

Uses a custom `@intrinsic` (`_duckdb_fetch_chunk`) instead of `_call_lib_func` because the result struct must be passed by pointer on the stack rather than by value.

## Key Paths

- `numbduck/ducklib.py` — all DuckDB C API bindings
- `numbduck/duckdb_utils.py` — buffer allocators for DuckDB structs
- `numbduck/utils.py` — shared library loader
- `test/test_ducklib.py` — integration tests

## Error Handling

Bindings must mirror the DuckDB C API error-handling protocol exactly — return state codes (`DuckDBSuccess`/`DuckDBError`) and pointers, never inject exceptions. Callers do C-style return code checks. See [numbox issue #5](https://github.com/Goykhman/numbox/issues/5) for rationale.

## Preferences

- Never include "Co-Authored-By" in git commit messages
- Avoid shell variable substitution in bash — inline actual values directly into commands
- Prefer simpler approaches
- Always git pull before making edits
- Commit messages must not mention AI, Claude, Anthropic, or any AI tooling — only attribute to the user
- Keep all memories in both MEMORY.md and the project CLAUDE.md (CLAUDE.md is in git and survives OS reinstalls)
- Environment details go in MEMORY.md only (may change between OS installs)
- Always exclude CLAUDE.md and .github/workflows/numbduck_ci.yml from upstream PRs (use a dedicated branch based on upstream/main)
- Always use a feature branch — never commit directly to main
- Never merge to main locally — only merge via PR on GitHub after all Actions pass
- Never merge local feature branches into main — main must always match upstream/main (exception: CLAUDE.md)
- Feature branches: base off origin/main (has CLAUDE.md); upstream PR branches: base off upstream/main (no CLAUDE.md)
- Do all coding work on the feature branch (has CLAUDE.md + fork CI), then cherry-pick to the upstream PR branch when ready
- Always enable GitHub Actions on forked repos
- Never assume a reviewer's comment is wrong — always verify claims against actual runtime before responding
- Before posting PR comments, check for pending reviews with existing comments (`GET /pulls/{pr}/reviews/{id}/comments`) — never silently delete a pending review, as it destroys all bundled draft comments
- Preface all AI-authored GitHub comments with "From the fake Slim Shady:"
- Never guess about things that can be verified — check the source of truth before making claims
- Always clean `__pycache__` and numba cache (`~/.cache/numba`) before every pytest run — stale JIT artifacts cause false failures
- Never put implementation planning details (task numbers, phase references, internal tracking) into code comments — comments must be context-independent
- When checking PR comments, show all comments — only skip nelson2005 comments containing "fake Slim Shady" (AI-authored); any other nelson2005 comment is from the user
- When told to "address" review feedback, implement your own recommendations — push back on items you assessed as not worth doing
- Always show PR review comments verbatim — never summarize or paraphrase
- Always include links to specific changed lines when responding to PR review comments
