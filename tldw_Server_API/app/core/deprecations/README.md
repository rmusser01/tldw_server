# deprecations

`deprecations` is a support package for runtime compatibility and deprecation logging. It is not a user-facing feature module; it centralizes registry-backed deprecation messages so compatibility warnings are deliberate and emitted once per runtime cycle.

## Start Here

- `runtime_registry.py` contains the registry loader and runtime deprecation logging helpers.
- `__init__.py` re-exports the public helpers.
- Related consumer: `app/core/LLM_Calls/deprecation.py`. The registry (`COMPAT_PATHS`) is currently empty: the three paths past their sunset dates (`web_scraping_legacy_fallback`, `llm_chat_legacy_session`, `auth_db_execute_compat`) were removed along with their call sites in `chat_calls.py`, `web_scraping_service.py`, and `auth_service.py`.
- Related tests: `tests/Services/test_compatibility_registry_contract.py` and `tests/lint/test_no_new_runtime_compat_markers.py`.

## Responsibilities

- Load the compatibility/deprecation registry used by runtime callers.
- Compose runtime deprecation messages from approved registry entries.
- Log each deprecation key at most once per runtime cycle.
- Reset runtime deprecation cycle state for tests.
- Keep compatibility-warning emission behind a small reviewed API.

## Module Map

- `runtime_registry.py` - registry loading, message composition, once-per-cycle state, and logging helper.
- `__init__.py` - public re-exports.

## How It Connects

- `app/core/LLM_Calls/deprecation.py` calls into this package when emitting runtime compatibility warnings.
- Compatibility registry contract tests verify that registry-backed messages remain valid.
- Lint tests prevent new ad hoc runtime compatibility markers from being introduced outside the approved path, and fail if an expired key is re-added to the registry or referenced by a call site (`EXPIRED_COMPAT_KEYS` in `tests/lint/test_no_new_runtime_compat_markers.py`).

## Extension Points

- For a new runtime deprecation warning, add the registry entry first, then call `log_runtime_deprecation` from the runtime path.
- For message formatting changes, update `runtime_registry.py` and the compatibility registry contract tests.
- For tests that need a clean emission cycle, use the reset helper rather than mutating module internals.

## Testing

- `tests/Services/test_compatibility_registry_contract.py`
- `tests/lint/test_no_new_runtime_compat_markers.py`

## Gotchas

- Deprecation messages should not include secrets or raw request payloads.
- The once-per-cycle behavior is intentional; do not emit repeated warnings from loops or high-volume request paths.
- The lint contract expects runtime compatibility warnings to use the approved registry-backed API.
