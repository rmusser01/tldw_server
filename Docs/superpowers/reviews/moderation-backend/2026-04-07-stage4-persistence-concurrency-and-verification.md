# Stage 4 Persistence Concurrency and Verification

## Scope
Review moderation persistence, reload behavior, optimistic concurrency, and targeted verification for stateful claims.

## Files Reviewed
- `tldw_Server_API/app/core/Moderation/moderation_service.py`
- `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- `tldw_Server_API/tests/unit/test_moderation_etag_handling.py`
- `tldw_Server_API/tests/unit/test_moderation_user_override_contract.py`
- `tldw_Server_API/tests/unit/test_moderation_user_override_validation.py`
- `tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py`

## Tests Reviewed
- `tldw_Server_API/tests/unit/test_moderation_etag_handling.py`
- `tldw_Server_API/tests/unit/test_moderation_user_override_contract.py`
- `tldw_Server_API/tests/unit/test_moderation_user_override_validation.py`
- `tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py`
- Persistence and concurrency verification slice passed: `24 passed, 62 warnings in 2.18s`

## Validation Commands
- `rg -n "def (set_user_override|delete_user_override|list_user_overrides|get_blocklist_state|append_blocklist_line|delete_blocklist_index|set_blocklist_lines|reload|update_settings|get_settings|_persist|_save|etag|version|If-Match|write|persist)" tldw_Server_API/app/core/Moderation/moderation_service.py`
- `nl -ba tldw_Server_API/app/core/Moderation/moderation_service.py | sed -n '500,700p'`
- `nl -ba tldw_Server_API/app/core/Moderation/moderation_service.py | sed -n '1230,1425p'`
- `rg -n "_user_overrides_path|_blocklist_path|_runtime_overrides_file|RLock|threading|tempfile|replace\\(|os\\.replace|json\\.dump|sha256|version" tldw_Server_API/app/core/Moderation/moderation_service.py`
- `nl -ba tldw_Server_API/tests/unit/test_moderation_etag_handling.py | sed -n '1,260p'`
- `nl -ba tldw_Server_API/tests/unit/test_moderation_user_override_contract.py | sed -n '1,260p'`
- `nl -ba tldw_Server_API/tests/unit/test_moderation_user_override_validation.py | sed -n '1,240p'`
- `nl -ba tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py | sed -n '1,220p'`
- `source ../../.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/unit/test_moderation_etag_handling.py tldw_Server_API/tests/unit/test_moderation_user_override_contract.py tldw_Server_API/tests/unit/test_moderation_user_override_validation.py tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py`
- `source ../../.venv/bin/activate && python -c '<set_user_override persistence failure reproduction with _user_overrides_path pointed at a directory>'`
- `source ../../.venv/bin/activate && python -c '<delete_user_override persistence failure reproduction with _user_overrides_path pointed at a directory>'`
- `source ../../.venv/bin/activate && python -c '<update_settings(persist=True) failure reproduction with _runtime_overrides_path pointed at a directory>'`

## Confirmed Findings
- High, high confidence: user-override persistence failures still mutate the live in-memory policy state, so the service can return an error while the current process has already changed enforcement behavior. In [`moderation_service.py`](../../../../tldw_Server_API/app/core/Moderation/moderation_service.py) lines 1243-1274, `set_user_override()` writes the sanitized override into `self._user_overrides` before attempting the JSON file write; if the write fails it returns `{"ok": False, "persisted": False, "error_type": "persistence"}` but does not roll the in-memory change back. The deletion path has the same problem in lines 1276-1295: it removes the override from `self._user_overrides` before writing the file, and a persistence failure returns an error after the live state is already changed. Reproduction: pointing `_user_overrides_path` at a directory made `set_user_override("alice", {"enabled": True})` return a persistence error while `alice` was still present in `list_user_overrides()`, and made `delete_user_override("alice")` return a persistence error while the override list was already empty. Impact: callers can receive an error and retry or abort under the assumption that nothing changed, while the running server has already started enforcing the new override set until reload or restart. Current tests miss this exact failure mode: the endpoint contract tests use stubs in [`test_moderation_user_override_contract.py`](../../../../tldw_Server_API/tests/unit/test_moderation_user_override_contract.py) lines 49-69 and 133-237, and the real-service validation tests in [`test_moderation_user_override_validation.py`](../../../../tldw_Server_API/tests/unit/test_moderation_user_override_validation.py) lines 10-205 only cover validation rejection and load-time sanitization, not persistence failure after in-memory mutation.
- High, high confidence: runtime moderation settings fail open on persistence errors when `persist=True`. In [`moderation_service.py`](../../../../tldw_Server_API/app/core/Moderation/moderation_service.py) lines 562-584, `update_settings()` mutates `self._runtime_override`, calls `_save_runtime_overrides_file()` when `persist=True`, then recomputes the policy and returns success data. But `_save_runtime_overrides_file()` in lines 612-632 only logs and suppresses file-write failures, so the caller gets a normal success response even if nothing was persisted. The endpoint simply relays that data back to the client in [`moderation.py`](../../../../tldw_Server_API/app/api/v1/endpoints/moderation.py) lines 228-255. Reproduction: pointing `_runtime_overrides_path` at a directory and calling `update_settings(pii_enabled=True, persist=True)` produced a warning from `_save_runtime_overrides_file()`, returned `{"pii_enabled": True, ...}`, and left `self._runtime_override == {"pii_enabled": True}` in memory. Impact: clients can receive a 200-style success path and assume the setting is durable, but the change only exists in the current process and disappears on restart. Existing tests do not cover the failure path: [`test_moderation_runtime_overrides_bool.py`](../../../../tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py) lines 10-55 verifies parsing and clearing behavior only.

## Probable Risks
- Medium, medium confidence: user-override and runtime-override files are written with direct `open(..., "w")` plus `json.dump(...)`, not the temp-file-plus-rename pattern used for blocklist writes. The direct-write paths are in [`moderation_service.py`](../../../../tldw_Server_API/app/core/Moderation/moderation_service.py) lines 612-632, 1264-1269, and 1286-1288, while blocklist writes use `NamedTemporaryFile(..., delete=False)` plus `os.replace(...)` in lines 1316-1346. That means a crash, kill, or storage interruption during override or runtime-settings persistence can plausibly leave truncated JSON or a partially written file. I did not simulate mid-write process interruption, so this remains a probable risk rather than a confirmed defect.

## Improvements
- Roll back in-memory override mutations on persistence failure, or stage the new JSON on disk first and only swap the in-memory state after a successful write.
- Make `update_settings(persist=True)` return a structured error when `_save_runtime_overrides_file()` fails instead of silently succeeding with a non-durable in-memory change.
- Use the same atomic temp-file-and-`os.replace()` strategy for user overrides and runtime overrides that blocklist writes already use.
- Add real-service persistence-failure tests for `set_user_override()`, `delete_user_override()`, and `update_settings(persist=True)`.
- If strict RFC behavior matters, reconsider accepting weak validators in `_normalize_etag_list()`: the current endpoint tests in [`test_moderation_etag_handling.py`](../../../../tldw_Server_API/tests/unit/test_moderation_etag_handling.py) lines 117-129 explicitly allow `If-Match: W/"..."`.

## Open Questions
- None.

## Exit Note
The optimistic-concurrency and blocklist path itself is in relatively good shape for the cases reviewed: the managed endpoint quotes ETags, append and delete both re-check versions under the service lock, and the targeted ETag slice passed. The main Task 4 problems are failure semantics on override and runtime-settings persistence, where the service currently drifts live memory away from disk and, in one case, reports success after a failed save.
