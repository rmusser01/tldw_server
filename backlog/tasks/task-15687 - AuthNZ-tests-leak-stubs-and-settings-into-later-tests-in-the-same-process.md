---
id: TASK-15687
title: AuthNZ tests leak stubs and settings into later tests in the same process
status: To Do
labels:
- authnz
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while verifying the account lock (TASK-15679, PR #3664). Several AuthNZ tests change process-wide state and do not restore it, so a test that passes alone fails when it runs after them. Two were fixed in that PR (test_auth_integration_flow_extended.py and test_forgot_password_flow_integration.py assigned stub modules and an always-true is_postgres_backend without restoring them, which made later MFA sign-ins skip the second factor and sent SQLite reset-password requests down the PostgreSQL path). These remain.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 tests/AuthNZ_Postgres/test_auth_enhanced_mfa.py no longer leaves stub mfa_service and email_service modules in sys.modules after its tests
- [ ] #2 A test's environment patch cannot be cached into the core config for the next test: monkeypatch is set up before the root autouse fixtures (through _offline_provider_model_inventory) and so is undone after them, and any teardown that reads settings after clear_config_cache reloads them under the patched environment. Shown by tests/AuthNZ/test_websocket_route_auth_matrix.py::test_sandbox_single_user_key_keeps_the_ip_allowlist[X-API-KEY] (closed 1008 because signed sandbox URLs were still on) and by a teardown FileNotFoundError after tests/AuthNZ/unit/test_initialize_mcp_secrets.py::test_authnz_initializer_does_not_create_missing_exclusive_env_file, both seen once in a single-process run of tests/AuthNZ + tests/AuthNZ_Unit + tests/AuthNZ_Federation and both passing alone
- [ ] #3 The app module that reload_app_main and importlib.reload(auth) leave in sys.modules does not change later tests, or those tests stop reloading
- [ ] #4 The endpoint-patch baseline for api/v1/endpoints/auth.py is tightened to what the tests now do
- [ ] #5 tests/AuthNZ, tests/AuthNZ_Unit, tests/AuthNZ_Federation and tests/AuthNZ_SQLite pass together in one process, three runs in a row
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
