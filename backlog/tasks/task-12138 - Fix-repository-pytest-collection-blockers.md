---
id: TASK-12138
title: Fix repository pytest collection blockers
status: Done
created_date: 2026-07-04 01:35
labels:
- tests
- pytest
- infrastructure
priority: High
references:
- /Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-phase-0-inventory/backlog/tasks/task-12027
  - Implement-Web-Scraping-refactor-Phase-0-import-inventory-and-guardrails.md
modified_files:
- pyproject.toml
- tldw_Server_API/tests/_plugins/authnz_full_fixtures.py
- tldw_Server_API/tests/Logging/test_pytest_plugin_isolation.py
- tldw_Server_API/tests/AuthNZ/integration/test_magic_link_flow_integration.py
- tldw_Server_API/tests/AuthNZ/unit/test_telegram_approvals_repo.py
- tldw_Server_API/tests/AuthNZ/unit/test_telegram_runtime_repo.py
- tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_api_keys_repo_sqlite.py
- tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py
- tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_mfa_repo_sqlite.py
- tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_org_provider_secrets_repo_sqlite.py
- tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_sessions_repo_sqlite.py
- tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_user_provider_secrets_repo_sqlite.py
- tldw_Server_API/tests/AuthNZ_SQLite/test_byok_oauth_state_repo_sqlite.py
- tldw_Server_API/tests/AuthNZ_SQLite/test_org_stt_settings_repo_sqlite.py
- tldw_Server_API/tests/AuthNZ_Unit/test_mcp_hub_capability_adapter_repo.py
- tldw_Server_API/tests/AuthNZ_Unit/test_mcp_hub_repo.py
- tldw_Server_API/tests/Collections/test_collections_postgres_integration.py
- tldw_Server_API/tests/Evaluations/test_embeddings_abtest_repository_postgres.py
- tldw_Server_API/tests/MCP_Hub/test_mcp_slot_status.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_external_access.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_external_slot_access.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_policy_overrides.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_service.py
- tldw_Server_API/tests/MediaDB2/test_media_db_postgres.py
- tldw_Server_API/tests/MediaDB2/test_read_contract_postgres.py
- tldw_Server_API/tests/Tools/test_tools_permissions.py
- tldw_Server_API/tests/Watchlists/test_watchlists_postgres_integration.py
- tldw_Server_API/tests/wizard/test_db_postgres_integration.py
updated_date: 2026-07-04 01:42
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the repo-wide pytest collection failures that block the Web_Scraping Phase 0 branch integration: duplicate test module basename import mismatches in default pytest mode and duplicate AuthNZ conftest plugin registration when using importlib collection mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default project pytest collection no longer fails on duplicate test module basename import mismatches.
- [x] #2 Importlib collection mode no longer fails on duplicate AuthNZ conftest plugin registration.
- [x] #3 Focused regression checks demonstrate both collection blockers are fixed.
- [x] #4 Full project pytest gate is retried and outcome is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented pytest collection unblocker after the Web_Scraping Phase 0 rebase exposed repo-wide collection failures.

Changes:
- Moved default pytest import mode into addopts as --import-mode=importlib and removed the ineffective import_mode ini key.
- Added tldw_Server_API.tests._plugins.authnz_full_fixtures as a safe plugin bridge that re-exports AuthNZ fixtures without registering AuthNZ/conftest.py itself as a pytest plugin.
- Redirected module-level AuthNZ fixture opt-ins from tldw_Server_API.tests.AuthNZ.conftest to the safe plugin bridge.
- Added regression checks in tldw_Server_API/tests/Logging/test_pytest_plugin_isolation.py for import mode config and AuthNZ plugin hygiene.

Verification:
- RED before fix: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Logging/test_pytest_plugin_isolation.py -q --tb=short failed with 3 expected config/plugin hygiene failures.
- Focused regression: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Logging/test_pytest_plugin_isolation.py -q --tb=short passed, 6 passed.
- Duplicate basename collection: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/API_Deps/test_meetings_db_deps_error_mapping.py tldw_Server_API/tests/Meetings/test_meetings_db_deps_error_mapping.py --collect-only -q passed, 13 tests collected.
- AuthNZ plugin collection sample: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest --collect-only -q tldw_Server_API/tests/AuthNZ/integration/test_magic_link_flow_integration.py tldw_Server_API/tests/AuthNZ/unit/test_telegram_approvals_repo.py tldw_Server_API/tests/MCP_Hub/test_mcp_slot_status.py passed, 8 tests collected.
- Full repository collection retry: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest --collect-only -q passed, 35,362 tests collected in 30.10s. Output: /tmp/tldw_pytest_collect_after_collection_fix.txt.
- git diff --check passed.
- rg confirmed no remaining pytest_plugins references to tldw_Server_API.tests.AuthNZ.conftest.
- Bandit touched-scope scan with only B101 skipped returned existing low-severity B105/B106 dummy secret findings in pre-existing test code on lines unrelated to this change. Filtered touched-scope scan excluding test assert/dummy secret rules B101,B105,B106 passed with zero findings. Outputs: /tmp/bandit_pytest_collection_fix.json and /tmp/bandit_pytest_collection_fix_filtered.json.
Marked Done after focused regressions, full collect-only retry, diff check, and Bandit filtered touched-scope scan completed successfully. Broad Bandit scan findings were limited to existing test dummy secrets outside changed lines.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the repo-wide pytest collection blockers by making importlib mode apply to default pytest runs and replacing unsafe direct registration of AuthNZ/conftest.py with a safe shared plugin bridge. Added regression coverage for the pytest config/plugin hygiene and verified focused collection plus full repository collect-only now pass.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed or blockers documented.
- [x] #2 Focused regression checks recorded.
- [x] #3 Full project pytest retry recorded.
- [x] #4 Bandit run for touched Python code or documented skip if only config changes.
- [x] #5 Modified files recorded.
<!-- DOD:END -->
