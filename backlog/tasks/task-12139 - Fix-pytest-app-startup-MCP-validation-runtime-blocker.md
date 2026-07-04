---
id: TASK-12139
title: Fix pytest app startup MCP validation runtime blocker
status: Done
created_date: 2026-07-04 01:50
labels:
- tests
- pytest
- startup
- mcp
priority: High
references:
- /Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-phase-0-inventory/backlog/tasks/task-12027
  - Implement-Web-Scraping-refactor-Phase-0-import-inventory-and-guardrails.md
- /Users/appledev/Documents/GitHub/tldw_server/.worktrees/web-scraping-phase-0-inventory/backlog/tasks/task-12138
  - Fix-repository-pytest-collection-blockers.md
modified_files:
- tldw_Server_API/app/main.py
- tldw_Server_API/app/services/startup_mcp_validation.py
- tldw_Server_API/app/services/startup_pre_core.py
- tldw_Server_API/tests/Services/test_startup_mcp_validation.py
- tldw_Server_API/tests/Services/test_startup_pre_core.py
updated_date: 2026-07-04 02:09
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the next repo-wide pytest runtime blocker exposed after collection was repaired: app startup under pytest should not run MCP production validation just because app.main was imported before PYTEST_CURRENT_TEST was set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focused regression proves MCP startup validation is skipped in explicit pytest/test mode but still enforced for production-like startup.
- [x] #2 A representative failing Admin app startup test passes.
- [x] #3 The default broad pytest retry outcome is recorded after this blocker is addressed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: after pytest collection was unblocked, app-startup tests reached FastAPI lifespan. The root test conftest sets TEST_MODE=1 during import, but app.main froze _TEST_MODE at import time and required PYTEST_CURRENT_TEST to already be set. During collection/module import PYTEST_CURRENT_TEST is absent, so later lifespan startup passed test_mode=False to the startup sequence even though pytest was executing the test. MCP startup validation then ran production validation and raised RuntimeError: MCP configuration validation failed; refusing to start in production.

Fix:
- Added a test_mode parameter to startup_mcp_validation.validate_startup_mcp_configuration and skip production MCP validation when startup has already established test mode.
- Threaded prepare_startup_pre_core(..., test_mode=...) into _validate_startup_mcp_configuration.
- Added app.main._runtime_test_mode_active() so lifespan startup computes pytest/test mode at runtime instead of relying only on import-time _TEST_MODE.
- Added service-level regression coverage for MCP validation skipping in test mode and for startup_pre_core forwarding test_mode.

Verification:
- RED: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_startup_mcp_validation.py tldw_Server_API/tests/Services/test_startup_pre_core.py -q --tb=short failed before implementation because startup_mcp_validation did not accept test_mode and startup_pre_core did not forward it.
- Focused GREEN: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Services/test_startup_mcp_validation.py tldw_Server_API/tests/Services/test_startup_pre_core.py -q --tb=short passed, 8 passed.
- Representative Admin startup regression: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Admin/test_admin_budgets_endpoint.py::test_admin_list_budgets_returns_canonical_pagination -q --tb=short passed, 1 passed.
- Admin directory: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Admin -q --tb=short passed, 674 passed in 285.32s. Output: /tmp/tldw_admin_after_mcp_startup_fix.txt.
- Broader stop-on-first-failure retry: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q -x --tb=short progressed beyond the former Admin startup failures and stopped at the next unrelated blocker after 1,142 passed and 12 skipped: tldw_Server_API/tests/Agent_Client_Protocol/test_acp_websocket.py::TestACPRunnerClientPermissions::test_determine_permission_tier_batch expected fs.write to be batch but got individual. Output: /tmp/tldw_pytest_after_mcp_startup_fix_x.txt.
- /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m py_compile tldw_Server_API/app/main.py tldw_Server_API/app/services/startup_mcp_validation.py tldw_Server_API/app/services/startup_pre_core.py passed.
- Bandit touched-scope scan with B101 skipped passed with zero findings. Output: /tmp/bandit_pytest_mcp_startup_fix.json.
- git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the pytest app-startup MCP validation blocker by making runtime startup test-mode detection dynamic and passing that test-mode signal into MCP validation. The former Admin startup failures now pass, and the broader stop-on-first-failure retry advances to the next separate ACP permission-tier assertion.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Root cause documented in the task notes.
- [x] #2 Focused regression and representative pytest command recorded.
- [x] #3 Bandit run for touched Python code recorded.
- [x] #4 Modified files recorded.
- [x] #5 Final summary added.
<!-- DOD:END -->
