---
id: TASK-545
title: Implement MCP Unified Stage 3E discovery environment seam cleanup
status: Done
labels:
- mcp
- mcp-unified
- standalone
- stage3
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the remaining direct host testing-helper import from MCP discovery module by using the package-local MCP Unified environment helper for catalog_strict truthy parsing. Keep host-adapter imports unchanged and add focused import-boundary/behavior coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] MCP discovery no longer imports `tldw_Server_API.app.core.testing` directly.
- [x] `catalog_strict` string truthy parsing still reaches the protocol as a boolean value.
- [x] Focused import-boundary and discovery behavior tests cover the seam.
- [x] Verification includes focused pytest, Ruff, Bandit, and whitespace diff checks.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Switched `mcp_discovery_module.py` from the host-level testing helper to `MCP_unified.environment.is_truthy`.
- Added an extraction contract test that forbids the discovery implementation from importing `tldw_Server_API.app.core.testing`.
- Added behavior coverage proving string `catalog_strict` values such as `"yes"` are normalized before dispatch to `MCPProtocol._handle_tools_list`.
- Cleaned touched test typing annotations to satisfy the repo Ruff rules.
- Addressed PR review feedback by typing the new async test's `monkeypatch` fixture and return value, and by replacing unused-argument `del` statements in the fake handler with underscore-prefixed parameter names.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3E removed the remaining non-adapter MCP discovery import of the host testing helper while preserving the catalog strictness parsing behavior. Verification:

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_mcp_discovery_module.py tldw_Server_API/tests/MCP_unified/test_mcp_discovery_sanitization.py -q` -> 31 passed, 5 warnings
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check tldw_Server_API/app/core/MCP_unified/modules/implementations/mcp_discovery_module.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_mcp_discovery_module.py` -> All checks passed
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/MCP_unified/modules/implementations/mcp_discovery_module.py -f json -o /tmp/bandit_mcp_stage3e_discovery_env.json` -> 0 results, 0 errors
- `git diff --check` -> passed

PR review pass: after rebasing on `origin/dev` at `a01c7b7f50`, Qodo and Gemini feedback requested explicit type hints on the new test and idiomatic unused parameters in the fake protocol handler. Both were addressed before force-pushing the rebased branch.

No known blockers or verification skips.
<!-- SECTION:FINAL_SUMMARY:END -->

## Modified Files
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/mcp_discovery_module.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_mcp_discovery_module.py`
- `backlog/tasks/task-545 - Implement-MCP-Unified-Stage-3E-discovery-environment-seam-cleanup.md`

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
