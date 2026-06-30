---
id: TASK-2392
title: Address PR 2411 smoke harness review comments
status: Done
labels:
- mcp
- review
- smoke-harness
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2411
modified_files:
- mcp_unified/smoke/__init__.py
- mcp_unified/smoke/cli.py
- mcp_unified/smoke/client.py
- mcp_unified/smoke/exceptions.py
- mcp_unified/smoke/scenarios.py
- mcp_unified/smoke/transports.py
- mcp_unified/smoke/types.py
- tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py
- tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py
- backlog/completed/task-2387 - Design-MCP-smoke-client-transport-harness.md
- backlog/tasks/task-2281 - Add-LSP-backed-code-intelligence-MCP-tools.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address still-valid GitHub review comments on PR #2411 for the MCP smoke harness and related Backlog task metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Verify each PR #2411 review finding against current code.
- [x] Fix still-valid smoke harness and Backlog metadata issues with minimal changes.
- [x] Validate the changed smoke code and tests locally.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Kept smoke exceptions centralized inside mcp_unified.smoke.exceptions instead of importing from tldw_Server_API.app.core.exceptions, because the smoke package is part of the standalone mcp_unified package boundary and should not depend on the host app package.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the still-valid PR #2411 smoke harness review comments with focused tests and minimal implementation changes. Verification: focused red/green tests passed, full test_smoke_client.py passed with 73 tests, Ruff passed on touched smoke code/tests, py_compile passed on mcp_unified/smoke modules, Bandit reported no findings for touched smoke scope, and git diff --check passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
