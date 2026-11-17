---
id: TASK-15256
title: Fix PR 2534 backend-required safe-config unit tests
status: Done
priority: High
ordinal: 12051
references:
- https://github.com/rmusser01/tldw_server/pull/2534
- https://github.com/rmusser01/tldw_server/actions/runs/28281468907
modified_files:
- tldw_Server_API/tests/unit/test_mcp_unified_error_mapping.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update stale unit coverage after PR #2534 made malformed safe-config query parameters fail closed with HTTP 400 instead of continuing request handling. CI backend-required failed two tests in test_mcp_unified_error_mapping.py that still asserted the old behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Renumbered from TASK-12051 to TASK-15256 on 2026-10-31 (TASK-13960): TASK-12051 now names only "Fix PR 1982 current backend CI shard failures" (backlog/tasks/task-12051 - Fix-PR-1982-current-backend-CI-shard-failures.md). Before 2026-10-31 both tasks used TASK-12051; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the two stale MCP safe-config error-mapping unit tests to expect the current fail-closed HTTP 400 behavior for malformed config query parameters. Verification: reproduced the two failing tests before the edit; after the edit, the two-test CI subset passed, the full test_mcp_unified_error_mapping.py file passed, the existing HTTP invalid safe-config contract test passed, the full backend-required unit smoke command passed with 199 selected tests, git diff --check passed, and Bandit on the touched test file reported only low-severity test-file findings with 0 medium/high findings.
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
