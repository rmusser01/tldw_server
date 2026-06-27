---
id: TASK-12051
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

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the two stale MCP safe-config error-mapping unit tests to expect the current fail-closed HTTP 400 behavior for malformed config query parameters. Verification: reproduced the two failing tests before the edit; after the edit, the two-test CI subset passed, the full test_mcp_unified_error_mapping.py file passed, the existing HTTP invalid safe-config contract test passed, the full backend-required unit smoke command passed with 199 selected tests, git diff --check passed, and Bandit on the touched test file reported only low-severity test-file findings with 0 medium/high findings.
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
