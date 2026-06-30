---
id: TASK-12018
title: Fix PR 2523 coverage-required refresh token test failure
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-26 06:31'
labels:
  - ci
  - tests
  - pr-2523
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce the coverage-required failure locally.
- [x] #2 Update the failing unit test to match the current MCP refresh endpoint contract.
- [x] #3 Run the exact failing test and relevant local verification successfully.
- [x] #4 Push the fix and re-check PR CI for remaining failures.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Completion note: AC #4 was satisfied up to push and initial PR check re-check. Remaining current CI checks were intentionally not waited on further because the user explicitly instructed: "Ignore the current CI checks."
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Original reproduction/verification: exact CI-failing test failed locally with the missing request argument, then passed after the test update. Full tldw_Server_API/tests/unit/test_mcp_unified_error_mapping.py passed 10/10. Bandit was skipped because only test/backlog files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the PR #2523 coverage-required failure cause by updating the direct MCP refresh-token error-mapping unit test to match the current endpoint contract. The test now passes a loopback Request and enables the explicit demo-auth test prerequisites before exercising the sanitized rotation-failure path. Local verification passed for the exact failing test (`1 passed`) and the full error-mapping file (`10 passed`); `git diff --check` was clean. The fix was committed as 417bde9a34 and pushed. Current CI monitoring was stopped per user instruction to ignore the remaining pending checks.
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
