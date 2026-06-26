---
id: TASK-12018
title: Fix PR 2523 coverage-required refresh token test failure
status: In Progress
labels:
- ci
- tests
- pr-2523
priority: High
modified_files:
- tldw_Server_API/tests/unit/test_mcp_unified_error_mapping.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduce the coverage-required failure locally.
- [ ] #2 Update the failing unit test to match the current MCP refresh endpoint contract.
- [ ] #3 Run the exact failing test and relevant local verification successfully.
- [ ] #4 Push the fix and re-check PR CI for remaining failures.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reproduced the CI failure locally with `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/unit/test_mcp_unified_error_mapping.py::test_refresh_token_sanitizes_rotation_failure_log -q` (failed with missing `request`). Updated the direct endpoint test to pass a loopback Request and set the demo-auth env prerequisites required by the route. Verification now passes: exact failing test `1 passed`; full file `10 passed`. `git diff --check` is clean. Bandit skipped because only test/backlog files were changed.
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
