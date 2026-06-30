---
id: TASK-45.31.1
title: Address PR 1444 StatusDot review comments
status: Done
assignee:
  - Codex
created_date: '2026-05-09 20:59'
updated_date: '2026-05-09 21:04'
labels:
  - design-system
  - ui
  - product-state
  - review-fix
dependencies: []
references:
  - apps/packages/ui/src/components/Sidepanel/Chat/StatusDot.tsx
  - >-
    apps/packages/ui/src/components/Sidepanel/Chat/__tests__/StatusBadges.design-system.test.tsx
  - apps/packages/ui/src/hooks/useConnectionState.ts
parent_task_id: TASK-45.31
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 StatusDot maps real connection UX states to specific design-system keys including connected_degraded, error_auth, error_unreachable, setup/configuring, testing, and demo/ready.
- [x] #2 StatusDot tests use realistic uxState and mode values from the connection state model.
- [x] #3 Review-fix verification covers the focused StatusBadges suite, product-state guard tests, design-system verifier, diff check, and touched-file TypeScript filter.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update the StatusDot tests first to use real ConnectionUxState/mode fixtures and assert degraded/auth/unreachable mappings. 2. Watch the focused test fail against the current implementation. 3. Update StatusDot to destructure uxState and map it to canonical design-system keys before deriving Badge severity variants, preserving demo override and retry behavior. 4. Run focused tests plus guard/verifier/diff/type-filter checks, then push the review-fix commit and resolve or reply to the PR threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review fixes: verified the PR threads pointed to real issues. StatusDot now maps uxState directly: connected_degraded -> degraded, error_auth -> auth_required, error_unreachable -> unavailable, testing -> retrying, setup/configuring -> setup_required, and demo/connected_ok -> ready. Test fixtures now use real ConnectionUxState and mode values instead of synthetic failed/full states. Red run before implementation failed 4 tests on degraded/auth/unreachable/retry; green run passed 12/12. Additional verification: product-state guard test passed 49/49; verify:design-system-state passed with 511 baseline exceptions; git diff --check passed; UI tsc still exits 2 on existing repo-wide typing debt with 236 lines and no touched-file diagnostics. Bandit skipped because touched files are TS/TSX and Backlog markdown only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1444 review feedback by switching StatusDot from boolean-collapse mapping to real uxState-based canonical design-system keys, including degraded, auth-required, unavailable, retrying, setup-required, and ready states. Reworked StatusBadges tests to use realistic connection UX fixtures and added coverage for degraded, auth, unreachable, setup/configuring, testing, demo, and retry behavior.
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
