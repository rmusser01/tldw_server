---
id: TASK-12930
title: Stabilize route-focused characters harness timeouts
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-09 03:33'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current-head PR #2692 UI Package Characters Harness failed because two slow route-focused character onboarding tests retained 10s timeout caps while neighboring route-focus tests already use 30s. Align the two failing tests with the existing 30s harness timeout pattern.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The two current-head failing route-focused character onboarding tests use the same 30s timeout pattern as adjacent slow route-focus tests.
- [x] #2 The exact failing test names pass locally.
- [x] #3 The full UI package characters harness command passes locally.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: two route-focused onboarding tests still had explicit 10s timeout caps. CI showed both completed too slowly under the full harness load, while adjacent route-focus tests already use 30s. Aligned only those two timeout values to 30s.

Verification: exact failing route-focused tests passed locally; full bun run test:characters-harness -- --maxWorkers=1 --no-file-parallelism passed. Bandit not applicable because this change only adjusts TypeScript test timeout metadata and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned the two failing route-focused character onboarding tests with the existing 30s harness timeout pattern. Verified the exact failing test names and the full characters harness locally.
<!-- SECTION:FINAL_SUMMARY:END -->

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
