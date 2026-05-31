---
id: TASK-83
title: Address PR 1321 review comments
status: Done
assignee: []
created_date: '2026-05-05 18:55'
updated_date: '2026-05-05 19:12'
labels: []
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable Qodo and CodeRabbit review comments on PR #1321 for Watchlists Vitest stabilization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Modal.confirm mock requires explicit confirmation before delete assertions
- [x] #2 Output run-jump status-filter assertion proves the run click caused the null reset
- [x] #3 Focused Watchlists review-fix tests pass
- [x] #4 Frontend TypeScript and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify actionable PR #1321 review comments against current code. 2. Patch only the stale test assertions/mocks. 3. Run focused review-fix tests, relevant scale benchmark, TypeScript, and diff checks. 4. Push the PR update and resolve review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified two unresolved review threads: Qodo on auto-confirming Modal.confirm in JobsTab.undo-delete.test.tsx, and CodeRabbit on loose setRunsStatusFilter assertion in OutputsTab.relationship-jumps.test.tsx. Both findings were valid in current PR code.

Fix: Modal.confirm mock now records config without invoking onOk; delete tests explicitly await the captured onOk and assert no delete call happened before confirmation. Output run-jump test now captures setRunsStatusFilter call count before the run click and asserts exactly one additional last call with null.

Verification: focused review-fix Vitest passed 2 files / 4 tests; watchlists-scale-baseline bench passed 1 file / 4 tests; frontend tsc passed; git diff --check passed. Full Watchlists directory reruns exposed unrelated load-sensitive timeout flakes/perf noise outside the modified review comments; the isolated scale gate passed.

PR: https://github.com/rmusser01/tldw_server/pull/1321
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the two actionable PR #1321 review comments. JobsTab delete tests now capture Modal.confirm config and explicitly invoke onOk, with an assertion that deletion has not happened before confirmation. OutputsTab relationship-jump test now captures setRunsStatusFilter call count before the run jump and checks one new last call with null. Verification passed for focused review-fix tests, isolated Watchlists scale benchmark, frontend TypeScript, and git diff --check. Bandit skipped because the changes are frontend tests plus Backlog metadata.
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
