---
id: TASK-45.21.1
title: Address PR 1394 Evaluations StatusBadge review comments
status: Done
assignee: []
created_date: '2026-05-09 03:48'
updated_date: '2026-05-09 03:52'
labels:
  - design-system
  - webui
  - guard
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1394'
  - >-
    apps/packages/ui/src/components/Option/Evaluations/components/__tests__/StatusBadge.design-system.test.tsx
parent_task_id: TASK-45.21
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the open PR 1394 review feedback for the Evaluations StatusBadge design-system adapter without broadening the implementation scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unknown run-status fallback behavior is covered by focused StatusBadge tests and still renders through the shared Badge primitive.
- [x] #2 Focused tests and design-system guard verification are rerun after the review fix.
- [x] #3 Prototype-key statuses such as toString fall back safely instead of crashing.
- [x] #4 The running spinner regression test uses a stable test hook instead of a Tailwind class selector.
- [x] #5 StatusBadge does not add contradictory screen-reader-only state labels beside the visible run status.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added PR review regression coverage for unknown and hostile Evaluations run statuses. The tests now verify arbitrary unknown text stays visible, prototype-property keys such as constructor fall back without crashing, running spinner coverage uses a stable test id instead of the animate-spin class, and canonical state labels are not appended as extra hidden badge copy. Production changes use an own-property guard for STATUS_CONFIG lookups, keep the running icon aria-hidden with a stable test hook, and remove the additive srLabel from this adapter so the visible run status remains the accessible label. Verification passed: bunx vitest run src/components/Option/Evaluations/components/__tests__/StatusBadge.design-system.test.tsx --reporter=dot (9 tests); bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot (46 tests); bun run verify:design-system-state; git diff --check. bunx tsc --noEmit --pretty false was rerun and still fails on existing unrelated package-wide TypeScript errors; no reported errors are from the touched Evaluations StatusBadge files. Bandit was not run because this review fix only touches TS/TSX test code and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all PR 1394 review comments by adding unknown/prototype-key fallback coverage, hardening StatusBadge config lookup, removing the additive srLabel mismatch, and replacing the spinner class assertion with a stable test hook. Focused adapter tests, product-state guard tests, design-system verifier, and diff checks pass; broad tsc remains blocked by unrelated existing package-wide errors.
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
