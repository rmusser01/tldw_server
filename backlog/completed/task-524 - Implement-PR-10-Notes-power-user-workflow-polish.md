---
id: TASK-524
title: Implement PR 10 Notes power-user workflow polish
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 01:34'
labels:
  - notes
  - ux
  - webui
  - pr10
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the PR 10 /notes UX remediation slice from the approved plan: speed up repeated note workflows after reliability, saving, capture, import/export/offline, and accessibility slices. Scope is limited to focused /notes power-user affordances and directly related tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fast create is available from /notes without losing current context.
- [x] #2 Search/filter state is predictable and preserved across repeated create/save flows where existing state patterns support it.
- [x] #3 Repeated create/save/tag flows do not require unnecessary pointer movement.
- [x] #4 Any shortcut added does not fire while typing in text inputs/editors.
- [x] #5 Focused tests cover the chosen power-user affordance and shortcut conflicts.
- [x] #6 Browser smoke or equivalent verification records create -> tag -> search/filter -> reopen/edit behavior where feasible.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a desktop Save & new editor action and mobile overflow entry for repeated note capture.
- saveNote now returns a boolean so dependent flows only move context after a confirmed online save, queued offline save, or update. Failed saves keep the draft in place.
- Save & new starts a fresh draft after a successful save, resets the active list mode when needed, closes the mobile sidebar, and focuses the title input for the next capture.
- Added regression coverage for successful Save & new focus/reset and failed Save & new draft preservation. Existing shortcut tests continue to cover Ctrl/Cmd+S scoping.
- Product note: offline new-note queue currently has a single draft:new slot. Browser diagnostics showed offline Save & new preserves the queued draft rather than enabling multiple parallel offline new drafts; changing that requires a separate offline queue model decision.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PR10 power-user polish for /notes. The editor now has a Save & new action for fast repeated capture, with save success/failure represented as a boolean so failed saves preserve the active draft. The action starts a blank draft and focuses the title field after successful online save. Mobile users get the same command in the overflow Create group.

Verification recorded: focused Notes vitest suite passed 19/19; search/filter/tag organization vitest suite passed 12/12; extension compile passed; git diff --check passed. UI tsc with 8GB heap still fails on the existing Characters test GalleryCardDensity baseline at src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx:35. Bandit skipped because this slice touched only frontend TypeScript/TSX and Backlog markdown. Browser smoke was run during this PR10 pass for online Save & new with API config/tour dismissed and observed POST /api/v1/notes/ returning 201 followed by a blank focused draft; offline diagnostics preserved the single queued draft:new entry, which is documented as a product follow-up.
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
