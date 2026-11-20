---
id: TASK-15578
title: Preserve typing when Kanban shortcuts are registered
status: Done
references:
- https://github.com/rmusser01/tldw_server/pull/3533
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Kanban cancels matching single-key events before its action checks whether a form field is focused. Apply the dispatcher editable guard before cancellation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Create Board name and description accept ordinary typing, including n and l.
- [x] #2 Kanban shortcuts still work outside editable controls.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reproduced on public dev ae287ab655. The regression covers the real Create Board name and description inputs with Planning board, including l, n, and b. Opening the dialog with b verifies a shortcut still works outside the fields.
Independent PR review confirmed the editable guard runs before event cancellation. Added a real key-event test for n and l outside editable fields to close its only coverage finding. The expanded Kanban, keyboard, and editable-target tests passed 95 checks. The broader local UI run encountered a HistorySelectionReview React hook error that also reproduces on unchanged dev. Bandit does not apply to the TypeScript and Markdown changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Configured all Kanban shortcuts to use the dispatcher's editable guard before preventDefault. Both regression cases failed before the fix and passed after it. The complete Kanban and keyboard test scope passed (76 tests). The matching hosted backport passed 29 Kanban and keyboard tests.
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
