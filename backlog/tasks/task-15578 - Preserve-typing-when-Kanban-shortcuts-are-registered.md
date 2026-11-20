---
id: TASK-15578
title: Preserve typing when Kanban shortcuts are registered
status: Done
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Configured all Kanban shortcuts to use the dispatcher's editable guard before preventDefault. Both regression cases failed before the fix and passed after it. The complete Kanban and keyboard test scope passed (76 tests). The matching hosted backport passed 29 Kanban and keyboard tests.
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
