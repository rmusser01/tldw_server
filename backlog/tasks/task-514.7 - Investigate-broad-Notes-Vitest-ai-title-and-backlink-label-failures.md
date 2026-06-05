---
id: TASK-514.7
title: Investigate broad Notes Vitest ai-title and backlink-label failures
status: To Do
parent_task_id: TASK-514
documentation:
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage10.ai-title.test.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage26.backlink-labels.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on optional broad Notes/NotesDock Vitest failures found during TASK-514 closeout. The task-focused Notes task-backed todo suite passes, but `bunx vitest run src/components/Notes src/components/Common/NotesDock` fails in `NotesManagerPage.stage10.ai-title.test.tsx` and `NotesManagerPage.stage26.backlink-labels.test.tsx`; rerunning those two files in isolation reproduces 4 failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Determine whether the ai-title strategy dropdown failure is product code, test harness, or AntD interaction drift.
- [ ] #2 Determine whether the backlink label failures are product code, test harness, or fixture drift.
- [ ] #3 Restore the broad Notes/NotesDock Vitest sweep or document any intentional test updates with focused verification.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
