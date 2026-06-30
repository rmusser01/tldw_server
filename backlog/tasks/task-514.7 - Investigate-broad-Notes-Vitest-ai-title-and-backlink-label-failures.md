---
id: TASK-514.7
title: Investigate broad Notes Vitest ai-title and backlink-label failures
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 05:28'
labels: []
dependencies: []
documentation:
  - >-
    apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage10.ai-title.test.tsx
  - >-
    apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage26.backlink-labels.test.tsx
parent_task_id: TASK-514
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on optional broad Notes/NotesDock Vitest failures found during TASK-514 closeout. The task-focused Notes task-backed todo suite passes, but `bunx vitest run src/components/Notes src/components/Common/NotesDock` fails in `NotesManagerPage.stage10.ai-title.test.tsx` and `NotesManagerPage.stage26.backlink-labels.test.tsx`; rerunning those two files in isolation reproduces 4 failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Determine whether the ai-title strategy dropdown failure is product code, test harness, or AntD interaction drift.
- [x] #2 Determine whether the backlink label failures are product code, test harness, or fixture drift.
- [x] #3 Restore the broad Notes/NotesDock Vitest sweep or document any intentional test updates with focused verification.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause investigation:
- The originally recorded ai-title strategy dropdown and backlink-label failures no longer reproduce on current dev; both files pass in isolation and in the broad Notes/NotesDock sweep.
- The current broad sweep exposed one remaining failure in NotesManagerPage.stage48.first-time-ux.test.tsx. The test queried the list empty-state Create note CTA before the async first-time empty-state render had settled; the list region still only contained toolbar actions such as New note.
- Fixed the Stage 48 test harness to wait for the No notes yet empty-state title before querying and clicking the Create note CTA.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified the original TASK-514.7 ai-title and backlink-label failures no longer reproduce on current dev. Fixed the remaining broad Notes/NotesDock sweep failure by stabilizing the Stage 48 first-time UX test around the async list empty-state render. Focused and broad Vitest verification now pass.
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
