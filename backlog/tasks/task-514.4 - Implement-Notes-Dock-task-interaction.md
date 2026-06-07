---
id: TASK-514.4
title: Implement Notes Dock task interaction
status: Done
parent_task_id: TASK-514
references:
- TASK-512
- TASK-513
- TASK-514
documentation:
- Docs/superpowers/specs/2026-06-05-notes-task-backed-todo-lists-design.md
- Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Common/NotesDock/NotesDockPanel.tsx
- apps/packages/ui/src/store/notes-dock.tsx
- apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add compact task-backed checklist interaction to the Notes Dock, including clean backend toggles, dirty local toggles, refresh behavior, and pending remote change notices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute plan Task 8. Run `apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx` and record verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-05: Added a focused Notes Dock task-backed todo regression file covering clean backend status toggles, dirty local markdown toggles, save conflict preservation, and agent task activity dismissal.
- 2026-06-05: Wired `NotesDockPanel` to load active note tasks and task activity through the shared frontend task API helpers.
- 2026-06-05: Reused `TaskChecklistPreview` in compact mode inside the dock editor. Clean saved notes call `setNoteTaskStatus`, refresh the note snapshot, refresh task state, and invalidate the notes page cache. Dirty notes rewrite local markdown through `toggleChecklistItemMarker` and keep unsaved state.
- 2026-06-05: Added component-local task activity and reconciliation notices. Store changes were not needed because activity can be reloaded from the server on remount and no durable dock-only state is required for this slice.
- 2026-06-05: Verification: `cd apps/packages/ui && bunx vitest run src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx` first failed with missing checkbox/activity UI, then passed with 4 tests.
- 2026-06-05: Verification: `cd apps/packages/ui && bunx vitest run src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx src/components/Common/NotesDock/__tests__/NotesDockPanel.stage4.cache-sync.test.tsx src/components/Common/NotesDock/__tests__/NotesDockPanel.stage1.accessibility.test.tsx src/components/Common/NotesDock/__tests__/NotesDockPanel.stage2.accessibility-regression.test.tsx src/components/Notes/__tests__/TaskChecklistPreview.test.tsx` passed, 5 files / 12 tests.
- 2026-06-05: Verification: `git diff --check HEAD -- apps/packages/ui/src/components/Common/NotesDock/NotesDockPanel.tsx apps/packages/ui/src/store/notes-dock.tsx apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx` passed with no whitespace errors.
- 2026-06-05: Bandit skip: touched code is TypeScript/React test and component code only; no Python files were changed in TASK-514.4.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented compact task-backed checklist interaction in the Notes Dock. The dock now loads active-note tasks and agent task activity, renders the shared checklist preview above the text editor, sends clean saved toggles to the backend status endpoint, keeps dirty toggles local in markdown, preserves dirty drafts on save conflicts, refreshes task/note state after successful saves, and exposes dismissible agent task activity notices without adding store state.
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
