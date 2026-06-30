---
id: TASK-514.3
title: Implement Notes task WebUI client, renderer, and notes page interaction
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
- apps/packages/ui/src/services/notes-tasks.ts
- apps/packages/ui/src/services/tldw/openapi-guard.ts
- apps/packages/ui/src/components/Notes/task-markdown.ts
- apps/packages/ui/src/components/Notes/TaskChecklistPreview.tsx
- apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx
- apps/packages/ui/src/components/Notes/NotesEditorPane.tsx
- apps/packages/ui/src/components/Notes/notes-manager-types.ts
- apps/packages/ui/src/public/_locales/en/option.json
- apps/packages/ui/src/services/__tests__/notes-tasks.test.ts
- apps/packages/ui/src/components/Notes/__tests__/task-markdown.test.ts
- apps/packages/ui/src/components/Notes/__tests__/TaskChecklistPreview.test.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-backed-todos.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add typed WebUI task API helpers, shared task checklist renderer, `/notes` preview/split interaction, local dirty checkbox behavior, conflict copy, and task continuity copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute plan Tasks 6-7. Run focused Vitest suites for notes task services, markdown helpers, renderer, and NotesManagerPage task-backed todos; run OpenAPI guard if paths changed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-06-05: Added typed frontend task API helpers and OpenAPI guard entries for note task CRUD, task status updates, reconciliation, and activity dismissal.
- 2026-06-05: Added shared markdown checklist parsing/toggling helpers plus `TaskChecklistPreview` so clean saved notes can update durable task status and dirty notes can toggle local markdown without backend writes.
- 2026-06-05: Wired `/notes` preview and split modes to load task state, render task-backed checkboxes, surface task conflict/activity/reconciliation/continuity notices, refresh task state after saves, and preserve dirty local checklist edits across save conflicts.
- 2026-06-05: Kept direct edit-mode behavior as raw markdown and added a current-content ref in the editor hook so rapid edit-preview-checkbox interactions preserve unsaved lines while toggling checklist markers.
- 2026-06-05: Verification: `cd apps/packages/ui && bunx vitest run src/services/__tests__/notes-tasks.test.ts src/components/Notes/__tests__/task-markdown.test.ts src/components/Notes/__tests__/TaskChecklistPreview.test.tsx src/components/Notes/__tests__/NotesManagerPage.task-backed-todos.test.tsx` passed, 4 files / 14 tests.
- 2026-06-05: Verification: `cd apps/packages/ui && bunx vitest run src/components/Notes/__tests__/NotesManagerPage.stage2.editor-modes.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage9.stale-version-warning.test.tsx` passed, 2 files / 5 tests.
- 2026-06-05: Verification: `cd apps/packages/ui && bun run verify:openapi` passed, verified 266 client paths and 49 media fallback fields; the 10 reviewed billing/media exceptions are existing guard exceptions.
- 2026-06-05: Verification: `git diff --check HEAD -- <Task 7 frontend files>` passed with no whitespace errors.
- 2026-06-05: Bandit skip: touched implementation is TypeScript/React and locale JSON only for this task slice; no Python code was changed in TASK-514.3 Task 6-7.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the WebUI task checklist frontend slice for notes. The shared client/renderer now exposes typed note task operations and reusable checklist rendering, and the `/notes` page now shows task-backed checkboxes in preview/split modes, routes clean saved toggles through the backend status endpoint, keeps dirty toggles local in markdown, preserves local drafts on save conflicts, and shows task continuity/activity/reconciliation notices.
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
