---
id: TASK-482
title: Implement PR 1 Notes list reliability
status: Done
labels:
- notes
- ux
- webui
- pr1
modified_files:
- apps/packages/ui/src/components/Notes/NotesListPanel.tsx
- apps/packages/ui/src/components/Notes/NotesListPanelEmptyStates.tsx
- apps/packages/ui/src/components/Notes/NotesManagerPage.tsx
- apps/packages/ui/src/components/Notes/NotesSidebar.tsx
- apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx
- apps/packages/ui/src/components/Notes/hooks/useNotesListManagement.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesListPanel.stage46.list-states.test.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage46.list-reliability.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first /notes UX remediation slice from the approved notes plan: make the notes list trustworthy after create, delete, restore, filter, and search changes. Scope is limited to /notes list state, selected-state handling, and distinct loading/empty/search/error states. No sidebar default/customization changes. Approved plan reference in source checkout: Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md, TASK-481.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Deleted notes disappear from active/recent lists without requiring a page refresh.
- [x] #2 Restored notes reappear in the expected active list state.
- [x] #3 Creating a note updates the visible list immediately.
- [x] #4 Loading, empty library, empty search result, and error states are visually and semantically distinct.
- [x] #5 Search/filter state does not display stale notes after deletion or restore.
- [x] #6 Sidebar default/customization behavior is untouched.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented as a frontend-only PR 1 slice:

- Added recent-note cache pruning for single-note delete and bulk delete success paths.
- Exposed list query error state from `useNotesListManagement` and passed retry/error/no-results state through `NotesSidebar` to `NotesListPanel`.
- Split list states into accessible loading, first-time empty library, active-filter no-results, and retryable load error states.
- Added RED/GREEN coverage for single-delete and bulk-delete removal from Recent notes.
- Added list-panel state coverage for loading, empty library, no-results, and load error.

Verification:

- RED: `NotesManagerPage.stage46.list-reliability.test.tsx` failed before `removeRecentNotes` was added.
- RED: `NotesListPanel.stage46.list-states.test.tsx` failed before loading/no-results/error state handling was added.
- GREEN: `./node_modules/.bin/vitest run src/components/Notes/__tests__/NotesListPanel.stage46.list-states.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage46.list-reliability.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage1.editor-reliability.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage8.trash-restore.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage11.search-filtering.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage12.recent-notes.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage16.bulk-actions.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage32.delete-undo.test.tsx --maxWorkers=1 --no-file-parallelism` passed: 8 files, 25 tests.
- `git diff --check` passed.
- `./node_modules/.bin/tsc -p tsconfig.json --noEmit --pretty false` still fails on unrelated baseline errors outside the touched Notes files.
- Browser verification was attempted with a temporary Next dev server in advanced and quickstart modes at `http://127.0.0.1:18016/notes`, but the local app redirected to `/login`/`/settings/tldw` before the Notes workspace could render. Full browser workflow verification remains environment-limited for this slice.
- Bandit skipped: touched scope is TS/TSX only.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 1 /notes list reliability slice completed in the isolated worktree. The visible list and Recent notes no longer retain deleted notes, bulk deletes clean recent state, restored/create/search behavior is covered by existing focused Notes tests, and list empty/loading/no-results/error states are distinct. Verification: RED tests failed as expected before implementation; focused Vitest suite passed: 8 files, 25 tests. git diff --check passed. Package-wide tsc still fails on unrelated baseline errors outside the touched Notes files. Browser verification was attempted with a temporary Next dev server in advanced and quickstart modes, desktop and mobile, but the local app redirected /notes to /login/settings before the Notes workspace could render, so full browser workflow verification remains environment-limited.
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
