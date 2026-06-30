---
id: TASK-483
title: Implement PR 2 Notes save state and error recovery
status: Done
labels:
- notes
- ux
- webui
- pr2
modified_files:
- apps/packages/ui/src/components/Notes/NotesEditorHeader.tsx
- apps/packages/ui/src/components/Notes/NotesEditorPane.tsx
- apps/packages/ui/src/components/Notes/NotesManagerPage.tsx
- apps/packages/ui/src/components/Notes/NotesSaveStatus.tsx
- apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx
- apps/packages/ui/src/components/Notes/notes-manager-types.ts
- apps/packages/ui/src/components/Notes/notes-manager-utils.ts
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage47.save-state-recovery.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the second /notes UX remediation slice from the approved notes plan: make note dirty, saving, saved, failed, and conflict states visible and recoverable. Scope is limited to /notes editor save-state and error-recovery behavior. Preserve the existing autosave/navigation policy unless clarified by tests. Approved plan reference in source checkout: Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md, PR 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dirty editor state enables save and is represented by visible, accessible status text.
- [x] #2 Saving state prevents duplicate saves and announces progress.
- [x] #3 Failed save keeps unsaved edits visible and recoverable.
- [x] #4 Successful save clears dirty state and updates visible modified/version state.
- [x] #5 Conflict/version errors produce a clear next action.
- [x] #6 Navigation away from dirty notes follows one consistent clarified policy without replacing the autosave model.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented PR 2 save-state and recovery slice.

Changes:
- Added persistent save recovery state for generic save failures and version conflicts.
- Added accessible save feedback live region for dirty, saving, failed, and saved states.
- Prioritized saving/error over dirty in the header status badge so failed saves expose retry.
- Added an in-flight save ref and disabled save button while saving to prevent duplicate saves.
- Added conflict recovery notice with explicit reload-server-version action guarded by confirmation.

Verification:
- RED: new stage47 tests initially failed on missing save feedback, retry, and conflict reload affordance.
- GREEN: bunx vitest run src/components/Notes/__tests__/NotesManagerPage.stage47.save-state-recovery.test.tsx
- GREEN: bunx vitest run src/components/Notes/__tests__/NotesManagerPage.stage1.editor-reliability.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage9.stale-version-warning.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage34.keyword-partial-save-warning.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage41.offline-drafting-sync.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage47.save-state-recovery.test.tsx
- GREEN: git diff --check
- Typecheck attempted with local TypeScript binary; package-wide baseline has unrelated pre-existing TypeScript failures outside Notes and no Notes-matching errors in filtered output.
- Browser smoke blocked: no existing localhost server; Next Turbopack rejected the ignored apps/node_modules symlink, webpack fallback attempted dependency install and was stopped. Symlink restored afterward and Vitest re-verified.
- Bandit skipped: touched implementation is TS/TSX only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 2 Notes save-state and error-recovery behavior is implemented. Users now get accessible dirty/saving/saved/error feedback, duplicate saves are blocked while a save is pending, failed saves preserve editable draft content and expose retry, and version conflicts show a persistent recovery notice with a reload-server-version action. Existing autosave/navigation policy was preserved.
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
