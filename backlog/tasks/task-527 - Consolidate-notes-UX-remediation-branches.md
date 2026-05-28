---
id: TASK-527
title: Consolidate notes UX remediation branches
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 02:36'
labels:
  - notes
  - ux
  - integration
  - webui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a single integration branch for the approved /notes UX remediation slices, replaying the existing PR branches in plan order on current origin/dev, resolving conflicts narrowly, and recording focused verification. Scope is branch/history/file integration only; avoid new /notes feature work except conflict resolution required to make the existing slices coexist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A clean integration branch contains the completed /notes remediation slice commits in approved PR order or equivalent resolved content.
- [x] #2 Conflicts are resolved narrowly without reverting unrelated slice behavior.
- [x] #3 Backlog task records the source branches/commits, conflicts, verification commands, and known baseline failures.
- [x] #4 Focused /notes UI tests run on the integrated branch, or any failures are documented with evidence.
- [x] #5 Frontend-only/Python verification requirements are documented, including Bandit skip or run as applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integration branch/worktree:
- Branch: codex/notes-ux-remediation-integrated
- Worktree: .worktrees/notes-ux-remediation-integrated
- Started from origin/dev and merged current origin/dev at 731c365b5 after it advanced, preserving the current MCP/dev files.

Replayed notes remediation commits in plan order:
- PR1 a28546d8b -> ec0812984 Fix notes list reliability states
- PR2 9bd11628c -> e3447901e Fix notes save recovery states
- PR3 df74079e8 -> 25fcc4171 Fix notes first-time create flow
- PR4 e903e152d -> 4ece0f5d0 Fix notes tags terminology
- PR5 4e551ef70 -> 3fa0addf3 Fix notes capture inbox provenance
- PR6A 0f74c30fd -> f541e4d59 Add capture workspace destination picker
- PR6B 6472747f1 -> 2d4837b1a Add notes folder destination picker
- PR7 b65f7bce3 -> 11d056235 Add notes connection link clarity
- PR8 5da8b0a76 -> 3a8f7ef06 Add notes responsive accessibility hardening
- PR9 7b1a4b60e -> 271a385c2 Harden notes import export recovery
- PR9 review fix 806f00b1a -> 18378ddab Address notes recovery review feedback
- PR10 a6f4a2ccf -> 49146b210 Add notes save and new workflow

Conflict resolutions / integration fixes:
- NotesListPanel and NotesSidebar keep both folder-sync controls and the list reliability retry/clear/create-note controls.
- WebClipper save-flow test keeps durable captured provenance while preserving user-entered tags/keywords.
- NotesEditorPane keeps accessible save-status announcements plus save recovery/conflict notices.
- useNotesEditorState and NotesManagerPage keep both reloadSelectedNoteAfterConflict and saveAndStartNew; saveNote now returns Promise<boolean> and keeps the in-flight guard.
- Updated integrated tests for current contracts: AI title option label is AI-powered, note update mocks use the current PUT path/header contract, and list-panel test fixtures pass the required onCreateNote callback.

Verification:
- apps/packages/ui: broad notes/sidepanel/routes/services Vitest matrix passed: 74 files, 260 tests.
- apps/packages/ui: affected NotesListPanel fixture tests passed: 3 files, 11 tests.
- apps/extension: bun run compile passed.
- Backend: ../../.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_note_folders.py tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py -q passed: 23 tests.
- Bandit on touched Python notes scope passed with 0 findings and 0 errors: /tmp/bandit_notes_integration.json.
- UI package typecheck still exits 2 only on unchanged baseline src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx:35 GalleryCardDensity value "comfortable"; that file is not part of the notes integration diff.
- Backend tests generated two untracked watchlist template markdown files; they were not staged.

Post-task-update diff checks passed: git diff --check origin/dev..HEAD and git diff --check.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Consolidated the approved /notes UX remediation slices onto codex/notes-ux-remediation-integrated, rebased onto current origin/dev, resolved cross-slice conflicts narrowly, and added the small test-contract fixes required for the integrated branch. Focused UI, extension, backend, Bandit, and diff verification were recorded; the only remaining typecheck failure is an unrelated pre-existing CharacterListContent design-system test type mismatch.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Verification recorded
- [x] #3 Known blockers/skips documented
- [x] #4 Final summary added
<!-- DOD:END -->
