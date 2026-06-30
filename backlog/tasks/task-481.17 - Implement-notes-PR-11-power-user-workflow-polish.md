---
id: TASK-481.17
title: Implement notes PR 11 power-user workflow polish
status: Done
labels:
- notes
- ux
- webui
- power-user
parent_task_id: TASK-481
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 11 from the notes UX remediation plan: improve fast, repeatable note workflows after reliability, accessibility, saving, capture, and import/export hardening. Keep the slice focused on one concrete speed or repetition improvement with shortcut conflict coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `Alt + Shift + D` duplicates the selected note or unsaved draft through the existing duplicate action.
- [x] #2 The duplicate shortcut is ignored while focus is in text inputs or editors.
- [x] #3 Keyboard shortcut help documents the duplicate-note shortcut.
- [x] #4 Focused tests cover duplicate shortcut behavior, shortcut conflict guards, and help-modal discoverability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md#pr-11-power-user-workflow-polish
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a guarded `Alt + Shift + D` global shortcut that duplicates the current note or unsaved draft into a new dirty draft through the existing duplicate action.
- Kept the global shortcut guard so the duplicate shortcut does not fire while typing in text inputs/editors.
- Added duplicate-shortcut discoverability to the keyboard shortcuts modal.
- Verification:
  - Red: `bunx vitest run src/components/Notes/__tests__/NotesManagerPage.stage38.productivity-extensions.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage20.accessibility-shortcuts.test.tsx` failed for missing duplicate shortcut behavior and missing help-modal copy.
  - Green: same command passed 2 files / 9 tests.
  - Focused regression: `bunx vitest run src/components/Notes/__tests__/NotesManagerPage.stage1.editor-reliability.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage16.bulk-actions.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage20.accessibility-shortcuts.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage38.productivity-extensions.test.tsx` passed 4 files / 18 tests.
  - Full Notes sweep: `bunx vitest run src/components/Notes/__tests__` reported 66 files passed and 1 known unrelated failure in `NotesManagerPage.stage10.ai-title.test.tsx` (`LLM (quality)` selector not found), with 206 / 207 tests passing.
- Browser smoke: Needs verification in a running app/backend environment; no dedicated Notes browser-smoke harness was available in this frontend-only worktree.
- Bandit: N/A, no Python/backend files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PR11 power-user workflow polish by adding a documented duplicate-note keyboard shortcut (`Alt + Shift + D`) with conflict coverage and shortcut-help discoverability. Focused tests pass; the full Notes sweep is unchanged except for the known unrelated Stage 10 AI-title selector failure.
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
