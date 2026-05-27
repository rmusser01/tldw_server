---
id: TASK-523
title: Implement PR 9 Notes import export offline sync
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 20:29'
labels:
  - notes
  - ux
  - webui
  - pr9
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the PR 9 /notes UX remediation slice from the approved plan: make exposed import, export, and offline draft/sync workflows understandable, reliable, and recoverable. Scope is limited to /notes import/export/offline workflow UI and directly related tests unless backend behavior proves broken.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import flow explains accepted file types, duplicate handling, success, partial success, and failure.
- [x] #2 Export flow communicates scope, format, progress, partial failure, and download/print result.
- [x] #3 Offline draft status is visible without being noisy and distinguishes queued, syncing, synced, error, and conflict states.
- [x] #4 Offline drafts recover after reload and sync when online without overwriting newer remote content silently.
- [x] #5 Import/export/offline errors preserve user data and provide a clear next action.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED focused Vitest failed on the new parse-error submit guard, clearer export failed-batch copy, and expanded offline conflict recovery copy. GREEN focused Vitest passed: src/components/Notes/__tests__/NotesManagerPage.stage36.import-workflow.test.tsx, NotesManagerPage.stage30.export-progress.test.tsx, NotesListPanel.stage46.export-progress-copy.test.tsx, NotesManagerPage.stage41.offline-drafting-sync.test.tsx (4 files / 9 tests). git diff --check passed before staging. UI package typecheck with NODE_OPTIONS=--max-old-space-size=8192 is blocked by unrelated baseline error in src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx:35, Type 'comfortable' is not assignable to GalleryCardDensity. Bandit skipped because this slice touches only frontend TypeScript tests/components/hooks and Backlog task metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 9 focused slice complete: deterministic import parse errors are prevented before submit, partial export progress is clearer, and offline conflict copy gives a data-preserving next action.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed or explicitly split with rationale recorded.
- [x] #2 Focused tests or verification recorded.
- [x] #3 Documentation updated when relevant.
- [x] #4 Bandit run for touched Python scope when applicable or frontend-only skip documented.
- [x] #5 Final summary added.
- [x] #6 Known skips or blockers documented.
<!-- DOD:END -->
