---
id: TASK-523
title: Implement PR 9 Notes import export offline sync
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 21:31'
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
RED/GREEN verification for original PR 9 slice: focused Vitest failed on the new parse-error submit guard, clearer export failed-batch copy, and expanded offline conflict recovery copy, then passed 4 files / 9 tests after implementation. Code review follow-up: import help now explicitly mentions JSON, Markdown, and plain text (.txt), the file input accepts .txt explicitly, and export failed-batch progress copy now goes through the option translation function. Follow-up focused Vitest passed the same 4 files / 9 tests. git diff --check passed. UI package typecheck with NODE_OPTIONS=--max-old-space-size=8192 is blocked by unrelated baseline error in src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx:35, Type 'comfortable' is not assignable to GalleryCardDensity. Bandit skipped because this slice touches only frontend TypeScript tests/components/hooks and Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 9 focused slice complete with code-review follow-up: deterministic import parse errors are prevented before submit, import accepted-file copy matches JSON/Markdown/plain-text input, partial export progress is localized and clearer, and offline conflict copy gives a data-preserving next action.
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
