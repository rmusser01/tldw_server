---
id: TASK-481.16
title: Implement notes PR 10 import export and offline draft sync
status: Done
labels:
- notes
- ux
- webui
- import-export
- offline
parent_task_id: TASK-481
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 10 from the notes UX remediation plan: make import, export, and offline draft/sync workflows clearer, recoverable, and data-safe. Keep changes scoped to visible frontend behavior unless backend behavior is proven insufficient.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Import files with known client-side parse errors are blocked before submit and keep the import modal open for correction.
- [x] #2 Partial import results surface a warning without conflating parse-preview failures with valid backend submissions.
- [x] #3 Export progress, single-note export/copy, export preflight, print export, and offline drafting sync workflows have focused regression coverage.
- [x] #4 Browser/backend verification gaps are recorded when no live stack is started for this frontend-focused slice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md#pr-10-import-export-and-offline-draft-sync
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PR10 frontend import reliability hardening. The import workflow now blocks files with known client-side parse errors before submitting to `/api/v1/notes/import`, keeps the modal open, and tells the user to fix or remove parse-error files. The partial-results import test now covers valid submissions separately, so parse-preview errors and backend partial results are no longer conflated. Focused PR10 tests passed for export progress, single-note export/copy, export preflight, import workflow, print export, and offline drafting sync (13 tests). Full Notes component sweep was also run: 66/67 files and 205/206 tests passed; the remaining deterministic failure is unrelated to PR10 in `NotesManagerPage.stage10.ai-title.test.tsx` (`LLM (quality)` strategy dropdown option not found). Browser smoke remains needs-verification because no live API/WebUI stack was started. Bandit/backend tests were not applicable because no Python/backend files changed.
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
