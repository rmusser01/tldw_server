---
id: TASK-530.12
title: Implement Skills export metadata feedback
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-06-29 01:15'
labels:
  - skills
  - webui
  - safe-operations
  - frontend
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-29-skills-export-metadata-feedback-design.md
  - Docs/superpowers/plans/2026-06-29-skills-export-metadata-feedback.md
parent_task_id: TASK-530
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 Safe Operations after TASK-530.11 by preserving Skills export response metadata through the frontend client and using the server-provided filename for downloads and success feedback. Keep scope limited to export metadata, filename fallback/safety, and user feedback; do not add bulk export or permission/model metadata panels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Skills export API client returns both Blob and filename metadata parsed from Content-Disposition when available.
- [x] #2 Skills export API client falls back to a safe `<skill>.zip` filename when the header is missing, malformed, or unsafe.
- [x] #3 Skills manager uses the returned filename for the browser download and shows success feedback naming the actual file.
- [x] #4 Existing sanitized export failure feedback remains unchanged.
- [x] #5 Focused frontend tests cover metadata filename, fallback filename, success feedback, and error feedback.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Updated `workspaceApiMethods.exportSkill()` to preserve binary response metadata by requesting the full response wrapper and returning `{ blob, filename }`.
- Added safe filename resolution from `Content-Disposition`, preferring RFC 5987 `filename*` over plain `filename`, with fallback to a safe `<skill>.zip` name.
- Added a missing-payload guard so an otherwise successful export response without binary data reports an error instead of starting an empty download.
- Updated the Skills manager export flow to use the returned filename for the browser download and to show success feedback naming the actual file.
- Preserved sanitized export failure feedback and added regression coverage for the sanitized failure path.

## Modified Files
- `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- `apps/packages/ui/src/services/tldw/domains/__tests__/workspace-api.skills.test.ts`
- `apps/packages/ui/src/components/Option/Skills/Manager.tsx`
- `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
- `Docs/superpowers/plans/2026-06-29-skills-export-metadata-feedback.md`

## Verification
- `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.skills.test.ts --reporter=dot` - RED for missing-payload regression before guard: 1 failed, 13 passed.
- `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.skills.test.ts --reporter=dot` - GREEN after guard: 1 file passed, 14 tests passed.
- `bunx vitest run src/services/tldw/domains/__tests__/workspace-api.skills.test.ts src/components/Option/Skills/__tests__/Manager.test.tsx --reporter=dot` - 2 files passed, 50 tests passed.
- `git diff --check` - passed.

## Known Skips Or Blockers
- Bandit skipped: touched code is frontend TypeScript/TSX and Markdown only.
- Full package typecheck/build skipped: focused Vitest coverage targets the changed Skills export contract and manager workflow.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Skills export metadata feedback. The frontend API now preserves server filename metadata for export downloads, safely falls back when metadata is absent or unsafe, rejects missing binary payloads, and the Skills manager now reports successful export starts with the actual download filename while keeping sanitized error feedback intact.
<!-- SECTION:FINAL_SUMMARY:END -->

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
