---
id: TASK-12095
title: Implement chat document upload frontend processing service
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-09 05:14'
labels:
  - implementation
  - chat
  - frontend
  - documents
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 2 from the approved plan: add UploadedFile document-processing metadata, media client methods/openapi guard paths, pure chat-document-processing service helpers, send-time preparation behavior, and focused Vitest coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Task 2 frontend processing service slice.

Touched files:
- apps/packages/ui/src/db/dexie/types.ts
- apps/packages/ui/src/services/chat-document-processing.ts
- apps/packages/ui/src/services/__tests__/chat-document-processing.test.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.media-ingest.test.ts
- apps/packages/ui/src/services/tldw/domains/media.ts
- apps/packages/ui/src/services/tldw/openapi-guard.ts

Verification:
- RED: bunx vitest run ../packages/ui/src/services/__tests__/chat-document-processing.test.ts failed on missing service/helper exports before implementation.
- PASS: bunx vitest run ../packages/ui/src/services/__tests__/chat-document-processing.test.ts ../packages/ui/src/services/__tests__/tldw-api-client.media-ingest.test.ts -> 25 tests passed.
- PASS: git diff --check and git diff --cached --check.
- PASS with warnings: focused ESLint command exited 0; remaining warnings are existing shared-file any/no-useless-escape baseline.
- KNOWN BASELINE: bunx tsc --noEmit --pretty false --project tsconfig.json still fails in unrelated existing files (AudioStudio TimelineEditor, ScheduledTasks editor/control-plane, Skills Manager, mcp-hub readiness path, voice-cloning ArrayBuffer, e2e fixtures/flashcards); no touched document upload files appear in the error list.
- Formatting note: default Prettier was not retained because it rewrote existing shared-package style across unrelated lines; final formatting verification for this slice is git diff --check.
- Bandit: not applicable; frontend TypeScript only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added frontend document-processing metadata/types, pure chat-document-processing helpers, send-time preparation, ingest/draft cancellation helpers, media client preflight/draft methods, OpenAPI guard paths, and focused Vitest coverage for add-to-chat, OCR, ingest, mixed batches, overflow blocking, async ingest processing state, retries, and cancellation.
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
