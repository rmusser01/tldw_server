---
id: TASK-12933
title: Implement chat document upload WebUI composer decision surface
status: Done
assignee: []
created_date: '2026-07-09 05:15'
updated_date: '2026-07-09 05:29'
labels:
  - implementation
  - chat
  - frontend
  - documents
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 3 from the approved plan: stage document uploads with default add-to-chat decisions, preflight capabilities, render the WebUI Playground document processing choices, show selected mode/status in attachment chips, and add focused component coverage.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document uploads stage with add-to-chat defaults and backend preflight capability metadata
- [x] #2 Composer renders batch/per-file document processing choices with disabled backend reasons
- [x] #3 Attachment chips show document processing mode/status while preserving remove actions
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification evidence:
- RED: focused vitest failed before implementation because DocumentProcessingChoices did not exist, AttachmentsSummary did not render Blocked, and useFileUpload did not stage processing metadata.
- PASS: cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/services/__tests__/chat-document-processing.test.ts ../packages/ui/src/services/__tests__/tldw-api-client.media-ingest.test.ts ../packages/ui/src/components/Option/Playground/__tests__/DocumentProcessingChoices.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/AttachmentsSummary.integration.test.tsx ../packages/ui/src/hooks/chat/__tests__/useFileUpload.document-processing.test.tsx (33 tests).
- PASS: git diff --check.
- PASS: node JSON.parse check for both English playground locale files.
- PASS WITH BASELINE WARNINGS: focused ESLint via apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs exited 0; warnings are existing large-file/no-explicit-any/no-img baseline.
- BASELINE FAIL: cd apps/tldw-frontend && bunx tsc --noEmit --pretty false --project tsconfig.json still fails in existing unrelated AudioStudio, ScheduledTasks, Skills, scheduled-tasks services, mcp-hub, voice-cloning, and e2e files. No touched Task 3 files appear after fixing the setUploadedFiles wiring error.
- Bandit: not applicable; Task 3 touched frontend TypeScript/JSON/docs/task files only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 3 composer decision surface: document uploads now stage with add-to-chat defaults and backend preflight metadata, the Playground composer renders batch/per-file Add/OCR/Ingest choices, attachment chips expose processing mode/status, and English locale keys/tests were added.
<!-- SECTION:FINAL_SUMMARY:END -->
