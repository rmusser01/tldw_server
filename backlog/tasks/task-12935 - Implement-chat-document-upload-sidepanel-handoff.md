---
id: TASK-12935
title: Implement chat document upload sidepanel handoff
status: Done
assignee: []
created_date: '2026-07-09 06:03'
updated_date: '2026-07-09 06:23'
labels:
  - implementation
  - chat
  - frontend
  - extension
dependencies:
  - TASK-12092
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 5 from the approved chat document upload processing plan: preserve document-processing choices across browser-extension /chat sidepanel handoff into WebUI /chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sidepanel handoff payload preserves chatDocumentDraftId, ragMediaIds, and fileRetrievalEnabled
- [x] #2 Sidepanel Add/OCR/Ingest upload choices do not silently downgrade and are recoverable on handoff failure
- [x] #3 WebUI /chat imports document handoff fields or surfaces a recoverable expired/failed state
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Task 5 sidepanel/WebUI handoff wiring. Sidepanel document context files now use the shared document-processing choices, backend preflight, send-time document preparation, and server-backed draft creation when continuing attached documents in WebUI. The WebUI handoff payload now preserves chatDocumentDraftId, ragMediaIds, and fileRetrievalEnabled. Playground imports document drafts into uploaded files/context files and deletes the draft after successful import.

Verification: focused red tests failed before implementation; focused and broader Vitest handoff suites now pass. Touched-file ESLint exits 0 with existing warnings only. git diff --check passes. TypeScript still fails only on pre-existing baseline files outside this task. Bandit skipped because this task touched TS/TSX/test/docs only, no Python runtime code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed browser-extension /chat document-processing handoff support for Task 5. Added sidepanel choice/preflight/send wiring, WebUI fragment fields, document draft handoff import/cleanup, and focused contract/regression tests. Known limitation: failed WebUI opens keep the sidepanel draft editable and retryable through the existing Continue-in-WebUI action; there is no separate sidepanel document-draft manager for reusing or manually cancelling unexpired drafts.
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
