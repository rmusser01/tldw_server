---
id: TASK-12934
title: Implement chat document upload send-time processing and queue replay
status: Done
assignee: []
created_date: '2026-07-09 05:31'
updated_date: '2026-07-09 06:01'
labels:
  - implementation
  - chat
  - frontend
  - documents
dependencies:
  - TASK-12092
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 4 from the approved plan: create visible document-processing turn status, process selected Add/OCR/Ingest files on Send, pass explicit context/media overrides, and preserve replay intent for queued sends.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pressing Send creates and updates a visible document-processing user turn instead of silently waiting
- [x] #2 Selected Add/OCR/Ingest files are processed on Send and pass explicit context/media overrides without stale contextFiles
- [x] #3 Queued sends preserve document-processing source context and replay the same file intent
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented visible document-processing turns, send-time document preparation with explicit context/upload/RAG overrides, reserved-user-message pipeline replay, and queued document replay using stored uploaded file data. Verification: 59 focused Vitest tests passed; git diff --check passed; local ESLint on touched files exited 0 with baseline warnings; TypeScript still fails only known unrelated baseline files outside this task. Bandit skipped because Task 4 touched TS/TSX/JSON/Markdown only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 4 complete: /chat document sends now show processing status, send prepared Add/OCR/Ingest overrides, avoid stale contextFiles for ingest, and replay queued document uploads with their selected processing intent.
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
