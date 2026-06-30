---
id: TASK-233.9
title: Support OpenWebUI DB async Chatbooks import jobs
status: Done
assignee: []
created_date: '2026-05-10 23:14'
updated_date: '2026-05-10 23:17'
labels:
  - chatbooks
  - openwebui
  - backend
  - jobs
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-10-openwebui-db-chat-import-implementation-plan.md
  - Docs/superpowers/specs/2026-05-10-openwebui-db-chat-import-design.md
parent_task_id: TASK-233
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 4 async Jobs support for uploaded OpenWebUI database chat imports. Ensure selected_openwebui_user_id survives enqueue and worker execution, DB imports use upload path resolution rather than archive extraction, and cleanup removes uploaded temp DB files after worker completion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Async worker dispatches source_format=openwebui_db to ChatbookService.import_openwebui_db with selected user id.
- [x] #2 OpenWebUI DB job payloads require selected_openwebui_user_id and return a structured openwebui_db_result wrapper.
- [x] #3 DB import jobs use uploaded file path resolution and never ZIP/archive extraction.
- [x] #4 Uploaded temporary DB files are cleaned after successful or failed async worker execution.
- [x] #5 Existing chatbook and OpenWebUI JSON job behavior remains unchanged.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 4 async Jobs worker support for source_format=openwebui_db in tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py. The worker validates selected_openwebui_user_id, resolves uploaded DB files through _resolve_import_upload_path, calls ChatbookService.import_openwebui_db, wraps successful results under openwebui_db_result, marks failures in the Chatbooks import job, and removes the uploaded DB file in the same finally cleanup pattern used by JSON/archive imports.

RED evidence: focused worker tests failed because openwebui_db was rejected as unsupported. Verification: focused worker pytest passed 6 tests; overlapping Chatbooks regression pytest passed 35 tests; Bandit over jobs_worker.py reported 0 findings; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed async Jobs support for OpenWebUI database imports. Uploaded DB jobs now require the selected source user, dispatch through the DB import service instead of archive/JSON handlers, return structured openwebui_db_result data, and clean the uploaded DB file after success or failure while preserving existing chatbook and OpenWebUI JSON job behavior.
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
