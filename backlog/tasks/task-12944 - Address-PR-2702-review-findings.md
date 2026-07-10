---
id: TASK-12944
title: Address PR 2702 review findings
status: In Progress
references:
- https://github.com/rmusser01/tldw_server/pull/2702
modified_files:
- apps/packages/ui/src/services/chat-document-processing.ts
- apps/packages/ui/src/services/__tests__/chat-document-processing.test.ts
- tldw_Server_API/app/api/v1/endpoints/media/document_upload_processing.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/document_upload_drafts.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/document_upload_preflight.py
- tldw_Server_API/app/core/Sandbox/snapshots.py
- tldw_Server_API/tests/Media/test_document_upload_processing.py
- tldw_Server_API/tests/Media/test_document_upload_draft_store.py
- tldw_Server_API/tests/sandbox/test_snapshot_quota_enforcement.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve all actionable review feedback on release PR #2702: wait for asynchronous chat document ingest completion, move upload draft persistence and quotas into a shared core service, make concurrency tests deterministic, fix hashed snapshot-directory quota maintenance, and add requested docstrings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Playground document ingestion waits for a submitted ingest job to reach a terminal state before sending or reporting failure.
- [ ] #2 Document upload drafts are persisted in a shared SQLite-backed core service with TTL, payload-size, owner, and global quota enforcement.
- [ ] #3 Draft tests cover cross-instance visibility and deterministic concurrent quota enforcement without sleeps.
- [ ] #4 Snapshot maintenance enforces quotas against hashed session directories.
- [ ] #5 Requested module, endpoint, and helper docstrings are present.
- [ ] #6 Focused backend and frontend tests, typecheck, Bandit, and diff validation pass.
- [ ] #7 All actionable PR #2702 review threads are answered and resolved.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation complete locally. Added ingest-job polling before chat send; SQLite-backed draft persistence with TTL, payload limits, atomic owner/global quotas, and cross-instance visibility; direct quota enforcement for discovered hashed snapshot directories; deterministic Barrier-based concurrency coverage; and requested docstrings. Verification: 18 backend tests passed, 27 frontend tests passed, frontend typecheck and Python compilation passed, git diff --check passed, and Bandit reported 0 findings across 1,070 lines. Pending: push the commit and close all six review threads.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
