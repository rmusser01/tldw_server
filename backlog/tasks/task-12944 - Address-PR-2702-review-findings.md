---
id: TASK-12944
title: Address PR 2702 review findings
status: Done
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
- [x] #1 Playground document ingestion waits for a submitted ingest job to reach a terminal state before sending or reporting failure.
- [x] #2 Document upload drafts are persisted in a shared SQLite-backed core service with TTL, payload-size, owner, and global quota enforcement.
- [x] #3 Draft tests cover cross-instance visibility and deterministic concurrent quota enforcement without sleeps.
- [x] #4 Snapshot maintenance enforces quotas against hashed session directories.
- [x] #5 Requested module, endpoint, and helper docstrings are present.
- [x] #6 Focused backend and frontend tests, typecheck, Bandit, and diff validation pass.
- [x] #7 All actionable PR #2702 review threads are answered and resolved.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation and PR follow-up complete. Commit 3fee616ae2 pushed to PR #2702. The five Qodo threads auto-resolved against the new head; the Gemini thread received an inline implementation reply and was explicitly resolved. Final GraphQL audit reports all six threads resolved.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved every actionable review finding on PR #2702. Chat document ingestion now polls accepted jobs to a terminal media ID before sending; upload drafts use a shared, transaction-safe SQLite core store with TTL and quotas; snapshot maintenance correctly handles hashed directories; timing-dependent tests were replaced with deterministic cross-instance coverage; and requested docstrings were added. Verified with 18 backend tests, 27 frontend tests, frontend typecheck, Python compilation, git diff checks, and Bandit with zero findings. Commit 3fee616ae2 is pushed, all six review threads are resolved, and no unresolved PR review comments remain.
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
