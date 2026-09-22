---
id: TASK-13335
title: >-
  Blob upload sessions leak on failure and permanently disable attachments after
  eight retries
status: To Do
assignee: []
created_date: '2026-09-22 04:58'
labels:
  - bug
  - sync
  - notes
dependencies: []
references:
  - 'tldw_Server_API/app/api/v1/endpoints/notes.py:4883'
  - 'tldw_Server_API/app/core/Sync/v2/service.py:10990'
  - 'tldw_Server_API/app/core/DB_Management/Sync_DB.py:12604'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The upload sequence (create_blob_upload_session -> N x upload_blob_chunk -> complete_blob_upload -> coordinator.capture) is wrapped in one try with NO finally, no cancel_blob_upload and no compensation.

Sessions are capped at max_active_blob_uploads = 8. summarize_blob_quota counts WHERE status IN (created, uploading) with NO expiry predicate. expires_at defaults None, is never set at insert, and appears in no WHERE clause repo-wide. No reaper exists. The upload_id is created server-side and never returned to the client, so the cancel endpoint is unreachable for it. Retries without an Idempotency-Key mint a fresh uuid4 each time, so each retry creates a NEW session rather than resuming.

Result: eight ordinary transient failures (flaky network on a large attachment suffices) PERMANENTLY disable attachment upload for that user, through this endpoint and the sync API, with no self-service recovery. Each orphan also holds reserved_quota_bytes.

notes.py half is owner-only; the expiry half is not.

Source: synthesis F35
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Failed uploads release their session
- [ ] #2 expires_at is set at insert and honoured by the quota count
- [ ] #3 Orphaned sessions are reaped by the existing retention pass
- [ ] #4 Test drives a mid-upload failure and asserts the next upload succeeds
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
