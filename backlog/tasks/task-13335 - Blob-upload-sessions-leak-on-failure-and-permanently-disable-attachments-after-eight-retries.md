---
id: TASK-13335
title: >-
  Blob upload sessions leak on failure and permanently disable attachments after
  eight retries
status: Done
assignee: []
created_date: '2026-09-22 04:58'
updated_date: '2026-09-22 23:34'
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
- [x] #1 Failed uploads release their session
- [x] #2 expires_at is set at insert and honoured by the quota count
- [x] #3 Orphaned sessions are reaped by the existing retention pass
- [x] #4 Test drives a mid-upload failure and asserts the next upload succeeds
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in 22b80424f1.

AC1: notes.py:4882 now releases the session in its except clause before surfacing the error. Compensation failure is logged and never masks the original exception. (This path is under app/api/v1/**, which CONTRIBUTING.md:13-18 pauses for third-party PRs; these commits are authored by the repository owner, so the boundary does not apply here.)

AC2: sessions are created with expires_at set from a new SyncV2Settings.blob_upload_session_ttl_seconds (24h default), computed through the service's injectable clock so it stays testable. summarize_blob_quota excludes expired sessions from BOTH reserved_blob_bytes and active_upload_count.

AC3: SyncDatabase.expire_blob_upload_sessions marks timed-out sessions 'expired' (a status the cancel path already recognised as terminal) and is called from the blob-GC leg of retention_compact, gated on apply_blob_gc.

AC4: test_abandoned_upload_sessions_do_not_permanently_disable_attachments drives the whole scenario -- cap reached, next upload refused with "active upload limit", sessions aged out, slots and reserved quota released, next upload succeeds. Plus test_expiry_reaper_leaves_live_and_legacy_sessions_alone. Both red against the pre-fix source.

Two implementation notes worth recording:

1. Expiry is compared in Python, not SQL. expires_at is TEXT on SQLite (schema line 813) and TIMESTAMPTZ on PostgreSQL (line 1399), so a bound ISO string compares differently, or errors, across the two. This is already the file's convention -- the background-lease code at _acquire_background_lease reads the row and compares with _parse_iso_datetime rather than in SQL, and no SQL timestamp comparison existed anywhere in Sync_DB before this change. It cost the SUM/COUNT aggregation in summarize_blob_quota, which now selects the candidate rows instead; the row count is bounded by the active-upload cap, so this is cheap.

2. Rows with expires_at IS NULL are deliberately still counted and never reaped. They predate the TTL, and releasing them would be a silent change of meaning rather than a repair. A deployment wanting them cleared can cancel them explicitly.

3. SyncV2Store proxies each method explicitly rather than delegating via __getattr__, so the retention call needed a proxy added there too. The AC4 test caught this before it could fail at runtime in production.

Scope note on the task description: it states "Retries without an Idempotency-Key mint a fresh uuid4 each time, so each retry creates a NEW session". That is true of the sync API path. The Notes endpoint at notes.py:4895 does pass an idempotency_key derived from request_key, so its retries resume rather than multiply -- but its sessions still leaked on failure, which is what AC1 fixes.

Verification: Sync blob/attachment/retention/store suites (10 files) 2 failed / 510 passed with the change vs 4 / 508 without (stash-isolated); the difference is exactly these two tests and the other two are pre-existing. Notes attachment tests 91 passed. Bandit clean over Sync_DB.py, service.py, store.py and notes.py (run via uvx; bandit is CI-only, not a declared local dependency).

Known skip: the full tests/Sync suite runs about 3 hours in this environment (recorded in synthesis F8) and was not run end to end. The targeted subset covers every file touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Upload sessions now expire, are excluded from the quota once expired, are reaped by the retention pass, and are released when an upload fails. Eight transient failures no longer disable attachments permanently.
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
