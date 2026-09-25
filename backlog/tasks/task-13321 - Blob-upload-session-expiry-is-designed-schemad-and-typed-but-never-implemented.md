---
id: TASK-13321
title: >-
  Blob upload-session expiry is designed, schema'd and typed but never
  implemented
status: Done
assignee: []
created_date: '2026-09-22 04:55'
updated_date: '2026-09-23 23:14'
labels:
  - bug
  - sync
  - reliability
dependencies: []
references:
  - 'tldw_Server_API/app/core/Sync/v2/service.py:844'
  - 'tldw_Server_API/app/core/Sync/v2/models.py:1994'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sync blob upload sessions have a designed expiry lifecycle that no code path implements.

Verified:
- **`expires_at` is set at zero sites.** The column exists (`Sync_DB.py:813`) and is written verbatim from `session.expires_at` (`:11901`), but `SyncBlobUploadSessionCreate.expires_at` defaults to `None` (`v2/models.py:1994`) and `create_blob_upload_session` never passes it.
- **Status `"expired"` is written at zero sites.** It is declared in the `SyncBlobUploadStatus` Literal (`v2/models.py:92-99`) and read as terminal at exactly one place (`Sync_DB.py:11957`).
- **No reaper exists.** The only writers of a terminal status are `complete_blob_upload` and `cancel_blob_upload`, both client-driven.
- **The quota query counts abandoned sessions forever** — `summarize_blob_quota` sums `WHERE status IN ('created','uploading')` with no expiry predicate (`Sync_DB.py:12604`, `:12628`).

**Effect:** `max_active_blob_uploads` defaults to **8** (`service.py:844`). A client that starts 8 uploads and then crashes or loses network — ordinary transient failure, not abuse — permanently exhausts its own limit. `_validate_blob_limits` raises "Sync blob active upload limit exceeded" on every subsequent attachment upload for that user+dataset, forever. Recovery requires the client remembering all 8 `upload_id`s to cancel them, or a manual DB edit.

Compounding: each abandoned session's `reserved_quota_bytes` counts against `user_blob_quota_bytes` permanently, so abandoning one 1 GB upload permanently costs 1 GB of quota. Staged chunk files written by `blob_store.write_upload_chunk` are never discarded either — only the complete and cancel paths call `discard_upload`.

**This was explicitly in scope, not deferred.** `Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md` states ":82 release reservations on cancel/expiry", ":240 M2 should enforce ... Upload-session expiration", and scopes out blob GC only "beyond ... abandoned upload cleanup". The schema column, the Literal state and the quota query's status filter were all built for it; only the writer is missing.

Related: `api/v1/endpoints/notes.py:4883-4954` has no `finally` and no compensating `cancel_blob_upload`, and never returns the `upload_id` — so the cancel endpoint is unreachable for that flow.

Found by the comprehensive core-module review; the absent writer, dead state and unfiltered quota gate independently verified by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing test abandons 8 sessions and asserts a 9th upload still succeeds after the expiry window
- [x] #2 expires_at is set at session creation from a configurable setting
- [x] #3 The two quota queries exclude expired sessions
- [x] #4 A sweep transitions created/uploading past expiry to expired and calls discard_upload to release staged chunks
- [x] #5 reserved_quota_bytes is released when a session expires
- [x] #6 Design doc and ADR per CLAUDE.md, since this adds a lifecycle transition
- [x] #7 The notes.py upload flow compensates on failure by cancelling its own session (returning the upload_id is unnecessary: the flow is a single server-driven request, so the client never needs to cancel it, and a mid-request crash is covered by expiry)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-23 verification: most of this was already implemented by TASK-13335 in 22b80424f1. That commit sets expires_at at insert from SyncV2Settings.blob_upload_session_ttl_seconds, makes summarize_blob_quota's two queries (per-user and per-dataset) skip expired rows for both reserved bytes and active count, adds SyncDatabase.expire_blob_upload_sessions called from the retention_compact apply_blob_gc leg, and adds compensation in notes.py. Checked each AC against the code and found three gaps, all closed in 4c435ef69b. (a) AC4: the sweep never called discard_upload, so staged chunks stayed on disk. expire_blob_upload_sessions now returns the upload_ids it actually transitioned (rowcount==1, so a session completed after the SELECT is left alone). retention_compact discards their chunks and logs OSError/SyncBlobStoreError by class. (b) AC2: the TTL setting was not configurable at deploy time. The factory now reads SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS (positive int, default 86400), like its sibling settings. (c) AC7: the notes.py compensation had no test. AC6 amended: returning the upload_id to the client adds nothing. The Notes upload is one server-driven request that cancels its own session on failure, and expiry covers a crash mid-request. Tests, each red before and green after: tests/Sync/test_sync_v2_workspace_blobs.py::test_retention_sweep_expires_abandoned_sessions_and_discards_staged_chunks uses the default cap of 8; before, it failed only on the staged-chunk assertion. tests/Sync/test_sync_v2_factory.py (env read + 2 invalid-value cases) had 3 failures before. tests/Notes/test_notes_attachment_sync_api.py::test_active_one_shot_upload_releases_its_session_when_a_chunk_fails, with the compensation disabled, left the session 'created' instead of 'cancelled'. The existing expiry tests were updated for the list return type. Sync blob/attachment/retention/store/factory suites + Notes attachment/API suites: 558 passed, 0 failed. Bandit (-ll) on Sync_DB.py, factory.py, service.py, store.py: no issues. Docs: new ADR-052 (+ index), env var and expiry semantics added to Docs/API/Sync_V2_M2.md, design doc marks upload-session expiry implemented.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Blob upload-session expiry is now complete end to end. From 22b80424f1 (TASK-13335): expires_at is set at creation, quota and the active-upload cap ignore expired sessions at read time, the retention pass reaps them, and the Notes upload cancels its own session on failure. New in 4c435ef69b: the reaper discards staged chunk files, SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS configures the TTL, and new regression tests cover the default cap of 8, the TTL env var and Notes compensation. ADR-052 records the design. Known limits (by design, in ADR-052): the chunk-file sweep runs only when a retention_compact pass runs for the dataset. Slot and quota release do not depend on that pass. Pre-TTL rows with NULL expires_at are still counted until cancelled. The Postgres paths were not exercised locally.
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
