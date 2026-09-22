---
id: TASK-13321
title: >-
  Blob upload-session expiry is designed, schema'd and typed but never
  implemented
status: To Do
assignee: []
created_date: '2026-09-22 04:55'
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
- [ ] #1 A failing test abandons 8 sessions and asserts a 9th upload still succeeds after the expiry window
- [ ] #2 expires_at is set at session creation from a configurable setting
- [ ] #3 The two quota queries exclude expired sessions
- [ ] #4 A sweep transitions created/uploading past expiry to expired and calls discard_upload to release staged chunks
- [ ] #5 reserved_quota_bytes is released when a session expires
- [ ] #6 The notes.py upload flow returns the upload_id and compensates on failure so cancel is reachable
- [ ] #7 Design doc and ADR per CLAUDE.md, since this adds a lifecycle transition
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
