# Stage 4 — Blob transfer and retention/GC: the data-source boundary

## Scope

The two Sync surfaces that own resources outside the envelope log: the M2 blob upload/download
lifecycle (sessions, chunks, quota, storage namespaces) and the retention/GC pass (envelope
compaction, tombstone prune, binding release, blob GC). Sequential coupling (Axis 3) and
efficiency (Axis 5) are the productive axes here.

## Code Paths Reviewed

Blob lifecycle:
- `v2/service.py:create_blob_upload_session (6099-6170)` — quota gate at :6130,
  `SyncBlobUploadSessionCreate(...)` construction at :6153-6169
- `v2/service.py:upload_blob_chunk (6328-6392)`, `complete_blob_upload (6394-6474)`,
  `get_blob_upload_session (6311-6326)`, `cancel_blob_upload (6520-6537)`,
  `require_completed_notes_attachment_upload (6476-6518)`
- `v2/service.py:_validate_blob_limits (10967-10991)` — the active-upload cap at :10990-10991
- `v2/service.py:store_attachment (6019-6097)`, `blob_download_manifest (6539-6601)`,
  `iter_blob_bytes (6603-6631)`, `_normalize_download_chunk_size (10907-10913)`
- `v2/models.py:SyncBlobUploadSessionCreate (1975-1995)` — `expires_at: str | None = None`
- `core/DB_Management/Sync_DB.py:create_blob_upload_session` insert (11875-11910) — persists
  `session.expires_at`
- `core/DB_Management/Sync_DB.py:summarize_blob_quota (12592-12640)` — the active-upload count at
  :12604-12609
- `v2/blob_store.py:LocalSyncBlobStore` — lock wait loop at :601-614
- Callers of the lifecycle: `api/v1/endpoints/sync.py:2102, 2177, 2223, 2262` (the full protocol,
  including cancel) and `api/v1/endpoints/notes.py:4883-4954` (create → chunks → complete →
  capture, no cancel)
- `api/v1/endpoints/notes.py:_optional_attachment_idempotency_key (695-703)`

Retention:
- `v2/service.py:retention_compact (3188-3336)`, `retention_dry_run (3338-3497)`
- `v2/service.py:_apply_retention_domain_compactions (8471-8618)`,
  `_apply_retention_binding_releases (8707-8778)`, `_apply_retention_blob_gc (8780-8917)`
- `v2/service.py:_retention_latest_envelopes_by_object (8938-8948)`,
  `_retention_blob_candidates (9011-9044)`, `_retention_blob_candidate (9239-9368)`
- `v2/service.py:diagnostics (3101-3186)`, `_attachment_lifecycle_diagnostics (7859-8407)`
- `v2/service.py:SyncV2Settings.restore_manifest_scan_limit (864)` = 10_000;
  `max_envelope_payload_bytes (839)` = 262_144; `max_active_blob_uploads (844)` = 8
- Scan-limit consumers: `:3130` (diagnostics), `:3359` (retention), `:6010` (restore),
  `:7976` (attachment diagnostics, capped at min(limit, 1_000)), `:10610` (profile manager),
  `:10782` (restore mutation groups), `:11039` (latest cursor for domains)
- Endpoints: `api/v1/endpoints/sync.py:get_sync_v2_diagnostics (1412-1436)`,
  `dry_run_sync_v2_retention (1444-1469)`, `compact_sync_v2_retention (1476-1505)` — all `def`
  (thread-pool) handlers, so no event-loop blocking

ADRs consulted: `Docs/ADR/038-canonical-notes-attachment-registry-and-blob-lifecycle.md` (the
attachment registry and blob lifecycle) and `Docs/ADR/034` (retention's interaction with the
dataset projection fence). Neither mandates the current session-expiry behaviour; ADR-038
describes registry and lifecycle states, not the orphan-session policy.

## Tests Reviewed

- `tests/Sync/test_sync_v2_blob_store.py` — `LocalSyncBlobStore` storage semantics. Downgrades
  risk on the filesystem side, not on session accounting.
- `tests/Sync/test_sync_v2_workspace_blobs.py` — workspace-scoped blob access.
- `tests/Sync/test_sync_v2_retention.py` (2,215 LOC) — retention candidate classification,
  blockers, revalidation. Good behavioural coverage; asserts nothing about scan volume.
- `tests/Sync/test_sync_v2_diagnostics.py` — the diagnostics report shape.
- `tests/Sync/test_sync_v2_attachment_refs.py`, `..._attachment_materializer.py`,
  `..._notes_attachment_bootstrap.py`, `..._notes_attachment_coordinator.py` — the attachment
  registry.
- `tests/Notes/test_notes_attachment_sync_api.py` and `tests/e2e/test_notes_attachment_sync_v2.py`
  cover the `api/v1/endpoints/notes.py` compatibility upload path on its **success** path. Neither
  drives a failure between `create_blob_upload_session` and `complete_blob_upload`, which is why
  `sync-2` is not caught.
- None of these run against PostgreSQL.

## Validation Commands

```
$ python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_store.py \
      tldw_Server_API/tests/Sync/test_sync_v2_retention.py \
      tldw_Server_API/tests/Sync/test_sync_v2_blob_store.py -q -p no:randomly
FAILED tldw_Server_API/tests/Sync/test_sync_v2_store.py::test_postgres_personal_context_receipt_locks_binding_before_upsert
1 failed, 266 passed, 2 skipped, 13 warnings in 12.24s
       # the failure is the pre-existing `link_state` fixture drift recorded as sync-11;
       # retention and blob_store are fully green.

$ grep -rn "expires_at <\|expires_at >\|expires_at IS\|expired_upload\|purge_expired" \
      tldw_Server_API/app/core/DB_Management/Sync_DB.py
(no output)          # sync_blob_upload_sessions.expires_at is never read in any predicate

$ grep -n "expires_at" tldw_Server_API/app/core/Sync/v2/service.py
(no output)          # and create_blob_upload_session never sets it — the field defaults to None
                     # at v2/models.py:1994 and is inserted as NULL at Sync_DB.py:11901

$ grep -n "max_active_blob_uploads\|max_envelope_payload_bytes\|restore_manifest_scan_limit" \
      tldw_Server_API/app/core/Sync/v2/service.py | head -3
839:    max_envelope_payload_bytes: int = 262_144
844:    max_active_blob_uploads: int = 8
864:    restore_manifest_scan_limit: int = 10_000

$ grep -rn "cancel_blob_upload" tldw_Server_API/app/api/
tldw_Server_API/app/api/v1/endpoints/sync.py:2262     # the sync API exposes cancel
                                                      # api/v1/endpoints/notes.py: no hits
```

## Findings

### FINDING sync-2

```
axis:        correctness
class:       n/a
severity:    High
sites:       The leaking caller: api/v1/endpoints/notes.py:4883-4954 — a single `try:` block
               running create_blob_upload_session (:4884-4907) -> N x upload_blob_chunk
               (:4908-4920) -> complete_blob_upload (:4921-4925) -> coordinator.capture
               (:4927-4951), closed by `except Exception as exc: raise
               _notes_attachment_http_error(exc) from exc` (:4952-4953). No `finally`, no
               `cancel_blob_upload`, no compensating cleanup on any failure path.
             The cap that turns a leak into a wedge: v2/service.py:_validate_blob_limits
               (10967-10991), `if quota.active_upload_count >= self.settings.max_active_blob_uploads:
               raise SyncStoreError("Sync blob active upload limit exceeded")` (:10990-10991),
               with `max_active_blob_uploads: int = 8` (v2/service.py:844).
             The count it reads: core/DB_Management/Sync_DB.py:summarize_blob_quota (12604-12609),
               `COUNT(*) ... WHERE owner_user_id = ? AND status IN ('created','uploading')` —
               no expiry predicate, no age predicate.
             The expiry that does not exist: v2/models.py:SyncBlobUploadSessionCreate.expires_at
               (1994) defaults to None; v2/service.py:create_blob_upload_session (6153-6169)
               never sets it; core/DB_Management/Sync_DB.py:11901 inserts it as NULL; and no
               query anywhere reads `expires_at` in a predicate. There is no reaper.
             The reason it is unrecoverable from outside: the session's `upload_id` is created
               server-side inside the notes endpoint and never returned to the client (the
               handler returns `_canonical_to_legacy_attachment_response(...)` on success and
               raises on failure), so the client cannot call the
               api/v1/endpoints/sync.py:2262 cancel endpoint for it.
             The reason retries do not reuse the session:
               api/v1/endpoints/notes.py:_optional_attachment_idempotency_key (695-703) returns
               `uuid4().hex` when no Idempotency-Key header is supplied, so each retry produces a
               different `idempotency_key` at :4896-4899 and therefore a NEW session rather than
               resuming the existing one.
canonical:   NONE
destination: n/a — the fix is a `finally` (or an explicit `except` compensating branch) around
             the sequence, plus making the expiry column real.
knowledge:   n/a
scenario:    A user uploads an attachment through the legacy notes endpoint
             (`POST /api/v1/notes/{note_id}/attachments`) with no `Idempotency-Key` header — the
             documented-optional case.
               1. `create_blob_upload_session` succeeds; the row is inserted with
                  `status='created'` and `reserved_quota_bytes=size_bytes` (v2/service.py:6166).
               2. A chunk write fails — a chunk-hash mismatch, a full disk under
                  `LocalSyncBlobStore`, a `SyncMaterializationBusyError`, or simply the client
                  dropping mid-request. Control leaves via :4952 and the HTTP error is returned.
               3. The session row is still `status='created'`. Nothing cancels it, `expires_at`
                  is NULL, and no reaper exists.
               4. The user retries. `_optional_attachment_idempotency_key` mints a fresh UUID, so
                  step 1 creates a second session rather than resuming the first.
               5. On the 9th attempt `_validate_blob_limits` sees `active_upload_count == 8` and
                  raises "Sync blob active upload limit exceeded". From that point **every**
                  attachment upload for that user — through this endpoint and through the sync
                  API — fails, permanently, with a message that describes a limit the user cannot
                  see and cannot clear.
             Each orphan also holds `reserved_quota_bytes` against `user_blob_quota_bytes`
             (Sync_DB.py:12604, service.py:10986-10989), so a user with a quota configured loses
             that space too.
impact:      High. Eight ordinary transient failures permanently disable a user-facing feature,
             with no self-service recovery and no operator-visible signal short of querying
             `sync_blob_upload_sessions` by hand. It requires no malice: a flaky network on a
             large attachment is enough. The sync API path does better only because it exposes
             `cancel_blob_upload` and returns the `upload_id`, making cleanup the remote client's
             job — which is itself a sequential-coupling hazard (a client that never calls cancel
             leaks identically), just one with an escape hatch.
cost-driver: n/a
tests:       import-grep reachability, not measured coverage —
             tests/Notes/test_notes_attachment_sync_api.py, tests/e2e/test_notes_attachment_sync_v2.py,
             tests/Sync/test_sync_v2_blob_store.py, tests/Sync/test_sync_v2_notes_attachment_coordinator.py.
             All of them drive the success path. The regression test is small: inject a failing
             `upload_blob_chunk`, then assert the session is not left in `created`/`uploading`.
effort:      cheap for the compensating cleanup (a `finally` that cancels a session that never
             reached `completed`); moderate to also make `expires_at` real, which needs a TTL
             setting, a value at insert, an `expires_at` predicate in
             `summarize_blob_quota`, and a decision about who reaps (the retention pass is the
             obvious home — `_apply_retention_blob_gc` already runs there).
owner-only:  yes for the notes.py cleanup — api/v1/endpoints/notes.py is under
             `tldw_Server_API/app/api/v1/**`. The expiry work is in core/ and
             core/DB_Management/ and is not owner-only.
confidence:  confirmed (every link in the chain verified in source: the missing cleanup, the cap,
             the unfiltered count, the never-written expiry, the never-read predicate, the fresh
             UUID on retry, and the unreturned upload_id). Not executed end to end — labelled
             confirmed-by-construction rather than reproduced, unlike sync-1.
```

### FINDING sync-9

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       v2/service.py:retention_dry_run (3338-3497) — the scan at :3369-3378,
               `self.store.list_accepted_envelopes_for_replay(dataset_id, since_cursor=0,
               limit=remaining_limit)` where `remaining_limit` derives from
               `scan_limit = limit or self.settings.restore_manifest_scan_limit` (:3359)
             v2/service.py:retention_compact (3206-3216) — calls `retention_dry_run` first, so
               every apply pays the scan before doing any work, and then re-reads candidates
               inside three guarded revalidation passes (`_apply_retention_domain_compactions`
               8471-8618, `_apply_retention_binding_releases` 8707-8778,
               `_apply_retention_blob_gc` 8780-8917)
             v2/service.py:diagnostics (3101-3186) — same knob at :3130
             The one knob serving seven unrelated scans:
               v2/service.py:SyncV2Settings.restore_manifest_scan_limit (864) = 10_000, read at
               :3130 (diagnostics), :3359 (retention), :6010 (repair), :7976 (attachment
               diagnostics, further capped at min(limit, 1_000)), :10610 (profile manager),
               :10782 (restore mutation-group expansion), :11039 (latest cursor for domains)
             Endpoints: api/v1/endpoints/sync.py:1451 (dry-run), :1483 (compact), :1422
               (diagnostics)
canonical:   NONE
destination: n/a — bound the scan, and split the knob
knowledge:   n/a
scenario:    n/a
impact:      Medium. Correct, and bounded — but bounded by a number chosen for a different
             feature.
cost-driver: Peak memory and I/O of one synchronous request, scaling with the number of accepted
             envelopes in the dataset up to a hard 10,000, times the per-envelope payload size up
             to `max_envelope_payload_bytes` (262,144 bytes, v2/service.py:839). `SyncEnvelope`
             carries `payload`, `payload_clear` and `payload_ciphertext`, so the scan materialises
             whole envelope bodies, not a projection: worst case ~2.5 GB resident for one
             `POST /retention/dry-run`, and a realistic 4 KiB average still means ~40 MB per call
             held for the duration. `retention_compact` pays it before doing anything, and the
             scan always restarts at `since_cursor=0` — there is no incremental watermark, so the
             cost does not amortise across repeated calls. These are `def` handlers so FastAPI
             runs them in the thread pool and the event loop is not blocked; the cost is memory
             and thread-pool occupancy, not latency for other requests.
tests:       import-grep reachability, not measured coverage —
             tests/Sync/test_sync_v2_retention.py (2,215 LOC, green) and
             tests/Sync/test_sync_v2_diagnostics.py cover candidate classification and report
             shape at small fixture sizes. Nothing exercises a large dataset, so the cost is
             invisible to the suite.
effort:      moderate. Three separable pieces: (1) select only the columns retention actually
             reads (`server_cursor`, `domain`, `object_id`, `operation`, `deleted`,
             `client_timestamp`) instead of whole envelopes — this alone removes the payload
             term and is the highest-value change; (2) give retention and diagnostics their own
             settings instead of borrowing `restore_manifest_scan_limit`; (3) carry a
             per-dataset retention watermark so `since_cursor` is not always 0. Piece (1) touches
             core/DB_Management/Sync_DB.py and needs a projection row type, so it is not a
             one-liner.
owner-only:  no
confidence:  confirmed (the scan shape, the shared knob, the seven consumers, and the two
             settings values were all read directly); assumption (the worst-case memory figure —
             it is the product of two documented caps, not something observed)
```

## Suggested Refactor/Actions

1. `sync-2` — split into two PRs. The core/ half is not owner-only: give
   `SyncBlobUploadSessionCreate` a real `expires_at` from a new
   `SyncV2Settings.blob_upload_session_ttl_seconds`, add the `expires_at` predicate to
   `summarize_blob_quota`, and reap expired sessions inside the existing
   `_apply_retention_blob_gc` pass. The `api/v1/endpoints/notes.py` half (a `finally` that
   cancels any session that never reached `completed`) is **owner-only** and should be raised
   with the API owner rather than proposed as a drive-by. Both need a Backlog task; neither needs
   a design doc — this is defect repair, not a decision.
2. Add the regression test alongside: inject a failing `upload_blob_chunk` into the notes
   compat upload and assert no session is left in `created`/`uploading`.
3. `sync-9` — do piece (1) only (column projection on the retention scan). Pieces (2) and (3)
   are worth doing but should wait behind a measurement; proposing a watermark without a number
   is speculation. If any of this grows past the projection change it needs
   `Docs/Design/YYYY-MM-DD-sync-retention-scan-bounds-design.md`.
4. Note for whoever picks these up: everything in this stage is SQLite-only tested. The
   `summarize_blob_quota` change in particular is a `COUNT(*)` predicate that must behave
   identically on both backends, and `sync-11` is the prerequisite for being able to prove that.
