# ADR-048: Sync blob upload expiry is a read-time predicate, not a lifecycle transition

**Status:** Accepted
**Date:** 2026-09-23
**Task:** TASK-13321
**Design:** `Docs/Design/2026-09-23-sync-blob-upload-expiry-design.md`

## Decision

A Sync v2 blob upload session is expired iff `expires_at IS NOT NULL AND expires_at <= now`.
That condition is evaluated at every point where quota or liveness is judged. No background
job transitions rows, and the `"expired"` member of `SyncBlobUploadStatus` remains unwritten.

`create_blob_upload_session` stamps `expires_at` from
`SyncV2Settings.blob_upload_session_ttl_seconds` (default 86 400 s, env
`SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS`, `0` disabling expiry) using the service's injected
`clock`. Both branches of `summarize_blob_quota` exclude expired sessions from
`reserved_blob_bytes` and `active_upload_count`, bound to one `quota_as_of` timestamp per
call. `record_blob_chunk` refuses an expired session.

`expires_at IS NULL` means "no deadline", never "already expired". The comparison is
lexicographic and is correct only because every writer emits
`YYYY-MM-DDTHH:MM:SS.ffffff+00:00` via `utcnow_iso()` / `blob_upload_expires_at()`.

Any future code that judges whether a session is live must apply the predicate. It is not
sufficient to read `status`.

## Context

`Sync_V2_M2_Restore_Completeness_and_Blobs.md` placed upload-session expiry in scope. The
`expires_at` column, the `"expired"` status Literal, and the quota query's status filter were
all built for that lifecycle, but `expires_at` was set at zero call sites, `"expired"` was
written at zero call sites, and no reaper existed. `summarize_blob_quota` counted
`created`/`uploading` rows with no expiry predicate.

`max_active_blob_uploads` defaults to 8, so a client that started 8 uploads and then crashed
or lost network permanently exhausted its own limit: every later attachment upload for that
user+dataset raised "Sync blob active upload limit exceeded", indefinitely. Each abandoned
session's `reserved_quota_bytes` also counted against `user_blob_quota_bytes` permanently.
Recovery required the client to have retained all 8 `upload_id`s, or a manual database edit.

TASK-13321's acceptance criteria assumed a sweep that writes the `expired` status.

## Alternatives

- **A scheduled reaper writing `status = 'expired'`**, as the task assumed. Recovery latency
  becomes the sweep interval, and a sweep that is misconfigured, crashed or never deployed
  leaves the original defect silently in place — the same class of failure being fixed. It
  also introduces a scheduler, locking, batch sizing and retry policy for a question that is
  a comparison.
- **Zeroing `reserved_quota_bytes` on expiry.** Destroys the record of what was reserved,
  and still needs something to run.
- **Releasing quota at read time without closing the write path.** Unsound: a client could
  let a session expire, freeing its budget for another upload, then resume it and exceed the
  quota.
- **Treating `expires_at IS NULL` as expired** to avoid a migration. Would free quota for
  live uploads at the moment of upgrade, since every pre-existing row is `NULL`.

## Consequences

Quota and upload slots recover exactly at the deadline with nothing having run, and there is
no window in which the row and the quota disagree. `"expired"` stays unwritten, so no caller
can come to depend on a transition only a job performs.

The predicate must be repeated at each site that judges liveness — currently two SQL branches
and one write guard. The two SQL branches are pinned by separate tests because they are
separate code paths that drifted independently before.

Staged chunk files are **not** reclaimed by this change: `blob_store.write_upload_chunk`
output is still discarded only by the complete and cancel paths, so an expired session's
bytes stay on disk until something calls `discard_upload`. That remains open on TASK-13321 as
a cost concern rather than a correctness one.

`api/v1/endpoints/notes.py` still lacks compensation and does not return the `upload_id`, so
its cancel endpoint is unreachable. Expiry bounds that leak to the TTL instead of forever;
the flow itself is a separate owner-only change.
