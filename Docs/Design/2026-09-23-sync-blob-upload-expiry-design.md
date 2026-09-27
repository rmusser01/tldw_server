# Sync v2 blob upload-session expiry

**Status**: Implemented (partial — see [Out of scope](#out-of-scope))
**Task**: TASK-13321
**Date**: 2026-09-23

## The problem

`Sync_V2_M2_Restore_Completeness_and_Blobs.md` put upload-session expiry in scope — ":82
release reservations on cancel/expiry", ":240 M2 should enforce … Upload-session
expiration", and it scoped out blob GC only "beyond … abandoned upload cleanup". The schema,
the status Literal and the quota query's status filter were all built for it. **Only the
writer was missing.**

Measured before this change:

| Piece | State |
|---|---|
| `sync_blob_upload_sessions.expires_at` column | exists, written verbatim from `session.expires_at` |
| `SyncBlobUploadSessionCreate.expires_at` | defaults to `None`, **set at zero call sites** |
| status `"expired"` in `SyncBlobUploadStatus` | declared, read as terminal in one place, **written at zero call sites** |
| a reaper | **does not exist** |
| `summarize_blob_quota` | sums `WHERE status IN ('created','uploading')`, **no expiry predicate** |

**Effect.** `max_active_blob_uploads` defaults to 8. A client that starts 8 uploads and then
crashes or loses network — ordinary transient failure, not abuse — permanently exhausts its
own limit. Every later attachment upload for that user+dataset raises "Sync blob active
upload limit exceeded", forever. Each abandoned session's `reserved_quota_bytes` also counts
against `user_blob_quota_bytes` permanently, so abandoning one 1 GB upload costs 1 GB for
good. Recovery requires the client to have remembered all 8 `upload_id`s, or a manual DB
edit.

## Decision: expiry is a read-time predicate, not a background reaper

TASK-13321's AC#4 assumed a sweep that transitions rows to `expired`. **This
implementation does not add that transition**, and that is the design decision here.

A session is expired iff `expires_at IS NOT NULL AND expires_at <= now`. That is evaluated
where it matters — in the quota query, and in the write path — rather than materialised into
a status by a job.

**Why the predicate rather than the reaper:**

1. **Correct without anything having run.** A crashed client's budget comes back the instant
   its deadline passes. With a reaper, recovery latency is the sweep interval, and a sweep
   that is misconfigured, crashed or never deployed means the defect silently persists —
   which is exactly the failure this task is fixing.
2. **No new failure mode.** A reaper is a scheduler, a lock, a batch size, a retry policy
   and an alert. None of that is needed to answer "is this past its deadline".
3. **Idempotent by construction.** There is no window where the row says `uploading` but the
   quota says released, or vice versa.
4. **The status Literal stays honest.** `"expired"` remains unwritten, so no code can come
   to depend on a transition that only a background job performs.

**Cost of the choice.** The predicate must be applied everywhere quota or liveness is
judged. Today that is two SQL branches in `summarize_blob_quota` and one guard in
`record_blob_chunk`. Adding a third reader of session liveness means adding the predicate
there too — the tests pin both SQL branches separately for that reason.

## Implementation

- `Sync/v2/models.py` — `blob_upload_expires_at(ttl_seconds, now=None)` composes the
  deadline; `blob_upload_session_is_expired(expires_at, now=None)` answers the predicate.
  Placed here because `Sync_DB` already imports from this module and `service.py` imports
  `.models`, so one definition serves both layers without the service reaching into
  `DB_Management`.
- `Sync/v2/service.py` — `create_blob_upload_session` stamps `expires_at` from
  `settings.blob_upload_session_ttl_seconds` and the injected `clock`. Default 86 400 s
  (24 h): long enough for a resumable upload over a poor link, short enough that a crashed
  client recovers the same day.
- `Sync/v2/factory.py` — `SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS`, read through a new
  `_sync_v2_non_negative_int_env`. **0 must be accepted**, because it is how an operator
  turns expiry off deliberately; the existing `_sync_v2_positive_int_env` rejects 0 and so
  cannot express that.
- `DB_Management/Sync_DB.py` — both branches of `summarize_blob_quota` gain
  `AND (expires_at IS NULL OR expires_at > ?)`, bound to a single `quota_as_of` so one
  answer cannot straddle two instants. `record_blob_chunk` refuses an expired session.

### Two constraints that are easy to get wrong

**`NULL` means "no deadline", not "already expired".** Every session written before this
change has `NULL`. Treating `NULL` as expired would free quota for live uploads on upgrade.
Pinned by a test.

**The comparison is lexicographic.** That is only correct because every writer produces one
fixed format, `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`, via `utcnow_iso()` /
`blob_upload_expires_at()`. A `Z`-suffixed or offset-shifted value written into `expires_at`
would sort wrongly. Noted at the definition site.

**Closing the write path is what makes releasing quota sound.** If an expired session could
still accept chunks, a client could let one expire — freeing its budget for another upload —
then resume it and exceed the quota. `record_blob_chunk` therefore refuses expired sessions,
and a test asserts it.

## Out of scope

**Disk reclamation (AC#4's other half).** Staged chunk files written by
`blob_store.write_upload_chunk` are still only discarded by the complete and cancel paths.
An expired session's staged bytes remain on disk until something calls `discard_upload`.
That needs a sweep — but it is a **cost** problem, not a correctness one: the lockout and
the quota loss are fixed without it. Deliberately separated so the reliability fix is not
blocked on introducing a scheduled job. AC#4 stays open for this.

**`notes.py` compensation (AC#6).** `api/v1/endpoints/notes.py:4883-4954` has no `finally`
and no compensating `cancel_blob_upload`, and never returns the `upload_id`, so the cancel
endpoint is unreachable for that flow. Expiry now bounds the damage to the TTL rather than
forever, which is why this is no longer urgent — but it is a separate change, and it touches
`app/api/v1/**`, which the contribution gates mark owner-only.

## Verification

14 tests in `tests/Sync/test_sync_v2_blob_upload_expiry.py`, including AC#1's scenario:
8 abandoned sessions, then the ninth upload's limit check passes — with a control asserting
8 *live* sessions still hit the limit, so the protection was fixed rather than removed.

Probe-the-fix, six ways; each reverted half turns tests red:

| Reverted | Red |
|---|---|
| quota predicate, unscoped branch | 1 |
| quota predicate, dataset-scoped branch | 2 |
| write-side expiry guard | 1 |
| service stops stamping the deadline | 1 |
| TTL default set to 0 | 1 |
| `NULL` treated as expired | 1 |

The fourth row was found the hard way: with only the helper under test, deleting the
`expires_at=` argument from the create path left every test passing. The suite now exercises
`SyncV2Service.create_blob_upload_session` itself, because the defect being fixed *is* that
call site being unwired.
