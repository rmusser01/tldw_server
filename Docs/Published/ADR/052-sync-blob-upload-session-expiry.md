# ADR-052: Sync blob upload sessions expire, and expiry is decided at read time

**Status:** Accepted
**Date:** 2026-09-23
**Backfilled from:** not backfilled
**Decision owner:** repository owner
**Related task:** TASK-13321 (implementation landed with TASK-13335, 22b80424f1)
**Related spec/plan:** `Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md` (Quota And Abuse Controls); `Docs/API/Sync_V2_M2.md`

## Decision

Every new blob upload session gets `expires_at = now + SYNC_V2_BLOB_UPLOAD_SESSION_TTL_SECONDS`
(default 24h); quota and the active-upload cap ignore a session past its `expires_at` at read time,
and the retention pass (`retention_compact` with `apply_blob_gc`) later marks it `expired` and
discards its staged chunk files.

## Context

The session lifecycle had `expired` in the status set and an `expires_at` column, but nothing wrote
either. With `max_active_blob_uploads = 8` and no reaper, eight abandoned uploads (a client crash or a
flaky network) blocked that user's attachment uploads permanently and held their reserved quota.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Release only when the sweep runs | The sweep runs only when a client calls `/retention/compact`; a user who is blocked cannot be expected to know that. Read-time exclusion releases the slot without it. |
| Compare `expires_at` in SQL | The column is TEXT on SQLite and TIMESTAMPTZ on PostgreSQL, so a bound string compares differently on each. Rows are filtered in Python; the row count is bounded by the active-upload cap. |
| Also expire sessions with `expires_at IS NULL` | Those rows predate the TTL. Releasing them would change what existing rows mean. They stay counted until cancelled. |
| A scheduled background reaper | Not needed for correctness, because read-time exclusion already frees slot and quota. The only thing that waits for the sweep is removing staged chunk files. |

## Consequences

- An upload that is idle for longer than the TTL cannot be resumed. Clients must restart it.
- Staged chunk files for expired sessions stay on disk until a retention pass runs for that dataset.
- The server-driven Notes upload (`POST /api/v1/notes/{id}/attachments`) cancels its own session on
  failure. Its `upload_id` never reaches the client, so the client could not cancel it.
