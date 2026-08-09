# ADR-033: Canonical folder-link suppression preserves source provenance

**Status:** Accepted
**Date:** 2026-08-08
**Backfilled from:** not backfilled
**Decision owner:** TASK-13003 requester and implementation review
**Related task:** TASK-13003
**Related ADRs:** `Docs/ADR/031-notes-capability-sync-domains.md`, `Docs/ADR/032-durable-server-origin-sync-mutation-batches.md`
**Related spec/plan:** `Docs/superpowers/specs/2026-08-08-notes-organization-sync-design.md`, `Docs/superpowers/plans/2026-08-08-notes-organization-sync-implementation-plan.md`

## Decision

Treat `notes.folder_link` as the authority for the user-visible effective
note-to-folder relationship while preserving local source-ingestion provenance in
its existing tables.

ChaChaNotes schema v55 adds a server-owned projection table named
`note_folder_sync_suppressions`. It stores a unique `(note_id, folder_id)` pair for
each canonical folder-link tombstone that must hide a relationship still supported
by local source provenance. Suppression rows are derived product projection state;
they are not a seventh Sync domain and are never serialized as client payload.

The effective membership query is:

```text
(manual memberships UNION source memberships) MINUS sync suppressions
```

Applying a canonical `notes.folder_link` upsert removes the matching suppression
and ensures the manual projection row exists. Applying a tombstone removes the
manual projection row and inserts the matching suppression. Neither operation
deletes `note_folder_source_memberships` or `note_folder_source_keys`.

Source-managed mutations remain local provenance operations. When Sync is active,
their coordinator computes the prospective effective union and emits a canonical
link change only when the user-visible relationship changes. The origin-only
provenance delta and canonical projection commit in one ChaChaNotes transaction;
remote materializers ignore that provenance delta and still converge because the
canonical upsert clears, and the canonical tombstone creates, the local suppression.

Fresh SQLite and PostgreSQL schemas and the v54-to-v55 migration create the table
and its unique lookup index. Existing rows require no backfill: before organization
Sync history exists, no canonical tombstone exists to suppress them.

## Context

ChaChaNotes derives visible note-folder membership from the union of manual and
source-managed membership tables. TASK-13003 deliberately keeps source IDs and
folder keys out of portable Sync state while making that visible union canonical.

Task 2 review found an inconsistency in the first projection seam: snapshots
correctly included source-backed relationships, but a canonical tombstone deleted
only the manual row. A source-backed relationship therefore reappeared in the next
snapshot and remained visible, so the tombstone had not actually materialized.
Deleting the source row would make the view converge but would destroy local
ingestion bookkeeping and violate the approved provenance boundary.

The product database needs a local representation of canonical absence that can
mask provenance without owning or rewriting it. A suppression pair is the smallest
such projection and keeps the existing Sync object state, source tables, and REST
integer identifiers intact.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Delete manual and source memberships on a canonical tombstone | It destroys local ingestion provenance and makes a remote canonical event rewrite unrelated local bookkeeping. |
| Synchronize source IDs, folder keys, and source-membership rows | Those values are server-local ingestion state, not portable personal knowledge, and would create a competing authority. |
| Let source membership override a canonical tombstone | The visible relationship would remain present and the next snapshot would contradict accepted Sync history. |
| Query the Sync database from every folder read | It couples product reads to Sync-store availability and transaction semantics, and makes offline projection incomplete. |
| Defer the inconsistency to the later provenance task | Materializers and bootstrap would build on a projection seam that cannot apply every canonical relationship state. |

## Consequences

Canonical folder-link upserts and tombstones become idempotent for manual-only,
source-only, and mixed provenance states. Source provenance survives remote Sync
changes, while the visible organization converges to canonical history.

Folder list and snapshot queries must consistently subtract suppressions. Source
mutation coordination must clear or create suppressions through the same canonical
link apply path rather than mutating them independently. Repair and restore rebuild
suppression state by replaying canonical folder-link heads.

The product schema gains one small derived table and lookup index. Operators may
observe source rows whose corresponding relationship is intentionally hidden; this
is expected and repairable from canonical Sync history.

## Follow-up

- TASK-13003 Task 2 implements and tests the projection table on SQLite and PostgreSQL.
- TASK-13003 Task 9 applies origin-only provenance deltas and effective-union changes in one product transaction.
