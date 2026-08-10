# ADR-035: Canonical folder-link suppression preserves source provenance

**Status:** Accepted
**Date:** 2026-08-08
**Backfilled from:** not backfilled
**Decision owner:** TASK-13003 requester and implementation review
**Related task:** TASK-13003
**Related ADRs:** `Docs/ADR/020-db-management-per-user-paths-and-content-backend.md`, `Docs/ADR/031-notes-capability-sync-domains.md`, `Docs/ADR/034-durable-server-origin-sync-mutation-batches.md`
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

### 2026-08-09 amendment: tenant isolation for shared PostgreSQL storage

Notes organization projection rows belong to the authenticated owner bound to the
`CharactersRAGDB` instance. Envelope device IDs, server-origin labels, payload
fields, and routing metadata are never ownership authorities.

Every projection-store resource query and mutation explicitly filters by that
owner's `client_id`. Hierarchy traversal keeps both the child and every ancestor in
the same owner scope. Relationship reads and writes resolve both endpoints in that
scope before touching a link, provenance, or suppression row. These predicates are
required even when PostgreSQL row-level security is active because table owners and
privileged service roles can bypass RLS.

PostgreSQL RLS is also enabled and forced for the three organization resource
tables and all organization link, source-provenance, and suppression tables.
Resource policies compare `client_id` directly with
`app.current_user_id`. Tables without a duplicated owner column derive access from
their owned foreign-key endpoints; source-key rows derive access from their folder,
the only product endpoint they store. Link tables do not gain redundant
`client_id` columns, avoiding a second ownership value that could drift from the
resources it protects.

Schema v57, ordered after the independently owned Web Clipper v56 migration,
replaces global keyword, collection, and folder identity/name/path uniqueness with
owner-scoped uniqueness. Existing global constraints make this a non-destructive
constraint relaxation, so v57 does not merge or delete rows. Folder startup
deduplication partitions by owner before comparing case-folded paths.

The shared PostgreSQL migration acquires transaction-held `SHARE ROW EXCLUSIVE`
locks on the organization resources, Notes and conversation ownership endpoints,
and their organization relationship/provenance/suppression tables before its first
preflight read. This closes the validation-to-DDL race by temporarily blocking
concurrent writes until migration commits or rolls back. Before changing
constraints, it also rejects cross-owner keyword-collection and folder hierarchies,
keyword links, and manual/source/suppressed note-folder relationships. It never
repairs, deletes, or reassigns those rows implicitly.

After locking, v57 reads the PostgreSQL catalog for all eleven locked tables and
requires every relation to exist and be owned by the current migration role. It
records each table's enabled/forced RLS state, rejects an inconsistent forced-but-
disabled state, and temporarily applies `NO FORCE ROW LEVEL SECURITY` only to the
exact set that was forced when migration began. Schema ownership then provides the
unfiltered view needed by the global owner, collision, and cross-owner preflights.
The migration restores `FORCE ROW LEVEL SECURITY` to that exact set before
advancing the schema version. All catalog checks, temporary RLS changes, preflight
queries, DDL, restoration, and version advancement share one transaction, so any
failure rolls the entire sequence back to its original constraints, version, and
RLS state.

Once the v56 and v57 schemas and all PostgreSQL ensure steps exist, initialization
reinstalls the complete ChaCha RLS policy set in the same schema transaction. This
repairs startup ordering where an earlier optional RLS pass can roll back on legacy
v55 Web Clipper sidecars that do not yet have their v56 owner column, without
leaving the organization policies absent after migration.

Before changing constraints in shared PostgreSQL, v57 requires every organization
resource `client_id` to be the exact canonical authenticated-user identifier: a
positive decimal integer string with no surrounding whitespace, sign, zero value,
or leading zero. Legacy values such as `unknown`, device IDs, worker labels, or
`server-origin` are not guessed or rewritten. Content storage may be separate from
AuthNZ storage, so the migration validates the canonical identifier form without
assuming a co-located users table. If an owner cannot be determined from
authoritative data, migration stops with an operator-visible error and leaves the
old constraints intact.

SQLite remains one database per authenticated owner, so it does not rebuild its
tables merely to relax uniqueness. Its v56-to-v57 step only advances the schema
version and accepts existing local nonnumeric desktop `client_id` values. SQLite
still executes the same explicit owner predicates, which provides consistent
semantics and makes accidental mixed-owner rows invisible.

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
| Rely only on PostgreSQL RLS | Privileged connections may bypass RLS, and application hierarchy validation still has to reject cross-owner parents. |
| Copy `client_id` onto every link and provenance row | Redundant ownership can drift from either endpoint and substantially enlarges the migration and write surface. |
| Keep global resource identities and names | One tenant could block another tenant from using the same portable Sync identity, keyword, collection name, or folder path. |
| Infer legacy owners from device or server-origin metadata | Those values identify mutation origin, not the authenticated owner, and guessing would silently transfer data between tenants. |

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

Shared PostgreSQL deployments gain defense in depth: owner-scoped application SQL
continues to isolate data when RLS is bypassed, while RLS protects direct SQL paths.
Two tenants may use the same organization Sync IDs and case-folded names/paths.
Deployments with unattributable legacy ownership must repair that data from an
authoritative owner mapping before v57 can proceed.

## Follow-up

- TASK-13003 Task 2 implements and tests the projection table on SQLite and PostgreSQL.
- TASK-13003 Task 9 applies origin-only provenance deltas and effective-union changes in one product transaction.
- TASK-13003 final contract hardening applies the explicit owner predicates, RLS policies, and schema v57 owner-scoped constraints described by the 2026-08-09 amendment.
