# ADR-037: Canonical Notes link Sync and derived graph projections

- **Status:** Accepted
- **Date:** 2026-08-10
- **Task:** TASK-13004
- **Depends on:** ADR-031 and ADR-034
- **Design:** `Docs/superpowers/specs/2026-08-10-notes-link-sync-and-graph-lifecycle-design.md`

## Context

Notes already has an explicit `note_edges` product table and graph APIs, but manual
edges are physically deleted, have no optimistic lifecycle, and are not a Sync v2
domain. Graph reads parse wikilinks opportunistically and currently include deleted
notes. Shared PostgreSQL does not give the legacy edge table the ownership/RLS
contract required for synchronized multi-tenant data.

Synchronizing manual links while also synchronizing backlinks, orphan flags, or
graph summaries would create competing mutable authorities. Conversely, keeping
link lifecycle only in Sync bookkeeping would split product truth and make inactive
behavior, restore, and optimistic edits difficult to reason about.

## Decision

1. `notes.link` adapter version 1 is the sole canonical Sync domain for explicit
   note-to-note `manual` links. It supports `upsert` and `tombstone`.
2. Stable edge identity is independent of mutable label, weight, and properties.
   Source, target, type, and direction are immutable. Logical identity remains
   unique across tombstones.
3. Schema version 58 extends `note_edges` in place with versioned soft-delete state,
   bounded mutable presentation fields, explicit owner checks, and forced
   PostgreSQL RLS. Ownership is derived from authenticated dataset authority, not
   client payload.
4. Note trash leaves incident explicit links intact. Graph visibility requires both
   endpoints to be live, so restore reveals the same links without link churn.
5. Wikilinks, backlinks, orphan state, graph summaries, dirty queues, parser state,
   and graph revision are deterministic local projections and are never Sync
   domains.
6. Normal note-content mutations update the derived projection in the same product
   transaction. Owner-scoped bounded maintenance repairs direct-write/crash drift;
   read endpoints never perform maintenance writes.
7. Graph and lifecycle APIs resolve an optional default-personal dataset, enforce
   owner authorization, reject aliases to other same-owner datasets, and use
   bounded revision-bound pagination and cache keys.
8. Migration is transactional and fail-closed. It locks before inspection, verifies
   same-owner endpoints and exact RLS state, temporarily relaxes FORCE only when
   required under verified schema ownership, restores it before version advance,
   and never guesses, deletes, or silently repairs legacy rows.
9. `notes.link` uses a separate versioned `notes_link_v1` enrollment/bootstrap
   state. Existing organization-ready datasets atomically add and source-verify the
   new domain without reopening or blocking the six-domain
   `notes_organization_v1` group.
10. PostgreSQL `note_edges` policies validate the authenticated owner and both note
    endpoints in `USING` and `WITH CHECK`; soft-deleted endpoints remain valid
    identities but are excluded by graph visibility joins.
11. Link creation provenance is client-signable but not freely client-selectable:
    canonical `created_at`/`last_modified` bind to the submitted envelope timestamp
    and `created_by` binds to the authenticated device. Trusted bootstrap alone may
    preserve source-verified legacy provenance.

## Consequences

- Existing manual links keep their identity and API compatibility while gaining
  optimistic update, tombstone, and restore semantics.
- Already-ready default-personal datasets have a resumable upgrade path; this is
  not a fresh-install-only capability.
- Two clients cannot silently replace the same logical edge. Divergent edits and
  recreate/restore races are reviewable conflicts.
- A deleted note may still have canonical live links, but no graph surface exposes
  those links until both endpoints are live.
- Derived graph queries can temporarily return a retryable rebuilding state after
  migration, parser change, or detected direct-write drift.
- Schema v58 temporarily blocks affected Notes writes during its PostgreSQL
  preflight/DDL transaction and therefore requires explicit operational approval.
- Permanent hard purge requires a future explicit policy for incident link audit
  history; it is not silently cascaded here.

## Alternatives considered

### Create a replacement canonical links table

Rejected because it adds a second storage surface, migration/reconciliation risk,
and compatibility work without improving the public contract.

### Keep product rows unchanged and store lifecycle only in Sync

Rejected because authority would be split between `note_edges` and Sync state,
especially in inactive mode and during exact restore or crash repair.

### Synchronize wikilinks, backlinks, or graph summaries

Rejected because they are deterministic functions of note content and explicit
links. Synchronizing them would permit drift and duplicate conflict surfaces.

### Cascade link tombstones on note trash

Rejected because it destroys existing restore behavior, creates large mutation
groups, and introduces link churn for a visibility-only lifecycle change.

### Add arbitrary relationship targets or types in version 1

Rejected as unnecessary scope that would broaden identity, authorization, schema,
and conflict semantics before a concrete product requirement exists.
