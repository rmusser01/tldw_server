# ADR-036: Web Clipper external identity mapping

**Status:** Accepted
**Date:** 2026-08-09
**Decision owner:** TASK-13003 final implementation review
**Related task:** TASK-13003
**Related ADRs:** `Docs/ADR/031-notes-capability-sync-domains.md`, `Docs/ADR/034-durable-server-origin-sync-mutation-batches.md`, `Docs/ADR/035-canonical-folder-link-suppression-preserves-source-provenance.md`
**Related design:** `Docs/superpowers/specs/2026-04-03-browser-extension-web-clipper-design.md`

## Decision

Treat the Web Clipper's public `clip_id` as an owner-scoped external idempotency
key, not as a canonical Notes object identifier. Every clipper document maps that
external key to one separate, unique `notes.id`:

- `note_clipper_documents.clip_id` is the stable public lookup and response value;
- `note_clipper_documents.note_id` is a unique foreign key to the canonical note;
- a new clip's note ID is derived deterministically from the authenticated owner
  and `clip_id` with the existing `stable_note_id()` helper, producing a canonical
  lowercase UUIDv4-shaped value; and
- status, enrichment, attachment, workspace-placement, delete, and restore paths
  resolve the note ID through the document row instead of assuming equality.

SQLite keeps the authenticated per-user database as the ownership boundary for the
public key. Its sidecar remains keyed by `clip_id`; `client_id` is not persisted in
the sidecar because it can be a device or session label and may legitimately change
when the same owner-local database is reopened.

The request boundary trims `clip_id` and `clip_type` and rejects whitespace-only
values before any direct or active-Sync write. Path-based clip lookups apply the same
`clip_id` normalization, so padded saves, exact retries, and status requests resolve
one public identity.

PostgreSQL uses a shared ChaChaNotes database, so the ownership boundary must be
materialized in the sidecar itself. `note_clipper_documents` stores `client_id` and
uses `(client_id, clip_id)` as its primary key. Workspace placements store the same
owner and reference the document through `(client_id, clip_id)`. All PostgreSQL
sidecar reads and writes include the authenticated `client_id`, and both tables use
row-level security with matching `USING` and `WITH CHECK` predicates that verify the
sidecar owner and every referenced note, document, and workspace endpoint. The same
`clip_id` may therefore map to different note IDs for different owners. Public
responses return the normalized public `clip_id`, while the existing `note.id` and
compatibility `note_id` fields return the canonical note UUID.

Active-Sync saves use a privacy-safe stable key derived from owner plus `clip_id`
for the complete ADR-034 mutation group. The stored request fingerprint binds the
normalized note content and organization request to that key. An exact retry
reuses the same group, note version, and response identity; a different request
using the same `clip_id` is a stable HTTP 409 idempotency conflict. Server-authored
routing metadata contains hashes, bounded status markers, and the server-persisted
capture timestamp needed to repair a missing sidecar. The timestamp may be a validated
caller value or a server default; the routing structure itself is not caller-controlled.
It never contains raw clip content, the raw public key, credentials, or authorization
data.

Readiness, preflight, and idempotency conflicts are stable HTTP 409 responses.
Durable-append and incomplete-projection failures are retryable HTTP 503 responses.
An active dataset whose Notes organization group is partial, initializing, or
failed never falls back to a direct ChaChaNotes write.

ChaChaNotes schema v56 removes the `clip_id -> notes.id` foreign key and
`clip_id = note_id` check while retaining unique `note_id` ownership and the
workspace-placement foreign keys. During v55-to-v56 migration:

1. rebuild the document table with the decoupled constraints;
2. preserve an already canonical lowercase UUIDv4 note ID unchanged, because it
   may already identify canonical Sync history;
3. rekey a legacy non-UUID clipper-owned note to a new UUIDv4 before later Notes
   organization enrollment, updating the legacy local `sync_log` identity only when
   each payload is a JSON object with exactly one root text `id` equal to the old
   note identity, then verifying exactly one such `id` equals the replacement, and
   relying on verified `ON UPDATE CASCADE` relationships while preserving the
   original public `clip_id`; and
4. abort the transaction on collisions, missing note rows, broken references, or
   any state that cannot be migrated as one verified mapping.

The PostgreSQL migration additionally serializes schema initialization by locking
the ChaChaNotes schema-version row with `FOR UPDATE` before deciding which migrations
to run. Before any Web Clipper mapping, history, or constraint preflight, it takes
transaction-held write-blocking locks on `notes`, `workspaces`, `sync_log`, the
existing Web Clipper sidecars, and any co-located canonical Sync history tables. It derives every
document owner only from the referenced `notes.client_id`; blank, missing,
ambiguous, cross-owner source-note, or cross-owner workspace state fails closed.
The owner column is then made non-null and the composite document, placement, and
foreign-key constraints are installed before the schema version advances.

If `notes` has forced PostgreSQL row-level security, the schema-owner migration path
must first verify the catalog state and ownership while those locks are held. It may
temporarily issue `NO FORCE ROW LEVEL SECURITY` inside the migration transaction so
all owners' legacy mappings are visible, and must restore `FORCE ROW LEVEL SECURITY`
after global migration verification and before final constraint validation and the
version bump. Unverifiable catalog state or
insufficient permission aborts the transaction; rollback restores both data and RLS
state.

The automatic migration is intentionally narrower than "make every ID valid." A
parseable UUID that is not already canonical lowercase UUIDv4 fails closed for an
explicit migration plan. Before rekeying, the migration checks every declared
foreign key that currently references the legacy note; a live non-cascading
reference aborts the transaction. If Sync envelope or object-state tables are
co-located with the Notes database, a `notes.note` reference to the legacy ID also
aborts. The deterministic replacement UUID must also be absent from `notes`, the
owner's PostgreSQL `sync_log` (or the owner-local SQLite log), and every co-located
canonical Sync envelope/object-state table; prior replacement history fails closed
instead of merging two identities. These checks make the SQLite upgrade atomic and
idempotent: a failed check leaves schema version 55 and all old mappings intact,
while reopening a completed version 56 database performs no second rekey.

Strict `notes.note` validation cannot have accepted a non-UUID canonical object ID,
so the automatic non-UUID rekey does not rewrite valid Sync v2 history. If a future
or externally modified deployment supplies contrary evidence from a separate Sync
store, it must fail closed and use an explicit canonical migration plan; it must not
silently rewrite or orphan that history.

An active-Sync save can durably append and materialize the canonical note before the
separate Web Clipper sidecar write completes. Its idempotency fingerprint therefore
binds only normalized client intent, including source, requested content and filing,
workspace, attachment, and enhancement inputs. It excludes server-generated capture
timestamps and mutable note versions. On an exact retry with a durable manifest but
a missing sidecar, manifest replay happens before collision detection, returns the
original materialized note without another envelope or note version, recovers the
original capture date from bounded server-only routing metadata, and repairs the
sidecar with the original materialized note content. A
different request for the same owner-scoped key remains an idempotency conflict.
Failure of the first sidecar write after canonical projection is exposed as a safe,
retryable HTTP 503 rather than a generic 500.

Before any active-Sync append, save resolves the owner-scoped document mapping with
soft-deleted rows included. A deleted mapping returns a stable conflict requiring the
existing clip to be restored; it must not derive a second note ID or append an orphan
canonical note.

## Context

The original clipper design reused `clip_id` as `notes.id`, and the v55 sidecar
schema encoded that assumption with two database constraints. TASK-13003 makes
`notes.note` and note organization state canonical Sync v2 objects, whose IDs must
be canonical UUIDv4 strings. An ordinary extension idempotency key such as
`clip-123` is therefore a valid public clip key but an invalid Sync object ID.

The first active-Sync integration passed the public key directly to
`capture_note_upsert()`. It could also persist the same unsafe ID while Sync was
inactive, leaving a note that could not later participate in organization bootstrap
or links. Mapping at the sidecar boundary fixes both paths without changing the
extension contract.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Require public `clip_id` to be UUIDv4 | Breaks the established extension idempotency contract and existing callers. |
| Continue using `clip_id` as `notes.id` only while Sync is inactive | Leaves note-only captures unsafe for later enrollment and changes identity when Sync activates. |
| Generate a fresh random note UUID on every request | Network retries could fork one public clip into multiple canonical notes. |
| Store only a derived note UUID and discard `clip_id` | Breaks public status, enrichment, workspace, and idempotent retry lookups. |
| Rewrite every legacy UUID note during migration | Could orphan already valid canonical Sync history for no benefit. |
| Store raw request content or `clip_id` in Sync routing metadata | Expands durable sensitive-data exposure and is unnecessary for replay validation. |
| Namespace `clip_id` internally but keep a global PostgreSQL key | Hides collisions from some callers but does not enforce tenant ownership or protect direct database access. |
| Create one PostgreSQL sidecar table or schema per owner | Adds operational and migration complexity when composite keys and RLS provide the required isolation. |

## Consequences

The sidecar becomes the authoritative mapping between extension identity and Notes
identity. Code that starts with `clip_id` must load the owner-scoped sidecar before
touching the note, attachments, or placements. Code that starts with a note ID can
continue using the owner-scoped reverse lookup. PostgreSQL deployments pay the small
cost of an additional owner column and composite indexes; SQLite keeps its existing
owner-local layout and remains compatible across client-label changes.

The schema receives one linear migration on both SQLite and PostgreSQL. Legacy
non-UUID clip notes change their internal note ID once, while the public clip ID and
workspace/source identities remain stable. Canonical UUID notes are not rekeyed.

ADR-034 still governs append-before-projection and exact group replay. ADR-035 still
governs folder-link projection. This ADR changes only the Web Clipper public-to-
canonical identity boundary and its migration.
