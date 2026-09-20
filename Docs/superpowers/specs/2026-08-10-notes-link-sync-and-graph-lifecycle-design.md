# Notes link Sync and graph lifecycle design

- **Date:** 2026-08-10
- **Task:** TASK-13004
- **Status:** Implemented and verified on the TASK-13004 feature branch
- **Depends on:** TASK-13003 and ADR-031
- **Architecture decision:** `Docs/ADR/037-canonical-notes-link-sync-and-derived-graph-projections.md`
- **Reviewed server baseline:** `dev` at `7b48bcb04fe416ed34e5e1d8a83a1290ccbe49ce`

## Purpose

Make explicit note-to-note manual relationships first-class Sync v2 state while
keeping wikilinks, backlinks, graph summaries, and orphan reports deterministic
local projections. Complete the interaction between synchronized links and the
existing conflict-aware note trash/restore lifecycle without duplicating link
history or exposing one owner's graph data to another.

Focused baseline verification on the isolated task worktree passed before design
changes:

```text
66 passed, 21 warnings in 53.40s
```

The command covered the existing graph service/parser, graph edge API, and note
restore suites.

## Approved product boundaries

- `notes.link` covers explicit note-to-note edges only.
- Version 1 supports the existing `manual` edge type only.
- Wikilinks and backlinks are derived from note bodies and are never Sync domains.
- Source note, target note, type, and direction are immutable for an edge identity.
  Retargeting is a tombstone plus a new edge.
- Note trash does not tombstone incident links. Links remain durable but hidden
  while either endpoint is deleted, and become visible again on restore.
- `dataset_id` is optional. Active Sync resolves omission to the active
  default-personal Notes dataset; inactive Sync preserves legacy behavior when it
  is omitted.

## Goals

1. Advertise `notes.link` adapter version 1 with `upsert` and `tombstone`.
2. Preserve stable identity, endpoint semantics, mutable presentation data,
   ownership, optimistic ancestry, and lifecycle state across SQLite and
   PostgreSQL.
3. Route active-Sync REST mutations through canonical envelopes and retain the
   legacy product path when Sync is inactive.
4. Make duplicate creation, concurrent editing, delete/update, and
   recreate/restore races idempotent or reviewable without silent replacement.
5. Produce live-only, bounded, paginated graph, backlink, and orphan views from
   explicit links and deterministic derived projections.
6. Preserve tenant isolation under shared PostgreSQL and fail closed during
   migration when legacy ownership or relationship integrity is ambiguous.

## Non-goals

- Polymorphic links to conversations, messages, media, or external resources.
- Arbitrary user-defined relationship types.
- Synchronizing wikilinks, backlinks, orphan flags, graph summaries, projection
  queues, or graph revisions.
- Field-level automatic merge of concurrently edited edge properties.
- Permanent hard purge of notes or link audit history.
- A new background daemon, cursor-signing secret, distributed transaction, or
  client-side implementation.

## Canonical product model

ChaChaNotes schema version 58 extends `note_edges` in place. The canonical columns
are:

| Column | Rule |
| --- | --- |
| `edge_id` | Stable canonical lowercase UUIDv4; immutable primary identity |
| `user_id` | Authenticated owner derived from request/dataset authority |
| `from_note_id`, `to_note_id` | Canonical note UUIDs; immutable |
| `type` | Constant `manual` in version 1 |
| `directed` | Immutable boolean |
| `weight` | Mutable finite number in the inclusive range 0–1,000,000 |
| `label` | Mutable optional string, at most 256 Unicode characters |
| `properties` | Mutable bounded canonical JSON object |
| `version` | Positive product version; changes once per real mutation |
| `deleted`, `deleted_at` | Soft-delete lifecycle state |
| `created_at`, `last_modified`, `created_by` | Stable creation and mutation provenance |

Undirected endpoints are stored in canonical UUID string order. Directed endpoints
retain their source/target order. The unique logical identity
`(user_id, type, directed, from_note_id, to_note_id)` remains reserved across
tombstones so a recreated edge cannot silently acquire a second identity.

Both endpoint identities must exist and belong to the same owner. Public create
and retarget operations require both endpoints to be live, preserving the existing
REST contract. Historical Sync materialization may accept an owned soft-deleted
endpoint so an explicit link arriving after note trash remains durable but hidden.
Product reads and materializers validate ownership explicitly even on PostgreSQL,
where forced RLS provides defense in depth. The `note_edges` RLS `USING` and
`WITH CHECK` predicates require both `note_edges.user_id` and the `client_id` of
each referenced note to equal the authenticated database owner; note deletion state
does not affect this ownership check. This prevents a direct SQL writer from
creating a cross-owner edge even when it knows both IDs. Public soft-delete/restore
remains the supported note lifecycle; a future hard-purge workflow must deal with
incident link records explicitly.

Properties are limited to a JSON object with at most 64 top-level keys, depth 4,
and 16 KiB after canonical UTF-8 encoding. Non-finite numbers and non-object roots
are invalid; object keys are strings and the parsed mapping is canonicalized before
fingerprinting. A note may contain at most 1,024 distinct derived wikilink targets;
additional first-occurrence-order targets are marked truncated in local projection
state rather than blocking the note write.

## Version 58 migration

The migration is transactional and non-destructive:

1. Lock the schema-version authority.
2. Lock `notes`, `note_edges`, `chacha_keywords`, `note_keywords`, and
   `conversations` in a fixed
   backend-specific order before global inspection or DDL. The keyword tables are
   included because v58 installs graph-revision triggers on those writer surfaces.
3. On PostgreSQL, catalog-verify relation/schema ownership and RLS state. Temporarily
   remove FORCE RLS only where schema-owner global validation requires it, then
   restore the exact prior forced set before the version bump.
4. Compute the exact post-transform row in memory and validate it before DDL:
   canonical edge and endpoint identities, same-owner endpoint existence, no
   self-links, logical uniqueness, weight in the inclusive 0–1,000,000 range,
   extracted label at most 256 Unicode characters, and remaining properties as an
   object within the 64-key/depth-4/16-KiB bounds. Any malformed, non-finite, or
   otherwise noncanonical legacy value fails closed.
5. Backfill existing rows with version 1, active lifecycle state, and
   `last_modified = created_at`.
6. Move a string `metadata.label` to `label` and remove that key from properties.
   Preserve every other metadata member. A non-string legacy `label` remains an
   opaque property while top-level label is null.
7. Add/backfill nullable columns before setting constraints/defaults on PostgreSQL;
   rebuild the table transactionally on SQLite.
8. Install owner/live indexes, logical uniqueness, portable checks, and forced RLS
   for explicit and derived graph tables.
9. Create one bounded full-rebuild state per PostgreSQL owner; SQLite initializes
   owner state lazily from authenticated runtime context.
10. Restore exact RLS state, install the complete policy set, verify fresh/upgraded
    parity, then set schema version 58.

Any invalid or ambiguous row aborts and rolls back the complete migration. The
migration never guesses ownership, rekeys an identity, deletes a row, or silently
repairs a relationship. PostgreSQL locks temporarily block affected Notes writes;
the exact lock and temporary-RLS operation requires explicit user approval before
implementation.

## Sync domain and payload

`notes.link` uses adapter version 1, `server_trusted_v1`, and operations `upsert`
and `tombstone`. Unknown payload fields are rejected.

The envelope `object_id` is the stable edge UUID. Ownership comes from the
authorized dataset and is not client-selectable. The upsert payload contains:

- `source_note_id`
- `target_note_id`
- `type: "manual"`
- `directed`
- `weight`
- `label`
- `properties`
- `created_at`
- `last_modified`
- `created_by`

Undirected Sync payloads must already contain endpoints in canonical UUID string
order; noncanonical client envelopes are rejected before append. The legacy REST
compatibility route may accept either order but normalizes it before constructing
and hashing the envelope.

Tombstones retain the complete canonical edge snapshot plus deletion time and a
stable reason of at most 256 Unicode characters. Labels and properties stay in the
protected payload and never enter routing metadata, conflict messages, or public
diagnostics. Standard envelope base cursor/revision/hash fields carry optimistic
ancestry.

`created_at` and `created_by` are immutable protected fields copied forward from the
current product row. On an ordinary client-origin create, `created_at` must exactly
equal the envelope's already-submitted normalized `created_at_client` after standard
Sync timestamp validation, and `created_by` must exactly equal the authenticated
device ID. Server-origin create uses the stable server-origin device identity and
the same pre-append `created_at_client` clock value. Trusted bootstrap is the sole
exception: it preserves the source-verified legacy creation values. No service
enrichment or rehash occurs after submission. `last_modified` must equal the same
envelope `created_at_client` for each mutation. The envelope `object_revision` is
the resulting product `version`. Materialization first compares version plus the
full canonical postcondition: an exact postcondition is success without a write, so
a crash after the product commit cannot change timestamps, increment version, or
advance graph revision again during repair.

Endpoint dependencies have two distinct states: identity existence and live
visibility. An existing tombstoned note satisfies link identity during Sync replay
but not a new public create request. A note planned earlier in the same immutable
mutation group also satisfies identity. A completely missing endpoint defers while
a bounded provider can exist, otherwise yielding a safe dependency conflict.

## Conflict and idempotency rules

- First creation requires no prior edge head.
- A mutable update requires the exact current head and unchanged immutable fields.
- Exact replay is idempotent and does not increment product version twice.
- Concurrent divergent mutable edits produce `notes_link_concurrent_edit`.
- Delete against a stale update produces `notes_link_delete_update_conflict`.
- Restore requires `restore_intent=true`, the same identity fields, and the exact
  current tombstone head.
- A different edge ID for an existing tombstoned logical identity produces
  `notes_link_restore_required`.
- Retarget, direction, or type changes produce `notes_link_identity_immutable`.

Logical-edge collisions are checked under the dataset projection guard and the
database unique constraint. If two devices append distinct object IDs for one
logical edge, the first projection wins and the second becomes a durable reviewable
conflict without overwriting it. Conflict records contain only safe identity,
version, cursor, and error-code data.

REST exact replay requires an idempotency key whose durable fingerprint binds the
dataset, owner, canonical request, submitted expected version, and desired mutable
state. Only a hash appears in routing. Without an idempotency key, duplicate create
retains the existing 409 behavior.

## Server-origin capture and projection

A focused link adapter, materializer, DB store, and thin capture/planning helper
reuse TASK-13003's atomic append, dataset projection lock, CAS, conflict recording,
repair, bootstrap, and ordered restore infrastructure. No new general coordinator
or concurrency framework is introduced.

`notes.link` has its own versioned readiness record,
`metadata.notes_link_v1`, rather than being added to the immutable six-domain
`notes_organization_v1` group. This avoids invalidating or re-running an already
ready organization bootstrap. The Sync profile/enrollment flow performs a resumable
upgrade for both new and already-ready default-personal datasets:

1. Under the dataset row lock, add `notes.link` to the domain set, create its domain
   state, and set `notes_link_v1` to `initializing` with a stable bootstrap ID.
2. Page the owner's existing current explicit-edge rows, including tombstones, in
   immutable edge-ID order. Append a source-verified upsert or tombstone envelope
   matching each row only after its `notes.note` identities are enrolled. Bootstrap
   does not reapply product rows. Tombstones are required because a user may mutate
   links in legacy inactive mode after v58 and enable Sync later; their reserved
   logical identities and restore/recreate history must not disappear at activation.
3. Resume the same bootstrap ID after interruption and verify the captured count and
   canonical fingerprints before transitioning `notes_link_v1` to `ready`.
4. Keep the six existing organization domains writable throughout this link-only
   upgrade. Link writes fail closed with a safe not-ready response until step 3.

A dataset marked organization-ready but lacking `notes_link_v1` is therefore an
upgrade candidate, not a permanently incomplete dataset. Old clients remain valid:
they do not need to request or push the new domain, and capability negotiation tells
new clients when it is available.

Active-Sync link routes preflight dataset readiness before product writes, require
`expected_version` for update, tombstone, and restore, append a canonical envelope,
materialize it, and return success only when the envelope is durably applied.
Creation has no expected version. Inactive Sync applies the same product invariants
directly; the existing unversioned delete compatibility call remains accepted only
on that inactive legacy path. Link rows and graph revision changes commit in the
same product transaction. Sync apply bookkeeping remains a separate crash-repairable
transaction; the design does not claim cross-database atomicity.

Note trash and restore retain the existing `notes.note` capture path. They emit no
link envelope and do not alter link versions. Graph visibility changes solely
through live endpoint joins. Bootstrap adds explicit link heads after endpoint note
identities but never includes derived graph state.

Restore preview and local inventory add `notes.link` to their explicit public
allowlists. A live link upsert depends on the identity of both `notes.note`
endpoints and is ordered after their selected providers, including a provider that
restores a deleted endpoint. A link tombstone has no live endpoint dependency.
Complete mutation groups remain immutable, contradictions or cycles fail closed,
and ordered actions retain dataset/group metadata so clients can execute the one
canonical cross-domain sequence.

## Deterministic derived projection

Version 58 adds local tables for derived wikilinks, coalesced dirty notes,
projection/rebuild status, and graph revision. PostgreSQL keys every row by owner
and protects it with explicit predicates plus forced RLS. SQLite is already one
authenticated user's database, so its queue/state keys are database-local rather
than inferred from legacy row provenance. None is a Sync domain.

The pure canonical-ID wikilink parser moves to a neutral Notes module. For normal
note-content writes, parsing occurs before the product transaction; the note row,
derived outgoing targets, queue-generation clear, and graph revision then commit in
one transaction. Database triggers coalesce unexpected/direct write changes by
`(owner, note_id)` and advance a generation counter.

Maintenance uses the repository's existing task/startup facility. PostgreSQL work
always runs under an explicit authenticated owner scope and claims bounded batches
with `FOR UPDATE SKIP LOCKED`; no unscoped global worker is allowed. SQLite uses its
database-local write transaction. A worker deletes a dirty row only when its
generation is still the claimed value. Rebuilds page through immutable note IDs.
Parser-version changes mark an owner/database for rebuild and advance the visible
graph revision only when that projection is current.

Projection rows may retain syntactically valid unresolved target UUIDs. Graph reads
join both endpoints against same-owner live notes, so a later-created target becomes
visible without reparsing. Self-wikilinks are omitted. A truncated source exposes a
safe projection warning.

Read endpoints never perform maintenance writes. Manual-only graph queries remain
available while derived projection is pending. Wikilink, backlink, and orphan
queries return a retryable projection-rebuilding response until the owner's global
projection is current.

## Graph and lifecycle APIs

Existing graph, neighbor, create-link, delete-link, trash, note-delete, and
note-restore routes remain compatible and gain optional `dataset_id`. Additive APIs
provide owner-scoped link list/detail, mutable PATCH, explicit link restore, and
`/notes/graph/orphans`.

The canonical product and projection tables are owner-scoped rather than
dataset-scoped. Consequently this feature authorizes exactly the user's single
active default-personal Chatbook dataset. A supplied `dataset_id` must identify that
same canonical dataset; another same-owner personal/workspace dataset is rejected
without inspecting or aliasing its history.

Dataset resolution follows these rules:

- active Sync + omitted ID: active default-personal Notes dataset;
- active Sync + supplied ID: the same active default-personal Notes dataset, with
  reserved Notes metadata and the ready `notes_link_v1` domain;
- inactive Sync + omitted ID: legacy product behavior;
- inactive Sync + supplied ID: safe conflict rather than silent ignore.

Missing and unauthorized datasets are indistinguishable. Dataset authorization is
always performed independently of cursors.

Link and orphan listings use immutable-ID keyset ordering, default limit 50, and
maximum 200. Full properties are returned only by the owner-authorized link detail
route. Graph/list responses return bounded summaries. Legacy `metadata` is accepted
and returned only on the existing create compatibility path; disagreeing legacy
and canonical labels are rejected.

Graph cursors are versioned untrusted pagination hints. The encoded cursor is at
most 8 KiB and its decoded JSON object is at most 4 KiB. Cursors bind
hashes of dataset and normalized request, the parser version, owner graph revision,
and stable traversal keys. They are not authorization tokens and require no new
signing secret. Radius-two resume deterministically recomputes the bounded prefix at
the same revision. Malformed, mismatched, oversized, or stale cursors fail instead
of silently restarting.

Every graph-cache lookup resolves authorization and the current revision first; the
cache key includes canonical dataset ID, owner graph revision, parser/projection
version, and normalized request. No unrevisioned user/query cache entry may serve a
response after trash, restore, link, or projection changes.

The owner graph revision advances transactionally for every graph-visible change:
note create/content update/trash/restore, explicit-link lifecycle or presentation
change, and any keyword/source membership or resource change included by the
normalized graph request. Manual, wikilink, and backlink edges appear only when
their two note endpoints are live. Existing owner-scoped tag/source nodes and their
membership edges remain in the graph for live notes; this task does not remove or
change that compatibility surface.
An orphan is a live note with no live inbound or outbound manual/wikilink edge;
tags and sources do not affect orphan status. Backlinks are the reverse view of the
same wikilink projection, not separately persisted mutable state.

## Stable error contract

- **400:** malformed identity, properties, immutable-field request, or cursor
- **404:** owned resource/dataset absent, without cross-owner enumeration
- **409:** duplicate, restore-required, stale version/cursor, supplied dataset while
  Sync is inactive, readiness/incomplete/preflight/idempotency, or reviewable conflict
- **413:** bounded payload/request limit exceeded
- **428:** active-Sync mutation missing `expected_version`
- **503:** transient append/materialization/projection busy or derived projection
  rebuilding, with safe retry guidance

Raw database errors, labels, properties, content, local integer IDs, filesystem
paths, and tenant details never appear in public errors or conflict evidence.

## Verification strategy

Implementation follows red-green-refactor for each seam. Required evidence covers:

- SQLite and server-free PostgreSQL fresh/v57-to-v58 migration parity, lock/RLS
  ordering, rollback stages, and concurrent initializer serialization;
- optional live PostgreSQL migration and two-owner RLS behavior when configured;
- product store and adapter vectors for every lifecycle/conflict rule;
- barrier races for duplicate creation and concurrent edit/delete;
- real materializer, bootstrap, repair, restore-preview, and conflict resolution;
- active, inactive, not-ready, exact replay, and no-incomplete-success server capture;
- note trash/restore with unchanged link heads/versions and deterministic graph and
  orphan visibility;
- direct-write recovery, queue generations, unresolved targets, parser upgrades,
  truncation, and bounded rebuild;
- live-only graph joins, global backlinks, orphan semantics, dataset authorization,
  stale/mismatched cursors, and radius-two resume;
- query plans, query counts, hard caps, dense/high-degree fixtures, and bounded
  projection backlog behavior rather than fragile wall-clock thresholds; and
- existing Notes graph/lifecycle, Sync v2, capture, restore/repair, migration, and
  PostgreSQL contract regression suites.

Touched/new Ruff and formatting scopes, Bandit production scope, compile checks, and
diff checks are recorded exactly. Known whole-file legacy baselines are reported
separately and never described as green. Independent spec review precedes planning;
independent correctness and security review precede completion.

## ADR check

```text
ADR required: yes
ADR path: Docs/ADR/037-canonical-notes-link-sync-and-derived-graph-projections.md
Reason: TASK-13004 changes the durable Notes schema, Sync domain and conflict
        contract, tenant ownership/RLS boundary, public lifecycle API, and the
        authority boundary between canonical and derived graph state.
```
