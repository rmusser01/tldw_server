# Notes Moodboard and Studio Sync Design

**Status:** Proposed; independently reviewed, requester approval pending
**Date:** 2026-08-24  
**Task:** `TASK-13007`  
**Delivery tasks:** `TASK-13007.1` through `TASK-13007.4`  
**Governing ADRs:** `Docs/ADR/031-notes-capability-sync-domains.md`,
`Docs/ADR/034-durable-server-origin-sync-mutation-batches.md`, and proposed
`Docs/ADR/040-synchronized-moodboards-and-studio-authority.md`

## Summary

Synchronize Notes moodboards, explicit note placements, and accepted persisted
Studio documents as three independently versioned Sync v2 domains:

- `notes.moodboard`
- `notes.moodboard_note`
- `notes.studio_document`

The existing product tables remain authoritative. Moodboards gain portable UUID
identity while preserving their integer REST identifiers. A placement remains
unique for one `(moodboard, note)` pair and uses a deterministic relationship
identity. A Studio document remains exactly one sidecar per note and uses the note
UUID as its object identity.

Only explicit manual placements synchronize. Smart-rule matches remain derived
from synchronized source state on each replica. Only accepted state that was
successfully persisted enters the Studio domain. Generation requests, prompts,
failed results, previews, credentials, and other transient operation state never
become synchronized objects.

All three domains are implemented dormant first. Moodboard and placement readiness
and writable advertisement are coupled. Studio readiness and writable
advertisement are independent, but require `notes.note`. No public capability is
advertised until storage, capture, bootstrap, repair, conflict, security, and live
PostgreSQL evidence are complete.

## Goals

- Preserve moodboard identity, description, smart rule, canvas configuration, and
  lifecycle across SQLite and PostgreSQL replicas.
- Preserve one explicit placement per `(moodboard, note)`, including position,
  size, order, display metadata, and tombstone/restore state.
- Preserve one accepted Studio sidecar per note, including its source binding,
  structured rendering state, document type, revision lineage, and accepted-output
  provenance.
- Keep note title and Markdown content authoritative in `notes.note` while making
  note-plus-Studio saves durable ordered mutation groups.
- Make server-origin and client-origin mutations use the same validation,
  canonical envelope, materialization, repair, and conflict boundaries.
- Make legacy migration, enrollment, bootstrap, pagination, RLS, and rollback
  behavior explicit and testable.

## Non-goals

- Synchronizing computed smart-board membership.
- Allowing the same note to appear more than once in one moodboard.
- Introducing multiple Studio documents per note or a Studio document hierarchy.
- Introducing an append-only Studio revision-history product model.
- Synchronizing generation prompts, transient previews, provider credentials,
  failures, or raw unaccepted model responses.
- Synchronizing device-local canvas viewport, selection, hover, focus, or zoom UI
  state.
- Adding Sync payload chunking for oversized Studio state.
- Making Sync and ChaChaNotes product projection a distributed transaction.
- Defining physical erasure or retention policy beyond the existing Sync retention
  boundary.

## Current state and constraints

### Moodboards

The current `moodboards` product table has an integer primary key, name,
description, smart-rule JSON, timestamps, soft-delete flag, `client_id`, and
product version. It has no portable resource UUID, dataset binding, canonical
revision/hash, or canvas state.

The current `moodboard_notes` table stores only `(moodboard_id, note_id,
created_at)`. Unpinning physically deletes the relationship. It has no portable
relationship identity, lifecycle state, concurrency version, canonical lineage,
or layout metadata.

The current hybrid list combines explicit placements with smart-rule matches.
Smart matches are queries over mutable Notes and conversation state. The accepted
schema includes notebook collection IDs, but the current query builder does not
apply that filter. Local integer collection IDs are not portable.

### Studio

The current `note_studio_documents` sidecar is keyed by `note_id` and contains
structured payload, template and handwriting choices, optional source-note and
excerpt bindings, diagram manifest, companion hash, render version, and
timestamps. It has no explicit owner/dataset scope, product version, canonical
revision/hash, lifecycle flag, or accepted-output provenance.

The table is currently created by a runtime ensure helper as well as by the wider
ChaChaNotes schema lifecycle. TASK-13007 makes the versioned ChaChaNotes migration
the only schema authority; the runtime helper becomes compatibility verification
or delegates to that authority.

Studio is an aggregate product surface. The normal Notes row remains authoritative
for title and Markdown content. The sidecar remains authoritative for Studio
rendering state. TASK-13007 must not duplicate these authorities merely to make a
single envelope self-contained.

### Existing Sync guarantees

ADR-031 requires independently mutable Notes capabilities to use independent
versioned domains. ADR-034 atomically appends a complete server-origin mutation
plan, then projects it in ordered, resumable steps under a dataset materialization
guard.

ADR-034 does not provide a distributed product transaction. Product projection
may commit an applied prefix before a later step fails. The canonical plan remains
complete and immutable, later projection is blocked, and idempotent repair resumes
at the first unapplied step. API success therefore means every step was applied,
not merely that the canonical plan was appended.

The existing `notes.note` payload does not carry a portable modification time and
ordinary note projection currently falls back to a replica-local clock. That is
insufficient for a convergent smart rule with an `updated` predicate. TASK-13007.2
therefore standardizes server-bound `canonical_modified_at` routing metadata for
every accepted `notes.note` envelope. Clients cannot choose or override it. New
accepted mutations use the server acceptance time; trusted legacy bootstrap uses
the strictly normalized source `last_modified`; old accepted envelopes without the
field use their immutable `received_at_server` value. The complete mutation plan
and exact-retry check cover the chosen value, and every note materializer projects
it instead of a local clock. Readiness remains blocked until existing note heads
and product rows have been verified or repaired to that portable value.

## Authority and identity

### Product authority

The existing product tables remain canonical projection stores:

| Sync domain | Product authority | Portable object identity |
| --- | --- | --- |
| `notes.moodboard` | `moodboards` | New canonical lowercase UUIDv4 `sync_id` |
| `notes.moodboard_note` | `moodboard_notes` | Deterministic relationship hash of moodboard `sync_id` and note UUID |
| `notes.studio_document` | `note_studio_documents` | The Studio note UUID |

Integer moodboard IDs remain local REST compatibility keys and are never placed in
canonical envelopes. Materializers resolve portable IDs to local integer rows.

### Placement identity

One note has at most one placement in a moodboard. The v1 relationship identity is
the lowercase SHA-256 digest of canonical UTF-8 JSON with this semantic shape:

```json
{
  "domain": "notes.moodboard_note",
  "members": ["<moodboard-sync-id>", "<note-id>"],
  "schema_version": 1
}
```

The public object ID is namespaced consistently with existing Notes relationship
domains. Clients cannot choose an unrelated placement ID. The adapter recomputes
and verifies it from the payload members.

### Studio identity

A Studio sidecar is one-to-one with its Notes row. `object_id`, payload `note_id`,
and envelope `parent_id` all bind to the same canonical note UUID. An optional
`source_note_id` identifies the source of a derived document; it does not change
the Studio object's identity.

New or changed Studio state with `source_note_id` requires a known live source note
in the same owner and dataset scope. An existing valid reference remains retained
if that source is later tombstoned and becomes usable again on restore; new
derivation from a tombstoned source is rejected. Unknown and cross-scope references
use the same non-enumerating error. Legacy unknown, ambiguous, or cross-scope
references receive a bounded diagnostic and block readiness rather than being
discarded or guessed.

### Ownership and dataset scope

Ownership is server-bound and excluded from canonical user payloads. Product rows
carry direct `owner_user_id` and `dataset_id` scope for authorization, bounded
queries, and PostgreSQL RLS. Existing `client_id` remains product attribution and
is not overloaded as portable resource identity.

Direct scope on placements and Studio rows must match their moodboard/note parents.
Composite database constraints are used where the parent schema supports them;
otherwise the product store verifies the parent in the same transaction. RLS alone
is not considered a scope-consistency constraint.

## Canonical v1 payloads

Outer domain contracts are strict and reject unknown fields. Canvas and placement
display metadata are bounded canonical JSON extension maps with explicitly allowed
scalar/list/object value types. Studio state is stricter: `payload_json`, diagram
manifest, and provenance use closed schemas versioned by `render_version`; unknown
fields are rejected recursively.

Studio `payload_json` v1 permits exactly `sections`; each section permits bounded
`id`, `kind`, `title`, and exactly `items` for `cue` or `content` for
`notes`/`summary`. Note title is injected from the current or planned `notes.note`
authority when rendering. Source note, template, handwriting mode, and render
version are injected from their outer Studio fields. Nested `meta` and `layout`
cannot become competing authorities.

The diagram manifest permits only the documented diagram type, selected section
IDs, closed canonical source graph, diagram text, format, status, render hash, and
a server-produced sanitized cache. Client caches are rejected or deterministically
rebuilt. Provider output is first reduced to these accepted product schemas; raw
provider dictionaries are never copied into canonical state. Legacy unknown fields
are diagnosed and block readiness instead of being silently retained or dropped.

All nested validators also reject excessive depth/counts, invalid UTF-8,
non-finite numbers, unsupported versions, secret-pattern fields, and values beyond
the configured envelope limit. The closed shape, rather than key-name heuristics
alone, enforces the transient/secret boundary.

### `notes.moodboard`

```json
{
  "moodboard_id": "canonical-uuid-v4",
  "name": "Board name",
  "description": null,
  "smart_rule": {
    "query": null,
    "keyword_tokens": [],
    "collection_sync_ids": [],
    "sources": [],
    "updated": {"after": null, "before": null}
  },
  "canvas": {
    "layout_mode": "masonry",
    "metadata": {}
  }
}
```

`smart_rule` is nullable. Set-like values are Unicode-normalized, deduplicated,
and sorted by their canonical comparison key. Timestamps are normalized to UTC.
Equivalent smart rules therefore hash identically.

The canonical contract uses collection `sync_id` values. REST schemas continue to
accept local integer collection IDs and translate them at the boundary. Unknown,
ambiguous, or cross-tenant references fail closed.

Canvas metadata belongs to the board and synchronizes. Device viewport, current
selection, hover state, temporary drag state, and other view preferences do not.

### `notes.moodboard_note`

```json
{
  "moodboard_id": "canonical-uuid-v4",
  "note_id": "canonical-note-uuid-v4",
  "x": 0,
  "y": 0,
  "width": 320,
  "height": 220,
  "order_index": 0,
  "display": {}
}
```

Coordinates, dimensions, and ordering are bounded JS-safe integers. Width and
height are positive. Sorting uses `(order_index, object_id)` so equal order values
remain deterministic and moving one card never requires renumbering every sibling.
Freeform coordinates are ignored while the moodboard uses masonry layout.

Smart-only matches never receive fabricated placement payloads. A note that is both
manual and smart uses the explicit placement payload and reports membership source
`both` through the existing hybrid product API.

### `notes.studio_document`

```json
{
  "note_id": "canonical-note-uuid-v4",
  "source_note_id": null,
  "payload_json": {"sections": []},
  "template_type": "lined",
  "handwriting_mode": "accented",
  "excerpt_snapshot": null,
  "excerpt_hash": null,
  "diagram_manifest_json": null,
  "companion_content_hash": "sha256:...",
  "render_version": 1,
  "note_revision": 4,
  "note_hash": "sha256:...",
  "accepted_provenance": {
    "kind": "manual",
    "attestation": "server",
    "provider": null,
    "model": null,
    "accepted_at": "2026-08-24T00:00:00Z",
    "source_revision": null,
    "source_hash": null,
    "result_hash": "sha256:..."
  }
}
```

`note_revision` and `note_hash` bind the sidecar to the accepted `notes.note` head.
They do not duplicate the note title or Markdown body. An accepted save that changes
both objects binds to the planned new note revision/hash. A sidecar-only change
binds to the currently applied note head. Note lifecycle delete/restore groups are
the deliberate exception: they preserve the complete prior Studio payload and its
binding until a later accepted Studio save.

Server AI provenance is stamped from the provider and model actually executed.
Client-origin AI provenance is marked `client_declared` and is bound to the
authenticated device by server routing metadata; it cannot claim server
attestation. Server-attested `accepted_at` is server-stamped. A client-declared
timestamp is strictly normalized but remains explicitly untrusted; the immutable
server receipt time is recorded separately in routing metadata. Legacy rows use
`trusted_bootstrap_v1`. Manual changes carry null provider/model values. Restore
preserves the prior accepted provenance rather than claiming that restore generated
the content.

No provenance object may contain prompts, authorization values, credentials,
tokens, raw unaccepted output, or arbitrary request metadata.

### Revision and hash rules

All three domains use whole-object compare-and-swap lineage:

- positive canonical object revision;
- exact base cursor, base revision, and base hash for updates and tombstones;
- explicit `restore_intent` for resurrection;
- canonical object hash over adapter version, domain, identity, lifecycle, complete
  normalized payload, and revision; and
- exact retry returns the existing accepted envelope while a changed retry identity
  is an idempotency conflict.

Tombstones carry the complete last accepted payload. This preserves placement
layout and Studio state for deterministic restore and permits a fresh replica to
materialize retained children beneath a deleted parent. Tombstone does not mean
physical erasure; existing append-only Sync history already retains earlier
payloads.

An exact semantic no-op returns current state without appending another envelope or
incrementing canonical revision.

## Product storage

### Moodboards

The moodboard schema retains local `id`, REST `version`, timestamps, `client_id`,
and soft-delete behavior, and adds:

- `owner_user_id`
- `dataset_id`
- `sync_id`
- `canonical_revision`
- `canonical_hash`
- `canvas_json`
- bounded source diagnostic fields when legacy conversion is not canonical

Scoped uniqueness is enforced for `(owner_user_id, dataset_id, sync_id)`.

### Supporting note projection

`notes` stores or exposes the server-bound `canonical_modified_at` used by portable
updated rules plus versioned NFC/casefold search projections for title/content.
Keyword and conversation-source authorities receive the equivalent versioned
normalized comparison values where they do not already have them. These are
derived query projections, not new canonical user fields. Materialization updates
them transactionally with their product authority; migration/backfill is bounded,
resumable on PostgreSQL, and verified before moodboard readiness.

### Placements

`moodboard_notes` becomes a first-class soft-deletable placement row with:

- direct owner and dataset scope;
- local moodboard integer ID and note UUID;
- position, dimensions, order, and display JSON;
- product version and timestamps;
- deleted lifecycle state; and
- canonical revision/hash.

The scoped `(owner_user_id, dataset_id, moodboard_id, note_id)` key preserves the
approved uniqueness. Unpin updates this row to a tombstone instead of deleting it.
Repin restores the same relationship identity.

### Studio sidecars

`note_studio_documents` retains its existing sidecar fields and adds:

- direct owner and dataset scope;
- product version;
- canonical revision/hash;
- deleted lifecycle state; and
- accepted provenance JSON.

Soft note deletion retains the row and payload. Active-domain hard deletion cannot
silently cascade it away.

### PostgreSQL RLS

All three tables enable and force RLS. Owner policies include both `USING` and
`WITH CHECK` predicates. Materializers, bootstrap, repair, and compatibility paths
set the expected owner context. Tests use a non-table-owner role and prove that
cross-owner reads, writes, relationship injection, and same-owner wrong-dataset
access fail.

Catalog verification checks exact columns, types, nullability, constraints,
foreign keys, indexes, RLS flags, and policy predicates. Existing-but-drifted
objects fail closed rather than being treated as installed.

## Smart-rule boundary

Only explicit placements synchronize. Smart matching is a deterministic local
projection over synchronized state.

Canonical rules translate local notebook collection IDs to collection `sync_id`
values. The product query builder must implement collection filtering before the
domain can become ready. Keyword-token matching requires the Notes organization
group to be ready. Source filtering depends on conversation source data; a board
with a source filter requires compatible `chat.conversation` state and verified
conversation references. If that dependency is absent, enrollment blocks with a
stable privacy-safe reason instead of claiming convergent smart results.

Smart matching v1 is backend-independent. Its compatibility identifier is
`nfc-casefold-ucd-<runtime-unicode-data-version>-v1`, so NFC/casefold behavior and
the exact Unicode Character Database version are one readiness contract. Text is
normalized with that algorithm before storage or comparison. `query` is a literal
substring of normalized title or content; keyword tokens are literal substrings of
normalized keyword values and match with OR semantics; sources are exact normalized
source values; collection IDs match membership in any listed collection; and
non-empty filter categories combine with AND semantics. `%`, `_`, and other SQL
metacharacters have no special meaning. Updated bounds are inclusive RFC 3339 UTC
comparisons against the server-bound `canonical_modified_at` described above. No
rule evaluation relies on backend `LOWER`, locale collation, or wildcard `LIKE`
behavior.

Candidate discovery uses owner/dataset-scoped relationship, source, modified-time,
and note-ID indexes with bounded keyset pages. Literal Unicode matching is applied
to the stored portable normalized values in application code. Enrollment records
the complete compatibility identifier; a server or device with a different Unicode
data version cannot advertise the moodboard pair as writable. Cross-runtime
conformance vectors cover normalization and casefold edge cases. SQLite/PostgreSQL
Unicode, wildcard, timestamp-boundary, collection, keyword, and source parity is a
required activation matrix.

Known tombstoned collection identities may remain in a rule and produce no match.
Unknown or cross-scope identities are invalid. Smart results may temporarily differ
while their synchronized dependencies are still applying, but the moodboard pair
is not writable or advertised until prerequisite readiness is satisfied.

Smart recomputation creates no Sync envelopes. A smart-only note becomes a
synchronized placement only after an explicit pin/place action.

## Mutation flows

### Active server-origin mutations

When a domain is inactive, existing direct product behavior remains compatible.
When enrollment has enabled fail-closed capture, REST mutations use this sequence:

1. Resolve the sole owner-scoped default-personal dataset and readiness state.
2. Validate request data, product version, canonical base, dependencies, payload
   size, and optional idempotency identity.
3. Build the complete singleton or ordered mutation group.
4. Atomically append the complete canonical plan under dataset append authority.
5. Materialize it in order under the dataset projection guard.
6. Reload product postconditions and return success only if every step applied.

A retryable projection failure returns a safe `503` with the durable group identity
and retryability. A reviewable conflict returns `409`. The product mutation is
never performed first with best-effort capture afterward.

### Client-origin mutations

Client pushes use the same domain contracts and product materializers. Exact base
lineage is authoritative. A sidecar-only Studio envelope is accepted normally. A
client change that also changes note title/content is submitted as one Studio
compound command: its strictly validated routing intent contains the complete
`notes.note` operation, payload, base revision/hash, and lifecycle intent. This
command is not appended as canonical state. The server validates the complete
overlay and deterministically expands it into primitive `notes.note` then
`notes.studio_document` envelopes, analogously to the existing task expansion.

The synthesized group lookup identity is stable for dataset, authenticated device,
and client envelope identity. Its stored plan hash is compared separately: exact
replay returns the prior outcome and changed intent under the same lookup identity
conflicts. Append is all-or-none, group fields remain response-only, and clients
cannot inject group IDs, step numbers, server timestamps, or attested provenance.
A separate note envelope that overlaps the same compound command in one push is
rejected as ambiguous.

In TASK-13007.4, a client `notes.note` tombstone or explicit restore for a note that
has a retained Studio sidecar is likewise treated as a lifecycle command and
expanded server-side into the ordered note-plus-Studio group. A normal note upsert
outside Studio remains a singleton and may intentionally make an existing Studio
binding stale.

Public push results retain existing Sync semantics:

- contract failures appear in `rejected[]` with stable error codes;
- dependencies that can be satisfied by ordering are retryable rejections;
- stale accepted bases and materialization collisions appear in `conflicts[]`; and
- request-level HTTP errors are not substituted for per-envelope outcomes.

### Moodboard lifecycle

- Create allocates one random UUIDv4 and binds it to the accepted idempotency
  record; replay returns that stored identity rather than deriving a non-v4 UUID.
- Update changes the whole canonical board under exact base lineage.
- Delete creates a whole-object moodboard tombstone.
- Restore requires explicit restore intent and the exact tombstoned base.
- Deleting a board does not fan out over every placement. Placements remain retained
  and hidden; restoring the board reveals them.

### Placement lifecycle

- Pin creates a placement or exactly replays an existing live placement.
- Layout patch updates the whole placement under optimistic concurrency.
- Unpin creates a whole-object placement tombstone.
- Repin restores the same relationship identity.
- A new placement requires live same-scope board and note dependencies.
- An existing placement may remain retained beneath a tombstoned board or note but
  cannot be edited until the parent is restored.
- Soft note deletion hides placements without rewriting or tombstoning them.

### Studio lifecycle

Successful accepted Studio changes synchronize after validation and before direct
product mutation:

- a manual accepted sidecar save;
- a successful derive operation that persists a new note and sidecar;
- a successful regenerate operation that persists its note/sidecar result; and
- a successful diagram operation that persists a new sidecar manifest.

Generation requests remain operations. A failed generation, returned-only preview,
prompt, provider request body, or unpersisted suggestion creates no envelope.

A sidecar-only change is a singleton Studio envelope bound to the current note
head. A change that also modifies note title or Markdown uses an ordered group:

1. `notes.note`
2. `notes.studio_document`

Note delete and restore use the same ordering when a Studio sidecar exists. The
adapter evaluates the complete group overlay, so the child step can validate the
planned parent lifecycle. If delete projection stops after the note step, the
deleted parent safely hides the retained sidecar until repair resumes.

Delete and restore do not rewrite the Studio payload's prior `note_revision` or
`note_hash`. The Studio tombstone and restored upsert preserve the complete last
accepted sidecar payload and provenance, including any pre-existing stale binding.
Only a later accepted Studio save rebinds the document to the then-current or
planned note head. A retained stale binding is valid review state, not malformed
bootstrap state, and is surfaced through the existing Studio staleness boundary.

Standalone Studio deletion is not introduced in v1. A Studio tombstone is valid
only as the Studio step of the corresponding note lifecycle group.

## Concurrency, conflicts, and recovery

The v1 conflict policy is intentionally whole-object and reviewable:

- concurrent board metadata edits conflict on stale base;
- concurrent placement moves conflict rather than silently field-merging;
- equal placement order values remain deterministic through object-ID tie-breaking;
- a Studio change whose bound note head changed becomes a stale-document conflict;
- restoring a live object or updating a tombstoned object without restore intent is
  rejected; and
- cross-owner, cross-dataset, missing dependency, and identity-mismatch attempts
  fail closed.

The design does not introduce CRDTs, per-field merge, or a Studio revision-history
product. Existing Sync conflict records and overwrite/skip/duplicate resolution
remain the review boundary.

Materializers are idempotent and verify product postconditions before writing.
Product commit followed by Sync bookkeeping failure is repaired by exact replay.
An unresolved conflict blocks later canonical projection according to ADR-034.

## REST compatibility and API additions

Existing integer moodboard routes remain available. Responses add portable and
canonical fields without removing legacy fields.

Moodboard changes include:

- optional canvas state on create/update and canvas plus `sync_id` in responses;
- a restore route with optimistic concurrency;
- optional placement data on pin;
- a placement layout patch route;
- placement version preconditions for patch and unpin; and
- optional placement metadata on the existing hybrid note-list response.

The existing hybrid note-list ordering remains compatible. Internal bootstrap and
Sync scans do not reuse public offset pagination as canonical progress.

Studio persistence routes gain an expected Studio or note revision as appropriate.
When capture is active, server-generated creates such as moodboard creation and
derive-style Studio note creation require the repository-standard
`Idempotency-Key`; a missing key fails before UUID allocation, append, provider
execution, or product state change. Identity-stable create-like mutations such as
pinning one deterministic placement may omit the key and rely on exact identity
and base semantics, while supplying a key opts into exact response replay. When
Sync is inactive, missing new concurrency or idempotency metadata preserves legacy
behavior. Every active update/delete/restore still requires its exact revision and
hash precondition.

REST failures use stable safe mappings:

- `409` for version/base conflict, changed idempotency replay, incomplete domain
  group, or review-required state;
- `413` for canonical payloads beyond the active Sync envelope limit;
- `422` for malformed identity, layout, canonical payload, or provenance; and
- `428` for a missing required idempotency or revision/hash precondition; and
- `503` when a durable plan cannot fully project or required components are
  unhealthy/unavailable.

## Schema migration and legacy conversion

### Schema authority

The ChaChaNotes schema-version migration owns fresh creation and upgrade for all
three product tables. SQLite uses transactional table rebuild/copy/verify/rename
where required. PostgreSQL uses explicit lock and statement timeouts, bounded
keyset backfill, and exact catalog verification. Fault-injection checkpoints cover
each schema, copy, index, constraint, RLS, and version-advance boundary.

The Studio runtime ensure helper no longer emits a competing legacy definition.

### Local-unbound scope

Schema migration can prove owner identity but cannot assume a Sync dataset already
exists. Legacy rows therefore receive:

- proven `owner_user_id`; and
- the reserved noncanonical `dataset_id = "local-unbound"`.

A scope-authority table records at most one bound default-personal dataset per
owner. Explicit enrollment acquires the dataset fence, proves the complete owner
graph, and atomically rebinds it from local-unbound scope. Local-unbound rows support
inactive legacy behavior but can never enter canonical capture, bootstrap, or
writable capability state.

Placement ownership is derived from both board and note and must agree. Studio
ownership is derived from its note. Ambiguous or mismatched ownership aborts
migration rather than guessing.

### Legacy canonicalization

- Moodboards receive canonical UUIDv4 `sync_id` values.
- Valid moodboard product versions seed canonical revision; canonical hash includes
  normalized smart rule and default canvas state.
- Existing boards default to masonry layout.
- Placements begin at revision 1, use `(created_at, note_id)` as a documented stable
  pin-order approximation, receive default card dimensions, zero freeform
  coordinates, and empty display metadata.
- Existing hard-deleted link rows cannot be reconstructed and are not fabricated.
- Local collection IDs translate to collection `sync_id` values. Missing or
  ambiguous references produce bounded source diagnostics and block readiness.
- Studio sidecars begin at revision 1 with `trusted_bootstrap_v1` provenance,
  stored modification time, and a verified accepted-result hash.
- Legacy Studio nested title, source, and layout fields are removed only after
  exact comparison with the authoritative note and outer sidecar fields; a mismatch
  receives a diagnostic and blocks readiness instead of choosing either value.
- A sidecar whose parent note is already deleted bootstraps as tombstoned.

Malformed JSON, unsupported render state, invalid timestamps, duplicate portable
identity, oversized canonical state, or invalid references are retained with
privacy-safe diagnostic codes/hashes where possible and block readiness. They are
never silently dropped or rewritten into plausible state.

## Enrollment, bootstrap, and readiness

No global environment activation flag is added. Explicit dataset enrollment and
per-domain readiness are the rollout boundary.

Each readiness record uses the established state machine:

```text
not_enrolled -> enrolling -> bootstrapping -> verifying -> ready
```

Failure enters `blocked` with a privacy-safe reason, resume phase, last verified
keyset cursor, count, and fingerprint. Retry resumes from that cursor.

Before source scanning:

- enrollment is limited to the Chatbook default-personal dataset with the
  server-materializable `server_trusted_v1` encryption policy;
- moodboard and placement capture enable together;
- Studio capture enables independently;
- external device writes remain disabled; and
- REST writes pass through fail-closed prebootstrap canonical capture under the
  dataset materialization fence.

Bootstrap is non-destructive, source-verified, and resumable:

| Domain | Keyset order |
| --- | --- |
| `notes.moodboard` | indexed `(sync_id)` |
| `notes.moodboard_note` | indexed `(moodboard_sync_id, note_id)` |
| `notes.studio_document` | indexed `(note_id)` |

Placement relationship hashes are computed in application code, not in SQL scan
predicates. Each page records source count, cursor, and privacy-safe fingerprint.
Final verification rechecks the full bounded aggregate, accepted envelope heads,
and product postconditions. Source drift captures a verified correction or leaves
the domain blocked; it never marks stale state ready.

### Writable predicates

The moodboard pair is writable only when:

- `notes.note` and the Notes organization group are ready;
- the dataset uses the supported default-personal `server_trusted_v1` policy and
  all note heads have portable `canonical_modified_at` projection state;
- both moodboard domains are enrolled at supported adapter versions;
- source-filtered boards have verified compatible `chat.conversation` dependency
  state;
- both source scans and final verification are complete;
- capture, materializers, repair, scope authority, and RLS are healthy; and
- no malformed source, unresolved drift, oversized object, or projection blocker
  remains.

Studio is writable independently only when:

- `notes.note` is ready;
- the dataset uses the supported default-personal `server_trusted_v1` policy;
- Studio is enrolled at a supported adapter version;
- Studio bootstrap, parent binding, and same-scope source-note verification are
  complete;
- singleton and note-plus-Studio capture and repair are healthy; and
- no malformed source, oversized object, or projection blocker remains.

Public server-supported capabilities omit all three domains through
`TASK-13007.1`–`.3`. `TASK-13007.4` may advertise all supported adapter versions,
while each dataset writable map advertises the moodboard pair together or neither
and advertises Studio independently.

Once a domain has published canonical history, enrollment cannot silently remove
it or return to unsynchronized direct mutation.

## Payload limits

The current Studio API permits state larger than the default Sync envelope limit.
TASK-13007 does not add chunking. When a domain is active or capture-enabled:

- the exact canonical JSON byte size is measured before append or product write;
- an oversized REST mutation returns a stable `413`;
- client push returns a stable rejected outcome;
- bootstrap records an actionable privacy-safe oversized-object blocker; and
- readiness cannot become `ready` while any current object is oversized.

Inactive legacy behavior keeps its existing product limit. Activation documentation
must describe the stricter synchronized-object limit.

## Hard delete, retention, and rollback

Soft delete and explicit restore are the public lifecycle. After activation,
hard-delete paths for moodboards, placements, notes with Studio sidecars, or Studio
sidecars fail closed unless invoked through an existing authorized retention or
erasure workflow that understands canonical Sync history.

No destructive downgrade migration is provided.

- `TASK-13007.1`–`.3` are safe to revert while domains remain dormant and no
  canonical history has been published.
- After `TASK-13007.4` activates a dataset, an arbitrary older binary is unsafe
  because it cannot understand the new capture gates.
- Post-activation rollback requires a compatibility build that retains the new
  gates, or maintenance mode plus restoration of a pre-activation database state.
- Operators must never run a pre-TASK-13007 binary with writes enabled against an
  activated dataset.

These restrictions are part of the operator documentation and activation tests.

## Performance and bounded-query requirements

- Bootstrap uses indexed keyset pages, never unbounded table loads.
- Placement list and conflict reads are scoped by owner and dataset.
- Board and Studio lookups use scoped portable-identity indexes.
- Retained-child checks and hard-delete gates use bounded existence queries.
- Public legacy offset pagination remains compatible, but canonical bootstrap and
  repair progress never depends on offsets.
- SQLite `EXPLAIN QUERY PLAN` and PostgreSQL plan assertions prove expected indexes
  for source scans, dependency checks, pagination, and RLS-scoped lookups.
- Payload depth, field counts, collection counts, and metadata byte sizes are
  bounded before canonical serialization.

## Security and privacy

- Owner and dataset identity are server-bound.
- Portable IDs and relationship members are strictly validated.
- All SQL values are parameterized; only fixed internal identifiers appear in
  migration/catalog SQL.
- PostgreSQL tables force RLS and verify both read and write predicates.
- Client provenance cannot claim server attestation.
- Credentials, prompts, authorization values, tokens, and raw failed/unaccepted
  model outputs are forbidden from canonical payload and routing metadata.
- Readiness and error diagnostics expose bounded codes, counts, cursors, and hashes
  rather than note or Studio content.
- Tombstone and retention documentation does not imply physical erasure.

## Delivery decomposition

### `TASK-13007.1` — Contracts and storage

- create ADR-040 and this authoritative design;
- add strict dormant contracts, canonical hash helpers, catalog declarations, and
  readiness parsing;
- migrate and verify SQLite/PostgreSQL product storage, scope authority, lineage,
  RLS, and indexes;
- remove parallel Studio schema authority; and
- keep all three domains absent from public supported/writable capabilities.

### `TASK-13007.2` — Moodboards and placements

- implement dormant adapters, materializers, coordinator, capture, repair, and
  bootstrap for the coupled pair;
- add portable smart-rule translation and missing collection-filter behavior;
- standardize portable note modification time and backend-independent smart-match
  evaluation;
- add canvas/placement REST compatibility surfaces, restore, concurrency, and
  idempotency;
- prove smart-match exclusion and retained-child lifecycle; and
- keep the pair absent from public capabilities.

### `TASK-13007.3` — Studio documents

- implement the dormant Studio adapter, materializer, provenance validation,
  capture, repair, and bootstrap;
- bind Studio state to exact note heads;
- implement server-origin and client-origin accepted-save group synthesis and
  capture accepted persisted manual/derive/regenerate/diagram results;
- reject oversized and transient/secret-bearing state; and
- keep Studio absent from public capabilities.

### `TASK-13007.4` — Lifecycle integration and activation

- implement note-plus-Studio delete/restore lifecycle groups;
- complete enrollment, capture flags, readiness, and capability advertisement;
- gate hard delete and document post-activation rollback;
- prove two-client convergence, crash repair, conflicts, and pagination on SQLite
  and live PostgreSQL; and
- update OpenAPI, generated client types, operator docs, and close `TASK-13007`.

Each child is independently mergeable. The first three are dormant foundations.
The fourth is the first public writable state and cannot merge without the complete
security, repair, bootstrap, and live-backend evidence.

## Verification strategy

All implementation follows focused RED-GREEN-REFACTOR and fresh verification
before each commit and completion claim.

### `TASK-13007.1`

- fresh and upgrade migration on SQLite and live PostgreSQL;
- rollback fault injection at every migration boundary;
- exact columns, constraints, indexes, scope authority, foreign keys, catalog,
  forced RLS, and policy drift;
- legacy owner proof, local-unbound binding, malformed JSON, invalid identities,
  duplicate IDs, and oversized diagnostics;
- strict closed Studio contracts, canonicalization, tombstone, hash,
  `server_trusted_v1`, and readiness matrices;
- runtime Studio helper compatibility; and
- proof that no public capability exposes the domains.

### `TASK-13007.2`

- moodboard and placement create/update/tombstone/restore, exact retry, changed
  retry, stale base, deleted parent, and cross-scope matrices;
- deterministic relationship identity and integer layout bounds;
- smart-rule normalization, collection translation/filtering, keyword and
  conversation-source dependencies, portable note time, exact Unicode-data-version
  compatibility/conformance vectors, backend parity, and smart-match exclusion;
- bodyless legacy pin compatibility, placement patch, unpin tombstone, and restore;
- interrupted/resumed bootstrap, source drift, projection split repair, pull, ack,
  and bounded keyset scans; and
- SQLite plus live PostgreSQL RLS, pagination, and plan evidence.

### `TASK-13007.3`

- Studio strict outer payload and versioned nested JSON validation;
- legacy nested-title/source/layout equality, canonical removal, and mismatch
  blockers;
- note-head binding, stale note, exact retry, changed retry, tombstone, restore,
  and cross-scope matrices;
- same-scope live source-note acceptance plus retained tombstoned-source behavior;
- server-attested versus client-declared versus legacy provenance;
- accepted manual/derive/regenerate/diagram capture and proof that previews,
  failures, prompts, and credentials are excluded;
- client compound-command synthesis, overlap rejection, exact replay, changed
  intent, and all-or-none append;
- payload-limit preflight and readiness blocking;
- interrupted/resumed bootstrap and product/Sync split repair; and
- SQLite plus live PostgreSQL RLS and bounded plan evidence.

### `TASK-13007.4`

- complete note-plus-Studio group ordering, crash windows, repair, delete, and
  restore;
- board/note tombstones with retained placements and explicit placement tombstones;
- activation predicates, partial-domain failure, enrollment retry, readiness drift,
  and capability maps;
- hard-delete denial and post-activation rollback guard documentation;
- two-client concurrent layout and Studio revisions, deletion, restore, conflicts,
  pull, apply, replay, and repair;
- broad Notes and Sync regressions on SQLite and live PostgreSQL;
- Ruff, changed-scope Bandit, compilation, and `git diff --check`; and
- OpenAPI drift, generated client types, and final documentation checks.

Live PostgreSQL is a required gate in every implementation PR. An unavailable
suitable server blocks the PR rather than converting the evidence into an accepted
skip.

## Acceptance mapping

| `TASK-13007` criterion | Design coverage |
| --- | --- |
| Advertise three versioned upsert/tombstone domains | Dormant catalog plus final readiness-gated activation |
| Preserve moodboard and placement identity/layout/base | Separate whole-object domains with UUID/resource and deterministic relationship identity |
| Preserve Studio identity/source/title/content/type/revision | `notes.studio_document` owns sidecar state while ordered `notes.note` groups preserve title/content without duplicate authority |
| Capture server-origin mutations | Append-before-materialize coordinators for REST and accepted Studio operations |
| Keep generation requests transient | Accepted-persistence boundary and attested provenance contract |
| Idempotent or reviewable concurrent outcomes with bounded queries | Exact lineage, whole-object conflicts, dataset guard, keyset scans, RLS, and plan assertions |

## Resolved design choices

- One Studio sidecar per note; note UUID is the Studio identity.
- One placement per `(moodboard, note)`; duplicate placements are not introduced.
- Only explicit/manual placements synchronize; smart matches remain derived.
- Existing product tables remain authoritative; no Sync-owned shadow tables.
- Latest accepted Studio state synchronizes; no new product revision-history model.
- Note title and Markdown remain in `notes.note`; Studio does not duplicate them.
- Whole-object tombstones retain payload for deterministic restore.
- Moodboard pair activation is coupled; Studio activation is independent.
- Product projection is ordered and repairable, not distributed-transaction atomic.
- Client note-plus-Studio changes are server-expanded compound commands; clients do
  not supply mutation-group metadata.
- Updated smart rules use server-bound portable note time and versioned
  backend-independent matching semantics with an exact Unicode-data-version gate.
- Studio structured state uses closed render-versioned schemas without nested
  title, source, or layout authorities.
- Enrollment is limited to default-personal `server_trusted_v1` datasets.
- No Sync chunking is added for oversized Studio objects.
