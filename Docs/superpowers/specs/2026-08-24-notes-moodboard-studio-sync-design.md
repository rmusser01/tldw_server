# Notes Moodboard and Studio Sync Design

**Status:** Approved after independent re-review
**Date:** 2026-08-24
**Task:** `TASK-13007`
**Delivery tasks:** `TASK-13007.1` through `TASK-13007.5`
**Governing ADRs:** `Docs/ADR/031-notes-capability-sync-domains.md`,
`Docs/ADR/034-durable-server-origin-sync-mutation-batches.md`,
`Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md`, and proposed
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

All three domains are implemented behind dormant internal coordinators first. No
production capture or enrollment path can invoke them or publish canonical history;
isolated internal harnesses exercise bootstrap and repair. Earlier tasks may add
compatible product-query fields and routes, but product mutations remain on their
existing direct path until the activation task wires fail-closed capture.
Moodboard and placement readiness and writable advertisement
are coupled. Studio readiness and writable advertisement are independent, but
require `notes.note`. No public capability is advertised until storage, capture,
bootstrap, repair, conflict, security, and live PostgreSQL evidence are complete.

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
therefore implements dormant support for server-bound `canonical_modified_at`
routing metadata on accepted `notes.note` envelopes. Clients cannot choose or
override it. New accepted mutations use the server acceptance time; trusted legacy
bootstrap uses the strictly normalized source `last_modified`; old accepted
envelopes without the field use their immutable `received_at_server` value. The
complete mutation plan and exact-retry check cover the chosen value, and every note
materializer projects it into the existing product `notes.last_modified` column
instead of a local clock. No second canonical note-time field is introduced.

TASK-13007.2 does not stamp this metadata on production `notes.note` history;
isolated harnesses prove parsing/materialization and existing product smart views
remain local compatibility behavior. TASK-13007.5 first enables stamping after the
fleet and readiness gates pass, repairs/verifies existing note heads and product
rows, and permanently retains parsing/projection/fallback compatibility thereafter.

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

The public object ID is exactly
`notes.moodboard_note:sha256:<64-lowercase-hex-digest>`, matching the existing
Notes relationship namespace. Its envelope `parent_id` is exactly the moodboard
`sync_id`; dependencies name both that board and the note UUID. Clients cannot
choose an unrelated placement ID. The adapter recomputes and verifies it from the
payload members.

### Studio identity

A Studio sidecar is one-to-one with its Notes row. `object_id`, payload `note_id`,
and envelope `parent_id` all bind to the same canonical note UUID. An optional
`source_note_id` identifies the source of a derived document; it does not change
the Studio object's identity and cannot equal `note_id`. `excerpt_snapshot` and
`excerpt_hash` are null together or non-null together; a non-null excerpt requires
`source_note_id`, an exact source revision/hash in provenance, and exact membership
in that accepted source revision after line-ending normalization.

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

TASK-13007 evolves the existing schema-v60 `note_task_scope_authority` row into the
single owner-to-default-personal-dataset authority despite its legacy task-specific
name. It does not create a second scope-authority table. The existing key and
immutable `dataset_id` remain, and the row gains exact non-null Boolean
`task_graph_bound`, `moodboard_graph_bound`, and `studio_graph_bound` columns.
For additive mixed-version safety, `task_graph_bound` has DDL default `true` while
the two new-domain flags default `false`; new task binders also write true
explicitly. Existing TASK-13006 authority rows migrate with only
`task_graph_bound=true` after the migration verifies the complete task graph
already matches that dataset; inconsistency aborts instead of blessing drift.

Absence means every graph remains local-unbound. A graph enrollment transaction may
insert the immutable target with every flag supplied explicitly, rekey only its graph, and
set only its flag before commit; an empty graph still sets its flag. Each flag is a
one-way `false -> true` transition made in the same transaction as complete graph
verification/rekey and can never return to false. Same-dataset, already-bound replay
is idempotent and a different target is rejected. Task
compatibility callers resolve the authority dataset only when
`task_graph_bound=true`; otherwise they continue to resolve local-unbound state.
Moodboard and Studio callers obey their corresponding flags. The table retains
owner-only forced PostgreSQL RLS with exact `USING` and `WITH CHECK` predicates,
and catalog verification covers columns, constraints, flags, and policy definition.
Interleaved first-enrollment races by different graph units are serialized and
tested: one immutable dataset wins and a conflicting target rolls back without a
partial rekey.

Moodboard/Studio binders never rely on DDL defaults: they insert
`task_graph_bound=false` and both new-domain flags explicitly, then set only their
own flag in the same transaction. TASK-13007.5 is gated until no row-presence-era
server remains, so an old task caller can never observe a non-task-first authority
row and mistake it for task binding.

Direct scope on placements and Studio rows must match their moodboard/note parents.
Composite database constraints are used where the parent schema supports them;
otherwise the product store verifies the parent in the same transaction. RLS alone
is not considered a scope-consistency constraint.

## Canonical v1 payloads

Canonical v1 JSON is produced only after schema normalization. It is UTF-8 JSON
with object keys sorted lexicographically, compact separators `,` and `:`,
`ensure_ascii=false`, and `allow_nan=false`. Every schema key is fixed safe ASCII;
extension-map keys must match `[A-Za-z0-9_.-]{1,64}`. Booleans are not integers,
floating-point values are forbidden, and integers must be in
`[-9007199254740991, 9007199254740991]`. ASCII-only key ordering and integer-only
numbers make the exact serialization portable across Python and JavaScript without
a second canonicalization library. Strings that participate in set comparison or
literal matching are NFC-normalized as specified below; other user-visible strings
retain their accepted Unicode content. Hashes use exactly those bytes. Timestamps
are canonical RFC 3339 UTC with a `Z` suffix and no insignificant fractional zeros.
UUIDs are canonical lowercase RFC-4122 strings. Cross-runtime vectors cover key
ordering, Unicode string escaping, integer bounds, timestamps, and rejected floats.

Outer domain contracts are strict and reject unknown fields. Canvas and placement
display metadata are bounded canonical JSON extension maps with explicitly allowed
scalar/list/object value types. Studio state is stricter: `payload_json`, diagram
manifest, and provenance use closed schemas versioned by `render_version`; unknown
fields are rejected recursively.

The canonical Sync `payload_json` v1 permits exactly `sections`; each section
permits bounded `id`, `kind`, `title`, and exactly `items` for `cue` or `content`
for `notes`/`summary`. Note title is injected from the current or planned
`notes.note` authority when rendering. Source note, template, handwriting mode,
and render version are injected from their outer Studio fields. Nested `meta` and
`layout` cannot become competing canonical authorities. This sections-only shape
is a Sync contract, not a breaking change to the legacy REST representation; the
compatibility serializer and input rules are defined below.

The canonical diagram manifest permits only the documented diagram type, selected
section IDs, closed canonical source graph, diagram text, format, status, and
render hash. `cached_svg` and every other rendered cache are disposable local
projections: they are rejected from canonical input, excluded from object and
result hashes, and rebuilt per replica for compatible REST responses. Diagram text
and Mermaid are untrusted content and are escaped/sanitized at every render and
export boundary. Provider output is first reduced to these accepted product
schemas; raw provider dictionaries are never copied into canonical state. Legacy
unknown fields are diagnosed and block readiness instead of being silently
retained or dropped.

All nested validators also reject excessive depth/counts, invalid UTF-8,
floating-point numbers, unsupported versions, secret-pattern fields, and values
beyond the configured envelope limit. The closed shape, rather than key-name
heuristics alone, enforces the transient/secret boundary.

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
attestation. `accepted_at` is server-stamped for every newly accepted server or
client transition and is not client-selectable. Legacy bootstrap uses the strictly
normalized product modification time under `trusted_bootstrap_v1`. Manual changes
carry null provider/model values. Restore preserves the prior accepted provenance
rather than claiming that restore generated the content.

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

Content hashes are normative and always use the `sha256:<64-lowercase-hex>` form:

- `note_hash` is the exact accepted `notes.note.payload_hash` at
  `note_revision`; it is copied, never recomputed from a product row;
- `excerpt_hash` is SHA-256 over the UTF-8 bytes of the exact accepted
  `excerpt_snapshot` after converting CRLF and CR to LF, with null paired only
  with null;
- `companion_content_hash` applies the same line-ending normalization to the exact
  accepted Notes Markdown body;
- diagram `render_hash` hashes canonical JSON with exactly `diagram_type`,
  `context`, and `diagram` after CRLF/CR-to-LF normalization of the latter two
  strings; and
- provenance `result_hash` hashes canonical JSON for the accepted Studio
  content-bearing state: `note_id`, `source_note_id`, sections-only
  `payload_json`, template, handwriting mode, excerpt snapshot/hash, canonical
  diagram manifest without caches, companion content hash, render version, and
  note revision/hash. Lifecycle, envelope lineage, timestamps, and provenance
  itself are excluded so the field is not recursive.

Canonical diagram context visits selected sections in document order, appends each
non-empty trimmed title, then appends trimmed content (`cue` items joined by LF),
and joins all appended parts with LF. If no part exists it is exactly
`Notes Studio diagram`.

The canonical object hash remains separate. Its canonical JSON has exactly
`adapter_version`, `domain`, `identity`, `lifecycle`, `payload`, and `revision`.
Lifecycle is `live` or `tombstone`. Identity is exactly
`{"moodboard_id": ...}` for a board,
`{"moodboard_id": ..., "note_id": ..., "placement_id": ...}` for a placement,
or `{"note_id": ...}` for Studio; `placement_id` is the recomputed namespaced
relationship object ID. Payload is the complete normalized domain payload and
revision is a positive integer. The stored value is the SHA-256 of those canonical
bytes in the same prefixed form.

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

The existing `notes.last_modified` projects server-bound `canonical_modified_at`
for portable updated rules. Notes, keywords, and conversation-source authorities
also expose versioned NFC/casefold comparison values where they do not already
have them. These are derived query projections, not new canonical user fields.
Materialization updates them transactionally with their product authority;
migration/backfill is bounded, resumable on PostgreSQL, and verified before
moodboard readiness.

`moodboard_smart_matches` and its dirty/rebuild state are local disposable
projections modeled on the existing Notes graph dirty/rebuild pattern. Rows are
scoped by owner/dataset and identify board, note, board revision, algorithm ID,
dependency fingerprint, and completed generation. They are never product or Sync
authority, never appear in canonical envelopes, and may be dropped and rebuilt.

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

All tenant-carrying product, smart-match, and rebuild-state tables enable and force
RLS. Owner/dataset policies include both `USING` and `WITH CHECK` predicates;
`note_task_scope_authority` deliberately remains owner-only as described above.
Materializers, bootstrap, repair, rebuild, and compatibility paths set the expected
owner/dataset context. Tests use a non-table-owner role and prove that cross-owner
reads, writes, relationship injection, and same-owner wrong-dataset access fail.

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

Smart-rule evaluation never scans an owner's complete note set on a request.
Narrow relationship/rule changes mark one indexed board dirty; broad note or
normalization changes increment an owner/dataset dependency epoch rather than
updating every board. A bounded keyset scheduler discovers stale board fingerprints.
A bounded resumable worker then discovers candidates with owner/dataset-scoped
relationship, source, modified-time, and note-ID indexes, applies literal Unicode
matching to stored normalized values, and publishes a complete new projection
generation atomically only after its final keyset page. Every scheduler and rebuild
turn has explicit row and wall-clock budgets plus a durable continuation cursor.
Hybrid reads use only the current completed generation.

While no current generation exists, the existing hybrid endpoint returns manual
placements, `smart_results_complete=false`, `smart_projection_status="pending"`,
and `total=null`; it never labels a partial scan as complete. Once published it
returns `smart_results_complete=true`, an exact indexed total, and the projection
generation/revision. A rule or dependency change invalidates the old generation
for public completeness and schedules another bounded rebuild. This additive
status is the only temporary compatibility change; computed matches remain
derived and never synchronize.

Enrollment records the complete compatibility identifier; a server or device with
a different Unicode data version cannot advertise the moodboard pair as writable.
Cross-runtime conformance vectors cover normalization and casefold edge cases.
SQLite/PostgreSQL Unicode, wildcard, timestamp-boundary, collection, keyword,
source, worst-case sparse-query, count, and high-page parity is a required
activation matrix.

Known tombstoned collection identities may remain in a rule and produce no match.
Unknown or cross-scope identities are invalid. Smart results may temporarily differ
while their synchronized dependencies are still applying, but the moodboard pair
is not writable or advertised until prerequisite readiness is satisfied.

Smart recomputation creates no Sync envelopes. A smart-only note becomes a
synchronized placement only after an explicit pin/place action.

## Mutation flows

### Active server-origin mutations

When a domain is inactive, existing direct product behavior remains compatible.
The dormant coordinators delivered by `TASK-13007.3` and `TASK-13007.4` are
reachable only from isolated internal tests/materialization harnesses: they cannot
enable production capture or publish production history. `TASK-13007.5` is the
sole change that wires enrollment and fail-closed capture into production REST
mutations, which then use this sequence:

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
and client envelope identity. Before server expansion or timestamp/provenance
stamping, the service hashes and persists a separate canonical fingerprint of only
the validated client-controlled command fields. Replay recomputes that intent
fingerprint: equality returns the prior outcome and a changed command under the
same lookup identity conflicts. The separately stored plan hash verifies immutable
expanded-plan integrity only and is never used as an incoming-intent comparator.
Append is all-or-none, group fields remain response-only, and clients cannot inject
group IDs, step numbers, server timestamps, or attested provenance. A separate note
envelope that overlaps the same compound command in one push is rejected as
ambiguous.

In `TASK-13007.5`, a client `notes.note` tombstone or explicit restore for a note that
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

AI title suggestion and summarization remain operations. Accepting a suggested
title changes only `notes.note` unless the same accepted action also changes Studio
state, in which case it uses the ordinary compound-save boundary above.

For a note with a retained Studio sidecar, lifecycle behavior is explicit:

| Studio state | Server REST note delete/restore | External client lifecycle command |
| --- | --- | --- |
| Not enrolled; capture off | Preserve existing Notes behavior; retain/hide the local sidecar and create no Studio envelope. Later bootstrap represents an already-deleted parent with a Studio tombstone. | Existing `notes.note` behavior only; no claim of Studio synchronization. |
| Enrolling/bootstrapping with capture on | Prebootstrap capture appends and applies the complete note-plus-Studio group before returning. | Reject with retryable `notes_studio_bootstrap_incomplete` until verification finishes. |
| Ready and healthy | Expand and apply the complete group. A note with no sidecar remains a note singleton. | Accepted only when the device advertises the supported Studio adapter; otherwise reject with retryable `notes_studio_device_capability_required` rather than causing an unseen sidecar lifecycle change. |
| Studio readiness degraded but the canonical predecessor chain is clear | Fail the sidecar-bearing delete/restore with `503 notes_studio_lifecycle_unavailable`; an ordinary note upsert may proceed only if `notes.note` capture is healthy and may make Studio stale. | Reject the lifecycle command with retryable `notes_studio_lifecycle_unavailable`. |
| Any pending/failed/conflicting accepted predecessor exists | ADR-034's dataset barrier blocks later append/projection; permit only exact replay, repair, or explicit conflict resolution. | Reject later work with the established predecessor-blocked result; never bypass history ordering. |

Restore preserves the tombstoned sidecar payload and its old note binding. A
bootstrap that first observes an already-deleted note emits the note/Studio heads
in compatible tombstoned lifecycle state. Capability mismatch, prebootstrap REST
capture, degraded readiness, predecessor blocking, already-deleted bootstrap, and
restore are mandatory state-matrix tests.

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
product. Existing Sync conflict records remain the review boundary. `overwrite`
and `skip` are supported for all three domains. `duplicate_rename` is rejected with
`sync_resolution_action_unsupported`: it is incompatible with deterministic
placement and note-bound Studio identity, while duplicating only a board without
its placements would introduce an unapproved partial-copy product operation.

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

A new owner/dataset-scoped
`GET /api/v1/notes/moodboards/{moodboard_id}/placements` endpoint exposes canonical
manual placement order. It uses an opaque authenticated keyset cursor over
`(order_index, object_id)`, defaults to 50 rows, permits at most 200, and accepts
`state=live|tombstoned|all`. The legacy hybrid endpoint deliberately retains its
existing `(last_modified DESC, id DESC)` ordering; no Sync or bootstrap logic
mistakes that compatibility order for placement order.

Studio REST responses retain the established product view. Its serializer
rehydrates `payload_json.meta.title` from the authoritative Notes row,
`meta.source_note_id` from the outer sidecar field, and
`payload_json.layout.{template_type,handwriting_mode,render_version}` from the
outer render fields. It may also rehydrate the legacy diagram aliases
`canonical_source` and `generation_status` and include a locally rebuilt sanitized
`cached_svg`.
Canonical Sync responses remain sections-only and cache-free.

Legacy REST writes may include nested `meta` or `layout` only when every supplied
value exactly equals the current or planned authoritative title/source/render
field. Equal compatibility fields are stripped before canonical persistence;
mismatch returns `422`. New canonical Sync input rejects `meta`, `layout`, and
`cached_svg` outright. Render, regenerate, diagram, and export services receive
the authoritative outer values explicitly rather than depending on nested legacy
fields. Existing REST fetch/save/render/regenerate/diagram/export representations
have regression coverage.

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
- `422` for malformed identity, layout, canonical payload, or provenance;
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

The existing `note_task_scope_authority` table records at most one bound
default-personal dataset per owner and is the only scope authority reused here.
Explicit enrollment acquires the dataset fence and checks or inserts that
authority inside the graph's verification/rekey transaction. Moodboard enrollment
then proves and atomically rekeys only moodboards plus placements before setting
`moodboard_graph_bound`; Studio enrollment independently proves and atomically
rekeys only Studio sidecars before setting `studio_graph_bound`. TASK-13006 task
binding does the same with `task_graph_bound`, and task compatibility resolution
does not infer binding from row presence alone. The same owner authority guarantees
all units target the same dataset, but a malformed unrelated unit does not block
the healthy unit's enrollment. Wrong-dataset state or a conflicting authority
blocks the affected enrollment. Local-unbound rows support inactive legacy behavior
but can never enter canonical capture, bootstrap, or writable capability state.

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
- Legacy diagram `canonical_source` and `generation_status` are removed only when
  equal to `source_graph` and `status`; mismatch blocks readiness. `cached_svg` is
  discarded as a derived cache and rebuilt locally after the canonical manifest
  validates.
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
- every active server uses per-graph authority semantics; any row-presence-era
  server blocks moodboard/Studio enrollment;
- the owner binding is checked or inserted through the existing
  `note_task_scope_authority` row;
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
- the shared authority targets this dataset and `moodboard_graph_bound=true`;
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
- the shared authority targets this dataset and `studio_graph_bound=true`;
- Studio bootstrap, parent binding, and same-scope source-note verification are
  complete;
- singleton and note-plus-Studio capture and repair are healthy; and
- no malformed source, oversized object, or projection blocker remains.

Public server-supported capabilities omit all three domains through
`TASK-13007.1`–`.4`. `TASK-13007.5` may advertise all supported adapter versions,
while each dataset writable map advertises the moodboard pair together or neither
and advertises Studio independently.

Once a domain has published canonical history, enrollment cannot silently remove
it or return to unsynchronized direct mutation.

## Payload limits

V1 uses these hard contract bounds in addition to the effective Sync envelope
limit (currently 262,144 bytes by default):

| Area | Closed bounds and enums |
| --- | --- |
| Moodboard | `name` 1–255 characters; description at most 2,000; `layout_mode` is `masonry` or `freeform`; query at most 2,000; at most 100 keyword tokens and 100 collection UUIDs; at most 50 sources; every token/source at most 255 characters; updated `after <= before`. |
| Canvas metadata | Object only; at most 64 keys, depth 4, 16 KiB canonical JSON; every key matches `[A-Za-z0-9_.-]{1,64}` and values are null, bool, JS-safe integer, string, or recursively bounded arrays/objects; floats plus reserved authority, identity, lifecycle, credential, and transient UI keys are rejected. |
| Placement | `x`, `y`, and `order_index` are signed JS-safe integers; width/height are 1–1,000,000; display is an object with at most 32 keys, depth 4, and 8 KiB canonical JSON under the same value/key restrictions. |
| Studio sections | At most 100 sections; unique IDs 1–128 characters; kind is exactly `cue`, `notes`, or `summary`; title 1–500 characters. `cue` has 0–200 items of at most 2,000 characters and no `content`; other kinds have one content string at most 65,536 UTF-8 bytes and no `items`. |
| Studio outer fields | `template_type` is `lined`, `grid`, or `cornell`; `handwriting_mode` is `off` or `accented`; `render_version` is exactly 1; excerpts are at most 5,000,000 characters before the stricter envelope preflight; source and note IDs are canonical UUIDs. |
| Diagram manifest | Null or exactly `diagram_type`, `source_section_ids`, `source_graph`, `diagram`, `format`, `status`, and `render_hash`. Type is `flowchart`, `sequence`, `class`, `state`, `er`, `gantt`, or `pie`; at most 50 unique existing section IDs; source graph contains only the selected `{id,title,kind,content}` rows; diagram is at most 131,072 UTF-8 bytes; format is `mermaid`; status is `ready`. Caches and legacy aliases are noncanonical. |
| Provenance | Exactly `kind`, `attestation`, `provider`, `model`, `accepted_at`, `source_revision`, `source_hash`, and `result_hash`. Kind is `manual`, `derive`, `regenerate`, `diagram`, or `legacy_bootstrap`; attestation is `server`, `client_declared`, or `trusted_bootstrap_v1`; `accepted_at` is the server acceptance time for new state and normalized product time only for bootstrap; provider/model are at most 100/200 characters and required together only for `derive` and `diagram` and null for `manual`, current deterministic `regenerate`, and bootstrap; source revision/hash are null together or positive/valid-hash together; bootstrap requires trusted-bootstrap attestation and no other kind may use it. |

Duplicate section IDs, duplicate selected IDs, invalid cross-field combinations,
unknown recursive keys, and hashes that do not match their accepted source are
rejected. `server` attestation is assigned only after server execution;
authenticated client transitions are assigned `client_declared` regardless of
their submitted claim. Diagram `source_section_ids` are normalized to document
section order; `source_graph` must exactly equal the canonical projection of those
current sections and `render_hash` is recomputed rather than trusted. The current
Studio API permits state larger than the default Sync envelope limit. TASK-13007
does not add chunking. When a domain is active or capture-enabled:

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

- `TASK-13007.1` and `.2` advance the ChaChaNotes schema for product and derived
  projection storage. A binary that predates the resulting schema cannot reopen
  the upgraded database even while the domains are dormant.
- Before `TASK-13007.5`, the domain behavior is safe to deactivate because no
  production history exists, but code rollback requires a forward-compatible build
  that still understands the upgraded schema or restoration of a pre-migration
  database before deploying the older binary.
- After `TASK-13007.5` activates a dataset, an arbitrary older binary is unsafe
  because it cannot understand the new capture gates or portable `notes.note`
  modification-time metadata. Every compatibility build permanently retains that
  parser, projection, and old-envelope fallback.
- Post-activation rollback requires a compatibility build that retains the new
  gates, or maintenance mode plus restoration of a pre-activation database state.
- Operators must never run a pre-TASK-13007 binary with writes enabled against an
  activated dataset.

These restrictions are part of the operator documentation and activation tests.

## Performance and bounded-query requirements

- Bootstrap uses indexed keyset pages, never unbounded table loads.
- Smart projection rebuilds use bounded row/time work turns and resumable keyset
  cursors; request paths read only indexed completed generations.
- Placement list and conflict reads are scoped by owner and dataset.
- Board and Studio lookups use scoped portable-identity indexes.
- Retained-child checks and hard-delete gates use bounded existence queries.
- Public legacy offset pagination remains compatible, but canonical bootstrap and
  repair progress never depends on offsets.
- SQLite `EXPLAIN QUERY PLAN` and PostgreSQL plan assertions prove expected indexes
  for source scans, smart projection rebuild/read/count, dependency checks,
  pagination, and RLS-scoped lookups.
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
- migrate and verify SQLite/PostgreSQL product storage, reuse of the existing sole
  scope authority with per-graph binding flags and task-caller compatibility,
  lineage, RLS, and indexes;
- remove parallel Studio schema authority; and
- keep all three domains absent from public supported/writable capabilities.

### `TASK-13007.2` — Portable smart-match foundation

- implement dormant server-bound note-time acceptance/projection into existing
  `last_modified` without stamping production note history;
- add versioned NFC/casefold comparison projections and exact dependency
  fingerprints;
- implement missing portable collection filtering and backend-independent rule
  evaluation;
- add the disposable bounded smart-match projection, dirty/rebuild state, and
  additive completeness status on the legacy hybrid endpoint;
- update that response's OpenAPI fingerprint, generated client types, and API docs;
- prove worst-case sparse-query, count, high-page, and SQLite/PostgreSQL parity;
  and
- publish no new Sync-domain history or capability.

### `TASK-13007.3` — Moodboards and placements

- implement dormant adapters, materializers, repair, bootstrap, and internal-only
  capture planners for the coupled pair;
- add canvas/placement REST compatibility surfaces, canonical placement keyset
  listing, restore, concurrency, and idempotency;
- update their OpenAPI fingerprints, generated client types, and API docs;
- prove smart-match exclusion and retained-child lifecycle; and
- leave production REST capture unwired and the pair absent from public
  capabilities.

### `TASK-13007.4` — Studio documents

- implement the dormant Studio adapter, materializer, provenance validation,
  repair, bootstrap, and internal-only capture/group planners;
- bind sections-only canonical state to exact note heads while preserving the
  established REST representation and rebuilding derived diagram caches;
- synthesize accepted manual/derive/regenerate/diagram save plans in an internal
  harness and reject oversized or transient/secret-bearing state;
- update Studio contract OpenAPI fingerprints, generated client types, and API
  docs; and
- leave production REST capture unwired and Studio absent from public
  capabilities.

### `TASK-13007.5` — Lifecycle integration and activation

- wire note-plus-Studio delete/restore groups and the complete lifecycle/capability
  mismatch matrix;
- enable permanent production `notes.note` canonical-time stamping/projection and
  verify or repair existing heads;
- activate production enrollment, fail-closed capture, readiness, repair, and
  capability advertisement for the completed domains;
- gate hard delete and document post-activation rollback;
- prove two-client convergence, crash repair, conflicts, and pagination on SQLite
  and live PostgreSQL; and
- update activation/capability OpenAPI contracts and operator docs, then close
  `TASK-13007`.

Each child is independently mergeable. The first four provide no production path
that can publish canonical history for these domains and remain safely dormant.
The fifth is the first public writable state and cannot merge without the complete
security, repair, bootstrap, and live-backend evidence.

## Verification strategy

All implementation follows focused RED-GREEN-REFACTOR and fresh verification
before each commit and completion claim.

### `TASK-13007.1`

- fresh and upgrade migration on SQLite and live PostgreSQL;
- rollback fault injection at every migration boundary;
- exact columns, constraints, indexes, reuse and owner-only policy of
  `note_task_scope_authority`, foreign keys, catalog, forced RLS, and policy drift;
- per-graph authority migration, task local-unbound compatibility, immutable flag
  transitions, additive defaults/old task-binder inserts, empty graphs, and
  interleaved first-enrollment races;
- legacy owner proof, local-unbound binding, malformed JSON, invalid identities,
  duplicate IDs, and oversized diagnostics;
- strict closed Studio contracts, canonicalization, tombstone, hash,
  cross-runtime JSON vectors, `server_trusted_v1`, and readiness matrices;
- runtime Studio helper compatibility; and
- proof that no public capability exposes the domains.

### `TASK-13007.2`

- portable note acceptance time, trusted bootstrap/fallback time, exact retry, and
  dormant projection into existing `notes.last_modified` with production note
  history unchanged;
- smart-rule normalization, portable collection translation/filtering, keyword
  and conversation-source dependencies, exact Unicode-data-version compatibility,
  and cross-runtime conformance vectors;
- dirty marking, bounded work budgets, continuation, crash/resume, atomic
  generation publication, and invalidation;
- manual-only pending responses plus complete indexed result/count responses;
- Unicode/wildcard/timestamp boundaries, sparse matches, high pages, and backend
  parity;
- SQLite plus live PostgreSQL plan evidence with no new domain history; and
- hybrid-response OpenAPI fingerprint, generated-client, and API-doc drift checks.

### `TASK-13007.3`

- moodboard and placement create/update/tombstone/restore, exact retry, changed
  retry, stale base, deleted parent, and cross-scope matrices;
- overwrite/skip plus duplicate-rename rejection;
- exact namespaced relationship identity, parent/dependency binding, integer
  layout bounds, and `(order_index, object_id)` keyset listing;
- bodyless legacy pin compatibility, placement patch, unpin tombstone, and restore;
- interrupted/resumed bootstrap, source drift, projection split repair, pull, ack,
  and bounded keyset scans;
- manual-placement-only synchronization and legacy hybrid-order compatibility;
- SQLite plus live PostgreSQL RLS, pagination, plan evidence, and proof production
  capture remains unwired; and
- placement/canvas OpenAPI fingerprint, generated-client, and API-doc drift checks.

### `TASK-13007.4`

- Studio strict outer payload and versioned nested JSON validation;
- legacy nested-title/source/layout equality, canonical removal, and mismatch
  blockers plus compatibility rehydration for fetch/save/render/regenerate/
  diagram/export;
- note-head binding, stale note, exact retry, changed retry, tombstone, restore,
  and cross-scope matrices;
- same-scope live source-note acceptance plus retained tombstoned-source behavior;
- server-attested versus client-declared versus legacy provenance;
- exact content hashes, canonical JSON, field limits, cross-field rules, cache-free
  diagrams, unambiguous render-hash framing, cache rebuild, and render/export
  sanitization;
- accepted manual/derive/regenerate/diagram capture and proof that previews,
  failures, prompts, title suggestions, and credentials are excluded;
- client compound-command synthesis, overlap rejection, canonical incoming-intent
  fingerprint replay, stored-plan integrity, changed intent, and all-or-none append;
- payload-limit preflight and readiness blocking;
- interrupted/resumed bootstrap and product/Sync split repair;
- SQLite plus live PostgreSQL RLS and bounded plan evidence with production capture
  still unwired; and
- Studio OpenAPI fingerprint, generated-client, and API-doc drift checks.

### `TASK-13007.5`

- complete note-plus-Studio group ordering, crash windows, repair, delete, and
  restore;
- production canonical note-time activation, mixed old/new envelope fallback,
  existing-head verification, and rollback compatibility;
- unbound, bootstrapping, ready, unhealthy, already-deleted-bootstrap, and
  device-capability mismatch lifecycle matrices plus ADR-034 predecessor blocking;
- board/note tombstones with retained placements and explicit placement tombstones;
- activation predicates, partial-domain failure, enrollment retry, readiness drift,
  independent binding units, mixed-fleet rejection, shared authority, and
  capability maps;
- hard-delete denial, schema-compatible pre-activation rollback, and
  post-activation rollback guard documentation;
- two-client concurrent layout and Studio revisions, deletion, restore, conflicts,
  pull, apply, replay, and repair;
- broad Notes and Sync regressions on SQLite and live PostgreSQL;
- Ruff, changed-scope Bandit, compilation, and `git diff --check`; and
- activation/capability OpenAPI drift and final operator-documentation checks.

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
- Smart matching is served from a disposable bounded local projection with honest
  completeness status, never from an unbounded request scan.
- Existing product tables remain authoritative; no Sync-owned shadow tables.
- Existing `note_task_scope_authority` is the sole owner/dataset binding authority;
  moodboard and Studio graphs enroll independently against it.
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
- Legacy Studio REST rehydrates its nested product view; canonical Sync remains
  sections-only and diagram caches remain derived local state.
- Enrollment is limited to default-personal `server_trusted_v1` datasets.
- No Sync chunking is added for oversized Studio objects.
