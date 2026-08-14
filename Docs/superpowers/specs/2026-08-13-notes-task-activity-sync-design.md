# Notes task and activity Sync v2 design

- **Date:** 2026-08-13
- **Task:** TASK-13006
- **Status:** Approved design; implementation planning and execution remain separate gates
- **Depends on:** TASK-13005, ADR-031, ADR-034, ADR-037, and ADR-038
- **Architecture decision:** `Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md`
- **Reviewed server baseline:** `dev` at `8f94369e51`

## Purpose

Make first-class Notes tasks and their immutable activity history converge across
devices without making Markdown checklist projections or local read state competing
authorities. Preserve existing REST behavior and data while adding strict Sync v2
contracts, exact optimistic lifecycle semantics, resumable bootstrap, and
reviewable reconciliation drift.

The work is deliberately split into four atomic pull requests. No partially
implemented domain is advertised writable.

## Current server reference

The reviewed server already provides:

- schema-v48 `note_tasks`, `task_events`, `task_event_read_state`,
  `task_note_projections`, and `note_task_reconciliation_state` tables;
- bounded task/event stores and REST endpoints;
- an existing Notes task reconciler for Markdown checklist projections;
- strict `notes.note` Sync lifecycle, durable server-origin mutation groups,
  materialization ordering, bootstrap/readiness, restore, repair, and per-version
  cursor/ack infrastructure; and
- SQLite and PostgreSQL backends with established exact-catalog and forced-RLS
  migration patterns.

The current task payload is narrower than the required wire contract. `note_tasks`
stores a stable ID, parent note, `text`, open/done status, metadata, projection
status, soft delete, timestamps, client ID, and a projection-influenced REST row
version. The existing
metadata supports due date, priority, and estimate. There is no `notes.task` or
`notes.task_activity` adapter, activity tombstone, or explicit task-table
PostgreSQL RLS contract.

Focused baseline verification found one stale migration fixture that simulated
schema v47 without removing the later v59 attachment table. The production v59
collision guard correctly failed closed. The fixture was repaired without
weakening production, then the full task baseline passed:

```text
74 passed, 2 warnings in 45.20s
```

## Goals

1. Define strict version-1 `notes.task` and `notes.task_activity` Sync domains.
2. Preserve stable task identity, structured task fields, exact lifecycle lineage,
   immutable activity history, and existing REST compatibility.
3. Enforce owner, dataset, parent-note, and task-note authority on SQLite and
   PostgreSQL with bounded indexed queries and forced RLS.
4. Capture normal server-origin REST, MCP, and reconciliation mutations as the same
   canonical envelopes clients produce.
5. Bootstrap existing tasks and events resumably before capability advertisement.
6. Converge explicit tasks and Markdown checklists deterministically without silent
   overwrite.
7. Make concurrent changes, replay, restore, completion, recurrence-state edits,
   and delete races idempotent or reviewable.

## Non-goals

- Automatically generating future task instances or adding a recurrence scheduler.
- Synchronizing Markdown text as an independent task authority.
- Synchronizing projection cursors, reconciliation state, event read state, FTS,
  counters, cache rows, or other derived/device-local data.
- Replacing the existing REST `text` field or integer/internal identifiers exposed
  by compatibility routes.
- Mutable task-history edits or activity restore.
- Distributed transactions between the Sync and ChaChaNotes databases.
- A global feature environment flag in addition to dataset enrollment/readiness.
- Broad task assignment, notification, calendar, or reminder workflows.

## Authority and identity

### Canonical product authorities

| State | Authority | Sync treatment |
| --- | --- | --- |
| Current task | `note_tasks` | `notes.task` v1 whole-object upsert/tombstone |
| User-visible task history | `task_events` | `notes.task_activity` v1 immutable create/one-way tombstone |
| Checklist text and projection cursor | managed marker in note Markdown plus `task_note_projections` cache | Derived; rebuild/reconcile locally |
| Reconciliation bookkeeping | `note_task_reconciliation_state` | Local only |
| Activity read/dismiss state | `task_event_read_state` | Device/user UI state; never synchronized |
| Projection drift review | `task_projection_drifts` | Local review state; never synchronized |

Each task and activity uses a stable opaque UUID identity. The client may submit
the resource ID but cannot select owner, authenticated actor, or trusted source
device. Task parent-note identity is immutable. Activity parent identities are
immutable.

### Canonical task fields

`note_tasks.text` is the storage authority for wire `title`. Existing REST routes
continue to expose and accept `text`; adapters translate at the boundary without
changing stored meaning.

The version-1 task payload is exact; every field is required even when nullable:

| Field | Canonical type and constraint |
| --- | --- |
| `task_id`, `note_id` | lowercase canonical UUIDv4 strings; immutable and equal to the envelope object/parent identity |
| `title` | 1–2,000 Unicode code points, no CR/LF/control characters, already stripped; REST keeps its existing strip behavior before capture |
| `description` | null or 0–16,000 Unicode code points with no disallowed controls; content is otherwise preserved |
| `status` | exactly `open` or `done` |
| `completed_at` | null RFC 3339 UTC timestamp for `open`; required RFC 3339 UTC timestamp for `done` |
| `priority` | null or `low`, `medium`, `high` |
| `due_date` | null or a real calendar date in canonical `YYYY-MM-DD` form |
| `estimate` | null or `[0-9]{1,6}[mhd]`, preserving the existing accepted syntax |
| `recurrence` | null or `{frequency, interval, by_weekday, until, state, occurrence_index}` using the bounded rules below |
| `assignee_id` | null or the authenticated personal-dataset owner; shared assignment needs a later adapter version |
| `tags` | at most 32 already-trimmed NFKC strings of 1–64 code points, no controls, unique under casefold, stored/hash-sorted by `(casefold, value)` |
| `custom` | JSON object with at most 32 non-reserved keys, keys 1–64 safe characters, depth at most 4, and canonical UTF-8 JSON at most 16 KiB |

Recurrence is data, not a scheduler. `frequency` is `daily`, `weekly`, `monthly`,
or `yearly`; `interval` is 1–365; `by_weekday` is an ordered unique subset of
`mo` through `su` and is allowed only for weekly recurrence; `until` is null or a
canonical real date; `state` is `active`, `paused`, or `completed`; and
`occurrence_index` is an integer from 0 through 2,147,483,647. Contradictory or
unknown keys are rejected.

Existing core fields stay in existing columns. Extended fields remain in
`metadata_json`, but they are first-class validated wire fields rather than an
untyped pass-through. Reserved wire keys cannot appear again in custom metadata.
Unknown fields, noncanonical values, duplicate tags, invalid recurrence data, and
out-of-bound strings/collections are rejected before hashing or persistence.

The product mapping is fixed:

| Wire state | Product state |
| --- | --- |
| `title` | `note_tasks.text` |
| `status` | `note_tasks.status` |
| `completed_at` | `note_tasks.completed_at` |
| `description`, `priority`, `due_date`, `estimate`, `recurrence`, `assignee_id`, `tags`, `custom` | reserved canonical keys in `metadata_json` |
| portable revision/hash | new `canonical_revision`, `canonical_hash` columns |
| REST/projection optimistic row version | existing `version` column |

Canonical task mutation increments both `version` and `canonical_revision` and
recomputes `canonical_hash`. Projection-only locator/status changes may preserve
their current REST `version` behavior but never change `canonical_revision` or
`canonical_hash`. Bootstrap initializes `canonical_revision` to the existing
positive `version` and hashes the current canonical row, so no REST version moves
backward. Sync envelope `object_revision` always equals canonical revision, never
the projection-influenced REST version.

Legacy `due_date`, `priority`, and `estimate` keys map directly after exact
validation. A legacy reserved key with the wrong type/value, an unknown top-level
legacy key, or an oversized value blocks readiness and reports only row ID plus a
reason code/hash; migration does not guess or silently relocate it into `custom`.

The recurrence contract stores a validated rule and recurrence state only. It does
not interpret wall-clock time to create another task. A future scheduler requires a
separate task and architecture decision.

### Canonical activity fields

An activity payload contains:

- stable `activity_id`;
- required `note_id` and nullable `task_id` parent identities;
- actor identity and source device as server-bound provenance;
- client event timestamp preserved as submitted metadata;
- transition/event type, source, and bounded structured metadata; and
- optional task revision/result references needed to verify the transition.

The exact value contract is:

| Field | Canonical type and constraint |
| --- | --- |
| `activity_id`, `note_id` | required lowercase UUIDv4 strings |
| `task_id` | null or lowercase UUIDv4; when present it belongs to `note_id` |
| `event_type` | one closed value listed below |
| `actor_type` | server-bound `user`, `agent`, `tool`, `system`, or `legacy` |
| `actor_id` | null or an authorized opaque ID of 1–128 safe characters; user actors equal the authenticated owner |
| `source_device_id` | registered canonical device UUID for client origin; null only for trusted server/bootstrap sources |
| `client_occurred_at` | canonical RFC 3339 UTC timestamp; preserved but not ordering authority |
| `source_kind` | `client`, `rest`, `mcp`, `markdown_reconciliation`, `repair`, or `trusted_bootstrap_v1` |
| `corrects_activity_id` | required same-scope UUID only for `corrected`; otherwise null |
| `old_value`, `new_value` | null or event-schema-validated canonical JSON objects, each at most 16 KiB and depth 4 |
| `metadata` | canonical JSON object at most 8 KiB, 16 keys, and depth 3; credentials and raw Markdown are forbidden |

The stable ID is opaque, not a digest. A canonical fingerprint over the validated
immutable content protects exact replay. Server cursor and activity ID define
deterministic order. Client timestamps may be displayed but never resolve ordering
or conflict.

The note is always present and authorized. If the task is present, it must belong
to that note. A task identity from another owner or dataset is indistinguishable
from missing at public boundaries.

The product mapping adds owner/dataset, `sync_revision`, `sync_object_hash`,
`sync_server_cursor`, `source_device_id`, `client_occurred_at`, `source_kind`,
`corrects_activity_id`, and one-way deleted/deleted-at/delete-reason columns to
`task_events`. Existing `created_at` remains the server-created time. Existing
actor/tool/policy/approval and old/new JSON fields remain source data for the
canonical event metadata. Activity pages keyset-order by
`(sync_server_cursor, activity_id)`; `created_at` and `client_occurred_at` never
order Sync history.

Version 1 event types are the closed set `created`, `updated`, `completed`,
`reopened`, `deleted`, `restored`, `projection_linked`, `projection_unlinked`,
`projection_drift`, and `corrected`. A `corrected` event requires a
`corrects_activity_id` in the same owner/dataset/note and, when present, task.
Other event types require that field to be null. The fingerprint binds adapter
version, lifecycle, activity ID, parent IDs, event type, correction target, actor,
source device, client timestamp, source kind, transition values, and bounded
metadata; it excludes server cursor, server-created time, and read state.

## Domain contracts

### `notes.task` version 1

Supported operations are `upsert` and `tombstone`.

- Create requires an empty current head and the complete canonical payload.
- Update, completion, reopen, recurrence-state change, and any other mutation require
  the exact current base cursor, object revision, and object hash.
- Tombstone requires the exact live current base.
- Restore is an upsert with explicit restore intent, exact current tombstone base,
  and the complete canonical payload.
- Ordinary upsert against a tombstone, restore against a live task, stale base,
  changed identity, or changed parent note is a reviewable whole-object conflict.
- Exact accepted replay returns the existing result and never creates a second
  task mutation or activity.

The canonical object hash binds adapter version, operation/lifecycle state,
identity, immutable parent, all mutable fields, and provenance fields that are
part of portable state. It excludes server cursor and local projection data.

### `notes.task_activity` version 1

Activity creation uses an `upsert` envelope only as the common wire verb; product
semantics are immutable create.

- A previously unseen envelope object/activity ID creates one event; the ID is
  always required.
- Exact stable ID plus canonical fingerprint replay is idempotent.
- Reusing an ID with changed content is a stable idempotency conflict.
- An existing event cannot be updated.
- Tombstone requires the exact current cursor/revision/hash base, advances the
  immutable activity lifecycle revision by one, and is irreversible.
- Tombstoned activity cannot be restored or recreated under the same ID.
- Corrections are represented as a new activity that references the earlier event
  where the event taxonomy permits it.

Exact replay of the original accepted create or tombstone returns its existing
result. A second create after tombstone, a distinct second tombstone, or any reuse
with changed fingerprint conflicts. Tombstones retain the immutable event content
for audit but exclude it from ordinary activity surfaces.

Create always has activity revision 1. Its payload is the complete immutable field
set above and its object hash is the canonical fingerprint. Tombstone always has
revision 2 and the exact payload `{note_id, task_id, deleted_at, delete_reason}`:
parent IDs must equal the create, `deleted_at` is the canonical envelope timestamp,
and `delete_reason` is `user_request`, `correction`, or `policy`. The tombstone hash
binds adapter version, revision/lifecycle, original create fingerprint, parent IDs,
deleted timestamp, and reason. The product row retains create fields and stores the
same deleted-at/reason/hash/cursor. No revision greater than 2 is valid.

Task transitions that require user-visible history create exactly one canonical
activity. A retry or crash repair cannot duplicate it because the event ID and
mutation-group step are stable.

## Storage, migration, and RLS

The first pull request extends the existing task schema rather than creating a
parallel product authority. Because client-chosen IDs are dataset-scoped, migration
rebuilds the five existing task-side tables under the schema lock with composite
`(owner_user_id, dataset_id, id)` authority and matching composite foreign keys;
an internal surrogate may be used only if it preserves the same collision-safe
contract. It also adds `task_projection_drifts` with the same scope. Legacy rows are
mapped only after their owning note and dataset are proven. The swap is
transactional, source-count/hash verified, and version-last.

Both backends must enforce:

- stable UUID identities, lifecycle and version checks, bounded metadata, and
  immutable parent references;
- owner/dataset/note/task indexes for point lookup, current task pages, activity
  pages, bootstrap, and reconciliation;
- count and cursor/fingerprint queries that remain bounded and index-backed;
- owner and parent-note authorization in every read/write predicate; and
- exact current catalog verification before advertising readiness.

These rules apply to `note_tasks`, `task_events`, `task_note_projections`,
`task_event_read_state`, `note_task_reconciliation_state`, and
`task_projection_drifts`. Read-state requires the authenticated `user_id` to equal
the owner and joins an authorized event before any insert/update. Projection and
reconciliation rows may contain raw Markdown or state that reveals task existence,
so they receive the same boundary despite not being Sync domains. Every public
predicate begins with owner and dataset before resource identity or limit.

PostgreSQL uses forced RLS on all six tables. Task policies require authenticated
owner/dataset and an owned parent note in `USING` and `WITH CHECK`. Activity,
projection, read-state, reconciliation, and drift policies join through the same
owned note/task/event boundary. The current-version verifier enumerates every
required column, constraint, composite key/FK, index, policy, role, command, and
canonical expression and rejects additional permissive policy drift.

Migration is transactional and authority-locked before inspection. It never
guesses at malformed legacy rows, rewrites existing task meaning, or deletes
history. A schema collision or incompatible catalog aborts without advancing the
version. Live PostgreSQL migration/RLS verification is mandatory for the storage
PR; an unavailable suitable server is a blocker, not a silent pass.

Before any new domain is writable, runtime reconciliation proves that legacy REST
reads and writes remain compatible and that all existing rows can be represented
canonically. Unrepresentable source data fails readiness with bounded diagnostics.

## Authorization and privacy

Every operation binds the authenticated principal, device, dataset, parent note,
and resource identity before revealing whether the resource exists. Public errors
are stable and sanitized. They do not expose another owner's note/task/event,
payload, Markdown, actor details, metadata, or projection drift.

Diagnostics may include bounded counts, opaque identifiers already authorized to
the caller, and hashes. Mutation-group metadata contains only opaque identities,
ordering, operation names, and plan hashes; it never contains title, description,
checklist text, credentials, or custom metadata.

Queries are owner-scoped and capped. Activity ordering uses server cursor and
stable ID. Tags and metadata limits are enforced before allocation or hashing.

## Server-origin and client-origin mutation flow

### Client origin

1. Validate envelope version, domain, operation, payload, and size.
2. Authorize device, dataset, parent note, task/activity identity, and enrollment.
3. Compare the complete optimistic base with the durable current head.
4. Deterministically expand any task mutation into its required activity and note
   projection steps before append. The activity ID and group ID derive from the
   authenticated input envelope ID plus transition kind, so retry produces the
   identical plan. Clients do not separately invent the transition event.
5. Validate and append the complete group under dataset authority, then project
   under the ADR-034 materialization fence.
6. Persist task/activity product state, object state, and apply result
   idempotently.
7. Return success only for an applied or exact terminal replay.

### Server origin

REST, MCP, and Markdown reconciliation call the same canonical adapter path when
Sync owns the domain. A compound action creates one stable ordered mutation plan,
for example:

1. update `notes.task`;
2. append `notes.task_activity`; and
3. update `notes.note` when the Markdown projection changes.

The complete plan is inserted atomically in the Sync store before product
materialization. The product write and Sync status commit cannot be globally
atomic. A product commit followed by a Sync status failure is repaired by replaying
the same stable plan and verifying the already-applied product postcondition.
Request success requires every step to be durably applied.

Client-origin compound groups use the same append authority and materialization
ordering. A conflict in one step blocks later projection rather than publishing a
partial accepted intent as success.

All portable task mutations emit one canonical activity, including metadata-only
updates. A task tombstone retains history and emits `deleted`; restore emits
`restored`. A Markdown projection change adds a `notes.note` step only when the
canonical note bytes change. Group synthesis, validation, append, repair, and
postcondition verification are complete before either domain is advertised.

Normal lifecycle activity is coordinator-derived, not separately submitted by the
client: actor/device/timestamp come from the authenticated task envelope and
`old_value`/`new_value` come from its exact base and accepted payload. A direct
client activity create is permitted only for `corrected` and must reference an
authorized existing event. Server-origin REST/MCP/reconciliation supplies the same
fields through trusted bindings, never caller-selectable provenance.

ADR-034's 1,000-envelope mutation-group cap is authoritative. Reconciliation
preflight parses and validates the complete note edit before appending any envelope
or mutating product state. One changed task consumes a task step plus one activity
step, and the note consumes at most one step, so a single request may change at
most 499 managed tasks (`2 * 499 + 1 = 999`). Any computed plan above 1,000 steps,
including future fixed coordinator steps, fails with the stable sanitized
`sync_task_projection_group_too_large` error before the `notes.note` envelope is
appended. The request is not split into sequential groups, because that would make
an accepted prefix durable without the user's complete intent.

## Bootstrap, capability advertisement, and rollout

No global environment flag is added. Existing dataset enrollment and readiness are
the rollout boundary. Public supported and writable capability maps omit both
domains through PRs 1–3; dormant adapters are not a compatibility promise.

Each domain has an independent readiness row and the state machine
`not_enrolled -> enrolling -> bootstrapping -> verifying -> ready`, with
`blocked` retaining its last verified keyset cursor and reason code. Retry resumes
`bootstrapping` or `verifying`; disabling enrollment returns to `not_enrolled` only
before a domain has published canonical history. Existing ready Notes groups are
not reopened.

Existing datasets require an explicit owner/admin enrollment action. New datasets
begin `not_enrolled`; a creation request may atomically request these domains only
after PR 4 exposes them, but no profile silently opts in. Adding a domain to an
already-ready default-personal dataset creates its independent readiness row in one
Sync transaction without changing earlier domain readiness.

Before either source scan, enrollment atomically enables fail-closed canonical
capture for both task and activity while external device writes remain disabled.
It cannot enable task capture alone. Each bootstrap page holds
the dataset materialization fence, reads an owner/dataset keyset page, appends
trusted source-verified envelopes, and records its cursor/count/fingerprint.
Concurrent REST changes therefore append after or before that page under the same
ordering authority rather than escaping capture.

`notes.task` becomes dataset-writable in PR 4 only when:

- `notes.note`, `notes.task`, and `notes.task_activity` are enrolled at supported
  adapter versions;
- source task bootstrap is complete, count/fingerprint verified, and resumable
  cursor state is `ready`;
- source activity bootstrap is complete and verified;
- atomic task/activity capture, deterministic group expansion, and both repair
  paths are enabled and healthy; and
- no reconciliation blocker or unresolved bootstrap drift exists.

`notes.task_activity` has the same coupled readiness predicate because ordinary
activity depends on an authorized task/note view and task mutations cannot succeed
without activity. PR 4 may then list both adapter versions in server-supported
capabilities. The selected dataset's writable map advertises both together or
neither; a later adapter version may relax that coupling only through a new ADR.

Bootstrap is non-destructive and keyset-paged. Stable existing task/event IDs are
preserved. Each page records its source count, cursor, and privacy-safe fingerprint.
Final readiness rechecks the complete bounded aggregate. Restart resumes from the
last verified page. Source drift either captures a verified correction under the
dataset fence or leaves the dataset not ready; it never marks stale source data
ready.

Legacy event provenance uses the explicit `trusted_bootstrap_v1` source kind,
preserves the stored actor/tool/policy fields after validation, uses stored
`created_at` as `client_occurred_at`, and records a null source device with a
`legacy_source_verified` marker. Bootstrap never fabricates a client device or
claims that a legacy timestamp ordered Sync history.

Legacy event conversion is exact:

| Existing `event_type` and values | Canonical event |
| --- | --- |
| `created` | `created`; old must be null and new must validate as the bounded task snapshot subset |
| `updated` | `updated`; old/new must be bounded objects and differ in at least one canonical task field |
| `status_changed`, `open -> done` | `completed` |
| `status_changed`, `done -> open` | `reopened` |
| `unlinked` | `projection_unlinked` |
| `deleted` | `deleted` |
| any other type, status transition, parent mismatch, malformed value, or invalid ID | block activity readiness; do not rewrite or drop the row |

A nullable legacy `note_id` is derived only from its same-scope authorized task;
both missing or inconsistent parents block readiness. `actor_type` must map to the
closed actor enum and actor ID must meet its bound. `tool_name`, `policy_mode`, and
`approval_id` become an optional `legacy_context` metadata object of separately
bounded strings (128 code points each). Stored old/new JSON is canonicalized under
the event-specific schema. A stored `idempotency_key` is not synchronized in raw
form; after type/length validation it becomes a SHA-256 request fingerprint in
metadata. Oversized, nested, or unknown values block readiness with a reason code
and row hash.

The source scan is keyset-ordered by `(created_at, id)` while holding the dataset
fence. Trusted bootstrap envelopes receive monotonic server cursors in that order;
after materialization all activity pages use `(sync_server_cursor, activity_id)`
with a maximum page size of 1,000. A concurrent new event is captured normally
under the same fence and therefore cannot be omitted between scan and final
count/fingerprint verification.

Legacy devices receive only versions they negotiated. Per-domain adapter-version
cursors and acknowledgments prevent a version change from skipping or
acknowledging incompatible envelopes.

## Markdown projection convergence

Markdown checklists are a deterministic view over explicit task authority. A
managed line ends with the protected marker
`<!-- tldw-task:v1:<task-uuid>:<canonical-revision>:<canonical-hash> -->`.
The marker is not user-visible when rendered, is included in the canonical note
bytes, and is updated only by the projection coordinator. Duplicate or malformed
markers are drift, never identity claims. Unmarked legacy checklist lines enter the
bounded bootstrap matcher once; after an unambiguous source-verified match or new
task creation, the coordinator rewrites the line with its marker.

The marker is the durable association and exact last-common task base tuple. The
named immutable `notes.task` envelope supplies the base snapshot after
`task_note_projections` loss. Projection cache rebuild parses markers, authorizes
their owner/dataset/note/task relation, verifies the referenced envelope
revision/hash, and regenerates offsets and hashes. If the historical envelope is
missing, superseded, unauthorized, or inconsistent, the line becomes drift and no
task or note is overwritten.

An applied task envelope referenced by a live marker is retention-protected until
the marker advances or unlinks. Sync maintenance may compact unrelated history but
must not remove the only last-common snapshot for a live projection.

When a group advances or removes a marker, its privacy-safe group metadata records
a projection anchor: task ID, linked/unlinked state, prior note envelope
cursor/hash, prior task revision/hash, and marker hash. It contains no Markdown or
task text. The anchor is durable Sync evidence, not the disposable locator cache,
and lets repair reconstruct whether a task was linked before deletion. A linked
task tombstone retains that anchor and both named envelopes until restore installs
a new verified marker, an exact explicit unlink resolution commits, or an
authorized hard-purge/restore-window workflow permanently releases the task. An
unlinked task records that state and needs no former-marker retention.

Only `title`, checkbox-derived `status`/`completed_at`, `due_date`, `priority`, and
`estimate` are Markdown-projectable in version 1. Description, recurrence,
assignee, tags, and custom metadata remain explicit-task-only fields and therefore
cannot be erased by Markdown. The last-common envelope provides the projected
field snapshot needed to distinguish task-only, Markdown-only, and compatible
changes.

For each bounded reconciliation unit:

| Markdown since base | Task since base | Result |
| --- | --- | --- |
| unchanged | unchanged | no-op |
| changed | unchanged | validate Markdown edit, create/update or unlink task projection, emit activity, then record new marker/base |
| unchanged | changed | rebuild Markdown from canonical task and record new projection |
| changed compatibly | changed compatibly | converge to one canonical task/projection and record the complete durable group |
| changed incompatibly | changed incompatibly | retain both authorities, create a privacy-safe review drift record, and do not overwrite |

Checklist reordering alone does not change stable task identity. A parser cannot
claim an explicit task by text coincidence alone; it must carry or resolve the
stable projection identity. Unsupported or ambiguous Markdown remains visible and
creates reviewable drift rather than disappearing.

Removing a managed line unlinks its projection and emits `projection_unlinked`; it
does not tombstone the explicit task. Task deletion is an explicit task operation,
removes the managed line in the same durable group when linked, and leaves the
activity history. Restore reprojects the line only when the stored linkage is live
and the parent note is live; an unlinked task remains unlinked until explicit
relink. Note trash hides projections without changing task lifecycle.

`task_projection_drifts` stores only owner/dataset/note/task IDs, marker base tuple,
bounded reason code, note/task head cursors and hashes, status
`open|resolved|dismissed`, and timestamps. It stores no raw title, description, or
Markdown. Resolution requires an exact opaque drift ID plus current note/task head
claims and chooses `keep_task`, `accept_markdown`, or `unlink`; changed claims create
a new review rather than applying stale intent. Resolved/dismissed rows remain
bounded audit metadata and are excluded from Sync.

Every open drift installs retention blockers for its referenced current/prior note
and task envelopes, including malformed or removed-marker cases that use the last
durable projection anchor. The blockers release only when an exact drift resolution
or dismissal commits and no live marker, linked tombstone, newer open drift, or
pending repair still references the envelope. Cache deletion never releases them.

Projection cache and drift bookkeeping never become Sync domains. After cache loss,
restore, migration, parser change, or detected corruption, association and the
last-common snapshot are reconstructed from the marker plus immutable Sync history.
When that proof is unavailable, rebuild stops at review. Read endpoints do not
perform maintenance writes.

## Conflict and failure semantics

The following produce stable reviewable conflicts or fail-closed readiness:

- stale task base, create-against-head, update-against-tombstone, or restore-against-live;
- changed task or activity identity/parent;
- activity stable-ID reuse with a different fingerprint;
- activity update or restore;
- task and note parent mismatch;
- duplicate recurrence transition or completion event with incompatible content;
- concurrent explicit task and Markdown edits that cannot be merged without data
  loss;
- unresolved earlier materialization conflict or incomplete mutation group;
- bootstrap source drift or unrepresentable legacy metadata; and
- missing, malformed, or unauthorized parent state.

Exact replay is idempotent. A retry after product commit but before Sync status
commit verifies the canonical product postcondition, repairs bookkeeping, and does
not create another event. Conflicts block later dependent projection according to
ADR-034 ordering.

Task deletion does not delete its activity history. Task restore restores only the
task; activity tombstones remain tombstoned. Parent-note trash controls visibility
but does not invent task or activity tombstones. Restore order requires the parent
note before a live task and the task before task-scoped activity visibility.

## Four atomic pull requests

### TASK-13006.1 — Contract and storage

- add ADR-039, canonical validators/hashes, lifecycle fields, migrations, RLS,
  exact catalog verification, and bounded indexes;
- separate portable canonical revision/hash from the projection-influenced REST
  row version and add marker/drift storage contracts;
- preserve current REST/data behavior;
- add fail-closed source-reconciliation and readiness state; and
- advertise neither new domain as supported or writable.

### TASK-13006.2 — `notes.task`

- implement strict upsert/tombstone adapter and idempotent materializer;
- enforce dataset, note, identity, exact base, completion/reopen, restore, and
  recurrence-state rules;
- implement dormant REST/MCP task capture primitives and split-commit repair;
- bootstrap existing tasks and prove pull/ack behavior; and
- keep both domains absent from public supported/writable capabilities.

### TASK-13006.3 — `notes.task_activity`

- implement immutable create and one-way tombstone;
- bind actor/device provenance and deterministic ordering;
- capture canonical events for task transitions with exact replay deduplication;
- bootstrap existing stable event IDs and exclude read state; and
- keep both domains absent from public supported/writable capabilities.

### TASK-13006.4 — Projection convergence

- implement the three-way task/Markdown matrix and privacy-safe drift records;
- implement deterministic task-to-activity expansion and durable multi-object
  mutation groups across note, task, and activity before any product mutation;
- activate public supported capabilities, explicit enrollment, bootstrap state
  transitions, and per-dataset writable advertisement only after all predicates
  pass;
- prove concurrency, pagination, repair, and two-client end-to-end convergence on
  SQLite and PostgreSQL; and
- publish final capability/API documentation and close TASK-13006.

Each PR must be independently mergeable. PRs 1–3 are compatibility-preserving
dormant foundations: no public capability map or enrollment API exposes either
domain. PR 4 is the first writable state and cannot merge until task/activity/note
group append, projection, bootstrap, repair, and live-PostgreSQL evidence are all
complete.

## Verification strategy

All implementation PRs use focused RED before production edits and fresh GREEN
evidence afterward.

### PR 1

- fresh and upgrade migration on SQLite and live PostgreSQL;
- exact columns, constraints, indexes, owner/parent RLS, policy drift, rollback,
  lock order, and schema-version authority;
- malformed legacy metadata and bounded readiness diagnostics;
- legacy REST/task-store/reconciler regression; and
- proof that neither domain is writable.

### PR 2

- strict task payload, canonical hashes, unknown/reserved metadata rejection;
- create/update/complete/reopen/delete/restore and stale-base matrices;
- exact replay, changed replay, concurrent edit, recurrence-state, and parent-note
  authorization;
- dormant server-origin REST/MCP capture, product/Sync split repair, bootstrap,
  pull, cursor, and acknowledgment; proof that public capabilities still omit both
  domains; and
- bounded query plans plus lifecycle/RLS/cross-owner/repair tests on SQLite and
  live PostgreSQL.

### PR 3

- event immutability, ID/fingerprint replay, provenance, ordering, correction, and
  irreversible tombstone;
- exact legacy event mapping/rejection, `(created_at, id)` bootstrap ordering,
  deleted-at/reason hashes, revision-1/create and revision-2/tombstone pages;
- transition-to-single-event behavior under retry and crash repair;
- task/note parent mismatch and cross-owner denial;
- resumable existing-event bootstrap and exclusion of read state; and
- batch ordering, pull/ack, restore visibility, and proof that public capabilities
  still omit both domains; and
- activity lifecycle/RLS/cross-owner/index/capture/repair tests on SQLite and live
  PostgreSQL.

### PR 4

- the complete three-way reconciliation matrix;
- explicit-task protection, stable checklist identity, ambiguous parser input, and
  privacy-safe drift lifecycle;
- projection-cache loss/rebuild from markers plus immutable envelope bases,
  retention protection, malformed/duplicate markers, line removal/unlink, task
  delete, restore, and explicit relink;
- linked-tombstone and open-drift compaction blockers/release, including cache loss;
- note/task/activity mutation-group crash windows and repair;
- 499-change boundary acceptance and pre-append rejection of 500 changes or any
  computed plan over ADR-034's 1,000-step cap;
- multi-device concurrent completion, recurrence-state, deletion, restore, and
  checklist edits;
- bounded pagination and plan assertions; and
- end-to-end SQLite and live PostgreSQL capability, bootstrap, pull, apply, replay,
  and repair.

Live PostgreSQL is a required gate in every PR: PR 1 proves migration/catalog/RLS;
PR 2 proves dormant task lifecycle, authorization, pagination, capture, and repair;
PR 3 proves dormant activity immutability, ordering, bootstrap, capture, and repair;
and PR 4 proves activation and end-to-end convergence. An unavailable suitable
server blocks the PR rather than converting the evidence to an accepted skip.

Every PR runs affected Notes/Sync regression tests, Ruff, Bandit on changed
production paths, `py_compile`, and `git diff --check`. The final PR runs the broad
Notes and Sync suites plus public capability/documentation checks.

## Acceptance mapping

| TASK-13006 criterion | Design coverage |
| --- | --- |
| Versioned task/activity domains; upsert/tombstone; immutable new activity IDs | Domain contracts; authority and identity |
| Stable task identity, complete fields, optimistic base, SQLite/PostgreSQL | Canonical task fields; storage/migration; task lifecycle |
| Ordered immutable activity with actor/time/transition/source | Canonical activity fields; activity contract |
| Deterministic rebuildable Markdown with review drift | Markdown projection convergence |
| Server-origin canonical envelopes while Sync active | Server-origin/client-origin flow; durable groups |
| Concurrency, recurrence, completion, delete, restore, checklist edits are idempotent or reviewable and bounded | Conflict semantics; rollout; verification strategy |

## ADR check

ADR required: yes

ADR path: `Docs/ADR/039-canonical-notes-task-sync-and-derived-checklist-projections.md`

Reason: TASK-13006 establishes two public Sync domains, product/storage authority,
immutable history semantics, projection ownership, PostgreSQL tenancy rules,
cross-database recovery behavior, and a long-lived recurrence boundary.
