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
status, soft delete, timestamps, client ID, and optimistic version. The existing
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
| Checklist text and projection cursor | `task_note_projections` plus note Markdown | Derived; rebuild/reconcile locally |
| Reconciliation bookkeeping | `note_task_reconciliation_state` | Local only |
| Activity read/dismiss state | `task_event_read_state` | Device/user UI state; never synchronized |

Each task and activity uses a stable opaque UUID identity. The client may submit
the resource ID but cannot select owner, authenticated actor, or trusted source
device. Task parent-note identity is immutable. Activity parent identities are
immutable.

### Canonical task fields

`note_tasks.text` is the storage authority for wire `title`. Existing REST routes
continue to expose and accept `text`; adapters translate at the boundary without
changing stored meaning.

The version-1 task payload contains:

- `task_id` and `note_id`;
- `title` and nullable `description`;
- `status` and explicit completion state;
- nullable `priority`, due date/time, recurrence rule/state, assignee, and estimate;
- bounded normalized tags; and
- bounded custom metadata.

Existing core fields stay in existing columns. Extended fields remain in
`metadata_json`, but they are first-class validated wire fields rather than an
untyped pass-through. Reserved wire keys cannot appear again in custom metadata.
Unknown fields, noncanonical values, duplicate tags, invalid recurrence data, and
out-of-bound strings/collections are rejected before hashing or persistence.

The recurrence contract stores a validated rule and recurrence state only. It does
not interpret wall-clock time to create another task. A future scheduler requires a
separate task and architecture decision.

### Canonical activity fields

An activity payload contains:

- stable `activity_id`;
- `task_id` and `note_id` parent identities;
- actor identity and source device as server-bound provenance;
- client event timestamp preserved as submitted metadata;
- transition/event type, source, and bounded structured metadata; and
- optional task revision/result references needed to verify the transition.

The stable ID is opaque, not a digest. A canonical fingerprint over the validated
immutable content protects exact replay. Server cursor and activity ID define
deterministic order. Client timestamps may be displayed but never resolve ordering
or conflict.

An activity must resolve to at least one authorized parent. If both task and note
are present, the task must belong to that note. A task identity from another owner
or dataset is indistinguishable from missing at public boundaries.

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

- A missing stable ID creates one event.
- Exact stable ID plus canonical fingerprint replay is idempotent.
- Reusing an ID with changed content is a stable idempotency conflict.
- An existing event cannot be updated.
- Tombstone requires exact identity/fingerprint lineage and is irreversible.
- Tombstoned activity cannot be restored or recreated under the same ID.
- Corrections are represented as a new activity that references the earlier event
  where the event taxonomy permits it.

Task transitions that require user-visible history create exactly one canonical
activity. A retry or crash repair cannot duplicate it because the event ID and
mutation-group step are stable.

## Storage, migration, and RLS

The first pull request extends the existing task schema rather than creating a
parallel product authority. Migration changes are additive except where an
explicit lifecycle field or constraint is required for immutable activity
tombstones and exact Sync bootstrap.

Both backends must enforce:

- stable UUID identities, lifecycle and version checks, bounded metadata, and
  immutable parent references;
- owner/dataset/note/task indexes for point lookup, current task pages, activity
  pages, bootstrap, and reconciliation;
- count and cursor/fingerprint queries that remain bounded and index-backed;
- owner and parent-note authorization in every read/write predicate; and
- exact current catalog verification before advertising readiness.

PostgreSQL uses forced RLS. Task policies require authenticated owner/dataset and an
owned parent note in `USING` and `WITH CHECK`. Activity policies additionally
require the referenced task, when present, to belong to the same owner, dataset,
and note. The current-version verifier enumerates every required column,
constraint, index, policy, role, command, and canonical expression and rejects
additional permissive policy drift.

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
4. Append under dataset authority and project under the ADR-034 materialization
   fence.
5. Persist task/activity product state, object state, and apply result
   idempotently.
6. Return success only for an applied or exact terminal replay.

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

## Bootstrap, capability advertisement, and rollout

No global environment flag is added. Existing dataset enrollment and readiness are
the rollout boundary.

Schema presence alone advertises neither domain as writable. Existing datasets are
not silently enrolled.

`notes.task` becomes dataset-writable only when:

- `notes.note` and `notes.task` are enrolled at supported adapter versions;
- source task bootstrap is complete, count/fingerprint verified, and resumable
  cursor state is `ready`;
- server-origin task capture and repair are enabled; and
- no reconciliation blocker or unresolved bootstrap drift exists.

`notes.task_activity` additionally requires activity enrollment, complete immutable
event bootstrap, and transition capture readiness. A server may list a completed
adapter version in supported capabilities while omitting it from the selected
dataset's writable map until these conditions hold.

Bootstrap is non-destructive and keyset-paged. Stable existing task/event IDs are
preserved. Each page records its source count, cursor, and privacy-safe fingerprint.
Final readiness rechecks the complete bounded aggregate. Restart resumes from the
last verified page. Source drift either captures a verified correction under the
dataset fence or leaves the dataset not ready; it never marks stale source data
ready.

Legacy devices receive only versions they negotiated. Per-domain adapter-version
cursors and acknowledgments prevent a version change from skipping or
acknowledging incompatible envelopes.

## Markdown projection convergence

Markdown checklists are a deterministic view over explicit task authority, with a
recorded last-common projection used for three-way reconciliation.

For each bounded reconciliation unit:

| Markdown since base | Task since base | Result |
| --- | --- | --- |
| unchanged | unchanged | no-op |
| changed | unchanged | validate Markdown edit, create/update/tombstone task, emit activity, then record new projection |
| unchanged | changed | rebuild Markdown from canonical task and record new projection |
| changed compatibly | changed compatibly | converge to one canonical task/projection and record the complete durable group |
| changed incompatibly | changed incompatibly | retain both authorities, create a privacy-safe review drift record, and do not overwrite |

Checklist reordering alone does not change stable task identity. A parser cannot
claim an explicit task by text coincidence alone; it must carry or resolve the
stable projection identity. Unsupported or ambiguous Markdown remains visible and
creates reviewable drift rather than disappearing.

Projection records and drift bookkeeping never become Sync domains. After restore,
migration, parser change, or detected corruption, they can be rebuilt from canonical
notes, tasks, and activity. Read endpoints do not perform unbounded maintenance
writes.

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
- preserve current REST/data behavior;
- add fail-closed source-reconciliation and readiness state; and
- advertise neither new domain writable.

### TASK-13006.2 — `notes.task`

- implement strict upsert/tombstone adapter and idempotent materializer;
- enforce dataset, note, identity, exact base, completion/reopen, restore, and
  recurrence-state rules;
- capture REST/MCP server-origin task mutations and repair split commits;
- bootstrap existing tasks and prove pull/ack behavior; and
- advertise only `notes.task` when its full readiness predicate is true.

### TASK-13006.3 — `notes.task_activity`

- implement immutable create and one-way tombstone;
- bind actor/device provenance and deterministic ordering;
- capture canonical events for task transitions with exact replay deduplication;
- bootstrap existing stable event IDs and exclude read state; and
- advertise activity only when bootstrap, capture, pull, and materialization are
  complete.

### TASK-13006.4 — Projection convergence

- implement the three-way task/Markdown matrix and privacy-safe drift records;
- use durable multi-object mutation groups across note, task, and activity;
- prove concurrency, pagination, repair, and two-client end-to-end convergence on
  SQLite and PostgreSQL; and
- publish final capability/API documentation and close TASK-13006.

Each PR must be independently mergeable. Dormant schema or code may land before a
domain, but capability advertisement cannot cross a later PR's boundary.

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
- server-origin REST/MCP capture, product/Sync split repair, bootstrap, pull,
  cursor, acknowledgment, and capability readiness; and
- bounded query plans on both backends.

### PR 3

- event immutability, ID/fingerprint replay, provenance, ordering, correction, and
  irreversible tombstone;
- transition-to-single-event behavior under retry and crash repair;
- task/note parent mismatch and cross-owner denial;
- resumable existing-event bootstrap and exclusion of read state; and
- batch ordering, pull/ack, restore visibility, and capability readiness.

### PR 4

- the complete three-way reconciliation matrix;
- explicit-task protection, stable checklist identity, ambiguous parser input, and
  privacy-safe drift;
- note/task/activity mutation-group crash windows and repair;
- multi-device concurrent completion, recurrence-state, deletion, restore, and
  checklist edits;
- bounded pagination and plan assertions; and
- end-to-end SQLite and live PostgreSQL capability, bootstrap, pull, apply, replay,
  and repair.

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
