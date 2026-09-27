# ADR-039: Canonical Notes task Sync and derived checklist projections

**Status:** Accepted
**Date:** 2026-08-13
**Backfilled from:** not backfilled
**Decision owner:** TASK-13006 requester and design review
**Related task:** TASK-13006
**Related spec/plan:** `Docs/superpowers/specs/2026-08-13-notes-task-activity-sync-design.md`
**Depends on:** ADR-031, ADR-034, ADR-037, and ADR-038
**Proposed amendment:** ADR-040 evolves the scope-authority row for independent
Notes graph binding when TASK-13007 is implemented.

## Decision

Synchronize Notes tasks and their immutable history as two independent versioned
Sync v2 domains: mutable `notes.task` objects and append-only
`notes.task_activity` objects. Keep Markdown checklists, projection cursors, read
state, and reconciliation bookkeeping as derived local state.

The canonical product authorities remain the existing ChaChaNotes task tables:

- `note_tasks` owns each task's stable identity, parent note, mutable fields,
  canonical Sync revision/hash, and soft-delete lifecycle. Its existing `version`
  remains the REST row version because projection-only operations already advance
  it; a new canonical revision/hash advances only for portable task mutations;
- `task_events` owns immutable user-visible task history; and
- `task_event_read_state`, `task_note_projections`, and
  `note_task_reconciliation_state` remain local projections or UI state and are
  never synchronized.

`notes.task` version 1 supports whole-object `upsert` and `tombstone`. Create
requires an empty Sync head. Update, completion, reopen, delete, and restore require
the exact current base cursor, object revision, and object hash. Restore is an
explicit upsert against the current tombstone. Task identity and parent-note
identity are immutable.

The version-1 task payload exposes stable task ID, parent note ID, title,
description, status, priority, due date, completion state, recurrence state,
assignee, tags, estimate, and bounded custom metadata. Existing product columns
remain authoritative where present. Existing `note_tasks.text` backs the wire
`title`; existing REST APIs retain their `text` compatibility field. Extended
fields remain in strictly validated `metadata_json` rather than creating parallel
columns. Custom metadata may not duplicate reserved fields.

Recurrence synchronization preserves only a validated recurrence rule and its
state. Version 1 does not schedule work or generate future task instances.

`notes.task_activity` version 1 creates one immutable event with a stable opaque
event ID, parent task and note identities, actor, client timestamp, transition,
source, and bounded metadata. Authenticated principal and source-device provenance
are server-bound, not client-selectable. Server cursor plus event ID defines
deterministic ordering; client timestamps never do. Exact ID and canonical
fingerprint replay is idempotent, while reuse of the ID with different content is
a conflict. Activity tombstone is one-way and cannot rewrite or restore event
content.

Every activity has a required authorized note and an optional task. When a task is
present, it must belong to that note. All five existing task-side tables, plus the
new drift table, gain collision-safe owner/dataset identity, forced PostgreSQL RLS,
and owner-scoped service predicates. Migrations are transactional and fail closed
on catalog drift. Existing data and REST behavior are preserved before a dataset
explicitly enrolls in the new domains.

Schema v60 also owns one private `note_task_scope_authority` relation. It records at
most one immutable real dataset for each owner and uses owner-only forced PostgreSQL
RLS. Absence of a row means the owner's task graph remains in the private
`local-unbound` scope. Binding inserts the authority row atomically with the graph
rekey, including when the graph is empty; replay to the same dataset is idempotent
and a different target is rejected. Compatibility callers resolve this indexed
product-owned row, never accept a client dataset selector, and all shared task-store
writes must match the recorded authority.

### Prospective ADR-040 amendment

ADR-040 preserves this relation as the sole immutable owner-to-default-personal-
dataset authority but adds one-way `task_graph_bound`, `moodboard_graph_bound`, and
`studio_graph_bound` state. Existing rows migrate with `task_graph_bound=true`.
The migration first verifies that the complete task graph matches the authority
dataset and aborts on inconsistency. After that migration, row presence alone no
longer proves that the task graph is bound: task compatibility reads and writes
resolve the authority dataset only when
`task_graph_bound=true`, otherwise they retain the private local-unbound behavior
defined here. Task binding changes its flag from false to true in the same
transaction as complete task-graph verification/rekey. Moodboard and Studio bind
their own graphs and flags independently against the same immutable dataset.

The additive migration defaults `task_graph_bound=true` so an older task binder's
owner/dataset-only insert still records the task rekey it performed. New binders
write all flags explicitly, and ADR-040 activation excludes row-presence-era task
servers before a moodboard/Studio-first authority insert can occur.

This amendment becomes effective only with TASK-13007.1's schema and task-caller
changes. Until then, schema-v60's original row-presence semantics above remain the
implemented contract.

Because the product migration cannot read the separate Sync database, preserved
inactive rows use a private owner-scoped local-unbound dataset sentinel. Explicit
enrollment source-verifies and atomically rekeys the complete task graph to the
real dataset; the sentinel is never client-selectable, writable through Sync, or
capability-ready.

Markdown checklist text is a deterministic projection, not a competing authority.
Each managed checklist line carries a hidden versioned marker containing its task
ID and exact last-projected task revision/hash. The marker reconstructs identity
after projection-cache loss; the immutable Sync envelope named by that tuple
reconstructs the last-common projected field snapshot. If that base is unavailable,
reconciliation fails safe to review rather than guessing by text. Reconciliation
compares that base with the current canonical task and current Markdown:

- Markdown-only changes become canonical task mutations plus activity;
- task-only changes rebuild the Markdown projection;
- compatible changes converge deterministically; and
- incompatible concurrent changes create a bounded, privacy-safe drift record for
  review instead of silently overwriting either side.

Server-origin REST, MCP, and reconciliation mutations use ADR-034 durable mutation
groups when they affect task, activity, and note projection together. The complete
ordered canonical envelope plan is appended atomically in the Sync store before
product materialization. Product and Sync databases do not share a distributed
transaction; idempotent repair closes a product-commit/status-commit split.

Rollout uses existing per-dataset enrollment and readiness. It adds no new global
environment flag. Neither domain is publicly advertised as supported or writable
until the fourth PR supplies deterministic task/activity group expansion,
projection convergence, bootstrap verification, and repair. Existing datasets are
never silently enrolled.

Because every portable task mutation emits immutable activity, task writability is
coupled to both domains: task and activity capture are enabled atomically, both
bootstraps and both repair paths must be ready, and neither domain is writable when
that invariant is incomplete.

TASK-13006 is delivered as four atomic pull requests:

1. contract, storage, migration, RLS, catalog verification, and dormant readiness;
2. a complete but dormant `notes.task` lifecycle and bootstrap;
3. complete but dormant immutable `notes.task_activity` capture and bootstrap; and
4. deterministic compound mutation groups, Markdown projection convergence,
   capability activation, end-to-end testing, and public documentation.

## Context

The server already stores Notes tasks, events, projection state, and read state,
but neither task authority nor task history is a Sync v2 domain. Existing task rows
have a stable identifier, parent note, text, open/done state, metadata, optimistic
version, and soft deletion. Existing events are append-only, while Markdown
checklists are reconciled through local projection tables.

Without an explicit ownership decision, synchronizing all of these tables would
create competing authorities. A Markdown edit could overwrite an explicit task, a
read marker could leak device-local UI state, and mutable event replication could
rewrite history. Conversely, treating Markdown as the only authority would lose
first-class task identity, recurrence state, structured metadata, and reliable
multi-device conflict handling.

ADR-031 requires independently mutable Notes resources to use independent Sync
domains. ADR-034 provides the durable ordered group required when one user action
changes a task, emits history, and rebuilds a note projection. ADR-037 establishes
the corresponding rule that deterministic projections are rebuilt rather than
synchronized.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Synchronize Markdown checklists as the task authority | Markdown lacks stable structured identity and would silently overwrite explicit task fields during concurrent edits. |
| Synchronize projection and read-state tables | They are deterministic or device-local state, not portable user authority. |
| Put task state and all activity inside one `notes.task` envelope | Append-only history would become a mutable whole-object conflict surface and grow without bound. |
| Permit mutable activity events | It would rewrite audit history and make exact replay unverifiable. Corrections are new events or one-way tombstones. |
| Generate future task instances from recurrence rules | It adds scheduler ownership, clock, deduplication, and offline-generation policy without an approved product requirement. |
| Add dedicated columns for every extended task field in the first slice | Existing bounded metadata can carry the validated fields without a disruptive compatibility migration or duplicate authority. |
| Make product and Sync writes one distributed transaction | The repositories already use ADR-034 durable plans and repair; a distributed transaction would add backend coupling without removing recovery requirements. |
| Advertise domains as soon as schema exists | Existing rows would be missing from initial pulls and server-origin mutations could bypass canonical capture before bootstrap completes. |

## Consequences

- Tasks gain exact multi-device lifecycle semantics while existing REST `text` and
  product storage remain compatible.
- Task history is portable, ordered, and immutable; device-local read state stays
  private.
- Recurring tasks synchronize their rule and state but require a later, separately
  approved scheduler if automatic instance creation is ever needed.
- A single user action may leave repairable cross-database work after a crash, but
  never an incomplete canonical mutation plan.
- Existing datasets require resumable source-verified bootstrap before either new
  domain is writable.
- Explicit task edits are never silently replaced by Markdown. Some concurrent
  edits therefore require user review.
- PostgreSQL upgrades and bootstrap verification can temporarily block affected
  writes and require live-backend verification before rollout.
- The four-PR split delays end-to-end checklist convergence until the final PR, but
  each earlier PR remains independently safe because incomplete domains are not
  advertised writable.

## Follow-up

- Implement the four TASK-13006 child PRs defined in the related design spec.
- Update public Sync capability documentation when each domain becomes writable.
- Keep automatic recurrence scheduling out of scope until a separate task and ADR
  define scheduler ownership and generation idempotency.
