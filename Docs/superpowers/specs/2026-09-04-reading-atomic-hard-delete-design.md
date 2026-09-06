# Reading atomic hard-delete design

Date: 2026-09-04 · Task: TASK-13153 · Status: approved, implementation in progress

## Goal and governing decision

Allow a client to permanently delete exactly the Reading capture it confirmed,
without removing newer changes, external Media/Notes, or unrelated artifacts.
The full normative contract is in
`backlog/decisions/003-reading-atomic-hard-delete.md` (ADR required: yes).

## API

`DELETE /api/v1/reading/items/{id}?hard=true&expected_revision=N`

| Condition | Response | Mutation |
| --- | --- | --- |
| Missing authentication | Existing auth response | None |
| Missing hard-delete revision | 428 | None |
| Non-positive/non-integer revision | 422 | None |
| Missing, wrong owner, or non-Reading row | 404 | None |
| Revision mismatch | 409 `reading_revision_conflict` | None |
| Ambiguous legacy artifact ownership | 409 `reading_artifact_ownership_conflict` | None |
| Hard-delete storage/schema/cleanup readiness unavailable | 503 `reading_delete_unavailable` | None |
| Matching revision | 200, cleanup complete or pending | Atomic logical deletion |

All Reading item DTOs expose `revision: int > 0`. Returning 409 never substitutes
the new revision and retries deletion automatically; the client must refresh and
confirm again. `hard=false` keeps existing archive semantics. Do not echo
record data or filesystem diagnostics in errors.

Example successful response:

```json
{"status":"deleted","item_id":42,"hard":true,"artifact_cleanup":"pending"}
```

## Revision and transaction coverage

Use a persistent database counter to allocate strictly increasing Reading tokens.
Unchanged normalized field/tag values allocate nothing. A committed compound
mutation advances once; separate committed operations may each advance. Gaps are
allowed. Rollback rolls back both state and token allocation. Initialization and
upgrades are idempotent and serialized on both backends.

Inventory every writer, including generic `items`/bulk paths, imports, highlight
reanchoring, note links, archive generation, generic output modification/deletion
and retention of owned outputs. They must lock and validate the Reading parent in
the same transaction as their writes. Deleting external links changes the Reading
association; editing external content does not. Generic non-Reading consumers keep
their existing API behavior.

An item response must not combine fields from one revision and tags/highlights
from another. Use a single read snapshot, including PostgreSQL isolation appropriate
to multi-statement reads. No filesystem or network work inside mutation transactions.
Reuse the existing `_read_snapshot()` and SQLite `BEGIN IMMEDIATE` helpers.

## Owned data and lifecycle

Delete item-local tags, highlights, note links, FTS entries, collection memberships
and structurally owned output records. Preserve shared tag definitions, collections,
Media and Notes. Proven ownership is not inferred solely from editable JSON.

Archive staging has a durable reserved path before file creation. Adoption verifies
the original item's revision and moves the reservation to owned-output state in one
transaction. Failed or stale adoption schedules disposal, never upserts the parent.
Expired unadopted reservations are reclaimed with a lease/fence so an active writer
cannot adopt a file concurrently being removed.
Additionally, hold a stable per-user OS storage lock across creation/write/close
and adoption, or removal and intent completion. Acquire storage lock before any
database transaction and recheck state after acquiring it. Database-only item
deletion never waits for the storage lock inside its transaction. A delayed writer
whose reservation has been retired aborts before file creation. Never delete the
lock file, including after successful cleanup. Lease expiry alone cannot authorize
unlink or prove that a writer has stopped.

Reservations and cleanup intents are keyed by storage namespace, user and path.
The namespace is a provisioned opaque volume identity with a verified marker;
workers cannot infer it from a shared PostgreSQL connection. Before treating a
missing path as success, validate the mounted namespace and acquire its storage
lock. An unavailable volume remains pending. Shared-volume deployments require
verified interprocess locking; distinct volumes require distinct namespace IDs.
Unsupported lock/storage configurations disable hard deletion. Legacy output
namespace assignment is an explicit upgrade/reconciliation step.

### Legacy ownership recovery

Manual archives and older save archives may legitimately lack a parent reference.
Do not silently drop these candidates or automatically trust their metadata.
Provide `scripts/reading_reconcile_artifacts.py` with dry-run manifest generation
and explicit apply mode. Operators review same-user item/output IDs and confirm
associations; the manifest captures the item revision and fingerprints of candidate
records (including ownership-relevant metadata and storage location). Apply rejects
changed rows, cross-user targets, existing conflicting ownership or unknown volumes.
It acquires the database fence, creates associations, advances the item revision
and commits all confirmed entries atomically. It does not delete or move artifacts.
Repeated apply of an already identical association is a no-op; conflicting state
still rejects. Keep manifests local with restrictive permissions and no payloads
in diagnostic logs. Document recovery from 409 with this workflow.

Deletion queues only files without surviving output references. A pending cleanup
path cannot be attached to a new output. Generic output persistence and purge must
honor this rule for Reading-managed paths; no broad output-system rewrite is needed.
Cleanup retries are idempotent, bounded per run, restart-safe and independent of the
retention feature flag. Invalid paths remain blocked and observable, never unlinked.

## Rollout and evidence

### Approved generic-output file policy

Hard deleting a structurally owned Reading archive through `/outputs/{id}` requires
`delete_file=true`; otherwise return 409 `reading_file_deletion_required` without
mutation. API/scheduled purges with `delete_files=false` skip managed archives.
Soft deletion and unrelated outputs keep their existing retention semantics.
Explicit managed file removal commits a durable intent before any unlink; the
existing `file_deleted`/`files_deleted` counts describe actual immediate removals,
not queued work. Permission and purge eligibility are enforced under the DB fence.
This user-approved clarification avoids introducing a retained-file lifecycle.

Managed archive files are immutable (user-approved clarification, 2026-09-05).
Title/retention edits change metadata only. A changed managed path or format
rejects the complete request with 409 `reading_archive_file_immutable`; unchanged
values are no-ops. Unowned output rename/conversion semantics are preserved.
Ownership dispatch and metadata mutation must be atomic. This guard is only one
checkpoint: late ownership registration during generic file operations and writes
through shared source/target paths must also be fenced before enabling rollout.
Managed format conversion would require a separate staged replacement lifecycle.

The reservation approach for generic file mutations was approved on 2026-09-05.
Its detailed storage-boundary amendment is
`Docs/superpowers/specs/2026-09-05-reading-output-file-reservations-design.md`
(independent review passed; user written-spec review pending). It supersedes any suggestion above that a DB
precheck alone suffices for generic filesystem writers. No production rollout or
capability activation is implied by approval of the approach.

Stop old writers before upgrade. Migrate before advertising the capability. Keep
the capability absent/false until all transaction and lifecycle tests pass, including
real PostgreSQL tests (a skipped backend suite is not evidence).
Capability derivation defaults false on missing/failed readiness and never inherits
a true shipped fallback. Readiness includes route enablement, schema/contract
version, volume/locking validation and healthy registered cleanup lifecycle.
Docs-info reads established readiness without migrations or filesystem I/O. Its
deployment-wide claim must be conservative if user stores differ. The endpoint
rechecks readiness for the authenticated user's store and rejects unavailable
hard deletion with 503 before any mutation, even if the client cached a true flag.
Existing pending cleanup remains recoverable when new deletions are disabled.

Focused evidence: migration twice; token persistence and ID reuse; each mutation
and normalized no-op; wrong user; missing item; stale token; rollback after each
delete phase; mutation/delete in both commit orders; concurrent child creation;
stale background completion; coherent reads; FTS removal; external preservation;
ambiguous ownership; shared files; unlink failure/restart; crash during staging;
path reservation races; sanitized logs; canonical service-factory HTTP request.
Also require delayed-writer-after-missing-file-cleanup, process-exit lock release,
two-volume/shared-database isolation, missing-mount recovery, manual/older archive
reconciliation and stale manifests, and capability startup/failure/recovery tests.

The implementation plan maps these requirements to staged red/green checks.
