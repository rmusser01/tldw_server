# DB-backed immutable report evidence

Date: 2026-09-05
Task: TASK-13153, prerequisite to output-reservations Task 7
Status: Direction, lifecycle and compatibility approved in conversation; written spec awaiting user review
ADR required: yes
ADR path: `backlog/decisions/003-reading-atomic-hard-delete.md` (amendment)
Reason: Changes evidence storage, ownership, deletion and migration boundaries.

## Purpose and scope

Replace newly generated Watchlists evidence JSON sidecars with bounded, immutable
Collections records shared by explicitly linked output variants. Evidence must
survive deletion of one variant, disappear transactionally with its last owner,
and remain readable when the output-file volume is offline.

Implementing this closes the Task 4c sidecar prerequisite. It does not activate output storage,
advertise optimistic deletion, migrate unrelated artifact systems, delete legacy
files, or complete the remaining producer inventory. No new public job, generic
blob store, general outbox or automatic directory sweep is introduced.

## Current behavior and alternatives

`watchlists.py:create_output` writes the primary output, writes a separate JSON
snapshot, then adds `report_snapshot_path` to output metadata. No structural row
owns the sidecar. Variant metadata is copied before that update: existing variant
relationships must not be presumed to establish evidence ownership.
Activated `_load_report_snapshot_for_user` currently returns sanitized 503 rather
than treating a metadata filename as permission to read a file.

The user selected shared DB evidence over two alternatives:

- Owned JSON files would require another shared-file publication, reference,
  recovery and migration lifecycle, including volume availability for reads.
- Copying the entire snapshot into every output's metadata duplicates potentially
  large payloads and blurs immutable evidence with editable presentation metadata.

The selected approach adds narrow relational evidence storage to Collections and
reuses its existing transaction fence and output identities.

## Records and authority

Use three narrowly scoped tables, with corresponding SQLite/PostgreSQL migrations:

| Record | Authority and essential data |
| --- | --- |
| Evidence policy | One row per user: positive finite maximum snapshot bytes and total retained evidence bytes; operator-managed, not inferred from filesystem settings |
| Evidence snapshot | Opaque generation identity, user, immutable canonical JSON, full SHA-256, UTF-8 byte size, schema version, historical primary output ID and creation time |
| Output evidence link | Same-user snapshot identity and exact output incarnation; at most one snapshot per output incarnation |

Use the existing unique `(user_id, file_incarnation)` output identity as the link
target. Numeric output IDs alone are insufficient: deletion and ID reuse cannot
reattach an old snapshot. Composite foreign keys reject cross-user links and
links to nonexistent output incarnations. Index lookup and ownership checks by
user and snapshot. Declare key columns explicitly NOT NULL on both backends.
Outputs also carry a structural `evidence_required` flag, default false for existing
rows. Only evidence publication/migration sets it, atomically with the link; generic
metadata updates cannot clear it. This makes a missing required link detectable
without trusting an editable metadata indicator.

The new opaque generation identity is internal. The existing payload's public
`snapshot_id` is retained for compatibility, but is not an ownership token or a
full-content digest. Do not deduplicate generations merely because their content,
title, source IDs or public snapshot IDs match. Explicit variants of one generation
share a record; unrelated users and independently requested generations do not.

Canonical JSON uses deterministic key ordering, UTF-8 and strict JSON values
(no NaN/infinity). Exclude only the response's `output_id` from this canonical body;
store the historical primary output ID separately when the primary commits. This
lets admission and the prepared digest precede numeric ID allocation without
reserializing a large snapshot inside the write transaction. Reads project that
immutable historical ID back into the unchanged response schema. Legacy import
extracts and validates the same field before canonicalizing its bounded payload.

The snapshot body and digest cannot be edited through ordinary output metadata
updates. Changing a report's evidence requires a new generation, not overwriting
history. Links likewise are not writable via public metadata fields.

## Atomic publication, replay and cancellation

Build and serialize the evidence with explicit bounds before logical output
publication. Rendering and serialization occur outside DB transactions. Reuse
the output producer's admitted operation and its bounded rendering interval;
no separate scratch file or unlinked persisted snapshot is needed.

For the primary output, one explicit DB connection and revision-clock fence cover
output insertion, evidence-policy admission, snapshot insertion, link insertion,
and the existing output operation's committed transition. Failure rolls back all
these logical changes. Files already published by the existing no-clobber protocol
retain its abort/recovery authority; they do not become a successful partial report.

Keep the snapshot body out of the file journal's small `intended_json`. The producer
passes bounded evidence to the logical commit boundary; the prepared operation
records only its generation identity, digest and intended link role. The commit
validates these against the supplied body. Recovery never reconstructs or invents
evidence to finish an uncommitted producer: prepared operations remain abortable.

Each later variant atomically creates its output and links the existing snapshot.
It must verify the expected same-user generation under the fence. If the last
owner was deleted before a delayed variant completes, reject that completion;
do not recreate the erased snapshot from the producer's old in-memory copy.

Replay of the same admitted operation or an existing producer idempotency key must
resolve its exact committed output/evidence association before attempting another
insert. A mismatched digest or association is a conflict, not an overwrite.
Reuse the file journal's conditional commit/outcome handling; unknown outcomes
remain unconfirmed. A retired token cannot mint a fresh generation. A genuinely
new generation request is not globally deduplicated against historical requests.

Preserve the current multi-variant success/failure response policy. Compensating
cleanup of committed variants uses their normal protected deletion boundary;
uncommitted variants abort their reservations. Cancellation drains in-flight
mutation workers. It never unlinks a sidecar or deletes a snapshot speculatively.
Compensation carries each committed output's incarnation and checks it under the
deletion fence; a reused numeric ID cannot redirect cleanup to a later output.

## Deletion and retention

Soft deletion retains the output's evidence link and its quota usage. Hard deletion,
including metadata-only hard deletion, removes that link. Delete the snapshot only
when no links remain, including links from soft-deleted outputs.

Perform link removal and last-reference deletion in the same transaction as output
removal, under the existing revision-clock fence. Use foreign-key cascade for link
removal and one shared DB helper for deleting newly unreferenced snapshots; do not
maintain a mutable reference counter or repair ordinary deletion with a sweeper.
Wire this helper into every output-deletion transaction, including journal removal,
legacy/managed deletion and Reading parent cascade. All attachment paths take the
same fence, so last-owner deletion and variant attachment cannot both win.

Never delete linked external Media, Notes, source items or their evidence. No
separate evidence-delete endpoint is added. Existing file-deletion permission
continues to govern files, not retention of DB evidence after its last output dies.

## Bounds and quota

Persist explicit operator-supplied limits per user before admitting DB-backed
evidence production or migration. No zero/unlimited default, schema-time activation,
or fallback to file-staging limits. A shared PostgreSQL database may be on a different
volume from the output files; its evidence budget is a distinct logical budget.

Measure canonical UTF-8 JSON bytes, not Python character count. Validate snapshot
shape and field/collection sizes during bounded construction and serialization;
checking size only after creating an arbitrarily large encoded object is insufficient.
Reject oversize rather than silently dropping evidence. Existing report-selection
rules that explicitly record excluded-item truncation remain unchanged.

Within the write fence, compute retained usage from unique snapshot rows, counting
shared snapshots once and soft-deleted owners normally. Admit the actual snapshot
size atomically with its first link. Concurrent commits cannot overspend the budget;
replay and additional links cannot charge it again. Reads and deletions do not need
a healthy filesystem or new producer admission.

Policy updates are explicit, fenced and reject limits below existing retained usage
or existing snapshot sizes; they do not evict or rewrite evidence. Logical quota
release does not promise immediate physical DB-file shrinkage. Actual DB failures
still roll back or use the existing uncertain-commit handling.

## Read API and compatibility

Keep `WatchlistOutputEvidenceResponse` and its snapshot schema: `output_id`,
`immutable_snapshot`, `snapshot` and `readiness` remain. Resolve the authenticated
user's accessible current output and its exact evidence link in one coherent DB
snapshot. Fetch evidence by that relationship, never by a metadata-supplied ID.
Return the envelope's requested output ID. Preserve the snapshot body's historical
primary `output_id`; it is provenance, not a lookup or permission for that old row.

DB-linked evidence reads perform no output-root resolution, marker check, namespace
activation, filesystem read or filesystem cleanup. They remain available with the
volume offline. Evidence endpoints must not invoke output purging as a read-side
effect; normal expiry/access checks return unavailable output without mutations.

For new or migrated reports, stop emitting `report_snapshot_path`; no fictional JSON
filename replaces it. Existing readiness/summary metadata stays available as a
projection, but the linked immutable snapshot is authoritative for evidence.
This is an intentional metadata change while the evidence response schema stays
compatible. Clients must use the evidence endpoint rather than opening sidecars.

Legacy outputs without any recorded immutable evidence keep the existing explicit
live-only response. A missing DB link when the structural flag requires evidence is
corruption, not permission to degrade silently. A bounded DB-evidence version indicator
may be projected into response metadata for diagnostics; it grants no ownership
authority and is not used to decide whether a missing link is acceptable.

For unmigrated sidecar-backed records, genuinely inactive stores may retain their
bounded legacy read path during rollout. Activated stores keep the current sanitized
503 until reconciliation succeeds. Never read an arbitrary metadata path from an
activated store or fabricate an immutable snapshot from today's live sources.

| Failure | Public behavior |
| --- | --- |
| Inaccessible, absent or expired output | Existing output-not-found 404 |
| Same operation with conflicting identity/body or late variant after last-owner deletion | 409 `report_evidence_conflict` |
| Snapshot exceeds configured per-snapshot bytes | 413 `report_evidence_size_limit` |
| Total evidence admission exceeds the user's retained-byte limit | 507 `report_evidence_capacity` |
| Producer/import policy unavailable or invalid; DB evidence unavailable/corrupt; commit outcome unconfirmed | Sanitized 503 `report_evidence_unavailable` |
| Unmigrated sidecar on an activated store | Existing 503 `output_storage_unavailable` |
| Legacy sidecar missing on an inactive store | Existing 404 `report_snapshot_missing` |

Log only stable categories and operational counts. Do not log titles, URLs,
payloads, raw DB errors, filesystem paths or sensitive migration manifest contents.

## Explicit legacy migration

Use an offline, dry-run-first reconciliation command with writers stopped. Schema
migration creates tables/constraints only: it reads no user files, assigns no links,
provisions no volume and enables no capability. Evidence policy provisioning is
explicit and does not activate the output-file protocol.
Dry run uses a non-initializing connection to an existing schema and reads policy
without provisioning it. If setup is missing, report the prerequisite; do not use
a convenience factory that creates databases, tables, policies or directories.

An operator selects the user, verified legacy root and policy, reviews a sensitive
manifest, and confirms exact output-to-snapshot mappings. Validate/provision volume
identity only through the existing explicit operator workflow, never from a reader.
Read through the held directory/file descriptors under the existing storage lock;
bound bytes before JSON parsing. Relative/absolute legacy names are candidates,
not authority to escape the chosen root or follow symlinks.

The manifest records output incarnation and complete-row digest, source identity,
source bytes/digest, a stable import-generation identity and intended same-user
mappings. Validate the snapshot schema and compare its provenance with those mappings.
A shared filename, `variant_of`,
matching title or partial public snapshot hash alone is not sufficient evidence.
An operator may confirm a legitimate mapping; contradictory or missing provenance
must be resolved, not silently inferred. Existing variants with no sidecar reference
are not automatically attached to the primary snapshot.

Apply reopens and verifies the same source under exclusion, then rechecks exact
output incarnations, unchanged records and policy under one short DB fence.
Reject rows with conflicting live file-operation claims; migration cannot bypass
the existing metadata mutation fence even when operators expect writers stopped.
Import the snapshot and the manifest's links together; remove obsolete path metadata
and set the structural evidence-required flag in that same transaction. Bound each migration
group; do not hold a DB transaction during file reads or parsing. Before opening
files on a repeat apply, resolve the manifest's import-generation identity and
verify all exact incarnations, links and immutable payload digest. An already-applied
identical group returns a DB-only no-op without rewriting current metadata or
charging again; missing/replaced owners never become an instruction to recreate them.
For a group not already applied, changed files, records, mappings or payloads reject
the group without partial changes.

Leave original files untouched, including after successful import. The command
does not delete them, normalize their contents or queue them for cleanup. It may
report them as retained legacy files in its sensitive operator output. Missing,
oversized, corrupt or ambiguous evidence blocks that group's migration and that
user's activation readiness; unrelated valid groups can still be migrated explicitly.

## Verification and delivery boundaries

Targeted tests must use real SQLite and PostgreSQL stores and cover:

- Migration twice, explicit NOT NULL/composite-FK constraints and cross-user isolation.
- Primary output/evidence/link rollback on failure, lost commit acknowledgement,
  replay, conflicting body and simultaneous same-generation commits.
- Variant linkage, primary deletion with a surviving variant, soft deletion, final
  hard deletion through every production boundary, ID reuse and late completion.
- Actual UTF-8 limits, malformed/deep/oversized data, bounded construction, concurrent
  quota admission, shared usage, deletion release and rejected policy lowering.
- DB evidence reads with a missing/replaced output volume and no file/purge side
  effects; metadata tampering cannot borrow another output's snapshot.
- Real Watchlists generation/evidence API flow, multi-variant failure/cancellation,
  expiry, legacy live-only behavior and the deliberate path-metadata compatibility change.
- Dry run without writes; apply/replay; changed records/files; wrong volume, path
  escape/symlink, missing/corrupt evidence; and preservation of all source files.

After written-spec approval, create a bounded implementation plan for storage and
deletion, producer/read integration, and offline migration. Verify those checkpoints
before resuming the rest of Task 7. Run scoped lint/format/security checks; do not
run a full suite or reprovision Docker without permission. The additional output
writers identified in the parent plan still require classification and coverage.
This amendment alone never enables `hasReadingOptimisticDeletesV1` or makes draft
PR #2903 merge-ready; the human-written Change summary gate remains.
