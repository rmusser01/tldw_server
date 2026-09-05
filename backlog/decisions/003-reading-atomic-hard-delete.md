# ADR-003: Revision-guarded Reading deletion and durable artifact cleanup

Date: 2026-09-04
Status: Accepted (implementation in progress; capability not yet active)
Task: TASK-13153

## Context

Reading captures are `content_items` with `origin=reading`, not external Media or
Notes. At base commit `59bbdd1bc990a86ee63d641a97be51c2bf6a81ed`, deletion uses
separate statements, omits highlights and archives, and has no revision guard.
Existing output purging is best-effort and cannot supply durable cleanup after
the owning row disappears. Archive ownership currently uses JSON metadata.

## Decision

Keep `DELETE /api/v1/reading/items/{id}`. Only `hard=true` requires a positive
integer `expected_revision`. Return 428 if absent, 422 if invalid, 404 for absent
or inaccessible Reading items, and 409 for stale revisions. Archive is unchanged.
Authenticate before disclosing item state. Non-Reading rows are not this API's
targets. A failed precondition removes nothing.

Add persisted positive integer revisions to content items and expose them in all
Reading summaries/details. Allocate Reading revisions from a transactional,
database-local monotonic counter, initialized above existing revisions. Gaps are
valid. The counter survives deletion, preventing a reused SQLite integer item ID
from matching an old confirmation token. Exhaustion fails closed, never wraps.

Every Reading-owned material mutation participates in the same transaction fence:
fields, tags, highlights, note associations, capture metadata and owned outputs.
Normalized no-ops do not advance the revision. External Media/Note edits do not
advance it. Lock the counter before the parent: SQLite takes the write transaction
before reading; PostgreSQL locks the counter row before the item. This intentionally
serializes Reading writes within a database. Keep I/O outside these short locks.
Read representations and their revision from a coherent database snapshot.
Reuse `CollectionsDatabase._read_snapshot()` and the SQLite backend's existing
`BEGIN IMMEDIATE` transaction helper, with explicit connection propagation; do not
introduce a second snapshot or transaction framework.

Hard delete checks owner, Reading origin and revision under that fence, then
deletes the parent with the exact revision predicate in the same transaction as
owned child rows, FTS cleanup, association removal, output rows and durable file
cleanup intents. Linked external Media and Notes survive. Roll back on cleanup SQL
failure; do not suppress FTS errors that would leave deleted content searchable.

Add structural Reading-output ownership, immutable to generic metadata updates.
Backfill only proven legacy associations using matching user, Reading parent,
archive type, parent archive reference and output metadata. Conflicting or
unprovable candidate ownership blocks hard delete with a documented 409
`reading_artifact_ownership_conflict`; it does not guess or silently leak artifacts.
The manual archive endpoint never established a parent archive reference, and the
save flow can retain older outputs beyond its latest reference. Neither case alone
is evidence of corruption. Provide an offline, dry-run-first reconciliation command
for these records: an operator explicitly confirms each same-user output/item
association from a reviewed manifest. Apply verifies the unchanged candidate
records and item revision in one transaction, records structural ownership and
advances the revision; it deletes no files or records. Mismatches fail closed.
The manifest is local sensitive data, not logs or a public API response. No automatic
ownership inference from filenames or editable JSON is added. Document the command
in the upgrade procedure so a 409 has an actionable repair path.
Shared files remain until their last output reference is removed. Shared tags and
containing collections survive; only item associations are removed.

Managed archive update policy (user approved 2026-09-05): title and retention
edits are metadata-only; a managed file's path and format are immutable. Reject
a compound update requesting a changed path or format with
`reading_archive_file_immutable` before changing any metadata or file. Repeating
the existing format/path is a no-op, not a conversion. Unowned outputs retain
their existing rename/conversion behavior. A staged replacement lifecycle for
managed conversions is deferred. Ownership dispatch and metadata updates share
the revision fence; generic filesystem writers still require exclusion against
late ownership registration and managed source/target aliases before rollout.

Record unlink intents transactionally before removing output rows. A small
Reading cleanup worker retries them after commit and on restart, independently
of optional retention purging. Missing files count as success. Unlink failures
remain pending with bounded retry backoff. Use existing confined output-path
validation; never log paths, URLs, titles, content or raw exception strings.
Reserve each queued `(user_id, storage_path)` against new output attachments until
cleanup completes; attachment and cleanup use the same database fence. Existing
shared references prevent scheduling unlink. New Reading archives use random,
exclusive-create names that are never intentionally reused. Do not hold the item
transaction during scraping, rendering or filesystem I/O.

Managed-output file policy (user approved 2026-09-05): generic output hard deletion
with `delete_file=false` rejects structurally owned Reading archives with 409
`reading_file_deletion_required`, without mutation. Purges with `delete_files=false`
skip these archives. Soft deletion retains ownership and files; unrelated outputs
keep their file-retention options. Explicit file deletion queues managed cleanup
in the deletion transaction, never unlinks first. Enforce permission under the
ownership fence, not only in an API precheck. Retention eligibility is rechecked
there using the requested grace period and retention selection. No retained-file
lifecycle is added: deleting ownership while promising indefinite file retention
would otherwise discard the authority needed for later capture cleanup.

Background completion must recheck the surviving parent and captured revision
under the write fence. It cannot recreate the item via upsert after deletion.
Stale completion rejects its item update and disposes of its privately staged
artifact. File staging and adoption must be restart-recoverable; record the
reserved path before writing so a crash cannot strand an untracked archive.
Lease expiry alone is not a filesystem fence. Creation and removal also acquire
the same stable, per-user storage lock, then recheck reservation state in a short
database transaction before touching the file. Hold the storage lock through file
creation/write/close or unlink and final state recording. Never acquire this lock
while holding a database transaction. Cleanup cannot retire a reservation while its
writer holds the lock; a writer delayed until after cleanup must recheck and abort
before opening the path. The lock file is persistent and is never unlinked/replaced
by cleanup. Process exit releases the OS lock; recovery still uses durable intents.

Every reservation/intent carries an opaque `storage_namespace_id` identifying the
actual output volume, not just a user/path or database. Workers only process their
validated namespace. A missing file is success only after validating that volume's
identity marker and holding its storage lock. Missing mounts, mismatched markers
and unavailable namespaces leave cleanup pending. Shared PostgreSQL does not imply
shared files: nodes on distinct volumes have distinct namespaces. Nodes sharing a
volume need the same marker and verified cross-process filesystem-lock semantics.
Use supported standard-library OS locking; unsupported storage/locking fails
readiness rather than falling back to lease-only deletion. Legacy records must be
bound to a verified volume during reconciliation, never to whichever worker sees
them first. Namespace identifiers/paths are not added to public responses.

Successful HTTP 200 reports `status=deleted`, `item_id`, `hard=true` and
`artifact_cleanup=complete|pending`. Pending means logical deletion is committed,
not that the user should issue DELETE again. Repeated DELETE returns 404. Do not
add a new public job/receipt API in this task. Operational pending counts and
sanitized error categories provide cleanup observability.

Advertise `hasReadingOptimisticDeletesV1=true` only after migrations, all writers,
the endpoint and enabled cleanup lifecycle satisfy this contract. Mixed old/new
writers are unsupported: upgrade with old processes stopped before enabling it.
This flag must not enter the existing true-valued shipped fallback. Derive it
fail-closed from enabled Reading routes, schema/contract version, validated storage
and lock support, and registered healthy cleanup lifecycle. Unknown state or failed
derivation is false. The DELETE endpoint independently checks the same readiness
and returns 503 `reading_delete_unavailable` without mutation when unavailable;
cached docs-info is not authority. Recovery of already-pending intents continues.
Docs-info performs no schema changes or filesystem probing; it reads established
readiness state. Where docs-info lacks user scope, it makes only a conservative
deployment-wide claim; the endpoint additionally checks the target user's store.

### Generic filesystem boundary amendment (2026-09-05)

The user approved extending durable reservations and existing storage exclusion
to generic file mutations, preserving unmanaged rename/conversion. The detailed
contract is in
`Docs/superpowers/specs/2026-09-05-reading-output-file-reservations-design.md`
(independent review passed; user written-spec review pending). It introduces a bounded output operation journal,
not fake Reading parents or a public job system. All conflicting output/ownership
writers honor its path and row reservations. Copy-before-commit preserves sources;
no-clobber publication, durable phases and identity-checked recovery preserve
cleanup authority. The existing OS-lock-before-DB ordering remains mandatory.

The detailed spec explicitly calls out occupied-destination rejection, missing
source handling and fail-closed activated-store behavior for user review. A
persisted per-user protocol/volume binding prevents runtime fallback to legacy
file-first operations; activation is a stopped-writer upgrade. No capability is
enabled by this amendment. Production code remains unchanged at this design stage.

## Alternatives and consequences

- Timestamp tokens fail to cover child-only writes and can collide.
- A service precheck cannot prevent a concurrent mutation; storage must enforce it.
- Per-item counters starting at one permit stale-token reuse after SQLite ID reuse.
- Filesystem-first deletion cannot roll back. Best-effort post-commit deletion
  loses retries. Durable intents are necessary, not a general job platform.
- A database-local counter is simpler than rebuilding shared item identity tables,
  but serializes Reading mutations. Measure contention before adding finer locks.
- Ambiguous legacy ownership requires repair before that item can be hard-deleted.
  Safety takes precedence over silently guessing ownership.

## Verification and scope

See `Docs/superpowers/specs/2026-09-04-reading-atomic-hard-delete-design.md` and
`Docs/superpowers/plans/2026-09-04-reading-atomic-hard-delete.md`.
Require real SQLite and PostgreSQL transaction races, rollback, authorization,
ID reuse, no-op, shared-file, staging-crash and cleanup-restart tests. No Chatbook
UI changes, facets, templates, digests, or legacy Collections migration are included.
