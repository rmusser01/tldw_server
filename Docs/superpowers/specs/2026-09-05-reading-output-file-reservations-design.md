# Reading-safe generic output file mutations

Date: 2026-09-05 · Task: TASK-13153
Status: amended spec independently reviewed and user approved; implementation planning
Governing decision: `backlog/decisions/003-reading-atomic-hard-delete.md`.
Amends the generic-writer portion of `2026-09-04-reading-atomic-hard-delete-design.md`.

## Purpose and scope

Preserve unmanaged output rename/conversion while ensuring that neither those
operations nor generation/deletion can damage a managed Reading archive. Extend
the existing durable-reservation and POSIX storage-exclusion design; do not add
a public job API, new dependency, or managed archive conversion support.

Isolated SQLite/endpoint probes at `2207a84fc1` reproduced: ownership registered
after dispatch causes a 409 after the source has moved; an unowned shared-source
rename moves an owned archive; a destination collision overwrites owned bytes.
The prior 146 passing tests cover the managed-only update checkpoint, not these
file operations. This document is a design, not verification or release evidence.

## User-visible contract

- Managed title/retention changes remain metadata-only. Actual managed format or
  path changes reject the complete request with `reading_archive_file_immutable`.
- Unowned outputs retain title-based renaming and md/html conversion. Changing
  title and format together is one operation: no intermediate rename is exposed.
- A managed source alias cannot be moved, converted or overwritten through an
  unowned row. Reject without mutation rather than guessing ownership.
- An occupied destination is never overwritten, including an unowned destination.
  Return 409 `output_path_conflict`; the user can choose another title. This is an
  explicit safety correction to the former POSIX rename/write overwrite behavior.
- Different spelling that aliases the source (including case-only names) must
  never be treated as a distinct disposable file. A title-only case change can
  update display metadata while retaining the physical spelling. Ambiguous
  physical changes reject without mutation.
- Missing source bytes for a requested physical change return 409
  `output_source_unavailable`; do not point metadata at a nonexistent new file.
  Pure metadata operations do not require filesystem access.
- Busy reservations/storage return retryable 409 `output_file_busy`; invalid or
  unavailable activated storage returns 503 `output_storage_unavailable`.
  Authentication/missing-row behavior remains unchanged. Errors contain no paths,
  filenames, titles, file bytes, namespace IDs or raw exception strings.
- A committed update remains successful if old-file cleanup is pending. Existing
  response fields describe the committed output, not cleanup completion. Actual
  unlink counters never count queued work. An uncertain HTTP outcome requires a
  GET before retry; never automatically rerun with a newly fetched record.

## Storage activation and authority

Use an internal persisted per-user protocol binding: authenticated user ID,
contract version and the explicitly provisioned namespace of that user's
authoritative generic output root. The current generic API already has one
configured output directory per user; this is not a new multi-volume output API.
Reading ownership/cleanup records retain their individual namespace identities.
Recovery of older intents on another verified volume remains independent.

Activation is an explicit stopped-writer upgrade, not a request-time fallback:
verify the output root and existing unowned records' file provenance, provision
its marker/lock, persist the binding under the DB clock, and deploy all participating
writers before admitting traffic. Do not infer ownership from file names or JSON.
Unknown/conflicting legacy locations require operator resolution. First-time store
initialization may explicitly provision a new root; missing existing state is never
silently recreated. No schema migration guesses a volume or activates the protocol.

For an activated user, all generic file mutations require that exact binding and
validated mounted volume. Missing/mismatched marker, unsupported locking, unknown
binding version or an unhealthy recovery worker fails closed. Never select the
legacy branch merely because a local file/marker is absent. A node on a different
volume cannot mutate the authoritative root's generic outputs. It may still drain
Reading intents explicitly assigned to its own validated volume.

For users not activated, existing unmanaged behavior can remain during rollout,
but production Reading ownership/staging cannot be enabled there. An apparently
inactive user with existing ownership/reservations is inconsistent, not permission
for legacy file writes. Activation requires old processes and in-flight legacy
requests stopped; runtime activation races are not supported. The optimistic-delete
capability remains absent until the full writer/lifecycle contract is verified.

## Bounded durable operation record

Add an internal output-file operation journal separate from item-owned
`reading_artifact_paths`; generic operations must not invent a Reading parent.
One record contains a non-null random token, user/namespace, operation kind,
optional output ID, bounded original-row snapshot and intended metadata changes,
exact source/destination/private-stage names and their conservative collision keys,
recorded source identity/fingerprint and staging/publication identity, phase,
lease, retry time/count and sanitized error category. Do not store output
bodies or credentials. Nullable output IDs have no cascading FK: cleanup authority
must survive logical output deletion. Identity constraints explicitly use NOT NULL
on SQLite and PostgreSQL.

Kinds are limited to create, replace (rename/conversion), and remove. Each holds
at most three path reservations: source, random private staging path, destination.
This bounded shape can use columns in the journal rather than a general resource
lock table. Existing clock serialization makes cross-column collision checks
atomic. Reserve a row identity as well as its paths so it cannot be deleted,
retargeted or assigned ownership by another operation while work is pending.
All records and checks are user-scoped. Unknown generic volume references are
conservative; never discard another namespace's cleanup authority by guessing.

Phases are `prepared`, `committed`, and `aborting`. A separate monotonic
`fs_done` flag distinguishes completed filesystem work from pending external
history delivery. A prepared operation owns its
reservations, not permission to delete its source. The output mutation and the
transition to committed occur on one connection in the same DB transaction.
Aborting may remove only privately created artifacts. Lease expiry authorizes a
recheck, not an unlink. After required cleanup/fsync or explicit preservation by
a surviving reference, set `fs_done` under the clock and release the file/row
reservations. Retain the operation's bounded pending history effects until their
delivery finishes. Delete the record only when both filesystem and delivery work
are complete. Ambiguous file identity stays blocked with durable diagnostic state.

## Shared database boundary

Every phase transition and relevant writer uses the existing revision-clock
fence, with explicit connection propagation. This includes generic output create,
update, soft/hard delete, retention and quota-affecting output mutations; Reading
ownership registration/reconciliation; archive reserve/adopt; and item deletion
that examines owned outputs. A request carrying the exact internal operation token
may perform only its recorded transition. Other callers cannot bypass reservations.
Never expose this token through public request fields or treat a caller-supplied
matching path as authorization. Metadata edits to a row reserved by another
operation reject too; they cannot be lost behind an older captured snapshot.
All conflict/cleanup queries exclude `fs_done` records from file and row claims;
such records can deliver history only, never mutate or inspect recycled paths.

Check source, destination and staging aliases against both journals, structural
ownership and surviving output references before granting reservations. Pending
Reading cleanup blocks new output attachments and generic writes. Generic claims
also block Reading reserve/adopt/ownership registration. A shared unowned source
may be copied, but remains while any other output references it. A managed source
alias rejects the physical update. Other users' records are neither exposed nor
modified. Reserved row identities prevent SQLite ID reuse until `fs_done` releases
the claims, not until eventual history-delivery retirement.

Comparison uses existing confined filename normalization, conservative case
collision keys and legacy absolute-path alias checks. Preserve exact spelling for
I/O and cleanup. Reject symlinks, special files and unexplained hardlinks; do not
equate basename or lowercase comparison with proof of identical file identity.
Internal staging names and storage lock/marker names cannot be generic targets.

## Filesystem protocol

Lock order is always verified per-user OS storage lock, then short DB transaction.
All I/O uses the held directory descriptor. No filesystem/network work runs inside
a DB mutation transaction. Validate request fields/formats and compute final names
before changing metadata; legacy path normalization must not write a DB field as
a hidden preliminary step. Rendering/network waits never hold the storage lock;
revalidate the original output snapshot when reserving/committing. Prepared path
and row reservations remain active between bounded write chunks, even though
the OS lock is released. Each reacquisition rechecks the token, lease and file
identities before any I/O; a retired/aborting token cannot resume writing.

1. Acquire the bound storage lock; under the clock read the active same-user row,
   validate ownership, original fields and all aliases, and persist the prepared
   operation/reservations. The source row and file are still unchanged.
2. Recheck the prepared token and current lease after acquiring all waits. Create
   a random reserved private file exclusively, without following links. Copy or
   convert source bytes into it in bounded chunks, flush/fsync, and persist its
   verified file identity
   before publication. Source bytes remain untouched. Before any logical commit
   permitting source disposal, persist the source's device/inode, regular-file
   type, expected link count, size and modification/change timestamps, collected
   outside the DB transaction through the held directory descriptor. Recheck
   those values before commit and every eventual unlink. Changed identity or
   fingerprint leaves blocked authority; never delete replacement bytes. Removal
   records this evidence too, although it does not create a staging file.
   Capture the initial source fingerprint before copying; every resumed chunk
   and final commit must still match that original evidence.
3. Publish the destination without replacement. Use descriptor-relative atomic
   no-clobber linking from the private file, retaining that private link as an
   identity witness until commit/recovery is resolved. Fsync the directory before
   DB commit. Platforms/filesystems without this primitive fail readiness; do not
   emulate it with exists-then-rename. Unexpected links/identities remain blocked.
4. Under the clock revalidate the operation, row snapshot, paths and ownership;
   atomically commit the recorded output changes and committed phase. Generation
   commits its output insert here; removal commits the output delete here. Only
   logically committed effects affect counters/history, with no double accounting
   on recovery. Same-store accounting joins this transaction; existing idempotent
   external history updates may be replayed, with durable pending work until
   acknowledged. Do not promise cross-database exactly-once delivery or apply
   additive external effects without an idempotency key.
   A definitely rejected/stale transaction may conditionally change prepared to
   aborting. An exception is not proof of rollback: reread the durable token/phase
   on a fresh connection after uncertain commit acknowledgement. Committed always
   wins and must never transition to aborting. If state cannot be established,
   preserve all files/reservations and return sanitized 503
   `output_update_unconfirmed`; the caller must GET before retrying.
5. While still holding storage exclusion, perform phase-specific cleanup. Aborting
   first unlinks only a destination proven to share the recorded private witness's
   inode, fsyncs the directory, then removes/fsyncs the private witness. Never remove
   that witness while an uncommitted publication still needs it. Committed cleanup
   preserves the published output and may remove the private witness; durable
   committed phase and recorded publication identity prove ownership on restart.
   After committed replacement/removal, clean unreferenced old sources only after
   verifying the recorded source identity/fingerprint. Check surviving references under
   the clock while reservations still prevent new attachments; unlink/fsync outside
   the transaction. Set `fs_done` and release reservations in a final transaction
   after cleanup. Delete the journal row only if external delivery is also complete.

For a removal there is no staging/destination; its source reservation remains
through post-commit cleanup. Metadata-only deletion explicitly preserving files
does not acquire disposal permission from this protocol. Managed removals continue
using their existing item-owned durable intents and explicit file permission.

Never overwrite an occupied destination, even on retry. For an uncommitted
publication, recovery requires the recorded identity and private-link witness;
for a committed publication, the durable committed phase and recorded identity
remain authoritative after witness cleanup. A matching filename or body is not
enough. Private stage creation/identity-recording
is also a crash boundary: if an existing stage cannot be proven to belong to the
operation, retain the record/files blocked for explicit operator verification.
Do not clear such records by age. The stopped-writer recovery procedure must list
blocked token/phase categories and require verified file identity before any
operator-authorized removal; no automatic deletion of ambiguous artifacts.

## Reader lookup and file opening

For activated stores, GET/download-by-ID, download-by-name, HEAD and any other
file reader acquire the same bound storage lock before their authoritative DB
lookup. Re-fetch the same-user current row after any wait, normalize without a
metadata write, and open the file descriptor relative to the verified directory
before releasing storage exclusion. Validate regular-file identity via `fstat`;
never return a response that will later reopen a pathname. All response bytes,
length, validators and range offsets must come from that same opened file.
During this protected lookup, compare structural ownership's namespace with the
verified reader volume before opening. Reading-owned outputs may retain another
namespace; a missing, ambiguous or mismatched ownership namespace fails closed
with sanitized storage-unavailable behavior, never a filename lookup in the
generic root or an opportunistic search across volumes. Unowned outputs require
the established generic binding and reconciled provenance described above.

Readers do not require a new durable reader journal or hold the OS lock for the
whole download. Stream the descriptor in bounded chunks after releasing the lock,
closing it on completion, error, cancellation or disconnect. An already-authorized
download may finish from its opened inode after a later rename/delete; requests
linearize at the protected lookup/open, not at response completion. A later
request rechecks the row and cannot read a recycled filename through an old lookup.
Preserve existing HEAD, conditional and range behavior using the opened descriptor.

A committed publication may still have its private witness link. Accept its
expected link count only after checking the same-user committed journal and exact
recorded destination/witness identities under storage exclusion. Other unexplained
hardlinks or uncommitted publications are not downloadable. Worker-health checks
govern new file mutations, not read availability: a valid volume and safe descriptor
open suffice for reads even while history delivery or recovery is temporarily down.

## External history delivery without filesystem locks

Record the finite required effect list and pending/acknowledged state atomically
with logical output commit. Delivery uses a stable `(operation token, effect kind)`
idempotency key and the exact original target identity. It neither takes the storage
lock nor holds file/row reservations after `fs_done`. A history-store outage leaves
delivery pending but does not make the filesystem recovery worker unhealthy or
prevent ordinary edits/deletion of the completed output. Bound due-work batches
and backoff; expose delivery backlog separately from blocked filesystem cleanup.

Never replay solely by a recyclable numeric output ID. Capture immutable history
row identities/versions or an original-output incarnation that the receiver can
validate; apply changes conditionally so a later output/history row cannot be
mistaken for the old one. An unavailable or ambiguous target is blocked delivery,
not permission to guess and not a reason to reacquire old path reservations.
The implementation plan must map the existing history API to this identity check;
its current update-by-output-ID helper alone is insufficient for delayed replay.

If delivery succeeds but acknowledgement is lost, safely repeat the same effect.
Acknowledgement and final journal retirement are conditional DB transitions.
Filesystem recovery skips `fs_done` even when the retained record names a path or
row now reused by newer work. No public/general-purpose outbox service is added.

## Resource admission and bounded staging

Use the existing operation journal for durable temporary-byte reservations; do
not build a second quota platform. Activation requires finite positive configured
limits for per-operation staging bytes, total per-user temporary/pending bytes,
active operation count and text-conversion input/output size, plus a minimum free
space margin. Invalid/missing resource policy fails activation rather than meaning
unlimited. Existing audiobook storage quotas still constrain committed artifacts;
they do not substitute for these temporary-space limits. Deployment values are
operator settings, not hard-coded assumptions about audiobook sizes.

Reserve a worst-case byte budget under the clock before starting a producer or
creating a private file. Unknown-length generation must have an explicit maximum.
Enforce the budget as bytes arrive, before each write. Include existing staging,
blocked files and committed-but-not-yet-cleaned old sources in admission; publication
hardlinks count their single allocation once. Mark completed file work and adjust
byte reservations atomically. Concurrent operations cannot both spend the same
per-user budget. Check actual destination-volume free space outside the DB lock
with the configured safety margin before admission and periodically while writing.
Free-space checks are advisory against other processes: ENOSPC still follows safe
abort/recovery, never a source-destructive fallback. Insufficient admission returns
507 `output_storage_capacity`; bytes over the declared operation limit return
413 `output_size_limit`, with no logical output changes.
An unlinked inode may still consume disk while an earlier reader holds it open;
logical budget release is not proof of reclaimed disk. Actual free-space checks
must include this possibility without blocking downloads until cleanup finishes.

Copy/audio streaming uses at most 1 MiB buffers; yield storage exclusion after at
most 8 MiB of writes or 50 ms observed between chunks, whichever comes first.
These are cooperative fairness boundaries, not hard deadlines for a blocked OS
call or fsync. Execute blocking I/O off the event loop. Before releasing the lock,
finish all writes and close writable descriptors; no background write can outlive
that lock interval. On reacquisition recheck phase, current lease, recorded source
fingerprint, stage identity and expected offset/length before continuing. Refresh
the lease only under the clock after acquiring the lock. Recovery can abort an
expired prepared operation; a paused producer must then stop without recreating it.
Large text conversions use their explicit bounded input/output limit when the
existing converter cannot stream; do not load unlimited content into memory.

Superseding the previous external-staging suggestion: output generation streams
directly into the journal-reserved private staging file in the authoritative
directory, with network/rendering waits outside storage exclusion. This removes
the extra untracked scratch-file lifecycle rather than adding another journal.
No output-specific producer may create a separate named scratch artifact before
admission. Adapt TTS/other producers to bounded byte streams or bounded in-memory
results; any unavoidable producer-owned named scratch path is a rollout blocker
until its durable ownership/recovery is included and reviewed. Cancellation changes
prepared to aborting conditionally and schedules identity-checked cleanup; process
death leaves the existing lease/journal for recovery. Do not use age-only directory
sweeps. Uncertain pre-identity file creation remains explicitly blocked as above.

## Recovery and rollout coverage

Reuse the existing cleanup lifecycle for bounded per-user batches, independent
of the optional outputs retention flag. Filesystem batches select only records
without `fs_done`: acquire the exact namespace lock, read current phase after
waiting, recheck due time, and process idempotently. A separate delivery pass
handles `fs_done` records without taking a filesystem lock or inspecting paths:

| Durable phase | Recovery action |
| --- | --- |
| Prepared, live lease | Leave unchanged |
| Prepared, expired lease | Transition to aborting under fence; preserve source |
| Aborting | Verify witness, unlink/fsync publication first, then unlink/fsync private witness; retain source |
| Committed | Preserve published output; clean private link and unreferenced source |
| Filesystem done, history pending | Retry only the recorded history effects; no filesystem access or reservations |
| Missing/wrong volume, busy lock, uncertain identity | Retain reservations; retry or block with sanitized category |

A writer paused before acquiring the lock must revalidate the token when it
resumes. A killed writer releases the OS lock but retains durable authority. A
crash after unlink/fsync but before `fs_done` safely repeats cleanup. Aborting work
has no committed history effects and can retire when filesystem cleanup completes.
Never declare filesystem completion for an absent-file record without validating
the actual volume/lock; later delivery-only retirement needs no volume validation.
No public job/receipt platform is introduced. Existing operational cleanup status
must distinguish retryable work from blocked identity verification.

All file writers in an activated output directory must participate before release,
not just PATCH. Inventory generic generation and failed-generation cleanup in
`outputs.py`, Watchlist output/briefing generation and cleanup in `watchlists.py`,
`outputs_service.py` TTS/text persistence and deletion, scheduled/API purges,
`audiobook_jobs_worker.py` and `endpoints/audio/audiobooks.py` artifact writers,
and production Reading archive creation/reconciliation. Long-running TTS/rendering
must use the bounded reserved staging protocol above; request cancellation cannot
run file-first cleanup.
Read-only routes may not normalize persisted paths through an unguarded write.
Any other discovered writer is an explicit rollout blocker, not an exception.

## Required evidence

Targeted real SQLite and PostgreSQL tests, no Docker reprovisioning or full suite
without permission. Include migration twice; NOT NULL/foreign-user constraints;
all three observed failures; prepared ownership/attachment/delete/metadata races
in both commit orders; both generic and Reading journals' alias conflicts; managed
and shared-unowned sources; occupied destinations; compound/no-op updates; missing
sources; stale snapshots; atomic same-store quota accounting, idempotent external
history replay; and unsupported storage.

Inject failure/crash before/after reservation, exclusive create, identity recording,
publication, file/directory fsync, DB commit, each unlink and retirement. Include
real process lock release and delayed-writer-after-retirement tests, missing/replaced
root, same DB/different volumes, case/absolute aliases, symlinks and hardlinks.
Also inject commit success followed by lost acknowledgement and unavailable
phase reads; never abort a committed operation. Crash between publication unlink,
directory fsync and witness removal; resume without losing cleanup authority.
Replace or modify a source after identity recording and before cleanup, including
across restart: preserve replacement bytes and leave the operation blocked.
Every rejected operation preserves source bytes/rows; every committed operation
has either completed cleanup or durable pending/blocked authority. Verify response
and log privacy, full route dispatch and generated-output cancellation paths.

Pause readers after initial lookup but before lock/open, replace/delete and reuse
the old path, then resume: never return another output's bytes. Pause after safe
open and verify the original bytes remain readable through subsequent unlink.
Cover committed witness links, HEAD/ranges/conditional requests and disconnect FD
cleanup. Put different bytes at the same filename on two same-user volumes and
prove a Reading-owned row cannot read the generic root's wrong-volume file.
Simulate history-store outage after file cleanup, allow newer updates and
ID/path reuse, then deliver/replay only the original effect. Crash around `fs_done`,
effect acknowledgement and journal deletion without lost delivery or renewed claims.
Test concurrent byte admission, exhausted capacity, provider overrun, bounded-memory
large audiobook rename, lock yielding with another reader/writer, ENOSPC, cancellation
between chunks, and killed producers with no untracked external scratch artifacts.

Recheck inactive/active/unknown bindings and prove that activated stores cannot
fall back to legacy mutation when storage or recovery is unavailable. Keep
`hasReadingOptimisticDeletesV1` absent through partial checkpoints. Review each
implementation slice, run scoped formatting/lint/Bandit, and record actual evidence
in TASK-13153 before any eventual capability activation.

## Design review record

Independent read-only review, 2026-09-05: first pass found three planning gaps
(uncertain commit acknowledgement, abort witness ordering, source identity before
disposal). All three were amended with explicit recovery rules and failure tests.
Second whole-spec review approved with no remaining serious issues. This records
design review only; no new production code or runtime verification is claimed.
User review of the written spec is the next gate before implementation planning.

User-requested follow-up review found reader lookup/open races, history delivery
holding/releasing filesystem authority ambiguously, and missing staging resource
bounds. The user requested all three amendments. The additions above define
descriptor-based reading, `fs_done`-separated delivery, and bounded journal-backed
streaming without a separate external scratch lifecycle. Re-review found one
related namespace gap; protected reads now reject ownership/volume mismatch and
require a two-volume same-filename regression. The final bounded independent
review approved with no remaining serious spec issues. These are documentation
changes, not implemented or runtime-verified safeguards. The user subsequently
approved the written spec after checkpoint `8dc255fcca`. Implementation planning
is recorded in `Docs/superpowers/plans/2026-09-05-reading-output-file-reservations.md`.
