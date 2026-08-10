# ADR-032: Durable server-origin Sync mutation batches

**Status:** Accepted
**Date:** 2026-08-08
**Decision owner:** TASK-13003 requester and implementation review
**Related task:** TASK-13003
**Related ADR:** `Docs/ADR/031-notes-capability-sync-domains.md`
**Related spec:** `Docs/superpowers/specs/2026-08-08-notes-organization-sync-design.md`

## Decision

Compound server-origin mutations that affect more than one Sync v2 object must
persist their complete, ordered envelope plan atomically in the Sync store before
materializing any step into a product database.

The Sync store will expose a batch insert operation with these properties:

- every envelope belongs to one dataset and one server-origin mutation group;
- each envelope has a stable group identifier, zero-based step number, total step
  count, and group-plan hash;
- the store validates domain enrollment, envelope contracts, group completeness,
  step uniqueness, and idempotency for the entire plan before inserting anything;
- all new envelopes are appended in one Sync-database transaction and receive
  ordered server cursors; and
- replaying the same group returns the existing identical plan, while reusing a
  group identifier for a different plan is an idempotency conflict.

After the append commits, the server materializes the group in step order. Applied
steps remain applied. A failed or conflicting step stops later steps, records its
normal per-envelope apply status, and leaves the remaining steps pending. A retry
or repair resumes at the first non-applied step after verifying the persisted group
shape and statuses. Retryable failures may resume automatically; an unresolved
conflict blocks the group until conflict resolution supplies an accepted outcome.
The initiating REST call reports success only when every step is applied.

### Concurrency amendment (2026-08-09)

Preflight is advisory; the append transaction is the authority. After acquiring
the dataset write lock, the Sync store compares every accepted envelope's
external base cursor/revision/hash with that object's durable current head. It
also advances an in-memory head through later steps for the same object in one
group. If any comparison is stale, the transaction appends none of the plan.
The same compare-and-append rule applies to client pushes, where a lost race is
returned as a normal reviewable conflict instead of a raw storage error.

Product projection is serialized with one durable lock row per dataset. This
coarse scope is intentional: Notes organization mutations can affect descendants,
merge targets, and relationship endpoints that are not named by the envelope's
primary object identifier, and client-origin projection must share the same
ordering boundary. Before projecting a mutation group or singleton, the service
refreshes that dataset lock row's timestamp, acquires it in one Sync transaction,
and reloads the canonical envelope statuses. It retains the lock through the
whole group while product mutations run and while object-state and envelope-status
bookkeeping use that same Sync transaction. The bounded one-row-per-dataset shape
also avoids an ever-growing lock table.

While holding the dataset lock, projection checks for every earlier accepted
envelope that is not applied, excluding only the current retried unit. Pending,
failed, and conflicting predecessors all prevent later projection. A pending or
failed predecessor returns a stable retryable projection-pending result; a
conflicting predecessor remains a review blocker. Thus append order is also
projection order, rather than merely mutual exclusion order. PostgreSQL uses a
row lock with a bounded local lock timeout; timeout/deadlock failures map to a
stable retryable projection result. SQLite's immediate write transaction provides
the corresponding serialization and uses its bounded busy timeout. Pending work
counts as replayable health debt so contention cannot be reported as a healthy
accepted projection. No correctness property relies on a process-local lock.

Trusted bootstrap capture is the narrow exception to predecessor readiness:
while holding the same dataset lock it verifies that the already-existing
product state matches each canonical step and records bookkeeping, but it does
not project product mutations. This permits interrupted bootstrap source drift
to capture and verify its correction before audit-reconciling the stale pending
capture; ordinary server-origin, repair, and client projection never bypass the
predecessor check.

An intra-group same-object base cursor of zero remains the immutable virtual-head
marker in the canonical plan and its fingerprint. Under the group lock, the
materialization view resolves that marker to the actual assigned cursor of the
preceding same-object group step. Repair acquires the same dataset guard once for
the complete group, reloads its statuses once, and retains the bound Sync
transaction across every remaining step.

Object-state advancement is monotonic by server cursor inside the held
projection transaction. A lower cursor, or divergent state at the same cursor,
is rejected; an identical same-cursor replay is idempotent. A crash releases the
database locks. The durable lock rows remain reusable, and the already-durable
canonical group resumes through idempotent materializers. Because the product
and Sync databases still do not share a transaction manager, a product commit
may precede a failed Sync commit; retry repairs that split state and must never
report the incomplete projection as success.

### Conflict-resolution and repair amendment (2026-08-09)

An accepted envelope whose product projection produced a conflict remains an
ordering blocker until that exact conflict is resolved. While such a conflict is
unresolved, ordinary client and server-origin appends fail before adding more
accepted history; clients receive the stable reviewable materialization-conflict
result for the existing blocker rather than a raw storage error. The rejected
attempt creates no envelope or additional conflict row. Exact idempotent replay
remains permitted. This prevents an unbounded queue from accumulating behind a
review decision. The same check runs under dataset append authority before an
adapter-preflight conflict is stored. If evaluation raced an accepted projection
conflict, the existing durable blocker wins and the preflight attempt creates
neither a conflict-status envelope nor another conflict record.

Conflict resolution does not weaken the general predecessor rule. Instead, it
acquires the dataset row as the append authority and then the dataset projection
sentinel, in that deterministic order, and retains both through one bound Sync
transaction. Overwrite or rename is appended under an exact opaque claim of the
conflict identifier and canonical source cursor. Readiness is evaluated at the
logical position of that source: every other accepted non-terminal cursor earlier
than the source must be resolved, but already-queued legacy cursors later than the
source do not block its replacement. Skip uses the same source-position boundary.
Those later rows cannot safely retain their old bases after the logical history
changes. In the same transaction that terminalizes the source, every later
accepted non-terminal row is converted, in cursor order, to the reviewable
`sync_rebase_required_after_conflict_resolution` conflict. Each conversion inserts
an idempotent conflict record and conditionally repoints or removes a current head
that still names the queued row. The scan is capped by the existing mutation-group
action limit; exceeding the cap aborts the whole resolution without terminalizing
the source. The earliest rebase conflict then blocks the dataset until a user
resolves or resubmits it. The claim, append, product projection, source
terminalization, legacy rebase conversion, current-head maintenance, and
conflict-record resolution therefore cannot interleave with another append or
projection for the dataset.

Conflict terminalization uses the explicit envelope apply status `superseded`.
It means the canonical envelope remains immutable audit history, but its payload
was never projected and must not remain the object's append base. If the current
head still names that source, terminalization conditionally repoints it to the
latest earlier accepted-and-applied envelope for the exact dataset, domain, and
object identity, or deletes the head when no projected predecessor exists. A
duplicate/rename resolution advances only its own target identity; it never makes
that target the source identity's head. Predecessor ordering, replay, health, and
mutation-group status treat `applied` and `superseded` as terminal; repair skips
superseded singleton and all-terminal groups without creating repair debt. Pull
excludes conflicting and superseded envelopes. While an unresolved accepted
materialization conflict exists, pull also excludes every later cursor so a
legacy queued dependency cannot reach a client before the source decision. Source
resolution atomically converts those rows to rebase conflicts, after which the
applied replacement remains visible even though its physical cursor is later.
Pagination advances across filtered cursors. This avoids the false assertion that
an unprojected payload was applied while preserving immutable cursor history and
a usable projected append base for later captures.

An accepted-conflict overwrite is a logical replacement, not a child of the
unprojected source. Its stored and client-visible base cursor, revision, and hash
must name the latest earlier accepted-and-applied envelope for the same strict
dataset/domain/object identity, or the empty projected base when no such envelope
exists. The conflict identifier and source cursor are separate opaque resolution
metadata and are validated exactly while append authority is held. Materializers
therefore consume the canonical stored resolution directly; no server-only virtual
base exists. A fresh client can pull the projected predecessor plus resolution and
apply them in order without receiving the superseded payload. After the resolution
applies, current head and object state both converge on the resolution cursor.
Ordinary appends continue to compare against the maintained current-head
projection; no generic append path is permitted to bypass that CAS.

Adapter evaluation for an accepted overwrite also occurs after the conflict claim
inside that same dataset guard. Its context substitutes the latest applied
predecessor only for the exact claimed source identity while all other head and
dependency reads use the bound canonical head projection. This lets the production
Notes and Notes-organization adapters validate the client-visible projected base
without making the rejected source look applied or weakening dependency checks.

The projected context is an immutable applied-head snapshot, not a mixture of one
substituted identity and live current heads. For an original accepted
materialization conflict, the snapshot contains the latest accepted-and-applied
head at or before the source cursor for every dataset/domain/object identity;
`get_head` and `list_heads` read that same snapshot, so later queued resources,
relationships, tombstones, and conflicting names cannot affect evaluation at the
source's logical position. A generated
`sync_rebase_required_after_conflict_resolution` conflict has different lineage:
both its conflict type and the source envelope's apply error code must carry that
stable marker, and overwrite/rename evaluates and appends against a guarded
snapshot of the current latest applied heads. That current snapshot includes an
applied replacement appended while resolving the original conflict, even when its
physical cursor follows the queued rebase source. A marker mismatch, owner or
dataset mismatch, or source domain/object mismatch fails closed. The append CAS
uses the same original-source versus rebase-current distinction as adapter
evaluation, preventing stored bases from diverging from the context that was
accepted.

Before any resolution product projection, the dataset guard stages the complete
bounded set of later accepted nonterminal rows and validates its cap, envelope
identities, and existing conflict-record compatibility. The staged bookkeeping
is then applied in the same bound Sync transaction after projection; no
deterministic rebase-plan error is first discovered after a product write. When
resolving one generated rebase conflict, any later exact unresolved and unclaimed
rebase-required record remains the original resolution's durable review item:
its conflict ID, source-conflict provenance, and audit metadata are preserved
unchanged rather than reparented. Claimed, resolved, or identity-incompatible
records fail closed.

Restore preview selects the latest non-superseded projected envelope for each
identity, consistent with the maintained current head. A superseded source cannot
hide its applied predecessor, and mutation-group expansion never emits a
superseded action or sibling. Surviving members of a terminally split group are
restored as primitive actions without the now-incomplete group metadata. Skip and
duplicate resolutions therefore preserve the restorable projected state without
exposing the rejected payload.

Materialization conflict records are bookkeeping for canonical history, not a
best-effort side channel. Group projection, client push, and repair insert them
in the same bound Sync transaction that records the envelope apply status. A
partial unique identity on `(dataset_id, local_envelope_id, server_sequence)` for
rows with both envelope identifiers makes crash retry idempotent on SQLite and
PostgreSQL. A retry returns the matching durable record; it never creates a
second review item for the same accepted envelope.

Before installing that unique conflict identity, schema upgrade groups legacy
rows by dataset, local envelope, and server cursor in one transaction. It keeps
the deterministic earliest row only when every durable conflict, resolution, and
outcome field matches; differing creation timestamps alone do not make otherwise
identical retry rows incompatible. Any resolution-divergent or otherwise
incompatible duplicate aborts migration with a stable storage error before rows
are deleted or the index is created. SQLite and PostgreSQL use the same comparison
and deletion contract. If the unique index already exists, startup skips the
legacy duplicate scan entirely.

Accepted server-origin capture reports success only after reloading an `applied`
envelope. Idempotent retry actively rematerializes legacy `pending` or `failed`
captures under the dataset guard; `conflict` remains review-blocked and
`superseded` remains terminal without being success. Dataset health counts every
accepted pending envelope as repair debt as well as failed envelopes.

Trusted bootstrap identifiers require a source-step verifier. Verification and
stale-source reconciliation acquire the same dataset guard and bind their Sync
bookkeeping to its transaction; a bootstrap identifier without a verifier fails
closed before append or product projection. Both backends maintain a partial
outstanding-work index on accepted envelopes whose apply status is neither
`applied` nor `superseded`, so ordered readiness and health scans remain bounded
to unresolved work.

This is deliberately not a distributed transaction between the Sync database and
ChaChaNotes. A projection failure can leave an applied prefix in ChaChaNotes, but
never a partial canonical plan in Sync. The canonical plan is durable and complete,
materializers are idempotent, and repair can converge the projection without
inventing or reconstructing missing intent.

Mutation-group metadata contains identifiers, ordering, operation names, and a
hash of the canonical plan. It must not contain plaintext note content, credentials,
or secret-bearing request values.

## Context

ADR-031 established Sync v2 as the ownership boundary for mutable Notes state.
TASK-13003 adds independently mutable keyword, collection, folder, and membership
objects. Ordinary operations may span several of those objects: creating a folder
path can create multiple ancestors, replacing a note's organization state can add
and remove several links, and merging keywords can move links before tombstoning
the source keyword.

At the reviewed `dev` revision `a495e252c1319a6e44c20a259e92fa94c0107627`,
`SyncV2Store` and `SyncDatabase` support only single-envelope insertion.
`capture_server_origin_mutation()` appends and materializes one envelope at a time.
Using that seam repeatedly could commit only a prefix of a compound canonical
mutation. Mutating ChaChaNotes first and capturing afterward has the opposite
failure: the product database can commit while the canonical Sync history is
missing.

The two databases do not share a transaction manager, and introducing one would
couple every backend and deployment to a distributed-transaction protocol. The
system already records per-envelope projection failures and supports idempotent
replay, so a durable ordered plan is the smallest boundary that preserves intent
and permits deterministic recovery.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Mutate ChaChaNotes transactionally, then capture envelopes | A crash after the product commit loses canonical Sync history and causes permanent device divergence. |
| Append and materialize each envelope before creating the next | A crash can leave both the Sync log and projection with an unrecoverable prefix because the missing suffix was never persisted. |
| Wrap the Sync and ChaChaNotes databases in one distributed transaction | It adds backend and operational complexity disproportionate to the feature and still needs recovery machinery for external failures. |
| Add a `merge` or other compound wire operation | Every client and restore path would need bespoke semantics. Primitive resource and membership envelopes are portable and replayable. |
| Store only a group marker and reconstruct steps during repair | Reconstruction depends on mutable database state and can produce a plan different from the user's accepted request. |
| Use process-local mutexes | They do not coordinate multiple workers or server processes and disappear on crash. |
| Use expiring leases without fencing | A slow owner can outlive the lease and race a replacement owner during product projection. |
| Lock only named objects in a compound group | Indirect footprints such as descendants, merge targets, and relationship endpoints can interleave even when named object sets do not overlap. Durable per-object rows also grow without a useful bound. |
| Serialize only server-origin groups by dataset | Client pushes and repair can project through the same product seams, so a server-only lock would leave the ordering hole open. |
| Add dependency-footprint discovery and sorted fine-grained locks | It preserves more PostgreSQL concurrency but duplicates materializer knowledge at the lock boundary. A one-row dataset lock is the smallest safe contract until measured throughput justifies that complexity. |

## Consequences

Clients and repair tools can observe the full canonical group after one commit and
can replay its primitive envelopes in server-cursor order. REST callers may receive
a retryable projection error after the canonical plan is durable; retrying with the
same idempotency identity resumes rather than duplicates it.

The Sync envelope schema and both storage backends gain nullable mutation-group
fields plus an index that enforces one step per dataset/group. Single-envelope and
client-origin flows remain valid with those fields unset. Both storage backends
also gain reusable materialization-lock rows. Batch preflight, authoritative
append-time head comparison, dataset-wide projection locking, and ordered resumption
become shared Sync infrastructure rather than Notes endpoint logic.

Materializers must remain idempotent, and the repair path must understand group
ordering. Operators must distinguish canonical append success from projection
completion in diagnostics. No API or documentation may describe compound product
projection as atomic across databases.
