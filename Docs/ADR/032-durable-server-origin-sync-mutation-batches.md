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

## Consequences

Clients and repair tools can observe the full canonical group after one commit and
can replay its primitive envelopes in server-cursor order. REST callers may receive
a retryable projection error after the canonical plan is durable; retrying with the
same idempotency identity resumes rather than duplicates it.

The Sync envelope schema and both storage backends gain nullable mutation-group
fields plus an index that enforces one step per dataset/group. Single-envelope and
client-origin flows remain valid with those fields unset. Batch preflight and
ordered resumption become shared Sync infrastructure rather than Notes endpoint
logic.

Materializers must remain idempotent, and the repair path must understand group
ordering. Operators must distinguish canonical append success from projection
completion in diagnostics. No API or documentation may describe compound product
projection as atomic across databases.
