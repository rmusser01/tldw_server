# ADR-060: Dormant Jobs-Rooted Embeddings Durability

**Status:** Accepted
**Date:** 2026-10-02
**Revised:** 2026-10-04
**Backfilled from:** not backfilled
**Decision owner:** repository requester; written approval on 2026-10-04
**Related task:** TASK-13450 (replaces this branch's colliding TASK-13421)
**Related spec/plan:** [Stage 3 design](../superpowers/specs/2026-10-02-embeddings-workflow-stage3-durable-store-jobs-runner-design.md)

## Decision

Build a dormant Embeddings durability foundation with Jobs-owned root lifecycle
and exact operation proofs, Embeddings-owned reference/generation/checkpoint state
on the configured Jobs database, and separate lease-fenced execution and
receipt-only recovery authority.

## Context

Merged Stage 2 extracted ordered preparation, provider attempts, fallback
coordination, immutable outcomes, and the canonical inline runner, leaving the
orchestrator as a compatibility facade. Stage 3 needs durable resume and recovery
without prematurely migrating the API, media Redis pipeline, or vector-store work.

ADR-003 governs using Jobs for operationally visible work. ADR-022 remains governing
for the existing Embeddings API and Jobs-root/Redis-stage media pipeline. This
foundation neither activates a replacement pipeline nor supersedes that record.

Existing prepared Jobs operations and DTOs enforce Admin Webhook-specific
identities; they are not a domain-neutral workflow capability. Cancellation clears
processing leases immediately. A latest root result marker is not a complete
operation journal. Ordinary vector-store upsert does not certify first-write
immutability or prevent late writes after superseded-output cleanup. The foundation
must establish stronger contracts before real adapters or entrypoints are migrated.
Generic Jobs admission also defaults to consuming failure retries on lease expiry;
retry-neutral infrastructure recovery needs explicit, replay-validated controls.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Independent Embeddings database and lifecycle authority | Adds operational ownership and prevents atomic Jobs/domain fence validation. |
| Jobs fan-out for every item/stage | Increases durable state and complicates retry, cancellation, and accounting authority. |
| Remove Admin Webhook identity restrictions and reuse its DTOs directly | Weakens an existing closed contract and misrepresents AuthNZ origin as generic workflow authority. |
| Treat stable output IDs plus ordinary upsert as sufficient idempotence | Allows conflicting replacement and late recreation after cleanup. |
| Require a live worker lease for every recovery acknowledgement | Strands already dispatched work when cancellation or lease loss revokes execution. |
| Activate the runner together with the persistence foundation | Expands migration risk before backend, crash-window, security, and adapter certification exist. |

## Consequences

- Jobs retains the sole coarse lifecycle/quarantine authority. Its new prepared
  capability must preserve exact effect acknowledgements across later transitions
  and archival without importing Embeddings schemas.
- New Embeddings roots explicitly pin `ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT` and
  validate stored execution controls on admission replay, binding, and resume.
  A separate durable limit of at most five consecutive no-progress acquisitions
  bounds yield/expiry cycling, including expiry before the handler starts. Only
  new, exact semantic-progress proofs reset it, not deferral acknowledgements.
  Exhaustion goes through Jobs failure/quarantine authority without inventing a
  provider attempt or retry; legacy Jobs defaults and Admin Webhook restrictions
  remain unchanged.
- Six owner-scoped Embeddings tables live on the configured Jobs backend behind
  DB_Management repository/migration abstractions. PostgreSQL requires FORCE RLS
  and a non-bypass runtime role; SQLite requires exact owner predicates and
  composite ownership constraints.
- Provider invocation remains at least once. Generations pin an embedding space;
  fallback to another space restarts the manifest rather than mixing vectors.
- Normal execution requires an exact live lease and atomic checkpoint fencing.
  Receipt-only recovery can settle existing effects after revocation but cannot
  call providers, reopen generations, or revive terminal Jobs roots.
- Sink certification requires atomic first-accepted writes, immutable receipts,
  writer-authoritative absence, expiring permits, generation closure, and scoped
  cleanup. Lagging lookup returns unavailable, never absence that authorizes new
  paid work. No production vector-store helper is automatically certified here.
- Accounting certification requires atomic per-attempt ordering and absorbing
  terminal records. Late reservations cannot recreate holds after release or
  settlement; proof retention covers all previously issued delivery authority.
- Explicit limits and deadlines bound manifests, calls, attempts, operations, and
  retained recovery state. Unsettled obligations consume capacity after Jobs expiry;
  deleting history does not implicitly delete completed sink outputs.
- Resolver and sink ports operate on bounded batches under a certified progress
  budget. Serial per-item deadline compliance alone does not certify a 64-item
  acquisition; the complete normal call plan must fit the 300-second quantum.
- Imports/startup stay inert. Production adapters, billing hookup, entrypoint
  migration, and worker/lifecycle registration remain separately approved work.
- ADR-051's existing retry helpers remain authoritative; the foundation does not
  introduce a bespoke retry schedule or duplicate durable DLQ.

## Follow-up

The requester approved the consolidated written spec and this record on 2026-10-04.
Deliver separate reviewable Jobs capability, repository,
coordinator/adapter, and certification units. Real adapters and production migration
remain later workstream stages; acceptance of this record is not activation approval.
Repository planning must select and test an atomic global capacity reservation
primitive compatible with owner-scoped FORCE RLS/non-bypass runtime roles. Runner
planning must map every private durable phase/terminal outcome to the existing
closed trace contracts without expanding inline literals or inferring completion
before Jobs acknowledgement.
This proposal was renumbered from ADR-058 and then ADR-059 as `dev` accepted those
numbers for Jobs completion row identity and the backlog-py task editor cutover.
ADR-060 is the next free number as of this revision. Recheck numbering before
integration; do not replace or rewrite accepted records.
