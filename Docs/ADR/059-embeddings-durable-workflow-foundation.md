# ADR-059: Dormant Jobs-Rooted Embeddings Durability

**Status:** Proposed
**Date:** 2026-10-02
**Revised:** 2026-10-03
**Backfilled from:** not backfilled
**Decision owner:** repository requester; consolidated record awaiting written review
**Related task:** TASK-13421
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

Obtain approval of the consolidated written spec and this Proposed record before
implementation planning. Deliver separate reviewable Jobs capability, repository,
coordinator/adapter, and certification units. Real adapters and production migration
remain later workstream stages; acceptance of this record is not activation approval.
This proposal was renumbered from ADR-058 after that number was accepted for Jobs
completion row identity on `dev`. Recheck the next free number before integration;
do not replace or rewrite that accepted record.
