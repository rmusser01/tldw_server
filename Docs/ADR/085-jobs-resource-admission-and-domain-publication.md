# ADR-085: Resource Admission In Jobs, Publication Fences In Domain Storage

**Status:** Proposed
**Date:** 2026-11-01
**Backfilled from:** not backfilled
**Decision owner:** Requester for product policy; implementation proposal pending written-design approval
**Related task:** TASK-15282
**Related spec/plan:** Docs/superpowers/specs/2026-11-01-claims-jobs-stage2c-cluster-rebuild-design.md

## Decision

For opt-in exclusive resource work, shared Jobs atomically arbitrates active
requests and assigns immutable monotonic publication authority, while domain
storage atomically fences output and retains replay receipts without managing
Jobs lifecycle.

## Context

Claims cluster rebuilds replace an owner's clusters. The requester selected one
active rebuild per owner: equal settings reuse the active Job and conflicting
settings return 409, with synchronous 200 retained by default and asynchronous
202 enabled explicitly.

Existing Jobs idempotency and owner admission are queue/job-type scoped and do
not provide this full resource contract. Lease/UUID checks protect Jobs row
completion but cannot stop an already-running synchronous domain transaction.
Jobs and Media DB may be separate databases. A Claims job registry or local
mutex would duplicate lifecycle ownership and fail across workers.

This extends ADR-003's ownership boundary and composes with ADR-058's row-identity
requirements; it supersedes neither accepted decision. The API and schema
details remain proposals until the accompanying design is approved.

## Alternatives Considered

| Option | Why not selected |
| --- | --- |
| Permanent owner/settings idempotency key | Completed work can suppress fresh requests, while different settings can still race |
| Queue every request or keep a latest pending request | Adds redundant execution or replacement semantics the requester did not choose |
| Claims-owned active-job table or process mutex | Recreates job management and cannot enforce cross-process lifecycle transitions |
| Trust Jobs terminal state to fence domain writes | Cancellation or lease loss does not stop a thread or roll back another database |
| Assume one transaction spans Jobs and domain storage | Not valid for supported independent SQLite/PostgreSQL deployments |

## Consequences

Jobs owns queue-independent resource admission, reuse/conflict decisions,
request receipts, generation authority and all reactivation/terminal transitions.
Existing non-resource Jobs behavior remains opt-out and unchanged.

Domain storage owns output ordering and committed-output receipts, not queued,
running, retry or lease state. A replay returns its committed result; an older
generation cannot overwrite newer applied output. Cancellation remains best
effort and cannot undo committed output.

Cluster replacement must validate existing references, including claims whose
ownership changed before its source snapshot. A DB_Management-owned integrity
guard remains effective under restricted-role PostgreSQL/RLS; hidden rows do not
prove absence. Foreign references fail publication without changes to domain
data, the publication head or receipts, and without cross-owner writes.
Transfer repair is separate work. Best-effort Watchlists evaluation also uses
existing-only storage access; missing storage or schema does not authorize
maintenance to provision it.

Resource counters survive ordinary pruning. Publication receipts do not pin Jobs
history and are retained with the owner's domain data in this stage; unexpired
Jobs request receipts protect their own replay targets. Supported restoration
requires a quiescent coordinated checkpoint and an operator pre-start check.
UUID checks alone cannot detect every same-UUID counter rewind or lost receipt.
Unprovable independent restores stay disabled for separately reviewed recovery,
rather than resetting a domain fence. Additional schema and backend parity tests
are accepted costs of the chosen concurrency guarantee.

Read-only owner status lives in Jobs; granting Claims permission must not grant
queue administration. The design proposes an owner-safe projection rather than
exposing arbitrary job payloads or introducing Claims queue-control APIs.
Cross-owner admin status links also bind the logical UUID using an optional
admin-detail read precondition. Numeric ID reuse must not redirect such a link
to another Job; existing admin/domain authorization and legacy reads are retained.

## Follow-Up

- Review the accompanying full design and its builder check before acceptance.
- After explicit approval, plan shared Jobs prerequisites before Claims adoption.
- Keep Stage 3 compatibility cleanup and unrelated prior-PR closeout separate.
