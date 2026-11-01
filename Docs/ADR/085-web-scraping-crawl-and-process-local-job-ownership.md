# ADR-085: Web Scraping Crawl And Process-Local Job Ownership

**Status:** Accepted
**Date:** 2026-10-04
**Last amended:** 2026-11-01
**Backfilled from:** not backfilled
**Decision owner:** Human requester in the Web Refactoring chat; written design approved 2026-11-01
**Related task:** TASK-15282
**Related spec/plan:** [Phase 5 design](../superpowers/specs/2026-10-04-web-scraping-phase-5-crawl-process-local-jobs-design.md); [implementation plan](../superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md) awaiting review

## Decision

Separate crawl and the existing process-local queue into state-owning components
with injected execution capabilities, composed by canonical article orchestration,
while retaining legacy compatibility entry points and deferring durable Jobs migration.

This is the approved ownership rule for Phase 5, not a claim that the split is implemented.

## Context

The earlier web-scraping phases established governed per-page execution, pre-scrape
analysis, runtime policy boundaries, and extraction worker lifetime management.
Traversal and queue state still live in large legacy/enhanced modules. Mechanical
relocation would preserve unclear cancellation, retry, completion, and continuation
ownership. Replacing everything at once would risk existing behavior and combine
the refactor with a separate persistence/backend migration.

The user chose the current queue now and durable Jobs later, with no functionality
loss. In particular, analyzer configuration/advice precedence, cookie/dedup behavior,
legacy exports, and per-entry-point traversal semantics must survive.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Relocate functions without changing ownership | Leaves the execution/state coupling and lifecycle races intact. |
| Have crawl/queue call legacy wrappers or application services | Introduces reverse dependencies and competing execution paths; canonical orchestration must compose these components. |
| Immediately replace the existing queue with central Jobs | Adds durable backend, storage, authorization, and deployment migration to a compatibility-sensitive split; explicitly deferred by the user. |
| Treat all progress JSON as resumable durable state | A snapshot lacks frontier/execution state, and a process-local queue plus periodic checkpoints cannot promise crash-safe exactly-once work. |

## Consequences

- `Web_Scraping/crawl/` owns discovery/frontiers and supported checkpoint state;
  `Web_Scraping/jobs/` owns process-local job state, retries, and completion.
  Neither imports orchestration, services, or legacy wrappers for page execution.
- Orchestration supplies execution capabilities and a live crawl-owned context;
  services retain trusted authorization, lifecycle, summaries, ingestion, and
  persistence. Existing public models and wrappers remain compatible adapters.
- Waiter cancellation, explicit job cancellation, crawl cancellation, and shared
  shutdown are distinct operations. Logical terminal state is distinct from
  physically draining thread/executor resources.
- Actual work is bounded independently of successful-page counts. Parent stop
  outcomes cannot be turned into local analyzer fail-open, fallback, or queue retry.
- Shared queue admission also has finite count and payload-byte limits across all
  callers/priorities. Overload rejects before accepting a job, retries reuse an
  admission lease, and draining work keeps its capacity until physical completion.
  Completed history has both byte and count ceilings independent of subscriber delivery.
- Batch producers submit through bounded windows and retain ownership of accepted
  children through cleanup. Queue capacity is not an artificial batch-size limit;
  overload handling cannot abandon earlier admitted siblings.
- Crawl outcomes survive service, ephemeral, persistent, and ingestion conversion.
  Storage success is not crawl completion; partial results remain available with
  their interruption reason. Existing structured failure lists remain compatible
  before and during traversal; a typed failure is used only where no established
  failure representation exists. Successful public shapes remain compatible.
- Terminal crawl outcomes are published once before service post-processing.
  Later service cancellation/failure is reported separately, not by rewriting
  the completed or partial crawl outcome.
- Public projections exclude execution credentials. Continuation state is versioned,
  bound to trusted scope, validated, and separate from progress observations.
- Legacy and incomplete crash history is disclosed; the phase does not claim
  durable queued jobs, exactly-once fetching, or crash-proof lifetime accounting.
- This is not an exception allowing new features to invent ad hoc queues.
  [ADR-003](003-jobs-vs-scheduler-default.md) remains the default for new work.
  [ADR-021](021-services-lifecycle-startup-and-shutdown.md),
  [ADR-026](026-security-outbound-egress-and-ssrf-policy.md), and
  [ADR-042](042-browser-transport-admission-and-attestation.md) remain governing.

## Follow-Up

- Written-spec/ADR approval was provided on 2026-11-01.
- Review the detailed staged implementation plan and its safety defaults before code.
- Reassess durable central Jobs migration as a separate later work item, including
  crash recovery, quotas/RLS, operational visibility, and compatibility evidence.

## Identifier Correction (2026-11-01)

The unpublished draft originally used ADR-063 and TASK-13452. Those identifiers
subsequently landed on `dev` for unrelated work. This document and its design task
were renumbered to ADR-085 and TASK-15282 after checking `dev` and all open PR file
pages. This is not a supersession or modification of the unrelated ADR-063.
