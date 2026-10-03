# ADR-060: MCP Durable Provider Accounting

**Status:** Proposed
**Date:** 2026-10-02
**Backfilled from:** not backfilled
**Decision owner:** MCP adapter workstream requester and reviewers
**Related task:** TASK-2294.3.2
**Related spec/plan:** `Docs/superpowers/specs/2026-07-23-mcp-skills-model-only-runner-design.md`; `Docs/superpowers/plans/2026-09-07-mcp-bounded-model-completion-adapter-implementation-plan.md`

## Decision

Bounded MCP completions reserve worst-case token and integer cost exposure durably before dispatch. Admission and settlement serialize on the explicit authenticated billing scope. Admission reads uncached canonical actual usage under that lock, adds all outstanding reservations, and fails closed on unavailable accounting state.

## Context

Legacy provider usage logging is post-call and best-effort. Concurrent requests can exceed quotas before their actual usage is recorded. A stale usage snapshot can also miss a reservation that reconciles immediately before admission. Cancellation after dispatch cannot prove that a provider did not execute the request, and replacing a valid completion with an accounting error can induce a duplicate paid call.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Post-call logging only | Does not bound concurrent unrecorded exposure. |
| Process-local reservation only | Loses conservative accounting on restart. |
| Read cached usage before acquiring the admission lock | Can undercount concurrent settlement. |
| Release on every cancellation or timeout | Cannot prove no billable dispatch occurred. |
| Return a retriable error after a valid completion if settlement fails | Can cause a duplicate paid request. |

## Consequences

Reservations move monotonically from reserved to released or dispatched, then from dispatched to reconciled or ambiguous. All unresolved rows remain chargeable, including across billing-period boundaries. Settlement inserts canonical `llm_usage_log` usage and transitions the reservation in one transaction. Ambiguous rows cannot accept late settlement. Valid completions survive post-call accounting failure while the worst-case reservation remains conservative.

Resource Governor release and bounded actual reconciliation reduce the associated daily-ledger operation rather than treating omitted actuals as fully consumed. Existing global governor policy remains governed by ADR-018 and ADR-056; the separate MCP reservation is the fail-closed authority.

Only the handle that inserted a durable daily-ledger row owns its downward settlement. Same-operation governor admission serializes through handle publication; durable replay after cache expiry or in another governor cannot acquire refund ownership over an earlier charge. Operation locks are removed after their final owner or waiter exits.

Completion concurrency uses the governor's existing `jobs` lease category. DB-backed Billing limit reads use the admission transaction connection, never a nested pool acquisition. Pre-dispatch cleanup attempts governor release even if durable storage is unavailable. Local release/dispatch markers make those boundaries mutually exclusive, and the durable state fence prevents refunds after dispatch begins. Failed release cannot authorize later dispatch on a refunded governor handle.

Hosted subscription limits apply only when the usage-quota master switch is enabled and a Billing repository is wired, following the host's Billing activation contract. OSS or inactive hosted Billing does not impose implicit free-plan limits. Explicit MCP operator bounds and durable usage/reservation recording remain mandatory for this certified execution path. Activation failures still fail closed rather than impersonating an inactive deployment.

The per-scope lock coordinates MCP admission and settlement, not legacy Chat calls that do not adopt this protocol. Operator pricing is captured by value; stored accounting excludes prompt, output, credentials, arbitrary metadata, and raw provider usage.

The canonical usage row determines which completed calls and billing period are counted. Its legacy floating-point USD columns remain compatible, but strict MCP cost enforcement reads the exact integer actual cost from the reservation audit row committed in the same transaction. A missing or inconsistent settlement fails closed instead of reconstructing MCP costs from lossy floats.

Canonical MCP timestamps are bound as UTC values independently of the database session timezone. PostgreSQL's exact numeric token aggregates are accepted only when finite, integral, non-negative, and within the signed-integer accounting bound; malformed values are never treated as zero exposure.

## Follow-Up

Stages 4 and 5 compose dispatch ownership, certified transport, and the host factory. Legacy provider callers may adopt the durable reservation protocol separately.
