# ADR-087: MCP Durable Provider Accounting

**Status:** Proposed
**Date:** 2026-10-02
**Backfilled from:** not backfilled
**Decision owner:** MCP adapter workstream requester and reviewers
**Related task:** TASK-2294.3.2
**Related spec/plan:** `Docs/superpowers/specs/2026-07-23-mcp-skills-model-only-runner-design.md`; `Docs/superpowers/plans/2026-09-07-mcp-bounded-model-completion-adapter-implementation-plan.md`

## Decision

Bounded MCP completions reserve worst-case token and integer cost exposure durably before dispatch. Admission and settlement take the durable invocation-user lock first, then the explicit authenticated billing-scope lock when distinct. Admission reads uncached canonical actual usage under those locks, adds all outstanding reservations, and fails closed on unavailable accounting state.

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

A separate opaque, server-generated per-attempt owner token is persisted with
each new reservation. A trusted synchronous insertion notification acknowledges
this attempt's insert before transaction exit, not commit. Pre-dispatch cleanup
must match both execution ID and that durable owner token. If this insert rolls
back and a waiting attempt reuses the execution ID, stale cleanup cannot refund
the replacement. A commit-success/connection-release failure still permits cleanup
of this attempt's committed row. Duplicate attempts and cancelled duplicate
waiters never acquire insertion ownership. Legacy provisional rows without an
owner token do not grant cleanup ownership to a new attempt; no token is inferred
from an execution ID, a callback acknowledgement, or an uncertain outcome.

Hosted subscription limits apply only when the usage-quota master switch is enabled and a Billing repository is wired, following the host's Billing activation contract. OSS or inactive hosted Billing does not impose implicit free-plan limits. Explicit MCP operator bounds and durable usage/reservation recording remain mandatory for this certified execution path. Activation failures still fail closed rather than impersonating an inactive deployment.

The per-scope lock coordinates MCP admission and settlement, not legacy Chat calls that do not adopt this protocol. Operator pricing is captured by value; stored accounting excludes prompt, output, credentials, arbitrary metadata, and raw provider usage.

The reservation table is an internal cross-user AuthNZ accounting authority,
following the native quota-ledger boundary rather than a per-user content table.
AuthNZ pools do not establish `app.current_user_id` sessions. A forced owner-only
RLS policy would deny normal accounting or hide other users' exposure in a shared
billing scope, weakening admission limits. The reason-backed RLS exemption is
explicit in `Helper_Scripts/ci/rls_coverage_exemptions.txt`, not the uncovered-table
baseline. Server-issued identity, immutable scope, exact scope/execution predicates
and durable scope locks remain the authority; no public row-list API is added.
SQLite/PostgreSQL cross-user admission tests preserve shared-capacity refusal and
isolation from unrelated billing scopes. A future public reservation-read surface
requires its own authorization design rather than inheriting unrestricted access.

The issued credential handle's immutable billing-source category is retained in the
reservation and canonical usage row. Only the vendor-billed `user` category is
exempt from hosted and native per-user LLM quotas (ADR-083); a user key pointed
at an operator endpoint remains `user_unmetered` and counts. Hosted admission
independently reads operator-paid actuals and unresolved reservations under the
same scope lock. Explicit MCP operator ceilings and Resource Governor limits
still cover every source, including own-key work; the exemption cannot weaken
those safety bounds or let own-key exposure consume a hosted allowance.

Within durable admission, operator-paid work checks the explicit invocation
user's current native profile precedence and monthly-token/daily-cost/monthly-cost
limits (ADR-064). The uncached resolver borrows the owning transaction connection,
not an ambient prechecked context or an additional pool slot. Period-bound paid
actuals, all unresolved paid MCP exposure across billing scopes, and the proposed
worst case are counted together. Reconciled reservations are not counted twice;
unresolved exposure never ages out. Admission and settlement share the fixed
user-first lock order so a concurrent settlement cannot disappear between usage
and reservation reads. A required lookup/counter failure remains fail-closed and
content-free. Legacy Chat's fail-open wrappers remain unchanged and do not join
this transaction/lock protocol; no atomic coordination with legacy Chat is claimed.

The canonical usage row determines which completed calls and billing period are counted. Its legacy floating-point USD columns remain compatible, but strict MCP cost enforcement reads the exact integer actual cost from the reservation audit row committed in the same transaction. A missing or inconsistent settlement fails closed instead of reconstructing MCP costs from lossy floats.

Canonical MCP timestamps are bound as UTC values independently of the database session timezone. PostgreSQL's exact numeric token aggregates are accepted only when finite, integral, non-negative, and within the signed-integer accounting bound; malformed values are never treated as zero exposure.

## Follow-Up

Stages 4 and 5 compose dispatch ownership, certified transport, and the host factory. Legacy provider callers may adopt the durable reservation protocol separately.
