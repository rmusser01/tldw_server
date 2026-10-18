# Workflow WebSocket load benchmark — TASK-13820

Baseline: origin/dev0566130cbed2670efca6bbfa3c309031d2d3bc3b.
ADR check: no new ADR for a benchmark or local responsiveness repair; preserve auth, owner checks, frames and caching policy. ADR-059 governs tracking.

## Stage 1: Baseline and measurements
**Goal**: Measure production ASGI WebSocket path over real loopback TCP with isolated data.
**Success Criteria**: Reproducible setup/snapshot/event percentiles and correctness at1/16/64 clients, runtime/source identity, lifecycle cleanup and clear limits.
**Tests**: Harness validation/smoke behavior; baseline private app run.
**Status**: Complete

## Stage 2: Evidence-driven repair
**Goal**: Repair only reproduced root causes and compare matched workloads. Follow-up: remove concurrent per-row SQLite transfers with one native JSON page, preserving the original row/decode path for legacy payloads, unavailable JSON functions and oversized aggregates. Replace per-frame bare yields with a loop-shared 1ms work budget and a timer checkpoint, retaining ordered individual frames. No global read admission limit or schema shortcut. Follow-up profiling identified repeated metric-label normalization per frame; remove only the redundant second normalization inside MetricsRegistry.record, retaining first-pass validation and raw-label lookup/export normalization. Add normalization-once and stored-label regression coverage plus existing collision/privacy/bridge/alias verification. Disposable workload ABBA improved replay/P99 medians; final full-matrix evidence remains required.

**Scope and ADR assessment**: Shared get_events callers include REST, webhooks, investigations, voice workflows and WebSockets; PostgreSQL remains on its current path. Compatibility tests cover stored payload types, filters/limits/order, missing JSON1, oversized values and genuine database errors. Scheduler tests cover cumulative work, cancellation, separate loops and cleanup. These are local retrieval/scheduling implementation details under existing persistence/auth/frame rules, so no new ADR is needed; ADR-059 governs tracking. Performance must be established against a freshly matched baseline before this repair is considered complete.
**Success Criteria**: Red/green regressions preserve auth, ownership, event delivery and safe diagnostics; paired before/after results.
**Tests**: Connected workflow/stream/auth suites and benchmark correctness/error cases.
**Status**: Complete — measured candidate136070c23e passes310 connected tests, scoped lint/format/Bandit and independent code/data review. Full six-case interrupted/resumed ABBA has72 valid trials, exact delivery and zero errors in both variants. At64 clients/history500 replayP95 856.82→802.14ms and loopP99 6.34→2.65ms; both orders improve. Report retains setup/single-client costs, desktop activity and resumed gap. Earlier rejected prototypes remain excluded.

## Stage 3: Review and integration
**Goal**: Deliver a reviewed new PR against latest dev.
**Success Criteria**: Scoped lint/Bandit, independent exact-head review, requester Change summary and hosted checks before merge.
**Tests**: All hosted checks; latest-dev ancestry and merged source verification.
**Status**: In Progress

Integration rebase: all checks passed on ff733d7976, but dev advanced before merge. Rebased unchanged patches onto e1f7770953; independent review confirms all four measured source hashes and the lifespan-disabled benchmark path are unchanged. Retain the timing evidence; rerun connected tests and require fresh exact-head review/hosted checks before merge.

CI follow-up: isolate the chat refund test exception-state injection to the endpoint module binding, preserving refund/cleanup assertions and Python/framework exception handling. Verify failing-first with current CI framework versions, then adjacent runtime lifecycle tests and the simplified endpoint suite. No production change or benchmark rerun.

Latest integration: clean8-patch rebase onto dev7c94a2efb1 after upstream sandbox teardown/test-quality changes; exact independent review and four measured hashes unchanged. CI-version chat suite217 passed/1 skipped and9 lifecycle cases passed; new quality ratchet and fresh hosted checks remain integration gates.
