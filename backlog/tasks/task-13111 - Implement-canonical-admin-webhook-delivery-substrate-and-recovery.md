---
id: TASK-13111
title: Implement canonical admin webhook delivery substrate and recovery
status: In Progress
assignee: []
created_date: '2026-08-23 03:15'
updated_date: '2026-08-29 00:04'
labels:
  - admin
  - webhooks
  - jobs
  - security
  - recovery
dependencies:
  - TASK-13014
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2806'
  - 'https://github.com/rmusser01/tldw_server/pull/2828'
documentation:
  - Docs/Design/2026-07-12-canonical-admin-outgoing-webhooks.md
  - >-
    Docs/superpowers/plans/2026-08-23-canonical-admin-webhook-delivery-substrate.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement upstream PR 2 from the approved canonical outgoing-webhook design. Deliver the synthetic-event data plane, recoverable AuthNZ/Jobs handshakes, supported Jobs extension contracts, secure attempt executor and worker, synchronous tests, manual redelivery, history APIs, recovery, retention, metrics, and health. This task excludes durable user/incident producers, final legacy-handler deletion, canonical activation, and public release.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An executable TDD plan maps every PR 2 requirement to exact files, interfaces, tests, commands, and reviewable commits.
- [ ] #2 Event, delivery, and append-only attempt persistence plus set-based expansion behave equivalently on SQLite and PostgreSQL with encrypted bounded event bodies.
- [ ] #3 Enqueue, disposition, cancellation, and lost-acknowledgement recovery pass every crash point for SQLite/SQLite, SQLite/PostgreSQL, PostgreSQL/SQLite, and PostgreSQL/PostgreSQL.
- [ ] #4 Supported Jobs contracts provide fail-closed acquisition, no-attempt deferral, exact retry delay, disposition recovery, lease-horizon enforcement, and a webhook-safe quarantine threshold.
- [ ] #5 The webhook worker and shared attempt executor enforce signing, SSRF-safe status-only egress, retries, expiry, hard network-attempt limits, retention, metrics, and health without leaking destinations or secrets.
- [ ] #6 Synchronous test, manual redelivery, delivery-history, audit, and operational service contracts are implemented with durable idempotency and stale-screen preconditions.
- [ ] #7 All PR 2 gates pass on supported backend combinations while durable domain producers, final route cutover, and canonical activation remain absent.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add migration-095 AuthNZ recovery/heartbeat fields and persisted Jobs lease/quarantine controls.
2. Implement dual-backend event, fanout, delivery, attempt, disposition, health, and retention repositories.
3. Add backward-compatible Jobs prepared dispositions, no-attempt lease recovery, and lease horizon.
4. Extend peer-verified egress with no-buffer status-only mode and build the one-attempt executor.
5. Implement lifecycle cancellation, enqueue/disposition recovery, worker, test, redelivery, history, runtime, metrics, and health.
6. Run all four backend crash matrices, PostgreSQL-required, security, static, OpenAPI, and review gates.
Detailed plan: Docs/superpowers/plans/2026-08-23-canonical-admin-webhook-delivery-substrate.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The initial draft was created from an earlier reviewed PR 1 head. Planning review identified and incorporated additive schema-extension readiness while preserving canonical schema version 1, durable disposition tokens and absolute not-before timestamps, per-attempt persisted request timeout, runtime heartbeat persistence, queued cancellation without a worker lease, infrastructure-only pre-attempt deferral, and persisted Jobs expired-lease/quarantine controls. Final gate audit added Jobs migration compatibility/parity coverage, exact synchronous-test replay/preflight/reservation ordering, immutable enqueue controls, and first-canonical-activity traceability.

2026-08-28: PR #2806 and tracking PR #2828 are merged into dev. Recovered the existing two-file planning commit onto codex/task-13111-delivery-plan at dev merge commit 9fd2246157ce8a32ae6a6691a75efab788229f77. Reconciled the plan against the final PR 1 implementation: final reviewed head f37d4c448ace69b56e208ca1f9bda94c571d86f3 is present, canonical schema migration 094 is merged, and PR 2 now allocates migration 095. No PR 2 runtime code has started.

2026-08-28: Independent plan review found nine actionable gaps, all corrected before runtime work: lookup-only orphan Jobs cancellation, transactional structural-restore closure on accepted manual redelivery, RUN_JOBS=1 on every pytest gate, reacquired lost-ack reconciliation, typed Jobs admission plus fixed queue registration, complete Bandit path coverage, append-only Backlog command usage with preserved references, deterministic 30-second infrastructure deferral, and removal of the obsolete migration instruction.

2026-08-28: Focused re-review found two remaining P1 executability gaps. The plan now requires typed admission and create_job to share the complete existing validation/transformation/side-effect pipeline with facade parity tests, and requires first-application infrastructure deferral to derive and persist its absolute 30-second schedule inside the Jobs backend transaction while keeping explicit stale-attempt scheduling separate.

2026-08-28: Final consistency review separated AuthNZ disposition acknowledgement from infrastructure/recovery no-ack defers, added their crash/reacquisition proof, and verified a later exact AuthNZ disposition is not blocked by historical defer evidence. Final focused re-review reported no findings.

2026-08-28: PR 2 implementation baseline at 1ad2f1e5b30c49ea75396e4b713496b73e875fec completed. The scoped baseline suite collected 760 tests: 642 passed and 117 PostgreSQL-dependent tests skipped locally. The only sandbox failure was test_public_http_hop_uses_real_socket_without_following_redirect because the sandbox denied a 127.0.0.1 ephemeral bind; rerunning that exact test with host loopback permission passed (1 passed, 2 warnings). This is an environment-conditioned baseline pass, not a product failure.

2026-08-28: Started Task 1 delivery contract and migration-095 implementation in codex/admin-webhooks-delivery-substrate. Scope includes the preflight-ruled AdminWebhookRepository.delivery_schema_ready() probe and focused coverage.

2026-08-28: Task 1 added migration-095 recovery tokens, per-attempt timeout, durable runtime heartbeats, fixed delivery settings/types, and the delivery schema readiness probe while preserving canonical schema version 1. Focused gate: 98 passed, 5 PostgreSQL tests skipped locally because no PostgreSQL instance was available; Ruff and git diff --check passed.

2026-08-28: Started Task 1 Fix Round 1. Addressing fail-closed delivery-schema structural preflight, closed heartbeat runtime-reason catalog, migration boundary coverage, required PostgreSQL verification, and warning triage.

2026-08-28: Task 1 Fix Round 1 complete. Hardened delivery_schema_ready() against backend-specific column, check, primary-key, and index/predicate drift; added closed heartbeat runtime reasons and 64-lowercase-hex disposition tokens; PostgreSQL-focused suite ran required with zero skips. Focused suite: 108 passed, 18 pre-existing warnings; Ruff and git diff --check pass.

2026-08-28: Started Task 1 Fix Round 2. Binding readiness constraints and indexes to owning tables, correcting heartbeat NULL reason SQL semantics, adding equivalent PostgreSQL instance bounds, and re-verifying migration 095 without a new migration.

2026-08-28: Task 1 Fix Round 2 complete. delivery_schema_ready now binds delivery/attempt/heartbeat checks and named indexes to their owning tables, validates PostgreSQL relation ownership, and migration 095 rejects unready heartbeat rows with NULL reason_code. Added SQLite/PostgreSQL decoy-index and wrong-table-check coverage plus PostgreSQL 129-character instance rejection. Full required suite: 112 passed, 0 skipped (PostgreSQL required); Ruff and diff checks passed. Warning triage: 2 environment/dependency configuration warnings and 20 existing deprecated shared-fixture shutdown warnings; no Task 1 warning introduced.

2026-08-28: Task 1 Fix Round 3 complete. PostgreSQL delivery_schema_ready now validates per-key pg_index.indoption DESC flags for every required recovery index: both delivery recovery indexes are all ascending and heartbeat freshness is component/ready ascending plus heartbeat_at descending. Added expected-name wrong-DESC tests for both recovery indexes; full required suite 114 passed, 0 skipped with PostgreSQL required; Ruff and diff checks passed.
2026-08-28: Started Task 2 repository/UoW persistence and crypto-validation implementation on codex/admin-webhooks-delivery-substrate. Following strict backend-neutral contract-test RED before implementation, then required SQLite/PostgreSQL parity and static verification.
2026-08-28: Task 2 complete. Implemented dual-backend encrypted event capture/set-based fanout, bounded history/readback, enqueue/attempt/disposition CAS, stale recovery, cancellation, expiry, runtime heartbeat, and ordered retention repositories plus bounded ProtectedValue validation. Strict RED: 3 expected collection errors, 5 warnings. Required final focused suite: 42 passed, 0 skipped, 78 pre-existing environment/dependency/shared-fixture warnings with PostgreSQL required. Crypto regression: 35 passed. Ruff and git diff --check passed. Bandit had only 22 low-confidence B608 reports from fixed module SQL fragments/allowlisted table interpolation; a scan excluding the separately triaged B608 class passed.
2026-08-28: Started Task 2 Fix Round 1. Scope: cancellation locking/full-snapshot CAS and processing preservation; disposition scheduling invariants; canonical UUIDv4/token/runtime coordinates; atomic disposition acknowledgement; mandatory retention, attempt-budget, synchronous-test, and malformed-row contract coverage. Strict TDD RED will be recorded before production changes.
2026-08-28: Task 2 Fix Round 1 complete. Cancellation now locks PostgreSQL pre-reservation candidates, uses SQLite write transactions, excludes processing, and applies exact state/current-attempt/Jobs/enqueue-coordinate/version CAS with rollback-visible stale-state errors. Disposition scheduling, canonical UUIDv4/token/runtime coordinates, and atomic attempt+delivery acknowledgement are fail closed. Added dual-backend race, processing-preservation, all-kind scheduling, acknowledgement, retention-order/nonterminal, fifth-attempt no-mutation, synchronous terminal-readback, and malformed-row coverage. Required five-file suite: 66 passed, 0 skipped, 118 pre-existing environment/dependency/shared-fixture warnings with PostgreSQL required. Crypto: 35 passed, 0 skipped, 2 pre-existing warnings. Ruff and diff checks pass. Raw Bandit reports 24 medium-severity/low-confidence B608 findings, all fixed column fragments or closed identifier selections/allowlists with bound caller values; excluding reviewed B608, Bandit passes.
2026-08-28: Started Task 2 Fix Round 2. Test-focused scope: prove real SQLite/PostgreSQL rollback when acknowledgement loses the delivery-marker CAS after updating the terminal attempt, and complete backend-backed malformed persisted coordinate/constraint evidence for event, delivery, attempt, redelivery, synchronous test token, and pending disposition token. Production changes only if RED exposes a defect.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-08-28: Task 2 Fix Round 2 complete. Added backend-neutral real-transaction acknowledgement rollback evidence and full malformed persisted-coordinate/constraint coverage on SQLite and PostgreSQL. No production defect was exposed, so production and crypto code were unchanged. RED: 2 collection errors from missing shared contracts. Corrective subset: 5 passed, 0 skipped, 6 warnings. Full PostgreSQL-required Task 2 suite: 70 passed, 0 skipped, 122 warnings. Ruff and git diff --check passed; Bandit/crypto were not required because production did not change.

Task 3 complete: added backend-neutral exact prepared dispositions, no-attempt defer, lease-horizon and identity lookup operations; unified typed/legacy admission; persisted expired-lease recovery controls; and SQLite/PostgreSQL parity. Mandatory Task 3 plus lifecycle suite: 263 passed, 0 skipped. Focused Ruff and Bandit passed. Full Jobs Ruff remains blocked only by pre-existing I001 findings in unchanged audit_bridge.py, metrics.py, and tracing.py.

Task 3 Fix Round 1/5 started at FIX_BASE 803ae280f66e990f7b4ffdf29e31cae311d648d7. Scope: canonical identity boundary, exact-token replay fact conflicts, Slides replay control validation, compressed archive identity lookup, and defer event reason evidence. Strict focused RED before production edits; mandatory PostgreSQL zero-skip gate required.

2026-08-28: Task 3 Fix Round 1/5 complete. Hardened canonical admin_webhooks:delivery admission/facade/backend identity and controls; exact-token reason/delay replay facts via bounded internal fingerprint; Slides immutable control replay/race validation; real compressed archive lookup; and current defer reason event evidence. Added strict marker/schedule forgery rejection. RED: 68 failed/95 passed, then 4 focused semantic failures. Final PostgreSQL-required Task 3/lifecycle gate: 343 passed, 0 skipped, 2 baseline warnings. Focused Ruff, Bandit, and diff checks pass. Full Jobs Ruff remains only the three unchanged baseline I001 findings; unrelated PostgreSQL Slides nullable-parameter audit issue remains unchanged.

2026-08-28: Started Task 3 Fix Round 2/5 at FIX_BASE 56fc110798fb44200e0fe6bd90b7973e059bdf2e. Scope: revalidate final canonical admission payload after shared transforms; exact backend-precision schedule comparison; durable no-attempt expired-lease recovery evidence across acquisition/sweeper; and bounded strict archive decompression. Strict focused RED precedes production edits; PostgreSQL-required zero-skip final gate.

2026-08-28 Fix Round 2/5 complete at pre-commit tree: canonical admin-webhook payload is revalidated after shared transforms; schedules use exact backend storage precision; no-attempt recovery now preserves bounded durable evidence and consumes it on reacquisition/cancel; archive decompression is strictly framed and bounded. RED: 25 failed, 139 passed, 0 skipped. Final mandatory PostgreSQL-required gate: 368 passed, 0 skipped, 2 warnings. Focused Ruff and Bandit pass; diff check clean. Evidence: .superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-3-report.md.

2026-08-28: Started Task 3 Fix Round 3/5 at FIX_BASE 45a1fbd90d09af8d45c43b88e59ec8a335ac9c63. Scope: make every compressed-column decode failure unambiguously invalid; reject raw JSON bytes/text and noncanonical standard-base64 spelling; preserve valid bounded SQLite gzip64 and PostgreSQL gzip-byte archive lookup. Strict focused RED precedes production edits; PostgreSQL-required zero-skip final gate.

2026-08-28: Task 3 Fix Round 3/5 complete at pre-commit tree. Compressed decode failures now return a private invalid sentinel instead of reinterpreting raw compressed-column data as JSON; gzip64 requires exact standard-base64 re-encoding equality. RED: 8 failed, 4 passed, 0 skipped. Focused GREEN: 14 passed, 0 skipped. Final PostgreSQL-required Task 3/lifecycle gate: 376 passed, 0 skipped, 2 warnings. Focused Ruff/Bandit and diff checks pass; full Jobs Ruff remains only three unchanged baseline I001 findings. Evidence: .superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-3-report.md.

2026-08-28: Started Task 3 Fix Round 4/5 at FIX_BASE 2be720aea8b39fdc99997c72a03da3814b0d7998. Scope: replace compressed archive invalid-sentinel leakage with one explicit closed normalization failure contract across JobManager Slides lookup, SQLite/PostgreSQL receipt replay, and migration collision/integrity callers. Strict focused RED precedes production edits; PostgreSQL-required zero-skip final gate.

2026-08-28: Task 3 Fix Round 4/5 complete at pre-commit tree. Replaced shared Slides archive invalid-sentinel leakage with a reload-stable zero-argument normalization exception and mapped every caller to its closed lookup, replay, migration, or collision contract. RED: 17 failed/2 passed, then 1 decoder-context failure and 1 reload-stability failure. Final focused PostgreSQL-required suite: 20 passed, 0 skipped, 137 deselected. Mandatory Task 3/lifecycle gate: 377 passed, 0 skipped. Relevant Slides/idempotency regressions: 109 passed, 0 skipped. Focused Ruff and Bandit pass; diff checks clean. Full production Jobs Ruff remains only the three unchanged I001 baseline findings; known PostgreSQL Slides nullable-parameter audit issue remains unchanged. Evidence: .superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-3-report.md.

2026-08-28: Started Task 3 Fix Round 5/5 at FIX_BASE e336b30578a4993f3f5a3c876cdc8bbe5dd0f13b. Scope: validate every non-null Slides archive compressed sidecar against primary JSON, close all public remap exception chains, and update the PostgreSQL audit source contract. Strict focused RED precedes production edits; PostgreSQL-required zero-skip final gate.

2026-08-28: Task 3 Fix Round 5/5 complete at pre-commit tree. Every non-null Slides archive sidecar is now strictly decoded and must exactly match retained primary JSON; all manager and SQLite/PostgreSQL idempotency public remaps have empty cause/context; canonical lookup remains CONFLICT; the PostgreSQL audit contract asserts _SLIDES_PG_AUDIT_EXCEPTIONS. Host RED: 53 failed/7 passed/0 skipped. Focused GREEN: 60 passed/0 skipped. Exact audit contract: 1 passed. Mandatory PostgreSQL-required Task 3/lifecycle gate: 391 passed/0 skipped. Stable complete Slides/idempotency gate: 130 passed/0 skipped. Focused Ruff, Bandit, and diff check pass. The known unchanged PostgreSQL Slides nullable-parameter audit defect remains documented in the Task 3 report.

2026-08-28: Started Task 3 post-Fix-Round-5 breaker remediation at FIX_BASE 82a29ee11697b39eb07396174b0db06908aee14a. Scope: centralized type-exact JSON comparison including non-finite floats, raw SQL JSON-null presence evidence, and fail-closed parity across migration, collision/prune, replay, and identity paths. Strict focused RED precedes production edits; PostgreSQL-required zero-skip and all mandatory gates remain required.

2026-08-28: Task 3 post-Fix-Round-5 breaker remediation complete at pre-commit tree. Added one recursive type-exact archive comparator (including SQLite NaN and signed infinity semantics), separated SQL-column presence from parsed JSON null with bounded private query evidence, and applied exact comparison across migration, collision/prune, receipt replay/prune, and generic identity paths without metadata leakage. RED: 49 failed/25 passed/0 skipped, plus corrected direct subset 5 failed/3 passed. Focused breaker GREEN: 74 passed/0 skipped; prior malformed/framing/reload/secrecy GREEN: 74 passed/0 skipped. Mandatory PostgreSQL-required Task 3/lifecycle gate: 407 passed/0 skipped. Stable complete SQLite Slides plus both idempotency suites: 176 passed/0 skipped. Focused Ruff passed 15 changed Python files; Bandit passed 8 production files; diff check clean. Known optional PostgreSQL audit IndeterminateDatatype baseline remains intentionally unchanged. Evidence: .superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-3-report.md.

2026-08-28: Task 4 complete at pre-commit tree. Added fail-closed prepared WorkerSDK path with observable renewal/horizon state, closed handler-error evidence, one typed disposition application, origin-aware bounded callbacks, and no default double finalization; legacy run guard semantics remain green. RED: 31 failed/38 passed/2 warnings because run_prepared was absent. GREEN: 69 passed/2 baseline warnings. Focused Ruff, Bandit, and diff checks pass. Evidence: .superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-4-report.md.

2026-08-28: Started Task 4 Fix Round 1/5 at FIX_BASE 3e9a946bfa144df0eaa4236bcdb4b3cca9def179. Scope: authoritative acquired-deadline renewal scheduling, outer-cancellation-safe renewal teardown, immutable prepared CAS facts with defensive callback/factory copies, and adversarial command/horizon/guard coverage. Strict RED precedes production edits; prepared plus legacy full gate remains required.

2026-08-28: Completed Task 4 Fix Round 1/5 at fix base 3e9a946b. Prepared renewal now schedules from authoritative leased_until, immediately ensures unsafe initial horizons, bounds short-lease renewal to avoid busy loops, and applies jitter earlier only. Renewal teardown preserves outer cancellation after cancellation-resistant child cleanup using a Python 3.10-compatible stop event. CAS facts are frozen before user code and factory/callback jobs are defensive copies. Added adversarial coverage for capped/missing/malformed horizons, sticky renewal loss, guard/teardown cancellation, and factory mutation. Verification: 81 passed, 2 baseline warnings; focused Ruff, Bandit, git diff --check, and Python 3.10 py_compile all passed. Report: .superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-4-report.md

2026-08-28: Started Task 4 Fix Round 2/5 at FIX_BASE 0780bac3950cdb50cbd0753cbc713218a9063308. Scope: deterministic RED tests and prepared-path fixes for wall-clock-independent cap-aware renewal scheduling and unambiguous shielded cleanup cancellation; frozen CAS facts, callback routing, exactly-one apply, and legacy run() remain unchanged.

2026-08-28: Completed Task 4 Fix Round 2/5 at fix base 0780bac3950c. Prepared renewal now always obtains an authoritative typed relative lease guarantee before sleeping, reloads and clamps the current manager cap on every cycle, uses only positive event-loop-relative intervals with earlier-only jitter, and fails closed with sticky renewal loss plus class-only logging on invalid configuration. Teardown uses a separately shielded cleanup task so simultaneous/repeated parent cancellation is retained and re-raised after child cleanup. Frozen CAS facts, defensive job copies, exactly-one apply, callback routing, and legacy run() remain unchanged. RED: 10 failed/6 passed/35 deselected, plus a cap-reload characterization failed with [30, 30, 30]. Final gate: 90 passed/2 baseline warnings. Focused Ruff, Bandit, git diff --check, and Python 3.10 py_compile passed. Evidence: .superpowers/sdd/2026-08-23-canonical-admin-webhook-delivery-substrate/task-4-report.md
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
