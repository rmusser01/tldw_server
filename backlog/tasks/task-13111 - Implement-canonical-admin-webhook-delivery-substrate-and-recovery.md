---
id: TASK-13111
title: Implement canonical admin webhook delivery substrate and recovery
status: In Progress
assignee: []
created_date: '2026-08-23 03:15'
updated_date: '2026-08-28 15:54'
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-08-28: PR 2 implementation baseline at 1ad2f1e5b30c49ea75396e4b713496b73e875fec completed. The scoped baseline suite collected 760 tests: 642 passed and 117 PostgreSQL-dependent tests skipped locally. The only sandbox failure was test_public_http_hop_uses_real_socket_without_following_redirect because the sandbox denied a 127.0.0.1 ephemeral bind; rerunning that exact test with host loopback permission passed (1 passed, 2 warnings). This is an environment-conditioned baseline pass, not a product failure.

2026-08-28: Started Task 1 delivery contract and migration-095 implementation in codex/admin-webhooks-delivery-substrate. Scope includes the preflight-ruled AdminWebhookRepository.delivery_schema_ready() probe and focused coverage.

2026-08-28: Task 1 added migration-095 recovery tokens, per-attempt timeout, durable runtime heartbeats, fixed delivery settings/types, and the delivery schema readiness probe while preserving canonical schema version 1. Focused gate: 98 passed, 5 PostgreSQL tests skipped locally because no PostgreSQL instance was available; Ruff and git diff --check passed.

2026-08-28: Started Task 1 Fix Round 1. Addressing fail-closed delivery-schema structural preflight, closed heartbeat runtime-reason catalog, migration boundary coverage, required PostgreSQL verification, and warning triage.

2026-08-28: Task 1 Fix Round 1 complete. Hardened delivery_schema_ready() against backend-specific column, check, primary-key, and index/predicate drift; added closed heartbeat runtime reasons and 64-lowercase-hex disposition tokens; PostgreSQL-focused suite ran required with zero skips. Focused suite: 108 passed, 18 pre-existing warnings; Ruff and git diff --check pass.
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
