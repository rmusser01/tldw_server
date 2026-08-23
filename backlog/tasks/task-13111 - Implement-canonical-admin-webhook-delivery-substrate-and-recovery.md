---
id: TASK-13111
title: Implement canonical admin webhook delivery substrate and recovery
status: To Do
assignee: []
created_date: '2026-08-23 03:15'
updated_date: '2026-08-23 03:59'
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
1. Add migration-092 AuthNZ recovery/heartbeat fields and persisted Jobs lease/quarantine controls.
2. Implement dual-backend event, fanout, delivery, attempt, disposition, health, and retention repositories.
3. Add backward-compatible Jobs prepared dispositions, no-attempt lease recovery, and lease horizon.
4. Extend peer-verified egress with no-buffer status-only mode and build the one-attempt executor.
5. Implement lifecycle cancellation, enqueue/disposition recovery, worker, test, redelivery, history, runtime, metrics, and health.
6. Run all four backend crash matrices, PostgreSQL-required, security, static, OpenAPI, and review gates.
Detailed plan: Docs/superpowers/plans/2026-08-23-canonical-admin-webhook-delivery-substrate.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created on a local planning branch based on reviewed PR 1 head 89f03feda2. Do not push or open a stacked PR until PR #2806 merges; then rebase this branch onto current dev before implementation. Planning review identified and incorporated additive schema-extension readiness while preserving canonical schema version 1, durable disposition tokens and absolute not-before timestamps, per-attempt persisted request timeout, runtime heartbeat persistence, queued cancellation without a worker lease, infrastructure-only pre-attempt deferral, and persisted Jobs expired-lease/quarantine controls. Final gate audit added Jobs migration compatibility/parity coverage, exact synchronous-test replay/preflight/reservation ordering, immutable enqueue controls, and first-canonical-activity traceability. The planning artifact is ready; implementation remains blocked on the merge and rebase gate.
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
