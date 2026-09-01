---
id: TASK-13145
title: Implement canonical admin webhook durable producers and final activation
status: In Progress
assignee: []
created_date: '2026-08-31 22:44'
updated_date: '2026-09-01 00:49'
labels:
  - admin
  - webhooks
  - authnz
  - incidents
  - frontend
  - security
  - activation
dependencies:
  - TASK-13111
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2842'
  - 'https://github.com/rmusser01/tldw_server/pull/2846'
documentation:
  - Docs/Design/2026-07-12-canonical-admin-outgoing-webhooks.md
  - >-
    Docs/superpowers/plans/2026-08-31-canonical-admin-webhook-durable-producers-activation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement upstream PR 3 from the approved canonical outgoing-webhook design. Add all six durable user and incident event producers, source-identity deduplication and reconciliation, transactional event_capture marking, final canonical route activation and legacy-handler removal, the complete operational admin Webhooks UI, receiver/operator documentation, and activation/end-to-end proof. Canonical mode must remain fail closed until every PR 3 gate passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An executable TDD plan maps every PR 3 requirement to exact files, interfaces, tests, verification commands, and reviewable commits.
- [ ] #2 user.created and user.deleted are inserted and expanded in the same AuthNZ transaction as their source mutation, with stable command identities, privacy bounds, and no source commit when event capture cannot succeed.
- [ ] #3 incident.created, incident.updated, incident.resolved, and incident.notify use encrypted file markers, stable source identities, crash-convergent reconciliation, strict key handling, and no duplicate canonical events.
- [ ] #4 Every producer transactionally records event_capture activity, enforces the exact six-event catalog and 64 KiB body limit, and creates version-pinned automatic deliveries without plaintext sensitive data.
- [ ] #5 The final runtime mounts exactly one canonical route per method and path, removes temporary legacy compatibility handlers, and enables on mode only after complete schema, migration, key, Jobs, worker, reconciler, and backlog preflight.
- [ ] #6 The operational admin Webhooks UI uses only the canonical API and provides catalog-driven registration management, one-time secret handling, status, history, test, redelivery, rotation, disable, and deletion workflows without persisting secrets or replay keys.
- [ ] #7 SQLite and PostgreSQL producer, activation, security, admin UI, and controlled-receiver end-to-end gates pass with automatic delivery, duplicate, retry, signature, test-header, and manual-redelivery proof.
- [ ] #8 Public receiver and operator documentation records event payloads, signing and deduplication, migration and key operations, rollout, disable, retention, forward-fix, rollback boundaries, and at-least-once unordered semantics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract shared deterministic event preparation and freeze six privacy-bounded payload contracts.
2. Add transactional user producers and atomic encrypted incident markers.
3. Reconcile incident markers with source-deduplicated crash recovery.
4. Remove legacy admin webhook routing and add two-phase activation checks.
5. Complete canonical Webhooks and incident-notify admin UI workflows.
6. Run controlled-receiver, SQLite/PostgreSQL, security, documentation, and release gates.
Detailed TDD plan: Docs/superpowers/plans/2026-08-31-canonical-admin-webhook-durable-producers-activation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-31 planning: Inspected merged PR #2842 boundaries across canonical delivery/domain/repository/runtime, user registration/deactivation transactions, file-backed incidents, route selection, admin Webhooks UI, and incident notification UI. The executable PR 3 plan preserves status compatibility while deleting runtime selection, separates durable incident webhook notification from stakeholder email, requires pending-file and reconciled-DB idempotency checks, and splits activation into migrate-mode predeploy plus no-traffic on-canary live gates.

Planning PR: https://github.com/rmusser01/tldw_server/pull/2846 at reviewed plan commit 62e4d6e9e5fe0badd84ff91bad8d78894c7a7594.

2026-09-01 execution start: Planning PR #2846 merged into dev as 8cb4d2dfdfdf04abee5bee6c08ae959092d413ce. Runtime branch codex/admin-webhooks-durable-producers-runtime starts exactly at that commit. Dependency PR #2842 merge 7b1450c927de9001975fe50694f37d91eb4ef8d6 is an ancestor of origin/dev. Pre-change verification is in progress.

Pre-change baseline at 8cb4d2df: backend command completed with TLDW_TEST_NO_DOCKER=1 after the unguarded run repeatedly attempted unavailable Docker/PostgreSQL provisioning and was interrupted without a test failure. Guarded result: 695 passed, 150 skipped, 1204 warnings in 388.60s; skips are the plan-permitted local PostgreSQL paths. Admin UI focused baseline: 3 files and 69 tests passed in 5.64s. bun run typecheck passed. bun install --frozen-lockfile installed the existing lockfile dependencies without tracked changes.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests and verification evidence recorded
- [ ] #3 Documentation and release notes updated
- [ ] #4 Bandit and frontend static/build gates completed
- [ ] #5 Independent review findings resolved
- [ ] #6 Final summary and PR link recorded
- [ ] #7 Known skips, blockers, and residual risks documented
<!-- DOD:END -->
