---
id: TASK-13145
title: Implement canonical admin webhook durable producers and final activation
status: In Progress
assignee: []
created_date: '2026-08-31 22:44'
updated_date: '2026-09-01 02:16'
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
- [x] #2 user.created and user.deleted are inserted and expanded in the same AuthNZ transaction as their source mutation, with stable command identities, privacy bounds, and no source commit when event capture cannot succeed.
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

2026-09-01 Task 1 complete pending commit: Added six closed privacy-bounded production payload builders; extracted deterministic JSON snapshot/body validation, encrypted EventInsert preparation, stored-body validation, and exact source/body replay proof from synthetic delivery; added mode/key/migration preflight and transaction-bound AdminWebhookEventProducer; added caller-owned AdminWebhookRepository.unit_of_work(connection); and consolidated synthetic and production capture on one writable-key gate. Strict TDD reds were observed for missing events module, validation gaps, missing event preparation/replay interfaces, missing repository factory, and missing production producer. Final focused matrix: 76 passed, 3 skipped, 88 warnings in 35.77s (skips are existing environment-gated cases). The caller-owned SQLite transaction contract passed separately. Ruff passed all changed production/test paths; git diff --check passed. Self-review found no unresolved correctness or privacy issue; no subagent reviewer was available in this execution environment. PostgreSQL producer integration remains a Task 2 required gate.

2026-09-01 Task 2 complete: Added transactional user.created capture to every RegistrationService path and moved admin user deactivation to a service-owned AuthNZ transaction for user.deleted. Both flows generate stable command coordinates before mutation, read persisted status/timestamps/profile version, capture encrypted privacy-bounded events plus version-pinned automatic fanout before commit, and fail closed on migration/key/rotation/fanout failures. Admin create/delete endpoints normalize and forward request IDs; delete no longer owns a route-level get_db_transaction dependency. Already-inactive deletion preserves the existing success response as an effective no-op with no second event or audit. SQLite coverage proves active/inactive creation, role/audit/event/activity/fanout atomicity, mode off/migrate compatibility, four preflight failures with zero writes, fanout rollback and directory cleanup, deactivation success/no-op, deactivation fanout rollback, payload privacy, and endpoint request IDs. Existing admin/registration and route/OpenAPI compatibility suites passed. Verification: focused SQLite 12 passed; expanded Task 2 matrix 93 passed, 1 environment-gated PostgreSQL skip; admin route matrix 32 passed; required Docker-backed PostgreSQL parity 2 passed with zero skips; Ruff passed; production Bandit passed; git diff --check pending staged final. Self-review found no unresolved correctness/privacy issue; no subagent reviewer was available.

2026-09-01 Task 3 complete pending commit: Added versioned incident.created/updated/resolved production capture using encrypted PendingIncidentWebhookMarker records published atomically with system_ops.json mutations. Existing versionless incidents normalize to 1 and first effective writes persist 2; no-op updates do not save, bump, or emit. Resolved is emitted only on transition into resolved; reopen/timeline mutations emit updated. Marker records use strict closed shape with ciphertext/key ID/body size/source metadata and exclude title, summary, tags, timeline, notes, recipients, and stakeholder results. Mode-on migration/key/rotation preflight occurs before the file lock, atomic-save failure preserves prior bytes, and stakeholder email now preflights before any outbound email side effect. Replaced legacy direct incident dispatch with durable POST /admin/incidents/{incident_id}/notify-webhooks requiring a bounded Idempotency-Key and optional 4096-character narrative; response is 202 acceptance metadata only. Pending-file and reconciled-database replay checks enforce same command/body replay and changed-body conflict. Extended strict key-rotation coverage for duplicate IDs, authenticated identity substitution, and key loss without file mutation. Final exact Task 3 matrix: 69 passed, 2 warnings in 49.33s. Expanded route/OpenAPI/incident API/production-event/repository matrix: 69 passed, 6 warnings in 25.38s. Ruff passed all 12 changed Python paths; py_compile passed six production paths; git diff --check passed. Bandit adds zero findings: repository remains at the exact HEAD baseline of 43 medium/low-confidence B608 reports, and the other touched-file enum warning is pre-existing; no high-severity finding. Self-review found and fixed one preflight-order defect and one stale legacy-route test. Task 4 remains responsible for marker-to-database reconciliation, crash convergence, and runtime recovery; acceptance criterion 3 intentionally remains open until that work passes.
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
