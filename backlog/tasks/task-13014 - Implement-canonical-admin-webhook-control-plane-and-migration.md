---
id: TASK-13014
title: Implement canonical admin webhook control plane and migration
status: In Progress
assignee: []
created_date: '2026-08-21 20:41'
updated_date: '2026-08-22 16:17'
labels:
  - admin
  - webhooks
  - security
  - migrations
dependencies:
  - TASK-13013
documentation:
  - Docs/Design/2026-07-12-canonical-admin-outgoing-webhooks.md
  - Docs/superpowers/plans/2026-08-21-canonical-admin-webhook-control-plane.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement upstream PR 1 from the approved canonical outgoing-webhook design. Deliver the canonical admin API, shared schemas and repository contracts, contextual encryption/key handling, SQLite and PostgreSQL schema migration, deterministic legacy import and encrypted rollback backup, and temporary route-selection controls that isolate the existing legacy handlers. This task stops before network delivery, Jobs workers, durable event producers, final legacy-handler deletion, or canonical feature activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An executable TDD implementation plan maps every PR 1 requirement to exact files, interfaces, tests, commands, and commits.
- [ ] #2 Canonical registration and idempotency persistence works on SQLite and PostgreSQL with revision ETags, bounded registrations, tombstones, and context-bound encrypted URL and secret material.
- [ ] #3 Legacy JSON and database records import deterministically with collision handling, encrypted backup provenance, resumability, and documented rollback limits.
- [ ] #4 Focused unit, migration, repository, API, authorization, encryption, and legacy-import tests pass on supported databases; security and documentation verification is recorded.
- [ ] #5 Temporary routing mounts either the isolated legacy webhook router or the canonical webhook router for a process, never duplicate method/path handlers; final compatibility deletion remains assigned to PR 3.
- [ ] #6 Create, list, get, update, delete, rotate-secret, catalog, and status API contracts enforce platform-admin authorization, preconditions, idempotency ordering, one-time secret disclosure, and generic lost-secret responses; test/redelivery remain unavailable until PR 2.
- [ ] #7 The admin UI consumes the catalog and ETag contracts, handles create/rotate one-time secrets without browser persistence, reuses one in-memory key only for same-command transport retries, and recovers lost responses through a new rotation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define canonical contracts and dedicated encryption.
2. Add equivalent SQLite/PostgreSQL persistence.
3. Implement repository, lifecycle, rotation, and importer.
4. Expose exclusive API routing and update the admin UI.
5. Run migration, security, UI, and OpenAPI gates.
Detailed plan: Docs/superpowers/plans/2026-08-21-canonical-admin-webhook-control-plane.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-21: Created after approval of TASK-13013. Planning scope is upstream PR 1 only; delivery substrate/recovery and durable producers/final activation remain separate future tasks.

Planning correction: the approved design assigns final legacy-handler deletion and final canonical mount to PR 3. PR 1 therefore extracts/isolate-selects legacy routes and introduces canonical routes behind an exclusive startup selector; it does not delete compatibility behavior.

2026-08-21 planning review completed against origin/dev 2e0815c1e4577902a220044619822ab6b1cb395f. Added the executable TDD plan at Docs/superpowers/plans/2026-08-21-canonical-admin-webhook-control-plane.md and revalidated the approved design. Review corrections cover exact route selection, mandatory mutation/operational audit ordering, central egress-policy composition, contextual key loading and rotation, deterministic dual-backend migration, crash-safe artifacts and rollback retirement, literal report approval, durable activity boundaries, ETag/idempotency proxy contracts, synchronous browser secret clearing, required disposable-PostgreSQL gates, and no-network/Jobs scope enforcement. Documentation-only verification: git diff --check passed; all 12 numbered tasks are present; Markdown fence counts are even (design 14, plan 130); extracted bash blocks pass bash -n; stale-version/path and placeholder scans returned no matches. Bandit/runtime tests are not applicable to this docs-only planning unit; TASK-13014 remains In Progress for implementation.

2026-08-21: Draft planning PR opened: https://github.com/rmusser01/tldw_server/pull/2797. Repository policy requires the human requester to replace the explicit Change summary placeholder with their own explanation of what changed and why before the PR is marked ready or merged.

2026-08-22: Execution started from merged origin/dev ec333fda4e74f49fefc41907ea1fb23e215f13ed in isolated worktree .worktrees/admin-webhooks-control-plane on branch codex/admin-webhooks-control-plane. Qodo review on merged planning PR #2797 contained no actionable findings; its completed assessment recommended the current design. Migration 091 remains unclaimed.

2026-08-22 baseline: Backend focused regression matrix passed 138 tests with 2 warnings. Admin proxy baseline passed 14/14 tests. Existing admin webhook page baseline passed 2/8 and failed 6/8 because legacy test fixtures omit event_types; page.tsx raises TypeError at webhook.event_types.map. This predates TASK-13014 implementation and is directly superseded by Task 10 canonical page/fixture replacement; retain as a named baseline failure until that task turns the focused suite green.

2026-08-22 Task 1 complete: added immutable mode/settings/catalog/domain contracts, strong registration ETags, request-ID normalization, domain-separated idempotency digests, strict target validation/redaction, and the focused global platform-webhook egress adapter. Corrected existing WORKFLOWS_WEBHOOK wildcard normalization (*.example now normalizes to example) and added production-detector parity for lowercase tldw_production. TDD evidence: missing-package RED; targeted production-parity RED; final focused matrix 130 passed with 2 existing warnings. Ruff and Bandit passed; git diff --check passed.
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
