---
id: TASK-13013
title: Prepare a verifiable core release candidate for downstream deployments
status: In Progress
assignee: []
created_date: '2026-08-21 19:48'
updated_date: '2026-08-21 20:02'
labels:
  - release-readiness
  - security
  - ci
  - deployment
  - epic
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-08-21-core-release-readiness-program-design.md
  - >-
    Docs/superpowers/plans/2026-08-21-core-release-readiness-backlog-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Produce a reusable public-core release candidate from dev with enforced CI, closed release-blocking security findings, production-safe reference configuration, deterministic provenance, and a downstream handoff contract. This epic deliberately excludes private hosted infrastructure, proprietary overlays, billing, legal terms, customer operations, and commercial launch decisions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The exact release candidate passes every documented required gate and cannot merge while required checks are red or stale.
- [ ] #2 Release version, changelog, GitHub release lineage, and package distribution state are internally consistent.
- [ ] #3 Reusable authentication, tenant isolation, logging, dependency, deployment, and data-lifecycle blockers are resolved or explicitly accepted before handoff.
- [ ] #4 The handoff records immutable source and artifact identity plus configuration, migration, and rollback compatibility without referencing private downstream repositories.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created from remote origin/dev 2e0815c1e4577902a220044619822ab6b1cb395f on 2026-08-21. Ten focused child tasks own the reusable readiness work; TASK-12116 remains the frontend-safety dependency; TASK-12983 now stops at the public downstream handoff.

Backlog CLI validation passed. The selected public graph contains 12 program and handoff records with a 13-record dependency closure, no missing dependency, and no cycle. Staged diff checking and prohibited-private-identifier scanning passed. Application tests and Bandit were not run because this change creates task and documentation records only and changes no executable code.

Initial task-graph materialization committed as eaa3d8e91f on codex/core-release-readiness-backlog. The epic remains In Progress because the release-readiness child tasks are intentionally not executed by this backlog-only change.
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
