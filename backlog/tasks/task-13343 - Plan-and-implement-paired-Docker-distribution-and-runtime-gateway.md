---
id: TASK-13343
title: Plan and implement paired Docker distribution and runtime gateway
status: In Progress
assignee: []
created_date: '2026-09-23 06:15'
updated_date: '2026-09-23 06:23'
labels:
  - distribution
  - docker
  - webui
dependencies:
  - TASK-13265
references:
  - Docs/Design/2026-09-20-complete-app-distribution-design.md
documentation:
  - Docs/superpowers/plans/2026-09-22-complete-app-wp1-paired-docker-gateway.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
WP1 of TASK-13265: establish the shared manifest/artifact contract, runtime gateway, managed WebUI mode, and paired prebuilt Docker bundle with idempotent initialization. Plan and implement in reviewable tested slices. Keep protected frontend artifacts local/CI until the separate publication gate is satisfied.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Write an executable WP1 plan with exact file responsibilities, test cycles, and G2/G4/G10/G12 coverage.
- [ ] #2 Create a runtime-configured gateway and managed WebUI build with authenticated routing and same-origin behavior.
- [ ] #3 Build a paired Docker bundle with pinned artifacts, idempotent initialization, and Docker-only host helpers.
- [ ] #4 Verify a fresh Docker setup and networking/security paths outside a repository checkout before publication.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the eight tasks and four stages in Docs/superpowers/plans/2026-09-22-complete-app-wp1-paired-docker-gateway.md, with a failing behavioral test, focused pass, review, and scoped commit for each task. Preserve the frontend publication freeze; qualify candidate images in a job-local ephemeral registry only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Written specification approved on 2026-09-22. Planning WP1 as a separate reviewable slice; initial codebase mapping found build-time Next rewrites, fixed CSRF cookie names in backend/browser, and existing Docker Compose dependency on source checkout plus Postgres/Redis. The WP1 plan will cover those boundaries and local CI artifact qualification; publication remains gated.

WP1 implementation plan completed and self-reviewed: four stages, eight tasks, file/interface map, TDD steps, exact commands, and G2/G4/G10/relevant-G12 mapping. Corrected CSRF accessor signature, WebSocket upgrade authorization, and release signing responsibility. Verified plan structure, placeholders, whitespace and scoped diff. Documentation/task-only update: application tests and Bandit do not apply to this planning commit. Implementation criteria AC2-AC4 remain open.
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
