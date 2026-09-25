---
id: TASK-13343
title: Plan and implement paired Docker distribution and runtime gateway
status: In Progress
assignee: []
created_date: '2026-09-23 06:15'
updated_date: '2026-09-25 16:52'
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

In isolated worktree codex/complete-app-wp1, Task 1 signed-manifest verifier implemented test-first. The focused test first failed because the Release module was absent, then passed 15 cases after implementation. Existing release-helper baseline passed 43 tests. Black formatting applied; Bandit found 0 issues in new Release source. The verifier uses only stdlib plus cryptography so WP3 can package its single maintained source in the small launcher without importing the backend; the plan records this handoff. WP1 implementation remains in progress.

WP1 Task 2 complete: managed same-origin WebUI mode and production build wrapper. Six focused Vitest files (64 tests) pass; managed Turbopack standalone build, token sync, and bundle budget pass; four URL/credential sentinels absent from generated static/server/standalone files. Full frontend typecheck remains red with 93 diagnostics in untouched files (baseline qualification issue). Task 3 cookie isolation in progress.

WP1 Task 3 cookie isolation complete: backend CSRF cookie setting and logout clearing, Next runtime config/session cookie pair, browser runtime accessor and all affected readers. Verification: 32 AuthNZ unit tests; 6 focused HTTP/logout integration tests; 179 Next/runtime browser tests; 219 affected service tests. Two same-host ports filter to their own cookie pair. Frontend lint has one pre-existing any warning, typecheck remains 93 unrelated diagnostics. Bandit: 11 existing B106 findings in auth.py identical to HEAD; zero new findings. Gateway-level two-instance browser exercise is deferred to Task 4.

WP1 Task 4 gateway code ready: route table, Host/Origin checks, stripping forwarded/Next control/hop headers, private Next hop, streaming proxy including cancellation and WebSocket upgrade, read-only maintenance status. http-proxy-middleware 4.2.0 is pinned. Twelve real-socket Node checks and 145 Next runtime/session tests pass; rebuilt standalone Next through gateway proves direct runtime auth unavailable and gateway cookie exchange successful without forwarding the master key on normal backend routes. Bun frozen lock, lint, syntax, token sync, and bundle budget pass. Full frontend typecheck still has the same 93 unrelated diagnostics. Real FastAPI extracted-bundle and two-instance browser smoke remain WP1 qualification gates.
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
