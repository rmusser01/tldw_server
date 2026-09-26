---
id: TASK-13343
title: Plan and implement paired Docker distribution and runtime gateway
status: In Progress
assignee: []
created_date: '2026-09-23 06:15'
updated_date: '2026-09-26 01:23'
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
- [x] #2 Create a runtime-configured gateway and managed WebUI build with authenticated routing and same-origin behavior.
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

WP1 Task 5 code/control slice: control tests red then 10 passed; full manifest+control 25 passed. Managed WebUI target now Node 24 with no build-time private API origin/key; gateway has dedicated frozen 14-package lock and Node 24 image; one-shot control image embeds a build-context trusted public key set. Docker image builds, inspect, and live asset smoke remain open: Docker Desktop socket on this host responds "Docker Desktop is unable to start" despite supported CLI/direct launch attempts. CI qualification in Task 7 must run those checks before acceptance. Scoped Bandit: 0 findings.

WP1 Task 6 bundle/helper source ready: signed-control verify/init precedes Compose pull/up; fixed digest/key placeholders are filled only during candidate packaging; shell and PowerShell start/stop/status use persisted state and project ID. Compose publishes only loopback gateway, persists backend DB/config volumes, and has no default Postgres/Redis. 35 manifest/control/helper tests pass, shell syntax passes, Compose config validates for amd64 and arm64, Bandit 0 new findings. Local live Docker and Windows PowerShell execution remain unverified because Docker Desktop cannot start here and pwsh is unavailable; Task 7 CI must exercise those gates.

WP1 Task 7 candidate tooling and manual CI lane implemented: 45 lean Release tests pass; signed exact-byte manifest, per-platform artifact roles/hashes, source-revision image labels, pinned local-registry refs, and G2/G4/G10/G12 evidence are checked. Manual GitHub Actions lane builds native amd64/arm64 candidates in separate job-local registries, runs shell/PowerShell syntax checks and an extracted-bundle smoke, and requires both jobs. No protected image or install catalog is published. Local Docker daemon remains unable to start, so neither image builds nor live smoke/actual runtime patch/size evidence has run here; both-platform CI must pass before TASK-13343 acceptance. Whole-frontend typecheck baseline remains 93 diagnostics in untouched files.

WP1 final self-review recorded in Docs/superpowers/reviews/2026-09-25-complete-app-wp1-acceptance.md. AC2 gateway/managed WebUI implementation is complete and locally tested. AC3/AC4 and G2/G4/G10/G12 acceptance remain open: Docker Desktop cannot start on this host; native amd64/arm64 manual CI candidate runs have not executed; no measured image sizes/runtime patches/startup times or two-instance browser setup proof exists. Manual CI now keeps G2/G4/G12 false and verifies the promotion gate refuses provisional candidates. No protected publication occurred. Shell start now waits for Compose health before printing/opening the browser; PowerShell source matches but runtime is untested.

WP1 review follow-up: 46 lean Release tests pass, Black/shell syntax/Compose config for amd64 and arm64/diff check pass, and scoped Bandit reports zero findings. Start helpers wait for service health and tear down partial services after failed readiness while retaining state. Candidate lane deliberately leaves G2/G4/G12 false and its promotion check must refuse it; live CI and full browser/two-instance/runtime evidence remain open.

Docker Desktop recovered for a local linux/arm64 run. Four images built and a signed extracted bundle passed control verify/init; Compose failed because backend became unhealthy. Isolated reproduction showed ModuleNotFoundError for tldw_profile_core. Added that local package to the backend builder and an early image-import guard. Corrected image imports the package, includes 2 schemas/45 fixtures, and became healthy on fresh volumes in 30.29 seconds. Full corrected signed-bundle smoke, two-instance browser checks, amd64 CI, Windows runtime, and download/installed size measurements remain open. Local disk headroom is about 15 GiB after repeated builds; no protected publication occurred.

User authorized branch push and native CI. Branch-only workflow trigger bootstraps the new lane; pinned Black 25.1.0 fixes the first CI formatter drift. Run 36206663211 at 754c9dd521 passed focused release/browser/gateway checks and Windows syntax, built all four native amd64/arm64 images, verified/init extracted signed bundles, and reached healthy backend/WebUI/gateway services. Both then exited 22 on a silent HTTP request. Found missing WebUI AUTH_MODE in Compose: regression red, real standalone Next/gateway returns 503 omitted versus 204/two cookies with single_user. Fixed configuration; 47 release tests, shell/Compose checks, production Bandit zero findings pass. Added HTTP error line/status and failure-evidence upload excluding private key. Corrected CI smoke pending; G2/G4/G10/G12 remain open. Local Docker content-store I/O error and 4.5 GiB host free prevent further local builds.
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
