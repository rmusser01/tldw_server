---
id: TASK-13265
title: Design complete WebUI and server distribution for Docker and native installs
status: In Progress
assignee: []
created_date: '2026-09-21 05:14'
updated_date: '2026-09-21 05:54'
labels:
  - packaging
  - distribution
  - webui
  - pypi
  - design
dependencies: []
references:
  - backlog/tasks/task-12158 - Plan-tldw-frontend-static-PyPI-bundle.md
  - Docs/ADR/029-tldw-frontend-static-pypi-bundle.md
  - Docs/Development/Packaging_and_Distribution_Strategy.md
  - Dockerfiles/Dockerfile.webui
  - tldw_Server_API/cli/wizard/profiles.py
documentation:
  - Docs/Design/2026-09-20-complete-app-distribution-review.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resume the expired installation brainstorm and design a complete application distribution that new users can install without a repository clone, frontend compilation, or manual connection and credential wiring. This task covers design and implementation decomposition; publication and runtime implementation are separate follow-up work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Record the approved native and Docker distribution architecture and supported-platform requirements.
- [ ] #2 Agree on first-run behavior, lifecycle, data ownership, updates, optional components, and the tested OS and architecture matrix.
- [ ] #3 Write and review a design under Docs/Design with repository evidence, security boundaries, failure recovery, release constraints, and acceptance checks.
- [ ] #4 Define implementation slices for Docker release bundles, lean native application packaging, and guided optional processing components.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved by the user in this conversation: native installation without Docker; Windows, macOS, and Linux; full WebUI and core workflows by default with guided setup for heavy local engines; small PyPI launcher plus verified matched release bundles and a private Node runtime; prebuilt Docker distribution from the same product release; retain the API-only tldw-server package. tldw-app and tldw start are provisional names. The user approved the distribution architecture, not the remaining detailed design. Next section: installation and everyday startup/shutdown. Existing TASK-12158 and ADR-029 propose static export inside the backend wheel; the newly approved architecture takes a different approach. Existing packaging documentation records a frontend publication licensing freeze; preserve this as a release constraint while continuing design and local preparation. No runtime implementation or publication performed. MCP search calls were unresponsive; used repository searches and the official CLI fallback.

User approved the installation and everyday lifecycle section: uv is the documented bootstrap prerequisite and can supply Python; provisional uv tool install tldw-app then tldw start; first start downloads and verifies matched components and creates private configuration with visible progress; open the WebUI after services and the authenticated connection are ready; loopback-only default; existing provider setup wizard; background operation survives terminal/browser closure; repeated start reuses the running instance; stop/status/logs commands; login autostart is optional; application data is outside replaceable install files; retry interrupted installation and clean up child processes after failed startup; later starts reuse the installed release. No changes to runtime code. Next section, still proposed: explicit coordinated product upgrades with candidate environments and pre-migration recovery snapshots; preserve the active installation on failed downloads or optional engine setup; retain data on uninstall; distinguish application-bundle rollback from database restoration. Preserve the stable browser origin and account for browser-local drafts/settings in backup coverage and upgrade testing. Exact CPU/OS support matrix and release acceptance checks remain open.

User accepted the data/update/optional-engine policy in principle and requested an issue and improvement review before further design. Reviewing the proposed contract against source plus an independent recovery/lifecycle review; no implementation. Early findings: build-time WebUI rewrites need a runtime-routing decision; browser IndexedDB/local storage is outside server backup coverage; install and update require an independent lifecycle owner; Docker update authority must remain outside the app container; repository-dependent paths and heavyweight dependencies prevent a simple launcher-only solution; current WebUI Docker runtime is Node 20 (upstream now EOL). Review recommendations will be recorded in a design-review document and remain distinguishable from approved product requirements.

Design review completed and saved in Docs/Design/2026-09-20-complete-app-distribution-review.md. Retain the architecture; ten correction areas cover lean core/runtime prerequisites, build-time routing, coherent server and browser backup coverage, isolated candidate startup and maintenance gates, isolated optional installs, independent supervisor and Docker update authority, paths/bootstrap, authenticated matched releases and supported runtimes/platforms, and the existing publication policy. Independent backend review corroborated recovery and lifecycle findings, which were verified against source. These are recommendations, not implemented fixes. Report structure/source-location/whitespace checks passed. Bandit and application tests are not applicable to documentation/task-only edits. No installation, runtime changes, registry actions, or publication performed. Continue the design only after presenting this requested review; supported-platform details and final spec remain open.
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
