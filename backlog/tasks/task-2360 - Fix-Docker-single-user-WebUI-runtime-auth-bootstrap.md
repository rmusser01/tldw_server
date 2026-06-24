---
id: TASK-2360
title: Fix Docker single-user WebUI runtime auth bootstrap
status: In Progress
labels:
- docker
- webui
- auth
- setup
priority: High
documentation:
- Docs/superpowers/specs/2026-06-24-docker-webui-runtime-auth-bootstrap-design.md
- Docs/superpowers/plans/2026-06-24-docker-webui-runtime-auth-bootstrap-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-24-docker-webui-runtime-auth-bootstrap-design.md
- Docs/superpowers/plans/2026-06-24-docker-webui-runtime-auth-bootstrap-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address Docker single-user startup/auth issues by designing and implementing a runtime WebUI auth bootstrap, setup remote-write configuration, and related Docker/docs/test updates. Track stale mcp_unified Docker guidance separately in the design rather than adding a nonexistent root package copy in this branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Runtime-auth WebUI bootstrap works without requiring Docker users to bake `NEXT_PUBLIC_X_API_KEY` into the WebUI image.
- [ ] #2 Runtime auth takes precedence over stale build-time public env auth and does not overwrite user-managed credentials.
- [ ] #3 Docker single-user compose enables authenticated setup writes from the WebUI container via `TLDW_SETUP_ALLOW_REMOTE=1`.
- [ ] #4 Docker WebUI compose passes runtime auth env to the WebUI service while preserving loopback host port binding.
- [ ] #5 Setup onboarding write calls remain authenticated; no unauthenticated `noAuth` regression is introduced.
- [ ] #6 Docs clarify runtime auth bootstrap as the Docker quickstart default and `NEXT_PUBLIC_X_API_KEY` as advanced/static-build compatibility.
- [ ] #7 Stale root `mcp_unified` Docker guidance is handled with branch-accurate verification rather than an unconditional nonexistent package copy.
- [ ] #8 Focused tests or verification cover runtime endpoint guards, bootstrap precedence, compose wiring, and active MCP package/import shape.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-24-docker-webui-runtime-auth-bootstrap-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-24 design review follow-up: tightened the design so runtime auth explicitly outranks stale build-time `NEXT_PUBLIC_X_API_KEY`, `_app.tsx` awaits a named bootstrap promise before auth-state checks, forwarded request headers disable default runtime-auth exposure, and the task has concrete acceptance criteria after the MCP edit helper did not populate the AC block.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
