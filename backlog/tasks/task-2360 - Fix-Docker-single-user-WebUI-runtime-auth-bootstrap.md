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
- apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts
- apps/tldw-frontend/__tests__/pages/api/runtime-config.test.ts
- apps/tldw-frontend/extension/shims/runtime-bootstrap.ts
- apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts
- apps/tldw-frontend/__tests__/auth.mode.test.ts
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
2026-06-24 implementation update: Task 1 runtime config endpoint completed and reviewed. Added WebUI-local /api/_tldw-webui/runtime-config guard path with quickstart-only runtime auth exposure, loopback/Docker-gateway peer handling, forwarded-header rejection, placeholder/short/whitespace API key rejection, and 40 focused Vitest cases. Local verification: bunx vitest run __tests__/pages/api/runtime-config.test.ts passed.

2026-06-24 Task 2 update: Web runtime bootstrap now exports runtimeBootstrapReady, fetches /api/_tldw-webui/runtime-config before env/storage seeding on http/https pages, sets runtime API key precedence, preserves manual stored tldwConfig keys, replaces runtime-owned or stale build-time keys, and writes tldwRuntimeAuthMetadata with a non-secret fingerprint. Verification: bunx vitest run __tests__/extension/runtime-bootstrap.test.ts passed. Required exact command bunx vitest run __tests__/extension/runtime-bootstrap.test.ts __tests__/auth.mode.test.ts __tests__/auth.logout.test.ts is blocked by pre-existing auth.mode.test.ts import-time API base config failure when NEXT_PUBLIC_API_URL is unset; diagnostic with NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 passed all 25 tests. Bandit not run: touched implementation is TypeScript only.

2026-06-24 Task 2 blocker follow-up: Patched auth.mode.test.ts with a hoisted @web/lib/api mock so getAuthMode tests do not import real API base-url resolution. Required verification now passes without env workarounds: bunx vitest run __tests__/extension/runtime-bootstrap.test.ts __tests__/auth.mode.test.ts __tests__/auth.logout.test.ts. Bandit not run: test-only TypeScript harness change.

2026-06-24 Task 2 code-quality follow-up: Fixed runtime auth ownership so a manual stored key equal to the current runtime key does not become runtime-owned without prior metadata, and aligned persisted placeholder replacement with the runtime-config endpoint invalid-key policy. Verification: bunx vitest run __tests__/extension/runtime-bootstrap.test.ts passed; bunx vitest run __tests__/extension/runtime-bootstrap.test.ts __tests__/auth.mode.test.ts __tests__/auth.logout.test.ts passed.
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
