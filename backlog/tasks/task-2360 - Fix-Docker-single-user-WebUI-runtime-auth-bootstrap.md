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
- apps/packages/ui/src/services/tldw/runtime-auth-override.ts
- apps/packages/ui/src/services/tldw/request-core.ts
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/tldw/TldwAuth.ts
- apps/tldw-frontend/pages/_app.tsx
- apps/tldw-frontend/__tests__/app/app-layout.test.tsx
- Dockerfiles/docker-compose.webui.yml
- Dockerfiles/docker-compose.single-user.yml
- apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts
- apps/tldw-frontend/__tests__/pr-916-review-followups.test.ts
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

2026-06-24 Task 2 runtime-auth preservation follow-up: Added regression coverage for manual multi-user tldwConfig with accessToken and updated runtime bootstrap persistence ownership so runtime API key still wins request precedence without overwriting stored multi-user credentials or writing runtime metadata. Red/green verification: bunx vitest run __tests__/extension/runtime-bootstrap.test.ts failed on the new regression before the fix and passed after the fix. Required verification passed: bunx vitest run __tests__/extension/runtime-bootstrap.test.ts __tests__/auth.mode.test.ts __tests__/auth.logout.test.ts (29 tests). Bandit not run: touched implementation is TypeScript frontend code only.

2026-06-24 Task 2 shared-client follow-up: added a non-persistent shared runtime single-user auth override consumed by runtime-bootstrap, request-core, TldwApiClient, and TldwAuth so runtime auth takes request precedence even when persisted manual single-user or multi-user credentials are preserved. Added runtime-bootstrap regressions for shared tldwRequest headers over manual single-user and multi-user configs. Verification: bunx vitest run __tests__/extension/runtime-bootstrap.test.ts __tests__/auth.mode.test.ts __tests__/auth.logout.test.ts ../packages/ui/src/services/tldw/__tests__/request-core.quickstart.test.ts ../packages/ui/src/services/tldw/__tests__/request-core.hosted.test.ts passed (34 tests); git diff --check clean.

2026-06-24 Task 3 update: `_app.tsx` now imports and awaits runtimeBootstrapReady before reading build-time env auth or persisted configured auth state, preventing the first-load app shell from resolving unauthenticated before runtime auth bootstrap completes. Added a delayed-bootstrap app-layout regression that failed before the production change and passed after it. Verification: bunx vitest run __tests__/app/app-layout.test.tsx passed (10 tests); bunx vitest run __tests__/app/app-layout.test.tsx __tests__/extension/runtime-bootstrap.test.ts __tests__/auth.mode.test.ts __tests__/auth.logout.test.ts ../packages/ui/src/services/tldw/__tests__/request-core.quickstart.test.ts ../packages/ui/src/services/tldw/__tests__/request-core.hosted.test.ts passed (44 tests). Bandit not run: touched implementation is TypeScript frontend code only.

2026-06-24 Task 4 update: Docker WebUI compose now passes runtime AUTH_MODE, SINGLE_USER_API_KEY, and TLDW_WEBUI_EXPOSE_RUNTIME_AUTH into the WebUI container while preserving the 127.0.0.1:8080:3000 host binding and quickstart internal API origin. Docker single-user compose now sets TLDW_SETUP_ALLOW_REMOTE=${TLDW_SETUP_ALLOW_REMOTE:-1} on the app service. Added compose assertions and aligned the PR-916 Dockerfile expectation with the current build:prod command. Red verification failed on missing compose env as expected; docker compose -f Dockerfiles/docker-compose.single-user.yml -f Dockerfiles/docker-compose.webui.yml config >/tmp/tldw_single_webui_runtime_auth_compose.yml succeeded; bunx vitest run __tests__/frontend-quickstart-networking.test.ts __tests__/pr-916-review-followups.test.ts passed (13 tests). Bandit not run: touched implementation is Docker YAML and TypeScript tests only.
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
