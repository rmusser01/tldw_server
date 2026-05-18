---
id: TASK-433
title: Add WebUI setup readiness client hook
status: Done
labels:
- implementation
- setup
- frontend
- webui
documentation:
- Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
- Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md
modified_files:
- apps/packages/ui/src/services/tldw/setup-readiness.ts
- apps/packages/ui/src/components/Option/Setup/hooks/useSetupReadiness.ts
- apps/packages/ui/src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx
- apps/packages/ui/src/services/tldw/openapi-guard.ts
- apps/packages/ui/src/services/tldw/index.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the sixth slice from Docs/superpowers/plans/2026-05-18-first-time-readiness-setup-implementation-plan.md: a shared WebUI setup readiness API client, polling hook, and OpenAPI guard entries for the first-run/admin setup readiness endpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Shared WebUI setup readiness client wraps first-run and admin profiles/status/preview/provision/verify endpoints.
- [x] React hook loads profiles and status without provisioning, maps first-run guard failures to `/setup` fallback, and uses admin endpoints in admin mode.
- [x] Preview and provisioning remain separate explicit actions; provisioning sends `confirmed=true` only from the provision action.
- [x] OpenAPI `ClientPath` allowlist includes all first-run and admin setup readiness endpoints.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing hook tests for first-run load, guard fallback, admin mode, and preview/provision separation.
2. Add typed setup readiness service client and useSetupReadiness hook.
3. Add setup readiness paths to the OpenAPI guard and export the client surface.
4. Run focused frontend/backend verification and record inherited TypeScript baseline debt separately.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Implemented `apps/packages/ui/src/services/tldw/setup-readiness.ts` with typed first-run/admin path selection and response-envelope error handling.
- Implemented `apps/packages/ui/src/components/Option/Setup/hooks/useSetupReadiness.ts` with parallel profiles/status load, guard mapping, polling while provisioning, preview/provision/verify actions, and `/setup` fallback metadata.
- Added hook coverage in `apps/packages/ui/src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx`, including explicit separation between preview and provision.
- Updated `apps/packages/ui/src/services/tldw/openapi-guard.ts` and `apps/packages/ui/src/services/tldw/index.ts`.
- Verification: `bunx vitest run src/components/Option/Setup/hooks/__tests__/useSetupReadiness.test.tsx` -> 4 passed.
- Verification: `bun run verify:openapi` -> passed; verifier reported only the existing reviewed OSS exception paths.
- Verification: backend setup readiness pytest slice -> 26 passed.
- Known skip: `bunx tsc --noEmit --pretty false` still fails on inherited UI TypeScript debt outside the setup readiness files; no reported errors referenced this task's new files.
- Bandit: skipped for this frontend-only task slice; prior backend setup readiness Bandit checks remain recorded in the plan.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the WebUI setup readiness client/hook slice. The hook supports first-run and admin endpoint modes, loads readiness profiles/status without provisioning, maps setup guard failures to the backend /setup fallback, keeps preview and Provision Now separate, and polls active provisioning status.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
