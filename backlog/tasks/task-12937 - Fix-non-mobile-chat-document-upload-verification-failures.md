---
id: TASK-12937
title: Fix non-mobile chat document upload verification failures
status: Done
labels:
- bugfix
- frontend
- tests
priority: High
modified_files:
- apps/tldw-frontend/e2e/smoke/composer-preference-server-sync.spec.ts
- apps/tldw-frontend/e2e/smoke/composer-variants-preview.spec.ts
- apps/packages/ui/src/components/Option/AudioStudio/TimelineEditor.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskAutomationDefinitionEditor.tsx
- apps/packages/ui/src/components/Option/Skills/Manager.tsx
- apps/packages/ui/src/services/scheduled-tasks-control-plane.ts
- apps/packages/ui/src/services/tldw/openapi-guard.ts
- apps/packages/ui/src/services/tldw/voice-cloning.ts
- apps/tldw-frontend/e2e/fixtures/knowledge-qa-live.ts
- apps/tldw-frontend/e2e/workflows/tier-2-features/flashcards.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the non-mobile failures found after chat document upload processing work: composer server-sync smoke setup, preview-route notification/CORS noise, and TypeScript compile errors. Mobile UI failures are intentionally out of scope pending further UX/UI work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Composer server preference smoke test seeds auth/config and verifies the server-provided value.
- [x] #2 Composer variants preview smoke test isolates app-wide notification traffic without hiding preview route console/runtime errors.
- [x] #3 Reported non-mobile TypeScript compile errors are resolved.
- [x] #4 Mobile viewport composer UI specs remain out of scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Acceptance criteria for this narrow cleanup:
- Composer server preference smoke test seeds auth/config and verifies the server-provided value.
- Composer variants preview smoke test isolates app-wide notification traffic without hiding preview route console/runtime errors.
- TypeScript compile errors from the reported failure list are resolved.
- Mobile viewport composer UI specs remain unchanged and out of scope pending further UX/UI work.

Root cause summary before edits:
- Server sync mocked /api/v1/users/me/profile, but auth/config was not seeded, so the profile request short-circuited before the route mock could respond.
- The preview route mounts the global notification bridge; unauthenticated notification requests used credentials and hit backend CORS failures unrelated to the preview harness.
- The TypeScript failures were stale or overly narrow types: unsupported audio referrerPolicy prop, unknown save result assigned to a concrete response type, AntD checkbox aria attr typing, query param helper requiring Record, missing MCP readiness path, ArrayBufferLike copying, assertion function declaration style, and obsolete TestInfo.titlePath callable compatibility.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented minimal non-mobile fixes only. Reused the existing smoke seedAuth helper in the server-sync and preview specs; stubbed only notification endpoints in the preview route test; fixed the reported TypeScript errors with local type-alignment changes. Verification run:
- bunx tsc --noEmit --pretty false --project tsconfig.json: exit 0
- npx playwright test e2e/smoke/composer-preference-server-sync.spec.ts --reporter=line: 1 passed
- npx playwright test e2e/smoke/composer-variants-preview.spec.ts --reporter=line: 2 passed
- git diff --check: exit 0
Bandit skipped: frontend/test-only TypeScript changes, no Python touched. Known skip: mobile viewport composer UI specs intentionally not addressed.
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
