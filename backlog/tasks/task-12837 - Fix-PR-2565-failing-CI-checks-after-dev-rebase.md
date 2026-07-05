---
id: TASK-12837
title: Fix PR 2565 failing CI checks after dev rebase
status: Done
labels:
- ci
- mcp
- pr-2565
priority: High
documentation:
- Docs/superpowers/plans/2026-07-01-pr2565-ci-fixes-after-dev-rebase.md
modified_files:
- Docs/superpowers/plans/2026-07-01-pr2565-ci-fixes-after-dev-rebase.md
- .github/workflows/frontend-ux-gates.yml
- apps/packages/ui/src/hooks/usePostOnboardingMediaReadiness.ts
- apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx
- apps/tldw-frontend/e2e/workflows/onboarding-ingestion-first.spec.ts
- tldw_Server_API/app/api/v1/endpoints/audio/audio_voices.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py
- tldw_Server_API/app/core/Monitoring/notification_service.py
- tldw_Server_API/app/core/Sandbox/service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase codex/mcp-docs-stage1 onto latest dev, reproduce actionable GitHub Actions failures from PR #2565, fix root causes with focused tests, run verification including Bandit for touched backend paths, then push the rebased branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-01-pr2565-ci-fixes-after-dev-rebase.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased codex/mcp-docs-stage1 onto origin/dev successfully with no conflicts. Reproduced/fixed actionable backend CI failures: audio preflight missing string import, Fish S2 provider error message preservation, Guardian notify_generic timestamp mutation, and sandbox queued-claim immediate renewal. Chatbooks and persona alias focused tests passed on rebased baseline, so no changes were needed there. Frontend fixes cover the standalone smoke start command and runtime single-user auth readiness for the first-source onboarding milestone. Added a Vitest regression for runtime auth and updated the onboarding Playwright spec to use shared auth seeding plus runtime/model stubs. Verification: focused backend pytest set passed 24 tests; focused Vitest set passed 99 tests across 6 files; onboarding Playwright passed 2 tests; Bandit reported 0 findings on touched backend files; git diff --check passed. Local bun run build:prod was inconclusive: next build stayed silent with stagnant CPU for several minutes, and after interruption token-sync failed against partial .next output.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2565 branch onto origin/dev and fixed the actionable CI failures found after the rebase. Backend fixes cover audio preflight title generation, Fish S2 provider error propagation, Guardian generic notification timestamp mutation, and sandbox queued-claim renewal timing. Frontend fixes cover the UX smoke standalone server command, runtime single-user auth readiness for first-source onboarding, and the onboarding evidence spec's auth/runtime/model stubs. Verification passed for focused backend pytest (24 tests), focused Vitest (99 tests across 6 files), onboarding Playwright (2 tests), Bandit (0 findings), and git diff --check. Local bun run build:prod was inconclusive because next build stalled and the interrupted partial .next caused token-sync to fail.
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
