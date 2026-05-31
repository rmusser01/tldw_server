---
id: TASK-576
title: Address unified solo onboarding UAT blockers
status: In Progress
labels:
- onboarding
- uat
- webui
- quick-ingest
documentation:
- Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md
- Docs/superpowers/plans/2026-05-31-unified-solo-onboarding-uat-repair-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md
- Docs/superpowers/plans/2026-05-31-unified-solo-onboarding-uat-repair-implementation-plan.md
- Makefile
- Dockerfiles/docker-compose.webui.yml
- apps/tldw-frontend/pages/_app.tsx
- apps/tldw-frontend/__tests__/app/app-layout.test.tsx
- apps/tldw-frontend/__tests__/frontend-quickstart-networking.test.ts
- apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts
- apps/packages/ui/src/routes/option-index.tsx
- apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx
- apps/packages/ui/src/hooks/usePostOnboardingMediaReadiness.ts
- apps/packages/ui/src/components/Option/Onboarding/PostSetupApiRecovery.tsx
- apps/packages/ui/src/utils/quick-ingest-open.ts
- apps/packages/ui/src/utils/__tests__/quick-ingest-open.test.ts
- apps/packages/ui/src/components/Common/QuickIngest/presets.ts
- apps/packages/ui/src/components/Layouts/__tests__/QuickIngestButton.resume.test.tsx
- apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts
- apps/packages/ui/src/services/tldw/quick-ingest-batch.ts
- apps/packages/ui/src/services/__tests__/quick-ingest-batch.test.ts
- tldw_Server_API/app/api/v1/API_Deps/setup_deps.py
- tldw_Server_API/tests/integration/test_setup_guard.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Plan and implement repairs required for a clean first-time solo-user walkthrough: root setup entry, WebUI auth handoff, first chat completion, first-source ingest, and fresh-install UAT verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Stage 0 cleanup/preflight: completed', 'Stage 1 first-run route repair: completed', 'Stage 2 quickstart WebUI auth handoff: completed', 'Stage 3 post-onboarding readiness gate: completed', 'Stage 4 first-source Quick Ingest defaults: completed', 'Stage 5 web/text ingest routing repair: completed', 'Stage 6 focused regression verification: completed', 'Stage 7 real UAT walkthrough: in progress']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Stage 6 verification passed: apps/tldw-frontend vitest suite for frontend-quickstart-networking, app-layout, runtime-bootstrap: 35 tests passed.', 'Stage 6 verification passed: apps/packages/ui vitest suite for option-index unified setup, UnifiedSetupWizard, quick-ingest-batch: 43 tests passed.', 'Additional touched UI regressions passed: tldw-api-client.quickstart-auth, quick-ingest-open, QuickIngestButton.resume, AddContentStep.url-detection: 19 tests passed.', 'git diff --check passed.', 'Bandit not applicable before Stage 7: touched scope was TS/React/Makefile/docs only; no backend Python files touched.', 'Stage 7 UAT found an additional backend blocker: local Next quickstart rewrites send x-forwarded-host/proto without a client IP header, and the setup guard rejected first-run state/metadata as remote.', 'Added regression coverage in test_setup_guard for local rewrite metadata without forwarded IP (RED) and remote spoof protection; implemented guard handling and verified 6 setup guard tests passed.']
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
