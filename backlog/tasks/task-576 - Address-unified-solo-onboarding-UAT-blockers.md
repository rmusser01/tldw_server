---
id: TASK-576
title: Address unified solo onboarding UAT blockers
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-31 23:54
labels:
- onboarding
- uat
- webui
- quick-ingest
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md
- Docs/superpowers/plans/2026-05-31-unified-solo-onboarding-uat-repair-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-05-31-unified-solo-onboarding-uat-repair-design.md
- Docs/superpowers/plans/2026-05-31-unified-solo-onboarding-uat-repair-implementation-plan.md
- Makefile
- Dockerfiles/docker-compose.webui.yml
- apps/tldw-frontend/pages/_app.tsx
- apps/tldw-frontend/next.config.mjs
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
['Stage 0 cleanup/preflight: completed', 'Stage 1 first-run route repair: completed', 'Stage 2 quickstart WebUI auth handoff: completed', 'Stage 3 post-onboarding readiness gate: completed', 'Stage 4 first-source Quick Ingest defaults: completed', 'Stage 5 web/text ingest routing repair: completed', 'Stage 6 focused regression verification: completed', 'Stage 7 real UAT walkthrough: completed']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Stage 6 verification passed: apps/tldw-frontend vitest suite for frontend-quickstart-networking, app-layout, runtime-bootstrap: 35 tests passed.', 'Stage 6 verification passed: apps/packages/ui vitest suite for option-index unified setup, UnifiedSetupWizard, quick-ingest-batch: 43 tests passed.', 'Additional touched UI regressions passed: tldw-api-client.quickstart-auth, quick-ingest-open, QuickIngestButton.resume, AddContentStep.url-detection: 19 tests passed.', 'git diff --check passed.', 'Bandit not applicable before Stage 7: touched scope was TS/React/Makefile/docs only; no backend Python files touched.', 'Stage 7 UAT found an additional backend blocker: local Next quickstart rewrites send x-forwarded-host/proto without a client IP header, and the setup guard rejected first-run state/metadata as remote.', 'Added regression coverage in test_setup_guard for local rewrite metadata without forwarded IP (RED) and remote spoof protection; implemented guard handling and verified 6 setup guard tests passed.', 'Bandit passed for tldw_Server_API/app/api/v1/API_Deps/setup_deps.py with no findings.', 'Stage 7 UAT reached and passed the first-chat completion gate with OpenAI; after redirect to Companion, post-onboarding readiness hit a quickstart rewrite/CORS blocker on GET /api/v1/media?results_per_page=1 because the backend redirected to the internal API origin.', 'Added Next quickstart rewrite coverage for /api/v1/media and implemented an internal /api/v1/media -> /api/v1/media/ rewrite; frontend-quickstart-networking test now passes 11 tests.', 'Real UAT walkthrough completed on ports 18101/18102 using the existing project OpenAI key, pocket-tts, and onnx-parakeet. The wizard flowed through solo/local install, privacy/security, OpenAI provider, ingest defaults, audio, optional advanced, and successful first chat.', 'Post-onboarding first-source milestone completed: uploaded tldw-uat-onboarding-source.md through Quick Ingest, backend processed one markdown document, created one chunk, persisted media ID 1, media search returned the source, and RAG generated the answer containing cobalt pine 731.', 'Final focused verification passed: pytest tldw_Server_API/tests/integration/test_setup_guard.py -q: 6 passed; bunx vitest run __tests__/frontend-quickstart-networking.test.ts --reporter=default: 11 passed; Bandit setup_deps.py: 0 findings; git diff --check: clean.', 'Browser note: the in-app browser later landed on a Chromium localhost error page for /media and Browser Use blocked navigation back to localhost from that internal data URL. Remaining value validation was completed through backend API evidence and server logs rather than bypassing that browser policy.']
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the first-time solo onboarding blockers found during real UAT. The WebUI now routes generic first-run users into the unified setup, quickstart mode carries the single-user API key into the frontend, post-onboarding readiness waits for authenticated backend state, Quick Ingest defaults to a low-friction document/web-source flow, local setup rewrites are accepted by the backend guard without weakening remote spoof protection, and quickstart media readiness stays same-origin to avoid backend redirect/CORS failures. Real UAT completed with OpenAI first chat, pocket-tts/onnx-parakeet audio choices, markdown source ingest, media search retrieval, and RAG answer generation from the ingested source.
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
