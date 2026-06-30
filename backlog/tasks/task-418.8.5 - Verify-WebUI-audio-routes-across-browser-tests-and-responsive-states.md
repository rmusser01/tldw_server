---
id: TASK-418.8.5
title: Verify WebUI audio routes across browser tests and responsive states
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-20 06:43
labels:
- ux
- webui
- extension
- audio
- verification
dependencies: []
documentation:
- Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md
parent_task_id: TASK-418.8
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/1890
modified_files:
- apps/packages/ui/src/components/Option/Audio/__tests__/audio-error-classification.test.ts
- apps/packages/ui/src/components/Option/Audio/audio-error-classification.ts
- apps/packages/ui/src/components/Option/AudiobookStudio/AudiobookStudioPage.tsx
- apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.audio-source.test.tsx
- apps/tldw-frontend/e2e/smoke/stage7-audio-regression.spec.ts
- apps/tldw-frontend/e2e/utils/page-objects/AudiobookStudioPage.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP11A Task 6 from the WebUI audio routes implementation plan. Verify /audio, /speech, /stt, /tts, and /audiobook-studio across route tests, focused component tests, Playwright workflows, and responsive browser observations; add only scoped verification assertions or minimal route-local fixes needed by observed failures, without backend API changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verified the root audio route group after PR #1887 merged. Added hermetic audio smoke stubs so browser tests cover /audio, /speech, /stt, /tts, and /audiobook-studio without background bootstrap noise; fixed the Audiobook Studio save-status live region so Playwright can observe the project save state; aligned the Audiobook Studio page object with the accessible Save button label; and added capture-busy audio error classification coverage preserving the active capture owner.

Verification recorded:
- PASS: bunx vitest run ../packages/ui/src/components/Option/Speech/__tests__ ../packages/ui/src/components/Option/STT/__tests__ ../packages/ui/src/components/Option/TTS/__tests__ ../packages/ui/src/components/Option/AudiobookStudio/__tests__ ../packages/ui/src/components/Option/Audio/__tests__/audio-error-classification.test.ts (18 files, 121 tests; rerun after rebase)
- PASS: bunx vitest run ../packages/ui/src/routes/__tests__/audio-route-jobs.test.ts ../packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx ../packages/ui/src/routes/__tests__/option-audio-hosted-message.test.tsx ../packages/ui/src/routes/__tests__/option-route-visibility.test.ts (4 files, 13 tests; rerun after rebase)
- PASS: git diff --check HEAD~1..HEAD
- PASS: TLDW_WEB_URL=http://localhost:18085 TLDW_WEB_CMD='bun run dev -- -p 18085' bunx playwright test e2e/smoke/stage7-audio-regression.spec.ts --reporter=line --workers=1 (6 passed)
- PASS: TLDW_WEB_URL=http://localhost:18087 TLDW_WEB_CMD='bun run dev -- -p 18087' bunx playwright test e2e/smoke/stage7-audio-regression.spec.ts e2e/workflows/tier-2-features/speech-playground.spec.ts e2e/workflows/tier-2-features/stt-transcription.spec.ts e2e/workflows/tier-2-features/tts-synthesis.spec.ts e2e/workflows/tier-2-features/audiobook-studio.spec.ts --reporter=line --workers=1 (17 passed, 3 skipped; rerun after rebase)
- PARTIAL: bunx tsc --noEmit --pretty false still fails on existing unrelated TypeScript debt outside this slice: MediaReadAlongPopover, Evaluations EmbeddingsModelSelectionConfig, WorkspacePlayground StudioPane, keyboard shortcut config, and tier-4 llama.cpp admin tests.
- SKIP: Bandit is not applicable because this slice only changes frontend TypeScript/Playwright files and a Backlog task record.
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
