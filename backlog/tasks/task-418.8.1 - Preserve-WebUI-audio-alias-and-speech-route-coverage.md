---
id: TASK-418.8.1
title: Preserve WebUI audio alias and speech route coverage
status: Done
labels:
- webui
- ux-audit
- audio
- wp11a
- e2e
priority: medium
parent_task_id: TASK-418.8
documentation:
- Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md
modified_files:
- apps/tldw-frontend/e2e/workflows/tier-2-features/audio-alias.spec.ts
- apps/tldw-frontend/e2e/workflows/tier-2-features/speech-playground.spec.ts
- backlog/tasks/task-418.8.1 - Preserve-WebUI-audio-alias-and-speech-route-coverage.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP11A Task 2 from the WebUI audio routes implementation plan. Preserve /audio as a UI-free alias to /speech and add focused browser coverage for the /speech first-screen route identity, combined workflow modes, recording/source readiness, provider readiness, and output/history state without changing backend APIs or redesigning the audio UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /audio has focused E2E coverage proving it lands on the canonical /speech route and renders the speech playground.
- [x] #2 /audio remains UI-free unless browser coverage exposes a real alias failure.
- [x] #3 /speech E2E coverage verifies first-screen identity, combined workflow modes, source or recording readiness, TTS provider readiness near synthesis controls, and generated audio history or empty output state.
- [x] #4 Focused Playwright verification and git diff checks are recorded in the task final summary.
- [x] #5 No backend APIs or unrelated routes are changed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added focused Tier 2 Playwright coverage for /audio as a UI-free alias to /speech. Extended /speech page-load coverage to assert first-screen route identity, combined workflow modes, input source readiness, TTS readiness status, and empty speech history state. No production UI, backend API, or unrelated route files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation: added e2e/workflows/tier-2-features/audio-alias.spec.ts and extended speech-playground.spec.ts page-load assertions. Verification: bunx playwright test e2e/workflows/tier-2-features/audio-alias.spec.ts e2e/workflows/tier-2-features/speech-playground.spec.ts --reporter=line (3 passed, 1 skipped by existing TTS API guard); git diff --check (passed). Bandit skipped: touched files are frontend Playwright/Backlog markdown only.
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
