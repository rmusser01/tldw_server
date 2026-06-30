---
id: TASK-592
title: Implement first-value starter questions after onboarding
status: Done
labels:
- onboarding
- webui
- uat
- first-value
priority: High
references:
- TASK-514
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR3 of the solo onboarding V2 roadmap: extend the onboarding UAT harness for first-source value behavior, then show safe starter questions only after first-source readiness is confirmed and route selected starters through the existing grounded chat/source workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Starter questions appear only after first-source media readiness is confirmed.
- [x] #2 Clicking a starter question dispatches the existing media chat handoff with the selected prompt and grounded media context.
- [x] #3 First-source idle, processing, error, and not-ready states do not show starter questions.
- [x] #4 Onboarding UAT includes a first-source starter-question scenario with screenshots, JSON summary, and diagnostics.
- [x] #5 Focused frontend tests and UAT verification are recorded; Bandit is run or explicitly skipped if no backend code is touched.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-02-first-value-starter-questions-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Branch/worktree: `codex/onboarding-diagnostics-recovery-clean` in `.worktrees/onboarding-uat-harness`.
- Baseline focused suite before RED passed: `bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx --reporter=dot` (2 files, 19 tests).
- RED coverage failed as expected after adding starter-question expectations: component prompt and route tests could not find `Starter questions`.
- Implemented fixed starter templates only for first-source ready state: `Summarize this source.`, `List the key claims.`, `What should I remember?`.
- Routed selected starter text through the existing `tldw:discuss-media` handoff as `content`, preserving `mode: "rag_media"` and the first-source media id/title.
- Extended UAT `first-source-after-chat` to process real pasted text through Quick Ingest, wait for a successful first-source media id, close the Quick Ingest modal, assert the starter buttons, and capture the selected starter handoff.
- GREEN focused suite after implementation passed: 2 files, 20 tests.
- Final focused suite passed: `bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx ../packages/ui/src/utils/__tests__/quick-ingest-open.test.ts ../packages/ui/src/hooks/__tests__/usePostOnboardingMediaReadiness.test.tsx --reporter=dot` (4 files, 28 tests).
- Final UAT passed: `bun run e2e:onboarding:uat -- --scenario first-source-after-chat --viewport desktop --mock-config hosted-success.json`.
- UAT artifacts: `apps/tldw-frontend/test-results/onboarding-uat/2026-06-02T21-07-43-211Z-c6kmj4/summary.json`, screenshots under the same run directory, and backend/frontend/mock logs under `logs/`.
- UAT handoff JSON recorded `mediaId: "1"`, `title: "pasted-text.txt"`, `mode: "rag_media"`, and `content: "Summarize this source."`.
- Bandit skipped: frontend TypeScript, Playwright UAT, Backlog, and plan files only; no backend Python files changed.
- Whitespace check passed: `git diff --check`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented first-value starter questions for the solo onboarding first-source milestone. The prompt now shows three fixed starter questions only after first-source readiness is confirmed, and the selected starter is dispatched through the existing grounded media chat handoff with the first-source media context. The real onboarding UAT harness now completes a pasted first-source ingest against the local backend/mock API stack, verifies the modal closes, captures screenshots/JSON/logs, and asserts the starter handoff payload.
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
