---
id: TASK-418.9.3
title: Implement WP11B flashcards and quiz study mode clarity
status: Done
labels:
- ux
- webui
- extension
- wp11b
- study
- flashcards
- quiz
- testing
priority: High
parent_task_id: TASK-418.9
documentation:
- Docs/superpowers/plans/2026-05-17-webui-study-safety-specialized-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- apps/packages/ui/src/components/Flashcards/FlashcardsWorkspace.tsx
- apps/packages/ui/src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.queue-state.test.tsx
- apps/packages/ui/src/components/Quiz/QuizWorkspace.tsx
- apps/packages/ui/src/components/Quiz/QuizPlayground.tsx
- apps/packages/ui/src/components/Quiz/__tests__/QuizWorkspace.connection-state.test.tsx
- apps/packages/ui/src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx
- apps/packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx
- apps/packages/ui/src/components/Quiz/tabs/__tests__/ResultsTab.filters-retake.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP11B Task 3 from the WebUI study/safety/specialized implementation plan. Make `/flashcards` and `/quiz` clearly behave as study workspaces for first-time and returning users: preserve route identity and headings across connection/setup/demo/error states, expose the expected tab/mode set, keep first-use empty states productive without hiding expert tabs, and preserve existing sidepanel handoff and route ownership. Keep scope frontend-focused and avoid new backend APIs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Flashcards route keeps study workspace identity, heading, and expected modes visible across ready, empty, setup, unavailable, and sidepanel handoff states.
- [x] Quiz route keeps assessment workspace identity, heading, and expected modes visible across ready, demo, setup, unavailable, and empty states.
- [x] First-time states point users toward productive start actions without hiding expert tabs or repeated-use controls.
- [x] Unsupported/unconfigured/error states preserve route headings and recovery actions using existing shared state patterns.
- [x] Focused Flashcards, Quiz, route-boundary, design-system, and browser or E2E verification is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented route-owned study frames for Flashcards and Quiz so setup/demo/unavailable/ready states keep the route heading and expected study modes visible. Kept existing shared connection and feature-unavailable states inside the route frame. Made Quiz reset-current-tab control consistently visible before quizzes or attempts exist so first-time and power-user toolbar behavior stays stable. Updated focused unit tests and repaired stale test harness mocks for recent study sessions, router MemoryRouter, and remediation conversion hooks encountered by expanded verification.

Verification:
- bunx vitest run src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx src/components/Quiz/__tests__/QuizWorkspace.connection-state.test.tsx src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx src/components/Quiz/__tests__/quiz-ftux.test.tsx -> 67 passed.
- bunx vitest run src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx src/components/Flashcards/__tests__/FlashcardsManager.consistency.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.orientation.test.tsx src/components/Flashcards/tabs/__tests__/ReviewTab.queue-state.test.tsx src/components/Flashcards/tabs/__tests__/ImportExportTab.llm-gating.test.tsx -> 32 passed.
- bunx vitest run src/components/Quiz/__tests__/QuizWorkspace.connection-state.test.tsx src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx src/components/Quiz/__tests__/quiz-ftux.test.tsx src/components/Quiz/tabs/__tests__/TakeQuizTab.start-flow.test.tsx src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx src/components/Quiz/tabs/__tests__/CreateTab.save-progress.test.tsx src/components/Quiz/tabs/__tests__/ResultsTab.filters-retake.test.tsx -> 73 passed.
- bunx vitest run src/routes/__tests__/study-safety-specialized-route-boundaries.test.tsx -> 8 passed.
- bun run verify:design-system-state -> passed with 306 existing allowed baseline exceptions.
- TLDW_WEB_CMD='bun run dev -- -p 18080 -H localhost' TLDW_WEB_URL=http://localhost:18080 bunx playwright test e2e/workflows/tier-2-features/flashcards.spec.ts e2e/workflows/tier-2-features/quiz.spec.ts e2e/workflows/journeys/notes-flashcards.spec.ts --reporter=line -> 14 passed, 6 skipped.
- git diff --check -> passed.
- NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false -> failed on repo-wide pre-existing baseline type errors outside this slice; plain heap run OOMed first.
- Bandit not run: this slice touched only frontend TypeScript/TSX and Backlog.md files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added route-owned Flashcards and Quiz study workspace frames that preserve route headings and mode landmarks across setup, demo, unsupported, and ready states. Kept existing recovery actions and demo previews intact, made Quiz reset-current-tab consistently visible in the toolbar, and refreshed focused test harness mocks uncovered by expanded verification. Verification covered focused Flashcards/Quiz units, route-boundary tests, design-system state guard, Playwright browser specs, diff checks, and TypeScript baseline observation.
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
