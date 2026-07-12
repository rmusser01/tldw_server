---
id: TASK-12144
title: Validate generated Research Workspace media artifacts
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 20:32'
labels:
  - research-workspace
  - testing
  - artifacts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the slides fail-closed artifact validation pattern to other Research Workspace file/media creation outputs so placeholder text or invalid sentinel content cannot complete as a generated artifact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Research Workspace generated file/media artifacts fail instead of completing when output is empty, invalid, or placeholder-only.
- [x] #2 Tests cover the affected artifact types through the existing Research Workspace generation flow.
- [x] #3 Verification includes focused frontend tests, diff check, and security scan for touched backend code if any backend files change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the shared placeholder guard in the existing StudioPane artifact generation path. Quiz/flashcard placeholders are rejected before persisted study-material records are created; audio placeholder scripts are rejected before TTS and empty TTS buffers are rejected; data tables must parse and contain at least one non-placeholder data cell.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extended Research Workspace artifact validation beyond slides for file/media-like outputs. Added failing-then-passing Stage 2 tests for placeholder-only quiz questions, flashcards, audio scripts, and data table cells. Verification: focused red run failed on the four new cases; after implementation `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx ../packages/ui/src/services/__tests__/quiz-flashcards-handoff.test.ts ../packages/ui/src/services/__tests__/flashcards-generate-handoff.test.ts ../packages/ui/src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.list-controls.test.tsx` passed 7 files / 137 tests. `git diff --check` passed. Bandit skipped because this slice touches frontend TypeScript and Backlog/plan docs only; no backend Python changed. Known unrelated untracked watchlist template files remain untouched.
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
