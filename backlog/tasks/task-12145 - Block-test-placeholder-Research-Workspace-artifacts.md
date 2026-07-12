---
id: TASK-12145
title: Block test-placeholder Research Workspace artifacts
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-04 20:40'
labels:
  - research-workspace
  - artifacts
  - testing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the literal test placeholder sentinel to Research Workspace generated artifact validation so outputs such as 'this is a test' cannot complete as generated quiz, flashcard, audio, table, slide, or mindmap artifacts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shared generated artifact placeholder guard rejects the literal 'this is a test' style sentinel.
- [x] #2 A focused Research Workspace generation test fails before the guard change and passes after it.
- [x] #3 Focused frontend verification and diff check are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added the exact 'this is a test' sentinel to the existing shared Research Workspace generated artifact placeholder set. The regression uses data table generation because it directly exercises the shared artifact finalizer used by the file/media-like outputs.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The literal 'this is a test' placeholder now fails closed through the shared Research Workspace artifact guard. Red verification: `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx -t "test-placeholder"` failed before the sentinel because the test table completed. Green verification: the same command passed after the one-line sentinel change. Broader verification: `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx ../packages/ui/src/services/__tests__/quiz-flashcards-handoff.test.ts ../packages/ui/src/services/__tests__/flashcards-generate-handoff.test.ts ../packages/ui/src/components/Quiz/__tests__/QuizPlayground.navigation.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/TakeQuizTab.list-controls.test.tsx` passed 7 files / 138 tests. `git diff --check` passed. Bandit skipped because only frontend TypeScript and Backlog task files changed.
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
