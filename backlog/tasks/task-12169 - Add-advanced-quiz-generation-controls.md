---
id: TASK-12169
title: Add advanced quiz generation controls
status: In Progress
references:
- Spec review approved by subagent 019f3dfe-9e1d-7762-8bf9-88a8e354e13f
- Final spec review approved by subagent 019f3e03-825a-7d80-aa7d-3c5c27de712f
- Plan review approved by subagent 019f3e20-d73f-70d2-8bf9-1074c1c4c5bc
modified_files:
- Docs/superpowers/specs/2026-07-07-advanced-quiz-generation-controls-design.md
- Docs/superpowers/plans/2026-07-07-advanced-quiz-generation-controls-implementation-plan.md
- apps/packages/ui/src/services/quizzes.ts
- apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx
- apps/packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the first-pass Advanced Quiz Studio controls for generated quizzes: exact per-type question counts and configurable MCQ option counts, including 5-option MCQs. Reuse the existing quiz generation endpoint, schemas, and WebUI quiz creation flow; do not add visual-question generation in this task.

Acceptance criteria:
- Quiz generation accepts a structured question plan or equivalent fields for per-type counts.
- MCQ generation supports at least 4-option and 5-option questions without truncating to four choices.
- Existing default quiz generation behavior remains backward compatible.
- WebUI exposes the new controls in the quiz creation flow.
- Tests cover 5-option MCQ generation/parsing and mixed per-type counts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-07-advanced-quiz-generation-controls-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 4 WebUI controls: added frontend question_plan type support, replaced GenerateTab's legacy question type/count controls with fixed five-row plan state, and submitted num_questions plus question_plan from enabled rows only.

Verification:
- RED: `bunx vitest run ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx` from `apps/tldw-frontend` failed with 6 missing-plan UI assertions.
- GREEN: `bunx vitest run ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.question-plan.test.tsx ../packages/ui/src/components/Quiz/tabs/__tests__/GenerateTab.media-selection.test.tsx` from `apps/tldw-frontend` passed 15 tests.
- Bandit skipped: frontend-only TypeScript/TSX changes.

Known caveat: the plain root-cwd Vitest command does not load the frontend alias/setup config in this worktree; tests were run from `apps/tldw-frontend`, matching existing frontend test config.
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
