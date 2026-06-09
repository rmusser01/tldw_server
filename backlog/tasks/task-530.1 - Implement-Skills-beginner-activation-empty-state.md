---
id: TASK-530.1
title: Implement Skills beginner activation empty state
status: Done
labels:
- skills
- webui
- ux
priority: medium
parent_task_id: TASK-530
modified_files:
- apps/packages/ui/src/components/Option/Skills/Manager.tsx
- apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next scoped Skills UX slice from TASK-530: add a restrained /skills page summary with count and replace the zero-skill generic table state with a Skills-specific beginner activation state. Keep scope limited to Manager UI/tests unless a tiny support change is required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-08-skills-beginner-activation-and-guided-authoring.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Rebased codex/skills-beginner-empty-state on latest origin/dev.', 'Addressed PR #2319 review feedback: load failures now render an explicit Alert with retry instead of the beginner empty state; stale pagination clamps to the last valid page after totals shrink; non-empty libraries with empty current pages show a distinct table empty message; beginner empty-state import action now says Import from text to avoid duplicate Import labels.', 'Added regression coverage for load-error handling, stale pagination clamping, empty-current-page messaging, and duplicate Import label avoidance.', 'Verification: bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --maxWorkers=1 passed 9 tests; bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx src/components/Option/Skills/__tests__/SkillsWorkspace.test.tsx --maxWorkers=1 passed 15 tests; git diff --check passed; Bandit via repo .venv against apps/packages/ui/src/components/Option/Skills produced 0 findings / 0 LOC because touched scope is TypeScript.']
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2319 on latest dev and addressed review feedback. The Skills manager now shows an explicit load-error alert with retry instead of the beginner empty state, clamps stale pagination when totals shrink, distinguishes empty current pages from empty libraries, and gives the beginner import action a unique accessible label. Added regression tests for each review item and verified the focused Skills suites, diff check, and Bandit touched-scope report.
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
