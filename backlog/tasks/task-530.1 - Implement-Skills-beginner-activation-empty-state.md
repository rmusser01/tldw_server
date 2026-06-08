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

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the next scoped Skills UX slice: the active /skills manager now shows a visible Skills heading, a short workflow summary, and the current total count above the toolbar. The zero-skill table state now explains what Skills are and gives first actions to seed built-ins, open skill creation, or import text, while searched-empty states stay as simple no-match feedback.

Verification:
- `bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx --maxWorkers=1` passed 6 tests after red/green.
- `bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx src/components/Option/Skills/__tests__/SkillsWorkspace.test.tsx --maxWorkers=1` passed 12 tests.
- `git diff --check` exited 0.
- Browser smoke on `http://127.0.0.1:18002/skills` confirmed the Skills heading, workflow summary, `0 skills` count, and beginner empty-state actions in the accessibility snapshot. The global backend reachability modal appeared because no tldw server was running.
- Bandit skipped: frontend-only TypeScript/TSX slice.
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
