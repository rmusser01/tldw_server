---
id: TASK-12099
title: Clean up frontend TypeScript baseline blockers for Visual Identity branch finish
status: Done
references:
- TASK-12090.5
- codex/visual-identity-expression-packs
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the residual apps/packages/ui TypeScript diagnostics that remain after dependency restoration so the Visual Identity expression pack branch can be finished with a clean frontend typecheck. Keep changes scoped to diagnostics reported by `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` and avoid altering the Visual Identity implementation behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 apps/packages/ui full TypeScript check passes after dependency restoration.
- [x] #2 Focused Visual Identity frontend tests still pass.
- [x] #3 Touched frontend behavior is covered by existing or focused regression tests when behavior changes are required.
- [x] #4 Verification notes are recorded, including any diagnostics intentionally left out of scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented frontend TypeScript baseline cleanup after dependency restoration. Root causes: stale Notes/ResearchWorkspace/Setup/Dexie test fixtures, invalid React audio `referrerPolicy` prop despite required DOM attribute behavior, overly wide scheduled-task editor callback return types, typed list-param objects passed to a `Record<string, unknown>` query helper, AntD v6 row-checkbox excess-property typing for ARIA labels, an MCP hub path not cast through the existing helper, `Uint8Array.buffer` returning `ArrayBufferLike`, and `background.ts` upload response inference preserving a success-only union member. Verification: full `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed. Focused stale-test slice passed 22 tests. Focused touched production and Visual Identity slice passed 103 tests. `git diff --check` passed. Bandit not applicable: frontend TypeScript-only cleanup.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Cleaned up the frontend TypeScript baseline blockers that remained after restoring dependencies. The full `apps/packages/ui` typecheck now passes, and focused tests for stale fixtures, touched production modules, and Visual Identity frontend behavior pass. The cleanup keeps source changes narrow and separates this verification unblocker from the Visual Identity feature commits.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Backlog task has final summary and verification notes.
- [x] #8 No unrelated untracked files are staged.
<!-- DOD:END -->
