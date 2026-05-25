---
id: TASK-506
title: Resolve UI tsc errors with tests included
status: Done
labels:
- frontend
- typescript
- testing
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the remaining apps/packages/ui TypeScript errors reported by the main tsconfig while keeping tests included in src/**/*. Work in small batches across production typings and test fixture/mocking drift until NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false passes from apps/packages/ui.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Main apps/packages/ui tsconfig continues to include test files under src/**/*; no tsconfig exclusion workaround is used.
- [x] #2 NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false passes from apps/packages/ui.
- [x] #3 Production TypeScript errors in ResearchWorkspace, Sources, Watchlists, Integrations, and TldwModels are resolved through local type/model fixes.
- [x] #4 Test TypeScript errors are resolved by updating fixtures, helpers, and mocks to the current production contracts.
- [x] #5 git diff --check passes and verification notes are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started from the TASK-505 post-syntax-fix baseline where full UI tsc reported 239 errors with tests included. Fixed source-level typing drift in ResearchWorkspace/Sources/Watchlists/Integrations/Writing/route metadata/TldwModels and updated test fixtures/mocks to current contracts across chat, prompt, watchlists, persona, web clipper, workspace, dictation, and Tldw API client tests. Verified apps/packages/ui/tsconfig.json still has include: ["src/**/*"]. Verification passed: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false from apps/packages/ui, and git diff --check from the worktree root. Bandit skipped because this task touched TypeScript/TSX tests, UI source, docs, and backlog only; no Python touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the main apps/packages/ui TypeScript check while keeping tests in the main tsconfig. Production drift and test fixture/mock drift are now aligned with current contracts, and verification passed with heap-sized tsc plus git diff --check.
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
