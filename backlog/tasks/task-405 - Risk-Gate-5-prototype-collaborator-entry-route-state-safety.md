---
id: TASK-405
title: Risk Gate 5 prototype collaborator entry route-state safety
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-16 06:47'
labels:
  - prototype-workspaces
  - risk-gate
  - frontend
  - product
dependencies:
  - TASK-324
  - TASK-399
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1457'
  - 'https://github.com/rmusser01/tldw_server/issues/1440'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1457 under tracker #1440. Harden the Frontend/Product collaborator public-share entry path so prototype collaborator sessions use explicit exchanged session context, never fall back to stale owner workspace state, map frozen backend contract error categories to user-facing route states, and add focused frontend coverage for public-link exchange and route-state isolation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frontend tests cover public link exchange and route-state isolation.
- [x] #2 Collaborator session context is explicit and is not inferred from stale local owner state.
- [x] #3 User-facing states map to backend error categories from the contract matrix.
- [x] #4 The slice records any skipped browser-observed public-share handoff proof for Risk Gate 8 if not practical now.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-prototype-risk-gate-5-collaborator-entry-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-05-16: Created isolated worktree .worktrees/prototype-risk-gate-5-collaborator-entry on branch codex/prototype-risk-gate-5-collaborator-entry from origin/dev a304f5f58 after Risk Gate 4 PR #1739 merged.

2026-05-16: Baseline focused frontend verification passed: bunx vitest run src/components/Option/__tests__/PublicShare.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/hooks/__tests__/usePrototypeWorkspaces.test.tsx src/hooks/__tests__/useSharing.auth.test.tsx --maxWorkers=1 --no-file-parallelism reported 4 files and 17 tests passed. Initial run failed before bun install because the fresh worktree lacked workspace node_modules; bun install in apps restored dependencies without tracked lockfile changes.

2026-05-16: Added staged implementation plan at Docs/superpowers/plans/2026-05-16-prototype-risk-gate-5-collaborator-entry-plan.md.

2026-05-16: Added route-scoped collaborator entry handling. Share/session URL entries no longer use stale activeWorkspaceId or mismatched stored collaborator session state, and successful collaborator branch-session creation replaces token-bearing route state with the resolved workspace URL.

2026-05-16: Added structured API error preservation for sharing/prototype workspace helpers and mapped frozen prototype contract frontend states into collaborator-entry UI messages with retryability.

2026-05-16: Focused verification passed: bunx vitest run src/components/Option/__tests__/PublicShare.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspaceSessionView.test.tsx src/hooks/__tests__/usePrototypeWorkspaces.test.tsx src/hooks/__tests__/useSharing.auth.test.tsx --maxWorkers=1 --no-file-parallelism reported 5 files and 23 tests passed. git diff --check passed.

2026-05-16: Package typecheck command ./node_modules/.bin/tsc --noEmit -p tsconfig.json was attempted and failed on existing unrelated repo-wide TypeScript baseline errors outside this touched slice, including audio/composer/flashcards/playground/study-suggestions tests and existing domain exports. Bandit was not run because this slice touched frontend TypeScript/TSX and documentation only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

2026-05-16: Rebased codex/prototype-risk-gate-5-collaborator-entry onto latest origin/dev 41c27e6af. Post-rebase focused verification passed with 5 files and 23 tests; git diff --check HEAD~1..HEAD passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Risk Gate 5 frontend/product hardening for prototype collaborator entry. Token-bearing collaborator routes now avoid stale owner workspace/session fallback, successful branch-session creation clears tokenized route state, structured API error details are preserved, and collaborator-entry UI maps frozen backend frontend_state/category/retryable fields to stable user-facing states. Added focused component and hook regression coverage; browser/E2E proof remains a later release-gate item.
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
