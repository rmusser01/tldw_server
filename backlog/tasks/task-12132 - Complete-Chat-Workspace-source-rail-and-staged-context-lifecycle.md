---
id: TASK-12132
title: Complete Chat Workspace source rail and staged-context lifecycle
status: Done
assignee: []
created_date: 2026-07-02 03:33
updated_date: 2026-07-03 19:52
labels:
- WebUI
- Front-End
- ChatWorkspace
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/2032
- https://github.com/rmusser01/tldw_server/issues/1239
- https://github.com/rmusser01/tldw_server/pull/2595
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #2032. Finish the /chat-workspace source rail and staged-context lifecycle so real workspace sources can be browsed, staged, unstaged, and sent without leaking state across workspace switches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Source rail is wired to real workspace sources for browse/open workflows.
- [x] #2 Add source and open library actions either work or route to the canonical source-management flow.
- [x] #3 Users can stage and unstage individual sources without clearing all context.
- [x] #4 Ready, processing, error, and unavailable states are rendered with clear actions and non-actionable states disabled.
- [x] #5 Empty states distinguish no workspace sources, filtered-out sources, and source loading/error states.
- [x] #6 Staged context sends ready media through structured media ids when possible.
- [x] #7 Staged context inserts a readable fallback summary when sources cannot be carried structurally.
- [x] #8 Switching workspace clears stale browsed/staged source state and cannot leak prior workspace context.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan file: Docs/superpowers/plans/2026-07-02-chat-workspace-source-rail-lifecycle-plan.md

Design review adjustments before implementation:
- Ready sources with invalid/missing mediaId remain stageable and use fallback summary text.
- Processing/error/unavailable sources cannot be staged, but already-staged sources can always be unstaged.
- Browse primes workspace source focus without implicit navigation; Add/Open actions route explicitly.
- Source loading/error states come from the workspace store.
- Individual unstage guards against stale prior-workspace callbacks.
- Duplicate source titles keep disambiguated accessible action names.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan after design review. Worktree-local tracker supersedes an accidentally-created base-checkout duplicate that was removed before behavior edits.

Implementation completed in the chat-workspace-live-flow worktree.

Touched #2032 files:
- apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceRail.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/ContextStagingCard.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspaceConsole.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/staging.ts
- apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/WorkspaceRail.test.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ContextStagingCard.test.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx
- apps/packages/ui/src/components/Option/ChatWorkspace/__tests__/staging.test.ts
- apps/tldw-frontend/e2e/smoke/chat-workspace-live-backend.spec.ts
- Docs/superpowers/plans/2026-07-02-chat-workspace-source-rail-lifecycle-plan.md

Verification:
- RED: bunx vitest run src/components/Option/ChatWorkspace/__tests__/staging.test.ts src/components/Option/ChatWorkspace/__tests__/WorkspaceRail.test.tsx src/components/Option/ChatWorkspace/__tests__/ContextStagingCard.test.tsx src/components/Option/ChatWorkspace/__tests__/ChatWorkspacePage.test.tsx failed for expected missing unstage/link/state behavior under apps/packages/ui.
- GREEN: same focused Vitest command passed: 4 files, 36 tests.
- Browser: TLDW_WEB_URL=http://localhost:18080 TLDW_WEB_CMD="bun run dev -- -p 18080" npx playwright test e2e/smoke/chat-workspace-live-backend.spec.ts --project=chromium passed: 4 tests. Escalation was required because sandbox blocked binding the fresh local dev server; fresh port was needed because 8080 had a stale reused server.
- Whitespace: git diff --check passed.
- Bandit: not applicable; #2032 touched TypeScript/TSX/docs/test files only, no Python.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed #2032 locally. Chat Workspace source rail uses real workspace source loading/error state, routes Add source/Open library to canonical source-management surfaces, primes workspace source focus on browse, supports individual unstage from both rail and staged context, renders ready/processing/error/unavailable source states, distinguishes loading/error/no-source/filter-empty states, preserves workspace-switch stale-callback guards, and keeps ready sources with invalid media ids stageable for fallback-summary sends.

Post-review update: internal source-management links now use React Router `Link` when a router context is present, with the existing anchor fallback preserved for isolated renders/tests.

Verification: focused Chat Workspace/Playground/hooks/settings Vitest passed 11 files / 127 tests; Playwright chat-workspace live-backend smoke passed 4 tests; Stage 5 Chat Workspace release gate passed 1 test; git diff --check passed. Bandit remains not applicable because no Python files were touched.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused component tests cover unstage and source state variants.
- [x] #8 At least one browser workflow covers stage, unstage, and send with staged context.
- [x] #9 git diff --check passes.
- [x] #10 Bandit is run on touched Python scope or explicitly marked not applicable.
<!-- DOD:END -->
