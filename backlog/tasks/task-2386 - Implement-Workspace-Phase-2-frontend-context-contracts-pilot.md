---
id: TASK-2386
title: Implement Workspace Phase 2 frontend context contracts pilot
status: In Progress
labels:
- workspace
- phase2
- frontend
- acp
priority: High
references:
- https://github.com/rmusser01/tldw_server/issues/1993
- https://github.com/rmusser01/tldw_server/issues/1984
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track #1993 implementation for server-authoritative frontend Workspace context contracts, using Research Workspace and ACP Playground as the pilot surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents the server Workspace model as the authoritative frontend contract source.
- [x] #2 Shared frontend contract types/helpers normalize server workspace, membership, active context, eligibility, and recovery responses without inventing parallel semantics.
- [x] #3 Research Workspace pilot consumes the shared contract for active workspace context/recovery copy.
- [x] #4 ACP Playground pilot consumes the shared contract for session workspace state and mismatch/recovery copy.
- [x] #5 Tests prove global browse/list rendering is not filtered by active workspace context.
- [x] #6 Focused frontend tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- User clarification: goal is to unify on the server Workspace model. Client-local Research Workspace state can cache/hydrate and decorate, but server workspace identity, memberships, eligibility decisions, and recovery reason codes are authoritative.
- Keep #1993 scoped to frontend/client contracts plus Research Workspace and ACP Playground pilots. Do not build #1994 activity/index UI in this slice.
- Design spec: `Docs/superpowers/specs/2026-06-18-workspace-frontend-server-context-contract-design.md`.
- Implementation plan: `Docs/superpowers/plans/2026-06-18-workspace-frontend-server-context-contract.md`.
- Touched files:
  - `apps/packages/ui/src/services/workspace-context/contracts.ts`
  - `apps/packages/ui/src/services/workspace-context/normalizers.ts`
  - `apps/packages/ui/src/services/workspace-context/hooks.tsx`
  - `apps/packages/ui/src/services/workspace-context/index.ts`
  - `apps/packages/ui/src/services/workspace-context/__tests__/normalizers.test.ts`
  - `apps/packages/ui/src/services/workspace-context/__tests__/hooks.test.tsx`
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`
  - `apps/packages/ui/src/components/Option/ACPPlayground/ACPWorkspacePanel.tsx`
  - `apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPWorkspacePanel.test.tsx`
- Verification:
  - `./node_modules/.bin/vitest run src/services/workspace-context/__tests__/normalizers.test.ts src/services/workspace-context/__tests__/hooks.test.tsx src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx src/components/Option/ACPPlayground/__tests__/ACPWorkspacePanel.test.tsx --maxWorkers=1` passed: 4 files, 68 tests.
  - `git diff --check` passed.
- Visible UI evidence:
  - Testing Library render assertions cover the Research Workspace server-context indicator for ready, failed, archived, stable i18n status-key, and global-browser-list states, and cover ACP Workspace panel aligned, mismatch, no-session, active-only ID-chip, and recovery fallback-link states.
  - Browser attempt: `../node_modules/.bin/playwright test e2e/workflows/research-workspace.parity.spec.ts --project=chromium --reporter=line --workers=1` from `apps/tldw-frontend` failed before browser launch because the worktree frontend app has no local `next` binary (`next: command not found`). The main checkout has `next`, but its workspace `@tldw/ui` symlink resolves to the main checkout package, so it was not used as evidence for this PR worktree.
- Bandit: not applicable; this slice touched frontend TypeScript/tests/docs/Backlog only and no Python production paths.
- Typecheck: skipped because no `tsc` binary is exposed in this worktree or the linked frontend dependency roots; focused Vitest transpilation and behavior checks passed.
- Local test environment note: added ignored symlink `apps/node_modules -> /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/node_modules` so existing package symlinks resolve in this worktree. This is not tracked by git.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented a server-authoritative frontend Workspace context contract for #1993. Added shared normalizers, recovery copy, action eligibility helpers, and an active-context hook over existing server Workspace DTOs. Piloted the contract in Research Workspace and ACP Playground, including ACP session/active Workspace mismatch copy. Added tests for contract normalization, hook behavior, pilot rendering, and the guard that active server context does not filter the Workspace browser list.
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
