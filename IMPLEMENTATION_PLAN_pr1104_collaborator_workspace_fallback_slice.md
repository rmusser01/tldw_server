## Stage 1: Verify Review Finding
**Goal**: Confirm token-only collaborator entry can inherit a stale active owner workspace id.
**Success Criteria**: A focused frontend test fails when `PrototypeWorkspaceSessionView` receives the previous `activeWorkspaceId` for a share-token entry without a `workspace` query param.
**Tests**: Focused Vitest for `PrototypeWorkspacePage`.
**Status**: Complete

## Stage 2: Implement Collaborator-Safe Resolution
**Goal**: Prevent collaborator-entry flows from falling back to the owner active workspace id.
**Success Criteria**: Owner views still resolve `workspace ?? activeWorkspaceId`, while collaborator entries pass only explicit `workspace` or `null`.
**Tests**: Focused Vitest and frontend typecheck where practical.
**Status**: Complete

Notes:
- Regression test first failed with `Workspace:pw_stale_owner` for a token-only share entry.
- Implementation now uses `workspaceId ?? null` for collaborator entries and preserves `workspaceId ?? activeWorkspaceId` for owner entries.

## Stage 3: Verify and Publish
**Goal**: Run focused frontend tests, TypeScript check if available, diff checks, then push/reply to the relevant PR thread.
**Success Criteria**: Local verification passes and the addressed PR #1104 review thread has a reply.
**Tests**: Focused Vitest, TypeScript check, `git diff --check`.
**Status**: Complete

Notes:
- `bun run --cwd packages/ui test -- src/components/Option/PrototypeWorkspace/__tests__/PrototypeWorkspacePage.test.tsx` passed with 1 file and 4 tests.
- `git diff --check` passed.
- `apps/packages/ui` exposes `test` but no package-local `typecheck` script or `tsc` binary was available in this worktree.
- `bun install` in `apps` was interrupted after `node scripts/wxt-prepare.mjs` hung, but the partial install provided enough dependencies for focused Vitest.
- Backlog MCP was unavailable in this session; tracking task `TASK-61` was created with the CLI in the Backlog-enabled default checkout because this old PR worktree has no Backlog project files.
