## Stage 1: Capture And Group Diagnostics
**Goal**: Reproduce the `apps/packages/ui` TypeScript baseline failure after dependency restoration and group diagnostics by root cause.
**Success Criteria**: Full `tsc` failure is captured, grouped, and linked to `TASK-12099`.
**Tests**: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`
**Status**: Complete

## Stage 2: Fix Stale Test Contracts
**Goal**: Update stale test fixtures and mocks to match current component/service contracts without changing production behavior.
**Success Criteria**: Notes, Research Workspace, Setup, and Dexie test diagnostics are removed.
**Tests**: Focused Vitest files plus full `tsc`.
**Status**: Complete

## Stage 3: Fix Narrow Production Type Contracts
**Goal**: Resolve remaining production type mismatches with minimal source changes.
**Success Criteria**: AudioStudio, Skills Manager, Scheduled Tasks, MCP hub, voice cloning, and background diagnostics are removed.
**Tests**: Full `tsc`; focused tests where behavior changes.
**Status**: Complete

## Stage 4: Verify And Commit
**Goal**: Verify the cleanup and keep it separate from the Visual Identity implementation commits.
**Success Criteria**: Full `tsc` passes, focused Visual Identity frontend tests pass, Backlog task is updated, and only intended files are staged.
**Tests**: Full `tsc`, Visual Identity Vitest slice, `git diff --check`.
**Status**: Complete
