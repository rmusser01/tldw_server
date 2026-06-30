## Stage 1: Contract And Red Tests
**Goal**: Lock the Agent Tasks workspace-native behavior before implementation.
**Success Criteria**: Tests fail for workspace query handoff, canonical workspace project filtering, and missing execution workspace setup guidance.
**Tests**: Focused Vitest for `AgentTasksPage` plus `WorkspaceHeader` navigation handoff.
**Status**: Complete

## Stage 2: Agent Tasks Workspace Filter
**Goal**: Derive workspace filter options from backend `canonical_workspace` metadata and incoming route query params without changing existing project/task APIs.
**Success Criteria**: Projects and selected tasks narrow to the chosen canonical workspace, unfiltered behavior remains unchanged, and selected project state is corrected when filters change.
**Tests**: Focused Vitest verifies filtered project list and only the filtered project task endpoint is requested.
**Status**: Complete

## Stage 3: Workspace Setup Gap Surfacing
**Goal**: Show actionable setup gaps when the selected canonical workspace has no usable ACP execution workspace context or a non-linked bridge status.
**Success Criteria**: Agent Tasks presents root/env/MCP readiness guidance and links users back to WorkspacePlayground/ACP setup surfaces before dispatch.
**Tests**: Focused Vitest covers URL-provided workspace with no linked project and conflict/unlinked metadata.
**Status**: Complete

## Stage 4: Verification And Closeout
**Goal**: Verify the focused slice and record the outcome for #1540 continuation.
**Success Criteria**: Focused Vitest, targeted static check or documented lint fallback, `git diff --check`, and Bandit touched-scope rationale are recorded in `TASK-314`.
**Tests**: `bunx vitest run` targeted files, `git diff --check`, and backend Bandit skip rationale if no Python is touched.
**Status**: Complete
