---
id: TASK-314
title: Implement Agent Tasks canonical workspace filter for ACP issue 1540
status: In Progress
assignee:
  - '@codex'
created_date: '2026-05-13 05:49'
updated_date: '2026-05-13 14:30'
labels:
  - ACP
  - workspace
  - frontend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1540'
  - 'https://github.com/rmusser01/tldw_server/pull/1625'
  - 'https://github.com/rmusser01/tldw_server/pull/1627'
documentation:
  - Docs/Design/ACP_Workspace_Integration_Decision_2026_05.md
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent Tasks can filter tasks by canonical workspace metadata without dropping existing project/task behavior
- [x] #2 Agent Tasks surfaces workspace-linked setup gaps for missing root/env/MCP readiness when the selected workspace has no usable ACP execution workspace context
- [x] #3 Focused frontend tests cover canonical workspace filter requests and setup-gap display behavior
- [x] #4 Verification includes focused Vitest, lint or targeted ESLint, git diff --check, and Bandit touched-scope rationale
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
## Stage 1: Contract And Red Tests
Goal: lock the Agent Tasks workspace-native behavior before implementation.
Success Criteria: tests fail for workspace query handoff, canonical workspace project filtering, and missing execution workspace setup guidance.
Tests: focused Vitest for AgentTasksPage plus WorkspaceHeader navigation handoff.
Status: Complete

## Stage 2: Agent Tasks Workspace Filter
Goal: derive workspace filter options from backend canonical_workspace metadata and from incoming route query params without changing existing project/task APIs.
Success Criteria: projects and selected tasks narrow to the chosen canonical workspace, unfiltered behavior remains unchanged, and selected project state is corrected when filters change.
Tests: focused Vitest verifies filtered project list and only the filtered project task endpoint is requested.
Status: Complete

## Stage 3: Workspace Setup Gap Surfacing
Goal: show actionable setup gaps when the selected canonical workspace has no usable ACP execution workspace context or a non-linked bridge status.
Success Criteria: Agent Tasks presents root/env/MCP readiness guidance and links users back to WorkspacePlayground/ACP setup surfaces before dispatch.
Tests: focused Vitest covers URL-provided workspace with no linked project and conflict/unlinked metadata.
Status: Complete

## Stage 4: Verification And Closeout
Goal: verify the focused slice and record the outcome for #1540 continuation.
Success Criteria: focused Vitest, targeted static check or documented lint fallback, git diff --check, and Bandit touched-scope rationale are recorded in TASK-314.
Tests: bunx vitest run targeted files; git diff --check; backend Bandit skip rationale if no Python touched.
Status: Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Continuation after merged PR #1625. Implements the next #1540 slice: Agent Tasks canonical workspace filter plus setup-gap surfacing, using the backend bridge metadata from PR #1615 and WorkspacePlayground handoff from PR #1625.

Red test run completed before production changes. Command: bunx vitest run src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx src/components/Option/WorkspacePlayground/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1 --no-file-parallelism. Expected failures: route workspace filter not applied, workspace setup gap alert absent, WorkspaceHeader navigates to /agent-tasks without workspace query.

Implementation notes: Agent Tasks now reads workspace/workspace_id/canonical_workspace_id from the route, derives canonical workspace filter options from project metadata, filters visible projects/tasks without changing existing backend project/task endpoints, and shows workspace-specific setup guidance when the selected canonical workspace lacks a usable ACP execution project or has a non-linked bridge status. WorkspacePlayground handoff now opens /agent-tasks with the current workspace query.

Verification: red run failed as expected before implementation. Final focused Vitest passed: bunx vitest run src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx src/components/Option/WorkspacePlayground/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1 --no-file-parallelism => 2 files, 38 tests passed. Targeted TypeScript pass used /private/tmp/acp-agent-tasks-tsconfig.json with touched files plus ambient/test globals. Full UI tsc remains blocked by existing unrelated baseline errors across many tests/services. git diff --check passed. Bandit not run because touched implementation is TypeScript/frontend docs/backlog only and no Python files are in scope.

Post-rebase verification on current origin/dev: focused Vitest passed again with 2 files and 38 tests; targeted TypeScript pass using /private/tmp/acp-agent-tasks-tsconfig.json passed; git diff --check origin/dev...HEAD passed.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1627. Kept draft because the AI-generated PR policy requires a human-owned Change summary before merge and #1540 has later history/diagnostic-link slices remaining.

PR #1627 review sweep: addressed 6 unresolved review threads. Changes: switched Agent Tasks workspace filter parsing to React Router location/navigation for MemoryRouter compatibility; synchronized manual workspace filter changes back to router search; suppressed workspace setup warnings unless project data loaded successfully; reused filteredProjects in workspace setup calculation; only reports unlinked bridge warning when no matching linked project exists; removed redundant active-workspace Tag. Added regression tests for MemoryRouter route updates, project-load failure diagnostics, and mixed linked/stale project state. Verification: focused Vitest now 2 files/40 tests passed; targeted touched-file TypeScript passed; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
