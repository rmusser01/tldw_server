---
id: TASK-318
title: Implement ACP workspace history and diagnostic links for issue 1540
status: Done
assignee: []
created_date: '2026-05-13 14:46'
updated_date: '2026-05-13 19:19'
labels:
  - ACP
  - workspace
  - frontend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1540'
  - 'https://github.com/rmusser01/tldw_server/pull/1643'
documentation:
  - Docs/Design/ACP_Workspace_Integration_Decision_2026_05.md
  - Docs/Plans/IMPLEMENTATION_PLAN_acp_run_history_1475.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Slice 4 from Docs/Design/ACP_Workspace_Integration_Decision_2026_05.md for issue 1540. WorkspacePlayground should make recent ACP runs discoverable from the current canonical workspace and route users to existing Agent Tasks and ACP Playground diagnostic/detail views without introducing a parallel ACP workspace browser or new backend data model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WorkspacePlayground exposes recent ACP run history for the current canonical workspace when linked Agent Tasks data is available
- [x] #2 Run history entries include task/project context, status, session id when available, and counts or previews from the existing enriched task-detail contract
- [x] #3 Users can navigate from a workspace history entry to Agent Tasks scoped to the current workspace and to ACP session detail/diagnostics/artifacts/audit views when those links are available
- [x] #4 The implementation handles loading, empty, unsupported orchestration, and backend-error states without showing misleading setup guidance
- [x] #5 Focused frontend tests cover successful history rendering, diagnostic navigation, empty state, and failure/unsupported behavior
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-13-acp-workspace-history-diagnostic-links-plan.md

Stages:
1. Add red WorkspaceHeader tests for recent ACP run history, diagnostic navigation, empty state, and backend-error behavior.
2. Implement a WorkspaceACPHistoryModal that reuses existing Agent Orchestration and ACP Playground contracts.
3. Wire Workspace settings and update docs/task evidence.
4. Run focused Vitest, targeted TypeScript, and git diff checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created isolated worktree .worktrees/acp-workspace-history-diagnostic-links-1540 from origin/dev after PR #1627 merged. TASK-318 covers #1540 Slice 4: workspace history and diagnostic links.

Implemented WorkspaceACPHistoryModal and WorkspaceHeader menu wiring for ACP run history. The modal reuses Agent Orchestration projects/tasks/task-detail data, filters by canonical workspace id, and links to Agent Tasks plus ACP Playground session/diagnostics/artifacts/audit views.

Verification: bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1 --no-file-parallelism passed 34 tests; bunx vitest run src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx --maxWorkers=1 --no-file-parallelism passed 10 tests; bunx tsc --noEmit -p /private/tmp/acp-workspace-history-tsconfig.json --pretty false exited 0; git diff --check exited 0.

Bandit: skipped because this slice changes TypeScript UI, tests, docs, and Backlog task metadata only; no Python backend files were touched.

Known skips/blockers: no Python files were touched, so Bandit was not run. Merge readiness still depends on the repository AI-generated PR policy requiring a human-authored Change summary before merge.

PR opened: https://github.com/rmusser01/tldw_server/pull/1643
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented #1540 Slice 4 by adding WorkspacePlayground ACP run history. The UI reuses existing Agent Orchestration project/task/task-detail responses, filters runs by canonical workspace id, and routes users to scoped Agent Tasks plus ACP Playground session diagnostics, artifacts, and audit views. Focused tests cover success, empty, backend-error, and unsupported endpoint states.
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
