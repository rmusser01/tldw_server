---
id: TASK-311
title: Implement WorkspacePlayground ACP task handoff
status: Done
assignee:
  - '@codex'
created_date: '2026-05-13 02:10'
updated_date: '2026-05-13 04:52'
labels:
  - ACP
  - workspace
  - frontend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1540'
  - 'https://github.com/rmusser01/tldw_server/pull/1615'
  - 'https://github.com/rmusser01/tldw_server/pull/1625'
  - 'https://github.com/rmusser01/tldw_server/issues/1512'
  - 'https://github.com/rmusser01/tldw_server/issues/1513'
documentation:
  - Docs/Design/ACP_Workspace_Integration_Decision_2026_05.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Slice 2 from Docs/Design/ACP_Workspace_Integration_Decision_2026_05.md for issue 1540. WorkspacePlayground should let a user start an ACP agent task from the current canonical workspace without inventing a parallel ACP workspace model. The flow must use the backend canonical bridge endpoint from PR 1615 to find or create the ACP execution workspace before creating the AgentProject and AgentTask.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WorkspacePlayground exposes a create-agent-task action for the current canonical workspace
- [x] #2 The flow calls the canonical bridge endpoint and uses the returned ACP execution workspace ID when creating the agent project/task
- [x] #3 Created project/task state exposes the canonical workspace link and gives the user a clear path to Agent Tasks or ACP diagnostics
- [x] #4 Missing root/allowlist/setup failures are surfaced as actionable setup gaps rather than silent task creation failures
- [x] #5 Focused frontend tests cover successful handoff and bridge/setup failure behavior
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented WorkspacePlayground ACP handoff slice in isolated worktree .worktrees/acp-workspace-task-handoff-1540. Added Workspace settings action and modal that calls canonical bridge, creates an AgentProject bound to returned ACP workspace ID, then creates the AgentTask with workspace metadata.

Verification: bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkspaceHeader.test.tsx src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx passed from apps/packages/ui. git diff --check passed. Bandit skipped/documented because touched source is TypeScript/TSX frontend only.

PR #1625 review-fix pass reopened this task in the branch worktree. Scope: resolve still-actionable Gemini, CodeRabbit, and Qodo review threads; preserve Ant Design v6 title/destroyOnHidden props where local tests and guardrails prove those are the current API; add regression tests for config loading, rollback, and in-flight cancellation guards.

Review fixes now also document that WorkspacePlayground ACP history/retention must coordinate with #1512 and #1513 rather than creating a separate retention/redaction path.

Review-fix verification: focused Vitest passed for WorkspaceHeader, AgentTasks connection, and AntD modal prop guards (40 tests). bun run lint from apps/tldw-frontend exited 0 with existing warnings only. Targeted ESLint on changed package files exited 0 with warnings only. git diff --check passed. Bandit on the touched frontend path wrote /tmp/bandit_task311_pr1625_review_frontend_path.json with zero findings and loc=0 because no Python files were touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PR #1625 review fixes for WorkspacePlayground ACP task handoff. The modal now awaits async pre-submit persistence, respects connection-config loading before enabling task creation, rolls back an ACP AgentProject if downstream AgentTask creation fails, and blocks cancel/close/keyboard dismissal while submission is in flight. The touched WorkspaceHeader modal now uses Ant Design v6 destroyOnHidden, and the ACP workspace decision doc explicitly ties retention/redaction behavior to #1512 and #1513. Verification passed for the focused Vitest set, frontend lint, targeted ESLint, git diff --check, and Bandit frontend-path scan.
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
