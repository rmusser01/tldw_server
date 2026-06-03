---
id: TASK-606
title: Thread Codex ACP sessions through workspace diagnostics
status: Done
labels:
- ACP
- Codex
- workspaces
- diagnostics
priority: High
documentation:
- Docs/superpowers/specs/2026-06-01-acp-codex-orchestration-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Stage 3 Codex ACP follow-up from the approved orchestration design. Scope: persist or expose workspace/worktree/MCP/sandbox/approval/adapter/certification context on Codex ACP sessions and diagnostics, make Research Workspace and Shared Workspaces link to the same session history/diagnostic surface, and validate a workspace-launched Codex ACP run path without introducing Codex app-server or generic runner-adapter support.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ACP session persistence preserves workspace/worktree-adjacent context needed for diagnostics, including workspace identifiers, MCP server evidence, runtime policy summary, and sandbox session/run IDs when available.
- [x] #2 `GET /api/v1/acp/sessions` supports a `workspace_id` filter and session detail/diagnostics expose a bounded workspace/runtime context envelope without leaking raw MCP env secrets or full local paths in diagnostics.
- [x] #3 Research Workspace ACP history can surface direct workspace-bound ACP sessions and link them to ACP Playground diagnostics/artifacts/audit views, not only Agent Tasks project runs.
- [x] #4 Codex `external_acp_adapter` session context includes adapter/runtime certification metadata when available, without introducing Codex app-server or generic runner-adapter support.
- [x] #5 Focused backend/frontend tests, Bandit for touched Python, and diff hygiene are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-02-acp-codex-workspace-diagnostics-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `sandbox_session_id` and `sandbox_run_id` persistence to ACP sessions with schema version 16 migration support.
- Extended ACP session list filtering with `workspace_id` using a static optional-filter SQL query.
- Added bounded `workspace_context` to ACP session list/detail and diagnostics responses; context includes workspace IDs, MCP server count/names, sandbox IDs, policy snapshot fields, and Codex adapter/certification metadata without exposing MCP env, args, commands, or cwd.
- Updated Research Workspace ACP run history to fetch direct workspace-bound ACP sessions in parallel with Agent Tasks history and link direct sessions to ACP Playground session, diagnostics, artifacts, and audit views.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-606 is complete. Codex ACP sessions are now workspace-filterable and carry bounded workspace/runtime context through backend persistence, API responses, diagnostics, and the Research Workspace history modal. Direct workspace-bound ACP sessions render alongside Agent Tasks runs without adding Codex app-server or generic runner-adapter support.

Verification:
- Backend: `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_session_store.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py -q` passed, 86 tests.
- Frontend: `./node_modules/.bin/vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx` passed, 40 tests.
- Bandit: `/tmp/bandit_task_606_acp_workspace_diagnostics.json` contains existing `ACP_Sessions_DB.py` baseline findings only; no new findings remain in the changed `list_sessions` query path.
- Diff hygiene: `git diff --check` passed.
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
