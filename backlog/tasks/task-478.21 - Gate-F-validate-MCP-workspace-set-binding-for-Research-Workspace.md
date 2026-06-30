---
id: TASK-478.21
title: 'Gate F: validate MCP workspace-set binding for Research Workspace'
status: Done
labels:
- research-workspace
- mcp
- shared-workspaces
- uat
- workspace-model
priority: High
milestone: Research Workspace UAT Remediation
ordinal: 21
parent_task_id: TASK-478
references:
- Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/superpowers/specs/2026-05-10-mcp-hub-workflow-first-control-panel-design.md
- TASK-478.7
- TASK-478.13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the explicit Research Workspace UAT matrix gap for live Shared Workspaces/MCP workspace-set binding. The canonical Research Workspace model says MCP Hub Shared Workspaces/workspace sets are the path/tool trust registry for agent and tool workflows, but the current matrix marks the live Research Workspace -> MCP workspace-set binding as Partial. Scope this slice to making the active Research Workspace's canonical workspace ID visible/usable in the MCP Hub workspace-set/path-trust workflow, validating the handoff live with backend + WebUI + CDP/Playwright, and updating the matrix honestly. Do not add /workspace-playground aliases or redirects, and do not duplicate Research Workspace source membership inside MCP Hub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Active Research Workspace can expose or navigate to the relevant MCP Hub workspace-set/path-trust state using the canonical workspace ID or a documented empty/not-configured state.
- [x] #2 MCP Hub Shared Workspaces/workspace-set UI or API distinguishes no binding, existing binding, and unavailable MCP service states with actionable copy and management links.
- [x] #3 No Research Workspace route, metadata, API, or tests reintroduce workspace-playground aliases, redirects, or active workspace_playground labels.
- [x] #4 Focused backend/frontend tests cover the binding/handoff contract or the explicit no-binding state.
- [x] #5 Live backend + WebUI validation via CDP/Playwright records evidence in the UAT matrix, moving RW-UAT-021 only as far as the evidence supports.
- [x] #6 Bandit is run for touched backend Python paths or explicitly skipped for frontend/docs-only scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Selected approach C by user decision: MCP Hub owns workspace-set/shared-workspace/path-trust state and management. Research Workspace should only carry the active canonical workspace context into MCP Hub via route/query state and contextual links. This slice should not add a Research Workspace-owned MCP binding projection unless implementation proves existing MCP Hub APIs cannot represent the state truthfully. Avoid duplicate research-source membership, new banners, and any workspace-playground aliases or redirects.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added the approach C design and implementation plan under `Docs/superpowers/`, keeping MCP Hub as the owner of workspace-set/shared-workspace/path-trust state.
- Updated Research Workspace MCP remediation links to deep-link to `/mcp-hub?workflow=workspaces&view=workspace-sets&workspace_id=<id>&source=research-workspace` using the canonical `capabilities.workspace_id`; no `/workspace-playground` alias, redirect, or label was introduced.
- Updated MCP Hub to parse the focused `workspace_id` query context and pass it into Workspace Sets. Workspace Sets derives existing/no-binding state from `listWorkspaceSetObjects()` and `listWorkspaceSetMembers()`, while API failures continue to show the existing load-error state.
- Added focused unit coverage for encoded workspace IDs, MCP Hub route hydration, existing workspace-set matches, no-binding state, and load-error suppression of the contextual callout.
- Added a live Playwright/CDP route assertion in `mcp-hub.spec.ts` and updated `RW-UAT-021` in the live matrix as `Partial`: the handoff/no-binding or existing-binding state is validated, but downstream policy/tool execution is not yet validated.
- Bandit skipped: no backend Python files were changed in this frontend/docs/e2e slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approach C is implemented. Research Workspace now hands canonical workspace context to MCP Hub without owning or duplicating MCP binding state. MCP Hub Workspace Sets shows whether the focused Research Workspace ID is already included in an MCP workspace set or not yet bound, and the live UAT matrix records the validated handoff while keeping full policy/tool execution as remaining `Partial` scope.

Verification:
- `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/WorkspaceCapabilityRemediation.test.tsx src/components/Option/MCPHub/__tests__/McpHubPage.test.tsx src/components/Option/MCPHub/__tests__/WorkspaceSetsTab.test.tsx --maxWorkers=1` passed: 20 tests.
- Live backend health on `http://127.0.0.1:18002/api/v1/health` returned 200 OK.
- `TLDW_WEB_URL=http://localhost:18080 TLDW_WEB_CMD='bun run dev -- -p 18080' TLDW_E2E_SERVER_URL=http://127.0.0.1:18002 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY npx playwright test e2e/workflows/tier-2-features/mcp-hub.spec.ts --project=tier-2 --grep "Research Workspace context" --reporter=line` passed: 1 test.
- Active-code route guard found no `workspace-playground`, `workspace_playground`, or `Workspace Playground` matches in the touched Research Workspace/MCP Hub surfaces; remaining matches are existing negative tests.
- `git diff --check` passed.
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
