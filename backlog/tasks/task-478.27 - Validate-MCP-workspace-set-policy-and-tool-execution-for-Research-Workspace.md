---
id: TASK-478.27
title: Validate MCP workspace-set policy and tool execution for Research Workspace
status: Done
labels:
- research-workspace
- mcp
- shared-workspaces
- uat
priority: High
milestone: Research Workspace UAT Remediation
ordinal: 27
parent_task_id: TASK-478
references:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md
- TASK-478.21
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the remaining RW-UAT-021 Partial gap with a fixture-backed live validation that creates or binds an MCP Hub workspace set for the active Research Workspace ID, exercises downstream policy/tool availability, and records honest matrix evidence. Preserve MCP Hub as the canonical owner of tool/path trust and do not duplicate MCP policy state inside Research Workspace.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Live fixture creates or binds an MCP Hub workspace set for the active Research Workspace canonical workspace ID.
- [x] #2 Workspace-set policy/path trust is resolved through MCP Hub APIs or UI, not duplicated in Research Workspace state.
- [x] #3 At least one downstream tool/policy availability path is exercised and records request/response evidence.
- [x] #4 Research Workspace handoff remains `/research-workspace` only; no `/workspace-playground` alias, redirect, or active UI label is introduced.
- [x] #5 RW-UAT-021 is updated only as far as live backend + WebUI + CDP evidence supports.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a focused Playwright/CDP E2E in `apps/tldw-frontend/e2e/workflows/tier-2-features/mcp-hub.spec.ts` for the Research Workspace -> MCP Hub handoff.
- The fixture creates a canonical Research Workspace ID, registers that ID as an MCP Hub Shared Workspace with a trusted root, creates a team-scoped MCP workspace set, adds the workspace as a member, creates a named policy assignment, resolves effective policy, executes the MCP virtual CLI `run` tool under `x-tldw-workspace-id`/`x-tldw-cwd`, then opens the MCP Hub workspace-set deep link and asserts the active Research Workspace is included in an MCP workspace set.
- The test keeps MCP Hub as the canonical owner of workspace sets, path trust, and policy/tool execution state. Research Workspace supplies only the canonical workspace ID and `source=research-workspace` deep-link context.
- The UI assertions explicitly reject `workspace-playground` in the URL/context text.
- Updated RW-UAT-021 to `Pass` with only the verified MCP evidence. RW-UAT-020 remains `Partial` until ACP and Sandbox fixture-backed validations close.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Live backend/WebUI validation now covers the MCP Hub Shared Workspaces slice of the Research Workspace model. The focused E2E passed against `127.0.0.1:18001` and `127.0.0.1:18002` after tightening the tool assertion to require HTTP 2xx: `1 passed (2.6s)`. An API-only confirmation returned workspace 200, Shared Workspace 201, workspace set 201, set member 201, policy assignment 201, effective policy 200 with `selected_workspace_source_mode: named` and `selected_workspace_trust_source: shared_registry`, and MCP tool execution 200 from the virtual CLI `run` tool.
- No Python code changed; Bandit is not applicable for this slice.

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
