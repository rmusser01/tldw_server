---
id: TASK-478.27
title: Validate MCP workspace-set policy and tool execution for Research Workspace
status: To Do
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
- [ ] #1 Live fixture creates or binds an MCP Hub workspace set for the active Research Workspace canonical workspace ID.
- [ ] #2 Workspace-set policy/path trust is resolved through MCP Hub APIs or UI, not duplicated in Research Workspace state.
- [ ] #3 At least one downstream tool/policy availability path is exercised and records request/response evidence.
- [ ] #4 Research Workspace handoff remains `/research-workspace` only; no `/workspace-playground` alias, redirect, or active UI label is introduced.
- [ ] #5 RW-UAT-021 is updated only as far as live backend + WebUI + CDP evidence supports.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
