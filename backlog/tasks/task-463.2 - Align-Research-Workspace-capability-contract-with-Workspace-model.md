---
id: TASK-463.2
title: Align Research Workspace capability contract with Workspace model
status: Done
labels:
- research-workspace
- workspace
- capabilities
- mcp
- acp
- sandbox
priority: high
parent_task_id: TASK-463
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Phase A workspace slice by aligning /api/v1/workspaces/{workspace_id}/capabilities and the Research Workspace trust panel with the approved minimum Workspace model contract: content/source readiness plus sharing, MCP/tools, ACP/agents, sandbox, provider/model, governance, and expanded action gates. Preserve no /workspace-playground aliases or redirects.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Workspace capabilities API exposes top-level Phase A fields for workspace_kind, effective_access_level, migration_state, sharing_state, mcp_state, acp_state, sandbox_state, provider_state, source_summary, and allowed_actions.
- [ ] #2 Allowed actions include add/edit/delete sources, ask grounded questions, generate outputs, share, use tools, start agents, and run sandboxed actions with fail-closed reason codes.
- [ ] #3 Existing workspace_services remains available for compatibility and maps to the same summarized state.
- [ ] #4 Research Workspace trust panel surfaces sharing/migration and Workspace model readiness, not only MCP/ACP/Sandbox/Provider.
- [ ] #5 Focused backend and frontend tests cover the expanded contract and no /workspace-playground redirects or aliases are introduced.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned Research Workspace capabilities and picker contracts with the Workspace model vocabulary. Added backend/frontend regression coverage, live HTTP validation against the FastAPI app, Bandit validation, and route guard verification for no /workspace-playground aliasing.
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
