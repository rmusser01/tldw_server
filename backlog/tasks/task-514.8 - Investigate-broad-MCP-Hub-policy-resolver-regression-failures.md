---
id: TASK-514.8
title: Investigate broad MCP Hub policy resolver regression failures
status: To Do
parent_task_id: TASK-514
documentation:
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_policy_overrides.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_shared_workspace_registry.py
- tldw_Server_API/tests/MCP_unified/test_mcp_hub_workspace_set_objects.py
- tldw_Server_API/tests/MCP_unified/test_tool_catalogs_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on optional broad MCP Unified regression failures found during TASK-514 closeout. The Notes task MCP tool suite passes, but `python -m pytest tldw_Server_API/tests/Notes_NEW tldw_Server_API/tests/MCP_unified -v` fails three persistent MCP Hub policy resolver assertions where `resolved_policy_document` includes empty `tool_tier_overrides` and `conditions` that are absent from `authored_policy_document`; `test_tool_catalogs_flow` also failed only in the full broad sweep with shutdown_in_progress and passed in isolation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Resolve or intentionally update the policy resolver authored/resolved document equality expectations.
- [ ] #2 Determine whether the tool catalog shutdown_in_progress failure is test order/shared lifecycle leakage.
- [ ] #3 Restore the broad Notes_NEW plus MCP_unified pytest sweep or document any accepted skips with focused verification.
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
