---
id: TASK-2373
title: Expose MCP policy explain admin API routes
status: In Progress
labels:
- mcp
- policy
- implementation
- admin-api
modified_files:
- mcp_unified/gateway/fastapi.py
- mcp_unified/gateway/policy_explain.py
- tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 4 from the MCP effective permission explain implementation plan: mount optional standalone gateway admin API routes for policy explain and profile tool preview using the policy explain service and admin permission seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 create_gateway_router/create_gateway_app accept optional policy explain management dependencies without changing default behavior.
- [ ] #2 POST /mcp/policy/explain requires admin auth when enabled, checks mcp.policy.explain permission, returns stable policy explain responses, and audits successful requests.
- [ ] #3 POST /mcp/profiles/{profile_id}/tool-preview returns preview rows including denied installed tools through the runtime catalog provider.
- [ ] #4 GatewayPolicyExplainError and GatewayAdminPermissionError map to stable JSON error envelopes without becoming 500s.
- [ ] #5 Focused admin API tests cover auth required, success audit, denied tool preview, and at least one permission/audit failure path.
- [ ] #6 Task 4 focused pytest passes or blockers are documented.
- [ ] #7 Changes are committed separately for Task 4.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 4 implemented with opt-in FastAPI route mounting in mcp_unified/gateway/fastapi.py. Added route tests for admin-auth enforcement, valid admin-key audit, runtime catalog denied-tool preview, GatewayPolicyExplainError stable envelope mapping, review-gap coverage for injected GatewayAdminPermissionError route-level 403 mapping, conflicting preview profile_id validation, and injected-service route actor/runtime catalog binding. Verification run before commit: focused policy explain API pytest passed; gateway admin auth pytest passed; Bandit initially found a test-only hardcoded /tmp path, then passed after changing fixture data; git diff --check passed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started Task 4 implementation under the approved subagent-driven workflow.
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
