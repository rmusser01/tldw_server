---
id: TASK-2372
title: Implement MCP gateway admin permission seam
status: In Progress
labels:
- mcp
- policy
- implementation
- admin-auth
modified_files:
- mcp_unified/gateway/admin_auth.py
- tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from the MCP effective permission explain implementation plan: add GatewayAdminIdentity, a permission checker seam, permission errors/responses, and focused auth seam tests without mounting policy explain API routes yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 GatewayAdminIdentity exists with a local_admin helper that includes mcp.policy.explain permission when auth is disabled/local.
- [ ] #2 GatewayAdminPermissionError, GatewayAdminPermissionChecker protocol, and DefaultGatewayAdminPermissionChecker are implemented with stable reason codes.
- [ ] #3 An identity-producing admin dependency helper exists while preserving existing GatewayAdminAuthError, gateway_admin_auth_dependencies, and auth error responses.
- [ ] #4 Focused tests cover local admin identity permission and missing-permission denial.
- [ ] #5 Task 3 focused pytest passes or blockers are documented.
- [ ] #6 Changes are committed separately for Task 3.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 3 seam implemented in mcp_unified/gateway/admin_auth.py with GatewayAdminIdentity, GatewayAdminPermissionError, GatewayAdminPermissionChecker protocol, DefaultGatewayAdminPermissionChecker, default local identity/dependency helper, and permission error response helper. Focused auth seam tests added in tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py. Verification passed: focused pytest (4 tests), existing app/core gateway admin auth pytest (7 tests), Bandit touched scope with B101 skipped, and git diff --check. Requested tldw_Server_API/tests/MCP_unified/test_standalone_gateway_admin_auth.py path does not exist. Awaiting controller review; task status remains In Progress.
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
