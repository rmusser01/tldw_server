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
Task 3 quality-review fix: _gateway_admin_identity_dependency now preserves local identity only for disabled admin auth and returns a distinct generic authenticated gateway admin identity (actor_id=gateway-admin, source=gateway_admin_auth) after successful enabled admin auth without deriving from the credential. Added focused async test coverage for valid enabled auth identity plus missing/invalid GatewayAdminAuthError reason codes. Verification for review fix passed: focused policy explain API seam pytest (5 passed), existing app/core gateway admin auth pytest (7 passed), Bandit touched scope with B101 skipped (no issues), and git diff --check. Awaiting controller review; task status remains In Progress.
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
