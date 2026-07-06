---
id: TASK-12752
title: Implement MCP gateway admin permission seam
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-17 05:54'
labels:
  - mcp
  - policy
  - implementation
  - admin-auth
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from the MCP effective permission explain implementation plan: add GatewayAdminIdentity, a permission checker seam, permission errors/responses, and focused auth seam tests without mounting policy explain API routes yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GatewayAdminIdentity exists with a local_admin helper that includes mcp.policy.explain permission when auth is disabled/local.
- [x] #2 GatewayAdminPermissionError, GatewayAdminPermissionChecker protocol, and DefaultGatewayAdminPermissionChecker are implemented with stable reason codes.
- [x] #3 An identity-producing admin dependency helper exists while preserving existing GatewayAdminAuthError, gateway_admin_auth_dependencies, and auth error responses.
- [x] #4 Focused tests cover local admin identity permission and missing-permission denial.
- [x] #5 Task 3 focused pytest passes or blockers are documented.
- [x] #6 Changes are committed separately for Task 3.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 3 quality-review fix: _gateway_admin_identity_dependency now preserves local identity only for disabled admin auth and returns a distinct generic authenticated gateway admin identity (actor_id=gateway-admin, source=gateway_admin_auth) after successful enabled admin auth without deriving from the credential. Added focused async test coverage for valid enabled auth identity plus missing/invalid GatewayAdminAuthError reason codes. Verification for review fix passed: focused policy explain API seam pytest (5 passed), existing app/core gateway admin auth pytest (7 passed), Bandit touched scope with B101 skipped (no issues), and git diff --check.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Final review results: latest spec compliance review passed; latest code-quality review found no Critical or Important issues. The only reviewer note was final task bookkeeping, addressed here.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 3 added the standalone gateway admin permission seam. It introduced GatewayAdminIdentity, GatewayAdminPermissionError, GatewayAdminPermissionChecker, DefaultGatewayAdminPermissionChecker, identity-producing admin dependency helpers, stable permission-denied responses, and focused tests. A review fix now distinguishes disabled local admin identity from enabled authenticated gateway admin identity without exposing credential material.

Verification from current HEAD using the root checkout virtualenv:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py -v -> 5 passed, 7 warnings
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_admin_auth.py -v -> 7 passed, 5 warnings
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway/admin_auth.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py -s B101 -f json -o /tmp/bandit_task2372.json -> 0 findings
- git diff --check HEAD~2..HEAD -> exit 0

Known caveat: tldw_Server_API/tests/MCP_unified/test_standalone_gateway_admin_auth.py does not exist in this worktree; the existing package-level admin auth regression test was run instead.
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
