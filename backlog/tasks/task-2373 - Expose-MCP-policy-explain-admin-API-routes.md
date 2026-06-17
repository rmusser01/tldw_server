---
id: TASK-2373
title: Expose MCP policy explain admin API routes
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-17 07:02'
labels:
  - mcp
  - policy
  - implementation
  - admin-api
modified_files:
  - mcp_unified/gateway/fastapi.py
  - mcp_unified/gateway/policy_explain.py
  - mcp_unified/gateway/profile_runtime.py
  - mcp_unified/gateway/tool_use_reporting.py
  - tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 4 from the MCP effective permission explain implementation plan: mount optional standalone gateway admin API routes for policy explain and profile tool preview using the policy explain service and admin permission seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 create_gateway_router/create_gateway_app accept optional policy explain management dependencies without changing default behavior.
- [x] #2 POST /mcp/policy/explain requires admin auth when enabled, checks mcp.policy.explain permission, returns stable policy explain responses, and audits successful requests.
- [x] #3 POST /mcp/profiles/{profile_id}/tool-preview returns preview rows including denied installed tools through the runtime catalog provider.
- [x] #4 GatewayPolicyExplainError and GatewayAdminPermissionError map to stable JSON error envelopes without becoming 500s.
- [x] #5 Focused admin API tests cover auth required, success audit, denied tool preview, and at least one permission/audit failure path.
- [x] #6 Task 4 focused pytest passes or blockers are documented.
- [x] #7 Changes are committed separately for Task 4.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 4 implemented with opt-in FastAPI route mounting in mcp_unified/gateway/fastapi.py. Added route tests for admin-auth enforcement, valid admin-key audit, runtime catalog denied-tool preview, GatewayPolicyExplainError stable envelope mapping, review-gap coverage for injected GatewayAdminPermissionError route-level 403 mapping, route-level audit_store_unavailable mapping, conflicting preview profile_id validation, injected-service route actor/runtime catalog binding, and profile-aware runtime unfiltered admin catalog discovery. Verification run before commit: focused policy explain API pytest passed; gateway admin auth pytest passed; Bandit initially found a test-only hardcoded /tmp path, then passed after changing fixture data; git diff --check passed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started Task 4 implementation under the approved subagent-driven workflow.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Final review results: latest spec compliance review passed; latest code-quality review found no Critical runtime/security issues. The remaining Important reviewer note was Backlog closeout, addressed here. Expanded Task 4 touched scope includes mcp_unified/gateway/profile_runtime.py and mcp_unified/gateway/tool_use_reporting.py for the explicit unfiltered admin catalog path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 4 exposed opt-in MCP policy explain admin API routes. It added POST /mcp/policy/explain and POST /mcp/profiles/{profile_id}/tool-preview, wired admin identity plus mcp.policy.explain permission checks, stable policy/permission/audit error envelopes, route-bound service actor/catalog context, body/path profile_id conflict validation, and unfiltered admin catalog discovery for profile-aware and tool-use-reporting runtimes so denied installed tools remain visible in admin previews.

Verification from current HEAD using the root checkout virtualenv:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py -v -> 14 passed, 7 warnings
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_admin_auth.py -v -> 7 passed, 5 warnings
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r mcp_unified/gateway/fastapi.py mcp_unified/gateway/policy_explain.py mcp_unified/gateway/profile_runtime.py mcp_unified/gateway/tool_use_reporting.py tldw_Server_API/tests/MCP_unified/test_standalone_policy_explain_api.py -s B101 -f json -o /tmp/bandit_task2373_final.json -> 0 findings
- git diff --check HEAD~5..HEAD -> exit 0

Known non-blocking follow-up: add an explicit disabled-route 404 regression test if this surface is touched again.
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
