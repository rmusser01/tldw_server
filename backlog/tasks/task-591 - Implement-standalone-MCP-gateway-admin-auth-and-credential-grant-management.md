---
id: TASK-591
title: Implement standalone MCP gateway admin auth and credential grant management
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-02 02:20
labels:
- mcp-unified
- standalone-gateway
- security
dependencies: []
priority: medium
modified_files:
- mcp_unified/gateway/admin_auth.py
- mcp_unified/gateway/credential_grants.py
- mcp_unified/gateway/__init__.py
- mcp_unified/gateway/bootstrap.py
- mcp_unified/gateway/config.py
- mcp_unified/gateway/fastapi.py
- mcp_unified/gateway/cli.py
- mcp_unified/interfaces/storage.py
- mcp_unified/storage/sqlite.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_admin_auth.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_credential_grants.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a package-owned admin auth seam plus credential-grant metadata manager, FastAPI routes, and CLI commands for the standalone MCP gateway. Keep secret material out of persistence and responses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Management routes can require standalone admin auth without importing tldw_Server_API.
- [x] #2 JSON-RPC /request and /ws routes are not accidentally gated by admin auth.
- [x] #3 Credential grants support list/show/create/patch/delete through manager, FastAPI, and CLI.
- [x] #4 Credential grants persist only broker references, slots, scopes, metadata, and provenance; secret-looking values are rejected or omitted before persistence.
- [x] #5 External-server delete guards continue to block deletion when enabled grants reference the server.
- [x] #6 Focused tests and Bandit on touched package files are recorded.
- [x] #7 Standalone config can enable admin auth and resolve the admin key from an environment variable without persisting the key.
- [x] #8 Credential-grant create rejects duplicate ids atomically instead of replacing existing grants.
- [x] #9 Admin auth errors return direct stable JSON payloads rather than framework-wrapped detail payloads.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-02-mcp-gateway-admin-auth-credential-grants.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented standalone gateway admin auth and credential-grant management. Added optional package-owned admin auth for management routes with direct JSON 401/403 payloads, config support that resolves admin keys from env vars without persisting them, and bootstrap propagation. Added credential-grant manager validation, recursive secret-key rejection, atomic create semantics, FastAPI routes, CLI commands, and SQLite create_grant support. Verification: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_admin_auth.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_credential_grants.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -v` passed with 241 tests; `python -m bandit -r mcp_unified/gateway/admin_auth.py mcp_unified/gateway/credential_grants.py mcp_unified/gateway/config.py mcp_unified/gateway/bootstrap.py mcp_unified/gateway/fastapi.py mcp_unified/gateway/cli.py mcp_unified/storage/sqlite.py mcp_unified/interfaces/storage.py -f json -o /tmp/bandit_mcp_gateway_admin_config.json` passed; `git diff --check` passed.
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
