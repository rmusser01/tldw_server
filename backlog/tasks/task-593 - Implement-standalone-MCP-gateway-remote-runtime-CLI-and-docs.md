---
id: TASK-593
title: Implement standalone MCP gateway remote runtime CLI and docs
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 04:39'
labels:
  - mcp-unified
  - standalone-gateway
  - cli
dependencies:
  - TASK-591
  - TASK-592
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add remote CLI commands that call a running standalone gateway for external runtime lifecycle operations, and document the end-to-end standalone admin workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CLI can call a running gateway for runtime list/start/stop/restart/refresh/reconcile/install/update operations.
- [x] #2 CLI runtime commands require an explicit --gateway-url or MCP_UNIFIED_GATEWAY_URL.
- [x] #3 Admin auth uses an environment-provided value such as MCP_UNIFIED_GATEWAY_ADMIN_KEY; command-line secret arguments are avoided.
- [x] #4 CLI preserves the gateway JSON payloads and reason codes.
- [x] #5 Docs explain local store commands versus remote runtime commands and include a safe credential-grant example.
- [x] #6 No runtime CLI command starts an upstream process that becomes orphaned when the CLI exits.
- [x] #7 Gateway URL semantics are explicit: the URL is the mounted gateway base path, such as http://host/mcp, and the client does not auto-add /mcp.
- [x] #8 Remote CLI preserves JSON reason_code payloads from HTTP 4xx/5xx response bodies.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-02-mcp-gateway-remote-runtime-cli-docs.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a package-owned remote admin client for mounted gateway base URLs, env/header-based admin auth, sanitized malformed/connection failures, and HTTP JSON error payload preservation.

Added flat runtime CLI commands for list/start/stop/restart/refresh/reconcile/install/update that call the running gateway over HTTP rather than starting local transports.

Added standalone admin docs covering local store commands, remote runtime commands, gateway URL prefix semantics, env-only admin keys, safe credential grants, snapshots, and runtime examples.

Touched files:
- mcp_unified/gateway/remote_admin.py
- mcp_unified/gateway/cli.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py
- Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md
- Docs/superpowers/plans/2026-06-02-mcp-gateway-remote-runtime-cli-docs.md
- backlog/tasks/task-593 - Implement-standalone-MCP-gateway-remote-runtime-CLI-and-docs.md

Verification passed:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py -v: 24 passed
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_remote_runtime_cli.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -v: 98 passed
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r mcp_unified/gateway/remote_admin.py mcp_unified/gateway/cli.py mcp_unified/gateway/fastapi.py -f json -o /tmp/bandit_mcp_gateway_remote_runtime_cli.json: 0 results
- git diff --check: passed

Known skips or blockers: none.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the standalone gateway remote runtime CLI and docs. Runtime commands now call the mounted gateway admin endpoints using --gateway-url or MCP_UNIFIED_GATEWAY_URL, preserve gateway JSON reason_code payloads, use MCP_UNIFIED_GATEWAY_ADMIN_KEY for admin auth without command-line secret arguments, and keep process ownership with the running gateway. Documentation now explains local store operations versus remote runtime operations, safe credential grant metadata, snapshots, admin auth, and mounted base URL semantics.
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
