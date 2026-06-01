---
id: TASK-582
title: Implement MCP upstream stdio external server transport
status: Done
labels:
- mcp-unified
- external-servers
- runtime
- stdio
- security
documentation:
- Docs/superpowers/specs/2026-06-01-mcp-upstream-stdio-transport-design.md
modified_files:
- mcp_unified/federation/stdio_transport.py
- mcp_unified/federation/__init__.py
- tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py
- Docs/superpowers/specs/2026-06-01-mcp-upstream-stdio-transport-design.md
- Docs/superpowers/plans/2026-06-01-mcp-upstream-stdio-transport-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the package-owned upstream stdio process transport for external MCP servers so the standalone gateway runtime manager can launch, discover, call, health-check, and stop configured stdio MCP server processes without depending on tldw_Server_API host code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Package-owned stdio transport launches configured external MCP server commands with bounded cwd and explicit environment allowlist behavior.
- [x] #2 Transport implements connect, list_tools, call_tool, health_check, and close against JSON-RPC over stdio with deterministic timeouts and structured failures.
- [x] #3 Credential broker runtime_auth is injected only for per-call process environment/request metadata supported by the transport and never logged or persisted.
- [x] #4 Focused tests cover launch validation, initialization/discovery, tool calls, health failures, process exit cleanup, timeout behavior, and secret redaction/no leakage.
- [x] #5 Ruff, focused pytest, Bandit on touched Python source, and git diff --check pass before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-01-mcp-upstream-stdio-transport-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `mcp_unified.federation.stdio_transport.StdioExternalTransport` with shell-free `asyncio.create_subprocess_exec` argv launch, cwd validation, environment allowlisting, JSON-RPC initialize/list/call/ping handling, deterministic timeouts, safe structured errors, and idempotent subprocess cleanup.
- Added package factory/export helpers so gateway runtime callers can inject the package-owned stdio transport without host `tldw_Server_API` dependencies.
- Added subprocess-backed tests for validation, package boundary import behavior, env allowlisting, discovery normalization, tool success/error calls, runtime auth `_meta` forwarding, timeout redaction, and exited-process health.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the package-owned upstream stdio transport in `mcp_unified.federation.stdio_transport`, exported the factory/helpers, and added subprocess-backed coverage for validation, package-boundary imports, environment allowlisting, discovery/calls, runtime-auth metadata, timeout redaction, and exited-process health.

Verification recorded:
- `../../.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py -q` -> 38 passed, 3 warnings
- `../../.venv/bin/python -m ruff check mcp_unified/federation/stdio_transport.py mcp_unified/federation/__init__.py tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py` -> All checks passed
- `../../.venv/bin/python -m bandit -r mcp_unified/federation/stdio_transport.py -f json -o /tmp/bandit_mcp_stdio_transport.json` -> exit 0
- `git diff --check` -> exit 0

Known skips/blockers: none.
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
