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
- Reopened for PR review remediation and rebase onto latest `origin/dev`.
- Rebased onto latest `origin/dev` (`f15f92809ae38420abd82ba77c4ea56b0718c112`) and addressed PR review feedback from Qodo, cubic, and CodeRabbit.
- Fixed timeout cleanup so request timeouts raise immediately and schedule subprocess cleanup outside `_request_lock`.
- Moved process capture/validation inside `_request_lock`, hardened process terminate/kill/wait races, added MCP initialize `capabilities`, sent `notifications/initialized`, rejected bare executable names without PATH allowlisting, and rejected non-numeric/non-finite timeout values.
- Added module/helper/test docstrings and regression tests for initialize lifecycle, timeout cleanup bounds, PATH diagnostics, and timeout validation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented package upstream stdio transport and PR review remediation. The rebased branch now includes MCP-compliant initialize payloads plus `notifications/initialized`, non-blocking timeout cleanup, request-lock process capture, race-hardened close behavior, bare-command/PATH diagnostics, structured invalid-timeout handling, and docstrings for the new test module/functions.

Verification recorded after review remediation:
- `../../.venv/bin/python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py -q` -> 45 passed, 3 warnings
- `../../.venv/bin/python -m ruff check mcp_unified/federation/stdio_transport.py mcp_unified/federation/__init__.py tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py` -> All checks passed
- `../../.venv/bin/python -m bandit -r mcp_unified/federation/stdio_transport.py -f json -o /tmp/bandit_mcp_stdio_transport.json` -> exit 0
- `git diff --check` -> exit 0

Known skips/blockers: GitHub checks were pending before the remediation push.
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
