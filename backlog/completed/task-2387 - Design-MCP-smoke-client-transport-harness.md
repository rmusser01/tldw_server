---
id: TASK-2387
title: Design MCP smoke client transport harness
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-19 16:08'
labels:
  - mcp
  - testing
  - smoke-client
  - gateway
  - agentic-execution
dependencies: []
references:
  - TASK-2281
  - 'https://code.claude.com/docs/en/tools-reference'
documentation:
  - >-
    Docs/superpowers/specs/2026-06-19-mcp-smoke-client-transport-harness-design.md
  - >-
    Docs/superpowers/plans/2026-06-19-mcp-smoke-client-transport-harness-implementation-plan.md
  - Docs/MCP/Unified/Smoke_Client.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design a simple MCP smoke client that runs the standalone/tldw MCP server through protocol, catalog, tool-call, resources, prompts, profile filtering, denial, and error-contract scenarios before continuing LSP-backed MCP tool implementation. The client must support deterministic in-process testing plus live HTTP, WebSocket, and stdio subprocess transports.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec covers in-process, live HTTP, live WebSocket, and stdio subprocess transports.
- [x] #2 Report contract is bounded and redacted for PR, CI, and Backlog use.
- [x] #3 Stdio subprocess execution is argv-based and never shell-based.
- [x] #4 TASK-2281 LSP implementation references the smoke client as its prerequisite harness.
- [x] #5 Baseline scenarios cover initialize, initialized notification, tools/list, ping, safe tools/call, unknown tool, resources, prompts, JSON-RPC batch, malformed request, profile filtering, and policy denial where fixtures support it.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-19-mcp-smoke-client-transport-harness-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Created design spec Docs/superpowers/specs/2026-06-19-mcp-smoke-client-transport-harness-design.md with deterministic in-process plus live HTTP, WebSocket, and stdio subprocess transport coverage.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the MCP smoke client harness with deterministic in-process coverage plus live HTTP, live WebSocket, and argv-only stdio subprocess transports. Added a redacted/bounded report contract, baseline scenario coverage, CLI entrypoint mcp-unified-smoke, operator docs at Docs/MCP/Unified/Smoke_Client.md, and tests for exit-code mapping, transport response correlation, stdio cleanup, stderr redaction, WebSocket profile/header behavior, and dash-prefixed stdio subprocess arguments. Final verification: python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py -q passed with 60 tests; python -m py_compile mcp_unified/smoke/*.py passed; python -m ruff check mcp_unified/smoke tldw_Server_API/app/core/MCP_unified/tests/test_smoke_client.py tldw_Server_API/app/core/MCP_unified/tests/fixtures/smoke_stdio_server.py passed; python -m bandit -r mcp_unified/smoke -f json -o /tmp/bandit_mcp_smoke_client.json reported 0 findings; git diff --check passed. Spec-review follow-up fixed stdio dash-prefixed subprocess argument handling so documented forms like --arg -m reach the stdio transport, and documented PATH allowlisting for command-name resolution.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
