---
id: TASK-2295
title: Add tool availability and deferred tool-search parity
status: In Progress
labels:
- mcp
- tool-discovery
- profiles
- agentic-execution
- tools
references:
- https://code.claude.com/docs/en/tools-reference
documentation:
- Docs/superpowers/specs/2026-06-09-mcp-tool-availability-deferred-search-design.md
- Docs/superpowers/plans/2026-06-09-mcp-tool-availability-deferred-search-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement Claude-style tool availability introspection and deferred ToolSearch parity for MCP profiles. Cover exact loaded-tool reporting, profile-filtered available tools, installation/readiness status, permission-rule summaries, deferred loading/search by category/name/BM25, WaitForMcpServers interaction, and safe exposure through chat/ACP sessions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Direct profile categories remain visible in `tools/list`; deferred installed categories are hidden from initial tool exposure.
- [x] #2 `profile.tools.list`, `tool_search`, and `tool_describe` include direct, deferred installed, and recommendation-only entries that remain profile-filtered.
- [x] #3 Catalog payloads expose safe availability metadata for direct, deferred installed, and recommended-unavailable tools without leaking denied tools.
- [x] #4 `tool_call` remains available when deferred tools exist and delegates installed deferred tools through existing profile policy.
- [x] #5 Focused runtime/discovery tests, py_compile, Bandit, and `git diff --check` pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/superpowers/specs/2026-06-09-mcp-tool-availability-deferred-search-design.md`.
- Added `Docs/superpowers/plans/2026-06-09-mcp-tool-availability-deferred-search-implementation-plan.md`.
- Extended `mcp_unified.gateway.tool_discovery` with direct/deferred/recommended exposure classification, safe per-tool availability reason codes, category availability counts, and whole-catalog availability counts.
- Added package-local helpers for direct profile backend tools and deferred-tool detection so runtime listing can reuse catalog classification.
- Updated `ProfileAwareGatewayRuntime.list_tools()` to expose only direct installed backend tools plus discovery bridge tools. Deferred installed tools remain searchable/describable and callable through `tool_call`.
- Preserved bridge-name collision behavior for direct backend tools; deferred tools with bridge-reserved names no longer suppress bridge helpers.
- Updated package user-guide documentation for direct, deferred, and recommended-unavailable profile discovery behavior.

Verification:
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q` -> 183 passed, 6 warnings.
- `source ../../.venv/bin/activate && python -m ruff check mcp_unified/gateway/tool_discovery.py mcp_unified/gateway/profile_runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py` -> passed.
- `source ../../.venv/bin/activate && python -m py_compile mcp_unified/gateway/tool_discovery.py mcp_unified/gateway/profile_runtime.py` -> passed.
- `source ../../.venv/bin/activate && python -m bandit -r mcp_unified/gateway/tool_discovery.py mcp_unified/gateway/profile_runtime.py -f json -o /tmp/bandit_mcp_tool_availability_search.json` -> 0 findings.
- `git diff --check` -> passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented MCP profile tool availability and deferred search parity for the standalone gateway. Profiles now classify visible tools as direct, deferred installed, or recommended unavailable. Initial `tools/list` returns direct installed tools plus discovery bridge helpers, while `profile.tools.list`, `tool_search`, and `tool_describe` expose the full profile-filtered catalog with safe availability metadata. Deferred installed tools can still be delegated through `tool_call` under the existing profile policy path; recommendation-only tools remain discoverable setup hints and return `tool_not_enabled`.
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
