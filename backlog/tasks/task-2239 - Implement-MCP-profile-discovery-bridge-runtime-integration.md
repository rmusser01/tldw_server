---
id: TASK-2239
title: Implement MCP profile discovery bridge runtime integration
status: Done
labels:
- mcp-unified
- gateway
- profiles
- tools
- implementation
priority: medium
modified_files:
- mcp_unified/gateway/profile_runtime.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 4 from the MCP default profile tooling implementation plan: expose profile-scoped discovery bridge tools in ProfileAwareGatewayRuntime and intercept tool_categories.list, profile.tools.list, tool_search, tool_describe, and tool_call.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Resolved profiles expose read-only discovery bridge tools.
- [x] `tool_call` is exposed only for profiles with deferred categories.
- [x] Bridge calls are intercepted before ordinary backend execution.
- [x] Installed `tool_call` delegates to the resolved backend tool through policy.
- [x] Recommended-unavailable and denied tools are not made executable.
- [x] Non-list backend tool discovery does not crash bridge catalog behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added synthetic tool descriptors in `mcp_unified/gateway/profile_runtime.py` with `metadata.category == "tool_discovery"` and fixed small input schemas.
- Integrated Task 3 discovery helpers without modifying `mcp_unified/gateway/tool_discovery.py`.
- Added strict runtime validation for `tool_call` arguments with `invalid_tool_call_arguments` denials for missing fields, unknown fields, non-string `tool_id`, and non-object delegated arguments.
- Updated profile runtime tests to account for synthetic discovery helpers while preserving ordinary backend profile filtering assertions.
- Review fix: backend tools whose names collide with bridge-reserved names now win when profile policy allows them; mixed-type unknown argument keys no longer crash validation; profile-runtime list assertions now check exact visible tool-name sets.
- Re-review fix: when backend discovery fails for bridge-reserved names, malformed synthetic bridge arguments are validated before surfacing the backend discovery exception.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented profile-scoped MCP discovery bridge integration in ProfileAwareGatewayRuntime. list_tools now appends synthetic discovery descriptors for resolved profiles, including tool_call only when deferred categories are configured. call_tool now intercepts tool_categories.list, profile.tools.list, tool_search, tool_describe, and tool_call before backend execution. Installed tool_call requests resolve to the real backend tool and reuse the existing profile policy path; recommended-only or hidden tools return structured normal result payloads instead of executing. Added runtime tests for bridge exposure, profile-scoped search/catalog results, denied descriptions, recommended-unavailable rejection, installed delegation, and strict invalid tool_call argument handling.

Verification:
- RED: source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q failed with 23 expected bridge-related failures before implementation.
- Focused: source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q passed: 174 passed, 6 warnings.
- git diff --check passed.
- Bandit: source .venv/bin/activate && python -m bandit -r mcp_unified/gateway/profile_runtime.py -f json -o /tmp/bandit_mcp_profile_runtime_bridge.json passed with 0 findings.

Review follow-up verification:
- RED: source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q failed with 2 expected failures: bridge-name collision returned the synthetic descriptor, and mixed-type unknown keys raised TypeError.
- Focused: source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q passed: 176 passed, 6 warnings.
- git diff --check passed.
- Bandit: source .venv/bin/activate && python -m bandit -r mcp_unified/gateway/profile_runtime.py -f json -o /tmp/bandit_mcp_profile_runtime_bridge.json passed with 0 findings.

Re-review follow-up verification:
- RED: source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q failed with the new regression raising the backend RuntimeBackendError before invalid bridge argument validation.
- Focused: source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q passed: 177 passed, 6 warnings.
- git diff --check passed.
- Bandit: source .venv/bin/activate && python -m bandit -r mcp_unified/gateway/profile_runtime.py -f json -o /tmp/bandit_mcp_profile_runtime_bridge.json passed with 0 findings.
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
