---
id: TASK-2238
title: Implement MCP profile-scoped tool discovery
status: Done
labels:
- mcp-unified
- gateway
- profiles
- tools
- implementation
priority: medium
modified_files:
- mcp_unified/gateway/tool_discovery.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 3 from the MCP default profile tooling implementation plan: add profile-scoped tool discovery helpers, deterministic filter-first/BM25 ranking, visible tool description, and bridge tool resolution primitives.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Documentation changes are not required for this pure helper/test slice; bridge
  runtime documentation remains part of the follow-up runtime integration task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented profile-scoped gateway tool discovery helpers in mcp_unified/gateway/tool_discovery.py and focused coverage in tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py.

Summary:
- Added installed backend tool discovery filtered through build_effective_policy_result().
- Added recommendation-only profile metadata visibility for recommended unavailable tools while keeping bridge resolution non-callable with tool_not_enabled.
- Added deterministic category filtering, installed-before-unavailable ordering, standard-library BM25 scoring metadata, category/id tie-breaks, describe, list, and resolve helpers.
- Added tests for policy-before-ranking filtering, installed/recommended ranking, describe visibility, bridge resolution, invalid descriptor handling, no semantic-search dependency metadata, and package-boundary cleanliness.

Verification:
- RED: source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py -q failed at collection with ImportError because mcp_unified.gateway.tool_discovery was missing.
- GREEN: source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py -q passed: 7 passed, 5 warnings.
- git diff --check passed with no output.
- source .venv/bin/activate && python -m bandit -r mcp_unified/gateway/tool_discovery.py -f json -o /tmp/bandit_mcp_tool_discovery.json passed with 0 findings.
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
