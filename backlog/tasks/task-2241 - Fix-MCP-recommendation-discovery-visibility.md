---
id: TASK-2241
title: Fix MCP recommendation discovery visibility
status: Done
labels:
- mcp-unified
- gateway
- profiles
- review-fix
modified_files:
- mcp_unified/gateway/tool_discovery.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Final review fix for PR 2251: recommended setup-dependent tools from profile metadata must be visible as recommended_unavailable in profile discovery/search/describe while remaining non-callable through tool_call. Current discovery filters recommendations through executable policy and hides bundled recommendations without explicit grants.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-03-mcp-default-profile-tooling-presets-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD RED: added/updated gateway discovery coverage so recommendations without executable grants, including bundled/default descriptors with no capability, must be discoverable as recommended_unavailable and resolve as unavailable/tool_not_enabled. Initial focused pytest failed as intended because search returned [] under the old _recommendation_visible() executable-policy gate.

Implementation: removed the recommendation executable-policy visibility gate from _recommended_entry(); recommendations still require a valid normalized id, remain non-callable without an installed backend entry, and installed backend tools still use _installed_tool_allowed()/build_effective_policy_result().
Verification:
- source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py::test_recommendations_without_executable_grants_are_discoverable_not_callable -q -> RED before production change: failed because results were [] instead of the expected recommendation ids; GREEN after fix: 1 passed.
- source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py -q -> 9 passed.
- source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_tool_discovery.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py -q -> 196 passed.
- source .venv/bin/activate && python -m bandit -r mcp_unified/gateway/tool_discovery.py -f json -o /tmp/bandit_mcp_recommendation_visibility.json -> exit 0, JSON report written.
- git diff --check -> exit 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposed profile recommendation metadata in discovery/search/describe without requiring executable policy grants. Recommendation descriptors still require a valid id, normalize safely, show as recommended_unavailable, and resolve/call as unavailable with tool_not_enabled unless an installed backend tool is actually policy-granted. Installed backend tool visibility remains filtered by build_effective_policy_result(), preserving denial behavior and existing BM25 filter-first ranking.
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
