---
id: TASK-589
title: Expose MCP external runtime tools through gateway runtime
status: Done
labels:
- mcp-unified
- gateway
- external-runtime
priority: medium
documentation:
- Docs/superpowers/plans/2026-06-02-mcp-gateway-external-runtime-adapter-plan.md
modified_files:
- mcp_unified/gateway/external_runtime_adapter.py
- mcp_unified/gateway/profile_runtime.py
- mcp_unified/gateway/external_runtime.py
- mcp_unified/gateway/__init__.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime_adapter.py
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_external_runtime.py
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- Docs/superpowers/plans/2026-06-02-mcp-gateway-external-runtime-adapter-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Adapt GatewayExternalRuntimeManager into the standalone gateway GatewayRuntime surface so active external virtual tools can be listed and executed through tools/list and tools/call with profile policy, credential brokering, and audit behavior preserved. Keep transport serving and package installer execution out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 External virtual tools are exposed through GatewayRuntime tools/list descriptors with external metadata and safe nullable schema defaults.
- [x] #2 GatewayRuntime tools/call dispatches active external virtual tool names through GatewayExternalRuntimeManager while preserving local base-runtime delegation.
- [x] #3 ProfileAwareGatewayRuntime forwards resolved effective policy metadata without mutating the original GatewayRequestContext, including metadata=None callers.
- [x] #4 External runtime call routing uses a direct virtual-tool membership check instead of list/sort/deep-copy catalog scans.
- [x] #5 Installer timeout and operation failures retain sanitized diagnostic context without exposing secret-looking values in public payloads or test log arguments.
- [x] #6 Focused MCP tests, Ruff, Bandit, and diff hygiene checks are recorded and passing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-02-mcp-gateway-external-runtime-adapter-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2219 review follow-up complete. Rebased on current origin/dev (already up to date). Addressed Gemini defensive-null findings by defaulting nullable external input schemas to {}, reading request metadata through safe fallbacks in the adapter/profile runtime, and covering metadata=None regressions. Addressed Qodo findings by adding a direct GatewayExternalRuntimeManager.has_virtual_tool() ownership check for call routing, removing the adapter's call-time list/sort/deep-copy scan, restoring timeout traceback logging, and adding sanitized traceback-frame diagnostics plus sanitized exception causes for installer operation failures without exposing do-not-leak secret text in public payloads/log arguments. Verification: targeted RED tests failed before fixes; targeted follow-up tests now pass; focused MCP suite reports 206 passed, 5 warnings; Ruff reports All checks passed; Bandit on touched gateway code reports 0 findings; git diff --check is clean.
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
