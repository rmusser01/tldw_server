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
- tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
- Docs/superpowers/plans/2026-06-02-mcp-gateway-external-runtime-adapter-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Adapt GatewayExternalRuntimeManager into the standalone gateway GatewayRuntime surface so active external virtual tools can be listed and executed through tools/list and tools/call with profile policy, credential brokering, and audit behavior preserved. Keep transport serving and package installer execution out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
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
Added a package-owned ExternalRuntimeGatewayRuntime adapter that exposes active GatewayExternalRuntimeManager virtual tools through GatewayRuntime tools/list and tools/call. ProfileAwareGatewayRuntime now copies resolved effective policy data into delegated request context metadata so external server grants and credential grants reach the external runtime manager without mutating the original context. Also fixed adjacent installer timeout/error wrapping regressions covered by existing tests. Verification: focused MCP pytest suite passed with 201 passed and 5 warnings; Ruff reported All checks passed; Bandit JSON at /tmp/bandit_mcp_gateway_external_runtime_adapter.json reported 0 results; git diff --check passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
