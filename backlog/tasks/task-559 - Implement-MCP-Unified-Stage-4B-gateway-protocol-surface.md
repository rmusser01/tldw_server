---
id: TASK-559
title: Implement MCP Unified Stage 4B gateway protocol surface
status: Done
labels:
- mcp-unified
- standalone-extraction
- gateway
priority: medium
references:
- https://github.com/rmusser01/tldw_server/pull/2139
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
- Docs/superpowers/plans/2026-05-30-mcp-unified-stage4a-gateway-entrypoint-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next narrow Stage 4B standalone gateway slice after Stage 4A: extend the package-owned FastAPI JSON-RPC gateway to route resources, prompts, and module discovery/health methods through the injected runtime protocol while preserving package isolation and host compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Gateway runtime contract supports resources/list, resources/read, prompts/list, prompts/get, modules/list, and modules/health without importing tldw_Server_API.
- [x] #2 Gateway JSON-RPC dispatch validates params for the new methods and maps missing required fields to JSON-RPC invalid params responses.
- [x] #3 Focused gateway tests prove the new method surface, context propagation, package isolation, unsupported method behavior, and notification/batch semantics remain intact.
- [x] #4 Host extraction and HTTP mapping compatibility tests continue to pass.
- [x] #5 Ruff, Bandit, and git diff hygiene are recorded before PR handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started from merged Stage 4A baseline on origin/dev commit 71f0edfe8c. Next narrow slice is gateway protocol surface parity for resources, prompts, and module discovery/health through the injected runtime, without storage, stdio, external lifecycle, or host route wiring.

Implemented the Stage 4B protocol surface by extending the package-owned `GatewayRuntime` protocol and FastAPI JSON-RPC dispatch for `resources/list`, `resources/read`, `prompts/list`, `prompts/get`, `modules/list`, and `modules/health`. Added focused tests that verify runtime delegation, request context propagation, parameter validation, and preservation of existing Stage 4A behavior.

Verification recorded:
- RED gateway test run: `4 failed, 11 passed, 3 warnings` before implementation; new methods returned `Method not found`.
- GREEN gateway test run: `15 passed, 3 warnings`.
- Host compatibility: `47 passed, 4 warnings` for extraction contracts and HTTP mapping tests.
- Ruff: `All checks passed!`.
- Bandit: `/tmp/bandit_mcp_stage4b_gateway_protocol_surface.json` reported `"results": []`.
- `git diff --check` exited cleanly.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4B adds the standalone gateway protocol surface for resources, prompts, and module discovery/health through the injected runtime contract. The slice remains package-isolated, preserves host compatibility tests, and intentionally leaves storage wiring, stdio transport, external lifecycle, profile policy, and host route integration for later stages. No known skips or blockers.
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
