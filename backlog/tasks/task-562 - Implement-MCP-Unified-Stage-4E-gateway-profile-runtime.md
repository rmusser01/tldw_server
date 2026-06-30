---
id: TASK-562
title: Implement MCP Unified Stage 4E gateway profile runtime
status: Done
labels:
- mcp-unified
- standalone-extraction
- gateway
priority: medium
references:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
- Docs/superpowers/plans/2026-05-30-mcp-unified-stage4e-gateway-profile-runtime-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next narrow Stage 4E standalone gateway slice after Stage 4D: add a package-owned profile-aware gateway runtime wrapper that resolves standalone default or explicit profiles before discovery/execution and delegates allowed calls to an injected backend runtime. This slice intentionally avoids external MCP lifecycle, upstream process spawning, SQLite CLI/config commands, and tldw_server host route integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Package gateway exposes a profile-aware runtime wrapper without importing `tldw_Server_API`.
- [x] #2 No-profile standalone tool discovery returns no executable tools and tool execution denies with structured `profile_required` data when no default profile is configured.
- [x] #3 Explicit or default profiles filter `tools/list` and allow/deny `tools/call` by profile policy before delegating to the backend runtime.
- [x] #4 FastAPI transports can select a profile through lightweight request metadata without changing existing no-header behavior.
- [x] #5 Focused gateway tests cover default-profile allow, explicit-profile header selection, no-profile fail-closed behavior, and deny reason payloads.
- [x] #6 Host extraction and HTTP mapping compatibility tests, Ruff, Bandit, and whitespace checks are recorded before PR handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started from merged Stage 4D baseline on origin/dev commit 4ae10d2a7e. Baseline gateway package tests passed with `29 passed, 4 warnings`.

Added Stage 4E RED coverage for a profile-aware gateway runtime wrapper: no-profile fail-closed discovery/execution, default-profile allow/filter behavior, explicit HTTP header profile selection, and WebSocket query profile selection. RED run failed as expected with `4 failed, 29 passed, 4 warnings` because `mcp_unified.gateway.profile_runtime` did not exist.

Added `GatewayPolicyDenied` for machine-readable gateway policy denials, mapped it to JSON-RPC server error `-32001`, and added `ProfileAwareGatewayRuntime`. The wrapper resolves explicit or default profiles through existing structured profile resolver/store primitives, returns no tools for unresolved standalone profiles, filters `tools/list`, denies `tools/call` before backend delegation when profile policy does not allow the tool, and delegates resources/prompts/modules unchanged for this slice. FastAPI HTTP and WebSocket transports now propagate optional profile ids from `X-MCP-Profile` / `X-MCP-Profile-Id` headers or `profile_id` / `profileId` query params into request metadata.

Verification recorded:
- GREEN gateway package tests: `33 passed, 4 warnings`.
- Host compatibility: `47 passed, 4 warnings` for extraction contracts and HTTP mapping tests.
- Ruff: `All checks passed!`.
- Bandit: `/tmp/bandit_mcp_stage4e_gateway_profile_runtime.json` reported `0` findings and no errors.
- `git diff --check` exited cleanly.

PR review closeout after rebasing onto origin/dev commit 112108c103:
- Verified current Qodo and Gemini review threads against rebased code.
- Fixed Qodo docstring coverage for new profile-runtime test helpers.
- Fixed denied tool calls so explicit tool-policy denials do not invoke backend discovery.
- Added capability-policy fail-closed handling when backend metadata discovery raises.
- Added defensive handling for non-list backend tool discovery payloads, invalid tool descriptors, and transport doubles missing headers/query_params.
- Re-ran focused validation: gateway package tests `38 passed, 4 warnings`; extraction/http compatibility tests `47 passed, 4 warnings`; Ruff passed; Bandit reported `0` findings; `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4E adds a package-owned profile-aware gateway runtime wrapper for standalone gateway tool discovery and execution. It keeps the existing transport-neutral JSON-RPC and FastAPI/WebSocket behavior intact while allowing gateway callers to select a profile through lightweight metadata and enabling default-profile standalone behavior through existing profile resolver/store primitives. Tool discovery now filters through effective profile policy, and tool execution fails closed with structured `profile_required` / `tool_not_allowed` denial data before backend delegation. This slice remains package-isolated and intentionally leaves external MCP lifecycle, upstream process spawning, SQLite CLI/config commands, preset duplication flows, and host route integration for later stages. No known skips or blockers.

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
