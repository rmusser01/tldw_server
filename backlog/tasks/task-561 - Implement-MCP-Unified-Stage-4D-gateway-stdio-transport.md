---
id: TASK-561
title: Implement MCP Unified Stage 4D gateway stdio transport
status: Done
labels:
- mcp-unified
- standalone-extraction
- gateway
priority: medium
references:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
- Docs/superpowers/plans/2026-05-30-mcp-unified-stage4d-gateway-stdio-transport-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next narrow Stage 4D standalone gateway slice after Stage 4C: add a package-owned client-facing stdio JSON-RPC transport skeleton that reuses the gateway dispatcher and preserves package isolation. This slice intentionally avoids SQLite/profile enforcement, external MCP lifecycle, process spawning, and host route integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Package gateway exposes a stdio JSON-RPC line handler without importing `tldw_Server_API`.
- [x] #2 Stdio initialize, ping, batch, notification-only, and malformed JSON behavior matches the shared gateway JSON-RPC envelope semantics.
- [x] #3 FastAPI HTTP and WebSocket gateway behavior remains compatible after shared dispatcher extraction.
- [x] #4 Focused gateway tests cover stdio success, parse error, notification suppression, batch filtering, and stdio request context metadata.
- [x] #5 Host extraction and HTTP mapping compatibility tests, Ruff, Bandit, and whitespace checks are recorded before PR handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started from merged Stage 4C baseline on origin/dev commit 887ebeeb28. Baseline gateway package tests passed with `22 passed, 3 warnings`.

Added Stage 4D RED coverage for stdio line-delimited JSON-RPC handling: initialize response serialization, notification suppression, mixed batch response filtering, malformed JSON parse errors, and stdio request-context metadata. RED run failed as expected with `4 failed, 22 passed, 3 warnings` because `mcp_unified.gateway.stdio` did not exist.

Extracted the shared gateway JSON-RPC models, validation, parsing, dispatch, error mapping, notification sentinel, and JSON-safe response serialization into `mcp_unified.gateway.jsonrpc`. Updated the FastAPI HTTP/WebSocket transport to use the shared dispatcher while preserving HTTP 204 notification behavior and WebSocket suppression semantics. Added `mcp_unified.gateway.stdio` with `GatewayStdioServer` and `handle_stdio_line(...)` for stdin-style single-line JSON-RPC payloads. Added lazy FastAPI helper exports so importing the stdio submodule does not eagerly load the FastAPI transport path.

Verification recorded:
- GREEN gateway package tests: `27 passed, 4 warnings`.
- Host compatibility: `47 passed, 4 warnings` for extraction contracts and HTTP mapping tests.
- Ruff: initial import-order issues were fixed with `ruff check --fix`; final Ruff reported `All checks passed!`.
- Bandit: `/tmp/bandit_mcp_stage4d_gateway_stdio_transport.json` reported `0` findings and no errors.
- `git diff --check` exited cleanly.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4D adds a package-owned client-facing stdio gateway transport skeleton for line-delimited JSON-RPC requests. HTTP, WebSocket, and stdio now share a transport-neutral gateway dispatcher, keeping response envelopes, validation, parse errors, notification suppression, batch filtering, and runtime exception mapping consistent across transports. The stdio submodule can be imported without eagerly loading the FastAPI transport. This slice remains package-isolated and intentionally leaves SQLite/profile enforcement, external MCP lifecycle, upstream stdio process spawning, auth/session policy, and host route integration for later stages. No known skips or blockers.
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
