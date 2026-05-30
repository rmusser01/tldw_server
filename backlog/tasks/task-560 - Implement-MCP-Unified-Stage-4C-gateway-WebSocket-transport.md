---
id: TASK-560
title: Implement MCP Unified Stage 4C gateway WebSocket transport
status: Done
labels:
- mcp-unified
- standalone-extraction
- gateway
priority: medium
references:
- https://github.com/rmusser01/tldw_server/pull/2141
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
- Docs/superpowers/plans/2026-05-30-mcp-unified-stage4b-gateway-protocol-surface-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next narrow Stage 4C standalone gateway slice after Stage 4B: add package-owned FastAPI WebSocket JSON-RPC transport that reuses the existing gateway dispatcher and preserves package isolation and host compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Gateway FastAPI router exposes a package-owned WebSocket endpoint for JSON-RPC messages without importing tldw_Server_API.
- [x] #2 WebSocket transport handles initialize, ping, resources/prompts/modules/tools requests using the same dispatcher and response envelope behavior as HTTP.
- [x] #3 WebSocket transport maps malformed JSON text frames to JSON-RPC parse errors and suppresses notification responses consistently with HTTP notification behavior.
- [x] #4 Focused gateway tests cover WebSocket success, parse error, notification/no-response, batch behavior, and package isolation.
- [x] #5 Host extraction and HTTP mapping compatibility tests continue to pass; Ruff, Bandit, and git diff hygiene are recorded before PR handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started from merged Stage 4B baseline on origin/dev commit 27f2b4b7eb. Implemented the next narrow Stage 4C gateway transport slice by adding package-owned `/ws` WebSocket JSON-RPC handling in `mcp_unified.gateway.fastapi`.

The WebSocket route accepts text frames, reuses the existing JSON-RPC parser/dispatcher/error mapping, serializes the existing Pydantic response models for `send_json`, and suppresses notification-only responses consistently with HTTP notification behavior.

Verification recorded:
- Baseline gateway package tests before edits: `16 passed, 3 warnings`.
- RED WebSocket test run: `4 failed, 16 passed, 3 warnings`; all new WebSocket tests failed because `/mcp/ws` was not registered.
- GREEN gateway package tests: `20 passed, 3 warnings`.
- Host compatibility: `47 passed, 4 warnings` for extraction contracts and HTTP mapping tests.
- Ruff: `All checks passed!`.
- Bandit: `/tmp/bandit_mcp_stage4c_gateway_websocket_transport.json` reported `"results": []`.
- `git diff --check` exited cleanly.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4C adds the standalone package FastAPI WebSocket transport for JSON-RPC messages while keeping runtime dispatch, validation, notification behavior, and batch response filtering shared with the HTTP transport. This slice remains package-isolated and intentionally leaves auth/session policy, stdio, SQLite wiring, external lifecycle, and host route integration for later stages. No known skips or blockers.
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
