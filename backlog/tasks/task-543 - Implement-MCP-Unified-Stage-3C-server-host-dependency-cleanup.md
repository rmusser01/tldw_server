---
id: TASK-543
title: Implement MCP Unified Stage 3C server host-dependency cleanup
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-29 01:47'
labels:
  - mcp
  - mcp-unified
  - standalone
  - stage3
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue MCP Unified standalone extraction after Stage 3B merged by removing server.py import-time host dependencies for testing helpers, AuthNZ exceptions, and WebSocket stream construction. Preserve existing tldw_server WebSocket/auth/test behavior through runtime adapters and focused compatibility tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP server.py no longer imports tldw_server testing helpers, AuthNZ exception classes, or WebSocketStream at module import time.
- [x] #2 Runtime dependencies expose host-neutral environment flag and WebSocket stream construction seams, with tldw_runtime preserving current behavior.
- [x] #3 AuthNZ websocket token failures are handled inside the tldw auth adapter so server.py does not need AuthNZ exception classes in its noncritical exception tuple.
- [x] #4 Focused tests cover server import-boundary cleanup, runtime dependency exports, WebSocket stream factory injection, and AuthNZ adapter fail-closed behavior.
- [x] #5 Focused pytest, Ruff, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-29-mcp-unified-stage3c-server-host-deps-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: RED extraction contracts failed for the expected missing seams; focused pytest passed with 71 tests; Ruff passed on touched files; Bandit scoped scan wrote /tmp/bandit_mcp_stage3c_server_host_deps.json with empty results.

Test hygiene: switched touched WebSocket security tests to monkeypatch.setenv so MCP_WS_* values do not leak into later WebSocket workspace tests.

Review fix pass after PR #2105 comments: added _handle_websocket_messages -> None annotation, restored AuthNZ-token MCP-JWT fallback gate when the adapter returns None, and added a host-neutral WebSocketStream protocol for the stream factory contract.

Review verification: focused pytest passed with 74 tests; Ruff passed on touched files; Bandit scoped scan wrote /tmp/bandit_mcp_stage3c_server_host_deps_review.json with empty results.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented MCP Unified Stage 3C server host-dependency cleanup and addressed PR #2105 review feedback. server.py no longer imports tldw_server testing helpers, AuthNZ exception classes, or WebSocketStream at module import time. Runtime dependencies now expose host-neutral environment flag and WebSocket stream contracts, the tldw runtime adapter preserves current behavior, AuthNZ websocket verification failures fail closed without allowing MCP-JWT fallback for AuthNZ tokens, and focused tests cover the boundary, contract, fallback, and WebSocket test-hygiene paths.
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
