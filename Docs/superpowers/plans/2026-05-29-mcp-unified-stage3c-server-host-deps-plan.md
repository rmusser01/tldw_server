# MCP Unified Stage 3C Server Host Dependency Cleanup Plan

Backlog: TASK-543

## Stage 1: Contract Tests
**Goal**: Capture the remaining `server.py` host dependency leaks before implementation.
**Success Criteria**: Focused extraction tests fail for import-boundary, missing runtime exports, WebSocket stream factory injection, and AuthNZ websocket fail-closed behavior.
**Tests**: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py -q`
**Status**: Complete

## Stage 2: Runtime Adapter Seams
**Goal**: Move env/test helper access and WebSocket stream construction behind runtime dependency protocols.
**Success Criteria**: `server.py` no longer imports host testing helpers or `WebSocketStream`; default tldw runtime dependencies preserve current behavior.
**Tests**: Focused MCP Unified extraction and server tests.
**Status**: Complete

## Stage 3: Verification And PR
**Goal**: Verify the slice, update task records, and publish a reviewable PR.
**Success Criteria**: Focused pytest, Ruff, and Bandit pass; Backlog task includes verification evidence; branch is committed, pushed, and PR opened.
**Tests**: Focused pytest set, `ruff check`, and scoped Bandit.
**Status**: Complete
