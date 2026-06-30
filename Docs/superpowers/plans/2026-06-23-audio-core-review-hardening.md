# Audio Core Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify and fix validated security, quota, streaming, tokenizer, and helper-ownership findings from the `tldw_Server_API/app/core/Audio` review.

**Architecture:** Keep `core.Audio` as the owner of shared audio service behavior. Endpoint shims may preserve test/backwards-compat patch points, but they should delegate to core implementations rather than duplicating logic. Fixes must fail closed for unsafe auth and local model-loading policy, while preserving explicit compatibility switches where existing clients may depend on legacy behavior.

**Tech Stack:** Python, FastAPI WebSockets, pytest, Loguru, Backlog.md task `TASK-2407`.

---

## Stage 1: Verify Findings
**Goal**: Confirm each review item still applies to current code.
**Success Criteria**: Each finding is marked validated or not applicable in task notes.
**Tests**: Static call-path checks with `rg` and targeted source reads.
**Status**: Complete

- [x] Check WebSocket auth query-token paths in `tldw_Server_API/app/core/Audio/streaming_service.py`.
- [x] Check quota exception tuples in `tldw_Server_API/app/core/Audio/quota_helpers.py` and their fail-open callers.
- [x] Check `_stream_tts_to_websocket` task cancellation behavior.
- [x] Check tokenizer loader local-only fallback behavior and endpoint-controlled model inputs.
- [x] Check tokenizer malformed payload behavior.
- [x] Check BYOK and fail-open helper duplication between core and audio endpoint shims.

## Stage 2: Write Failing Tests
**Goal**: Add focused tests for validated behavior before production edits.
**Success Criteria**: New tests fail for the expected reasons on current code.
**Tests**: Targeted pytest invocations for audio core tests.
**Status**: Complete

- [x] Add tests proving query-token auth is rejected by default but can be enabled by an explicit compatibility flag.
- [x] Add tests proving `NameError` is not in expected DB/Redis quota exceptions.
- [x] Add a test proving TTS streaming cancels the producer when the WebSocket send path fails.
- [x] Add tests proving tokenizer local-only loading does not fall back to download-capable calls.
- [x] Add tests proving invalid base64, odd raw PCM, and invalid sample rates become HTTP 400 errors.
- [x] Add tests proving endpoint BYOK wrappers delegate to the core helper.

## Stage 3: Implement Fixes
**Goal**: Make the smallest production changes that satisfy the failing tests.
**Success Criteria**: All new tests pass without broad refactors.
**Tests**: Same targeted pytest invocations from Stage 2.
**Status**: Complete

- [x] Gate WebSocket `?token=` auth behind `AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH`.
- [x] Remove `NameError` from quota exception tuples.
- [x] Cancel the TTS producer when the consumer finishes early, while letting normal producer completion drain queued audio.
- [x] Enforce tokenizer local-only policy by failing closed when a backend cannot honor it.
- [x] Convert malformed tokenizer inputs into structured 400 errors.
- [x] Delegate endpoint BYOK helper wrappers to `core.Audio.tts_service._resolve_tts_byok`.

## Stage 4: Verification and Cleanup
**Goal**: Prove the fixes and update tracking records.
**Success Criteria**: Targeted tests, Bandit on touched scope, and task notes are complete.
**Tests**: `python -m pytest` for touched audio tests and `python -m bandit` for touched source files.
**Status**: Complete

- [x] Run targeted pytest commands for new and impacted audio tests.
- [x] Run Bandit on touched source paths.
- [x] Update Backlog task `TASK-2407` with verification results and final summary.
