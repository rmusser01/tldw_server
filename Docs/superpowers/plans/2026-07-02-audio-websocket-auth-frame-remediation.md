# Audio WebSocket Auth Frame Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remediate `TASK-12137` by making browser audio WebSocket clients authenticate with the backend-supported initial auth frame instead of default-rejected query-string tokens.

**Architecture:** Add a shared frontend audio WebSocket auth helper that builds bare audio WebSocket URLs and sends `{"type":"auth","token":"..."}` as the first frame after `open`. Update the affected TTS, STT, and voice-chat clients to call that helper before sending prompt, config, or audio frames. Add focused tests that prove default query-token-disabled backend behavior and frontend first-frame ordering.

**Tech Stack:** TypeScript, React hooks, Vitest, FastAPI audio WebSocket auth helpers, pytest, Bandit.

**Backlog:** `TASK-12137`

**Audit References:**
- `AUDIT-2026-06-27-WEBUI-002`
- `AUDIT-2026-06-27-APIWEB-001`

---

## File Map

- Create `apps/packages/ui/src/services/tldw/audio-websocket-auth.ts`: shared frontend helper for bare audio WebSocket URLs and auth-frame sending.
- Create `apps/packages/ui/src/services/__tests__/audio-websocket-auth.test.ts`: helper-level Vitest coverage for URL construction and first-frame auth.
- Modify `apps/packages/ui/src/services/tldw/voice-conversation.ts`: return a bare voice-chat WebSocket URL instead of embedding `?token=`.
- Modify `apps/packages/ui/src/hooks/useVoiceChatStream.tsx`: send the auth frame before the existing config/audio flow.
- Modify `apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx`: open the bare TTS route and send auth before the prompt payload.
- Modify `apps/packages/ui/src/entries/background.ts`: open the bare STT route and send auth before forwarding microphone audio.
- Modify focused frontend tests near existing call sites as needed, including `apps/packages/ui/src/services/__tests__/voice-conversation.test.ts` and `apps/packages/ui/src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx`.
- Modify `tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py`: expand backend auth contract coverage for TTS, STT, and voice chat default query-token rejection plus auth-frame acceptance.

## Stage 1: Prove The Contract With Failing Tests

**Goal:** Capture the intended backend and frontend contract before production code changes.

**Success Criteria:** Tests fail because frontend helpers/clients still embed `?token=` or do not send auth first, and backend coverage demonstrates all three audio routes reject query tokens by default.

**Tests:**
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py -k "audio_ws" -v`
- `cd apps/packages/ui && npm test -- src/services/__tests__/audio-websocket-auth.test.ts src/services/__tests__/voice-conversation.test.ts src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx`

**Status:** Complete

- [x] Add backend test cases for `/api/v1/audio/stream/transcribe`, `/api/v1/audio/chat/stream`, and `/api/v1/audio/stream/tts`.
- [x] Add helper/client Vitest expectations for bare URLs and `auth` as the first frame.
- [x] Run the focused tests and record the expected failures in `TASK-12137`.

## Stage 2: Add Shared Frontend Auth Helper

**Goal:** Centralize audio WebSocket URL and first-frame auth behavior.

**Success Criteria:** A small helper builds audio WebSocket URLs without query tokens and sends a JSON auth frame consistently.

**Tests:**
- `cd apps/packages/ui && npm test -- src/services/__tests__/audio-websocket-auth.test.ts`

**Status:** Complete

- [x] Implement `buildAudioWebSocketUrl(baseUrl, path)` so it preserves the existing HTTP-to-WS base conversion and rejects token-in-query construction.
- [x] Implement `sendAudioWebSocketAuthFrame(ws, token)` so callers can authenticate immediately in `onopen`.
- [x] Keep the helper intentionally narrow to the audio WebSocket routes in scope.

## Stage 3: Update TTS, STT, And Voice Chat Clients

**Goal:** Replace query-token audio WebSocket clients with the shared auth-frame flow.

**Success Criteria:** Speech playground TTS, extension STT, and voice chat all open bare routes and send auth before prompt, config, or audio frames.

**Tests:**
- `cd apps/packages/ui && npm test -- src/services/__tests__/audio-websocket-auth.test.ts src/services/__tests__/voice-conversation.test.ts src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx`

**Status:** Complete

- [x] Update `voice-conversation.ts` and `useVoiceChatStream.tsx` so voice chat builds a token-free URL and sends auth before config.
- [x] Update `SpeechPlaygroundPage.tsx` so TTS sends auth before prompt.
- [x] Update `background.ts` so extension STT sends auth before audio frames.
- [x] Adjust focused tests to assert no audio WebSocket URL contains `token=`.

## Stage 4: Verify, Document, And Commit

**Goal:** Prove the remediation and record remaining risk.

**Success Criteria:** Focused frontend/backend tests, Bandit for touched Python production scope or a documented no-production-Python result, and `git diff --check` are recorded in `TASK-12137`.

**Tests:**
- `cd apps/packages/ui && npm test -- src/services/__tests__/audio-websocket-auth.test.ts src/services/__tests__/voice-conversation.test.ts src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx`
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py -k "audio_ws" -v`
- `source .venv/bin/activate && python -m bandit -r <touched_python_production_paths> -f json -o /tmp/bandit_audio_ws_auth.json`
- `git diff --check`

**Status:** Complete

- [x] Update `TASK-12137` with verification results, touched files, residual risks, and final summary.
- [x] Mark completed acceptance criteria and Definition of Done items that are satisfied by the evidence.
- [x] Commit implementation changes without pushing.
