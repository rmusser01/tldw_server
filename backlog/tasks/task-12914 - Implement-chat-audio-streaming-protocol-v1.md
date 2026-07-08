---
id: TASK-12914
title: Implement chat audio streaming protocol v1
status: Done
labels:
- webui
- audio
- implementation
priority: High
documentation:
- Docs/superpowers/plans/2026-07-08-chat-audio-streaming-protocol-v1.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved v1 protocol plan for WebUI and browser-extension chat audio streaming, dictation, turn detection, and VAD behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend strict parser validates v1 config and normalizes PCM16 to Float32 before handlers.
- [x] #2 Transcribe and chat websocket endpoints enforce endpoint modes and strict config.
- [x] #3 Frontend voice chat, dictation, and extension STT send strict JSON PCM16 frames.
- [x] #4 Focused backend/frontend tests, Bandit, and diff checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete: added strict audio protocol parser and parser unit tests. Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py -q` passed with 10 tests.
Task 2 complete: `/audio/stream/transcribe` now requires a strict v1 post-auth config frame, rejects wrong endpoint modes with 4400, and normalizes PCM16 JSON audio to Float32 before downstream accounting/processing. Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py tldw_Server_API/tests/Audio/test_ws_fallbacks.py tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py -q` passed with 30 tests.
Task 3 complete: enforced strict v1 validation on /api/v1/audio/chat/stream, decodes PCM16 JSON audio through the shared parser before quota/VAD/transcriber handling, added push_to_talk_release commit handling with commit_source payloads, made push-to-talk ignore VAD auto-commit, and replaced obsolete protocol_version=2 behavior tests with strict rejection coverage. Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py tldw_Server_API/tests/Audio/test_ws_fallbacks.py tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py -q` passed with 51 tests and 6 warnings.
Task 4 complete: voice chat now keeps PCM16 capture by default, sends strict v1 top-level config fields, and uses the voice_chat microphone capture owner. Split audio capture owners from audio source preference feature groups so new owners do not imply missing source-preference storage keys. Verification: `cd apps/packages/ui && bun run test src/hooks/__tests__/useMicStream.test.tsx src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx` passed with 25 tests.
Task 5 complete: server dictation now streams PCM16 frames over /api/v1/audio/stream/transcribe with strict dictate config, dictation mic owner, selected-device forwarding, partial preview callbacks, final transcript delivery, and websocket stop cleanup. Composer dictation wiring now keeps server partials separate from final transcript commits. Verification: `cd apps/packages/ui && bun run test src/hooks/__tests__/useMicStream.test.tsx src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx src/hooks/__tests__/useServerDictation.source.test.tsx src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx` passed with 42 tests.
Task 6 complete: browser-extension STT now sends auth, strict captions config, then open, and wraps extension audio chunks as JSON base64 PCM16 frames instead of raw binary. Verification: `cd apps/packages/ui && bun run test src/hooks/__tests__/useMicStream.test.tsx src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx src/hooks/__tests__/useServerDictation.source.test.tsx src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx src/entries/__tests__/background.stt-protocol.test.ts` passed with 43 tests.
Task 7 complete: updated API/product/generated docs and comments to describe strict v1 config and PCM16 JSON audio; updated the design spec status to implemented; found and fixed stale v2 WebSocket integration/metrics tests so strict v1 rejection is covered. Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py tldw_Server_API/tests/Audio/test_ws_fallbacks.py tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py tldw_Server_API/tests/Audio/test_ws_transcribe_control_v2_integration.py tldw_Server_API/tests/Audio/test_ws_metrics_audio.py -q` passed with 57 tests; `cd apps/packages/ui && bun run test src/hooks/__tests__/useMicStream.test.tsx src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx src/hooks/__tests__/useServerDictation.source.test.tsx src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx src/entries/__tests__/background.stt-protocol.test.ts` passed with 43 tests; `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/audio_stream_protocol.py tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py -f json -o /tmp/bandit_chat_audio_protocol_v1.json` passed after replacing a runtime assert with an explicit guard. Known skip: `cd apps/packages/ui && bun run typecheck` failed because `apps/packages/ui` does not define a `typecheck` script.
Diff checks: full-worktree `git diff --check` reports an unrelated pre-existing whitespace issue in `Docs/Design/Tool-Calling.md:6`. Scoped `git diff --check -- <task files>` passed, and `git diff --cached --check` passed before staging.
Note: generated site HTML under `Docs/site` and `Docs/Docs/site` is gitignored in this checkout, so canonical committed documentation changes are in the source docs; the ignored generated copies were patched locally only to clear stale-string scans in the working tree.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented chat audio streaming protocol v1 across backend, WebUI, and browser-extension STT paths. The rollout uses one strict backend parser, existing websocket endpoints, required first post-auth config, endpoint-specific mode allowlists, PCM16 mono 16 kHz JSON wire frames, server-side Float32 normalization before quota/VAD/STT handling, push-to-talk release commits, streaming server dictation with partial preview, and extension captions/STT JSON audio frames. Final cleanup aligned public docs and stale tests with strict v1 behavior.
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
