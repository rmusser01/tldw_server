---
id: TASK-12914
title: Implement chat audio streaming protocol v1
status: In Progress
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
- [ ] #1 Backend strict parser validates v1 config and normalizes PCM16 to Float32 before handlers.
- [ ] #2 Transcribe and chat websocket endpoints enforce endpoint modes and strict config.
- [ ] #3 Frontend voice chat, dictation, and extension STT send strict JSON PCM16 frames.
- [ ] #4 Focused backend/frontend tests, Bandit, and diff checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete: added strict audio protocol parser and parser unit tests. Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py -q` passed with 10 tests.
Task 2 complete: `/audio/stream/transcribe` now requires a strict v1 post-auth config frame, rejects wrong endpoint modes with 4400, and normalizes PCM16 JSON audio to Float32 before downstream accounting/processing. Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py tldw_Server_API/tests/Audio/test_ws_fallbacks.py tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py -q` passed with 30 tests.
Task 3 complete: enforced strict v1 validation on /api/v1/audio/chat/stream, decodes PCM16 JSON audio through the shared parser before quota/VAD/transcriber handling, added push_to_talk_release commit handling with commit_source payloads, made push-to-talk ignore VAD auto-commit, and replaced obsolete protocol_version=2 behavior tests with strict rejection coverage. Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py tldw_Server_API/tests/Audio/test_ws_fallbacks.py tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py -q` passed with 51 tests and 6 warnings.
Task 4 complete: voice chat now keeps PCM16 capture by default, sends strict v1 top-level config fields, and uses the voice_chat microphone capture owner. Split audio capture owners from audio source preference feature groups so new owners do not imply missing source-preference storage keys. Verification: `cd apps/packages/ui && bun run test src/hooks/__tests__/useMicStream.test.tsx src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx` passed with 25 tests.
Task 5 complete: server dictation now streams PCM16 frames over /api/v1/audio/stream/transcribe with strict dictate config, dictation mic owner, selected-device forwarding, partial preview callbacks, final transcript delivery, and websocket stop cleanup. Composer dictation wiring now keeps server partials separate from final transcript commits. Verification: `cd apps/packages/ui && bun run test src/hooks/__tests__/useMicStream.test.tsx src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx src/hooks/__tests__/useServerDictation.source.test.tsx src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx` passed with 42 tests.
Task 6 complete: browser-extension STT now sends auth, strict captions config, then open, and wraps extension audio chunks as JSON base64 PCM16 frames instead of raw binary. Verification: `cd apps/packages/ui && bun run test src/hooks/__tests__/useMicStream.test.tsx src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx src/hooks/__tests__/useServerDictation.source.test.tsx src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx src/entries/__tests__/background.stt-protocol.test.ts` passed with 43 tests.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
