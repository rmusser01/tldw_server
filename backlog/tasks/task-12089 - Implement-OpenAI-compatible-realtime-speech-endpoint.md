---
id: TASK-12089
title: Implement OpenAI-compatible realtime speech endpoint
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-01 20:55
labels:
- audio
- realtime
- implementation
dependencies: []
references:
- TASK-12088
- https://github.com/huggingface/speech-to-speech
- https://developers.openai.com/api/docs/guides/realtime
- https://developers.openai.com/api/docs/guides/realtime-conversations#handling-audio-with-websockets
documentation:
- Docs/superpowers/specs/2026-07-01-openai-realtime-speech-endpoint-design.md
- Docs/superpowers/plans/2026-07-01-openai-realtime-speech-endpoint-implementation-plan.md
- Docs/Audio_Streaming_Protocol.md
- Docs/Product/Realtime_Voice_Latency_PRD.md
- Docs/superpowers/specs/2026-07-01-openai-realtime-speech-endpoint-design.md
- backlog/tasks/task-12088 - Design-OpenAI-compatible-realtime-speech-endpoint.md
modified_files:
- backlog/tasks/task-12089 - Implement-OpenAI-compatible-realtime-speech-endpoint.md
- Docs/superpowers/plans/2026-07-01-openai-realtime-speech-endpoint-implementation-plan.md
- tldw_Server_API/app/core/Audio/Realtime/__init__.py
- tldw_Server_API/app/core/Audio/Realtime/constants.py
- tldw_Server_API/app/core/Audio/Realtime/models.py
- tldw_Server_API/app/core/Audio/Realtime/protocol.py
- tldw_Server_API/app/core/Audio/Realtime/capabilities.py
- tldw_Server_API/app/core/Audio/Realtime/pipeline.py
- tldw_Server_API/app/core/Audio/Realtime/session.py
- tldw_Server_API/app/core/Audio/Realtime/persistence.py
- tldw_Server_API/app/core/Audio/Realtime/auth.py
- tldw_Server_API/app/core/Audio/Realtime/handler.py
- tldw_Server_API/app/core/Audio/Realtime/default_pipeline.py
- tldw_Server_API/app/core/Audio/streaming_service.py
- tldw_Server_API/app/core/TTS/realtime_session.py
- tldw_Server_API/app/api/v1/endpoints/audio/audio_realtime.py
- tldw_Server_API/app/api/v1/endpoints/realtime_compat.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/api/v1/router_groups/minimal.py
- tldw_Server_API/Config_Files/README.md
- tldw_Server_API/Config_Files/privilege_catalog.yaml
- tldw_Server_API/Config_Files/resource_governor_policies.yaml
- tldw_Server_API/tests/Audio/test_realtime_protocol_adapter.py
- tldw_Server_API/tests/Audio/test_realtime_capabilities.py
- tldw_Server_API/tests/Audio/test_realtime_session.py
- tldw_Server_API/tests/Audio/test_realtime_persistence.py
- tldw_Server_API/tests/Audio/test_realtime_auth.py
- tldw_Server_API/tests/Audio/test_realtime_websocket.py
- tldw_Server_API/tests/Audio/test_realtime_default_pipeline.py
- tldw_Server_API/tests/Audio/test_realtime_live_smoke.py
- tldw_Server_API/tests/Resource_Governance/test_realtime_route_policy.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the adapter-first OpenAI GA Realtime-compatible speech-to-speech WebSocket endpoint plan, including protocol/capabilities, session orchestration, auth and route integration, default pipeline adapter, docs, tests, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-01-openai-realtime-speech-endpoint-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 1 complete. Implemented protocol constants, dataclass command/event models, OpenAI GA protocol parser/serializer, capabilities metadata, and provider-free tests. Verification: baseline focused tests passed before implementation (21 passed); Stage 1 tests passed after fixes (43 passed, 3 warnings); spec compliance review passed; code-quality review passed with no Critical or Important findings. Bandit production Realtime package reported errors=0 results=0. Minor hardening candidate: reject stray top-level beta audio fields consistently across event types.

Stage 2 complete. Implemented provider-free realtime pipeline event protocol, internal session orchestrator, manual audio turn lifecycle, response generation/cancellation guards, stale-output suppression, metadata merging, and optional persistence boundary. Verification: focused Stage 2 session+persistence tests passed locally (19 passed, 3 warnings); implementer reported expanded focused slice 72 passed; spec compliance review passed at HEAD 25c6a585; code-quality re-review passed with no Critical or Important findings; Bandit on tldw_Server_API/app/core/Audio/Realtime reported errors=0 results=0. The final persistence fix snapshots RealtimePersistenceConfig before yielding response.done so late session.update cannot misattribute a completed turn.

Stage 3 complete. Implemented realtime WebSocket auth adapter and handler, native /api/v1/audio/realtime capabilities + WS route, OpenAI-compatible /v1/realtime WS route, audio-realtime router gating in content/minimal groups, audio.realtime privilege, and Resource Governor by_route/by_path policy entries. Refactored _audio_ws_authenticate with allow_initial_auth_message defaulting true so existing audio WS routes keep first-message auth fallback while realtime routes do not consume session.update. Stage 3 handler uses monkeypatchable module-level pipeline/persistence factories and keeps the production pipeline as a clear Stage 4 placeholder. Verification: required focused Stage 3 pytest command passed (12 passed, 3 warnings); regression auth/route toggle slice passed (16 passed, 3 warnings); Bandit on touched production scope completed with errors=0 and findings=0; git diff --check passed.

Stage 3 spec-review fix complete. Removed realtime handler filtering so every internal RealtimeSession event is serialized through to_openai_server_event, including all content_part.added/content_part.done and output_item.done frames. Changed imported /v1 realtime compat router specs in content/minimal groups to tags=("audio-realtime",). Updated Stage 3 tests to assert the full emitted OpenAI event order and compat spec tags. Verification: required Stage 3 pytest command passed (12 passed, 3 warnings); Bandit Stage 3 production scope reported errors=0 results=0; git diff --check passed.

Stage 3 quality-review fix complete. Realtime auth denial now closes pre-accept sockets without direct websocket JSON sends when outer_stream is None, while accepted audio routes with outer_stream retain error JSON behavior. Added dummy and TestClient regressions for close-only 4401 unauthenticated realtime denial and hardened the realtime WebSocket receive timeout helper. Verification: required Stage 3 pytest suite plus tldw_Server_API/tests/Audio/test_audio_streaming_service_core.py passed (20 passed, 3 warnings); Bandit Stage 3 production scope reported errors=0 results=0; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented an adapter-first OpenAI-compatible realtime speech WebSocket surface with native and `/v1/realtime` routes, strict Stage 1 JSON event parsing/serialization, capabilities metadata, authenticated WebSocket handling, route/resource-governor integration, internal realtime session orchestration, optional persistence, default STT -> chat -> TTS pipeline wiring, provider-free regression coverage, documentation, and an opt-in live smoke marker. The design preserves existing custom audio routes while keeping OpenAI wire compatibility at the edge and carrying `generation_id` for future interruption/cancellation hardening.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 4 complete. Added DefaultRealtimePipeline with injected STT/chat/TTS dependencies, lazy production factory construction, PCM16-to-WAV STT adapter wrapper, streaming/non-streaming chat delta normalization, realtime TTS session integration, buffered fallback coverage, typed pipeline events, and stage-specific RealtimePipelineError wrapping. Wired native and OpenAI-compatible realtime routes to the default pipeline factory through a handler helper that preserves no-arg fake factories. Updated BufferedRealtimeSession to propagate target_sample_rate from realtime config extras into OpenAISpeechRequest. Initial Stage 4 code-quality review found abnormal-exit cleanup leaks and fragile realtime TTS opener kwargs; fixed by adding cleanup regressions, close/abort/cancel cleanup with audio-task cancellation, BufferedRealtimeSession.close() for aborting uncommitted buffered text, and signature-filtered open_realtime_session kwargs. Verification: red test first failed with ModuleNotFoundError for default_pipeline.py; focused default pipeline tests passed (7 passed, 3 warnings); Stage 3 realtime WebSocket regression passed (8 passed, 3 warnings); post-fix focused suite passed (20 passed, 5 warnings); code-quality re-review passed with no Critical or Important findings; Bandit touched production scope reported errors=[] results=0; git diff --check passed.

Stage 5 complete. Documented the OpenAI-compatible realtime speech routes, handshake auth behavior, audio contract, supported client/server events, explicit Stage 1 unsupported features, and tldw quota semantics. Updated the latency PRD to record the Stage 1 route support and deferred latency/interruption benchmarks, accepted the design spec, completed TASK-12088, and added an opt-in provider-backed live smoke marker that is skipped unless `TLDW_REALTIME_LIVE_SMOKE=1` plus explicit STT/LLM/TTS provider env vars are set. Verification: live smoke marker collection reported 1 skipped; focused realtime suite passed (96 passed, 3 warnings); route/config regression suite passed (11 passed, 4 warnings); Bandit on touched implementation paths plus TTS realtime session wrote `/tmp/bandit_audio_realtime.json` with errors=[] and results=0; git diff --check passed.

Pre-PR review follow-up complete. Rebased the feature branch onto current `origin/dev` so the PR diff is limited to realtime/task/doc files. Fixed review blockers by making response generation cancellable through the WebSocket receive loop, adding an active-generation cancellation integration test, rejecting unimplemented `response.create` and `session.modalities` overrides instead of silently accepting them, adding explicit beta `input_audio_format` rejection, validating session scalar field types, exposing persistence/deferred feature metadata in capabilities, serializing capabilities with `asdict`, splitting oversized TTS audio chunks before protocol serialization, splitting chat and TTS provider hints, and correcting the opt-in live smoke session shape. Verification: focused realtime suite passed (110 passed, 3 warnings); route/config regression suite passed (11 passed, 4 warnings); Bandit wrote `/tmp/bandit_audio_realtime_reviewfix_final.json` with errors=[] and results=0; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
