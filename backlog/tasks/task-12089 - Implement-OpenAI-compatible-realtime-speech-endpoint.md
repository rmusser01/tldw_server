---
id: TASK-12089
title: Implement OpenAI-compatible realtime speech endpoint
status: Done
assignee: []
created_date: ''
updated_date: '2026-09-10 03:29'
labels:
  - audio
  - realtime
  - implementation
dependencies: []
references:
  - TASK-12088
  - 'https://github.com/rmusser01/tldw_server/pull/2572'
  - 'https://github.com/huggingface/speech-to-speech'
  - 'https://developers.openai.com/api/docs/guides/realtime'
  - >-
    https://developers.openai.com/api/docs/guides/realtime-conversations#handling-audio-with-websockets
documentation:
  - Docs/superpowers/reviews/2026-09-09-pr2572-rebase-review.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the adapter-first OpenAI GA Realtime-compatible speech-to-speech WebSocket endpoint plan, including protocol/capabilities, session orchestration, auth and route integration, default pipeline adapter, docs, tests, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch includes latest dev with conflicts resolved without losing current behavior
- [x] #2 Confirmed review defects are fixed with regression coverage and unsupported suggestions are explained
- [x] #3 Focused realtime and shared audio/auth regressions pass; touched code passes Bandit and diff checks
- [x] #4 Merge usefulness, limitations, CI state and human Change summary gate are documented
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

2026-09-09 PR 2572 follow-up: rebased onto origin/dev 40345571a2; three already-applied design commits skipped and both documentation conflicts resolved preserving Voice Chat v1. Original head saved at codex/pr2572-before-rebase-20260909. Reviewing current-dev applicability and all existing inline comments; regression tests and Bandit required before completion. Backlog MCP workflow read/search did not respond, using official CLI fallback.

Earlier implementation history and touched-file inventory preserved during CLI normalization:

Stage 4 complete. Added DefaultRealtimePipeline with injected STT/chat/TTS dependencies, lazy production factory construction, PCM16-to-WAV STT adapter wrapper, streaming/non-streaming chat delta normalization, realtime TTS session integration, buffered fallback coverage, typed pipeline events, and stage-specific RealtimePipelineError wrapping. Wired native and OpenAI-compatible realtime routes to the default pipeline factory through a handler helper that preserves no-arg fake factories. Updated BufferedRealtimeSession to propagate target_sample_rate from realtime config extras into OpenAISpeechRequest. Initial Stage 4 code-quality review found abnormal-exit cleanup leaks and fragile realtime TTS opener kwargs; fixed by adding cleanup regressions, close/abort/cancel cleanup with audio-task cancellation, BufferedRealtimeSession.close() for aborting uncommitted buffered text, and signature-filtered open_realtime_session kwargs. Verification: red test first failed with ModuleNotFoundError for default_pipeline.py; focused default pipeline tests passed (7 passed, 3 warnings); Stage 3 realtime WebSocket regression passed (8 passed, 3 warnings); post-fix focused suite passed (20 passed, 5 warnings); code-quality re-review passed with no Critical or Important findings; Bandit touched production scope reported errors=[] results=0; git diff --check passed.

Stage 5 complete. Documented the OpenAI-compatible realtime speech routes, handshake auth behavior, audio contract, supported client/server events, explicit Stage 1 unsupported features, and tldw quota semantics. Updated the latency PRD to record the Stage 1 route support and deferred latency/interruption benchmarks, accepted the design spec, completed TASK-12088, and added an opt-in provider-backed live smoke marker that is skipped unless `TLDW_REALTIME_LIVE_SMOKE=1` plus explicit STT/LLM/TTS provider env vars are set. Verification: live smoke marker collection reported 1 skipped; focused realtime suite passed (96 passed, 3 warnings); route/config regression suite passed (11 passed, 4 warnings); Bandit on touched implementation paths plus TTS realtime session wrote `/tmp/bandit_audio_realtime.json` with errors=[] and results=0; git diff --check passed.

Pre-PR review follow-up complete. Rebased the feature branch onto current `origin/dev` so the PR diff is limited to realtime/task/doc files. Fixed review blockers by making response generation cancellable through the WebSocket receive loop, adding an active-generation cancellation integration test, rejecting unimplemented `response.create` and `session.modalities` overrides instead of silently accepting them, adding explicit beta `input_audio_format` rejection, validating session scalar field types, exposing persistence/deferred feature metadata in capabilities, serializing capabilities with `asdict`, splitting oversized TTS audio chunks before protocol serialization, splitting chat and TTS provider hints, and correcting the opt-in live smoke session shape. Verification: focused realtime suite passed (110 passed, 3 warnings); route/config regression suite passed (11 passed, 4 warnings); Bandit wrote `/tmp/bandit_audio_realtime_reviewfix_final.json` with errors=[] and results=0; git diff --check passed.

PR opened against `dev`: https://github.com/rmusser01/tldw_server/pull/2572. PR body records the human-authored Change summary merge-gate requirement for this AI-authored change.

Original modified files:
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

2026-09-09 follow-up complete: reviewed all 23 inline comments and current-dev integration; fixed confirmed streaming/backpressure, cancellation, lifecycle, metadata, cookie-auth, quota, credential-policy, redaction and dialogue-context defects. Independent final review found no additional must-fix findings. Focused suite: 140 passed, 1 intentional live-provider smoke skip. Shared regression slice: 92 passed. Ruff/Black, repository guards, compilation, diff check and Bandit passed (0 findings, 0 errors). Review report documents applicability, unsupported suggestions, experimental protocol limits, historical CI failures and the human Change summary merge gate. Main checkout preserved. Temporary follow-up plan completed and its outcome retained in the report. Remote current-head CI remains a pre-merge condition.

2026-09-09 requested follow-up: rebase onto latest dev again, wait for Qodo reviews on the resulting PR head, address all verified findings and CI issues, then request merge approval. User supplied a Change summary in the task conversation. Continuing isolated worktree and official Backlog CLI fallback; no merge is authorized yet.

Qodo follow-up implementation: rebased cleanly onto dev 456eafb7a6; Qodo confirmed the prior code findings resolved and retained three live-smoke policy findings. Moved real-provider verification to an explicit standalone command with a spoken-WAV input and server-configured providers; replaced skipped pytest smoke with deterministic fake-transport tests. Independent review identified oversized WAV frames, reproduced with a seven-second clip and fixed via chunking plus a 30-second input limit. Final focused suite: 150 passed, no skips. Refreshed the OpenAPI fingerprint after reproducing CI contract drift (2086 -> 2087 paths; no schema count changes) and regenerated ignored frontend types. Ruff/Black, compilation, repository guards, diff check and Bandit passed. Awaiting posted current-head Qodo/CI before asking for merge approval.

Final Qodo follow-up validation: d61c9b8573 received Qodo zero bugs/zero rule violations and all 50 CI checks passed (26 workflow skips). Local backend unit smoke: 403 passed; deployment-shaped startup smoke passed. Dev advanced during CI to f0248aaa00 through PR 2613 (three task documents and one audio-download test only), so the PR was rebased again without conflicts. Tree comparison confirms production code, manual smoke helper, and API fingerprint are unchanged from the fully green head. Focused realtime/route suite plus updated download regression: 153 passed, no skips; Bandit zero findings/errors. Refreshed-head checks run again before any merge. User merge decision and implementation-rationale sentence remain pending. Temporary follow-up plan outcomes are retained in the report; the plan file was removed.

2026-09-10 merge instruction: requester supplied the Change summary and explicitly authorized merging PR 2572; no further approval or rationale is pending. GitHub strict status checks rejected the first merge because dev advanced through PR 2599 (audio.cpp TTS). Rebased cleanly onto dev 6b61b5074c. Feature implementation and review fixes are complete; the requested merge will proceed after GitHub required checks on the updated head.

Final merge freshness update: dev advanced again through PR 2612 to 751563a966 (media original-file cleanup). Rebased cleanly and verified the combined realtime, route, audio.cpp registry, audio-download, and original-storage regression slice: 201 passed, 24 warnings. Bandit on realtime production/manual-helper scope again reported zero findings and errors. Requester merge authorization remains in effect; finishing the required GitHub checks and merge.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation and Qodo review of PR 2572 are complete; the requester supplied the Change summary and authorized merge. Branch includes dev 751563a966, with 201 focused and inherited-media regression tests passing and zero Bandit findings. Current-head GitHub required checks are the remaining merge prerequisite. Live-provider interoperability and latency remain unverified.
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
