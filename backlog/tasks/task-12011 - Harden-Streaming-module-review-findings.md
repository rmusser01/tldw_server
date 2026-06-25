---
id: TASK-12011
title: Harden Streaming module review findings
status: Done
assignee: []
created_date: '2026-06-24'
updated_date: '2026-06-24'
labels:
  - streaming
  - security
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the accepted code review findings for `tldw_Server_API/app/core/Streaming`, focusing on the current module code rather than git diffs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Speech chat action execution fails closed unless an explicit allowlist is configured.
- [x] #2 Audio request size and format limits are checked before decoding/parsing audio content.
- [x] #3 Speech chat STT/LLM configuration fields are passed through to the underlying providers where supported.
- [x] #4 TTS exception mapping does not return internal paths or raw provider details to clients.
- [x] #5 SSE provider control-line filtering honors the configured passthrough/filter policy.
- [x] #6 WebSocket stream lifecycle handles accept failures and terminal states without leaked background tasks.
- [x] #7 Focused regression tests and touched-scope security scan pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Manual Backlog task file created because the Backlog MCP was unavailable and the CLI hung on search/list/create operations. The user approved a manual task-file exception for this review fix.

Implemented:
- Speech chat actions now require `AUDIO_CHAT_ALLOWED_ACTIONS` to explicitly include the requested action before any MCP module lookup occurs.
- Speech chat validates declared input format before base64 decoding and enforces byte size before parsing audio with soundfile.
- Speech chat passes supported STT model configuration via `whisper_model` and forwards guarded LLM extra params while blocking internal/security-sensitive keys.
- Speech chat TTS validation and voice-reference failures now return generic client details instead of raw exception text.
- SSE raw provider lines now use the shared provider-line normalizer and honor provider control passthrough/filter settings.
- WebSocket streams now propagate real accept failures, tolerate known already-accepted Starlette state errors, and stop ping/idle background tasks on terminal frames.
- SSE heartbeat enqueue uses forced enqueue to avoid deadlock under a full queue.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the Streaming module review findings and added focused regression coverage.

Verification:
- Initial focused red run: 10 expected failures in speech-chat and stream lifecycle tests.
- Clean-worktree focused final run: `python -m pytest tldw_Server_API/tests/Audio/test_speech_chat_service.py tldw_Server_API/tests/Streaming/test_streams.py -q` -> 52 passed.
- Clean-worktree touched-scope security scan: `python -m bandit -r tldw_Server_API/app/core/Streaming -f json -o /tmp/bandit_streaming_12011_worktree.json` -> 0 findings.

Residual note: the optional audio WebSocket ping checks still fail in the clean worktree even with `MINIMAL_TEST_INCLUDE_AUDIO=1`, because `/api/v1/audio/stream/transcribe` explicitly constructs its `WebSocketStream` with `heartbeat_interval_s=0`. That endpoint-level behavior is outside this `app/core/Streaming` hardening change.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests passing
- [x] #3 Security scan run on touched scope
- [x] #4 Final summary recorded
<!-- DOD:END -->
