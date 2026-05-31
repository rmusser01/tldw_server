---
id: TASK-432
title: Fix Audio usage-event test lifecycle timeout in full suite
status: Done
labels:
- ci
- audio
- tests
- pr-1846
priority: High
modified_files:
- tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py
- tldw_Server_API/tests/Audio/test_audio_usage_events.py
- tldw_Server_API/tests/Audio/test_http_quota_validation.py
- tldw_Server_API/tests/Audio/test_stream_limits_endpoint.py
- tldw_Server_API/tests/Audio/test_stream_status_endpoint.py
- tldw_Server_API/tests/Audio/test_transcript_segmentation_endpoint.py
- tldw_Server_API/tests/Audio/test_ws_concurrent_streams.py
- tldw_Server_API/tests/Audio/test_ws_diarization_persistence_status.py
- tldw_Server_API/tests/Audio/test_ws_failopen_runtime.py
- tldw_Server_API/tests/Audio/test_ws_idle_metrics_audio.py
- tldw_Server_API/tests/Audio/test_ws_invalid_json_error.py
- tldw_Server_API/tests/Audio/test_ws_metrics_audio.py
- tldw_Server_API/tests/Audio/test_ws_pings_audio.py
- tldw_Server_API/tests/Audio/test_ws_quota.py
- tldw_Server_API/tests/Audio/test_ws_quota_close_toggle.py
- tldw_Server_API/tests/Audio/test_ws_quota_compat_and_close.py
- tldw_Server_API/tests/Audio/test_ws_vad_turn_detection.py
- tldw_Server_API/tests/Audio/ws_test_helpers.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #1846 full-suite CI is failing during the backend Audio module. Local reproduction shows tldw_Server_API/tests/Audio/test_audio_usage_events.py::test_tts_usage_event_logged passes alone but times out when run after the Audio suite prefix because the test starts the full FastAPI TestClient app just to verify audio.tts usage logging. Replace that fragile lifecycle-heavy unit test with a direct endpoint-function test using fake dependencies, preserving behavior coverage while avoiding app startup cross-test contamination.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Audio full-suite no longer hangs/cancels during FastAPI TestClient shutdown.
- [x] Narrow Audio endpoint tests avoid starting the full app lifecycle when direct endpoint-function coverage is sufficient.
- [x] Unified streaming WebSocket does not emit duplicate terminal full_transcript frames after an auto/manual commit with no new transcript state.
- [x] Focused and full Audio verification are recorded.
- [x] Bandit/diff hygiene checks are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Converted lifecycle-heavy HTTP endpoint tests to direct endpoint-function calls with fake request/user/service dependencies.
- Added `ws_client_without_lifespan` for Audio WebSocket route tests that need routing but not FastAPI app startup/shutdown.
- Suppressed duplicate terminal `full_transcript` frames by tracking whether transcript state changed since the last full transcript emission.
- Verification:
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Audio/test_ws_vad_turn_detection.py -vv --tb=short --timeout=30` -> 9 passed, 1 skipped.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Audio -x -vv --tb=short --timeout=60` -> 468 passed, 4 skipped.
  - `git diff --check` -> passed.
  - Bandit production touched file -> passed.
  - Bandit touched test scope -> expected low-severity pytest assert noise plus pre-existing test try/continue patterns.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reworked Audio tests that were starting the full FastAPI lifespan for narrow endpoint/WebSocket assertions so the full Audio suite no longer hangs during shutdown. Added a no-lifespan WebSocket TestClient helper for route-level websocket tests and converted lifecycle-heavy HTTP tests to direct endpoint-function calls with fake dependencies. Fixed the unified streaming handler to suppress duplicate terminal full_transcript frames after an auto/manual commit when no transcript state changed, while preserving legacy stop-only final transcript emission. Verification: focused VAD file passed; full Audio suite passed with 468 passed, 4 skipped. Diff check passed. Bandit production touched file passed; whole touched test scope only reported expected low-severity pytest assert noise and pre-existing test try/continue patterns.
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
