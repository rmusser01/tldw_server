---
id: TASK-12934
title: Fix audio WebSocket legacy config rejection on main CI
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-09 16:14'
labels:
  - ci
  - audio
  - websocket
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Main CI run 28994210037 is failing media-audio and media-ingestion-modification shards because legacy/internal unified audio WebSocket config frames without explicit strict v1 protocol fields are rejected with `protocol_version must be 1` before the transcriber initializes. Prepare a minimal local patch on `codex/fix-main-guardian-notify-ts` and do not push until all tests complete.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Legacy unified WebSocket config frames without explicit protocol fields are accepted as v1 defaults.
- [x] #2 Explicit protocol frames still use strict v1 validation and reject protocol_version 2.
- [x] #3 Targeted direct handler CI-failure tests pass locally.
- [x] #4 Bandit and diff checks are clean.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: handle_unified_websocket applied the strict public v1 audio protocol validator to every first config frame. Legacy/internal unified streaming clients send only transcription settings, so frames without protocol_version/mode/audio_format/channels were rejected before transcriber setup, quota, VAD, or diarization behavior could run. Fix: synthesize strict v1 defaults only when protocol_version is omitted; explicit protocol frames remain strict.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prepared local unpushed commit for main CI audio WebSocket failures. Verification: red test `test_websocket_server_integration` failed before the patch and passed after; direct handler CI-failure set passed (`16 passed, 1 skipped`); strict protocol guards passed (`12 passed`); `git diff --check` passed; Bandit on `Audio_Streaming_Unified.py` reported 0 findings. Local endpoint quota batch was not used as final verification because after this protocol fix it reached macOS `parakeet_mlx` initialization and aborted in the local environment, which is a separate dependency/runtime path from the CI bad_request failures.
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
