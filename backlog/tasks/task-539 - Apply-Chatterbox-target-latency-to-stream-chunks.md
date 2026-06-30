---
id: TASK-539
title: Apply Chatterbox target latency to stream chunks
status: Done
labels:
- tts
- chatterbox
- streaming
- config
references:
- https://github.com/devnen/Chatterbox-TTS-Server
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py
- tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Use ChatterboxAdapter target_latency_ms / chatterbox_target_latency_ms for actual progressive stream chunk duration instead of hardcoded 0.2 second chunks in TTS and voice-conversion streaming paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Add failing adapter tests proving configured target_latency_ms controls chunk_duration_sec for Chatterbox TTS streaming and voice-conversion streaming, implement a small duration helper used by both stream_encoded_waveform calls, then verify with focused tests, the full adapter mock suite, Bandit, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
ChatterboxAdapter now applies target_latency_ms / chatterbox_target_latency_ms to the actual stream_encoded_waveform chunk duration used by both TTS streaming and voice-conversion streaming. Added red/green coverage showing configured 125 ms latency maps to 0.125 second chunks in both paths. Verification: focused tests failed before the fix and passed after it; full Chatterbox adapter mock suite passed; Bandit on chatterbox_adapter.py reported no findings; git diff --check passed.
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
