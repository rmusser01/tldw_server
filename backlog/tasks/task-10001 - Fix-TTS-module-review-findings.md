---
id: TASK-10001
title: Fix TTS module review findings
status: Done
assignee: []
created_date: '2026-06-23 21:55'
updated_date: '2026-06-23 22:09'
labels:
  - tts
  - security
  - review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the current-code review findings in tldw_Server_API/app/core/TTS: VibeVoice non-stream config propagation, VibeVoice shared model-state races, blocking local TTS generation in async paths, ffmpeg subprocess timeouts, voice-reference conversion failures, realtime websocket egress policy enforcement, TTS config secret redaction, and stale Anthropic env mapping.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VibeVoice non-stream generation forwards request generation config and guards shared model state.
- [x] #2 Local blocking generation calls are offloaded from the event loop.
- [x] #3 Audio conversion subprocesses are bounded by timeouts.
- [x] #4 Voice-reference conversion errors surface instead of silently using raw bytes.
- [x] #5 Realtime websocket TTS URLs pass central egress policy checks before connecting.
- [x] #6 TTS config export redacts provider API keys by default and removes non-TTS Anthropic key mapping.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: Add focused regression coverage for reviewed TTS defects.
Stage 2: Repair VibeVoice config propagation, model-state locking, and async offload boundaries.
Stage 3: Bound audio converter subprocesses and surface voice-reference conversion failures.
Stage 4: Enforce realtime websocket egress policy and redact TTS config secrets by default.
Stage 5: Run focused tests, Bandit on touched TTS scope, and record verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Touched production files: tldw_Server_API/app/core/TTS/utils.py, tts_config.py, audio_utils.py, audio_converter.py, adapters/vibevoice_adapter.py, adapters/vibevoice_realtime_adapter.py, adapters/chatterbox_adapter.py, adapters/dia_adapter.py, adapters/higgs_adapter.py, adapters/kokoro_adapter.py. Added/updated focused tests under tldw_Server_API/tests/TTS and tldw_Server_API/tests/TTS_NEW.

Verification passed: focused 10-test regression set; nearby 43-test TTS utility/config/VibeVoice suite; 77-test Chatterbox/Higgs/Kokoro mock/flow suite; py_compile on touched production TTS files; git diff --check on touched scope; Bandit touched production TTS scope wrote /tmp/bandit_tts_review_fixes.json with errors=[] and results_count=0. No separate user-facing docs were needed for these internal hardening fixes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the TTS review findings: VibeVoice now forwards non-stream generation config and serializes model variant reload/generation state; local blocking generation calls in Chatterbox, Dia, Higgs, VibeVoice, and Kokoro stream iteration now run through thread offload helpers; ffmpeg/ffprobe calls in audio_converter are timeout-bounded; voice-reference conversion opts into strict failures; VibeVoice realtime websocket sessions enforce central egress policy before connecting; TTS config export redacts provider API keys by default and ignores ANTHROPIC_API_KEY; touched-scope Bandit warnings were cleaned up.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused regression tests cover the repaired review findings.
- [x] #8 Bandit runs on touched TTS production scope.
<!-- DOD:END -->
