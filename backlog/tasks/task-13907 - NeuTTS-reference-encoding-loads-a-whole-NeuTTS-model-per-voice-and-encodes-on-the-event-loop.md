---
id: TASK-13907
title: NeuTTS reference encoding loads a whole NeuTTS model per voice and encodes
  on the event loop
status: To Do
labels:
- tts
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
VoiceManager._encode_neutts_reference (core/TTS/voice_manager.py:1236-1257 on dev e06eeec68c) builds a new NeuTTSAdapter for every reference it encodes, awaits ensure_initialized() (a full model load; after TASK-13893 it queues behind any other NeuTTS load of the same repos instead of loading beside it), never closes the adapter, and calls adapter._engine.encode_reference on the event loop (:1247). Recorded in the TASK-13856 plan (Review Focus 6) as pre-existing and out of scope; never filed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reference encoding reuses the registry's NeuTTS adapter, or closes the one it builds
- [ ] #2 encode_reference runs off the event loop
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
