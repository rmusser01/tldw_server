---
id: TASK-13308
title: >-
  Resampler fails open and caller relabels sample rate producing 3x-speed
  transcription garbage
status: To Do
assignee: []
created_date: '2026-09-22 04:53'
updated_date: '2026-09-22 05:12'
labels:
  - bug
  - ingestion
  - audio
dependencies: []
references:
  - >-
    tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Buffered_Transcription.py:534
  - >-
    tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py:336
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_resample returns the input unchanged on ImportError (:540-541) and the callers at :384-386 and :752-754 then set sample_rate = 16000 UNCONDITIONALLY.

Deploy without librosa (the code treats it as optional): a client streams 48 kHz, the log says "librosa not available, returning original audio", and the next line declares the 48 kHz array to be 16 kHz. Every chunk boundary derived from chunk_samples_at_16k covers one third of the intended audio and the model is fed 48 kHz samples as 16 kHz - 3x-speed garbage, all timestamps 3x short, HTTP 200.

Three of six resamplers fail open this way. Audio_Transcription_Lib.py:336-355 is correct - it linear-interpolates rather than lying. Canonical post-condition: return audio at target_sr, or raise.

Source: synthesis F10
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Resample either returns audio at the requested rate or raises
- [ ] #2 Unconditional sample_rate relabels removed at both call sites
- [ ] #3 Test runs the buffered transcriber with librosa absent
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PARTIAL. Audio_Buffered_Transcription._resample now honours its post-condition: returns audio AT target_sr, falling back to linear interpolation when librosa is absent (matching Audio_Transcription_Lib._resample_audio_if_needed) rather than returning the input unchanged. Contract test added at tests/Audio/test_buffered_transcriber_resample_contract.py: red before, green after; 56 existing audio tests still pass.
STILL OPEN: the three other fail-open resamplers named in the finding - Audio_Streaming_Unified._resample_audio_if_needed (282-308), Audio_Transcription_Qwen3ASR._maybe_resample (192-210), Audio_Transcription_VibeVoice._maybe_resample (179-196) - and the shared Audio/audio_resample.py extraction.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
