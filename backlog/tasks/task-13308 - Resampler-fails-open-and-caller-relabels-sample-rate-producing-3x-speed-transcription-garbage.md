---
id: TASK-13308
title: >-
  Resampler fails open and caller relabels sample rate producing 3x-speed
  transcription garbage
status: Done
assignee: []
created_date: '2026-09-22 04:53'
updated_date: '2026-09-23 20:40'
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
- [x] #1 Resample either returns audio at the requested rate or raises
- [x] #2 Unconditional sample_rate relabels removed at both call sites
- [x] #3 Test runs the buffered transcriber with librosa absent
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PARTIAL. Audio_Buffered_Transcription._resample now honours its post-condition: returns audio AT target_sr, falling back to linear interpolation when librosa is absent (matching Audio_Transcription_Lib._resample_audio_if_needed) rather than returning the input unchanged. Contract test added at tests/Audio/test_buffered_transcriber_resample_contract.py: red before, green after; 56 existing audio tests still pass.
STILL OPEN: the three other fail-open resamplers named in the finding - Audio_Streaming_Unified._resample_audio_if_needed (282-308), Audio_Transcription_Qwen3ASR._maybe_resample (192-210), Audio_Transcription_VibeVoice._maybe_resample (179-196) - and the shared Audio/audio_resample.py extraction.

2026-09-23 reconciliation: AC1 met for the buffered transcriber - commit 7c348a05ae; Audio_Buffered_Transcription._resample (:534) falls back to np.interp linear resampling when librosa is absent instead of returning the input; tests/Audio/test_buffered_transcriber_resample_contract.py asserts 48k->16k yields ~1/3 length with librosa import blocked (2 passed). Of the other resamplers the description counts: Qwen3ASR/VibeVoice _maybe_resample return the TRUE (unchanged) rate on failure, so they do not mislabel (premise weaker than stated, as long as callers use the returned rate - not checked here); Audio_Streaming_Unified._resample_audio_if_needed still returns input unchanged if np.interp raises (unlikely path). AC2 NOT met - the relabels 'sample_rate = 16000' remain at :386 and :769 right after _resample; they are now TRUTHFUL because of the post-condition, so the bug is closed, but the AC as written (remove/derive the rate) is not done. AC3 NOT met - the contract test calls _resample unbound; no test runs BufferedTranscriber / the :704-769 entry point end-to-end with librosa absent and checks chunk timing. Remaining: either close AC2 as superseded by the post-condition or derive sample_rate from the resampler; add an end-to-end buffered-transcriber test with librosa blocked and a stub model. Bandit on Audio_Buffered_Transcription.py: no findings. Note: status is still To Do although work landed; left as is.

2026-09-23 completion (commit a28ba8f5fe): AC2 - both 'sample_rate = 16000' lines after _resample removed (Audio_Buffered_Transcription.py process_audio and the mlx structured branch of transcribe_long_audio); the variable was dead after that point, so _resample alone now determines the rate. AC3 - tests/Audio/test_buffered_transcriber_resample_contract.py::test_transcribe_long_audio_48k_without_librosa_keeps_real_timing[onnx|mlx] runs transcribe_long_audio on a 12 s 48 kHz WAV with sys.modules['librosa']=None and stub ONNX/MLX models; asserts 192000 resampled samples, 4 chunks/model calls, last chunk end 12.0 s, mlx token starts < 12 s. Red against 7c348a05ae~1 (576000 samples) and green now. Loose end - Audio_Streaming_Unified._resample_audio_if_needed now raises ValueError instead of returning the input when np.interp fails (both callers already catch the noncritical tuple); test_streaming_resample_raises_instead_of_returning_wrong_rate, red before/green after. Suite (tests/Audio + 9 other files importing the touched modules): before 52 failed/1668 passed/12 skipped/14 errors, after 52 failed/1671 passed/12 skipped/14 errors, identical failure id set (pre-existing, env-related: Persona whisper, hotwords, TTS policy etc.). uvx bandit -q -ll on both sources: no findings. Docs: none needed (internal behaviour). Not done here: Qwen3ASR/VibeVoice _maybe_resample callers' use of the returned rate was not audited; shared audio_resample.py extraction not attempted.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resamplers in the buffered transcriber and the unified streaming path now either return audio at the target rate or raise; the unconditional sample_rate relabels are gone. End-to-end test with librosa blocked proves 48 kHz input is chunked and timestamped at its true 12 s duration (fails against pre-fix code). Commits 7c348a05ae (resample post-condition) and a28ba8f5fe (relabels removed, streaming resampler raises, e2e test). Open: Qwen3ASR/VibeVoice caller audit and a shared resample helper were out of scope.
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
