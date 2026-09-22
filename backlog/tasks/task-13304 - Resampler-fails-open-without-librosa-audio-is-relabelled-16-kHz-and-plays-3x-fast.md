---
id: TASK-13304
title: >-
  Resampler fails open without librosa; audio is relabelled 16 kHz and plays 3x
  fast
status: To Do
assignee: []
created_date: '2026-09-22 04:52'
labels:
  - bug
  - audio
  - transcription
dependencies: []
references:
  - >-
    tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Buffered_Transcription.py:534
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Audio/Audio_Buffered_Transcription.py:_resample` returns its input **unchanged** when librosa is missing:

```python
def _resample(self, audio, orig_sr, target_sr):
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        logger.warning("librosa not available, returning original audio")
        return audio
```

Both callers then set the rate **unconditionally**:

```python
if sample_rate != 16000:
    audio_data = self._resample(audio_data, sample_rate, 16000)
    sample_rate = 16000        # <- asserted regardless of whether resampling happened
```

Verified at `:385-386` and `:753-754`.

**Effect:** without librosa installed, 48 kHz audio is declared to be 16 kHz. The transcriber receives samples 3x too dense: the transcript is garbage and every timestamp is 3x short. The request returns **HTTP 200** with a warning buried in the logs.

Three of six resamplers in the module fail open this way; `Audio_Transcription_Lib.py:336-355` handles it correctly.

Found by the comprehensive core-module review; independently verified by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failing test simulates a missing librosa and asserts the pipeline does not claim 16 kHz for un-resampled audio
- [ ] #2 _resample raises or returns a signal the caller must handle, rather than returning input unchanged
- [ ] #3 Both callers only set sample_rate = 16000 when resampling actually occurred
- [ ] #4 The other two fail-open resamplers in the module are corrected in the same pass
- [ ] #5 A missing resampling dependency surfaces as an error to the client, not a warning in the log
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
