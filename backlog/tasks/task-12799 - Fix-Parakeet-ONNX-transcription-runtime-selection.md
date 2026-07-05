---
id: TASK-12799
title: Fix Parakeet ONNX transcription runtime selection
status: Done
labels:
- audio
- transcription
- onnx
- parakeet
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the Parakeet ONNX transcription failure where the runtime loads the wrong ONNX graph from a multi-graph RNNT export and raises missing input feed errors such as required inputs ['encoder_outputs', 'input_states_1', 'input_states_2'] missing from ['targets', 'target_length'].
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Loader downloads Parakeet ONNX sidecars needed for runtime and decoding, including external data files and vocab/config metadata.
- [x] Multi-graph Parakeet TDT exports load through upstream `onnx-asr` instead of selecting the first `.onnx` file.
- [x] Transcription dispatches multi-graph bundles through the upstream adapter and does not feed waveform/features into decoder-only inputs such as `targets`.
- [x] Existing single-session ONNX fallback remains available for non-bundle exports.
- [x] Verification includes automated tests, Bandit, and a no-mock local ONNX Runtime smoke.
- [x] Multi-graph Parakeet TDT decoding uses upstream `onnx-asr` instead of a locally maintained greedy TDT decoder.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased PR #2524 onto latest dev and addressed active review comments. Existing local Parakeet artifact directories no longer trigger Hugging Face downloads, missing TDT bundle vocab fails closed with a clear log, upstream bundle chunking now honors middle overlap trimming, and new review tests have explicit type hints.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Fixed the Parakeet ONNX transcription runtime selection issue by treating official Parakeet TDT exports as upstream `onnx-asr` model bundles instead of arbitrary single ONNX sessions.
- Removed the locally maintained greedy TDT decoder path; graph bundles now use `onnx_asr.load_model("nemo-conformer-tdt", path=..., quantization=...)` behind a small project adapter.
- Rebased PR #2524 onto current `dev` and addressed active review comments: existing local artifact directories are not passed as Hugging Face repo ids, missing bundle `vocab.txt` fails closed with a clear log, upstream bundle chunking honors `merge_algo="middle"`, and new tests have explicit type hints.
- Verified with focused pytest coverage, Bandit, `git diff --check`, and a no-mock upstream smoke against local cached Parakeet graphs. `pip check` still reports an unrelated local venv conflict between `typer-slim 0.24.0` and `typer 0.16.1`.
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
