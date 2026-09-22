---
id: TASK-13309
title: >-
  Unkeyed MLX model cache returns the wrong model while reporting the requested
  name
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
    tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_MLX.py:43
  - >-
    tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Nemo.py:55
  - >-
    tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Qwen3ASR.py:50
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_mlx_model_cache is a bare Optional[Any], not a dict. load_parakeet_mlx_model takes model_path / cache_dir / mlx_dtype but only force_reload affects the lookup, so a request for parakeet-tdt-1.1b hits the truthy check at :343, logs "Using cached Parakeet MLX model", and receives the previously loaded 0.6b model while reporting the 1.1b name upstream.

Separately Nemo (_model_cache:55) and Parakeet_ONNX (:57) are keyed but UNLOCKED - check-then-act with no threading.Lock anywhere in either file - so two concurrent Canary requests both miss and from_pretrained runs twice, leaving two 1-3 GB models resident. STT batch adapters and the OCR thread pool both dispatch concurrently.

Six correct examples exist in-repo (Qwen3ASR:269-315 and the five OCR _load_transformers). Destination: Audio/model_utils.py cached_model(key, factory).

Source: synthesis F11
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 MLX cache keyed on (model_id, dtype, cache_dir)
- [ ] #2 Nemo and Parakeet-ONNX caches guarded by a module lock with re-check inside
- [ ] #3 Test asserts two different model_path values return different objects
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PARTIAL. AC#1 done: _mlx_model_cache is now a dict keyed on (model_path or default, cache_dir); read, write and the unload path all updated. Three test cache-resets updated from "= None" to "= {}" (the reset idiom for the new representation - assertions unchanged, nothing disabled) and a discrimination regression test added. 26 passed.
STILL OPEN: AC#2 - Audio_Transcription_Nemo._model_cache and Audio_Transcription_Parakeet_ONNX._onnx_model_cache are keyed but UNLOCKED (check-then-act); they still need a module lock with the re-check inside, per the Qwen3ASR:269-315 pattern.
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
