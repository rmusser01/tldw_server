---
id: TASK-13305
title: >-
  Unkeyed MLX model cache returns the wrong model while reporting the requested
  name
status: Done
assignee: []
created_date: '2026-09-22 04:52'
updated_date: '2026-09-22 14:28'
labels:
  - bug
  - audio
  - transcription
dependencies: []
references:
  - >-
    tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_MLX.py:43
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Audio/Audio_Transcription_Parakeet_MLX.py:43` declares a single global cache that is **not keyed by model id**:

```python
_mlx_model_cache: Optional[Any] = None
_DEFAULT_MLX_MODEL_ID = "mlx-community/parakeet-tdt-0.6b-v3"
```

The load path at `:343` returns it regardless of what was asked for:

```python
if _mlx_model_cache and not force_reload:
    return _mlx_model_cache
```

**Effect:** the first model loaded in the process wins for every subsequent request. Ask for `parakeet-tdt-1.1b` after `0.6b` has been cached and you receive the **0.6b** model, while the log says "Using cached model" and the requested name is reported upstream. Transcription quality silently differs from what the caller selected and what the response claims — with no error at any layer.

The only escape is `force_reload`, which callers do not set per model.

Related, same class: the `Nemo` and `Parakeet_ONNX` caches **are** keyed but unlocked (check-then-act), so two concurrent Canary loads can resident two 1-3 GB models simultaneously.

Six correctly-keyed model caches already exist elsewhere in this repo to copy from.

Found by the comprehensive core-module review; independently verified by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failing test loads model A then requests model B and asserts B is returned, not A
- [ ] #2 The cache is keyed by model id (and any other parameter that changes the loaded artifact)
- [ ] #3 The reported model name always matches the model actually used
- [ ] #4 The keyed-but-unlocked Nemo and Parakeet_ONNX caches gain a lock so concurrent loads cannot double-resident
- [ ] #5 Cache eviction still works with the keyed structure
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed test-first and merged to dev in PR #2980 (merge commit 8045fa2956). A failing test reproduced the defect before any code changed, with controls pinning the behaviour that had to stay unchanged. Qodo review then found follow-on defects in three of this batch's fixes; those were corrected in the same PR before merge.
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
