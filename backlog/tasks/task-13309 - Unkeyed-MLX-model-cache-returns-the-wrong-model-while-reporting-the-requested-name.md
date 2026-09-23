---
id: TASK-13309
title: >-
  Unkeyed MLX model cache returns the wrong model while reporting the requested
  name
status: Done
assignee: []
created_date: '2026-09-22 04:53'
updated_date: '2026-09-23 21:08'
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
- [x] #1 Nemo and Parakeet-ONNX caches guarded by a module lock with re-check inside
- [x] #2 Test asserts two different model_path values return different objects
- [x] #3 MLX cache keyed on (resolved model_id, resolved cache_dir)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PARTIAL. AC#1 done: _mlx_model_cache is now a dict keyed on (model_path or default, cache_dir); read, write and the unload path all updated. Three test cache-resets updated from "= None" to "= {}" (the reset idiom for the new representation - assertions unchanged, nothing disabled) and a discrimination regression test added. 26 passed.
STILL OPEN: AC#2 - Audio_Transcription_Nemo._model_cache and Audio_Transcription_Parakeet_ONNX._onnx_model_cache are keyed but UNLOCKED (check-then-act); they still need a module lock with the re-check inside, per the Qwen3ASR:269-315 pattern.

2026-09-23 reconciliation (no ACs checked):
AC1 NOT checked - substantively fixed but the AC as written is not met. Commit 7c348a05ae made _mlx_model_cache a dict keyed on (model_path or _DEFAULT_MLX_MODEL_ID, cache_dir) (Audio_Transcription_Parakeet_MLX.py:46,346). dtype is NOT in the key; the AC premise is partly wrong: load_parakeet_mlx_model has no mlx_dtype parameter (signature: force_reload, model_path, cache_dir, allow_download, execution_route) and dtype is hardcoded bfloat16, so a dtype key component would be constant. Also the key uses the raw arg, not the resolved model_id (config mlx_model_id) / resolved cache dir. Recommend amending AC1 to (resolved model_id, cache_dir) or keying on the resolved values.
AC2 NOT met - no threading.Lock anywhere in Audio_Transcription_Nemo.py (_model_cache :55, check-then-act at :477, :565) or Audio_Transcription_Parakeet_ONNX.py (_onnx_model_cache :57, :835). model_utils.py has no cached_model helper.
AC3 NOT met - test_parakeet_mlx.py::test_mlx_cache_discriminates_by_model_path only writes two sentinels into the dict and reads them back; it never calls load_parakeet_mlx_model with two model_path values, so it would still pass if the loader ignored model_path when computing the key. Need a loader-level test (fake parakeet_mlx.from_pretrained returning distinct objects per model_id).
Also: tests/Audio/test_stt_execution_plan_local.py:200,208,1116 still reset _mlx_model_cache to None / an object(), leaving a non-dict in the module global after teardown; latent order-dependence (a later loader call would hit None.get). Verification: test_parakeet_mlx.py + test_stt_execution_plan_local.py 90 passed.

2026-09-23 completion (commit 765610dcb5 on top of 0eacb750fd):
AC amended: original AC1 'MLX cache keyed on (model_id, dtype, cache_dir)' replaced by 'MLX cache keyed on (resolved model_id, resolved cache_dir)'. The backlog CLI has no in-place AC edit, so it was removed and re-added and now sits at #3. dtype is excluded because load_parakeet_mlx_model has no dtype parameter: it always passes mx.bfloat16 (when supported), so a dtype key component would be a constant. Add it only if a dtype parameter is ever introduced.
MLX (#3): config resolution (mlx_model_id / mlx_cache_dir fallbacks) moved ahead of the cache lookup; cache_key = (model_id, model_cache_dir), exactly what from_pretrained receives.
Nemo/ONNX (#1): Audio_Transcription_Nemo gets module-level _model_cache_lock (threading.Lock) around the three check-then-load sites (_load_controlled_nemo_model, load_canary_model, load_parakeet_model), with a re-check inside the lock and the lock-free fast path for hits kept. Audio_Transcription_Parakeet_ONNX gets _onnx_model_cache_lock; the load body moved unchanged into _load_parakeet_onnx_uncached, called under the lock after a re-check. One lock per module (ponytail comment notes that loads of different models serialise too). The Audio/model_utils.cached_model helper was NOT built: two plain locks are a smaller diff than a helper plus rewiring three differently shaped loaders.
Tests (#2): test_parakeet_mlx.py::test_mlx_loader_cache_discriminates_by_model_path (replaces the dict-poking test) stubs parakeet_mlx.from_pretrained to return a new object per id, loads two model_path values and asserts different objects, same path twice returns the same object, factory called twice. test_mlx_loader_cache_keys_on_resolved_config_model_id covers the resolved-key fix. test_nemo_transcription.py::test_concurrent_canary_loads_call_factory_once: two threads, factory called once, same object.
Red before / green after: with HEAD sources, the Canary concurrency test fails (assert 2 == 1) and the resolved-config test fails (same object returned); the model_path test fails against the pre-7c348a05ae single-slot loader (7c348a05ae^) and passes on HEAD. All 3 pass with the fix.
Also fixed the test leak from the earlier reconciliation: test_stt_execution_plan_local.py fixture now .clear()s _mlx_model_cache and the planned-MLX test uses monkeypatch.setattr with a dict, so no None/object() is left in the module global.
Suite (tests/Audio + every test file that imports these modules, 12 targets): HEAD 1742 passed / 58 failed / 14 errors / 17 skipped; after 1746 passed / 56 failed / 14 errors / 17 skipped. The after failure set is a strict subset of HEAD's (no new failures). The two that now pass are TestParakeetMLX::test_model_loading and test_model_loading_with_custom_path, which had been broken by the None-cache leak. The remaining failures are pre-existing and unrelated (hotwords, TTS policy, WS, persona whisper errors, and nemo/onnx tests that need the real toolkits).
Bandit: uvx bandit -q -ll on the three touched source files reported no issues (only pre-existing nosec B615 notices).
Docs: no user-facing behaviour change, so no doc update needed.

2026-09-23 follow-up ef819cbc1f: the MLX cache left unlocked in the close-out now has _mlx_model_cache_lock with re-check; test_mlx_loader_concurrent_same_key_loads_once red on HEAD (2 loads), green now; test_parakeet_mlx.py 28 passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Parakeet MLX cache now keys on the resolved (model_id, cache_dir) the loader actually loads, so a config model-id change no longer serves the old model. Nemo and Parakeet-ONNX model caches are guarded by a module threading.Lock with a re-check inside it, so concurrent misses load a 1-3 GB model once. Loader-level MLX tests and a two-thread Canary test were shown red before and green after; no new failures in the audio/STT suite (1746 passed vs 1742 on HEAD). AC1 was amended to drop dtype, which is hard-coded bfloat16 and not a parameter. The cached_model helper was not built; per-module locks were smaller. Known open item: the MLX cache is still unlocked, which was out of AC scope.
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
