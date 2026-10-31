---
id: TASK-13893
title: 'Local TTS model loads: the per-adapter-instance lock lets a new adapter load
  beside an orphaned load thread'
status: Done
labels:
- tts
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13856 gave the local TTS adapters hold_tts_engine (core/TTS/utils.py), and TASK-13860 (PR #3291) moved Kokoro's model load onto it: a cancelled or timed-out caller leaves the adapter's lock with the load thread until it returns. The lock is per adapter instance, so a new adapter instance gets a fresh lock and can start a second load of the same model beside a thread still running on an earlier one. Three paths do that today, shared by every hold_tts_engine site, not just Kokoro: (1) request-owned adapters from TTSAdapterRegistry.create_adapter_with_overrides (one per request carrying provider overrides, tts_service_v2._get_adapter); (2) an unload or re-registration (a generation bump) while a load thread still runs; (3) a registry retry after adapter_failure_retry_seconds (300 in tts_providers_config.yaml) when the timed-out load's thread is still running. None is new in #3291; dev already loaded beside orphaned threads on these paths. The fix named in the TASK-13860 notes and #3291's clean review (Minor 3) is a per-model-path lock (keyed on provider and model path, shared across instances), held through hold_tts_engine the same way. Also in scope, from the same review, for a decision or a fix: (a) a hung ONNX Kokoro load holds _model_lock with no bound, because init_timeout_sec is PyTorch-only (one condition in KokoroAdapter._ensure_model_loaded); (b) Kokoro's loop-bound PyTorch work, i.e. the sync fallback after a failed load thread, the generic torch.jit.load/torch.load path and the device move, runs on the event loop under the lock: it cannot overlap a load but blocks the loop; (c) review Minor 5: the no-voices ONNX branch in KokoroAdapter._initialize_onnx (the else of 'if voices_json_arg', with its TypeError fallback) is dead code, since voices_json_arg is always a path or the method raises first, so delete it; (d) review Minor 6: _assert_cancelled_call_keeps_the_engine in tests/TTS/unit/test_tts_engine_lock_cancellation.py checks the second caller after fixed sleep(0.05)/sleep(0.1) margins, which can only let a regression pass on a slow runner; check the second caller's structural state instead (for example that it is waiting on the lock). Source: the clean-session review of PR #3291 (Minor 3, 5 and 6), summarised in a PR comment there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Two adapter instances for the same provider and model path never run their model loads at the same time, including after a caller is cancelled or times out mid-load, with a test for a plain task cancel and an anyio cancel scope
- [x] #2 The hung-ONNX bound and the loop-bound Kokoro PyTorch fallbacks are each fixed or recorded as accepted by the owner
- [x] #3 The dead no-voices ONNX branch in KokoroAdapter._initialize_onnx is removed
- [x] #4 _assert_cancelled_call_keeps_the_engine no longer relies on fixed sleep margins to show the second caller has not started
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed (AC1): core/TTS/utils.py gains tts_model_load_lock(provider, model_key), one asyncio.Lock per (event loop, provider, model key) in a WeakValueDictionary, shared by every adapter instance and held through hold_tts_engine, so a cancelled or timed-out load keeps it until its thread returns. Adopted at all 11 in-process sites that load a model on a worker thread: Kokoro (_ensure_model_loaded), Echo-TTS (_ensure_models_loaded), KittenTTS (_load_runtime_for_model, keyed per model name), Qwen3 pipeline (_get_pipeline) and Qwen3 MLX (_get_model), which swap their per-instance lock; and LuxTTS, PocketTTS, NeuTTS, IndexTTS2, Supertonic and Supertonic2, whose initialize() loaded with a bare asyncio.to_thread and no lock at all (two concurrent override requests loaded the same model at once even without a cancel). Keys are provider_key plus the model identity each adapter gives the resource manager. Not changed: OmniVoice (its runtime lives in the sidecar process, one per process, behind one shared supervisor), and VibeVoice, Chatterbox, Dia and Higgs, which load on the event loop, so no load can overlap; that each of those loads stalls the server is filed as TASK-13908. AC2a (owner decision 1A, approved 2026-10-23): init_timeout_sec now bounds ONNX callers too. AC2b: Kokoro's PyTorch device move and generic torch.jit.load/torch.load path run on the worker thread, and the sync fallback that re-ran a failed load on the loop is gone. AC3: the unreachable no-voices ONNX branch in _initialize_onnx is deleted. AC4: _assert_cancelled_call_keeps_the_engine takes the held lock and asserts, once the cancel has unwound, that the next caller is queued on it (asyncio.Lock._waiters), instead of sleeping fixed margins.
Tests (tests/TTS/unit/test_tts_engine_lock_cancellation.py, measured on this branch from origin/dev 617094d014, whose TTS code matches the plan's e06eeec68c): 48 -> 76 passed. New: 24 cross-instance cases (12 sites x task cancel / anyio scope), test_kokoro_load_on_a_new_adapter_waits_for_a_timed_out_load (the registry-retry path), test_kokoro_onnx_load_honours_init_timeout_sec (replaces the ONNX-ignores-the-bound pin), test_kokoro_failed_pytorch_load_is_not_retried_on_the_event_loop, test_kokoro_pytorch_load_runs_off_the_event_loop x2. RED before each fix: Task 1 25 failed, 48 passed (24 assert 2 == 1, 1 assert ['load', 'load'] == ['load']); Task 3 1 failed (TimeoutError: the ONNX caller waited past its bound); Task 4 3 failed (two KModel attempts, the second on the loop; .cpu() on the loop; jit.load/eval on the loop). AC4 evidence: with a mutant that releases the lock at cancel and starts every engine thread 0.3 s late, the old helper gave 4 failed, 69 passed (all 66 helper-based cases passed); the new one gives 70 failed, 3 passed (all 66 fail with assert 2 == 1). Mutations: release at cancel fails 70 of 76; a fresh lock per call (the per-instance lock again) fails 54 (24 cross-instance, the timed-out case, 26 single-instance load cases, 3 past-timeout); each of the four Kokoro mutants (ONNX exemption, sync fallback, device move on the loop, generic load on the loop) fails only its own test. Stable: 76 passed in 3 sequential runs and one -n 4 run. The 40 test files that reference a touched adapter: 3 failed, 412 passed, 12 skipped (dev: 3 failed, 384 passed, 12 skipped; the 3 are ModuleNotFoundError: av on both). Test-quality and private-coercion ratchets: 5 passed. Ruff on the 13 touched files: no new findings (5 pre-existing in kitten_tts_adapter.py and qwen3_runtime_mlx.py; the whole core/TTS directory plus the test file gives 65 on dev and after). Bandit -ll (uvx) on the 12 touched product files: clean. ADR assessment: none needed; a process-local concurrency fix through the existing hold_tts_engine helper, ADR-011 untouched. Docs: none describe the load locks; Env_Vars.md lists KOKORO_INIT_TIMEOUT_SEC by name only; refresh_docs_published.sh gives no diff. Known limits: the lock is per process and per event loop, keyed on configured strings (a relative and an absolute path to one file are two keys); a hung load now stalls every instance of that model, with a caller bound only for Kokoro (opt-in); a cancelled load's model is still discarded and loaded again after it; one Kitten instance or Qwen3 MLX runtime can now load different models at once (at most the 4 pinned Kitten repos plus the configured one, at most 2 MLX backend ids), so its transient memory peak is the cached model plus every model loading, where dev loaded one at a time per instance. VibeVoice, Chatterbox, Dia and Higgs still load on the event loop (no overlap is possible, but each load stalls the server): TASK-13908. Follow-ups: TASK-13905, TASK-13906, TASK-13907, TASK-13908.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A local TTS model load now holds one lock per model, shared by every adapter instance, until its load thread returns. A request-owned override adapter, a re-registration or a registry retry can no longer load a model beside another instance's running or orphaned load, across all 11 adapters that load on a worker thread (6 of which had no load lock at all). Kokoro's init_timeout_sec now bounds ONNX callers too, its PyTorch load runs wholly off the event loop, and its dead no-voices ONNX branch is gone. The shared test helper checks the waiting caller's state, not elapsed time.
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
