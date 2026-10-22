---
id: TASK-13860
title: Kokoro's init_timeout_sec abandons a running model load, so a second load can
  start alongside it
status: Done
labels:
- tts
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
KokoroAdapter._ensure_model_loaded holds _model_lock (core/TTS/adapters/kokoro_adapter.py:359) around _load_onnx_model / _load_pytorch_model, which load the model with asyncio.to_thread. Two ways a second load can start alongside a running one. (1) Timeout: asyncio.wait_for(..., timeout=self.init_timeout_sec) (:366, :389; from initialize() and from _ensure_model_loaded()) wraps only the PyTorch load; it cancels its own wait at the timeout while the load thread keeps running, and the lock passes to the next caller. (2) Client disconnect, independent of any timeout: a request cancelled during the lazy load (generate() -> _ensure_model_loaded()) releases _model_lock while the load thread runs, so the next request starts a second load. This includes the ONNX path (:362-363, _initialize_onnx -> to_thread at :511-526), which has no timeout at all. Found while fixing TASK-13856, which gave the other local adapters hold_tts_engine (core/TTS/utils.py); Kokoro needs its timeout reworked before that helper fits. (With lazy init initialize() does not load, :375-382; without it generate() waits on ensure_initialized() at :770 before _ensure_model_loaded() at :776, which then returns early, so no two-locks interleaving is claimed.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Kokoro load that outlasts init_timeout_sec (from initialize() or _ensure_model_loaded()) does not let a second load start while the first thread runs, with a test for a plain cancel and an anyio cancel scope
- [x] #2 The Kokoro load is held through hold_tts_engine, or the filing explains why it cannot be
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed: every Kokoro model load goes through KokoroAdapter._ensure_model_loaded, which holds _model_lock with hold_tts_engine (core/TTS/utils.py) and passes run to _load_onnx_model/_load_pytorch_model -> _initialize_onnx/_initialize_pytorch. Those construct the ONNX Kokoro (kokoro_adapter.py:511/519/526 on dev) and the KModel (:666) through run instead of asyncio.to_thread, so a cancelled or timed-out caller leaves the lock with the load thread until it returns. initialize() now loads through _ensure_model_loaded(); on dev it loaded with no _model_lock at all (:384-394), guarded only by ensure_initialized's _init_lock, which a cancelled caller releases. init_timeout_sec (owner decision: option A) is now an asyncio.timeout around the whole hold: it bounds each caller's wait, queueing behind another load included; a load that outlasts it keeps running and keeps the lock, its model is discarded, and the next caller loads again after it. Still PyTorch-only (ONNX, the shipped backend, ignores it) and off by default. Known limits: the lock is per adapter instance, so a load on a new adapter can run beside a thread still running on an earlier one: request-owned adapters from create_adapter_with_overrides (one per request carrying provider overrides, pre-existing and shared by every local adapter), an unload or re-registration, and a registry retry after adapter_failure_retry_seconds (300 s shipped). The PyTorch sync fallback, the generic torch.load fallback and the device move run on the event loop, so they block it but cannot overlap a load.
Tests: tests/TTS/unit/test_tts_engine_lock_cancellation.py gains 22 Kokoro cases: 18 = {ONNX, PyTorch unbounded, PyTorch init_timeout_sec 30 s} x {ensure_initialized(), initialize(), _ensure_model_loaded()} x {task cancel, anyio cancel scope}; 3 init_timeout_sec cases (0.1 s, PyTorch, one per entry); and test_kokoro_onnx_load_ignores_init_timeout_sec, which pins ONNX ignoring the bound (passes on dev too). RED on dev: 21 failed (18 assert 2 == 1, 3 assert ['load', 'load'] == ['load']; a second load started while the first thread ran), 1 passed (the ONNX pin). GREEN: 22 passed. Mutations: releasing the lock at cancel fails the 21 lock cases; bounding only the load and not the queue wait fails exactly the 3 timeout cases (the second caller waits on the stuck thread); applying the bound to ONNX fails exactly the ONNX pin. test_persona_live_voice_runtime.py:323 now loads through _ensure_model_loaded(); it called _initialize_onnx() with no run. The 12 test files that reference the Kokoro adapter: 208 passed, 8 skipped (dev: 186 passed, 8 skipped). tests/TTS, tests/TTS_NEW, tests/Persona: 7 failed, 3010 passed, 28 skipped, 1 xfailed, 1 xpassed; the 7 fail on unchanged dev too (ModuleNotFoundError: PyAV/torch not installed). Ruff: no new findings (one pre-existing I001 at test_persona_live_voice_runtime.py:86, also on dev). Bandit -ll (uvx): clean. ADR assessment: none needed; a local fix in one adapter with an existing helper, and ADR-011 is untouched. Docs: none describe init_timeout_sec beyond the Env_Vars.md name list; _ensure_model_loaded's docstring states the meaning. Plan reviewed independently before execution; review folded in.
Correction after the clean-session review of PR #3291 (0 Critical, 0 Important, 6 Minor; ready to merge). Minor 1: init_timeout_sec bounds each caller's wait on the model load, including queueing on _model_lock. It does not bound queueing on ensure_initialized's _init_lock: two concurrent ensure_initialized() callers on one uninitialized adapter queue there, so the n-th can wait about n x the bound (production's registry runs one shielded init per adapter, so this does not arise there). The earlier 'queueing included' wording in these notes and in the PR body meant _model_lock only; _ensure_model_loaded's docstring now says so. Minor 2: the bound's expiry is told apart from a TimeoutError raised by the load itself with asyncio.timeout's deadline.expired(), so a load's own TimeoutError (a hub download, say) now propagates even with a bound set, instead of being logged as the bound expiring (test_kokoro_load_timeout_error_is_not_reported_as_the_bound; RED before: DID NOT RAISE). Minor 4: _cleanup_resources now drops kokoro_pt_model and kokoro_pt_pipelines; since initialize() reuses a loaded model, a closed PyTorch adapter otherwise kept its KModel and re-initialised without loading (test_kokoro_close_releases_the_pytorch_model; RED before: the KModel survived close()). Minor 3, 5 and 6 (the per-adapter-instance lock, the hung-ONNX bound, the loop-bound PyTorch fallbacks, the dead no-voices ONNX branch, the helper's margin sleeps) are filed as TASK-13893. After the changes: lock-cancellation file 48 passed (24 Kokoro); the 12 Kokoro test files 210 passed, 8 skipped; test_test_quality_ratchet.py + tests/Docs 213 passed; ruff only the pre-existing I001; bandit -ll clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A Kokoro model load now keeps its lock until its load thread returns, through hold_tts_engine, on both backends and on every load path (ensure_initialized/initialize and the lazy load). Neither a cancelled request nor an init_timeout_sec expiry lets a second load start beside a running one on the same adapter. init_timeout_sec now bounds each caller's whole wait; a load that outlasts it keeps the lock and is loaded again after it.
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
