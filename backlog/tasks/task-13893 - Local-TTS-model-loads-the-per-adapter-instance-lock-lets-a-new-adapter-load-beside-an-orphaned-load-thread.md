---
id: TASK-13893
title: 'Local TTS model loads: the per-adapter-instance lock lets a new adapter load
  beside an orphaned load thread'
status: To Do
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
- [ ] #1 Two adapter instances for the same provider and model path never run their model loads at the same time, including after a caller is cancelled or times out mid-load, with a test for a plain task cancel and an anyio cancel scope
- [ ] #2 The hung-ONNX bound and the loop-bound Kokoro PyTorch fallbacks are each fixed or recorded as accepted by the owner
- [ ] #3 The dead no-voices ONNX branch in KokoroAdapter._initialize_onnx is removed
- [ ] #4 _assert_cancelled_call_keeps_the_engine no longer relies on fixed sleep margins to show the second caller has not started
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
