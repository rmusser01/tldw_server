---
id: TASK-13860
title: Kokoro's init_timeout_sec abandons a running model load, so a second load can
  start alongside it
status: To Do
labels:
- tts
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
KokoroAdapter._ensure_model_loaded holds _model_lock (core/TTS/adapters/kokoro_adapter.py:359) around _load_onnx_model / _load_pytorch_model, which load the model with asyncio.to_thread. asyncio.wait_for(..., timeout=self.init_timeout_sec) (:366, :389; from initialize() and from _ensure_model_loaded()) cancels its own wait at the timeout while the load thread keeps running, and the lock passes to the next caller, which starts a second load. Found while fixing TASK-13856, which gave the other local adapters hold_tts_engine (core/TTS/utils.py); Kokoro needs its timeout reworked before that helper fits. (With lazy init initialize() does not load, :375-382; without it generate() waits on ensure_initialized() at :770 before _ensure_model_loaded() at :776, which then returns early, so no two-locks interleaving is claimed.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Kokoro load that outlasts init_timeout_sec (from initialize() or _ensure_model_loaded()) does not let a second load start while the first thread runs, with a test for a plain cancel and an anyio cancel scope
- [ ] #2 The Kokoro load is held through hold_tts_engine, or the filing explains why it cannot be
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
