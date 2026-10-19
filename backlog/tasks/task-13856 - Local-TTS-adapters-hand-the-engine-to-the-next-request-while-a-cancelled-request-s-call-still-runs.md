---
id: TASK-13856
title: Local TTS adapters hand the engine to the next request while a cancelled request's
  call still runs
status: In Progress
labels:
- tts
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Several local TTS adapters guard their engine with an asyncio lock around a thread-offloaded call: async with lock: await asyncio.to_thread(...) or run_in_executor. When the awaiting request is cancelled (a client disconnect; Starlette cancels a disconnected stream through an anyio cancel scope), the lock is released at once while the thread keeps running, so the next queued request starts a second engine call alongside it, which the lock exists to prevent. Found in the clean review of TASK-13855 for pocket_tts_adapter.py:317 and :523-535 (the lock is held across a stream) and luxtts_adapter.py:298-305; the same shape appears in omnivoice_runtime.py, qwen3_runtime_mlx.py, qwen3_tts_adapter.py, supertonic_adapter.py and supertonic2_adapter.py. TASK-13855 fixed NeuTTS by releasing the lock from the call's own completion (NeuTTSAdapter._engine_call / _release_engine_when_done). Waiting for the thread inside the cancelled task does not hold, because anyio re-cancels that wait. More likely now that the long-form TTS jobs worker runs in-process by default (TASK-13849).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A shared helper in core/TTS/utils.py runs an engine call on a worker thread and releases the caller's lock only when the thread returns, including for a stream that holds the lock across chunks, and NeuTTS uses it
- [ ] #2 Each local adapter that holds a lock around a thread-offloaded engine call uses the helper, with a test that cancels a request mid-call (directly and through an anyio cancel scope) while another waits, and checks the second call does not start until the first returns
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
