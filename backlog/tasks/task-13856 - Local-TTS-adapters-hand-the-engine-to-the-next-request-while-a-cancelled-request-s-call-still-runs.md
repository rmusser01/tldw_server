---
id: TASK-13856
title: Local TTS adapters hand the engine to the next request while a cancelled request's
  call still runs
status: Done
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
- [x] #1 A shared helper in core/TTS/utils.py runs an engine call on a worker thread and releases the caller's lock only when the thread returns, including for a stream that holds the lock across chunks, and NeuTTS uses it
- [x] #2 Each local adapter that holds a lock around a thread-offloaded engine call uses the helper, with a test that cancels a request mid-call (directly and through an anyio cancel scope) while another waits, and checks the second call does not start until the first returns
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed: core/TTS/utils.py gains hold_tts_engine(lock), an async context manager that holds a local engine's asyncio lock and yields run(func, *args, **kwargs), which runs one call on a worker thread (through run_tts_blocking_call) under asyncio.shield. On exit it releases the lock at once or, when the holder leaves while its last call still runs on its thread, from that call's completion (TASK-13855's NeuTTS pattern, lifted; its finally never awaits, because anyio re-cancels a cancelled holder at every await). One context manager covers single calls, holds spanning several calls, and streams (await run(next, iterator, sentinel)), so there is no separate stream variant. Adopted at every lock held around a thread-offloaded call in the local adapters: NeuTTS (_engine_call and the stream, replacing _release_engine_when_done), PocketTTS generate (pocket_tts_adapter.py:317) and stream (:523-536), LuxTTS (luxtts_adapter.py:298-314), Supertonic (:313), Supertonic2 (:322), the OmniVoice runtime's synthesis (omnivoice_runtime.py:113) and load (:59-85), the Qwen3 MLX model load (qwen3_runtime_mlx.py:133-136) and the Qwen3 pipeline load (qwen3_tts_adapter.py:638-641). Listed sites verified: qwen3_runtime_mlx.py and qwen3_tts_adapter.py hold their lock around a model load only (their generation runs unlocked); the shape and the fix are the same. Three unlisted sites have the shape and are fixed: VibeVoice (vibevoice_adapter.py:963-1059, where a cancelled generate let the next request's variant reload free the model under the running thread), the Echo-TTS model load (echo_tts_adapter.py:479-517) and the KittenTTS runtime load (kitten_tts_adapter.py:94-110). Kokoro's _model_lock is filed as TASK-13860: its own wait_for(init_timeout_sec) abandons a running load thread, so a second load can start. A cancelled model load's result is discarded, so the next caller loads again after it (ponytail comment in the helper). Plan: Docs/superpowers/plans/2026-10-18-tts-cancellation-registry.md (PR A of two; committed with TASK-13857's PR, which merges first; this PR then needs neutts_adapter.py's import line to read from ..utils import hold_tts_engine, parse_bool, run_tts_blocking_call).
Tests: tests/TTS/unit/test_tts_engine_lock_cancellation.py, 24 cases (12 sites, each with a plain task cancel and an anyio cancel scope), every one RED on dev with assert 2 == 1 (the next call's engine call started while the cancelled one still ran) and GREEN after. Deviations from the plan's test code, for the test-quality ratchet (tests/CI/test_test_quality_ratchet.py, which passes): _Gate.enter raises RuntimeError instead of asserting its release guard (an assert inside a fake class is tautology_suspect), and the LuxTTS and VibeVoice fakes are wired in helpers (_luxtts_adapter, _vibevoice_adapter), as PocketTTS's already were, rather than inside the test functions (stub_injection). Mutations: releasing at cancel fails all 24 plus 4 NeuTTS tests; awaiting the thread in the holder's finally fails exactly the 12 anyio cases plus the NeuTTS anyio and error-log tests; dropping the exception retrieval fails test_neutts_cancelled_request_engine_error_is_not_logged_raw. test_local_adapter_async_offload.py now checks that VibeVoice offloads through hold_tts_engine. tests/TTS, tests/TTS_NEW, tests/Docs, the two other test files that reference touched modules and the ratchet: 7 failed, 2111 passed, 28 skipped, 1 xfailed, 1 xpassed; the 7 are the plan's baseline (No module named 'av'; torch not installed). New file and test_neutts_adapter.py: 44 passed in three runs. Ruff: no new findings. Bandit -ll (uvx): clean (only the existing nosec B615 warning at vibevoice_adapter.py:610). ADR assessment: none needed; a local fix inside the TTS adapters, no durable rule changes (ADR-011 governs adapter init failure retry, untouched). No docs describe these locks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A cancelled TTS request (a client disconnect) no longer hands a local engine to the next request while its call still runs on a worker thread. The completion-released lock NeuTTS got in TASK-13855 is now hold_tts_engine in core/TTS/utils.py, used by NeuTTS, PocketTTS, LuxTTS, Supertonic, Supertonic2, VibeVoice and the OmniVoice runtime for synthesis, and by the OmniVoice runtime, Qwen3 MLX, the Qwen3 pipeline, Echo-TTS and KittenTTS for model loads. Kokoro's init_timeout_sec abandons a running load and is tracked separately (TASK-13860).
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
