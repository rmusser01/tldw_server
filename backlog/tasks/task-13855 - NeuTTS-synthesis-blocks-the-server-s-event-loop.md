---
id: TASK-13855
title: NeuTTS synthesis blocks the server's event loop
status: Done
labels:
- tts
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
NeuTTSAdapter calls the engine synchronously inside its async methods: generate runs encode_reference and infer on the event loop (core/TTS/adapters/neutts_adapter.py:250,260), and generate_stream runs encode_reference and iterates infer_stream there (:344,:361). The other local adapters offload through run_tts_blocking_call / run_tts_blocking_next in core/TTS/utils.py. Since TASK-13849 the long-form TTS jobs worker runs inside the server by default, so a NeuTTS job freezes that server process for the whole synthesis: other requests and the job's own lease renewal stall with it. Found in the clean-session review of PR #3262.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 NeuTTS generate runs encode_reference and infer off the event loop, and a test fails when either runs on the loop thread
- [x] #2 NeuTTS generate_stream encodes the reference and pulls each infer_stream chunk off the event loop, and a test fails when any runs on the loop thread
- [x] #3 Docs/Operations/Env_Vars.md says the in-process TTS jobs worker shares the per-process TTS generation limit (performance.max_concurrent_generations, default 4) with /audio/speech
- [x] #4 Engine calls never overlap, including after a request is cancelled mid-call (a client disconnect), and loading the model does not block the loop
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed: NeuTTSAdapter loads the model through run_tts_blocking_call, runs encode_reference and infer on worker threads, and pulls infer_stream chunks through run_tts_blocking_next (core/TTS/utils.py). On worker threads, concurrent requests would share one engine: a llama-cpp context is not thread-safe, and nothing limits NeuTTS per provider (only the global performance.max_concurrent_generations of 4). So _engine_lock serialises engine calls, and a stream holds it until it ends, since llama-cpp keeps the stream's state in the shared context. asyncio.to_thread cannot stop a thread, so a request cancelled mid-call (Starlette cancels a disconnected stream through an anyio cancel scope) must not hand the engine on while its call still runs: _engine_call and the stream release the lock from the call's own completion (_release_engine_when_done), not from the cancelled caller. Waiting for the thread inside the cancelled caller, as the clean review first suggested, does not hold: anyio re-cancels that wait (probed: the wait left with the thread still running). Env_Vars.md and its Published mirror say the in-process TTS jobs worker takes one of the process's performance.max_concurrent_generations slots (default 4) that /audio/speech uses; tts_jobs_worker calls get_tts_service_v2().generate_speech, which holds that semaphore. First filed as TASK-13854 on an unpushed branch; renumbered because PR #3263 took that id.
Tests (tests/TTS/adapters/test_neutts_adapter.py): test_neutts_runs_engine_calls_off_the_event_loop RED on dev (load and all five engine calls on the loop thread); test_neutts_runs_one_engine_call_at_a_time RED without the lock (peak 4 concurrent calls); test_neutts_cancelled_request_keeps_the_engine_until_its_call_returns (generate, stream, stream through an anyio cancel scope) RED with the lock released by the cancelled caller (a second call started); test_neutts_stream_keeps_the_engine_between_chunks pins the stream-long lock. Eight mutations each fail a test: lock released at cancel, the wait-on-cancel variant (fails only the anyio case), stream locked per chunk, encode_reference unlocked, infer unlocked, model loaded on the loop, infer on the loop, stream on the loop. The file gives 19 passed in three runs. Every test that references NeuTTS plus tests/Docs: 292 passed, 4 skipped, 1 failed (tests/TTS_NEW/unit/test_tts_service.py::TestTextGeneration::test_chunked_retry_metadata, No module named 'av'). tests/TTS, tests/TTS_NEW and tests/Docs: 7 failed, 2062 passed, 28 skipped; all 7 fail on unchanged dev too (av and torch are not installed locally). Ruff clean on the adapter; the test file's I001 and B010 are on dev already. Bandit -ll clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
NeuTTS no longer blocks the server's event loop: model loading, reference encoding, synthesis and each streamed chunk run on worker threads, one engine call at a time, and a cancelled request keeps the engine until its call returns. Env_Vars.md notes that in-process long-form TTS jobs share the per-process TTS generation slots with /audio/speech. Follow-up TASK-13856 covers the same cancellation gap in the other local adapters.
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
