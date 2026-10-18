---
id: TASK-13848
title: 'Audio jobs pipeline stops after conversion: the transcribe stage''s queue
  is not allowed'
status: Done
labels:
- audio
- jobs
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev, every POST /api/v1/audio/jobs/submit job stops after audio_convert. The in-process audio worker (app/services/audio_jobs_worker.py) queues the next stage, audio_transcribe, on the queue 'transcribe', which the Jobs audio domain does not allow (JobManager.DOMAIN_ALLOWED_QUEUES has no audio entry for it; standard queues are default, high, low). create_job raises 'Queue transcribe not allowed for domain audio', the worker logs 'Audio worker loop error' and nothing further runs, while the audio_convert job reads completed. Probe (stubbed convert_to_wav, worker run for 2 s, no JOBS_ALLOWED_QUEUES_AUDIO): jobs = audio_convert default completed; error as above. Registering 'transcribe' alone is not enough: only the separate GPU worker container (app/services/audio_transcribe_gpu_worker.py, Dockerfiles/Dockerfile.audio_gpu_worker) reads that queue, and the in-process worker, which has an audio_transcribe handler, reads only default. The existing tests miss it because they create audio_transcribe jobs on default by hand, and the opt-in pipeline smoke test asserts nothing. Found while fixing TASK-13641.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A job submitted to /api/v1/audio/jobs/submit is transcribed by the in-process audio worker when no GPU worker is configured, with a test that runs the real hand-off from audio_convert
- [x] #2 A deployment that runs the GPU transcription worker can still route audio_transcribe to it, and the queue it uses is allowed for the audio domain
- [x] #3 A failure to queue the next stage fails the job visibly instead of leaving the previous stage completed and the job stranded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the clean-session review of PR #3259 (TASK-13641): core/TTS/utils.tts_jobs_queue() refuses default and transcribe because the audio pipeline workers read those queues without a job-type filter; the names are literals copied from audio_jobs_worker.py and audio_transcribe_gpu_worker.py. Any queue this task routes the pipeline through must be added to that refusal set (or the names shared as constants), or the guard goes stale and a TTS queue could again be one an unfiltered worker reads.
Fixed in app/services/audio_jobs_worker.py: _next_stage_queue() puts every next stage on default, except audio_transcribe on transcribe when AUDIO_JOBS_GPU_TRANSCRIBE is set (the owner's choice: in-process by default, GPU by opt-in flag where the audio jobs worker runs); transcribe is registered for the audio domain. The worker queues the next stage before completing the current one, with idempotency key audio-stage:<job id>:<next type>; a failed enqueue logs a warning and fails the current job with next_stage_enqueue_failed, permanently only when the queue is refused (a quota or transient error is retried). JobManager.create_job/admit_job gained fair_share_exempt (default False), which the hand-off passes: the next stage replaces the running job, which would otherwise count against its own hand-off under JOBS_MAX_PER_USER (the plan review's probe: with a limit of 1 the job failed). The scope-predicate line baseline was regenerated (7 manager.py entries moved, none added). The flag is a boolean rather than a queue name, so tts_jobs_queue()'s refusal of default and transcribe stays complete. Tests: RED 5 failed, 1 skipped on dev, GREEN 5 passed, 1 skipped; with the exemption switched off only test_audio_pipeline_runs_under_a_per_user_job_limit_of_one fails; every test file touching the changed code plus tests/AudioJobs, tests/Jobs, tests/Docs and tests/lint 1 failed, 2737 passed, 7 skipped. Ruff: only the two I001 in test_audio_worker_smoke.py that dev has. Bandit -ll: none on the worker; only the pre-existing B608 in Jobs/manager.py. Docs/API-related/Audio_Jobs_API.md and Docs/Operations/Env_Vars.md document the routing, the flag, the GPU prerequisites, the upgrade step and next_stage_enqueue_failed (mirrored). Upgrade: a GPU deployment that relied on JOBS_ALLOWED_QUEUES_AUDIO=transcribe must now set AUDIO_JOBS_GPU_TRANSCRIBE=true as well; Jobs whose audio_convert completed before the deploy have no next stage and must be re-submitted.
Whole-branch review fix: fair_share_exempt skipped the entire fair-share block in admit_job, including the priority adjustment, so an exempt hand-off kept priority 5 while new submissions got the lowered one (1 in the probe) and every new first stage ran before any in-flight later stage. It now gates only the can_submit refusal; the priority adjustment still applies. Test tests/Jobs/test_fair_share_exempt.py: RED 1 failed (assert 5 == 1), GREEN 1 passed. Pipeline tests 5 passed, 1 skipped; lint test_scope_predicate_ratchet.py 13 passed after regenerating the baseline (manager.py entries only).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The audio jobs pipeline transcribes in the in-process worker by default; AUDIO_JOBS_GPU_TRANSCRIBE=true routes transcription to the GPU worker's transcribe queue, now registered for the audio domain. Each stage queues the next before completing, under an idempotency key and exempt from per-user fair-share, so a failed hand-off fails the job with next_stage_enqueue_failed instead of stranding it.
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
