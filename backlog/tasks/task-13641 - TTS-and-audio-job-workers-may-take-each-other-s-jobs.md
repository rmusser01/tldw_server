---
id: TASK-13641
title: TTS and audio job workers may take each other's jobs
status: Done
labels:
- audio
- tts
- jobs
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
core/TTS/tts_jobs_worker.py (WorkerSDK, no job_type filter) and app/services/audio_jobs_worker.py both acquire from domain audio, queue default with no job_type filter. If both run, the TTS worker would fail audio_* stage jobs as unsupported and the audio worker would fail tts_longform jobs as an unknown job_type. Not verified at runtime: the TTS worker appears to run only standalone (python -m). Found while fixing TASK-13637.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Verify whether both workers can run against the same queue; if so, each acquires only its own job types (separate queues or a job_type filter), with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified on dev f456e5c9bc: the collision is real and does not need the TTS worker. POST /api/v1/audio/speech/jobs enqueued tts_longform on the audio domain's default queue, which the in-process audio pipeline worker (AUDIO_JOBS_WORKER_ENABLED) reads unfiltered; it took a submitted speech job within half a second and failed it as an unknown job type until it was quarantined. The standalone TTS worker read the same queue unfiltered and failed an audio_convert job as unsupported. Nothing in the repository starts the TTS worker; it runs only as python -m tldw_Server_API.app.core.TTS.tts_jobs_worker. Fixed: core/TTS/utils.tts_jobs_queue() (TTS_JOBS_QUEUE, default tts) is the queue for long-form TTS jobs, used by the endpoint and the worker; tts is registered for the audio domain in JobManager.DOMAIN_ALLOWED_QUEUES; the TTS worker passes job_type=tts_longform to WorkerSDK.run; tts_jobs_queue() refuses default and transcribe (the queues the audio pipeline workers read unfiltered), so an old TTS_JOBS_QUEUE=default fails at worker startup and at the endpoint instead of reviving the collision or stranding jobs. A custom TTS_JOBS_QUEUE must be set the same on the API server and the worker and registered in JOBS_ALLOWED_QUEUES_AUDIO, otherwise the endpoint fails with a 500 (it used to strand jobs silently). TTS jobs already queued on default at deploy are not picked up by the TTS worker; move them with UPDATE jobs SET queue='tts' WHERE domain='audio' AND job_type='tts_longform' AND status='queued' AND queue='default'. Tests: RED 6 failed on dev (the endpoint queued on default; the audio worker quarantined the speech job; the TTS worker failed the audio_convert job; no tts_jobs_queue; no job-type filter), GREEN 6 passed; every test file touching the changed code plus tests/AudioJobs, tests/Jobs and tests/Docs 1 failed, 3365 passed, 6 skipped, the failure (test_retry_admission_fixture_registration's terminal-summary check) also failing on unchanged dev. Ruff: only the six findings dev already has in two test files. Bandit -ll: only the pre-existing B608 in Jobs/manager.py. Docs/Architecture/audio.md documents the queue. Found and filed: TASK-13848 (the audio pipeline stops after conversion; the transcribe queue is not allowed and only the GPU worker reads it) and TASK-13849 (nothing starts the TTS jobs worker, but the WebUI offers long-form TTS).
Clean-session review of PR #3259 (2026-10-17): ready to merge, 0 Critical, 0 Important, 5 Minor. Fixed: the audio-worker isolation test gains a positive control (an audio_store job on default that the worker must complete), since the worker loop swallows errors (still fails on dev f456e5c9bc: quarantined; passes 3 of 3 on the branch; the three TTS test files 152 passed); noted on TASK-13848 that any queue it routes the pipeline through must join tts_jobs_queue()'s refusal set; the PR's operator note covers jobs in flight at deploy (stop the old TTS worker first, re-run the UPDATE after the lease period). Stands: a refused or unregistered TTS_JOBS_QUEUE surfaces per request as a generic 500 with the reason in the ERROR log (the plan's accepted behaviour; the worker fails at startup); test_tts_job_dispatch_marker still creates TTS jobs on default directly (cosmetic).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Long-form TTS jobs go on their own tts queue (TTS_JOBS_QUEUE, registered for the audio domain), chosen in one place for the endpoint and the TTS worker, so the audio pipeline worker no longer takes and quarantines them; the TTS worker acquires only tts_longform jobs, so it no longer fails audio pipeline jobs.
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
