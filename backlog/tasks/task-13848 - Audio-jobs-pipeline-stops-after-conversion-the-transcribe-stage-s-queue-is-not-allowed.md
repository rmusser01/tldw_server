---
id: TASK-13848
title: 'Audio jobs pipeline stops after conversion: the transcribe stage''s queue
  is not allowed'
status: To Do
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
- [ ] #1 A job submitted to /api/v1/audio/jobs/submit is transcribed by the in-process audio worker when no GPU worker is configured, with a test that runs the real hand-off from audio_convert
- [ ] #2 A deployment that runs the GPU transcription worker can still route audio_transcribe to it, and the queue it uses is allowed for the audio domain
- [ ] #3 A failure to queue the next stage fails the job visibly instead of leaving the previous stage completed and the job stranded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the clean-session review of PR #3259 (TASK-13641): core/TTS/utils.tts_jobs_queue() refuses default and transcribe because the audio pipeline workers read those queues without a job-type filter; the names are literals copied from audio_jobs_worker.py and audio_transcribe_gpu_worker.py. Any queue this task routes the pipeline through must be added to that refusal set (or the names shared as constants), or the guard goes stale and a TTS queue could again be one an unfiltered worker reads.
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
