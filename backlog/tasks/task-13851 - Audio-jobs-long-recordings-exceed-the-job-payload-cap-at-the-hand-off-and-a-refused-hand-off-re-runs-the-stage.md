---
id: TASK-13851
title: 'Audio jobs: long recordings exceed the job payload cap at the hand-off, and
  a refused hand-off re-runs the stage'
status: To Do
labels:
- audio
- jobs
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The audio pipeline payload carries the transcript about 4-5 times: text, segments, normalized_stt.text and normalized_stt.segments, plus chunks when chunking is on (app/services/audio_jobs_worker.py). JOBS_MAX_JSON_BYTES defaults to 1048576, so by the PR #3260 review's estimate recordings over roughly 3.5 hours (2.5 with chunking) cannot be handed off past transcription, and audio_store's complete_job(result=...) hits the same cap (the job then stays processing until lease recovery). Separately, since PR #3260 a failed hand-off is permanent only when the queue is refused, which no shipped config does; any other refusal (a payload over JOBS_MAX_JSON_BYTES, JOBS_ALLOWED_JOB_TYPES_AUDIO, a JOBS_SECRET_REJECT match) is retried by re-running the whole stage, so a long transcription runs twice before the job is quarantined (clean-review probe: STT twice). A transient create_job error also re-runs the stage, although the enqueue is idempotent and could be retried on its own.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A multi-hour recording's pipeline payload stays under the job payload cap, for example by carrying the transcript once or storing it outside the payload
- [ ] #2 A hand-off refusal that will always repeat fails the stage without re-running it, and a transient enqueue error is retried without re-running the stage, with tests
- [ ] #3 audio_store cannot leave a job stuck processing because its result is too large
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the clean-session review of PR #3261: JobManager.create_job and complete_job also honour JOBS_JSON_TRUNCATE, which replaces an oversized payload with {"_truncated": true} and silently destroys the transcript. Enabling it is not a fix for the payload cap; the transcript must stay intact.
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
