---
id: TASK-13852
title: 'Audio jobs: cancelling a running stage does not stop the pipeline'
status: Done
labels:
- audio
- jobs
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When an audio job is cancelled while a stage runs, that stage is marked cancelled, but the worker ignores complete_job's False return, and the next stage is already queued (since PR #3260 it is queued before the current one completes), so the later stages still run to completion. The PR #3260 review probe (cancel_mid_stage) showed audio_convert cancelled and audio_transcribe and audio_store completed; dev behaved the same before that PR. The submit response returns only the first stage's job id, so a caller can only cancel that stage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cancelling an audio job while a stage runs stops the stages after it, with a test
- [x] #2 A caller can cancel a submitted audio pipeline as a whole, or the API documents how
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the clean-session review of PR #3261: app/api/v1/endpoints/audio/audio_jobs.py has no cancel endpoint, so AC #2 means choosing an outcome: add a whole-pipeline cancel (for example one keyed by the first stage's job id), or document the existing Jobs cancel path for each stage.
RED on dev+PR1: a stage cancelled while running still handed off and audio_store completed; no cancel route (3 AttributeErrors). Mutations: without hand_off's batch_group copy, the route finds no later stage (400) and the race test cancels nothing; with a one-pass route, the race test's stage queued between the list and the cancel is left queued (cancelled 0). GREEN: the queued next stage is cancelled when the current one turns out cancelled; DELETE /api/v1/audio/jobs/{job_id} handles only the six pipeline job types (404 otherwise, so a tts_longform job stays queued) and repeats list-then-cancel, at most 6 passes, through a per-pipeline batch_group (Decision 2, owner). OpenAPI fingerprint regenerated (schema_count 3301 -> 3302; paths.d.ts unchanged). Bandit -ll clean. Out of scope: cancelling long-form TTS jobs, and a rate limit on the route (the other audio-jobs routes have none).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Cancelling a running stage now cancels the stage it queues, so nothing after it runs. DELETE /api/v1/audio/jobs/{job_id} (owner or admin) cancels every queued or running stage of the pipeline, found through the batch_group that submit sets and every hand-off copies, including a stage queued while it cancels. Other audio-domain jobs, such as long-form TTS, get 404.
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
