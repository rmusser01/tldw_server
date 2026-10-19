---
id: TASK-13851
title: 'Audio jobs: long recordings exceed the job payload cap at the hand-off, and
  a refused hand-off re-runs the stage'
status: Done
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
- [x] #1 A multi-hour recording's pipeline payload stays under the job payload cap, for example by carrying the transcript once or storing it outside the payload
- [x] #2 A hand-off refusal that will always repeat fails the stage without re-running it, and a transient enqueue error is retried without re-running the stage, with tests
- [x] #3 audio_store cannot leave a job stuck processing because its result is too large
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the clean-session review of PR #3261: JobManager.create_job and complete_job also honour JOBS_JSON_TRUNCATE, which replaces an oversized payload with {"_truncated": true} and silently destroys the transcript. Enabling it is not a fix for the payload cap; the transcript must stay intact.
RED on dev+PR1: refused hand-off quarantined (retry_count 2) after running STT twice, for all three refusals (payload cap, job-type allowlist, payload not JSON); transient enqueue error re-ran the stage (retry_count 1); GPU worker left a refused hand-off's stage completed, and ran an ownerless job as user 0; 3-hour chunked recording refused at transcribe->chunk (ValueError, the payload cap; the transcribe job failed at once after Task 4); store result over the cap left the job processing; helper test pinned the duplicated artifact. GREEN: all pass, including the analysis plan's Decision 3 test (hand_off passes result=analysis_error to complete_job). Mutation: catching only ValueError quarantines the payload-not-json case. hand_off logs only the error type. Decision 1 (owner): normalized_stt keeps the artifact's metadata only; about 4 h chunked, 6 h unchunked at the default cap. The GPU worker now shares the transcribe handler and hand_off, and fails an ownerless job as the in-process worker does. Sweep: 476 passed, 70 skipped; RUN_JOBS=1: 272 passed; tests/Docs plus API-contract: 323 passed; required ratchets: 53 passed (an earlier run under load hit the route lints' intermittent child-process abort at exit, libc++abi recursive_mutex lock failed). Ruff: only the helper test's I001, which exists on dev. Bandit -ll clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Both workers hand off through one hand_off(): a ValueError or TypeError from create_job (payload over the cap or not JSON, refused queue or job type, secret match) fails the stage at once; any other error is retried twice in place before the stage is retried. The stage completes with the result its caller passes (analysis_error for a failed analysis). The payload carries the transcript once, so a multi-hour recording fits the default cap, and an audio_store result over the cap fails with result_too_large instead of staying processing.
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
