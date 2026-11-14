---
id: TASK-13847
title: 'Audio jobs: a failed analysis loses the transcript'
status: Done
labels:
- audio
- jobs
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the audio_analyze stage fails without retry (for example a missing model or rejected credentials, which since PR #3257 fail at once) or is quarantined, the job never reaches audio_store, so the transcript produced by the earlier stages is not stored as the job result. Before PR #3257 the same jobs ended quarantined, with the same loss. A user who asked for transcription plus analysis loses the transcription to an analysis setup error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When analysis fails for good, the transcript is still stored, with the analysis failure and its error_code recorded next to it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-18, branch fix/audio-jobs-analysis-stage, plan Docs/superpowers/plans/2026-10-18-audio-jobs-analysis-stage.md. In the audio_analyze branch, a failure that a retry cannot fix, or a retryable one (rate_limit, provider_failure) once retry_count >= max_retries, hands the transcript to audio_store with analysis_error = {code, message} and no analysis. Owner decisions (2026-10-18): (1) an LLM quota refusal stores the transcript with analysis_error.code llm_quota_exceeded and message 'LLM quota exceeded (<category>)', no retry; (2) an empty transcript is stored with analysis_error.code empty_transcript; (3) the analyze job completes with result={'analysis_error': ...}, so GET /api/v1/jobs/{id} shows the code, and a warning is logged; the job.failed event and failures-by-code metric no longer fire for these. An analysis reply starting with 'Error:' is now a coded provider_failure, retried at 5 s and stored on the last attempt.
RED: 10 failed (9x 'audio_analyze did not hand over to audio_store' with fail_job terminal dist={'failed': 1}; the Error:-reply retry test got ('quarantined', 2, None)). GREEN: tests/AudioJobs, tests/Audio/test_audio_jobs_worker_helper.py, tests/Usage/test_llm_roots_non_sdk.py, tests/Services/test_main_shutdown_job_pollers.py, tests/Services/test_shutdown_audio_jobs_worker.py: 85 passed, 4 skipped; with tests/Jobs/test_jobs_quarantine_sqlite.py after the ruff import fixes: 90 passed, 4 skipped. tests/Docs + tests/CI/test_test_quality_ratchet.py: 213 passed. Scope-predicate ratchet: baseline exact. Docs: Audio_Jobs_API.md and Jobs_Module.md (both mirrored; Docs/Published refreshed), Operations/Usage_Quotas.md:95. Not changed: the GPU worker's older hand-off order (TASK-13850/13851 own it); audio_store stays a placeholder (TASK-13851).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A failed analysis no longer loses the transcript. Setup failures, LLM quota refusals and empty transcripts store it at once; rate limits and provider failures store it when their retries run out. The audio_store payload records analysis_error {code, message} next to the transcript, and the analyze job completes with the same analysis_error as its result. Docs/API-related/Audio_Jobs_API.md and Docs/Operations/Usage_Quotas.md describe the new behaviour.
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
