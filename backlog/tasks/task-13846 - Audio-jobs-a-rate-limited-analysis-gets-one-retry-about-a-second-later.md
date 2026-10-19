---
id: TASK-13846
title: 'Audio jobs: a rate-limited analysis gets one retry about a second later'
status: Done
labels:
- audio
- jobs
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The audio jobs worker fails a stage with JobManager.fail_job's default backoff_seconds=1, and the Jobs failure streak quarantines a job after JOBS_QUARANTINE_THRESHOLD (default 2) identical failures in a row. So an audio_analyze job that hits a provider rate limit is retried once, about a second later, and a second rate limit quarantines it (confirmed by probe in the review of PR #3257, on dev and on the branch). A rate limit usually needs a longer wait, and the provider may say how long.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A rate-limited analysis waits a meaningful backoff (or the provider's retry-after) before retrying
- [x] #2 Rate limits do not quarantine a job after two attempts
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-18, branch fix/audio-jobs-analysis-stage, plan Docs/superpowers/plans/2026-10-18-audio-jobs-analysis-stage.md. JobManager.fail_job gains quarantine: bool = True (default unchanged for its other callers). quarantine=False sets the streak threshold to 10**9 on both backends; on SQLite it sits after the test-mode block that resets the threshold when JOBS_QUARANTINE_THRESHOLD is set. Retries still stop at max_retries. The audio analyze stage retries rate_limit with backoff_seconds=30 and provider_failure with 5 (fail_job doubles per retry: 30/60/120 s and 5/10/20 s), quarantine=False; _StageFailure carries both and the worker's final fail_job passes them.
RED: test_fail_job_without_quarantine_keeps_retrying[quarantine-off] TypeError (unknown keyword 'quarantine'), its default-quarantine control passing; test_audio_analyze_rate_limit_waits_and_is_not_quarantined 'assert 0.0 >= 25' (available_at NULL: in test mode the old 1 s backoff becomes no delay). GREEN: tests/AudioJobs + helper + tests/Jobs/test_jobs_quarantine_sqlite.py 53 passed, 4 skipped. tests/Jobs (SQLite, TLDW_TEST_NO_DOCKER=1): 1 failed, 1329 passed, 631 skipped; the failure is test_retry_admission_fixture_registration.py::test_bridge_resolution_preserves_jobs_loop_and_isolation (pytester 'Plugin already registered under a different name: asyncio'), environmental, failing the same way on unchanged dev per the plan review's dry run. Postgres (Docker, RUN_JOBS=1): test_jobs_quarantine_postgres.py + test_jobs_quarantine_flow_postgres.py 5 passed; a mutation disabling the Postgres override fails the quarantine-off case ('quarantined' != 'queued').
Not done: reading the provider's retry-after. SummaryProviderError does not carry it; the fixed doubling backoff meets AC #1 (out of scope in the plan).
Clean-session review of PR #3273: 0 Critical, 0 Important, 2 Minor, both fixed. M1: added test_audio_analyze_rate_limited_one_attempt_before_the_last_is_retried (retry_count == max_retries - 1 is retried, not stored; fails under retry_count + 1 < max_retries) and asserted the D3 result on the retries-exhausted path. M2: documented in the fail_job docstring and Jobs_Module.md that quarantine=False still advances the failure streak.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
A rate-limited audio analysis now waits 30 s, then 60 s, then 120 s before retrying (a provider failure 5, 10, 20 s), and these failures never quarantine the job: fail_job(quarantine=False), new on both Jobs backends, prevents quarantine for those failures but still advances the failure streak; retries still stop at max_retries. When the retries run out, TASK-13847 stores the transcript with analysis_error. Documented in Docs/Code_Documentation/Jobs_Module.md (whose quarantine default now reads 2, the code's) and Docs/API-related/Audio_Jobs_API.md.
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
