---
id: TASK-13850
title: 'Audio jobs: renew the job lease while a long stage runs'
status: Done
labels:
- audio
- jobs
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
app/services/audio_jobs_worker.py acquires each stage with JOBS_LEASE_SECONDS (default 120) and never renews the lease while the stage runs. Since PR #3260 (TASK-13848) transcription runs in this worker by default, and a transcription can easily run longer than the lease. Dockerfiles/Dockerfile.prod defaults UVICORN_WORKERS to 4, and each server process runs the in-process audio worker when AUDIO_JOBS_WORKER_ENABLED is on, so another process can recover the expired lease and run the same stage again, consuming a retry. The stage idempotency key (audio-stage:<job id>:<next type>) keeps the pipeline single-track, but the STT work is duplicated. core/Jobs workers such as core_jobs_worker.py renew leases while they work (renew_job_lease). Found in the whole-branch review of PR #3260.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A stage that runs longer than JOBS_LEASE_SECONDS keeps its lease while it runs and is not run again by another worker, with a test
- [x] #2 Lease renewal stops when the stage ends, whether it completes or fails
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the clean-session review of PR #3261: lease-renewal patterns to follow are in core/TTS/tts_jobs_worker.py (via WorkerSDK) and core/Jobs/worker_sdk.py (renew_job_lease), besides core_jobs_worker.py. The claim that an expired lease consumes a retry follows the ExpiredLeasePolicy.CONSUME_RETRY default on create_job; confirm it in the lease-recovery path when implementing.
RED on dev: 4 failed (STT ran twice under a 2 s lease with two worker loops; no renewal at all, twice; STT ran twice under a 3 s lease whose first renewal raised sqlite3.OperationalError). GREEN: 4 passed. Mutations: removing the stop in the loop's finally fails both renewal cases (2 renewals); returning on a renewal error, as WorkerSDK._auto_renew does, fails the renewal-error test (STT ran twice). Renewal is a local loop in audio_jobs_worker.py: it returns when a renewal is refused and logs a backend error and tries again. Sweep: 464 passed, 70 skipped; RUN_JOBS=1: 272 passed (Postgres reachable); required ratchets: 53 passed. Bandit -ll: workers clean; manager.py has one pre-existing B608 (dev line 9802), nothing new. Known flake: the route lints' child process intermittently aborts at exit (libc++abi recursive_mutex lock failed, assert -6 == 0) on a different test each run; the failing tests pass alone and the full ratchet set passed on rerun.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Both audio workers renew the acquired job's lease every half lease while its stage runs and stop renewing when the stage ends, so a stage longer than JOBS_LEASE_SECONDS is not recovered and run again by another worker. A backend error during a renewal is logged and the next renewal tried; a refused renewal (lost lease, cancelled job) ends the loop.
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
