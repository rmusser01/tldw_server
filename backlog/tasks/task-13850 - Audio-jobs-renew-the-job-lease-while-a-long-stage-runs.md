---
id: TASK-13850
title: 'Audio jobs: renew the job lease while a long stage runs'
status: To Do
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
- [ ] #1 A stage that runs longer than JOBS_LEASE_SECONDS keeps its lease while it runs and is not run again by another worker, with a test
- [ ] #2 Lease renewal stops when the stage ends, whether it completes or fails
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the clean-session review of PR #3261: lease-renewal patterns to follow are in core/TTS/tts_jobs_worker.py (via WorkerSDK) and core/Jobs/worker_sdk.py (renew_job_lease), besides core_jobs_worker.py. The claim that an expired lease consumes a retry follows the ExpiredLeasePolicy.CONSUME_RETRY default on create_job; confirm it in the lease-recovery path when implementing.
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
