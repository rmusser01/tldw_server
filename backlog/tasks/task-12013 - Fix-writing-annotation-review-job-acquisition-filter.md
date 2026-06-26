---
id: TASK-12013
title: Fix writing annotation review job acquisition filter
status: Done
labels:
- bug
- writing
- jobs
priority: High
modified_files:
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/Jobs/worker_sdk.py
- tldw_Server_API/app/services/writing_annotation_review_jobs_worker.py
- tldw_Server_API/tests/Jobs/test_worker_sdk.py
- tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address Task 5 spec-review feedback: writing annotation review worker must lease only writing_scene_annotation_review jobs by exposing an optional job_type filter through WorkerSDK.run and JobManager.acquire_next_job, with focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the Task 5 spec-review issue by adding an optional job_type filter to JobManager.acquire_next_job and WorkerSDK.run, applying the predicate across the Postgres and SQLite acquisition query paths, and passing writing_scene_annotation_review from the writing annotation review worker. Added focused regression coverage for WorkerSDK/JobManager filtering and the writing worker SDK.run call contract.

Verification:
- Red: targeted new tests failed before implementation with WorkerSDK.run() unexpected job_type and missing service run_kwargs['job_type'].
- Green: ../../.venv/bin/python -m pytest tldw_Server_API/tests/Jobs/test_worker_sdk.py tldw_Server_API/tests/Services/test_writing_annotation_review_jobs_worker.py -q -> 12 passed.
- Green: ../../.venv/bin/python -m pytest tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py tldw_Server_API/tests/Writing/test_manuscript_annotations_api.py tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py -q -> 57 passed.
- git diff --check -> clean.
- ../../.venv/bin/python -m bandit -r touched app files -f json -o /tmp/bandit_task12013.json -> exit 0, zero findings.
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
