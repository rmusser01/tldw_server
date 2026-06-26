---
id: TASK-12014
title: Fix Task 5 writing annotation review code-quality issues
status: Done
references:
- codex/writing-manuscript-annotations-design
- 030135992e0fc2570e88c2c3a8622562eb7f8619
modified_files:
- tldw_Server_API/app/core/Writing/manuscript_annotation_jobs.py
- tldw_Server_API/app/core/DB_Management/ManuscriptDB.py
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/tests/Writing/test_manuscript_annotation_review_jobs.py
- tldw_Server_API/tests/Writing/test_manuscript_annotations_db.py
- tldw_Server_API/tests/Jobs/test_jobs_manager_acquire.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address code review issues for writing manuscript annotation review jobs: owner-scoped idempotency, in-memory duplicate candidate suppression, and JobManager.acquire_next_jobs job_type filtering. Use TDD and keep changes scoped away from unrelated watchlist templates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Task 5 review issues with focused TDD coverage. Red tests showed cross-user writing review idempotency collision (different owner returned job id 1), same-batch duplicate annotation candidates were both retained, and acquire_next_jobs rejected job_type. Implemented owner-scoped idempotency via a stable owner digest without adding owner data to payload, in-memory duplicate suppression while retaining candidates, and acquire_next_jobs job_type forwarding. Verification: focused red tests failed before implementation; focused green tests passed (3 passed); required pytest groups passed (27 passed, 63 passed); git diff --check passed; Bandit on touched app files exited 0 with 0 findings.
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
