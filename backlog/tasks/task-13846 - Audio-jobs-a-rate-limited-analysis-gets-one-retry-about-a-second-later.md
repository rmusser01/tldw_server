---
id: TASK-13846
title: 'Audio jobs: a rate-limited analysis gets one retry about a second later'
status: To Do
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
- [ ] #1 A rate-limited analysis waits a meaningful backoff (or the provider's retry-after) before retrying
- [ ] #2 Rate limits do not quarantine a job after two attempts
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
