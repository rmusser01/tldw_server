---
id: TASK-13853
title: 'Audio jobs: stage hand-offs count against the per-minute and queued-job submit
  quotas'
status: To Do
labels:
- audio
- jobs
- quotas
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Each audio stage hand-off goes through JobManager.create_job admission, so with JOBS_QUOTA_SUBMITS_PER_MIN, JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO or JOBS_QUOTA_MAX_QUEUED set, the worker's internal hand-offs count as the user's submissions. With JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO=1 every pipeline is quarantined at its first hand-off (PR #3260 review probe); before that PR these jobs stopped silently. PR #3260's fair_share_exempt exempts hand-offs only from the per-user concurrency limit, not from these quotas.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A stage hand-off does not count against the user's submit quotas (or the decision to count it is documented), with a test
- [ ] #2 With JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO=1 a submitted audio job runs through every stage
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
