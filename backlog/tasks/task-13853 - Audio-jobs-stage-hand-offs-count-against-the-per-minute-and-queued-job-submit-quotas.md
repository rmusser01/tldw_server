---
id: TASK-13853
title: 'Audio jobs: stage hand-offs count against the per-minute and queued-job submit
  quotas'
status: Done
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
- [x] #1 A stage hand-off does not count against the user's submit quotas (or the decision to count it is documented), with a test
- [x] #2 With JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO=1 a submitted audio job runs through every stage
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED on dev: 6 failed with a local Postgres reachable (TypeError quota_exempt on both quota cases on SQLite and on Postgres; pipeline quarantined under JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO=1; hand-off refused by JOBS_QUOTA_MAX_QUEUED_AUDIO=1 with another queued job). GREEN: 6 passed on both backends. Stage rows still count toward the user's next submission, documented in Audio_Jobs_API.md and Jobs_Module.md (owner Decision 3). scope_predicate_baseline.txt regenerated (manager.py lines +5; 28 predicates, baseline exact). Bandit -ll: one pre-existing B608 in manager.py, nothing new.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
create_job(quota_exempt=True) skips the queued and per-minute submit quotas; both audio workers pass it on every stage hand-off, so a submitted pipeline runs through every stage under JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO=1. The docs say that stage rows still count toward the user's next submission.
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
