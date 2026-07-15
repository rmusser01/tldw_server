---
id: TASK-12969
title: Coordinate Jobs admission hardening and lease lifecycle extraction
status: In Progress
created_date: 2026-07-15 01:18
labels:
- Jobs
- stability
- refactor
- postgres
priority: High
references:
- Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md
- Docs/superpowers/plans/2026-07-04-jobs-admission-operations-extraction-plan.md
- 'PR #2527'
- 'PR #2611'
- origin/dev 132037dd075090c295003d6885ac4276a9640916
documentation:
- Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md
modified_files:
- Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md
updated_date: 2026-07-15 01:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the two-week Jobs stability stream validated on current origin/dev. PR 1 hardens admission transaction recovery, secret rejection, and atomic quotas. PR 2 begins only after PR 1 is merged and extracts single-job lease acquisition. PR 3 begins only after PR 2 is merged and extracts single-job renewal/release. All phases preserve the JobManager facade and backend parity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An execution-ready plan records exact files, tests, commands, sequencing, merge gates, and rollback boundaries for all three PRs.
- [ ] #2 Admission hardening is an independently reviewable deliverable with regression coverage for all three validated findings on SQLite and real PostgreSQL where applicable.
- [ ] #3 Lease acquisition is a separate gated deliverable and does not start until admission hardening is merged and green.
- [ ] #4 Lease renewal/release is a separate gated deliverable and does not start until acquisition is merged and green.
- [ ] #5 Every implementation PR includes the requester-owned Change summary required by project policy.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use Docs/superpowers/plans/2026-07-14-jobs-admission-hardening-and-lease-lifecycle.md. PR 1 is child .1 admission hardening. PR 2 is dependent child .2 single-job acquisition extraction. PR 3 is dependent child .3 single-job renewal/release extraction.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Planning worktree: codex/jobs-admission-hardening at .worktrees/jobs-admission-hardening, rebased to origin/dev 132037dd075090c295003d6885ac4276a9640916. Intervening PR #2731 changed only Research Workspace/frontend files and no Jobs source/tests. Latest-head diagnostics reproduced all three defects: secret reject DID NOT RAISE ValueError; optional PostgreSQL counter failure raised InFailedSqlTransaction at event insertion; concurrent max-queued=1 PostgreSQL admission created 2 jobs instead of 1. Latest clean controls: focused SQLite/contracts/parity 22 passed with 52 warnings; real PostgreSQL parity 7 passed with 20 warnings and no skips. A Backlog ID collision with merged Research Workspace TASK-12968 was detected and resolved by reallocating this stream to TASK-12969 and children .1-.3. Final design review split acquisition from renewal/release; PR 3 is stretch scope if merge latency prevents its fresh-base gate. Planning only touched docs/Backlog; Bandit is not applicable until production code changes.
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
