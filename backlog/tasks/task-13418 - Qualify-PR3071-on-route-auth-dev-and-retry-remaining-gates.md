---
id: TASK-13418
title: Qualify PR3071 on route-auth dev and retry remaining gates
status: In Progress
created_date: 2026-10-02 05:59
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/3071
documentation:
- IMPLEMENTATION_PLAN_pr3071_route_auth_dev_retry_2026_10_01.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue approved PR3071 retry on frozen dev3caebcfc. Preserve the prior TASK-13417 qualification history and incoming route-map TASK-13417 without active ID collision. Retain source-equivalent Chat Workspace and current-head CI successes; run incoming route-auth/benchmark gates and actual read-only Chrome acceptance without mocks. Retry PostgreSQL only through official fixtures; do not restart Docker, remove shared containers, fabricate a database or bypass gates. Publish to the existing draft PR with requester Change summary unchanged and no merge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preserve the prior tracker byte-identically and integrate frozen dev3caebcfc without losing either branch history.
- [ ] #2 Incoming authentication/ratchet checks, scoped Bandit and actual Chat Workspace acceptance are qualified, with unchanged-source evidence reused only where byte-identical.
- [ ] #3 Retry official PostgreSQL and head-bound CI, record real remaining limits, and normally publish the existing draft PR without merging or changing requester summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Archive the collided qualification record intact and merge frozen dev. 2. Run incoming gate tests and review, scoped Bandit, and source-equivalent native Chrome checks. 3. Retry official PG/current-head CI and publish evidence without claiming incomplete gates passed.
<!-- SECTION:PLAN:END -->

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
