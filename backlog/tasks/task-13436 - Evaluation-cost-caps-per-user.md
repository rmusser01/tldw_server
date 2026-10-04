---
id: TASK-13436
title: Evaluation cost caps per user
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from spec 2. Cost caps never fired: every caller passes estimated_cost=0.0 and the monthly cap is never compared. Needs real cost estimation, then limits.evaluation_cost_per_day_usd / _per_month_usd. Parent: TASK-13434.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A per-user daily and monthly evaluation cost cap blocks once recorded cost reaches it
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
