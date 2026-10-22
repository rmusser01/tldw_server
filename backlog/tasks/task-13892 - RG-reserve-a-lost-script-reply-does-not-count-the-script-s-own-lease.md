---
id: TASK-13892
title: 'RG reserve: a lost script reply does not count the script''s own lease'
status: To Do
labels:
- resource-governance
- redis
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the clean review of #3288 (M3). After TASK-13859, when the lease script or the multi-key rate script runs but its reply is lost (a redis-py TimeoutError after the 5 s socket_timeout, or a connection reset), reserve falls through to the Python path, which counts the members the script just wrote. Leases: with a limit of 1 the request is refused with retry_after=120, and the script's member blocks the user's only slot until it expires; with a limit of 2 the request is admitted holding two slots. Rates: the multi-key script's members are charged again under new member ids and never refunded. Before TASK-13859, dev admitted with no lease and the member was stranded the same way. ADR-078's Consequences record it. Fix: before counting in Python, best-effort ZREM the members planned for that script; they are this reserve's own UUIDs, so removing them is always safe. Needs Redis to stall past socket_timeout, so it is unlikely.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a lease or multi-key rate script whose reply is lost, reserve removes that reserve's planned members before the Python fallback counts, with a real-Redis test where the script runs and the reply raises redis.exceptions.TimeoutError
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
