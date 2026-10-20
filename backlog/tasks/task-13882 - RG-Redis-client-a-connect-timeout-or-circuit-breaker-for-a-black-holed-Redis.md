---
id: TASK-13882
title: 'RG Redis client: a connect timeout or circuit breaker for a black-holed Redis'
status: To Do
labels:
- resource-governance
- redis
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After TASK-13859, RG applies its fail mode on Redis errors instead of raising, so a request makes several Redis calls during an outage, where it used to fail at the first. The reviewer counted, with Redis down: ingress 7 (fallback_memory), 14-15 (fail_open) or 3-8 (fail_closed) calls, against 1 on dev; a quota acquire 2 or 8; a shielded release 3. A refused connection fails in about 10 ms, but redis_factory sets no socket_connect_timeout, so with a black-holed Redis each call waits for the OS connect timeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The RG Redis client has a short connect timeout, or RG stops calling Redis for a while after a failure, with a test on an unroutable address
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
