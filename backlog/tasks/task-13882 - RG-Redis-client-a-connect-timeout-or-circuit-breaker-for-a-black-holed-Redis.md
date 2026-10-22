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
Correction from the clean review of #3288 (M4, 2026-10-20): the description's 'waits for the OS connect timeout' is wrong for the installed redis-py. redis-py 8.1's from_url defaults socket_connect_timeout (and socket_timeout) to 5 s: on the black-holed address 10.255.255.1 one command waited 5.00 s, then raised redis.exceptions.TimeoutError. Measured Redis calls per request with Redis down (production from_url client): ingress 6 under fallback_memory (not 7), 14 under fail_open, and 7 on the first request then 2 under fail_closed; a quota acquire 2 (fallback_memory) or 8 (fail_open, fail_closed); renew 2; release 3; dev 1 for any of these. With a black-holed Redis a governed request therefore takes about 30 s (fallback_memory), 70 s (fail_open), or 35 s then 10 s (fail_closed), against 5 s and then a 500 on dev. Older redis-py versions, which the redis>=4.5.0 pin still allows, may wait for the OS connect timeout instead.
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
