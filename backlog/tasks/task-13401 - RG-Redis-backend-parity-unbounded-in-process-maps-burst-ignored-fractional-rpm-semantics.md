---
id: TASK-13401
title: >-
  RG Redis backend parity: unbounded in-process maps, burst ignored,
  fractional-rpm semantics
status: To Do
assignee: []
created_date: '2026-09-30 09:45'
updated_date: '2026-10-01 03:03'
labels:
  - rate-limit
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the final review of the RG ingress safety net (plan 2026-09-29-rg-ingress-safety-net).
- governor_redis.py _requests_accept_window and the floor maps get one entry per accepted (policy, entity) and are never evicted. Memory got idle eviction for the same 'every /api/ route, many IPs' growth; Redis did not.
- Redis ignores burst: its requests window is rpm per 60 s. Every safety-net policy (burst 2.0) therefore has half the headroom on Redis that it has on memory.
- Fractional rpm: Redis rounds up to max(1, ceil(rpm)) per minute. authnz.magic_link.email (0.3/10, meant as about 3 per 10 min) admits 10 per 10 min on Redis, while memory allows 3 up front and then 1 per 200 s.
- For fractional rpm, the memory decision details (effective_limit = int(rpm)) and the middleware header fallback report a limit of 0.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Redis in-process maps are bounded (idle eviction like memory)
- [ ] #2 Redis applies burst, or ADR-057 and the troubleshooting page document the difference
- [ ] #3 Fractional-rpm policies behave the same on both backends within one window, with a test
- [ ] #4 Rate-limit headers never report a limit of 0 for a fractional policy
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR reference: AC #2's 'ADR-057' means the RG safety-net ADR, now Docs/ADR/056-resource-governor-safety-net.md.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
