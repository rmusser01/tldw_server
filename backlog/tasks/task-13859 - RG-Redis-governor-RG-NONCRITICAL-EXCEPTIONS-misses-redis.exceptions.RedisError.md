---
id: TASK-13859
title: 'RG Redis governor: _RG_NONCRITICAL_EXCEPTIONS misses redis.exceptions.RedisError'
status: To Do
labels:
- resource-governance
- redis
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the TASK-13435 plan review (B1). redis-py's ConnectionError and TimeoutError derive from redis.exceptions.RedisError, not from the builtins, so _RG_NONCRITICAL_EXCEPTIONS in Resource_Governance/governor_redis.py (builtins only) does not catch them. Once a worker has connected, a Redis outage surfaces as RedisError through RG's best-effort guards in reserve/renew/commit, ingress included. concurrency_quota catches RedisError itself (TASK-13435); RG does not. Fix: add RedisError (imported where redis is available, as Infrastructure/redis_factory.py does) to _RG_NONCRITICAL_EXCEPTIONS, with tests that raise redis.exceptions.ConnectionError.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 _RG_NONCRITICAL_EXCEPTIONS includes redis.exceptions.RedisError when redis is importable
- [ ] #2 A test raising redis.exceptions.ConnectionError from the Redis client shows reserve, renew and commit honour the governor's fail mode instead of raising
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
