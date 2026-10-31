---
id: TASK-13859
title: 'RG Redis governor: _RG_NONCRITICAL_EXCEPTIONS misses redis.exceptions.RedisError'
status: Done
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
- [x] #1 _RG_NONCRITICAL_EXCEPTIONS includes redis.exceptions.RedisError when redis is importable
- [x] #2 A test raising redis.exceptions.ConnectionError from the Redis client shows reserve, renew and commit honour the governor's fail mode instead of raising
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-20-rg-redis-error-noncritical.md (branch fix/rg-redis-error-noncritical).
Fixed:
- governor_redis.py: _RG_NONCRITICAL_EXCEPTIONS gains redis.exceptions.RedisError when redis is importable, so its 146 guards (74 around Redis calls) treat a dropped Redis as they treat the builtin ConnectionError.
- peek_with_policy reports unknown headroom when Redis cannot be read; before, ingress answered 500 on admitted requests and the character rate limiter's usage stats raised.
- middleware_simple.py: _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS gains RedisError, so a Redis error that still escapes reserve gets the fail-mode answer (503 under fail_closed), not a 500.
- commit and renew fall back to the worker's copy of the lease when Redis cannot be read; before, a released slot stayed taken until TTL, and a long request's lease lapsed (fail_open, fail_closed).
- A script Redis cannot run (NOSCRIPT after a restart, OOM, READONLY) drops its cached SHA. The lease path counts in Python instead of denying every quota request until the worker restarts, or dropping the limit under fail_open: against Redis after a restart; in each worker while Redis rejects writes. On dev the same error made ingress answer 500 after every Redis restart.
- policy_eval.log_redis_failure logs one WARNING per worker process and namespace per minute wherever a Redis error leaves the worker counting a rate window or a lease on its own: Redis unreachable, READONLY or OOM (owner Decision 2).
Outage behaviour, measured with a real redis-py client on a closed port:
- requests and tokens: fail_open admits; fail_closed denies with Retry-After up to 60; fallback_memory uses the memory governor.
- streams and jobs: every mode counts leases per worker.
- renew and commit never raise. Ingress answers 200 or 429.
Docs: ADR-078 amends ADR-076 (Decision 1). fail_closed keeps its 429 (Decision 3) and still admits concurrency leases (Decision 4).
Tests (real redis-py errors from real clients):
- T1 RED 9 failed, 3 passed; GREEN 12 passed.
- T2 RED 4 failed, 31 passed; GREEN 35 passed.
- T3 RED 6 failed, 16 passed; GREEN 22 passed.
- T4 RED 4 failed, 1 passed (real Redis); GREEN 5 passed.
- T5 RED 5 failed, 27 passed (real Redis); GREEN 32 passed.
- Sweep: 1 failed, 1910 passed, 4 skipped, 2 xfailed; the one failure (test_required_workflow_contracts::test_critical_e2e_budget_preserves_other_policies_and_still_enforces_limits) runs a workflow step that calls python, which is not on this machine's PATH (exit 127); with a venv python wrapper on PATH it passes on this branch. Docs 212 passed. Test-quality ratchet OK; scope-predicate baseline exact.
- Ruff clean. Bandit: only the B113 false positive dev already has in policy_eval.py.
Follow-up: TASK-13882, a connect timeout or circuit breaker for the RG Redis client.
Clean review of #3288 (2026-10-20): 0 Critical, 2 Important, 6 Minor; all folded in.
- I1: a READONLY Redis made fail_open requests/tokens reserves raise ReadOnlyError from a redundant, unguarded _purge_and_count at the head of the real-Redis add loop; the line is deleted.
- I2: under OOM a rejected window ZADD was swallowed, so no fail mode held the tokens per-minute limit. _add_members now re-raises after logging, and the add loop applies the fail mode as an unreachable Redis does (fail_open admits, fail_closed denies for the window, fallback_memory hands the request to the memory governor). test_redis_rejecting_writes_follows_the_fail_mode (READONLY and OOM x 3 modes x requests/tokens, real Redis): RED 5 failed, 7 passed; GREEN 12 passed.
- M1: renew falls back to the worker's copy when Redis has no record, as commit does (RED 2 failed; GREEN).
- M2: _tokens_lua_sha is cleared when its script fails, at the check and the peek estimate (RED [60, 60] == [60, 30] x2; GREEN).
- M3: a lost script reply counts the script's own lease in the Python fallback; recorded in ADR-078 and filed as TASK-13892.
- M4: TASK-13882's latency premise corrected (redis-py 8.1 from_url waits 5 s per call).
- M5: ADR-078 text fixed (one-sentence Decision, fallback_memory READONLY lease holder, rates, narrowed 500 claim, Follow-up section).
- M6: test_renew_and_commit_survive_redis_raising_connection_error asserts the slot frees; mutants (commit without fallback, tuple without RedisError) fail it.
Verification: RG suite + Usage/test_concurrency_quota.py 484 passed, 2 xfailed (one earlier run had 2 Postgres setup errors from the shared tldw_test database being dropped mid-run; both tests pass on rerun). tests/Docs passed; test-quality ratchet OK; ruff clean; bandit clean on governor_redis.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The Redis Resource Governor and the ingress middleware now catch redis.exceptions.RedisError, so once Redis drops, reserve, renew, commit, peek_with_policy and ingress apply RG's fail mode instead of raising or answering 500. commit and renew act on the worker's copy of a lease during an outage, a script Redis cannot run falls back to counting in Python and reloads, and a worker that enforces limits on its own logs it once a minute. Concurrency leases count per worker while Redis is down (ADR-078, amending ADR-076).
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
