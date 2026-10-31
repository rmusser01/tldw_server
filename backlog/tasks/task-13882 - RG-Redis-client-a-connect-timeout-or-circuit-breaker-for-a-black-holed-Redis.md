---
id: TASK-13882
title: 'RG Redis client: a connect timeout or circuit breaker for a black-holed Redis'
status: Done
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
- [x] #1 The RG Redis client has a short connect timeout, or RG stops calling Redis for a while after a failure, with a test on an unroutable address
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Correction from the clean review of #3288 (M4, 2026-10-20): the description's 'waits for the OS connect timeout' is wrong for the installed redis-py. redis-py 8.1's from_url defaults socket_connect_timeout (and socket_timeout) to 5 s: on the black-holed address 10.255.255.1 one command waited 5.00 s, then raised redis.exceptions.TimeoutError. Measured Redis calls per request with Redis down (production from_url client): ingress 6 under fallback_memory (not 7), 14 under fail_open, and 7 on the first request then 2 under fail_closed; a quota acquire 2 (fallback_memory) or 8 (fail_open, fail_closed); renew 2; release 3; dev 1 for any of these. With a black-holed Redis a governed request therefore takes about 30 s (fallback_memory), 70 s (fail_open), or 35 s then 10 s (fail_closed), against 5 s and then a 500 on dev. Older redis-py versions, which the redis>=4.5.0 pin still allows, may wait for the OS connect timeout instead.
Plan: Docs/superpowers/plans/2026-10-22-rg-redis-unreachable-cooldown.md (branch fix/rg-redis-unreachable-cooldown, with TASK-13892).
Fixed (AC's second option: RG stops calling Redis for a while after a failure):
- governor_redis.py: _client_get wraps every real client's execute_command (_stop_calling_redis_while_unreachable). A command that fails with redis-py ConnectionError or TimeoutError (refused, reset or timed out) is followed by one PING, which normally takes the connection just dropped and reconnects; if that fails too, each command raises ConnectionError at once for RG_REDIS_UNREACHABLE_COOLDOWN_SEC (default 10 s, 0 = off), which RG's guards answer by the fail mode. Then one command probes while the rest keep failing fast; any answer, an error reply included, ends the cooldown. The stub is not wrapped.
- RG's client now waits at most 5 s to connect and for each reply (redis-py 8's defaults, pinned so older releases the redis>=4.5.0 pin allows wait no longer); socket_timeout and socket_connect_timeout options in REDIS_URL override them.
- Owner Decision 1 (b): a PING confirms a TimeoutError as well as a ConnectionError.
- The PING is the review's B1 fix. Without it, a connection Redis closed while idle started the cooldown: after an idle spell fail_closed refused every new user with deny(60), where dev admitted them. Measured with CONFIG SET timeout 1 on a private Redis: dev ok x3, without the PING deny(60) x3, with it ok x3.
- Not a shorter timeout: at 0.5 s a fail_open ingress request still waits 14 x 0.5 s, every request. Not the shared Infrastructure CircuitBreaker: it logs a WARNING on every trip for each of the 9 governors, and a registry breaker reads a database on every call.
Measured (black-holed 10.255.255.1, 5 s timeouts): on dev an ingress request took 30 s (fallback_memory) or 70 s (fail_open) and a quota request 10 s or 65 s, every request. Now the first request takes 10.00 s and the next 0.00 s (re-measured for the PR, all six mode/kind lines); while Redis stays down, one request per governor waits 10 s every 20 s. RG retries 10 s after the last failed probe, so its fail mode lasts up to about 15 s after Redis returns (measured 14.8 s worst case, planner); under fail_closed a user refused then keeps a Retry-After of up to 60 s, so 429s can last about 75 s after recovery. One 6 s stall: fail_closed admits everyone, as dev (option (a) would have refused every new user for about 10 s). One 11 s stall starts the cooldown.
Tests: 6 tests, 22 cases in test_rg_fail_modes_across_categories.py (unroutable/stopped/silent x 3 modes; a 0 cooldown; the env var, 8 values; the pinned timeouts; recovery after the cooldown, reply and error reply; a connection closed while idle). RED 21 failed, 26 passed (dev sends 28/15/11 commands over 3 reserves, where 2 are expected; no pinned timeouts; the helpers are missing); GREEN 47 passed. The plan's dry run: 11 mutants each fail a test.
Sweep (tests/Resource_Governance, tests/Usage, Services/test_startup_resource_governor.py, private Redis): 785 passed, 1 skipped, 2 xfailed. Docs and test-quality ratchet: 213 passed; triage OK; ruff clean; bandit clean on governor_redis.py.
Docs: ADR-081 amends ADR-078 (when the fail modes apply, the fail_closed bound, the recovery lag, the knob); Env_Vars.md (RG_REDIS_FAIL_MODE and RG_REDIS_UNREACHABLE_COOLDOWN_SEC, mirrored), Rate_Limits_Troubleshooting.md (a fail_closed row), Usage_Quotas.md and the RG README.
Follow-up: TASK-13909 (rg_unique_ns leaks keys past the first SCAN page).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Once a Redis command and a PING right after it both fail, the Redis Resource Governor stops calling Redis for RG_REDIS_UNREACHABLE_COOLDOWN_SEC (10 s by default) and answers by its fail mode, then probes with one command at a time; its client waits at most 5 s per connect or reply. A black-holed Redis now costs the first governed request 10 s and the rest nothing, where each request used to wait 30-70 s; the fail mode lasts up to about 15 s after Redis returns, and fail_closed's 429s up to about 75 s (ADR-081).
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
