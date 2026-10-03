---
id: TASK-13430
title: 'RG Redis window keys (rg:win:*) never expire'
status: Done
assignee: []
created_date: '2026-10-02 02:02'
updated_date: '2026-10-03 01:20'
labels:
  - rate-limit
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing TASK-13401. The Redis server's own requests/tokens sliding-window keys (`rg:win:<policy>:<category>:<scope>:<entity>`) never get an EXPIRE. Old members are trimmed only when the same key is accessed again (ZREMRANGEBYSCORE in `_purge_and_count` and in the Lua scripts), and an empty ZSET is never deleted. So one key per (policy, category, scope, entity) stays on the Redis server forever after its last request. This is the same 'every /api/ route, many IPs' growth TASK-13401 bounded in process, but here it is on the Redis server and is shared by all workers.

Add paths that create window keys (tldw_Server_API/app/core/Resource_Governance/governor_redis.py at commit dbb48312f4):
- `_add_members` (:542; ZADD at :545). Called from the stub-client add path in `reserve()` (:1686) and the Python real-Redis fallback add path (:1720).
- The multi-key reserve Lua script from `_ensure_multi_reserve_lua` (:653; ZADD at :712). Run from `reserve()` via EVALSHA at :1607.
- The tokens Lua script from `_ensure_tokens_lua` (:614; ZADD at :632). Today it only runs with a full window (`_allow_requests_sliding_check_only`, `peek_with_policy`), so it does not add in practice, but its script can.

Suggested fix: on every add, EXPIRE each window key at its policy window plus a small margin. Use `policy_eval.requests_window` for requests (60 s, or 60 * limit / rpm below 1 rpm) and 60 s for tokens. Cover both paths: in Lua, pass the window and call EXPIRE after the ZADD loop; in Python, call EXPIRE after `_add_members` (or pipeline ZADD + EXPIRE). Lease keys (`rg:lease:*`) and handle keys could get the same treatment; handles already EXPIRE after 86400 s.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Redis window key carries a TTL of at least its policy window
- [x] #2 A test proves an idle entity's window keys expire
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Every add to a window ZSET now EXPIREs the key at ceil(window) + 5 s (governor_redis._window_ttl): 65 s at 60 s windows, longer when rpm < 1. The TTL is never shorter than the window, so an idle entity's rg:win:* keys expire once their window has passed.
Multi-key reserve Lua: per-key ARGV group (limit, window, units, ttl, csv), EXPIRE after the ZADD loop. Tokens Lua: ttl is ARGV[4], EXPIRE after ZADD. Python paths (stub and real-Redis fallback) EXPIRE in _add_members. The in-memory stub's tokens-script emulation (_eval_rate_limiter) mirrors the EXPIRE.
The changed script has a new SHA, so workers on old and new code run side by side during a rolling deploy.
Tests: stub TTL tests in test_governor_redis.py (requests, tokens, rpm 0.3 / 600 s, tokens script) and real-Redis integration/test_redis_real_window_ttl.py (Lua TTLs, idle keys gone after a 1 s patched TTL). RG suite with real Redis (-n 4): 415 passed, 2 xfailed. Docs + redis_factory tests: 223 passed. Docs/Deployment/horizontal-scaling.md key table corrected (tokens are ZSETs, not fixed-window INCRBY) and Published refreshed.
Known limits: lease keys (rg:lease:*) get no TTL, because renew can shorten it and a correct TTL needs EXPIRE GT (Redis 7); only crashed processes leave lease keys behind. Keys written before this deploy get a TTL on their next write; keys of entities idle since before the deploy stay until deleted by hand. A reload that lengthens a requests window can leave an unwritten key with the old, shorter TTL, which allows a brief over-admit bounded by that window.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
RG Redis sliding-window keys now expire: every add sets a TTL of ceil(window) + 5 s on both the Lua paths and the Python paths, and the in-memory stub mirrors it. Real-Redis tests prove an idle entity's keys disappear. RG suite: 415 passed with real Redis.
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
