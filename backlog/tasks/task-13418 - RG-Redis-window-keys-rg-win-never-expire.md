---
id: TASK-13418
title: 'RG Redis window keys (rg:win:*) never expire'
status: To Do
assignee: []
created_date: '2026-10-02 02:02'
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
- [ ] #1 Every Redis window key carries a TTL of at least its policy window
- [ ] #2 A test proves an idle entity's window keys expire
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
