---
id: TASK-13401
title: >-
  RG Redis backend parity: unbounded in-process maps, burst ignored,
  fractional-rpm semantics
status: Done
assignee: []
created_date: '2026-09-30 09:45'
updated_date: '2026-10-02 01:30'
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
- [x] #1 Redis in-process maps are bounded (idle eviction like memory)
- [x] #2 Redis applies burst, or ADR-057 and the troubleshooting page document the difference
- [x] #3 Fractional-rpm policies behave the same on both backends within one window, with a test
- [x] #4 Rate-limit headers never report a limit of 0 for a fractional policy
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR reference: AC #2's 'ADR-057' means the RG safety-net ADR, now Docs/ADR/056-resource-governor-safety-net.md.

### Changes (branch fix/13401-rg-redis-parity)

AC1 (Redis in-process maps bounded): `governor_redis.py::_maybe_evict_idle`, called at the top of `reserve()`, reuses the memory backend's constants (`_EVICT_INTERVAL_SEC` 60 s, batch `max(_EVICT_BATCH, len // 10)`) and its rotate-in-place sweep. It drops expired entries from `_requests_deny_until` and `_stub_backoff_until` (until <= now), `_requests_accept_window` (window closed) and `_stub_leases` (every lease lapsed). The idle TTL is each entry's own expiry, not the memory backend's 600 s: an expired entry changes no decision, so eviction is lossless. To make that exact, the accept-window tuple now stores `(window_end, limit, count)` instead of `(start, ...)`, and the stub-rate tail smoothing in `reserve()` only applies inside the window (before, it could fire after the window closed, which made an expired tracker differ from no tracker under `RG_TEST_FORCE_STUB_RATE`). Not covered: `_local_handles` / `_fallback_handles` are per-reservation and removed on commit/release, and `_test_cleared_keys` only grows when the clock is below 1.0 (tests).

AC2 (burst on Redis): ruling kept. From 1 rpm up, Redis enforces `max(1, ceil(rpm))` per 60 s and does not apply burst. ADR-056 now says outright that the difference is deliberate, and `Docs/Operations/Rate_Limits_Troubleshooting.md` already documented it. Met through the "or documents the difference" branch.

AC3 (fractional rpm parity): `policy_eval.requests_window(policy) -> (limit, window_seconds)`. For rpm < 1 it returns `ceil(rpm * burst)` per `ceil(60 * burst)` s, with burst taken after `effective_policy`. Both values are `round(x, 6)` before the ceil so float noise such as `0.07 * 100 == 7.000000000000001` can't round up to 8. From 1 rpm up it returns `max(1, ceil(rpm))` per 60 s. It replaces `_requests_limit`, and `governor_redis._rate_window(policy, category)` sends requests through it (tokens stay `per_min` per 60 s).

Every Redis requests-window site and what happened to it:
- `_bootstrap_accept_window_from_zset`: the `start + 60.0` check and `_purge_and_count(window=60)` now use the policy window (new `window` parameter).
- `check()`: the requests branch's `rpm = _requests_limit; window = 60` became `limit, window = requests_window(pol)`. That one `window` already fed the accept-window deny, its `retry_after`, the stub-smoothing step, the deny floor and the sliding check.
- `check()`: the tokens branch's `window = 60` is unchanged (tokens).
- `reserve()`: the bootstrap call now passes the policy window.
- `reserve()`: the early accept-window guard's `start + 60.0`, step `60 / limit`, `60 - step` and `floor_until` now use `window_e` and `window_end`.
- `reserve()`: the stub-rate smoothing step `60 / lim` and its `60 - step` tail now use `window_e`, and the tail is limited to inside the window.
- `reserve()`: the early-deny decision's `limit` now comes from `requests_window`.
- `reserve()`: the denial-path floor (`rpm_b`, `win` read from a nonexistent `requests.window` policy key, defaulting to 60) now takes `rpm_b, win` from `requests_window`.
- `reserve()`: the real-Redis multi-key Lua ARGV window, the stub pre-check window, and the python-path add window (three `window = 60` sites) now go through `_rate_window`.
- `reserve()`: the stub-add limit and both denial-detail `lim` sites now go through `_rate_window`.
- `reserve()`: the rollback floor's `ra_df = 60` fallback and `floor_df` full-window fallback now use the policy window.
- `reserve()`: accept-window tracking's `start + 60.0` reset and `floor_until` now use `window_end`.
- `peek_with_policy`: `window = 60` (purge, Lua reset, fallback reset) now goes through `_rate_window`.
- Unchanged: concurrency `ttl_sec or 60` (6 sites), op/handle TTLs (86400/3600), and tokens quantum `_window_members` (requests quantum is always 1).

AC4 (headers never 0): the memory backend's `effective_limit = int(rpm)` became `requests_window(policy)[0]`. The Redis decision details already used the helper limit. The `middleware_simple.py` deny and success header fallbacks use `_policy_requests_limit` (that same limit, or 0 when the policy has no requests block, as before). Rule: `X-RateLimit-Limit` = `requests_window` limit on both backends (`{0.3, 10}` → 3, `{0.3}` → 1). This changes one thing for non-integer rpm ≥ 1: memory now reports `ceil(rpm)` (what Redis enforces) instead of `int(rpm)`.

Tests (written first; RED shown before the fix):
- `test_governor_safety_net.py`, on both backends:
  - `{0.3, 10}` admits exactly 3 then denies, and again after 600 s
  - `{0.5, 1}` admits 1 and denies, still denies at 119 s, admits at 120 s
  - decision limit is `[3, 1]` for `{0.3, 10}` / `{0.3}`
- `test_governor_safety_net.py`, Redis only:
  - peek and `retry_after` use the 600 s window
  - maps bounded under 50 entities, with the stub lease purge disabled, and an evicted entity behaves fresh
  - sweeps rotate through every entry
  - the batch scales with a flood
- `test_middleware_simple.py`: `X-RateLimit-Limit == 3` for `{rpm 0.3}` on memory, Redis and the no-limit fallback (allow and deny).
- `test_fractional_rpm_admits_then_refills` encoded the old Redis 1/min rounding. It was replaced by the parity tests above.

Docs: ADR-056 (Redis eviction; the explicit burst sentence plus the fractional window and header rule) and `Rate_Limits_Troubleshooting.md` (the fractional-rpm paragraph).

Bandit (uvx bandit -ll on the 4 touched source files): one Medium/Low B113 false positive on a pre-existing line in `policy_eval.effective_policy` (a dict named `requests`); nothing new.

Final run (`tldw_Server_API/tests/Resource_Governance tldw_Server_API/tests/AuthNZ_Unit -n 4`, `--timeout 300`): 1572 passed, 1 skipped, 2 xfailed, 5 failed. All 5 failures are Postgres-backed RG policy-store tests:
- `test_policy_admin_put_optimistic_concurrency_conflict`
- `test_policy_admin_upsert_delete_postgres`
- `test_policy_admin_list_count_and_metadata_postgres`
- `test_authnz_policy_store_postgres`
- `test_db_policy_loader_merges_route_map_from_file_on_postgres`

Each one hangs in the Docker Postgres fixture until the 300 s timeout kills its xdist worker. They fail the same way on an archived origin/dev snapshot (5 failed in 320 s), so they predate this change. The cause is the environment: stale `docker rm -f tldw_postgres_test` processes and no Postgres on :5432.

Known skips / follow-up: the Redis server-side window ZSET keys (`rg:win:*`) get no EXPIRE, so a key per (policy, entity) stays after it empties. That is Redis-server growth, not an in-process map, so it is outside AC1.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Redis in-process maps (deny floors, backoffs, acceptance windows, stub leases) are now bounded by a lossless expired-entry sweep that reuses the memory backend's interval, batch and rotation. Fractional-rpm policies now get the same immediate burst and long-run average on both backends through policy_eval.requests_window: ceil(rpm*burst) per 60*burst s, with every Redis requests-window site routed through it, including peek and retry_after. X-RateLimit-Limit reports that limit on both backends and in the middleware fallback, so it is never 0. Integer-rpm Redis semantics (burst not applied) are unchanged and documented as deliberate in ADR-056. Tests were added first on both backends, and docs were updated. 5 Postgres policy-store tests fail in this environment (the Docker Postgres fixture hangs) and fail identically on origin/dev.
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
