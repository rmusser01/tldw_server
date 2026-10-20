# RG Redis Governor Treats redis-py Errors as Redis Failures Implementation Plan

> **Review folded in (2026-10-20).** An independent review and dry run (`review_plan_fu_rg_redis_error.md`) found 1 Blocker, 3 Important and 8 Minor findings. All are folded in:
> - **B1:** Task 4 is new. A lease or rate script that Redis cannot run (NOSCRIPT after a restart, OOM, READONLY) falls back to the Python count and loads again; before, it denied every quota request, or dropped the limit, until the worker restarted.
> - **I1:** Task 3 gains the per-worker counting test.
> - **I2:** Decision 2 and Task 5 add the outage warning.
> - **I3:** Task 6 numbers the ADR when it runs, paging every PR's file list.
> - **Minor findings:**
>   - M1: the paging is folded into Task 6, Step 1.
>   - M2: the `fail_closed` 429 is Decision 3.
>   - M3: latency is Review Focus 4, with a follow-up.
>   - M4: `peek_with_policy` catches the error at its root, in Task 2.
>   - M5: `renew` is fixed in Task 3.
>   - M6: Redis not seeing the fallback leases is in Review Focus 2 and the ADR.
>   - M7: the expected outputs are corrected.
>   - M8: scratch files go to the session scratchpad.
>
> Every RED and GREEN count below was re-measured in a scratch worktree after the fold, on `origin/dev` `454974e812`.
>
> **Re-review folded in (2026-10-20).** A scoped re-review (`rereview_plan_fu_rg_and_anthropic.md`) found it ready with fixes: 1 Important and 4 Minor findings, all folded in.
> - **I1:** while Redis rejects writes (OOM, READONLY), the probe still answers, so each worker counts its leases on its own. The reviewer's two-worker probe admitted 4 against a limit of 2. ADR-078 and Review Focus 1 now say so.
> - **I1, logging:** Task 5 is reworked, test first. The warning now fires wherever a Redis error leaves the worker enforcing a limit on its own: a rate check or a rate window add, and a lease count or a lease add. That covers the reads failing under READONLY and the writes failing under OOM and READONLY, as well as an outage. The probe hook is gone, because these four sites cover an outage too.
> - **M1:** the warning is limited per worker process and namespace.
> - **M2:** this plan and the duplicate-id plan run one after the other.
> - **M3:** a renumbered ADR reaches every place in Task 6 and the plan copy.
> - **M4:** the admin diagnostic peek endpoint is named as the third caller of `peek_with_policy`.
>
> Task 5's counts changed; Tasks 1-4 are unchanged. Everything was re-measured on `origin/dev` `e521605198`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Once a worker has connected to Redis, a Redis outage reaches the Resource Governor (RG) as `redis.exceptions.RedisError`. RG then applies its fail mode in `reserve`, `renew`, `commit`, `peek_with_policy` and at ingress, instead of raising. A lease script that Redis cannot run falls back to counting in Python, and the outage is logged (TASK-13859).

**Architecture:**
- **`governor_redis.py`:**
  - `_RG_NONCRITICAL_EXCEPTIONS` gains `redis.exceptions.RedisError`, imported where redis is importable, the way `Usage/concurrency_quota.py` does. Every one of its 146 guards then catches redis-py's errors as it already catches the builtin `ConnectionError`.
  - `peek_with_policy` reports unknown headroom when Redis cannot be read. Before, it raised by design, which failed ingress headers and the character rate limiter's usage stats.
  - `commit` and `renew` fall back to the worker's own copy of the lease (`_local_handles`) when Redis cannot be read.
  - A script failure drops the cached SHA. For leases, it also falls through to the Python count, instead of denying or skipping.
- **`middleware_simple.py`:** `_RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS` gains the same, for any Redis error that still escapes `reserve`.
- **`policy_eval.py`:** gains `log_redis_failure`, a WARNING at most once a minute per worker process and namespace. The governor calls it wherever a Redis error leaves the worker enforcing a limit on its own: Redis unreachable, a read-only replica, or out of memory. The governor's own logger is off unless `RG_DEBUG` is set.
- **Docs:** ADR-078 (numbered at execution) records that concurrency leases now count per worker while Redis is down, which amends ADR-076. `Docs/Operations/Usage_Quotas.md` says the same.

**Tech Stack:** Python, redis-py (`redis.asyncio`, 8.1.0 in the venv, `redis>=4.5.0` in `pyproject.toml`), loguru, FastAPI `TestClient`, pytest, a real local Redis for the integration tests.

**Spec:** `backlog/tasks/task-13859 - RG-Redis-governor-RG-NONCRITICAL-EXCEPTIONS-misses-redis.exceptions.RedisError.md`:
- AC #1: "_RG_NONCRITICAL_EXCEPTIONS includes redis.exceptions.RedisError when redis is importable".
- AC #2: "A test raising redis.exceptions.ConnectionError from the Redis client shows reserve, renew and commit honour the governor's fail mode instead of raising".

## What was verified

First measured on `origin/dev` `a6719bf24f`; the fold was re-measured on `454974e812`. No commit in between touches a file this plan changes.

- **redis-py's errors are not the builtins.**
  - `redis.exceptions.ConnectionError.__mro__` is `(redis.exceptions.ConnectionError, RedisError, Exception, BaseException, object)`, and the same holds for its `TimeoutError`.
  - `NoScriptError`, `ReadOnlyError` and the OOM error are `ResponseError`s, which are `RedisError`s too.
  - A real `redis.asyncio.Redis` pointed at a closed port raises `redis.exceptions.ConnectionError`, which is neither a builtin `ConnectionError` nor an `OSError`. With `retry=Retry(NoBackoff(), 0)` it fails in under 1 ms, so the tests turn retries off. The production client (`redis_factory`, `from_url`) has no retry object and fails in about 10 ms on a refused connection.
- **Today a dropped Redis raises out of RG** (scratch probe `p13859_matrix.py`, not committed: a real client on port 1, all three fail modes, `requests`, `tokens` and `jobs`):
  - `reserve`, `renew` and `commit` raise `redis.exceptions.ConnectionError` in every case.
  - The same probe with a client that raises the builtin `ConnectionError` applies the fail modes. With `RedisError` added to the tuple, the redis-py run gives exactly the builtin run's results.
- **Ingress answers 500 today, in every fail mode,** over 7 requests per mode:
  - after Redis drops, because `reserve` raises and the middleware's guard misses `RedisError`;
  - after every Redis restart, because the cached multi-key script SHA is gone, and the resulting `NoScriptError` escapes `reserve` (the reviewer's probe `rv13859_noscript_ingress.py`).
- **With only the governor fixed, ingress still answers 500 for an admitted request.** The middleware's `_send_wrapped` calls `peek_with_policy` to build the rate-limit headers, and that calls `_purge_and_count`, which raises by design. It has two other callers:
  - `character_rate_limiter.get_usage_stats` (`characters_endpoint.py:1031`) guards with `RG_RUNTIME_EXCEPTIONS`, which also lacks `RedisError`;
  - the admin diagnostic `GET /resource-governor/diag/peek` (`endpoints/resource_governor.py:383`) answers its generic 500 during an outage. After Task 2 it answers `status: ok` with `remaining: null`.

  Task 2 catches the error in `peek_with_policy`, the shared root.
- **`commit` and `renew` miss the worker's copy of a lease.**
  - Both return as soon as `hgetall` raises.
  - `commit` never removes the lease from RG's in-process map, so under `fail_open` and `fail_closed`, which count leases from that map while Redis is down, a released slot stays taken until the TTL. Measured: `retry_after=59` with `max_concurrent: 1`.
  - `renew` never extends it, so a lease held past its TTL stops counting. `renew`'s copy also stores `categories` as a dict, which its `json.loads` rejects.
  - Builtin errors do the same today. Task 3 fixes both.
- **A Redis restart breaks the lease script path** (reviewer's probe `rv13859_noscript_probe.py`; Task 4's tests reproduce it):
  - The worker caches `_concurrency_lua_sha`, so after a restart every `evalsha` raises `NoScriptError` while Redis is healthy.
  - On dev, the error escapes and `concurrency_quota` admits with no lease.
  - With only Task 1, the guard at L1884 (on `origin/dev`) catches it. Under `fallback_memory` and `fail_closed` it then denies every quota request (`Retry-After: 120`) until the worker restarts. Under `fail_open` it skips the lease, so there is no limit.
  - The same guard catches OOM and READONLY, where Redis answers reads but rejects the script.
- **After Task 4 the limit holds across workers only after a restart** (the re-reviewer's probe `rr2/probe_oom_readonly.py`: two governors on one namespace, a limit of 2, faults injected on writes):
  - **NOSCRIPT** admits 2 in every mode: `A:ok B:ok A:no B:no`. The Python path counts against Redis, and the script reloads.
  - **READONLY and OOM** each admit 4 in every mode: `A:ok B:ok A:ok B:ok`.
    - Under READONLY, the Python path's `ZREMRANGEBYSCORE` fails, so `active_real` is 0 and only the worker's own map counts. Under `fallback_memory`, `check()` already hands the lease to the memory governor.
    - Under OOM, the reads work but `ZADD` fails, so the lease never reaches Redis.
  - **Logging.** In both cases the probe (`ZCARD`) still answers, so a warning on the probe alone would miss them. Task 5 logs at the four places a Redis error leaves the worker counting on its own.
- **Concurrency quotas change from "admit with no lease" to "count per worker"** (scratch probe `p13859_quota.py`, not committed, and `test_leases_count_per_worker_while_redis_is_down`; `acquire_slot` with a limit of 1 and Redis down):
  - Today every request is admitted with no lease, in every mode.
  - After this plan, the first request takes a lease, and an overlapping second one is refused (`retry_after` 119 or 120). After the first releases, the next is admitted. This holds in all three modes.
  - ADR-076 says the opposite ("If Redis drops later, a reserve that fails admits the request"). See Decision 1.
- **The outage is silent after the plan unless Task 5 lands.**
  - Today `concurrency_quota` logs a WARNING on every request, and ingress answers 500s: 9 WARNING lines in the reviewer's 3 acquires × 3 modes.
  - After Tasks 1-4, RG absorbs the error and logs nothing, because its logger is disabled without `RG_DEBUG`, and no RG metric counts backend errors. See Decision 2.
  - Task 5's warning was re-measured by the re-reviewer at its first placement (`rr2/probe_warning.py`): exactly 1 line for each of 12 outage cases, 0 on a healthy Redis and 0 on the in-memory stub. Thread races can only add a duplicate line. The four sites it moved to are covered case by case by `test_redis_rejecting_writes_is_logged`.
- **Ingress after this plan** (scratch probe `p13859_ingress.py`, not committed, policy `rpm: 5`, 7 requests per mode):
  - `fail_open`: five 200s, then 429 with `Retry-After: 59`. Each worker's acceptance window still applies the requests limit.
  - `fallback_memory`: five 200s, then 429 with `Retry-After: 12`, from the memory governor.
  - `fail_closed`: every request is 429, with `Retry-After: 60` on the first and 59 after it. See Decision 3.
- **Nothing on the default path relies on the 500.** The shipped `config.txt` sets `redis_fail_mode = fallback_memory`, and `rg_redis_fail_mode()` defaults to it too.

## Where `_RG_NONCRITICAL_EXCEPTIONS` is used, and what a caught `RedisError` does

- **Count.** It is used only in `governor_redis.py`: one definition and 146 guards (`except` or `contextlib.suppress`), counted with an AST walk on `origin/dev`.
  - 72 guards wrap no `await`, so no Redis call can raise inside them. Adding `RedisError` changes nothing there.
  - The other 74 wrap Redis calls.
- **Line numbers** below are on `origin/dev`. Task 1 adds 9 lines above them.

| Function | Guards (around Redis calls) | When a `RedisError` is caught |
| --- | --- | --- |
| module import; `_RedisKeys.backoff`; `__init__`; `_reg`; `_accept_window_enabled`; `_force_stub_rate`; `_use_stub_rate`; `_stub_lease_purge_and_count`; `_lookup_policy`; `_effective_fail_mode`; `_fallback_allowed`; `_consume_daily_caps_for_reserve` | 16 (0) | Pure Python; no change |
| `_maybe_test_purge_leases`, `test_force_clear_windows`, `_maybe_test_purge_windows_once` | 15 (11) | Test-only purges are skipped. They run only with the stub client or `RG_TEST_PURGE_LEASES_BEFORE_RESERVE` |
| `_bootstrap_accept_window_from_zset` | 3 (2) | The acceptance-window bootstrap is skipped |
| `_purge_and_count` | 1 (1) | Only its FakeTime cleanup branch. The purge and count raise to the caller by design |
| `_add_members`, `_zrem_members` | 2 (2) | The add or remove is skipped (an add is logged at debug). After Task 5, a failed add is logged as a warning |
| `_allow_requests_sliding_check_only` | 4 (4) | **The fail mode, for `requests` and `tokens`:** `fallback_memory` raises `_FallbackToMemory`, and `check()` hands the request to the memory governor; `fail_open` allows; `fail_closed` denies with `retry_after` = the window (60 s). After Task 5, logged as a warning |
| `_ensure_tokens_lua`, `_ensure_multi_reserve_lua`, `_ensure_concurrency_reserve_lua` | 3 (3) | Returns `None`, so no Lua script is used |
| `_is_real_redis` | 2 (2) | Returns `False`, so the dead client takes the stub path: Python sliding windows, which apply the fail mode |
| `_concurrency_retry_after_for_deficit` | 2 (1) | `Retry-After` comes from the worker's own leases |
| `check` | 18 (3) | L906: the client probe treats the client as a stub. L1027: the Redis backoff key is not set. **L1125, `streams` and `jobs`:** `fallback_memory` raises `_FallbackToMemory`; `fail_open` and `fail_closed` count the worker's own leases (`active_real = 0`). After Task 5, logged as a warning |
| `reserve` | 49 (27) | Idempotency (`op:` keys) reads and writes are skipped, so an `op_id` replay is not remembered during an outage. **L1651:** the multi-key Lua path falls back to Python; Task 4 also drops the cached SHA. L1750: the real-Redis add loop continues under `fail_open` and denies otherwise; it is reached only when Redis passed the probe and then failed. **L1884, the lease script:** reached on *every* reserve after a Redis restart (NOSCRIPT), and while Redis answers reads but rejects the script (OOM, READONLY). After Task 1 alone it denies, or skips under `fail_open`; Task 4 makes it drop the cached SHA and count leases in the Python path. L1901: leases are counted from the worker's own map. L1933: the lease `zadd` is skipped, keeping the worker's copy; after Task 5, logged as a warning. L2035: the handle `hset` is skipped, keeping `_local_handles` |
| `commit` | 16 (8) | The `op:` key, lease `zrem` and refund removals are skipped. **L2269, the outer guard,** returns in both modes without reading the handle, so the worker's copy is not released. Task 3 fixes this |
| `refund` | 8 (6) | Skipped |
| `renew` | 5 (2) | The `zadd` is skipped. **L2440:** nothing is renewed, including the worker's copy. Task 3 fixes this |
| `peek_with_policy` | 1 (1) | Only the Lua reset estimate. `_purge_and_count` raises to the caller; Task 2 catches it here |
| `capabilities` | 1 (1) | Reports `real_redis: false` |

**By fail mode, once Redis has dropped (after this plan):**
- **`requests` and `tokens`.**
  - `fail_open` admits. Each worker's acceptance window still holds `requests` to its limit.
  - `fail_closed` denies with `Retry-After` up to 60.
  - `fallback_memory` (the default) uses the worker's memory governor.
- **`streams` and `jobs`.**
  - Every mode counts leases per worker: `fallback_memory` in the memory governor, `fail_open` and `fail_closed` in RG's in-process map.
  - `fail_closed` does not refuse a lease. Builtin errors behave the same today; see Decision 4.
- **`renew` and `commit`** never raise, and they act on the worker's copy of the lease.
- **Ingress** answers 200 or 429 by fail mode, never 500.
- **The outage** is logged as a WARNING, at most once a minute per worker process and namespace. So is Redis rejecting writes (READONLY, OOM).

## Global Constraints

- **Branch:** `fix/rg-redis-error-noncritical` from `origin/dev`. The PR targets `dev`. One PR.
- **The import, verbatim, in both modules:**
  ```python
  try:
      from redis.exceptions import RedisError
  except ImportError:  # pragma: no cover - redis is an optional deployment dependency
      _REDIS_ERRORS: tuple[type[Exception], ...] = ()
  else:
      # redis-py's ConnectionError and TimeoutError are not the builtins: they derive from RedisError.
      _REDIS_ERRORS = (RedisError,)
  ```
  Each tuple ends with `*_REDIS_ERRORS,`. Never widen either tuple to `Exception`.
- **Tests raise real redis-py errors from real clients:**
  - Outage tests use `redis.asyncio.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0))`. Nothing listens on port 1 and retries are off, so every command raises `redis.exceptions.ConnectionError` at once.
  - The script tests (Task 4) run on the real local Redis through the `real_redis` and `rg_unique_ns` fixtures. They set a cached SHA that Redis does not know (`"f" * 40`). Never `SCRIPT FLUSH` a shared Redis.
  - No fake client is defined in a test module, and there is no `RuntimeError` stand-in for the Redis error.
- **The governor is "connected" by assigning `rg._client`.** Outage tests never reach Redis on 6379.
- **`RG_TEST_PURGE_LEASES_BEFORE_RESERVE`.** The RG `conftest.py` sets it for every RG test, and its test-only purge drops the worker's leases once `SCAN` fails. Tests about counting leases during an outage delete it, as production never sets it.
- **Test first:** each task's RED run happens before its source change, and fails for the reason given.
- **Test-quality ratchet:** `Helper_Scripts/ci/test_quality_triage.py --enforce` must stay at `OK`.
  - The ingress test reuses `_fractional_app` and asserts a header, not only the status.
  - The `fail_closed` 503 test reuses `_ExplodingGov`, now with an error parameter.
  - An earlier draft that built its own `_Loader` app and asserted only the status failed the ratchet: `status_only=1 (new)`, `stub_injection=4 (baseline 3)`.
- **Scratch files:** `SCRATCH` is this session's scratchpad directory. Save diffs and the PR body there, never in `/tmp`.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit ids, `git add -A backlog/tasks`.
- **Mirrored docs:** ADRs are mirrored. Run `bash Helper_Scripts/refresh_docs_published.sh` and commit `Docs/Published/ADR`. `Docs/Operations/Usage_Quotas.md` is not mirrored.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`.
- **Before merge:**
  - The PR needs the owner's Change summary (ADR-004).
  - A clean-subagent `/superpowers:requesting-code-review` must run.
  - Check `gh variable get MERGE_QUEUE` for the merge procedure.
- **Out of scope, recorded in Review Focus:**
  - `fail_closed` refusing concurrency leases (Decision 4);
  - the `reserve` race at L1739;
  - a connect timeout on the RG client (follow-up, Task 6).

## Decisions for the owner

1. **Concurrency leases count per worker during a Redis outage, which amends ADR-076.**
   - **Recommended:** accept it, recorded as ADR-078 (Task 6). Concurrency limits keep holding per worker, as they already do when Redis is down at first connect, and no request fails.
   - **Today:** a quota request is admitted with no lease, because the `RedisError` escapes RG. ADR-076's Consequences say so, and accepted ADRs are immutable, so the change needs a new ADR.
   - **Alternative:** keep "admit with no lease". RG has no fail mode that admits a lease without counting it, so `concurrency_quota` would need its own path around RG, and a user could start unlimited work for the length of an outage. Not recommended.
2. **The outage must not be silent.**
   - **Recommended:** Task 5 logs a WARNING, from `policy_eval.py`, which is not disabled.
     - It is rate-limited to once a minute per worker process and Redis namespace, so N workers log N lines a minute.
     - It fires wherever a Redis error leaves the worker enforcing a limit on its own: a rate check or a rate window add (`_allow_requests_sliding_check_only`, `_add_members`), and a lease count or a lease add (`check`, `reserve`).
     - That covers an unreachable Redis, and also OOM and READONLY, where the probe still answers. A restart (NOSCRIPT) is not logged, because the limit still holds across workers.
     - `Usage_Quotas.md` keeps "a warning is logged", which is then true again.
   - **No counter:** none of RG's metrics (`rg_decisions_total`, `rg_denials_total`, `rg_refunds_total`, `rg_concurrency_active`, `rg_shadow_decision_mismatch_total`) fits a backend error, and a new metric is more than this task needs.
   - **Alternative:** drop Task 5, record "the outage is silent in RG" in ADR-078 and `Usage_Quotas.md`, and file a follow-up. Operators then cannot tell that quotas have gone per-worker, or why `fail_closed` answers 429 everywhere.
3. **Under `fail_closed`, ingress answers 429 while Redis is down.**
   - **Recommended:** keep the 429 (`rate_limited`, `Retry-After` up to 60). It needs no new plumbing, and the `Resource_Governor_PRD.md` already warns that `fail_closed` deployments 429 on an outage.
   - **Alternative:** a 503 `resource_governance_unavailable`, which the middleware already sends for a governor that raises. That needs the governor to mark denials caused by a backend failure.
   - **Today:** a 500.
4. **`fail_closed` still admits `streams` and `jobs` leases during an outage**, counting them per worker.
   - **Recommended:** keep it, with no follow-up unless you want concurrency to fail closed.
   - The PRD says `fail_closed` denies on Redis errors. `check()` has always counted leases in process instead (L1125), for builtin errors too.
   - Flipping it would refuse every media ingest, audio stream and transcription that has a quota, while Redis is down under `RG_REDIS_FAIL_MODE=fail_closed`.

## ADR assessment

- **ADR-076** (Accepted) states the outage behaviour this plan changes. **A new ADR is required** (Decision 1). It amends ADR-076's "Lease-store failures" consequence, and the ADR index marks ADR-076 "Amended in part by" it.
- **Its number is set when Task 6 runs.**
  - Task 6, Step 1 takes the next free number after `origin/dev` and every open PR, paging each PR's file list.
  - On 2026-10-20 that was 078: dev's highest is 076, and open PR #3269 holds 077.
  - The duplicate-id renumbering plan also numbers its ADR when it runs, so there is no static reservation to avoid.
  - #3269 adds its index row after ADR-076 too, so whichever merges second has a one-line conflict in `Docs/ADR/README.md`.
- **ADR-056** (RG safety net) is upheld:
  - **No configuration produces a permanent 429.** Before Task 4, a Redis restart under `fail_closed` or `fallback_memory` made every quota reserve deny until the worker restarted, which was permanent in effect. Task 4 removes that.
  - `fail_closed` 429s only while Redis is down, and for up to 60 s after, as each worker's deny floor runs out. The PRD already warns about this.
  - `RG_ENABLED=false` still disables ingress.

## Review Focus

1. **A Redis restart, failover, OOM or READONLY while the worker keeps its connection.**
   - Redis answers reads but cannot run the cached scripts.
   - **Leases after a restart (NOSCRIPT):** after Task 4, the reserve counts leases in the Python path against Redis (`ZCARD`), so the limit still holds across workers. The script loads again on the next reserve.
   - **Leases while Redis rejects writes (OOM, READONLY):** each worker counts on its own, as during an outage, so N workers admit up to N times the limit.
     - The re-reviewer measured 4 admitted against a limit of 2 on two workers.
     - Each reserve also loads the script and fails it again for as long as the condition lasts.
     - Task 5 logs it, because the probe still answers; `test_redis_rejecting_writes_is_logged` covers 4 cases.
   - **Rates:** after a restart, the request falls back to the Python windows once, then reloads. While Redis rejects writes (OOM or READONLY), the failed write or read gets the fail mode, as while Redis is unreachable; either way, Task 5 logs it. *(Updated in review of PR #3288: before that fold, a rejected OOM write was swallowed and no fail mode applied.)*
   - **Retry estimates:** a failing `evalsha` sets a denial's `Retry-After` to the full window once, and `_tokens_lua_sha` is cleared so the next estimate reloads the script. *(Updated in review of PR #3288.)*
   - Covered by `test_a_lost_lease_script_falls_back_to_counting_in_python` (3 modes) and `test_a_lost_rate_script_falls_back_once_then_reloads`.
2. **Leases on the other side of an outage.**
   - During the outage, a worker counting on its own does not see leases taken before Redis dropped.
   - After Redis returns, Redis does not see the leases the memory governor took during the outage (`fallback_memory`), until those requests end.
   - Either way, a user can briefly exceed the limit by the leases the other side holds.
   - ADR-078 records it. No test: it needs two stores in different states.
3. **Ingress under `fail_closed` answers 429 (`Retry-After` up to 60), not the middleware's 503** (Decision 3).
   - The per-worker deny floor holds for up to 60 s after Redis returns.
   - `test_ingress_honours_the_fail_mode_when_redis_raises_connection_error[fail_closed-429]` pins the 429.
4. **Latency with a black-holed Redis.**
   - **More Redis calls per request.** With Redis down, the reviewer counted:
     - ingress: 1 call on dev; with the plan, 7 (`fallback_memory`), 14-15 (`fail_open`) or 3-8 (`fail_closed`);
     - a quota acquire: 1 on dev; with the plan, 2 or 8;
     - a shielded release: 0 on dev; with the plan, 3.
   - **Refused connections** fail in about 10 ms, so this is harmless.
   - **Black-holed connections** are not: redis-py 8.1's `from_url` waits 5 s per call, so a governed request takes about 30-70 s. *(Updated in review of PR #3288: not the OS connect timeout, as first written; see TASK-13882.)*
   - Task 6 files a follow-up for a connect timeout, or a short circuit breaker, on the RG client. No test.
5. **A Redis error that still escapes `reserve`.**
   - The real-Redis add loop called `_purge_and_count` unguarded at L1739. *(Updated in review of PR #3288: a read-only Redis reached it on every request, not only in a race; the line is deleted. A Redis that drops between `check()` and the add now gets the fail mode in `reserve`, so nothing escapes.)*
   - The middleware turns that error into its fail-mode answer (Task 2): 503 under `fail_closed`, otherwise the request passes through. `concurrency_quota` already admits on it.
   - `test_middleware_fail_closed_on_reserve_error_when_policy_requires_it[redis]` pins the 503 for a `RedisError` from `reserve`. No test reproduces the race itself.

---

Set once for every task:
- `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`;
- `export TLDW_TEST_NO_DOCKER=1`;
- `SCRATCH=` this session's scratchpad directory.

The Task 4 and Task 5 integration tests need a local Redis on 6379, as `ci.yml` provides. The RG `conftest.py` defaults `REDIS_URL` to it. Without one they skip, and their RED step shows `skipped` instead of `failed`.

### Task 1: RG catches redis-py's errors in reserve, renew and commit

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/governor_redis.py:34-47`: the import, and `*_REDIS_ERRORS` at the end of `_RG_NONCRITICAL_EXCEPTIONS`.
- Test: `tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py`:
  - `_redis_down_client()` and `_governor(policy, ns, time_source=time.time)`;
  - `test_reserve_honours_the_fail_mode_when_redis_raises_connection_error` (6 cases);
  - `test_renew_and_commit_survive_redis_raising_connection_error` (3 cases).
- Add: this plan, as `Docs/superpowers/plans/2026-10-20-rg-redis-error-noncritical.md`.

**Interfaces:**
- Produces:
  - `governor_redis._REDIS_ERRORS: tuple[type[Exception], ...]`;
  - in the test module, `_redis_down_client() -> redis.asyncio.Redis` and `_governor(policy: dict, ns: str, time_source=time.time) -> RedisResourceGovernor`. Tasks 2, 3 and 5 reuse both.

- [ ] **Step 1: Branch, and claim the task**

```bash
git fetch -q origin dev
git switch -c fix/rg-redis-error-noncritical origin/dev
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13859 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-10-20-rg-redis-error-noncritical.md (branch fix/rg-redis-error-noncritical)."
```

Copy this plan to `Docs/superpowers/plans/2026-10-20-rg-redis-error-noncritical.md`.

- [ ] **Step 2: Write the failing tests**

Save this diff as `$SCRATCH/t1_tests.diff` and apply it with `git apply $SCRATCH/t1_tests.diff`. Every later diff is saved and applied the same way.

```diff
diff --git a/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py b/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
index e42aba325e..8b35faed4e 100644
--- a/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
+++ b/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
@@ -1,7 +1,13 @@
+import time
+
 import pytest
+import redis.asyncio as aioredis
+from redis.asyncio.retry import Retry
+from redis.backoff import NoBackoff

 pytestmark = pytest.mark.rate_limit

+from tldw_Server_API.app.core.Infrastructure.redis_factory import InMemoryAsyncRedis
 from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest


@@ -70,3 +76,46 @@ async def test_fallback_memory_requests_allows_when_redis_broken(monkeypatch):
     d2, h2 = await rg.reserve(req, op_id="f2")
     # Memory fallback should enforce rpm=1
     assert (not d2.allowed) and (h2 is None)
+
+
+def _redis_down_client() -> aioredis.Redis:
+    """A real redis-py client whose server is gone: every command raises redis.exceptions.ConnectionError.
+
+    Nothing listens on port 1, and retries are off, so each command fails at once.
+    """
+    return aioredis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0), decode_responses=True)
+
+
+def _governor(policy: dict, ns: str, time_source=time.time) -> RedisResourceGovernor:
+    class _Loader:
+        def get_policy(self, pid):
+            return policy
+
+    return RedisResourceGovernor(policy_loader=_Loader(), ns=ns, time_source=time_source)
+
+
+@pytest.mark.asyncio
+@pytest.mark.parametrize("category, limits", [("requests", {"rpm": 5}), ("tokens", {"per_min": 5})])
+@pytest.mark.parametrize("fail_mode, admitted", [("fail_open", True), ("fallback_memory", True), ("fail_closed", False)])
+async def test_reserve_honours_the_fail_mode_when_redis_raises_connection_error(category, limits, fail_mode, admitted):
+    rg = _governor({category: limits, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_{category}_{fail_mode}")
+    rg._client = _redis_down_client()  # the worker connected once; Redis has gone away since
+    req = RGRequest(entity="user:x", categories={category: {"units": 1}}, tags={"policy_id": "p"})
+
+    decision, handle = await rg.reserve(req)
+
+    assert (decision.allowed, handle is not None) == (admitted, admitted)
+
+
+@pytest.mark.asyncio
+@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
+async def test_renew_and_commit_survive_redis_raising_connection_error(fail_mode):
+    rg = _governor({"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_lease_{fail_mode}")
+    rg._client = InMemoryAsyncRedis()
+    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})
+    decision, handle = await rg.reserve(req)
+    assert decision.allowed and handle
+
+    rg._client = _redis_down_client()  # Redis goes away while the lease is held
+    await rg.renew(handle, 60)
+    await rg.commit(handle)
```

- [ ] **Step 3: Run them to see them fail**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
```

Expected:
- **The summary** is `9 failed, 3 passed`.
- **The 9 failures** are every case of the two new tests. Each ends in `redis.exceptions.ConnectionError: Error 61 connecting to 127.0.0.1:1. Connect call failed ('127.0.0.1', 1).` The errno and wording differ by platform.
- **Where each escapes:**
  - the `requests` cases, from `reserve` → `_bootstrap_accept_window_from_zset` → `_is_real_redis` (`client.zcard`);
  - the `tokens` cases, from `reserve` → `_maybe_test_purge_leases` → `_scan_keys`. That purge is test-only and runs because of the RG conftest's env;
  - the renew/commit cases, from `renew` → `client.hgetall`, so `commit` is never reached.
- **The 3 passes** are the existing `RuntimeError` tests.

- [ ] **Step 4: Catch `RedisError` in RG**

```diff
diff --git a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
index 65c342832f..bd0d1bd4dc 100644
--- a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
+++ b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
@@ -31,6 +31,14 @@ from .policy_eval import (
 )
 from .tenant import hash_entity

+try:
+    from redis.exceptions import RedisError
+except ImportError:  # pragma: no cover - redis is an optional deployment dependency
+    _REDIS_ERRORS: tuple[type[Exception], ...] = ()
+else:
+    # redis-py's ConnectionError and TimeoutError are not the builtins: they derive from RedisError.
+    _REDIS_ERRORS = (RedisError,)
+
 TimeSource = Callable[[], float]

 _RG_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
@@ -44,6 +52,7 @@ _RG_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
     UnicodeDecodeError,
     ValueError,
     json.JSONDecodeError,
+    *_REDIS_ERRORS,
 )

 try:
```

- [ ] **Step 5: Run the tests to see them pass**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
```

Expected: `12 passed`.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
git commit -m "fix(rg): the Redis governor catches redis-py errors and applies its fail mode (TASK-13859)

redis-py's ConnectionError and TimeoutError derive from RedisError, not
from the builtins, so _RG_NONCRITICAL_EXCEPTIONS (builtins only) let a
dropped Redis raise out of reserve, renew and commit in every fail mode.
RedisError now joins the tuple where redis is importable, so all 146
best-effort guards treat it as they treat the builtin ConnectionError.
Tests raise a real redis.exceptions.ConnectionError from a real client
on a closed port.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: Ingress and header builders survive a Redis outage

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/governor_redis.py` (`peek_with_policy`, around `_purge_and_count`; line 2489 after Task 1).
- Modify: `tldw_Server_API/app/core/Resource_Governance/middleware_simple.py:30-42`: the same import, and `*_REDIS_ERRORS` at the end of `_RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS`.
- Test: `tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py`, `test_peek_reports_unknown_headroom_when_redis_raises_connection_error`.
- Test: `tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py`:
  - `_redis_down_governor(loader)`;
  - `test_ingress_honours_the_fail_mode_when_redis_raises_connection_error` (3 cases);
  - `_ExplodingGov(error=None)`, with `test_middleware_fail_closed_on_reserve_error_when_policy_requires_it` parametrized over a builtin and a redis-py error.

**Interfaces:**
- Consumes: Task 1's tuple. Without it, `reserve` raises before ingress reaches the header code.
- Produces:
  - `peek_with_policy(...)` returns `{"remaining": None, "reset": None}` for a rate category whose window cannot be read;
  - `middleware_simple._REDIS_ERRORS`.

- [ ] **Step 1: Write the failing tests**

```diff
diff --git a/tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py b/tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py
index dbf7c411cf..b11267d44d 100644
--- a/tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py
+++ b/tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py
@@ -5,6 +5,7 @@ from typing import Any
 import pytest
 from fastapi import FastAPI
 from fastapi.testclient import TestClient
+from redis import exceptions as redis_exceptions

 from tldw_Server_API.app.core.Resource_Governance.governor import RGDecision
 from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
@@ -61,8 +62,11 @@ class _Gov:


 class _ExplodingGov:
+    def __init__(self, error: Exception | None = None):
+        self._error = error or RuntimeError("resource governor unavailable")
+
     async def reserve(self, req, op_id=None):
-        raise RuntimeError("resource governor unavailable")
+        raise self._error


 class _CaptureGov:
@@ -206,7 +210,12 @@ async def test_middleware_resolves_new_domain_paths(path: str, pattern: str, pol


 @pytest.mark.asyncio
-async def test_middleware_fail_closed_on_reserve_error_when_policy_requires_it():
+@pytest.mark.parametrize(
+    "error",
+    [RuntimeError("resource governor unavailable"), redis_exceptions.ConnectionError("Connection refused")],
+    ids=["builtin", "redis"],
+)
+async def test_middleware_fail_closed_on_reserve_error_when_policy_requires_it(error):
     app = FastAPI()
     app.add_middleware(RGSimpleMiddleware)

@@ -216,7 +225,7 @@ async def test_middleware_fail_closed_on_reserve_error_when_policy_requires_it()

     route_map = {"by_path": {"/api/v1/fail-closed": "fail.closed"}}
     app.state.rg_policy_loader = _Loader(route_map, policies={"fail.closed": {"fail_mode": "fail_closed"}})
-    app.state.rg_governor = _ExplodingGov()
+    app.state.rg_governor = _ExplodingGov(error)

     with TestClient(app) as c:
         r = c.get("/api/v1/fail-closed")
@@ -353,3 +362,29 @@ def test_fractional_policy_header_fallback_reports_capacity_not_zero(allowed: bo
     with TestClient(_fractional_app(lambda _loader: _NoLimitGov(allowed))) as c:
         r = c.get("/api/v1/echo")
     assert (r.status_code, r.headers.get("X-RateLimit-Limit")) == (status, "3")
+
+
+def _redis_down_governor(loader: _Loader) -> Any:
+    """A Redis-backend governor whose server has gone: every command raises redis.exceptions.ConnectionError."""
+    import redis.asyncio as aioredis
+    from redis.asyncio.retry import Retry
+    from redis.backoff import NoBackoff
+
+    from tldw_Server_API.app.core.Resource_Governance.governor_redis import RedisResourceGovernor
+
+    gov = RedisResourceGovernor(policy_loader=loader, ns="rg_t_mw_redis_down")
+    # Nothing listens on port 1, and retries are off, so each command fails at once.
+    gov._client = aioredis.Redis(host="127.0.0.1", port=1, socket_connect_timeout=0.2, retry=Retry(NoBackoff(), 0))
+    return gov
+
+
+@pytest.mark.unit
+@pytest.mark.parametrize("fail_mode, status", [("fail_open", 200), ("fallback_memory", 200), ("fail_closed", 429)])
+def test_ingress_honours_the_fail_mode_when_redis_raises_connection_error(
+    fail_mode: str, status: int, monkeypatch: pytest.MonkeyPatch
+) -> None:
+    monkeypatch.setenv("RG_REDIS_FAIL_MODE", fail_mode)
+    with TestClient(_fractional_app(_redis_down_governor)) as c:
+        r = c.get("/api/v1/echo")
+    # The governor decided (the policy's limit is in the headers): not a 500, and not the 503 for a governor error.
+    assert (r.status_code, r.headers.get("X-RateLimit-Limit")) == (status, "3")
diff --git a/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py b/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
index 8b35faed4e..e44d891208 100644
--- a/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
+++ b/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
@@ -119,3 +119,12 @@ async def test_renew_and_commit_survive_redis_raising_connection_error(fail_mode
     rg._client = _redis_down_client()  # Redis goes away while the lease is held
     await rg.renew(handle, 60)
     await rg.commit(handle)
+
+
+@pytest.mark.asyncio
+async def test_peek_reports_unknown_headroom_when_redis_raises_connection_error():
+    rg = _governor({"requests": {"rpm": 5}, "scopes": ["user"]}, ns="rg_down_peek")
+    rg._client = _redis_down_client()
+
+    # Ingress headers and the character rate limiter's usage stats read this; it must not raise.
+    assert await rg.peek_with_policy("user:x", ["requests"], "p") == {"requests": {"remaining": None, "reset": None}}
```

- [ ] **Step 2: Run them to see them fail**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py
```

Expected:
- **The summary** is `4 failed, 31 passed`.
- **Three failures** raise `redis.exceptions.ConnectionError: Error 61 ...`:
  - `test_peek_reports_unknown_headroom_when_redis_raises_connection_error`, from `peek_with_policy` → `_purge_and_count`;
  - `test_ingress_honours_the_fail_mode_when_redis_raises_connection_error[fail_open-200]` and `[fallback_memory-200]`, from `middleware_simple.py` `_send_wrapped` → `peek_with_policy`.
- **`test_middleware_fail_closed_on_reserve_error_when_policy_requires_it[redis]`** fails with `redis.exceptions.ConnectionError: Connection refused`, which escapes the middleware.
- **`[fail_closed-429]` already passes:** a denial builds its headers from the decision and never calls `peek_with_policy`.

- [ ] **Step 3: Catch the error in `peek_with_policy` and in the middleware**

```diff
diff --git a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
index bd0d1bd4dc..0f3a21af40 100644
--- a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
+++ b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
@@ -2486,7 +2486,12 @@ class RedisResourceGovernor(ResourceGovernor):
             for sc, ev in self._scope_pairs(pol, entity_scope, entity_value):
                 current_cnt = 0
                 key = self._keys.win(policy_id, category, sc, ev)
-                current_cnt = await self._purge_and_count(key=key, now=now, window=window)
+                try:
+                    current_cnt = await self._purge_and_count(key=key, now=now, window=window)
+                except _RG_NONCRITICAL_EXCEPTIONS:
+                    # Redis cannot be read: this category's headroom is unknown.
+                    remainings, resets = [], []
+                    break
                 if current_cnt >= limit_m:
                     try:
                         res = await self._ensure_tokens_lua()
diff --git a/tldw_Server_API/app/core/Resource_Governance/middleware_simple.py b/tldw_Server_API/app/core/Resource_Governance/middleware_simple.py
index a1ae85ac17..6f5cb9b5ee 100644
--- a/tldw_Server_API/app/core/Resource_Governance/middleware_simple.py
+++ b/tldw_Server_API/app/core/Resource_Governance/middleware_simple.py
@@ -27,6 +27,14 @@ from .governor import RGRequest
 from .policy_eval import requests_window
 from .tenant import TenantScopeConfig, get_tenant_id, parse_tenant_config

+try:
+    from redis.exceptions import RedisError
+except ImportError:  # pragma: no cover - redis is an optional deployment dependency
+    _REDIS_ERRORS: tuple[type[Exception], ...] = ()
+else:
+    # redis-py's ConnectionError and TimeoutError are not the builtins: they derive from RedisError.
+    _REDIS_ERRORS = (RedisError,)
+
 _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
     AttributeError,
     ConnectionError,
@@ -39,6 +47,7 @@ _RG_MIDDLEWARE_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
     UnicodeDecodeError,
     ValueError,
     re.error,
+    *_REDIS_ERRORS,
 )

 # ponytail: per-process identity cache. A revoked credential keeps charging its old
```

- [ ] **Step 4: Run the tests to see them pass**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py
```

Expected: `35 passed`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/app/core/Resource_Governance/middleware_simple.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py
git commit -m "fix(rg): ingress answers by fail mode, not 500, when Redis drops (TASK-13859)

peek_with_policy let Redis errors through by design, so building the
rate-limit headers of an admitted request answered 500 once Redis
dropped, and the character rate limiter's usage stats raised. It now
reports unknown headroom for a window it cannot read. The middleware's
own guard also gains RedisError, so a Redis error that still escapes
reserve gets the fail-mode answer (503 under fail_closed) instead of a
500.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 3: commit and renew act on the worker's copy of a lease when Redis is down

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/governor_redis.py`:
  - `commit`, around `client.hgetall` (line 2137 after Task 1);
  - `renew`, around `client.hgetall` and `json.loads(categories)` (line 2406 after Tasks 1-2).
- Test: `tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py`, 3 cases each:
  - `test_a_lease_committed_while_redis_is_down_frees_its_slot`;
  - `test_leases_count_per_worker_while_redis_is_down`;
  - `test_renew_while_redis_is_down_keeps_the_lease`.

**Interfaces:**
- Consumes: `_redis_down_client()` and `_governor(policy, ns, time_source)` from Task 1.
- Produces:
  - `commit` falls back to `self._local_handles[handle_id]` when the handle cannot be read from Redis, as it already does when Redis has no record;
  - `renew` does the same only when the read raises. When Redis answers with no record, it still returns without renewing.

- [ ] **Step 1: Write the failing tests**

```diff
diff --git a/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py b/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
index e44d891208..bbc2cf8303 100644
--- a/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
+++ b/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
@@ -128,3 +128,59 @@ async def test_peek_reports_unknown_headroom_when_redis_raises_connection_error(

     # Ingress headers and the character rate limiter's usage stats read this; it must not raise.
     assert await rg.peek_with_policy("user:x", ["requests"], "p") == {"requests": {"remaining": None, "reset": None}}
+
+
+@pytest.mark.asyncio
+@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
+async def test_a_lease_committed_while_redis_is_down_frees_its_slot(fail_mode, monkeypatch):
+    # The RG conftest's test-only lease purge would drop the in-process lease for us; production never runs it.
+    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
+    rg = _governor({"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_free_{fail_mode}")
+    rg._client = InMemoryAsyncRedis()
+    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})
+    _decision, handle = await rg.reserve(req)
+
+    rg._client = _redis_down_client()
+    await rg.commit(handle)
+    decision, again = await rg.reserve(req)
+
+    assert decision.allowed and again
+
+
+@pytest.mark.asyncio
+@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
+async def test_leases_count_per_worker_while_redis_is_down(fail_mode, monkeypatch):
+    # The RG conftest's test-only lease purge drops the in-process leases once Redis is down; production never runs it.
+    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
+    rg = _governor({"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_down_count_{fail_mode}")
+    rg._client = _redis_down_client()
+    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})
+
+    first, handle = await rg.reserve(req)
+    second, _ = await rg.reserve(req)
+    await rg.commit(handle)
+    third, _ = await rg.reserve(req)
+
+    assert [first.allowed, second.allowed, third.allowed] == [True, False, True]
+
+
+@pytest.mark.asyncio
+@pytest.mark.parametrize("fail_mode", ["fail_open", "fail_closed", "fallback_memory"])
+async def test_renew_while_redis_is_down_keeps_the_lease(fail_mode, monkeypatch):
+    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
+    now = [1000.0]
+    rg = _governor(
+        {"jobs": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"], "fail_mode": fail_mode},
+        ns=f"rg_down_renew_{fail_mode}",
+        time_source=lambda: now[0],
+    )
+    rg._client = _redis_down_client()
+    req = RGRequest(entity="user:x", categories={"jobs": {"units": 1}}, tags={"policy_id": "p"})
+    first, handle = await rg.reserve(req)  # the lease runs to 1060
+
+    now[0] = 1050.0
+    await rg.renew(handle, 60)  # now to 1110
+    now[0] = 1070.0
+    second, _ = await rg.reserve(req)
+
+    assert first.allowed and not second.allowed
```

The tests delete `RG_TEST_PURGE_LEASES_BEFORE_RESERVE` (see Global Constraints).

- [ ] **Step 2: Run them to see them fail**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
```

Expected:
- **The summary** is `6 failed, 16 passed`. The failures are the `[fail_open]` and `[fail_closed]` cases of the three new tests.
- **`test_a_lease_committed_while_redis_is_down_frees_its_slot`:** `AssertionError: assert (False)` on `RGDecision(allowed=False, retry_after=59, ... 'jobs': {'allowed': False, 'limit': 1, 'remaining': 0, ...`. The released lease still holds the only slot.
- **`test_leases_count_per_worker_while_redis_is_down`:** `assert [True, False, False] == [True, False, True]`. The overlapping second lease is already refused, but the commit did not free the slot.
- **`test_renew_while_redis_is_down_keeps_the_lease`:** `assert (True and not True)`. The renewal did not extend the lease, so it lapsed at 1060 and the reserve at 1070 was admitted.
- **The `[fallback_memory]` cases pass:** the memory governor holds those leases and handles them itself.

- [ ] **Step 3: Use the worker's copy when Redis cannot be read**

```diff
diff --git a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
index 0f3a21af40..ffcb8761f4 100644
--- a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
+++ b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
@@ -2134,7 +2134,11 @@ class RedisResourceGovernor(ResourceGovernor):
                         return
         try:
             hkey = self._keys.handle(handle_id)
-            data = await client.hgetall(hkey)
+            try:
+                data = await client.hgetall(hkey)
+            except _RG_NONCRITICAL_EXCEPTIONS:
+                # Redis is down: release from this process's copy, or the lease blocks its slot until TTL.
+                data = None
             if not data:
                 data = self._local_handles.get(handle_id) or {}
                 if not data:
@@ -2399,14 +2403,19 @@ class RedisResourceGovernor(ResourceGovernor):
         client = await self._client_get()
         try:
             hkey = self._keys.handle(handle_id)
-            data = await client.hgetall(hkey)
+            try:
+                data = await client.hgetall(hkey)
+            except _RG_NONCRITICAL_EXCEPTIONS:
+                # Redis is down: renew this process's copy, or the lease lapses after its TTL while still held.
+                data = self._local_handles.get(handle_id)
             if not data:
                 return
             policy_id = data.get("policy_id") or "default"
             entity = data.get("entity") or ""
             entity_scope, entity_value = self._parse_entity(entity)
             pol = self._get_policy(policy_id)
-            cats = json.loads(data.get("categories") or "{}")
+            cats_raw = data.get("categories")
+            cats = json.loads(cats_raw or "{}") if isinstance(cats_raw, str) else dict(cats_raw or {})
             members_raw = data.get("members")
             try:
                 members = json.loads(members_raw or "{}") if isinstance(members_raw, str) else (members_raw or {})
```

- [ ] **Step 4: Run the tests to see them pass**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
```

Expected: `22 passed`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
git commit -m "fix(rg): commit and renew act on the worker's copy of a lease while Redis is down (TASK-13859)

commit and renew returned as soon as the handle read failed, so the
worker's own copy of a lease was never released or renewed. Under
fail_open and fail_closed, which count that copy while Redis is down, a
released slot stayed taken until the TTL (a concurrency quota of 1
refused the user's next request for up to 120 s), and a long request's
lease lapsed while it ran. Both now fall back to the worker's copy when
Redis cannot be read; renew also reads that copy's categories dict,
which its json.loads rejected.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 4: A script Redis cannot run falls back to counting in Python

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/governor_redis.py`:
  - `reserve`'s multi-key Lua guard (`origin/dev` L1651; 1660 after Tasks 1-3), which drops `_multi_lua_sha`;
  - the lease Lua guard (`origin/dev` L1884; 1893 after Tasks 1-3), which drops `_concurrency_lua_sha` and falls through to the `if not used_concurrency_lua:` Python path, instead of denying (or skipping under `fail_open`).
- Test: `tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py`, following `test_redis_real_lua.py`:
  - `_Policy(fail_mode, **body)`;
  - `test_a_lost_lease_script_falls_back_to_counting_in_python` (3 cases);
  - `test_a_lost_rate_script_falls_back_once_then_reloads`.

**Interfaces:**
- Consumes: Task 1's tuple, which already catches `NoScriptError`.
- Produces: after any script failure, the next reserve reloads the script. A failed lease script never denies by itself.

- [ ] **Step 1: Write the failing tests**

```diff
diff --git a/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py b/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
index d79b776e7d..c212dfbb96 100644
--- a/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
+++ b/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
@@ -37,3 +37,56 @@ async def test_request_limit_holds_across_two_workers(real_redis, rg_unique_ns)
     await worker_a.release(h1)
     d3, h3 = await worker_b.reserve(_lease())
     assert d3.allowed and h3
+
+
+class _Policy:
+    """One policy for any id: ``body``, with the scope and fail mode the test names."""
+
+    def __init__(self, fail_mode: str, **body: dict) -> None:
+        """Remember the policy's fail mode and its category blocks."""
+        self._policy = {"scopes": ["user"], "fail_mode": fail_mode, **body}
+
+    def get_policy(self, _policy_id: str) -> dict:
+        """The same policy for any id."""
+        return dict(self._policy)
+
+
+# What a Redis restart or failover leaves in a worker: a cached SHA the server no longer knows (NOSCRIPT).
+# Never SCRIPT FLUSH a shared Redis to get there.
+_LOST_SHA = "f" * 40
+
+
+@pytest.mark.parametrize("fail_mode", ["fallback_memory", "fail_closed", "fail_open"])
+async def test_a_lost_lease_script_falls_back_to_counting_in_python(real_redis, rg_unique_ns, fail_mode) -> None:
+    """A lease reserve that cannot run its script still counts leases, then loads the script again (TASK-13859)."""
+    gov = RedisResourceGovernor(policy_loader=_Policy(fail_mode), ns=rg_unique_ns)
+    if not await gov._is_real_redis():
+        pytest.skip("Redis client is not real; using in-memory stub")
+    lease = RGRequest(
+        entity="user:42",
+        categories={"jobs": {"units": 1, "max_concurrent": 2, "ttl_sec": 60}},
+        tags={"policy_id": "limits.test_concurrent"},
+    )
+    first, _ = await gov.reserve(lease)  # loads the script
+    gov._concurrency_lua_sha = _LOST_SHA
+
+    second, _ = await gov.reserve(lease)
+    third, _ = await gov.reserve(lease)
+
+    assert [first.allowed, second.allowed, third.allowed] == [True, True, False]
+
+
+async def test_a_lost_rate_script_falls_back_once_then_reloads(real_redis, rg_unique_ns) -> None:
+    """A rate reserve that cannot run its script uses the Python windows once, then loads the script again (TASK-13859)."""
+    gov = RedisResourceGovernor(policy_loader=_Policy("fallback_memory", requests={"rpm": 5}), ns=rg_unique_ns)
+    if not await gov._is_real_redis():
+        pytest.skip("Redis client is not real; using in-memory stub")
+    req = RGRequest(entity="user:42", categories={"requests": {"units": 1}}, tags={"policy_id": "rate.test"})
+    first, _ = await gov.reserve(req)  # loads the script
+    gov._multi_lua_sha = _LOST_SHA
+
+    second, _ = await gov.reserve(req)
+    third, _ = await gov.reserve(req)
+
+    assert [first.allowed, second.allowed, third.allowed] == [True, True, True]
+    assert gov._multi_lua_sha not in (None, _LOST_SHA)
```

- [ ] **Step 2: Run them to see them fail**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
```

Expected (needs the local Redis):
- **The summary** is `4 failed, 1 passed`.
- **`test_a_lost_lease_script_falls_back_to_counting_in_python`:**
  - `[fallback_memory]` and `[fail_closed]` fail with `assert [True, False, False] == [True, True, False]`, because the lost script denies the second lease;
  - `[fail_open]` fails with `assert [True, True, True] == [True, True, False]`, because it skips the lease and drops the limit.
- **`test_a_lost_rate_script_falls_back_once_then_reloads`** fails with `assert 'ffff...' not in (None, 'ffff...')`. The lost SHA stays cached, so every later request pays a failing `evalsha`.
- **The 1 pass** is the existing two-worker test.

- [ ] **Step 3: Drop the lost SHA, and count leases in Python**

```diff
diff --git a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
index ffcb8761f4..cbedae0318 100644
--- a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
+++ b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
@@ -1658,9 +1658,10 @@ class RedisResourceGovernor(ResourceGovernor):
                                 pass
                             add_failed = True
         except _RG_NONCRITICAL_EXCEPTIONS:
-            # fall through to Python fallback
+            # fall through to Python fallback; a lost script (NOSCRIPT after a restart) loads again next time
             used_lua = False
             self._last_used_multi_lua = False
+            self._multi_lua_sha = None
         if not used_lua:
             if await self._is_stub_client():
                 # Pre-check across all scopes/categories
@@ -1891,12 +1892,9 @@ class RedisResourceGovernor(ResourceGovernor):
                                     denial_retry_after = max(denial_retry_after, int(ttl_sec or 1))
                                 break
                     except _RG_NONCRITICAL_EXCEPTIONS:
-                        if self._effective_fail_mode(pol, category) == "fail_open":
-                            continue
-                        concurrency_failed = True
-                        concurrency_failed_category = category
-                        denial_retry_after = max(denial_retry_after, int(ttl_sec or 1))
-                        break
+                        # Redis answered the probe but not the script (NOSCRIPT after a restart, OOM,
+                        # READONLY): load it again next time, and count leases in the Python path below.
+                        self._concurrency_lua_sha = None

                 if not used_concurrency_lua:
                     active_by_scope: dict[tuple[str, str], int] = {}
```

- [ ] **Step 4: Run the tests to see them pass**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_lua.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py tldw_Server_API/tests/Resource_Governance/test_governor_redis.py
```

Expected: `53 passed, 1 xfailed`. The concurrency file alone gives `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
git commit -m "fix(rg): a script Redis cannot run falls back to counting in Python and reloads (TASK-13859)

The worker caches its Lua SHAs. After a Redis restart or failover the
scripts are gone, so every evalsha raises NoScriptError, as it does at
OOM or READONLY while Redis still answers reads. The lease path then
denied every quota request until the worker restarted (fallback_memory,
fail_closed) or skipped the lease and dropped the limit (fail_open). It
now drops the cached SHA and counts leases in the Python path: against
Redis after a restart, or in the worker while Redis rejects writes (OOM,
READONLY); the rate path drops its SHA too, so the next reserve reloads.
On dev the same NoScriptError escaped reserve and ingress answered 500
after every Redis restart.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 5: Log when a worker enforces limits on its own (Decision 2)

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/policy_eval.py`: `import time`, `_REDIS_WARNING_INTERVAL_SEC`, `_last_redis_warning` and `log_redis_failure(ns, exc)`, after `log_lookup_failure`.
- Modify: `tldw_Server_API/app/core/Resource_Governance/governor_redis.py`: import `log_redis_failure`, and call it in four guards:
  - `_add_members`: a rate window add;
  - `_allow_requests_sliding_check_only`: a rate window check;
  - `check`'s lease count;
  - `reserve`'s lease add.
- Test: `tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py`, `test_a_redis_outage_is_logged_as_a_warning_once_a_minute`.
- Test: `tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py`:
  - `_rejecting(error)`, `_READ_ONLY` and `_OUT_OF_MEMORY`;
  - `test_redis_rejecting_writes_is_logged`, with 4 cases: `lease-readonly`, `lease-oom`, `rate-readonly` and `rate-oom`.
  - On the real local Redis, each patches the client's write commands to raise `ReadOnlyError`, or the memory-growing ones to raise `OutOfMemoryError`. Reads, including the probe, still answer.

**Interfaces:**
- Consumes: `_Policy` from Task 4.
- Produces: `policy_eval.log_redis_failure(ns: str, exc: BaseException) -> None`.
  - It logs at most one WARNING per 60 s per worker process and namespace. `_last_redis_warning` is a module dict.
  - It names the exception type only, never its message, which could carry a URL.
  - Each of the four cases fails if its own call is removed (a mutation check on the prototype).

- [ ] **Step 1: Write the failing tests**

```diff
diff --git a/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py b/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
index c212dfbb96..0ef9081d71 100644
--- a/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
+++ b/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
@@ -1,6 +1,7 @@
 """On real Redis a request's own max_concurrent holds across workers (TASK-13435)."""

 import pytest
+from redis.exceptions import OutOfMemoryError, ReadOnlyError

 from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest

@@ -90,3 +91,57 @@ async def test_a_lost_rate_script_falls_back_once_then_reloads(real_redis, rg_un

     assert [first.allowed, second.allowed, third.allowed] == [True, True, True]
     assert gov._multi_lua_sha not in (None, _LOST_SHA)
+
+
+def _rejecting(error: Exception):
+    """A Redis command that fails with ``error``, as a replica (READONLY) or a full server (OOM) answers a write."""
+
+    async def _command(*_args, **_kwargs):
+        raise error
+
+    return _command
+
+
+# Redis still answers the probe (ZCARD) but rejects writes. A replica (READONLY) rejects every write; a full
+# server (OOM) rejects the commands that grow memory, scripts included, but still trims windows and leases.
+_READ_ONLY = (ReadOnlyError("You can't write against a read only replica."), ("evalsha", "zadd", "zremrangebyscore", "set", "hset", "expire", "delete"))
+_OUT_OF_MEMORY = (OutOfMemoryError("command not allowed when used memory > 'maxmemory'."), ("evalsha", "zadd", "set", "hset"))
+
+
+@pytest.mark.parametrize(
+    ("fail_mode", "body", "categories", "fault"),
+    [
+        ("fallback_memory", {}, {"jobs": {"units": 1, "max_concurrent": 2, "ttl_sec": 60}}, _READ_ONLY),
+        ("fail_closed", {}, {"jobs": {"units": 1, "max_concurrent": 2, "ttl_sec": 60}}, _OUT_OF_MEMORY),
+        ("fail_closed", {"requests": {"rpm": 5}}, {"requests": {"units": 1}}, _READ_ONLY),
+        ("fail_open", {"requests": {"rpm": 5}}, {"requests": {"units": 1}}, _OUT_OF_MEMORY),
+    ],
+    ids=["lease-readonly", "lease-oom", "rate-readonly", "rate-oom"],
+)
+async def test_redis_rejecting_writes_is_logged(real_redis, rg_unique_ns, monkeypatch, fail_mode, body, categories, fault) -> None:
+    """A Redis that answers reads but rejects writes makes each worker count on its own; that is logged (TASK-13859)."""
+    from loguru import logger
+
+    from tldw_Server_API.app.core.Resource_Governance import policy_eval
+
+    gov = RedisResourceGovernor(policy_loader=_Policy(fail_mode, **body), ns=rg_unique_ns)
+    if not await gov._is_real_redis():
+        pytest.skip("Redis client is not real; using in-memory stub")
+    monkeypatch.setattr(policy_eval, "_last_redis_warning", {}, raising=False)  # no warning sent yet
+    error, commands = fault
+    client = await gov._client_get()
+    for command in commands:
+        monkeypatch.setattr(client, command, _rejecting(error))
+    req = RGRequest(entity="user:42", categories=categories, tags={"policy_id": "write.test"})
+    seen = []
+    sink = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
+    try:
+        await gov.reserve(req)
+        await gov.reserve(req)
+    finally:
+        logger.remove(sink)
+
+    assert [m for m in seen if "Redis backend" in m] == [
+        f"Resource Governor Redis backend {rg_unique_ns!r} is failing ({type(error).__name__}); "
+        "this worker enforces its limits on its own until it recovers"
+    ]
diff --git a/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py b/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
index bbc2cf8303..8ce31f0f58 100644
--- a/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
+++ b/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
@@ -184,3 +184,26 @@ async def test_renew_while_redis_is_down_keeps_the_lease(fail_mode, monkeypatch)
     second, _ = await rg.reserve(req)

     assert first.allowed and not second.allowed
+
+
+@pytest.mark.asyncio
+async def test_a_redis_outage_is_logged_as_a_warning_once_a_minute(monkeypatch):
+    from loguru import logger
+
+    from tldw_Server_API.app.core.Resource_Governance import policy_eval
+
+    monkeypatch.setattr(policy_eval, "_last_redis_warning", {}, raising=False)  # no warning sent yet
+    rg = _governor({"requests": {"rpm": 5}, "scopes": ["user"], "fail_mode": "fail_open"}, ns="rg_down_warn")
+    rg._client = _redis_down_client()
+    req = RGRequest(entity="user:x", categories={"requests": {"units": 1}}, tags={"policy_id": "p"})
+    seen = []
+    sink = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
+    try:
+        await rg.reserve(req)
+        await rg.reserve(req)
+    finally:
+        logger.remove(sink)
+
+    assert [m for m in seen if "Redis" in m] == [
+        "Resource Governor Redis backend 'rg_down_warn' is failing (ConnectionError); this worker enforces its limits on its own until it recovers"
+    ]
```

- [ ] **Step 2: Run them to see them fail**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
```

Expected (needs the local Redis):
- **The summary** is `5 failed, 27 passed`.
- **Each failure** is `assert [] == ['Resource Go... it recovers']`: nothing is logged. The failures are the outage test and all 4 cases of `test_redis_rejecting_writes_is_logged`.

- [ ] **Step 3: Warn, once a minute, where a worker falls back to enforcing a limit on its own**

```diff
diff --git a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
index cbedae0318..6b08b2a536 100644
--- a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
+++ b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
@@ -26,6 +26,7 @@ from .policy_eval import (
     effective_policy,
     lease_limits,
     log_lookup_failure,
+    log_redis_failure,
     requests_window,
     scope_pairs,
 )
@@ -577,8 +578,9 @@ class RedisResourceGovernor(ResourceGovernor):
         try:
             await client.zadd(key, dict.fromkeys(members, now))
             await client.expire(key, _window_ttl(window))
-        except _RG_NONCRITICAL_EXCEPTIONS:
+        except _RG_NONCRITICAL_EXCEPTIONS as exc:
             logger.debug("RG window add/expire failed for key={}", key, exc_info=True)
+            log_redis_failure(self._keys.ns, exc)  # the window lives only in this worker's acceptance tracker

     async def _zrem_members(self, *, key: str, members: list[str]) -> None:
         client = await self._client_get()
@@ -640,7 +642,8 @@ class RedisResourceGovernor(ResourceGovernor):
                 # Fallback to conservative window
                 ra = window
             return False, int(ra), count
-        except _RG_NONCRITICAL_EXCEPTIONS:
+        except _RG_NONCRITICAL_EXCEPTIONS as exc:
+            log_redis_failure(self._keys.ns, exc)
             if fail_mode == "fallback_memory":
                 raise _FallbackToMemory from None
             if fail_mode == "fail_open":
@@ -1132,6 +1135,7 @@ class RedisResourceGovernor(ResourceGovernor):
                                 await client.zremrangebyscore(key, float("-inf"), now)
                                 active_real = int(await client.zcard(key))
                             except _RG_NONCRITICAL_EXCEPTIONS as exc:
+                                log_redis_failure(self._keys.ns, exc)
                                 if cat_fail == "fallback_memory":
                                     raise _FallbackToMemory from exc
                                 active_real = 0
@@ -1937,8 +1941,9 @@ class RedisResourceGovernor(ResourceGovernor):
                             bucket[mem] = float(expires_at)
                         try:
                             await client.zadd(key, {mem: float(expires_at) for mem in members})
-                        except _RG_NONCRITICAL_EXCEPTIONS:
-                            pass
+                        except _RG_NONCRITICAL_EXCEPTIONS as exc:
+                            # Redis rejects the write (OOM, READONLY): only this worker counts the lease.
+                            log_redis_failure(self._keys.ns, exc)

                 expires_at = now + max(1, int(ttl_sec))
                 for (sc, ev), members in planned.items():
diff --git a/tldw_Server_API/app/core/Resource_Governance/policy_eval.py b/tldw_Server_API/app/core/Resource_Governance/policy_eval.py
index 4d380d26dd..c52fc64e69 100644
--- a/tldw_Server_API/app/core/Resource_Governance/policy_eval.py
+++ b/tldw_Server_API/app/core/Resource_Governance/policy_eval.py
@@ -12,6 +12,7 @@ safety-net rule is that no configuration can produce a permanent 429.
 from __future__ import annotations

 import math
+import time
 from collections.abc import Callable, Mapping
 from typing import Any

@@ -39,6 +40,27 @@ def log_lookup_failure(policy_id: str, exc: BaseException) -> None:
         logger.error("Resource Governor policy store lookup for {!r} failed: {!r}; treating it as undefined", policy_id, exc)


+_REDIS_WARNING_INTERVAL_SEC = 60.0
+_last_redis_warning: dict[str, float] = {}
+
+
+def log_redis_failure(ns: str, exc: BaseException) -> None:
+    """Warn that the Redis backend ``ns`` is failing, at most once a minute per worker process and namespace.
+
+    Called where a Redis error leaves this worker enforcing limits on its own: Redis unreachable, or
+    answering reads but rejecting writes (READONLY, OOM). The Redis governor's own logger is off unless
+    RG_DEBUG is set, so the warning is logged here.
+    """
+    now = time.monotonic()
+    if now - _last_redis_warning.get(ns, -math.inf) >= _REDIS_WARNING_INTERVAL_SEC:
+        _last_redis_warning[ns] = now
+        logger.warning(
+            "Resource Governor Redis backend {!r} is failing ({}); this worker enforces its limits on its own until it recovers",
+            ns,
+            type(exc).__name__,
+        )
+
+
 def _lookup(get_policy: Callable[[str], Mapping[str, Any] | None], policy_id: str) -> dict[str, Any]:
     try:
         pol = get_policy(policy_id)
```

- [ ] **Step 4: Run the tests to see them pass**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py tldw_Server_API/tests/Resource_Governance/test_policy_eval.py tldw_Server_API/tests/Resource_Governance/test_governor_safety_net.py
```

Expected: `116 passed`. The two test files of Step 2 alone give `32 passed`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/policy_eval.py tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
git commit -m "feat(rg): log when a Redis failure leaves a worker enforcing limits on its own (TASK-13859)

Once RG absorbs Redis errors, an outage left no trace: the governor's
logger is off unless RG_DEBUG is set, RG has no backend-error metric,
and concurrency_quota no longer sees the error it used to warn about.
A Redis that answers reads but rejects writes (READONLY, OOM) is
quieter still, since the probe passes. policy_eval.log_redis_failure
now logs one WARNING per worker process and namespace per minute,
naming the exception type, wherever a Redis error leaves the worker
counting a rate window or a lease on its own.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 6: Record the outage behaviour, verify, file the follow-up, and close the task

**Files:**
- Create:
  - `Docs/ADR/078-concurrency-leases-count-per-worker-while-redis-is-unreachable.md`, numbered in Step 1;
  - its mirror under `Docs/Published/ADR/`, generated by the refresh.
- Modify:
  - `Docs/ADR/README.md`: ADR-076's row, and a new ADR row;
  - its mirror `Docs/Published/ADR/README.md`, generated;
  - `Docs/Operations/Usage_Quotas.md`, the Redis line under the concurrency limits.
- Modify: `backlog/tasks/task-13859*.md`, and create one follow-up task (backlog-py only).

**Interfaces:**
- Consumes: Tasks 1-5.

- [ ] **Step 1: Number the ADR**

```bash
{
  git ls-tree -r --name-only origin/dev Docs/ADR/
  for n in $(gh pr list --state open --limit 500 --json number --jq '.[].number'); do
    gh api --paginate "repos/rmusser01/tldw_server/pulls/$n/files" --jq '.[].filename'
  done
} | sed -nE 's|^Docs/ADR/([0-9]{3})-.*|\1|p' | sort -n | tail -1
```

`gh pr list --json files` returns at most 100 files per PR (#3269 has 1168), so the loop pages every PR through `gh api --paginate`. Expected: `077` on 2026-10-20 (#3269), so the ADR is 078.

If it prints a higher number N, then N+1 replaces 078 everywhere in Task 6 and in the plan copy, `Docs/superpowers/plans/2026-10-20-rg-redis-error-noncritical.md`. That covers:
- the Files list;
- the ADR's file name and title, Steps 2 and 3, and the index rows;
- Step 3's `git add -N` and Step 6's `git add` paths;
- the commit message and the notes;
- every mention of `ADR-078` in the plan copy that Step 8 commits, which ADR-078's "Related spec/plan" points to.

**Run this plan and `plan_dupids_v2.md` one after the other.** That plan also numbers its ADR at execution, from dev and open PRs, so if both run at once they take the same number. Nothing catches a duplicate ADR number. Open this PR (Step 9) before that plan computes its number, or compute this one after its PR is open.

- [ ] **Step 2: Write the ADR**

Create `Docs/ADR/078-concurrency-leases-count-per-worker-while-redis-is-unreachable.md` with exactly this content, setting `**Date:**` to the day you write it:

```markdown
# ADR-078: Concurrency leases count per worker while Redis is unreachable

**Status:** Accepted
**Date:** 2026-10-20
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13859
**Related spec/plan:** `Docs/superpowers/plans/2026-10-20-rg-redis-error-noncritical.md`
**Amends:** ADR-076 (Consequences, "Lease-store failures": a reserve after Redis drops no longer admits without a lease)

## Decision

The Resource Governor's Redis backend treats `redis.exceptions.RedisError` as a Redis failure and answers it by its fail mode. So while Redis is unreachable, a per-user concurrency lease is counted in the worker that holds it, whether Redis was down when the worker first connected or dropped later. When Redis answers but cannot run the lease script, leases are counted in Python, and the script loads again on the next reserve:
- after a Redis restart (NOSCRIPT), against Redis, so the limit holds across workers;
- while Redis rejects writes (OOM, READONLY), in each worker on its own.

## Context

ADR-076 says that if Redis drops after a worker has connected, "a reserve that fails admits the request". It did, because RG's best-effort guards (`_RG_NONCRITICAL_EXCEPTIONS` in `governor_redis.py`) listed only builtin exceptions, and redis-py's `ConnectionError` and `TimeoutError` derive from `RedisError`, not from the builtins. The error escaped RG, and `concurrency_quota` admitted the request with no lease. The same escape made ingress (`RGSimpleMiddleware`) answer every governed request with HTTP 500 once Redis dropped, whatever the fail mode, and again after every Redis restart, when the cached script SHAs no longer exist (TASK-13859).

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep admitting quota requests with no lease while Redis is down | No RG fail mode admits a lease without counting it: `fallback_memory` counts it in the worker's memory governor, and `fail_open` and `fail_closed` count it in RG's in-process lease map. Keeping ADR-076's behaviour needs a lease path that bypasses RG for quotas alone, and lets a user start unlimited concurrent work for as long as Redis is down. |
| Refuse quota requests while Redis is down | Turns a Redis outage into an outage of media ingest, audio streaming and transcription. |
| Deny, or skip, a lease whose script fails while Redis still answers | After a Redis restart every reserve fails the script, so denying refuses every quota request until the worker restarts, and skipping drops the limit. |
| Catch `RedisError` only in RG's callers (ingress, `concurrency_quota`) | RG's fail modes would still never run for a Redis error, and every new caller would have to know to catch it. |

## Consequences

- **Per-worker counting.** While Redis is unreachable or rejects writes, each worker counts its own leases, so with N workers a user can run up to N times the limit.
  - That is how the memory backend works, and what ADR-076 already says for a worker that never connected.
  - Under the default `fallback_memory`, the worker's memory governor holds the lease. Under `fail_open` and `fail_closed`, RG's in-process lease map does. While Redis rejects writes, the lease is in that map but not in Redis.
  - `fail_closed` does not refuse a lease during an outage.
  - While Redis rejects writes, each reserve also loads the lease script and fails it again.
- **Leases on the other side.** A worker counting on its own does not see leases taken before Redis dropped. When Redis returns, Redis does not see the leases the memory governor took during the outage, until those requests end. Either way, a user can briefly exceed the limit by the leases the other side holds.
- **Release and renewal.** A lease released or renewed during the outage acts on the worker's own copy of the lease. Redis leases the worker could not release expire within 120 s.
- **Logging.** Every Redis error that leaves the worker enforcing a limit on its own is logged: unreachable, READONLY or OOM. Each worker process logs one WARNING per Redis namespace at most once a minute. A restart (NOSCRIPT) is not logged, because the limit still holds.
- **No more 500s.** A Redis error no longer answers a request with HTTP 500. Ingress answers by fail mode:
  - `fail_open` and `fallback_memory` admit, within the requests limit each worker keeps on its own;
  - `fail_closed` answers 429 (`Retry-After` up to 60 s) while Redis is unreachable, and for up to 60 s after, while each worker's deny floor runs out.
```

- [ ] **Step 3: Index it, and update the operator page**

```diff
diff --git a/Docs/ADR/README.md b/Docs/ADR/README.md
index 0478246fd5..84263c8b85 100644
--- a/Docs/ADR/README.md
+++ b/Docs/ADR/README.md
@@ -101,6 +101,7 @@ Small bug fixes, local implementation details, product copy, temporary experimen
 | [ADR-070](070-native-chat-retained-byte-quota-authority.md) | Proposed | Reserve native retained bytes in existing AuthNZ authority; keep legacy counters separate and report fresh aggregate usage once. |
 | [ADR-071](071-conversation-source-incarnation.md) | Proposed | Bind original conversation records to server-issued incarnation, independently of reused IDs, reset counters and Sync journals. |
 | [ADR-075](075-native-messages-and-mcp-tools-use-the-provider-recorder.md) | Accepted | A provider call no `ChatProvider` method makes (the Messages API native path) records through the provider recorder's public wrappers, and an MCP tool's LLM calls are recorded for its caller even under a caller that records itself. |
-| [ADR-076](076-per-user-concurrency-quotas.md) | Accepted | Per-user synchronous concurrency (media ingest, audio streams, direct transcription) is a `limits.*` usage quota held as one Resource Governor lease per request; the request carries its own limit, the lease store follows `RG_BACKEND`, and `RG_ENABLED` does not turn it off. |
+| [ADR-076](076-per-user-concurrency-quotas.md) | Accepted | Per-user synchronous concurrency (media ingest, audio streams, direct transcription) is a `limits.*` usage quota held as one Resource Governor lease per request; the request carries its own limit, the lease store follows `RG_BACKEND`, and `RG_ENABLED` does not turn it off. Amended in part by [ADR-078](078-concurrency-leases-count-per-worker-while-redis-is-unreachable.md): while Redis is unreachable, a lease is counted per worker instead of admitted without one. |
+| [ADR-078](078-concurrency-leases-count-per-worker-while-redis-is-unreachable.md) | Accepted | The Resource Governor's Redis backend answers redis-py's `RedisError` by its fail mode, so while Redis is unreachable a per-user concurrency lease is counted in the worker that holds it, and a lease script Redis cannot run falls back to counting in Python. |

 The original backfill inventory is [the 2026-06-03 decision inventory](inventory/2026-06-03-decision-inventory.md). See the [2026-09-25 reconciliation](inventory/2026-09-25-decision-inventory-reconciliation.md) for current dispositions of its unresolved rows.
diff --git a/Docs/Operations/Usage_Quotas.md b/Docs/Operations/Usage_Quotas.md
index 8188038845..42815facc0 100644
--- a/Docs/Operations/Usage_Quotas.md
+++ b/Docs/Operations/Usage_Quotas.md
@@ -121,7 +121,7 @@ What `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd` and `limits.ll
 - A lease nobody releases, for example when a worker is killed mid-request, expires within 2 minutes. A crash can hold one of the user's slots for up to that long.
 - `Retry-After` on a refusal is when the oldest held lease would expire if it stopped renewing, up to 2 minutes. A slot usually frees sooner, when one of the user's requests finishes, so a client may retry earlier.
 - Lowering a limit does not stop running requests. New ones are refused until the user is below the new limit.
-- If Redis is unreachable when a worker's lease store first connects, that worker counts on its own for the life of the process, as on `memory`. If Redis drops later, requests are admitted, a warning is logged, and leases the worker could not release expire within 2 minutes.
+- If Redis is unreachable when a worker's lease store first connects, that worker counts on its own for the life of the process, as on `memory`. If Redis drops later, or rejects writes (a read-only replica, or out of memory), the worker counts on its own until Redis recovers, and a warning is logged (ADR-078). It does not see leases taken before the drop, a lease released during the outage frees its slot at once, and leases the worker could not release in Redis expire within 2 minutes. After a Redis restart, leases are counted as before.

 Two other `limits.*` entries in the catalog are not part of this scheme:

```

If #3269 has merged, its ADR-077 row also follows ADR-076. Keep both rows, in number order.

Then refresh the mirror:

```bash
bash Helper_Scripts/refresh_docs_published.sh
git add -N Docs/ADR/078-concurrency-leases-count-per-worker-while-redis-is-unreachable.md Docs/Published/ADR/078-concurrency-leases-count-per-worker-while-redis-is-unreachable.md
git status --short Docs
```

Expected:
- **The refresh** prints `Refreshed curated docs in: .../Docs/Published`.
- **`git status --short Docs`** lists exactly these:
  - `M Docs/ADR/README.md`;
  - `M Docs/Operations/Usage_Quotas.md`;
  - `M Docs/Published/ADR/README.md`;
  - the two `A` ADR files;
  - the untracked plan copy, `?? Docs/superpowers/plans/2026-10-20-rg-redis-error-noncritical.md`, which Step 8 commits.

- [ ] **Step 4: Run the sweep, the Docs tests and the ratchet**

```bash
FILES=$(git grep -l -E "governor_redis|RedisResourceGovernor|middleware_simple|RGSimpleMiddleware|concurrency_quota|peek_with_policy|character_rate_limiter|Resource_Governance(\.| import )policy_eval" -- 'tldw_Server_API/tests/*.py' \
  | grep -v -E "^tldw_Server_API/tests/(Resource_Governance|Usage)/|/conftest\.py$")
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance tldw_Server_API/tests/Usage $FILES tldw_Server_API/tests/lint/test_rg_route_map_lint.py
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs
$PY Helper_Scripts/ci/test_quality_triage.py --enforce 2>&1 | grep "test-triage\] \(OK\|FAIL\)"
```

The first run covers:
- the RG suite, its real-Redis integration tests included;
- the Usage suite;
- the RG route-map lint;
- every other test file that references the changed modules, the `peek_with_policy` callers or `policy_eval` (29 files on `origin/dev`).

Expected, measured on the prototype:
- **The first run:** `1 failed, 1910 passed, 4 skipped, 2 xfailed`, in about 23-31 minutes. The Postgres-backed RG tests skip without a reachable Postgres.
  - **The one failure** is `tests/CI/test_required_workflow_contracts.py::test_critical_e2e_budget_preserves_other_policies_and_still_enforces_limits`, with `CalledProcessError ... returned non-zero exit status 127`.
  - The workflow snippet it runs calls `python`, which is not on this machine's PATH. It fails the same way on unchanged `origin/dev`, and it passes where `python` exists, as in CI.
- **Docs:** `212 passed`.
- **The ratchet:** `[test-triage] OK — no offenses beyond the baseline.`

`integration/test_policy_loader_route_map_postgres.py` has flaked once with `asyncpg ... ConnectionDoesNotExistError` against a local Postgres, and it passes on rerun on both trees. If anything else fails, run the same file on `origin/dev` before going on.

- [ ] **Step 5: Lint and security**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/app/core/Resource_Governance/middleware_simple.py tldw_Server_API/app/core/Resource_Governance/policy_eval.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py tldw_Server_API/tests/Resource_Governance/test_middleware_simple.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
uvx bandit -q -ll tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/app/core/Resource_Governance/middleware_simple.py tldw_Server_API/app/core/Resource_Governance/policy_eval.py
```

Expected:
- **Ruff** prints `All checks passed!`.
- **Bandit** reports exactly one finding and exits 1: `B113:request_without_timeout`, Medium severity and Low confidence, at `policy_eval.py:99`.
  - It is a false positive on `requests.get("burst")`, a dict read in `requests_window`.
  - `origin/dev` reports the same at line 77; Task 5's 22 lines move it.
  - On its first run, `uvx` may print package download lines first.

- [ ] **Step 6: Commit the docs**

```bash
git add Docs/ADR/README.md Docs/ADR/078-concurrency-leases-count-per-worker-while-redis-is-unreachable.md Docs/Operations/Usage_Quotas.md Docs/Published/ADR/README.md Docs/Published/ADR/078-concurrency-leases-count-per-worker-while-redis-is-unreachable.md
git commit -m "docs(adr): ADR-078, concurrency leases count per worker while Redis is unreachable (TASK-13859)

ADR-076 said a quota reserve that fails after Redis drops admits the
request. It did, because RG let redis-py's errors escape. Now that RG
applies its fail mode, each worker counts its own leases during an
outage, as it already did when Redis was down at first connect, and
while Redis rejects writes. A lease script lost to a restart falls back
to counting in Python against Redis, and both cases of counting alone
are logged. ADR-078 records that and amends ADR-076 in part;
Usage_Quotas.md says the same.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: File the latency follow-up**

Set `M` to the highest task id across `origin/dev` and open PRs, with the command from AGENTS.md's Backlog section. Keep this shell open for Step 8, which reuses `M`.

```bash
git fetch origin dev --quiet
M=$({ git ls-tree -r --name-only origin/dev backlog/; gh pr list --state open --limit 500 --json files --jq '.[].files[].path'; } \
  | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1)
echo "$M"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "RG Redis client: a connect timeout or circuit breaker for a black-holed Redis" --id TASK-$((M+1)) \
  -d "After TASK-13859, RG applies its fail mode on Redis errors instead of raising, so a request makes several Redis calls during an outage, where it used to fail at the first. The reviewer counted, with Redis down: ingress 7 (fallback_memory), 14-15 (fail_open) or 3-8 (fail_closed) calls, against 1 on dev; a quota acquire 2 or 8; a shielded release 3. A refused connection fails in about 10 ms, but redis_factory sets no socket_connect_timeout, so with a black-holed Redis each call waits for the OS connect timeout." \
  --ac "The RG Redis client has a short connect timeout, or RG stops calling Redis for a while after a failure, with a test on an unroutable address" \
  -l resource-governance,redis
```

Expected: `echo` prints a number of at least `13870`, and the create prints `TASK-<id> - RG Redis client: ... (To Do)`.

- [ ] **Step 8: Close the task and commit the plan**

The notes quote the measured results. If any RED, GREEN or verification step printed different lines, put the actual ones in. Run this in the shell from Step 7.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13859 --append-notes "Fixed:
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
- Sweep: 1 failed, 1910 passed, 4 skipped, 2 xfailed; the one failure (test_required_workflow_contracts, python not on PATH, exit 127) fails on origin/dev too. Docs 212 passed. Test-quality ratchet OK.
- Ruff clean. Bandit: only the B113 false positive dev already has in policy_eval.py.
Follow-up: TASK-$((M+1)), a connect timeout or circuit breaker for the RG Redis client." --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "The Redis Resource Governor and the ingress middleware now catch redis.exceptions.RedisError, so once Redis drops, reserve, renew, commit, peek_with_policy and ingress apply RG's fail mode instead of raising or answering 500. commit and renew act on the worker's copy of a lease during an outage, a script Redis cannot run falls back to counting in Python and reloads, and a worker that enforces limits on its own logs it once a minute. Concurrency leases count per worker while Redis is down (ADR-078, amending ADR-076)." --status Done
git add -A backlog/tasks
git add Docs/superpowers/plans/2026-10-20-rg-redis-error-noncritical.md
git commit -m "docs(rg): plan for catching redis-py errors in RG; close TASK-13859, file its follow-up

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: `git show --stat HEAD` lists:
- the TASK-13859 file;
- the follow-up task file;
- the plan.

- [ ] **Step 9: Open the PR**

```bash
git push -u origin fix/rg-redis-error-noncritical
gh pr create --base dev --title "fix(rg): the Redis governor and ingress catch redis-py errors (TASK-13859)" --body-file $SCRATCH/pr_body_13859.md
```

Write `$SCRATCH/pr_body_13859.md` from the commit messages above. Include:
- the outage list from "By fail mode";
- the four owner decisions;
- the dev bug Task 4 also fixes: ingress answered 500 after every Redis restart;
- a `## Change summary` section that the owner writes (ADR-004);
- a last line of `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

Then run `/superpowers:requesting-code-review` in a clean subagent. Follow `gh variable get MERGE_QUEUE` for the merge.
