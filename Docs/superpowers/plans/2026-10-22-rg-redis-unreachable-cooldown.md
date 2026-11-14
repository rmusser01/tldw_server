# RG Redis: A Cooldown for an Unreachable Redis, and Lost Script Replies Implementation Plan

> **Re-review folded in (r3, 2026-10-22).** The scoped re-review of r2 (`<scratchpad>/rereview_plan_fu2_rg_redis.md`) found it ready with fixes: 0 Blocker, 0 Important, 5 Minor. It reproduced every r2 count, and confirmed B1 fixed and the two-timeout stall boundary. r2 is kept as `<scratchpad>/plan_fu2_rg_redis.r2.md`. All five minors are folded in:
> - **M1:** option (a)'s test edit now also accepts a network that answers `10.255.255.1` "unreachable" at once (Decision 1). Review Focus 6 says which claim holds under which option. With the edit, (a) gives `47 passed` here.
> - **M2:** the docstring and the commit text now say what is true: the PING normally takes the connection just dropped (redis-py's pool is LIFO), so it reconnects. Review Focus 2 records the deterministic alternative, and why it is not taken.
> - **M3:** Decision 1's table and the cost of (b) now list the PINGs at the onset of an outage, one per failing request: the re-review measured 50 for 50 concurrent requests. Review Focus 2 names coalescing as the upgrade.
> - **M4:** `socket_timeout` and `socket_connect_timeout` options in `REDIS_URL` override the pinned 5 s. This was measured (`?socket_timeout=30` gives 30.0), and it is now said in the code comment, the `Env_Vars.md` entry, the RG README and ADR-081.
> - **M5:** ADR-081 gains a "Release and renewal" consequence: "renewed after Redis returns" now means after the cooldown ends, and a release in the lag window leaves a Redis lease to lapse within 120 s. "Read when the server starts" becomes "read when each governor is built (at startup, or on a limiter's first use)", in `Env_Vars.md` and the ADR.
> - **Re-run on r3,** against a private `redis-server` on port 6433, never the shared Redis:
>   - Task 1's module: `47 passed`; with option (a)'s edits, `47 passed`.
>   - Task 2's module: `31 passed`; the two together: `78 passed`.
>   - The 11 mutants are all caught.
>   - Docs and the ratchet: `213 passed`. Triage OK. ruff, bandit and `diff --check` clean.
>   - The Docs step from the plan's own numbered diff, refreshed and staged: `212 passed`.
>   - r3 changes no executable code, only a comment, a docstring and docs. So Task 1's RED, Task 2's RED, the sweep (`785 passed, 1 skipped, 2 xfailed`) and the latency lines stand as measured on r2.

> **Review folded in (r2, 2026-10-22).** The independent review and dry run (`<scratchpad>/review_plan_fu2_rg_redis.md`, probes in `<scratchpad>/rv_fu2rg/`) found 1 Blocker, 3 Important and 4 Minor findings. r1 is kept as `<scratchpad>/plan_fu2_rg_redis.r1.md`. All findings are folded in, following the coordinator's rulings:
> - **B1, fixed.** In r1 a connection the server closed while idle started the cooldown: after an idle spell, `fail_closed` refused every new user with `deny(60)`, where dev admits them. A redis-py `ConnectionError` now starts the cooldown only if a PING right after it also fails; the failed connection was dropped, so the PING reconnects. "What was verified" and Review Focus 1 and 2 are corrected. Task 1 gains a regression test (a Redis stand-in that closes its first connection), and the other tests count the PING.
> - **`TimeoutError`, owner Decision 1.** Both options are measured on a black-holed address, on one 6 s and one 11 s stall, on recovery, and on a real lost reply. The plan implements the recommendation (b): a PING confirms a `TimeoutError` too. Decision 1 gives the exact edits for (a). The socket timeout is part of the decision: 5 s, now set explicitly on RG's client.
> - **I1, fixed.** The real cycle is stated everywhere, in the docs, the task notes, the commit text and the PR body. RG retries 10 s after the last failed probe, so the fail mode lasts up to about 15 s after Redis returns (measured 14.5 s and 14.8 s). Under `fail_closed`, refusals can last about 75 s after recovery.
> - **I2, settled.**
>   - A knob, `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` (default 10, `0` turns the cooldown off), documented in `Env_Vars.md` with its mirror. The `0` path is tested.
>   - A new ADR-081, numbered at execution, amends ADR-078's Consequences (Task 3).
> - **I3, fixed.** The `RG_REDIS_FAIL_MODE` entry gains the cooldown and the recovery lag, with its mirror. `Rate_Limits_Troubleshooting.md` gains a `fail_closed` row.
> - **Minor findings:**
>   - **M1:** TASK-13892's notes now say that with option (a) a real lost reply starts the cooldown, and how `fail_closed` then answers. Under (b), a reply lost to a stall shorter than two timeouts is cleaned up and counted correctly (measured).
>   - **M2:** the recovery test uses a 0.5 s cooldown and a 0.6 s sleep.
>   - **M3:** ids are taken under the round's id lock, with the paging pipeline, right before use, after a rebase.
>   - **M4:** 150 guard sites, and "embeddings" is two of the nine governors.
> - **Follow-ups:** the redis-py follow-up from r1 is dropped, because the timeouts are now pinned. Only the `rg_unique_ns` follow-up remains.
> - **Re-measured:** everything was re-measured on `origin/dev` `e06eeec68c` (unchanged since r1), against a private `redis-server` on port 6421 and never the shared Redis.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two follow-ups that PR #3288 (TASK-13859, ADR-078) filed against the Redis Resource Governor (RG).
- **TASK-13882:** once Redis cannot be reached, RG stops calling it for a cooldown and answers by its fail mode, so a black-holed Redis no longer costs 30-70 s per governed request.
- **TASK-13892:** when the lease script or the multi-key rate script fails after it may have run (its reply lost), `reserve` removes the members it planned for that script before the Python fallback counts, so a request no longer counts or charges itself twice.

**Architecture:**
- **`governor_redis.py`, TASK-13882:**
  - New `_stop_calling_redis_while_unreachable(client, cooldown_s)`. It wraps the real client's `execute_command`, through which every redis-py command RG sends runs.
    - **Starting the cooldown.** A command that fails with redis-py's `ConnectionError` or `TimeoutError` is followed by one PING on a fresh connection. Only if that PING fails too does the cooldown start, for `cooldown_s`.
    - **During the cooldown,** each command raises redis-py `ConnectionError` at once. RG's 150 guard sites already answer that by the fail mode (TASK-13859).
    - **After the cooldown,** one command probes Redis while the rest keep failing fast. Any answer, an error reply included, ends the cooldown.
  - `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` (default 10, `0` = off) is read when each governor is built (`_redis_unreachable_cooldown_s`).
  - `_client_get` passes `socket_connect_timeout` and `socket_timeout` of 5 s (`_REDIS_SOCKET_TIMEOUT_S`) to `create_async_redis_client`, and wraps the real client unless the cooldown is `0`. The in-memory stub is never wrapped.
- **`governor_redis.py`, TASK-13892:** two `except` branches in `reserve` (the multi-key rate script, and the lease script) `ZREM` the members planned for the script, one call per key and best effort, before the Python path counts.
- **Docs:**
  - ADR-081 amends ADR-078.
  - The `RG_REDIS_FAIL_MODE` entry and the new `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` entry in `Env_Vars.md`, with its mirror.
  - A `fail_closed` row in `Rate_Limits_Troubleshooting.md`.
  - A bullet in `Usage_Quotas.md`, and the RG `README.md`.

**Tech Stack:** Python, redis-py (`redis.asyncio`; 8.1.0 in the venv, `redis>=4.5.0` in `pyproject.toml`), pytest, a private `redis-server` (8.6.2 here) for the TASK-13892 tests and the sweep.

**Spec:**
- `backlog/tasks/task-13882 - RG-Redis-client-a-connect-timeout-or-circuit-breaker-for-a-black-holed-Redis.md`
  - AC #1: "The RG Redis client has a short connect timeout, or RG stops calling Redis for a while after a failure, with a test on an unroutable address".
  - The plan takes the second option. Its notes correct the latency premise (M4 of the #3288 clean review): redis-py 8.1 waits 5 s per command.
- `backlog/tasks/task-13892 - RG-reserve-a-lost-script-reply-does-not-count-the-script-s-own-lease.md`
  - AC #1: "After a lease or multi-key rate script whose reply is lost, reserve removes that reserve's planned members before the Python fallback counts, with a real-Redis test where the script runs and the reply raises redis.exceptions.TimeoutError".
- Sources:
  - PR #3288 (merged), and ADR-078, whose Follow-up section names both tasks;
  - the #3288 clean review `<scratchpad>/clean3288_review.md` (M3, M4) and its re-review `<scratchpad>/rereview3288.md`;
  - this plan's review, `<scratchpad>/review_plan_fu2_rg_redis.md`.

## What was verified

**Where:** `origin/dev` `e06eeec68c`, in this planner's own worktree, reset afterwards.
- **Redis:** a private `redis-server` on 127.0.0.1:6421 (`--save "" --appendonly no --enable-debug-command local`), and `10.255.255.1` for a black hole. The shared Redis was never touched.
- **Probes,** none committed:
  - `<scratchpad>/fu2rg/` (r1);
  - `<scratchpad>/fu2rg/r2/` (r2), each run through `variants.py`, which swaps in `origin/dev`'s, r1's, option (a)'s or option (b)'s `governor_redis.py` and restores the tree.

- **redis-py 8.1's own timeouts** (`fu2rg/probe_timeouts.py`, the `from_url` client `redis_factory` builds):
  - `socket_connect_timeout` and `socket_timeout` default to 5 s, and a pool connection has no retries.
  - On the black-holed `10.255.255.1`, each command waits 5.00 s and raises `redis.exceptions.TimeoutError`.
  - On a closed port it fails in 0.00 s with `redis.exceptions.ConnectionError`.
- **dev's cost per governed request with a black-holed Redis** (`fu2rg/probe_blackhole_latency.py`, default 5 s timeouts):

  | Request | dev, per request |
  | --- | --- |
  | Ingress (reserve, peek, commit), `fallback_memory` | 30.01 s, every request |
  | Ingress, `fail_open` | 70.01 s, every request |
  | Quota (reserve, renew, release), `fallback_memory` | 10.00 s, every request |
  | Quota, `fail_open` | 65.01 s, every request |

- **Commands sent with Redis unreachable on dev** (Task 1's RED run, 3 reserves): 28 under `fail_open`, 15 under `fallback_memory` and 11 under `fail_closed`. The counts are the same for an unroutable, a stopped and a silent Redis.
- **B1, reproduced and fixed** (`r2/p4_idle.py`; `CONFIG SET timeout 1` on 6421, a 2.5 s idle spell, then 3 new users, `fail_closed`):

  | Variant | After the idle spell |
  | --- | --- |
  | dev | `ok(0)` ×3 |
  | r1 | `deny(60)` ×3 |
  | r2 | `ok(0)` ×3 |

  - **What happens:** `ConnectionPool.ensure_connection` does not see the server's close. The first command on the stale connection raises `ConnectionError: Connection closed by server.`
  - **In r2:** the PING after it normally takes the connection just dropped, reconnects, and gets an answer, so no cooldown starts (Review Focus 2).
  - **Correction to r1:** r1's claim that "an idle reset is not a failed command" was wrong.
- **Both `TimeoutError` options, measured.** These are the numbers behind Decision 1; the table is there.
  - **(a):** one `TimeoutError` starts the cooldown, and a PING confirms only a `ConnectionError`.
  - **(b):** a PING confirms both.
- **One stall shorter than two socket timeouts does not start (b)'s cooldown.**
  - The timed-out command's PING waits for the stall to end, and gets an answer.
  - So under (b), TASK-13892's `ZREM` reaches Redis for a reply lost to such a stall (`r2/p5_lost_reply.py`).
- **Why a cooldown, not a shorter timeout.** The AC allows either.
  - At 0.5 s, a `fail_open` ingress request would still wait 14 × 0.5 s on every request.
  - A shorter reply timeout turns more slow replies into lost ones (TASK-13892).
- **Why not the shared `Infrastructure/circuit_breaker.py`:**
  - `_transition` logs a WARNING on every trip, with no switch. That would be every governor (9 construction sites: ingress via `governor_factory`, the quota lease store, and the chat, character, two embeddings, evaluations, MCP and web-scraping limiters) every cooldown, against TASK-13859's owner Decision 2: one WARNING per worker a minute (`log_redis_failure`).
  - A registry breaker reads a database on every call when persistence is on, the default outside pytest.
  - Every success increments a metric.
- **TASK-13892 on dev:**
  - **Simulated reply loss** (Task 2's RED, `evalsha` runs the script and then raises): a lease at a limit of 1 is refused, and its lease stays after `release`, `(False, 1, 1)`. A `requests` reserve leaves 2 window members.
  - **Real reply loss** (`r2/p5_lost_reply.py`: `DEBUG SLEEP 6` lands just before the `EVALSHA`, and the 5 s timeout fires):
    - a lease is refused (`retry_after=60`), with 1 lease in Redis;
    - a rate reserve is admitted, with 2 members in Redis;
    - the same under `fallback_memory` and `fail_closed`.
- **The task's "limit of 2" claim is inaccurate** (`fu2rg/probe_lost_reply_limit2.py`). The Python path re-adds the same member ids, so the request never holds two slots.
  - With no other lease held, it is admitted holding one slot.
  - With one lease held, it is refused (`retry_after=119`).
- **Mutation check:** each of these 11 mutants fails at least one new test (`r2/mutate_r2.py`):
  - no wrap in `_client_get`;
  - no PING (r1);
  - a PING confirms `ConnectionError` only (option (a) against (b)'s tests);
  - no reset after a reply, or after an error reply;
  - no probe push;
  - a `0` cooldown still wraps;
  - no pinned timeouts;
  - no finiteness check on the env var;
  - no lease `ZREM`, or no rate `ZREM`.
- **Nothing else is in the way:**
  - None of the 44 open PRs touches the code or test files.
  - `scope_predicate_baseline.txt` lists none of them.
  - `Usage_Quotas.md`, `Rate_Limits_Troubleshooting.md` and the RG README are not mirrored. `Env_Vars.md` and the ADRs are.
- **Found while planning:** the `rg_unique_ns` fixture's cleanup reads one `SCAN` page (`count=1000`). The shared Redis holds about 1150 keys, 520 of them leaked `rg_it_*` keys. Filed as a follow-up (Task 4, Step 4). Never delete other runs' keys by hand.

## Global Constraints

- **Branch and PR:** `fix/rg-redis-unreachable-cooldown` from `origin/dev`, one PR to `dev`. Both tasks change `governor_redis.py`, and the ADR covers how they interact.
- **Redis:** run the Redis-backed tests against a private `redis-server` on a port of your own, never the shared Redis on 6379:
  ```bash
  RPORT=6431  # any free port
  redis-server --port $RPORT --bind 127.0.0.1 --save "" --appendonly no --daemonize yes --dir $SCRATCH --pidfile $SCRATCH/redis$RPORT.pid --logfile $SCRATCH/redis$RPORT.log
  export REDIS_URL=redis://127.0.0.1:$RPORT
  ```
  - Shut it down when done: `redis-cli -p $RPORT shutdown nosave`.
  - Never `FLUSHALL`, `FLUSHDB` or `SCRIPT FLUSH`, and never `CONFIG SET` any Redis but your own.
  - CI provides its own Redis through `REDIS_URL`.
  - Task 1's tests need no Redis. They use `10.255.255.1:6379` (unroutable), a free local port (stopped), a socket that listens and never answers (silent), and a local Redis stand-in that closes its first connection.
- **Ids under the round's id lock, taken right before use, after `git fetch origin dev && git rebase origin/dev`:**
  - the ADR number in Task 3, Step 1;
  - the follow-up task id in Task 4, Step 4.
  - Each is computed with the paging pipeline, which also reads the file lists of PRs with more than 100 files, and checked free before it is used.
- **`NNN`:** the docs diff in Task 3 writes the ADR number as `NNN`. Task 3, Step 1 replaces it before applying. `NNN` appears nowhere else in that diff.
- **Test first:** each task's RED run happens before its source change, and fails for the reason given.
- **`RG_TEST_PURGE_LEASES_BEFORE_RESERVE`:** the RG `conftest.py` sets it for every RG test. The lease test deletes it, because production never sets it.
- **Embedded diffs:** save each one to `$SCRATCH` exactly as shown and apply it with `git apply`. Blank context lines are empty, which `git apply` accepts; the dry run applied these exact diffs.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit ids, `git add -A backlog/tasks`.
- **Mirrored docs:** run `bash Helper_Scripts/refresh_docs_published.sh` and commit the `Docs/Published` changes in the same commit. `git add` new files before running `tests/Docs`, because it compares the mirror against tracked files.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`, never force-push.
- **Before merge:**
  - The PR needs the owner's `## Change summary` (ADR-004). No recorded waiver covers TASK-13882 or TASK-13892, so ask.
  - A clean-subagent `/superpowers:requesting-code-review` must run.
  - Check `gh variable get MERGE_QUEUE` for the merge procedure.

## Decisions for the owner

### Decision 1: does one `TimeoutError` start the cooldown, or must a PING confirm it?

**Recommended: (b), a PING confirms a `TimeoutError` as it confirms a `ConnectionError`.** The plan's diffs implement (b).

**The socket timeout is 5 s for both options.**
- The plan sets it explicitly on RG's client (`_REDIS_SOCKET_TIMEOUT_S`). It is redis-py 8's default, so a healthy but slow Redis behaves as it does on dev.
- RG's commands take well under a millisecond, so 5 s fires only on a stall or an outage.
- A shorter timeout turns more slow replies into lost replies (TASK-13892) and more single failures into fail-mode answers. Under (b) it would also make the PING's answer less likely to cover a stall.
- A longer timeout lengthens every wait during an outage.
- Setting it explicitly makes these numbers hold on older redis-py releases the `redis>=4.5.0` pin allows, which TASK-13882's notes say may wait for the OS connect timeout.

Measured on `e06eeec68c` with 5 s timeouts (dev shown for reference):

| | (a) one timeout starts it | (b) a PING confirms it | dev |
| --- | --- | --- | --- |
| First request on a black-holed Redis (`r2/p1_blackhole.py`) | 5.00 s | 10.00 s | 30-70 s, every request |
| Requests arriving at the start of an outage, one a second (`r2/p6_onset.py`) | the first 6 wait 5.0 s each, then 0.0 s | the first 11 wait 10.0 s each, then 0.0 s | every one waits |
| While Redis stays down | one request per governor waits 5 s every 15 s | one request per governor waits 10 s every 20 s | every request waits |
| PINGs at the onset of an outage | one per refused or reset command; a timeout sends none | one per request whose command fails before the first PING fails: 50 concurrent requests sent 50 commands and 50 PINGs, then the next 50 sent nothing (re-review, `probe_herd.py`, 0.2 s timeouts) | none |
| Lag after Redis returns, worst case (`r2/p3_recovery.py`) | 14.51 s | 14.8 s (4.46 s when Redis returned mid-cooldown) | none |
| `fail_closed`, one 6 s stall (`r2/p2_stall.py`) | the user hit and 3 new users `deny(60)`; new users 11 s later `ok` | all `ok`, as dev | all `ok` |
| `fail_closed`, one 11 s stall | the same as for 6 s (not run: it starts on the first timeout) | the user hit and 3 new users `deny(60)`; new users 11 s later `ok` | all `ok` |
| `fallback_memory`, one 6 s stall | admitted, counted per worker for about 10 s | admitted, counted in Redis | admitted |
| A real lost reply (`DEBUG SLEEP 6` before the `EVALSHA`, `r2/p5_lost_reply.py`) | the cooldown starts. Lease admitted per worker, its script lease left in Redis. Rate: `fallback_memory` admitted, charged once; `fail_closed` refused (`retry_after=60`), its one charge left in Redis | no cooldown: Task 2's `ZREM` runs. Lease and rate admitted and counted once, in both modes | lease refused (`retry_after=60`); rate charged twice |

**Why (b):**
- The cooldown is there to stop paying timeouts during an outage, and one stall is not an outage.
- Under (a), one stall between 5 and 10 s turns into about 10 s of `deny(60)` for every new user of that governor under `fail_closed`, each of them keeping a 60 s `Retry-After`. That is the same harm as B1, from a timeout instead of a closed connection.
- (b) applies one rule to both errors, and lets TASK-13892's cleanup work for a lost reply.

**Cost of (b):**
- 5 s more on the first request of an outage, and on each probe request: 10 s instead of 5 s.
- Requests arriving in the first 10 s of an outage wait 10 s each, not 5 s.
- Each of them also sends its own PING, so during that first window a Redis that does not answer gets twice the connection attempts. Coalescing the PINGs would remove that (Review Focus 2).
- Both are still a fraction of dev's 30-70 s on every request.

**If the owner picks (a), make these changes and nothing else.** The dry run checked the code and test edits: with them, Task 1's module gives `47 passed` (r3, with M1's test line).
- **Task 1, Step 4's diff.** Replace
  ```python
          except (RedisConnectionError, RedisTimeoutError):
              blocked_until = time.monotonic() + cooldown_s if await _unreachable() else 0.0
              raise
  ```
  with
  ```python
          except RedisTimeoutError:
              blocked_until = time.monotonic() + cooldown_s  # a timeout already cost one full wait: do not risk a second
              raise
          except RedisConnectionError:
              blocked_until = time.monotonic() + cooldown_s if await _unreachable() else 0.0
              raise
  ```
  In the docstring, change "A command that fails to reach Redis (refused, reset or timed out) starts the cooldown only if a PING right after it fails too" to "A command that times out starts the cooldown; a refused or reset one starts it only if a PING right after it fails too", and "a connection Redis closed while idle, or one stall, does not stop RG" to "a connection Redis closed while idle does not stop RG". The comment on `_REDIS_UNREACHABLE_COOLDOWN_S` then reads "Longer than the socket timeout, so one probe at a time waits".
- **Task 1, Step 2's diff.** In `test_rg_stops_calling_an_unreachable_redis_until_the_cooldown_ends`, replace the last line with
  ```python
      # A refused command is confirmed with a PING; a timeout starts the cooldown at once. A network that answers
      # 10.255.255.1 "unreachable" at once refuses it too.
      pinged = kind == "stopped" or (kind == "unroutable" and "PING" in sent)
      assert ([d.allowed for d in decisions], len(sent), sent[1:2]) == ([admitted] * 3, 2 if pinged else 1, ["PING"] if pinged else [])
  ```
- **Task 3's diff.**
  - Wherever it says "Once a (Redis) command and a PING right after it both fail", say instead "Once a command times out, or a refused or reset command and a PING after it both fail".
  - Replace "requests in the first 10 seconds wait up to 10 seconds each (two 5-second timeouts), then about one request every 20 seconds does" with "requests in the first 5 seconds wait up to 5 seconds each, then about one request every 15 seconds does".
  - In the ADR's Latency bullet, say "up to 5 s each" and "every 15 s or so".
  - In the ADR's Alternatives table, put "(b) confirm a timeout with a PING too: 5 s more on the first request of an outage and on each probe" in place of the third row.
  - In its Follow-up section, say that a lost reply starts the cooldown, so the cleanup does not reach Redis.
- **Task 4's notes.** In the TASK-13882 notes and the final summary, change "a PING right after it" accordingly. Change "10.00 s" to "5.00 s" and "10 s every 20 s" to "5 s every 15 s". In TASK-13892's notes, keep the "With option (a)" paragraph and drop the "With option (b)" one.

## ADR assessment

**ADR-081 is required**, by the coordinator's ruling on the review's I2. It is numbered at execution (Task 3, Step 1) and amends ADR-078's Consequences, as ADR-078 amended ADR-076's.
- **Why ADR-078 needs amending.** It applies the fail modes "while Redis is unreachable or rejects writes", and says that `fail_closed` refuses "for up to 60 s after".
- **What the cooldown changes:**
  - the fail mode also applies for the cooldown once a failure is confirmed, and for up to about 15 s after Redis returns;
  - `fail_closed`'s refusals can last about 75 s after recovery;
  - there is a new knob.
- **What ADR-081 records:** when the fail modes apply under the cooldown, the `fail_closed` bound, the recovery lag, the latency, the 5 s timeouts, and the knob.
- **The index:** `Docs/ADR/README.md` gains its row, and ADR-078's row gains "Amended in part by ADR-081".
- **ADR-076 is upheld:** leases count per worker while Redis is unreachable, and unreleased leases lapse within 120 s.
- **ADR-056 is upheld:** no configuration produces a permanent 429. The cooldown ends 10 s after the last failed probe, and `0` turns it off.
- **ADR-078's Consequences describe a lost reply counting its own lease.** Task 2 removes that. Accepted ADRs are not rewritten; ADR-078's Follow-up already points at TASK-13892, and ADR-081's Follow-up says when the cleanup reaches Redis.

## Review Focus

1. **The fail mode lasts longer than the outage.**
   - **The cycle:** RG retries 10 s after the last failed probe. A probe waits up to 5 s for its command and 5 s for its PING. So the fail mode goes on for up to about 15 s after Redis returns (measured 14.51 s for (a) and 14.8 s for (b)).
   - **Per mode:**
     - `fallback_memory` counts per worker for that long. The memory governor's leases are invisible to Redis (ADR-078's "Leases on the other side").
     - `fail_open` admits `tokens` uncounted for that long.
     - `fail_closed` refuses every new user for that long. Each keeps a `Retry-After` of up to 60 s, so 429s can last about 75 s after recovery.
   - **Where it is stated:** ADR-081, `Env_Vars.md`, `Rate_Limits_Troubleshooting.md` and `Usage_Quotas.md`.
   - **Tests:** `test_rg_calls_redis_again_once_it_answers_after_the_cooldown`, with a 0.5 s cooldown.
2. **What starts the cooldown.** A command must fail (refused, reset or timed out) and a PING right after it must fail too.
   - **An idle close:** a connection Redis or a proxy closed while idle fails one command, as on dev, and the PING reconnects, so no cooldown starts. Tests: `test_a_connection_redis_closed_while_idle_does_not_start_the_cooldown`, and `r2/p4_idle.py` at RG level.
   - **A stall:** a stall shorter than about two timeouts is covered by the PING (Decision 1).
   - **Error replies** (NOSCRIPT, READONLY, OOM) never start the cooldown, and they end one (the `error-reply` case).
   - **Cancellation:** a cancelled probe leaves its cooldown standing, so the next probe comes 10 s later.
   - **Which connection the PING takes:** redis-py drops the failed connection and releases it to a LIFO pool, so the PING normally takes that same connection and reconnects. Under concurrency it can take another pooled connection, which could in theory be stale too and fail the PING.
     - The re-review saw no such case in 6 bursts and ramps against 20-40 stale connections.
     - **Not taken:** `await client.connection_pool.disconnect(inuse_connections=False)` before the PING would make it deterministic, at the cost of dropping every idle connection on any single failure.
   - **No coalescing:** while a failure is being confirmed, other commands still reach Redis, and each of them that fails sends its own PING. That is why the first 10 s of an outage cost 10 s per request, and why a silent Redis gets one PING per failing request then (Decision 1).
     - **Upgrade, if the onset matters:** share one in-flight confirmation, so a later failure awaits the first PING instead of sending its own (about 5 lines).
3. **One cooldown per governor, not per worker.**
   - Each of the 9 governors in a worker has its own client, so each pays its own first wait and sends its own probes.
   - The `ponytail:` comment on `_REDIS_UNREACHABLE_COOLDOWN_S` names sharing one per Redis URL as the upgrade.
   - Not tested: it needs two governors on a black-holed Redis at once.
4. **The knob and the timeouts.**
   - `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` is read when a governor is built: the module-level limiters and the app's governor are built once, so the value holds from startup.
   - An unset, negative, non-numeric, infinite or NaN value logs a warning and uses 10 s (`test_the_cooldown_comes_from_its_env_var`, 8 cases).
   - `0` leaves the client unwrapped, which is dev's behaviour (`test_a_zero_cooldown_keeps_calling_an_unreachable_redis`).
   - Values below twice the socket timeout let probes overlap on a Redis that does not answer.
   - The 5 s timeouts are pinned (`test_rg_builds_its_redis_client_with_5_s_timeouts`). `create_async_redis_client`'s startup PING uses them too.
   - `socket_timeout` and `socket_connect_timeout` options in `REDIS_URL` override them: `?socket_timeout=30` gives 30.0. The docs say so.
5. **A reply lost to a long outage, beneath the cooldown.**
   - Once the PING also fails, the `ZREM` and the Python count both fail fast, and the request gets its fail mode.
   - The script's members stay in Redis until they expire: a rate member when its 60 s window ends; a lease at its TTL (up to 120 s).
   - The claim that a later commit removes the lease once Redis is back, because the handle carries the same ids, is not measured.
   - For a stall shorter than two timeouts, the `ZREM` reaches Redis (`r2/p5_lost_reply.py` under (b)).
6. **The unroutable case depends on the network.**
   - Here `10.255.255.1` black-holes.
   - Where the network answers "unreachable" at once, the case raises `ConnectionError`.
     - Under (b) it then still sends 2 commands and passes.
     - Under (a), Decision 1's test line accepts 2 for it.
   - The `silent` case covers `TimeoutError` everywhere.
7. **The extra `ZREM`s.** Task 2 adds one `ZREM` per key, only after a script fails while Redis answered `_is_real_redis`'s probe: once per worker per script after a Redis restart (NOSCRIPT).
8. **The cooldown's state** is a closure variable on one client, used from one event loop, as RG's `redis.asyncio` clients already are. The check and the probe push are synchronous, so one probe runs at a time.

---

Set once for every task:
- `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`;
- `export TLDW_TEST_NO_DOCKER=1`;
- `SCRATCH=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad`, this session's scratchpad;
- the private Redis and `REDIS_URL` from Global Constraints.

### Task 1: RG stops calling an unreachable Redis for a cooldown (TASK-13882)

**Files:**
- Modify `tldw_Server_API/app/core/Resource_Governance/governor_redis.py`:
  - L35-41: import redis-py's `ConnectionError` and `TimeoutError` in the existing `try`;
  - after `_FallbackToMemory` (L66-67): `_REDIS_SOCKET_TIMEOUT_S`, `_REDIS_UNREACHABLE_COOLDOWN_S`, `_redis_unreachable_cooldown_s` and `_stop_calling_redis_while_unreachable`;
  - `__init__` (L157): read the cooldown;
  - `_client_get` (L430-436): the pinned timeouts and the wrap.
- Test `tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py`:
  - helpers `_unreachable_redis(kind)`, `_redis_gone_since_connect(address, sent)` and `_redis_that_closes_its_first_connection()`;
  - `test_rg_stops_calling_an_unreachable_redis_until_the_cooldown_ends` (9 cases);
  - `test_a_zero_cooldown_keeps_calling_an_unreachable_redis`;
  - `test_the_cooldown_comes_from_its_env_var` (8 cases);
  - `test_rg_builds_its_redis_client_with_5_s_timeouts`;
  - `test_rg_calls_redis_again_once_it_answers_after_the_cooldown` (2 cases);
  - `test_a_connection_redis_closed_while_idle_does_not_start_the_cooldown`.
- Add: this plan, as `Docs/superpowers/plans/2026-10-22-rg-redis-unreachable-cooldown.md`.

**Interfaces:**
- **Produces:**
  - `governor_redis._REDIS_SOCKET_TIMEOUT_S: float = 5.0`;
  - `governor_redis._REDIS_UNREACHABLE_COOLDOWN_S: float = 10.0`;
  - `governor_redis._redis_unreachable_cooldown_s() -> float`;
  - `governor_redis._stop_calling_redis_while_unreachable(client: Any, cooldown_s: float) -> Any`. It replaces `client.execute_command` in place and returns the same client;
  - `RedisResourceGovernor._unreachable_cooldown_s: float`.
- **`_client_get`** keeps its signature.
- **Env var:** `RG_REDIS_UNREACHABLE_COOLDOWN_SEC`.

- [ ] **Step 1: Branch, and claim both tasks**

```bash
git fetch -q origin dev
git switch -c fix/rg-redis-unreachable-cooldown origin/dev
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13882 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-10-22-rg-redis-unreachable-cooldown.md (branch fix/rg-redis-unreachable-cooldown, with TASK-13892)."
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13892 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-10-22-rg-redis-unreachable-cooldown.md (branch fix/rg-redis-unreachable-cooldown, with TASK-13882)."
```

Copy this plan, as reviewed, to `Docs/superpowers/plans/2026-10-22-rg-redis-unreachable-cooldown.md`. If the owner chose (a) in Decision 1, make Decision 1's edits in each step below as you reach it.

- [ ] **Step 2: Write the failing tests**

Save this diff as `$SCRATCH/fu2rg_t1_tests.diff` and apply it with `git apply $SCRATCH/fu2rg_t1_tests.diff`. Every later diff is saved and applied the same way.

```diff
diff --git a/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py b/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
index 4cf19fd2cd..c55175d817 100644
--- a/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
+++ b/tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
@@ -1,9 +1,14 @@
+import asyncio
+import contextlib
+import socket
 import time

 import pytest
 import redis.asyncio as aioredis
 from redis.asyncio.retry import Retry
 from redis.backoff import NoBackoff
+from redis.exceptions import ConnectionError as RedisConnectionError
+from redis.exceptions import NoScriptError

 pytestmark = pytest.mark.rate_limit

@@ -215,6 +220,184 @@ async def test_renew_after_redis_returns_keeps_a_lease_taken_during_the_outage(f
     assert first.allowed and not second.allowed


+@contextlib.contextmanager
+def _unreachable_redis(kind: str):
+    """Yield ``host:port`` of a Redis RG cannot reach: unroutable, stopped (nothing listens) or silent (never answers)."""
+    if kind == "unroutable":
+        # Black-holed: a connect waits for the connect timeout, unless the network answers "unreachable" at once.
+        yield "10.255.255.1:6379"
+        return
+    with socket.socket() as sock:
+        sock.bind(("127.0.0.1", 0))
+        address = f"127.0.0.1:{sock.getsockname()[1]}"
+        if kind == "silent":
+            sock.listen(8)  # the kernel completes the connect; nothing reads or replies, so each command times out
+            yield address
+            return
+    yield address  # stopped: the port is free again, so a connect is refused
+
+
+def _redis_gone_since_connect(address: str, sent: list):
+    """A stand-in for ``create_async_redis_client``: the client it hands RG was built before Redis went away (so no
+    ping), with 0.2 s timeouts to keep the test fast, and it records each command it sends, PINGs included."""
+
+    async def _create(**_kwargs):
+        client = aioredis.from_url(f"redis://{address}", socket_connect_timeout=0.2, socket_timeout=0.2, decode_responses=True)
+        run = client.execute_command
+
+        async def _counted(*args, **options):
+            sent.append(args[0])
+            return await run(*args, **options)
+
+        client.execute_command = _counted
+        return client
+
+    return _create
+
+
+@pytest.mark.asyncio
+@pytest.mark.parametrize("kind", ["unroutable", "stopped", "silent"])
+@pytest.mark.parametrize("fail_mode, admitted", [("fail_open", True), ("fallback_memory", True), ("fail_closed", False)])
+async def test_rg_stops_calling_an_unreachable_redis_until_the_cooldown_ends(kind, fail_mode, admitted, monkeypatch):
+    """One command and the PING after it fail to reach Redis; RG answers the rest by its fail mode without calling Redis (TASK-13882)."""
+    from tldw_Server_API.app.core.Resource_Governance import governor_redis
+
+    sent = []
+    with _unreachable_redis(kind) as address:
+        monkeypatch.setattr(governor_redis, "create_async_redis_client", _redis_gone_since_connect(address, sent))
+        rg = _governor({"requests": {"rpm": 5}, "scopes": ["user"], "fail_mode": fail_mode}, ns=f"rg_unreachable_{fail_mode}")
+        req = RGRequest(entity="user:x", categories={"requests": {"units": 1}}, tags={"policy_id": "p"})
+
+        decisions = [(await rg.reserve(req))[0] for _ in range(3)]
+
+    assert ([d.allowed for d in decisions], len(sent), sent[1:2]) == ([admitted] * 3, 2, ["PING"])
+
+
+@pytest.mark.asyncio
+async def test_a_zero_cooldown_keeps_calling_an_unreachable_redis(monkeypatch):
+    """RG_REDIS_UNREACHABLE_COOLDOWN_SEC=0 turns the cooldown off: every reserve calls Redis, and nothing PINGs it (TASK-13882)."""
+    from tldw_Server_API.app.core.Resource_Governance import governor_redis
+
+    monkeypatch.setenv("RG_REDIS_UNREACHABLE_COOLDOWN_SEC", "0")
+    sent = []
+    with _unreachable_redis("stopped") as address:
+        monkeypatch.setattr(governor_redis, "create_async_redis_client", _redis_gone_since_connect(address, sent))
+        rg = _governor({"requests": {"rpm": 5}, "scopes": ["user"], "fail_mode": "fail_open"}, ns="rg_unreachable_off")
+        req = RGRequest(entity="user:x", categories={"requests": {"units": 1}}, tags={"policy_id": "p"})
+        per_reserve = []
+        for _ in range(3):
+            before = len(sent)
+            await rg.reserve(req)
+            per_reserve.append(len(sent) - before)
+
+    assert (min(per_reserve) > 0, "PING" in sent) == (True, False)
+
+
+@pytest.mark.parametrize(
+    "raw, seconds",
+    [(None, 10.0), ("", 10.0), ("0", 0.0), ("2.5", 2.5), ("-1", 10.0), ("ten", 10.0), ("inf", 10.0), ("nan", 10.0)],
+)
+def test_the_cooldown_comes_from_its_env_var(raw, seconds, monkeypatch):
+    """RG_REDIS_UNREACHABLE_COOLDOWN_SEC sets the cooldown; unset, or not a number of seconds >= 0, it is 10 s (TASK-13882)."""
+    from tldw_Server_API.app.core.Resource_Governance import governor_redis
+
+    if raw is None:
+        monkeypatch.delenv("RG_REDIS_UNREACHABLE_COOLDOWN_SEC", raising=False)
+    else:
+        monkeypatch.setenv("RG_REDIS_UNREACHABLE_COOLDOWN_SEC", raw)
+
+    assert governor_redis._redis_unreachable_cooldown_s() == seconds
+
+
+@pytest.mark.asyncio
+async def test_rg_builds_its_redis_client_with_5_s_timeouts(monkeypatch):
+    """RG's client waits at most 5 s to connect and for each reply, on any redis-py release the pin allows (TASK-13882)."""
+    from tldw_Server_API.app.core.Resource_Governance import governor_redis
+
+    async def _create(**kwargs):
+        # Builds the client from the governor's options, as redis_factory does; it never connects in this test.
+        return aioredis.from_url("redis://127.0.0.1:1", **(kwargs.get("redis_kwargs") or {}))
+
+    monkeypatch.setattr(governor_redis, "create_async_redis_client", _create)
+    client = await _governor({"scopes": ["user"]}, ns="rg_timeouts")._client_get()
+    options = client.connection_pool.connection_kwargs
+
+    assert (options.get("socket_connect_timeout"), options.get("socket_timeout")) == (5.0, 5.0)
+
+
+@pytest.mark.asyncio
+# Any answer from Redis shows it is reachable, an error reply too: after a restart the probe is often NOSCRIPT.
+@pytest.mark.parametrize("probe_answer", ["PONG", NoScriptError("No matching script.")], ids=["reply", "error-reply"])
+async def test_rg_calls_redis_again_once_it_answers_after_the_cooldown(probe_answer):
+    """After the cooldown one command goes to Redis while the others still fail fast; once Redis answers, all go (TASK-13882)."""
+    from tldw_Server_API.app.core.Resource_Governance import governor_redis
+
+    # The failed command, the PING that confirms Redis is down, the probe after the cooldown, the next command.
+    answers = [RedisConnectionError("Connection refused"), RedisConnectionError("Connection refused"), probe_answer, "after"]
+    sent = []
+
+    class _Redis:
+        async def execute_command(self, *args, **_options):
+            sent.append(args[0])
+            await asyncio.sleep(0)  # in flight: a concurrent command runs now
+            answer = answers.pop(0)
+            if isinstance(answer, Exception):
+                raise answer
+            return answer
+
+    client = governor_redis._stop_calling_redis_while_unreachable(_Redis(), cooldown_s=0.5)
+    with pytest.raises(RedisConnectionError):
+        await client.execute_command("ZCARD")  # Redis is down, and a PING confirms it
+    with pytest.raises(RedisConnectionError):
+        await client.execute_command("ZCARD")  # within the cooldown: not sent
+    await asyncio.sleep(0.6)
+    probe, other = await asyncio.gather(client.execute_command("ZCARD"), client.execute_command("ZCARD"), return_exceptions=True)
+
+    assert (probe is probe_answer, type(other), await client.execute_command("ZCARD"), sent) == (True, RedisConnectionError, "after", ["ZCARD", "PING", "ZCARD", "ZCARD"])
+
+
+@contextlib.asynccontextmanager
+async def _redis_that_closes_its_first_connection():
+    """Yield ``host:port`` of a Redis stand-in that answers every command +PONG, and closes its first connection after
+    the first PING, as Redis's ``timeout`` setting (or a proxy's idle timeout) closes an idle client."""
+    connections = []
+
+    async def _serve(reader, writer):
+        connections.append(writer)
+        first = len(connections) == 1
+        while data := await reader.read(65536):
+            writer.write(b"+PONG\r\n" * data.count(b"*"))  # one reply per command; redis-py's connect handshake accepts it
+            await writer.drain()
+            if first and b"PING" in data:
+                break
+        writer.close()
+
+    server = await asyncio.start_server(_serve, "127.0.0.1", 0)
+    try:
+        yield f"127.0.0.1:{server.sockets[0].getsockname()[1]}"
+    finally:
+        server.close()
+
+
+@pytest.mark.asyncio
+async def test_a_connection_redis_closed_while_idle_does_not_start_the_cooldown():
+    """Redis closed the pooled connection while it sat idle: that command fails, as before, but the PING after it gets an
+    answer, so RG keeps calling Redis. Starting the cooldown here would refuse every user under fail_closed (TASK-13882)."""
+    from tldw_Server_API.app.core.Resource_Governance import governor_redis
+
+    async with _redis_that_closes_its_first_connection() as address:
+        raw = aioredis.from_url(f"redis://{address}", socket_timeout=1)
+        client = governor_redis._stop_calling_redis_while_unreachable(raw, cooldown_s=10)
+        assert await client.ping()  # the connection the stand-in then closes
+        await asyncio.sleep(0.05)
+        with pytest.raises(RedisConnectionError):
+            await client.ping()  # that closed connection fails, as it does without the cooldown
+        answered = await client.ping()
+        await raw.aclose()
+
+    assert answered is True
+
+
 @pytest.mark.asyncio
 async def test_a_redis_outage_is_logged_as_a_warning_once_a_minute(monkeypatch):
     from loguru import logger
```

- [ ] **Step 3: Run them to see them fail**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py 2>&1 | grep -E "^E   (assert|AttributeError|AssertionError)|passed|failed" | sed -E 's/module .tldw_Server_API.app.core.Resource_Governance.governor_redis./governor_redis/' | sort | uniq -c
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py -k unreachable 2>&1 | grep -E "At index 1 diff" | sort | uniq -c
```

Expected (measured on `origin/dev` `e06eeec68c`; about 25 s, because dev waits 0.2 s on each command to the unroutable and silent addresses):
```text
   1 ================== 21 failed, 26 passed, 6 warnings in 24.50s ==================
   1 E   assert (None, None) == (5.0, 5.0)
   3 E   AssertionError: assert ([False, Fals...ANGEBYSCORE']) == ([False, Fals..., 2, ['PING'])
   6 E   AssertionError: assert ([True, True,...ANGEBYSCORE']) == ([True, True,..., 2, ['PING'])
   8 E   AttributeError: governor_redis has no attribute '_redis_unreachable_cooldown_s'
   3 E   AttributeError: governor_redis has no attribute '_stop_calling_redis_while_unreachable'
```
and
```text
   3 E     At index 1 diff: 11 != 2
   3 E     At index 1 diff: 15 != 2
   3 E     At index 1 diff: 28 != 2
```

**Why they fail:**
- Dev sends every command to the unreachable Redis: 28 commands over 3 reserves under `fail_open`, 15 under `fallback_memory` and 11 under `fail_closed`, for each of the three kinds. The plan expects 2, the first command and its PING.
- Dev's client has no pinned timeouts.
- The env var helper and the wrapper do not exist yet.
- `test_a_zero_cooldown_keeps_calling_an_unreachable_redis` passes on dev, because `0` is dev's behaviour. It is among the 26 passed, and it keeps the off switch honest.

- [ ] **Step 4: Add the cooldown, the knob and the pinned timeouts**

Save as `$SCRATCH/fu2rg_t1_src.diff` and apply.

```diff
diff --git a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
index b52947885b..07fdb967d4 100644
--- a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
+++ b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
@@ -33,7 +33,9 @@ from .policy_eval import (
 from .tenant import hash_entity

 try:
+    from redis.exceptions import ConnectionError as RedisConnectionError
     from redis.exceptions import RedisError
+    from redis.exceptions import TimeoutError as RedisTimeoutError
 except ImportError:  # pragma: no cover - redis is an optional deployment dependency
     _REDIS_ERRORS: tuple[type[Exception], ...] = ()
 else:
@@ -67,6 +69,79 @@ class _FallbackToMemory(Exception):
     """Signal that Redis operations failed and fallback_memory should be used."""


+# RG's Redis client waits at most this long to connect, and for each reply (TASK-13882). It is
+# redis-py 8's default, set here so older releases the redis>=4.5.0 pin allows wait no longer.
+# socket_timeout and socket_connect_timeout options in REDIS_URL override it.
+_REDIS_SOCKET_TIMEOUT_S = 5.0
+# How long RG stops sending commands to a Redis it could not reach (TASK-13882), unless
+# RG_REDIS_UNREACHABLE_COOLDOWN_SEC says otherwise (0 turns the cooldown off). At least twice the
+# socket timeout, so a probe and its confirming PING finish before the next probe starts.
+# ponytail: one cooldown per governor's client; share one per Redis URL if each governor's first wait matters.
+_REDIS_UNREACHABLE_COOLDOWN_S = 10.0
+
+
+def _redis_unreachable_cooldown_s() -> float:
+    """``RG_REDIS_UNREACHABLE_COOLDOWN_SEC``, or the default when it is unset or not a number of seconds >= 0."""
+    raw = (os.getenv("RG_REDIS_UNREACHABLE_COOLDOWN_SEC") or "").strip()
+    if not raw:
+        return _REDIS_UNREACHABLE_COOLDOWN_S
+    try:
+        seconds = float(raw)
+    except ValueError:
+        seconds = -1.0
+    if not math.isfinite(seconds) or seconds < 0:
+        logger.warning("RG_REDIS_UNREACHABLE_COOLDOWN_SEC={!r} is not a number of seconds >= 0; using {}", raw, _REDIS_UNREACHABLE_COOLDOWN_S)
+        return _REDIS_UNREACHABLE_COOLDOWN_S
+    return seconds
+
+
+def _stop_calling_redis_while_unreachable(client: Any, cooldown_s: float) -> Any:
+    """Make ``client`` fail fast for ``cooldown_s`` once Redis cannot be reached (TASK-13882).
+
+    A black-holed Redis makes each command wait the socket timeout, and a governed request sends
+    several commands while Redis is down. A command that fails to reach Redis (refused, reset or
+    timed out) starts the cooldown only if a PING right after it fails too. redis-py drops the
+    failed connection and its pool hands the most recently released one out first, so the PING
+    normally takes that connection and reconnects; a connection Redis closed while idle, or one
+    stall, does not stop RG. Within the cooldown a command raises redis ConnectionError at once,
+    which RG answers by its fail mode as it does a refused connection. After it, one command goes
+    to Redis while the rest keep failing fast; any answer from Redis, an error reply included,
+    ends the cooldown.
+    """
+    run = client.execute_command
+    blocked_until = 0.0
+
+    async def _unreachable() -> bool:
+        try:
+            await run("PING")
+        except (RedisConnectionError, RedisTimeoutError):
+            return True
+        except RedisError:
+            return False  # an error reply: Redis answered
+        return False
+
+    async def execute_command(*args: Any, **options: Any) -> Any:
+        nonlocal blocked_until
+        now = time.monotonic()
+        if now < blocked_until:
+            raise RedisConnectionError("Redis was unreachable; the Resource Governor retries it after a cooldown")
+        if blocked_until:
+            blocked_until = now + cooldown_s  # this command probes; the rest fail fast meanwhile
+        try:
+            result = await run(*args, **options)
+        except (RedisConnectionError, RedisTimeoutError):
+            blocked_until = time.monotonic() + cooldown_s if await _unreachable() else 0.0
+            raise
+        except RedisError:
+            blocked_until = 0.0  # Redis answered (NOSCRIPT, READONLY, OOM): it is reachable
+            raise
+        blocked_until = 0.0
+        return result
+
+    client.execute_command = execute_command
+    return client
+
+
 def _rate_window(policy: dict[str, Any], category: str) -> tuple[int, int]:
     """``(limit, window_seconds)`` of a requests or tokens sliding window."""
     if category == "requests":
@@ -155,6 +230,7 @@ class RedisResourceGovernor(ResourceGovernor):
         self._client_lock = asyncio.Lock()
         self._concurrency_lock = asyncio.Lock()
         self._fail_mode = rg_redis_fail_mode()
+        self._unreachable_cooldown_s = _redis_unreachable_cooldown_s()
         self._local_handles: dict[str, dict[str, Any]] = {}
         self._tokens_lua_sha: str | None = None
         self._multi_lua_sha: str | None = None
@@ -432,7 +508,14 @@ class RedisResourceGovernor(ResourceGovernor):
             return self._client
         async with self._client_lock:
             if self._client is None:
-                self._client = await create_async_redis_client(context="resource_governor", fallback_to_fake=True)
+                client = await create_async_redis_client(
+                    context="resource_governor",
+                    fallback_to_fake=True,
+                    redis_kwargs={"socket_connect_timeout": _REDIS_SOCKET_TIMEOUT_S, "socket_timeout": _REDIS_SOCKET_TIMEOUT_S},
+                )
+                if self._unreachable_cooldown_s > 0 and not getattr(client, "_tldw_is_stub", False):
+                    _stop_calling_redis_while_unreachable(client, self._unreachable_cooldown_s)
+                self._client = client
         return self._client

     async def _scan_keys(self, pattern: str, *, count: int = 1000) -> list[Any]:
```

**Notes:**
- Without redis installed, the `except` clauses name unbound classes. That is safe: `create_async_redis_client(fallback_to_fake=True)` then returns the stub, which is never wrapped.
- Tests that assign `rg._client` directly, as every TASK-13859 outage test does, bypass the wrapper, so they behave as before.
- So do tests that patch a command on the client from `_client_get()`: the instance attribute is found before `execute_command` is reached.

- [ ] **Step 5: Run the tests to see them pass**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py 2>&1 | tail -1
```

Expected: `47 passed`, in about 6 s.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py
git commit -m "fix(rg): stop calling an unreachable Redis for a cooldown (TASK-13882)

Since TASK-13859, RG answers a Redis error by its fail mode instead of
raising, so a governed request goes on making Redis calls during an
outage. With a black-holed Redis each call waits redis-py's 5 s
timeout: an ingress request took 30 s under fallback_memory and 70 s
under fail_open, every request.

A command that fails to reach Redis (refused, reset or timed out) is
now followed by one PING, which normally takes the connection just
dropped, so it reconnects. If the PING fails too, the
governor's client raises ConnectionError at once for
RG_REDIS_UNREACHABLE_COOLDOWN_SEC (10 s by default, 0 turns it off),
which RG's guards already answer by the fail mode, then lets one
command probe Redis. A connection Redis closed while idle, or one
stall shorter than two timeouts, gets an answer to the PING and does
not start the cooldown. RG's client now waits at most 5 s to connect
and for each reply, on any redis-py release the pin allows, unless
REDIS_URL sets socket_timeout or socket_connect_timeout.

A black-holed Redis now costs the first request 10 s and the rest
nothing until the next probe. RG retries 10 s after the last failed
probe, so its fail mode lasts up to about 15 s after Redis returns;
under fail_closed, a user refused then keeps a Retry-After of up to
60 s, so 429s can last about 75 s after recovery. The ADR that
records this follows in its own commit.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```


### Task 2: A lost script reply does not count the reserve's own members (TASK-13892)

**Files:**
- Modify `tldw_Server_API/app/core/Resource_Governance/governor_redis.py`, in `reserve`. Line numbers are after Task 1:
  - L1701-1715: `tmp_members` moves above the `try`;
  - L1753-1755: the multi-key rate script's `except` removes `tmp_members`;
  - L2006-2008: the lease script's `except` removes `planned`.
- Test `tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py`:
  - `_reply_lost(run)`;
  - `test_a_lost_lease_script_reply_does_not_count_its_own_lease`;
  - `test_a_lost_rate_script_reply_charges_the_request_once`.

**Interfaces:** none new. `tmp_members` keeps its type, `list[tuple[str, str, str, str, list[str]]]`; it is only bound earlier.

- [ ] **Step 1: Write the failing tests**

Save as `$SCRATCH/fu2rg_t2_tests.diff` and apply.

```diff
diff --git a/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py b/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
index 3a54d4f676..d6cfa87290 100644
--- a/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
+++ b/tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
@@ -5,6 +5,7 @@ import redis.asyncio as aioredis
 from redis.asyncio.retry import Retry
 from redis.backoff import NoBackoff
 from redis.exceptions import OutOfMemoryError, ReadOnlyError
+from redis.exceptions import TimeoutError as RedisTimeoutError

 from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest

@@ -118,6 +119,47 @@ async def test_a_lost_window_estimate_script_reloads_after_one_failure(real_redi
     assert [await reset_estimate(), await reset_estimate()] == [60, 30]


+def _reply_lost(run):
+    """Run the script on Redis, then lose its reply, as a redis-py socket timeout or a connection reset does."""
+
+    async def _command(*args, **kwargs):
+        await run(*args, **kwargs)
+        raise RedisTimeoutError("Timeout reading from socket")
+
+    return _command
+
+
+async def test_a_lost_lease_script_reply_does_not_count_its_own_lease(real_redis, rg_unique_ns, monkeypatch) -> None:
+    """The lease script takes the lease, then its reply is lost: the user's only slot is still this request's (TASK-13892)."""
+    monkeypatch.delenv("RG_TEST_PURGE_LEASES_BEFORE_RESERVE")
+    gov = RedisResourceGovernor(policy_loader=_NoJobsBlock(), ns=rg_unique_ns)
+    if not await gov._is_real_redis():
+        pytest.skip("Redis client is not real; using in-memory stub")
+    client = await gov._client_get()
+    monkeypatch.setattr(client, "evalsha", _reply_lost(client.evalsha))
+    key = gov._keys.lease("limits.test_concurrent", "jobs", "user", "42")
+
+    decision, handle = await gov.reserve(_lease())
+    held = await client.zcard(key)
+    await gov.release(handle)
+
+    assert (decision.allowed, held, await client.zcard(key)) == (True, 1, 0)
+
+
+async def test_a_lost_rate_script_reply_charges_the_request_once(real_redis, rg_unique_ns, monkeypatch) -> None:
+    """The multi-key rate script charges the window, then its reply is lost: the window holds one charge, not two (TASK-13892)."""
+    gov = RedisResourceGovernor(policy_loader=_Policy("fallback_memory", requests={"rpm": 5}), ns=rg_unique_ns)
+    if not await gov._is_real_redis():
+        pytest.skip("Redis client is not real; using in-memory stub")
+    client = await gov._client_get()
+    monkeypatch.setattr(client, "evalsha", _reply_lost(client.evalsha))
+    req = RGRequest(entity="user:42", categories={"requests": {"units": 1}}, tags={"policy_id": "rate.test"})
+
+    decision, _ = await gov.reserve(req)
+
+    assert (decision.allowed, await client.zcard(gov._keys.win("rate.test", "requests", "user", "42"))) == (True, 1)
+
+
 def _rejecting(error: Exception):
     """A Redis command that fails with ``error``, as a replica (READONLY) or a full server (OOM) answers a write."""

```

The patched `evalsha` sits above the client's `execute_command`, so this lost reply never reaches the cooldown wrapper. That is the AC's case. Decision 1's table and Review Focus 5 cover a reply lost beneath the wrapper.

- [ ] **Step 2: Run them to see them fail**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py 2>&1 | grep -E "^E   assert|passed|failed"
```

Expected (measured after Task 1 against the private Redis, and the same on unchanged `origin/dev`):
```text
E   assert (False, 1, 1) == (True, 1, 0)
E   assert (True, 2) == (True, 1)
================== 2 failed, 29 passed, 37 warnings in 3.39s ===================
```

**Why they fail:**
- **Lease:** the Python count includes the lease the script just took, so the user's only slot refuses the request. Its `release(None)` frees nothing, and the script's lease stays.
- **Rate:** the Python path charges the window again under new member ids, so one request holds two members.

- [ ] **Step 3: Remove the script's members before counting in Python**

Save as `$SCRATCH/fu2rg_t2_src.diff` and apply.

```diff
diff --git a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
index 07fdb967d4..85fe21ec31 100644
--- a/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
+++ b/tldw_Server_API/app/core/Resource_Governance/governor_redis.py
@@ -1701,6 +1701,8 @@ class RedisResourceGovernor(ResourceGovernor):
         fallback_to_memory = False
         denial_retry_after = 0
         used_lua = False
+        # The script's members, kept to populate added_members on success and to remove on failure
+        tmp_members: list[tuple[str, str, str, str, list[str]]] = []  # (category, sc, ev, key, members)

         # Attempt real-Redis multi-key Lua script when available (disabled when forcing stub rate)
         try:
@@ -1710,8 +1712,6 @@ class RedisResourceGovernor(ResourceGovernor):
                 argv: list[Any] = []
                 # ARGV[1]=now, ARGV[2]=kcount; rest per-key quads
                 now_f = float(now)
-                # We'll build a temporary structure to also populate added_members on success
-                tmp_members: list[tuple[str, str, str, str, list[str]]] = []  # (category, sc, ev, key, members)
                 for category, cfg in req.categories.items():
                     units = int(cfg.get("units") or 0)
                     if units <= 0:
@@ -1753,6 +1753,11 @@ class RedisResourceGovernor(ResourceGovernor):
             used_lua = False
             self._last_used_multi_lua = False
             self._multi_lua_sha = None
+            # A script that ran but lost its reply charged these members: take them out, or the
+            # Python path below counts them and charges the request again (TASK-13892).
+            for _category, _sc, _ev, win_key, members in tmp_members:
+                with contextlib.suppress(_RG_NONCRITICAL_EXCEPTIONS):
+                    await client.zrem(win_key, *members)
         if not used_lua:
             if await self._is_stub_client():
                 # Pre-check across all scopes/categories
@@ -2006,6 +2011,11 @@ class RedisResourceGovernor(ResourceGovernor):
                         # Redis answered the probe but not the script (NOSCRIPT after a restart, OOM,
                         # READONLY): load it again next time, and count leases in the Python path below.
                         self._concurrency_lua_sha = None
+                        # A script that ran but lost its reply left this reserve's leases in Redis:
+                        # take them out, or the count below includes them (TASK-13892).
+                        for (sc, ev), members in planned.items():
+                            with contextlib.suppress(_RG_NONCRITICAL_EXCEPTIONS):
+                                await client.zrem(self._keys.lease(policy_id, category, sc, ev), *members)

                 if not used_concurrency_lua:
                     active_by_scope: dict[tuple[str, str], int] = {}
```

**Notes:**
- `client` is the one `reserve` takes at its start (L1357 after Task 1).
- One variadic `ZREM` per key, not `_zrem_members`, which sends one command per member: a `tokens` key can hold about 1000.
- **Leases:** the Python path re-adds the same `planned` ids if it admits, so the lease ends up held once.
- **Under READONLY:** the `ZREM` is rejected and suppressed.

- [ ] **Step 4: Run the tests to see them pass**

```bash
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py 2>&1 | tail -1
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py 2>&1 | tail -1
```

Expected: `31 passed`, then `78 passed`.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
git commit -m "fix(rg): a lost script reply no longer counts the reserve's own members (TASK-13892)

When the lease script or the multi-key rate script ran but its reply
was lost, reserve fell through to the Python path, which counted the
members the script had just written. A lease at a limit of 1 was
refused and its lease blocked the user's only slot for up to two
minutes; a rate reserve was charged twice, and the script's charge was
never refunded.

reserve now removes the members it planned for the failed script, one
ZREM per key, before the Python path counts. They are this reserve's
own UUIDs, so removing them is always safe, and a no-op when the
script never ran. With TASK-13882's cooldown, a reply lost to a stall
shorter than two socket timeouts does not start the cooldown, so this
cleanup reaches Redis and the request is counted once.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 3: Record the cooldown: ADR-081, the env vars and the operator pages

**Files:**
- Create: `Docs/ADR/081-resource-governor-stops-calling-an-unreachable-redis-for-a-cooldown.md`.
- Modify:
  - `Docs/ADR/README.md`: a new row, and "Amended in part by" on ADR-078's row;
  - `Docs/Operations/Env_Vars.md`: `RG_REDIS_FAIL_MODE` and the new `RG_REDIS_UNREACHABLE_COOLDOWN_SEC`;
  - `Docs/Operations/Rate_Limits_Troubleshooting.md`: a `fail_closed` row;
  - `Docs/Operations/Usage_Quotas.md:125`: a new bullet;
  - `tldw_Server_API/app/core/Resource_Governance/README.md:28`: a sub-bullet.
- Generated by the refresh script:
  - `Docs/Published/ADR/081-...md`;
  - `Docs/Published/ADR/README.md`;
  - `Docs/Published/Env_Vars.md`.

- [ ] **Step 1: Number the ADR, under the round's id lock**

```bash
git fetch -q origin dev && git rebase origin/dev
N=$({ git ls-tree --name-only origin/dev Docs/ADR/
  gh pr list --state open --limit 500 --json number,changedFiles,files \
    --jq '.[] | if .changedFiles > 100 then "PR#\(.number)" else .files[].path end' \
    | while IFS= read -r line; do
        if [[ $line == PR#* ]]; then
          gh api --paginate "repos/rmusser01/tldw_server/pulls/${line#PR#}/files" --jq '.[].filename'
        else
          printf '%s\n' "$line"
        fi
      done
} | sed -nE 's|^Docs/ADR/([0-9]{3})-.*|\1|p' | sort -n | tail -1)
N=$(printf '%03d' $((10#$N + 1)))
echo "$N"; ls Docs/ADR | grep "^$N-" || echo "ADR-$N is free"
```

**Expected:**
- `echo` prints a number of at least `081`, and the check prints `ADR-<N> is free`.
- If the rebase moved `Docs/ADR/README.md`, the row diff below may need `git apply -3`.

- [ ] **Step 2: Apply the docs**

Save this diff as `$SCRATCH/fu2rg_t3_docs.diff`. Number it, set the ADR's **Date** to today, and apply it:

```bash
sed "s/NNN/$N/g; s/^+\*\*Date:\*\* 2026-10-22/+**Date:** $(date +%F)/" $SCRATCH/fu2rg_t3_docs.diff > $SCRATCH/fu2rg_t3_docs.$N.diff
git apply $SCRATCH/fu2rg_t3_docs.$N.diff
```

```diff
diff --git a/Docs/ADR/081-resource-governor-stops-calling-an-unreachable-redis-for-a-cooldown.md b/Docs/ADR/081-resource-governor-stops-calling-an-unreachable-redis-for-a-cooldown.md
new file mode 100644
index 0000000000..4d151ad697
--- /dev/null
+++ b/Docs/ADR/081-resource-governor-stops-calling-an-unreachable-redis-for-a-cooldown.md
@@ -0,0 +1,44 @@
+# ADR-081: The Resource Governor stops calling an unreachable Redis for a cooldown
+
+**Status:** Accepted
+**Date:** 2026-10-22
+**Backfilled from:** not backfilled
+**Decision owner:** Repository owner (@rmusser01)
+**Related task:** TASK-13882
+**Related spec/plan:** `Docs/superpowers/plans/2026-10-22-rg-redis-unreachable-cooldown.md`
+**Amends:** ADR-078 (Consequences: when the fail modes apply, and the `fail_closed` bound after Redis returns)
+
+## Decision
+
+Once a Redis command and a PING right after it both fail to reach Redis, the Resource Governor's Redis backend stops calling Redis for `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` (10 s by default, `0` turns it off) and answers by its fail mode, then probes Redis with one command at a time.
+
+## Context
+
+ADR-078 made RG answer every redis-py error by its fail mode instead of raising. A governed request then goes on making Redis calls during an outage, and with a Redis that does not answer (a network partition, or a host gone without a reset) each call waits redis-py's 5 s timeout. Measured on a black-holed address, an ingress request took 30 s under `fallback_memory` and 70 s under `fail_open`, and a quota request 10 s and 65 s, for every request of the outage (TASK-13882).
+
+## Alternatives considered
+
+| Option | Why rejected |
+| --- | --- |
+| A shorter socket timeout only | At 0.5 s a `fail_open` ingress request still waits 7 s, every request, and a shorter reply timeout turns more slow replies into lost ones (TASK-13892). |
+| Start the cooldown on any one failed command | A connection Redis or a proxy closed while idle, or one stall past the timeout, would start it. Under `fail_closed` that refused every new user for 10 s, where RG had admitted them, and each refused user kept a 60 s `Retry-After`. |
+| Start the cooldown on one timeout, and confirm only a refused or reset command with a PING | One stall between 5 and 10 s then refuses every new user under `fail_closed` for about 10 s, and a script reply lost to such a stall cannot be cleaned up (TASK-13892). It saves 5 s on the first request of an outage. |
+| The shared `Infrastructure/circuit_breaker.py` | It logs a WARNING on every trip, for each of the nine governors in a worker, against ADR-078's one warning per worker a minute; a registry breaker reads a database on every call. |
+
+## Consequences
+
+- **When the fail modes apply.** ADR-078 applies them while Redis is unreachable or rejects writes. With the cooldown:
+  - a single failed command still gets the fail mode for that call alone, as before;
+  - once a PING confirms the failure, every call of that governor gets the fail mode for the cooldown, without reaching Redis;
+  - after Redis returns, they go on for up to the cooldown plus one socket timeout: about 15 s by default.
+- **`fail_closed` after Redis returns.** A user refused in that time keeps a `Retry-After` of up to 60 s, so 429s can last about 75 s after Redis recovers. ADR-078 said up to 60 s.
+- **Latency while Redis does not answer.** Requests in the first 10 s of an outage wait up to 10 s each, a call's timeout and its PING's. Then one request per governor waits up to 10 s every 20 s or so, as it probes Redis.
+- **Timeouts.** RG's client waits at most 5 s to connect and for each reply, redis-py 8's defaults, set explicitly so older releases the `redis>=4.5.0` pin allows wait no longer. `socket_timeout` and `socket_connect_timeout` options in `REDIS_URL` override them.
+- **Release and renewal.** ADR-078's "renewed after Redis returns" now means after the cooldown ends: a lease in RG's lease map reaches Redis with its first renewal after that. A lease released in the lag window is released in the worker's copy only; if Redis holds it, it lapses there within 120 s.
+- **Per governor.** Each governor (ingress, the quota lease store, each module's limiter) has its own client, cooldown and probes.
+- **The knob.** `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` is read when each governor is built: at startup, or on a limiter's first use. A shorter cooldown shortens the lag after recovery; below 10 s, probes can overlap on a Redis that does not answer. `0` restores the ADR-078 behaviour.
+- **Unchanged from ADR-078:** what each fail mode does, per-worker lease counting, the once-a-minute warning, and the error replies (NOSCRIPT, READONLY, OOM), which never start the cooldown and end one.
+
+## Follow-up
+
+- TASK-13892 removes a failed script's own members before RG counts in Python. Because one stall does not start the cooldown, that cleanup reaches Redis for a reply lost to a stall shorter than about two timeouts.
diff --git a/Docs/ADR/README.md b/Docs/ADR/README.md
index 958208bc47..7eea7e8bee 100644
--- a/Docs/ADR/README.md
+++ b/Docs/ADR/README.md
@@ -103,9 +103,10 @@ Small bug fixes, local implementation details, product copy, temporary experimen
 | [ADR-075](075-native-messages-and-mcp-tools-use-the-provider-recorder.md) | Accepted | A provider call no `ChatProvider` method makes (the Messages API native path) records through the provider recorder's public wrappers, and an MCP tool's LLM calls are recorded for its caller even under a caller that records itself. |
 | [ADR-076](076-per-user-concurrency-quotas.md) | Accepted | Per-user synchronous concurrency (media ingest, audio streams, direct transcription) is a `limits.*` usage quota held as one Resource Governor lease per request; the request carries its own limit, the lease store follows `RG_BACKEND`, and `RG_ENABLED` does not turn it off. Amended in part by [ADR-078](078-concurrency-leases-count-per-worker-while-redis-is-unreachable.md): while Redis is unreachable, a lease is counted per worker instead of admitted without one. |
 | [ADR-077](077-selected-existing-input-chat-regeneration.md) | Proposed | Settle an assistant reply against a selected existing chat question while retaining owner-bound capture and recovery. |
-| [ADR-078](078-concurrency-leases-count-per-worker-while-redis-is-unreachable.md) | Accepted | The Resource Governor's Redis backend answers redis-py's `RedisError` by its fail mode, so while Redis is unreachable a per-user concurrency lease is counted in the worker that holds it, and a lease script Redis cannot run falls back to counting in Python. |
+| [ADR-078](078-concurrency-leases-count-per-worker-while-redis-is-unreachable.md) | Accepted | The Resource Governor's Redis backend answers redis-py's `RedisError` by its fail mode, so while Redis is unreachable a per-user concurrency lease is counted in the worker that holds it, and a lease script Redis cannot run falls back to counting in Python. Amended in part by [ADR-081](081-resource-governor-stops-calling-an-unreachable-redis-for-a-cooldown.md): once a command and a PING both fail, the fail mode applies for a cooldown, and under `fail_closed` for up to about 75 s after Redis returns. |
 | [ADR-079](079-explicit-postgresql-database-recovery-scope.md) | Proposed | Require explicit global scope, platform admission and overwrite confirmation for PostgreSQL database recovery covering all users and Jobs. |
 | [ADR-080](080-chacha-postgres-checkouts-owned-by-entry-points.md) | Accepted | On PostgreSQL each non-HTTP unit of work (a WebSocket step or persona call, a job owner or cancel action, a background pass) owns its ChaCha checkouts through `chacha_operation` or `@owns_chacha_checkouts`, not centrally in `WorkerSDK`; ChaCha transactions never run concurrently inside one operation; the thread-local fallback is returned on thread exit, a dropped handle's only at cyclic GC and after its thread exits, and `close_all_connections()` never closes the shared content pool. |
+| [ADR-081](081-resource-governor-stops-calling-an-unreachable-redis-for-a-cooldown.md) | Accepted | Once a Redis command and a PING right after it both fail, the Resource Governor's Redis backend stops calling Redis for `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` (10 s by default, `0` off) and answers by its fail mode, then probes with one command at a time. |


 The original backfill inventory is [the 2026-06-03 decision inventory](inventory/2026-06-03-decision-inventory.md). See the [2026-09-25 reconciliation](inventory/2026-09-25-decision-inventory-reconciliation.md) for current dispositions of its unresolved rows.
diff --git a/Docs/Operations/Env_Vars.md b/Docs/Operations/Env_Vars.md
index 169ac10d8d..3a83a02d07 100644
--- a/Docs/Operations/Env_Vars.md
+++ b/Docs/Operations/Env_Vars.md
@@ -346,7 +346,8 @@ The Resource Governor (RG) is the **primary enforcement path** for all rate limi
 - `RG_POLICY_RELOAD_ENABLED`: Enable hot-reload of policy changes (`true|false`). Default `true`.
 - `RG_POLICY_RELOAD_INTERVAL_SEC`: Policy reload check interval in seconds. Default `10`.
 - `RG_ROUTE_MAP_AUDIT`: When `true` (the default), the startup audit logs routes not covered by the RG route map.
-- `RG_REDIS_FAIL_MODE`: Behavior when Redis is unavailable (`fail_open` | `fail_closed` | `fallback_memory`). Default `fallback_memory`.
+- `RG_REDIS_FAIL_MODE`: Behavior when Redis is unavailable (`fail_open` | `fail_closed` | `fallback_memory`). Default `fallback_memory`. Once a Redis command and a PING right after it both fail, RG stops calling Redis for `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` and answers by this fail mode, so the fail mode lasts up to about 15 s after Redis returns (the cooldown after the last failed probe, plus one 5 s timeout); under `fail_closed`, a user refused in that time keeps a `Retry-After` of up to 60 s, so 429s can last about 75 s after Redis recovers (ADR-081).
+- `RG_REDIS_UNREACHABLE_COOLDOWN_SEC`: Seconds the RG Redis backend stops calling a Redis it could not reach before it probes again. Default `10`; `0` turns the cooldown off, so every Redis call waits its own 5 s timeout while Redis does not answer. Below `10`, probes can overlap on a Redis that does not answer. An unset, negative or non-numeric value uses the default. Read when each governor is built (at startup, or on a limiter's first use). RG's Redis client waits at most 5 s to connect and for each reply; `socket_timeout` and `socket_connect_timeout` options in `REDIS_URL` override that (ADR-081).

 ### Client Identity
 - `RG_TRUSTED_PROXIES`: Comma-separated trusted-proxy host/CIDR list. This is opt-in: forwarding is used only when the physical peer is a valid IP in this list.
diff --git a/Docs/Operations/Rate_Limits_Troubleshooting.md b/Docs/Operations/Rate_Limits_Troubleshooting.md
index e58bad6212..090f9a1922 100644
--- a/Docs/Operations/Rate_Limits_Troubleshooting.md
+++ b/Docs/Operations/Rate_Limits_Troubleshooting.md
@@ -7,6 +7,7 @@ See also: [ADR-056](../ADR/056-resource-governor-safety-net.md) for why the Reso
 | Response | Layer | Where it lives | How to tune |
 |---|---|---|---|
 | `{"error":"rate_limited","policy_id":"...","retry_after":N}`, HTTP 429 | RG ingress (`RGSimpleMiddleware`) | `Resource_Governance/middleware_simple.py` | Edit the named policy's `requests` block in `tldw_Server_API/Config_Files/resource_governor_policies.yaml` (hot-reloads within `RG_POLICY_RELOAD_INTERVAL_SEC`, default 10s), or set `RG_ENABLED=false` to disable RG ingress (and every other RG-governed path except the auth brute-force floor below). |
+| `{"error":"rate_limited",...}`, HTTP 429 for many users at once, while Redis is unreachable or soon after it returns | RG's Redis backend under `RG_REDIS_FAIL_MODE=fail_closed` | `Resource_Governance/governor_redis.py` | `fail_closed` refuses while RG cannot reach Redis, and for up to about 15 s after Redis returns: once a command and a PING both fail, RG stops calling Redis for `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` (default 10 s), and the probe after that waits up to one 5 s timeout. A user refused then keeps a `Retry-After` of up to 60 s, so the 429s can last about 75 s after recovery. Check Redis first. To admit meanwhile, use `fallback_memory` (the default: each worker counts on its own) or `fail_open`. A lower `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` shortens the lag after recovery; `0` turns the cooldown off, at up to 5 s per Redis call while Redis does not answer (ADR-081). |
 | `Rate limit exceeded (ResourceGovernor policy=chat.default); retry_after=Ns`, HTTP 429 | Chat token bucket (`_maybe_enforce_with_rg_chat` / the `/chat/completions` RG reserve) | `Chat/rate_limiter.py`, `api/v1/endpoints/chat.py` | Edit `chat.default`'s `tokens` block (`per_min`, `burst`) in `resource_governor_policies.yaml`. |
 | `Rate limit exceeded for resource: <resource>`, HTTP 429 | RBAC resource-aware limiter (`enforce_rbac_rate_limit`) | `api/v1/API_Deps/auth_deps.py` | Edit the resource's `rate_limit_class` (`requests_per_min`, `burst`) under `rate_limit_classes` in `tldw_Server_API/Config_Files/privilege_catalog.yaml`, or the per-user/per-role override in the `rbac_user_rate_limits` / `rbac_role_rate_limits` tables. |
 | `Rate limit exceeded for endpoint: <path>` (general) or `Auth rate limit exceeded for endpoint: <path>` (auth routes), HTTP 429 | `auth_deps.py` ingress fallback (`check_rate_limit` / `check_auth_rate_limit`) | `api/v1/API_Deps/auth_deps.py` | General (`check_rate_limit`): `AUTH_DEPS_FALLBACK_RATE_LIMIT` (default 120/min), or `RG_ENABLED=false` — this one honors the RG switch and turns off with it. Auth (`check_auth_rate_limit`): `AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT` (default 30/min) — this is a brute-force floor for auth endpoints and stays on even with RG disabled. Either fallback only runs when RG ingress did **not** already resolve a policy for the request (`request.state.rg_policy_id` unset) — a route outside `/api/`, or any route while RG is disabled. Both are skipped only for a request that already resolved an authenticated single-user principal, or in test mode — **not** simply because `AUTH_MODE=single_user`: an anonymous single-user-mode request to `/auth/login`, `/auth/register`, or `/auth/single-user/session` is still floored. |
diff --git a/Docs/Operations/Usage_Quotas.md b/Docs/Operations/Usage_Quotas.md
index 539271bd10..b62ec54f12 100644
--- a/Docs/Operations/Usage_Quotas.md
+++ b/Docs/Operations/Usage_Quotas.md
@@ -123,6 +123,7 @@ What `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd` and `limits.ll
 - `Retry-After` on a refusal is when the oldest held lease would expire if it stopped renewing, up to 2 minutes. A slot usually frees sooner, when one of the user's requests finishes, so a client may retry earlier.
 - Lowering a limit does not stop running requests. New ones are refused until the user is below the new limit.
 - If Redis is unreachable when a worker's lease store first connects, that worker counts on its own for the life of the process, as on `memory`. If Redis drops later, or rejects writes (a read-only replica, or out of memory), the worker counts on its own until Redis recovers, and a warning is logged (ADR-078). It does not see leases taken before the drop, a lease released during the outage frees its slot at once, and leases the worker could not release in Redis expire within 2 minutes. After a Redis restart, leases are counted as before.
+- Once a Redis command and a PING right after it both fail (refused, reset or timed out), each limiter in the worker stops calling Redis for `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` (10 seconds by default) and answers by the fail mode, then lets one command through to see whether Redis is back. A single failed command, such as one on a connection Redis closed while idle, does not start this. With a Redis that stops answering (a network partition, say), requests in the first 10 seconds wait up to 10 seconds each (two 5-second timeouts), then about one request every 20 seconds does, where every request used to wait 30 to 70 seconds. The worker keeps counting on its own for up to about 15 seconds after Redis returns (ADR-081).

 Two other `limits.*` entries in the catalog are not part of this scheme:

diff --git a/tldw_Server_API/app/core/Resource_Governance/README.md b/tldw_Server_API/app/core/Resource_Governance/README.md
index 36f2583c1f..4753a035c6 100644
--- a/tldw_Server_API/app/core/Resource_Governance/README.md
+++ b/tldw_Server_API/app/core/Resource_Governance/README.md
@@ -26,6 +26,7 @@ Centralized rate limiting and concurrency control with policy-based configuratio
 ## Backend Selection (env)
 - `RG_BACKEND`: `memory` (default) or `redis`
 - `RG_REDIS_FAIL_MODE`: `fallback_memory` (default) | `fail_closed` | `fail_open`
+  - Once a Redis command and a PING right after it both fail (refused, reset or timed out), the governor stops calling Redis for `RG_REDIS_UNREACHABLE_COOLDOWN_SEC` seconds (default 10; `0` turns it off) and answers by this fail mode, then probes Redis with one command at a time. The fail mode then lasts up to about 15 s after Redis returns. RG's client waits at most 5 s to connect and for each reply, unless `REDIS_URL` sets `socket_timeout` or `socket_connect_timeout` (ADR-081, TASK-13882).
 - `REDIS_URL`: Redis connection URL (used when backend=redis). If unset, defaults to `redis://127.0.0.1:6379`.
 - Determinism (tests/dev): `RG_TEST_FORCE_STUB_RATE=1` prefers in‑process rails for requests/tokens when running Redis backend, stabilizing burst vs steady retry‑after behavior in CI.

```

Then number this plan's copy too: `sed -i '' "s/NNN/$N/g" Docs/superpowers/plans/2026-10-22-rg-redis-unreachable-cooldown.md`.

- [ ] **Step 3: Refresh the mirror, and run the Docs tests**

```bash
bash Helper_Scripts/refresh_docs_published.sh
git add Docs/ADR Docs/Operations/Env_Vars.md Docs/Operations/Rate_Limits_Troubleshooting.md Docs/Operations/Usage_Quotas.md tldw_Server_API/app/core/Resource_Governance/README.md Docs/Published
git status --short
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs 2>&1 | tail -1
```

**Expected:**
- The refresh prints `Refreshed curated docs in: .../Docs/Published`.
- `git status --short` lists exactly:
  - the five edited docs;
  - `Docs/ADR/<N>-resource-governor-stops-calling-an-unreachable-redis-for-a-cooldown.md` and its copy under `Docs/Published/ADR/`;
  - `Docs/Published/ADR/README.md` and `Docs/Published/Env_Vars.md`.
- The Docs tests print `212 passed`.

`git add` comes first: `test_refresh_output_matches_tracked_published_files` compares the mirror against tracked files, so an untracked new ADR fails it.

- [ ] **Step 4: Commit**

```bash
git commit -m "docs(adr): ADR-$N, the Resource Governor stops calling an unreachable Redis for a cooldown (TASK-13882)

ADR-078 applies RG's fail modes while Redis is unreachable. With the
cooldown they also apply once a command and a PING both fail, for
RG_REDIS_UNREACHABLE_COOLDOWN_SEC (10 s by default), and for up to
about 15 s after Redis returns, so fail_closed's 429s can last about
75 s after recovery. ADR-$N records that and amends ADR-078 in part.
Env_Vars.md documents the knob and the lag next to RG_REDIS_FAIL_MODE,
Rate_Limits_Troubleshooting.md gains a fail_closed row, and
Usage_Quotas.md and the RG README say the same.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 4: Verify, file the follow-up, close both tasks, open the PR

**Files:**
- Modify: both task files, through backlog-py.
- Create: one follow-up task file, through backlog-py.
- Add: `Docs/superpowers/plans/2026-10-22-rg-redis-unreachable-cooldown.md`, copied in Task 1, Step 1.

- [ ] **Step 1: Run the sweep, the ratchet and the triage**

```bash
git grep -l -E "governor_redis|RedisResourceGovernor|_stop_calling_redis_while_unreachable|RG_REDIS_UNREACHABLE_COOLDOWN_SEC" -- 'tldw_Server_API/tests/*.py' | grep -v -E "^tldw_Server_API/tests/(Resource_Governance|Usage)/"
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance tldw_Server_API/tests/Usage tldw_Server_API/tests/Services/test_startup_resource_governor.py 2>&1 | tail -1
$PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/CI/test_test_quality_ratchet.py tldw_Server_API/tests/Docs 2>&1 | tail -1
$PY Helper_Scripts/ci/test_quality_triage.py --enforce 2>&1 | grep "test-triage\] \(OK\|FAIL\)"
```

Expected, measured on the prototype with `REDIS_URL` on the private Redis:
- **The grep** prints only `tldw_Server_API/tests/Services/test_startup_resource_governor.py`.
- **The sweep:** `785 passed, 1 skipped, 2 xfailed`, in about 5 minutes. The Postgres-backed RG tests skip without a reachable Postgres.
- **Docs and ratchet:** `213 passed`.
- **Triage:** `[test-triage] OK — no offenses beyond the baseline.`

If anything else fails, run the same test on `origin/dev` before going on.

- [ ] **Step 2: Lint and security**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/tests/Resource_Governance/test_rg_fail_modes_across_categories.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
uvx bandit -q -ll tldw_Server_API/app/core/Resource_Governance/governor_redis.py; echo "bandit exit=$?"
git diff --check origin/dev...HEAD && echo "no whitespace errors"
```

Expected:
- `All checks passed!`;
- `bandit exit=0`, with nothing else printed (`uvx` may print package download lines on its first run);
- `no whitespace errors`.

- [ ] **Step 3: Measure the black-holed latency for the PR body**

Save as `$SCRATCH/fu2rg_latency.py`. It is a scratch probe, not committed.

```python
"""Wall time per governed request with a black-holed Redis and redis-py's default 5 s timeouts (TASK-13882).

The governor's client is built through _client_get (so the cooldown applies), from from_url on 10.255.255.1 with no
ping, as a client built before Redis went away. Ingress = reserve + peek_with_policy + commit; quota = reserve + renew + release.
"""
import asyncio
import os
import time
import uuid

os.environ.pop("RG_TEST_PURGE_LEASES_BEFORE_RESERVE", None)

import redis.asyncio as aioredis

from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest, governor_redis


async def _blackholed(**_kw):
    return aioredis.from_url("redis://10.255.255.1:6379", decode_responses=True)


governor_redis.create_async_redis_client = _blackholed


class _P:
    def __init__(self, pol):
        self.p = pol

    def get_policy(self, _):
        return dict(self.p)


async def main():
    for fail_mode in ("fallback_memory", "fail_open", "fail_closed"):
        gov = RedisResourceGovernor(policy_loader=_P({"scopes": ["user"], "fail_mode": fail_mode, "requests": {"rpm": 50}}), ns=f"fu2rg_{uuid.uuid4().hex[:6]}")
        times = []
        for _ in range(3):
            t = time.monotonic()
            d, h = await gov.reserve(RGRequest(entity="user:1", categories={"requests": {"units": 1}}, tags={"policy_id": "p"}), op_id=str(uuid.uuid4()))
            await gov.peek_with_policy("user:1", ["requests"], "p")
            if h:
                await gov.commit(h, actuals=None)
            times.append(f"{'ok' if d.allowed else 'deny'}:{time.monotonic() - t:.2f}s")
        print(f"ingress {fail_mode:16} {times}", flush=True)
    for fail_mode in ("fallback_memory", "fail_open", "fail_closed"):
        gov = RedisResourceGovernor(policy_loader=_P({"scopes": ["user"], "fail_mode": fail_mode}), ns=f"fu2rg_{uuid.uuid4().hex[:6]}")
        times = []
        for _ in range(3):
            t = time.monotonic()
            d, h = await gov.reserve(RGRequest(entity="user:1", categories={"jobs": {"units": 1, "max_concurrent": 5, "ttl_sec": 120}}, tags={"policy_id": "limits.x"}))
            await gov.renew(h, 120)
            await gov.release(h)
            times.append(f"{'ok' if d.allowed else 'deny'}:{time.monotonic() - t:.2f}s")
        print(f"quota   {fail_mode:16} {times}", flush=True)


asyncio.run(main())
```

```bash
PYTHONPATH=. $PY $SCRATCH/fu2rg_latency.py 2>&1 | grep -E "^(ingress|quota)"
```

Expected (measured; about 60 s):
```text
ingress fallback_memory  ['ok:10.00s', 'ok:0.00s', 'ok:0.00s']
ingress fail_open        ['ok:10.01s', 'ok:0.00s', 'ok:0.00s']
ingress fail_closed      ['deny:10.00s', 'deny:0.00s', 'deny:0.00s']
quota   fallback_memory  ['ok:10.00s', 'ok:0.00s', 'ok:0.00s']
quota   fail_open        ['ok:10.00s', 'ok:0.00s', 'ok:0.00s']
quota   fail_closed      ['ok:10.00s', 'ok:0.00s', 'ok:0.00s']
```

**Notes:**
- Under option (a), each first request reads `5.00s`.
- On `origin/dev`, every request took 30.01 s (`fallback_memory` ingress), 70.01 s (`fail_open` ingress), 10.00 s and 65.01 s (quota).
- If this network answers `10.255.255.1` "unreachable" at once, every line reads `0.00s`. Say so in the PR body instead of quoting it.

- [ ] **Step 4: File the follow-up, under the round's id lock**

Right before the create, rebase and take the next id with the paging pipeline. It is Task 3, Step 1's pipeline, reading `backlog/` instead of `Docs/ADR/`. Check that the id is free.

```bash
git fetch -q origin dev && git rebase origin/dev
M=$({ git ls-tree -r --name-only origin/dev backlog/
  gh pr list --state open --limit 500 --json number,changedFiles,files \
    --jq '.[] | if .changedFiles > 100 then "PR#\(.number)" else .files[].path end' \
    | while IFS= read -r line; do
        if [[ $line == PR#* ]]; then
          gh api --paginate "repos/rmusser01/tldw_server/pulls/${line#PR#}/files" --jq '.[].filename'
        else
          printf '%s\n' "$line"
        fi
      done
} | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1)
F=$((M+1)); echo "$F"; ls backlog/tasks | grep "^task-$F " || echo "TASK-$F is free"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "RG integration fixture rg_unique_ns deletes only the keys on the first SCAN page" --id TASK-$F \
  -d "tests/Resource_Governance/integration/conftest.py cleans a test's namespace with one client.scan(0, match=..., count=1000) call, before and after the test, and never follows the cursor. On a Redis with more than about 1000 keys the namespace's keys are often on a later page, so they stay. Found while planning TASK-13882/TASK-13892 (2026-10-22): the shared local Redis held about 1150 keys, 520 of them rg_it_* keys from past runs (328 namespaces, no TTL). CI's Redis is ephemeral, so this only fills shared developer Redis instances." \
  --ac "rg_unique_ns deletes every key in its namespace whatever the keyspace size (follow the SCAN cursor, for example with scan_iter), with a check against a Redis holding more than 1000 other keys" \
  -l resource-governance,redis,tests
```

**Expected:**
- `echo` prints a number of at least `13901` (the reviewer's run on 2026-10-22), and the check prints `TASK-<F> is free`.
- The create prints `TASK-<F> - RG integration fixture ... (To Do)`.

- [ ] **Step 5: Close both tasks and commit the plan**

The notes quote the measured results. If any step printed different lines, put the actual ones in. Run this in the shell from Steps 3-4 (`N` and `F` set).

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13882 --append-notes "Fixed (AC's second option: RG stops calling Redis for a while after a failure):
- governor_redis.py: _client_get wraps every real client's execute_command (_stop_calling_redis_while_unreachable). A command that fails with redis-py ConnectionError or TimeoutError (refused, reset or timed out) is followed by one PING on a fresh connection; if that fails too, each command raises ConnectionError at once for RG_REDIS_UNREACHABLE_COOLDOWN_SEC (default 10 s, 0 = off), which RG's guards answer by the fail mode. Then one command probes while the rest keep failing fast; any answer, an error reply included, ends the cooldown. The stub is not wrapped.
- RG's client now waits at most 5 s to connect and for each reply (redis-py 8's defaults, pinned so older releases the redis>=4.5.0 pin allows wait no longer); socket_timeout and socket_connect_timeout options in REDIS_URL override them.
- Owner Decision 1: a PING confirms a TimeoutError as well as a ConnectionError.
- The PING is the review's B1 fix. Without it, a connection Redis closed while idle started the cooldown: after an idle spell fail_closed refused every new user with deny(60), where dev admitted them. Measured with CONFIG SET timeout 1 on a private Redis: dev ok x3, without the PING deny(60) x3, with it ok x3.
- Not a shorter timeout: at 0.5 s a fail_open ingress request still waits 14 x 0.5 s, every request. Not the shared Infrastructure CircuitBreaker: it logs a WARNING on every trip for each of the 9 governors, and a registry breaker reads a database on every call.
Measured (black-holed 10.255.255.1, 5 s timeouts): on dev an ingress request took 30 s (fallback_memory) or 70 s (fail_open) and a quota request 10 s or 65 s, every request. Now the first request takes 10.00 s and the next 0.00 s; while Redis stays down, one request per governor waits 10 s every 20 s. RG retries 10 s after the last failed probe, so its fail mode lasts up to about 15 s after Redis returns (measured 14.8 s worst case); under fail_closed a user refused then keeps a Retry-After of up to 60 s, so 429s can last about 75 s after recovery. One 6 s stall: fail_closed admits everyone, as dev (option (a) would have refused every new user for about 10 s). One 11 s stall starts the cooldown.
Tests: 6 tests, 22 cases in test_rg_fail_modes_across_categories.py (unroutable/stopped/silent x 3 modes; a 0 cooldown; the env var, 8 values; the pinned timeouts; recovery after the cooldown, reply and error reply; a connection closed while idle). RED 21 failed, 26 passed (dev sends 28/15/11 commands over 3 reserves, where 2 are expected; no pinned timeouts; the helpers are missing); GREEN 47 passed. 11 mutants each fail a test.
Sweep (tests/Resource_Governance, tests/Usage, Services/test_startup_resource_governor.py, private Redis): 785 passed, 1 skipped, 2 xfailed. Docs 212 passed; test-quality ratchet OK; ruff clean; bandit clean on governor_redis.py.
Docs: ADR-$N amends ADR-078 (when the fail modes apply, the fail_closed bound, the recovery lag, the knob); Env_Vars.md (RG_REDIS_FAIL_MODE and RG_REDIS_UNREACHABLE_COOLDOWN_SEC, mirrored), Rate_Limits_Troubleshooting.md (a fail_closed row), Usage_Quotas.md and the RG README.
Follow-up: TASK-$F (rg_unique_ns leaks keys past the first SCAN page)." --check-ac 1 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "Once a Redis command and a PING right after it both fail, the Redis Resource Governor stops calling Redis for RG_REDIS_UNREACHABLE_COOLDOWN_SEC (10 s by default) and answers by its fail mode, then probes with one command at a time; its client waits at most 5 s per connect or reply. A black-holed Redis now costs the first governed request 10 s and the rest nothing, where each request used to wait 30-70 s; the fail mode lasts up to about 15 s after Redis returns, and fail_closed's 429s up to about 75 s (ADR-$N)." --status Done
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13892 --append-notes "Fixed: when the lease script or the multi-key rate script fails, reserve removes the members it planned for that script, one ZREM per key and best effort, before the Python path counts. They are this reserve's own UUIDs, so removing them is always safe, and a no-op when the script never ran.
Measured on dev (private Redis): with evalsha patched to run the script and then raise redis TimeoutError, a lease at a limit of 1 was refused and the script's lease stayed after release, (allowed, held, after release) == (False, 1, 1); a requests reserve left 2 window members for 1 request. A real lost reply (DEBUG SLEEP 6 landing just before the EVALSHA, 5 s socket timeout) did the same: lease refused (retry_after 60), rate charged twice.
Correction to the description: with a limit of 2 the request never holds two slots, because the Python path re-adds the same member ids. With no other lease it was admitted holding one slot; with one lease held it was refused (retry_after 119). So at a limit of N the request was refused once the user held N-1.
With option (b) of TASK-13882 (a PING confirms a timeout): a reply lost to a stall shorter than two socket timeouts does not start the cooldown, so this ZREM reaches Redis. Measured with the real lost reply: lease and rate admitted and counted once, under fallback_memory and fail_closed. In a longer outage the PING fails too, the ZREM and the Python count both skip Redis, and the request gets its fail mode; the script's members then expire (a rate member with its 60 s window, a lease at its TTL).
With option (a) (one timeout starts the cooldown), the review measured: lease admitted per worker with its script lease left in Redis; rate admitted under fallback_memory, refused under fail_closed (retry_after 60) with its one charge left in Redis. Keep the paragraph for the option the owner chose.
Tests: test_a_lost_lease_script_reply_does_not_count_its_own_lease and test_a_lost_rate_script_reply_charges_the_request_once (real Redis). RED 2 failed, 29 passed; GREEN 31 passed. Each ZREM's mutant fails its test.
Docs: ADR-$N's Follow-up says when this cleanup reaches Redis; ADR-078 is not rewritten (accepted ADRs are not).
Verification: shared with TASK-13882 (same PR): sweep 785 passed, 1 skipped, 2 xfailed; Docs, ratchet, ruff and bandit clean." --check-ac 1 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "When the lease script or the multi-key rate script fails after it may have run, reserve removes the members it planned for that script before counting in Python. A request whose script reply is lost no longer counts or charges itself twice: a user's only slot goes to that request, instead of being refused and blocked for up to two minutes." --status Done
git add -A backlog/tasks
git add Docs/superpowers/plans/2026-10-22-rg-redis-unreachable-cooldown.md
git commit -m "docs(rg): plan for the unreachable-Redis cooldown and lost script replies; close TASK-13882 and TASK-13892, file the follow-up

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Before you run it, delete from TASK-13892's notes the option paragraph (a) or (b) that the owner did not choose. Expected: `git show --stat HEAD` lists:
- the TASK-13882 and TASK-13892 files;
- the follow-up task file;
- the plan.

- [ ] **Step 6: Open the PR, and stop the private Redis**

```bash
git push -u origin fix/rg-redis-unreachable-cooldown
gh pr create --base dev --title "fix(rg): stop calling an unreachable Redis for a cooldown; a lost script reply no longer counts itself (TASK-13882, TASK-13892)" --body-file $SCRATCH/pr_body_fu2rg.md
redis-cli -p $RPORT shutdown nosave
```

Write `$SCRATCH/pr_body_fu2rg.md` from the three fix commits. Include:
- the dev latency table and Step 3's lines for this branch;
- the cycle: "RG retries 10 s after the last failed probe, so the fail mode lasts up to about 15 s after Redis returns; under `fail_closed`, 429s can last about 75 s after recovery";
- the knob, and ADR-<N>;
- why the PING (B1, with the idle-close numbers), Decision 1 and its measurements, and why not a shorter timeout or the shared breaker;
- Review Focus 1-5;
- the follow-up id;
- a `## Change summary` section for the owner to write (ADR-004);
- a last line of `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

Then run `/superpowers:requesting-code-review` in a clean subagent, and fold in its findings before merge. Follow `gh variable get MERGE_QUEUE` for the merge.

## PR split and commits

**One PR**, `fix/rg-redis-unreachable-cooldown` into `dev`. The tasks are not independent enough to split:
- both change `governor_redis.py`;
- ADR-081 and Decision 1 cover how they interact;
- together they are about 90 lines of product code.

Commits:
1. `fix(rg): stop calling an unreachable Redis for a cooldown (TASK-13882)`: source and tests.
2. `fix(rg): a lost script reply no longer counts the reserve's own members (TASK-13892)`: source and tests.
3. `docs(adr): ADR-<N>, the Resource Governor stops calling an unreachable Redis for a cooldown (TASK-13882)`: the ADR, the index, Env_Vars, the troubleshooting page, Usage_Quotas, the RG README, and the mirror.
4. `docs(rg): plan ...; close TASK-13882 and TASK-13892, file the follow-up`: both task files, the new task file, the plan.

## Follow-ups (filed in Task 4, Step 4)

- **TASK-<N>:** `rg_unique_ns` cleans only the first `SCAN` page, so on a shared Redis with more than 1000 keys the integration tests leak their namespaces (520 `rg_it_*` keys found on 2026-10-22).
- **Dropped from r1:** the redis-py timeout follow-up. RG's client now pins 5 s connect and reply timeouts (Task 1).
