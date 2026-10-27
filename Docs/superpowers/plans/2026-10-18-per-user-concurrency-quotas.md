# Per-User Concurrency Quotas (TASK-13435) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Independent review and dry run (2026-10-18): 1 Blocker, 3 Important, 15 Minor; all folded in.

**Goal:** With `USAGE_QUOTAS_ENABLED` on, a platform admin can cap how many media ingests, audio streams and direct transcriptions one user runs at once, per user, team or org; with no value set, nothing is limited.

**Architecture:**
- **The governors learn a per-request limit.** A `streams`/`jobs` entry in `RGRequest.categories` may carry its own `max_concurrent` and `ttl_sec`. They replace the policy's values and charge only the caller's own bucket. One helper, `policy_eval.lease_limits`, decides this for both backends. The Redis backend also stops skipping the release of a lease whose policy has no block for it, and its `renew()` refreshes only the buckets a lease charged (Review M4).
- **One slot module.** `core/Usage/concurrency_quota.py` resolves the user's `limits.*` value, reserves one lease, renews it every 30 s while the request's task runs, and releases it under a shielded cancel scope. Leases live in a process-wide lease store built from `RG_BACKEND`, so on Redis one limit holds across workers. Every lease-store call catches `redis.exceptions.RedisError` as well as the builtin errors, so a Redis outage never fails a request (Review B1).
- **Three sites.** `/media/add` and `/audio/transcriptions` take a slot right before the `try` whose `finally` gives it back; the release is the last statement of that `finally`, so the cleanup before it always runs (Review B1). The six audio stream handlers keep the `can_start_stream`/`finish_stream` hooks; the hooks now hold a real slot, bound to the calling task through a `ContextVar`.

**Tech Stack:** Python 3.12, FastAPI/Starlette 1.6, anyio 4 (`CancelScope(shield=True)`, already used in `endpoints/persona.py`), redis-py asyncio and Lua (already used by the RG Redis backend), pytest (`asyncio_mode = "auto"`), loguru, backlog-py, `gh`.

**Spec:**
- `backlog/tasks/task-13435 - Per-user-synchronous-concurrency-quotas-media-ingest-audio-streams-direct-transcription.md` (on `origin/dev`).
- Parent: `backlog/tasks/task-13434 - Usage-quotas-off-by-default-set-per-user-team-org-spec-2.md`.
- `Docs/Design/2026-10-02-usage-quota-posture-design.md`: Non-goals ("Deferred... per-user concurrency on synchronous paths... which needs an RG per-request limit") and §4 "What becomes unlimited". Review ruling R1 is recorded in `Docs/superpowers/plans/2026-10-03-usage-quotas-pr-b-per-user-values.md`, "Rulings carried into this plan".
- Governing ADRs: `Docs/ADR/064-usage-quotas-per-user-limits.md`, `Docs/ADR/056-resource-governor-safety-net.md`.
- Owner posture: quotas are off unless an admin configures them (`Docs/Design/2026-09-29-rg-ingress-safety-net-design.md`, Out of scope; spec 2 owner decision 4: "Every quota is blank (unlimited) by default, concurrency caps included").

## Decisions for the owner

The plan is written for the recommendation in each case. The alternative's delta is stated so the executor can switch without re-planning.

1. **D1. Does `RG_ENABLED=false` turn these limits off?** **Recommendation: no. They follow `USAGE_QUOTAS_ENABLED` only, like every other `limits.*` quota.**
   - Why: ADR-064 and spec 2 §1 say the usage-quota switch "does not depend on `RG_ENABLED`". Every other per-user limit (minutes, MB/day, workflow runs) works with RG off. Under the alternative, an admin's assigned limit would silently stop applying on an install that turned RG off.
   - Against it (Review M13): ADR-056's rationale is that `RG_ENABLED` is an honest switch, with no governor reachable when it is off, and it explicitly rejected "governor object still exists". The recommendation adds a third governor build site that ignores the switch. With RG off and `RG_BACKEND=redis`, a worker still connects to Redis once a user has a value set. ADR-076 records this cost (Task 6).
   - Cost: ADR-056 and `Env_Vars.md` currently list "media job concurrency" and "audio concurrency" under `RG_ENABLED`, so the new ADR amends ADR-056 (Task 6).
   - Alternative (Review M13): check `rg_enabled()` only where `_lease_governor()` builds the store (inside `if _governor is None:`), return `None` there when it is false, and have `acquire_slot` treat a `None` store as unlimited (about five lines in Task 2). Do not put the check at the top of `_lease_governor()` or in `acquire_slot`: `rg_enabled()` returns `False` under pytest unless `RG_ENABLED` is set (`config.py:2867-2876`), so every enforcement test in Tasks 2–5 would become unlimited. Those tests inject `concurrency_quota._governor`, which skips the build. `test_store_follows_rg_backend_even_with_rg_disabled` then asserts `None`, and the Env_Vars edit in Task 6 is dropped.
2. **D2. Which key limits direct transcription?** The task title names direct transcription; its ACs name only media ingest and audio streams. **Recommendation: its own key, `limits.audio_concurrent_transcriptions`, covering `POST /audio/transcriptions` and `/audio/translations` (translations delegate to transcriptions). Task 0 adds it as AC #3.**
   - Why: reusing `limits.audio_concurrent_jobs` would mix two counters under one key: the audio Jobs worker enforces that key as a count of `processing` rows, while a direct transcription is one open request.
   - History (Review M14): this is not a restored split. Before spec 2, the RG `jobs` lease that `can_start_job` took on this path was also taken by the three audio workers (`1012f9af9a^`: `audio_jobs_worker.py:252`, `audio_transcribe_gpu_worker.py:111`, `audiobook_jobs_worker.py:1251`), so it was never a direct-transcription-only counter. Its 429 said `Concurrent job limit reached`, and spec §4 says a denial keeps the site's existing message. The new message, `Concurrent transcription limit reached (N)`, departs from that on purpose: "job" now means a queued job under `limits.audio_concurrent_jobs`, so the old text would point at the wrong key. The new key and message are a reasonable reading of the task.
   - Alternative (a): count direct transcriptions under `limits.media_concurrent_jobs`. Task 4 then uses `MEDIA_CONCURRENT_KEY`, and the catalog entry and docs rows for the third key are dropped.
   - Alternative (b): leave direct transcription unlimited. Drop Task 4, the third catalog key, AC #3, and the docs rows.
3. **D3. Keep the key name `limits.media_concurrent_jobs`?** **Recommendation: keep it; the task named it** (Review M11).
   - It limits synchronous `/media/add` requests, including its research-discovery handoff, which ingests in one `add_media_persist` call (`research_discovery_handoff.py:141`). It does not limit queued `/media/ingest/jobs` (`endpoints/media/ingest_jobs.py:569`) or `/media/process-*`. Next to `limits.audio_concurrent_jobs`, which counts queued jobs, the name reads the other way, so the catalog description (Task 2), ADR-076 and `Usage_Quotas.md` (Task 6) say what it does not cover.
   - Alternative: rename it now, before any admin sets a value, for example to `limits.media_concurrent_ingests`. Replace the key in Task 2 (catalog, `NEW_KEYS`, the test's `KEY`), Task 5 (`MEDIA_CONCURRENT_KEY` and the `one_ingest_at_a_time` fixture) and Task 6 (ADR-076 and the docs).

None of these decisions blocks Tasks 1–2. Record the answers in the task notes in Task 0.

**Owner answers (2026-10-18):** D1, D2 and D3 as recommended; the new ADR is accepted.

## What was measured, and what it changed (origin/dev `32eb11ae03`, 2026-10-18)

Measured in a detached worktree of `origin/dev` at `32eb11ae03`, with a second clean worktree at the same commit for baselines. The first draft was measured at `e1f7770953`; PR #3243 merged in between and changed none of this plan's files, but it made the test-quality ratchet a required check (Review I1). A prototype of Tasks 1–6 was built from this plan's code blocks, and every test in this plan was run RED on unchanged `origin/dev` and GREEN on the prototype. Line numbers below are on `32eb11ae03`.

- **Nothing reserves a concurrency lease today.**
  - `git grep '"jobs": {"units"' origin/dev -- 'tldw_Server_API/app/*.py'` finds nothing. PR B removed the media reserve (commit `c03a94fc93`) and the audio RG handle machinery (`47f6ba73c0`).
  - `can_start_job`, `finish_job`, `can_start_stream`, `finish_stream`, `heartbeat_stream` and `heartbeat_jobs` are no-ops (`core/Usage/audio_quota.py:762-780, 838-880`).
- **`RGRequest` can't carry a limit, and the governors read it only from the policy.**
  - `RGRequest` (`Resource_Governance/governor.py:53-57`) has `entity`, `categories` and `tags`.
  - Memory reads `max_concurrent`/`ttl_sec` from the policy block (`governor.py:408-410` and `665-667`); Redis does the same (`governor_redis.py:1093-1094`, `1846-1847`, `1963-1965`).
  - A missing block means unbounded. With no block, `ttl_sec` defaults to 60 s, so a long ingest or stream would stop counting after a minute.
- **The Redis backend would leak every such lease.** `RedisResourceGovernor.commit` (which `release` calls) skips any concurrency category the policy lacks (`governor_redis.py:2147-2152`). A lease reserved with only a request limit would never be released; it would just lapse.
  - Measured on the prototype: reverting that one line makes `test_request_limit_applies_without_a_policy_block[redis]` and the real-Redis cross-worker test fail on the post-release reserve.
- **The Redis `renew()` would charge `global` (Review M4).** It walks `self._scope_pairs(pol, ...)` (`governor_redis.py:2403-2405`) and, for a bucket the handle never charged, ZADDs a fallback member `"{handle_id}:{sc}:{ev}"` (`2413-2414`). A request-limited lease charged only the caller's bucket, so renewing it would add it to the policy's `global` bucket. Measured: with a policy of `jobs.max_concurrent: 1` and scopes `[global, user]`, renewing user 1's request-limited lease made user 2's policy-limited reserve fail on Redis, not on memory (memory's `renew()` only touches leases that exist, `governor.py:837-852`).
- **Scope and bucket keys.**
  - `scope_pairs` (`policy_eval.py:104-118`) adds a `("global", "*")` bucket when a policy lists `global`. A per-user limit must not charge it.
  - Lease buckets key on `(policy_id, category, scope, entity)` (`governor.py:239-240`; Redis `_RedisKeys.lease`, `governor_redis.py:95-96`). Two quotas that shared a policy id and category would share one count, so each quota key is its own policy id.
- **RG off means no app governor.** `app.state.rg_governor` is `None` when `RG_ENABLED` is false (`services/startup_resource_governor.py:100-103`). This drives D1 and the dedicated lease store.
- **Two release gaps in existing code become real once slots are real.**
  - `websocket_transcribe` takes its slot at `audio_streaming.py:1481`, 312 lines before the `try:` (`1793`) whose `finally:` (`1837`) gives it back. Measured: a setup error in between (the new test patches `_get_failopen_cap_minutes` to raise) gives `started=1, finished=0`.
  - `create_transcription` releases its job slot only on the minutes-denial path (`audio_transcriptions.py:912`) and in an inner `finally:` around the provider call (`1117-1124`). Upload (413), media-type and WAV-conversion errors never reach either.
  - The other five stream handlers already take their slot inside the `try` whose `finally` releases it: `POST /audio/chat` `1210-1234`, `/chat/stream` `2061-2076`/`3671-3690`, `/stream/tts` `3798-3809`/`3942-3945`, `/stream/tts/realtime` `4118-4125`/`4401-4429`, and `core/Audio/Realtime/handler.py:52-90`.
- **redis-py's errors are not the builtins (Review B1).** `redis.exceptions.ConnectionError.__mro__` is `(redis.exceptions.ConnectionError, redis.exceptions.RedisError, Exception, ...)`: it is not a builtin `ConnectionError`, `TimeoutError` or `OSError`. `RedisResourceGovernor` uses a raw `redis.asyncio` client, so once a worker has connected, a Redis outage surfaces as `RedisError` out of `reserve`, `renew` and `commit`. RG's own `_RG_NONCRITICAL_EXCEPTIONS` (`governor_redis.py:29-40`) lists only builtins, so its "best-effort" guards miss it too; Task 7 files that as a follow-up. The reviewer's probes: `acquire_slot` raised `redis.exceptions.ConnectionError` to the caller, and so did `slot.release()`.
- **The job heartbeat leaks on early exits (Review I3).** `create_transcription` starts a `while True: sleep(interval)` task (`_hb_loop`, `audio_transcriptions.py:618-629`, started at `660-663`) before the slot is taken. Only the minutes-denial path (`921-924`) and the inner provider `finally` (`1117-1122`) cancel it. Measured: three refused requests left three `_hb_loop` tasks running, and three 413 requests did the same.
- **One harness pins that offline email ingestion never touches AuthNZ (Review I2).** `tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py::test_offline_upload_does_not_enter_authnz_pool` sets `USAGE_QUOTAS_ENABLED=true` and asserts `get_db_pool` is never called. With the media slot in place, `acquire_slot` → `user_quota` → `_load_limits` calls it once. Its `offline_client` fixture (line 47) already keeps every other quota path out ("Quota remains outside this native parser/persistence harness", line 124).
- **A timing-sensitive WebSocket test feels the slot lookup.** `tests/Audio/test_ws_idle_metrics_audio.py` sets a 0.1 s idle timeout and sleeps 0.25 s. With the session's quotas on (`tests/conftest.py:45`), the moved `/stream/transcribe` acquire now makes one resolver lookup before the idle loop starts: measured at 45–80 ms, and in a failing run it had not finished when the client disconnected. It failed 2 of 15 prototype runs and 0 of 6 runs on clean `origin/dev`; with `concurrency_quota.user_quota` stubbed in that test, 0 of 10. Task 3 stubs it, as I2 does for the email harness.
- **The test-quality ratchet is a required check (Review I1).** `tests/CI/test_test_quality_ratchet.py` runs `Helper_Scripts/ci/test_quality_triage.py --enforce` in `backend-required` (wired by #3243). It fails a test whose asserts touch only `.status_code` (`status_only`) or only a collector its own fakes append to (`tautology_suspect`). The first draft's tests added three such offenses; the revised tests add none (measured: `[test-triage] OK — no offenses beyond the baseline.`).
- **`can_start_job` can't carry the transcription limit.** Three workers also call `can_start_job`/`finish_job`: `services/audio_jobs_worker.py:296`, `services/audio_transcribe_gpu_worker.py:111` and `services/audiobook_jobs_worker.py:1251`. Queued jobs are limited separately, by `limits.audio_concurrent_jobs`.
- **Existing release tests use the user-keyed hooks.** These patch `can_start_stream`/`finish_stream` and assert on them, so keeping those hook names keeps them valid:
  - `tests/Audio/test_realtime_governance.py` (`test_admitted_disconnect_releases_slot`, `test_admitted_stream_releases_slot_on_factory_failure`);
  - `tests/Audio/test_ws_tts_endpoint.py::test_websocket_tts_disconnect_releases_stream_slot`;
  - `tests/Audio/test_ws_concurrent_streams.py` (denial message `Concurrent streams limit reached (1)`).
- **ADRs.** ADR-064 lists "synchronous concurrency" under "Unchanged guardrails... not usage quotas". ADR-056 and `Docs/Operations/Env_Vars.md:341` list "media job concurrency" and "evaluations/audio concurrency" as paths `RG_ENABLED` turns off.
- **Test environment.**
  - A real Redis answers at `127.0.0.1:6379`, and `tests/Resource_Governance/conftest.py` defaults `REDIS_URL` to it. The `[redis]` tests and `integration/test_redis_*` ran against real Redis, not the stub.
  - RG suite on the prototype after Task 1: `421 passed, 2 xfailed`.
  - Audio + AudioJobs + Usage on the prototype after Task 3: one failure that also fails on clean `origin/dev` (`tests/Audio/test_nemo_streaming_error_sentinel.py`, `ModuleNotFoundError: No module named 'pyaudio'`), and the idle-timeout flake above until Task 3 stubs it.

## The shape chosen, and why it differs from the task's sketch

The task suggests "an optional `RGRequest.max_concurrent` honored by the memory and Redis governors". This plan keeps the idea and changes its shape:

| Task sketch | This plan | Why |
|---|---|---|
| `RGRequest.max_concurrent: int \| None` | `RGRequest(categories={"jobs": {"units": 1, "max_concurrent": N, "ttl_sec": T}})` | A request can carry several categories; one request-level value is ambiguous. The lease TTL needs a home too, because the policies no longer have `jobs`/`streams` blocks and the 60 s default would drop long requests. Policies already spell the same limit as `jobs: {max_concurrent, ttl_sec}`, and `categories` is already `dict[str, dict[str, int]]`, so no dataclass change is needed. |
| Use the app's governor | A dedicated lease store per process, built from `RG_BACKEND` with a stub policy loader | `app.state.rg_governor` is `None` with RG off (D1). The operator's policy store could add a `global` bucket. Unknown policy ids log `policy ... is not defined` on that governor. |
| (not specified) | Each `limits.*` key is the lease policy id | It gives each quota its own bucket. |
| (not specified) | Renew every 30 s while the owning task runs; TTL 120 s | It holds a slot for a long ingest or stream, and bounds the lockout after a crash or a missed release to 2 minutes. |

Rejected outright:
- **A limit encoded in the policy id (`key:limit`).** It needs no governor change, but changing a user's limit opens a new, empty bucket.
- **A new lease store outside RG.** It duplicates the Lua acquire, the TTL purge and the retry-after logic.

## Lease release on every exit path

| Exit path | What releases the lease | Test that pins it |
|---|---|---|
| Success | The site's `finally`, whose last statement is `slot.release()`, so the file cleanup and metrics before it always run (Review B1) | `test_a_finished_ingest_gives_its_slot_back` (Task 5); `test_limit_is_enforced_and_release_frees_a_slot` (Task 2) |
| Error (HTTPException or another exception) | The same `finally` | `test_a_rejected_ingest_gives_its_slot_back` (Task 5); `test_a_failed_request_gives_its_slot_back` (Task 4); `test_setup_failure_after_admission_leaves_no_slot_held` (Task 3) |
| Client disconnect or task cancellation | `CancelledError` (or `WebSocketDisconnect`) runs the `finally`. `release()` runs under `CancelScope(shield=True)`, so anyio's level cancellation can't abort it. | `test_cancelled_request_gives_its_slot_back`, `test_release_completes_inside_a_cancelled_scope` (Task 2); the media cancel test (Task 5); the existing WebSocket disconnect tests above (Task 3) |
| Graceful shutdown | Uvicorn closes WebSockets and cancels in-flight tasks, which is the same path as cancellation. Memory-backend leases end with the process. | Same as cancellation |
| Hard kill (SIGKILL, OOM, crash) | Nothing runs. A Redis lease lapses `LEASE_TTL_SEC` (120 s) after its last renewal; a memory lease dies with the process. | `test_request_ttl_lets_an_unreleased_lease_lapse[memory\|redis]` (Task 1) |
| A missed release (a bug) | The keep-alive stops renewing once the owning task ends, so the lease lapses within 120 s. | `test_keep_alive_renews_until_the_owner_ends` (Task 2) |
| Cancellation inside `acquire_slot` after the store granted the lease (Review M15) | Nothing: the handle reaches no caller and no keep-alive was started, so the lease lapses within 120 s. | `test_request_ttl_lets_an_unreleased_lease_lapse[memory\|redis]` (Task 1) |
| Redis drops while the slot is held | redis-py raises its own `ConnectionError`, a `RedisError`, not the builtin. `_keep_alive` and `release()` catch it, log, and carry on; the lease lapses within 120 s. The request's own result stands (Review B1). | `test_a_redis_outage_never_escapes_renew_or_release` (Task 2) |

## Global Constraints

- **Branch and PR:** branch `feat/per-user-concurrency-quotas` from `origin/dev`; one PR into `dev`. Never `git stash`; never `--no-verify`.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- **Backlog:**
  - Edit only through backlog-py: `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13435 ...`.
  - Use `--append-notes`, never `--notes`.
  - Stage with `git add -A backlog/tasks`.
- **Python** is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`.
  - Run tests as `TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q <paths>`.
  - The Redis tests are real only when a Redis answers at `REDIS_URL`; `tests/Resource_Governance/conftest.py` defaults it to `redis://127.0.0.1:6379`. Without one, `[redis]` cases run on the in-memory stub and `integration/test_redis_*` skip. Record which happened.
- **A failure counts as environmental** only if it also fails on a clean detached `origin/dev` checkout. Known: `tests/Audio/test_nemo_streaming_error_sentinel.py` (no `pyaudio`).
- **The test session runs with quotas ON:** `tldw_Server_API/tests/conftest.py:45` sets `LIMIT_ENFORCEMENT_ENABLED=true`. New tests patch `concurrency_quota.user_quota` rather than relying on the DB. So do the two existing tests the new resolver lookup would disturb: the offline email harness (Task 5, Review I2) and the idle-timeout metric test (Task 3).
- **Fakes that can't fail:** never assert inside a fake that a broad `except` would swallow. Use call-recording fakes and assert on the record. A fake that stands in for a Redis failure raises `redis.exceptions.ConnectionError`, never the builtin `ConnectionError` (Review B1).
- **Lease-store errors:** every lease-store call in `concurrency_quota` catches `_NONCRITICAL`, which includes `redis.exceptions.RedisError` (imported where redis is available). Never catch bare `Exception` (Review B1).
- **Test-quality ratchet:** `Helper_Scripts/ci/test_quality_triage.py --enforce` is a required check (since #3243). Every new test asserts on real output or real state, not only on a status code or on a collector its own fake fills. Never raise `Helper_Scripts/ci/test_quality_baseline.txt` (Review I1).
- **Keys:** every new `limits.*` key has `default: null`, `minimum: 0`, `editable_by: [platform_admin]`, type integer. Unset means unlimited at no cost; `0` blocks.
- **Placement rule at every site:** take the slot directly before the `try:` whose `finally:` releases it, with nothing between them that can raise. Make `slot.release()` the last statement of that `finally:`, so the cleanup before it always runs (Review B1).
- **Denials keep each site's status and shape:** 429 with `RateLimit-*`/`Retry-After` headers (from `quota_checks.rate_limit_headers`) on HTTP; the existing error frame and close on WebSockets.
- **No new dependencies.** anyio ships with Starlette.
- **Mirrored docs:** any edit to `Docs/ADR/*`, `Docs/Operations/Env_Vars.md` or `Docs/API-related/*` needs `bash Helper_Scripts/refresh_docs_published.sh`, with the `Docs/Published` diff in the same commit. Stage new mirrored files before running `tests/Docs` (Review M2).
- **Line-number baselines:** none of the touched files is in `Helper_Scripts/ci/scope_predicate_baseline.txt` (checked on `32eb11ae03`). `test_email_offline_ingestion.py` is in `test_quality_baseline.txt` (`mock_density=1`); the fixture edit adds no test function, and the ratchet passes.

## Review Focus

These are the input classes most likely to bite a user that the main tests don't cover. Each has a pinning test, named at the end of its line, in the task that owns the code.

1. **A UI that fires parallel uploads or streams for one user** must get exactly `limit` admissions, never more, on both backends. Task 1, `test_simultaneous_requests_get_exactly_the_limit[memory|redis]`; Task 2, `test_parallel_requests_get_exactly_the_limit`.
2. **An admin lowers a user's limit while their requests run.** Running requests finish; new ones are refused until usage is under the new limit, and then admitted. Nobody is stuck. Task 2, `test_lowering_the_limit_refuses_new_requests_until_usage_drops`.
3. **Quotas switched off, or the value removed, while slots are held:** the next request is admitted at once. Task 2, `test_switching_quotas_off_admits_at_once`.
4. **Redis drops after the worker connected:** redis-py raises `redis.exceptions.ConnectionError`, which is not the builtin. A reserve that fails admits the request; renewals keep trying; the release logs a warning; the request's own response is unaffected; the lease lapses within 120 s. Task 2, `test_store_error_admits_the_request` and `test_a_redis_outage_never_escapes_renew_or_release`, both raising redis-py's error (Review B1).
5. **A principal with no numeric user id** (an API-key principal, or `None`) is never limited and never crashes the request, even with a value of 0 set. Task 2, `test_a_non_numeric_user_is_unlimited`.
6. **A client at its limit that keeps retrying** must not leave work behind on each refusal. A refused or failed transcription leaves no `_hb_loop` heartbeat task running. Task 4, `test_refused_with_429_while_the_users_slot_is_held` and `test_a_failed_request_gives_its_slot_back` (Review I3).

Covered elsewhere or out of reach for a unit test, and written into the docs instead:
- **Memory backend with N workers** admits up to N × the limit.
- **An event loop blocked for more than 120 s** lets a lease lapse, which under-counts by one. That is the safe direction.
- **Redis unreachable when a worker first connects:** the lease store falls back to the in-memory stub for the life of that process, so the limit holds per worker, as on the memory backend (Review M7).
- **`Retry-After` is the lease TTL, not when a slot frees:** leases renew, so a refusal's `Retry-After` is about 119–120 s although a slot usually frees in seconds. Documented, not changed (Review M8).

## ADR assessment

- **ADR required: yes. ADR path: `Docs/ADR/076-per-user-concurrency-quotas.md`** (the number Task 0 computed: on 2026-10-18 the highest claimed is ADR-075, by open PR #3270, with every open PR's files paged).
- Why: the change makes synchronous concurrency a usage quota, which ADR-064's Consequences ("synchronous concurrency are not usage quotas and are not touched") rule out. It also moves media and audio concurrency out from under `RG_ENABLED` (ADR-056).
- **A partial amendment, by precedent (Review M12).** `Docs/ADR/README.md` rule 8 says not to rewrite accepted rationale and, read literally, prescribes supersession for a changed decision. Supersession would retire all of ADR-064 and ADR-056, though only one consequence of each changes, and there is no "amended" status. The repo precedent is ADR-056 (`**Amends:** ADR-018 and ADR-044`): a new ADR with an `**Amends:**` line, and the old records' text left alone. This plan follows it. The `Docs/ADR/README.md` index rows for ADR-056 and ADR-064 gain a forward pointer ("Amended in part by ADR-076"); the index is not ADR rationale.
- `Status: Accepted`, dated 2026-10-18: the owner approved D1, D2 and D3 as recommended and accepted this ADR with the plan, in both the ADR and its index row.

## File map

| File | Change | Task |
|---|---|---|
| `tldw_Server_API/app/core/Resource_Governance/policy_eval.py` | Add `lease_limits` | 1 |
| `tldw_Server_API/app/core/Resource_Governance/governor.py` | Memory: request limit in `check`/`reserve` | 1 |
| `tldw_Server_API/app/core/Resource_Governance/governor_redis.py` | Redis: request limit in `check`/`reserve`/denial details; `commit` releases block-less leases; `renew` refreshes only the charged buckets (Review M4) | 1 |
| `tldw_Server_API/app/core/Resource_Governance/README.md` | One bullet | 1 |
| `tldw_Server_API/tests/Resource_Governance/test_request_concurrency_limit.py` | New | 1 |
| `tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py` | New | 1 |
| `tldw_Server_API/app/core/Usage/concurrency_quota.py` | New | 2 |
| `tldw_Server_API/Config_Files/user_profile_catalog.yaml` | Three keys | 2 |
| `tldw_Server_API/tests/Usage/test_concurrency_quota.py` | New | 2 |
| `tldw_Server_API/tests/UserProfile/test_user_profile_limits.py` | `NEW_KEYS` | 2 |
| `tldw_Server_API/app/core/Usage/audio_quota.py` | Stream hooks hold slots; `concurrent_streams` reported | 3 |
| `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py` | Move the `/stream/transcribe` acquire down to its `try`; `/stream/limits` docstring and comment (Review M5) | 3 |
| `tldw_Server_API/app/api/v1/schemas/audio_schemas.py` | `StreamingLimitsResponse` field descriptions (Review M5) | 3 |
| `tldw_Server_API/tests/Usage/test_audio_stream_slots.py` | New | 3 |
| `tldw_Server_API/tests/Audio/test_ws_transcribe_stream_slot_release.py` | New | 3 |
| `tldw_Server_API/tests/Usage/test_audio_quota_per_user.py` | Fixture, one expectation, one test rewritten | 3 |
| `tldw_Server_API/tests/Audio/test_ws_idle_metrics_audio.py` | Stub the slot lookup (re-measured on `32eb11ae03`) | 3 |
| `tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py` | Slot around the outer `try`; the heartbeat is cancelled on the 429 and in the outer `finally` (Review I3) | 4 |
| `tldw_Server_API/tests/Audio/test_transcription_concurrency_slot.py` | New | 4 |
| `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py` | Slot around the `add_media_orchestrate` `try` | 5 |
| `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_cancellation_and_quota.py` | Three tests and a fixture | 5 |
| `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py` | Stub the slot lookup in `offline_client` (Review I2) | 5 |
| `Docs/ADR/076-per-user-concurrency-quotas.md`, `Docs/ADR/README.md`, `Docs/Operations/Usage_Quotas.md`, `Docs/Operations/Env_Vars.md`, `Docs/Operations/Rate_Limits_Troubleshooting.md`, `Docs/API-related/Audio_Transcription_API.md` (Review M5), `Docs/Published/**` | Docs | 6 |
| `backlog/tasks/task-<N> - ...` | Follow-up: RG's `_RG_NONCRITICAL_EXCEPTIONS` misses `RedisError` (Review B1) | 7 |

---

## PR — Per-user concurrency quotas (one PR)

It is one PR because every task after Task 2 depends on Tasks 1–2, and the whole change is about 310 added lines of app code and config and 760 of tests (measured on the prototype). A split would ship either inert machinery or a half-documented quota.

### Task 0: Branch, plan, ledger

**Files:** `Docs/superpowers/plans/2026-10-18-per-user-concurrency-quotas.md` (this plan), `backlog/tasks/task-13435 ...`.

- [ ] **Step 1: Branch**

From the repository root (or a fresh worktree of it), run every command in this plan:

```bash
git fetch origin dev --quiet
git checkout -b feat/per-user-concurrency-quotas origin/dev
```

- [ ] **Step 2: Compute the ADR number**

`gh pr list --json files` returns at most 100 files per PR, and nine open PRs are larger, so page through each PR's files (Review B1, recipe note):

```bash
{ git ls-tree -r --name-only origin/dev Docs/ADR
  for n in $(gh pr list --state open --limit 500 --json number --jq '.[].number'); do
    gh api --paginate "repos/rmusser01/tldw_server/pulls/$n/files" --jq '.[].filename'
  done
} | sed -nE 's|^Docs/ADR/([0-9]{3})-.*|\1|p' | sort -n | tail -1
```

Measured at execution: `075` (on 2026-10-18, claimed by open PR #3270; #3226 holds 072-074), so the ADR is 076, used throughout this plan's code and docs.

- [ ] **Step 3: Commit the plan and move the task to In Progress**

Copy this plan to `Docs/superpowers/plans/2026-10-18-per-user-concurrency-quotas.md`. Then run the command below, filling in the owner's answers to D1, D2 and D3:

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13435 \
  -s "In Progress" \
  --ac "A per-user concurrency limit on direct transcription (POST /audio/transcriptions and /audio/translations) is enforced on both RG backends" \
  --ac "A held slot is released on success, error, client disconnect or cancellation, and graceful shutdown; a lease nobody releases lapses within 120 s" \
  --append-notes "Plan: Docs/superpowers/plans/2026-10-18-per-user-concurrency-quotas.md (one PR). ADR check: ADR required: yes, ADR path: Docs/ADR/076-per-user-concurrency-quotas.md. The change makes synchronous concurrency a usage quota, which ADR-064's consequences ruled out, and takes media/audio concurrency out from under RG_ENABLED (ADR-056); a partial amendment by the ADR-056 precedent. Owner decisions: D1 (RG_ENABLED does not turn them off) = <answer>; D2 (direct transcription key limits.audio_concurrent_transcriptions) = <answer>; D3 (keep the name limits.media_concurrent_jobs) = <answer>."
git add -A backlog/tasks Docs/superpowers/plans/2026-10-18-per-user-concurrency-quotas.md
git commit -m "docs(plan): per-user concurrency quotas; TASK-13435 in progress

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

If D2 is answered (b), drop the first `--ac`. Expected: one commit, and the task file shows ACs #3 and #4 with status In Progress.

### Task 1: The governors honor a per-request concurrency limit (memory and Redis)

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/policy_eval.py` (after `scope_pairs`, which ends at line 118)
- Modify: `tldw_Server_API/app/core/Resource_Governance/governor.py:32, 397-414, 507-516, 664-669`
- Modify: `tldw_Server_API/app/core/Resource_Governance/governor_redis.py:24, 1092-1107, 1840-1853, 1958-1966, 2147-2152, 2402-2405`
- Modify: `tldw_Server_API/app/core/Resource_Governance/README.md:120`
- Test: `tldw_Server_API/tests/Resource_Governance/test_request_concurrency_limit.py` (new)
- Test: `tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `policy_eval.lease_limits(policy: Mapping[str, Any], category: str, request_cfg: Mapping[str, Any], entity_scope: str, entity_value: str) -> tuple[int, int, list[tuple[str, str]]]`, returning `(limit, ttl_sec, buckets)`.
  - The contract: a `streams` or `jobs` entry in `RGRequest.categories` may carry `max_concurrent: int` and `ttl_sec: int`. They replace the policy's values and charge only `(entity_scope, entity_value)`.
  - `release()`/`commit()` on either backend frees such a lease even when the policy has no block for the category.
  - `renew()` on either backend refreshes only the buckets the reservation charged; renewing a request-limited lease never adds it to `global` (Review M4).

Untouched on purpose: the Redis early-deny and rate-rollback paths that rebuild `per_category` for `streams`/`jobs` (`governor_redis.py:1393-1396`, `1685-1688`, `1793-1796`). They run only when a `requests` or `tokens` category was denied, and quota reservations carry neither.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Resource_Governance/test_request_concurrency_limit.py`. Each test takes `redis_ns`, a namespace of its own whose keys the fixture deletes afterwards, as `integration/conftest.py::rg_unique_ns` does; that fixture can't be used here because it skips without real Redis (Review M10). `test_renewing_a_request_limited_lease_never_charges_global` pins the Redis `renew()` fix (Review M4).

```python
"""A request that carries its own max_concurrent is held to it on both backends (TASK-13435)."""

import asyncio
import uuid

import pytest
from redis.exceptions import RedisError

from tldw_Server_API.app.core.Infrastructure.redis_factory import create_async_redis_client
from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor, RedisResourceGovernor, RGRequest

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]


class FakeTime:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        """Start at t=1000 (the Redis governor wipes leases once when the clock reads < 1)."""
        self.t = 1000.0

    def __call__(self) -> float:
        """Now."""
        return self.t


class _Loader:
    """Serve one policy for every id."""

    def __init__(self, policy: dict) -> None:
        """Keep the policy."""
        self.policy = policy

    def get_policy(self, _policy_id: str) -> dict:
        """The same policy for any id."""
        return dict(self.policy)


@pytest.fixture()
async def redis_ns():
    """A Redis namespace of its own, whose keys are deleted after the test, as rg_unique_ns does."""
    ns = f"rg_rcl_{uuid.uuid4().hex[:8]}"
    yield ns
    try:
        client = await create_async_redis_client(fallback_to_fake=False, context="rg_rcl_cleanup")
    except (OSError, RedisError):
        return  # No real Redis: the [redis] cases ran on the in-memory stub.
    async for key in client.scan_iter(match=f"{ns}:*"):
        await client.delete(key)
    await client.aclose()


def _governor(backend: str, policy: dict, clock: FakeTime, ns: str):
    """A fresh governor of the given backend; Redis uses the test's own namespace."""
    if backend == "memory":
        return MemoryResourceGovernor(policy_loader=_Loader(policy), time_source=clock)
    return RedisResourceGovernor(policy_loader=_Loader(policy), time_source=clock, ns=ns)


def _lease(entity: str, limit: int, ttl_sec: int = 120) -> RGRequest:
    """One jobs lease, held to the request's own limit and TTL."""
    return RGRequest(
        entity=entity,
        categories={"jobs": {"units": 1, "max_concurrent": limit, "ttl_sec": ttl_sec}},
        tags={"policy_id": "limits.test_concurrent"},
    )


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_request_limit_applies_without_a_policy_block(backend: str, redis_ns: str) -> None:
    """A policy with no jobs block still holds a request to the request's own max_concurrent; release frees a slot."""
    rg = _governor(backend, {"scopes": ["user"]}, FakeTime(), redis_ns)
    d1, h1 = await rg.reserve(_lease("user:1", 2))
    d2, h2 = await rg.reserve(_lease("user:1", 2))
    d3, h3 = await rg.reserve(_lease("user:1", 2))
    assert (d1.allowed, d2.allowed, d3.allowed) == (True, True, False)
    assert h3 is None
    await rg.release(h1)
    d4, h4 = await rg.reserve(_lease("user:1", 2))
    assert d4.allowed and h4


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_request_limit_charges_only_the_callers_bucket(backend: str, redis_ns: str) -> None:
    """A request's own limit is that caller's: another user's leases and the policy's global bucket don't count."""
    rg = _governor(backend, {"jobs": {"max_concurrent": 1}, "scopes": ["global", "user"]}, FakeTime(), redis_ns)
    d1, _h1 = await rg.reserve(_lease("user:1", 1))
    d2, _h2 = await rg.reserve(_lease("user:2", 1))
    assert d1.allowed and d2.allowed


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_renewing_a_request_limited_lease_never_charges_global(backend: str, redis_ns: str) -> None:
    """Renewing user 1's request-limited lease leaves the policy's global bucket free for user 2's policy-limited lease."""
    rg = _governor(backend, {"jobs": {"max_concurrent": 1}, "scopes": ["global", "user"]}, FakeTime(), redis_ns)
    _d1, h1 = await rg.reserve(_lease("user:1", 1))
    await rg.renew(h1, 120)
    policy_limited = RGRequest(entity="user:2", categories={"jobs": {"units": 1}}, tags={"policy_id": "limits.test_concurrent"})
    d2, h2 = await rg.reserve(policy_limited)
    assert d2.allowed and h2


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_request_ttl_lets_an_unreleased_lease_lapse(backend: str, redis_ns: str) -> None:
    """A lease that is never released lapses after the request's ttl_sec, then the slot is free again."""
    clock = FakeTime()
    rg = _governor(backend, {"scopes": ["user"]}, clock, redis_ns)
    d1, _h1 = await rg.reserve(_lease("user:1", 1, ttl_sec=30))
    d2, _h2 = await rg.reserve(_lease("user:1", 1, ttl_sec=30))
    clock.t += 31
    d3, h3 = await rg.reserve(_lease("user:1", 1, ttl_sec=30))
    assert (d1.allowed, d2.allowed, d3.allowed) == (True, False, True)
    assert h3


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_simultaneous_requests_get_exactly_the_limit(backend: str, redis_ns: str) -> None:
    """Six requests racing for a limit of 2 (a UI firing parallel uploads): exactly two are admitted."""
    rg = _governor(backend, {"scopes": ["user"]}, FakeTime(), redis_ns)
    results = await asyncio.gather(*(rg.reserve(_lease("user:1", 2)) for _ in range(6)))
    assert sum(1 for decision, _handle in results if decision.allowed) == 2


@pytest.mark.parametrize("backend", ["memory", "redis"])
async def test_renew_keeps_a_lease_past_its_first_ttl(backend: str, redis_ns: str) -> None:
    """Renewing a held lease pushes its expiry out, so the limit still holds after the first ttl_sec."""
    clock = FakeTime()
    rg = _governor(backend, {"scopes": ["user"]}, clock, redis_ns)
    _d1, h1 = await rg.reserve(_lease("user:1", 1, ttl_sec=30))
    clock.t += 20
    await rg.renew(h1, 30)
    clock.t += 20
    d2, h2 = await rg.reserve(_lease("user:1", 1, ttl_sec=30))
    assert not d2.allowed and h2 is None
```

`tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py`:

```python
"""On real Redis a request's own max_concurrent holds across workers (TASK-13435)."""

import pytest

from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest

pytestmark = pytest.mark.rate_limit


class _NoJobsBlock:
    """A policy without a jobs block, as the usage-quota lease store uses."""

    def get_policy(self, _policy_id: str) -> dict:
        """The same block-less policy for any id."""
        return {"scopes": ["user"]}


def _lease() -> RGRequest:
    """One jobs lease held to a limit of 1."""
    return RGRequest(
        entity="user:42",
        categories={"jobs": {"units": 1, "max_concurrent": 1, "ttl_sec": 60}},
        tags={"policy_id": "limits.test_concurrent"},
    )


async def test_request_limit_holds_across_two_workers(real_redis, rg_unique_ns) -> None:
    """Two governors on one Redis namespace (two workers) share the user's limit; a release frees it for the other."""
    worker_a = RedisResourceGovernor(policy_loader=_NoJobsBlock(), ns=rg_unique_ns)
    worker_b = RedisResourceGovernor(policy_loader=_NoJobsBlock(), ns=rg_unique_ns)
    if not await worker_a._is_real_redis():
        pytest.skip("Redis client is not real; using in-memory stub")
    d1, h1 = await worker_a.reserve(_lease())
    d2, h2 = await worker_b.reserve(_lease())
    assert d1.allowed and h1
    assert not d2.allowed and h2 is None
    await worker_a.release(h1)
    d3, h3 = await worker_b.reserve(_lease())
    assert d3.allowed and h3
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_request_concurrency_limit.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py`

Expected (measured on `origin/dev` `32eb11ae03` with real Redis): `13 failed`. Every test fails because the request limit is ignored:
- `assert (True, True, True) == (True, True, False)` (three admitted at a limit of 2);
- `assert (True and False)` (the policy's global bucket refused user 2);
- `assert (False)` (user 1's lease filled the global bucket, so user 2's policy-limited lease was refused);
- `assert (not True)` (the renew test admitted past the limit);
- the real-Redis test admits the second worker.

Without a real Redis the last test skips: `12 failed, 1 skipped`.

- [ ] **Step 3: Implement**

In `policy_eval.py`, add after `scope_pairs` (after line 118):

```python
def lease_limits(
    policy: Mapping[str, Any],
    category: str,
    request_cfg: Mapping[str, Any],
    entity_scope: str,
    entity_value: str,
) -> tuple[int, int, list[tuple[str, str]]]:
    """Return ``(limit, ttl_sec, buckets)`` for a streams/jobs reservation.

    A request that carries its own ``max_concurrent`` (a per-user usage quota,
    TASK-13435) uses that limit and its own ``ttl_sec``, and charges only the
    caller's bucket: the limit is that caller's, not the policy's. Otherwise the
    policy's category block decides, as before. A limit of 0 is unbounded.
    """
    own = request_cfg.get("max_concurrent")
    if own is not None:
        return int(own), int(request_cfg.get("ttl_sec") or 60), [(entity_scope, entity_value)]
    cfg = policy.get(category) or {}
    return int(cfg.get("max_concurrent") or 0), int(cfg.get("ttl_sec") or 60), scope_pairs(policy, entity_scope, entity_value)
```

In `governor.py`, make these changes.

1. Replace the `policy_eval` import at line 32 with the wrapped form; ruff's isort wraps it once `lease_limits` is added:

```python
from .policy_eval import (
    clamp_token_units,
    effective_policy,
    lease_limits,
    log_lookup_failure,
    requests_window,
    scope_pairs,
)
```

2. In `_acquire_concurrency`, add a `request_cfg` parameter after `now`, and replace the first lines of the body. Before:

```python
        units: int,
        now: float,
    ) -> tuple[bool, int, dict[str, Any]]:
        cfg = self._category_limits(policy, category)
        limit = int(cfg.get("max_concurrent") or 0)
        ttl_sec = int(cfg.get("ttl_sec") or 60)
        if limit <= 0:
            return True, 0, {"limit": 0, "remaining": 10**9, "unbounded": True}

        scope_keys = scope_pairs(policy, entity_scope, entity_value)

        remainings = []
```

After:

```python
        units: int,
        now: float,
        request_cfg: dict[str, Any],
    ) -> tuple[bool, int, dict[str, Any]]:
        limit, ttl_sec, scope_keys = lease_limits(policy, category, request_cfg, entity_scope, entity_value)
        if limit <= 0:
            return True, 0, {"limit": 0, "remaining": 10**9, "unbounded": True}

        remainings = []
```

3. In `check()`, the `elif category in ("streams", "jobs"):` call to `self._acquire_concurrency(...)` gains `request_cfg=cfg,` after `now=now,`. Here `cfg` is the loop variable from `for category, cfg in req.categories.items():`.

4. In `reserve()`, replace the start of the `elif category in ("streams", "jobs"):` branch. Before:

```python
            elif category in ("streams", "jobs"):
                cfgc = self._category_limits(pol, category)
                limit = int(cfgc.get("max_concurrent") or 0)
                ttl_sec = int(cfgc.get("ttl_sec") or 60)
                # Acquire for global and entity scopes (when configured)
                for sc, ev in scope_pairs(pol, entity_scope, entity_value):
```

After (`cfg` is again the request's category config from the enclosing loop):

```python
            elif category in ("streams", "jobs"):
                limit, ttl_sec, lease_scopes = lease_limits(pol, category, cfg, entity_scope, entity_value)
                # Acquire for global and entity scopes (when configured)
                for sc, ev in lease_scopes:
```

`commit()` and `renew()` need no change. They walk `scope_pairs(pol, ...)`, which always includes the entity bucket, and match leases by handle prefix.

In `governor_redis.py`, make these changes.

1. Replace the `policy_eval` import at line 24 with the same wrapped form as in `governor.py`.

2. In `check()`, the `elif category in ("streams", "jobs"):` branch (lines 1092-1107). Before:

```python
                elif category in ("streams", "jobs"):
                    limit = int((pol.get(category) or {}).get("max_concurrent") or 0)
                    ttl_sec = int((pol.get(category) or {}).get("ttl_sec") or 60)
                    cat_fail = self._effective_fail_mode(pol, category)
```

After:

```python
                elif category in ("streams", "jobs"):
                    limit, ttl_sec, scope_keys = lease_limits(pol, category, cfg, entity_scope, entity_value)
                    cat_fail = self._effective_fail_mode(pol, category)
```

Then delete these two lines at the start of that branch's `else:` block:

```python
                        scope_keys = self._scope_pairs(pol, entity_scope, entity_value)

```

3. In `reserve()`'s concurrency block (lines 1843-1853). Before:

```python
                units = int((cfg or {}).get("units") or 0)
                if units <= 0:
                    continue
                limit = int((pol.get(category) or {}).get("max_concurrent") or 0)
                ttl_sec = int((pol.get(category) or {}).get("ttl_sec") or 60)
                if limit <= 0:
                    # Missing max_concurrent is unbounded; mirror memory's semantics (spec §4).
                    continue
                # Named "pairs", not "scope_pairs": that name would shadow the
                # policy_eval.scope_pairs import for the rest of this function.
                pairs = self._scope_pairs(pol, entity_scope, entity_value)
```

After:

```python
                units = int((cfg or {}).get("units") or 0)
                if units <= 0:
                    continue
                # Named "pairs", not "scope_pairs": that name would shadow the
                # policy_eval.scope_pairs import for the rest of this function.
                limit, ttl_sec, pairs = lease_limits(pol, category, cfg or {}, entity_scope, entity_value)
                if limit <= 0:
                    # Missing max_concurrent is unbounded; mirror memory's semantics (spec §4).
                    continue
```

4. In the `if concurrency_failed:` block (lines 1958-1966), report the limit that was applied. Before:

```python
            failed = concurrency_failed_category or "streams"
            failed_info = dict(per_category.get(failed) or {})
            failed_info.update({
                "allowed": False,
                "limit": int((pol.get(failed) or {}).get("max_concurrent") or 0),
                "retry_after": int(denial_retry_after or (pol.get(failed) or {}).get("ttl_sec") or 1),
                "ttl_sec": int((pol.get(failed) or {}).get("ttl_sec") or 60),
            })
```

After:

```python
            failed = concurrency_failed_category or "streams"
            failed_limit, failed_ttl, _pairs = lease_limits(pol, failed, req.categories.get(failed) or {}, entity_scope, entity_value)
            failed_info = dict(per_category.get(failed) or {})
            failed_info.update({
                "allowed": False,
                "limit": failed_limit,
                "retry_after": int(denial_retry_after or failed_ttl or 1),
                "ttl_sec": failed_ttl,
            })
```

5. In `commit()` (lines 2147-2152), release leases the handle recorded even when the policy has no block for the category. `members` is the handle's persisted member map, parsed a few lines above. Before:

```python
            for category in ("streams", "jobs"):
                try:
                    if not (pol.get(category) or {}):
                        continue
```

After:

```python
            for category in ("streams", "jobs"):
                try:
                    # A request-limited lease (TASK-13435) has no policy block; its members say it exists.
                    if not (pol.get(category) or members.get(category)):
                        continue
```

6. In `renew()` (lines 2402-2405), refresh only the buckets the reservation charged (Review M4). `members` maps each category to the `"scope:value"` keys it charged, as `commit()` reads it; a handle without that record keeps the old walk. Before:

```python
            now = self._time()
            for category in cats:
                if category in ("streams", "jobs"):
                    for sc, ev in self._scope_pairs(pol, entity_scope, entity_value):
```

After:

```python
            now = self._time()
            for category in cats:
                if category in ("streams", "jobs"):
                    # Refresh only the buckets the reservation charged: a request-limited
                    # lease (TASK-13435) charged the caller's bucket, never `global`.
                    charged = [tuple(scope_key.split(":", 1)) for scope_key in (members.get(category) or {})]
                    for sc, ev in charged or self._scope_pairs(pol, entity_scope, entity_value):
```

The memory backend needs no change: its `renew()` only extends leases that already exist (`governor.py:837-852`).

In `Resource_Governance/README.md`, add this bullet after the `daily_cap` bullet (line 120):

```markdown
- A `streams`/`jobs` reservation can carry its own limit: `RGRequest(categories={"jobs": {"units": 1, "max_concurrent": 2, "ttl_sec": 120}})`. That limit and TTL replace the policy's `max_concurrent`/`ttl_sec` for the request, and only the caller's own bucket is charged (never `global`). Per-user concurrency quotas use this (`core/Usage/concurrency_quota.py`, ADR-076).
```

- [ ] **Step 4: Run the new tests, then the whole RG suite**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Resource_Governance/test_request_concurrency_limit.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py`

Expected: `13 passed`, or `12 passed, 1 skipped` without a real Redis.

Then show that the two Redis fixes are each pinned. Revert each change below, one at a time, then put it back; each revert must make exactly the named tests fail (measured):
- In `commit()`, `members.get(category)` → `{}`: `test_request_limit_applies_without_a_policy_block[redis]` and `test_request_limit_holds_across_two_workers` fail (`2 failed, 11 passed`).
- In `renew()`, `charged or self._scope_pairs(...)` → `self._scope_pairs(...)`: `test_renewing_a_request_limited_lease_never_charges_global[redis]` fails (`1 failed, 12 passed`).

Run: `redis-cli --scan --pattern 'rg_rcl_*' | wc -l`

Expected: `0`. The tests left no keys in Redis (Review M10).

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -n 4 tldw_Server_API/tests/Resource_Governance`

Expected: `421 passed, 2 xfailed` (measured on the prototype, with real Redis). Any failure must be shown on clean `origin/dev`.

Run (Review M1: list the touched RG files, because `Resource_Governance/tenant.py` has a pre-existing I001 on `origin/dev`): `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check tldw_Server_API/app/core/Resource_Governance/policy_eval.py tldw_Server_API/app/core/Resource_Governance/governor.py tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/tests/Resource_Governance/test_request_concurrency_limit.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py`

Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Resource_Governance/policy_eval.py tldw_Server_API/app/core/Resource_Governance/governor.py tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/app/core/Resource_Governance/README.md tldw_Server_API/tests/Resource_Governance/test_request_concurrency_limit.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py
git commit -m "feat(rg): a streams/jobs reservation can carry its own max_concurrent and ttl_sec on both backends (TASK-13435)

The Redis backend also releases a lease whose policy has no block for its
category; before, such a lease was never released, only lapsed. Its renew()
now refreshes only the buckets a lease charged, so renewing a request-limited
lease no longer adds it to the policy's global bucket.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 2: Concurrency slots and the three `limits.*` keys

**Files:**
- Create: `tldw_Server_API/app/core/Usage/concurrency_quota.py`
- Modify: `tldw_Server_API/Config_Files/user_profile_catalog.yaml` (after the `limits.chatbooks_concurrent_jobs` entry, lines 368-376)
- Modify: `tldw_Server_API/tests/UserProfile/test_user_profile_limits.py:25-37` (`NEW_KEYS`)
- Test: `tldw_Server_API/tests/Usage/test_concurrency_quota.py` (new)

**Interfaces:**
- Consumes: Task 1's request-limit contract; `quota_resolver.user_quota(user_id, key) -> int | float | None`; `quota_checks.as_quota_user_id(value) -> int | None`.
- Produces:
  - `concurrency_quota.acquire_slot(user_id: Any, key: str, category: str) -> ConcurrencySlot`. It raises `ConcurrencyQuotaExceeded`, which has `.limit: int` and `.retry_after: int | None` (`None` when the limit is 0).
  - `ConcurrencySlot.release() -> None`: async, idempotent, shielded, and it never raises a lease-store error, redis-py's included (Review B1).
  - `LEASE_TTL_SEC = 120`, `RENEW_EVERY_SEC = 30.0`.
  - `_NONCRITICAL`: the builtin errors plus `redis.exceptions.RedisError` when redis is importable (Review B1).
  - Test seams: `concurrency_quota._governor` (the lease store, settable), `concurrency_quota._QuotaPolicies`, and `concurrency_quota._lease_governor()`.
  - Catalog keys `limits.media_concurrent_jobs`, `limits.audio_concurrent_streams` and `limits.audio_concurrent_transcriptions`.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_concurrency_quota.py`. The store-failure fakes raise `redis.exceptions.ConnectionError`, which is what a dropped Redis raises; a builtin `ConnectionError` would pass while the real path fails (Review B1). `test_a_redis_outage_never_escapes_renew_or_release` pins both `_keep_alive` and `release()`. Every test asserts on real output or state, so the ratchet passes: `test_no_limit_is_unlimited_and_never_touches_the_store` also asserts each slot's `_handle_id is None` (Review I1).

```python
"""Per-user concurrency slots: unset is unlimited, 0 blocks, every exit path gives the slot back (TASK-13435)."""

import asyncio

import anyio
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Resource_Governance.governor_redis import RedisResourceGovernor
from tldw_Server_API.app.core.Usage import concurrency_quota

pytestmark = pytest.mark.unit
KEY = "limits.media_concurrent_jobs"


@pytest.fixture()
def limits(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Per-key limits the test fills in, and a fresh in-memory lease store."""
    values: dict = {}

    async def _user_quota(user_id: int | None, key: str) -> object:
        """Serve the test's limit for this key; like the real resolver, no user id means unlimited."""
        return None if user_id is None else values.get(key)

    monkeypatch.setattr(concurrency_quota, "user_quota", _user_quota)
    monkeypatch.setattr(concurrency_quota, "_governor", MemoryResourceGovernor(policy_loader=concurrency_quota._QuotaPolicies()))
    return values


class _RecordingGovernor:
    """A lease store that admits everything and records what it was asked."""

    def __init__(self) -> None:
        """Start with empty records."""
        self.calls: list[tuple[str, object]] = []

    async def reserve(self, req):
        """Admit and record the request."""
        self.calls.append(("reserve", req))
        return type("D", (), {"allowed": True, "retry_after": None})(), f"h{len(self.calls)}"

    async def renew(self, handle_id, ttl_s):
        """Record a renewal."""
        self.calls.append(("renew", handle_id))

    async def release(self, handle_id):
        """Record a release after one scheduling point, as a networked store would."""
        await asyncio.sleep(0)
        self.calls.append(("release", handle_id))


class _DroppedRedisGovernor(_RecordingGovernor):
    """Admits, then Redis goes away: every renewal and release fails the way redis-py fails."""

    async def renew(self, handle_id, ttl_s):
        """Record the attempt, then fail like a dropped connection."""
        self.calls.append(("renew", handle_id))
        raise RedisConnectionError("Connection closed by server.")

    async def release(self, handle_id):
        """Record the attempt, then fail like a dropped connection."""
        self.calls.append(("release", handle_id))
        raise RedisConnectionError("Connection closed by server.")


async def test_no_limit_is_unlimited_and_never_touches_the_store(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A user with no value gets a slot that holds nothing; the lease store is never built."""
    built: list[str] = []

    def _recording_store() -> _RecordingGovernor:
        """Record that the store was built."""
        built.append("store")
        return _RecordingGovernor()

    monkeypatch.setattr(concurrency_quota, "_lease_governor", _recording_store)
    slots = [await concurrency_quota.acquire_slot(7, KEY, "jobs") for _ in range(5)]
    assert all(slot._handle_id is None for slot in slots)
    for slot in slots:
        await slot.release()
    assert built == []


async def test_limit_is_enforced_and_release_frees_a_slot(limits: dict) -> None:
    """With a limit of 1 the second slot is refused with a retry hint; releasing the first admits the next."""
    limits[KEY] = 1
    first = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    with pytest.raises(concurrency_quota.ConcurrencyQuotaExceeded) as exc:
        await concurrency_quota.acquire_slot(7, KEY, "jobs")
    assert exc.value.limit == 1 and exc.value.retry_after >= 1
    await first.release()
    second = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await second.release()


async def test_zero_blocks_without_a_retry_hint(limits: dict) -> None:
    """A limit of 0 refuses every request, and there is no point retrying."""
    limits[KEY] = 0
    with pytest.raises(concurrency_quota.ConcurrencyQuotaExceeded) as exc:
        await concurrency_quota.acquire_slot(7, KEY, "jobs")
    assert exc.value.limit == 0 and exc.value.retry_after is None


async def test_users_and_keys_do_not_share_slots(limits: dict) -> None:
    """One user's held slot neither blocks another user nor a different concurrency key."""
    limits[KEY] = 1
    limits["limits.audio_concurrent_transcriptions"] = 1
    held = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    other_user = await concurrency_quota.acquire_slot(8, KEY, "jobs")
    other_key = await concurrency_quota.acquire_slot(7, "limits.audio_concurrent_transcriptions", "jobs")
    for slot in (held, other_user, other_key):
        await slot.release()


async def test_release_is_idempotent(limits: dict) -> None:
    """Releasing twice frees one slot, not two."""
    limits[KEY] = 2
    a = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    b = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await a.release()
    await a.release()
    c = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    with pytest.raises(concurrency_quota.ConcurrencyQuotaExceeded):
        await concurrency_quota.acquire_slot(7, KEY, "jobs")
    for slot in (b, c):
        await slot.release()


async def test_cancelled_request_gives_its_slot_back(limits: dict) -> None:
    """A request cancelled mid-flight (client gone) releases in its finally, and the slot is free again."""
    limits[KEY] = 1
    started = asyncio.Event()

    async def _request() -> None:
        """Hold a slot until cancelled."""
        slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            await slot.release()

    task = asyncio.create_task(_request())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await slot.release()


async def test_release_completes_inside_a_cancelled_scope(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A server that cancels the request's scope (anyio level cancellation) cannot abort the release."""
    limits[KEY] = 1
    store = _RecordingGovernor()
    monkeypatch.setattr(concurrency_quota, "_governor", store)
    slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    with anyio.CancelScope() as scope:
        scope.cancel()
        try:
            await anyio.sleep(1)
        finally:
            await slot.release()
    assert ("release", "h1") in store.calls


async def test_keep_alive_renews_until_the_owner_ends(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A slot renews its lease while its task runs, and stops once the task ends without releasing."""
    limits[KEY] = 1
    store = _RecordingGovernor()
    monkeypatch.setattr(concurrency_quota, "_governor", store)
    monkeypatch.setattr(concurrency_quota, "RENEW_EVERY_SEC", 0.01)

    async def _leaky_request() -> None:
        """Take a slot, run a while, and end without releasing it (a bug the backstop covers)."""
        await concurrency_quota.acquire_slot(7, KEY, "jobs")
        await asyncio.sleep(0.05)

    await asyncio.create_task(_leaky_request())
    renewals = sum(1 for call in store.calls if call[0] == "renew")
    await asyncio.sleep(0.05)
    assert renewals >= 2
    assert sum(1 for call in store.calls if call[0] == "renew") == renewals


async def test_store_error_admits_the_request(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lease store that fails open: the request goes ahead with a slot that holds nothing."""
    limits[KEY] = 1

    class _Broken:
        """A lease store whose Redis connection has dropped."""

        async def reserve(self, req):
            """Fail the way redis-py does: its ConnectionError is not the builtin one."""
            raise RedisConnectionError("Connection closed by server.")

    monkeypatch.setattr(concurrency_quota, "_governor", _Broken())
    slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    assert slot._handle_id is None
    await slot.release()


async def test_parallel_requests_get_exactly_the_limit(limits: dict) -> None:
    """Six requests from one user racing for a limit of 2: two get slots, four get the 429 refusal."""
    limits[KEY] = 2
    results = await asyncio.gather(
        *(concurrency_quota.acquire_slot(7, KEY, "jobs") for _ in range(6)), return_exceptions=True
    )
    held = [r for r in results if isinstance(r, concurrency_quota.ConcurrencySlot)]
    refused = [r for r in results if isinstance(r, concurrency_quota.ConcurrencyQuotaExceeded)]
    assert (len(held), len(refused)) == (2, 4)
    for slot in held:
        await slot.release()


async def test_lowering_the_limit_refuses_new_requests_until_usage_drops(limits: dict) -> None:
    """Running requests keep their slots when an admin lowers the limit; new ones wait until usage is under it."""
    limits[KEY] = 3
    running = [await concurrency_quota.acquire_slot(7, KEY, "jobs") for _ in range(3)]
    limits[KEY] = 1
    await running[0].release()
    with pytest.raises(concurrency_quota.ConcurrencyQuotaExceeded):
        await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await running[1].release()
    await running[2].release()
    slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await slot.release()


async def test_switching_quotas_off_admits_at_once(limits: dict) -> None:
    """With the user's slot held, turning the limit off (switch off or value removed) admits the next request."""
    limits[KEY] = 1
    held = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    limits.pop(KEY)
    unlimited = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await unlimited.release()
    await held.release()


async def test_a_redis_outage_never_escapes_renew_or_release(limits: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redis drops mid-request: renewals keep trying, the release does not raise, and the request's own outcome stands."""
    limits[KEY] = 1
    store = _DroppedRedisGovernor()
    monkeypatch.setattr(concurrency_quota, "_governor", store)
    monkeypatch.setattr(concurrency_quota, "RENEW_EVERY_SEC", 0.01)
    slot = await concurrency_quota.acquire_slot(7, KEY, "jobs")
    await asyncio.sleep(0.05)
    assert sum(1 for call in store.calls if call[0] == "renew") >= 2
    assert not slot._keepalive.done()
    await slot.release()
    assert ("release", "h1") in store.calls


async def test_a_non_numeric_user_is_unlimited(limits: dict) -> None:
    """A principal without a numeric user id is never limited and never crashes the request."""
    limits[KEY] = 0
    for user_id in ("api-key-7", None):
        slot = await concurrency_quota.acquire_slot(user_id, KEY, "jobs")
        await slot.release()


def test_store_follows_rg_backend_even_with_rg_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lease store is Redis when RG_BACKEND=redis, and RG_ENABLED=false does not remove it (ADR-076)."""
    monkeypatch.setattr(concurrency_quota, "_governor", None)
    monkeypatch.setenv("RG_ENABLED", "false")
    monkeypatch.setenv("RG_BACKEND", "redis")
    assert isinstance(concurrency_quota._lease_governor(), RedisResourceGovernor)
    monkeypatch.setattr(concurrency_quota, "_governor", None)
    monkeypatch.setenv("RG_BACKEND", "memory")
    assert isinstance(concurrency_quota._lease_governor(), MemoryResourceGovernor)
```

In `tldw_Server_API/tests/UserProfile/test_user_profile_limits.py`, add these three entries to the `NEW_KEYS` set, after `"limits.chatbooks_concurrent_jobs",`:

```python
    "limits.media_concurrent_jobs",
    "limits.audio_concurrent_streams",
    "limits.audio_concurrent_transcriptions",
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_concurrency_quota.py`

Expected: `1 error` during collection: `ImportError: cannot import name 'concurrency_quota'`.

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/UserProfile/test_user_profile_limits.py::test_catalog_has_every_limit_key_platform_admin_only`

Expected: `1 failed`. The diff names the three missing keys (measured).

- [ ] **Step 3: Implement**

Create `tldw_Server_API/app/core/Usage/concurrency_quota.py`:

```python
"""Per-user synchronous concurrency quotas (TASK-13435, ADR-076).

A request holds one lease for as long as it runs. The limit is the user's
``limits.*`` value: ``None`` is unlimited and costs nothing, ``0`` blocks
(ADR-064). Leases live in a Resource Governor built from ``RG_BACKEND``: on Redis
one limit holds across workers; on the memory backend it holds per worker process.

Every exit path calls ``ConcurrencySlot.release()``. Two backstops bound a missed
release: a lease lapses ``LEASE_TTL_SEC`` after its last renewal, and a slot stops
renewing once the task that took it has finished.
"""

from __future__ import annotations

import asyncio
from typing import Any

from anyio import CancelScope
from loguru import logger

from tldw_Server_API.app.core.config import rg_backend
from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor, ResourceGovernor, RGRequest
from tldw_Server_API.app.core.Usage.quota_checks import as_quota_user_id
from tldw_Server_API.app.core.Usage.quota_resolver import user_quota

try:
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis is an optional deployment dependency
    _REDIS_ERRORS: tuple[type[Exception], ...] = ()
else:
    # redis-py's ConnectionError and TimeoutError are not the builtins: they derive from RedisError.
    _REDIS_ERRORS = (RedisError,)

LEASE_TTL_SEC = 120
RENEW_EVERY_SEC = 30.0

_NONCRITICAL = (
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    *_REDIS_ERRORS,
)
_governor: ResourceGovernor | None = None


class ConcurrencyQuotaExceeded(Exception):
    """Every one of the user's slots is held. ``retry_after`` is None when the limit is 0."""

    def __init__(self, limit: int, retry_after: int | None) -> None:
        """Record the limit that was hit and when a slot may free up."""
        super().__init__(f"Concurrency limit reached ({limit})")
        self.limit = limit
        self.retry_after = retry_after


class _QuotaPolicies:
    """Quota leases carry their own limit and TTL, so every policy id resolves to one bare policy."""

    def get_policy(self, _policy_id: str) -> dict[str, Any]:
        """A policy with no limits of its own."""
        return {"scopes": ["user"]}


def _lease_governor() -> ResourceGovernor:
    """This process's lease store, built once from RG_BACKEND. RG_ENABLED does not turn it off (ADR-076)."""
    global _governor
    if _governor is None:
        if rg_backend() == "redis":
            from tldw_Server_API.app.core.Resource_Governance.governor_redis import RedisResourceGovernor

            _governor = RedisResourceGovernor(policy_loader=_QuotaPolicies())
        else:
            _governor = MemoryResourceGovernor(policy_loader=_QuotaPolicies())
    return _governor


class ConcurrencySlot:
    """One held lease, or nothing for an unlimited user. ``release()`` is idempotent."""

    def __init__(self, governor: ResourceGovernor | None = None, handle_id: str | None = None) -> None:
        """Hold ``handle_id`` and renew it while the calling task runs."""
        self._governor = governor
        self._handle_id = handle_id
        self._owner = asyncio.current_task()
        self._keepalive = asyncio.create_task(self._keep_alive()) if handle_id else None

    async def _keep_alive(self) -> None:
        """Renew the lease while the task that took it runs; once that task ends, let the lease lapse."""
        while True:
            await asyncio.sleep(RENEW_EVERY_SEC)
            if self._owner is None or self._owner.done() or self._handle_id is None:
                return
            try:
                await self._governor.renew(self._handle_id, LEASE_TTL_SEC)
            except _NONCRITICAL as exc:
                logger.debug("Concurrency lease renew failed: {!r}", exc)

    async def release(self) -> None:
        """Give the slot back. Completes even inside a cancelled request scope."""
        handle_id, self._handle_id = self._handle_id, None
        if handle_id is None:
            return
        if self._keepalive is not None:
            self._keepalive.cancel()
        with CancelScope(shield=True):
            try:
                await self._governor.release(handle_id)
            except _NONCRITICAL as exc:
                logger.warning("Concurrency lease release failed; it lapses within {} s: {!r}", LEASE_TTL_SEC, exc)


async def acquire_slot(user_id: Any, key: str, category: str) -> ConcurrencySlot:
    """Take one of the user's ``key`` slots, or raise ConcurrencyQuotaExceeded.

    ``category`` is the lease kind, ``"jobs"`` or ``"streams"``; ``key`` names both the
    ``limits.*`` value and the lease bucket. A user id that is not numeric is unlimited.
    A lease-store error admits the request and logs a warning (fail open, like the resolver).
    """
    uid = as_quota_user_id(user_id)
    limit = await user_quota(uid, key)
    if limit is None:
        return ConcurrencySlot()
    if int(limit) < 1:
        raise ConcurrencyQuotaExceeded(0, retry_after=None)
    governor = _lease_governor()
    try:
        decision, handle_id = await governor.reserve(
            RGRequest(
                entity=f"user:{uid}",
                categories={category: {"units": 1, "max_concurrent": int(limit), "ttl_sec": LEASE_TTL_SEC}},
                tags={"policy_id": key, "endpoint": key},
            )
        )
    except _NONCRITICAL as exc:
        logger.warning("Concurrency quota {} unavailable for user {}; admitting: {!r}", key, uid, exc)
        return ConcurrencySlot()
    if not decision.allowed or not handle_id:
        raise ConcurrencyQuotaExceeded(int(limit), retry_after=int(decision.retry_after or 1))
    return ConcurrencySlot(governor, handle_id)
```

Notes for the implementer:
- `user_quota` returns `None` when the switch is off, when no value is set, or when the lookup fails (it fails open and is cached for 60 s). That is why an unlimited user never builds the store.
- A limit of 0 is refused before the store, because RG treats a 0 limit as unbounded.
- `_NONCRITICAL` adds `redis.exceptions.RedisError` because redis-py's `ConnectionError` and `TimeoutError` derive from it, not from the builtins. The import is guarded, as `Infrastructure/redis_factory.py` guards its own. Do not widen this to bare `Exception` (Review B1).
- `reserve()` gets no `op_id`: the id would be random, so it buys no idempotency, and on Redis it costs a GET and a 24 h key per acquire (Review M9).

In `user_profile_catalog.yaml`, add after the `limits.chatbooks_concurrent_jobs` entry:

```yaml
  - key: limits.media_concurrent_jobs
    label: Media Concurrent Ingests
    description: Media ingest requests (/media/add) running at once. Queued ingest jobs (/media/ingest/jobs) and /media/process-* are not counted.
    type: integer
    minimum: 0
    default: null
    editable_by: [platform_admin]
    sensitivity: internal
    ui: number

  - key: limits.audio_concurrent_streams
    label: Audio Concurrent Streams
    description: Audio streaming sessions (transcription, TTS, voice chat, realtime) open at once.
    type: integer
    minimum: 0
    default: null
    editable_by: [platform_admin]
    sensitivity: internal
    ui: number

  - key: limits.audio_concurrent_transcriptions
    label: Audio Concurrent Transcriptions
    description: Direct transcription and translation requests running at once.
    type: integer
    minimum: 0
    default: null
    editable_by: [platform_admin]
    sensitivity: internal
    ui: number
```

- [ ] **Step 4: Run the tests**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_concurrency_quota.py tldw_Server_API/tests/UserProfile/test_user_profile_limits.py tldw_Server_API/tests/UserProfile/test_user_profile_catalog.py`

Expected: `23 passed` (`15 passed` for the new file), measured.

Then show that these tests can fail. Revert each change below, one at a time, then put it back; each revert must make exactly the named tests fail (measured):
- `CancelScope(shield=True)` → `CancelScope(shield=False)`: `test_release_completes_inside_a_cancelled_scope` fails (`1 failed, 14 passed`).
- `if self._owner is None or self._owner.done() or self._handle_id is None:` → `if self._handle_id is None:`: `test_keep_alive_renews_until_the_owner_ends` fails (`1 failed, 14 passed`).
- Delete the `*_REDIS_ERRORS,` line from `_NONCRITICAL`: `test_store_error_admits_the_request` and `test_a_redis_outage_never_escapes_renew_or_release` fail (`2 failed, 13 passed`) (Review B1).

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Usage/concurrency_quota.py tldw_Server_API/Config_Files/user_profile_catalog.yaml tldw_Server_API/tests/Usage/test_concurrency_quota.py tldw_Server_API/tests/UserProfile/test_user_profile_limits.py
git commit -m "feat(quotas): per-user concurrency slots held as RG leases; limits.media_concurrent_jobs, audio_concurrent_streams, audio_concurrent_transcriptions (TASK-13435)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 3: Audio streams hold a slot

**Files:**
- Modify: `tldw_Server_API/app/core/Usage/audio_quota.py`: imports (lines 21 and 30), a key and a `ContextVar` before `_UNLIMITED_AUDIO_LIMITS` (line 314), `get_limits_for_user` (lines 324-333), and `can_start_stream`/`finish_stream` (lines 772-779).
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`: `websocket_transcribe`, moving lines 1481-1487 down to just above the `try:` at line 1793; the `streaming_limits` docstring (lines 4535-4536) and comment (line 4597) (Review M5).
- Modify: `tldw_Server_API/app/api/v1/schemas/audio_schemas.py`: `StreamingLimitsResponse.active_streams` and `can_start_stream` descriptions (lines 339-345) (Review M5).
- Modify: `tldw_Server_API/tests/Usage/test_audio_quota_per_user.py`: lines 5, 10-21, 24-30 and 88-91.
- Modify: `tldw_Server_API/tests/Audio/test_ws_idle_metrics_audio.py`: lines 1-5 and 15 (re-measured on `32eb11ae03`; see "What was measured").
- Test: `tldw_Server_API/tests/Usage/test_audio_stream_slots.py` (new)
- Test: `tldw_Server_API/tests/Audio/test_ws_transcribe_stream_slot_release.py` (new)

**Interfaces:**
- Consumes: `concurrency_quota.acquire_slot`, `ConcurrencyQuotaExceeded`, `ConcurrencySlot`.
- Produces:
  - `audio_quota.STREAMS_KEY = "limits.audio_concurrent_streams"`.
  - `can_start_stream(user_id) -> tuple[bool, str]` holds a slot for the calling task, or returns `(False, "Concurrent streams limit reached (N)")`.
  - `finish_stream(user_id) -> None` releases the calling task's slot, and is a no-op when the task holds none.
  - `get_limits_for_user` reports `concurrent_streams`.
  - The hook names and signatures are unchanged, so the six handlers, the shim (`audio_streaming._audio_shim_attr`) and the existing tests that patch them keep working.
  - `GET /audio/stream/limits` keeps `active_streams: null` and `can_start_stream: true`; its docstring and schema now say why (Review M5). A live answer would need a count of open streams, which nothing reports; a partial one (false only at a limit of 0) would still mislead at any other limit, so this plan does not compute it.

Why a `ContextVar`: every handler calls `can_start_stream` and `finish_stream` in the same task, and a task-local binds each release to that stream's own slot. The pre-spec-2 code kept a per-user stack and popped the newest handle. With leases that renew while their owner runs, popping another stream's slot would leave the first stream's lease to lapse 2 minutes later, while that stream still ran. `test_finish_releases_the_finishing_streams_own_slot` pins this by reading the real lease store: after stream `a` finishes, only stream `b`'s lease is held (Review I1).

`heartbeat_stream` and `heartbeat_jobs` stay no-ops, because slots renew themselves. `can_start_job` and `finish_job` stay no-ops too: they serve the queued-job workers, which `limits.audio_concurrent_jobs` limits.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_audio_stream_slots.py`:

```python
"""Audio streams hold a per-user slot from can_start_stream to finish_stream (TASK-13435)."""

import asyncio

import pytest

from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Usage import audio_quota, concurrency_quota

pytestmark = pytest.mark.unit


@pytest.fixture()
def streams_limit(monkeypatch: pytest.MonkeyPatch) -> dict:
    """A per-key limit the test sets, served to both the slot store and the audio limits view."""
    values: dict = {}

    async def _user_quota(_user_id: int, key: str) -> object:
        """Serve the test's limit for this key."""
        return values.get(key)

    monkeypatch.setattr(concurrency_quota, "user_quota", _user_quota)
    monkeypatch.setattr(audio_quota, "user_quota", _user_quota)
    monkeypatch.setattr(concurrency_quota, "_governor", MemoryResourceGovernor(policy_loader=concurrency_quota._QuotaPolicies()))
    return values


async def _in_own_task(coro):
    """Run a coroutine in a task of its own, as each WebSocket handler does."""
    return await asyncio.create_task(coro)


async def test_second_stream_is_refused_at_the_limit_and_admitted_after_finish(streams_limit: dict) -> None:
    """With limits.audio_concurrent_streams = 1 a second stream is refused until the first finishes."""
    streams_limit[audio_quota.STREAMS_KEY] = 1
    held = asyncio.Event()
    done = asyncio.Event()

    async def _first_stream() -> None:
        """Hold a stream until told to finish."""
        assert await audio_quota.can_start_stream(7) == (True, "OK")
        held.set()
        await done.wait()
        await audio_quota.finish_stream(7)

    first = asyncio.create_task(_first_stream())
    await held.wait()
    assert await _in_own_task(audio_quota.can_start_stream(7)) == (False, "Concurrent streams limit reached (1)")
    done.set()
    await first
    assert await _in_own_task(audio_quota.can_start_stream(7)) == (True, "OK")


async def test_finish_releases_the_finishing_streams_own_slot(streams_limit: dict) -> None:
    """Of two open streams, the one that finishes gives back its own lease; the other's lease stays held in the store."""
    streams_limit[audio_quota.STREAMS_KEY] = 2
    store = concurrency_quota._governor
    a_open, b_open, a_done, b_done = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
    handles: dict[str, str] = {}

    async def _stream(name: str, opened: asyncio.Event, finish: asyncio.Event) -> None:
        """Open a stream, note its lease, finish when told."""
        await audio_quota.can_start_stream(7)
        handles[name] = audio_quota._stream_slot.get()._handle_id
        opened.set()
        await finish.wait()
        await audio_quota.finish_stream(7)

    a = asyncio.create_task(_stream("a", a_open, a_done))
    await a_open.wait()
    b = asyncio.create_task(_stream("b", b_open, b_done))
    await b_open.wait()
    a_done.set()
    await a
    held = store._leases[(audio_quota.STREAMS_KEY, "streams", "user", "7")]
    assert {lease_id.split(":")[0] for lease_id in held} == {handles["b"]}
    b_done.set()
    await b


async def test_finish_without_a_slot_is_a_noop(streams_limit: dict) -> None:
    """finish_stream in a task that never started a stream does nothing and does not raise."""
    assert await _in_own_task(audio_quota.finish_stream(7)) is None


async def test_limits_view_reports_the_streams_limit(streams_limit: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /audio/stream/limits shows limits.audio_concurrent_streams as concurrent_streams."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    streams_limit[audio_quota.STREAMS_KEY] = 3
    assert (await audio_quota.get_limits_for_user(7))["concurrent_streams"] == 3
```

`tldw_Server_API/tests/Audio/test_ws_transcribe_stream_slot_release.py`:

```python
"""/audio/stream/transcribe never leaves a stream slot held when session setup fails (TASK-13435)."""

import pytest
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.tests.Audio.ws_test_helpers import ws_client_without_lifespan

pytestmark = pytest.mark.unit


def test_setup_failure_after_admission_leaves_no_slot_held(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure while the session is being set up must not strand a slot: every start has its finish."""
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_streaming as audio_streaming
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.main import app

    started: list[int] = []
    finished: list[int] = []

    async def _start(user_id: int) -> tuple[bool, str]:
        """Admit and record the start."""
        started.append(user_id)
        return True, "OK"

    async def _finish(user_id: int) -> None:
        """Record the finish."""
        finished.append(user_id)

    def _broken_setup() -> float:
        """Fail during session setup, after admission on the old code path."""
        raise RuntimeError("setup failed")

    monkeypatch.setattr(audio_ep, "can_start_stream", _start)
    monkeypatch.setattr(audio_ep, "finish_stream", _finish)
    monkeypatch.setattr(audio_streaming, "_get_failopen_cap_minutes", _broken_setup)
    token = get_settings().SINGLE_USER_API_KEY

    with ws_client_without_lifespan(app) as client:
        try:
            with client.websocket_connect(f"/api/v1/audio/stream/transcribe?token={token}") as ws:
                ws.receive_json()
        except (WebSocketDisconnect, RuntimeError):
            pass

    assert len(started) == len(finished)
```

Make these changes in `tldw_Server_API/tests/Usage/test_audio_quota_per_user.py`:
- Import line 5 becomes `from tldw_Server_API.app.core.Usage import audio_quota, concurrency_quota`.
- In the `limits` fixture, after `monkeypatch.setattr(audio_quota, "user_quota", _user_quota)`, add `monkeypatch.setattr(concurrency_quota, "user_quota", _user_quota)`.
- Replace the body of `test_limits_come_from_the_resolver` with:

```python
    limits.update({
        "limits.audio_daily_minutes": 45, "limits.transcription_minutes_per_month": 600,
        "limits.audio_concurrent_jobs": 3, "limits.audio_concurrent_streams": 2,
    })
    assert await audio_quota.get_limits_for_user(1) == {
        "daily_minutes": 45, "monthly_minutes": 600, "concurrent_jobs": 3,
        "concurrent_streams": 2, "max_file_size_mb": None,
    }
```

- Replace `test_synchronous_concurrency_is_unlimited` (its docstring says the work is deferred) with:

```python
async def test_unset_concurrency_keys_are_unlimited(limits: dict) -> None:
    """With no limits.audio_concurrent_streams set, streams are admitted; queued-job hooks always admit (TASK-13435)."""
    assert await audio_quota.can_start_job(1) == (True, "OK")
    for _ in range(3):
        assert await audio_quota.can_start_stream(1) == (True, "OK")
    await audio_quota.finish_stream(1)
```

In `tldw_Server_API/tests/Audio/test_ws_idle_metrics_audio.py`, keep the new resolver lookup out of the test's 0.25 s window, as Task 5 does for the email harness (see "What was measured"; found re-measuring on `32eb11ae03`, Review I1). Replace the imports (lines 1-5) with:

```python
import time
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.app.core.Usage import concurrency_quota
from tldw_Server_API.tests.Audio.ws_test_helpers import ws_client_without_lifespan, ws_session_or_skip
```

and after `monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_S", "0")` (line 15) add:

```python
    # Quotas are outside this test: the stream-slot lookup must not spend the 0.25 s on the AuthNZ database.
    monkeypatch.setattr(concurrency_quota, "user_quota", AsyncMock(return_value=None))
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_audio_stream_slots.py tldw_Server_API/tests/Audio/test_ws_transcribe_stream_slot_release.py tldw_Server_API/tests/Usage/test_audio_quota_per_user.py`

Expected (measured with Tasks 1–2 applied): 5 failed, the rest passed.
- 3 failures in `test_audio_stream_slots.py`: `AttributeError: module ... 'audio_quota' has no attribute 'STREAMS_KEY'`. `test_finish_without_a_slot_is_a_noop` passes; it pins the no-op case.
- `test_setup_failure_after_admission_leaves_no_slot_held`: `assert 1 == 0`.
- `test_limits_come_from_the_resolver`: `concurrent_streams` is `None`, not 2.

- [ ] **Step 3: Implement**

In `audio_quota.py`, make these changes.

1. Add the imports. After `from collections.abc import Awaitable, Callable` add:

```python
from contextvars import ContextVar
```

After `from tldw_Server_API.app.core.exceptions import AudioQuotaStoreUnavailable` add:

```python
from tldw_Server_API.app.core.Usage import concurrency_quota
```

2. Immediately before the `# Every audio limit when usage quotas are off: no tier applies (spec 2).` comment (line 314), add:

```python
STREAMS_KEY = "limits.audio_concurrent_streams"
# The stream slot the current request task holds. can_start_stream and finish_stream
# always run in the same handler task, so a task-local binds each release to its own slot.
_stream_slot: ContextVar[concurrency_quota.ConcurrencySlot | None] = ContextVar("audio_stream_slot", default=None)


```

3. In `get_limits_for_user`, add this line after the `concurrent_jobs` line:

```python
    limits["concurrent_streams"] = await user_quota(uid, STREAMS_KEY)
```

4. Replace `can_start_stream` and `finish_stream` with:

```python
async def can_start_stream(user_id: int) -> tuple[bool, str]:
    """Hold one of the user's audio stream slots (``limits.audio_concurrent_streams``) for this task."""
    try:
        slot = await concurrency_quota.acquire_slot(user_id, STREAMS_KEY, "streams")
    except concurrency_quota.ConcurrencyQuotaExceeded as exc:
        _metrics_increment("audio_quota_violations_total", {"type": "concurrent_streams"})
        return False, f"Concurrent streams limit reached ({exc.limit})"
    _stream_slot.set(slot)
    return True, "OK"


async def finish_stream(user_id: int) -> None:
    """Give back the stream slot this task took in ``can_start_stream``; a no-op when it holds none."""
    slot = _stream_slot.get()
    _stream_slot.set(None)
    if slot is not None:
        await slot.release()
```

In `audio_streaming.py`, in `websocket_transcribe`, delete these lines (1481-1488):

```python
        ok_stream, msg_stream = await _can_start_stream(user_id_for_usage)
        if not ok_stream:
            if _outer_stream:
                await _outer_stream.send_json({"type": "error", "message": msg_stream})
            await websocket.close()
            return
        acquired_stream = True

```

Then, directly above the `try:` that begins `await handle_unified_websocket(` (line 1793, right after the `_on_heartbeat` definition), insert:

```python
        # Take the stream slot only here: the finally below is what gives it back (TASK-13435).
        ok_stream, msg_stream = await _can_start_stream(user_id_for_usage)
        if not ok_stream:
            if _outer_stream:
                await _outer_stream.send_json({"type": "error", "message": msg_stream})
            await websocket.close()
            return
        acquired_stream = True
```

Leave the other five stream sites alone. Each already takes its slot inside the `try` whose `finally` releases it; see "What was measured" for the line numbers.

Then correct the stale `/audio/stream/limits` contract text (Review M5). In `audio_streaming.py`, `streaming_limits`, replace these docstring lines (4535-4536):

```python
            - active_streams (None): Per-user stream concurrency is not tracked.
            - _can_start_stream (bool): Always True; stream concurrency is not limited per user.
```

with:

```python
            - active_streams (None): The count of the user's open streams is not reported.
            - _can_start_stream (bool): Always True. It is not a live check: open streams are not
              counted, so a stream over `limits.audio_concurrent_streams` is refused when it starts.
```

and the comment above `can_start = True` (line 4597):

```python
    # Stream concurrency isn't limited per user.
```

with:

```python
    # Not a live check: open streams aren't counted; a stream over the limit is refused when it starts.
```

In `tldw_Server_API/app/api/v1/schemas/audio_schemas.py`, `StreamingLimitsResponse`, replace the `active_streams` description (line 341):

```python
        description="Always null: per-user stream concurrency is not tracked.",
```

with:

```python
        description="Always null: the count of the user's open streams is not reported.",
```

and the `can_start_stream` field (lines 343-345):

```python
    can_start_stream: bool = Field(
        ..., description="Whether another stream can be started."
    )
```

with:

```python
    can_start_stream: bool = Field(
        ...,
        description=(
            "Always true. Not a live check: open streams are not counted, so a stream over the "
            "user's limit is refused when it starts."
        ),
    )
```

No test or snapshot carries either description (`git grep` on `32eb11ae03` finds them only in the schema and `Usage_Quotas.md`, which Task 6 updates).

- [ ] **Step 4: Run the tests and the audio suites**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Usage/test_audio_stream_slots.py tldw_Server_API/tests/Audio/test_ws_transcribe_stream_slot_release.py tldw_Server_API/tests/Usage/test_audio_quota_per_user.py tldw_Server_API/tests/Usage/test_usage_quotas_audio.py tldw_Server_API/tests/Audio/test_stream_limits_endpoint.py`

Expected: `17 passed` (measured).

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -n 6 tldw_Server_API/tests/Audio tldw_Server_API/tests/AudioJobs tldw_Server_API/tests/Usage`

Expected: all pass except `tests/Audio/test_nemo_streaming_error_sentinel.py::test_nemo_streaming_suppresses_error_sentinel` (`No module named 'pyaudio'`, which also fails on `origin/dev`). With every task applied the prototype measured `1 failed, 1878 passed, 4 skipped`; Task 4 adds the only two of those tests that don't exist yet, so expect `1 failed, 1876 passed, 4 skipped` here. Without the idle-test stub above, `test_ws_idle_metrics_audio.py` failed one run in this suite too.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Usage/audio_quota.py tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py tldw_Server_API/app/api/v1/schemas/audio_schemas.py tldw_Server_API/tests/Usage/test_audio_stream_slots.py tldw_Server_API/tests/Audio/test_ws_transcribe_stream_slot_release.py tldw_Server_API/tests/Usage/test_audio_quota_per_user.py tldw_Server_API/tests/Audio/test_ws_idle_metrics_audio.py
git commit -m "feat(quotas): audio streams hold a limits.audio_concurrent_streams slot; /stream/transcribe takes it inside the try that releases it (TASK-13435)

/audio/stream/limits now says that can_start_stream and active_streams are
not live: open streams are not counted.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 4: Direct transcription holds a slot

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py`: an import after line 60, a constant before `router = APIRouter(` (line 75), the acquire before the outer `try:` (after line 689), and the heartbeat cancel and the release as the last statements of the outer `finally:` (lines 1393-1409) (Review B1, I3).
- Test: `tldw_Server_API/tests/Audio/test_transcription_concurrency_slot.py` (new)

**Interfaces:**
- Consumes: `concurrency_quota.acquire_slot`/`ConcurrencyQuotaExceeded`, and `quota_checks.rate_limit_headers(limit, remaining, reset_seconds) -> dict[str, str]`.
- Produces: `audio_transcriptions.TRANSCRIPTIONS_KEY = "limits.audio_concurrent_transcriptions"`. At the limit, `POST /audio/transcriptions` and `/audio/translations` answer 429 `Concurrent transcription limit reached (N)` with `RateLimit-*`, `X-RateLimit-*` and `Retry-After` headers (none when N is 0).
- No request leaves its `_hb_loop` job-heartbeat task running: the 429 branch cancels it, and so does the outer `finally`, which also closes the existing leak on the 413, 400 and WAV-conversion paths (Review I3).

The existing `can_start_job`, job-heartbeat, `increment_jobs_started` and `finish_job` calls in `create_transcription` stay. They are no-ops on this path, and other tests patch and assert on them (`test_audio_transcriptions_hotwords.py::test_audio_transcriptions_sanitizes_heartbeat_jobs_failure_log`, among others). The new slot doesn't use them.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Audio/test_transcription_concurrency_slot.py`. `_heartbeat_loops` runs on the app's event loop through the TestClient's portal and lists the `_hb_loop` tasks still running (Review I3). The 413 test asserts on the response body, not only on status codes, so the ratchet passes (Review I1).

```python
"""POST /audio/transcriptions holds a per-user slot (limits.audio_concurrent_transcriptions) for the whole request (TASK-13435)."""

import asyncio
import io
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor, RGRequest
from tldw_Server_API.app.core.Usage import concurrency_quota

pytestmark = pytest.mark.unit
TEST_API_KEY = "test-api-key-1234567890"
KEY = "limits.audio_concurrent_transcriptions"


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, bypass_api_limits: Any):
    """The audio router for single user 1, a limit of 1 transcription at a time, and a fresh lease store."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", TEST_API_KEY)
    import tldw_Server_API.app.api.v1.endpoints.audio.audio as audio_ep
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _user() -> User:
        """The single-user principal."""
        return User(id=1, username="single_user")

    async def _one_at_a_time(_user_id: int, key: str) -> object:
        """A limit of 1 for direct transcription; every other key unset."""
        return 1 if key == KEY else None

    store = MemoryResourceGovernor(policy_loader=concurrency_quota._QuotaPolicies())
    monkeypatch.setattr(concurrency_quota, "user_quota", _one_at_a_time)
    monkeypatch.setattr(concurrency_quota, "_governor", store)
    app = FastAPI()
    app.dependency_overrides[get_request_user] = _user
    app.include_router(audio_ep.router, prefix="/api/v1/audio")
    with bypass_api_limits(app), TestClient(app) as test_client:
        test_client.store = store
        yield test_client


def _post(client: TestClient):
    """Upload a small file for transcription."""
    return client.post(
        "/api/v1/audio/transcriptions",
        headers={"X-API-KEY": TEST_API_KEY},
        files={"file": ("sample.wav", io.BytesIO(b"RIFF" + b"\0" * 64), "audio/wav")},
        data={"model": "external:stub", "response_format": "json"},
    )


async def _heartbeat_loops() -> list[asyncio.Task]:
    """The request job-heartbeat loops still running on the app's event loop."""
    return [task for task in asyncio.all_tasks() if task.get_coro().__name__ == "_hb_loop"]


def test_refused_with_429_while_the_users_slot_is_held(client: TestClient) -> None:
    """With the user's one slot taken by another request, a transcription is refused with 429 and Retry-After."""
    held = RGRequest(
        entity="user:1",
        categories={"jobs": {"units": 1, "max_concurrent": 1, "ttl_sec": 120}},
        tags={"policy_id": KEY},
    )
    decision, _handle = asyncio.run(client.store.reserve(held))
    assert decision.allowed
    resp = _post(client)
    assert resp.status_code == 429, resp.text
    assert "Concurrent transcription limit reached (1)" in resp.text
    assert int(resp.headers["Retry-After"]) >= 1
    assert client.portal.call(_heartbeat_loops) == []


def test_a_failed_request_gives_its_slot_back(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request that fails after admission (413 here) releases its slot: the next one fails on its own merits, not 429."""
    import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_tx

    monkeypatch.setattr(audio_tx, "_max_upload_bytes", lambda _limits: 8)
    first, second = _post(client), _post(client)
    assert (first.status_code, second.status_code) == (413, 413)
    assert second.json()["detail"] == "File size exceeds maximum of 0MB"
    assert "Concurrent transcription limit" not in second.text
    assert client.portal.call(_heartbeat_loops) == []
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Audio/test_transcription_concurrency_slot.py`

Expected (measured with Tasks 1–3 applied): `2 failed`.
- `test_refused_with_429_while_the_users_slot_is_held` fails with `assert 400 == 429`: the request reached WAV conversion.
- `test_a_failed_request_gives_its_slot_back` gets both 413s with the upload-size detail, then fails on the heartbeat assertion: `assert [<Task pending name=... coro=<create_transcription.<locals>._maybe_start_job_heartbeat.<locals>._hb_loop() ...>, ...] == []`. That is the existing leak on the 413 path (Review I3). Its slot assertions pass now and pin the release once the slot exists.

- [ ] **Step 3: Implement**

Make these changes in `audio_transcriptions.py`.

1. After `from tldw_Server_API.app.core.testing import is_test_mode` (line 60), add:

```python
from tldw_Server_API.app.core.Usage import concurrency_quota, quota_checks
```

2. Directly above `router = APIRouter(` (line 75), add:

```python
TRANSCRIPTIONS_KEY = "limits.audio_concurrent_transcriptions"

```

3. After the policy-resolution block, which ends with these lines (686-689):

```python
    except Exception:
        _emit_error_metrics(status_label="internal_error", reason="policy_resolution_failed")
        raise
```

and directly above the outer `try:` that begins `first_chunk = await file.read(upload_chunk_size)`, insert:

```python
    # One of the user's direct-transcription slots, held until the finally below (TASK-13435).
    try:
        transcription_slot = await concurrency_quota.acquire_slot(
            current_user.id, TRANSCRIPTIONS_KEY, "jobs"
        )
    except concurrency_quota.ConcurrencyQuotaExceeded as exc:
        if job_heartbeat_task:
            job_heartbeat_task.cancel()
        _emit_error_metrics(status_label="quota_exceeded", reason="quota")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Concurrent transcription limit reached ({exc.limit})",
            headers=quota_checks.rate_limit_headers(exc.limit, 0, exc.retry_after) if exc.retry_after else None,
        ) from exc
```

The metric labels (`quota_exceeded`/`quota`) are the ones the endpoint already uses for its other 429s (lines 1362-1363). The heartbeat started at lines 660-663 is cancelled before the 429 is raised; nothing else would ever cancel it (Review I3).

4. End the outer `finally:` (lines 1393-1409) with the heartbeat cancel and then the release, after the file cleanup. Every statement before them is already guarded, so the release always runs, and a release error can no longer skip the temp-file removal (Review B1, I3). Before (the last two lines of that `finally:`):

```python
                except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed (audio tempfile_remove_failed)")
```

After:

```python
                except _AUDIO_TRANSCRIPTIONS_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed (audio tempfile_remove_failed)")
        if job_heartbeat_task:
            job_heartbeat_task.cancel()
        # Last, so every cleanup above has run whatever the release does (TASK-13435).
        await transcription_slot.release()
```

`cancel()` on a task the inner `finally` already cancelled is a no-op.

- [ ] **Step 4: Run the tests and the transcription suites**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Audio/test_transcription_concurrency_slot.py`

Expected: `2 passed`.

Then show each new line is pinned. Make each change below, one at a time, rerun, then restore it; each must make exactly the named test fail (measured, `1 failed, 1 passed` each):
- `await transcription_slot.release()` → `pass`: `test_a_failed_request_gives_its_slot_back` fails on the second POST (429).
- Delete the two heartbeat-cancel lines in the 429 branch: `test_refused_with_429_while_the_users_slot_is_held` fails on the heartbeat assertion.
- Delete the two heartbeat-cancel lines at the end of the outer `finally:`: `test_a_failed_request_gives_its_slot_back` fails on the heartbeat assertion.

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -n 6 tldw_Server_API/tests/Audio tldw_Server_API/tests/AudioJobs tldw_Server_API/tests/Usage`

Expected: `1 failed, 1878 passed, 4 skipped` (measured); the one failure is the `pyaudio` test.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py tldw_Server_API/tests/Audio/test_transcription_concurrency_slot.py
git commit -m "feat(quotas): direct transcription holds a limits.audio_concurrent_transcriptions slot for the whole request (TASK-13435)

The job-heartbeat task is now cancelled on the new 429 and at the end of the
outer finally, so refused and failed requests no longer leave it running.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 5: Media ingest holds a slot

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`: the import (line 83), a constant before `_enforce_and_record_media_bytes` (line 278), the acquire directly above the `try:` in `add_media_orchestrate` (line 2843), and the release as the last statement of its `finally:` (lines 3539-3548) (Review B1).
- Test: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_cancellation_and_quota.py` (append; it reuses the file's `_patch_upload`, `_orchestrate`, `_BlockingQuotaService` and `_RejectingQuotaService`).
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py`: the `offline_client` fixture (lines 53 and 127) (Review I2).

**Interfaces:**
- Consumes: `concurrency_quota.acquire_slot`/`ConcurrencyQuotaExceeded`, and `quota_checks.rate_limit_headers`.
- Produces: `persistence.MEDIA_CONCURRENT_KEY = "limits.media_concurrent_jobs"`. At the limit, `/media/add` answers 429 `Media ingestion concurrency limit reached.`, the pre-spec-2 message, with `RateLimit-*`, `X-RateLimit-*` and `Retry-After` headers. The refusal comes before any work.

- [ ] **Step 1: Write the failing tests**

Append to `test_persistence_cancellation_and_quota.py`:

```python


@pytest.fixture()
def one_ingest_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """limits.media_concurrent_jobs = 1 for every user, with a fresh in-memory lease store (TASK-13435)."""
    from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
    from tldw_Server_API.app.core.Usage import concurrency_quota

    async def _one(_user_id: int, key: str) -> object:
        """A limit of 1 for concurrent media ingests; every other key unset."""
        return 1 if key == "limits.media_concurrent_jobs" else None

    monkeypatch.setattr(concurrency_quota, "user_quota", _one)
    monkeypatch.setattr(concurrency_quota, "_governor", MemoryResourceGovernor(policy_loader=concurrency_quota._QuotaPolicies()))


@pytest.mark.asyncio
async def test_second_ingest_is_refused_while_the_first_runs_and_admitted_once_it_is_cancelled(
    monkeypatch: pytest.MonkeyPatch, one_ingest_at_a_time: None
) -> None:
    """At limit 1 a second /media/add gets 429 before any work; cancelling the first frees the slot."""
    quota_service = _BlockingQuotaService()
    _patch_upload(monkeypatch, quota_service)
    first = asyncio.create_task(_orchestrate())
    await asyncio.wait_for(quota_service.entered.wait(), timeout=10)
    processed = _patch_upload(monkeypatch, _RejectingQuotaService())

    with pytest.raises(HTTPException) as refused:
        await _orchestrate()
    assert refused.value.status_code == 429
    assert refused.value.detail == "Media ingestion concurrency limit reached."
    assert int(refused.value.headers["Retry-After"]) >= 1

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, timeout=10)
    with pytest.raises(HTTPException) as after_cancel:
        await _orchestrate()
    assert after_cancel.value.status_code == 413
    assert processed == []


@pytest.mark.asyncio
async def test_a_finished_ingest_gives_its_slot_back(monkeypatch: pytest.MonkeyPatch, one_ingest_at_a_time: None) -> None:
    """At limit 1, two ingests one after the other both run: a successful request releases its slot."""

    class _AdmittingQuotaService:
        """A storage quota with room for everything."""

        async def check_quota(self, user_id: int, new_bytes: int, raise_on_exceed: bool = False):
            """Admit the upload."""
            return True, {"current_usage_mb": 0, "new_size_mb": 1, "quota_mb": None, "available_mb": None}

    processed = _patch_upload(monkeypatch, _AdmittingQuotaService())
    await _orchestrate()
    await _orchestrate()
    assert len(processed) == 2


@pytest.mark.asyncio
async def test_a_rejected_ingest_gives_its_slot_back(monkeypatch: pytest.MonkeyPatch, one_ingest_at_a_time: None) -> None:
    """An ingest that fails after admission (413) releases its slot: the next one fails on its own merits, not 429."""
    _patch_upload(monkeypatch, _RejectingQuotaService())
    for _ in range(2):
        with pytest.raises(HTTPException) as exc_info:
            await _orchestrate()
        assert exc_info.value.status_code == 413
```

In `tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py`, keep the slot lookup out of the `offline_client` harness, next to its other quota stubs (Review I2). After `from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib` (line 53) add:

```python
    from tldw_Server_API.app.core.Usage import concurrency_quota
```

and after `monkeypatch.setattr(persistence, "_enforce_and_record_media_bytes", AsyncMock())` (line 127) add:

```python
    monkeypatch.setattr(concurrency_quota, "user_quota", AsyncMock(return_value=None))
```

Without it, `test_offline_upload_does_not_enter_authnz_pool` fails once the slot exists (`Called 1 times`), because the slot's `user_quota` lookup opens the AuthNZ pool. Skipping the slot for `email_request` instead would be a product change; this plan does not make it.

- [ ] **Step 2: Run them and confirm the first fails**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_cancellation_and_quota.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py`

Expected (measured with Tasks 1–4 applied): `1 failed, 31 passed`. The cancel test fails with `assert 413 == 429`: the second request ran instead of being refused. The success and 413 tests pass now and pin the release once the slot exists. All 25 email tests pass.

- [ ] **Step 3: Implement**

Make these changes in `persistence.py`.

1. Line 83 becomes:

```python
from tldw_Server_API.app.core.Usage import concurrency_quota, quota_checks
```

2. Directly above `async def _enforce_and_record_media_bytes(` (line 278), add:

```python
MEDIA_CONCURRENT_KEY = "limits.media_concurrent_jobs"


```

3. In `add_media_orchestrate`, between `loop = asyncio.get_running_loop()` and the `try:` whose first line is `# --- 3. Setup Temporary Directory ---`, insert:

```python
    # One of the user's media-ingest slots, held until the finally below (TASK-13435).
    try:
        ingest_slot = await concurrency_quota.acquire_slot(
            getattr(current_user, "id", None), MEDIA_CONCURRENT_KEY, "jobs"
        )
    except concurrency_quota.ConcurrencyQuotaExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Media ingestion concurrency limit reached.",
            headers=quota_checks.rate_limit_headers(exc.limit, 0, exc.retry_after) if exc.retry_after else None,
        ) from exc

```

4. Make `await ingest_slot.release()` the last statement of that `try`'s `finally:` (lines 3539-3548), after the two metric calls, which swallow their own errors (Review B1). Before (the end of that `finally:`):

```python
            processor="media_add_orchestrate",
            duration_seconds=time.monotonic() - request_started_at,
        )
```

After:

```python
            processor="media_add_orchestrate",
            duration_seconds=time.monotonic() - request_started_at,
        )
        # Last, so the metrics above are recorded whatever the release does (TASK-13435).
        await ingest_slot.release()
```

The `processor="media_add_orchestrate",` line also appears at line 2650 with `duration_seconds=0.0`; the three lines above match only once.

- [ ] **Step 4: Run the tests and the media unit suite**

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_cancellation_and_quota.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py`

Expected: `32 passed` (7 + 25, measured).

Then show the release and the email stub are pinned. Make each change below, one at a time, rerun the same two files, then restore it (measured):
- `await ingest_slot.release()` → `pass`: the three new release tests fail (`3 failed, 29 passed`).
- Delete the `concurrency_quota.user_quota` stub from `offline_client`: `test_offline_upload_does_not_enter_authnz_pool` fails (`Called 1 times`; `1 failed, 31 passed`) (Review I2).

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -n 6 tldw_Server_API/tests/MediaIngestion_NEW/unit tldw_Server_API/tests/UserProfile`

Expected: `951 passed, 8 skipped, 1 xfailed` (measured with every task applied).

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_cancellation_and_quota.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py
git commit -m "feat(quotas): /media/add holds a limits.media_concurrent_jobs slot for the whole request (TASK-13435)

The offline email harness stubs the slot lookup, as it stubs every other
quota path, so it still never touches AuthNZ.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 6: ADR-076 and the operator docs

**Files:**
- Create: `Docs/ADR/076-per-user-concurrency-quotas.md`
- Modify: `Docs/ADR/README.md` (an index row after ADR-071, and forward pointers on the ADR-056 and ADR-064 rows, lines 90 and 95) (Review M12)
- Modify: `Docs/Operations/Usage_Quotas.md`, `Docs/Operations/Env_Vars.md` (lines 325, 327 and 341), `Docs/Operations/Rate_Limits_Troubleshooting.md`
- Modify: `Docs/API-related/Audio_Transcription_API.md` (lines 509-510) (Review M5)
- Regenerate: `Docs/Published/**` via `bash Helper_Scripts/refresh_docs_published.sh`

**Interfaces:** none (documentation). The key names, messages and behaviour must match Tasks 2–5 exactly.

- [ ] **Step 1: Write the ADR**

Write `Status: Accepted` and the index row's `Accepted`: the owner answered D1, D2 and D3 and accepted the ADR on 2026-10-18. The ADR records the counter-argument to D1 under **Switches** (Review M13), the coverage of `limits.media_concurrent_jobs` under **Keys** (Review M11), `Retry-After` (Review M8) and the lease-store failure modes (Review M7, B1).

`Docs/ADR/076-per-user-concurrency-quotas.md`:

```markdown
# ADR-076: Per-user concurrency quotas are usage quotas held as Resource Governor leases

**Status:** Accepted
**Date:** 2026-10-18
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13435
**Related spec/plan:** `Docs/Design/2026-10-02-usage-quota-posture-design.md` (Non-goals, deferred), `Docs/superpowers/plans/2026-10-18-per-user-concurrency-quotas.md`
**Amends:** ADR-064 (synchronous concurrency is now a usage quota) and ADR-056 (`RG_ENABLED` no longer covers media and audio concurrency)

## Decision

Per-user synchronous concurrency (media ingest, audio streams, direct transcription) is a usage quota: a `limits.*` value resolved and switched like every other (ADR-064), enforced as one lease held for the life of each request in a Resource Governor lease store that the request carries its own limit into, built from `RG_BACKEND` and not turned off by `RG_ENABLED`.

## Context

Spec 2 stopped the media and audio sites reserving `jobs` and `streams` leases from RG policies, because those were defaults nobody had assigned. That left synchronous concurrency unlimited, and the per-user limit was deferred because `RGRequest` could not carry one (TASK-13435). The owner's commercial offering needs it per user, team or org, blank by default. The governors read a lease limit only from the policy's category block; the Redis backend released a lease only when that block existed; and a policy listing `global` charged every lease to the server-wide bucket as well.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| A request-level `RGRequest.max_concurrent` field | One value for a request that can carry several categories, and no place for the lease TTL. Per-category `max_concurrent`/`ttl_sec` in `RGRequest.categories` mirror how policies already spell the limit. |
| Reserve on the app's governor (`app.state.rg_governor`) | It is `None` when `RG_ENABLED` is off, so an assigned limit would silently stop applying; the operator's policy store could add a `global` bucket; the quota keys are not policies, so lookups log "policy not defined". |
| Encode the limit in the policy id, with no governor change | Changing a user's limit opens a new, empty bucket, so the user briefly runs the old plus the new limit. |
| A separate lease store (a dict plus a Redis ZSET) outside RG | Duplicates the TTL purge, the atomic Lua acquire and the retry-after logic RG already runs on both backends. |
| Lease TTL only, no renewal | A long ingest or stream outlives a short TTL and stops counting; a long TTL locks a user out that long after a crash. |
| Turn the limits off with `RG_ENABLED` | Every other `limits.*` quota works with RG off (ADR-064); an assigned limit would silently stop applying. |

## Consequences

- **Keys:**
  - `limits.media_concurrent_jobs` covers `/media/add`, including its research-discovery handoff. It does not count queued ingest jobs (`/media/ingest/jobs`) or `/media/process-*`.
  - `limits.audio_concurrent_streams` covers the audio WebSockets, the realtime sockets and `POST /audio/chat`.
  - `limits.audio_concurrent_transcriptions` covers `/audio/transcriptions` and `/audio/translations`.
  - Unset is unlimited and costs nothing; `0` blocks.
- **The request carries the limit.** A `streams`/`jobs` entry in `RGRequest.categories` may carry `max_concurrent` and `ttl_sec`. They replace the policy's values and charge only the caller's bucket, on both backends (`policy_eval.lease_limits`); renewing such a lease refreshes only that bucket.
- **Memory vs Redis.** On the memory backend a limit holds per worker process, so N workers admit up to N times the limit, as for every memory-backend limit (ADR-056). `RG_BACKEND=redis` makes it exact across workers.
- **Release and renewal.** A request releases its lease as the last step of its `finally`, on success, error, cancellation and graceful shutdown, shielded from the request's cancel scope. A held lease is renewed every 30 s while its task runs.
- **Unreleased leases lapse within 120 s.** That covers a killed worker, a missed release, a cancellation that lands after the store granted the lease, and an event loop blocked for two minutes. The worst case is a two-minute lockout of one slot, or briefly one extra admission.
- **Retry-After is the lease TTL.** A refusal's `Retry-After` is when the oldest held lease would lapse without renewal, up to 120 s. A slot usually frees sooner, when a request finishes.
- **Lease-store failures.** If Redis is unreachable when a worker's lease store first connects, that worker counts on its own for the life of the process, as on the memory backend. If Redis drops later, a reserve that fails admits the request, and a renewal or release that fails is logged and the lease lapses within 120 s. Redis errors never fail the request.
- **Switches.** `RG_ENABLED=false` no longer turns off media or audio concurrency limits; `USAGE_QUOTAS_ENABLED` does. The cost: the lease store is a governor that `RG_ENABLED` does not remove, which ADR-056 had ruled out for RG enforcement. With RG off and `RG_BACKEND=redis`, a worker connects to Redis once a user has a value set.
```

If the owner chose the D1 alternative, replace the Decision's last clause with "and turned off, like the rest of RG, by `RG_ENABLED`". Then delete the last Alternatives row and the last Consequences bullet, and drop "and ADR-056 ..." from **Amends**. If D2 was (a) or (b), edit the Keys bullet to match. If D3 was the rename, use the new key name throughout.

In `Docs/ADR/README.md`, add this row after the ADR-071 row:

```markdown
| [ADR-076](076-per-user-concurrency-quotas.md) | Accepted | Per-user synchronous concurrency (media ingest, audio streams, direct transcription) is a `limits.*` usage quota held as one Resource Governor lease per request; the request carries its own limit, the lease store follows `RG_BACKEND`, and `RG_ENABLED` does not turn it off. |
```

Then give the ADR-056 and ADR-064 index rows a forward pointer; their ADR files stay as they are (Review M12). In the ADR-056 row (line 90), replace `and one switch disables enforcement everywhere. |` with:

```markdown
and one switch disables enforcement everywhere. Amended in part by [ADR-076](076-per-user-concurrency-quotas.md): `RG_ENABLED` no longer covers per-user media and audio concurrency. |
```

In the ADR-064 row (line 95), replace `then the most generous org value; none is set by default. |` with:

```markdown
then the most generous org value; none is set by default. Amended in part by [ADR-076](076-per-user-concurrency-quotas.md): synchronous concurrency is now a usage quota. |
```

- [ ] **Step 2: Update `Docs/Operations/Usage_Quotas.md`**

The rows and the section state what `limits.media_concurrent_jobs` does not cover (Review M11), each socket's actual refusal (Review M6), what `Retry-After` means (Review M8), and what a Redis outage does (Review M7).

1. In "The `limits.*` keys" table, add these rows after the `limits.chatbooks_concurrent_jobs` row:

```markdown
| `limits.media_concurrent_jobs` | Media ingest requests running at once: `/media/add`, including its research-discovery handoff. Queued ingest jobs (`/media/ingest/jobs`) and `/media/process-*` are not counted | One lease per running request (see "Concurrency limits" below) | HTTP 429 `Media ingestion concurrency limit reached.`, with `RateLimit-*`, `X-RateLimit-*` and `Retry-After` headers |
| `limits.audio_concurrent_streams` | Audio streaming sessions open at once: `/audio/stream/transcribe`, `/audio/chat/stream`, `/audio/stream/tts`, `/audio/stream/tts/realtime`, the realtime speech sockets, and `POST /audio/chat` | The same leases | Refused when the stream starts. `/audio/stream/transcribe`: an error frame `Concurrent streams limit reached (N)`, then close code 1000. `/audio/chat/stream`, `/audio/stream/tts` and `/audio/stream/tts/realtime`: the same message in an error frame, then close code 4003 (1008 with `AUDIO_WS_QUOTA_CLOSE_1008`). The realtime speech sockets: close code 4003 before the socket is accepted, with no error frame. `POST /audio/chat`: HTTP 429 with the same message |
| `limits.audio_concurrent_transcriptions` | Direct transcriptions running at once (`/audio/transcriptions`, `/audio/translations`) | The same leases | HTTP 429 `Concurrent transcription limit reached (N)`, with `RateLimit-*`, `X-RateLimit-*` and `Retry-After` headers |
```

2. Directly before "Two other `limits.*` entries in the catalog are not part of this scheme:", add:

```markdown
### Concurrency limits

`limits.media_concurrent_jobs`, `limits.audio_concurrent_streams` and `limits.audio_concurrent_transcriptions` count requests running at once, not usage over time.

- Each admitted request holds one lease until it ends: on success, on error, when the client disconnects, and when the server shuts down gracefully. A refused request holds nothing.
- Leases live in the Resource Governor's lease store, chosen by `RG_BACKEND`. On `memory` (the default) a limit holds per worker process, so with N workers a user can run up to N times the limit. On `redis` it is exact across workers.
- `RG_ENABLED=false` does not turn these limits off; `USAGE_QUOTAS_ENABLED` does (ADR-076).
- A lease nobody releases, for example when a worker is killed mid-request, expires within 2 minutes. A crash can hold one of the user's slots for up to that long.
- `Retry-After` on a refusal is when the oldest held lease would expire if it stopped renewing, up to 2 minutes. A slot usually frees sooner, when one of the user's requests finishes, so a client may retry earlier.
- Lowering a limit does not stop running requests. New ones are refused until the user is below the new limit.
- If Redis is unreachable when a worker's lease store first connects, that worker counts on its own for the life of the process, as on `memory`. If Redis drops later, requests are admitted, a warning is logged, and leases the worker could not release expire within 2 minutes.

```

3. In "What is not a quota", delete the bullet `- **Synchronous concurrency**: media ingest requests, audio streams and direct transcription have no per-user cap.`

4. In "Where users see their limits", replace the `GET /api/v1/audio/stream/limits` row's last sentence, "`active_streams` is `null` because per-user stream concurrency is not tracked", with: "`limits.concurrent_streams` is the `limits.audio_concurrent_streams` value. `active_streams` stays `null` and `can_start_stream` stays `true`: open streams are not counted, and a stream over the limit is refused when it starts" (Review M5).

- [ ] **Step 3: Update `Docs/Operations/Env_Vars.md`**

1. Line 325, the keys list: after `` `limits.chatbooks_concurrent_jobs`, `` insert `` `limits.media_concurrent_jobs` (media ingests at once), `limits.audio_concurrent_streams` (audio streams at once), `limits.audio_concurrent_transcriptions` (direct transcriptions at once), ``.
2. Line 327: delete the sentence `Synchronous concurrency (media ingest requests, audio streams, direct transcription) is not limited per user.`. Then add this bullet after that line:

```markdown
- The three concurrency keys hold one lease per running request. With `RG_BACKEND=memory` (the default) a limit holds per worker process; `RG_BACKEND=redis` makes it exact across workers. `RG_ENABLED` does not turn them off (ADR-076).
```

3. Line 341 (`RG_ENABLED`): change `evaluations/audio concurrency, the workflows daily cap, media job concurrency, and the general` to `evaluations concurrency, the workflows daily cap, and the general`. At the end of that bullet, add: ` Per-user concurrency quotas (`limits.media_concurrent_jobs`, `limits.audio_concurrent_streams`, `limits.audio_concurrent_transcriptions`) are usage quotas and stay on with RG disabled; `USAGE_QUOTAS_ENABLED` governs them (ADR-076).`

- [ ] **Step 4: Update `Docs/Operations/Rate_Limits_Troubleshooting.md`**

1. Add these rows after the Chatbooks row. The streams row names the realtime sockets' frameless 4003 close (Review M6):

```markdown
| `Media ingestion concurrency limit reached.`, HTTP 429 | Per-user concurrent media ingests | `Ingestion_Media_Processing/persistence.py` (`add_media_orchestrate`), `Usage/concurrency_quota.py` | Raise or remove the user's `limits.media_concurrent_jobs`, or wait for one of their ingests to finish. Only applies when `USAGE_QUOTAS_ENABLED` is on. |
| `Concurrent transcription limit reached (N)`, HTTP 429 | Per-user concurrent direct transcriptions | `api/v1/endpoints/audio/audio_transcriptions.py` (`create_transcription`) | Raise or remove the user's `limits.audio_concurrent_transcriptions`. Only applies when `USAGE_QUOTAS_ENABLED` is on. |
| `Concurrent streams limit reached (N)` in a WebSocket error frame, then a close; a realtime speech socket closing with 4003 before it is accepted; HTTP 429 on `POST /audio/chat` | Per-user concurrent audio streams | `Usage/audio_quota.py` (`can_start_stream`) | Raise or remove the user's `limits.audio_concurrent_streams`, or close another of the user's streams. Only applies when `USAGE_QUOTAS_ENABLED` is on. |
```

2. In "Memory-backend buckets are per worker process", add after the first paragraph:

```markdown
The per-user concurrency quotas (`limits.media_concurrent_jobs`, `limits.audio_concurrent_streams`, `limits.audio_concurrent_transcriptions`) follow the same rule: with N workers on the memory backend, a user can run up to N times the limit.
```

- [ ] **Step 5: Update `Docs/API-related/Audio_Transcription_API.md`**

Its "Auth & Close Codes" quota bullets say concurrent-stream violations emit `quota: concurrent_streams` and close 4003, and that "Redis is used when available"; neither is true of the new slot (Review M5). Replace these two lines (509-510):

```markdown
  - Concurrent streams and daily minutes enforced per user; Redis is used when available for cross-process counters; otherwise in-process.
  - On quota violations, the server emits `{ "type":"error", "error_type":"quota_exceeded", "quota":"daily_minutes|concurrent_streams" }` and closes with code `4003`.
```

with:

```markdown
  - Daily minutes and concurrent streams are per-user usage quotas (`limits.audio_daily_minutes`, `limits.audio_concurrent_streams`), enforced only when `USAGE_QUOTAS_ENABLED` is on and a value is set. Open streams are counted across workers with `RG_BACKEND=redis`; with `memory` (the default) each worker process counts its own.
  - On a daily-minutes violation, the server emits `{ "type":"error", "error_type":"quota_exceeded", "quota":"daily_minutes" }` and closes with code `4003`. A stream over the concurrent-streams limit is refused when it starts; "Concurrency limits" in `Docs/Operations/Usage_Quotas.md` gives each socket's error frame and close code.
```

The close-code list below it (`4003` for daily minutes and concurrent streams) stays: four of the six stream sites close with 4003.

- [ ] **Step 6: Refresh the mirror and run the docs tests**

```bash
bash Helper_Scripts/refresh_docs_published.sh
git status --short Docs/Published
```

Expected: `Docs/Published/ADR/076-per-user-concurrency-quotas.md` (new), and `Docs/Published/ADR/README.md`, `Docs/Published/Env_Vars.md` and `Docs/Published/API-related/Audio_Transcription_API.md` changed (measured).

`test_refresh_output_matches_tracked_published_files` compares the refresh output with `git ls-files Docs/Published`, so stage the two new ADR files before the docs tests (Review M2):

```bash
git add Docs/ADR/076-per-user-concurrency-quotas.md Docs/Published/ADR/076-per-user-concurrency-quotas.md
```

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs`

Expected: `212 passed` (measured). `test_docs_published_refresh.py` fails if the mirror is stale or a new mirror file is unstaged.

Check that no trailing whitespace was introduced in the docs or the plan:

```bash
git diff --check origin/dev -- Docs
```

Expected: no output (measured).

- [ ] **Step 7: Commit**

```bash
git add Docs/ADR/076-per-user-concurrency-quotas.md Docs/ADR/README.md Docs/Operations/Usage_Quotas.md Docs/Operations/Env_Vars.md Docs/Operations/Rate_Limits_Troubleshooting.md Docs/API-related/Audio_Transcription_API.md Docs/Published
git commit -m "docs(quotas): ADR-076 and the operator docs for per-user concurrency quotas (TASK-13435)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

### Task 7: Verify, close the task, open the PR

**Files:** `backlog/tasks/task-13435 ...` and the follow-up task file (Step 3); no code.

- [ ] **Step 1: Rebase check and full sweep**

The sweep includes `tests/CI/test_test_quality_ratchet.py`, the required test-quality ratchet (Review I1):

```bash
git fetch origin dev --quiet && git rebase origin/dev
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -n 6 \
  tldw_Server_API/tests/Resource_Governance tldw_Server_API/tests/Usage tldw_Server_API/tests/Audio tldw_Server_API/tests/AudioJobs \
  tldw_Server_API/tests/UserProfile tldw_Server_API/tests/MediaIngestion_NEW tldw_Server_API/tests/Docs tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py \
  tldw_Server_API/tests/CI/test_test_quality_ratchet.py
redis-cli --scan --pattern 'rg_rcl_*' | wc -l
```

Expected: all pass, except `test_nemo_streaming_error_sentinel.py` (`pyaudio`). The prototype on `32eb11ae03` measured `1 failed (pyaudio), 3774 passed, 16 skipped, 3 xfailed`, with real Redis; `test_offline_upload_does_not_enter_authnz_pool` and `test_audio_ws_idle_timeout_increments_metric` pass. `redis-cli` prints `0` (Review M10). Prove any other failure on a clean detached `origin/dev` worktree, or fix it. Record the counts, and whether the `[redis]` cases ran against a real Redis.

The global-default rule: no global default changes here. Every key is unset by default and the switch is untouched, so no repo-wide `git grep` sweep is needed beyond the suites above. The hook names are unchanged, so every test that patches `can_start_stream`/`finish_stream` is in `tests/Audio` or `tests/Usage`, verified by `git grep -l "can_start_stream\|finish_stream" tldw_Server_API/tests`.

- [ ] **Step 2: Static checks**

The first ruff command lists the touched RG files, not the directory: `Resource_Governance/tenant.py` has a pre-existing I001 on `origin/dev` (Review M1).

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check tldw_Server_API/app/core/Usage/concurrency_quota.py tldw_Server_API/app/core/Resource_Governance/policy_eval.py tldw_Server_API/app/core/Resource_Governance/governor.py tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/app/api/v1/schemas/audio_schemas.py tldw_Server_API/tests/Usage/test_concurrency_quota.py tldw_Server_API/tests/Usage/test_audio_stream_slots.py tldw_Server_API/tests/Resource_Governance/test_request_concurrency_limit.py tldw_Server_API/tests/Resource_Governance/integration/test_redis_real_request_concurrency.py tldw_Server_API/tests/Audio/test_transcription_concurrency_slot.py tldw_Server_API/tests/Audio/test_ws_transcribe_stream_slot_release.py
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check --statistics tldw_Server_API/app/core/Usage/audio_quota.py tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_cancellation_and_quota.py tldw_Server_API/tests/Usage/test_audio_quota_per_user.py tldw_Server_API/tests/Audio/test_ws_idle_metrics_audio.py
/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/test_quality_triage.py --enforce
uvx bandit -q -ll tldw_Server_API/app/core/Usage/concurrency_quota.py tldw_Server_API/app/core/Resource_Governance/policy_eval.py tldw_Server_API/app/core/Resource_Governance/governor.py tldw_Server_API/app/core/Resource_Governance/governor_redis.py tldw_Server_API/app/core/Usage/audio_quota.py tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py tldw_Server_API/app/api/v1/schemas/audio_schemas.py tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
```

Expected (measured on the prototype):
- The first ruff command prints `All checks passed!`.
- The second prints `2 I001` and `1 SIM114`, all pre-existing: `audio_transcriptions.py` has one I001 and the SIM114, and `test_ws_idle_metrics_audio.py` keeps one in-function I001. `origin/dev` prints `3 I001` and `1 SIM114` for the same files; Task 3's import edit sorts that test's top-level imports.
- The ratchet exits 0 and logs `[test-triage] OK — no offenses beyond the baseline.` (Review I1). Never fix a failure here by raising `Helper_Scripts/ci/test_quality_baseline.txt`.
- Bandit (not in the venv; `uvx` fetches it) reports exactly one Medium: `policy_eval.py:77 B113`, a false positive on a dict named `requests`. It is unchanged from `origin/dev`.

- [ ] **Step 3: File the RG follow-up (Review B1)**

`concurrency_quota` now catches redis-py's errors, but RG's own `_RG_NONCRITICAL_EXCEPTIONS` (`governor_redis.py`, lines 29-40 on `origin/dev`) still lists only builtins, so its best-effort guards (ingress included) let a `RedisError` through. That is pre-existing and out of this task's scope; file it.

Compute an explicit id with the AGENTS.md recipe, paging every open PR's files: `gh pr list --json files` returns at most 100 files per PR, and on 2026-10-18 nine open PRs were larger.

```bash
git fetch origin dev --quiet
{ git ls-tree -r --name-only origin/dev backlog/
  for n in $(gh pr list --state open --limit 500 --json number --jq '.[].number'); do
    gh api --paginate "repos/rmusser01/tldw_server/pulls/$n/files" --jq '.[].filename'
  done
} | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1
```

Expected: a number such as `13857` (the highest on 2026-10-18). Use the next one, written `<N>` below.

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "RG Redis governor: _RG_NONCRITICAL_EXCEPTIONS misses redis.exceptions.RedisError" \
  --id TASK-<N> \
  -l resource-governance,redis \
  -d "Found in the TASK-13435 plan review (B1). redis-py's ConnectionError and TimeoutError derive from redis.exceptions.RedisError, not from the builtins, so _RG_NONCRITICAL_EXCEPTIONS in Resource_Governance/governor_redis.py (builtins only) does not catch them. Once a worker has connected, a Redis outage surfaces as RedisError through RG's best-effort guards in reserve/renew/commit, ingress included. concurrency_quota catches RedisError itself (TASK-13435); RG does not. Fix: add RedisError (imported where redis is available, as Infrastructure/redis_factory.py does) to _RG_NONCRITICAL_EXCEPTIONS, with tests that raise redis.exceptions.ConnectionError." \
  --ac "_RG_NONCRITICAL_EXCEPTIONS includes redis.exceptions.RedisError when redis is importable" \
  --ac "A test raising redis.exceptions.ConnectionError from the Redis client shows reserve, renew and commit honour the governor's fail mode instead of raising"
git add -A backlog/tasks
git commit -m "docs(backlog): file the RG RedisError follow-up from TASK-13435

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: one new task file, `backlog/tasks/task-<N> - RG-Redis-governor-...md`, in To Do.

- [ ] **Step 4: Close the task**

Fill in the counts from Steps 1–2 and the per-task RED/GREEN counts. The counts measured while planning (on `32eb11ae03`):
- T1: RED 13 failed, GREEN 13 passed.
- T2: RED 1 collection error + 1 failed, GREEN 15 passed (23 with the two catalog files).
- T3: RED 5 failed, GREEN 17 passed.
- T4: RED 2 failed, GREEN 2 passed.
- T5: RED 1 failed, GREEN 32 passed (7 media + 25 email).

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13435 \
  --append-notes "Implemented on feat/per-user-concurrency-quotas (PR #<n>). RED/GREEN: T1 governors 13 failed -> 13 passed (real Redis); T2 slots 1 collection error + catalog 1 failed -> 15 passed + catalog; T3 streams 5 failed -> 17 passed; T4 transcription 2 failed -> 2 passed; T5 media 1 failed -> 32 passed (with the email harness). Release, heartbeat-cancel and RedisError lines each shown to fail their tests when removed. Sweep: <counts>; only failure test_nemo_streaming_error_sentinel (no pyaudio, fails on origin/dev). Test-quality ratchet: OK. Ruff tallies no worse than origin/dev; Bandit -ll: only the pre-existing B113 false positive at policy_eval.py:77. ADR-076 written. Follow-up: TASK-<N> (RG _RG_NONCRITICAL_EXCEPTIONS misses RedisError)." \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Per-user concurrency quotas: limits.media_concurrent_jobs (/media/add), limits.audio_concurrent_streams (audio WebSockets, realtime, POST /audio/chat) and limits.audio_concurrent_transcriptions (/audio/transcriptions, /audio/translations). Unset is unlimited, 0 blocks, resolved per user/team/org (ADR-064). Each request holds one RG lease: an RGRequest streams/jobs entry may now carry max_concurrent/ttl_sec, honored on the memory and Redis backends; the Redis backend now releases leases whose policy has no block and renews only the buckets a lease charged. Slots renew every 30 s while their task runs, release shielded as the last statement of finally on success/error/disconnect/shutdown, and lapse within 120 s if never released. Redis errors never fail a request. The lease store follows RG_BACKEND and is not turned off by RG_ENABLED (ADR-076). Also fixed: /stream/transcribe took its slot 312 lines before the try that released it, and create_transcription leaked its job-heartbeat task on early exits. Known skips: test_nemo_streaming_error_sentinel (no pyaudio). Not done: an active-stream count in /audio/stream/limits (active_streams stays null, can_start_stream stays true); RG's own RedisError gap is TASK-<N>." \
  -s Done
git add -A backlog/tasks
git commit -m "docs(backlog): close TASK-13435

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

If D2 was (b), Task 0 added only the release criterion, which is AC #3, so drop `--check-ac 4` (Review M3).

- [ ] **Step 5: Review, then open the PR**

1. Run `/superpowers:requesting-code-review` in a clean subagent on the branch diff, and address what it finds before the PR merges.
2. Push and open the PR:

```bash
git push -u origin feat/per-user-concurrency-quotas
gh pr create --base dev --title "feat(quotas): per-user concurrency quotas for media ingest, audio streams and direct transcription (TASK-13435)" --body-file <body.md>
```

The body covers:
- the three keys and where each is enforced;
- the RG change, including the Redis release and renew fixes;
- the exit-path table from this plan;
- D1, D2 and D3 as decided;
- the ADR check (ADR-076, a partial amendment of ADR-064 and ADR-056);
- the follow-up task from Step 3;
- the test, ratchet and static-check results.

It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

3. ADR-004: no Change-summary waiver is on record for TASK-13435, so ask the owner for their own `## Change summary` before merge.
4. Merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset.
