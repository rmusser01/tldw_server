# Backend Performance Remediation — Batch 5: Event Loop, Workers, Schedulers & MCP Memory Implementation Plan (2026-10-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Backlog task:** TASK-13517 · **Index:** [2026-10-06-perf-remediation-coordination-index.md](2026-10-06-perf-remediation-coordination-index.md)

**Goal:** Stop the continuous event-loop starvation (~15 workers polling with sync SQLite at 1Hz, sync schedulers every 300s, sync destructive endpoints) and the unbounded MCP in-memory growth.

**Architecture:** Wrap sync work in `asyncio.to_thread` at the worker boundary, collapse per-tick connections, make scheduler rescans diff-based, convert destructive loops to bulk SQL, and bound the MCP metric/token structures. No behavior changes beyond latency; job semantics preserved.

**Tech Stack:** `asyncio.to_thread`, shared connection providers, APScheduler job-args hashing, Prometheus histogram counters, Batch-0 connection counter.

## Global constraints

- Backend only; activate venv; one commit per stage referencing TASK-13517.
- Poll/backoff changes must remain configurable via existing env/config knobs (or add one with a default equal to today's semantics where risky).
- Worker changes must preserve job-claim atomicity (the claim SQL/transaction is not touched — only where it runs and how many connections it costs).
- Bandit: `python -m bandit -r tldw_Server_API/app/core/Jobs tldw_Server_API/app/core/Workflows tldw_Server_API/app/services tldw_Server_API/app/core/MCP_unified -f json -o /tmp/bandit_perf_b5.json`.
- Line numbers from the 2026-10-06 review; re-locate by symbol.

---

## Stage 1: Job-worker acquire tick — off-loop, one connection, no duplicate reconcile

**Goal:** The 1Hz empty-poll tick stops opening 5-7 SQLite connections and running duplicate dependency reconciliation on the event loop.

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/manager.py` (`acquire_next_job` ~6159-6278; duplicate `_reconcile_terminal_dependents` calls ~6199-6202 and ~6246-6252; `_connect` usages inside the tick: `_reconcile_terminal_dependents` ~4326, quota count ~6211, `_recover_expired_processing_jobs` ~5823, main acquire ~6253)
- Modify: worker call sites to thread offload: `tldw_Server_API/app/services/core_jobs_worker.py` (~86-96), `audio_jobs_worker.py` (~186), `reminder_jobs_worker.py` (~51), `connectors_worker.py` (~601), `agent_task_jobs_worker.py` (~83), `audiobook_jobs_worker.py` (~2111)
- Test: `tldw_Server_API/tests/Jobs/test_acquire_tick_efficiency.py` (new)

**Change:**
1. Inside `acquire_next_job`: open **one** connection and pass it through the four sub-steps (extend their signatures with `conn=None` following the existing `conn` convention used elsewhere in the codebase).
2. Delete the second `_reconcile_terminal_dependents(**reconciliation_scope)` call; if recovery may invalidate it, gate the re-run on `recovered > 0` from `_recover_expired_processing_jobs`.
3. Worker loops: `job = await asyncio.to_thread(jm.acquire_next_job, ...)` (it is fully sync).
4. Empty-poll backoff: when no job was acquired, sleep grows 1s → 2s → cap 5s (reset on acquire); make the cap configurable (`JOBS_EMPTY_POLL_MAX_SLEEP_SEC`, default 5). This is a deliberate, documented behavior change — record it in the task notes.

**Success criteria:** `bench_jobs_acquire_tick` (Batch 0) reports `connections_per_empty_tick == 1` (from 5-7) and sub-10ms median; reconcile query count per tick halves.

**Tests:**
- [ ] `test_empty_tick_single_connection` — Batch-0 `count_connections` around one acquire on an empty queue == 1.
- [ ] `test_reconcile_runs_once_per_tick` — statement-prefix counter on the dependency-reconcile query == 1 per tick.
- [ ] `test_backoff_grows_and_resets` — monkeypatched sleep durations sequence [1,2,5,5] then reset after an acquire.
- [ ] Existing Jobs suite green (claim atomicity, recovery, quota tests).

**Status:** Not Started

## Stage 2: SLO gauge service — histograms + cadence + thread

**Goal:** Stop rescanning 24h of completed jobs and re-sorting percentiles in Python every 5 seconds.

**Files:**
- Modify: `tldw_Server_API/app/services/jobs_metrics_service.py` (SLO loop ~271-325)
- Test: `tldw_Server_API/tests/test_jobs_slo_gauges.py` (new)

**Change:** derive P50/P90/P99 from the histogram buckets already maintained by `Jobs/metrics.py` (or, where a true percentile is required, one SQL `percentile_cont`-style query — SQLite: single sorted read of the two latency columns with `LIMIT/OFFSET` windowing at most once per 60s); interval 5s → 60s (config knob `JOBS_SLO_INTERVAL_SEC` default 60); DB access via `asyncio.to_thread`.

**Tests:**
- [ ] `test_slo_loop_no_full_history_scan` — 10k completed jobs; one tick issues bounded queries (row ceiling ≤ 2× bucket count or configured sample), zero full-table `SELECT` statements.
- [ ] `test_gauge_values_within_tolerance` — histogram-derived P50/P90 vs brute-force percentiles on a fixture within 5%.

**Status:** Not Started

## Stage 3: Workflow pause-wait — backoff + heartbeat decoupling

**Goal:** A paused run stops polling the DB 5×/second.

**Files:**
- Modify: `tldw_Server_API/app/core/Workflows/engine.py` (`_wait_if_paused` ~515-529)
- Test: `tldw_Server_API/tests/Workflows/test_pause_wait_backoff.py` (new)

**Change:** poll interval 0.2s → 1.0s growing to 5s cap (config: `WORKFLOWS_PAUSE_POLL_SEC`, default 1.0); heartbeat write only at its TTL cadence (interval/5) rather than every loop; sync DB reads moved to `to_thread` (they already sit in async context). Unpause/cancel detection latency stays ≤ poll interval.

**Tests:**
- [ ] `test_paused_run_polls_at_backoff_cadence` — 10s pause → ≤ 10 polls + ≤ 2 heartbeat writes (mock clock).
- [ ] `test_cancel_detected_within_one_interval` — cancel flag set mid-wait returns promptly.

**Status:** Not Started

## Stage 4: Scheduler rescans — shared enumeration + diff-based re-registration

**Goal:** Seven schedulers stop re-enumerating every user directory and re-adding unchanged APScheduler jobs every 300s.

**Files:**
- Create: `tldw_Server_API/app/services/_scheduler_common.py` — `enumerate_user_ids_cached(ttl: float = 60.0) -> list[str]` (mtime-aware cache over `DatabasePaths.get_user_db_base_dir().iterdir()`) and `job_signature(*args, **kwargs) -> str` (stable hash of job args).
- Modify: `services/workflows_scheduler.py` (~235-408: single `user_id=None` listing for shared backend; skip `remove_job/add_job` and the `update_schedule`/`set_history` writes when the job signature is unchanged), `services/reminders_scheduler.py` (~147-179), `services/scheduled_task_recurring_question_scheduler.py` (~130-209), `services/scheduled_task_automation_scheduler.py` (~315-334), `services/quality_eval_scheduler.py`, `services/claims_alerts_scheduler.py`, `services/reading_digest_scheduler.py`, `app/core/Watchlists/websub.py` (user scan)
- Test: `tldw_Server_API/tests/test_scheduler_rescan_diffing.py` (new)

**Change:** every scheduler uses the shared enumerator; per schedule, compute `job_signature`; if an APScheduler job with the same id and signature exists, skip remove/add and both DB writes. Workflows shared-backend path lists once from one handle instead of per-user-handle full listings.

**Tests:**
- [ ] `test_rescan_skips_unchanged_jobs` — two consecutive rescans: `add_job`/`remove_job` and `update_schedule`/`set_history` call counts == 0 the second time.
- [ ] `test_changed_cron_reregisters` — modified cron → exactly one remove+add.
- [ ] `test_user_enumeration_cached` — 7 schedulers in one process within TTL → 1 `iterdir` scan (mock counter).

**Status:** Not Started

## Stage 5: Destructive endpoints — bulk operations off the loop

**Goal:** Deleting a long conversation or emptying the trash stops freezing the server.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py` (new `soft_delete_all_messages_for_conversation(conversation_id)` — one bulk `UPDATE messages SET deleted=1, version=version+1 ... WHERE conversation_id=? AND deleted=0`, preserving the soft-delete columns/semantics of `soft_delete_message`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py` (`delete_chat_session` ~7818-7860) — call the bulk method via `asyncio.to_thread`; delete the paged loop
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/legacy_maintenance.py` (add `permanently_delete_items(ids)` — chunked batch DELETE + set-based FTS cleanup + one RAG invalidation per chunk)
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/listing.py` (`empty_media_trash_endpoint` ~677-700) — chunked (500) batch delete via `asyncio.to_thread`
- Test: `tldw_Server_API/tests/test_bulk_deletes.py` (new)

**Tests:**
- [ ] `test_delete_session_bulk_single_statement` — 5,000-message conversation → 1 UPDATE statement, loop gone; message rows soft-deleted identically (compare row state vs sequential reference on small fixture).
- [ ] `test_empty_trash_chunked_offloop` — 2,000 items → statements ≈ ceil(2000/500) × per-chunk count; endpoint is `async def` and does no direct sync DB calls (assert via mock that work ran in a thread).

**Status:** Not Started

## Stage 6: Sync I/O off the event loop (exports, attachments, alert-rules DB)

**Goal:** Remaining sync file/DB work in async handlers moves to threads.

**Files + changes:**
1. `endpoints/prompts.py` (~1130-1210): export generation + file read + base64 wrapped in `run_in_threadpool`; skip the temp-file roundtrip where the exporter can return content directly.
2. `endpoints/notes.py` (~4959): `await asyncio.to_thread(target_path.write_bytes, payload)`.
3. `app/core/DB_Management/watchlist_alert_rules_db.py` (~54-193): replace per-call `sqlite3.connect` with a lazily-created shared connection provider (pattern from the other DB modules); document thread-safety (check usage context — if called from multiple threads, use a lock or thread-local connection).

**Tests:**
- [ ] `test_export_runs_in_threadpool` — assert `run_in_threadpool` used (mock) and endpoint stays async.
- [ ] `test_alert_rules_reuses_connection` — 10 rule evaluations → connection opens ≤ 1 (mock counter).

**Status:** Not Started

## Stage 7: MCP metrics cardinality + JWT token pruning

**Goal:** Unbounded metric keys and refresh/revocation sets stop growing for process lifetime.

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/monitoring/metrics.py` (~91, ~310, ~378, ~599-694)
- Modify: `tldw_Server_API/app/core/MCP_unified/auth/jwt_manager.py` (~72-73, ~184, ~300, ~316, ~327)
- Test: `tldw_Server_API/tests/MCP_unified/test_metrics_and_jwt_bounds.py` (new)

**Change:**
1. Metrics: whitelist/sanitize label values before they become dict keys (module, operation from fixed sets; `reason`/error strings bucketed to a small enum-ish set, e.g., first token or "other"); `get_internal_metrics` serves summaries from the Prometheus counters/histograms already maintained in the same class instead of re-aggregating every deque; keep deques only for recent-event display at reduced depth (e.g., 100).
2. JWT manager: store `exp` alongside each refresh token / revoked jti; evict expired entries lazily on insert (small heap or periodic sweep on a counter boundary); `revoke_all_user_tokens` iterates only unexpired entries.

**Tests:**
- [ ] `test_metric_keys_bounded` — 10k varied reasons → distinct key count ≤ configured cap.
- [ ] `test_expired_tokens_pruned` — insert tokens with past exp → subsequent insert prunes them; lookup of expired returns absent.
- [ ] `test_revoked_token_still_enforced_while_valid` — unexpired jti revocation remains authoritative.

**Status:** Not Started

---

## Batch verification

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Jobs tldw_Server_API/tests/Workflows tldw_Server_API/tests/MCP_unified tldw_Server_API/tests/test_jobs_slo_gauges.py tldw_Server_API/tests/test_scheduler_rescan_diffing.py tldw_Server_API/tests/test_bulk_deletes.py -v -m "not external_api and not local_llm_service"
python -m tldw_Server_API.tests.perf.benchmarks.bench_jobs_acquire_tick   # expect connections_per_empty_tick == 1
python -m bandit -r tldw_Server_API/app/core/Jobs tldw_Server_API/app/core/Workflows tldw_Server_API/app/services tldw_Server_API/app/core/MCP_unified -f json -o /tmp/bandit_perf_b5.json
```

Record deltas in `Docs/Reviews/PERF_BASELINE_2026_10.md`; update TASK-13517 (notes, touched files, verification, final summary, DOD — document the two intentional behavior changes: empty-poll backoff caps and SLO cadence).
