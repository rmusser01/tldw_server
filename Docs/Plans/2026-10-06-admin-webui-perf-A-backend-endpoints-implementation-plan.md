# Admin WebUI Perf — Plan A: Backend Admin Endpoints Implementation Plan (2026-10-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Backlog task:** TBD — `backlog task create` crashes with "Maximum call stack size exceeded" (CLI 1.44.0, 2026-10-06). Create the task before execution begins and update this line + stage commit messages with the real ID.

**Spec:** [ADMIN_WEBUI_PERF_REVIEW_2026_10_06.md](../Reviews/ADMIN_WEBUI_PERF_REVIEW_2026_10_06.md) — implements findings **F1–F7**.

**Goal:** Make the backend endpoints that serve the admin webui correct (billing routes, dead `llm_usage_v2` table) and efficient as data grows (indexes, sargable queries, cached backup scans, thread offload).

**Architecture:** Four independent fixes plus one route-contract repair. No API shape changes except adding the admin billing routes the frontend already calls (client contract is source of truth) and new indexes via the existing AuthNZ migration helpers.

**Tech Stack:** FastAPI, SQLite + PostgreSQL dual-dialect SQL (existing `is_pg` branches), `starlette.concurrency.run_in_threadpool`, `asyncio.to_thread`, AuthNZ migrations.

## Global constraints

- Backend only: never modify `apps/**`. Activate venv first: `source .venv/bin/activate` (repo root).
- TDD: failing test first, then the fix; behavior-preserving except where a stage explicitly says otherwise.
- Every SQL change must ship in **both** dialects (PG + SQLite) wherever the file has an `is_pg` split; respect the existing parameterization style (`?` vs `$N`).
- Bandit before finishing: `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/admin tldw_Server_API/app/api/v1/endpoints/billing.py tldw_Server_API/app/api/v1/endpoints/llamacpp.py tldw_Server_API/app/services/admin_data_ops_service.py tldw_Server_API/app/services/admin_system_service.py tldw_Server_API/app/services/admin_usage_service.py tldw_Server_API/app/services/admin_system_ops_service.py -f json -o /tmp/bandit_admin_perf_a.json`
- One commit per stage, message references the Backlog task ID.
- Line numbers come from the 2026-10-06 review; re-locate by symbol name, they drift.
- Route changes: this repo lints served routes (`rg_route_map_lint`, see TASK-13417 era tooling) — if the billing path changes, run/update the route map artifacts per their README before committing Stage 1.

---

## Stage 1: Billing admin route contract (F1)

**Goal:** `GET /api/v1/admin/billing/overview|subscriptions|events` exist and return the shapes `BillingDashboardPage` already expects; subscriptions filters and paginates in SQL.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/router_groups/admin.py:63-69` (billing `ImportedRouterSpec` prefix)
- Modify: `tldw_Server_API/app/api/v1/endpoints/billing.py` (add `/overview`, `/events`; rework `/subscriptions`; keep legacy alias)
- Modify: `tldw_Server_API/app/services/billing_repo.py` (add `list_subscriptions(status, limit, offset)`, `list_billing_events(limit, offset)`, `get_overview()`; org names via one JOIN)
- Test: `tldw_Server_API/tests/test_admin_billing_endpoints.py` (new) + `apps/packages/ui/src/components/Option/Admin/__tests__/BillingDashboardPage.test.tsx` (already mocks the client — must stay green untouched)

**Change:**
1. Change the billing `ImportedRouterSpec` prefix from `f"{API_V1_PREFIX}"` to `f"{API_V1_PREFIX}/admin"` so `billing.py`'s `prefix="/billing"` yields `/api/v1/admin/billing/*`. Keep a deprecated duplicate route for the old path: in `billing.py`, register `GET /subscriptions` on a second router mounted at the legacy `/api/v1/billing` (one release, log a deprecation warning) so existing API consumers don't break.
2. `GET /overview` — new endpoint, admin-guarded like subscriptions. SQL over `org_subscriptions`:
   ```sql
   SELECT
     COALESCE(SUM(total_cost_usd), 0)      AS mrr,                -- adjust to the revenue column billing_repo actually exposes; if none, mrr = 0
     COUNT(*) FILTER (WHERE status = 'active')    AS active_subscriptions,
     COUNT(*) FILTER (WHERE status = 'canceled')  AS canceled_subscriptions,
     COUNT(*) FILTER (WHERE status = 'past_due')  AS past_due_subscriptions
   FROM org_subscriptions
   ```
   (SQLite: `SUM(CASE WHEN status='active' THEN 1 ELSE 0 END)` etc.) Response: `{"mrr", "active_subscriptions", "canceled_subscriptions", "past_due_subscriptions"}` — exactly the mock at `BillingDashboardPage.test.tsx:88-93`. Pick the real revenue column by reading `billing_repo.py`'s subscription model first; if no cost column exists, return `mrr: 0` and note it.
3. `GET /subscriptions` — push `status`, `limit` (default 100), `offset` into SQL (`WHERE status = ?` only when provided), `ORDER BY created_at DESC LIMIT ? OFFSET ?`, resolve org names with a `LEFT JOIN organizations` in the same query (drop the `list_organizations(limit=len+50)` name-map fetch). Return `{items, total}` with truthful `COUNT(*)` — the UI pages client-side today and can adopt `total` later.
4. `GET /events` — check `billing_repo` / `admin_usage_service` for an existing billing/ledger events source (e.g. usage events, credit ledger). If a real source exists, page it in SQL. If none exists, return `{"items": [], "total": 0}` with a `501`-free honest empty shape and record the gap in the task notes — do NOT invent data.

**Tests:**
- [ ] `test_overview_counts_by_status` — seed 2 active + 1 canceled subscriptions; assert the four fields.
- [ ] `test_subscriptions_status_filter_in_sql` — `status=active` returns only active rows; assert one query (statement counter) — no Python-side filtering loop.
- [ ] `test_subscriptions_paginates_with_truthful_total` — 5 rows, `limit=2&offset=0` → 2 items, `total=5`.
- [ ] `test_legacy_billing_subscriptions_alias_alive` — old path still 200s with deprecation log.
- [ ] UI suite green: `cd apps/packages/ui && bun vitest run src/components/Option/Admin/__tests__/BillingDashboardPage.test.tsx` (run only when touching shared schemas; otherwise backend-only stage).

**Status:** Not Started

## Stage 2: Replace dead `llm_usage_v2` reads with `llm_usage_log` (F2)

**Goal:** Admin token-stats-today and cost attribution return real numbers from the live, indexed `llm_usage_log` table, with sargable date filters.

**Files:**
- Modify: `tldw_Server_API/app/services/admin_system_service.py:253-276` (`tokens_today` query)
- Modify: `tldw_Server_API/app/services/admin_usage_service.py:1240-1290` (cost attribution)
- Test: `tldw_Server_API/tests/test_admin_llm_usage_reads.py` (new)

**Change:**
1. `admin_system_service.py` — both dialect branches become a scan of `llm_usage_log` keyed on `ts` (index `idx_llm_usage_log_ts` exists, `migrations.py:2625`):
   ```sql
   SELECT
     COALESCE(SUM(prompt_tokens), 0)     AS prompt,
     COALESCE(SUM(completion_tokens), 0) AS completion,
     COALESCE(SUM(total_tokens), 0)      AS total
   FROM llm_usage_log
   WHERE ts >= CURRENT_DATE            -- PG branch
   -- SQLite branch: WHERE ts >= date('now')
   ```
   Never wrap the column in `date()`/`datetime()`.
2. `admin_usage_service.py` cost attribution — switch `FROM llm_usage_v2` to `FROM llm_usage_log`, `created_at` → `ts`; drop the `datetime()` wrapper (`WHERE ts >= datetime('now', ?)` stays sargable because the transformation is on the *literal* side only — verify with EXPLAIN QUERY PLAN in the test). `group_field` maps to existing columns (`user_id` / `key_id`). Org scoping: `llm_usage_log` has no `org_id`; when `org_ids is not None`, JOIN through the pattern already used at `admin_usage_service.py:296-300` (`JOIN org_members om ON om.user_id = llm_usage_log.user_id AND om.org_id IN (...)`) in both dialects.
3. Delete the `except Exception` swallow on the tokens-today block only if the surrounding error contract allows; otherwise keep the try/except but add `logger.warning` so a future regression isn't silent. (Verify current behavior first; the review only guarantees silence, not the handler.)

**Tests:**
- [ ] `test_tokens_today_reads_llm_usage_log` — insert 2 rows into `llm_usage_log` (one today, one 3 days old); assert sums include only today's row.
- [ ] `test_tokens_today_query_is_sargable` — `EXPLAIN QUERY PLAN` output mentions `idx_llm_usage_log_ts` (SQLite), no `SCAN` on the table.
- [ ] `test_cost_attribution_groups_by_user` — seed 3 users' rows; `group_by=user` returns per-user sums, ordered by total DESC, limit respected.
- [ ] `test_cost_attribution_org_scoped_uses_join` — org filter returns only members' rows; EXPLAIN shows no `SCAN org_members` once Stage 3's index lands (assert index usage after Stage 3; until then assert correctness only).

**Status:** Not Started

## Stage 3: AuthNZ indexes `sessions(created_at)` + `org_members(org_id, user_id)` (F3, F4)

**Goal:** Time-windowed session queries and org-scoped JOINs stop full-scanning.

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py` (SQLite; alongside existing `sessions` indexes ~`:127-130` and `org_members` index ~`:2716`)
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py` (PG; alongside `:1091-1094` and `:1250`)
- Test: `tldw_Server_API/tests/AuthNZ/test_admin_perf_indexes.py` (new)

**Change:** Add `CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)` and `CREATE INDEX IF NOT EXISTS idx_org_members_org_user ON org_members(org_id, user_id)` in both migration files, following the file-local convention for idempotent index creation (mirror the `llm_usage_log` index block at `migrations.py:2625-2631`). Migration ordering: append to the migration step that owns the corresponding table — do not invent a new migration step unless the helper requires one; if a versioned-migration helper exists, use it with the next free number.

**Tests:**
- [ ] `test_sessions_created_at_index_exists` — run migrations on a fresh temp DB; `PRAGMA index_list(sessions)` (SQLite) / `pg_indexes` (PG fixture) contains both new indexes.
- [ ] `test_activity_query_uses_index` — EXPLAIN QUERY PLAN of the Stage-2-style `WHERE created_at >= ?` on `sessions` reports the new index, not `SCAN sessions`.

**Status:** Not Started

## Stage 4: Backups endpoint — TTL cache + scan-level early exit (F5)

**Goal:** `GET /admin/backups` stops walking the entire backup tree on every request; repeated calls within the TTL are served from cache; pagination no longer requires the full sort of every file.

**Files:**
- Modify: `tldw_Server_API/app/services/admin_data_ops_service.py:152-191`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_data_ops.py:224` (pass through; no signature change)
- Test: `tldw_Server_API/tests/test_admin_backups_cache.py` (new)

**Change:**
1. Add a module-level cache: `_backup_cache: dict[tuple[str | None, int | None], tuple[float, list[BackupFile]]]` guarded by the TTL constant `BACKUP_SCAN_CACHE_TTL_SEC = 30.0`. `list_backup_items` checks `(dataset, user_id)` key against `time.monotonic()`; on hit, skip the scan. Cache access under a `threading.Lock` (the scan runs in `asyncio.to_thread` today — keep it that way).
2. Invalidate (`_backup_cache.clear()`) in every write path of the same service (backup creation, restore, delete — locate by symbol).
3. Cheap pre-pagination: collect `(path, mtime)` from `scandir` entries without building full `BackupFile` objects, sort by mtime desc, slice to `[offset:offset+limit]`, then construct `BackupFile` (with `stat()` details) only for the page. Total = the collected length. This turns per-request stat-detail work from O(all files) to O(page size) while the TTL cache absorbs the directory walk itself.
4. Long-term (out of scope, note in task): persist backup metadata to a table at creation time.

**Tests:**
- [ ] `test_second_call_within_ttl_skips_scan` — monkeypatch `os.scandir` with a counter; two calls within TTL → scandir invoked only for the first.
- [ ] `test_cache_invalidated_on_backup_write` — call create-backup helper; next list call re-scans.
- [ ] `test_pagination_returns_page_and_truthful_total` — 25 files, `limit=10&offset=20` → 5 items, total 25, sorted by mtime desc.
- [ ] `test_stats_only_page_files` — monkeypatch `DirEntry.stat` counter; building a page of 10 from 25 files stats ≤ 10 entries more than the scan requires for mtime (scandir-cached stats may count; assert the *object construction* count via a side-channel spy on `BackupFile` creation).

**Status:** Not Started

## Stage 5: Event-loop offload — llamacpp inventory + system_ops reads (F6, F7)

**Goal:** No sync filesystem work on the event loop in the admin-serving endpoints.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/llamacpp.py:573-580` (inventory endpoint)
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_ops.py:263-270,441-467,555-583` (maintenance, feature-flags, incidents)
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_api_keys.py:165-183` (usage top)
- Test: `tldw_Server_API/tests/test_admin_event_loop_offload.py` (new)

**Change:**
1. `llamacpp.py` inventory: wrap both calls exactly like the assets endpoint at `:407-415`:
   ```python
   config_state = await run_in_threadpool(llamacpp_config_service.get_config_state, llm_manager)
   inventory = await run_in_threadpool(llamacpp_inventory_service.scan_inventory, config_state)
   ```
   Add no cache in this stage (config-state-keyed TTL cache is a possible follow-up; keep the stage minimal).
2. `admin_ops.py` + `admin_api_keys.py`: wrap the sync service calls (`svc_get_maintenance_state()`, feature-flag/incident list functions, `svc_list_api_key_usage(limit=...)`) in `asyncio.to_thread(...)` — the repo already uses `asyncio.to_thread` in sibling admin endpoints (`sandbox.py:2326-2340`), match that style.

**Tests:**
- [ ] `test_inventory_uses_threadpool` — monkeypatch `run_in_threadpool` recorder; GET inventory routes through it (both calls).
- [ ] `test_system_ops_reads_offloaded` — for each of the four endpoints, monkeypatch `asyncio.to_thread` recorder; the service call is wrapped.
- [ ] Existing endpoint suites stay green: `python -m pytest tldw_Server_API/tests -k "admin_ops or feature_flag or incident or maintenance or api_key_usage or llamacpp" -q`

**Status:** Not Started

---

## Verification (whole plan)

- [ ] `source .venv/bin/activate && python -m pytest tldw_Server_API/tests -q` (full suite green)
- [ ] Bandit command from Global constraints: zero new findings
- [ ] Self-review diff; update the Backlog task with touched files, verification results, and final summary
