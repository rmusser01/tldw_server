# Backend Performance Remediation — Batch 4: SQL/FTS Pushdown & Indexes Implementation Plan (2026-10-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Backlog task:** TASK-13516 · **Index:** [2026-10-06-perf-remediation-coordination-index.md](2026-10-06-perf-remediation-coordination-index.md)

**Goal:** Make SQLite do what it already has indexes for: FTS5 MATCH instead of leading-wildcard LIKE over full content, SQL GROUP BY instead of Python bucketing, indexed prefix lookups, and one missing composite index.

**Architecture:** Each stage converts one scan path to an indexed path. The only schema change is Stage 7 (index addition on both backends). **Run after Batch 3** — both touch `Prompts_DB.py` / `ChaChaNotes_DB.py`.

**Tech Stack:** SQLite FTS5 (`MATCH`, quoted literals, bm25), `EXPLAIN QUERY PLAN` assertions, `strftime` bucketing, schema migration machinery (coordinate with TASK-13403).

## Global constraints

- Backend only; activate venv; one commit per stage referencing TASK-13516.
- FTS queries must quote user input as a literal phrase (`'"' + query.replace('"', '""') + '"'`) to avoid FTS syntax errors — reuse the escaping pattern already in `message_store.search_messages_by_content` (~1597-1662).
- Every FTS swap keeps a LIKE fallback **only** for FTS-syntax-error exceptions, never for zero matches.
- Ordering/equivalence: result sets may legitimately differ (FTS tokenization vs substring); tests must encode the *new* expected semantics, and any user-visible behavior change must be listed in the task notes.
- Bandit: `python -m bandit -r tldw_Server_API/app/core/DB_Management tldw_Server_API/app/api/v1/endpoints -f json -o /tmp/bandit_perf_b4.json`.
- Line numbers from the 2026-10-06 review; re-locate by symbol.

---

## Stage 1: RAG chat-history retriever → FTS5

**Goal:** The chat-history RAG retriever (runs on every RAG-enabled chat query) stops `LIKE '%q%'`-scanning every message's content.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/chat_history_queries.py` (~41-52)
- Test: `tldw_Server_API/tests/RAG/test_chat_history_retriever_fts.py` (new)

**Change:** replace the LIKE clause with `messages_fts MATCH ?` (quoted literal), joining `messages_fts` to `messages` on the rowid/id mapping used by `search_messages_by_content`; keep all existing filters (deleted, owner/client, conversation type) and ordering; bm25 rank available if the existing sort wants it.

**Tests:**
- [ ] `test_retriever_uses_fts_index` — `EXPLAIN QUERY PLAN` for the generated SQL contains `messages_fts` (no `SCAN messages`).
- [ ] `test_retriever_matches_expected_documents` — fixture DB with 1,000 messages; query matches the 3 containing the phrase; ordering by timestamp preserved.
- [ ] `test_fts_error_falls_back_to_like` — malformed FTS input path degrades to LIKE (exception-only fallback).

**Status:** Not Started

## Stage 2: Prompts search naive fallback — remove the zero-match trigger

**Goal:** A legitimate "no results" search must not degrade into a full-table dump with per-prompt N+1.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Prompts_DB.py` (`search_prompts` ~2672-2711; `do_naive` trigger ~2670)
- Test: `tldw_Server_API/tests/test_prompts_search_fallback.py` (new)

**Change:**
1. Zero FTS matches → return empty page (delete the `or not combined` style trigger).
2. `do_naive` allowed **only** for explicit test-mode / no-search-fields (i.e., plain listing) — not for FTS errors or misses.
3. When naive runs (listing mode): SQL-side filtering + `LIMIT/OFFSET`; keyword enrichment via Batch 3's `fetch_keywords_for_prompts`.

**Tests:**
- [ ] `test_zero_fts_matches_returns_empty_fast` — 10k-prompt fixture; miss query → 0 rows returned, statement count ≤ 3, no full scan (`EXPLAIN QUERY PLAN` shows FTS + LIMIT).
- [ ] `test_listing_mode_still_paginates_in_sql` — naive path respects offset/per_page in SQL.

**Status:** Not Started

## Stage 3: Chat analytics → SQL GROUP BY

**Goal:** `get_chat_analytics` stops materializing every conversation in a 180-day range.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py` (new `count_conversations_grouped(client_id, start, end, group_expr)`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` (~7781-7817)
- Test: `tldw_Server_API/tests/test_chat_analytics_groupby.py` (new)

**Change:** bucket keys via `strftime('%Y-%m-%d', last_modified)` (day) / state / topic as the endpoint's current Python bucketing does; one query per dimension (≤3) with the same date-range and client filters; Python re-bucketing removed; response shape unchanged.

**Tests:**
- [ ] `test_analytics_groupby_matches_python_impl` — randomized fixture; SQL results equal the old Python bucketing (keep old code in test as reference).
- [ ] `test_analytics_never_loads_rows` — `count_statements` shows only aggregate queries; zero `SELECT c.*` executions.

**Status:** Not Started

## Stage 4: Keyword autocomplete + prompt keyword listing

**Goal:** Autocomplete stops loading the whole Keywords table.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/api.py` (add `search_keywords(query: str, limit: int)` — `LIKE 'q%'` on the normalized column first, `%q%` second, always LIMIT)
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/listing.py` (~234-252)
- Modify: `tldw_Server_API/app/api/v1/endpoints/prompts.py` (`list_all_keywords` ~1085) — pagination params + LIMIT
- Test: `tldw_Server_API/tests/test_keyword_autocomplete.py` (new)

**Tests:**
- [ ] `test_autocomplete_bounded_query` — 10k keywords fixture; query → ≤ `limit` rows, 1 statement.
- [ ] `test_prefix_preferred_over_substring` — prefix matches rank before infix matches.

**Status:** Not Started

## Stage 5: Moodboard smart rules + skill registry search

**Goal:** Two remaining leading-wildcard LOWER(...) scans use FTS/prefix indexes.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` (moodboard rule query ~34271-34289 — FTS via `notes_fts` pattern from `note_store.search_notes_with_keywords` ~2547; skill registry ~26338 — prefix `LIKE 'q%'` on an indexed `COLLATE NOCASE` column, add the index)
- Test: `tldw_Server_API/tests/test_moodboard_and_skill_search.py` (new)

**Tests:**
- [ ] `test_moodboard_rule_uses_fts` — EXPLAIN QUERY PLAN shows `notes_fts`; keyword tokens use exact/prefix match, not `LOWER(k.keyword) LIKE '%tok%'`.
- [ ] `test_skill_search_prefix_indexed` — EXPLAIN QUERY PLAN shows the new index (SEARCH, not SCAN).

**Status:** Not Started

## Stage 6: Media search keyword subquery — drop LOWER()

**Goal:** The must-have-keywords correlated subquery becomes index-usable.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/repositories/media_search_repository.py` (~291-300)
- Test: `tldw_Server_API/tests/test_media_search_keyword_filter.py` (new)

**Change:** precondition test first — assert `keywords_repository.add` normalizes to lowercase (it does per review; encode as an invariant test). Then resolve keyword ids once (single indexed `WHERE keyword IN (...)` SELECT), and use `EXISTS (... mk.keyword_id IN (ids))` per media row instead of `LOWER(k_mh.keyword) IN (...)`.

**Tests:**
- [ ] `test_keywords_stored_lowercase_invariant` — add mixed-case keyword, read back lowercase.
- [ ] `test_keyword_filter_uses_keyword_index` — EXPLAIN QUERY PLAN: no `SCAN Keywords`; results identical to old query on fixture.

**Status:** Not Started

## Stage 7: `conversations(client_id, last_modified)` index (schema change)

**Goal:** Conversation filters/sorts stop scanning on shared-table (Postgres) deployments; SQLite deployments get covering-index benefit.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` (SQLite DDL + migration version bump) and the PostgreSQL backend DDL counterpart (locate via the schema-parity machinery; **coordinate with TASK-13403** — same migration files/version line; if TASK-13403 is mid-flight, land this as a stage on top of its latest schema version).
- Test: `tldw_Server_API/tests/DB_Management/test_conversations_client_id_index.py` (new)

**Change:** `CREATE INDEX IF NOT EXISTS idx_conversations_client_lastmod ON conversations(client_id, last_modified);` in both backends + migration version bump + upgrade path for existing DBs (follow the existing migration pattern exactly — grep the last index-adding migration and copy it).

**Tests:**
- [ ] `test_index_exists_after_migration` — open pre-index-version DB fixture, run migrations, assert index present (both backends where testable locally; Postgres via the existing isolated-test fixture).
- [ ] `test_conversation_search_plan_uses_index` — EXPLAIN QUERY PLAN on the search filters shows the new index.

**Status:** Not Started

## Stage 8: Jobs/queue aggregation pushdown

**Goal:** Background aggregations collapse to single GROUP BYs and one bounded query.

**Files + changes:**
1. `tldw_Server_API/app/services/jobs_metrics_service.py` (~105-176): per-group COUNT loop → one `GROUP BY domain, queue, job_type, status` feeding counters/gauges; drop the per-gauge reconnect.
2. `tldw_Server_API/app/services/audio_jobs_worker.py` (~110-181): owner-strict candidate selection → single aggregate query (candidates LEFT JOIN per-owner running counts, filtered by limits, `LIMIT 1`).
3. `tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py` (`_collect_jobs_for_batch` ~478-539): drop the full-table paging fallback; rely on the `batch_group` filter (verify it is indexed; add the index if not — same migration discipline as Stage 7 but in the Jobs schema).

**Tests:**
- [ ] `test_metrics_reconcile_single_groupby` — 100 groups → 1 aggregate statement (+0 per-group).
- [ ] `test_audio_owner_selection_single_query` — 50 candidates → 1 statement, same owner chosen as the sequential reference impl.
- [ ] `test_batch_detail_no_full_scan` — old/partial batch id → no paging loop over all jobs (statement count bounded).

**Status:** Not Started

---

## Batch verification

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG/test_chat_history_retriever_fts.py tldw_Server_API/tests/test_prompts_search_fallback.py tldw_Server_API/tests/test_chat_analytics_groupby.py tldw_Server_API/tests/test_keyword_autocomplete.py tldw_Server_API/tests/test_moodboard_and_skill_search.py tldw_Server_API/tests/test_media_search_keyword_filter.py tldw_Server_API/tests/DB_Management/test_conversations_client_id_index.py -v -m "not external_api"
# full owners of touched modules:
python -m pytest tldw_Server_API/tests/DB_Management tldw_Server_API/tests/RAG -v -m "not external_api and not local_llm_service" -x
python -m bandit -r tldw_Server_API/app/core/DB_Management tldw_Server_API/app/api/v1/endpoints tldw_Server_API/app/services -f json -o /tmp/bandit_perf_b4.json
```

Record EXPLAIN/query-count deltas in `Docs/Reviews/PERF_BASELINE_2026_10.md`; update TASK-13516 (notes, touched files, verification, final summary, DOD). Schema notes: record the migration version(s) touched for the TASK-13403 parity ledger.
