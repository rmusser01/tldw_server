# Backend Performance Remediation — Batch 3: N+1 Batching Implementation Plan (2026-10-06)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan stage-by-stage. Steps use checkbox (`- [ ]`) syntax for tracking.

**Backlog task:** TASK-13515 · **Index:** [2026-10-06-perf-remediation-coordination-index.md](2026-10-06-perf-remediation-coordination-index.md)

**Goal:** Eliminate the N+1 enrichment loops across the DB and endpoint layers — in almost every case by adding an `IN (...)`-batched helper modeled on one that already exists in the same module, then swapping the loop.

**Architecture:** Repeated pattern per stage: (1) add `*_for_ids` / `*_map` / `*_counts` batch helper next to the existing per-item function, (2) swap the loop at the call site, (3) query-count test proves 2 queries instead of 1+N. No API changes; returned data shapes identical.

**Tech Stack:** SQLite `WHERE id IN (?, ...)`, `GROUP BY`, `GROUP_CONCAT`, Batch-0 `count_statements` helper.

## Global constraints

- Backend only: never modify `apps/**`. Activate venv: `source .venv/bin/activate`. One commit per stage referencing TASK-13515.
- TDD: query-count test first (it fails against the loop), then the helper + swap.
- Verify each helper handles: empty id list (return {} / [] without querying), missing ids (absent from map, not errors), >999 ids (SQLite parameter limit — chunk the IN list at 900).
- Behavior-preserving: same ordering semantics as the loops they replace (note per stage where ordering matters).
- Bandit: `python -m bandit -r tldw_Server_API/app/core/DB_Management tldw_Server_API/app/api/v1/endpoints -f json -o /tmp/bandit_perf_b3.json`.
- Line numbers from the 2026-10-06 review; re-locate by symbol.

---

## Stage 1: ChaCha message-metadata maps (citations + RAG-context)

**Goal:** Two call paths issue up to 1,001 and 101 queries respectively; both become 2.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` (`get_conversation_citations` ~25594-25617)
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py` (`get_messages_with_rag_context` ~2025-2032; reuse `get_message_metadata_map` ~1800)
- Test: `tldw_Server_API/tests/DB_Management/test_metadata_map_batching.py` (new)

**Change:** both loops become `metadata_map = self.get_message_metadata_map(ids)` then a plain Python grouping of `retrieved_documents` / `rag_context` from the map, preserving current output structure.

**Tests:**
- [ ] `test_citations_two_queries_for_500_messages` — `count_statements` ≤ 3 around `get_conversation_citations`.
- [ ] `test_rag_context_two_queries` — same for `get_messages_with_rag_context` (default limit 100).
- [ ] Output-equality golden tests against pre-change structure for conversations with mixed present/absent metadata.

**Status:** Not Started

## Stage 2: Character-session list (settings batch + COUNT fallback)

**Goal:** `list_chat_sessions` stops doing per-conversation settings queries (≤200/page) and never falls back to materializing 1,000 messages per conversation.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py` or `message_store.py` (add `get_conversation_settings_for_ids(ids) -> dict[str, dict]`, mirroring `count_messages_for_conversations`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py` (~7299-7332)
- Test: `tldw_Server_API/tests/test_character_sessions_batching.py` (new)

**Change:** settings loop → one map fetch; the `except _CHAR_CHAT_SESSIONS_NONCRITICAL_EXCEPTIONS` fallback for message counts switches to per-conversation `COUNT(*)` (bounded, no content fetch) — or surfaces the error if COUNT also fails; never `get_messages_for_conversation(limit=1000)`.

**Tests:**
- [ ] `test_include_settings_one_batched_query` — 50 conversations + `include_settings=true` → ≤3 statements.
- [ ] `test_count_fallback_never_materializes_messages` — force the batch count helper to raise; assert fallback issues COUNT queries and zero `SELECT ... content` statements.

**Status:** Not Started

## Stage 3: Prompt keyword enrichment batch

**Goal:** Prompt list/search pages stop issuing one keyword-JOIN query per prompt.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Prompts_DB.py` (add `fetch_keywords_for_prompts(ids) -> dict[int, list[str]]` single `IN` + JOIN; swap loops at ~2380, ~2528, ~2579, ~2662)
- Test: `tldw_Server_API/tests/test_prompts_keyword_batch.py` (new)

**Change:** one `SELECT pkl.prompt_id, k.keyword FROM PromptKeywordLinks pkl JOIN PromptKeywordsTable k ON ... WHERE pkl.prompt_id IN (...) AND pkl.deleted = 0` grouped in Python; per-page keyword lists identical (same order as `fetch_keywords_for_prompt` produced — verify ordering source, ORDER BY if needed).

**Tests:**
- [ ] `test_list_page_two_queries` — 100 prompts/page → ≤4 statements for the enrichment phase.
- [ ] `test_keywords_identical_to_per_prompt_fetch` — compare map values vs legacy per-prompt calls on a fixture DB.

**Status:** Not Started

## Stage 4: Watchlist source-tag batching + limits

**Goal:** Source listings stop doing one tags query per row (callers pass `limit=10000` today → up to 10,001 queries).

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py` (add `_fetch_tags_for_source_ids(ids)` modeled on `Collections_DB._fetch_tags_for_item_ids`; swap loops at ~1803, ~2212, ~2657; add LIMIT to `list_sources_by_group_ids` ~2641)
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py` (~2258, ~3733) and `collections_feeds.py` (~484): cap `limit` at a sane max (e.g., 500) or paginate — keep API backward-compatible by documenting the cap; add `get_sources_by_ids(ids)` for the report/check-now paths (~1528, ~2401-2446, eliminating the double fetch)
- Test: `tldw_Server_API/tests/test_watchlists_tag_batching.py` (new)

**Tests:**
- [ ] `test_list_sources_two_queries` — 100 sources with tags → ≤3 statements.
- [ ] `test_group_listing_has_limit` — `list_sources_by_group_ids` respects a limit.
- [ ] `test_check_now_single_batched_fetch` — mock run returns statuses; source fetch count == 1 batch.

**Status:** Not Started

## Stage 5: Notes, kanban, collections, media-keyword batches

**Goal:** The remaining per-item loops.

**Files + changes (one sub-commit each is fine):**
1. `endpoints/notes.py` (~2844, ~2906): add `get_notes_by_ids(ids)`; export loops use it. Conversation-keyword links (~3963): single JOIN with SQL `LIMIT/OFFSET` (mirror the no-IDs branch at ~3980-3992).
2. `endpoints/kanban/kanban_boards.py` (~185) + `kanban_lists.py` (~115): add `get_card_counts_for_list_ids(ids)` (`GROUP BY list_id`); swap per-list COUNT loops.
3. `Collections_DB.py` (`ensure_collection_tag_ids` ~2893) + `Watchlists_DB.py` (~1900): batch resolve (one `IN` SELECT over names) + `executemany` INSERT for misses, keep unique-violation retry fallback. `_replace_item_tags` (~2922): `executemany`.
4. `media_db/repositories/keywords_repository.py` (~184 with `add()` ~59): batch `replace_keywords` — one `IN` SELECT for existing keywords, `executemany` INSERT for new, single FTS refresh.
5. `endpoints/prompts.py` (`bulk_update_prompt_keywords` ~1553): batch existence check by ids (no full prompt fetches); fetch only keyword links.

**Tests (one per sub-change):**
- [ ] `test_notes_export_single_query` — 50 ids → 1 note query + existing keyword batch.
- [ ] `test_kanban_counts_grouped` — board with 10 lists → 1 count query.
- [ ] `test_tag_resolve_batched` — 20 tag names → ≤2 statements (+retries preserved).
- [ ] `test_keyword_write_batched` — 20 keywords on update → statements ≤ 6 (was ~40-80).
- [ ] `test_bulk_prompt_keywords_no_full_fetch` — SELECT count touching prompt bodies == 0.

**Status:** Not Started

## Stage 6: Endpoint gathers and refetch elimination

**Goal:** Sequential awaits → gathers; redundant per-id refetches → reuse.

**Files + changes:**
1. `endpoints/vector_stores_openai.py` (~444-505): `asyncio.gather` per-store stats with a semaphore (8); dedupe meta-DB pass vs Chroma fallback pass.
2. `endpoints/sharing.py` (~1786-1810): gather owner-DB opens; batch `get_workspaces_by_ids` per owner (new helper).
3. `endpoints/writing.py` (~1541-1556): drop the per-session `get_writing_session` refetch — the list query already returned the rows.
4. `endpoints/characters_endpoint.py` (~3164): batch world-book + entries fetch per export (2N → 2).

**Tests:**
- [ ] `test_vector_store_stats_concurrent` — 10 stores, each stats mocked at 50ms → wall time < 300ms (and result identical).
- [ ] `test_shared_with_me_no_per_item_workspace_query` — query-count assertion.
- [ ] `test_writing_export_no_refetch` — `get_writing_session` call count == 0 during export.

**Status:** Not Started

---

## Stage 7: Conversation clustering + unbounded query guards

**Goal:** Clustering stops loading every conversation as full rows and writing per-conversation updates in a loop; two unbounded read paths get guards.

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/conversation_enrichment.py` (`cluster_conversations_for_user` ~317: load via projection `SELECT id, topic_label, cluster_id, last_modified FROM conversations WHERE ...` instead of full `search_conversations` rows; batch the cluster/topic writes — chunked `UPDATE ... WHERE id IN (...)` or `executemany`, replacing the per-conversation `_update_conversation_with_retry` loop)
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py` (`search_conversations` ~1692-1736: accept `limit`/`offset` kwargs defaulting to the current unbounded behavior for compatibility; internal callers that don't need everything pass limits)
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/legacy_content_queries.py` (`get_all_content_from_database` ~30-35: make `limit` a required keyword arg and drop `content` from the default projection — callers wanting content opt in explicitly; update the `DB_Manager` facade accordingly)
- Test: `tldw_Server_API/tests/Chat/test_conversation_clustering_batch.py` (new)

**Tests:**
- [ ] `test_clustering_projection_and_batched_writes` — 500-conversation fixture: clustering issues ≤ (1 read + ceil(500/chunk)) statements; no `SELECT c.*` execution.
- [ ] `test_get_all_content_requires_limit` — calling without `limit` raises `TypeError`; with `limit=10` returns 10 rows without `content` unless requested.

**Status:** Not Started

---

## Batch verification

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests -k "batch or batching or counts_grouped or export_single" -v -m "not external_api and not local_llm_service"
# plus the full suites owning touched modules:
python -m pytest tldw_Server_API/tests/DB_Management tldw_Server_API/tests/test_prompts_keyword_batch.py tldw_Server_API/tests/test_watchlists_tag_batching.py -v -m "not external_api"
python -m bandit -r tldw_Server_API/app/core/DB_Management tldw_Server_API/app/api/v1/endpoints -f json -o /tmp/bandit_perf_b3.json
```

Record query-count deltas in `Docs/Reviews/PERF_BASELINE_2026_10.md`; update TASK-13515 (notes, touched files, verification, final summary, DOD).
