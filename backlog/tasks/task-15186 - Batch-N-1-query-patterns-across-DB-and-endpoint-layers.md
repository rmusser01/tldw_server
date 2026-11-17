---
id: TASK-15186
title: Batch N+1 query patterns across DB and endpoint layers
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 3. Plan: Docs/Plans/2026-10-06-perf-batch-3-n-plus-one-batching-implementation-plan.md. Citations metadata map (ChaChaNotes_DB.py:25594); conversation settings IN-batch (character_chat_sessions.py:7327); message-count COUNT fallback (:7299); prompt keyword enrichment batch (Prompts_DB.py:2380); watchlist source tags batch incl. limit=10000 callers (Watchlists_DB.py:2212, endpoints/watchlists.py:2258); notes export IN query; kanban group counts; RAG-context metadata map (message_store.py:2025); collections tag resolve batch; media keyword write batch; sharing and vector-store gathers; writing export; bulk prompt keyword existence check; clustering projection + batched writes (conversation_enrichment.py:317); search_conversations limit params; get_all_content limit guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Renumbered from TASK-13515 to TASK-15186 on 2026-10-31 (TASK-13960): TASK-13515 now names only "Preserve pooled SQLite handles across Media DB migrations" (backlog/tasks/task-13515 - Preserve-pooled-SQLite-handles-across-Media-DB-migrations.md). Before 2026-10-31 both tasks used TASK-13515; Docs/Development/Backlog_Task_Id_Renumbering.md lists every renumbered id.
W4 Stage 4 (webui-perf batch 4, TASK-13524) consolidated [BE] request note — client-side mitigations landed on branch codex/webui-perf-w4-http-arch; server endpoints requested from backend perf batch 3: (1) batch POST /api/v1/media/details/batch {ids} to replace the per-source media-details fan-out in ResearchWorkspace ChatPane ask-with-sources and StudioPane artifact generation (WP-45; client now reuses an in-memory workspace-store cache keyed by mediaId+version and fetches only misses); (2) GET /api/v1/flashcards/due-counts aggregate to replace the per-deck due-count N+1 (WP-46; client caches per-deck counts with react-query staleTime 60s until then); (3) per-book worldbook statistics endpoint to replace the entries fetch-all behind the stats panel (WP-47; client gates the fetch-all behind the Global Statistics panel and the per-book stats tab until then); (4) KnowledgeQA rag-context whole-tail payload so retrySync can post the message+context tail in one request instead of N message writes plus M rag-context writes (WP-48; client now defers rag-context writes off the parent-chained message sequence and settles them in parallel until then).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
