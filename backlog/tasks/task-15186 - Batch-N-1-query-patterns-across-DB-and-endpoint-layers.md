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
