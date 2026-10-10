---
id: TASK-13516
title: Push search and aggregation into SQL/FTS5 and add missing indexes
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 4. Plan: Docs/Plans/2026-10-06-perf-batch-4-sql-fts-pushdown-implementation-plan.md. Chat-history retriever FTS5 MATCH (chacha/chat_history_queries.py:41); remove or bound prompts naive fallback (Prompts_DB.py:2672); chat analytics SQL GROUP BY (endpoints/chat.py:7781); keyword autocomplete LIKE+LIMIT (media/listing.py:234); moodboard FTS (ChaChaNotes_DB.py:34271); skill registry prefix match (:26338); drop LOWER() in media keyword subquery (media_search_repository.py:291); conversations(client_id,last_modified) index on both backends + schema version bump (coordinate TASK-13403); jobs metrics single GROUP BY; audio worker owner-strict single query; drop ingest_jobs full-scan fallback (media/ingest_jobs.py:478).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
