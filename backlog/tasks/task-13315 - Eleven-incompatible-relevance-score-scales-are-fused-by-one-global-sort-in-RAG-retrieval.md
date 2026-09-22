---
id: TASK-13315
title: >-
  Eleven incompatible relevance score scales are fused by one global sort in RAG
  retrieval
status: To Do
assignee: []
created_date: '2026-09-22 04:54'
labels:
  - bug
  - rag
dependencies: []
references:
  - 'tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py:4930'
  - 'tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py:3054'
  - 'tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py:4949'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
MultiDatabaseRetriever.retrieve sorts and caps across sources whose scores are produced on eleven incompatible scales: min-max normalised (media, chunk FTS, vector), constant 1.0 (both notes paths), constant 0.5 (chat history, character cards, SQL), and 0.6/0.4 constants (claims).

Scenario: sources=["media_db","notes"], top_k=10, include_note_ids of 20 notes. _retrieve_allowed_notes_via_sql returns notes ordered last_modified DESC with NO text match required (its own docstring says so) and stamps every one score=1.0. Media is min-max normalised so exactly one media doc reaches 1.0. The global sort puts 20 constant-1.0 notes at or above every media doc and documents[:max_results] returns a top-10 of notes only - zero media docs, including high-BM25 matches. Generation then answers from documents never scored for relevance.

Second instance: min-max maps every source best hit to exactly 1.0 and list.sort is stable, so which source wins the top slot is decided by dict insertion order.

retrieve_with_fusion (4949) already exists and the main path does not use it. Fix is to route through it - rank-based RRF needs no calibration. Needs design doc: ordering changes for every caller.

Source: synthesis F16
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Main retrieval path uses rank-based fusion rather than a global sort over raw scores
- [ ] #2 Test asserts cross-source ordering for a mixed media+notes query with an include list
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
