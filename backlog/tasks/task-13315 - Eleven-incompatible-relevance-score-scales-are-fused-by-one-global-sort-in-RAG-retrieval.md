---
id: TASK-13315
title: >-
  Eleven incompatible relevance score scales are fused by one global sort in RAG
  retrieval
status: Done
assignee: []
created_date: '2026-09-22 04:54'
updated_date: '2026-09-22 23:16'
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
- [x] #1 Main retrieval path uses rank-based fusion rather than a global sort over raw scores
- [x] #2 Test asserts cross-source ordering for a mixed media+notes query with an include list
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in 2f3b59fd67. Design record: Docs/ADR/049-rag-cross-source-fusion.md.

AC1: MultiDatabaseRetriever.retrieve now keeps each source's results in their own list and orders across them with reciprocal rank fusion (k=60) in _order_across_sources, instead of flattening and sorting by raw score.

Two things the task's suggested fix could not do as written:
1. retrieve_with_fusion was NOT reusable. It calls retr.retrieve(query) with no config and no per-source restrictions, so routing the main path through it would have silently dropped allowed_media_ids, allowed_note_ids, index_namespace and the entire RetrievalConfig -- including the include list that the reported scenario depends on. Fusion had to move into retrieve rather than the caller moving to fusion.
2. Raw RRF scores (~0.016 at rank 1) break callers. unified_pipeline.py re-sorts by score and caps to top_k at 4458, 4516 and 4766, and applies a [0,1]-bounded boost min(1.0, score * 1.1 + 0.02) at 5069; Research/providers/local.py:117 returns the score in an API response. So fused scores are rescaled onto (0,1], which keeps the fused order stable under those re-sorts while leaving the range intact.

Also rejected: keeping each document's original score and fusing only the order. unified_pipeline re-sorts by score immediately afterwards, which would have undone the fusion completely.

Scope deliberately limited to the cross-source case: with fewer than two sources returning results, behaviour is byte-for-byte unchanged. Single-source scores are meaningful within one scale and rank-based scores would be a loss of fidelity. min_score is unaffected -- every retriever applies it internally, before results reach fusion.

Behaviour change beyond ordering: a document returned by two sources now appears once with its ranks summed, where the global sort listed it twice. Standard RRF, matching the existing _reciprocal_rank_fusion, but a multi-source result set can be shorter than before for the same inputs. Recorded in ADR-049.

AC2: TestCrossSourceFusion in tests/RAG_NEW/unit/test_retrieval.py, 7 tests, all red without the fix. Includes the review's exact scenario (20 constant-1.0 notes vs min-max media, top_k=10) both as a unit test and end-to-end through retrieve() with two stubbed sources and the max_results cap, plus insertion-order ties, the (0,1] range, single-source passthrough, and the duplicate-document case.

Note on test quality: the end-to-end test first asserted only "any media document in the top-10". That passed against the unfixed code -- the old stable sort left exactly one media document (the single one min-max mapped to 1.0) above the twenty notes, satisfying `any` while still shutting media out of the other nine slots. Caught by running the new tests against the stashed source; it now asserts interleaving, with the reason in a comment.

Follow-up filed as TASK-13346: _retrieve_allowed_notes_via_sql returns notes ordered by last_modified with no text match required, stamped score=1.0. Fusion stops them dominating a multi-source result, but in a single-source notes query the caller still gets recency ranking presented as relevance ranking.

Verification: tests/RAG + tests/RAG_NEW 12 failed / 2020 passed with the change vs 18 / 2014 without (stash-isolated); the difference is exactly these tests and there are no new failures. The 12 are pre-existing. tests/RAG/test_restricted_postgres_media_retrieval.py hangs in this environment (no local Postgres) and was excluded from both runs -- an environment limitation, not a code defect. Bandit clean on database_retrievers.py (run via uvx; bandit is CI-only, not a declared local dependency).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Multi-source retrieval now orders by reciprocal rank fusion rescaled to (0,1], so constant-scored notes can no longer shut min-max-scored media out of the top-k. Single-source retrieval is unchanged. retrieve_with_fusion could not be reused as the task suggested -- it drops the caller's config -- so fusion moved into retrieve.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
