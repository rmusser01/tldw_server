---
id: TASK-13371
title: RAG analytics maps 1-star feedback to 0.2 instead of 0.0
status: Done
assignee: []
created_date: '2026-09-24 00:00'
updated_date: '2026-09-28 00:09'
labels:
  - evaluations
  - rag
  - consistency
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
core/RAG/rag_service/analytics_system.py:959,968 normalises user feedback stars by stars/5, so the worst rating stores as 0.2 while Evaluations now maps every judge scale linearly with min->0 (TASK-13328, scoring.py). Changing it shifts the stored analytics trend series, so decide: migrate (rewrite or version the history) or keep stars/5 and document why analytics differs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Decision recorded: linear min->0 mapping via scoring.py, or documented exception
- [x] #2 If migrated, stored history is versioned or rewritten so trends stay comparable
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Decision: keep stars/5 (documented exception), no migration. The stored analytics events (record_search_quality -> AnalyticsEvent.metrics.quality_score, record_document_performance.relevance_score) are unversioned JSON with no scale marker, so switching to scoring.py's linear min->0 map would silently shift every stored trend with no way to tell old rows from new. Evaluations needed min->0 because judge scores feed pass/fail thresholds; analytics never thresholds this value and only compares trends, and stars/5 has a coherent meaning (share of the top rating). AC2 is N/A (not migrated). Comment added at analytics_system.py above the search-quality record; test_submit_feedback_keeps_stars_over_five_for_stored_trends pins 1->0.2 and 5->1.0 so a future consistency refactor fails loudly. tests/RAG/test_analytics_backend.py: 11 passed. Bandit on analytics_system.py: 0 issues (pre-existing file, comment-only change). Side observation, not changed: rating = 1 if helpful else 0 records a thumbs-up as rating 1 in the same field as 1-5 stars.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Kept stars/5 as a documented exception to TASK-13328's min->0 mapping (unversioned stored trends; analytics does not threshold); pinned by a test.
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
