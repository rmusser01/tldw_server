---
id: TASK-13346
title: '_retrieve_allowed_notes_via_sql returns notes with no text match, scored 1.0'
status: Done
assignee: []
created_date: '2026-09-22 22:56'
updated_date: '2026-09-23 05:20'
labels:
  - bug
  - rag
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py:_retrieve_allowed_notes_via_sql returns notes ordered by last_modified DESC with NO text match required -- its own docstring says so -- and stamps every row score=1.0.

ADR-049 (cross-source rank fusion) stops those notes dominating a multi-source result set, which was the acute failure: a top-10 of notes only, zero media documents. But the underlying problem remains -- when an include list is supplied, the notes returned are the most recently modified, not the most relevant, and they carry a score that claims perfect relevance.

Within a single-source notes query there is no fusion to rescue the ordering, so the caller gets recency ranking presented as relevance ranking.

Source: found while fixing TASK-13315 / synthesis F16.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Notes returned via the include-list SQL path are scored by text match, or the absence of scoring is made explicit to the caller
- [x] #2 Single-source notes retrieval with an include list orders by relevance, not last_modified
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Done in 4a20c4b23d.

SCOPE CORRECTION: the task named only _retrieve_allowed_notes_via_sql, but its ChaChaNotes sibling _retrieve_allowed_notes_via_chacha had the identical defect, and that is the one actually used whenever chacha_db is configured -- i.e. the common deployment. Fixing only the named path would have left the live one broken.

ROOT CAUSE, which the task did not state: neither function RECEIVED the query. They took (allowed_note_ids, notebook_id) only, so scoring by relevance was not possible, not merely omitted. Both now take the query, and the caller passes it.

AC1: rows are scored by text match. The formula is the one the unrestricted notes path has always used -- title match 1.0, content match 0.5 -- which was inline in that path and absent from the other two. It is now one _text_match_score helper shared by all three, so the third copy is gone as well.

AC2: both include-list paths sort by score descending. last_modified remains the tie-break, still applied by the database, with relevance applied above it where the query text exists.

DESIGN POINTS worth keeping:
- A note matching nothing is still RETURNED, scored 0.0. The include list is the filter; no text match is required, which is the point of an include list. Returning it with an honest 0.0 is right, hiding it would not be.
- An empty query scores every row equally at 1.0. There is no relevance to measure, and claiming a difference would be worse than claiming none.

Verification: 8 new tests in tests/RAG_NEW/unit/test_notes_include_list_relevance.py, covering the four score combinations, the empty-query case, relevance reordering away from the database's last_modified order, the zero-scored row still being returned, and both include-list paths agreeing. RAG + RAG_NEW 12 failed / 2028 passed -- the same 12 pre-existing failures as before the change. test_restricted_postgres_media_retrieval.py hangs without a local Postgres and was excluded from both runs.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Both include-list notes paths now score and order by relevance instead of stamping 1.0 and returning last_modified order. Neither received the query at all, which is why neither could score; the task named only the SQL path, but the ChaChaNotes sibling had the same defect and is the one used in the common deployment.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
