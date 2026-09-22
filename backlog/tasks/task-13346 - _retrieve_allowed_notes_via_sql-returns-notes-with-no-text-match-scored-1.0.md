---
id: TASK-13346
title: '_retrieve_allowed_notes_via_sql returns notes with no text match, scored 1.0'
status: To Do
assignee: []
created_date: '2026-09-22 22:56'
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
- [ ] #1 Notes returned via the include-list SQL path are scored by text match, or the absence of scoring is made explicit to the caller
- [ ] #2 Single-source notes retrieval with an include list orders by relevance, not last_modified
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
