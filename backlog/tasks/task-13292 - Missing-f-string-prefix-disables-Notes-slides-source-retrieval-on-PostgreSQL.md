---
id: TASK-13292
title: Missing f-string prefix disables Notes slides-source retrieval on PostgreSQL
status: To Do
assignee: []
created_date: '2026-09-22 04:36'
labels:
  - bug
  - rag
  - database
dependencies: []
references:
  - 'tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py:2719'
  - 'tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py:3652'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`NotesDBRetriever.retrieve_slides_source_candidates_v1` in `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py` assigns its PostgreSQL SQL **without an f-string prefix**, while the body interpolates a brace expression:

- `:2719` -> `sql = """` (no `f`)
- `:2722` -> `LENGTH({formatted_text}) AS _standalone_source_full_chars`
- No `.format()` call follows; only a params tuple is bound.
- `:2743` -> the SQLite branch **is** `sql = f"""` and works.

Verified: the block contains the literal text `{formatted_text}`, and the assignment has no `f` prefix. PostgreSQL therefore receives `LENGTH({formatted_text})` verbatim and raises a syntax error at `{`. The bare `except Exception` at `:2778` converts it to `RAGDatabaseError("Note source candidate retrieval failed.", database_name="notes")` — a message naming neither the backend nor the cause.

**Effect: on a PostgreSQL-backed ChaChaNotes deployment, Notes contribute zero slides-source candidates, 100% of the time.** The same request on SQLite works. Chat history is unaffected — its sibling `ChatHistoryRetriever.retrieve_slides_source_candidates_v1` has the `f` prefix on *both* branches (`:3652`, `:3683`). A third method, `project_slides_source_documents_v1`, uses `.format()` (`:2853`) and is immune. Three methods, three different splicing conventions.

Caller: `unified_pipeline.py:1633`.

**Introduced** 2026-07-16 in `fc7838d2e3` ("feat(slides): snapshot bounded generation sources (TASK-12115)") — after the 2026-04-07 RAG review ledger, which is why no prior stage caught it.

**Why no test caught it:** the only two tests touching these methods (`tests/RAG_NEW/unit/test_slides_source_retrieval_hardening.py`, `tests/RAG_NEW/unit/test_rag_profiles.py`) replace them with `AsyncMock`, so neither SQL branch ever executes. Worse, TASK-12115 (Done) records verification as "preserves behavior on SQLite and PostgreSQL" while its own evidence notes 21-26 PostgreSQL fixture skips per run. Parity was claimed; every PostgreSQL test was skipped.

Found by the comprehensive core-module review (RAG reviewer); independently verified by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A PostgreSQL-backed test reproduces the failure before the fix (tests/RAG/conftest.py:dual_backend_env exists and is the starting point)
- [ ] #2 The f-string prefix is added at database_retrievers.py:2719
- [ ] #3 The three sibling methods use one consistent splicing convention, or each divergence is documented
- [ ] #4 The bare except at :2778 no longer hides the backend and cause -- the error names both
- [ ] #5 dual_backend_env coverage is extended to the Notes slides-source path so this class of divergence fails loudly next time
- [ ] #6 Bandit run for touched scope
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
