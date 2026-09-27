---
id: TASK-13292
title: Missing f-string prefix disables Notes slides-source retrieval on PostgreSQL
status: Done
assignee: []
created_date: '2026-09-22 04:36'
updated_date: '2026-09-23 23:46'
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
- [x] #1 A PostgreSQL-backed test reproduces the failure before the fix (tests/RAG/conftest.py:dual_backend_env exists and is the starting point)
- [x] #2 The f-string prefix is added at database_retrievers.py:2719
- [x] #3 The three sibling methods use one consistent splicing convention, or each divergence is documented
- [x] #4 The bare except at :2778 no longer hides the backend and cause -- the error names both
- [x] #5 dual_backend_env coverage is extended to the Notes slides-source path so this class of divergence fails loudly next time
- [x] #6 Bandit run for touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Premise partly already fixed: 9061081c0c (earlier on this branch) added the f prefix at database_retrievers.py:2719 (AC2) and tests/RAG_NEW/unit/test_slides_source_sql_interpolation.py, which captures SQL from a stub (no real PostgreSQL). Remaining work done in d561c83999.

AC1/AC5: new test_dual_backend_notes_slides_source_retrieval in tests/RAG/test_dual_backend_end_to_end.py runs notes retrieve_slides_source_candidates_v1 + project_slides_source_documents_v1 against real SQLite and PostgreSQL via dual_backend_env (local Postgres on :5432, not skipped). Red check: with the f prefix removed, postgres FAILED (RAGDatabaseError 'Note source candidate retrieval failed.') and sqlite passed; with the fix, 2 passed.

AC4: all six slides-source except blocks (media/notes/chats x candidates/projection) now go through _slides_source_db_error, which appends backend and exception class, e.g. '... failed. (backend=postgresql, cause=SyntaxError)'. The driver message is deliberately excluded (it can echo SQL/params); 'from None' kept. Test test_slides_source_db_error_names_backend_and_cause: RED on ea1cbc6941 (message lacked 'postgresql'), GREEN after; also asserts SQL text is not leaked. AC3: the ticket's 'three conventions' is now two, applied consistently by role across all three retrievers (f-strings for per-backend candidate SQL, str.format for the shared projection template); documented in the helper docstring.

Suites: RAG_NEW/unit + RAG/test_dual_backend_end_to_end.py + test_sql_retriever_hardening.py + test_dual_backend_rag_flow.py. Before (ea1cbc6941 retriever): 4 failed / 1209 passed = 3 pre-existing (prompt_loader concurrent transition, pgvector psycopg fallback, flashrank local bundle) + the new red test. After: RAG_NEW/unit 4 failed / 1197 passed = same 3 pre-existing + test_generation_executor capacity test, which is an xdist flake unrelated to this file (152/152 passed in 3 isolated runs; it does not touch database_retrievers); RAG files 12 passed. Bandit -ll on database_retrievers.py: no findings. Ruff: only a pre-existing I001 in test_dual_backend_end_to_end.py.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Notes slides-source retrieval works on PostgreSQL (f prefix, 9061081c0c) and is now covered by a real dual-backend test that fails on PostgreSQL if the defect returns. Slides-source DB errors now name the backend and cause type without leaking SQL. Splicing convention documented. Known: test_generation_executor capacity test flakes under xdist (unrelated).
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
