---
id: TASK-13290
title: Prompt Studio optimizations endpoint returns 500 on every SQLite deployment
status: To Do
assignee: []
created_date: '2026-09-22 04:34'
updated_date: '2026-09-22 15:29'
labels:
  - bug
  - prompt-studio
  - database
dependencies: []
references:
  - 'tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:2122'
  - 'tldw_Server_API/app/core/DB_Management/PromptStudioDatabase.py:7378'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`PromptStudioDatabase` selects `_SQLitePromptStudioDatabase` unless the backend is PostgreSQL (`PromptStudioDatabase.py:7156-7167`), so SQLite is the default. But `list_optimizations` is implemented **only** on the PostgreSQL class:

- `_BackendPromptStudioDatabase.list_optimizations` — present (`:2122-2183`)
- `_SQLitePromptStudioDatabase.list_optimizations` — **absent**
- `PromptStudioDatabase.list_optimizations` — present (`:7378`), delegates blindly to `self._impl`

Verified by AST: 59 method names are implemented on both classes, 9 are PostgreSQL-only (incl. `list_optimizations`), 7 are SQLite-only.

Call path: the facade method calls `self._impl.list_optimizations(...)` -> AttributeError -> the endpoint's `_OPTIMIZATION_NONCRITICAL_EXCEPTIONS` tuple (`prompt_studio_optimization.py:80-91`) swallows it -> HTTP 500 "Failed to list optimizations". The real cause is never logged.

So `GET` project-optimizations is **100% broken on the default backend and works on PostgreSQL**.

Why no test caught it: both `tests/prompt_studio/integration/test_api_endpoints.py:369-380` and `tests/prompt_studio/unit/test_optimization_endpoint_error_mapping.py:32-37` substitute a stub class that defines `list_optimizations(self, *_args, **_kwargs)`. The stub is the reason the gap is invisible; there is no test of the happy path against a real `_SQLitePromptStudioDatabase`.

Root cause is the dual-backend triplication in this file (see the separate consolidation task): ~5,565 LOC of paired method bodies with nothing enforcing that the pair stays complete.

Found by the comprehensive core-module review; independently verified by the orchestrator via AST.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failing test calls the endpoint against a real _SQLitePromptStudioDatabase and reproduces the 500 before any fix
- [ ] #2 list_optimizations is implemented on the SQLite class with behaviour matching the PostgreSQL body
- [ ] #3 A parity guard test asserts the two backend classes expose the same public method set, or documents each intentional asymmetry (currently 9 PG-only and 7 SQLite-only)
- [ ] #4 The endpoint logs the underlying exception rather than only the generic message
- [ ] #5 The stub-based tests no longer mask a missing method (stub derives from or is checked against the real class)
- [ ] #6 Bandit run for touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DUAL-BACKEND TESTING GAP (2026-09-22) -- correcting an earlier assumption.

I had assumed the blocker for verifying dual-backend defects was that CI lacks a PostgreSQL fixture. That is wrong. Measured against ci.yml:

  full-suite-linux-312-shards    postgres=True   RAG_NEW=True  prompt_studio=True
  full-suite-linux-313-shards    postgres=True   RAG_NEW=True  prompt_studio=True
  full-suite-macos-312-shards    postgres=False  ...
  full-suite-windows-312-shards  postgres=False  ...

So the service exists, TEST_DATABASE_URL is exported, and both RAG_NEW and prompt_studio already run in Postgres-enabled shards on Linux.

The real gap is test design, not infrastructure:
  - tests/RAG_NEW/unit/test_slides_source_retrieval_hardening.py:88 replaces retrieve_slides_source_candidates_v1 with an AsyncMock, so NO SQL executes on either backend. That is why task-13292 (a PostgreSQL-only SQL defect, live since 2026-07-16) went unnoticed.
  - tests/RAG/conftest.py:dual_backend_env is the sqlite/postgres parametrized fixture and exists already -- but it is used by exactly TWO files (test_dual_backend_end_to_end.py, test_dual_backend_characters_retriever.py), neither covering Notes, ChatHistory or WorldBooks.

ACTIONABLE: extend dual_backend_env coverage to the retriever paths that carry per-backend SQL, rather than provisioning anything new. The same argument applies to this task's own defect: PromptStudioDatabase has 59 paired methods of which 51 are never called in any postgres-marked test, so porting list_optimizations without adding backend-parity coverage just resets the clock.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
