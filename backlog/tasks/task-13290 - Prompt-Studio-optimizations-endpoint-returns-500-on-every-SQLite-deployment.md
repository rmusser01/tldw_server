---
id: TASK-13290
title: Prompt Studio optimizations endpoint returns 500 on every SQLite deployment
status: Done
assignee: []
created_date: '2026-09-22 04:34'
updated_date: '2026-09-23 23:09'
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
- [x] #1 A failing test calls the endpoint against a real _SQLitePromptStudioDatabase and reproduces the 500 before any fix
- [x] #2 list_optimizations is implemented on the SQLite class with behaviour matching the PostgreSQL body
- [x] #3 A parity guard test asserts the two backend classes expose the same public method set, or documents each intentional asymmetry (currently 9 PG-only and 7 SQLite-only)
- [x] #4 The endpoint logs the underlying exception rather than only the generic message
- [x] #5 The stub-based tests no longer mask a missing method (stub derives from or is checked against the real class)
- [x] #6 Bandit run for touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
DONE (core fix). list_optimizations implemented on _SQLitePromptStudioDatabase, inserted beside the other optimization methods. Signature mirrors _BackendPromptStudioDatabase keyword-for-keyword - verified at runtime, inspect.signature equality is True - so the facade *args/**kwargs delegation cannot produce a TypeError. Uses the SQLite class conventions: get_connection()/cursor, "deleted = 0" (matching get_optimization, not the backend FALSE), and _row_to_dict(cursor, row) for the 2-arg arity.

Verification: new test tldw_Server_API/tests/prompt_studio/test_list_optimizations_sqlite.py (4 tests: existence, round-trip, filter+paginate, bad pagination) red before / green after. FULL prompt_studio suite: 1077 passed, 79 skipped.

The retry loops follow the established idiom of this class and carry a comment pointing at the backoff consolidation task; they should collapse when that lands.
NOTE: the endpoint at app/api/v1/endpoints/prompt_studio/prompt_studio_optimization.py is untouched - the defect was entirely in core. Remaining: Bandit on touched scope.

2026-09-23 reconciliation: AC2 met (7c348a05ae added it on SQLite; dee169a794/TASK-13318 then moved all optimization methods into prompt_studio_db/repositories/optimizations.py so both backends share one body; tests/prompt_studio/test_list_optimizations_sqlite.py passes). AC3 met (tests/DB_Management/test_prompt_studio_backend_parity.py ratchets signatures and documents the 3 intentional one-sided public methods; with test_backend_behaviour_parity.py: 24 passed, 0 skipped). AC6 met (uvx bandit on PromptStudioDatabase.py + repositories/optimizations.py: no issues). AC1 NOT met: the regression test calls PromptStudioDatabase directly, not the endpoint; no test drives GET /api/v1/prompt-studio/optimizations against a real SQLite DB. AC4 NOT met: prompt_studio_optimization.py:1017/1020 still log only 'Database error listing optimizations'/'Unexpected error listing optimizations' with no exception type/detail (endpoint untouched; note earlier commit 290e4fe7b6 deliberately sanitized these logs, so a fix should log at least the exception class). AC5 NOT met: _BrokenListOptimizations*Db in tests/prompt_studio/unit/test_optimization_endpoint_error_mapping.py:31-37 and _OptimizationDbWithoutPagination in integration/test_api_endpoints.py:373 are still free-standing stubs not derived from/checked against the real class.

2026-09-23 close-out (commit bcb8b9adea): AC1 - tests/prompt_studio/integration/test_api_endpoints.py::test_list_optimizations_endpoint_against_real_sqlite_db drives GET /api/v1/prompt-studio/optimizations/list/{project_id} through TestClient against a real SQLite PromptStudioDatabase. RED: with the facade restored to the pre-fix blind 'return self._impl.list_optimizations(...)' -> 500, AttributeError: '_SQLitePromptStudioDatabase' object has no attribute 'list_optimizations'. GREEN on current code (200, one row). Side finding while writing it: a row created with optimization_config=None/max_iterations=None also 500s (OptimizationResponse requires both); the create endpoints always set them, so the test uses realistic rows and this is not a live bug. AC4 - both list error branches now log the exception class ('... listing optimizations: {}', type(exc).__name__); message text stays out, preserving 290e4fe7b6 sanitization; sanitization tests updated to pin the class name. AC5 - new tests/prompt_studio/db_stub_contract.py::unservable_stub_methods(): a stub method must be implemented on the facade itself, or (if the facade forwards to self._impl explicitly or via __getattr__) exist on BOTH backend impls. Applied to every *Db stub in unit/test_optimization_endpoint_error_mapping.py and to _OptimizationDbWithoutPagination. RED under the old forwarder facade: 3 failed (stub contract test, stub pagination test, real-SQLite endpoint test); GREEN after. Suite: tests/prompt_studio + tests/DB_Management/test_prompt_studio_backend_parity.py -> 1162 passed, 17 skipped, 0 failed. Bandit on prompt_studio_optimization.py: no issues (only pre-existing nosec warnings). No doc change needed (internal test/log change).


Notes recorded on dev by the parallel core-review work (merged 2026-09-23):
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
FIXED 2026-09-22 in PR #2981. list_optimizations ported to _SQLitePromptStudioDatabase, mirroring the PostgreSQL implementation; the only dialect difference is `deleted = 0` rather than `deleted = FALSE`.
Verified against a real SQLite database rather than a stub -- the two existing endpoint tests substitute a stub defining list_optimizations(self, *_args, **_kwargs), which is exactly why the gap was invisible. 4 new tests, all red before the change. Full prompt_studio suite: 1145 passed, 17 skipped.
Also added a parity test asserting the two backend classes expose the same public surface, with the eight genuinely PostgreSQL-only methods listed by name. That is the cheap guard against the next missing method; it would have caught this one at edit time.
STILL OPEN, tracked separately: the decomposition itself. 59 method names implemented twice (~5,565 LOC of paired bodies) plus a third *args/**kwargs delegating facade, and 51 of the 59 are never called in any postgres-marked test. Porting one method does not address that -- the parity test is a guard, not a fix.
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
GET /api/v1/prompt-studio/optimizations/list/{project_id} was 500 on every SQLite deployment because list_optimizations existed only on the PostgreSQL backend. Fixed in 7c348a05ae, then unified in dee169a794 (TASK-13318) so both backends share one repository body; backend parity ratchet in test_prompt_studio_backend_parity.py. This close-out (bcb8b9adea) adds the endpoint-level regression against a real SQLite DB, logs the exception class on both list error paths (still no message text), and a stub contract check so hand-written DB stubs can no longer mask a one-backend gap. Known skip: 17 Postgres-dependent tests skip locally.
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
