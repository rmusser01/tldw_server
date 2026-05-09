## Stage 1: Verify Repository SQL Review Findings
**Goal**: Confirm the actionable prototype repository comments against current code.
**Success Criteria**: Focused tests fail for PostgreSQL table discovery, pre-read state updates, and Python-side active-session filtering.
**Tests**: Focused pytest tests in `tldw_Server_API/tests/PrototypeWorkspaces/test_prototype_repo.py`.
**Status**: Complete

Notes:
- Focused red run failed all three new tests for the expected query-shape/pre-read reasons.

## Stage 2: Implement Backend-Aware Repository Queries
**Goal**: Keep placeholder handling delegated to `DatabasePool` while fixing backend-specific table discovery and query-level filtering/update behavior.
**Success Criteria**: `ensure_tables()` uses `information_schema` on PostgreSQL, state updates preserve existing values with SQL-level `COALESCE`, and `find_active_session()` filters active candidates in SQL.
**Tests**: Focused pytest and full `PrototypeWorkspaces` regression.
**Status**: Complete

Notes:
- `DatabasePool` already translates `?` placeholders to `$n` on the PostgreSQL path, so the placeholder-only review item is addressed with a technical reply rather than duplicating placeholder conversion in this repo.
- `ensure_tables()` now branches table discovery for PostgreSQL instead of querying `sqlite_master`.
- `update_workspace_state()` now preserves unspecified fields with SQL-level `COALESCE` instead of reading the row and writing a merged copy.
- `find_active_session()` now uses a static SQL query with optional actor filters and active-session predicates, avoiding broad workspace reads and dynamic SQL construction.

## Stage 3: Verify and Publish
**Goal**: Run focused/regression tests, Bandit on touched backend code, diff checks, then push/reply to relevant PR threads.
**Success Criteria**: Local verification passes and repo SQL review threads have technical replies or fixes.
**Tests**: Focused pytest, full PrototypeWorkspaces pytest, Bandit, `git diff --check`.
**Status**: Complete

Notes:
- Focused pytest for the three new repository tests passed.
- Full `tldw_Server_API/tests/PrototypeWorkspaces` regression passed with 63 tests.
- `ruff check` passed for the touched repository and test files.
- Bandit passed on `tldw_Server_API/app/core/AuthNZ/repos/prototype_workspaces_repo.py` after replacing the dynamic query with a static optional-filter query.
- `git diff --check` passed.
