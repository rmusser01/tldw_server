# PR 908 Open Review Cleanup Design

## Context

PR 908 already landed the larger Jobs and metering persistence boundary redesign, but four review threads remain open:

- `JobManager.create_job()` performs the Postgres fair-share count before `_pg_cursor()` applies RLS session state.
- `Jobs_Repository.py` lacks module/class/method docstrings that explain the new transaction boundary.
- `AuthNZ_Metering_Repository.py` lacks module/class/method docstrings that explain the repository contracts.
- `TrackingJobsRepository` in the fair-share integration tests omits type hints on new helper methods.

The first item is a correctness bug. The other three are narrow compliance and maintainability gaps.

## Goals

- Preserve the architectural direction already merged into PR 908.
- Fix the Postgres fair-share bug without broadening the scope of the repository redesign.
- Close the documentation and typing review threads with minimal churn.
- Add regression coverage that proves fair-share counting runs after Postgres cursor setup.

## Non-Goals

- Reworking the repository boundaries introduced in PR 908.
- Expanding the repository API beyond what the current orchestration path needs.
- Refactoring unrelated `JobManager` create-time branches or metering flows.

## Root Cause

`JobManager.create_job()` currently opens a raw connection, immediately constructs a `JobsSession`, and runs the fair-share count against that session before entering `with self._pg_cursor(conn)`. On PostgreSQL, `_pg_cursor()` is the code path that applies `SET ROLE` and the `app.*` session variables used by RLS. That means the repository count query may execute without the required RLS context. If it fails, `_count_active_jobs_for_user()` catches the database exception and returns `0`, which silently disables fair-share admission control.

SQLite is unaffected because it does not use `_pg_cursor()` for access control.

## Recommended Approach

### 1. Split fair-share timing by backend

Keep the existing early fair-share block for SQLite, but defer the fair-share count for PostgreSQL until after `with self._pg_cursor(conn)` has been entered. This keeps the current repository-backed session reuse while ensuring the Postgres session has the correct RLS state before the repository issues the count query.

### 2. Reuse one repository session for count and insert

Continue to use a single `JobsSession` object for the create path so the fair-share count and repository insert stay on the same live connection. This preserves the boundary redesign’s main benefit without requiring a larger session-factory refactor in this PR.

### 3. Add boundary documentation in place

Add concise module, class, and method docstrings to:

- `tldw_Server_API/app/core/DB_Management/Jobs_Repository.py`
- `tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py`

The docstrings should describe backend differences, transaction ownership, and the returned row shapes rather than restating obvious implementation details.

### 4. Tighten the regression tests

Extend `tldw_Server_API/tests/Jobs/test_fair_share_integration.py` with:

- explicit type hints on `TrackingJobsRepository`
- a regression test that proves the Postgres branch performs the fair-share count only after `_pg_cursor()` has been entered

The regression can be implemented with a fake Postgres connection and instrumented `_pg_cursor()` / repository methods rather than a full live Postgres fixture. The point is sequencing, not SQL correctness.

## Data Flow After The Fix

### SQLite create path

1. `create_job()` opens the repository session.
2. Fair-share count runs immediately using that session.
3. Priority and admission checks complete.
4. Existing SQLite insert path runs.

### Postgres create path

1. `create_job()` opens the connection and repository session.
2. Code enters `with conn: with self._pg_cursor(conn)`.
3. Fair-share count runs using the same repository session, now with RLS state applied.
4. Quota, idempotency, and insert logic run on that same connection.

## Testing Strategy

- Add the new Postgres-ordering regression first and confirm it fails against the current code.
- Keep the existing fair-share integration coverage green for SQLite behavior.
- Re-run the repository and Stripe metering tests after the docstring changes to make sure the cleanup does not affect behavior.
- Run Bandit on the touched Jobs and metering files before pushing.

## Risks

- Moving the fair-share block must not change SQLite behavior or create double evaluation on Postgres.
- The Postgres regression test should be narrow and deterministic; it should assert sequencing rather than replicate the full create flow.
- Docstring additions must stay concise enough that they do not become stale quickly.

## Implementation Notes

- `JobManager` now centralizes fair-share submission enforcement in `_apply_fair_share_submission_policy(...)`.
- SQLite still performs fair-share evaluation immediately after the repository session is opened.
- PostgreSQL now performs the same check only after `_pg_cursor()` has applied RLS session state, while reusing the same `JobsSession` for the active-job count and insert.
- The fair-share integration tests now include a fake-Postgres regression that proves the repository count happens after `_pg_cursor()` is entered.
- `Jobs_Repository.py` and `AuthNZ_Metering_Repository.py` now include module/class/method docstrings describing the persistence boundary and normalized return contracts.

## Verification

- `python -m pytest tldw_Server_API/tests/Billing/test_authnz_metering_repository.py tldw_Server_API/tests/Jobs/test_fair_share_integration.py tldw_Server_API/tests/Jobs/test_jobs_repository.py tldw_Server_API/tests/test_stripe_metering.py -v`
  - Result: `41 passed, 5 warnings`
- `python -m bandit -r tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/core/DB_Management/Jobs_Repository.py tldw_Server_API/app/core/DB_Management/AuthNZ_Metering_Repository.py -f json -o /tmp/bandit_pr908_open_review_cleanup.json`
  - Result: `0` findings, `0` errors
