# AuthNZ Follow-Up Remediation Design

## Summary

This spec covers follow-up remediation for three AuthNZ issues found after the
initial review fixes:

1. Lockout persistence is only identifier-scoped even though the public API is
   attempt-type scoped.
2. Fresh/bootstrap API key schemas still default omitted `scope` to `read`.
3. SQLite schema harmonization and API key schema validation can fail open in
   persisted runtimes.

The goal is to correct the storage contract, align bootstrap and test fixtures,
and keep explicit test-mode and in-memory flows flexible.

## Scope

In scope:

- [`tldw_Server_API/app/core/AuthNZ/lockout_tracker.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/lockout_tracker.py)
- [`tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py)
- [`tldw_Server_API/app/core/AuthNZ/migrations.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/migrations.py)
- [`tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py)
- [`tldw_Server_API/app/core/AuthNZ/database.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/database.py)
- [`tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py)
- [`tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql)
- AuthNZ tests and schema fixtures that currently encode the old contracts
- Schema-dependent tests outside `tldw_Server_API/tests/AuthNZ` that create or
  query AuthNZ `api_keys` rows and currently rely on omitted `scope`

Out of scope:

- Broad endpoint redesign
- Rewriting historical explicit `'read'` API key rows
- Changing password-history semantics in this pass
- Non-AuthNZ modules except for tests that directly depend on AuthNZ schema

## Locked Decisions

- Lockouts become attempt-type scoped end-to-end.
- Legacy `account_lockouts` rows migrate forward as `attempt_type='login'`.
- API key schemas stop granting implicit `read` for omitted `scope` going
  forward.
- Existing explicit `'read'` rows are preserved.
- SQLite persisted runtime schema drift becomes fail-fast.
- Explicit pytest/test-mode and in-memory SQLite paths remain permissive.

## Approaches Considered

### Recommended: Schema-First Remediation

Fix the storage contract, align bootstrap schema sources, and update tests to
the corrected behavior.

Pros:

- Removes the underlying drift instead of masking symptoms.
- Keeps runtime and schema behavior consistent.
- Makes future review findings easier to reason about.

Cons:

- Requires migration work and fixture alignment.
- Touches more files than a caller-only patch.

### Alternative: Runtime Guards Only

Keep existing schema shapes and compensate in code.

Pros:

- Lower migration risk in the short term.

Cons:

- Leaves storage semantics inconsistent.
- Preserves future maintenance traps.

### Rejected: Minimal Hotfixes

Patch only the most visible callers and add narrow tests.

Reason rejected:

- Too likely to leave the same blind spots in bootstrap and persistence layers.

## Design

### 1. Lockout Scoping

`account_lockouts` will store `attempt_type` and enforce uniqueness on
`(identifier, attempt_type)`.

SQLite:

- Add AuthNZ migration `84` to rebuild or migrate `account_lockouts` into the
  new shape.
- Legacy identifier-only rows are copied into the new table with
  `attempt_type='login'`.

PostgreSQL/backstop:

- [`rate_limits_repo.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py)
  schema ensure logic will detect the old identifier-only shape and upgrade it
  to attempt-type scoped form.
- That upgrade must be idempotent and safe under concurrent first-use
  initialization, rather than assuming a single-writer bootstrap path.

Runtime:

- `record_failed_attempt_and_lockout(...)` persists lockouts per
  `(identifier, attempt_type)`.
- `get_active_lockout(...)` looks up by `(identifier, attempt_type)`.
- `reset_failed_attempts_and_lockout(...)` only clears the matching attempt
  type.
- [`lockout_tracker.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/lockout_tracker.py)
  will stop dropping `attempt_type` on lookup and reset.
- The repo contract changes accordingly: `get_active_lockout(...)` and
  `reset_failed_attempts_and_lockout(...)` become attempt-type aware signatures,
  and test doubles/stubs that currently model identifier-only lockouts must be
  updated with the new method shape.

Fixture alignment:

- AuthNZ Postgres schema builders in
  [`tldw_Server_API/tests/AuthNZ/conftest.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/AuthNZ/conftest.py)
  must move to the new `account_lockouts` shape.
- DDL expectation tests such as
  [`test_rate_limiter_bootstrap.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_bootstrap.py)
  and
  [`test_authnz_rate_limits_repo_backend_selection.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py)
  must be updated to assert attempt-type-aware SQL.

### 2. API Key Scope Defaults

Future omitted `scope` values must not implicitly authorize `read`.

SQLite:

- Add AuthNZ migration `85` to remove the `DEFAULT 'read'` behavior from the
  persisted `api_keys.scope` column.
- Migration preserves existing row values and only changes future omission
  behavior.

PostgreSQL:

- Backstop/bootstrap DDL drops the default from `api_keys.scope` in place.

Bootstrap and schema sources:

- Update
  [`sqlite_users.sql`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql)
  to remove `DEFAULT 'read'`.
- Update AuthNZ Postgres test schema builders in
  [`tldw_Server_API/tests/AuthNZ/conftest.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/AuthNZ/conftest.py)
  to remove the same default.

Runtime expectations:

- Production repo insert paths already pass explicit `scope`, so repo logic
  should need little or no behavioral change.
- Tests or fixtures that insert `api_keys` without `scope` must be updated to
  pass explicit scope where they intend read-only behavior.

### 3. SQLite Fail-Fast Hardening

SQLite persisted runtime schema drift should fail early instead of surfacing as
later query-time breakage.

The runtime/test gate must be defined once and reused, not reimplemented in
multiple modules. Introduce a canonical helper or predicate in the AuthNZ
database layer and have both [`database.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/database.py)
and [`api_keys_repo.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py)
consume that shared rule.

In
[`database.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/database.py):

- `ensure_authnz_tables(...)` harmonization failures will raise for persisted
  runtime SQLite paths.
- The permissive path remains only when one of these is true:
  - explicit test mode is active
  - explicit pytest runtime is active
  - SQLite path is `:memory:`

In
[`api_keys_repo.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py):

- SQLite schema readiness must validate required columns, not just table
  existence.
- Missing required columns should produce a clear bootstrap/schema error in
  persisted runtime paths.
- Test-mode and in-memory contexts may keep the lighter behavior if needed by
  existing harnesses.

### 4. Migration Numbering

Reserve these new AuthNZ migration versions:

- `84`: attempt-type scoped `account_lockouts`
- `85`: remove implicit API key `scope` default

No other migration renumbering is part of this design.

## Testing Strategy

This work will be implemented with TDD.

Required failing tests first:

- Lockout isolation by `attempt_type`
- Lockout reset only clears the requested `attempt_type`
- Legacy lockout migration maps identifier-only rows to `login`
- Fresh/bootstrap API key schema no longer assigns implicit `read`
- SQLite persisted-schema drift fails fast
- Explicit test-mode or `:memory:` SQLite paths remain permissive

Test categories:

- Unit tests for repo/tracker behavior
- Migration-sensitive tests for SQLite schema evolution
- Bootstrap/schema tests for API key readiness checks
- Fixture/schema expectation updates where old DDL is asserted
- Schema-dependent tests outside `tldw_Server_API/tests/AuthNZ` that rely on
  omitted `scope` must be updated or explicitly justified

## Risks And Mitigations

### Risk: Test Fixture Fallout

Many AuthNZ fixtures still encode the old schema defaults.

Mitigation:

- Treat fixture alignment as a first-class workstream, not cleanup.

### Risk: Runtime/Test Gate Too Broad

Using path shape alone would incorrectly hard-fail many temp-file tests.

Mitigation:

- Use existing runtime detection helpers instead of inventing a new heuristic.

### Risk: Historical Migration Ambiguity

Editing only old migrations can produce confusing fresh-install versus migrated
behavior.

Mitigation:

- Use new migrations for persisted behavior changes and align bootstrap/schema
  sources separately.

## Verification Contract

Before completion:

- Run focused AuthNZ tests covering new lockout, migration, API key schema, and
  SQLite fail-fast behavior.
- Re-run the earlier remediation tests to ensure no regression in session key,
  refresh rotation, API key validation, and MFA backend support.
- Attempt Bandit on touched AuthNZ paths if the tool is available in the venv;
  otherwise report the environment limitation with evidence.

## Non-Goals

- No broad auth endpoint refactor
- No password-history fail-open remediation in this spec
- No data rewrite of existing explicit `'read'` API key rows
