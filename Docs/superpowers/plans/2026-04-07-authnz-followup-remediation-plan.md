# AuthNZ Follow-Up Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix AuthNZ lockout scoping, remove implicit API-key scope defaults from persisted/bootstrap schema, and make persisted SQLite schema drift fail fast without breaking explicit test-mode or in-memory flows.

**Architecture:** Add two new AuthNZ migrations (`84` and `85`) to correct persisted SQLite schema, update Postgres/backstop DDL to the same contract, and introduce one shared SQLite schema-strictness predicate reused by both `DatabasePool` and `AuthnzApiKeysRepo`. The runtime code stays storage-first: repo/tracker APIs follow the corrected schema, and schema files plus fixtures are aligned so fresh installs and migrated installs behave the same.

**Tech Stack:** Python 3.12+, FastAPI/AuthNZ core, SQLite, PostgreSQL/asyncpg, aiosqlite, pytest

---

## File Map

- `tldw_Server_API/app/core/AuthNZ/lockout_tracker.py`
  Responsibility: pass `attempt_type` through all lockout lookups/resets.
- `tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py`
  Responsibility: own DDL/backstop logic and DB queries for `failed_attempts` and `account_lockouts`.
- `tldw_Server_API/app/core/AuthNZ/migrations.py`
  Responsibility: define SQLite migrations `84` and `85`, then register them.
- `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
  Responsibility: align Postgres API-key bootstrap DDL with no `scope` default.
- `tldw_Server_API/app/core/AuthNZ/database.py`
  Responsibility: host the canonical SQLite schema-strictness predicate and fail-fast harmonization behavior.
- `tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py`
  Responsibility: validate required SQLite `api_keys`/`api_key_audit_log` columns under the shared strictness gate.
- `tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql`
  Responsibility: fresh SQLite bootstrap schema, must match post-migration behavior.
- `tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py`
  Responsibility: lockout tracker semantics using stubs.
- `tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_lockout_reset.py`
  Responsibility: backward-compatible rate-limiter lockout behavior.
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py`
  Responsibility: assert backend-specific lockout SQL.
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py`
  Responsibility: DB-backed Postgres lockout behavior.
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py`
  Responsibility: migration-level SQLite API-key schema coverage.
- `tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py`
  Responsibility: Postgres API-key bootstrap DDL coverage.
- `tldw_Server_API/tests/AuthNZ/conftest.py`
  Responsibility: Postgres fixture schema builders; must stop encoding old lockout/API-key contracts.
- `tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py`
  Responsibility: representative non-AuthNZ SQLite test that inserts `api_keys`.
- `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py`
  Responsibility: representative SQLite repo test that inserts `api_keys`.
- `tldw_Server_API/tests/AuthNZ/unit/test_database_sqlite_schema_fail_fast.py`
  Create: focused unit coverage for `DatabasePool` SQLite strictness gate.
- `tldw_Server_API/tests/AuthNZ/unit/test_api_keys_repo_schema_strictness.py`
  Create: focused unit coverage for `AuthnzApiKeysRepo.ensure_tables()` strictness behavior.
- `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_lockout_scope.py`
  Create: migration-level SQLite coverage for `account_lockouts` scoping.

### Task 1: Write Lockout Regression Tests

**Files:**
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_lockout_scope.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_lockout_reset.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_bootstrap.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py`
- Modify: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py`

- [ ] **Step 1: Write the failing tracker-scoping regression**

```python
@pytest.mark.asyncio
async def test_check_lockout_is_scoped_by_attempt_type():
    tracker = _make_tracker()
    identifier = "user:alice"

    for _ in range(3):
        await tracker.record_failed_attempt(identifier, attempt_type="login")

    login_locked, _ = await tracker.check_lockout(identifier, attempt_type="login")
    reset_locked, _ = await tracker.check_lockout(identifier, attempt_type="password_reset")

    assert login_locked is True
    assert reset_locked is False


@pytest.mark.asyncio
async def test_reset_failed_attempts_only_clears_matching_attempt_type():
    tracker = _make_tracker()
    identifier = "user:bob"

    for _ in range(3):
        await tracker.record_failed_attempt(identifier, attempt_type="login")
        await tracker.record_failed_attempt(identifier, attempt_type="password_reset")

    await tracker.reset_failed_attempts(identifier, attempt_type="login")

    login_locked, _ = await tracker.check_lockout(identifier, attempt_type="login")
    reset_locked, _ = await tracker.check_lockout(identifier, attempt_type="password_reset")

    assert login_locked is False
    assert reset_locked is True
```

- [ ] **Step 2: Write the failing migration/backstop regression**

```python
def test_migration_084_scopes_account_lockouts_and_maps_legacy_rows() -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import (
        migration_011_add_enhanced_auth_tables,
        migration_084_scope_account_lockouts_by_attempt_type,
    )

    conn = sqlite3.connect(":memory:")
    try:
        migration_011_add_enhanced_auth_tables(conn)
        conn.execute(
            """
            INSERT INTO account_lockouts (identifier, locked_until, reason)
            VALUES ('legacy-user', '2030-01-01T00:00:00+00:00', 'legacy')
            """
        )

        migration_084_scope_account_lockouts_by_attempt_type(conn)

        cols = {row[1] for row in conn.execute("PRAGMA table_info(account_lockouts)").fetchall()}
        assert "attempt_type" in cols

        row = conn.execute(
            """
            SELECT identifier, attempt_type, reason
            FROM account_lockouts
            WHERE identifier = 'legacy-user'
            """
        ).fetchone()
        assert row == ("legacy-user", "login", "legacy")
    finally:
        conn.close()
```

- [ ] **Step 3: Update backend-selection expectations to the new SQL shape**

```python
ddl_blob = "\n".join(pool.conn.queries)
assert "CREATE TABLE IF NOT EXISTS account_lockouts" in ddl_blob
assert "attempt_type TEXT NOT NULL" in ddl_blob
assert "PRIMARY KEY (identifier, attempt_type)" in ddl_blob
```

```python
assert "insert into account_lockouts" in lockout_queries[0].lower()
assert "attempt_type" in lockout_queries[0].lower()
assert "values (?, ?, ?, ?)" in lockout_queries[0].lower()

query, params = conn.fetchrow_calls[0]
assert "where identifier = $1 and attempt_type = $2 and locked_until > $3" in query.lower()
assert params[:2] == ("id-3", "login")
```

- [ ] **Step 4: Run the focused lockout regressions and verify they fail for the right reason**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_lockout_reset.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_bootstrap.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_lockout_scope.py
```

Expected:

- tracker/reset tests fail because lockouts are still identifier-only
- bootstrap DDL assertions fail because the generated lockout table is still identifier-only
- backend-selection assertions fail because SQL still omits `attempt_type`
- migration test fails because migration `84` does not exist yet

- [ ] **Step 5: Commit only the red lockout tests if you are working in an isolated branch**

```bash
git add -- \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_lockout_scope.py \
  tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_lockout_reset.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_bootstrap.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py
git commit -m "test: add authnz lockout scoping regressions"
```

### Task 2: Implement Lockout Scoping End-To-End

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/lockout_tracker.py`
- Modify: `tldw_Server_API/tests/AuthNZ/conftest.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_bootstrap.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_lockout_scope.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_lockout_reset.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py`

- [ ] **Step 1: Add SQLite migration `84` and register it after migration `83`**

```python
def migration_084_scope_account_lockouts_by_attempt_type(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS account_lockouts_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT NOT NULL,
            attempt_type TEXT NOT NULL,
            locked_until TIMESTAMP NOT NULL,
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(identifier, attempt_type)
        )
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO account_lockouts_new
            (identifier, attempt_type, locked_until, reason, created_at)
        SELECT identifier,
               'login',
               locked_until,
               reason,
               COALESCE(created_at, CURRENT_TIMESTAMP)
        FROM account_lockouts
        """
    )
    conn.execute("DROP TABLE IF EXISTS account_lockouts")
    conn.execute("ALTER TABLE account_lockouts_new RENAME TO account_lockouts")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_lockouts_identifier_attempt_type "
        "ON account_lockouts(identifier, attempt_type)"
    )
    conn.commit()


Migration(
    84,
    "Scope account lockouts by attempt type",
    migration_084_scope_account_lockouts_by_attempt_type,
)
```

- [ ] **Step 2: Make repo DDL and queries attempt-type aware**

```python
CREATE TABLE IF NOT EXISTS account_lockouts (
    identifier TEXT NOT NULL,
    attempt_type TEXT NOT NULL,
    locked_until TEXT NOT NULL,
    reason TEXT,
    PRIMARY KEY (identifier, attempt_type)
)
```

```python
await conn.execute(
    """
    INSERT INTO account_lockouts (identifier, attempt_type, locked_until, reason)
    VALUES ($1, $2, $3, $4)
    ON CONFLICT (identifier, attempt_type) DO UPDATE SET
        locked_until = $3,
        reason = $4
    """,
    identifier,
    attempt_type,
    lockout_expires,
    reason,
)
```

> **NOTE:** The snippet below uses PostgreSQL-specific features (`information_schema.columns`,
> `LOCK TABLE`). Guard this code path with a backend-type check (e.g.
> `if backend.is_postgres:`) and use the appropriate SQLite introspection
> (`PRAGMA table_info(...)`) when running against SQLite.

```python
# IMPORTANT: Only execute this block when the backend is PostgreSQL.
# For SQLite use PRAGMA table_info('account_lockouts') instead of information_schema.
if not backend.is_postgres:
    raise RuntimeError("This migration path is PostgreSQL-only; see SQLite migration 84/85.")

columns = await conn.fetch(
    """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = 'account_lockouts'
    """
)
column_names = {row["column_name"] for row in columns}
if "attempt_type" not in column_names:
    await conn.execute("LOCK TABLE account_lockouts IN ACCESS EXCLUSIVE MODE")
    await conn.execute("ALTER TABLE account_lockouts RENAME TO account_lockouts_legacy")
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS account_lockouts (
            identifier TEXT NOT NULL,
            attempt_type TEXT NOT NULL,
            locked_until TIMESTAMPTZ NOT NULL,
            reason TEXT,
            PRIMARY KEY (identifier, attempt_type)
        )
        """
    )
    await conn.execute(
        """
        INSERT INTO account_lockouts (identifier, attempt_type, locked_until, reason)
        SELECT identifier, 'login', locked_until, reason
        FROM account_lockouts_legacy
        ON CONFLICT (identifier, attempt_type) DO NOTHING
        """
    )
    await conn.execute("DROP TABLE IF EXISTS account_lockouts_legacy")
```

```python
row = await conn.fetchrow(
    """
    SELECT locked_until
    FROM account_lockouts
    WHERE identifier = $1
      AND attempt_type = $2
      AND locked_until > $3
    """,
    identifier,
    attempt_type,
    now,
)
return row["locked_until"] if row else None
```

```python
await conn.execute(
    "DELETE FROM account_lockouts WHERE identifier = $1 AND attempt_type = $2",
    identifier,
    attempt_type,
)
await conn.execute(
    "DELETE FROM failed_attempts WHERE identifier = $1 AND attempt_type = $2",
    identifier,
    attempt_type,
)
```

- [ ] **Step 3: Thread the new signatures through `LockoutTracker` and update fixture DDL**

```python
locked_until = await repo.get_active_lockout(
    identifier=identifier,
    attempt_type=attempt_type,
    now=datetime.now(timezone.utc),
)
```

```python
await repo.reset_failed_attempts_and_lockout(
    identifier=identifier,
    attempt_type=attempt_type,
)
```

```sql
CREATE TABLE IF NOT EXISTS account_lockouts (
    identifier TEXT NOT NULL,
    attempt_type TEXT NOT NULL,
    locked_until TIMESTAMPTZ NOT NULL,
    reason TEXT,
    PRIMARY KEY (identifier, attempt_type)
)
```

- [ ] **Step 4: Run the lockout suite again and verify it is green**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_lockout_reset.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_bootstrap.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_lockout_scope.py
```

Expected:

- `0 failed`
- new migration and SQL-shape assertions pass
- reset only clears the requested `attempt_type`

- [ ] **Step 5: Commit the lockout remediation**

```bash
git add -- \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py \
  tldw_Server_API/app/core/AuthNZ/lockout_tracker.py \
  tldw_Server_API/tests/AuthNZ/conftest.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_bootstrap.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_lockout_scope.py \
  tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_lockout_reset.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py
git commit -m "fix: scope authnz lockouts by attempt type"
```

### Task 3: Write API-Key Default And SQLite Strictness Regressions

**Files:**
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_database_sqlite_schema_fail_fast.py`
- Create: `tldw_Server_API/tests/AuthNZ/unit/test_api_keys_repo_schema_strictness.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py`

- [ ] **Step 1: Add the failing SQLite migration/default regression**

```python
def test_migration_085_removes_api_keys_scope_default() -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import (
        migration_001_create_users_table,
        migration_003_create_api_keys_table,
        migration_004_create_api_key_audit_log,
        migration_085_remove_api_keys_scope_default,
    )

    conn = sqlite3.connect(":memory:")
    try:
        migration_001_create_users_table(conn)
        migration_003_create_api_keys_table(conn)
        migration_004_create_api_key_audit_log(conn)

        migration_085_remove_api_keys_scope_default(conn)

        columns = conn.execute("PRAGMA table_info(api_keys)").fetchall()
        scope_row = next(row for row in columns if row[1] == "scope")
        assert scope_row[4] is None
    finally:
        conn.close()
```

- [ ] **Step 2: Add the failing Postgres/bootstrap and strictness regressions**

```python
@pytest.mark.asyncio
async def test_ensure_api_keys_tables_pg_does_not_emit_scope_default() -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_api_keys_tables_pg

    pool = _StubPostgresPool()
    ok = await ensure_api_keys_tables_pg(pool)

    assert ok is True
    ddl_blob = "\n".join(pool.executed_sql)
    assert "scope VARCHAR(50) DEFAULT 'read'" not in ddl_blob
    assert "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS scope VARCHAR(50) DEFAULT 'read'" not in ddl_blob
```

```python
@pytest.mark.asyncio
async def test_database_pool_raises_when_sqlite_harmonization_fails_in_strict_mode(monkeypatch, tmp_path):
    settings = Settings(AUTH_MODE="multi_user", DATABASE_URL=f"sqlite:///{tmp_path / 'strict.db'}", JWT_SECRET_KEY="x" * 64)
    pool = DatabasePool(settings)

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.ensure_authnz_tables", lambda _path: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.is_test_mode", lambda: False)
    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.is_explicit_pytest_runtime", lambda: False)

    with pytest.raises(RuntimeError, match="boom"):
        await pool.initialize()
```

```python
@pytest.mark.asyncio
async def test_api_keys_repo_ensure_tables_raises_on_missing_scope_column_in_strict_mode():
    pool = _StrictSqlitePool(
        api_keys_columns={"id", "user_id", "key_hash"},
        audit_columns={"id", "key_id", "action"},
        strict=True,
    )
    repo = AuthnzApiKeysRepo(pool)

    with pytest.raises(RuntimeError, match="scope"):
        await repo.ensure_tables()
```

```python
@pytest.mark.asyncio
async def test_database_pool_allows_sqlite_harmonization_failure_in_test_mode(monkeypatch, tmp_path):
    settings = Settings(AUTH_MODE="multi_user", DATABASE_URL=f"sqlite:///{tmp_path / 'test-mode.db'}", JWT_SECRET_KEY="x" * 64)
    pool = DatabasePool(settings)

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.ensure_authnz_tables", lambda _path: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.is_test_mode", lambda: True)
    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.is_explicit_pytest_runtime", lambda: False)

    await pool.initialize()
```

```python
@pytest.mark.asyncio
async def test_api_keys_repo_ensure_tables_allows_missing_scope_when_shared_gate_is_off(monkeypatch):
    pool = _StrictSqlitePool(
        api_keys_columns={"id", "user_id", "key_hash"},
        audit_columns={"id", "key_id", "action"},
        strict=True,
    )
    repo = AuthnzApiKeysRepo(pool)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo.should_enforce_sqlite_schema_strictness",
        lambda _path: False,
    )

    await repo.ensure_tables()
```

- [ ] **Step 3: Run the focused default/strictness regressions and verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_database_sqlite_schema_fail_fast.py \
  tldw_Server_API/tests/AuthNZ/unit/test_api_keys_repo_schema_strictness.py
```

Expected:

- migration `85` test fails because the migration does not exist yet
- Postgres DDL test fails because emitted SQL still includes `DEFAULT 'read'`
- strictness tests fail because harmonization/column validation still fail open
- permissive-mode tests fail until both modules consume the shared gating rule

- [ ] **Step 4: Audit schema-dependent tests outside `tests/AuthNZ` for omitted `scope`**

Run:

```bash
rg -n "INSERT INTO api_keys \\(" \
  tldw_Server_API/tests/Usage \
  tldw_Server_API/tests/AuthNZ_SQLite \
  tldw_Server_API/tests/AuthNZ/integration \
  tldw_Server_API/tests/AuthNZ/unit
```

Expected:

- identify any test setup that semantically expects a read-capable API key when `scope` is omitted
- record exact files that need explicit `'read'` added during Task 4

- [ ] **Step 5: Commit only the red regressions if you are working in an isolated branch**

```bash
git add -- \
  tldw_Server_API/tests/AuthNZ/unit/test_database_sqlite_schema_fail_fast.py \
  tldw_Server_API/tests/AuthNZ/unit/test_api_keys_repo_schema_strictness.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py
git commit -m "test: add authnz schema hardening regressions"
```

### Task 4: Implement API-Key Schema Hardening And Shared SQLite Strictness

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/database.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py`
- Modify: `tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql`
- Modify: `tldw_Server_API/tests/AuthNZ/conftest.py`
- Modify: `tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py`
- Modify: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_database_sqlite_schema_fail_fast.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_api_keys_repo_schema_strictness.py`

- [ ] **Step 1: Add migration `85` and align bootstrap schema sources**

```python
def migration_085_remove_api_keys_scope_default(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            key_hash TEXT UNIQUE NOT NULL,
            key_id TEXT,
            key_prefix TEXT,
            name TEXT,
            description TEXT,
            scope TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            last_used_at TIMESTAMP,
            last_used_ip TEXT,
            usage_count INTEGER DEFAULT 0,
            rate_limit INTEGER,
            allowed_ips TEXT,
            metadata TEXT,
            rotated_from INTEGER REFERENCES api_keys(id),
            rotated_to INTEGER REFERENCES api_keys(id),
            revoked_at TIMESTAMP,
            revoked_by INTEGER,
            revoke_reason TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO api_keys_new (
            id, user_id, key_hash, key_id, key_prefix, name, description, scope,
            status, created_at, expires_at, last_used_at, last_used_ip, usage_count,
            rate_limit, allowed_ips, metadata, rotated_from, rotated_to, revoked_at,
            revoked_by, revoke_reason
        )
        SELECT
            id, user_id, key_hash, key_id, key_prefix, name, description, scope,
            status, created_at, expires_at, last_used_at, last_used_ip, usage_count,
            rate_limit, allowed_ips, metadata, rotated_from, rotated_to, revoked_at,
            revoked_by, revoke_reason
        FROM api_keys
        """
    )
    conn.execute("DROP TABLE IF EXISTS api_keys")
    conn.execute("ALTER TABLE api_keys_new RENAME TO api_keys")
    conn.commit()
```

```sql
scope TEXT,
```

```sql
scope VARCHAR(50),
```

- [ ] **Step 2: Introduce one shared SQLite schema-strictness predicate in `database.py`**

```python
def should_enforce_sqlite_schema_strictness(sqlite_fs_path: str | None) -> bool:
    if not sqlite_fs_path or sqlite_fs_path == ":memory:":
        return False
    if is_test_mode() or is_explicit_pytest_runtime():
        return False
    return True
```

```python
if self._sqlite_fs_path and self._sqlite_fs_path != ":memory:":
    try:
        await asyncio.to_thread(ensure_authnz_tables, Path(self._sqlite_fs_path))
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS:
        if should_enforce_sqlite_schema_strictness(self._sqlite_fs_path):
            raise
        logger.debug(
            "Skipping SQLite AuthNZ schema harmonization failure in permissive mode for {}",
            self._sqlite_fs_path,
        )
```

- [ ] **Step 3: Harden `AuthnzApiKeysRepo.ensure_tables()` under the shared gate**

```python
required_api_keys_columns = {
    "id",
    "user_id",
    "key_hash",
    "scope",
    "status",
    "created_at",
}
required_audit_columns = {"id", "key_id", "action", "created_at"}
```

```python
if should_enforce_sqlite_schema_strictness(getattr(self.db_pool, "_sqlite_fs_path", None)):
    api_key_cols = {
        row["name"] if isinstance(row, dict) else row[1]
        for row in await self.db_pool.fetchall("PRAGMA table_info(api_keys)")
    }
    missing = sorted(required_api_keys_columns - api_key_cols)
    if missing:
        raise RuntimeError(f"SQLite api_keys schema missing required columns: {', '.join(missing)}")
```

- [ ] **Step 4: Align fixture/test schema builders and explicit-scope inserts, then run the focused green suite**

```sql
scope VARCHAR(50),
```

```python
await pool.execute(
    "INSERT INTO api_keys (user_id, key_hash, name, scope) VALUES (?, ?, ?, ?)",
    1,
    key_hash,
    "DerivedName",
    "read",
)
```

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_database_sqlite_schema_fail_fast.py \
  tldw_Server_API/tests/AuthNZ/unit/test_api_keys_repo_schema_strictness.py \
  tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py
```

Expected:

- `0 failed`
- migration `85` removes the SQLite default
- Postgres DDL no longer emits `DEFAULT 'read'`
- strict mode raises on persisted drift but not on test/in-memory contexts

- [ ] **Step 5: Commit the API-key/default and SQLite strictness remediation**

```bash
git add -- \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/AuthNZ/database.py \
  tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py \
  tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql \
  tldw_Server_API/tests/AuthNZ/conftest.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_database_sqlite_schema_fail_fast.py \
  tldw_Server_API/tests/AuthNZ/unit/test_api_keys_repo_schema_strictness.py \
  tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py
git commit -m "fix: harden authnz api key schema bootstrap"
```

### Task 5: Final Verification And Review

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-07-authnz-followup-remediation-plan.md`
- Verify: all touched files from Tasks 1-4

- [ ] **Step 1: Run the complete focused AuthNZ remediation suite**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_lockout_scope.py \
  tldw_Server_API/tests/AuthNZ/unit/test_lockout_tracker.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_lockout_reset.py \
  tldw_Server_API/tests/AuthNZ/unit/test_rate_limiter_bootstrap.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_rate_limits_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_rate_limits_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_api_keys.py \
  tldw_Server_API/tests/AuthNZ/unit/test_database_sqlite_schema_fail_fast.py \
  tldw_Server_API/tests/AuthNZ/unit/test_api_keys_repo_schema_strictness.py \
  tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py
```

Expected:

- `0 failed`

- [ ] **Step 2: Re-run the earlier remediation regressions to prove no rollback**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py \
  tldw_Server_API/tests/AuthNZ/integration/test_jwt_refresh_rotation_blacklist.py \
  tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py \
  tldw_Server_API/tests/AuthNZ/unit/test_api_key_manager_validation.py \
  tldw_Server_API/tests/AuthNZ/unit/test_mfa_backend_support.py \
  tldw_Server_API/tests/AuthNZ/integration/test_mfa_service.py
```

Expected:

- `0 failed`

- [ ] **Step 3: Run Bandit on the touched AuthNZ Python files**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/AuthNZ/lockout_tracker.py \
  tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/AuthNZ/database.py \
  tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py
```

Expected:

- either clean output, or a captured environment limitation such as `No module named bandit`

- [ ] **Step 4: Update this plan file status and self-review the diff**

Run:

```bash
git diff -- \
  tldw_Server_API/app/core/AuthNZ/lockout_tracker.py \
  tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/AuthNZ/database.py \
  tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py \
  tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql \
  tldw_Server_API/tests/AuthNZ \
  tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py
```

Expected:

- only intended lockout/API-key/default/strictness changes are present

- [ ] **Step 5: Commit the completed remediation**

```bash
git add -- \
  tldw_Server_API/app/core/AuthNZ/lockout_tracker.py \
  tldw_Server_API/app/core/AuthNZ/repos/rate_limits_repo.py \
  tldw_Server_API/app/core/AuthNZ/migrations.py \
  tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py \
  tldw_Server_API/app/core/AuthNZ/database.py \
  tldw_Server_API/app/core/AuthNZ/repos/api_keys_repo.py \
  tldw_Server_API/Databases/SQLite/Schema/sqlite_users.sql \
  tldw_Server_API/tests/AuthNZ \
  tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py \
  Docs/superpowers/plans/2026-04-07-authnz-followup-remediation-plan.md
git commit -m "fix: harden authnz schema contracts"
```
