# DB_Management Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confirmed DB_Management findings in one fail-closed remediation branch, align direct callers with the corrected contracts, and add focused regression coverage for every corrected behavior.

**Architecture:** Land the work in four slices. First harden security- and schema-critical flows (`pg_rls_policies.py`, `db_migration.py`, `UserDatabase_v2.py`). Then fix shared backend lifecycle and trusted-path boundaries. After that, tighten `media_db.api` caller-visible error semantics and the shared backend FTS contract. Finish with targeted verification plus a Bandit pass on the touched scope.

**Tech Stack:** Python 3.14, FastAPI, SQLite/PostgreSQL backend abstraction, pytest, `unittest.mock`, Loguru, Bandit

---

## File Map

- Modify: `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`
  - Replace best-effort RLS application with a shared all-or-nothing installer helper that raises `DatabaseError` on PostgreSQL partial failure.
- Modify: `tldw_Server_API/app/main.py`
  - Move the PG RLS startup branch behind a small helper so startup logging matches the new exception contract.
- Create: `tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py`
  - Unit coverage for partial-statement failure, PostgreSQL no-op behavior, and startup helper logging.
- Modify: `tldw_Server_API/tests/DB/integration/test_pg_rls_apply.py`
  - Tighten the smoke expectation from “`True` or `False`” to the new explicit contract.

- Modify: `tldw_Server_API/app/core/DB_Management/db_migration.py`
  - Raise on malformed or duplicate migrations, validate contiguous upgrade and rollback ranges, and preflight rollback `down_sql` coverage.
- Modify: `tldw_Server_API/app/core/DB_Management/migrate_db.py`
  - Thread `create_backup` from CLI to `DatabaseMigrator.migrate_to_version(...)`.
- Modify: `tldw_Server_API/tests/DB_Management/test_db_migration_loader.py`
  - Update duplicate-version behavior and malformed-artifact expectations.
- Create: `tldw_Server_API/tests/DB_Management/test_db_migration_planning.py`
  - Focused upgrade-gap and rollback-gap tests.
- Modify: `tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py`
  - Prove `--no-backup` is wired through the CLI helper.

- Modify: `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py`
  - Make required schema normalization and required baseline RBAC seed verification fail closed with `UserDatabaseError`.
- Create: `tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py`
  - Focused unit coverage for required-column failures and missing required RBAC state.

- Modify: `tldw_Server_API/app/core/DB_Management/content_backend.py`
  - Centralize close-before-replace and close-before-clear cache semantics.
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py`
  - Stop nulling cache globals directly; call the hardened cache-clear helper.
- Modify: `tldw_Server_API/app/core/DB_Management/DB_Manager.py`
  - Keep reset/shutdown wiring aligned with the hardened backend lifecycle.
- Modify: `tldw_Server_API/tests/DB_Management/test_content_backend_cache.py`
  - Add superseded-pool closure assertions.

- Modify: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
  - Make trusted DB path resolution symlink-safe while still allowing new file creation.
- Modify: `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`
  - Replace relative-path-only checks with the shared trusted path helper.
- Modify: `tldw_Server_API/app/core/DB_Management/watchlist_alert_rules_db.py`
  - Validate helper `db_path` arguments through the shared trusted path helper.
- Modify: `tldw_Server_API/app/core/DB_Management/Voice_Registry_DB.py`
  - Replace custom containment logic with the shared trusted path helper.
- Modify: `tldw_Server_API/tests/DB_Management/test_db_path_utils.py`
  - Add symlink escape coverage.
- Create: `tldw_Server_API/tests/DB_Management/test_db_management_helper_path_contracts.py`
  - Prove the representative helper modules all call the shared trust helper.
- Modify: `tldw_Server_API/tests/test_watchlist_alert_rules.py`
  - Add a helper-level path-validation regression.

- Modify: `tldw_Server_API/app/core/DB_Management/media_db/api.py`
  - Raise `DatabaseError` instead of normalizing backend/query faults to benign values.
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py`
  - Catch `DatabaseError` explicitly where degraded retrieval fallback is intentional.
- Modify: `tldw_Server_API/app/core/StudyPacks/source_resolver.py`
  - Catch `DatabaseError` explicitly and fall back to transcript behavior.
- Modify: `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
  - Catch `DatabaseError` explicitly and continue to heuristic section fallback.
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/navigation.py`
  - Narrow section-lookup fallback to the typed DB error path.
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py`
  - Add error-propagation regressions for the package-level helpers.
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_media_retrieval.py`
  - Cover MCP fallback when chunk helpers raise `DatabaseError`.
- Modify: `tldw_Server_API/tests/StudyPacks/test_source_resolver.py`
  - Cover transcript fallback after chunk lookup `DatabaseError`.
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker.py`
  - Cover heuristic section fallback after typed DB lookup failure.
- Modify: `tldw_Server_API/tests/Media/test_media_navigation.py`
  - Cover typed section-lookup failures without falling through to a 500.

- Modify: `tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py`
  - Normalize shared `FTSQuery.query` at the backend boundary for SQLite.
- Modify: `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py`
  - Normalize shared `FTSQuery.query` at the backend boundary for PostgreSQL and fail on empty normalized tsquery.
- Modify: `tldw_Server_API/tests/DB_Management/test_database_backends.py`
  - Add backend-normalization regressions for `fts_search(...)`.

## Stage Overview

### Stage 1: Security And Schema Contracts

- Task 1: Harden PostgreSQL RLS installers and startup auto-ensure.
- Task 2: Harden migration loading, planning, rollback validation, and CLI backup wiring.
- Task 3: Make `UserDatabase_v2` bootstrap and baseline RBAC seeding fail closed.

### Stage 2: Lifecycle And Storage Boundaries

- Task 4: Close superseded shared PostgreSQL backend pools during cache reset and replacement.
- Task 5: Make trusted DB path validation symlink-safe and reuse it across representative helper modules.

### Stage 3: Read And Search Contracts

- Task 6: Make `media_db.api` surface typed DB errors and align direct callers with explicit fallback behavior.
- Task 7: Normalize `fts_search(...)` queries at the backend abstraction boundary.

### Stage 4: Verification

- Task 8: Run the focused pytest matrix and a Bandit scan on the touched scope.

### Task 1: Harden PostgreSQL RLS Installers And Startup Caller

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py`
- Modify: `tldw_Server_API/app/main.py`
- Create: `tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py`
- Modify: `tldw_Server_API/tests/DB/integration/test_pg_rls_apply.py`

- [ ] **Step 1: Write the failing tests**

```python
# tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py
import contextlib
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import ensure_prompt_studio_rls


class _FailingCursor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, _sql: str) -> None:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("boom")


class _TxnConn:
    def __init__(self) -> None:
        self.cursor_obj = _FailingCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cursor_obj

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class _Backend:
    backend_type = SimpleNamespace(name="POSTGRESQL")

    def __init__(self, conn: _TxnConn) -> None:
        self._conn = conn

    def transaction(self):
        @contextlib.contextmanager
        def _ctx():
            yield self._conn

        return _ctx()


def test_ensure_prompt_studio_rls_raises_on_partial_failure():
    conn = _TxnConn()

    with pytest.raises(DatabaseError, match="prompt_studio"):
        ensure_prompt_studio_rls(_Backend(conn))

    assert conn.committed is False
    assert conn.rolled_back is True


def test_run_pg_rls_auto_ensure_logs_success_only_after_both_installers_pass(monkeypatch, caplog):
    import tldw_Server_API.app.main as main_mod

    monkeypatch.setattr(main_mod, "ensure_prompt_studio_rls", lambda _backend: True, raising=False)
    monkeypatch.setattr(main_mod, "ensure_chacha_rls", lambda _backend: True, raising=False)

    main_mod._run_pg_rls_auto_ensure(object())

    assert "PG RLS ensure invoked" in caplog.text
```

```python
# tldw_Server_API/tests/DB/integration/test_pg_rls_apply.py
def test_apply_rls_policies_smoke(pg_database_config: DatabaseConfig):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    applied = ensure_prompt_studio_rls(backend)
    assert applied is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py \
  -q
```

Expected: FAIL because `ensure_prompt_studio_rls()` currently swallows statement errors, commits best-effort work, and `app.main` does not expose a small helper with the hardened logging contract.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py
def _ensure_rls_policy_set(
    backend: DatabaseBackend,
    *,
    name: str,
    statements: list[str],
) -> bool:
    try:
        if not hasattr(backend, "backend_type") or backend.backend_type.name != "POSTGRESQL":
            return False
    except Exception:
        return False

    with backend.transaction() as conn:
        # Let the transaction context manager handle commit/rollback.
        # Manual conn.commit() / conn.rollback() inside a context-managed
        # transaction block is risky and can cause double-commit or
        # mask the original exception during rollback.
        cur = conn.cursor()
        for index, statement in enumerate(statements, start=1):
            try:
                cur.execute(statement)
            except Exception as exc:
                raise DatabaseError(f"{name} RLS statement {index} failed: {exc}") from exc
        return True


def ensure_prompt_studio_rls(backend: DatabaseBackend) -> bool:
    return _ensure_rls_policy_set(
        backend,
        name="prompt_studio",
        statements=build_prompt_studio_rls_sql(),
    )


def ensure_chacha_rls(backend: DatabaseBackend) -> bool:
    return _ensure_rls_policy_set(
        backend,
        name="chacha",
        statements=build_chacha_rls_sql(),
    )
```

```python
# tldw_Server_API/app/main.py
def _run_pg_rls_auto_ensure(backend) -> tuple[bool, bool]:
    prompt_ok = ensure_prompt_studio_rls(backend)
    chacha_ok = ensure_chacha_rls(backend)
    logger.info(
        "PG RLS ensure invoked (prompt_studio_applied={}, chacha_applied={})",
        prompt_ok,
        chacha_ok,
    )
    return prompt_ok, chacha_ok


# inside startup branch
_run_pg_rls_auto_ensure(_backend)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py \
  tldw_Server_API/tests/DB/integration/test_pg_rls_apply.py \
  -q
```

Expected: PASS or `SKIPPED` only if the integration fixture is unavailable; the unit contract test must pass.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py \
  tldw_Server_API/app/main.py \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py \
  tldw_Server_API/tests/DB/integration/test_pg_rls_apply.py
git commit -m "fix: fail closed on postgres rls setup"
```

### Task 2: Harden Migration Loading, Planning, Rollback Validation, And CLI Wiring

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/db_migration.py`
- Modify: `tldw_Server_API/app/core/DB_Management/migrate_db.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_db_migration_loader.py`
- Create: `tldw_Server_API/tests/DB_Management/test_db_migration_planning.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py`

- [ ] **Step 1: Write the failing tests**

```python
# tldw_Server_API/tests/DB_Management/test_db_migration_loader.py
import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator, MigrationError


def test_load_migrations_raises_on_duplicate_versions(tmp_path: Path):
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    db_path = tmp_path / "app.db"
    db_path.touch()

    (migrations_dir / "001_first.json").write_text(json.dumps({"version": 1, "name": "first", "up_sql": "SELECT 1"}))
    (migrations_dir / "001_second.json").write_text(json.dumps({"version": 1, "name": "second", "up_sql": "SELECT 2"}))

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))

    with pytest.raises(MigrationError, match="Duplicate migration version 1"):
        migrator.load_migrations()
```

```python
# tldw_Server_API/tests/DB_Management/test_db_migration_planning.py
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator, MigrationError


def test_migrate_to_version_rejects_missing_intermediate_versions(tmp_path: Path):
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    db_path = tmp_path / "app.db"
    db_path.touch()

    (migrations_dir / "001_first.json").write_text(json.dumps({"version": 1, "name": "first", "up_sql": "SELECT 1"}))
    (migrations_dir / "003_third.json").write_text(json.dumps({"version": 3, "name": "third", "up_sql": "SELECT 3"}))

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))

    with pytest.raises(MigrationError, match="Missing migration versions: \\[2\\]"):
        migrator.migrate_to_version(3, create_backup=False)


def test_migrate_to_version_rejects_rollback_without_down_sql(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "unused"))

    monkeypatch.setattr(migrator, "get_current_version", lambda: 2)
    monkeypatch.setattr(
        migrator,
        "load_migrations",
        lambda: [
            SimpleNamespace(version=1, name="first", up_sql="SELECT 1", down_sql="SELECT 1", checksum="a", idempotent=False),
            SimpleNamespace(version=2, name="second", up_sql="SELECT 2", down_sql=None, checksum="b", idempotent=False),
        ],
    )

    with pytest.raises(MigrationError, match="down_sql"):
        migrator.migrate_to_version(0, create_backup=False)
```

```python
# tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py
from types import SimpleNamespace

from tldw_Server_API.app.core.DB_Management import migrate_db as migrate_db_cli


def test_migrate_threads_create_backup_flag(monkeypatch):
    calls = {}

    class _FakeMigrator:
        def __init__(self, db_path: str):
            calls["db_path"] = db_path

        def get_current_version(self):
            return 0

        def load_migrations(self):
            return [SimpleNamespace(version=5)]

        def migrate_to_version(self, target_version, create_backup=True):
            calls["target_version"] = target_version
            calls["create_backup"] = create_backup
            return {
                "status": "success",
                "previous_version": 0,
                "current_version": target_version,
                "target_version": target_version,
                "migrations_applied": [],
                "total_execution_time": 0.0,
                "backup_path": None,
            }

    monkeypatch.setattr(migrate_db_cli, "DatabaseMigrator", _FakeMigrator)

    migrate_db_cli.migrate("/tmp/app.db", target_version=5, create_backup=False)

    assert calls == {
        "db_path": "/tmp/app.db",
        "target_version": 5,
        "create_backup": False,
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_db_migration_loader.py \
  tldw_Server_API/tests/DB_Management/test_db_migration_planning.py \
  tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py \
  -q
```

Expected: FAIL because duplicate versions are currently ignored, missing intermediate versions are tolerated, rollback preflight does not validate contiguous `down_sql` coverage, and the CLI helper does not accept or thread `create_backup=False`.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/DB_Management/db_migration.py
def load_migrations(self) -> list[Migration]:
    migrations: list[Migration] = []
    seen_versions: dict[int, Path] = {}
    errors: list[str] = []

    for filepath in sorted(Path(self.migrations_dir).glob("*")):
        suffix = filepath.suffix.lower()
        try:
            if suffix == ".json":
                migration = Migration.from_file(filepath)
            elif suffix == ".sql":
                migration = self._load_sql_migration(filepath)
            else:
                continue
        except Exception as exc:
            errors.append(f"{filepath.name}: {exc}")
            continue

        if migration is None:
            continue
        if migration.version in seen_versions:
            raise MigrationError(
                f"Duplicate migration version {migration.version}: "
                f"{seen_versions[migration.version].name} and {filepath.name}"
            )
        migrations.append(migration)
        seen_versions[migration.version] = filepath

    if errors:
        raise MigrationError("Invalid migration set: " + "; ".join(errors))

    migrations.sort(key=lambda item: item.version)
    return migrations


def _validate_contiguous_versions(
    self,
    *,
    available_versions: list[int],
    start_version: int,
    end_version: int,
) -> None:
    missing = [
        version
        for version in range(start_version, end_version + 1)
        if version not in set(available_versions)
    ]
    if missing:
        raise MigrationError(f"Missing migration versions: {missing}")
```

```python
# tldw_Server_API/app/core/DB_Management/migrate_db.py
def migrate(
    db_path: str,
    target_version: Optional[int] = None,
    create_backup: bool = True,
):
    migrator = DatabaseMigrator(db_path)
    ...
    result = migrator.migrate_to_version(target, create_backup=create_backup)


# inside main()
elif args.command == "migrate":
    migrate(args.db_path, args.version, create_backup=not args.no_backup)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_db_migration_loader.py \
  tldw_Server_API/tests/DB_Management/test_db_migration_planning.py \
  tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py \
  -q
```

Expected: PASS with the loader raising on invalid migration sets, planning failing fast on version gaps, and the CLI helper passing `create_backup=False`.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/db_migration.py \
  tldw_Server_API/app/core/DB_Management/migrate_db.py \
  tldw_Server_API/tests/DB_Management/test_db_migration_loader.py \
  tldw_Server_API/tests/DB_Management/test_db_migration_planning.py \
  tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py
git commit -m "fix: harden db migration contracts"
```

### Task 3: Make `UserDatabase_v2` Bootstrap Fail Closed

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py`
- Create: `tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py`

- [ ] **Step 1: Write the failing tests**

```python
# tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.UserDatabase_v2 import UserDatabase, UserDatabaseError
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


class _Result:
    def __init__(self, rows):
        self.rows = rows


def test_ensure_core_columns_raises_when_required_column_add_fails():
    class _Backend:
        backend_type = BackendType.SQLITE

        def execute(self, sql, params=None):
            if sql == "PRAGMA table_info(users)":
                return _Result([{"name": "id"}])
            if sql.startswith("ALTER TABLE users ADD COLUMN uuid"):
                raise RuntimeError("no alter")
            return _Result([])

    db = UserDatabase.__new__(UserDatabase)
    db.backend = _Backend()

    with pytest.raises(UserDatabaseError, match="uuid"):
        db._ensure_core_columns()


def test_seed_default_data_raises_when_required_role_state_missing():
    class _Backend:
        backend_type = BackendType.SQLITE

        def execute(self, sql, params=None):
            if sql.startswith("SELECT id FROM roles"):
                return _Result([])
            if sql.startswith("SELECT id FROM permissions"):
                return _Result([])
            return _Result([])

    db = UserDatabase.__new__(UserDatabase)
    db.backend = _Backend()

    with pytest.raises(UserDatabaseError, match="admin"):
        db._seed_default_data()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py \
  -q
```

Expected: FAIL because `_ensure_core_columns()` currently logs and continues after required schema errors, and `_seed_default_data()` currently tolerates missing required baseline roles and permissions.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py
def _ensure_core_columns(self) -> None:
    try:
        ...
    except Exception as exc:
        raise UserDatabaseError(f"Required user schema normalization failed: {exc}") from exc

    try:
        ...
    except Exception as exc:
        raise UserDatabaseError(f"Required registration_codes normalization failed: {exc}") from exc


def _seed_default_data(self) -> None:
    required_roles = {"admin", "user", "viewer"}
    required_permissions = {
        "media.read",
        "media.create",
        "media.delete",
        "sql.read",
        "sql.target:media_db",
        "system.configure",
        "users.manage_roles",
    }
    ...
    missing_roles = [name for name in required_roles if _get_id(sel_role_id, name) is None]
    missing_permissions = [name for name in required_permissions if _get_id(sel_perm_id, name) is None]
    if missing_roles or missing_permissions:
        raise UserDatabaseError(
            f"Required RBAC seed state missing: roles={missing_roles}, permissions={missing_permissions}"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py \
  tldw_Server_API/tests/test_authnz_backends_improved.py \
  -q
```

Expected: PASS, or a small number of follow-up failures in `test_authnz_backends_improved.py` that reveal real bootstrap assumptions to fix in the same task before committing.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py \
  tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py \
  tldw_Server_API/tests/test_authnz_backends_improved.py
git commit -m "fix: fail closed on user database bootstrap"
```

### Task 4: Close Superseded Shared PostgreSQL Backend Pools

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/content_backend.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py`
- Modify: `tldw_Server_API/app/core/DB_Management/DB_Manager.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_content_backend_cache.py`

- [ ] **Step 1: Write the failing tests**

```python
# tldw_Server_API/tests/DB_Management/test_content_backend_cache.py
class _FakePool:
    def __init__(self) -> None:
        self.closed = 0

    def close_all(self) -> None:
        self.closed += 1


class _FakeBackend:
    def __init__(self) -> None:
        self.pool = _FakePool()

    def get_pool(self):
        return self.pool


def test_get_content_backend_closes_superseded_cached_backend(monkeypatch) -> None:
    old_backend = _FakeBackend()
    new_backend = _FakeBackend()

    monkeypatch.setattr(content_backend, "_cached_backend", old_backend)
    monkeypatch.setattr(content_backend, "_cached_backend_signature", ("old",))
    monkeypatch.setattr(
        content_backend.DatabaseBackendFactory,
        "create_backend",
        staticmethod(lambda _cfg: new_backend),
    )

    cfg = _make_config(password="pw-new", sslmode="prefer")
    backend = content_backend.get_content_backend(cfg)

    assert backend is new_backend
    assert old_backend.pool.closed == 1


def test_reset_media_runtime_defaults_closes_cached_backend(monkeypatch) -> None:
    cached_backend = _FakeBackend()
    monkeypatch.setattr(content_backend, "_cached_backend", cached_backend)
    monkeypatch.setattr(content_backend, "_cached_backend_signature", ("cached",))

    media_runtime_defaults._clear_content_backend_cache()

    assert cached_backend.pool.closed == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_content_backend_cache.py \
  -q
```

Expected: FAIL because cache replacement and cache clearing currently overwrite the cached backend reference without closing the displaced pool.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/DB_Management/content_backend.py
def _close_cached_backend(backend) -> None:
    if backend is None:
        return
    try:
        backend.get_pool().close_all()
    except Exception as exc:
        logger.warning("Failed to close superseded content backend pool: {}", exc)


def clear_cached_backend() -> None:
    global _cached_backend, _cached_backend_signature
    old_backend = _cached_backend
    _cached_backend = None
    _cached_backend_signature = None
    _close_cached_backend(old_backend)


def get_content_backend(config: ConfigParser):
    ...
    with _cache_lock:
        if _cached_backend and _cached_backend_signature == signature:
            return _cached_backend
        old_backend = _cached_backend
        backend = DatabaseBackendFactory.create_backend(settings.database_config)
        _cached_backend = backend
        _cached_backend_signature = signature
        if old_backend is not None and old_backend is not backend:
            _close_cached_backend(old_backend)
        return backend
```

```python
# tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py
def _clear_content_backend_cache() -> None:
    import tldw_Server_API.app.core.DB_Management.content_backend as cb

    if hasattr(cb, "_cache_lock") and cb._cache_lock:
        with cb._cache_lock:
            cb.clear_cached_backend()
    else:
        cb.clear_cached_backend()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_content_backend_cache.py \
  -q
```

Expected: PASS with direct evidence that both replacement and cache clearing close the displaced pool once.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/content_backend.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py \
  tldw_Server_API/app/core/DB_Management/DB_Manager.py \
  tldw_Server_API/tests/DB_Management/test_content_backend_cache.py
git commit -m "fix: close superseded content backend pools"
```

### Task 5: Make Trusted DB Paths Symlink-Safe And Reuse Them Across Helper Modules

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- Modify: `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/watchlist_alert_rules_db.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Voice_Registry_DB.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_db_path_utils.py`
- Create: `tldw_Server_API/tests/DB_Management/test_db_management_helper_path_contracts.py`
- Modify: `tldw_Server_API/tests/test_watchlist_alert_rules.py`

- [ ] **Step 1: Write the failing tests**

```python
# tldw_Server_API/tests/DB_Management/test_db_path_utils.py
def test_resolve_trusted_database_path_rejects_symlink_escape(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    trusted_dir = project_root / "Databases"
    outside_dir = tmp_path / "outside"
    project_root.mkdir()
    trusted_dir.mkdir()
    outside_dir.mkdir()

    (trusted_dir / "escape").symlink_to(outside_dir, target_is_directory=True)

    monkeypatch.setattr(db_path_utils, "get_project_root", lambda: str(project_root))
    monkeypatch.setattr(db_path_utils, "_is_test_context", lambda: False)

    with pytest.raises(InvalidStoragePathError):
        db_path_utils.resolve_trusted_database_path(trusted_dir / "escape" / "users.db")
```

```python
# tldw_Server_API/tests/DB_Management/test_db_management_helper_path_contracts.py
def test_topic_monitoring_db_uses_shared_trusted_path(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management import TopicMonitoring_DB as topic_mod

    calls = []
    safe_path = tmp_path / "safe-topic.db"
    monkeypatch.setattr(
        topic_mod,
        "resolve_trusted_database_path",
        lambda db_path, **kwargs: calls.append((db_path, kwargs["label"])) or safe_path,
    )

    db = topic_mod.TopicMonitoringDB(str(tmp_path / "input-topic.db"))

    assert db.db_path == str(safe_path)
    assert calls == [(str(tmp_path / "input-topic.db"), "topic monitoring db")]


def test_watchlist_alert_rules_helpers_use_shared_trusted_path(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management import watchlist_alert_rules_db as alert_rules_db

    calls = []
    safe_path = tmp_path / "safe-alerts.db"
    monkeypatch.setattr(
        alert_rules_db,
        "resolve_trusted_database_path",
        lambda db_path, **kwargs: calls.append((db_path, kwargs["label"])) or safe_path,
    )

    alert_rules_db.ensure_watchlist_alert_rules_table(str(tmp_path / "input-alerts.db"))

    assert calls == [(str(tmp_path / "input-alerts.db"), "watchlist alert rules db")]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_db_path_utils.py \
  tldw_Server_API/tests/DB_Management/test_db_management_helper_path_contracts.py \
  tldw_Server_API/tests/test_watchlist_alert_rules.py \
  -q
```

Expected: FAIL because `resolve_trusted_database_path()` currently uses lexical containment and the representative helper modules do not all call the shared trust helper.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/DB_Management/db_path_utils.py
def _resolve_candidate_for_containment(candidate: Path) -> Path:
    # Resolve the entire path, not just the parent, so that symlinks
    # and '..' segments anywhere in the path are fully resolved.
    return candidate.resolve(strict=False)


def resolve_trusted_database_path(...):
    ...
    if candidate.is_absolute():
        normalized = _resolve_candidate_for_containment(candidate.expanduser())
    else:
        normalized = _resolve_candidate_for_containment(project_root / candidate.expanduser())

    for root in unique_roots:
        resolved_root = root.resolve(strict=False)
        try:
            normalized.relative_to(resolved_root)
            return normalized
        except ValueError:
            continue
    raise InvalidStoragePathError("invalid_path")
```

```python
# tldw_Server_API/app/core/DB_Management/watchlist_alert_rules_db.py
from pathlib import Path

from tldw_Server_API.app.core.DB_Management.db_path_utils import resolve_trusted_database_path


def _trusted_db_path(db_path: str | Path) -> str:
    return str(resolve_trusted_database_path(db_path, label="watchlist alert rules db"))


def ensure_watchlist_alert_rules_table(db_path: str) -> None:
    with sqlite3.connect(_trusted_db_path(db_path)) as conn:
        conn.executescript(ALERT_RULES_TABLE_SQL)
```

```python
# tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py
from tldw_Server_API.app.core.DB_Management.db_path_utils import resolve_trusted_database_path


class TopicMonitoringDB:
    def __init__(self, db_path: str = "Databases/monitoring_alerts.db") -> None:
        safe_path = resolve_trusted_database_path(db_path, label="topic monitoring db")
        self.db_path = str(safe_path)
        ...
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_db_path_utils.py \
  tldw_Server_API/tests/DB_Management/test_db_management_helper_path_contracts.py \
  tldw_Server_API/tests/test_watchlist_alert_rules.py \
  -q
```

Expected: PASS with symlink escapes rejected and all representative helper modules routing through the shared trusted-path helper.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/db_path_utils.py \
  tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py \
  tldw_Server_API/app/core/DB_Management/watchlist_alert_rules_db.py \
  tldw_Server_API/app/core/DB_Management/Voice_Registry_DB.py \
  tldw_Server_API/tests/DB_Management/test_db_path_utils.py \
  tldw_Server_API/tests/DB_Management/test_db_management_helper_path_contracts.py \
  tldw_Server_API/tests/test_watchlist_alert_rules.py
git commit -m "fix: harden trusted db path handling"
```

### Task 6: Surface Typed `media_db.api` Errors And Align Direct Callers

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/api.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py`
- Modify: `tldw_Server_API/app/core/StudyPacks/source_resolver.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/media/navigation.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_media_retrieval.py`
- Modify: `tldw_Server_API/tests/StudyPacks/test_source_resolver.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker.py`
- Modify: `tldw_Server_API/tests/Media/test_media_navigation.py`

- [ ] **Step 1: Write the failing tests**

```python
# tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError


def test_has_unvectorized_chunks_raises_database_error_on_query_failure():
    class StubDb:
        def execute_query(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    with pytest.raises(DatabaseError, match="has_unvectorized_chunks"):
        media_db_api.has_unvectorized_chunks(StubDb(), 9)
```

```python
# tldw_Server_API/tests/StudyPacks/test_source_resolver.py
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError


def test_media_source_falls_back_to_transcript_after_chunk_lookup_database_error(monkeypatch: pytest.MonkeyPatch):
    _, resolver_mod = _load_study_modules()

    monkeypatch.setattr(resolver_mod, "get_media_by_id", lambda *_args, **_kwargs: {"id": 9, "title": "Fallback Doc"})
    monkeypatch.setattr(
        resolver_mod,
        "get_unvectorized_chunks_in_range",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(DatabaseError("chunk lookup failed")),
    )
    monkeypatch.setattr(
        resolver_mod,
        "get_latest_transcription",
        lambda *_args, **_kwargs: "Transcript fallback still works.",
    )

    bundle = _resolver(media_db=SimpleNamespace()).resolve(
        [_selection(source_type="media", source_id="9", locator={"chunk_index": 2})]
    )

    assert "Transcript fallback" in bundle.items[0].evidence_text
```

```python
# tldw_Server_API/app/core/MCP_unified/tests/test_media_retrieval.py
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError


class BrokenMediaDB(FakeMediaDB):
    def has_unvectorized_chunks(self, media_id: int) -> bool:
        raise DatabaseError("prechunked lookup failed")


@pytest.mark.asyncio
async def test_media_get_chunk_with_siblings_falls_back_after_database_error():
    mod = MediaModule(ModuleConfig(name="media"))
    mod._open_media_db = lambda ctx: BrokenMediaDB()  # type: ignore[attr-defined]
    context = SimpleNamespace(user_id="1", metadata={})

    out = await mod.execute_tool(
        "media.get",
        {
            "media_id": 42,
            "retrieval": {"mode": "chunk_with_siblings", "max_tokens": 25, "chars_per_token": 1, "loc": {"approx_offset": 12}},
        },
        context=context,
    )

    assert isinstance(out, dict)
    assert out["content"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_media_retrieval.py \
  tldw_Server_API/tests/StudyPacks/test_source_resolver.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker.py \
  tldw_Server_API/tests/Media/test_media_navigation.py \
  -q
```

Expected: FAIL because the package-level helpers currently normalize backend/query failures to benign values, and the direct callers have not been converted to explicit typed-error fallback or explicit propagation.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/DB_Management/media_db/api.py
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError


def _raise_read_error(operation: str, exc: Exception) -> None:
    if isinstance(exc, DatabaseError):
        raise
    raise DatabaseError(f"{operation} failed: {exc}") from exc


def has_unvectorized_chunks(db: MediaDbLike | MediaDbReadLike, media_id: int) -> bool:
    ...
    if is_media_database_like(db_instance):
        try:
            cursor = db_instance.execute_query(...)
            return cursor.fetchone() is not None
        except Exception as exc:
            _raise_read_error("has_unvectorized_chunks", exc)
```

```python
# tldw_Server_API/app/core/StudyPacks/source_resolver.py
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

...
try:
    chunks = get_unvectorized_chunks_in_range(self.media_db, media_id, chunk_index, chunk_index)
except DatabaseError as exc:
    logger.warning("Chunk lookup failed for media {}: {}", media_id, exc)
    chunks = []
```

```python
# tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

...
except DatabaseError:
    pass
```

```python
# tldw_Server_API/app/api/v1/endpoints/media/navigation.py
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

...
except DatabaseError as exc:
    logger.debug("Section heading lookup failed for media {}: {}", media_id, exc)
    lookup = None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_media_retrieval.py \
  tldw_Server_API/tests/StudyPacks/test_source_resolver.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker.py \
  tldw_Server_API/tests/Media/test_media_navigation.py \
  -q
```

Expected: PASS with the API layer raising typed DB errors and the reviewed direct callers making an explicit fallback decision.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/media_db/api.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py \
  tldw_Server_API/app/core/StudyPacks/source_resolver.py \
  tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py \
  tldw_Server_API/app/api/v1/endpoints/media/navigation.py \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_media_retrieval.py \
  tldw_Server_API/tests/StudyPacks/test_source_resolver.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker.py \
  tldw_Server_API/tests/Media/test_media_navigation.py
git commit -m "fix: surface media db read failures explicitly"
```

### Task 7: Normalize `fts_search(...)` Queries At The Backend Boundary

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py`
- Modify: `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py`
- Modify: `tldw_Server_API/tests/DB_Management/test_database_backends.py`

- [ ] **Step 1: Write the failing tests**

```python
# tldw_Server_API/tests/DB_Management/test_database_backends.py
from tldw_Server_API.app.core.DB_Management.backends.base import QueryResult


def test_sqlite_backend_fts_search_normalizes_query(monkeypatch, sqlite_config):
    backend = SQLiteBackend(sqlite_config)
    backend.create_tables("CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY, title TEXT, content TEXT)")
    backend.create_fts_table("docs_fts", "docs", ["title", "content"])

    calls = []
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.backends.sqlite_backend.FTSQueryTranslator.normalize_query",
        lambda query, backend_name: calls.append((query, backend_name)) or "hello OR world",
    )
    monkeypatch.setattr(backend, "execute", lambda sql, params=(), connection=None: QueryResult(rows=[]))

    backend.fts_search(FTSQuery(query="hello world", table="docs_fts"))

    assert calls == [("hello world", "sqlite")]


def test_postgresql_backend_fts_search_raises_when_normalized_query_is_empty(monkeypatch, pg_config):
    from tldw_Server_API.app.core.DB_Management.backends.postgresql_backend import PostgreSQLBackend

    backend = PostgreSQLBackend(pg_config)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.backends.postgresql_backend.FTSQueryTranslator.normalize_query",
        lambda query, backend_name: "",
    )

    with pytest.raises(DatabaseError, match="normalized to empty"):
        backend.fts_search(FTSQuery(query="!!!", table="docs_fts"))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_database_backends.py \
  -k "fts_search_normalizes_query or normalized_query_is_empty" \
  -q
```

Expected: FAIL because `fts_search(...)` currently passes `FTSQuery.query_text` straight through to the backend SQL without shared-boundary normalization.

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py
normalized_query = FTSQueryTranslator.normalize_query(fts_query.query_text, "sqlite") or fts_query.query_text
params.append(normalized_query)
```

```python
# tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py
normalized_query = FTSQueryTranslator.normalize_query(fts_query.query_text, "postgresql")
if not normalized_query:
    raise DatabaseError("FTS query normalized to empty for postgresql")

query_parts = [
    f"SELECT *, ts_rank({self.escape_identifier(fts_column)}, query) AS rank",
    f"FROM {self.escape_identifier(source_table)},",
    "to_tsquery('english', %s) query",
    f"WHERE {self.escape_identifier(fts_column)} @@ query",
]
params = [normalized_query]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_database_backends.py \
  -k "fts_search_normalizes_query or normalized_query_is_empty" \
  -q
```

Expected: PASS with translator invocation locked in at the shared backend boundary and PostgreSQL failing fast on empty normalized tsquery input.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py \
  tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py \
  tldw_Server_API/tests/DB_Management/test_database_backends.py
git commit -m "fix: normalize shared backend fts queries"
```

### Task 8: Run Final Verification And Security Scan

**Files:**
- No new source files expected in this task.
- If any verification command fails, fix the concrete defect in the task that introduced it before re-running this verification task.

- [ ] **Step 1: Run the focused regression suite**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_pg_rls_policies_contract.py \
  tldw_Server_API/tests/DB/integration/test_pg_rls_apply.py \
  tldw_Server_API/tests/DB_Management/test_db_migration_loader.py \
  tldw_Server_API/tests/DB_Management/test_db_migration_planning.py \
  tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py \
  tldw_Server_API/tests/DB_Management/test_userdatabase_v2_bootstrap_failclosed.py \
  tldw_Server_API/tests/DB_Management/test_content_backend_cache.py \
  tldw_Server_API/tests/DB_Management/test_db_path_utils.py \
  tldw_Server_API/tests/DB_Management/test_db_management_helper_path_contracts.py \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  tldw_Server_API/tests/DB_Management/test_database_backends.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_media_retrieval.py \
  tldw_Server_API/tests/StudyPacks/test_source_resolver.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker.py \
  tldw_Server_API/tests/Media/test_media_navigation.py \
  tldw_Server_API/tests/test_watchlist_alert_rules.py \
  -q
```

Expected: PASS, with only environment-dependent integration tests allowed to skip.

- [ ] **Step 2: Run the wider DB_Management follow-up slices from the audit**

Run:
```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py \
  tldw_Server_API/tests/DB_Management/test_media_db_bootstrap_lifecycle_ops.py \
  tldw_Server_API/tests/DB_Management/test_media_db_connection_cleanup.py \
  tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py \
  tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py \
  tldw_Server_API/tests/DB_Management/test_media_db_scope_resolution_ops.py \
  tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py \
  tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py \
  tldw_Server_API/tests/DB_Management/test_media_db_v2_regressions.py \
  tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py \
  tldw_Server_API/tests/DB_Management/test_media_db_postgres_rls_ops.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py \
  tldw_Server_API/tests/DB_Management/test_media_postgres_support.py \
  -q
```

Expected: PASS or explicit `SKIPPED` where PostgreSQL-only fixtures are unavailable.

- [ ] **Step 3: Run Bandit on the touched scope**

Run:
```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/DB_Management \
  tldw_Server_API/app/core/StudyPacks \
  tldw_Server_API/app/core/RAG/rag_service \
  tldw_Server_API/app/core/MCP_unified/modules/implementations \
  tldw_Server_API/app/api/v1/endpoints/media \
  tldw_Server_API/app/main.py \
  -f json -o /tmp/bandit_db_management_remediation.json
```

Expected: `bandit` exits `0` or only reports pre-existing accepted findings outside the touched diff. Any new finding in the touched code must be fixed before completion.

- [ ] **Step 4: Inspect the final diff and status**

Run:
```bash
git status --short
git diff --stat
```

Expected: only the intended remediation files are modified, with no accidental unrelated edits or missing test files.
