import builtins
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.UserDatabase_v2 import (
    UserDatabase,
    UserDatabaseError,
)


class _Result:
    def __init__(self, rows):
        self.rows = rows


def test_initialize_schema_raises_when_required_schema_apply_fails(monkeypatch):
    secret = "postgres://schema-user:secret@example.invalid/authnz"
    db = UserDatabase.__new__(UserDatabase)
    db.backend = SimpleNamespace(backend_type=BackendType.SQLITE)

    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr(db, "_default_schema_statements", lambda: ["CREATE TABLE users (id INTEGER)"])
    monkeypatch.setattr(
        db,
        "_apply_schema_statements",
        lambda _statements: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    monkeypatch.setattr(
        db,
        "_ensure_core_columns",
        lambda: pytest.fail("_ensure_core_columns should not run after schema failure"),
    )
    monkeypatch.setattr(
        db,
        "_seed_default_data",
        lambda: pytest.fail("_seed_default_data should not run after schema failure"),
    )

    with pytest.raises(UserDatabaseError, match="schema initialization") as raised:
        db._initialize_schema()

    assert raised.value.__cause__ is None
    assert secret not in str(raised.value)


def test_initialize_schema_sanitizes_schema_file_read_failure(monkeypatch) -> None:
    secret = "schema path=/private/auth/users.sql password=secret"
    db = UserDatabase.__new__(UserDatabase)
    db.backend = SimpleNamespace(backend_type=BackendType.SQLITE)

    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(secret)),
    )
    monkeypatch.setattr(db, "_default_schema_statements", lambda: ["CREATE TABLE"])
    monkeypatch.setattr(db, "_apply_schema_statements", lambda _statements: None)
    monkeypatch.setattr(db, "_ensure_sqlite_profile_version_schema", lambda: None)
    monkeypatch.setattr(db, "_ensure_core_columns", lambda: None)
    monkeypatch.setattr(db, "_seed_default_data", lambda: None)
    output = io.StringIO()
    sink = logger.add(output, format="{message} {extra}")
    try:
        db._initialize_schema()
    finally:
        logger.remove(sink)

    assert secret not in output.getvalue()
    assert "/private/auth/users.sql" not in output.getvalue()


def test_initialize_schema_raises_for_unsupported_backend_type():
    db = UserDatabase.__new__(UserDatabase)
    db.backend = SimpleNamespace(backend_type=SimpleNamespace(value="unsupported"))

    with pytest.raises(UserDatabaseError, match="Unsupported backend type"):
        db._initialize_schema()


def test_postgres_initialize_runs_profile_readiness_before_core_normalization(
    monkeypatch,
):
    events = []
    connection = object()

    class _Transaction:
        def __enter__(self):
            events.append("transaction.enter")
            return connection

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            events.append("transaction.exit")
            return False

    backend = SimpleNamespace(
        backend_type=BackendType.POSTGRESQL,
        transaction=lambda: _Transaction(),
    )
    db = UserDatabase.__new__(UserDatabase)
    db.backend = backend

    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr(db, "_default_schema_statements", lambda: ["CREATE TABLE"])
    monkeypatch.setattr(
        db,
        "_apply_schema_statements",
        lambda statements: events.append(("schema", statements)),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.UserDatabase_v2.ensure_postgres_profile_version_sync",
        lambda executor, *, connection: events.append(
            ("readiness", executor, connection)
        ),
    )
    monkeypatch.setattr(db, "_ensure_core_columns", lambda: events.append("core"))
    monkeypatch.setattr(db, "_seed_default_data", lambda: events.append("seed"))

    db._initialize_schema()

    assert events == [
        ("schema", ["CREATE TABLE"]),
        "transaction.enter",
        ("readiness", backend, connection),
        "transaction.exit",
        "core",
        "seed",
    ]


def test_postgres_embedded_schema_uses_canonical_users_relation():
    statements = UserDatabase._default_schema_statements_postgres()
    users_statements = [statement for statement in statements if "users" in statement]
    candidate_tables = (
        "org_members",
        "team_members",
        "user_config_overrides",
        "org_config_overrides",
        "team_config_overrides",
    )

    assert any("CREATE TABLE IF NOT EXISTS public.users" in sql for sql in statements)
    assert all(
        any(f"CREATE TABLE IF NOT EXISTS public.{table}" in sql for sql in statements)
        for table in candidate_tables
    )
    assert all("REFERENCES users(" not in sql for sql in users_statements)
    assert all(" ON users" not in sql for sql in users_statements)


def test_postgres_candidate_remediation_uses_public_schema() -> None:
    statements = UserDatabase._profile_candidate_table_statements_postgres()
    ddl = "\n".join(statements)

    assert all("CREATE TABLE IF NOT EXISTS public." in sql for sql in statements)
    assert "public.organizations" in ddl
    assert "public.teams" in ddl
    assert "PRIMARY KEY (org_id, user_id)" in ddl
    assert "PRIMARY KEY (team_id, user_id)" in ddl
    assert "PRIMARY KEY (user_id, key)" in ddl
    assert "REFERENCES public.users(id)" in ddl
    assert "REFERENCES public.organizations(id)" in ddl
    assert "REFERENCES public.teams(id)" in ddl


def test_initialize_schema_real_sqlite_bootstrap_seeds_required_state(tmp_path):
    db_path = tmp_path / "users.db"
    db = UserDatabase(
        config=DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(db_path),
        ),
        client_id="test_suite",
    )

    tables = {
        row["name"]
        for row in db.backend.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).rows
    }
    expected_tables = {
        "users",
        "org_members",
        "team_members",
        "user_config_overrides",
        "org_config_overrides",
        "team_config_overrides",
        "roles",
        "permissions",
        "role_permissions",
        "registration_codes",
        "auth_audit_log",
    }
    missing_tables = expected_tables - tables
    if missing_tables:
        pytest.fail(f"expected SQLite bootstrap to create tables: {sorted(missing_tables)}")

    user_columns = {
        row["name"] if isinstance(row, dict) else row[1]
        for row in db.backend.execute("PRAGMA table_info(users)").rows
    }
    expected_columns = {
        "uuid",
        "metadata",
        "failed_login_attempts",
        "locked_until",
        "is_superuser",
        "profile_version",
    }
    missing_columns = expected_columns - user_columns
    if missing_columns:
        pytest.fail(f"expected SQLite bootstrap to create columns: {sorted(missing_columns)}")

    role_names = {
        row["name"]
        for row in db.backend.execute("SELECT name FROM roles").rows
    }
    expected_roles = {"admin", "user", "viewer"}
    missing_roles = expected_roles - role_names
    if missing_roles:
        pytest.fail(f"expected SQLite bootstrap to seed roles: {sorted(missing_roles)}")

    permission_names = {
        row["name"]
        for row in db.backend.execute("SELECT name FROM permissions").rows
    }
    expected_permissions = {
        "media.read",
        "media.create",
        "media.delete",
        "sql.read",
        "sql.target:media_db",
        "system.configure",
        "users.manage_roles",
    }
    missing_permissions = expected_permissions - permission_names
    if missing_permissions:
        pytest.fail(
            f"expected SQLite bootstrap to seed permissions: {sorted(missing_permissions)}"
        )

    admin_role = db.backend.execute(
        "SELECT id FROM roles WHERE name = ?",
        ("admin",),
    ).rows[0]["id"]
    manage_roles_permission = db.backend.execute(
        "SELECT id FROM permissions WHERE name = ?",
        ("users.manage_roles",),
    ).rows[0]["id"]
    admin_link = db.backend.execute(
        "SELECT 1 FROM role_permissions WHERE role_id = ? AND permission_id = ?",
        (admin_role, manage_roles_permission),
    ).rows
    if not admin_link:
        pytest.fail("expected SQLite bootstrap to seed admin users.manage_roles mapping")


def test_metadata_only_update_preserves_success_without_advancing_profile_version(
    tmp_path,
):
    db = UserDatabase(
        config=DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "users.db"),
        ),
        client_id="test_suite",
    )
    user_id = db.create_user(
        username="metadata-user",
        email="metadata@example.com",
        password_hash="not-a-real-hash",
    )
    before = db.backend.execute(
        "SELECT profile_version FROM users WHERE id = ?",
        (user_id,),
    ).rows[0]["profile_version"]

    assert db.update_user(user_id, metadata={"source": "test"}) is True

    after = db.backend.execute(
        "SELECT profile_version FROM users WHERE id = ?",
        (user_id,),
    ).rows[0]["profile_version"]
    assert after == before


def test_visible_update_of_missing_user_preserves_false_result(tmp_path):
    db = UserDatabase(
        config=DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "users.db"),
        ),
        client_id="test_suite",
    )

    assert db.update_user(999, email="missing@example.com") is False


def test_ensure_core_columns_handles_real_legacy_sqlite_uuid_migration(tmp_path):
    db_path = tmp_path / "users.db"
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(db_path),
        )
    )
    backend.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            profile_version TEXT NOT NULL DEFAULT (
                STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')
            )
        )
        """
    )
    backend.execute(
        "INSERT INTO users (username) VALUES (?)",
        ("legacy-user",),
    )
    backend.execute("CREATE TABLE roles (id INTEGER PRIMARY KEY)")
    backend.execute("CREATE TABLE registration_codes (id INTEGER PRIMARY KEY)")

    db = UserDatabase.__new__(UserDatabase)
    db.backend = backend

    db._ensure_core_columns()

    user_columns = {
        row["name"] if isinstance(row, dict) else row[1]
        for row in backend.execute("PRAGMA table_info(users)").rows
    }
    if "uuid" not in user_columns:
        pytest.fail("expected users.uuid column to be added for legacy SQLite bootstrap")

    candidate_tables = {
        row["name"]
        for row in backend.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).rows
    }
    expected_candidate_tables = {
        "organizations",
        "teams",
        "org_members",
        "team_members",
        "user_config_overrides",
        "org_config_overrides",
        "team_config_overrides",
    }
    assert expected_candidate_tables <= candidate_tables

    org_member_columns = {
        row["name"]
        for row in backend.execute("PRAGMA table_info(org_members)").rows
    }
    assert {"org_id", "user_id", "role", "status", "added_at"} <= org_member_columns
    org_member_pk = [
        row["name"]
        for row in sorted(
            backend.execute("PRAGMA table_info(org_members)").rows,
            key=lambda row: row["pk"],
        )
        if row["pk"]
    ]
    assert org_member_pk == ["org_id", "user_id"]
    org_member_fk_targets = {
        row["table"]
        for row in backend.execute("PRAGMA foreign_key_list(org_members)").rows
    }
    assert org_member_fk_targets == {"organizations", "users"}

    uuid_value = backend.execute("SELECT uuid FROM users WHERE username = ?", ("legacy-user",)).rows[0]["uuid"]
    if not uuid_value:
        pytest.fail("expected legacy SQLite user rows to receive a UUID backfill")


def test_ensure_core_columns_rejects_reduced_existing_candidate_schema(tmp_path):
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "users.db"),
        )
    )
    backend.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            profile_version TEXT NOT NULL DEFAULT (
                STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')
            )
        )
        """
    )
    backend.execute(
        "CREATE TABLE org_members "
        "(user_id INTEGER NOT NULL, org_id INTEGER NOT NULL, status TEXT)"
    )
    backend.execute("CREATE TABLE roles (id INTEGER PRIMARY KEY)")
    backend.execute("CREATE TABLE registration_codes (id INTEGER PRIMARY KEY)")
    db = UserDatabase.__new__(UserDatabase)
    db.backend = backend

    with pytest.raises(
        UserDatabaseError,
        match="profile candidate source tables",
    ):
        db._ensure_core_columns()


@pytest.mark.parametrize(
    ("table_name", "original", "replacement"),
    [
        ("teams", "org_id INTEGER NOT NULL", "org_id TEXT NOT NULL"),
        ("org_members", "role TEXT DEFAULT 'member'", "role TEXT DEFAULT 'owner'"),
        ("organizations", "name TEXT UNIQUE NOT NULL", "name TEXT NOT NULL"),
        (
            "team_members",
            "REFERENCES teams(id) ON DELETE CASCADE",
            "REFERENCES teams(id) ON DELETE SET NULL",
        ),
        (
            "user_config_overrides",
            "updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        ),
    ],
)
def test_candidate_readiness_rejects_noncanonical_sqlite_metadata(
    tmp_path,
    table_name: str,
    original: str,
    replacement: str,
) -> None:
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "candidate-metadata.db"),
        )
    )
    backend.execute("CREATE TABLE users (id INTEGER PRIMARY KEY)")
    for statement in UserDatabase._profile_candidate_table_statements_sqlite():
        if f"CREATE TABLE IF NOT EXISTS {table_name} " in statement:
            statement = statement.replace(original, replacement, 1)
        backend.execute(statement)

    db = UserDatabase.__new__(UserDatabase)
    db.backend = backend

    with pytest.raises(
        UserDatabaseError,
        match="profile candidate schema validation failed",
    ):
        db._validate_profile_candidate_tables_sqlite()


def test_postgres_candidate_readiness_inspects_complete_schema_contract() -> None:
    class _Backend:
        backend_type = BackendType.POSTGRESQL

        def __init__(self) -> None:
            self.queries: list[str] = []

        def execute(self, sql, params=None):
            self.queries.append(sql)
            return _Result([])

    backend = _Backend()
    db = UserDatabase.__new__(UserDatabase)
    db.backend = backend

    with pytest.raises(
        UserDatabaseError,
        match="profile candidate schema validation failed",
    ):
        db._validate_profile_candidate_tables_postgres()

    inspected_sql = "\n".join(backend.queries).lower()
    assert "data_type" in inspected_sql
    assert "is_nullable" in inspected_sql
    assert "column_default" in inspected_sql
    assert "constraint_type = 'unique'" in inspected_sql
    assert "foreign_column_name" in inspected_sql
    assert "foreign_table_schema" in inspected_sql
    assert "delete_rule" in inspected_sql
    assert "is_identity" in inspected_sql
    assert "identity_generation" in inspected_sql


def test_sqlite_candidate_bootstrap_rolls_back_all_new_tables_on_validation_failure(
    tmp_path,
) -> None:
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "candidate-atomicity.db"),
        )
    )
    backend.execute(UserDatabase._default_schema_statements_sqlite()[0])
    backend.execute(
        """
        CREATE TABLE teams (
            id INTEGER PRIMARY KEY,
            org_id INTEGER NOT NULL,
            name TEXT NOT NULL
        )
        """
    )
    db = UserDatabase.__new__(UserDatabase)
    db.backend = backend

    with pytest.raises(
        UserDatabaseError,
        match="profile candidate source tables",
    ):
        db._ensure_core_columns()

    assert not backend.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("organizations",),
    ).rows


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


def test_seed_default_data_raises_when_required_role_missing_after_seed():
    class _Backend:
        backend_type = BackendType.SQLITE

        def execute(self, sql, params=None):
            if sql.startswith("SELECT id FROM roles WHERE name = ?"):
                if params == ("admin",):
                    return _Result([])
                return _Result([{"id": 1}])
            if sql.startswith("SELECT id FROM permissions WHERE name = ?"):
                return _Result([{"id": 1}])
            return _Result([])

    db = UserDatabase.__new__(UserDatabase)
    db.backend = _Backend()

    with pytest.raises(UserDatabaseError, match="admin"):
        db._seed_default_data()


def test_seed_default_data_raises_when_required_role_permission_link_missing():
    class _Backend:
        backend_type = BackendType.SQLITE

        _roles = {
            "admin": 1,
            "user": 2,
            "viewer": 3,
        }
        _permissions = {
            "media.read": 10,
            "media.create": 11,
            "media.delete": 12,
            "sql.read": 13,
            "sql.target:media_db": 14,
            "system.configure": 15,
            "users.manage_roles": 16,
        }

        def execute(self, sql, params=None):
            if sql.startswith("SELECT id FROM roles WHERE name = ?"):
                return _Result([{"id": self._roles[params[0]]}])
            if sql.startswith("SELECT id FROM permissions WHERE name = ?"):
                return _Result([{"id": self._permissions[params[0]]}])
            if sql.startswith("SELECT 1 FROM role_permissions WHERE role_id = ? AND permission_id = ?"):
                if params == (1, 16):
                    return _Result([])
                return _Result([{"1": 1}])
            return _Result([])

    db = UserDatabase.__new__(UserDatabase)
    db.backend = _Backend()

    with pytest.raises(UserDatabaseError, match="users.manage_roles"):
        db._seed_default_data()
