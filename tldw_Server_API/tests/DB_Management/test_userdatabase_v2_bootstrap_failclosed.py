from pathlib import Path
from types import SimpleNamespace
import pytest

from tldw_Server_API.app.core.DB_Management.UserDatabase_v2 import (
    UserDatabase,
    UserDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


class _Result:
    def __init__(self, rows):
        self.rows = rows


def test_initialize_schema_raises_when_required_schema_apply_fails(monkeypatch):
    db = UserDatabase.__new__(UserDatabase)
    db.backend = SimpleNamespace(backend_type=BackendType.SQLITE)

    monkeypatch.setattr(Path, "exists", lambda self: False)
    monkeypatch.setattr(db, "_default_schema_statements", lambda: ["CREATE TABLE users (id INTEGER)"])
    monkeypatch.setattr(
        db,
        "_apply_schema_statements",
        lambda _statements: (_ for _ in ()).throw(RuntimeError("schema boom")),
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

    with pytest.raises(UserDatabaseError, match="schema initialization"):
        db._initialize_schema()


def test_initialize_schema_raises_for_unsupported_backend_type():
    db = UserDatabase.__new__(UserDatabase)
    db.backend = SimpleNamespace(backend_type=SimpleNamespace(value="unsupported"))

    with pytest.raises(UserDatabaseError, match="Unsupported backend type"):
        db._initialize_schema()


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


def test_ensure_core_columns_handles_real_legacy_sqlite_uuid_migration(tmp_path):
    db_path = tmp_path / "users.db"
    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(db_path),
        )
    )
    backend.execute(
        "CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)"
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

    uuid_value = backend.execute("SELECT uuid FROM users WHERE username = ?", ("legacy-user",)).rows[0]["uuid"]
    if not uuid_value:
        pytest.fail("expected legacy SQLite user rows to receive a UUID backfill")


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
