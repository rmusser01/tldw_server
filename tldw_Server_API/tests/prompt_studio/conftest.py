# conftest.py
# Pytest configuration for Prompt Studio tests

import os
import tempfile
import sqlite3
from pathlib import Path
import uuid
from unittest.mock import patch
import importlib

import pytest
from fastapi.testclient import TestClient
from typing import Any

# Prompt Studio routes are not included in minimal test app mode.
os.environ["MINIMAL_TEST_APP"] = "0"
# Ensure test flags are set before loading the app module.
os.environ["TEST_MODE"] = "true"
os.environ["AUTH_MODE"] = "single_user"
os.environ["CSRF_ENABLED"] = "false"

import tldw_Server_API.app.main as main_mod

if getattr(main_mod, "_MINIMAL_TEST_APP", True):
    main_mod = importlib.reload(main_mod)
fastapi_app = main_mod.app
from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import get_prompt_studio_db
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

# Environment variables already set before app import above.

# Postgres setup is unified via tests._plugins.postgres.

@pytest.fixture(scope="session", autouse=True)
def prompt_studio_jobs_test_env(tmp_path_factory: pytest.TempPathFactory):
    jobs_dir = tmp_path_factory.mktemp("prompt_studio_jobs")
    jobs_db_path = jobs_dir / "jobs.sqlite"
    os.environ["JOBS_DB_PATH"] = str(jobs_db_path)
    os.environ["JOBS_QUOTA_MAX_QUEUED_PROMPT_STUDIO"] = "0"
    os.environ["JOBS_QUOTA_MAX_INFLIGHT_PROMPT_STUDIO"] = "0"
    os.environ["JOBS_QUOTA_SUBMITS_PER_MIN_PROMPT_STUDIO"] = "0"
    yield
    for key in (
        "JOBS_DB_PATH",
        "JOBS_QUOTA_MAX_QUEUED_PROMPT_STUDIO",
        "JOBS_QUOTA_MAX_INFLIGHT_PROMPT_STUDIO",
        "JOBS_QUOTA_SUBMITS_PER_MIN_PROMPT_STUDIO",
    ):
        os.environ.pop(key, None)

@pytest.fixture(autouse=True)
def enable_test_mode(monkeypatch):
    """Enable test mode for all tests."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("CSRF_ENABLED", "false")


@pytest.fixture(autouse=True)
def isolate_prompt_studio_jobs_db(monkeypatch, tmp_path):
    """Keep durable Jobs rows from leaking across Prompt Studio tests."""

    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "prompt-studio-jobs.sqlite"))

@pytest.fixture
def mock_current_user():
    """Mock user for authentication."""
    return {
        "id": "test-user-123",
        "username": "testuser",
        "email": "test@example.com",
        "is_active": True,
        "is_authenticated": True
    }

@pytest.fixture(autouse=True)
def mock_auth_dependency(mock_current_user):
    """Automatically mock authentication for all tests."""
    with patch('tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_current_user') as mock_get_user:
        mock_get_user.return_value = mock_current_user
        with patch('tldw_Server_API.app.api.v1.API_Deps.auth_deps.get_current_active_user') as mock_active:
            mock_active.return_value = mock_current_user
            yield

@pytest.fixture(scope="session")
def test_db_path():
    """Session-scoped test database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_prompt_studio.db"
        yield str(db_path)

@pytest.fixture
def isolated_db(test_db_path):
    """Isolated database for each test."""
    import uuid

    unique_path = f"{test_db_path}_{uuid.uuid4()}.db"
    db = PromptStudioDatabase(unique_path, "test-client")

    yield db

    if hasattr(db, "close"):
        try:
            db.close()
        except Exception:
            _ = None

    if os.path.exists(unique_path):
        os.unlink(unique_path)

@pytest.fixture
def auth_headers():
    """Authentication headers for API tests."""
    return {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json"
    }

def _quote_ident(name: str) -> str:
    escaped_name = name.replace('"', '""')
    return f'"{escaped_name}"'

def _row_name(row) -> str:
    if isinstance(row, (list, tuple)) and row:
        value = row[0]
        return "" if value is None else str(value)
    if isinstance(row, dict):
        value = row.get("tablename") or row.get("name")
        return "" if value is None else str(value)
    if hasattr(row, "keys"):
        try:
            keys = set(row.keys())
            if "tablename" in keys:
                value = row["tablename"]
            elif "name" in keys:
                value = row["name"]
            else:
                value = None
            return "" if value is None else str(value)
        except Exception:
            return ""
    if hasattr(row, "get"):
        try:
            value = row.get("tablename") or row.get("name")
            return "" if value is None else str(value)
        except Exception:
            return ""
    return ""


def _reset_prompt_studio_tables(db: PromptStudioDatabase) -> None:
    if db.backend_type == BackendType.POSTGRESQL:
        with db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'prompt_studio_%'"
            )
            rows = cursor.fetchall()
            tables = [name for name in (_row_name(row) for row in rows) if name and "_fts" not in name]
            if tables:
                joined = ", ".join(_quote_ident(name) for name in tables)
                cursor.execute(f"TRUNCATE {joined} RESTART IDENTITY CASCADE")
        return

    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'prompt_studio_%'"
    )
    rows = cursor.fetchall()
    tables = [name for name in (_row_name(row) for row in rows) if name and "_fts" not in name]
    cursor.execute("PRAGMA foreign_keys=OFF")
    try:
        for name in tables:
            cursor.execute(f"DELETE FROM {_quote_ident(name)}")  # nosec B608
        cursor.execute(
            "DELETE FROM sync_log WHERE entity LIKE 'prompt_studio_%'"
        )
        if tables:
            placeholders = ", ".join("?" for _ in tables)
            try:
                cursor.execute(
                    f"DELETE FROM sqlite_sequence WHERE name IN ({placeholders})",  # nosec B608
                    tables,
                )
            except Exception:
                _ = None
    finally:
        cursor.execute("PRAGMA foreign_keys=ON")
    conn.commit()


@pytest.fixture(scope="session")
def prompt_studio_sqlite_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("prompt_studio_shared")
    db_path = base / "prompt_studio_shared.sqlite"
    db = PromptStudioDatabase(str(db_path), "prompt-studio-session")
    if hasattr(db, "close"):
        try:
            db.close()
        except Exception:
            _ = None
    return db_path


@pytest.fixture(scope="session")
def prompt_studio_pg_placeholder_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    base = tmp_path_factory.mktemp("prompt_studio_pg")
    return str(base / "prompt_studio_pg_placeholder.sqlite")


@pytest.fixture(scope="session")
def prompt_studio_pg_shared_db(
    pg_database_config_session: DatabaseConfig,
    prompt_studio_pg_placeholder_path: str,
) -> PromptStudioDatabase:
    backend = DatabaseBackendFactory.create_backend(pg_database_config_session)
    db_instance = PromptStudioDatabase(
        db_path=prompt_studio_pg_placeholder_path,
        client_id="dual-postgres",
        backend=backend,
    )
    try:
        yield db_instance
    finally:
        try:
            db_instance.close_connection()
        except Exception:
            _ = None
        try:
            db_instance.close()
        except Exception:
            _ = None
        try:
            backend.get_pool().close_all()
        except Exception:
            _ = None


@pytest.fixture(params=["sqlite", "postgres"])
def prompt_studio_dual_backend_db(
    request,
    prompt_studio_sqlite_db_path,
):
    """Provide a PromptStudioDatabase instance configured for the requested backend."""

    label: str = request.param

    if label == "sqlite":
        db_instance = PromptStudioDatabase(str(prompt_studio_sqlite_db_path), f"dual-{label}")
    else:
        db_instance = request.getfixturevalue("prompt_studio_pg_shared_db")

    try:
        try:
            _reset_prompt_studio_tables(db_instance)
        except sqlite3.DatabaseError:
            if label != "sqlite":
                raise
            try:
                db_instance.close_connection()
            except Exception:
                _ = None
            try:
                db_instance.close()
            except Exception:
                _ = None
            try:
                os.unlink(str(prompt_studio_sqlite_db_path))
            except Exception:
                _ = None
            db_instance = PromptStudioDatabase(str(prompt_studio_sqlite_db_path), f"dual-{label}")
            _reset_prompt_studio_tables(db_instance)
        yield label, db_instance
    finally:
        try:
            db_instance.close_connection()
        except Exception:
            _ = None
        if label == "sqlite":
            try:
                db_instance.close()
            except Exception:
                _ = None
        # No explicit drop needed; pg_database_config fixture handles DB cleanup


@pytest.fixture
def prompt_studio_dual_backend_client(
    prompt_studio_dual_backend_db,
    mock_current_user,
    tmp_path,
    monkeypatch,
):
    """Yield a FastAPI TestClient wired to the selected Prompt Studio backend."""

    from tldw_Server_API.app.core.config import settings as app_settings
    # Patch prompt_studio_deps.get_current_active_user directly; dependency_overrides
    # on auth_deps does not affect this symbol.
    from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps as ps_deps

    backend_label, db_instance = prompt_studio_dual_backend_db

    monkeypatch.setitem(app_settings, "USER_DB_BASE_DIR", tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")

    async def override_user():
        return User(
            id=mock_current_user.get("id", "test-user-123"),
            username=mock_current_user.get("username", "testuser"),
            email=mock_current_user.get("email", "test@example.com"),
            is_active=True,
        )

    async def override_db():
        try:
            yield db_instance
        finally:
            try:
                db_instance.close_connection()
            except Exception:
                _ = None

    _app: Any = fastapi_app  # appease static analyzers about dynamic attributes
    _app.dependency_overrides[get_request_user] = override_user
    _app.dependency_overrides[get_prompt_studio_db] = override_db
    # get_prompt_studio_user calls ps_deps.get_current_active_user directly; patch it
    monkeypatch.setattr(ps_deps, "get_current_active_user", lambda: mock_current_user, raising=False)

    try:
        with TestClient(_app) as client:
            yield backend_label, client, db_instance
    finally:
        _app.dependency_overrides.clear()
        monkeypatch.delenv("TEST_MODE", raising=False)
