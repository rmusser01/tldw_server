"""TASK-15640: self-service email change needs the current password, keeps the previous address, notifies it."""

import sqlite3
from io import StringIO
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import users
from tldw_Server_API.app.api.v2.endpoints import user_profiles as v2_profiles
from tldw_Server_API.app.core.AuthNZ import migrations
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.password_service import hash_password
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.user_profile_writes import update_user_email
from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo
from tldw_Server_API.app.core.UserProfiles.update_service import UserProfileUpdateService
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

OWNER_EMAIL = "owner@example.test"
OWNER_PASSWORD = "Owner-Pass-2026!"


def test_migration_108_adds_email_change_columns_once(tmp_path) -> None:
    with sqlite3.connect(tmp_path / "users.db") as conn:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL)")
        conn.execute("INSERT INTO users (id, email) VALUES (1, 'owner@example.test')")
        migrations.migration_108_add_users_email_change_columns(conn)
        migrations.migration_108_add_users_email_change_columns(conn)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
        row = conn.execute("SELECT email, previous_email, email_changed_at FROM users").fetchone()
    assert (columns.count("previous_email"), columns.count("email_changed_at")) == (1, 1)
    assert row == ("owner@example.test", None, None)


def test_migration_108_skips_without_users_table(tmp_path) -> None:
    """Synthetic databases migrate to the latest version without a users table (cf. migrations 100 and 107)."""
    with sqlite3.connect(tmp_path / "synthetic.db") as conn:
        migrations.migration_108_add_users_email_change_columns(conn)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "users" not in tables
