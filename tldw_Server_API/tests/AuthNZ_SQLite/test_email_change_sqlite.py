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

OWNER_EMAIL = "owner@example.com"
OWNER_PASSWORD = "Owner-Pass-2026!"


def test_migration_108_adds_email_change_columns_once(tmp_path) -> None:
    with sqlite3.connect(tmp_path / "users.db") as conn:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL)")
        conn.execute("INSERT INTO users (id, email) VALUES (1, 'owner@example.com')")
        migrations.migration_108_add_users_email_change_columns(conn)
        migrations.migration_108_add_users_email_change_columns(conn)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
        row = conn.execute("SELECT email, previous_email, email_changed_at FROM users").fetchone()
    assert (columns.count("previous_email"), columns.count("email_changed_at")) == (1, 1)
    assert row == ("owner@example.com", None, None)


def test_migration_108_skips_without_users_table(tmp_path) -> None:
    """Synthetic databases migrate to the latest version without a users table (cf. migrations 100 and 107)."""
    with sqlite3.connect(tmp_path / "synthetic.db") as conn:
        migrations.migration_108_add_users_email_change_columns(conn)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "users" not in tables


@pytest.fixture
async def owner_db(tmp_path, monkeypatch, fresh_services):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("JWT_SECRET_KEY", "synthetic-signing-key-for-regression-tests-only")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    reset_settings()
    pool = await get_db_pool()
    user_id = await create_authnz_test_user(
        pool, username="owner", email=OWNER_EMAIL, password_hash=hash_password(OWNER_PASSWORD), is_verified=True
    )
    try:
        yield SimpleNamespace(pool=pool, user_id=user_id)
    finally:
        reset_settings()


async def _columns(db) -> tuple:
    row = dict(await db.pool.fetchone("SELECT email, previous_email, email_changed_at FROM users WHERE id = ?", db.user_id))
    return row["email"], row["previous_email"], row["email_changed_at"]


def _raw(db, statement: str) -> None:
    """Write outside the AuthNZ write guard, as an operator or the passage of time would."""
    with sqlite3.connect(db.pool.db_path) as conn:
        conn.execute(statement, (db.user_id,))


async def _change(db, email: str):
    async with db.pool.transaction() as conn:
        return await update_user_email(conn, backend="sqlite", user_id=db.user_id, email=email)


@pytest.mark.asyncio
async def test_email_change_keeps_the_previous_address_and_time(owner_db):
    await _change(owner_db, "second@example.com")
    email, previous, changed_at = await _columns(owner_db)
    assert (email, previous) == ("second@example.com", OWNER_EMAIL)
    assert changed_at is not None


@pytest.mark.asyncio
async def test_second_change_within_seven_days_keeps_the_original_previous_address(owner_db):
    await _change(owner_db, "typo@example.com")
    await _change(owner_db, "fixed@example.com")
    assert (await _columns(owner_db))[:2] == ("fixed@example.com", OWNER_EMAIL)


@pytest.mark.asyncio
async def test_change_after_seven_days_records_the_immediately_previous_address(owner_db):
    await _change(owner_db, "second@example.com")
    _raw(owner_db, "UPDATE users SET email_changed_at = datetime('now', '-8 days') WHERE id = ?")
    await _change(owner_db, "third@example.com")
    assert (await _columns(owner_db))[:2] == ("third@example.com", "second@example.com")


@pytest.mark.asyncio
async def test_same_address_in_any_case_changes_nothing(owner_db):
    result = await _change(owner_db, OWNER_EMAIL.upper())
    assert result.affected_user_ids == ()
    assert await _columns(owner_db) == (OWNER_EMAIL, None, None)


@pytest.mark.asyncio
async def test_own_profile_email_edit_records_the_previous_address(owner_db):
    async with owner_db.pool.transaction() as conn:
        result = await UserProfileUpdateService(owner_db.pool).apply_updates(
            user_id=owner_db.user_id, updates=(("identity.email", "Second@Example.com"),), roles={"user"},
            dry_run=False, db_conn=conn, updated_by=owner_db.user_id,
        )
    assert result.applied == ["identity.email"]
    assert (await _columns(owner_db))[:2] == ("second@example.com", OWNER_EMAIL)


class _FixedStorage:
    async def user_quota_status(self, user_id: int) -> dict[str, float]:
        return {"quota_mb": 5120, "used_mb": 0.0}

    async def native_usage(self, user_id: int) -> dict[str, int]:
        return {"bytes": 0}


@pytest.fixture
async def email_api(owner_db, monkeypatch):
    async def _principal():
        return AuthPrincipal(kind="user", user_id=owner_db.user_id, username="owner", email=OWNER_EMAIL, roles=["user"])

    async def _storage():
        return _FixedStorage()

    async def _no_rate_limit():
        return None

    monkeypatch.setattr("tldw_Server_API.app.services.storage_quota_service.get_storage_service", _storage)
    app = FastAPI()
    app.include_router(users.router, prefix="/api/v1")
    app.include_router(v2_profiles.router, prefix="/api/v2")
    app.dependency_overrides[get_auth_principal] = _principal
    app.dependency_overrides[check_rate_limit] = _no_rate_limit
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client


EMAIL_UPDATE = {"updates": [{"key": "identity.email", "value": "new@example.com"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("blank_hash", "password", "code"),
    [
        (False, None, "current_password_required"),
        (False, "Wrong-Pass-2026!", "current_password_invalid"),
        (True, OWNER_PASSWORD, "email_change_requires_admin"),
    ],
    ids=["missing", "wrong", "no-password-account"],
)
async def test_put_email_change_without_the_owners_password_is_refused(owner_db, email_api, blank_hash, password, code):
    if blank_hash:
        _raw(owner_db, "UPDATE users SET password_hash = '' WHERE id = ?")
    body = {"email": "new@example.com", **({"current_password": password} if password else {})}
    response = await email_api.put("/api/v1/users/me", json=body)
    assert (response.status_code, response.json()["error_code"]) == (403, code)
    assert (await _columns(owner_db))[:2] == (OWNER_EMAIL, None)


@pytest.mark.asyncio
async def test_put_email_change_with_the_owners_password_succeeds(owner_db, email_api):
    response = await email_api.put("/api/v1/users/me", json={"email": "new@example.com", "current_password": OWNER_PASSWORD})
    assert response.status_code == 200, response.text
    assert response.json()["email"] == "new@example.com"
    assert (await _columns(owner_db))[:2] == ("new@example.com", OWNER_EMAIL)


@pytest.mark.asyncio
async def test_patch_v1_email_change_requires_the_owners_password(owner_db, email_api):
    refused = await email_api.patch("/api/v1/users/me/profile", json=EMAIL_UPDATE)
    assert (refused.status_code, refused.json()["error_code"]) == (403, "current_password_required")
    accepted = await email_api.patch("/api/v1/users/me/profile", json={**EMAIL_UPDATE, "current_password": OWNER_PASSWORD})
    assert accepted.status_code == 200, accepted.text
    assert "identity.email" in accepted.json()["applied"]
    assert (await _columns(owner_db))[:2] == ("new@example.com", OWNER_EMAIL)


@pytest.mark.asyncio
async def test_patch_v2_email_change_requires_the_owners_password(owner_db, email_api):
    response = await email_api.patch("/api/v2/users/me/profile", json={**EMAIL_UPDATE, "current_password": "Wrong-Pass-2026!"})
    assert response.status_code == 403
    assert response.json()["detail"]["error_code"] == "current_password_invalid"
    assert (await _columns(owner_db))[0] == OWNER_EMAIL


@pytest.mark.asyncio
async def test_refused_email_change_applies_none_of_the_other_updates(owner_db, email_api):
    body = {
        "updates": [{"key": "preferences.ui.theme", "value": "paper"}, *EMAIL_UPDATE["updates"]],
        "current_password": "Wrong-Pass-2026!",
    }
    response = await email_api.patch("/api/v1/users/me/profile", json=body)
    assert response.status_code == 403
    overrides = await UserProfileOverridesRepo(owner_db.pool).list_overrides_for_user(owner_db.user_id)
    assert all(row["key"] != "preferences.ui.theme" for row in overrides)


@pytest.mark.asyncio
async def test_unchanged_address_in_other_case_needs_no_password(owner_db, email_api):
    body = {"updates": [{"key": "identity.email", "value": OWNER_EMAIL.upper()}]}
    response = await email_api.patch("/api/v1/users/me/profile", json=body)
    assert response.status_code == 200, response.text
    assert await _columns(owner_db) == (OWNER_EMAIL, None, None)


@pytest.mark.asyncio
async def test_any_changed_identity_email_in_the_request_needs_the_password(owner_db, email_api):
    body = {
        "updates": [
            {"key": "identity.email", "value": "new@example.com"},
            {"key": "identity.email", "value": OWNER_EMAIL},
        ]
    }
    response = await email_api.patch("/api/v1/users/me/profile", json=body)
    assert (response.status_code, response.json()["error_code"]) == (403, "current_password_required")
    assert await _columns(owner_db) == (OWNER_EMAIL, None, None)


@pytest.mark.asyncio
async def test_dry_run_email_change_needs_no_password(owner_db, email_api):
    response = await email_api.patch("/api/v1/users/me/profile", json={**EMAIL_UPDATE, "dry_run": True})
    assert response.status_code == 200, response.text
    assert (await _columns(owner_db))[0] == OWNER_EMAIL


@pytest.mark.asyncio
async def test_single_user_mode_needs_no_password(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import email_change

    monkeypatch.setenv("AUTH_MODE", "single_user")
    reset_settings()
    try:
        # db_conn=None: single-user mode must decide without reading anything.
        await email_change.authorize_own_email_change(
            db_conn=None, is_postgres=False, user_id=1, new_emails=("new@example.com",), current_password=None
        )
    finally:
        reset_settings()


@pytest.fixture
def sent_notices(owner_db, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import email_change

    notices: list[dict] = []

    class _Mail:
        async def send_email_changed_email(self, **kwargs):
            with sqlite3.connect(owner_db.pool.db_path) as conn:
                committed = conn.execute("SELECT email FROM users WHERE id = ?", (owner_db.user_id,)).fetchone()[0]
            notices.append({**kwargs, "committed_email": committed})
            return True

    monkeypatch.setattr(email_change, "get_email_service", lambda: _Mail())
    return notices


@pytest.mark.asyncio
async def test_email_change_notifies_the_previous_address_after_commit(owner_db, email_api, sent_notices):
    response = await email_api.put("/api/v1/users/me", json={"email": "new@example.com", "current_password": OWNER_PASSWORD})
    assert response.status_code == 200, response.text
    assert [(n["to_email"], n["new_email"], n["committed_email"]) for n in sent_notices] == [
        (OWNER_EMAIL, "new@example.com", "new@example.com")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/v1/users/me/profile", "/api/v2/users/me/profile"])
async def test_profile_email_change_notifies_the_previous_address(owner_db, email_api, sent_notices, path):
    response = await email_api.patch(path, json={**EMAIL_UPDATE, "current_password": OWNER_PASSWORD})
    assert response.status_code == 200, response.text
    assert [n["to_email"] for n in sent_notices] == [OWNER_EMAIL]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [EMAIL_UPDATE, {**EMAIL_UPDATE, "dry_run": True}, {"updates": [{"key": "identity.email", "value": OWNER_EMAIL.upper()}]}],
    ids=["refused", "dry-run", "unchanged"],
)
async def test_no_notice_without_a_committed_change(owner_db, email_api, sent_notices, body):
    await email_api.patch("/api/v1/users/me/profile", json=body)
    assert sent_notices == []


@pytest.mark.asyncio
async def test_failed_notice_does_not_fail_the_change(owner_db, email_api, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import email_change

    class _BrokenMail:
        async def send_email_changed_email(self, **_kwargs):
            raise OSError(f"smtp unreachable for {OWNER_EMAIL}")

    monkeypatch.setattr(email_change, "get_email_service", lambda: _BrokenMail())
    buffer = StringIO()
    sink = logger.add(buffer, level="DEBUG")
    try:
        response = await email_api.put("/api/v1/users/me", json={"email": "new@example.com", "current_password": OWNER_PASSWORD})
    finally:
        logger.remove(sink)
    assert response.status_code == 200, response.text
    assert (await _columns(owner_db))[0] == "new@example.com"
    warnings = [line for line in buffer.getvalue().splitlines() if "Email-change notice" in line]
    assert len(warnings) == 1
    assert OWNER_EMAIL not in warnings[0] and "new@example.com" not in warnings[0]
