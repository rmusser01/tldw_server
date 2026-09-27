"""Verify profile account-scope assertions against the real PostgreSQL API."""

from __future__ import annotations

import uuid

import asyncpg
import pytest
import pytest_asyncio

pytestmark = pytest.mark.integration

PROFILE_PATH = "/api/v1/users/me/profile"
DEFAULT_KEY = "preferences.chat.default_character_id"


@pytest_asyncio.fixture
async def profile_accounts(isolated_test_environment):
    """Create and authenticate two users in the official isolated database."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService

    client, _db_name = isolated_test_environment
    pool = await get_db_pool()
    password = "ProfileScope@Test2026!"  # nosec B105 - isolated test credential
    connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    accounts = {}
    try:
        for name in ("alice", "bob"):
            user_id = await connection.fetchval(
                """
                INSERT INTO users (uuid, username, email, password_hash, is_active, is_verified)
                VALUES ($1, $2, $3, $4, TRUE, TRUE)
                RETURNING id
                """,
                uuid.uuid4(),
                name,
                f"{name}@example.test",
                PasswordService().hash_password(password),
            )
            accounts[name] = {"id": user_id}
    finally:
        await connection.close()

    for name, account in accounts.items():
        login = client.post("/api/v1/auth/login", data={"username": name, "password": password})
        assert login.status_code == 200
        account["headers"] = {"Authorization": f"Bearer {login.json()['access_token']}"}
        response = client.patch(
            PROFILE_PATH,
            headers=account["headers"],
            json={"updates": [{"key": DEFAULT_KEY, "value": f"{name}-original"}]},
        )
        assert response.status_code == 200

    return client, accounts


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "PATCH"])
async def test_stale_profile_owner_is_rejected_without_changing_either_account(profile_accounts, method: str) -> None:
    """An Alice-captured request cannot run after authentication changes to Bob."""
    client, accounts = profile_accounts
    headers = {
        **accounts["bob"]["headers"],
        "X-TLDW-Expected-User-ID": str(accounts["alice"]["id"]),
    }
    kwargs = (
        {"json": {"updates": [{"key": DEFAULT_KEY, "value": "alice-stale-write"}]}}
        if method == "PATCH"
        else {"params": {"sections": "preferences"}}
    )
    response = client.request(method, PROFILE_PATH, headers=headers, **kwargs)

    assert response.status_code == 412
    assert response.json()["detail"]["code"] == "request_config_scope_changed"
    assert response.headers["cache-control"] == "no-store"
    for name, account in accounts.items():
        profile = client.get(PROFILE_PATH, headers=account["headers"], params={"sections": "preferences"})
        assert profile.status_code == 200
        assert profile.json()["preferences"][DEFAULT_KEY] == f"{name}-original"


@pytest.mark.asyncio
@pytest.mark.parametrize("include_expected_user", [False, True], ids=["legacy", "scoped"])
async def test_matching_and_legacy_profile_requests_set_and_clear_preferences(
    profile_accounts, include_expected_user: bool
) -> None:
    """The optional scope guard preserves existing profile set/clear behavior."""
    client, accounts = profile_accounts
    headers = dict(accounts["bob"]["headers"])
    if include_expected_user:
        headers["X-TLDW-Expected-User-ID"] = str(accounts["bob"]["id"])

    for value in ("bob-new", None):
        response = client.patch(
            PROFILE_PATH,
            headers=headers,
            json={"updates": [{"key": DEFAULT_KEY, "value": value}]},
        )
        assert response.status_code == 200
        assert DEFAULT_KEY in response.json()["applied"]
        profile = client.get(PROFILE_PATH, headers=headers, params={"sections": "preferences"})
        assert profile.status_code == 200
        assert profile.json()["preferences"].get(DEFAULT_KEY) == value
