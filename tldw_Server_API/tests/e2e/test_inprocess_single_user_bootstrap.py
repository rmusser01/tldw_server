"""
Regression guard for in-process single-user e2e setup.
"""

import asyncio
import os

import pytest

from . import conftest as e2e_conftest
from .fixtures import api_client


def _is_truthy_env(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "y", "on"}


@pytest.mark.critical
def test_inprocess_auth_mode_detection_uses_explicit_profile(monkeypatch):
    """In-process collection must not probe an external server for its auth mode."""
    monkeypatch.setenv("E2E_INPROCESS", "1")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("E2E_TEST_BASE_URL", "http://127.0.0.1:1")

    assert e2e_conftest._detect_auth_mode() == "single_user"


@pytest.mark.critical
def test_inprocess_single_user_profile_has_authnz_user_row(api_client):
    """Ensure fixed single-user profile has a concrete AuthNZ users row."""
    if not _is_truthy_env("E2E_INPROCESS"):
        pytest.skip("Only applicable to in-process e2e runs.")

    health = api_client.health_check()
    auth_mode = str(health.get("auth_mode") or os.getenv("AUTH_MODE", "")).lower()
    if auth_mode not in {"single_user", "single-user", "singleuser"}:
        pytest.skip(f"Single-user invariant check not applicable in auth mode: {auth_mode}")

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import get_users_db

    async def _read_user():
        settings = get_settings()
        users_db = await get_users_db()
        user = await users_db.get_user_by_id(int(settings.SINGLE_USER_FIXED_ID))
        return settings, user

    settings, user = asyncio.run(_read_user())
    if user is None:
        pytest.fail(
            f"Expected users row for SINGLE_USER_FIXED_ID={settings.SINGLE_USER_FIXED_ID}"
        )
    if int(user["id"]) != int(settings.SINGLE_USER_FIXED_ID):
        pytest.fail(
            "Single-user bootstrap row id mismatch: "
            f"{user['id']} != {settings.SINGLE_USER_FIXED_ID}"
        )
    if user.get("is_active") not in (True, 1):
        pytest.fail("Single-user bootstrap row exists but is not active")
