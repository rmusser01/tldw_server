"""Shared fixture: back a fake admin principal with real AuthNZ rows.

The shared-BYOK write path (PUT /telegram/admin/bot) authorizes the actor against
the users table and the org/team rows, never against the principal's claims. A
bare ``AuthPrincipal(user_id=202, roles=["admin"])`` therefore gets a 403.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.utils.chatops import telegram_support
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user


@pytest.fixture()
def back_principal(client_user_only, monkeypatch, tmp_path):
    """Return ``back(principal) -> principal`` that seeds the principal's user and scope rows."""
    pool = DatabasePool(Settings(
        AUTH_MODE="multi_user", DATABASE_URL=f"sqlite:///{tmp_path / 'telegram_authnz.db'}",
        JWT_SECRET_KEY="x" * 32,
    ))

    async def _get_pool():
        return pool

    # Patches get_db_pool for telegram_support only; the runtime repo uses the global pool, so two pools coexist in one test.
    monkeypatch.setattr(telegram_support, "get_db_pool", _get_pool)

    async def _seed(principal: AuthPrincipal) -> AuthPrincipal:
        await pool.initialize()
        user_id = await ensure_test_user(pool, f"telegram-actor-{principal.user_id}", role="admin")
        org_id = principal.active_org_id or (principal.org_ids or [1])[0]
        await pool.execute(
            "INSERT OR IGNORE INTO organizations (id, name) VALUES (?, ?)",
            org_id, f"telegram-org-{org_id}",
        )
        if principal.active_team_id:
            await pool.execute(
                "INSERT OR IGNORE INTO teams (id, org_id, name) VALUES (?, ?, ?)",
                principal.active_team_id, org_id, f"telegram-team-{principal.active_team_id}",
            )
        return principal.model_copy(update={"user_id": user_id})

    def back(principal: AuthPrincipal) -> AuthPrincipal:
        return client_user_only.portal.call(_seed, principal)

    yield back
    client_user_only.portal.call(pool.close)
