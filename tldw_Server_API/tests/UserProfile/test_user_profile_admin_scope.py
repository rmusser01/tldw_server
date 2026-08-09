from __future__ import annotations

import sqlite3
import uuid

import pytest

from tldw_Server_API.app.api.v1.endpoints.admin import _load_bulk_user_candidates
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    add_org_member,
    add_team_member,
    create_organization,
    create_team,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_team_admin_scope_bulk_candidates(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()
    await ensure_authnz_schema_ready_once()

    pool = await get_db_pool()
    suffix = uuid.uuid4().hex[:8]

    with sqlite3.connect(db_path) as maintenance:
        maintenance.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active)
            VALUES (?, ?, ?, 1)
            """,
            (f"team_admin_{suffix}", f"team_admin_{suffix}@example.com", "x"),
        )
        maintenance.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active)
            VALUES (?, ?, ?, 1)
            """,
            (f"team_member_{suffix}", f"team_member_{suffix}@example.com", "x"),
        )
        maintenance.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active)
            VALUES (?, ?, ?, 1)
            """,
            (f"org_member_{suffix}", f"org_member_{suffix}@example.com", "x"),
        )

    admin_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?",
        (f"team_admin_{suffix}",),
    )
    member_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?",
        (f"team_member_{suffix}",),
    )
    outsider_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?",
        (f"org_member_{suffix}",),
    )

    org = await create_organization(name=f"Scope Org {suffix}", owner_user_id=None)
    org_id = int(org["id"])
    await add_org_member(org_id=org_id, user_id=int(admin_id), role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
    await add_org_member(org_id=org_id, user_id=int(member_id), role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
    await add_org_member(org_id=org_id, user_id=int(outsider_id), role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)

    team = await create_team(org_id=org_id, name=f"Scope Team {suffix}")
    team_id = int(team["id"])
    await add_team_member(team_id=team_id, user_id=int(admin_id), role="lead", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
    await add_team_member(team_id=team_id, user_id=int(member_id), role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)

    principal = AuthPrincipal(
        kind="user",
        user_id=int(admin_id),
        roles=[],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
        active_org_id=None,
        active_team_id=None,
    )

    target_ids = await _load_bulk_user_candidates(
        principal=principal,
        org_id=None,
        team_id=None,
        role=None,
        is_active=None,
        search=None,
        user_ids=None,
    )
    assert set(target_ids) == {int(admin_id), int(member_id)}

    team_only_ids = await _load_bulk_user_candidates(
        principal=principal,
        org_id=org_id,
        team_id=team_id,
        role=None,
        is_active=None,
        search=None,
        user_ids=None,
    )
    assert set(team_only_ids) == {int(admin_id), int(member_id)}
