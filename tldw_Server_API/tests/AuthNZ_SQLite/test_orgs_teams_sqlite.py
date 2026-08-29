import os
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_orgs_teams_crud_sqlite(tmp_path, authnz_schema_ready):
    # Configure SQLite for AuthNZ
    os.environ['AUTH_MODE'] = 'single_user'
    db_path = tmp_path / 'users.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    # Reset singletons
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    reset_settings()
    await reset_db_pool()

    # Schema ensured by authnz_schema_ready; acquire pool for ops
    pool = await get_db_pool()

    # Create a dummy user for membership FKs
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active)
            VALUES (?, ?, ?, 1)
            """,
            ("alice", "alice@example.com", "x"),
        )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", "alice")

    # Use service helpers
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
        create_organization, list_organizations, create_team, add_team_member, list_team_members
    )

    org = await create_organization(name="Acme Corp", owner_user_id=user_id)
    assert org['id'] > 0 and org['name'] == 'Acme Corp'

    orgs = await list_organizations()
    assert any(o['name'] == 'Acme Corp' for o in orgs)

    team = await create_team(org_id=org['id'], name="Research")
    assert team['org_id'] == org['id'] and team['name'] == 'Research'

    member = await add_team_member(team_id=team['id'], user_id=user_id, role='member')
    assert member['team_id'] == team['id'] and member['user_id'] == user_id

    members = await list_team_members(team_id=team['id'])
    assert len(members) == 1 and members[0]['user_id'] == user_id
