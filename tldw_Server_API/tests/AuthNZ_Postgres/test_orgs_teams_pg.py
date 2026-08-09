import pytest

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway

_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orgs_teams_postgres(test_db_pool):
    pool = test_db_pool

    # Ensure org/team tables exist
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id SERIAL PRIMARY KEY,
            uuid VARCHAR(64) UNIQUE,
            name VARCHAR(255) UNIQUE NOT NULL,
            slug VARCHAR(255) UNIQUE,
            owner_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS teams (
            id SERIAL PRIMARY KEY,
            org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255),
            description TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (org_id, name)
        )
        """
    )
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS team_members (
            team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role VARCHAR(32) DEFAULT 'member',
            status VARCHAR(32) DEFAULT 'active',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team_id, user_id)
        )
        """
    )

    # Create user
    import uuid
    async with pool.transaction() as conn:
        await VersionedUserWriteGateway("postgres").insert_user(
            conn,
            values={
                "uuid": str(uuid.uuid4()),
                "username": "pgorguser",
                "email": "pgorguser@example.com",
                "password_hash": "x",
                "is_active": True,
            },
        )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = $1", "pgorguser")

    # Use services to exercise Postgres path
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
        add_team_member,
        create_organization,
        create_team,
        list_team_members,
    )
    org = await create_organization(name="PG Org", owner_user_id=user_id)
    assert org['id'] > 0
    team = await create_team(org_id=org['id'], name="PG Team")
    assert team['org_id'] == org['id']
    member = await add_team_member(
        team_id=team['id'],
        user_id=user_id,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )
    assert member['team_id'] == team['id'] and member['user_id'] == user_id
    members = await list_team_members(team_id=team['id'])
    assert any(m['user_id'] == user_id for m in members)
