import os

import pytest

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


@pytest.mark.asyncio
async def test_orgs_teams_crud_sqlite(tmp_path, authnz_schema_ready):
    # Configure SQLite for AuthNZ
    os.environ['AUTH_MODE'] = 'single_user'
    db_path = tmp_path / 'users.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    # Reset singletons
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    reset_settings()
    await reset_db_pool()

    # Schema ensured by authnz_schema_ready; acquire pool for ops
    pool = await get_db_pool()

    # Create a dummy user for membership FKs through the guarded write path.
    user_id = await create_authnz_test_user(
        pool, username="alice", email="alice@example.com"
    )

    # Use service helpers
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
        add_org_member,
        add_team_member,
        create_organization_with_owner_membership,
        create_team,
        list_organizations,
        list_team_members,
    )

    org = await create_organization_with_owner_membership(
        name="Acme Corp",
        owner_user_id=user_id,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )
    assert org['id'] > 0 and org['name'] == 'Acme Corp'

    orgs = await list_organizations()
    assert any(o['name'] == 'Acme Corp' for o in orgs)

    await add_org_member(
        org_id=org['id'],
        user_id=user_id,
        role='owner',
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )

    team = await create_team(org_id=org['id'], name="Research")
    assert team['org_id'] == org['id'] and team['name'] == 'Research'

    member = await add_team_member(
        team_id=team['id'],
        user_id=user_id,
        role='member',
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )
    assert member['team_id'] == team['id'] and member['user_id'] == user_id

    members = await list_team_members(team_id=team['id'])
    assert len(members) == 1 and members[0]['user_id'] == user_id
