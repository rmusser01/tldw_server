import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.tests.helpers.audit_helpers import await_audit_action, flush_audit_events
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user


@pytest.mark.real_audit
@pytest.mark.asyncio
async def test_team_membership_audit_events_sqlite(tmp_path, real_audit_service):
    # Configure single-user mode with API key; real_audit_service sets USER_DB_BASE_DIR
    os.environ['AUTH_MODE'] = 'single_user'
    os.environ['SINGLE_USER_API_KEY'] = 'audit-key-123'

    # Point AuthNZ DB at tmp path
    db_path = tmp_path / 'users.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

    # Reset singletons and init schema
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Create admin user (should be id=1) and a target user.
    admin_id = await create_authnz_test_user(
        pool, username="single_user", email="single@example.com"
    )
    target_id = await create_authnz_test_user(
        pool, username="victim", email="victim@example.com"
    )

    # Prepare app
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    headers = {"X-API-KEY": os.environ['SINGLE_USER_API_KEY']}

    with TestClient(app) as client:
        # Create org
        r = client.post("/api/v1/admin/orgs", json={"name": "Audit Org"}, headers=headers)
        assert r.status_code == 200, r.text
        org = r.json()

        # Create team
        r = client.post(f"/api/v1/admin/orgs/{org['id']}/teams", json={"name": "QA"}, headers=headers)
        assert r.status_code == 200, r.text
        team = r.json()

        # Add team member -> should log audit event for actor (single_user)
        r = client.post(
            f"/api/v1/admin/teams/{team['id']}/members",
            json={"user_id": target_id, "role": "member"},
            headers=headers,
        )
        assert r.status_code == 200, r.text

        flush_audit_events(client, int(admin_id))

    # Ensure audit services flush events before inspection.
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services
    await shutdown_all_audit_services()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    audit_db = DatabasePaths.get_audit_db_path(int(admin_id))
    assert audit_db.exists(), f"Audit DB not found: {audit_db}"

    cnt = await await_audit_action(audit_db, "team_member.add")
    assert cnt >= 1
