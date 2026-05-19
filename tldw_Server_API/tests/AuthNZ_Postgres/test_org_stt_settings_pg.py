from __future__ import annotations

import uuid

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_org_stt_settings_postgres(test_db_pool):
    pool = test_db_pool

    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_authnz_core_tables_pg
    from tldw_Server_API.app.core.AuthNZ.repos.org_stt_settings_repo import (
        AuthnzOrgSttSettingsRepo,
    )

    await ensure_authnz_core_tables_pg(pool)
    await pool.execute(
        "INSERT INTO users (uuid, username, email, password_hash, is_active) VALUES ($1, $2, $3, $4, TRUE)",
        str(uuid.uuid4()),
        "pg-org-stt",
        "pg-org-stt@example.com",
        "x",
    )
    user_id = await pool.fetchval("SELECT id FROM users WHERE username = $1", "pg-org-stt")
    await pool.execute(
        """
        INSERT INTO organizations (uuid, name, slug, owner_user_id, is_active, metadata)
        VALUES ($1, $2, $3, $4, TRUE, '{}'::jsonb)
        """,
        str(uuid.uuid4()),
        "PG STT Org",
        "pg-stt-org",
        user_id,
    )
    org_id = await pool.fetchval("SELECT id FROM organizations WHERE slug = $1", "pg-stt-org")

    repo = AuthnzOrgSttSettingsRepo(pool)
    await repo.ensure_tables()

    created = await repo.upsert_settings(
        org_id=int(org_id),
        delete_audio_after_success=False,
        audio_retention_hours=12.0,
        redact_pii=True,
        allow_unredacted_partials=False,
        redact_categories=["email", "phone"],
        updated_by=int(user_id),
    )
    assert created["org_id"] == int(org_id)
    assert created["redact_categories"] == ["email", "phone"]

    fetched = await repo.get_settings(int(org_id))
    assert fetched["delete_audio_after_success"] is False
    assert fetched["audio_retention_hours"] == 12.0
    assert fetched["redact_pii"] is True
    assert fetched["allow_unredacted_partials"] is False
    assert fetched["redact_categories"] == ["email", "phone"]
