from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _setup_env(tmp_path) -> None:
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "unit-test-api-key"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_test_retention_preview.db'}"
    os.environ["TLDW_DB_ALLOWED_BASE_DIRS"] = str(tmp_path)
    os.environ["TLDW_DB_BACKUP_PATH"] = str(tmp_path / "backups")
    os.environ["USER_DB_BASE_DIR"] = str(tmp_path / "user_dbs")


async def _seed_audit_log_preview_data() -> int:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

    pool = await get_db_pool()
    username = "retention_preview_user"
    email = "retention_preview_user@example.com"
    user_id = await ensure_test_user(pool, username, email)

    now = datetime.now(timezone.utc)
    stale_created_at = (now - timedelta(days=120)).isoformat()
    fresh_created_at = (now - timedelta(days=10)).isoformat()
    await pool.execute(
        """
        INSERT INTO audit_logs (user_id, action, resource_type, resource_id, ip_address, details, created_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        int(user_id),
        "retention.preview.old",
        "audit",
        1,
        "127.0.0.1",
        '{"ok": true}',
        stale_created_at,
    )
    await pool.execute(
        """
        INSERT INTO audit_logs (user_id, action, resource_type, resource_id, ip_address, details, created_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        int(user_id),
        "retention.preview.fresh",
        "audit",
        2,
        "127.0.0.1",
        '{"ok": true}',
        fresh_created_at,
    )
    return int(user_id)


@pytest.mark.asyncio
async def test_retention_preview_returns_authoritative_counts_and_signature(tmp_path) -> None:
    _setup_env(tmp_path)

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        await _seed_audit_log_preview_data()

        response = client.post(
            "/api/v1/admin/retention-policies/audit_logs/preview",
            json={"current_days": 180, "days": 90},
        )

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["key"] == "audit_logs"
        assert payload["current_days"] == 180
        assert payload["new_days"] == 90
        assert payload["counts"]["audit_log_entries"] == 1
        assert payload["counts"]["job_records"] == 0
        assert payload["counts"]["backup_files"] == 0
        assert isinstance(payload["preview_signature"], str)
        assert payload["preview_signature"]


@pytest.mark.asyncio
async def test_retention_preview_returns_unknown_policy_and_invalid_range(tmp_path) -> None:
    _setup_env(tmp_path)

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        unknown_resp = client.post(
            "/api/v1/admin/retention-policies/missing_policy/preview",
            json={"current_days": 180, "days": 90},
        )
        assert unknown_resp.status_code == 404, unknown_resp.text
        assert unknown_resp.json()["detail"] == "unknown_policy"

        invalid_resp = client.post(
            "/api/v1/admin/retention-policies/audit_logs/preview",
            json={"current_days": 180, "days": 1},
        )
        assert invalid_resp.status_code == 400, invalid_resp.text
        assert invalid_resp.json()["detail"] == "invalid_range"


@pytest.mark.asyncio
async def test_retention_update_requires_valid_preview_signature(tmp_path) -> None:
    _setup_env(tmp_path)

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()

    headers = {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}

    with TestClient(app, headers=headers) as client:
        await _seed_audit_log_preview_data()

        missing_resp = client.put(
            "/api/v1/admin/retention-policies/audit_logs",
            json={"days": 90},
        )
        assert missing_resp.status_code == 400, missing_resp.text
        assert missing_resp.json()["detail"] == "preview_signature_required"

        invalid_resp = client.put(
            "/api/v1/admin/retention-policies/audit_logs",
            json={"days": 90, "preview_signature": "not-a-real-signature"},
        )
        assert invalid_resp.status_code == 400, invalid_resp.text
        assert invalid_resp.json()["detail"] == "invalid_preview_signature"

        preview_resp = client.post(
            "/api/v1/admin/retention-policies/audit_logs/preview",
            json={"current_days": 180, "days": 90},
        )
        assert preview_resp.status_code == 200, preview_resp.text
        preview_signature = preview_resp.json()["preview_signature"]

        update_resp = client.put(
            "/api/v1/admin/retention-policies/audit_logs",
            json={"days": 90, "preview_signature": preview_signature},
        )
        assert update_resp.status_code == 200, update_resp.text
        assert update_resp.json()["key"] == "audit_logs"
        assert update_resp.json()["days"] == 90
