import asyncio
from importlib import import_module
from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from starlette.requests import Request


pytestmark = pytest.mark.integration


def test_admin_update_org_stt_settings(monkeypatch, tmp_path, authnz_schema_ready_sync):
    base_dir = tmp_path / "test_admin_org_stt_settings"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "authnz_admin_stt.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("STT_DELETE_AUDIO_AFTER_SUCCESS", "1")
    monkeypatch.setenv("STT_AUDIO_RETENTION_HOURS", "0")
    monkeypatch.setenv("STT_REDACT_PII", "0")

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    asyncio.run(reset_db_pool())

    mod = import_module("tldw_Server_API.app.main")
    app = getattr(mod, "app")
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

    async def _principal_override(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="single_user",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        request.state.auth = AuthContext(
            principal=principal,
            ip=None,
            user_agent=None,
            request_id=None,
        )
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override
    try:
        with TestClient(app) as client:
            create_response = client.post("/api/v1/admin/orgs", json={"name": "STT Org"})
            assert create_response.status_code == 200, create_response.text
            org_id = create_response.json()["id"]

            initial = client.get(f"/api/v1/admin/orgs/{org_id}/stt/settings")
            assert initial.status_code == 200, initial.text
            initial_payload = initial.json()
            assert initial_payload == {
                "org_id": org_id,
                "delete_audio_after_success": True,
                "audio_retention_hours": 0.0,
                "redact_pii": False,
                "allow_unredacted_partials": False,
                "redact_categories": [],
            }

            update_response = client.patch(
                f"/api/v1/admin/orgs/{org_id}/stt/settings",
                json={
                    "delete_audio_after_success": False,
                    "audio_retention_hours": 24.0,
                    "redact_pii": True,
                    "allow_unredacted_partials": False,
                    "redact_categories": ["email", "phone"],
                },
            )
            assert update_response.status_code == 200, update_response.text
            assert update_response.json() == {
                "org_id": org_id,
                "delete_audio_after_success": False,
                "audio_retention_hours": 24.0,
                "redact_pii": True,
                "allow_unredacted_partials": False,
                "redact_categories": ["email", "phone"],
            }

            fetched = client.get(f"/api/v1/admin/orgs/{org_id}/stt/settings")
            assert fetched.status_code == 200, fetched.text
            assert fetched.json() == {
                "org_id": org_id,
                "delete_audio_after_success": False,
                "audio_retention_hours": 24.0,
                "redact_pii": True,
                "allow_unredacted_partials": False,
                "redact_categories": ["email", "phone"],
            }
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)


def test_admin_org_stt_settings_404(monkeypatch, tmp_path, authnz_schema_ready_sync):
    base_dir = Path(tmp_path) / "test_admin_org_stt_settings_404"
    base_dir.mkdir(parents=True, exist_ok=True)
    db_path = base_dir / "authnz_admin_stt_404.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_MODE", "single_user")

    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    asyncio.run(reset_db_pool())

    mod = import_module("tldw_Server_API.app.main")
    app = getattr(mod, "app")
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

    async def _principal_override(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="single_user",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        request.state.auth = AuthContext(principal=principal, ip=None, user_agent=None, request_id=None)
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/admin/orgs/999/stt/settings")
            assert response.status_code == 404, response.text
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)
