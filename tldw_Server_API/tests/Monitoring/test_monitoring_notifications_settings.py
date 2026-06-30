import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import monitoring as monitoring_mod
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _single_user_test_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", "monitoring-test-key")
    reset_settings()
    yield
    reset_settings()


def _make_admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


def _build_app_with_overrides(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(monitoring_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        return _make_admin_principal()

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal

    class _FakeNotificationService:
        def __init__(self) -> None:
            self.updated = False

        def get_settings(self) -> dict:
            return {
                "enabled": False,
                "min_severity": "critical",
                "file": "Databases/monitoring_notifications.log",
                "webhook_url": "",
                "email_to": "",
                "smtp_host": "",
                "smtp_port": 587,
                "smtp_starttls": True,
                "smtp_user": "",
                "email_from": "",
            }

        def update_settings(self, **_kwargs) -> dict:
            self.updated = True
            return self.get_settings()

        def is_file_path_allowed(self, _path: str) -> bool:
            return True

    fake_service = _FakeNotificationService()
    monkeypatch.setattr(monitoring_mod, "get_notification_service", lambda: fake_service)
    return app, fake_service


def test_notification_settings_rejects_empty_file(monkeypatch: pytest.MonkeyPatch) -> None:
    app, fake_service = _build_app_with_overrides(monkeypatch)

    with TestClient(app) as client:
        resp = client.put("/api/v1/monitoring/notifications/settings", json={"file": ""})

    assert resp.status_code == 400
    detail = resp.json().get("detail", "")
    assert "file" in detail
    assert fake_service.updated is False


def test_notification_settings_rejects_disallowed_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    app, fake_service = _build_app_with_overrides(monkeypatch)
    fake_service.is_file_path_allowed = lambda _path: False  # type: ignore[method-assign]

    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/monitoring/notifications/settings",
            json={"file": "/etc/passwd"},
        )

    assert resp.status_code == 400
    detail = resp.json().get("detail", "")
    assert "Notification file path" in detail
    assert fake_service.updated is False
