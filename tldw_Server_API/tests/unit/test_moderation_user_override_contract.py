from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import moderation as moderation_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append((message, args, kwargs))


def _make_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
        is_admin=True,
        org_ids=[1],
        team_ids=[],
    )


def _build_app(stub: _StubModerationService) -> FastAPI:
    app = FastAPI()
    app.include_router(moderation_mod.router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = _make_principal()
        request.state.auth = AuthContext(
            principal=principal,
            ip=None,
            user_agent=None,
            request_id=None,
        )
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    moderation_mod.get_moderation_service = lambda: stub  # type: ignore[assignment]
    return app


class _StubModerationService:
    def __init__(self, overrides: dict[str, dict] | None = None, set_result: dict | None = None):
        self.overrides = dict(overrides or {})
        self.last_set: tuple[str, dict] | None = None
        self.set_result = dict(set_result) if set_result is not None else None

    def list_user_overrides(self) -> dict[str, dict]:
        return self.overrides

    def set_user_override(self, user_id: str, override: dict) -> dict:
        self.last_set = (user_id, dict(override))
        if self.set_result is not None:
            return dict(self.set_result)
        self.overrides[str(user_id)] = dict(override)
        return {"ok": True, "persisted": True, **dict(override)}

    def delete_user_override(self, user_id: str) -> dict:
        if str(user_id) not in self.overrides:
            return {"ok": False, "persisted": False, "error": "not found"}
        self.overrides.pop(str(user_id), None)
        return {"ok": True, "persisted": True}


@pytest.mark.unit
def test_get_user_override_returns_rules_payload():
    stub = _StubModerationService(
        {
            "alice": {
                "enabled": True,
                "rules": [
                    {
                        "id": "r1",
                        "pattern": "bad",
                        "is_regex": False,
                        "action": "block",
                        "phase": "both",
                    }
                ],
            }
        }
    )
    app = _build_app(stub)

    with TestClient(app) as client:
        resp = client.get("/api/v1/moderation/users/alice")

    assert resp.status_code == 200
    body = resp.json()
    assert body["exists"] is True
    assert body["override"]["enabled"] is True
    assert body["override"]["rules"][0]["action"] == "block"


@pytest.mark.unit
def test_put_user_override_persists_rules_payload():
    stub = _StubModerationService()
    app = _build_app(stub)

    payload = {
        "enabled": True,
        "rules": [
            {
                "id": "n1",
                "pattern": "heads up",
                "is_regex": False,
                "action": "warn",
                "phase": "both",
            }
        ],
    }

    with TestClient(app) as client:
        resp = client.put("/api/v1/moderation/users/alice", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["persisted"] is True
    assert body["rules"][0]["pattern"] == "heads up"
    assert body["rules"][0]["action"] == "warn"
    assert stub.last_set is not None
    assert stub.last_set[0] == "alice"
    assert stub.last_set[1]["rules"][0]["phase"] == "both"


@pytest.mark.unit
def test_put_user_override_returns_400_for_validation_error():
    stub = _StubModerationService(
        set_result={
            "ok": False,
            "persisted": False,
            "error": "dangerous regex in rule: r1",
            "error_type": "validation",
        }
    )
    app = _build_app(stub)

    payload = {
        "enabled": True,
        "rules": [
            {
                "id": "r1",
                "pattern": "(",
                "is_regex": True,
                "action": "block",
                "phase": "both",
            }
        ],
    }

    with TestClient(app) as client:
        resp = client.put("/api/v1/moderation/users/alice", json=payload)

    assert resp.status_code == 400
    assert resp.json().get("detail") == "dangerous regex in rule: r1"


@pytest.mark.unit
def test_put_user_override_returns_500_for_persistence_error():
    stub = _StubModerationService(
        set_result={
            "ok": False,
            "persisted": False,
            "error": "disk full",
            "error_type": "persistence",
        }
    )
    app = _build_app(stub)

    payload = {
        "enabled": True,
        "rules": [
            {
                "id": "r1",
                "pattern": "hello",
                "is_regex": False,
                "action": "warn",
                "phase": "both",
            }
        ],
    }

    with TestClient(app) as client:
        resp = client.put("/api/v1/moderation/users/alice", json=payload)

    assert resp.status_code == 500
    assert resp.json().get("detail") == "Failed to persist override"


@pytest.mark.unit
def test_put_user_override_persistence_error_log_is_sanitized(monkeypatch):
    stub = _StubModerationService(
        set_result={
            "ok": False,
            "persisted": False,
            "error": "moderation persistence exploded at /private/moderation-overrides.json",
            "error_type": "persistence",
        }
    )
    logger_stub = _LoggerStub()
    monkeypatch.setattr(moderation_mod, "logger", logger_stub)
    app = _build_app(stub)

    payload = {
        "enabled": True,
        "rules": [
            {
                "id": "r1",
                "pattern": "hello",
                "is_regex": False,
                "action": "warn",
                "phase": "both",
            }
        ],
    }

    with TestClient(app) as client:
        resp = client.put("/api/v1/moderation/users/private-user", json=payload)

    assert resp.status_code == 500
    assert resp.json().get("detail") == "Failed to persist override"
    assert logger_stub.errors == [("Moderation override persist failed", (), {})]
    rendered = " ".join([logger_stub.errors[0][0], *(str(arg) for arg in logger_stub.errors[0][1])])
    assert "private-user" not in rendered
    assert "exploded" not in rendered
    assert "/private/moderation-overrides.json" not in rendered


@pytest.mark.unit
def test_put_user_override_returns_400_for_legacy_dangerous_error_message():
    stub = _StubModerationService(
        set_result={
            "ok": False,
            "persisted": False,
            "error": "dangerous regex in rule: r1",
        }
    )
    app = _build_app(stub)

    payload = {
        "enabled": True,
        "rules": [
            {
                "id": "r1",
                "pattern": "(",
                "is_regex": True,
                "action": "block",
                "phase": "both",
            }
        ],
    }

    with TestClient(app) as client:
        resp = client.put("/api/v1/moderation/users/alice", json=payload)

    assert resp.status_code == 400
    assert resp.json().get("detail") == "dangerous regex in rule: r1"


@pytest.mark.unit
def test_get_user_override_missing_returns_empty_payload():
    stub = _StubModerationService()
    app = _build_app(stub)

    with TestClient(app) as client:
        resp = client.get("/api/v1/moderation/users/missing")

    assert resp.status_code == 200
    assert resp.json() == {"exists": False, "override": {}}
