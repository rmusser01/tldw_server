from __future__ import annotations

from typing import Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import moderation as moderation_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal


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


def _build_app(stub) -> FastAPI:
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
    def __init__(self, version: str = "abc123", invalid_lines: Optional[set[str]] = None):
        self.version = version
        self.invalid_lines = invalid_lines or set()
        self.append_called = False
        self.delete_called = False
        self.last_expected: Optional[str] = None
        self.lines: list[str] = []

    def get_blocklist_state(self) -> dict:
        return {"version": self.version, "items": []}

    def append_blocklist_line(self, expected_version: str, line: str):
        self.append_called = True
        self.last_expected = expected_version
        if expected_version and expected_version != self.version:
            return False, {"version": self.version, "conflict": True}
        return True, {"version": self.version, "items": [{"id": 0, "line": line}]}

    def delete_blocklist_index(self, expected_version: str, index: int):
        self.delete_called = True
        self.last_expected = expected_version
        if expected_version and expected_version != self.version:
            return False, {"version": self.version, "conflict": True}
        return True, {"version": self.version, "items": []}

    def lint_blocklist_lines(self, lines: list[str]) -> dict:
        items = []
        invalid_count = 0
        for idx, line in enumerate(lines or []):
            ok = line not in self.invalid_lines
            items.append({"index": idx, "line": line, "ok": ok})
            if not ok:
                invalid_count += 1
        return {"items": items, "valid_count": len(lines) - invalid_count, "invalid_count": invalid_count}

    def set_blocklist_lines(self, lines: list[str]) -> bool:
        self.lines = list(lines or [])
        return True


class _FailingBlocklistModerationService(_StubModerationService):
    def append_blocklist_line(self, expected_version: str, line: str):
        self.append_called = True
        self.last_expected = expected_version
        return False, {"version": self.version, "error": "moderation storage exploded"}

    def delete_blocklist_index(self, expected_version: str, index: int):
        self.delete_called = True
        self.last_expected = expected_version
        return False, {"version": self.version, "error": "moderation storage exploded"}


@pytest.mark.unit
def test_blocklist_managed_returns_quoted_etag():
    stub = _StubModerationService(version="v1")
    app = _build_app(stub)
    with TestClient(app) as client:
        resp = client.get("/api/v1/moderation/blocklist/managed")
    assert resp.status_code == 200
    assert resp.headers.get("ETag") == "\"v1\""


@pytest.mark.unit
def test_blocklist_append_accepts_quoted_if_match():
    stub = _StubModerationService(version="v2")
    app = _build_app(stub)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/blocklist/append",
            json={"line": "secret"},
            headers={"If-Match": "\"v2\""},
        )
    assert resp.status_code == 200
    assert stub.append_called is True
    assert stub.last_expected == "v2"


@pytest.mark.unit
def test_blocklist_append_accepts_weak_if_match():
    stub = _StubModerationService(version="v3")
    app = _build_app(stub)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/blocklist/append",
            json={"line": "secret"},
            headers={"If-Match": "W/\"v3\""},
        )
    assert resp.status_code == 200
    assert stub.append_called is True
    assert stub.last_expected == "v3"


@pytest.mark.unit
def test_blocklist_append_rejects_mismatched_if_match():
    stub = _StubModerationService(version="v4")
    app = _build_app(stub)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/blocklist/append",
            json={"line": "secret"},
            headers={"If-Match": "\"nope\""},
        )
    assert resp.status_code == 412
    assert stub.append_called is False


@pytest.mark.unit
def test_blocklist_append_rejects_invalid_line():
    stub = _StubModerationService(version="v6", invalid_lines={"bad"})
    app = _build_app(stub)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/blocklist/append",
            json={"line": "bad"},
            headers={"If-Match": "\"v6\""},
        )
    assert resp.status_code == 400
    assert stub.append_called is False


@pytest.mark.unit
def test_blocklist_append_sanitizes_internal_failure_detail():
    stub = _FailingBlocklistModerationService(version="v8")
    app = _build_app(stub)
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/moderation/blocklist/append",
            json={"line": "secret"},
            headers={"If-Match": "\"v8\""},
        )
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to append blocklist line"


@pytest.mark.unit
def test_blocklist_update_rejects_invalid_lines():
    stub = _StubModerationService(version="v7", invalid_lines={"bad"})
    app = _build_app(stub)
    with TestClient(app) as client:
        resp = client.put(
            "/api/v1/moderation/blocklist",
            json={"lines": ["good", "bad"]},
        )
    assert resp.status_code == 400


@pytest.mark.unit
def test_blocklist_delete_accepts_quoted_if_match():
    stub = _StubModerationService(version="v5")
    app = _build_app(stub)
    with TestClient(app) as client:
        resp = client.delete(
            "/api/v1/moderation/blocklist/0",
            headers={"If-Match": "\"v5\""},
        )
    assert resp.status_code == 200
    assert stub.delete_called is True
    assert stub.last_expected == "v5"


@pytest.mark.unit
def test_blocklist_delete_sanitizes_internal_failure_detail():
    stub = _FailingBlocklistModerationService(version="v9")
    app = _build_app(stub)
    with TestClient(app) as client:
        resp = client.delete(
            "/api/v1/moderation/blocklist/0",
            headers={"If-Match": "\"v9\""},
        )
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to delete blocklist line"
