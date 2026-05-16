"""Integration tests for public prototype private-link exchange."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import sharing as sharing_endpoints
from tldw_Server_API.app.api.v1.endpoints.sharing import router
from tldw_Server_API.app.core.AuthNZ.migrations import (
    migration_001_create_users_table,
    migration_077_create_sharing_tables,
    migration_086_create_prototype_workspace_tables,
    migration_087_expand_share_tokens_resource_type_for_prototypes,
)
from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
    PrototypeWorkspacesRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import SharedWorkspaceRepo
from tldw_Server_API.app.core.Prototype_Workspaces.access import PROTOTYPE_SHARED_ACTOR_COOKIE
from tldw_Server_API.app.core.Sharing.share_token_service import ShareTokenService

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_STATES_FIXTURE = _REPO_ROOT / "apps/tldw-frontend/e2e/fixtures/prototype-workspaces/contract-states.json"


def _assert_prototype_error(
    response,
    *,
    category: str,
    frontend_state: str,
    retryable: bool,
) -> None:
    detail = response.json()["detail"]
    assert detail["category"] == category
    assert detail["frontend_state"] == frontend_state
    assert detail["retryable"] is retryable
    assert isinstance(detail["message"], str)
    assert detail["message"]


def _assert_openapi_error_response(openapi: dict, path: str, method: str, status_code: str) -> None:
    schema = openapi["paths"][path][method]["responses"][status_code]["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("/PrototypeErrorResponse")


def _assert_openapi_error_or_validation_response(openapi: dict, path: str, method: str, status_code: str) -> None:
    schema = openapi["paths"][path][method]["responses"][status_code]["content"]["application/json"]["schema"]
    refs = {entry["$ref"].rsplit("/", maxsplit=1)[-1] for entry in schema["anyOf"]}
    assert {"PrototypeErrorResponse", "HTTPValidationError"} <= refs


class _FakePool:
    """Minimal DatabasePool stand-in backed by an in-memory SQLite connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, params: tuple = ()) -> None:
        self._conn.execute(sql, params)
        self._conn.commit()

    async def fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        cur = self._conn.execute(sql, params)
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row, strict=True))

    async def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self._conn.execute(sql, params)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in rows]


class _NoOpAuditService:
    async def log(self, *args, **kwargs) -> None:
        return None


@pytest.fixture
def exchange_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migration_001_create_users_table(conn)
    migration_077_create_sharing_tables(conn)
    migration_087_expand_share_tokens_resource_type_for_prototypes(conn)
    migration_086_create_prototype_workspace_tables(conn)
    conn.execute(
        "INSERT INTO users (id, username, email, password_hash) VALUES (1, 'owner', 'owner@test.com', 'hash')"
    )
    conn.execute(
        "INSERT INTO users (id, username, email, password_hash) VALUES (2, 'other', 'other@test.com', 'hash')"
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def fake_pool(exchange_db):
    return _FakePool(exchange_db)


@pytest.fixture
def sharing_repo(fake_pool):
    return SharedWorkspaceRepo(db_pool=fake_pool)


@pytest.fixture
def prototype_repo(fake_pool):
    return PrototypeWorkspacesRepo(db_pool=fake_pool)


@pytest.fixture
def token_service(sharing_repo):
    return ShareTokenService(sharing_repo)


@pytest.fixture
def prototype_workspace(prototype_repo):
    workspace = asyncio.run(
        prototype_repo.create_workspace(
            owner_user_id=1,
            title="Prototype Demo",
            creation_source="prompt",
            share_policy={"allow_browser_session_resume": True},
            runtime_policy={"external_collaborator_profile": "locked_collab"},
        )
    )
    return workspace


@pytest.fixture
def prototype_share_token(token_service, prototype_workspace):
    token = asyncio.run(
        token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=prototype_workspace["id"],
            owner_user_id=1,
            password="demo-pass",
        )
    )
    return token["raw_token"]


@pytest.fixture
def protected_single_use_prototype_share(token_service, prototype_workspace):
    return asyncio.run(
        token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=prototype_workspace["id"],
            owner_user_id=1,
            password="demo-pass",
            max_uses=1,
        )
    )


@pytest.fixture
def unprotected_prototype_share_token(token_service, prototype_workspace):
    token = asyncio.run(
        token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=prototype_workspace["id"],
            owner_user_id=1,
        )
    )
    return token["raw_token"]


@pytest.fixture
def revoked_prototype_share_token(token_service, prototype_workspace):
    token = asyncio.run(
        token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=prototype_workspace["id"],
            owner_user_id=1,
        )
    )
    asyncio.run(token_service.revoke_token(token["id"]))
    return token["raw_token"]


@pytest.fixture
def non_prototype_share_token(token_service):
    token = asyncio.run(
        token_service.generate_token(
            resource_type="workspace",
            resource_id="ws_regular_workspace",
            owner_user_id=1,
        )
    )
    return token["raw_token"]


@pytest.fixture
def second_link_same_workspace_token(token_service, prototype_workspace):
    token = asyncio.run(
        token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=prototype_workspace["id"],
            owner_user_id=1,
        )
    )
    return token["raw_token"]


@pytest.fixture
def second_prototype_workspace(prototype_repo):
    workspace = asyncio.run(
        prototype_repo.create_workspace(
            owner_user_id=1,
            title="Prototype Demo 2",
            creation_source="prompt",
            share_policy={"allow_browser_session_resume": True},
            runtime_policy={"external_collaborator_profile": "locked_collab"},
        )
    )
    return workspace


@pytest.fixture
def second_workspace_prototype_share_token(token_service, second_prototype_workspace):
    token = asyncio.run(
        token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=second_prototype_workspace["id"],
            owner_user_id=1,
        )
    )
    return token["raw_token"]


@pytest.fixture
def single_use_prototype_share(token_service, prototype_workspace):
    return asyncio.run(
        token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=prototype_workspace["id"],
            owner_user_id=1,
            max_uses=1,
        )
    )


@pytest.fixture
def resume_disabled_workspace(prototype_repo):
    workspace = asyncio.run(
        prototype_repo.create_workspace(
            owner_user_id=1,
            title="No Resume Prototype",
            creation_source="prompt",
            share_policy={"allow_browser_session_resume": False},
            runtime_policy={"external_collaborator_profile": "locked_collab"},
        )
    )
    return workspace


@pytest.fixture
def resume_disabled_prototype_share_token(token_service, resume_disabled_workspace):
    token = asyncio.run(
        token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=resume_disabled_workspace["id"],
            owner_user_id=1,
        )
    )
    return token["raw_token"]


@pytest.fixture
def archived_prototype_share_token(exchange_db, token_service, prototype_workspace):
    token = asyncio.run(
        token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=prototype_workspace["id"],
            owner_user_id=1,
        )
    )
    exchange_db.execute(
        "UPDATE prototype_workspaces SET archived_at = CURRENT_TIMESTAMP WHERE id = ?",
        (prototype_workspace["id"],),
    )
    exchange_db.commit()
    return token["raw_token"]


@pytest.fixture
def missing_workspace_prototype_share_token(token_service):
    token = asyncio.run(
        token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id="pws_missing_workspace",
            owner_user_id=1,
        )
    )
    return token["raw_token"]


@pytest.fixture
def mismatched_owner_prototype_share_token(token_service, prototype_workspace):
    token = asyncio.run(
        token_service.generate_token(
            resource_type="prototype_workspace",
            resource_id=prototype_workspace["id"],
            owner_user_id=2,
        )
    )
    return token["raw_token"]


@pytest.fixture
def test_app(monkeypatch, sharing_repo, prototype_repo, token_service):
    monkeypatch.setattr(sharing_endpoints, "_get_repo", lambda: sharing_repo)
    monkeypatch.setattr(sharing_endpoints, "_get_token_service", lambda: token_service)
    monkeypatch.setattr(
        sharing_endpoints,
        "_get_prototype_repo",
        lambda: prototype_repo,
        raising=False,
    )
    monkeypatch.setattr(sharing_endpoints, "_get_audit_service", lambda: _NoOpAuditService())
    monkeypatch.setattr(sharing_endpoints, "_check_public_rate_limit", lambda _request: None)

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


def test_access_service_requires_configured_stable_signing_secret(
    prototype_repo,
    monkeypatch,
):
    from tldw_Server_API.app.core.Prototype_Workspaces import access

    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    monkeypatch.setattr(
        access,
        "get_settings",
        lambda: SimpleNamespace(JWT_SECRET_KEY=None, SINGLE_USER_API_KEY=None),
    )

    with pytest.raises(RuntimeError, match="stable signing secret"):
        access.PrototypeAccessService(prototype_repo)


def test_public_prototype_exchange_openapi_declares_error_contract(test_app):
    openapi = test_app.openapi()

    _assert_openapi_error_response(
        openapi,
        "/api/v1/sharing/public/{token}/prototype-session",
        "post",
        "403",
    )
    _assert_openapi_error_response(
        openapi,
        "/api/v1/sharing/public/{token}/prototype-session",
        "post",
        "404",
    )
    _assert_openapi_error_or_validation_response(
        openapi,
        "/api/v1/sharing/public/{token}/prototype-session",
        "post",
        "422",
    )
    assert "429" in openapi["paths"]["/api/v1/sharing/public/{token}/prototype-session"]["post"]["responses"]


def test_contract_states_fixture_uses_structured_error_details():
    fixture = json.loads(_CONTRACT_STATES_FIXTURE.read_text(encoding="utf-8"))

    assert fixture["riskGate"] == "Risk Gate 4 frozen"
    for state in fixture["states"]:
        detail = state["mockResponse"]["detail"]
        assert detail["category"] == state["stableErrorCategory"]
        assert isinstance(detail["message"], str)
        assert detail["message"]
        assert isinstance(detail["frontend_state"], str)
        assert detail["retryable"] == state["retryable"]


def test_public_prototype_exchange_creates_shared_actor(client, prototype_share_token):
    resp = client.post(
        f"/api/v1/sharing/public/{prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM", "password": "demo-pass"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["actor_type"] == "external_collaborator"
    assert body["shared_actor_id"].startswith("psa_")
    assert body["session_token"]


def test_public_prototype_exchange_revoked_link_returns_404(client, revoked_prototype_share_token):
    resp = client.post(
        f"/api/v1/sharing/public/{revoked_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )

    assert resp.status_code == 404
    _assert_prototype_error(
        resp,
        category="invalid_or_unavailable_link",
        frontend_state="link_unavailable",
        retryable=False,
    )


def test_public_prototype_exchange_bad_password_returns_403(client, prototype_share_token):
    resp = client.post(
        f"/api/v1/sharing/public/{prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM", "password": "wrong-pass"},
    )

    assert resp.status_code == 403
    _assert_prototype_error(
        resp,
        category="invalid_password",
        frontend_state="password_rejected",
        retryable=True,
    )


def test_public_prototype_exchange_missing_password_returns_403(client, prototype_share_token):
    resp = client.post(
        f"/api/v1/sharing/public/{prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )

    assert resp.status_code == 403
    _assert_prototype_error(
        resp,
        category="password_required",
        frontend_state="password_required",
        retryable=True,
    )


def test_public_prototype_exchange_missing_display_name_returns_422(
    client,
    unprotected_prototype_share_token,
):
    resp = client.post(
        f"/api/v1/sharing/public/{unprotected_prototype_share_token}/prototype-session",
        json={},
    )

    assert resp.status_code == 422


def test_public_prototype_exchange_releases_claim_on_unexpected_error(
    client,
    monkeypatch,
    sharing_repo,
    single_use_prototype_share,
):
    class _ExplodingAccessService:
        async def can_resume_external_collaborator(self, **kwargs):
            return False

        async def exchange_external_collaborator(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        sharing_endpoints,
        "_get_prototype_access_service",
        lambda: _ExplodingAccessService(),
    )

    with pytest.raises(RuntimeError, match="boom"):
        client.post(
            f"/api/v1/sharing/public/{single_use_prototype_share['raw_token']}/prototype-session",
            json={"display_name": "Acme PM"},
        )

    token_row = asyncio.run(sharing_repo.get_token(single_use_prototype_share["id"]))
    assert token_row is not None
    assert token_row["use_count"] == 0


def test_public_prototype_exchange_retains_claim_on_post_exchange_failure(
    client,
    monkeypatch,
    sharing_repo,
    single_use_prototype_share,
):
    class _ExplodingAuditService:
        async def log(self, *args, **kwargs):
            raise RuntimeError("audit boom")

    monkeypatch.setattr(
        sharing_endpoints,
        "_get_audit_service",
        lambda: _ExplodingAuditService(),
    )

    with pytest.raises(RuntimeError, match="audit boom"):
        client.post(
            f"/api/v1/sharing/public/{single_use_prototype_share['raw_token']}/prototype-session",
            json={"display_name": "Acme PM"},
        )

    token_row = asyncio.run(sharing_repo.get_token(single_use_prototype_share["id"]))
    assert token_row is not None
    assert token_row["use_count"] == 1


def test_public_prototype_exchange_non_prototype_token_returns_422(client, non_prototype_share_token):
    resp = client.post(
        f"/api/v1/sharing/public/{non_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )

    assert resp.status_code == 422


def test_public_preview_reports_prototype_resource_type(
    client,
    unprotected_prototype_share_token,
):
    resp = client.get(f"/api/v1/sharing/public/{unprotected_prototype_share_token}")

    assert resp.status_code == 200
    assert resp.json()["resource_type"] == "prototype_workspace"


def test_public_prototype_exchange_reuses_same_actor_for_same_browser_session(
    client,
    unprotected_prototype_share_token,
):
    first = client.post(
        f"/api/v1/sharing/public/{unprotected_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert first.status_code == 200
    first_body = first.json()

    second = client.post(
        f"/api/v1/sharing/public/{unprotected_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM (renamed)"},
    )
    assert second.status_code == 200
    second_body = second.json()

    assert second_body["shared_actor_id"] == first_body["shared_actor_id"]


def test_public_prototype_exchange_updates_activity_timestamp_on_resume(
    client,
    prototype_repo,
    unprotected_prototype_share_token,
):
    first = client.post(
        f"/api/v1/sharing/public/{unprotected_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert first.status_code == 200
    actor_id = first.json()["shared_actor_id"]

    before = asyncio.run(prototype_repo.get_shared_actor(actor_id))
    time.sleep(0.02)
    second = client.post(
        f"/api/v1/sharing/public/{unprotected_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert second.status_code == 200
    after = asyncio.run(prototype_repo.get_shared_actor(actor_id))

    assert after is not None
    assert before is not None
    assert after["last_activity_at"] != before["last_activity_at"]


def test_public_prototype_exchange_rotates_resume_cookie_on_reuse(
    client,
    prototype_repo,
    unprotected_prototype_share_token,
):
    first = client.post(
        f"/api/v1/sharing/public/{unprotected_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert first.status_code == 200
    actor_id = first.json()["shared_actor_id"]
    first_cookie = first.cookies.get(PROTOTYPE_SHARED_ACTOR_COOKIE)
    assert first_cookie
    before = asyncio.run(prototype_repo.get_shared_actor(actor_id))

    second = client.post(
        f"/api/v1/sharing/public/{unprotected_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert second.status_code == 200
    second_cookie = second.cookies.get(PROTOTYPE_SHARED_ACTOR_COOKIE)
    after = asyncio.run(prototype_repo.get_shared_actor(actor_id))

    assert second_cookie
    assert second_cookie != first_cookie
    assert before is not None
    assert after is not None
    assert after["session_binding_id"] != before["session_binding_id"]


def test_public_prototype_exchange_allows_same_browser_resume_after_max_uses(
    client,
    test_app,
    sharing_repo,
    single_use_prototype_share,
):
    raw_token = single_use_prototype_share["raw_token"]

    first = client.post(
        f"/api/v1/sharing/public/{raw_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert first.status_code == 200
    first_actor_id = first.json()["shared_actor_id"]

    second = client.post(
        f"/api/v1/sharing/public/{raw_token}/prototype-session",
        json={},
    )
    assert second.status_code == 200
    assert second.json()["shared_actor_id"] == first_actor_id

    token_row = asyncio.run(sharing_repo.get_token(single_use_prototype_share["id"]))
    assert token_row is not None
    assert token_row["use_count"] == 1

    fresh_client = TestClient(test_app)
    third = fresh_client.post(
        f"/api/v1/sharing/public/{raw_token}/prototype-session",
        json={"display_name": "Another PM"},
    )
    assert third.status_code == 404


def test_public_prototype_exchange_password_link_resumes_without_replaying_password(
    client,
    test_app,
    sharing_repo,
    protected_single_use_prototype_share,
):
    raw_token = protected_single_use_prototype_share["raw_token"]

    first = client.post(
        f"/api/v1/sharing/public/{raw_token}/prototype-session",
        json={"display_name": "Acme PM", "password": "demo-pass"},
    )
    assert first.status_code == 200
    first_actor_id = first.json()["shared_actor_id"]

    second = client.post(
        f"/api/v1/sharing/public/{raw_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert second.status_code == 200
    assert second.json()["shared_actor_id"] == first_actor_id

    token_row = asyncio.run(sharing_repo.get_token(protected_single_use_prototype_share["id"]))
    assert token_row is not None
    assert token_row["use_count"] == 1

    fresh_client = TestClient(test_app)
    third = fresh_client.post(
        f"/api/v1/sharing/public/{raw_token}/prototype-session",
        json={"display_name": "Another PM"},
    )
    assert third.status_code == 403


def test_public_prototype_exchange_rejects_forged_resume_cookie(
    client,
    test_app,
    unprotected_prototype_share_token,
):
    first = client.post(
        f"/api/v1/sharing/public/{unprotected_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert first.status_code == 200
    first_actor_id = first.json()["shared_actor_id"]
    valid_cookie = first.cookies.get(PROTOTYPE_SHARED_ACTOR_COOKIE)
    assert valid_cookie

    forged_client = TestClient(test_app)
    forged_cookie = f"{valid_cookie}tampered"
    forged_client.cookies.set(PROTOTYPE_SHARED_ACTOR_COOKIE, forged_cookie)
    second = forged_client.post(
        f"/api/v1/sharing/public/{unprotected_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert second.status_code == 200
    second_actor_id = second.json()["shared_actor_id"]

    assert second_actor_id != first_actor_id


def test_public_prototype_exchange_does_not_resume_across_share_links(
    client,
    unprotected_prototype_share_token,
    second_link_same_workspace_token,
):
    first = client.post(
        f"/api/v1/sharing/public/{unprotected_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert first.status_code == 200
    first_actor_id = first.json()["shared_actor_id"]

    second = client.post(
        f"/api/v1/sharing/public/{second_link_same_workspace_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert second.status_code == 200
    second_actor_id = second.json()["shared_actor_id"]

    assert second_actor_id != first_actor_id


def test_public_prototype_exchange_does_not_resume_across_workspaces(
    client,
    unprotected_prototype_share_token,
    second_workspace_prototype_share_token,
):
    first = client.post(
        f"/api/v1/sharing/public/{unprotected_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert first.status_code == 200
    first_actor_id = first.json()["shared_actor_id"]

    second = client.post(
        f"/api/v1/sharing/public/{second_workspace_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert second.status_code == 200
    second_actor_id = second.json()["shared_actor_id"]

    assert second_actor_id != first_actor_id


def test_public_prototype_exchange_does_not_resume_when_policy_disabled(
    client,
    resume_disabled_prototype_share_token,
):
    first = client.post(
        f"/api/v1/sharing/public/{resume_disabled_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert first.status_code == 200
    first_actor_id = first.json()["shared_actor_id"]

    second = client.post(
        f"/api/v1/sharing/public/{resume_disabled_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )
    assert second.status_code == 200
    second_actor_id = second.json()["shared_actor_id"]

    assert second_actor_id != first_actor_id


def test_public_prototype_exchange_archived_workspace_returns_403(
    client,
    archived_prototype_share_token,
):
    resp = client.post(
        f"/api/v1/sharing/public/{archived_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )

    assert resp.status_code == 403
    _assert_prototype_error(
        resp,
        category="workspace_unavailable",
        frontend_state="workspace_unavailable",
        retryable=False,
    )


def test_public_prototype_exchange_missing_workspace_returns_404(
    client,
    missing_workspace_prototype_share_token,
):
    resp = client.post(
        f"/api/v1/sharing/public/{missing_workspace_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )

    assert resp.status_code == 404
    _assert_prototype_error(
        resp,
        category="invalid_or_unavailable_link",
        frontend_state="link_unavailable",
        retryable=False,
    )


def test_public_prototype_exchange_rejects_token_owner_mismatch(
    client,
    mismatched_owner_prototype_share_token,
):
    resp = client.post(
        f"/api/v1/sharing/public/{mismatched_owner_prototype_share_token}/prototype-session",
        json={"display_name": "Acme PM"},
    )

    assert resp.status_code == 404
    _assert_prototype_error(
        resp,
        category="invalid_or_unavailable_link",
        frontend_state="link_unavailable",
        retryable=False,
    )
