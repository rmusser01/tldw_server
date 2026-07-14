import asyncio
import base64
import json
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


async def _issue_access_token(
    user_row: dict,
    *,
    active_org_id: int | None = None,
    active_team_id: int | None = None,
) -> str:
    from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_memberships_for_user

    user_id = int(user_row["id"])
    memberships = await list_memberships_for_user(user_id)
    team_ids = sorted({m.get("team_id") for m in memberships if m.get("team_id") is not None})
    org_ids = sorted({m.get("org_id") for m in memberships if m.get("org_id") is not None})

    claims = {"team_ids": team_ids, "org_ids": org_ids}
    if active_org_id is not None:
        claims["active_org_id"] = int(active_org_id)
    if active_team_id is not None:
        claims["active_team_id"] = int(active_team_id)

    jwt_service = get_jwt_service()
    return jwt_service.create_access_token(
        user_id=user_id,
        username=str(user_row.get("username") or user_id),
        role=str(user_row.get("role") or "user"),
        additional_claims=claims,
    )


async def _setup_byok_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("BYOK_ALLOWED_BASE_URL_PROVIDERS", "openai")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890")
    monkeypatch.setenv("DEFAULT_MODEL_OPENAI", "gpt-4o-mini")
    monkeypatch.setenv("DEFAULT_MODEL_ANTHROPIC", "claude-3-haiku")
    monkeypatch.setenv("DEFAULT_MODEL_COHERE", "command-r")
    monkeypatch.setenv("DEFAULT_MODEL_GROQ", "llama-3.1-8b-instant")
    monkeypatch.setenv("DEFAULT_MODEL_OPENROUTER", "openrouter/test-model")

    if "pyotp" not in sys.modules:
        pyotp_stub = types.ModuleType("pyotp")

        class _StubTOTP:
            def __init__(self, *_args, **_kwargs):
                pass

            def now(self) -> str:
                return "000000"

            def verify(self, *_args, **_kwargs) -> bool:
                return True

            def provisioning_uri(self, *_args, **_kwargs) -> str:
                return "otpauth://totp/test"

        pyotp_stub.TOTP = _StubTOTP
        pyotp_stub.random_base32 = lambda *_args, **_kwargs: "A" * 32
        monkeypatch.setitem(sys.modules, "pyotp", pyotp_stub)

    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB, reset_users_db
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
        create_organization,
        create_team,
        add_org_member,
        add_team_member,
    )

    reset_settings()
    reset_jwt_service()
    await reset_db_pool()
    await reset_users_db()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    users_db = UsersDB(pool)
    await users_db.initialize()

    admin = await users_db.create_user(
        username="byok-admin",
        email="byok-admin@example.com",
        password_hash="hashed-admin",
        role="admin",
        is_active=True,
        is_verified=True,
        is_superuser=True,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user = await users_db.create_user(
        username="byok-user",
        email="byok-user@example.com",
        password_hash="hashed-user",
        role="user",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    users_repo = AuthnzUsersRepo(db_pool=pool)
    await users_repo.assign_role_if_missing(user_id=int(admin["id"]), role_name="admin")
    await users_repo.assign_role_if_missing(user_id=int(user["id"]), role_name="user")

    org = await create_organization(name="BYOK Org", owner_user_id=int(admin["id"]))
    team = await create_team(org_id=int(org["id"]), name="BYOK Team")

    await add_org_member(org_id=int(org["id"]), user_id=int(user["id"]), role="lead")
    await add_team_member(team_id=int(team["id"]), user_id=int(user["id"]), role="lead")

    return {
        "pool": pool,
        "admin": admin,
        "user": user,
        "org": org,
        "team": team,
    }


@pytest.mark.asyncio
async def test_byok_endpoints_sqlite(tmp_path, monkeypatch):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    org_id = int(state["org"]["id"])
    team_id = int(state["team"]["id"])

    from tldw_Server_API.app.main import app
    user_token = await _issue_access_token(
        state["user"],
        active_org_id=org_id,
        active_team_id=team_id,
    )
    org_only_token = await _issue_access_token(
        state["user"],
        active_org_id=org_id,
    )
    admin_token = await _issue_access_token(state["admin"])
    user_headers = _auth_headers(user_token)
    org_only_headers = _auth_headers(org_only_token)
    admin_headers = _auth_headers(admin_token)

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/users/keys",
            json={"provider": "unknown-provider", "api_key": "sk-unknown-0000"},
            headers=user_headers,
        )
        assert r.status_code == 403

        r = client.post(
            "/api/v1/users/keys",
            json={
                "provider": "openai",
                "api_key": "sk-invalid-0000",
                "credential_fields": {"unsupported": "value"},
            },
            headers=user_headers,
        )
        assert r.status_code == 400

        r = client.post(
            "/api/v1/users/keys",
            json={"provider": "oai", "api_key": "sk-user-openai-1234"},
            headers=user_headers,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["provider"] == "openai"
        assert body["status"] == "stored"
        assert body["key_hint"] == "1234"
        assert "api_key" not in body

        r = client.post(
            "/api/v1/users/keys",
            json={"provider": "cohere", "api_key": "sk-user-cohere-5678"},
            headers=user_headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["key_hint"] == "5678"

        r = client.post(
            "/api/v1/users/keys",
            json={"provider": "openai", "api_key": "invalid-test-key"},
            headers=user_headers,
        )
        assert r.status_code == 401

        r = client.post(
            "/api/v1/users/keys/test",
            json={"provider": "openai"},
            headers=user_headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "valid"

        listing = client.get("/api/v1/users/keys", headers=user_headers)
        assert listing.status_code == 200
        items = {item["provider"]: item for item in listing.json()["items"]}
        assert items["openai"]["source"] == "user"
        assert items["openai"]["has_key"] is True
        assert items["openai"]["auth_source"] == "api_key"
        assert items["cohere"]["source"] == "user"

        r = client.delete("/api/v1/users/keys/cohere", headers=user_headers)
        assert r.status_code == 204

        listing = client.get("/api/v1/users/keys", headers=user_headers)
        items = {item["provider"]: item for item in listing.json()["items"]}
        assert items["cohere"]["source"] != "user"

        r = client.post(
            f"/api/v1/orgs/{org_id}/keys/shared",
            json={"provider": "anthropic", "api_key": "sk-org-9999"},
            headers=user_headers,
        )
        assert r.status_code == 200, r.text
        org_resp = r.json()
        assert org_resp["scope_type"] == "org"
        assert org_resp["scope_id"] == org_id
        assert org_resp["key_hint"] == "9999"

        r = client.post(
            f"/api/v1/teams/{team_id}/keys/shared",
            json={"provider": "openrouter", "api_key": "sk-team-4321"},
            headers=user_headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["key_hint"] == "4321"

        r = client.post(
            f"/api/v1/orgs/{org_id}/keys/shared/test",
            json={"provider": "anthropic"},
            headers=user_headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "valid"

        r = client.post(
            f"/api/v1/teams/{team_id}/keys/shared/test",
            json={"provider": "openrouter"},
            headers=user_headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "valid"

        r = client.get(f"/api/v1/orgs/{org_id}/keys/shared", headers=user_headers)
        assert r.status_code == 200
        org_items = {item["provider"]: item for item in r.json()["items"]}
        assert "anthropic" in org_items

        r = client.get(f"/api/v1/teams/{team_id}/keys/shared", headers=user_headers)
        assert r.status_code == 200
        team_items = {item["provider"]: item for item in r.json()["items"]}
        assert "openrouter" in team_items

        listing = client.get("/api/v1/users/keys", headers=user_headers)
        items = {item["provider"]: item for item in listing.json()["items"]}
        assert items["anthropic"]["source"] == "org"
        assert items["anthropic"]["has_key"] is False
        assert items["openrouter"]["source"] == "team"

        org_only_listing = client.get(
            "/api/v1/users/keys",
            headers=org_only_headers,
        )
        assert org_only_listing.status_code == 200
        org_only_items = {item["provider"]: item for item in org_only_listing.json()["items"]}
        assert org_only_items["anthropic"]["source"] == "org"
        assert org_only_items["openrouter"]["source"] != "team"

        admin_list = client.get(f"/api/v1/admin/keys/users/{user_id}", headers=admin_headers)
        assert admin_list.status_code == 200
        admin_items = {item["provider"]: item for item in admin_list.json()["items"]}
        assert "openai" in admin_items
        assert admin_items["openai"]["allowed"] is True

        r = client.post(
            "/api/v1/admin/keys/shared/test",
            json={
                "scope_type": "org",
                "scope_id": org_id,
                "provider": "anthropic",
            },
            headers=admin_headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "valid"

        r = client.post(
            "/api/v1/admin/keys/shared",
            json={
                "scope_type": "org",
                "scope_id": org_id,
                "provider": "groq",
                "api_key": "sk-admin-3333",
            },
            headers=admin_headers,
        )
        assert r.status_code == 200, r.text

        r = client.get(
            "/api/v1/admin/keys/shared",
            params={"scope_type": "org", "scope_id": org_id},
            headers=admin_headers,
        )
        assert r.status_code == 200
        shared_items = {item["provider"]: item for item in r.json()["items"]}
        assert "groq" in shared_items

        r = client.delete(f"/api/v1/admin/keys/shared/org/{org_id}/groq", headers=admin_headers)
        assert r.status_code == 204

        r = client.delete(f"/api/v1/admin/keys/users/{user_id}/openai", headers=admin_headers)
        assert r.status_code == 204

        r = client.delete(f"/api/v1/orgs/{org_id}/keys/shared/anthropic", headers=user_headers)
        assert r.status_code == 204

        r = client.delete(f"/api/v1/teams/{team_id}/keys/shared/openrouter", headers=user_headers)
        assert r.status_code == 204

        listing = client.get("/api/v1/users/keys", headers=user_headers)
        items = {item["provider"]: item for item in listing.json()["items"]}
        assert items["openai"]["source"] != "user"


@pytest.mark.asyncio
async def test_openai_oauth_endpoints_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "1")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "oauth-client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "oauth-client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_AUTH_URL", "https://oauth.example.com/authorize")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example.com/token")
    monkeypatch.setenv(
        "OPENAI_OAUTH_REDIRECT_URI",
        "https://app.example.com/api/v1/users/keys/openai/oauth/callback",
    )
    monkeypatch.setenv("OPENAI_OAUTH_SCOPES", "openid profile api")
    monkeypatch.setenv("OPENAI_OAUTH_ALLOWED_RETURN_PATH_PREFIXES", "/settings,/profile")

    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    org_id = int(state["org"]["id"])
    team_id = int(state["team"]["id"])

    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.api.v1.endpoints import user_keys as user_keys_endpoints

    user_token = await _issue_access_token(
        state["user"],
        active_org_id=org_id,
        active_team_id=team_id,
    )
    user_headers = _auth_headers(user_token)

    class _FakeOAuthTokenResponse:
        def __init__(self, *, status_code: int, payload: dict):
            self.status_code = status_code
            self._payload = dict(payload)
            self.text = json.dumps(payload)

        def json(self):
            return dict(self._payload)

        async def aclose(self):
            return None

    token_call_log: list[dict] = []
    metric_counter_calls: list[dict] = []
    metric_histogram_calls: list[dict] = []
    audit_calls: list[dict] = []

    class _FakeAuditService:
        async def log_event(self, **kwargs):
            audit_calls.append(dict(kwargs))
            return "evt-test"

    async def _fake_get_audit_service_for_user_id_optional(_user_id):
        return _FakeAuditService()

    def _fake_increment_counter(metric_name: str, value: float = 1, labels: dict | None = None):
        metric_counter_calls.append(
            {
                "name": metric_name,
                "value": value,
                "labels": dict(labels or {}),
            }
        )

    def _fake_observe_histogram(metric_name: str, value: float, labels: dict | None = None):
        metric_histogram_calls.append(
            {
                "name": metric_name,
                "value": value,
                "labels": dict(labels or {}),
            }
        )

    async def _fake_http_afetch(**kwargs):
        token_call_log.append(dict(kwargs))
        data = kwargs.get("data") or {}
        grant_type = data.get("grant_type")
        if grant_type == "authorization_code":
            return _FakeOAuthTokenResponse(
                status_code=200,
                payload={
                    "access_token": "oauth-access-token-111",
                    "refresh_token": "oauth-refresh-token-111",
                    "token_type": "Bearer",
                    "scope": "api",
                    "expires_in": 3600,
                    "sub": "user-sub-123",
                },
            )
        if grant_type == "refresh_token":
            return _FakeOAuthTokenResponse(
                status_code=200,
                payload={
                    "access_token": "oauth-access-token-222",
                    "refresh_token": "oauth-refresh-token-222",
                    "token_type": "Bearer",
                    "scope": "api refreshed",
                    "expires_in": 1800,
                },
            )
        return _FakeOAuthTokenResponse(
            status_code=400,
            payload={"error": "unsupported_grant_type"},
        )

    monkeypatch.setattr(user_keys_endpoints, "_http_afetch", _fake_http_afetch)
    monkeypatch.setattr(user_keys_endpoints, "increment_counter", _fake_increment_counter)
    monkeypatch.setattr(user_keys_endpoints, "observe_histogram", _fake_observe_histogram)
    monkeypatch.setattr(
        user_keys_endpoints,
        "get_or_create_audit_service_for_user_id_optional",
        _fake_get_audit_service_for_user_id_optional,
    )

    with TestClient(app) as client:
        r = client.post(
            "/api/v1/users/keys",
            json={"provider": "openai", "api_key": "sk-user-openai-4321"},
            headers=user_headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["key_hint"] == "4321"

        listing = client.get("/api/v1/users/keys", headers=user_headers)
        assert listing.status_code == 200
        openai_item = {item["provider"]: item for item in listing.json()["items"]}["openai"]
        assert openai_item["auth_source"] == "api_key"

        r = client.post(
            "/api/v1/users/keys/openai/oauth/authorize",
            json={
                "credential_fields": {"org_id": "org_abc"},
                "return_path": "/settings/models",
            },
            headers=user_headers,
        )
        assert r.status_code == 200, r.text
        auth_body = r.json()
        assert auth_body["provider"] == "openai"
        parsed = urlparse(auth_body["auth_url"])
        parsed_qs = parse_qs(parsed.query)
        assert parsed_qs.get("client_id") == ["oauth-client-id"]
        assert parsed_qs.get("redirect_uri") == [
            "https://app.example.com/api/v1/users/keys/openai/oauth/callback"
        ]
        state_value = parsed_qs["state"][0]

        r = client.get(
            "/api/v1/users/keys/openai/oauth/callback",
            params={"code": "auth-code-123", "state": state_value},
        )
        assert r.status_code == 200, r.text
        callback_body = r.json()
        assert callback_body["auth_source"] == "oauth"
        assert callback_body["key_hint"] == "oauth"
        assert callback_body["expires_at"] is not None

        r = client.get("/api/v1/users/keys/openai/oauth/status", headers=user_headers)
        assert r.status_code == 200, r.text
        status_body = r.json()
        assert status_body["connected"] is True
        assert status_body["auth_source"] == "oauth"
        assert status_body["scope"] == "api"
        assert status_body["expires_at"] is not None

        listing = client.get("/api/v1/users/keys", headers=user_headers)
        openai_item = {item["provider"]: item for item in listing.json()["items"]}["openai"]
        assert openai_item["auth_source"] == "oauth"

        r = client.post(
            "/api/v1/users/keys/openai/source",
            json={"auth_source": "api_key"},
            headers=user_headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["auth_source"] == "api_key"

        r = client.post(
            "/api/v1/users/keys/openai/source",
            json={"auth_source": "oauth"},
            headers=user_headers,
        )
        assert r.status_code == 200, r.text
        assert r.json()["auth_source"] == "oauth"

        r = client.post("/api/v1/users/keys/openai/oauth/refresh", headers=user_headers)
        assert r.status_code == 200, r.text
        refresh_body = r.json()
        assert refresh_body["status"] == "refreshed"
        assert refresh_body["expires_at"] is not None

        r = client.get(
            "/api/v1/users/keys/openai/oauth/callback",
            params={"code": "bad-code", "state": "not-a-valid-state"},
        )
        assert r.status_code == 403

        r = client.delete("/api/v1/users/keys/openai/oauth", headers=user_headers)
        assert r.status_code == 204

        r = client.post("/api/v1/users/keys/openai/oauth/refresh", headers=user_headers)
        assert r.status_code == 404

        r = client.get("/api/v1/users/keys/openai/oauth/status", headers=user_headers)
        assert r.status_code == 200
        status_after_disconnect = r.json()
        assert status_after_disconnect["connected"] is False
        assert status_after_disconnect["auth_source"] == "api_key"

        listing = client.get("/api/v1/users/keys", headers=user_headers)
        openai_item = {item["provider"]: item for item in listing.json()["items"]}["openai"]
        assert openai_item["auth_source"] == "api_key"

        r = client.post(
            "/api/v1/users/keys/openai/source",
            json={"auth_source": "oauth"},
            headers=user_headers,
        )
        assert r.status_code == 409

        r = client.post(
            "/api/v1/users/keys/test",
            json={"provider": "openai"},
            headers=user_headers,
        )
        assert r.status_code == 200, r.text

    assert len(token_call_log) == 2
    assert token_call_log[0]["data"]["grant_type"] == "authorization_code"
    assert token_call_log[1]["data"]["grant_type"] == "refresh_token"
    metric_names = [entry["name"] for entry in metric_counter_calls]
    assert "byok_oauth_authorize_started_total" in metric_names
    assert "byok_oauth_callback_success_total" in metric_names
    assert "byok_oauth_callback_failure_total" in metric_names
    assert "byok_oauth_refresh_total" in metric_names
    refresh_outcomes = {
        entry["labels"].get("outcome")
        for entry in metric_counter_calls
        if entry["name"] == "byok_oauth_refresh_total"
    }
    assert "success" in refresh_outcomes
    assert "failure" in refresh_outcomes
    assert any(entry["name"] == "byok_oauth_refresh_latency_ms" for entry in metric_histogram_calls)

    audit_actions = [entry.get("action") for entry in audit_calls]
    assert "provider_oauth_authorize_started" in audit_actions
    assert "provider_oauth_connected" in audit_actions
    assert "provider_oauth_refreshed" in audit_actions
    assert "provider_oauth_disconnected" in audit_actions
    assert "provider_oauth_refresh_failed" in audit_actions


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_openai_oauth_refresh_disconnect_race_preserves_revocation_sqlite(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "1")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "oauth-client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "oauth-client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_AUTH_URL", "https://oauth.example.com/authorize")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example.com/token")
    monkeypatch.setenv(
        "OPENAI_OAUTH_REDIRECT_URI",
        "https://app.example.com/api/v1/users/keys/openai/oauth/callback",
    )
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])

    from fastapi import HTTPException
    from starlette.requests import Request

    from tldw_Server_API.app.api.v1.endpoints import user_keys as user_keys_endpoints
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        decrypt_byok_payload,
        dumps_envelope,
        encrypt_byok_payload,
        loads_envelope,
    )

    repo = AuthnzUserProviderSecretsRepo(state["pool"])
    now = datetime.now(timezone.utc)
    original_payload = {
        "credential_version": 2,
        "active_auth_source": "oauth",
        "credentials": {
            "oauth": {
                "access_token": "stale-access-token",
                "refresh_token": "refresh-token-race",
                "expires_at": (now + timedelta(seconds=30)).isoformat(),
            }
        },
    }
    await repo.upsert_secret(
        user_id=user_id,
        provider="openai",
        encrypted_blob=dumps_envelope(encrypt_byok_payload(original_payload)),
        key_hint="oauth",
        metadata=None,
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )

    token_request_started = asyncio.Event()
    release_token_response = asyncio.Event()
    issued_token = "issued-token-must-not-be-stored"

    async def _token_exchange(**_kwargs):
        token_request_started.set()
        await release_token_response.wait()
        return {"access_token": issued_token, "expires_in": 3600}

    async def _get_user_repo():
        return repo

    async def _ignore_audit_event(**_kwargs):
        return None

    monkeypatch.setattr(user_keys_endpoints, "_get_user_repo", _get_user_repo)
    monkeypatch.setattr(user_keys_endpoints, "_openai_oauth_token_exchange", _token_exchange)
    monkeypatch.setattr(
        user_keys_endpoints,
        "_emit_openai_oauth_audit_event",
        _ignore_audit_event,
    )

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/users/keys/openai/oauth/refresh",
            "headers": [],
        }
    )
    refresh_task = asyncio.create_task(
        user_keys_endpoints.refresh_openai_oauth(
            request,
            AuthPrincipal(kind="user", user_id=user_id),
        )
    )
    token_started_waiter = asyncio.create_task(token_request_started.wait())
    completed, _pending = await asyncio.wait(
        {refresh_task, token_started_waiter},
        return_when=asyncio.FIRST_COMPLETED,
    )
    assert token_started_waiter in completed
    assert await repo.delete_secret(user_id, "openai", revoked_by=user_id)
    release_token_response.set()

    with pytest.raises(HTTPException) as exc_info:
        await refresh_task

    assert exc_info.value.status_code == 404
    stored_row = await repo.fetch_secret_for_user(user_id, "openai", include_revoked=True)
    assert stored_row is not None
    assert stored_row["revoked_at"] is not None
    stored_payload = decrypt_byok_payload(loads_envelope(stored_row["encrypted_blob"]))
    assert stored_payload == original_payload
    assert issued_token not in json.dumps(stored_payload)


@pytest.mark.asyncio
async def test_legacy_user_alias_row_can_be_touched_and_revoked_sqlite(
    tmp_path,
    monkeypatch,
):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])
    pool = state["pool"]

    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )

    created_at = datetime.now(timezone.utc)
    await pool.execute(
        """
        INSERT INTO user_provider_secrets (
            user_id, provider, encrypted_blob, key_hint, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            "oai",
            "legacy-encrypted-blob",
            "legacy",
            created_at.isoformat(),
            created_at.isoformat(),
        ),
    )
    repo = AuthnzUserProviderSecretsRepo(pool)

    legacy_row = await repo.fetch_secret_for_user(user_id, "openai")
    assert legacy_row is not None
    assert legacy_row["provider"] == "oai"

    used_at = created_at + timedelta(seconds=1)
    await repo.touch_last_used(user_id, "openai", used_at)
    touched_row = await pool.fetchone(
        "SELECT last_used_at FROM user_provider_secrets WHERE user_id = ? AND provider = ?",
        (user_id, "oai"),
    )
    assert touched_row is not None
    assert touched_row["last_used_at"] == used_at.isoformat()

    assert await repo.delete_secret(user_id, "openai", revoked_by=user_id)
    assert await repo.fetch_secret_for_user(user_id, "openai") is None
    revoked_row = await repo.fetch_secret_for_user(
        user_id,
        "openai",
        include_revoked=True,
    )
    assert revoked_row is not None
    assert revoked_row["provider"] == "oai"
    assert revoked_row["revoked_at"] is not None


@pytest.mark.asyncio
async def test_shared_keys_scoped_requires_manager_sqlite(tmp_path, monkeypatch):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    pool = state["pool"]
    org_id = int(state["org"]["id"])
    team_id = int(state["team"]["id"])

    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import add_org_member, add_team_member
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo

    users_db = UsersDB(pool)
    await users_db.initialize()

    member = await users_db.create_user(
        username="byok-member",
        email="byok-member@example.com",
        password_hash="hashed-member",
        role="user",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    member_id = int(member["id"])
    await AuthnzUsersRepo(db_pool=pool).assign_role_if_missing(user_id=member_id, role_name="user")
    await add_org_member(org_id=org_id, user_id=member_id, role="member")
    await add_team_member(team_id=team_id, user_id=member_id, role="member")

    from tldw_Server_API.app.main import app

    member_token = await _issue_access_token(
        member,
        active_org_id=org_id,
        active_team_id=team_id,
    )
    member_headers = _auth_headers(member_token)

    with TestClient(app) as client:
        r = client.post(
            f"/api/v1/orgs/{org_id}/keys/shared",
            json={"provider": "openai", "api_key": "sk-org-0000"},
            headers=member_headers,
        )
        assert r.status_code == 403

        r = client.post(
            f"/api/v1/teams/{team_id}/keys/shared",
            json={"provider": "openai", "api_key": "sk-team-0000"},
            headers=member_headers,
        )
        assert r.status_code == 403
