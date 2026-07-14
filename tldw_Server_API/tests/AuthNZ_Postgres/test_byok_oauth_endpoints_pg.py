import base64
import json
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


def _auth_headers(token: str) -> dict[str, str]:
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

    claims: dict[str, object] = {"team_ids": team_ids, "org_ids": org_ids}
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_oauth_endpoints_postgres(test_db_pool, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("BYOK_ALLOWED_BASE_URL_PROVIDERS", "openai")
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
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings as reset_auth_settings
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
        add_org_member,
        add_team_member,
        create_organization,
        create_team,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        ProviderCredentialAliasConflictError,
    )
    from tldw_Server_API.app.main import app
    from tldw_Server_API.app.api.v1.endpoints import user_keys as user_keys_endpoints

    reset_auth_settings()
    reset_jwt_service()
    app_settings["CSRF_ENABLED"] = False

    name_suffix = uuid.uuid4().hex[:12]
    admin_username = f"byok-pg-admin-{name_suffix}"
    user_username = f"byok-pg-user-{name_suffix}"

    await test_db_pool.execute(
        """
        INSERT INTO users (
            uuid,
            username,
            email,
            password_hash,
            role,
            is_active,
            is_verified,
            is_superuser,
            storage_quota_mb
        ) VALUES ($1, $2, $3, $4, $5, TRUE, TRUE, TRUE, 5120)
        """,
        str(uuid.uuid4()),
        admin_username,
        f"{admin_username}@example.com",
        "hashed-admin",
        "admin",
    )
    await test_db_pool.execute(
        """
        INSERT INTO users (
            uuid,
            username,
            email,
            password_hash,
            role,
            is_active,
            is_verified,
            is_superuser,
            storage_quota_mb
        ) VALUES ($1, $2, $3, $4, $5, TRUE, TRUE, FALSE, 5120)
        """,
        str(uuid.uuid4()),
        user_username,
        f"{user_username}@example.com",
        "hashed-user",
        "user",
    )

    admin_row = await test_db_pool.fetchrow(
        "SELECT id, username, role FROM users WHERE username = $1",
        admin_username,
    )
    user_row = await test_db_pool.fetchrow(
        "SELECT id, username, role FROM users WHERE username = $1",
        user_username,
    )
    assert admin_row is not None
    assert user_row is not None

    admin_id = int(admin_row["id"])
    user_id = int(user_row["id"])

    org = await create_organization(name=f"BYOK Org {name_suffix}", owner_user_id=admin_id)
    team = await create_team(org_id=int(org["id"]), name=f"BYOK Team {name_suffix}")
    await add_org_member(org_id=int(org["id"]), user_id=user_id, role="lead")
    await add_team_member(team_id=int(team["id"]), user_id=user_id, role="lead")

    shared_repo = AuthnzOrgProviderSecretsRepo(test_db_pool)
    await shared_repo.ensure_tables()
    base_scope_id = 1_000_000 + (uuid.uuid4().int % 1_000_000)
    now = datetime.now(timezone.utc)
    written = await shared_repo.upsert_secret(
        scope_type="team",
        scope_id=base_scope_id,
        provider="oai",
        encrypted_blob="pg-canonical-write",
        key_hint="write",
        metadata=None,
        updated_at=now,
    )
    assert written["provider"] == "openai"

    legacy_scope_id = base_scope_id + 1
    await test_db_pool.execute(
        """
        INSERT INTO org_provider_secrets (
            scope_type, scope_id, provider, encrypted_blob, key_hint, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $6)
        """,
        "org",
        legacy_scope_id,
        "oai",
        "pg-legacy",
        "legacy",
        now,
    )
    legacy = await shared_repo.fetch_secret("org", legacy_scope_id, "openai")
    assert legacy is not None
    assert legacy["provider"] == "oai"
    await shared_repo.touch_last_used("org", legacy_scope_id, "openai", now)
    assert await shared_repo.delete_secret("org", legacy_scope_id, "openai")

    revoked_scope_id = base_scope_id + 2
    await test_db_pool.execute(
        """
        INSERT INTO org_provider_secrets (
            scope_type, scope_id, provider, encrypted_blob, key_hint,
            revoked_at, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $6, $6)
        """,
        "team",
        revoked_scope_id,
        "openai",
        "pg-revoked-canonical",
        "canonical",
        now,
    )
    await test_db_pool.execute(
        """
        INSERT INTO org_provider_secrets (
            scope_type, scope_id, provider, encrypted_blob, key_hint, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $6)
        """,
        "team",
        revoked_scope_id,
        "oai",
        "pg-active-alias",
        "alias",
        now,
    )
    assert await shared_repo.fetch_secret("team", revoked_scope_id, "openai") is None
    revoked_rows = await shared_repo.list_secrets(
        scope_type="team",
        scope_id=revoked_scope_id,
        include_revoked=True,
    )
    assert len(revoked_rows) == 1
    assert revoked_rows[0]["provider"] == "openai"

    conflict_scope_id = base_scope_id + 3
    for provider in ("custom-openai", "openai-compatible"):
        await test_db_pool.execute(
            """
            INSERT INTO org_provider_secrets (
                scope_type, scope_id, provider, encrypted_blob, key_hint, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $6)
            """,
            "org",
            conflict_scope_id,
            provider,
            provider,
            provider,
            now,
        )
    with pytest.raises(ProviderCredentialAliasConflictError):
        await shared_repo.list_secrets(scope_type="org", scope_id=conflict_scope_id)

    user_token = await _issue_access_token(
        dict(user_row),
        active_org_id=int(org["id"]),
        active_team_id=int(team["id"]),
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
                    "access_token": "oauth-access-token-pg-111",
                    "refresh_token": "oauth-refresh-token-pg-111",
                    "token_type": "Bearer",
                    "scope": "api",
                    "expires_in": 3600,
                    "sub": "user-sub-pg-123",
                },
            )
        if grant_type == "refresh_token":
            return _FakeOAuthTokenResponse(
                status_code=200,
                payload={
                    "access_token": "oauth-access-token-pg-222",
                    "refresh_token": "oauth-refresh-token-pg-222",
                    "token_type": "Bearer",
                    "scope": "api refreshed",
                    "expires_in": 1800,
                },
            )
        return _FakeOAuthTokenResponse(
            status_code=400,
            payload={"error": "unsupported_grant_type"},
        )

    async def _fake_test_provider_credentials(**_kwargs):
        return "gpt-4o-mini"

    monkeypatch.setattr(user_keys_endpoints, "_http_afetch", _fake_http_afetch)
    monkeypatch.setattr(user_keys_endpoints, "test_provider_credentials", _fake_test_provider_credentials)
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
            json={"provider": "openai", "api_key": "sk-user-openai-pg-4321"},
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
            params={"code": "auth-code-pg-123", "state": state_value},
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

    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )

    repo = AuthnzUserProviderSecretsRepo(test_db_pool)
    active_row = await repo.fetch_secret_for_user(user_id, "openai")
    assert active_row is not None
    expected_blob = str(active_row["encrypted_blob"])
    touched_at = datetime.now(timezone.utc)
    await repo.touch_last_used(user_id, "openai", touched_at)

    cas_at = touched_at + timedelta(seconds=1)
    replacement_blob = f"{expected_blob}-cas"
    assert await repo.update_secret_if_active_and_unchanged(
        user_id=user_id,
        provider="oai",
        encrypted_blob=replacement_blob,
        expected_encrypted_blob=expected_blob,
        key_hint="cas",
        metadata=None,
        updated_at=cas_at,
        updated_by=user_id,
    )
    assert not await repo.update_secret_if_active_and_unchanged(
        user_id=user_id,
        provider="openai",
        encrypted_blob="stale-write",
        expected_encrypted_blob=expected_blob,
        key_hint="cas",
        metadata=None,
        updated_at=cas_at + timedelta(seconds=1),
        updated_by=user_id,
    )
    assert await repo.delete_secret(user_id, "openai", revoked_by=user_id)
    assert not await repo.update_secret_if_active_and_unchanged(
        user_id=user_id,
        provider="openai",
        encrypted_blob="revoked-write",
        expected_encrypted_blob=replacement_blob,
        key_hint="cas",
        metadata=None,
        updated_at=cas_at + timedelta(seconds=2),
        updated_by=user_id,
    )
