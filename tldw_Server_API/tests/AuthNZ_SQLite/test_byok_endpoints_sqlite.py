import base64
import json
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
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
    admin_token = await _issue_access_token(state["admin"])
    user_headers = _auth_headers(user_token)
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
            json={"provider": "openai", "api_key": "sk-user-openai-1234"},
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


def _gateway_endpoint_spec(
    backend_id: str,
    *,
    enabled: bool = True,
    allow_user_api_key: bool = True,
    api_key: str | None = None,
    config_generation: str = "gateway-config-v1",
):
    return SimpleNamespace(
        backend_id=backend_id,
        enabled=enabled,
        allow_user_api_key=allow_user_api_key,
        api_key=api_key,
        config_generation=config_generation,
        base_url="https://gateway.example/v1/",
        models_path="models",
        headers=(),
        discovery_query=(),
        discovery=SimpleNamespace(enabled=True, timeout_seconds=1.0),
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"nested": [{"header": {"X-Api-Key": "attacker"}}]},
        {"nested": {"authorizationHeader": "Bearer attacker"}},
        {"nested": [{"apiHost": "attacker.example"}]},
        {"nested": {"authType": "basic"}},
        {"nested": [{"authSchemeName": "bearer"}]},
        {"nested": {"credentialSource": "metadata"}},
        {"nested": [{"apiKeyAlias": "attacker"}]},
        {"nested": {"bearerAuthority": "metadata"}},
        {"nested": [{"serviceEndpointOptions": {}}]},
        {"nested": {"discoveryURIValue": "https://attacker.example/v1"}},
        {"nested": [{"customURLValue": "https://attacker.example/v1"}]},
    ],
)
def test_gateway_metadata_authority_variants_are_rejected(metadata):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints.user_keys import (
        _validate_gateway_metadata,
    )

    with pytest.raises(HTTPException) as exc_info:
        _validate_gateway_metadata("gateway:voice-lab", metadata)

    assert exc_info.value.status_code == 400
    _validate_gateway_metadata("openrouter", metadata)


@pytest.mark.parametrize("uppercase", [False, True], ids=["lower", "upper"])
@pytest.mark.parametrize(
    "collapsed_key",
    [
        "authorizationheader",
        "authschemename",
        "authtype",
        "apikeyalias",
        "bearerauthority",
        "credentialsource",
        "serviceendpointoptions",
        "discoveryurivalue",
        "customurlvalue",
    ],
)
def test_gateway_metadata_collapsed_authority_aliases_are_rejected(
    collapsed_key,
    uppercase,
):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints.user_keys import (
        _validate_gateway_metadata,
    )

    key = collapsed_key.upper() if uppercase else collapsed_key
    with pytest.raises(HTTPException) as exc_info:
        _validate_gateway_metadata(
            "gateway:voice-lab",
            {"nested": [{"deeper": {key: "attacker"}}]},
        )

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "key",
    [
        "token_count",
        "token-count",
        "header_color",
        "HEADER-COLOR",
        "secret_santa",
        "secret.santa",
        "password_policy",
        "password policy",
        "author",
        "description",
        "model_family",
    ],
)
def test_gateway_metadata_benign_keys_are_allowed(key):
    from tldw_Server_API.app.api.v1.endpoints.user_keys import (
        _validate_gateway_metadata,
    )

    _validate_gateway_metadata(
        "gateway:voice-lab",
        {"nested": [{key: "informational"}]},
    )


@pytest.mark.asyncio
async def test_gateway_byok_endpoints_sqlite(tmp_path, monkeypatch):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    user_id = int(state["user"]["id"])

    from tldw_Server_API.app.api.v1.endpoints import user_keys as user_keys_endpoints
    from tldw_Server_API.app.core.AuthNZ import byok_helpers, byok_runtime
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        decrypt_byok_payload,
        loads_envelope,
    )
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAPIError
    from tldw_Server_API.app.main import app

    specs = {
        "gateway:voice-lab": _gateway_endpoint_spec("gateway:voice-lab"),
        "gateway:admin-fallback": _gateway_endpoint_spec(
            "gateway:admin-fallback",
            api_key="configured-admin-key",
        ),
        "gateway:unverified": _gateway_endpoint_spec("gateway:unverified"),
        "gateway:rejected": _gateway_endpoint_spec("gateway:rejected"),
        "openrouter": _gateway_endpoint_spec("openrouter"),
    }
    monkeypatch.setattr(byok_helpers, "get_byok_gateway_specs", lambda: specs)
    monkeypatch.setattr(
        user_keys_endpoints,
        "get_byok_gateway_spec",
        lambda provider: specs.get(provider),
        raising=False,
    )

    gateway_probe_calls: list[tuple[str, str]] = []
    generic_test_calls: list[dict[str, object]] = []

    async def _fake_probe(*, spec, api_key):
        gateway_probe_calls.append((spec.backend_id, api_key))
        if spec.backend_id == "gateway:rejected" or api_key.startswith("reject-"):
            return "rejected"
        if spec.backend_id == "gateway:unverified" or api_key.startswith("unverified-"):
            return "stored-unverified"
        return "verified"

    monkeypatch.setattr(
        user_keys_endpoints,
        "probe_gateway_credentials",
        _fake_probe,
        raising=False,
    )

    async def _fake_generic_test(
        *,
        provider,
        api_key,
        credential_fields=None,
        model=None,
    ):
        generic_test_calls.append(
            {
                "provider": provider,
                "api_key": api_key,
                "credential_fields": credential_fields,
                "model": model,
            }
        )
        if api_key == "openrouter-upstream-5xx" or model == "openrouter/5xx":
            raise ChatAPIError(
                "OpenRouter unavailable",
                status_code=503,
                provider="openrouter",
            )
        return model or "openrouter/default-model"

    monkeypatch.setattr(
        user_keys_endpoints,
        "test_provider_credentials",
        _fake_generic_test,
    )

    users_db = UsersDB(state["pool"])
    await users_db.initialize()
    second_user = await users_db.create_user(
        username="byok-gateway-user-two",
        email="byok-gateway-user-two@example.com",
        password_hash="hashed-user-two",
        role="user",
        is_active=True,
        is_verified=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    second_user_id = int(second_user["id"])

    user_headers = _auth_headers(await _issue_access_token(state["user"]))
    second_headers = _auth_headers(await _issue_access_token(second_user))
    admin_headers = _auth_headers(await _issue_access_token(state["admin"]))
    repo = AuthnzUserProviderSecretsRepo(state["pool"])

    with TestClient(app) as client:
        for headers in (user_headers, admin_headers):
            response = client.post(
                "/api/v1/users/keys",
                json={
                    "provider": "gateway:voice-lab",
                    "api_key": "authority-key",
                    "credential_fields": {"base_url": "https://attacker.example/v1"},
                },
                headers=headers,
            )
            assert response.status_code == 400

            for forbidden_metadata in (
                {"nested": [{"URL": "https://attacker.example/v1"}]},
                {"nested": {"baseURI": "https://attacker.example/v1"}},
                {"nested": {"endpoint": "https://attacker.example/v1"}},
                {"nested": {"headers": {"Authorization": "attacker"}}},
                {"nested": {"authorization": "Bearer attacker"}},
                {"nested": {"authScheme": "basic"}},
                {"nested": {"apiBaseUrl": "https://attacker.example/v1"}},
                {"nested": {"httpHeaders": {"X-Attacker": "yes"}}},
                {"nested": {"API-BASE_URL": "https://attacker.example/v1"}},
            ):
                response = client.post(
                    "/api/v1/users/keys",
                    json={
                        "provider": "gateway:voice-lab",
                        "api_key": "authority-key",
                        "metadata": forbidden_metadata,
                    },
                    headers=headers,
                )
                assert response.status_code == 400

        benign_metadata = {
            "author": "gateway administrator",
            "description": "voice lab credential",
            "model_family": "speech",
        }
        benign = client.post(
            "/api/v1/users/keys",
            json={
                "provider": "gateway:voice-lab",
                "api_key": "benign-metadata-key",
                "metadata": benign_metadata,
            },
            headers=user_headers,
        )
        assert benign.status_code == 200, benign.text

        admin_fallback_listing = client.get(
            "/api/v1/users/keys",
            headers=user_headers,
        )
        assert admin_fallback_listing.status_code == 200
        admin_fallback_items = {
            item["provider"]: item
            for item in admin_fallback_listing.json()["items"]
        }
        assert admin_fallback_items["gateway:admin-fallback"]["source"] == (
            "server_default"
        )
        assert admin_fallback_items["gateway:admin-fallback"]["has_key"] is False

        rejected = client.post(
            "/api/v1/users/keys",
            json={"provider": "gateway:rejected", "api_key": "reject-key"},
            headers=user_headers,
        )
        assert rejected.status_code == 401
        assert await repo.fetch_secret_for_user(user_id, "gateway:rejected") is None

        unverified = client.post(
            "/api/v1/users/keys",
            json={"provider": "gateway:unverified", "api_key": "unverified-key"},
            headers=user_headers,
        )
        assert unverified.status_code == 200, unverified.text
        assert unverified.json()["status"] == "stored"
        assert unverified.json()["verification_status"] == "stored-unverified"

        created = client.post(
            "/api/v1/users/keys",
            json={"provider": "gateway:voice-lab", "api_key": "first-user-key"},
            headers=user_headers,
        )
        assert created.status_code == 200, created.text
        assert created.json()["status"] == "stored"
        assert created.json()["verification_status"] == "verified"
        assert "credential_scope_token" not in created.text
        assert "first-user-key" not in created.text

        second_created = client.post(
            "/api/v1/users/keys",
            json={"provider": "gateway:voice-lab", "api_key": "second-user-key"},
            headers=second_headers,
        )
        assert second_created.status_code == 200, second_created.text

        first_resolved = await byok_runtime.resolve_gateway_byok_credentials(
            "gateway:voice-lab",
            user_id=user_id,
            gateway_spec=specs["gateway:voice-lab"],
        )
        second_resolved = await byok_runtime.resolve_gateway_byok_credentials(
            "gateway:voice-lab",
            user_id=second_user_id,
            gateway_spec=specs["gateway:voice-lab"],
        )
        assert first_resolved.credential_scope_token
        assert second_resolved.credential_scope_token
        assert first_resolved.credential_scope_token != second_resolved.credential_scope_token

        listing = client.get("/api/v1/users/keys", headers=user_headers)
        assert listing.status_code == 200
        items = {item["provider"]: item for item in listing.json()["items"]}
        assert items["gateway:voice-lab"]["source"] == "user"
        assert items["gateway:voice-lab"]["verification_status"] == "verified"
        assert "credential_scope_token" not in listing.text

        first_scope = first_resolved.credential_scope_token
        rotated = client.post(
            "/api/v1/users/keys",
            json={"provider": "gateway:voice-lab", "api_key": "unverified-rotated-key"},
            headers=user_headers,
        )
        assert rotated.status_code == 200, rotated.text
        assert rotated.json()["verification_status"] == "stored-unverified"
        rotated_resolved = await byok_runtime.resolve_gateway_byok_credentials(
            "gateway:voice-lab",
            user_id=user_id,
            gateway_spec=specs["gateway:voice-lab"],
        )
        assert rotated_resolved.credential_scope_token != first_scope

        probe_count_before_openrouter = len(gateway_probe_calls)
        openrouter_first = client.post(
            "/api/v1/users/keys",
            json={
                "provider": "openrouter",
                "api_key": "openrouter-first",
                "credential_fields": {
                    "org_id": "general-org",
                    "project_id": "general-project",
                },
                "metadata": {
                    "verification_status": "general-verified",
                    "scope": "general-llm",
                },
            },
            headers=user_headers,
        )
        assert openrouter_first.status_code == 200, openrouter_first.text
        assert openrouter_first.json()["verification_status"] is None
        assert generic_test_calls[-1] == {
            "provider": "openrouter",
            "api_key": "openrouter-first",
            "credential_fields": {
                "org_id": "general-org",
                "project_id": "general-project",
            },
            "model": None,
        }
        assert len(gateway_probe_calls) == probe_count_before_openrouter

        openrouter_second = client.post(
            "/api/v1/users/keys",
            json={"provider": "openrouter", "api_key": "unverified-openrouter-rotated"},
            headers=user_headers,
        )
        assert openrouter_second.status_code == 200, openrouter_second.text
        assert generic_test_calls[-1]["provider"] == "openrouter"
        assert len(gateway_probe_calls) == probe_count_before_openrouter

        openrouter_test = client.post(
            "/api/v1/users/keys/test",
            json={"provider": "openrouter", "model": "openrouter/requested-model"},
            headers=user_headers,
        )
        assert openrouter_test.status_code == 200, openrouter_test.text
        assert openrouter_test.json()["model"] == "openrouter/requested-model"
        assert openrouter_test.json()["verification_status"] is None
        assert generic_test_calls[-1]["model"] == "openrouter/requested-model"
        assert len(gateway_probe_calls) == probe_count_before_openrouter

        failed_openrouter_rotation = client.post(
            "/api/v1/users/keys",
            json={"provider": "openrouter", "api_key": "openrouter-upstream-5xx"},
            headers=user_headers,
        )
        assert failed_openrouter_rotation.status_code == 503

        failed_openrouter_test = client.post(
            "/api/v1/users/keys/test",
            json={"provider": "openrouter", "model": "openrouter/5xx"},
            headers=user_headers,
        )
        assert failed_openrouter_test.status_code == 503
        assert len(gateway_probe_calls) == probe_count_before_openrouter

        openrouter_row = await repo.fetch_secret_for_user(user_id, "openrouter")
        assert openrouter_row is not None
        openrouter_payload = decrypt_byok_payload(loads_envelope(openrouter_row["encrypted_blob"]))
        assert openrouter_payload["api_key"] == "unverified-openrouter-rotated"
        assert openrouter_payload["credential_fields"] == {
            "org_id": "general-org",
            "project_id": "general-project",
        }
        openrouter_metadata = json.loads(openrouter_row["metadata"])
        assert openrouter_metadata["verification_status"] == "general-verified"
        assert openrouter_metadata["scope"] == "general-llm"
        assert "tts_gateway_verification_status" not in openrouter_metadata

        del specs["gateway:voice-lab"]
        orphan_listing = client.get("/api/v1/users/keys", headers=user_headers)
        orphan_items = {
            item["provider"]: item for item in orphan_listing.json()["items"]
        }
        assert orphan_items["gateway:voice-lab"]["source"] == "disabled"

        orphan_replace = client.post(
            "/api/v1/users/keys",
            json={"provider": "gateway:voice-lab", "api_key": "must-not-store"},
            headers=user_headers,
        )
        assert orphan_replace.status_code == 403
        orphan_test = client.post(
            "/api/v1/users/keys/test",
            json={"provider": "gateway:voice-lab"},
            headers=user_headers,
        )
        assert orphan_test.status_code == 403
        orphan_delete = client.delete(
            "/api/v1/users/keys/gateway:voice-lab",
            headers=user_headers,
        )
        assert orphan_delete.status_code == 204


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
async def test_shared_keys_scoped_requires_manager_sqlite(tmp_path, monkeypatch):
    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    pool = state["pool"]
    org_id = int(state["org"]["id"])
    team_id = int(state["team"]["id"])

    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import add_org_member, add_team_member

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
