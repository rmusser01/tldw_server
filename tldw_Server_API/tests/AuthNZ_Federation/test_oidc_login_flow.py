from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.datastructures import URL

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    check_auth_rate_limit,
    get_auth_principal,
)
from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings

pytestmark = pytest.mark.integration
_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


def test_build_federation_redirect_uri_uses_public_base_without_request_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "x" * 32)
    monkeypatch.setenv("PUBLIC_WEB_BASE_URL", "https://app.example.com")
    reset_settings()

    from tldw_Server_API.app.api.v1.endpoints.auth import _build_federation_redirect_uri

    def _url_for(name: str, *, provider_slug: str) -> URL:
        assert name == "federation_callback"
        return URL(f"http://attacker.example/api/v1/auth/federation/{provider_slug}/callback")

    request = SimpleNamespace(url_for=_url_for)

    assert _build_federation_redirect_uri(request, "corp") == (
        "https://app.example.com/api/v1/auth/federation/corp/callback"
    )


@pytest.fixture
def federation_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    db_path = tmp_path / "oidc_login_flow.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "x" * 32)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AUTH_FEDERATION_ENABLED", "true")
    monkeypatch.setattr(
        Settings,
        "enterprise_federation_supported",
        property(lambda self: True),
        raising=False,
    )

    asyncio.run(reset_db_pool())
    reset_settings()

    from tldw_Server_API.app.main import app as fastapi_app

    async def _override_principal(request=None) -> AuthPrincipal:
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test-admin",
            token_type="test",
            jti=None,
            roles=["admin"],
            permissions=["claims.admin"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        return principal

    fastapi_app.dependency_overrides[get_auth_principal] = _override_principal

    try:
        with TestClient(fastapi_app) as client:
            yield client
    finally:
        fastapi_app.dependency_overrides.pop(get_auth_principal, None)
        asyncio.run(reset_db_pool())
        reset_settings()


def test_federation_login_redirects_to_provider_authorization_url(
    federation_client: TestClient,
) -> None:
    create_response = federation_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": True,
            "display_name": "Corp SSO",
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "claim_mapping": {
                "subject": "sub",
                "email": "email",
                "username": "preferred_username",
            },
            "provisioning_policy": {
                "jit_create": True,
                "allow_email_account_linking": True,
            },
        },
    )
    assert create_response.status_code == 200, create_response.text

    response = federation_client.get(
        "/api/v1/auth/federation/corp/login",
        follow_redirects=False,
    )
    assert response.status_code == 307, response.text

    location = response.headers["location"]
    parsed = urlparse(location)
    query = parse_qs(parsed.query)

    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://issuer.example.com/oauth2/authorize"
    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["client-123"]
    assert query["scope"] == ["openid email profile"]
    assert query["code_challenge_method"] == ["S256"]
    assert len(query["code_challenge"][0]) >= 32
    assert len(query["state"][0]) >= 20
    assert query["redirect_uri"] == ["http://testserver/api/v1/auth/federation/corp/callback"]


def test_federation_login_uses_public_web_base_url_for_redirect_uri(
    federation_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PUBLIC_WEB_BASE_URL", "https://app.example.com")
    reset_settings()

    create_response = federation_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp-public-base",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": True,
            "display_name": "Corp SSO",
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "claim_mapping": {
                "subject": "sub",
                "email": "email",
                "username": "preferred_username",
            },
            "provisioning_policy": {
                "jit_create": True,
                "allow_email_account_linking": True,
            },
        },
    )
    assert create_response.status_code == 200, create_response.text

    response = federation_client.get(
        "/api/v1/auth/federation/corp-public-base/login",
        headers={"host": "attacker.example"},
        follow_redirects=False,
    )
    assert response.status_code == 307, response.text

    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["redirect_uri"] == [
        "https://app.example.com/api/v1/auth/federation/corp-public-base/callback"
    ]


def test_federation_callback_supports_org_scoped_provider_resolution(
    federation_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.federation.oidc_service import OIDCFederationService
    from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import create_organization
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    async def _seed_org_scoped_user() -> tuple[int, int]:
        pool = await get_db_pool()
        users_db = UsersDB(pool)
        await users_db.initialize()
        password_hash = PasswordService(get_settings()).hash_password("OrgScoped!7Qx")
        user = await users_db.create_user(
            username="org-alice",
            email="org-alice@example.com",
            password_hash=password_hash,
            is_active=True,
            is_verified=True,
        )
        org = await create_organization(name="Org Scoped SSO Org", owner_user_id=None)
        return int(user["id"]), int(org["id"])

    existing_user_id, org_id = asyncio.run(_seed_org_scoped_user())

    create_response = federation_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp-org",
            "provider_type": "oidc",
            "owner_scope_type": "org",
            "owner_scope_id": org_id,
            "enabled": True,
            "display_name": "Corp Org SSO",
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "claim_mapping": {
                "subject": "sub",
                "email": "email",
                "username": "preferred_username",
            },
            "provisioning_policy": {
                "jit_create": True,
                "allow_email_account_linking": True,
            },
        },
    )
    assert create_response.status_code == 200, create_response.text
    created_provider = create_response.json()

    login_response = federation_client.get(
        f"/api/v1/auth/federation/corp-org/login?org_id={org_id}",
        follow_redirects=False,
    )
    assert login_response.status_code == 307, login_response.text
    login_query = parse_qs(urlparse(login_response.headers["location"]).query)
    state = login_query["state"][0]

    async def _fake_exchange(
        self,
        *,
        provider: dict,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        nonce: str | None = None,
    ) -> dict:
        assert int(provider["id"]) == int(created_provider["id"])
        assert code == "code-org-123"
        assert redirect_uri == "http://testserver/api/v1/auth/federation/corp-org/callback"
        assert code_verifier
        return {
            "sub": "external-org-subject-123",
            "email": "org-alice@example.com",
            "preferred_username": "org.alice.sso",
            "iss": "https://issuer.example.com",
        }

    monkeypatch.setattr(
        OIDCFederationService,
        "exchange_authorization_code",
        _fake_exchange,
        raising=False,
    )

    callback_response = federation_client.get(
        f"/api/v1/auth/federation/corp-org/callback?state={state}&code=code-org-123",
    )
    assert callback_response.status_code == 200, callback_response.text
    body = callback_response.json()
    payload = JWTService(get_settings()).verify_token(body["access_token"], token_type="access")
    assert int(payload["sub"]) == existing_user_id


def test_federation_callback_uses_auth_rate_limit_dependency(
    federation_client: TestClient,
) -> None:
    async def _block_rate_limit() -> None:
        raise HTTPException(status_code=429, detail="Too many requests")

    federation_client.app.dependency_overrides[check_auth_rate_limit] = _block_rate_limit
    try:
        response = federation_client.get(
            "/api/v1/auth/federation/corp/callback?state=test-state&code=test-code",
        )
    finally:
        federation_client.app.dependency_overrides.pop(check_auth_rate_limit, None)

    assert response.status_code == 429, response.text
    assert response.json()["detail"] == "Too many requests"


def test_federation_login_uses_auth_rate_limit_dependency(
    federation_client: TestClient,
) -> None:
    async def _block_rate_limit() -> None:
        raise HTTPException(status_code=429, detail="Too many requests")

    federation_client.app.dependency_overrides[check_auth_rate_limit] = _block_rate_limit
    try:
        response = federation_client.get(
            "/api/v1/auth/federation/corp/login",
            follow_redirects=False,
        )
    finally:
        federation_client.app.dependency_overrides.pop(check_auth_rate_limit, None)

    assert response.status_code == 429, response.text
    assert response.json()["detail"] == "Too many requests"


def test_federation_login_respects_auth_resource_governor_limits(
    federation_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    create_response = federation_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp-rg-login",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": True,
            "display_name": "Corp SSO",
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "claim_mapping": {
                "subject": "sub",
            },
            "provisioning_policy": {
                "jit_create": True,
            },
        },
    )
    assert create_response.status_code == 200, create_response.text

    async def _deny_rg(*args, **kwargs):  # noqa: ANN002, ANN003
        return False, 11

    monkeypatch.setattr(auth, "_reserve_auth_rg_requests", _deny_rg)

    response = federation_client.get(
        "/api/v1/auth/federation/corp-rg-login/login",
        follow_redirects=False,
    )

    assert response.status_code == 429, response.text
    assert response.headers["Retry-After"] == "11"


def test_federation_callback_respects_auth_resource_governor_limits(
    federation_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.api.v1.endpoints.auth as auth

    create_response = federation_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp-rg-callback",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": True,
            "display_name": "Corp SSO",
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "claim_mapping": {
                "subject": "sub",
            },
            "provisioning_policy": {
                "jit_create": True,
            },
        },
    )
    assert create_response.status_code == 200, create_response.text

    login_response = federation_client.get(
        "/api/v1/auth/federation/corp-rg-callback/login",
        follow_redirects=False,
    )
    assert login_response.status_code == 307, login_response.text
    state = parse_qs(urlparse(login_response.headers["location"]).query)["state"][0]

    async def _deny_rg(*args, **kwargs):  # noqa: ANN002, ANN003
        return False, 13

    monkeypatch.setattr(auth, "_reserve_auth_rg_requests", _deny_rg)

    response = federation_client.get(
        f"/api/v1/auth/federation/corp-rg-callback/callback?state={state}&code=test-code",
    )

    assert response.status_code == 429, response.text
    assert response.headers["Retry-After"] == "13"


def test_federation_callback_links_existing_user_and_returns_tokens(
    federation_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.federation.oidc_service import OIDCFederationService
    from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    from tldw_Server_API.app.core.AuthNZ.repos.federated_identity_repo import FederatedIdentityRepo
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    async def _seed_existing_user() -> int:
        pool = await get_db_pool()
        users_db = UsersDB(pool)
        await users_db.initialize()
        password_hash = PasswordService(get_settings()).hash_password("FedLink!7Qx")
        user = await users_db.create_user(
            username="alice",
            email="alice@example.com",
            password_hash=password_hash,
            is_active=True,
            is_verified=True,
        )
        return int(user["id"])

    existing_user_id = asyncio.run(_seed_existing_user())

    create_response = federation_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": True,
            "display_name": "Corp SSO",
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "claim_mapping": {
                "subject": "sub",
                "email": "email",
                "username": "preferred_username",
            },
            "provisioning_policy": {
                "jit_create": True,
                "allow_email_account_linking": True,
            },
        },
    )
    assert create_response.status_code == 200, create_response.text
    provider = create_response.json()
    provider_id = int(provider["id"])

    login_response = federation_client.get(
        "/api/v1/auth/federation/corp/login",
        follow_redirects=False,
    )
    assert login_response.status_code == 307, login_response.text
    login_query = parse_qs(urlparse(login_response.headers["location"]).query)
    state = login_query["state"][0]

    async def _fake_exchange(
        self,
        *,
        provider: dict,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        nonce: str | None = None,
    ) -> dict:
        assert int(provider["id"]) == provider_id
        assert code == "code-123"
        assert redirect_uri == "http://testserver/api/v1/auth/federation/corp/callback"
        assert code_verifier
        return {
            "sub": "external-subject-123",
            "email": "alice@example.com",
            "preferred_username": "alice.sso",
            "iss": "https://issuer.example.com",
        }

    monkeypatch.setattr(
        OIDCFederationService,
        "exchange_authorization_code",
        _fake_exchange,
        raising=False,
    )

    callback_response = federation_client.get(
        f"/api/v1/auth/federation/corp/callback?state={state}&code=code-123",
    )
    assert callback_response.status_code == 200, callback_response.text
    body = callback_response.json()

    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["expires_in"] > 0

    payload = JWTService(get_settings()).verify_token(body["access_token"], token_type="access")
    assert int(payload["sub"]) == existing_user_id
    assert payload["username"] == "alice"

    async def _fetch_link() -> dict | None:
        pool = await get_db_pool()
        links_repo = FederatedIdentityRepo(db_pool=pool)
        await links_repo.ensure_tables()
        return await links_repo.get_by_provider_subject(
            identity_provider_id=int(provider["id"]),
            external_subject="external-subject-123",
        )

    link = asyncio.run(_fetch_link())
    assert link is not None
    assert int(link["user_id"]) == existing_user_id
    assert link["external_email"] == "alice@example.com"


def test_federation_callback_jit_creates_user_when_policy_allows(
    federation_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.federation.oidc_service import OIDCFederationService
    from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
    from tldw_Server_API.app.core.AuthNZ.repos.federated_identity_repo import FederatedIdentityRepo
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    create_response = federation_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": True,
            "display_name": "Corp SSO",
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "claim_mapping": {
                "subject": "sub",
                "email": "email",
                "username": "preferred_username",
            },
            "provisioning_policy": {
                "jit_create": True,
                "allow_email_account_linking": False,
            },
        },
    )
    assert create_response.status_code == 200, create_response.text
    provider = create_response.json()

    login_response = federation_client.get(
        "/api/v1/auth/federation/corp/login",
        follow_redirects=False,
    )
    assert login_response.status_code == 307, login_response.text
    login_query = parse_qs(urlparse(login_response.headers["location"]).query)
    state = login_query["state"][0]

    async def _fake_exchange(
        self,
        *,
        provider: dict,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        nonce: str | None = None,
    ) -> dict:
        assert code == "code-create-123"
        assert redirect_uri == "http://testserver/api/v1/auth/federation/corp/callback"
        assert code_verifier
        return {
            "sub": "external-create-subject-123",
            "email": "newuser@example.com",
            "preferred_username": "newuser_sso",
            "iss": "https://issuer.example.com",
        }

    monkeypatch.setattr(
        OIDCFederationService,
        "exchange_authorization_code",
        _fake_exchange,
        raising=False,
    )

    callback_response = federation_client.get(
        f"/api/v1/auth/federation/corp/callback?state={state}&code=code-create-123",
    )
    assert callback_response.status_code == 200, callback_response.text
    body = callback_response.json()
    assert body["access_token"]
    assert body["refresh_token"]

    async def _fetch_created_user() -> dict | None:
        pool = await get_db_pool()
        users_db = UsersDB(pool)
        await users_db.initialize()
        return await users_db.get_user_by_email("newuser@example.com")

    created_user = asyncio.run(_fetch_created_user())
    assert created_user is not None
    assert created_user["username"] == "newuser_sso"
    assert bool(created_user["is_verified"]) is True

    payload = JWTService(get_settings()).verify_token(body["access_token"], token_type="access")
    assert int(payload["sub"]) == int(created_user["id"])

    async def _fetch_link() -> dict | None:
        pool = await get_db_pool()
        links_repo = FederatedIdentityRepo(db_pool=pool)
        await links_repo.ensure_tables()
        return await links_repo.get_by_provider_subject(
            identity_provider_id=int(provider["id"]),
            external_subject="external-create-subject-123",
        )

    link = asyncio.run(_fetch_link())
    assert link is not None
    assert int(link["user_id"]) == int(created_user["id"])


def test_federated_password_generation_retries_until_policy_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.exceptions import WeakPasswordError
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    password_service = PasswordService(get_settings())
    validate_password = password_service.validate_password_strength
    attempts = 0

    def _reject_first_candidate(password: str, username: str | None = None) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise WeakPasswordError("synthetic policy rejection")
        validate_password(password, username)

    monkeypatch.setattr(
        password_service,
        "validate_password_strength",
        _reject_first_candidate,
    )

    password = password_service.generate_secure_password(username="federated-user")

    assert attempts >= 2
    validate_password(password, "federated-user")


def test_federated_provisioning_fails_closed_when_password_policy_rejects_all_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.auth import _provision_federated_user
    from tldw_Server_API.app.core.AuthNZ.exceptions import WeakPasswordError
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    password_service = PasswordService(get_settings())

    def _reject_candidate(_password: str, _username: str | None = None) -> None:
        raise WeakPasswordError("synthetic policy rejection")

    monkeypatch.setattr(
        password_service,
        "validate_password_strength",
        _reject_candidate,
    )

    class _RegistrationService:
        def __init__(self) -> None:
            self.password_service = password_service
            self.register_calls = 0

        async def register_user(self, **_kwargs: Any) -> dict[str, Any]:
            self.register_calls += 1
            return {
                "user_id": 1,
                "username": "federated-user",
                "email": "federated@example.com",
                "role": "user",
            }

    registration_service = _RegistrationService()

    with pytest.raises(
        RuntimeError,
        match="Unable to generate a password that satisfies the configured policy",
    ):
        asyncio.run(
            _provision_federated_user(
                mapped_claims={
                    "email": "federated@example.com",
                    "username": "federated-user",
                },
                registration_service=registration_service,  # type: ignore[arg-type]
                default_role="user",
            )
        )

    assert registration_service.register_calls == 0


def test_federation_callback_applies_mapped_grants_in_jit_grant_only_mode(
    federation_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.federation.oidc_service import OIDCFederationService
    from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
        create_organization,
        create_team,
        list_memberships_for_user,
        list_org_memberships_for_user,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    async def _seed_user_and_scope() -> tuple[int, int, int]:
        pool = await get_db_pool()
        users_db = UsersDB(pool)
        await users_db.initialize()
        user = await users_db.create_user(
            username="carol",
            email="carol@example.com",
            password_hash="not-a-real-password-hash",
            is_active=True,
            is_verified=True,
        )
        org = await create_organization(name="Federated Grants Org", owner_user_id=None)
        team = await create_team(org_id=int(org["id"]), name="Federated Grants Team")
        return int(user["id"]), int(org["id"]), int(team["id"])

    existing_user_id, mapped_org_id, mapped_team_id = asyncio.run(_seed_user_and_scope())

    create_response = federation_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": True,
            "display_name": "Corp SSO",
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "claim_mapping": {
                "subject": "sub",
                "email": "email",
                "username": "preferred_username",
                "default_roles": ["admin"],
                "default_org_ids": [mapped_org_id],
                "default_team_ids": [mapped_team_id],
            },
            "provisioning_policy": {
                "jit_create": False,
                "allow_email_account_linking": True,
                "mode": "jit_grant_only",
            },
        },
    )
    assert create_response.status_code == 200, create_response.text

    login_response = federation_client.get(
        "/api/v1/auth/federation/corp/login",
        follow_redirects=False,
    )
    assert login_response.status_code == 307, login_response.text
    login_query = parse_qs(urlparse(login_response.headers["location"]).query)
    state = login_query["state"][0]

    async def _fake_exchange(
        self,
        *,
        provider: dict,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        nonce: str | None = None,
    ) -> dict:
        assert code == "code-grants-123"
        assert redirect_uri == "http://testserver/api/v1/auth/federation/corp/callback"
        assert code_verifier
        return {
            "sub": "external-grants-subject-123",
            "email": "carol@example.com",
            "preferred_username": "carol.sso",
            "iss": "https://issuer.example.com",
        }

    monkeypatch.setattr(
        OIDCFederationService,
        "exchange_authorization_code",
        _fake_exchange,
        raising=False,
    )

    callback_response = federation_client.get(
        f"/api/v1/auth/federation/corp/callback?state={state}&code=code-grants-123",
    )
    assert callback_response.status_code == 200, callback_response.text
    body = callback_response.json()

    payload = JWTService(get_settings()).verify_token(body["access_token"], token_type="access")
    assert int(payload["sub"]) == existing_user_id
    assert mapped_org_id in payload["org_ids"]
    assert mapped_team_id in payload["team_ids"]

    async def _fetch_grants() -> tuple[list[dict], list[dict], list[str]]:
        org_memberships = await list_org_memberships_for_user(existing_user_id)
        team_memberships = await list_memberships_for_user(existing_user_id)
        role_rows = AuthnzRbacRepo().get_user_roles(existing_user_id)
        return org_memberships, team_memberships, [str(row["name"]) for row in role_rows]

    org_memberships, team_memberships, role_names = asyncio.run(_fetch_grants())
    assert any(int(row["org_id"]) == mapped_org_id for row in org_memberships)
    assert any(int(row["team_id"]) == mapped_team_id for row in team_memberships)
    assert "admin" in role_names


@pytest.mark.parametrize("provisioning_mode", ["jit_grant_and_revoke", "sync_managed_only"])
def test_federation_callback_revokes_only_provider_managed_grants_in_safe_revoke_modes(
    federation_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    provisioning_mode: str,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.federation.oidc_service import OIDCFederationService
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
        add_org_member,
        add_team_member,
        create_organization,
        create_team,
        list_memberships_for_user,
        list_org_memberships_for_user,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    async def _seed_user_and_scope() -> tuple[int, int, int, int, int]:
        pool = await get_db_pool()
        users_db = UsersDB(pool)
        await users_db.initialize()
        user = await users_db.create_user(
            username="dana",
            email="dana@example.com",
            password_hash="not-a-real-password-hash",
            is_active=True,
            is_verified=True,
        )
        managed_org = await create_organization(name="Federated Managed Org", owner_user_id=None)
        managed_team = await create_team(org_id=int(managed_org["id"]), name="Federated Managed Team")
        manual_org = await create_organization(name="Federated Manual Org", owner_user_id=None)
        manual_team = await create_team(org_id=int(manual_org["id"]), name="Federated Manual Team")
        await add_org_member(org_id=int(manual_org["id"]), user_id=int(user["id"]), role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
        await add_team_member(team_id=int(manual_team["id"]), user_id=int(user["id"]), role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
        return (
            int(user["id"]),
            int(managed_org["id"]),
            int(managed_team["id"]),
            int(manual_org["id"]),
            int(manual_team["id"]),
        )

    user_id, managed_org_id, managed_team_id, manual_org_id, manual_team_id = asyncio.run(_seed_user_and_scope())

    create_response = federation_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": True,
            "display_name": "Corp SSO",
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "claim_mapping": {
                "subject": "sub",
                "email": "email",
                "username": "preferred_username",
                "groups": "groups",
                "role_mappings": {
                    "eng-admins": "admin",
                },
                "org_mappings": {
                    "eng-admins": managed_org_id,
                },
                "team_mappings": {
                    "eng-admins": managed_team_id,
                },
            },
            "provisioning_policy": {
                "jit_create": False,
                "allow_email_account_linking": True,
                "mode": provisioning_mode,
            },
        },
    )
    assert create_response.status_code == 200, create_response.text

    async def _fake_exchange(
        self,
        *,
        provider: dict,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        nonce: str | None = None,
    ) -> dict:
        assert redirect_uri == "http://testserver/api/v1/auth/federation/corp/callback"
        assert code_verifier
        if code == "code-managed-grants":
            return {
                "sub": "external-revoke-subject-123",
                "email": "dana@example.com",
                "preferred_username": "dana.sso",
                "groups": ["eng-admins"],
                "iss": "https://issuer.example.com",
            }
        if code == "code-managed-revoke":
            return {
                "sub": "external-revoke-subject-123",
                "email": "dana@example.com",
                "preferred_username": "dana.sso",
                "groups": [],
                "iss": "https://issuer.example.com",
            }
        raise AssertionError(f"Unexpected code {code}")

    monkeypatch.setattr(
        OIDCFederationService,
        "exchange_authorization_code",
        _fake_exchange,
        raising=False,
    )

    first_login = federation_client.get(
        "/api/v1/auth/federation/corp/login",
        follow_redirects=False,
    )
    assert first_login.status_code == 307, first_login.text
    first_state = parse_qs(urlparse(first_login.headers["location"]).query)["state"][0]
    first_callback = federation_client.get(
        f"/api/v1/auth/federation/corp/callback?state={first_state}&code=code-managed-grants",
    )
    assert first_callback.status_code == 200, first_callback.text

    async def _fetch_grants() -> tuple[list[dict], list[dict], list[str]]:
        org_memberships = await list_org_memberships_for_user(user_id)
        team_memberships = await list_memberships_for_user(user_id)
        role_rows = AuthnzRbacRepo().get_user_roles(user_id)
        return org_memberships, team_memberships, [str(row["name"]) for row in role_rows]

    org_memberships, team_memberships, role_names = asyncio.run(_fetch_grants())
    assert any(int(row["org_id"]) == managed_org_id for row in org_memberships)
    assert any(int(row["team_id"]) == managed_team_id for row in team_memberships)
    assert any(int(row["org_id"]) == manual_org_id for row in org_memberships)
    assert any(int(row["team_id"]) == manual_team_id for row in team_memberships)
    assert "admin" in role_names

    second_login = federation_client.get(
        "/api/v1/auth/federation/corp/login",
        follow_redirects=False,
    )
    assert second_login.status_code == 307, second_login.text
    second_state = parse_qs(urlparse(second_login.headers["location"]).query)["state"][0]
    second_callback = federation_client.get(
        f"/api/v1/auth/federation/corp/callback?state={second_state}&code=code-managed-revoke",
    )
    assert second_callback.status_code == 200, second_callback.text

    org_memberships, team_memberships, role_names = asyncio.run(_fetch_grants())
    assert not any(int(row["org_id"]) == managed_org_id for row in org_memberships)
    assert not any(int(row["team_id"]) == managed_team_id for row in team_memberships)
    assert any(int(row["org_id"]) == manual_org_id for row in org_memberships)
    assert any(int(row["team_id"]) == manual_team_id for row in team_memberships)
    assert "admin" not in role_names
