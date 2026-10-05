from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
from tldw_Server_API.app.core.AuthNZ.federation import oidc_service as oidc_module
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

pytestmark = pytest.mark.integration
_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


@pytest.fixture
def admin_identity_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    db_path = tmp_path / "admin_identity_providers.db"
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


def _install_fetch_json_stub(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[tuple[str, str], dict],
) -> None:
    async def _fake_afetch_json(*, method: str, url: str, **kwargs) -> dict:  # noqa: ANN003
        return dict(responses[(method.upper(), url)])

    monkeypatch.setattr(oidc_module, "afetch_json", _fake_afetch_json, raising=False)


def _create_local_user(username: str, email: str) -> int:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    async def _create() -> int:
        db = UsersDB(await get_db_pool())
        user = await db.create_user(
            username=username,
            email=email,
            password_hash="hashed-password",
            role="user",
            is_active=True,
            is_verified=True,
        )
        return int(user["id"])

    return int(asyncio.run(_create()))


def test_identity_provider_mapping_preview_returns_derived_memberships(
    admin_identity_client: TestClient,
) -> None:
    create_response = admin_identity_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": False,
            "issuer": "https://issuer.example.com",
            "claim_mapping": {
                "email": "email",
                "groups": "groups",
                "role_mappings": {
                    "eng-admins": "admin",
                    "eng-members": "member",
                },
            },
            "provisioning_policy": {
                "mode": "jit_grant_only",
            },
        },
    )
    assert create_response.status_code == 200, create_response.text
    provider = create_response.json()

    preview_response = admin_identity_client.post(
        f"/api/v1/admin/identity/providers/{provider['id']}/mappings/preview",
        json={
            "claims": {
                "sub": "abc-123",
                "email": "alice@example.com",
                "groups": ["eng-admins"],
            }
        },
    )
    assert preview_response.status_code == 200, preview_response.text
    body = preview_response.json()

    assert body["provider_id"] == provider["id"]
    assert body["subject"] == "abc-123"
    assert body["email"] == "alice@example.com"
    assert body["derived_roles"] == ["admin"]
    assert body["groups"] == ["eng-admins"]


def test_identity_provider_test_resolves_discovery_runtime_config(
    admin_identity_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery_url = "https://issuer.example.com/.well-known/openid-configuration"
    _install_fetch_json_stub(
        monkeypatch,
        {
            ("GET", discovery_url): {
                "issuer": "https://issuer.example.com",
                "authorization_endpoint": "https://issuer.example.com/oauth2/authorize",
                "token_endpoint": "https://issuer.example.com/oauth2/token",
                "jwks_uri": "https://issuer.example.com/.well-known/jwks.json",
            }
        },
    )

    response = admin_identity_client.post(
        "/api/v1/admin/identity/providers/test",
        json={
            "provider": {
                "slug": "corp-test",
                "provider_type": "oidc",
                "owner_scope_type": "global",
                "enabled": False,
                "issuer": "https://issuer.example.com",
                "discovery_url": discovery_url,
                "client_id": "client-123",
            }
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["issuer"] == "https://issuer.example.com"
    assert body["authorization_url"] == "https://issuer.example.com/oauth2/authorize"
    assert body["token_url"] == "https://issuer.example.com/oauth2/token"
    assert body["jwks_url"] == "https://issuer.example.com/.well-known/jwks.json"
    assert body["client_id"] == "client-123"


def test_identity_provider_list_rejects_invalid_owner_scope_type(
    admin_identity_client: TestClient,
) -> None:
    response = admin_identity_client.get(
        "/api/v1/admin/identity/providers",
        params={"owner_scope_type": "user"},
    )

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid owner_scope_type"


def test_identity_provider_dry_run_reports_email_link_action_when_policy_allows(
    admin_identity_client: TestClient,
) -> None:
    user_id = _create_local_user("alice", "alice@example.com")

    response = admin_identity_client.post(
        "/api/v1/admin/identity/providers/dry-run",
        json={
            "provider": {
                "slug": "corp-dry-run",
                "provider_type": "oidc",
                "owner_scope_type": "global",
                "enabled": False,
                "issuer": "https://issuer.example.com",
                "authorization_url": "https://issuer.example.com/oauth2/authorize",
                "token_url": "https://issuer.example.com/oauth2/token",
                "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
                "client_id": "client-123",
                "claim_mapping": {
                    "email": "email",
                },
                "provisioning_policy": {
                    "allow_email_account_linking": True,
                    "jit_create": False,
                },
            },
            "claims": {
                "sub": "external-user-123",
                "email": "alice@example.com",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provisioning_action"] == "link_existing_user"
    assert body["matched_user_id"] == user_id
    assert body["identity_link_found"] is False
    assert body["email_match_found"] is True
    assert body["mapping"]["subject"] == "external-user-123"
    assert body["mapping"]["email"] == "alice@example.com"


def test_identity_provider_dry_run_reports_email_collision_when_linking_disabled(
    admin_identity_client: TestClient,
) -> None:
    user_id = _create_local_user("bob", "bob@example.com")

    response = admin_identity_client.post(
        "/api/v1/admin/identity/providers/dry-run",
        json={
            "provider": {
                "slug": "corp-dry-run-collision",
                "provider_type": "oidc",
                "owner_scope_type": "global",
                "enabled": False,
                "issuer": "https://issuer.example.com",
                "authorization_url": "https://issuer.example.com/oauth2/authorize",
                "token_url": "https://issuer.example.com/oauth2/token",
                "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
                "client_id": "client-123",
                "claim_mapping": {
                    "email": "email",
                },
                "provisioning_policy": {
                    "allow_email_account_linking": False,
                    "jit_create": True,
                },
            },
            "claims": {
                "sub": "external-user-456",
                "email": "bob@example.com",
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provisioning_action"] == "deny_email_collision"
    assert body["matched_user_id"] == user_id
    assert body["identity_link_found"] is False
    assert body["email_match_found"] is True
    assert body["mapping"]["subject"] == "external-user-456"
    assert body["mapping"]["email"] == "bob@example.com"


def test_identity_provider_dry_run_previews_safe_revoke_effects_for_existing_provider(
    admin_identity_client: TestClient,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
        add_org_member,
        add_team_member,
        create_organization,
        create_team,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.federated_identity_repo import FederatedIdentityRepo
    from tldw_Server_API.app.core.AuthNZ.repos.federated_managed_grant_repo import (
        FederatedManagedGrantRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo

    user_id = _create_local_user("dana", "dana@example.com")

    create_response = admin_identity_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp-dry-run-revoke",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": False,
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "claim_mapping": {
                "subject": "sub",
                "email": "email",
                "groups": "groups",
            },
            "provisioning_policy": {
                "allow_email_account_linking": True,
                "jit_create": False,
                "mode": "sync_managed_only",
            },
        },
    )
    assert create_response.status_code == 200, create_response.text
    provider = create_response.json()

    async def _seed_provider_state() -> tuple[int, int]:
        pool = await get_db_pool()
        managed_repo = FederatedManagedGrantRepo(db_pool=pool)
        identity_repo = FederatedIdentityRepo(db_pool=pool)
        users_repo = AuthnzUsersRepo(db_pool=pool)
        await managed_repo.ensure_tables()
        await identity_repo.ensure_tables()

        managed_org = await create_organization(name="Managed Dry Run Org", owner_user_id=None)
        managed_team = await create_team(org_id=int(managed_org["id"]), name="Managed Dry Run Team")
        manual_org = await create_organization(name="Manual Dry Run Org", owner_user_id=None)
        manual_team = await create_team(org_id=int(manual_org["id"]), name="Manual Dry Run Team")

        await add_org_member(org_id=int(managed_org["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
        await add_team_member(team_id=int(managed_team["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
        await add_org_member(org_id=int(manual_org["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
        await add_team_member(team_id=int(manual_team["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
        await users_repo.assign_role_if_missing(user_id=user_id, role_name="admin")

        await identity_repo.upsert_identity(
            identity_provider_id=int(provider["id"]),
            external_subject="external-user-789",
            user_id=user_id,
            external_email="dana@example.com",
            status="active",
        )
        await managed_repo.upsert_grant(
            identity_provider_id=int(provider["id"]),
            user_id=user_id,
            grant_kind="org",
            target_ref=str(int(managed_org["id"])),
        )
        await managed_repo.upsert_grant(
            identity_provider_id=int(provider["id"]),
            user_id=user_id,
            grant_kind="team",
            target_ref=str(int(managed_team["id"])),
        )
        await managed_repo.upsert_grant(
            identity_provider_id=int(provider["id"]),
            user_id=user_id,
            grant_kind="role",
            target_ref="admin",
        )
        return int(managed_org["id"]), int(managed_team["id"])

    managed_org_id, managed_team_id = asyncio.run(_seed_provider_state())

    response = admin_identity_client.post(
        "/api/v1/admin/identity/providers/dry-run",
        json={
            "provider_id": provider["id"],
            "provider": {
                "slug": "corp-dry-run-revoke",
                "provider_type": "oidc",
                "owner_scope_type": "global",
                "enabled": False,
                "issuer": "https://issuer.example.com",
                "authorization_url": "https://issuer.example.com/oauth2/authorize",
                "token_url": "https://issuer.example.com/oauth2/token",
                "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
                "client_id": "client-123",
                "claim_mapping": {
                    "subject": "sub",
                    "email": "email",
                    "groups": "groups",
                },
                "provisioning_policy": {
                    "allow_email_account_linking": True,
                    "jit_create": False,
                    "mode": "sync_managed_only",
                },
            },
            "claims": {
                "sub": "external-user-789",
                "email": "dana@example.com",
                "groups": [],
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["provisioning_action"] == "subject_already_linked"
    assert body["matched_user_id"] == user_id
    assert body["identity_link_found"] is True
    assert body["grant_sync"]["mode"] == "sync_managed_only"
    assert body["grant_sync"]["would_change"] is True
    assert body["grant_sync"]["grant_org_ids"] == []
    assert body["grant_sync"]["grant_team_ids"] == []
    assert body["grant_sync"]["grant_roles"] == []
    assert body["grant_sync"]["revoke_org_ids"] == [managed_org_id]
    assert body["grant_sync"]["revoke_team_ids"] == [managed_team_id]
    assert body["grant_sync"]["revoke_roles"] == ["admin"]


def test_create_enabled_identity_provider_requires_valid_runtime_configuration(
    admin_identity_client: TestClient,
) -> None:
    response = admin_identity_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp-invalid-enabled",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": True,
            "issuer": "https://issuer.example.com",
            "client_id": "client-123",
        },
    )

    assert response.status_code == 400, response.text
    assert "OIDC provider is missing" in response.json()["detail"]


def test_update_enabled_identity_provider_rejects_missing_env_client_secret_reference(
    admin_identity_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CORP_OIDC_CLIENT_SECRET", raising=False)
    create_response = admin_identity_client.post(
        "/api/v1/admin/identity/providers",
        json={
            "slug": "corp-update-enabled",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": False,
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "client_secret_ref": "env:CORP_OIDC_CLIENT_SECRET",
        },
    )
    assert create_response.status_code == 200, create_response.text
    provider = create_response.json()

    update_response = admin_identity_client.put(
        f"/api/v1/admin/identity/providers/{provider['id']}",
        json={
            "slug": "corp-update-enabled",
            "provider_type": "oidc",
            "owner_scope_type": "global",
            "enabled": True,
            "issuer": "https://issuer.example.com",
            "authorization_url": "https://issuer.example.com/oauth2/authorize",
            "token_url": "https://issuer.example.com/oauth2/token",
            "jwks_url": "https://issuer.example.com/.well-known/jwks.json",
            "client_id": "client-123",
            "client_secret_ref": "env:CORP_OIDC_CLIENT_SECRET",
        },
    )

    assert update_response.status_code == 400, update_response.text
    assert (
        update_response.json()["detail"]
        == "OIDC client_secret_ref environment variable is not set: CORP_OIDC_CLIENT_SECRET"
    )
