import asyncio
import base64
import json
import sys
import types
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)

_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


async def _execute_membership_fixture_sql(test_db_pool, query: str, *args) -> None:
    from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
        _execute_membership_scope_sql,
    )

    async with test_db_pool.transaction() as conn:
        await _execute_membership_scope_sql(
            conn,
            query,
            *args,
            backend="postgres",
        )


async def _insert_postgres_user(
    test_db_pool,
    *,
    username: str,
    email: str,
    password_hash: str,
    role: str = "user",
    is_superuser: bool = False,
) -> int:
    from tldw_Server_API.app.core.AuthNZ.profile_version import (
        VersionedUserWriteGateway,
    )

    async with test_db_pool.transaction() as conn:
        result = await VersionedUserWriteGateway("postgres").insert_user(
            conn,
            values={
                "uuid": str(uuid.uuid4()),
                "username": username,
                "email": email,
                "password_hash": password_hash,
                "role": role,
                "is_active": True,
                "is_verified": True,
                "is_superuser": is_superuser,
                "storage_quota_mb": 5120,
            },
        )
    return result.affected_user_ids[0]


async def _set_postgres_user_active(
    test_db_pool,
    *,
    user_id: int,
    value: bool | None,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.profile_version import (
        VersionedUserWriteGateway,
    )

    async with test_db_pool.transaction() as conn:
        await VersionedUserWriteGateway("postgres").execute_update(
            conn,
            user_id=user_id,
            profile_visible_fields=("is_active",),
            statement="UPDATE public.users SET is_active = $1 WHERE id = $2",
            parameters=(value, user_id),
        )


class _PostgresMutationConnectionGate:
    def __init__(self, connection, owner: "_PostgresMutationGatePool") -> None:
        self.connection = connection
        self.owner = owner

    async def execute(self, query: str, *args):
        if "pg_advisory_xact_lock" not in query:
            return await self.connection.execute(query, *args)
        if self.owner.role == "upsert":
            self.owner.upsert_attempted.set()
            result = await self.connection.execute(query, *args)
        else:
            result = await self.connection.execute(query, *args)
            self.owner.revoke_ready.set()
            await self.owner.release_revoke.wait()
        self.owner.identity_lock_count += 1
        return result

    async def fetchrow(self, query: str, *args):
        if self.owner.role == "upsert" and "for update" in query.lower():
            self.owner.upsert_attempted.set()
        return await self.connection.fetchrow(query, *args)

    def __getattr__(self, name: str):
        return getattr(self.connection, name)


class _PostgresMutationGatePool:
    """Coordinate legacy split statements and transaction-locked mutations."""

    def __init__(
        self,
        delegate,
        *,
        role: str,
        revoke_ready: asyncio.Event,
        release_revoke: asyncio.Event,
        upsert_attempted: asyncio.Event,
    ) -> None:
        self.delegate = delegate
        self.pool = delegate.pool
        self.role = role
        self.revoke_ready = revoke_ready
        self.release_revoke = release_revoke
        self.upsert_attempted = upsert_attempted
        self.identity_lock_count = 0

    @asynccontextmanager
    async def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ):
        async with self.delegate.transaction(
            acquire_timeout_seconds=acquire_timeout_seconds,
        ) as connection:
            yield _PostgresMutationConnectionGate(connection, self)

    async def fetchone(self, query: str, *args):
        if self.role == "upsert":
            self.upsert_attempted.set()
        row = await self.delegate.fetchone(query, *args)
        if self.role == "revoke" and row is not None:
            self.revoke_ready.set()
            await self.release_revoke.wait()
        return row

    async def execute(self, query: str, *args):
        if self.role == "upsert":
            self.upsert_attempted.set()
        return await self.delegate.execute(query, *args)

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)


class _PostgresAliasReadGatePool:
    """Place a revocation exactly between legacy split reads, if they exist."""

    def __init__(self, delegate, *, table: str) -> None:
        self.delegate = delegate
        self.pool = delegate.pool
        self.table = table
        self.read_started = asyncio.Event()
        self.release_read = asyncio.Event()
        self.alias_statement_count = 0
        self.last_alias_query = ""
        self._gated = False

    def _is_alias_read(self, query: str) -> bool:
        normalized = " ".join(query.split()).lower()
        return (
            normalized.startswith("select")
            and self.table in normalized
            and "provider" in normalized
        )

    async def fetchone(self, query: str, *args):
        if not self._is_alias_read(query):
            return await self.delegate.fetchone(query, *args)

        self.alias_statement_count += 1
        self.last_alias_query = query
        row = await self.delegate.fetchone(query, *args)
        if not self._gated:
            self._gated = True
            self.read_started.set()
            await self.release_read.wait()
        return row

    async def fetchall(self, query: str, *args):
        if not self._is_alias_read(query):
            return await self.delegate.fetchall(query, *args)

        self.alias_statement_count += 1
        self.last_alias_query = query
        if not self._gated:
            self._gated = True
            self.read_started.set()
            await self.release_read.wait()
        return await self.delegate.fetchall(query, *args)

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _capture_real_openai_adapter_headers(monkeypatch):
    """Install a fake transport beneath the real OpenAI adapter."""
    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter as adapter_module

    captured_headers: list[dict[str, str]] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, _url, *, headers, json):
            del json
            captured_headers.append(dict(headers))
            return FakeResponse()

    monkeypatch.setattr(adapter_module, "http_client_factory", lambda **_kwargs: FakeClient())
    return adapter_module.OpenAIAdapter(), captured_headers


async def _create_postgres_runtime_scope(test_db_pool) -> tuple[int, int, int]:
    """Create one active user with active org and team memberships."""
    suffix = uuid.uuid4().hex
    user_id = await _insert_postgres_user(
        test_db_pool,
        username=f"runtime-{suffix}",
        email=f"runtime-{suffix}@example.com",
        password_hash="hashed-password",
    )
    org = await test_db_pool.fetchrow(
        """
        INSERT INTO organizations (uuid, name, owner_user_id, is_active)
        VALUES ($1, $2, $3, TRUE)
        RETURNING id
        """,
        str(uuid.uuid4()),
        f"Runtime Org {suffix}",
        user_id,
    )
    org_id = int(org["id"])
    await _execute_membership_fixture_sql(
        test_db_pool,
        """
        INSERT INTO public.org_members (org_id, user_id, role, status)
        VALUES ($1, $2, 'lead', 'active')
        """,
        org_id,
        user_id,
    )
    team = await test_db_pool.fetchrow(
        """
        INSERT INTO teams (org_id, name, is_active)
        VALUES ($1, $2, TRUE)
        RETURNING id
        """,
        org_id,
        f"Runtime Team {suffix}",
    )
    team_id = int(team["id"])
    await _execute_membership_fixture_sql(
        test_db_pool,
        """
        INSERT INTO public.team_members (team_id, user_id, role, status)
        VALUES ($1, $2, 'lead', 'active')
        """,
        team_id,
        user_id,
    )
    return user_id, org_id, team_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_org_provider_secrets_use_public_schema_under_shadow_search_path(
    test_db_pool,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )

    _user_id, org_id, _team_id = await _create_postgres_runtime_scope(test_db_pool)
    schema = f"org_secret_shadow_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)

    async with test_db_pool.transaction() as conn:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(
            f'CREATE TABLE "{schema}".org_provider_secrets '
            "(LIKE public.org_provider_secrets INCLUDING ALL)"
        )
        await conn.fetchval(
            "SELECT set_config('search_path', $1, TRUE)",
            f'"{schema}", public',
        )

        class _ConnectionBoundPool:
            pool = object()

            @asynccontextmanager
            async def transaction(
                self,
                *,
                acquire_timeout_seconds: float | None = None,
            ):
                assert acquire_timeout_seconds is not None
                yield conn

            async def fetchall(self, query: str, *args):
                return await conn.fetch(query, *args)

            async def execute(self, query: str, *args):
                return await conn.execute(query, *args)

        repo = AuthnzOrgProviderSecretsRepo(_ConnectionBoundPool())  # type: ignore[arg-type]
        written = await repo.upsert_secret(
            scope_type="org",
            scope_id=org_id,
            provider="openai",
            encrypted_blob="public-only",
            key_hint="shadow-test",
            metadata=None,
            updated_at=now,
        )
        fetched = await repo.fetch_secret("org", org_id, "openai")

        assert written["provider"] == "openai"
        assert fetched is not None
        assert fetched["encrypted_blob"] == "public-only"
        assert await conn.fetchval(
            "SELECT COUNT(*) FROM public.org_provider_secrets "
            "WHERE scope_type = 'org' AND scope_id = $1",
            org_id,
        ) == 1
        assert await conn.fetchval(
            f'SELECT COUNT(*) FROM "{schema}".org_provider_secrets'
        ) == 0


async def _insert_postgres_user_payload(
    test_db_pool,
    *,
    user_id: int,
    provider: str,
    payload: dict,
) -> None:
    """Insert a provider spelling without repository canonicalization."""
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        dumps_envelope,
        encrypt_byok_payload,
        key_hint_for_api_key,
    )

    now = datetime.now(timezone.utc)
    await test_db_pool.execute(
        """
        INSERT INTO user_provider_secrets (
            user_id, provider, encrypted_blob, key_hint, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $5)
        """,
        user_id,
        provider,
        dumps_envelope(encrypt_byok_payload(payload)),
        "oauth" if payload.get("credential_version") == 2 else key_hint_for_api_key(
            str(payload.get("api_key") or "")
        ),
        now,
    )


async def _insert_postgres_shared_payload(
    test_db_pool,
    *,
    scope_type: str,
    scope_id: int,
    provider: str,
    payload: dict,
) -> None:
    """Insert a shared provider spelling without repository canonicalization."""
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        dumps_envelope,
        encrypt_byok_payload,
        key_hint_for_api_key,
    )

    now = datetime.now(timezone.utc)
    await test_db_pool.execute(
        """
        INSERT INTO org_provider_secrets (
            scope_type, scope_id, provider, encrypted_blob, key_hint,
            created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $6)
        """,
        scope_type,
        scope_id,
        provider,
        dumps_envelope(encrypt_byok_payload(payload)),
        key_hint_for_api_key(str(payload.get("api_key") or "")),
        now,
    )


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
@pytest.mark.concurrent
@pytest.mark.parametrize(
    "null_boundary",
    [
        "team_user",
        "team_membership",
        "team",
        "team_org",
        "org_user",
        "org_membership",
        "org",
    ],
)
async def test_authorized_shared_fetch_rejects_null_activity_boundaries_postgres(
    test_db_pool,
    null_boundary: str,
) -> None:
    """Legacy NULL activity state must never authorize a shared secret."""
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )

    user_id, org_id, team_id = await _create_postgres_runtime_scope(test_db_pool)
    scope_type = "team" if null_boundary.startswith("team") else "org"
    scope_id = team_id if scope_type == "team" else org_id
    repo = AuthnzOrgProviderSecretsRepo(test_db_pool)
    await repo.upsert_secret(
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai",
        encrypted_blob="opaque-test-payload",
        key_hint=None,
        metadata=None,
        updated_at=datetime.now(timezone.utc),
        created_by=user_id,
        updated_by=user_id,
    )
    assert await repo.fetch_authorized_secret_for_user(
        scope_type,
        scope_id,
        user_id,
        "openai",
    ) is not None

    if null_boundary in {"team_user", "org_user"}:
        await _set_postgres_user_active(
            test_db_pool,
            user_id=user_id,
            value=None,
        )
    elif null_boundary == "team_membership":
        await _execute_membership_fixture_sql(
            test_db_pool,
            "UPDATE public.team_members SET status = NULL "
            "WHERE team_id = $1 AND user_id = $2",
            scope_id,
            user_id,
        )
    elif null_boundary == "team":
        await test_db_pool.execute(
            "UPDATE teams SET is_active = NULL WHERE id = $1",
            scope_id,
        )
    elif null_boundary == "team_org":
        await test_db_pool.execute(
            "UPDATE organizations SET is_active = NULL WHERE id = $1",
            org_id,
        )
    elif null_boundary == "org_membership":
        await _execute_membership_fixture_sql(
            test_db_pool,
            "UPDATE public.org_members SET status = NULL "
            "WHERE org_id = $1 AND user_id = $2",
            scope_id,
            user_id,
        )
    else:
        await test_db_pool.execute(
            "UPDATE organizations SET is_active = NULL WHERE id = $1",
            scope_id,
        )

    assert await repo.fetch_authorized_secret_for_user(
        scope_type,
        scope_id,
        user_id,
        "openai",
    ) is None


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

    from tldw_Server_API.app.api.v1.endpoints import user_keys as user_keys_endpoints
    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
        add_org_member,
        add_team_member,
        create_organization_with_owner_membership,
        create_team,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings as reset_auth_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        ProviderCredentialAliasConflictError,
    )
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.main import app

    reset_auth_settings()
    reset_jwt_service()
    app_settings["CSRF_ENABLED"] = False

    name_suffix = uuid.uuid4().hex[:12]
    admin_username = f"byok-pg-admin-{name_suffix}"
    user_username = f"byok-pg-user-{name_suffix}"

    admin_id = await _insert_postgres_user(
        test_db_pool,
        username=admin_username,
        email=f"{admin_username}@example.com",
        password_hash="hashed-admin",
        role="admin",
        is_superuser=True,
    )
    user_id = await _insert_postgres_user(
        test_db_pool,
        username=user_username,
        email=f"{user_username}@example.com",
        password_hash="hashed-user",
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

    assert int(admin_row["id"]) == admin_id
    assert int(user_row["id"]) == user_id

    org = await create_organization_with_owner_membership(
        name=f"BYOK Org {name_suffix}",
        owner_user_id=admin_id,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )
    team = await create_team(org_id=int(org["id"]), name=f"BYOK Team {name_suffix}")
    await add_org_member(org_id=int(org["id"]), user_id=user_id, role="lead", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
    await add_team_member(team_id=int(team["id"]), user_id=user_id, role="lead", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)

    shared_repo = AuthnzOrgProviderSecretsRepo(test_db_pool)
    await shared_repo.ensure_tables()
    repo_org = await create_organization_with_owner_membership(
        name=f"BYOK Repo Org {name_suffix}",
        owner_user_id=admin_id,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )
    repo_team = await create_team(
        org_id=int(repo_org["id"]),
        name=f"BYOK Repo Team {name_suffix}",
    )
    base_scope_id = int(repo_team["id"])
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

    legacy_scope_id = int(repo_org["id"])
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

    revoked_scope_id = 1_000_000 + (uuid.uuid4().int % 1_000_000)
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

    for scope_type, scope_id in (
        ("team", int(team["id"])),
        ("org", int(org["id"])),
    ):
        await test_db_pool.execute(
            """
            INSERT INTO org_provider_secrets (
                scope_type, scope_id, provider, encrypted_blob, key_hint,
                revoked_at, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $6, $6)
            """,
            scope_type,
            scope_id,
            "custom-openai-api",
            f"pg-authorized-revoked-{scope_type}",
            "canonical",
            now,
        )
        await test_db_pool.execute(
            """
            INSERT INTO org_provider_secrets (
                scope_type, scope_id, provider, encrypted_blob, key_hint, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $6)
            """,
            scope_type,
            scope_id,
            "openai-compatible",
            f"pg-authorized-active-alias-{scope_type}",
            "alias",
            now,
        )

        authorized_revoked = await shared_repo.fetch_authorized_secret_for_user(
            scope_type,
            scope_id,
            user_id,
            "openai-compatible",
        )

        assert authorized_revoked is not None
        assert authorized_revoked["provider"] == "custom-openai-api"
        assert authorized_revoked["revoked_at"] is not None

    conflict_scope_id = 1_000_000 + (uuid.uuid4().int % 1_000_000)
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_mutation_lock_bound_repo_executes_delete_postgres(
    test_db_pool,
    monkeypatch,
):
    """The advisory-lock connection-bound repository supports real revocation."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )

    user_suffix = uuid.uuid4().hex
    user_id = await _insert_postgres_user(
        test_db_pool,
        username=f"lock-delete-{user_suffix}",
        email=f"lock-delete-{user_suffix}@example.com",
        password_hash="hashed-password",
    )
    repo = AuthnzUserProviderSecretsRepo(test_db_pool)
    await repo.upsert_secret(
        user_id=user_id,
        provider="openai",
        encrypted_blob="opaque-lock-delete-blob",
        key_hint="lock",
        metadata=None,
        updated_at=datetime.now(timezone.utc),
        created_by=user_id,
        updated_by=user_id,
    )

    async def get_test_pool():
        return test_db_pool

    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    monkeypatch.setattr(byok_runtime, "get_db_pool", get_test_pool)

    async with byok_runtime.openai_credential_mutation_lock(
        user_id=user_id,
        provider="openai",
    ) as locked_repo:
        assert locked_repo is not None
        assert await locked_repo.delete_secret(user_id, "OpenAI", revoked_by=user_id)

    revoked_row = await repo.fetch_secret_for_user(user_id, "openai", include_revoked=True)
    assert revoked_row is not None
    assert revoked_row["revoked_at"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("owner_kind", ["user", "org"])
async def test_alias_revoke_and_canonical_upsert_serialize_postgres(
    test_db_pool,
    owner_kind: str,
) -> None:
    """Alias revocation and canonical replacement form one serial DB history."""
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )

    now = datetime.now(timezone.utc)
    identity = 1_000_000 + (uuid.uuid4().int % 1_000_000)
    actor_user_id = identity
    if owner_kind == "user":
        user_suffix = uuid.uuid4().hex
        identity = await _insert_postgres_user(
            test_db_pool,
            username=f"alias-race-{user_suffix}",
            email=f"alias-race-{user_suffix}@example.com",
            password_hash="hashed-password",
        )
        await test_db_pool.execute(
            """
            INSERT INTO user_provider_secrets (
                user_id, provider, encrypted_blob, key_hint, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $5)
            """,
            identity,
            "oai",
            "legacy",
            "legacy",
            now,
        )
    else:
        actor_user_id, identity, _team_id = await _create_postgres_runtime_scope(
            test_db_pool
        )
        await test_db_pool.execute(
            """
            INSERT INTO org_provider_secrets (
                scope_type, scope_id, provider, encrypted_blob, key_hint,
                created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $6)
            """,
            "org",
            identity,
            "oai",
            "legacy",
            "legacy",
            now,
        )

    revoke_ready = asyncio.Event()
    release_revoke = asyncio.Event()
    upsert_attempted = asyncio.Event()
    revoke_pool = _PostgresMutationGatePool(
        test_db_pool,
        role="revoke",
        revoke_ready=revoke_ready,
        release_revoke=release_revoke,
        upsert_attempted=upsert_attempted,
    )
    upsert_pool = _PostgresMutationGatePool(
        test_db_pool,
        role="upsert",
        revoke_ready=revoke_ready,
        release_revoke=release_revoke,
        upsert_attempted=upsert_attempted,
    )

    if owner_kind == "user":
        revoke_repo = AuthnzUserProviderSecretsRepo(revoke_pool)
        upsert_repo = AuthnzUserProviderSecretsRepo(upsert_pool)
        revoke_task = asyncio.create_task(
            revoke_repo.delete_secret(identity, "openai", revoked_by=identity)
        )

        async def upsert():
            return await upsert_repo.upsert_secret(
                user_id=identity,
                provider="openai",
                encrypted_blob="canonical",
                key_hint="canonical",
                metadata=None,
                updated_at=now,
                created_by=identity,
                updated_by=identity,
            )

        final_query = (
            "SELECT provider, revoked_at FROM user_provider_secrets "
            "WHERE user_id = $1 ORDER BY provider"
        )
        final_args = (identity,)
    else:
        revoke_repo = AuthnzOrgProviderSecretsRepo(revoke_pool)
        upsert_repo = AuthnzOrgProviderSecretsRepo(upsert_pool)
        revoke_task = asyncio.create_task(
            revoke_repo.delete_secret(
                "org",
                identity,
                "openai",
                revoked_by=actor_user_id,
            )
        )

        async def upsert():
            return await upsert_repo.upsert_secret(
                scope_type="org",
                scope_id=identity,
                provider="openai",
                encrypted_blob="canonical",
                key_hint="canonical",
                metadata=None,
                updated_at=now,
                created_by=actor_user_id,
                updated_by=actor_user_id,
            )

        final_query = (
            "SELECT provider, revoked_at FROM org_provider_secrets "
            "WHERE scope_type = $1 AND scope_id = $2 ORDER BY provider"
        )
        final_args = ("org", identity)

    upsert_task: asyncio.Task | None = None
    try:
        await asyncio.wait_for(revoke_ready.wait(), timeout=2)
        upsert_task = asyncio.create_task(upsert())
        await asyncio.wait_for(upsert_attempted.wait(), timeout=2)
        release_revoke.set()
        revoked, written = await asyncio.wait_for(
            asyncio.gather(revoke_task, upsert_task),
            timeout=10,
        )
    finally:
        release_revoke.set()
        pending = [
            task
            for task in (revoke_task, upsert_task)
            if task is not None and not task.done()
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    assert revoked
    assert written["provider"] == "openai"
    assert revoke_pool.identity_lock_count == 1
    assert upsert_pool.identity_lock_count == 1
    rows = await test_db_pool.fetchall(final_query, *final_args)
    assert [(row["provider"], row["revoked_at"]) for row in rows] == [
        ("openai", None)
    ]


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("owner_kind", ["user", "team"])
async def test_alias_revoke_interleaving_fails_closed_before_openai_adapter_postgres(
    test_db_pool,
    monkeypatch,
    owner_kind: str,
) -> None:
    """A canonical tombstone created mid-lookup cannot expose a lower key."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
        ByokResolutionError,
        resolve_byok_credentials,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import build_secret_payload

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    reset_settings()

    async def get_test_pool():
        return test_db_pool

    monkeypatch.setattr(byok_runtime, "get_db_pool", get_test_pool)
    user_id, org_id, team_id = await _create_postgres_runtime_scope(test_db_pool)
    legacy_key = f"sk-{owner_kind}-legacy-pg-1111"
    lower_key = f"sk-{owner_kind}-lower-pg-2222"
    adapter, captured_headers = _capture_real_openai_adapter_headers(monkeypatch)

    if owner_kind == "user":
        await _insert_postgres_user_payload(
            test_db_pool,
            user_id=user_id,
            provider="oai",
            payload=build_secret_payload(legacy_key),
        )
        gate_pool = _PostgresAliasReadGatePool(
            test_db_pool,
            table="user_provider_secrets",
        )
        gated_repo = AuthnzUserProviderSecretsRepo(gate_pool)
        writer_repo = AuthnzUserProviderSecretsRepo(test_db_pool)

        async def get_gated_user_repo():
            return gated_repo

        monkeypatch.setattr(byok_runtime, "_get_user_repo", get_gated_user_repo)
        resolve_kwargs = {
            "team_ids": [],
            "org_ids": [],
            "required_source": "user",
            "server_config_snapshot": {"openai_api": {"api_key": lower_key}},
        }

        async def revoke():
            return await writer_repo.delete_secret(
                user_id,
                "openai",
                revoked_by=user_id,
            )

        final_rows_sql = (
            "SELECT provider, revoked_at FROM user_provider_secrets "
            "WHERE user_id = $1 ORDER BY provider"
        )
        final_rows_params = (user_id,)
    else:
        await _insert_postgres_shared_payload(
            test_db_pool,
            scope_type="team",
            scope_id=team_id,
            provider="oai",
            payload=build_secret_payload(legacy_key),
        )
        await _insert_postgres_shared_payload(
            test_db_pool,
            scope_type="org",
            scope_id=org_id,
            provider="openai",
            payload=build_secret_payload(lower_key),
        )
        gate_pool = _PostgresAliasReadGatePool(
            test_db_pool,
            table="org_provider_secrets",
        )
        gated_repo = AuthnzOrgProviderSecretsRepo(gate_pool)
        writer_repo = AuthnzOrgProviderSecretsRepo(test_db_pool)

        async def get_gated_org_repo():
            return gated_repo

        monkeypatch.setattr(byok_runtime, "_get_org_repo", get_gated_org_repo)
        resolve_kwargs = {
            "team_ids": [team_id],
            "org_ids": [org_id],
            "server_config_snapshot": {"openai_api": {"api_key": lower_key}},
        }

        async def revoke():
            return await writer_repo.delete_secret(
                "team",
                team_id,
                "openai",
                revoked_by=user_id,
            )

        final_rows_sql = (
            "SELECT provider, revoked_at FROM org_provider_secrets "
            "WHERE scope_type = $1 AND scope_id = $2 ORDER BY provider"
        )
        final_rows_params = ("team", team_id)

    async def resolve_and_dispatch():
        resolved = await resolve_byok_credentials(
            "openai",
            user_id=user_id,
            **resolve_kwargs,
        )
        return adapter.chat(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-test",
                "api_key": resolved.api_key,
                "app_config": resolved.app_config,
            }
        )

    resolution_task = asyncio.create_task(resolve_and_dispatch())
    await asyncio.wait_for(gate_pool.read_started.wait(), timeout=10)
    try:
        assert await revoke()
    finally:
        gate_pool.release_read.set()

    with pytest.raises(ByokResolutionError) as exc_info:
        await asyncio.wait_for(resolution_task, timeout=10)

    assert exc_info.value.code == "invalid_provider_credentials"
    assert gate_pool.alias_statement_count == 1
    assert (
        "provider = ANY(" in gate_pool.last_alias_query
        or "provider IN (" in gate_pool.last_alias_query
    )
    assert captured_headers == []
    rows = await test_db_pool.fetchall(final_rows_sql, *final_rows_params)
    assert [(row["provider"], row["revoked_at"] is not None) for row in rows] == [
        ("openai", True)
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_oai_oauth_refresh_migrates_before_openai_adapter_postgres(
    test_db_pool,
    monkeypatch,
) -> None:
    """A legacy OAuth row refreshes once and persists under the canonical name."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import resolve_byok_credentials
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        decrypt_byok_payload,
        loads_envelope,
    )

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "1")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    reset_settings()

    async def get_test_pool():
        return test_db_pool

    monkeypatch.setattr(byok_runtime, "get_db_pool", get_test_pool)
    user_id, _org_id, _team_id = await _create_postgres_runtime_scope(test_db_pool)
    await _insert_postgres_user_payload(
        test_db_pool,
        user_id=user_id,
        provider="oai",
        payload={
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": "expired-access-token",
                    "refresh_token": "legacy-refresh-token",
                    "expires_at": (
                        datetime.now(timezone.utc) - timedelta(minutes=1)
                    ).isoformat(),
                }
            },
        },
    )
    refresh_tokens: list[str] = []

    async def refresh_token(**kwargs):
        refresh_tokens.append(str(kwargs["refresh_token"]))
        return {
            "access_token": "fresh-access-token",
            "refresh_token": "fresh-refresh-token",
            "expires_in": 3600,
        }

    monkeypatch.setattr(byok_runtime, "_openai_oauth_token_refresh", refresh_token)
    adapter, captured_headers = _capture_real_openai_adapter_headers(monkeypatch)
    resolved = await resolve_byok_credentials(
        "openai",
        user_id=user_id,
        team_ids=[],
        org_ids=[],
        server_config_snapshot={"openai_api": {"api_key": "sk-server-sentinel"}},
    )
    response = adapter.chat(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "model": "gpt-test",
            "api_key": resolved.api_key,
            "app_config": resolved.app_config,
        }
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert refresh_tokens == ["legacy-refresh-token"]
    assert [headers["Authorization"] for headers in captured_headers] == [
        "Bearer fresh-access-token"
    ]
    rows = await test_db_pool.fetchall(
        """
        SELECT provider, encrypted_blob, revoked_at
        FROM user_provider_secrets
        WHERE user_id = $1
        ORDER BY provider
        """,
        user_id,
    )
    assert [row["provider"] for row in rows] == ["openai"]
    assert rows[0]["revoked_at"] is None
    stored_payload = decrypt_byok_payload(loads_envelope(rows[0]["encrypted_blob"]))
    assert stored_payload["credentials"]["oauth"]["access_token"] == "fresh-access-token"
    assert stored_payload["credentials"]["oauth"]["refresh_token"] == "fresh-refresh-token"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_oai_oauth_bound_repo_supports_set_read_postgres(
    test_db_pool,
    monkeypatch,
) -> None:
    """The advisory-lock connection-bound repository supports alias set reads."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import build_secret_payload

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    reset_settings()

    async def get_test_pool():
        return test_db_pool

    monkeypatch.setattr(byok_runtime, "get_db_pool", get_test_pool)
    user_id, _org_id, _team_id = await _create_postgres_runtime_scope(test_db_pool)
    await _insert_postgres_user_payload(
        test_db_pool,
        user_id=user_id,
        provider="oai",
        payload=build_secret_payload("sk-legacy-bound-read"),
    )

    async with byok_runtime.openai_credential_mutation_lock(
        user_id=user_id,
        provider="openai",
    ) as locked_repo:
        assert locked_repo is not None
        row = await locked_repo.fetch_secret_for_user(user_id, "openai")

    assert row is not None
    assert row["provider"] == "oai"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_legacy_oai_oauth_refresh_cannot_resurrect_revoke_postgres(
    test_db_pool,
    monkeypatch,
) -> None:
    """A revoke committed during refresh wins over canonicalizing CAS."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
        ByokResolutionError,
        resolve_byok_credentials,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        decrypt_byok_payload,
        loads_envelope,
    )

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "1")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    reset_settings()

    async def get_test_pool():
        return test_db_pool

    monkeypatch.setattr(byok_runtime, "get_db_pool", get_test_pool)
    user_id, _org_id, _team_id = await _create_postgres_runtime_scope(test_db_pool)
    await _insert_postgres_user_payload(
        test_db_pool,
        user_id=user_id,
        provider="oai",
        payload={
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": "expired-access-token",
                    "refresh_token": "single-use-refresh-token",
                    "expires_at": (
                        datetime.now(timezone.utc) - timedelta(minutes=1)
                    ).isoformat(),
                }
            },
        },
    )
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    refresh_calls = 0

    async def refresh_token(**_kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        refresh_started.set()
        await release_refresh.wait()
        return {
            "access_token": "must-not-be-persisted",
            "refresh_token": "must-not-resurrect",
            "expires_in": 3600,
        }

    monkeypatch.setattr(byok_runtime, "_openai_oauth_token_refresh", refresh_token)
    adapter, captured_headers = _capture_real_openai_adapter_headers(monkeypatch)

    async def resolve_and_dispatch():
        resolved = await resolve_byok_credentials(
            "openai",
            user_id=user_id,
            team_ids=[],
            org_ids=[],
            server_config_snapshot={},
        )
        return adapter.chat(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-test",
                "api_key": resolved.api_key,
                "app_config": resolved.app_config,
            }
        )

    resolution_task = asyncio.create_task(resolve_and_dispatch())
    await asyncio.wait_for(refresh_started.wait(), timeout=10)
    repo = AuthnzUserProviderSecretsRepo(test_db_pool)
    try:
        assert await repo.delete_secret(user_id, "openai", revoked_by=user_id)
    finally:
        release_refresh.set()

    with pytest.raises(ByokResolutionError) as exc_info:
        await asyncio.wait_for(resolution_task, timeout=10)

    assert exc_info.value.code == "invalid_provider_credentials"
    assert refresh_calls == 1
    assert captured_headers == []
    rows = await test_db_pool.fetchall(
        """
        SELECT provider, encrypted_blob, revoked_at
        FROM user_provider_secrets
        WHERE user_id = $1
        ORDER BY provider
        """,
        user_id,
    )
    assert [row["provider"] for row in rows] == ["openai"]
    assert rows[0]["revoked_at"] is not None
    stored_payload = decrypt_byok_payload(loads_envelope(rows[0]["encrypted_blob"]))
    assert stored_payload["credentials"]["oauth"]["access_token"] == "expired-access-token"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("inactive_value", [False, None], ids=["false", "null"])
async def test_inactive_user_blocks_overlapping_oauth_refresh_before_openai_adapter_postgres(
    test_db_pool,
    monkeypatch,
    inactive_value,
) -> None:
    """Deactivation wins against the refresher and its advisory-lock waiter."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
        ByokResolutionError,
        resolve_byok_credentials,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    monkeypatch.setenv("OPENAI_OAUTH_ENABLED", "1")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_ID", "client-id")
    monkeypatch.setenv("OPENAI_OAUTH_CLIENT_SECRET", "client-secret")
    monkeypatch.setenv("OPENAI_OAUTH_TOKEN_URL", "https://oauth.example/token")
    monkeypatch.setenv("OPENAI_OAUTH_REFRESH_LOCK_BACKEND", "db")
    reset_settings()

    async def get_test_pool():
        return test_db_pool

    monkeypatch.setattr(byok_runtime, "get_db_pool", get_test_pool)
    user_id, _org_id, _team_id = await _create_postgres_runtime_scope(test_db_pool)
    await _insert_postgres_user_payload(
        test_db_pool,
        user_id=user_id,
        provider="openai",
        payload={
            "credential_version": 2,
            "active_auth_source": "oauth",
            "credentials": {
                "oauth": {
                    "access_token": "expired-access-token",
                    "refresh_token": "single-use-refresh-token",
                    "expires_at": (
                        datetime.now(timezone.utc) - timedelta(minutes=1)
                    ).isoformat(),
                }
            },
        },
    )
    original_row = await test_db_pool.fetchrow(
        "SELECT encrypted_blob FROM user_provider_secrets WHERE user_id = $1 AND provider = $2",
        user_id,
        "openai",
    )
    assert original_row is not None
    original_blob = original_row["encrypted_blob"]

    real_repo = AuthnzUserProviderSecretsRepo(test_db_pool)
    initial_reads_ready = asyncio.Event()
    task_active_reads: dict[asyncio.Task, int] = {}
    initial_read_count = 0

    class CountingRepo:
        async def fetch_secret_for_active_user(self, *args, **kwargs):
            nonlocal initial_read_count
            task = asyncio.current_task()
            assert task is not None
            prior = task_active_reads.get(task, 0)
            task_active_reads[task] = prior + 1
            row = await real_repo.fetch_secret_for_active_user(*args, **kwargs)
            if prior == 0:
                assert row is not None
                initial_read_count += 1
                if initial_read_count == 2:
                    initial_reads_ready.set()
            return row

        def __getattr__(self, name):
            return getattr(real_repo, name)

    counting_repo = CountingRepo()

    async def get_counting_repo():
        return counting_repo

    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    refresh_calls = 0

    async def refresh_token(**_kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        refresh_started.set()
        await release_refresh.wait()
        return {
            "access_token": "must-not-be-persisted",
            "refresh_token": "must-not-be-used",
            "expires_in": 3600,
        }

    monkeypatch.setattr(byok_runtime, "_get_user_repo", get_counting_repo)
    monkeypatch.setattr(byok_runtime, "_openai_oauth_token_refresh", refresh_token)
    adapter, captured_headers = _capture_real_openai_adapter_headers(monkeypatch)
    adapter_calls = 0

    async def resolve_and_dispatch():
        nonlocal adapter_calls
        resolved = await resolve_byok_credentials(
            "openai",
            user_id=user_id,
            team_ids=[],
            org_ids=[],
            required_source="user",
            server_config_snapshot={},
        )
        adapter_calls += 1
        return adapter.chat(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-test",
                "api_key": resolved.api_key,
                "app_config": resolved.app_config,
            }
        )

    first_task = asyncio.create_task(resolve_and_dispatch())
    second_task = None
    try:
        await asyncio.wait_for(refresh_started.wait(), timeout=10)
        second_task = asyncio.create_task(resolve_and_dispatch())
        await asyncio.wait_for(initial_reads_ready.wait(), timeout=10)
        await asyncio.sleep(0)
        assert not second_task.done()
        await _set_postgres_user_active(
            test_db_pool,
            user_id=user_id,
            value=inactive_value,
        )
    finally:
        release_refresh.set()

    assert second_task is not None
    results = await asyncio.wait_for(
        asyncio.gather(first_task, second_task, return_exceptions=True),
        timeout=10,
    )
    errors = [result for result in results if isinstance(result, ByokResolutionError)]
    assert len(errors) == 2
    assert all(error.code == "invalid_provider_credentials" for error in errors)
    assert all(error.provider == "openai" for error in errors)
    assert all(error.__cause__ is None and error.__context__ is None for error in errors)
    assert initial_read_count == 2
    assert refresh_calls == 1
    assert adapter_calls == 0
    assert captured_headers == []

    stored_row = await test_db_pool.fetchrow(
        "SELECT encrypted_blob FROM user_provider_secrets WHERE user_id = $1 AND provider = $2",
        user_id,
        "openai",
    )
    assert stored_row is not None
    assert stored_row["encrypted_blob"] == original_blob


@pytest.mark.integration
@pytest.mark.asyncio
async def test_inactive_user_static_openai_key_fails_before_adapter_postgres(
    test_db_pool,
    monkeypatch,
) -> None:
    """Default-source lookup cannot dispatch a deactivated owner's API key."""
    from tldw_Server_API.app.core.AuthNZ import byok_runtime
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
        ByokResolutionError,
        resolve_byok_credentials,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import build_secret_payload

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("BYOK_ENABLED", "1")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))
    reset_settings()

    async def get_test_pool():
        return test_db_pool

    monkeypatch.setattr(byok_runtime, "get_db_pool", get_test_pool)
    user_id, _org_id, _team_id = await _create_postgres_runtime_scope(test_db_pool)
    await _insert_postgres_user_payload(
        test_db_pool,
        user_id=user_id,
        provider="openai",
        payload=build_secret_payload("sk-inactive-owner-must-not-dispatch"),
    )
    await _set_postgres_user_active(
        test_db_pool,
        user_id=user_id,
        value=False,
    )
    adapter, captured_headers = _capture_real_openai_adapter_headers(monkeypatch)
    adapter_calls = 0

    async def resolve_and_dispatch():
        nonlocal adapter_calls
        resolved = await resolve_byok_credentials(
            "openai",
            user_id=user_id,
            team_ids=[],
            org_ids=[],
            server_config_snapshot={},
        )
        adapter_calls += 1
        return adapter.chat(
            {
                "messages": [{"role": "user", "content": "hello"}],
                "model": "gpt-test",
                "api_key": resolved.api_key,
                "app_config": resolved.app_config,
            }
        )

    with pytest.raises(ByokResolutionError) as exc_info:
        await resolve_and_dispatch()

    assert vars(exc_info.value) == {
        "code": "invalid_provider_credentials",
        "provider": "openai",
    }
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert adapter_calls == 0
    assert captured_headers == []
