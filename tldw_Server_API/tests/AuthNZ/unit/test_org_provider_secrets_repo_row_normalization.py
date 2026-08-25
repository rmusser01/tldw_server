from __future__ import annotations

import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.exceptions import UserRegistrationException
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    MembershipAuthority,
    MembershipAuthorizationError,
    MembershipWriter,
)
from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    ProviderCredentialAliasConflictError,
)


def test_provider_alias_conflict_survives_authnz_transaction_boundary() -> None:
    assert issubclass(ProviderCredentialAliasConflictError, ValueError)
    assert issubclass(
        ProviderCredentialAliasConflictError,
        UserRegistrationException,
    )


class _AuthorizationConnection:
    def __init__(self, *, membership_status: str = "active") -> None:
        self.membership_status = membership_status
        self.queries: list[str] = []

    async def execute(self, query: str, *_args):
        self.queries.append(" ".join(query.split()))
        return "SELECT 1"

    async def fetch(self, query: str, *_args):
        self.queries.append(" ".join(query.split()))
        return []

    async def fetchrow(self, query: str, *_args):
        normalized = " ".join(query.split())
        self.queries.append(normalized)
        if "FROM public.users" in normalized:
            return {
                "id": 7,
                "is_active": True,
                "is_superuser": False,
                "role": "user",
            }
        if "FROM public.organizations" in normalized:
            return {"id": 9, "is_active": True}
        if "FROM public.teams" in normalized:
            return {"id": 12, "org_id": 9, "is_active": True}
        if "FROM public.org_members" in normalized:
            return {
                "role": "admin",
                "status": self.membership_status,
            }
        if "FROM public.team_members" in normalized:
            return {"role": "admin", "status": "active"}
        if normalized.startswith("INSERT INTO public.org_provider_secrets"):
            return {
                "id": 1,
                "scope_type": "org",
                "scope_id": 9,
                "provider": "openai",
            }
        return None


class _AuthorizationPool:
    pool = object()

    def __init__(self, conn: _AuthorizationConnection) -> None:
        self.conn = conn

    @asynccontextmanager
    async def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ):
        assert acquire_timeout_seconds is not None
        yield self.conn


@pytest.mark.asyncio
async def test_shared_secret_upsert_reauthorizes_manager_under_total_lock_order() -> None:
    conn = _AuthorizationConnection()
    repo = AuthnzOrgProviderSecretsRepo(_AuthorizationPool(conn))  # type: ignore[arg-type]

    await repo.upsert_secret(
        scope_type="org",
        scope_id=9,
        provider="openai",
        encrypted_blob="encrypted",
        key_hint=None,
        metadata=None,
        updated_at=datetime.now(timezone.utc),
        authorization_context=ActorMembershipWriteContext(
            actor_user_id=7,
            required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
        ),
    )

    user_lock = next(
        index for index, query in enumerate(conn.queries) if "public.users" in query
    )
    org_lock = next(
        index
        for index, query in enumerate(conn.queries)
        if "public.organizations" in query
    )
    membership_lock = next(
        index
        for index, query in enumerate(conn.queries)
        if "public.org_members" in query
    )
    insert = next(
        index
        for index, query in enumerate(conn.queries)
        if query.startswith("INSERT INTO public.org_provider_secrets")
    )
    assert user_lock < org_lock < membership_lock < insert


@pytest.mark.asyncio
async def test_shared_secret_platform_admin_locks_persisted_authority_before_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _has_platform_admin(*_args, **_kwargs) -> bool:
        return True

    monkeypatch.setattr(
        MembershipWriter,
        "has_persisted_platform_admin",
        _has_platform_admin,
    )
    conn = _AuthorizationConnection()
    repo = AuthnzOrgProviderSecretsRepo(_AuthorizationPool(conn))  # type: ignore[arg-type]

    await repo.upsert_secret(
        scope_type="org",
        scope_id=9,
        provider="openai",
        encrypted_blob="encrypted",
        key_hint=None,
        metadata=None,
        updated_at=datetime.now(timezone.utc),
        authorization_context=ActorMembershipWriteContext(
            actor_user_id=7,
            required_authority=MembershipAuthority.PLATFORM_ADMIN,
        ),
    )

    org_lock = next(
        index
        for index, query in enumerate(conn.queries)
        if "public.organizations" in query
    )
    authority_locks = [
        index
        for index, query in enumerate(conn.queries)
        if "FOR UPDATE OF" in query
    ]
    insert = next(
        index
        for index, query in enumerate(conn.queries)
        if query.startswith("INSERT INTO public.org_provider_secrets")
    )
    assert len(authority_locks) == 5
    assert org_lock < authority_locks[0] < authority_locks[-1] < insert


@pytest.mark.asyncio
async def test_shared_secret_upsert_rejects_inactive_manager_under_lock() -> None:
    conn = _AuthorizationConnection(membership_status="inactive")
    repo = AuthnzOrgProviderSecretsRepo(_AuthorizationPool(conn))  # type: ignore[arg-type]

    with pytest.raises(MembershipAuthorizationError):
        await repo.upsert_secret(
            scope_type="org",
            scope_id=9,
            provider="openai",
            encrypted_blob="encrypted",
            key_hint=None,
            metadata=None,
            updated_at=datetime.now(timezone.utc),
            authorization_context=ActorMembershipWriteContext(
                actor_user_id=7,
                required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
            ),
        )

    assert not any(
        query.startswith("INSERT INTO public.org_provider_secrets")
        for query in conn.queries
    )


@pytest.mark.asyncio
async def test_shared_secret_delete_rejects_inactive_manager_under_lock() -> None:
    conn = _AuthorizationConnection(membership_status="inactive")
    repo = AuthnzOrgProviderSecretsRepo(_AuthorizationPool(conn))  # type: ignore[arg-type]

    with pytest.raises(MembershipAuthorizationError):
        await repo.delete_secret(
            scope_type="org",
            scope_id=9,
            provider="openai",
            authorization_context=ActorMembershipWriteContext(
                actor_user_id=7,
                required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
            ),
        )

    assert not any(
        query.startswith("UPDATE public.org_provider_secrets")
        for query in conn.queries
    )


class _ManagerReadConnection(_AuthorizationConnection):
    async def fetch(self, query: str, *_args):
        normalized = " ".join(query.split())
        self.queries.append(normalized)
        if "FROM public.org_provider_secrets" in normalized:
            return [
                {
                    "id": 1,
                    "scope_type": "org",
                    "scope_id": 9,
                    "provider": "openai",
                    "encrypted_blob": "encrypted",
                    "revoked_at": None,
                }
            ]
        return []


@pytest.mark.asyncio
async def test_shared_secret_manager_reads_reauthorize_in_read_transaction() -> None:
    conn = _ManagerReadConnection()
    repo = AuthnzOrgProviderSecretsRepo(_AuthorizationPool(conn))  # type: ignore[arg-type]
    context = ActorMembershipWriteContext(
        actor_user_id=7,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )

    listed = await repo.list_secrets_for_manager(
        scope_type="org",
        scope_id=9,
        authorization_context=context,
    )
    fetched = await repo.fetch_secret_for_manager(
        scope_type="org",
        scope_id=9,
        provider="openai",
        authorization_context=context,
    )

    assert [row["provider"] for row in listed] == ["openai"]
    assert fetched and fetched["encrypted_blob"] == "encrypted"
    membership_locks = [
        query for query in conn.queries if "FROM public.org_members" in query
    ]
    assert len(membership_locks) == 2
    assert all("FOR UPDATE" in query for query in membership_locks)


@pytest.mark.asyncio
async def test_team_manager_read_requires_active_parent_org_membership() -> None:
    conn = _ManagerReadConnection(membership_status="inactive")
    repo = AuthnzOrgProviderSecretsRepo(_AuthorizationPool(conn))  # type: ignore[arg-type]

    with pytest.raises(MembershipAuthorizationError):
        await repo.fetch_secret_for_manager(
            scope_type="team",
            scope_id=12,
            provider="openai",
            authorization_context=ActorMembershipWriteContext(
                actor_user_id=7,
                required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
            ),
        )

    assert not any(
        "FROM public.org_provider_secrets" in query for query in conn.queries
    )


class _RowLike:
    def __init__(self, data: dict[str, object]) -> None:
        self._data = dict(data)

    def keys(self):
        return list(self._data.keys())

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        # Mimic sqlite3.Row iteration-by-values to guard against regression.
        return iter(self._data.values())


class _FakePool:
    pool = None

    async def fetchall(self, *_args, **_kwargs):
        return [
            _RowLike(
                {
                    "id": 1,
                    "scope_type": "org",
                    "scope_id": 9,
                    "provider": "openai",
                    "key_hint": "1234",
                }
            )
        ]


@pytest.mark.asyncio
async def test_list_secrets_normalizes_row_like_objects() -> None:
    repo = AuthnzOrgProviderSecretsRepo(db_pool=_FakePool())  # type: ignore[arg-type]
    rows = await repo.list_secrets(scope_type="org", scope_id=9)
    assert rows == [
        {
            "id": 1,
            "scope_type": "org",
            "scope_id": 9,
            "provider": "openai",
            "key_hint": "1234",
        }
    ]


class _PostgresQualificationConnection:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries

    def _record(self, query: str) -> None:
        self.queries.append(query)

    async def execute(self, query: str, *_args):
        self._record(query)
        return "UPDATE 1"

    async def fetch(self, query: str, *_args):
        self._record(query)
        return []

    async def fetchrow(self, query: str, *_args):
        self._record(query)
        normalized = " ".join(query.split())
        if "FROM public.organizations" in normalized:
            return {"id": 9, "is_active": True}
        if normalized.startswith("INSERT INTO"):
            return {
                "id": 1,
                "scope_type": "org",
                "scope_id": 9,
                "provider": "openai",
                "key_hint": "1234",
            }
        return None


class _PostgresQualificationPool:
    pool = object()

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.connection = _PostgresQualificationConnection(self.queries)

    @asynccontextmanager
    async def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ):
        assert acquire_timeout_seconds is not None
        yield self.connection

    async def fetchall(self, query: str, *_args):
        self.queries.append(query)
        return [
            {
                "id": 1,
                "scope_type": "org",
                "scope_id": 9,
                "provider": "openai",
                "encrypted_blob": "encrypted",
                "key_hint": "1234",
                "metadata": None,
                "created_at": None,
                "updated_at": None,
                "last_used_at": None,
                "created_by": 7,
                "updated_by": 7,
                "revoked_by": None,
                "revoked_at": None,
            }
        ]

    async def execute(self, query: str, *_args):
        self.queries.append(query)
        return "UPDATE 1"


@pytest.mark.asyncio
async def test_postgres_secret_queries_are_public_qualified() -> None:
    pool = _PostgresQualificationPool()
    repo = AuthnzOrgProviderSecretsRepo(db_pool=pool)  # type: ignore[arg-type]
    now = datetime.now(timezone.utc)

    await repo.upsert_secret(
        scope_type="org",
        scope_id=9,
        provider="openai",
        encrypted_blob="encrypted",
        key_hint="1234",
        metadata=None,
        updated_at=now,
        created_by=7,
        updated_by=7,
    )
    await repo.fetch_secret("org", 9, "openai")
    await repo.fetch_authorized_secret_for_user("org", 9, 7, "openai")
    await repo.fetch_authorized_secret_for_user("team", 9, 7, "openai")
    await repo.list_secrets(scope_type="org", scope_id=9)
    await repo.delete_secret("org", 9, "openai", revoked_by=7)
    await repo.touch_last_used("org", 9, "openai", now)

    secret_queries = [
        query
        for query in pool.queries
        if re.search(r"\borg_provider_secrets\b", query, re.IGNORECASE)
        and "pg_advisory_xact_lock" not in query
    ]
    assert secret_queries
    assert all(
        re.search(r"\bpublic\.org_provider_secrets\b", query, re.IGNORECASE)
        for query in secret_queries
    )
    authorized_queries = [query for query in secret_queries if " JOIN " in query]
    assert authorized_queries
    assert all("public.org_members" in query for query in authorized_queries)
    assert any("public.team_members" in query for query in authorized_queries)
    assert all("public.organizations" in query for query in authorized_queries)
    assert all("public.users" in query for query in authorized_queries)
