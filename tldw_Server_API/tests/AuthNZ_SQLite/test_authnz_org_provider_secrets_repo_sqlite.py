from __future__ import annotations

import asyncio
import base64
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from sqlite3 import IntegrityError

import pytest

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import ProfileUserWriteRejected

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


@pytest.fixture
async def shared_repo_state(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )
    from tldw_Server_API.tests.AuthNZ_SQLite.test_byok_endpoints_sqlite import (
        _setup_byok_sqlite,
    )

    state = await _setup_byok_sqlite(tmp_path, monkeypatch)
    return state, AuthnzOrgProviderSecretsRepo(state["pool"])


async def _insert_shared_row(
    pool,
    *,
    scope_type: str,
    scope_id: int,
    provider: str,
    encrypted_blob: str,
    revoked_at: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await pool.execute(
        """
        INSERT INTO org_provider_secrets (
            scope_type, scope_id, provider, encrypted_blob, key_hint,
            created_at, updated_at, revoked_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            scope_type,
            scope_id,
            provider,
            encrypted_blob,
            provider,
            now,
            now,
            revoked_at,
        ),
    )


def _scope_id(state, scope_type: str) -> int:
    return int(state[scope_type]["id"])


class _SQLiteMutationGatePool:
    """Coordinate old split-statement and new transactional mutation paths."""

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

    @asynccontextmanager
    async def transaction(self):
        if self.role == "upsert":
            self.upsert_attempted.set()
        async with self.delegate.transaction() as conn:
            if self.role == "revoke":
                self.revoke_ready.set()
                await self.release_revoke.wait()
            yield conn

    async def fetchone(self, query: str, *args):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_shared_alias_write_uses_canonical_provider(shared_repo_state, scope_type):
    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)

    row = await repo.upsert_secret(
        scope_type=scope_type,
        scope_id=scope_id,
        provider="oai",
        encrypted_blob="canonical-write",
        key_hint="write",
        metadata=None,
        updated_at=datetime.now(timezone.utc),
    )

    assert row["provider"] == "openai"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_shared_fetch_prefers_canonical_over_legacy_alias(shared_repo_state, scope_type):
    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai",
        encrypted_blob="canonical",
    )
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="oai",
        encrypted_blob="legacy",
    )

    row = await repo.fetch_secret(scope_type, scope_id, "oai")

    assert row is not None
    assert row["encrypted_blob"] == "canonical"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_shared_fetch_reads_one_legacy_alias(shared_repo_state, scope_type):
    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="oai",
        encrypted_blob="legacy",
    )

    row = await repo.fetch_secret(scope_type, scope_id, "openai")

    assert row is not None
    assert row["provider"] == "oai"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disabled_boundary",
    [
        "team_membership",
        "team",
        "team_org",
        "org_membership",
        "org",
    ],
)
async def test_authorized_shared_fetch_is_atomic_across_active_scope_boundaries(
    shared_repo_state,
    disabled_boundary,
) -> None:
    """One repository read binds the secret to current owner and entity activity."""

    state, repo = shared_repo_state
    scope_type = "team" if disabled_boundary.startswith("team") else "org"
    scope_id = _scope_id(state, scope_type)
    user_id = int(state["user"]["id"])
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai",
        encrypted_blob="authorized-secret",
    )

    authorized = await repo.fetch_authorized_secret_for_user(
        scope_type,
        scope_id,
        user_id,
        "openai",
    )
    assert authorized is not None
    assert authorized["encrypted_blob"] == "authorized-secret"
    assert await repo.fetch_authorized_secret_for_user(
        scope_type,
        scope_id,
        user_id + 999,
        "openai",
    ) is None

    if disabled_boundary == "team_membership":
        await state["pool"].execute(
            "UPDATE team_members SET status = 'suspended' WHERE team_id = ? AND user_id = ?",
            (scope_id, user_id),
        )
    elif disabled_boundary == "team":
        await state["pool"].execute(
            "UPDATE teams SET is_active = 0 WHERE id = ?",
            (scope_id,),
        )
    elif disabled_boundary == "team_org":
        await state["pool"].execute(
            "UPDATE organizations SET is_active = 0 WHERE id = ?",
            (int(state["org"]["id"]),),
        )
    elif disabled_boundary == "org_membership":
        await state["pool"].execute(
            "UPDATE org_members SET status = 'suspended' WHERE org_id = ? AND user_id = ?",
            (scope_id, user_id),
        )
    else:
        await state["pool"].execute(
            "UPDATE organizations SET is_active = 0 WHERE id = ?",
            (scope_id,),
        )

    assert await repo.fetch_authorized_secret_for_user(
        scope_type,
        scope_id,
        user_id,
        "openai",
    ) is None


@pytest.mark.asyncio
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
async def test_authorized_shared_fetch_rejects_null_activity_boundaries_sqlite(
    shared_repo_state,
    null_boundary: str,
) -> None:
    """Legacy NULL activity state must never authorize a shared secret."""

    state, repo = shared_repo_state
    scope_type = "team" if null_boundary.startswith("team") else "org"
    scope_id = _scope_id(state, scope_type)
    user_id = int(state["user"]["id"])
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai",
        encrypted_blob="opaque-test-payload",
    )
    assert await repo.fetch_authorized_secret_for_user(
        scope_type,
        scope_id,
        user_id,
        "openai",
    ) is not None

    if null_boundary in {"team_user", "org_user"}:
        # The managed write boundary rejects this legacy state before SQLite;
        # PostgreSQL coverage below exercises the nullable-row join directly.
        with pytest.raises(ProfileUserWriteRejected):
            await state["pool"].execute(
                "UPDATE users SET is_active = NULL WHERE id = ?",
                (user_id,),
            )
        return
    elif null_boundary == "team_membership":
        await state["pool"].execute(
            "UPDATE team_members SET status = NULL WHERE team_id = ? AND user_id = ?",
            (scope_id, user_id),
        )
    elif null_boundary == "team":
        await state["pool"].execute(
            "UPDATE teams SET is_active = NULL WHERE id = ?",
            (scope_id,),
        )
    elif null_boundary == "team_org":
        await state["pool"].execute(
            "UPDATE organizations SET is_active = NULL WHERE id = ?",
            (int(state["org"]["id"]),),
        )
    elif null_boundary == "org_membership":
        await state["pool"].execute(
            "UPDATE org_members SET status = NULL WHERE org_id = ? AND user_id = ?",
            (scope_id, user_id),
        )
    else:
        await state["pool"].execute(
            "UPDATE organizations SET is_active = NULL WHERE id = ?",
            (scope_id,),
        )

    assert await repo.fetch_authorized_secret_for_user(
        scope_type,
        scope_id,
        user_id,
        "openai",
    ) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_shared_fetch_and_list_cover_accepted_underscore_alias(
    shared_repo_state,
    scope_type,
):
    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="aws_bedrock",
        encrypted_blob="accepted-underscore-alias",
    )

    row = await repo.fetch_secret(scope_type, scope_id, "bedrock")
    assert row is not None
    assert row["provider"] == "aws_bedrock"
    listed = await repo.list_secrets(scope_type=scope_type, scope_id=scope_id)
    assert [(item["provider"], item["key_hint"]) for item in listed] == [
        ("bedrock", "aws_bedrock")
    ]

    revoked_at = datetime.now(timezone.utc).isoformat()
    await state["pool"].execute(
        """
        UPDATE org_provider_secrets
        SET revoked_at = ?, updated_at = ?
        WHERE scope_type = ? AND scope_id = ? AND provider = ?
        """,
        (revoked_at, revoked_at, scope_type, scope_id, "aws_bedrock"),
    )
    assert await repo.fetch_secret(scope_type, scope_id, "bedrock") is None
    tombstone = await repo.fetch_secret(
        scope_type,
        scope_id,
        "bedrock",
        include_revoked=True,
    )
    assert tombstone is not None
    assert tombstone["provider"] == "aws_bedrock"
    assert await repo.list_secrets(scope_type=scope_type, scope_id=scope_id) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_shared_fetch_rejects_conflicting_legacy_aliases(shared_repo_state, scope_type):
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        ProviderCredentialAliasConflictError,
    )

    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    for provider in ("custom-openai", "openai-compatible"):
        await _insert_shared_row(
            state["pool"],
            scope_type=scope_type,
            scope_id=scope_id,
            provider=provider,
            encrypted_blob=provider,
        )

    with pytest.raises(ProviderCredentialAliasConflictError):
        await repo.fetch_secret(scope_type, scope_id, "custom-openai-api")


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_revoked_canonical_shared_row_blocks_active_legacy_alias(shared_repo_state, scope_type):
    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai",
        encrypted_blob="revoked-canonical",
        revoked_at=datetime.now(timezone.utc).isoformat(),
    )
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="oai",
        encrypted_blob="active-legacy",
    )

    assert await repo.fetch_secret(scope_type, scope_id, "oai") is None
    revoked = await repo.fetch_secret(scope_type, scope_id, "oai", include_revoked=True)
    assert revoked is not None
    assert revoked["encrypted_blob"] == "revoked-canonical"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_authorized_revoked_canonical_shared_row_blocks_active_legacy_alias(
    shared_repo_state,
    scope_type,
):
    """An authorized read must preserve canonical revocation authority."""

    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    user_id = int(state["user"]["id"])
    revoked_at = datetime.now(timezone.utc).isoformat()
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="custom-openai-api",
        encrypted_blob="revoked-canonical",
        revoked_at=revoked_at,
    )
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai-compatible",
        encrypted_blob="active-legacy",
    )

    row = await repo.fetch_authorized_secret_for_user(
        scope_type,
        scope_id,
        user_id,
        "openai-compatible",
    )

    assert row is not None
    assert row["provider"] == "custom-openai-api"
    assert row["encrypted_blob"] == "revoked-canonical"
    assert row["revoked_at"] == revoked_at


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_authorized_shared_fetch_prefers_active_canonical_over_legacy_alias(
    shared_repo_state,
    scope_type,
):
    """Authorization joins must preserve canonical provider precedence."""

    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    user_id = int(state["user"]["id"])
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="custom-openai-api",
        encrypted_blob="canonical",
    )
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai-compatible",
        encrypted_blob="legacy",
    )

    row = await repo.fetch_authorized_secret_for_user(
        scope_type,
        scope_id,
        user_id,
        "openai-compatible",
    )

    assert row is not None
    assert row["provider"] == "custom-openai-api"
    assert row["encrypted_blob"] == "canonical"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_authorized_shared_fetch_reads_single_legacy_alias(
    shared_repo_state,
    scope_type,
):
    """One legacy row remains readable when no canonical row exists."""

    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    user_id = int(state["user"]["id"])
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai-compatible",
        encrypted_blob="legacy",
    )

    row = await repo.fetch_authorized_secret_for_user(
        scope_type,
        scope_id,
        user_id,
        "custom-openai-api",
    )

    assert row is not None
    assert row["provider"] == "openai-compatible"
    assert row["encrypted_blob"] == "legacy"


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_authorized_shared_alias_lookup_uses_one_sqlite_snapshot_during_rotation(
    shared_repo_state,
):
    """A concurrent canonical rotation cannot split an alias lookup across statements."""
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )

    state, _repo = shared_repo_state
    scope_type = "team"
    scope_id = _scope_id(state, scope_type)
    user_id = int(state["user"]["id"])
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai-compatible",
        encrypted_blob="legacy-before-rotation",
    )

    snapshot_read = asyncio.Event()
    release_lookup = asyncio.Event()

    class _CoordinatedPool:
        pool = None

        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.statement_count = 0
            self.query = ""

        async def fetchall(self, query: str, params: tuple[object, ...]):
            self.statement_count += 1
            self.query = query
            rows = await self.delegate.fetchall(query, params)
            snapshot_read.set()
            await release_lookup.wait()
            return rows

        async def fetchone(self, query: str, params: tuple[object, ...]):
            self.statement_count += 1
            self.query = query
            row = await self.delegate.fetchone(query, params)
            if self.statement_count == 1:
                snapshot_read.set()
                await release_lookup.wait()
            return row

    coordinated_pool = _CoordinatedPool(state["pool"])
    repo = AuthnzOrgProviderSecretsRepo(coordinated_pool)
    lookup = asyncio.create_task(
        repo.fetch_authorized_secret_for_user(
            scope_type,
            scope_id,
            user_id,
            "custom-openai-api",
        )
    )

    await asyncio.wait_for(snapshot_read.wait(), timeout=1)
    try:
        await _insert_shared_row(
            state["pool"],
            scope_type=scope_type,
            scope_id=scope_id,
            provider="custom-openai-api",
            encrypted_blob="revoked-canonical-after-snapshot",
            revoked_at=datetime.now(timezone.utc).isoformat(),
        )
    finally:
        release_lookup.set()

    row = await asyncio.wait_for(lookup, timeout=1)

    assert coordinated_pool.statement_count == 1
    assert "s.provider IN (" in coordinated_pool.query
    assert row is not None
    assert row["provider"] == "openai-compatible"
    assert row["encrypted_blob"] == "legacy-before-rotation"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_authorized_shared_fetch_rejects_conflicting_legacy_aliases(
    shared_repo_state,
    scope_type,
):
    """Multiple authorized legacy rows remain an explicit fail-closed conflict."""
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        ProviderCredentialAliasConflictError,
    )

    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    user_id = int(state["user"]["id"])
    for provider in ("custom-openai", "openai-compatible"):
        await _insert_shared_row(
            state["pool"],
            scope_type=scope_type,
            scope_id=scope_id,
            provider=provider,
            encrypted_blob=provider,
        )

    with pytest.raises(ProviderCredentialAliasConflictError):
        await repo.fetch_authorized_secret_for_user(
            scope_type,
            scope_id,
            user_id,
            "custom-openai-api",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_shared_legacy_alias_row_can_be_touched(shared_repo_state, scope_type):
    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="oai",
        encrypted_blob="legacy",
    )
    used_at = datetime.now(timezone.utc)

    await repo.touch_last_used(scope_type, scope_id, "openai", used_at)

    row = await state["pool"].fetchone(
        "SELECT last_used_at FROM org_provider_secrets WHERE scope_type = ? AND scope_id = ? AND provider = ?",
        (scope_type, scope_id, "oai"),
    )
    assert row is not None
    assert row["last_used_at"] == used_at.isoformat()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_shared_legacy_alias_row_can_be_revoked(shared_repo_state, scope_type):
    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="oai",
        encrypted_blob="legacy",
    )

    assert await repo.delete_secret(scope_type, scope_id, "openai")
    assert await repo.fetch_secret(scope_type, scope_id, "openai") is None


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize("owner_kind", ["user", "org"])
async def test_alias_revoke_and_canonical_upsert_serialize_sqlite(
    shared_repo_state,
    owner_kind: str,
) -> None:
    """One logical provider identity cannot retain a legacy row after a race."""
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import (
        AuthnzUserProviderSecretsRepo,
    )

    state, _repo = shared_repo_state
    pool = state["pool"]
    user_id = int(state["user"]["id"])
    scope_id = int(state["org"]["id"])
    now = datetime.now(timezone.utc)
    if owner_kind == "user":
        await pool.execute(
            """
            INSERT INTO user_provider_secrets (
                user_id, provider, encrypted_blob, key_hint, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, "oai", "legacy", "legacy", now.isoformat(), now.isoformat()),
        )
    else:
        await _insert_shared_row(
            pool,
            scope_type="org",
            scope_id=scope_id,
            provider="oai",
            encrypted_blob="legacy",
        )

    revoke_ready = asyncio.Event()
    release_revoke = asyncio.Event()
    upsert_attempted = asyncio.Event()
    revoke_pool = _SQLiteMutationGatePool(
        pool,
        role="revoke",
        revoke_ready=revoke_ready,
        release_revoke=release_revoke,
        upsert_attempted=upsert_attempted,
    )
    upsert_pool = _SQLiteMutationGatePool(
        pool,
        role="upsert",
        revoke_ready=revoke_ready,
        release_revoke=release_revoke,
        upsert_attempted=upsert_attempted,
    )

    if owner_kind == "user":
        revoke_repo = AuthnzUserProviderSecretsRepo(revoke_pool)
        upsert_repo = AuthnzUserProviderSecretsRepo(upsert_pool)
        revoke_task = asyncio.create_task(
            revoke_repo.delete_secret(user_id, "openai", revoked_by=user_id)
        )

        async def upsert():
            return await upsert_repo.upsert_secret(
                user_id=user_id,
                provider="openai",
                encrypted_blob="canonical",
                key_hint="canonical",
                metadata=None,
                updated_at=now,
                created_by=user_id,
                updated_by=user_id,
            )

        final_query = (
            "SELECT provider, revoked_at FROM user_provider_secrets "
            "WHERE user_id = ? ORDER BY provider"
        )
        final_params = (user_id,)
    else:
        revoke_repo = AuthnzOrgProviderSecretsRepo(revoke_pool)
        upsert_repo = AuthnzOrgProviderSecretsRepo(upsert_pool)
        revoke_task = asyncio.create_task(
            revoke_repo.delete_secret("org", scope_id, "openai", revoked_by=user_id)
        )

        async def upsert():
            return await upsert_repo.upsert_secret(
                scope_type="org",
                scope_id=scope_id,
                provider="openai",
                encrypted_blob="canonical",
                key_hint="canonical",
                metadata=None,
                updated_at=now,
                created_by=user_id,
                updated_by=user_id,
            )

        final_query = (
            "SELECT provider, revoked_at FROM org_provider_secrets "
            "WHERE scope_type = ? AND scope_id = ? ORDER BY provider"
        )
        final_params = ("org", scope_id)

    upsert_task: asyncio.Task | None = None
    try:
        await asyncio.wait_for(revoke_ready.wait(), timeout=1)
        upsert_task = asyncio.create_task(upsert())
        await asyncio.wait_for(upsert_attempted.wait(), timeout=1)
        release_revoke.set()
        revoked, written = await asyncio.wait_for(
            asyncio.gather(revoke_task, upsert_task),
            timeout=5,
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
    rows = await pool.fetchall(final_query, final_params)
    assert [(row["provider"], row["revoked_at"]) for row in rows] == [
        ("openai", None)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_shared_list_folds_canonical_and_single_legacy_rows(shared_repo_state, scope_type):
    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai",
        encrypted_blob="canonical",
    )
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="oai",
        encrypted_blob="shadowed-legacy",
    )
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai-compatible",
        encrypted_blob="single-legacy",
    )

    rows = await repo.list_secrets(scope_type=scope_type, scope_id=scope_id)
    by_provider = {row["provider"]: row for row in rows}

    assert set(by_provider) == {"openai", "custom-openai-api"}
    assert by_provider["openai"]["key_hint"] == "openai"
    assert by_provider["custom-openai-api"]["key_hint"] == "openai-compatible"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_shared_list_treats_revoked_canonical_as_authoritative(shared_repo_state, scope_type):
    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="openai",
        encrypted_blob="revoked-canonical",
        revoked_at=datetime.now(timezone.utc).isoformat(),
    )
    await _insert_shared_row(
        state["pool"],
        scope_type=scope_type,
        scope_id=scope_id,
        provider="oai",
        encrypted_blob="active-legacy",
    )

    assert await repo.list_secrets(scope_type=scope_type, scope_id=scope_id) == []
    rows = await repo.list_secrets(
        scope_type=scope_type,
        scope_id=scope_id,
        include_revoked=True,
    )
    assert len(rows) == 1
    assert rows[0]["provider"] == "openai"
    assert rows[0]["key_hint"] == "openai"


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["team", "org"])
async def test_shared_list_rejects_conflicting_legacy_aliases(shared_repo_state, scope_type):
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        ProviderCredentialAliasConflictError,
    )

    state, repo = shared_repo_state
    scope_id = _scope_id(state, scope_type)
    for provider in ("custom-openai", "openai-compatible"):
        await _insert_shared_row(
            state["pool"],
            scope_type=scope_type,
            scope_id=scope_id,
            provider=provider,
            encrypted_blob=provider,
        )

    with pytest.raises(ProviderCredentialAliasConflictError):
        await repo.list_secrets(scope_type=scope_type, scope_id=scope_id)


@pytest.mark.asyncio
async def test_org_provider_secrets_repo_sqlite(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import (
        AuthnzOrgProviderSecretsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        build_secret_payload,
        encrypt_byok_payload,
        key_hint_for_api_key,
    )
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    users_db = UsersDB(pool)
    await users_db.initialize()
    created_user = await users_db.create_user(
        username="byok-org",
        email="byok-org@example.com",
        password_hash="hashed-password",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
        uuid_value=uuid.uuid4(),
    )
    user_id = int(created_user["id"])

    repo = AuthnzOrgProviderSecretsRepo(pool)
    await repo.ensure_tables()

    payload = build_secret_payload("sk-test", {"org_id": "org-1"})
    envelope = encrypt_byok_payload(payload)
    encrypted_blob = json.dumps(envelope)
    key_hint = key_hint_for_api_key("sk-test")
    now = datetime.now(timezone.utc)

    await repo.upsert_secret(
        scope_type="org",
        scope_id=1,
        provider="OpenAI",
        encrypted_blob=encrypted_blob,
        key_hint=key_hint,
        metadata={"label": "org-shared"},
        updated_at=now,
        created_by=user_id,
        updated_by=user_id,
    )

    row = await repo.fetch_secret("org", 1, "openai")
    assert row is not None
    assert row["provider"] == "openai"
    assert row["encrypted_blob"] == encrypted_blob
    assert row["key_hint"] == key_hint
    assert row["created_by"] == user_id
    assert row["updated_by"] == user_id

    items = await repo.list_secrets(scope_type="org", scope_id=1)
    assert len(items) == 1
    assert items[0]["provider"] == "openai"

    items_filtered = await repo.list_secrets(scope_type="org", scope_id=1, provider="openai")
    assert len(items_filtered) == 1

    await repo.touch_last_used("org", 1, "openai", now)
    refreshed = await repo.fetch_secret("org", 1, "openai")
    assert refreshed is not None
    assert refreshed["last_used_at"] is not None

    deleted = await repo.delete_secret("org", 1, "openai")
    assert deleted
    missing = await repo.fetch_secret("org", 1, "openai")
    assert missing is None
    revoked_rows = await repo.list_secrets(scope_type="org", scope_id=1, include_revoked=True)
    assert len(revoked_rows) == 1
    assert revoked_rows[0]["revoked_at"] is not None
