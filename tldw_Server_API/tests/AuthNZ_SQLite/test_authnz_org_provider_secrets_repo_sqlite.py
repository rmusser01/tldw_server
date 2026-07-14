from __future__ import annotations

from datetime import datetime, timezone
import base64
import json
import uuid

import pytest

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
