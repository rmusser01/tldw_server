from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


async def _open_sqlite_override_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.llm_provider_overrides_repo import (
        AuthnzLLMProviderOverridesRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))
    repo = AuthnzLLMProviderOverridesRepo(pool)
    await repo.ensure_tables()
    return repo


async def _insert_raw_override(
    repo: Any,
    *,
    provider: str,
    secret_blob: str | None = None,
    api_key_hint: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await repo.db_pool.execute(
        """
        INSERT INTO llm_provider_overrides (
            provider, secret_blob, api_key_hint, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (provider, secret_blob, api_key_hint, now, now),
    )


@pytest.mark.asyncio
async def test_llm_provider_overrides_repo_sqlite(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        build_secret_payload,
        encrypt_byok_payload,
        key_hint_for_api_key,
    )

    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)

    payload = build_secret_payload("sk-test", {"base_url": "https://example.com"})
    envelope = encrypt_byok_payload(payload)
    secret_blob = json.dumps(envelope)
    key_hint = key_hint_for_api_key("sk-test")
    now = datetime.now(timezone.utc)

    await repo.upsert_override(
        provider="OpenAI",
        is_enabled=True,
        allowed_models=json.dumps(["gpt-4o"]),
        config_json=json.dumps({"default_model": "gpt-4o"}),
        secret_blob=secret_blob,
        api_key_hint=key_hint,
        updated_at=now,
    )

    row = await repo.fetch_override("openai")
    assert row is not None
    assert row["provider"] == "openai"
    assert row["allowed_models"] is not None
    assert row["config_json"] is not None
    assert row["secret_blob"] == secret_blob
    assert row["api_key_hint"] == key_hint

    items = await repo.list_overrides()
    assert len(items) == 1
    assert items[0]["provider"] == "openai"

    deleted = await repo.delete_override("openai")
    assert deleted
    missing = await repo.fetch_override("openai")
    assert missing is None


@pytest.mark.asyncio
async def test_sqlite_patch_override_preserves_concurrent_unsupplied_columns(
    tmp_path,
    monkeypatch,
) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)
    await repo.upsert_override(
        provider="openai",
        is_enabled=True,
        allowed_models=None,
        config_json=None,
        secret_blob="old-secret",
        api_key_hint="old-hint",
        updated_at=now,
    )

    ready = 0
    both_ready = asyncio.Event()

    async def patch(fields: dict[str, Any]) -> dict[str, Any]:
        nonlocal ready
        ready += 1
        if ready == 2:
            both_ready.set()
        await both_ready.wait()
        return await repo.patch_override(
            provider="openai",
            fields=fields,
            updated_at=datetime.now(timezone.utc),
        )

    await asyncio.gather(
        patch({"secret_blob": None, "api_key_hint": None}),
        patch({"is_enabled": False}),
    )

    row = await repo.fetch_override("openai")
    assert row is not None
    assert row["is_enabled"] == 0
    assert row["secret_blob"] is None
    assert row["api_key_hint"] is None


@pytest.mark.asyncio
async def test_sqlite_patch_override_rejects_stale_secret_compare_and_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)
    now = datetime.now(timezone.utc)
    await repo.upsert_override(
        provider="openai",
        is_enabled=True,
        allowed_models=None,
        config_json=None,
        secret_blob="old-secret",
        api_key_hint="old-hint",
        updated_at=now,
    )

    conflict = await repo.patch_override(
        provider="openai",
        fields={"secret_blob": "new-secret", "api_key_hint": "new-hint"},
        updated_at=datetime.now(timezone.utc),
        compare_secret_blob=True,
        expected_secret_blob="stale-secret",
    )
    assert conflict is None

    updated = await repo.patch_override(
        provider="openai",
        fields={"secret_blob": "new-secret", "api_key_hint": "new-hint"},
        updated_at=datetime.now(timezone.utc),
        compare_secret_blob=True,
        expected_secret_blob="old-secret",
    )
    assert updated is not None
    assert updated["secret_blob"] == "new-secret"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input_time", "expected_time"),
    [
        (
            datetime(2026, 7, 14, 9, 30),
            datetime(2026, 7, 14, 9, 30, tzinfo=timezone.utc),
        ),
        (
            datetime(
                2026,
                7,
                14,
                9,
                30,
                tzinfo=timezone(timedelta(hours=-7)),
            ),
            datetime(2026, 7, 14, 16, 30, tzinfo=timezone.utc),
        ),
    ],
    ids=("naive-is-utc", "aware-offset-converts-to-utc"),
)
async def test_sqlite_override_timestamps_use_utc_storage_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_time: datetime,
    expected_time: datetime,
) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)

    created = await repo.upsert_override(
        provider="openai",
        is_enabled=True,
        allowed_models=None,
        config_json=None,
        secret_blob=None,
        api_key_hint=None,
        updated_at=input_time,
    )
    patched = await repo.patch_override(
        provider="openai",
        fields={"is_enabled": False},
        updated_at=input_time,
    )

    assert created["created_at"] == expected_time.isoformat()
    assert created["updated_at"] == expected_time.isoformat()
    assert patched is not None
    assert patched["updated_at"] == expected_time.isoformat()


@pytest.mark.asyncio
async def test_sqlite_list_folds_one_legacy_alias_to_canonical_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)
    await _insert_raw_override(
        repo,
        provider="oai",
        secret_blob="legacy-secret",
        api_key_hint="hint",
    )

    rows = await repo.list_overrides()

    assert len(rows) == 1
    assert rows[0]["provider"] == "openai"
    assert rows[0]["secret_blob"] == "legacy-secret"


@pytest.mark.asyncio
async def test_sqlite_patch_migrates_legacy_alias_without_losing_unsupplied_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)
    await _insert_raw_override(
        repo,
        provider="oai",
        secret_blob="legacy-secret",
        api_key_hint="legacy-hint",
    )

    patched = await repo.patch_override(
        provider="openai",
        fields={"is_enabled": False},
        updated_at=datetime.now(timezone.utc),
    )

    assert patched is not None
    assert patched["provider"] == "openai"
    assert patched["secret_blob"] == "legacy-secret"
    raw_rows = await repo.db_pool.fetchall(
        "SELECT provider, secret_blob FROM llm_provider_overrides ORDER BY provider"
    )
    assert [dict(row) for row in raw_rows] == [
        {"provider": "openai", "secret_blob": "legacy-secret"}
    ]


@pytest.mark.asyncio
async def test_sqlite_legacy_upsert_replaces_alias_with_one_canonical_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)
    await _insert_raw_override(
        repo,
        provider="oai",
        secret_blob="legacy-secret",
        api_key_hint="legacy-hint",
    )

    stored = await repo.upsert_override(
        provider="OAI",
        is_enabled=False,
        allowed_models=None,
        config_json=None,
        secret_blob="replacement-secret",
        api_key_hint="replacement-hint",
        updated_at=datetime.now(timezone.utc),
    )

    assert stored["provider"] == "openai"
    assert stored["secret_blob"] == "replacement-secret"
    raw_rows = await repo.db_pool.fetchall(
        "SELECT provider, secret_blob FROM llm_provider_overrides ORDER BY provider"
    )
    assert [dict(row) for row in raw_rows] == [
        {"provider": "openai", "secret_blob": "replacement-secret"}
    ]


@pytest.mark.asyncio
async def test_sqlite_legacy_alias_secret_cas_migrates_and_updates_canonical_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)
    await _insert_raw_override(
        repo,
        provider="oai",
        secret_blob="legacy-secret",
        api_key_hint="legacy-hint",
    )

    patched = await repo.patch_override(
        provider="openai",
        fields={"secret_blob": "new-secret", "api_key_hint": "new-hint"},
        updated_at=datetime.now(timezone.utc),
        compare_secret_blob=True,
        expected_secret_blob="legacy-secret",
    )

    assert patched is not None
    assert patched["provider"] == "openai"
    assert patched["secret_blob"] == "new-secret"
    raw_rows = await repo.db_pool.fetchall(
        "SELECT provider, secret_blob FROM llm_provider_overrides ORDER BY provider"
    )
    assert [dict(row) for row in raw_rows] == [
        {"provider": "openai", "secret_blob": "new-secret"}
    ]


@pytest.mark.asyncio
async def test_sqlite_concurrent_legacy_alias_patches_preserve_both_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)
    await _insert_raw_override(
        repo,
        provider="oai",
        secret_blob="legacy-secret",
        api_key_hint="legacy-hint",
    )
    ready = 0
    both_ready = asyncio.Event()

    async def patch(
        fields: dict[str, Any],
        *,
        compare_secret_blob: bool = False,
    ) -> dict[str, Any] | None:
        nonlocal ready
        ready += 1
        if ready == 2:
            both_ready.set()
        await both_ready.wait()
        return await repo.patch_override(
            provider="openai",
            fields=fields,
            updated_at=datetime.now(timezone.utc),
            compare_secret_blob=compare_secret_blob,
            expected_secret_blob="legacy-secret" if compare_secret_blob else None,
        )

    metadata_result, secret_result = await asyncio.gather(
        patch({"is_enabled": False}),
        patch(
            {"secret_blob": "new-secret", "api_key_hint": "new-hint"},
            compare_secret_blob=True,
        ),
    )

    assert metadata_result is not None
    assert secret_result is not None
    row = await repo.fetch_override("openai")
    assert row is not None
    assert row["is_enabled"] == 0
    assert row["secret_blob"] == "new-secret"
    raw_providers = await repo.db_pool.fetchall(
        "SELECT provider FROM llm_provider_overrides ORDER BY provider"
    )
    assert [dict(row) for row in raw_providers] == [{"provider": "openai"}]


@pytest.mark.asyncio
async def test_sqlite_list_prefers_canonical_row_over_legacy_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)
    await _insert_raw_override(
        repo,
        provider="oai",
        secret_blob="legacy-secret",
        api_key_hint="legacy",
    )
    await _insert_raw_override(
        repo,
        provider="openai",
        secret_blob="canonical-secret",
        api_key_hint="canonical",
    )

    rows = await repo.list_overrides()

    assert len(rows) == 1
    assert rows[0]["provider"] == "openai"
    assert rows[0]["secret_blob"] == "canonical-secret"


@pytest.mark.asyncio
async def test_sqlite_delete_alias_removes_canonical_and_legacy_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)
    await _insert_raw_override(repo, provider="oai", secret_blob="legacy-secret")
    await _insert_raw_override(repo, provider="openai", secret_blob="canonical-secret")

    deleted = await repo.delete_override("OAI")

    assert deleted is True
    assert await repo.list_overrides() == []


@pytest.mark.asyncio
async def test_sqlite_list_rejects_ambiguous_legacy_alias_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        ProviderCredentialAliasConflictError,
    )

    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)
    await _insert_raw_override(repo, provider="aws-bedrock", secret_blob="first")
    await _insert_raw_override(repo, provider="amazon-bedrock", secret_blob="second")

    with pytest.raises(ProviderCredentialAliasConflictError):
        await repo.list_overrides()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["voyage", "elevenlabs", "unknown-provider"])
async def test_sqlite_list_rejects_unsupported_stored_provider_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)
    await _insert_raw_override(repo, provider=provider, secret_blob="must-not-load")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        await repo.list_overrides()


@pytest.mark.asyncio
async def test_admin_alias_put_is_visible_to_canonical_credential_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.admin import admin_llm_providers
    from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
        LLMProviderOverrideRequest,
    )
    from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
        capture_provider_override_call_snapshot,
    )
    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
        ProviderCredentialRuntime,
    )
    from tldw_Server_API.app.services import admin_llm_providers_service

    repo = await _open_sqlite_override_repo(tmp_path, monkeypatch)

    async def ready() -> None:
        return None

    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: ready,
    )
    response = await admin_llm_providers.admin_upsert_llm_provider_override(
        "OAI",
        LLMProviderOverrideRequest(api_key="alias-admin-key"),
        admin_llm_providers_service=admin_llm_providers_service,
    )

    stored = await repo.list_overrides()
    assert response.provider == "openai"
    assert [row["provider"] for row in stored] == ["openai"]

    runtime = ProviderCredentialRuntime(
        user_id=None,
        team_ids=None,
        org_ids=None,
        trusted_base_url_override=False,
        server_config_snapshot={},
        override_snapshot_resolver=capture_provider_override_call_snapshot,
    )
    try:
        credentials = await runtime.resolve("oai")
        assert credentials.provider == "openai"
        assert credentials.api_key == "alias-admin-key"
    finally:
        await runtime.close()
