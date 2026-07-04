from __future__ import annotations

from datetime import datetime, timezone
import base64
import json

import pytest

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


@pytest.mark.asyncio
async def test_llm_provider_overrides_repo_sqlite(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.llm_provider_overrides_repo import (
        AuthnzLLMProviderOverridesRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        build_secret_payload,
        encrypt_byok_payload,
        key_hint_for_api_key,
    )

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
