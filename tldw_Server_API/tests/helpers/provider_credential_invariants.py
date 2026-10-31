"""Storage invariants for provider credentials, shared by the SQLite and Postgres suites (TASK-13950.17)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.AuthNZ.repos.org_provider_secrets_repo import AuthnzOrgProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    key_hint_for_api_key,
    loads_envelope,
)

PLAINTEXT = "sk-t13950-17-invariant-0001"
# Written by the pre-move code (0d270b00e4) with BYOK_ENCRYPTION_KEY = base64(b"k" * 32).
GOLDEN_ENVELOPE = (
    '{"_enc": "aesgcm:v1", "nonce": "MRxmSm87PadxTUYn", '
    '"ct": "FJf6UufOTYiOtvbRSWKiv0Qh8K/pfvZQT7Mpplz6mUxheW+XH727k315", "tag": "y9c2ilJnbPceEmTDbCkJVg=="}'
)


def _blob() -> str:
    return dumps_envelope(encrypt_byok_payload(build_secret_payload(PLAINTEXT)))


def _assert_encrypted(blob: Any) -> None:
    assert isinstance(blob, str) and blob
    assert PLAINTEXT not in blob
    assert decrypt_byok_payload(loads_envelope(blob))["api_key"] == PLAINTEXT


def assert_pre_move_envelope_decrypts() -> None:
    assert decrypt_byok_payload(loads_envelope(GOLDEN_ENVELOPE)) == {"api_key": PLAINTEXT}


async def assert_user_secret_encrypted_and_owner_scoped(pool: Any, *, owner_id: int, other_id: int) -> None:
    repo = AuthnzUserProviderSecretsRepo(pool)
    await repo.ensure_tables()
    await repo.upsert_secret(
        user_id=owner_id,
        provider="openai",
        encrypted_blob=_blob(),
        key_hint=key_hint_for_api_key(PLAINTEXT),
        metadata=None,
        updated_at=datetime.now(timezone.utc),
    )
    raw = await pool.fetchval(
        "SELECT encrypted_blob FROM user_provider_secrets WHERE user_id = ? AND provider = ?", owner_id, "openai"
    )
    _assert_encrypted(raw)
    assert (await repo.fetch_secret_for_user(owner_id, "openai"))["encrypted_blob"] == raw
    assert await repo.fetch_secret_for_user(other_id, "openai") is None
    assert await repo.list_secrets_for_user(other_id) == []


async def assert_org_secret_encrypted_and_scope_bound(pool: Any, *, org_id: int, other_org_id: int) -> None:
    repo = AuthnzOrgProviderSecretsRepo(pool)
    await repo.ensure_tables()
    await repo.upsert_secret(
        scope_type="org",
        scope_id=org_id,
        provider="openai",
        encrypted_blob=_blob(),
        key_hint=key_hint_for_api_key(PLAINTEXT),
        metadata=None,
        updated_at=datetime.now(timezone.utc),
    )
    raw = await pool.fetchval(
        "SELECT encrypted_blob FROM org_provider_secrets WHERE scope_type = ? AND scope_id = ? AND provider = ?",
        "org",
        org_id,
        "openai",
    )
    _assert_encrypted(raw)
    assert await repo.fetch_secret("org", other_org_id, "openai") is None
    assert await repo.fetch_secret("team", org_id, "openai") is None
    assert await repo.list_secrets(scope_type="org", scope_id=other_org_id) == []
