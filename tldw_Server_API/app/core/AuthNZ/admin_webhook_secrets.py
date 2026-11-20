from __future__ import annotations

import json
import os
from dataclasses import dataclass

from tldw_Server_API.app.core.config_sections.auth import api_key_env, jwt_secret_key_env, single_user_api_key_env
from tldw_Server_API.app.core.Security.crypto import (
    decrypt_json_blob_with_key,
    encrypt_json_blob_with_key,
)


@dataclass(frozen=True)
class EncryptedAdminWebhookSecret:
    encrypted_blob: str
    key_id: str


def _iter_admin_webhook_secret_keys() -> list[tuple[str, str]]:
    raw_candidates = [
        ("byok_primary", (os.getenv("BYOK_ENCRYPTION_KEY") or "").strip()),
        ("byok_secondary", (os.getenv("BYOK_SECONDARY_ENCRYPTION_KEY") or "").strip()),
        ("session_encryption", (os.getenv("SESSION_ENCRYPTION_KEY") or "").strip()),
        ("jwt_secret", (jwt_secret_key_env() or "").strip()),
        ("single_user_api_key", (single_user_api_key_env() or api_key_env() or "").strip()),
    ]
    candidates: list[tuple[str, str]] = []
    seen_values: set[str] = set()
    for key_id, value in raw_candidates:
        if not value or value in seen_values:
            continue
        seen_values.add(value)
        candidates.append((key_id, value))
    return candidates


def encrypt_admin_webhook_secret(secret: str) -> EncryptedAdminWebhookSecret:
    if not secret:
        raise ValueError("Webhook secret cannot be empty")

    candidates = _iter_admin_webhook_secret_keys()
    if not candidates:
        raise ValueError(
            "Admin webhook secret encryption requires BYOK_ENCRYPTION_KEY, "
            "SESSION_ENCRYPTION_KEY, JWT_SECRET_KEY, or SINGLE_USER_API_KEY"
        )

    key_id, key_value = candidates[0]
    envelope = encrypt_json_blob_with_key({"secret": secret}, key_value)
    if not envelope:
        raise ValueError("Failed to encrypt admin webhook secret")

    return EncryptedAdminWebhookSecret(
        encrypted_blob=json.dumps(envelope, sort_keys=True),
        key_id=key_id,
    )


def decrypt_admin_webhook_secret(encrypted_blob: str) -> str:
    if not encrypted_blob:
        raise ValueError("Encrypted admin webhook secret cannot be empty")

    try:
        envelope = json.loads(encrypted_blob)
    except json.JSONDecodeError as exc:
        raise ValueError("Encrypted admin webhook secret is not valid JSON") from exc

    for _key_id, key_value in _iter_admin_webhook_secret_keys():
        payload = decrypt_json_blob_with_key(envelope, key_value)
        if isinstance(payload, dict) and isinstance(payload.get("secret"), str):
            return payload["secret"]

    raise ValueError("Failed to decrypt admin webhook secret")
