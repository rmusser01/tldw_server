"""Encrypted Calendar external-account secret storage."""

from __future__ import annotations

import json
import os
from typing import Any

from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase
from tldw_Server_API.app.core.Security.crypto import (
    decrypt_json_blob_with_key,
    encrypt_json_blob_with_key,
)

# Environment variable name, not a secret value.
_CALENDAR_SECRET_KEY_ENV = "CALENDAR_SECRET_ENCRYPTION_KEY"  # nosec B105


class CalendarSecretStore:
    """Encrypts external account credentials before storing opaque DB refs."""

    def __init__(self, *, db: CalendarDatabase, tenant_id: str = "default") -> None:
        self.db = db
        self.tenant_id = tenant_id

    def create_secret(
        self,
        *,
        owner_user_id: int,
        provider: str,
        payload: dict[str, Any],
    ) -> str:
        key = self._required_encryption_key()
        envelope = encrypt_json_blob_with_key(payload, key)
        if envelope is None:
            raise CalendarValidationError("Calendar secret encryption is unavailable")
        return self.db.create_secret_ref(
            tenant_id=self.tenant_id,
            user_id=owner_user_id,
            provider=provider,
            encrypted_payload=json.dumps(envelope, sort_keys=True),
        )

    def resolve_secret(self, *, owner_user_id: int, secret_ref: str) -> dict[str, Any]:
        key = self._required_encryption_key()
        encrypted_payload = self.db.resolve_secret_ref_for_user(
            secret_ref,
            tenant_id=self.tenant_id,
            user_id=owner_user_id,
        )
        try:
            envelope = json.loads(encrypted_payload)
        except json.JSONDecodeError as exc:
            raise CalendarValidationError("Calendar secret payload is not a valid encryption envelope") from exc
        payload = decrypt_json_blob_with_key(envelope, key)
        if payload is None:
            raise CalendarValidationError("Calendar secret could not be decrypted")
        return payload

    def delete_secret(self, *, owner_user_id: int, secret_ref: str) -> bool:
        try:
            self.db.delete_secret_ref_for_user(
                secret_ref,
                tenant_id=self.tenant_id,
                user_id=owner_user_id,
            )
        except CalendarValidationError:
            return False
        return True

    @staticmethod
    def _required_encryption_key() -> str:
        key = os.getenv(_CALENDAR_SECRET_KEY_ENV, "").strip()
        if not key:
            raise CalendarValidationError(
                f"{_CALENDAR_SECRET_KEY_ENV} must be set before storing Calendar external account credentials"
            )
        return key
