"""Database-layer key custody for encrypted Personal Context profiles."""

from __future__ import annotations

import base64
import binascii
import json
import os
import secrets
import sqlite3
from datetime import UTC, datetime

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.personal_context_crypto import (
    EnvelopeAuthenticationError,
    unwrap_key,
    wrap_key,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ProfileKeyAlreadyExistsError,
    ProfileKeyMaterial,
    ProfileStorageLockedError,
)


def _now_text() -> str:
    now = datetime.now(UTC).replace(microsecond=0)
    return now.isoformat().replace("+00:00", "Z")


class ServerProfileKeyProvider:
    """Wrap per-profile keys under one explicitly configured server root key."""

    ENV_NAME = "TLDW_PERSONAL_CONTEXT_MASTER_KEY"

    def __init__(self, database: PersonalizationDB) -> None:
        self._database = database

    def require_master_key(self) -> bytes:
        """Return strict base64-decoded 32-byte configured master key material."""

        raw = os.getenv(self.ENV_NAME, "").strip()
        try:
            key = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            raise ProfileStorageLockedError("invalid server profile master key") from None
        if len(key) != 32:
            raise ProfileStorageLockedError("server profile master key must be exactly 32 bytes")
        return key

    @staticmethod
    def _aad(profile_id: str, purpose: str, version: int) -> bytes:
        return json.dumps(
            {
                "domain": "tldw-personal-context-server-key-v1",
                "profile_id": profile_id,
                "purpose": purpose,
                "version": version,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def create(
        self,
        profile_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ProfileKeyMaterial:
        """Create and persist new wrapped profile keys exactly once."""

        if connection is None:
            with self._database.transaction(immediate=True) as owned:
                return self.create(profile_id, connection=owned)
        if connection.execute(
            "SELECT 1 FROM personal_context_profile_keys WHERE profile_id = ?",
            (profile_id,),
        ).fetchone():
            raise ProfileKeyAlreadyExistsError("profile key material already exists")

        master_key = self.require_master_key()
        material = ProfileKeyMaterial(
            encryption_key=secrets.token_bytes(32),
            integrity_key=secrets.token_bytes(32),
        )
        wrap_nonce, wrapped_profile_key = wrap_key(
            master_key,
            material.encryption_key,
            self._aad(profile_id, "encryption", material.key_version),
        )
        integrity_wrap_nonce, wrapped_integrity_key = wrap_key(
            master_key,
            material.integrity_key,
            self._aad(
                profile_id,
                "integrity",
                material.integrity_key_version,
            ),
        )
        now = _now_text()
        connection.execute(
            """
            INSERT INTO personal_context_profile_keys(
                profile_id, key_version, integrity_key_version,
                wrapped_profile_key, wrap_nonce,
                wrapped_integrity_key, integrity_wrap_nonce,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                material.key_version,
                material.integrity_key_version,
                wrapped_profile_key,
                wrap_nonce,
                wrapped_integrity_key,
                integrity_wrap_nonce,
                now,
                now,
            ),
        )
        return material

    def load(
        self,
        profile_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> ProfileKeyMaterial:
        """Load existing keys or fail closed without replacement."""

        master_key = self.require_master_key()
        if connection is None:
            with self._database.transaction() as owned:
                return self.load(profile_id, connection=owned)
        row = connection.execute(
            "SELECT * FROM personal_context_profile_keys WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
        if row is None:
            raise ProfileStorageLockedError("profile key material is unavailable")
        try:
            key_version = int(row["key_version"])
            integrity_key_version = int(row["integrity_key_version"])
            return ProfileKeyMaterial(
                encryption_key=unwrap_key(
                    master_key,
                    bytes(row["wrap_nonce"]),
                    bytes(row["wrapped_profile_key"]),
                    self._aad(profile_id, "encryption", key_version),
                ),
                integrity_key=unwrap_key(
                    master_key,
                    bytes(row["integrity_wrap_nonce"]),
                    bytes(row["wrapped_integrity_key"]),
                    self._aad(profile_id, "integrity", integrity_key_version),
                ),
                key_version=key_version,
                integrity_key_version=integrity_key_version,
            )
        except (EnvelopeAuthenticationError, KeyError, TypeError, ValueError):
            raise ProfileStorageLockedError("profile key material is unavailable") from None

    def replace_encryption_key(
        self,
        profile_id: str,
        *,
        encryption_key: bytes,
        integrity_key: bytes,
        expected_key_version: int,
        integrity_key_version: int,
        connection: sqlite3.Connection,
    ) -> ProfileKeyMaterial:
        """Replace wrapped encryption and integrity material atomically."""

        if len(encryption_key) != 32 or len(integrity_key) != 32:
            raise ValueError("profile keys must be exactly 32 bytes")
        new_key_version = expected_key_version + 1
        master_key = self.require_master_key()
        wrap_nonce, wrapped_profile_key = wrap_key(
            master_key,
            encryption_key,
            self._aad(profile_id, "encryption", new_key_version),
        )
        integrity_wrap_nonce, wrapped_integrity_key = wrap_key(
            master_key,
            integrity_key,
            self._aad(profile_id, "integrity", integrity_key_version),
        )
        updated = connection.execute(
            """
            UPDATE personal_context_profile_keys
            SET key_version = ?, wrapped_profile_key = ?, wrap_nonce = ?,
                integrity_key_version = ?, wrapped_integrity_key = ?,
                integrity_wrap_nonce = ?, updated_at = ?
            WHERE profile_id = ? AND key_version = ?
            """,
            (
                new_key_version,
                wrapped_profile_key,
                wrap_nonce,
                integrity_key_version,
                wrapped_integrity_key,
                integrity_wrap_nonce,
                _now_text(),
                profile_id,
                expected_key_version,
            ),
        )
        if updated.rowcount != 1:
            raise ProfileStorageLockedError("profile key material is unavailable")
        return ProfileKeyMaterial(
            encryption_key=encryption_key,
            integrity_key=integrity_key,
            key_version=new_key_version,
            integrity_key_version=integrity_key_version,
        )
