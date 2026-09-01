"""Database-layer encrypted Personal Context repository."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from tldw_profile_core import (
    ProfileManifest,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    ProposalState,
    RecordState,
    ScopeKind,
    canonical_bytes,
)
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.DB_Management.Personal_Context_Key_Store import (
    ServerProfileKeyProvider,
)
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.personal_context_crypto import (
    EncryptedEnvelope,
    EnvelopeAuthenticationError,
    EnvelopeCipher,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ConcurrentProfileUpdateError,
    ProfileAlreadyExistsError,
    ProfileIntegrityError,
    ProfileKeyMaterial,
    ProfileQuotaExceededError,
    ProfileSemanticKeyCollisionError,
    ProfileStorageLockedError,
    ProfileUnsupportedSchemaError,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_ENVELOPE_SCHEMA_VERSION = 1
_MAX_PENDING_PROPOSALS = 200
_MAX_PROPOSAL_HEADS = 1_000
_MAX_RECORD_HEADS = 1_000
_MAX_SCOPE_HEADS = 1_000
_MAX_LIST_ROWS = 1_000


def _now_text() -> str:
    now = datetime.now(UTC)
    now = now.replace(microsecond=now.microsecond // 1000 * 1000)
    return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class PersonalContextRepository:
    """Persist immutable canonical object versions under per-profile keys."""

    def __init__(self, database: PersonalizationDB) -> None:
        self._database = database
        self._keys = ServerProfileKeyProvider(database)

    @property
    def database(self) -> PersonalizationDB:
        """Expose the owning database for transaction-level integrations and tests."""

        return self._database

    def initialize_schema(self) -> None:
        """Verify the database wrapper has installed every canonical table."""

        required = {
            "personal_context_profile_keys",
            "personal_context_object_versions",
            "personal_context_object_heads",
            "personal_context_runtime_heads",
            "personal_context_receipts",
        }
        with self._database.transaction() as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if not required.issubset(tables):
            raise ProfileIntegrityError("Personal Context schema is incomplete")

    def close(self) -> None:
        """Release no persistent handles; each operation owns its connection."""

    @staticmethod
    def _aad(
        profile_id: str,
        object_type: str,
        object_id: str,
        version_id: str,
        schema_version: int,
    ) -> bytes:
        return canonical_json_bytes(
            {
                "envelope": "tldw-personal-context-server-v1",
                "object_id": object_id,
                "object_type": object_type,
                "profile_id": profile_id,
                "schema_version": schema_version,
                "version_id": version_id,
            }
        )

    @staticmethod
    def _canonical_payload(value: BaseModel | Mapping[str, Any]) -> bytes:
        if isinstance(value, BaseModel):
            return canonical_bytes(value)
        return canonical_json_bytes(value)

    @staticmethod
    def _integrity_tag(key: bytes, plaintext: bytes) -> str:
        digest = hmac.new(key, plaintext, hashlib.sha256).hexdigest()
        return f"hmac-sha256-v1:{digest}"

    def _insert_encrypted(
        self,
        connection: sqlite3.Connection,
        keys: ProfileKeyMaterial,
        *,
        profile_id: str,
        object_type: str,
        object_id: str,
        version_id: str,
        parent_version_id: str | None,
        value: BaseModel | Mapping[str, Any],
    ) -> None:
        plaintext = self._canonical_payload(value)
        aad = self._aad(
            profile_id,
            object_type,
            object_id,
            version_id,
            _ENVELOPE_SCHEMA_VERSION,
        )
        envelope = EnvelopeCipher(
            keys.encryption_key,
            key_version=keys.key_version,
        ).encrypt(plaintext, aad)
        connection.execute(
            """
            INSERT INTO personal_context_object_versions(
                profile_id, object_type, object_id, version_id,
                parent_version_id, schema_version, algorithm, key_version,
                nonce, wrapped_dek, wrapped_dek_nonce, ciphertext,
                integrity_tag, payload_size_bytes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile_id,
                object_type,
                object_id,
                version_id,
                parent_version_id,
                _ENVELOPE_SCHEMA_VERSION,
                envelope.algorithm,
                envelope.key_version,
                envelope.nonce,
                envelope.wrapped_dek,
                envelope.wrapped_dek_nonce,
                envelope.ciphertext,
                self._integrity_tag(keys.integrity_key, plaintext),
                len(plaintext),
                _now_text(),
            ),
        )

    def _decrypt_row(
        self,
        row: sqlite3.Row,
        keys: ProfileKeyMaterial,
    ) -> bytes:
        try:
            schema_version = int(row["schema_version"])
            if schema_version != _ENVELOPE_SCHEMA_VERSION:
                raise ProfileUnsupportedSchemaError("Encrypted object schema version is unsupported")
            key_version = int(row["key_version"])
            envelope = EncryptedEnvelope(
                algorithm=str(row["algorithm"]),
                nonce=bytes(row["nonce"]),
                wrapped_dek=bytes(row["wrapped_dek"]),
                wrapped_dek_nonce=bytes(row["wrapped_dek_nonce"]),
                ciphertext=bytes(row["ciphertext"]),
                key_version=key_version,
            )
            if key_version != keys.key_version:
                raise ProfileIntegrityError("Encrypted object key version is invalid")
            aad = self._aad(
                str(row["profile_id"]),
                str(row["object_type"]),
                str(row["object_id"]),
                str(row["version_id"]),
                schema_version,
            )
            plaintext = EnvelopeCipher(
                keys.encryption_key,
                key_version=key_version,
            ).decrypt(envelope, aad)
            expected = self._integrity_tag(keys.integrity_key, plaintext)
            if not hmac.compare_digest(expected, str(row["integrity_tag"])):
                raise ProfileIntegrityError("Canonical object integrity failed")
            if len(plaintext) != int(row["payload_size_bytes"]):
                raise ProfileIntegrityError("Encrypted object size is invalid")
            return plaintext
        except ProfileIntegrityError:
            raise
        except (EnvelopeAuthenticationError, KeyError, TypeError, ValueError):
            raise ProfileIntegrityError("Encrypted object authentication failed") from None

    @staticmethod
    def _set_head(
        connection: sqlite3.Connection,
        *,
        profile_id: str,
        object_type: str,
        object_id: str,
        version_id: str,
        expected_version_id: str | None,
    ) -> None:
        current = connection.execute(
            """
            SELECT current_version_id FROM personal_context_object_heads
            WHERE profile_id = ? AND object_type = ? AND object_id = ?
            """,
            (profile_id, object_type, object_id),
        ).fetchone()
        if current is None:
            if expected_version_id is not None:
                raise ConcurrentProfileUpdateError("object head changed concurrently")
            connection.execute(
                "INSERT INTO personal_context_object_heads VALUES (?, ?, ?, ?)",
                (profile_id, object_type, object_id, version_id),
            )
            return
        if current["current_version_id"] != expected_version_id:
            raise ConcurrentProfileUpdateError("object head changed concurrently")
        updated = connection.execute(
            """
            UPDATE personal_context_object_heads SET current_version_id = ?
            WHERE profile_id = ? AND object_type = ? AND object_id = ?
              AND current_version_id = ?
            """,
            (
                version_id,
                profile_id,
                object_type,
                object_id,
                expected_version_id,
            ),
        )
        if updated.rowcount != 1:
            raise ConcurrentProfileUpdateError("object head changed concurrently")

    @staticmethod
    def _validate_new_head_quota(
        connection: sqlite3.Connection,
        profile_id: str,
        object_type: str,
        maximum: int,
        *,
        expected_version_id: str | None,
    ) -> None:
        if expected_version_id is not None:
            return
        count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM personal_context_object_heads
                WHERE profile_id = ? AND object_type = ?
                """,
                (profile_id, object_type),
            ).fetchone()[0]
        )
        if count >= maximum:
            raise ProfileQuotaExceededError(f"{object_type} quota exceeded")

    @staticmethod
    def _prune_terminal_proposals_for_insert(
        connection: sqlite3.Connection,
        profile_id: str,
    ) -> None:
        """Retain a bounded proposal history by removing oldest terminal receipts."""

        count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM personal_context_object_heads
                WHERE profile_id = ? AND object_type = 'proposal'
                """,
                (profile_id,),
            ).fetchone()[0]
        )
        required = count - _MAX_PROPOSAL_HEADS + 1
        if required <= 0:
            return
        rows = connection.execute(
            """
            SELECT receipts.receipt_id
            FROM personal_context_receipts AS receipts
            JOIN personal_context_object_heads AS heads
              ON heads.profile_id = receipts.profile_id
             AND heads.object_type = 'proposal'
             AND heads.object_id = receipts.receipt_id
            WHERE receipts.profile_id = ?
            ORDER BY receipts.created_at, receipts.receipt_id
            LIMIT ?
            """,
            (profile_id, required),
        ).fetchall()
        if len(rows) != required:
            raise ProfileQuotaExceededError("proposal storage quota exceeded")
        receipt_ids = tuple(str(row["receipt_id"]) for row in rows)
        connection.executemany(
            """
            DELETE FROM personal_context_object_heads
            WHERE profile_id = ? AND object_type = 'proposal' AND object_id = ?
            """,
            ((profile_id, receipt_id) for receipt_id in receipt_ids),
        )
        connection.executemany(
            """
            DELETE FROM personal_context_object_versions
            WHERE profile_id = ? AND object_type = 'proposal' AND object_id = ?
            """,
            ((profile_id, receipt_id) for receipt_id in receipt_ids),
        )
        connection.executemany(
            """
            DELETE FROM personal_context_receipts
            WHERE profile_id = ? AND receipt_id = ?
            """,
            ((profile_id, receipt_id) for receipt_id in receipt_ids),
        )

    @staticmethod
    def _head_row(
        connection: sqlite3.Connection,
        profile_id: str,
        object_type: str,
        object_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT versions.*
            FROM personal_context_object_heads AS heads
            JOIN personal_context_object_versions AS versions
              ON versions.profile_id = heads.profile_id
             AND versions.object_type = heads.object_type
             AND versions.object_id = heads.object_id
             AND versions.version_id = heads.current_version_id
            WHERE heads.profile_id = ? AND heads.object_type = ?
              AND heads.object_id = ?
            """,
            (profile_id, object_type, object_id),
        ).fetchone()

    def _read_model(
        self,
        profile_id: str,
        object_type: str,
        object_id: str,
        model_type: type[_ModelT],
    ) -> _ModelT | None:
        with self._database.transaction() as connection:
            row = self._head_row(connection, profile_id, object_type, object_id)
            if row is None:
                return None
            keys = self._keys.load(profile_id, connection=connection)
            plaintext = self._decrypt_row(row, keys)
        try:
            return model_type.model_validate_json(plaintext)
        except ValidationError:
            raise ProfileIntegrityError("Canonical object validation failed") from None

    def create_profile(
        self,
        manifest: ProfileManifest,
        global_scope: ProfileScope,
        *,
        runtime_policy: Mapping[str, Any] | None = None,
        runtime_version_id: str | None = None,
    ) -> None:
        """Atomically create wrapped keys, manifest, and required global scope."""

        if global_scope.profile_id != manifest.profile_id or global_scope.kind is not ScopeKind.GLOBAL:
            raise ValueError("global scope must belong to the profile")
        if (runtime_policy is None) != (runtime_version_id is None):
            raise ValueError("runtime policy and version must be provided together")
        with self._database.transaction(immediate=True) as connection:
            if connection.execute("SELECT 1 FROM personal_context_profile_keys LIMIT 1").fetchone():
                raise ProfileAlreadyExistsError("a Personal Context profile exists")
            surviving_state = connection.execute(
                """
                SELECT 1 FROM personal_context_object_versions
                UNION ALL SELECT 1 FROM personal_context_object_heads
                UNION ALL SELECT 1 FROM personal_context_runtime_heads
                UNION ALL SELECT 1 FROM personal_context_receipts
                LIMIT 1
                """
            ).fetchone()
            if surviving_state is not None:
                raise ProfileStorageLockedError("existing profile key material is unavailable")
            keys = self._keys.create(manifest.profile_id, connection=connection)
            self._insert_encrypted(
                connection,
                keys,
                profile_id=manifest.profile_id,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                parent_version_id=None,
                value=manifest,
            )
            self._set_head(
                connection,
                profile_id=manifest.profile_id,
                object_type="manifest",
                object_id=manifest.profile_id,
                version_id=manifest.current_version_id,
                expected_version_id=None,
            )
            self._insert_encrypted(
                connection,
                keys,
                profile_id=manifest.profile_id,
                object_type="scope",
                object_id=global_scope.scope_id,
                version_id=global_scope.version_id,
                parent_version_id=None,
                value=global_scope,
            )
            self._set_head(
                connection,
                profile_id=manifest.profile_id,
                object_type="scope",
                object_id=global_scope.scope_id,
                version_id=global_scope.version_id,
                expected_version_id=None,
            )
            if runtime_policy is not None and runtime_version_id is not None:
                self._insert_encrypted(
                    connection,
                    keys,
                    profile_id=manifest.profile_id,
                    object_type="runtime_policy",
                    object_id="__profile__",
                    version_id=runtime_version_id,
                    parent_version_id=None,
                    value=runtime_policy,
                )
                self._set_runtime_head(
                    connection,
                    manifest.profile_id,
                    "__profile__",
                    runtime_version_id,
                    None,
                )

    def get_manifest(self, profile_id: str) -> ProfileManifest | None:
        """Return one authenticated manifest without cross-profile fallback."""

        return self._read_model(profile_id, "manifest", profile_id, ProfileManifest)

    def get_scope(self, profile_id: str, scope_id: str) -> ProfileScope | None:
        """Return one authenticated scope for the exact profile."""

        return self._read_model(profile_id, "scope", scope_id, ProfileScope)

    def profile_ids(self) -> tuple[str, ...]:
        """Return opaque profile IDs present in this authenticated user's database."""

        with self._database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT profile_id
                FROM personal_context_object_heads
                WHERE object_type = 'manifest' AND object_id = profile_id
                ORDER BY profile_id
                """
            ).fetchall()
        return tuple(str(row["profile_id"]) for row in rows)

    def has_profile_state(self) -> bool:
        """Return whether any encrypted profile state or key material exists."""

        with self._database.transaction() as connection:
            return (
                connection.execute(
                    """
                    SELECT 1 FROM personal_context_profile_keys
                    UNION ALL SELECT 1 FROM personal_context_object_versions
                    UNION ALL SELECT 1 FROM personal_context_object_heads
                    UNION ALL SELECT 1 FROM personal_context_runtime_heads
                    UNION ALL SELECT 1 FROM personal_context_receipts
                    LIMIT 1
                    """
                ).fetchone()
                is not None
            )

    def _list_models(
        self,
        profile_id: str,
        object_type: str,
        model_type: type[_ModelT],
        *,
        limit: int,
        offset: int = 0,
    ) -> tuple[_ModelT, ...]:
        if not 1 <= limit <= _MAX_LIST_ROWS or offset < 0:
            raise ValueError("encrypted object page is out of bounds")
        with self._database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT versions.*
                FROM personal_context_object_heads AS heads
                JOIN personal_context_object_versions AS versions
                  ON versions.profile_id = heads.profile_id
                 AND versions.object_type = heads.object_type
                 AND versions.object_id = heads.object_id
                 AND versions.version_id = heads.current_version_id
                WHERE heads.profile_id = ? AND heads.object_type = ?
                ORDER BY heads.object_id
                LIMIT ? OFFSET ?
                """,
                (profile_id, object_type, limit, offset),
            ).fetchall()
            if not rows:
                return ()
            keys = self._keys.load(profile_id, connection=connection)
            plaintext_rows = tuple(self._decrypt_row(row, keys) for row in rows)
        try:
            return tuple(model_type.model_validate_json(value) for value in plaintext_rows)
        except ValidationError:
            raise ProfileIntegrityError("Canonical object validation failed") from None

    def list_scopes(
        self,
        profile_id: str,
        *,
        limit: int = _MAX_SCOPE_HEADS,
        offset: int = 0,
    ) -> tuple[ProfileScope, ...]:
        """Return authenticated scope heads for exactly one profile."""

        return self._list_models(
            profile_id,
            "scope",
            ProfileScope,
            limit=limit,
            offset=offset,
        )

    def list_records(
        self,
        profile_id: str,
        *,
        limit: int = _MAX_RECORD_HEADS,
        offset: int = 0,
    ) -> tuple[ProfileRecord, ...]:
        """Return authenticated record heads for exactly one profile."""

        return self._list_models(
            profile_id,
            "record",
            ProfileRecord,
            limit=limit,
            offset=offset,
        )

    def list_proposals(
        self,
        profile_id: str,
        *,
        limit: int = _MAX_PENDING_PROPOSALS,
        offset: int = 0,
    ) -> tuple[ProfileProposal, ...]:
        """Return authenticated proposal heads for exactly one profile."""

        return self._list_models(
            profile_id,
            "proposal",
            ProfileProposal,
            limit=limit,
            offset=offset,
        )

    def list_unresolved_proposals(
        self,
        profile_id: str,
    ) -> tuple[ProfileProposal, ...]:
        """Return a bounded unresolved set without scanning terminal receipts."""

        with self._database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT versions.*
                FROM personal_context_object_heads AS heads
                JOIN personal_context_object_versions AS versions
                  ON versions.profile_id = heads.profile_id
                 AND versions.object_type = heads.object_type
                 AND versions.object_id = heads.object_id
                 AND versions.version_id = heads.current_version_id
                LEFT JOIN personal_context_receipts AS receipts
                  ON receipts.profile_id = heads.profile_id
                 AND receipts.receipt_id = heads.object_id
                WHERE heads.profile_id = ? AND heads.object_type = 'proposal'
                  AND receipts.receipt_id IS NULL
                ORDER BY heads.object_id
                LIMIT ?
                """,
                (profile_id, _MAX_PENDING_PROPOSALS),
            ).fetchall()
            if not rows:
                return ()
            keys = self._keys.load(profile_id, connection=connection)
            plaintext_rows = tuple(self._decrypt_row(row, keys) for row in rows)
        try:
            return tuple(ProfileProposal.model_validate_json(value) for value in plaintext_rows)
        except ValidationError:
            raise ProfileIntegrityError("Canonical object validation failed") from None

    @staticmethod
    def _validate_manifest_transition(
        current: ProfileManifest,
        manifest: ProfileManifest,
        *,
        expected_version_id: str,
        allow_purge_advance: bool = False,
    ) -> None:
        expected_purge_generation = current.purge_generation + 1 if allow_purge_advance else current.purge_generation
        if (
            manifest.profile_id != current.profile_id
            or manifest.current_version_id == expected_version_id
            or manifest.revision != current.revision + 1
            or manifest.purge_generation != expected_purge_generation
            or manifest.created_at != current.created_at
            or manifest.updated_at < current.updated_at
        ):
            raise ValueError("manifest revision is invalid")

    def _read_manifest_for_update(
        self,
        connection: sqlite3.Connection,
        profile_id: str,
        expected_version_id: str,
        keys: ProfileKeyMaterial,
    ) -> tuple[sqlite3.Row, ProfileManifest]:
        row = self._head_row(connection, profile_id, "manifest", profile_id)
        if row is None or str(row["version_id"]) != expected_version_id:
            raise ConcurrentProfileUpdateError("manifest head changed concurrently")
        try:
            current = ProfileManifest.model_validate_json(self._decrypt_row(row, keys))
        except ValidationError:
            raise ProfileIntegrityError("Canonical object validation failed") from None
        return row, current

    def _require_writable_manifest_state(
        self,
        connection: sqlite3.Connection,
        profile_id: str,
        expected_manifest_version: str,
        keys: ProfileKeyMaterial,
    ) -> None:
        """Fence standalone writes against manifest changes and purge barriers."""

        self._read_manifest_for_update(
            connection,
            profile_id,
            expected_manifest_version,
            keys,
        )
        scope_exists = connection.execute(
            """
            SELECT 1 FROM personal_context_object_heads
            WHERE profile_id = ? AND object_type = 'scope'
            LIMIT 1
            """,
            (profile_id,),
        ).fetchone()
        if scope_exists is None:
            raise ConcurrentProfileUpdateError("profile purge barrier changed concurrently")

    def _require_current_writable_manifest_state(
        self,
        connection: sqlite3.Connection,
        profile_id: str,
        keys: ProfileKeyMaterial,
    ) -> None:
        """Fence a standalone write against the live manifest and purge barrier."""

        manifest_row = self._head_row(
            connection,
            profile_id,
            "manifest",
            profile_id,
        )
        if manifest_row is None:
            raise ConcurrentProfileUpdateError("manifest head changed concurrently")
        self._require_writable_manifest_state(
            connection,
            profile_id,
            str(manifest_row["version_id"]),
            keys,
        )

    def _insert_manifest_revision(
        self,
        connection: sqlite3.Connection,
        keys: ProfileKeyMaterial,
        manifest: ProfileManifest,
        *,
        expected_version_id: str,
    ) -> None:
        self._insert_encrypted(
            connection,
            keys,
            profile_id=manifest.profile_id,
            object_type="manifest",
            object_id=manifest.profile_id,
            version_id=manifest.current_version_id,
            parent_version_id=expected_version_id,
            value=manifest,
        )
        self._set_head(
            connection,
            profile_id=manifest.profile_id,
            object_type="manifest",
            object_id=manifest.profile_id,
            version_id=manifest.current_version_id,
            expected_version_id=expected_version_id,
        )

    def commit_manifest_version(
        self,
        manifest: ProfileManifest,
        *,
        expected_version_id: str,
    ) -> None:
        """Commit the next manifest revision with an optimistic head check."""

        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(manifest.profile_id, connection=connection)
            _row, current = self._read_manifest_for_update(
                connection,
                manifest.profile_id,
                expected_version_id,
                keys,
            )
            self._validate_manifest_transition(
                current,
                manifest,
                expected_version_id=expected_version_id,
            )
            self._insert_manifest_revision(
                connection,
                keys,
                manifest,
                expected_version_id=expected_version_id,
            )

    def commit_scope_and_manifest(
        self,
        scope: ProfileScope,
        manifest: ProfileManifest,
        *,
        expected_scope_version: str | None,
        expected_manifest_version: str,
        runtime_policy: Mapping[str, Any] | None = None,
        runtime_version_id: str | None = None,
    ) -> None:
        """Atomically commit one scope revision and its manifest fence."""

        if scope.profile_id != manifest.profile_id:
            raise ValueError("scope and manifest must belong to the same profile")
        if (runtime_policy is None) != (runtime_version_id is None):
            raise ValueError("runtime policy and version must be provided together")
        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(scope.profile_id, connection=connection)
            _row, current_manifest = self._read_manifest_for_update(
                connection,
                scope.profile_id,
                expected_manifest_version,
                keys,
            )
            self._validate_manifest_transition(
                current_manifest,
                manifest,
                expected_version_id=expected_manifest_version,
            )
            current_scope = self._head_row(
                connection,
                scope.profile_id,
                "scope",
                scope.scope_id,
            )
            current_scope_version = None if current_scope is None else str(current_scope["version_id"])
            if current_scope_version != expected_scope_version:
                raise ConcurrentProfileUpdateError("scope head changed concurrently")
            self._validate_new_head_quota(
                connection,
                scope.profile_id,
                "scope",
                _MAX_SCOPE_HEADS,
                expected_version_id=expected_scope_version,
            )
            self._insert_encrypted(
                connection,
                keys,
                profile_id=scope.profile_id,
                object_type="scope",
                object_id=scope.scope_id,
                version_id=scope.version_id,
                parent_version_id=expected_scope_version,
                value=scope,
            )
            self._set_head(
                connection,
                profile_id=scope.profile_id,
                object_type="scope",
                object_id=scope.scope_id,
                version_id=scope.version_id,
                expected_version_id=expected_scope_version,
            )
            if runtime_policy is not None and runtime_version_id is not None:
                self._insert_encrypted(
                    connection,
                    keys,
                    profile_id=scope.profile_id,
                    object_type="runtime_policy",
                    object_id=scope.scope_id,
                    version_id=runtime_version_id,
                    parent_version_id=None,
                    value=runtime_policy,
                )
                self._set_runtime_head(
                    connection,
                    scope.profile_id,
                    scope.scope_id,
                    runtime_version_id,
                    None,
                )
            self._insert_manifest_revision(
                connection,
                keys,
                manifest,
                expected_version_id=expected_manifest_version,
            )

    def commit_scope(
        self,
        scope: ProfileScope,
        *,
        expected_version_id: str | None,
    ) -> None:
        """Commit a scope revision with an optimistic head check."""

        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(scope.profile_id, connection=connection)
            self._require_current_writable_manifest_state(
                connection,
                scope.profile_id,
                keys,
            )
            current_row = self._head_row(
                connection,
                scope.profile_id,
                "scope",
                scope.scope_id,
            )
            current_version_id = None if current_row is None else str(current_row["version_id"])
            if current_version_id != expected_version_id:
                raise ConcurrentProfileUpdateError("scope head changed concurrently")
            self._validate_new_head_quota(
                connection,
                scope.profile_id,
                "scope",
                _MAX_SCOPE_HEADS,
                expected_version_id=expected_version_id,
            )
            if current_row is not None:
                try:
                    current = ProfileScope.model_validate_json(self._decrypt_row(current_row, keys))
                except ValidationError:
                    raise ProfileIntegrityError("Canonical object validation failed") from None
                if (
                    scope.version_id == expected_version_id
                    or scope.kind is not current.kind
                    or scope.created_at != current.created_at
                    or scope.updated_at < current.updated_at
                ):
                    raise ValueError("scope revision is invalid")
            self._insert_encrypted(
                connection,
                keys,
                profile_id=scope.profile_id,
                object_type="scope",
                object_id=scope.scope_id,
                version_id=scope.version_id,
                parent_version_id=expected_version_id,
                value=scope,
            )
            self._set_head(
                connection,
                profile_id=scope.profile_id,
                object_type="scope",
                object_id=scope.scope_id,
                version_id=scope.version_id,
                expected_version_id=expected_version_id,
            )

    def commit_record_version(
        self,
        record: ProfileRecord,
        *,
        expected_version_id: str | None,
    ) -> None:
        """Insert an immutable record and compare-and-set its head atomically."""

        if record.parent_version_id != expected_version_id:
            raise ConcurrentProfileUpdateError("record parent does not match head")
        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(record.profile_id, connection=connection)
            self._require_current_writable_manifest_state(
                connection,
                record.profile_id,
                keys,
            )
            self._validate_new_head_quota(
                connection,
                record.profile_id,
                "record",
                _MAX_RECORD_HEADS,
                expected_version_id=expected_version_id,
            )
            self._insert_encrypted(
                connection,
                keys,
                profile_id=record.profile_id,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                parent_version_id=record.parent_version_id,
                value=record,
            )
            self._set_head(
                connection,
                profile_id=record.profile_id,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                expected_version_id=expected_version_id,
            )

    def commit_record_and_manifest(
        self,
        record: ProfileRecord,
        manifest: ProfileManifest,
        *,
        expected_record_version: str | None,
        expected_manifest_version: str,
    ) -> None:
        """Atomically commit one record revision and its manifest fence."""

        if record.profile_id != manifest.profile_id or record.parent_version_id != expected_record_version:
            raise ConcurrentProfileUpdateError("record parent does not match head")
        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(record.profile_id, connection=connection)
            _row, current_manifest = self._read_manifest_for_update(
                connection,
                record.profile_id,
                expected_manifest_version,
                keys,
            )
            self._validate_manifest_transition(
                current_manifest,
                manifest,
                expected_version_id=expected_manifest_version,
            )
            current_record = self._head_row(
                connection,
                record.profile_id,
                "record",
                record.record_id,
            )
            current_record_version = None if current_record is None else str(current_record["version_id"])
            if current_record_version != expected_record_version:
                raise ConcurrentProfileUpdateError("record head changed concurrently")
            self._validate_new_head_quota(
                connection,
                record.profile_id,
                "record",
                _MAX_RECORD_HEADS,
                expected_version_id=expected_record_version,
            )
            self._validate_semantic_key_available(
                connection,
                keys,
                record,
                excluding_record_id=(None if expected_record_version is None else record.record_id),
            )
            self._insert_encrypted(
                connection,
                keys,
                profile_id=record.profile_id,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                parent_version_id=expected_record_version,
                value=record,
            )
            self._set_head(
                connection,
                profile_id=record.profile_id,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                expected_version_id=expected_record_version,
            )
            self._insert_manifest_revision(
                connection,
                keys,
                manifest,
                expected_version_id=expected_manifest_version,
            )

    def get_record(self, profile_id: str, record_id: str) -> ProfileRecord | None:
        """Return one authenticated record for the exact profile."""

        return self._read_model(profile_id, "record", record_id, ProfileRecord)

    def version_exists(
        self,
        profile_id: str,
        object_type: str,
        object_id: str,
        version_id: str,
    ) -> bool:
        """Return whether one immutable version survived a transaction."""

        with self._database.transaction() as connection:
            return (
                connection.execute(
                    """
                    SELECT 1 FROM personal_context_object_versions
                    WHERE profile_id = ? AND object_type = ?
                      AND object_id = ? AND version_id = ?
                    """,
                    (profile_id, object_type, object_id, version_id),
                ).fetchone()
                is not None
            )

    def commit_proposal(
        self,
        proposal: ProfileProposal,
        *,
        expected_manifest_version: str,
    ) -> None:
        """Persist a new pending proposal under a generated storage version."""

        if proposal.state is not ProposalState.PENDING:
            raise ValueError("only pending proposals may be committed")
        version_id = str(uuid.uuid4())
        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(proposal.profile_id, connection=connection)
            self._require_writable_manifest_state(
                connection,
                proposal.profile_id,
                expected_manifest_version,
                keys,
            )
            if proposal.expires_at <= datetime.now(UTC):
                raise ValueError("proposal has expired")
            if (
                self._head_row(
                    connection,
                    proposal.profile_id,
                    "proposal",
                    proposal.proposal_id,
                )
                is not None
            ):
                raise ConcurrentProfileUpdateError("proposal head changed concurrently")
            self._prune_terminal_proposals_for_insert(
                connection,
                proposal.profile_id,
            )
            pending = 0
            rows = connection.execute(
                """
                SELECT versions.*
                FROM personal_context_object_heads AS heads
                JOIN personal_context_object_versions AS versions
                  ON versions.profile_id = heads.profile_id
                 AND versions.object_type = heads.object_type
                 AND versions.object_id = heads.object_id
                 AND versions.version_id = heads.current_version_id
                LEFT JOIN personal_context_receipts AS receipts
                  ON receipts.profile_id = heads.profile_id
                 AND receipts.receipt_id = heads.object_id
                WHERE heads.profile_id = ? AND heads.object_type = 'proposal'
                  AND receipts.receipt_id IS NULL
                LIMIT ?
                """,
                (proposal.profile_id, _MAX_PENDING_PROPOSALS + 1),
            ).fetchall()
            if len(rows) > _MAX_PENDING_PROPOSALS:
                raise ProfileQuotaExceededError("pending proposal quota exceeded")
            now = datetime.now(UTC)
            for row in rows:
                try:
                    existing = ProfileProposal.model_validate_json(self._decrypt_row(row, keys))
                except ValidationError:
                    raise ProfileIntegrityError("Canonical object validation failed") from None
                if existing.state is ProposalState.PENDING and existing.expires_at > now:
                    pending += 1
            if pending >= _MAX_PENDING_PROPOSALS:
                raise ProfileQuotaExceededError("pending proposal quota exceeded")
            self._insert_encrypted(
                connection,
                keys,
                profile_id=proposal.profile_id,
                object_type="proposal",
                object_id=proposal.proposal_id,
                version_id=version_id,
                parent_version_id=None,
                value=proposal,
            )
            self._set_head(
                connection,
                profile_id=proposal.profile_id,
                object_type="proposal",
                object_id=proposal.proposal_id,
                version_id=version_id,
                expected_version_id=None,
            )

    def get_proposal(
        self,
        profile_id: str,
        proposal_id: str,
    ) -> ProfileProposal | None:
        """Return one authenticated current proposal or terminal receipt."""

        return self._read_model(
            profile_id,
            "proposal",
            proposal_id,
            ProfileProposal,
        )

    def _replace_proposal_with_receipt(
        self,
        connection: sqlite3.Connection,
        keys: ProfileKeyMaterial,
        proposal_row: sqlite3.Row,
        proposal: ProfileProposal,
        state: ProposalState,
        *,
        version_id: str,
    ) -> ProfileProposal:
        """Replace one pending proposal body with a terminal receipt in-place."""

        if state is ProposalState.PENDING:
            raise ValueError("proposal resolution must be terminal")
        profile_id = proposal.profile_id
        proposal_id = proposal.proposal_id
        current_version_id = str(proposal_row["version_id"])
        receipt = ProfileProposal.model_validate(
            {
                **proposal.model_dump(mode="python"),
                "state": state,
                "proposed_record": None,
                "confidence": None,
            }
        )
        self._insert_encrypted(
            connection,
            keys,
            profile_id=profile_id,
            object_type="proposal",
            object_id=proposal_id,
            version_id=version_id,
            parent_version_id=current_version_id,
            value=receipt,
        )
        self._set_head(
            connection,
            profile_id=profile_id,
            object_type="proposal",
            object_id=proposal_id,
            version_id=version_id,
            expected_version_id=current_version_id,
        )
        connection.execute(
            """
            DELETE FROM personal_context_object_versions
            WHERE profile_id = ? AND object_type = 'proposal'
              AND object_id = ? AND version_id != ?
            """,
            (profile_id, proposal_id, version_id),
        )
        connection.execute(
            """
            INSERT INTO personal_context_receipts(
                profile_id, receipt_id, version_id, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (profile_id, proposal_id, version_id, _now_text()),
        )
        return receipt

    def resolve_proposal(
        self,
        profile_id: str,
        proposal_id: str,
        state: ProposalState,
    ) -> ProfileProposal:
        """Replace pending content with one content-free terminal receipt."""

        if state is ProposalState.PENDING:
            raise ValueError("proposal resolution must be terminal")
        version_id = str(uuid.uuid4())
        with self._database.transaction(immediate=True) as connection:
            row = self._head_row(connection, profile_id, "proposal", proposal_id)
            if row is None:
                raise KeyError(proposal_id)
            keys = self._keys.load(profile_id, connection=connection)
            current = ProfileProposal.model_validate_json(self._decrypt_row(row, keys))
            if current.state is not ProposalState.PENDING:
                raise ValueError("only pending proposals may be resolved")
            resolved = self._replace_proposal_with_receipt(
                connection,
                keys,
                row,
                current,
                state,
                version_id=version_id,
            )
        return resolved

    def reject_proposal(
        self,
        profile_id: str,
        proposal_id: str,
    ) -> ProfileProposal:
        """Reject a pending proposal and shred its prior encrypted body."""

        return self.resolve_proposal(
            profile_id,
            proposal_id,
            ProposalState.REJECTED,
        )

    def accept_proposal_and_record(
        self,
        profile_id: str,
        proposal_id: str,
        record: ProfileRecord,
        manifest: ProfileManifest,
        *,
        expected_record_version: str | None,
        expected_manifest_version: str,
    ) -> ProfileProposal:
        """Atomically apply a pending proposal, record, and manifest revision."""

        if (
            record.profile_id != profile_id
            or manifest.profile_id != profile_id
            or record.parent_version_id != expected_record_version
        ):
            raise ConcurrentProfileUpdateError("proposal target changed concurrently")
        receipt_version = str(uuid.uuid4())
        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(profile_id, connection=connection)
            proposal_row = self._head_row(
                connection,
                profile_id,
                "proposal",
                proposal_id,
            )
            if proposal_row is None:
                raise KeyError(proposal_id)
            try:
                proposal = ProfileProposal.model_validate_json(self._decrypt_row(proposal_row, keys))
            except ValidationError:
                raise ProfileIntegrityError("Canonical object validation failed") from None
            if proposal.state is not ProposalState.PENDING:
                raise ValueError("only pending proposals may be accepted")
            if proposal.expires_at <= datetime.now(UTC):
                return self._replace_proposal_with_receipt(
                    connection,
                    keys,
                    proposal_row,
                    proposal,
                    ProposalState.EXPIRED,
                    version_id=receipt_version,
                )
            _row, current_manifest = self._read_manifest_for_update(
                connection,
                profile_id,
                expected_manifest_version,
                keys,
            )
            self._validate_manifest_transition(
                current_manifest,
                manifest,
                expected_version_id=expected_manifest_version,
            )
            current_record = self._head_row(
                connection,
                profile_id,
                "record",
                record.record_id,
            )
            current_record_version = None if current_record is None else str(current_record["version_id"])
            if current_record_version != expected_record_version:
                raise ConcurrentProfileUpdateError("proposal target changed concurrently")
            self._validate_new_head_quota(
                connection,
                record.profile_id,
                "record",
                _MAX_RECORD_HEADS,
                expected_version_id=expected_record_version,
            )
            self._validate_semantic_key_available(
                connection,
                keys,
                record,
                excluding_record_id=(None if expected_record_version is None else record.record_id),
            )
            self._insert_encrypted(
                connection,
                keys,
                profile_id=profile_id,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                parent_version_id=expected_record_version,
                value=record,
            )
            self._set_head(
                connection,
                profile_id=profile_id,
                object_type="record",
                object_id=record.record_id,
                version_id=record.version_id,
                expected_version_id=expected_record_version,
            )
            self._insert_manifest_revision(
                connection,
                keys,
                manifest,
                expected_version_id=expected_manifest_version,
            )
            receipt = self._replace_proposal_with_receipt(
                connection,
                keys,
                proposal_row,
                proposal,
                ProposalState.ACCEPTED,
                version_id=receipt_version,
            )
        return receipt

    def _validate_semantic_key_available(
        self,
        connection: sqlite3.Connection,
        keys: ProfileKeyMaterial,
        record: ProfileRecord,
        *,
        excluding_record_id: str | None,
    ) -> None:
        """Enforce active same-scope semantic uniqueness inside the write lock."""

        if record.state is not RecordState.ACTIVE or record.semantic_key is None:
            return
        rows = connection.execute(
            """
            SELECT versions.*
            FROM personal_context_object_heads AS heads
            JOIN personal_context_object_versions AS versions
              ON versions.profile_id = heads.profile_id
             AND versions.object_type = heads.object_type
             AND versions.object_id = heads.object_id
             AND versions.version_id = heads.current_version_id
            WHERE heads.profile_id = ? AND heads.object_type = 'record'
            LIMIT ?
            """,
            (record.profile_id, _MAX_RECORD_HEADS + 1),
        ).fetchall()
        if len(rows) > _MAX_RECORD_HEADS:
            raise ProfileQuotaExceededError("record head quota exceeded")
        now = datetime.now(UTC)
        for row in rows:
            if str(row["object_id"]) == excluding_record_id:
                continue
            try:
                existing = ProfileRecord.model_validate_json(self._decrypt_row(row, keys))
            except ValidationError:
                raise ProfileIntegrityError("Canonical object validation failed") from None
            if (
                existing.state is RecordState.ACTIVE
                and existing.scope_id == record.scope_id
                and existing.kind is record.kind
                and existing.semantic_key == record.semantic_key
                and (existing.expires_at is None or existing.expires_at > now)
            ):
                raise ProfileSemanticKeyCollisionError("active semantic key already exists")

    def purge_profile(
        self,
        manifest: ProfileManifest,
        *,
        expected_manifest_version: str,
    ) -> None:
        """Advance the purge barrier and remove every readable profile body."""

        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(manifest.profile_id, connection=connection)
            _row, current = self._read_manifest_for_update(
                connection,
                manifest.profile_id,
                expected_manifest_version,
                keys,
            )
            self._validate_manifest_transition(
                current,
                manifest,
                expected_version_id=expected_manifest_version,
                allow_purge_advance=True,
            )
            self._insert_manifest_revision(
                connection,
                keys,
                manifest,
                expected_version_id=expected_manifest_version,
            )
            connection.execute(
                """
                DELETE FROM personal_context_object_heads
                WHERE profile_id = ? AND object_type != 'manifest'
                """,
                (manifest.profile_id,),
            )
            connection.execute(
                """
                DELETE FROM personal_context_object_versions
                WHERE profile_id = ? AND NOT (
                    object_type = 'manifest' AND object_id = ? AND version_id = ?
                )
                """,
                (
                    manifest.profile_id,
                    manifest.profile_id,
                    manifest.current_version_id,
                ),
            )
            connection.execute(
                "DELETE FROM personal_context_runtime_heads WHERE profile_id = ?",
                (manifest.profile_id,),
            )
            connection.execute(
                "DELETE FROM personal_context_receipts WHERE profile_id = ?",
                (manifest.profile_id,),
            )

    def proposal_version_count(self, profile_id: str, proposal_id: str) -> int:
        """Return retained proposal version count for privacy evidence."""

        with self._database.transaction() as connection:
            return int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM personal_context_object_versions
                    WHERE profile_id = ? AND object_type = 'proposal'
                      AND object_id = ?
                    """,
                    (profile_id, proposal_id),
                ).fetchone()[0]
            )

    def encrypted_version_material(
        self,
        profile_id: str,
        object_type: str,
        object_id: str,
    ) -> tuple[bytes, bytes]:
        """Return ciphertext and wrapped DEK solely for durable-owner tests."""

        with self._database.transaction() as connection:
            row = self._head_row(connection, profile_id, object_type, object_id)
            if row is None:
                raise KeyError(object_id)
            return bytes(row["ciphertext"]), bytes(row["wrapped_dek"])

    def encrypted_version_details(
        self,
        profile_id: str,
        object_type: str,
        object_id: str,
    ) -> dict[str, bytes | int]:
        """Return opaque envelope details solely for durable-owner tests."""

        with self._database.transaction() as connection:
            row = self._head_row(connection, profile_id, object_type, object_id)
            if row is None:
                raise KeyError(object_id)
            return {
                "ciphertext": bytes(row["ciphertext"]),
                "wrapped_dek": bytes(row["wrapped_dek"]),
                "key_version": int(row["key_version"]),
            }

    def key_material_for_test(self, profile_id: str) -> ProfileKeyMaterial:
        """Return decrypted key material solely for key-custody tests."""

        return self._keys.load(profile_id)

    def rotate_encryption_key(self, profile_id: str) -> ProfileKeyMaterial:
        """Atomically rewrap every DEK under a fresh profile encryption key."""

        new_encryption_key = secrets.token_bytes(32)
        with self._database.transaction(immediate=True) as connection:
            current_keys = self._keys.load(profile_id, connection=connection)
            rows = connection.execute(
                """
                SELECT * FROM personal_context_object_versions
                WHERE profile_id = ?
                """,
                (profile_id,),
            ).fetchall()
            new_key_version = current_keys.key_version + 1
            cipher = EnvelopeCipher(
                current_keys.encryption_key,
                key_version=current_keys.key_version,
            )
            for row in rows:
                try:
                    schema_version = int(row["schema_version"])
                except (KeyError, TypeError, ValueError):
                    raise ProfileUnsupportedSchemaError("Encrypted object schema version is unsupported") from None
                if schema_version != _ENVELOPE_SCHEMA_VERSION:
                    raise ProfileUnsupportedSchemaError("Encrypted object schema version is unsupported")
                envelope = EncryptedEnvelope(
                    algorithm=str(row["algorithm"]),
                    nonce=bytes(row["nonce"]),
                    wrapped_dek=bytes(row["wrapped_dek"]),
                    wrapped_dek_nonce=bytes(row["wrapped_dek_nonce"]),
                    ciphertext=bytes(row["ciphertext"]),
                    key_version=int(row["key_version"]),
                )
                aad = self._aad(
                    profile_id,
                    str(row["object_type"]),
                    str(row["object_id"]),
                    str(row["version_id"]),
                    schema_version,
                )
                try:
                    rewrapped = cipher.rewrap(
                        envelope,
                        aad,
                        new_encryption_key,
                        new_key_version=new_key_version,
                    )
                except EnvelopeAuthenticationError:
                    raise ProfileIntegrityError("Encrypted object authentication failed") from None
                updated = connection.execute(
                    """
                    UPDATE personal_context_object_versions
                    SET wrapped_dek = ?, wrapped_dek_nonce = ?, key_version = ?
                    WHERE profile_id = ? AND object_type = ? AND object_id = ?
                      AND version_id = ? AND key_version = ?
                    """,
                    (
                        rewrapped.wrapped_dek,
                        rewrapped.wrapped_dek_nonce,
                        rewrapped.key_version,
                        profile_id,
                        row["object_type"],
                        row["object_id"],
                        row["version_id"],
                        current_keys.key_version,
                    ),
                )
                if updated.rowcount != 1:
                    raise ConcurrentProfileUpdateError("encrypted object changed concurrently")
            return self._keys.replace_encryption_key(
                profile_id,
                encryption_key=new_encryption_key,
                integrity_key=current_keys.integrity_key,
                expected_key_version=current_keys.key_version,
                integrity_key_version=current_keys.integrity_key_version,
                connection=connection,
            )

    @staticmethod
    def _set_runtime_head(
        connection: sqlite3.Connection,
        profile_id: str,
        scope_id: str,
        version_id: str,
        expected_version_id: str | None,
    ) -> None:
        current = connection.execute(
            """
            SELECT current_version_id FROM personal_context_runtime_heads
            WHERE profile_id = ? AND scope_id = ?
            """,
            (profile_id, scope_id),
        ).fetchone()
        if current is None:
            if expected_version_id is not None:
                raise ConcurrentProfileUpdateError("runtime policy changed concurrently")
            connection.execute(
                "INSERT INTO personal_context_runtime_heads VALUES (?, ?, ?)",
                (profile_id, scope_id, version_id),
            )
            return
        if current["current_version_id"] != expected_version_id:
            raise ConcurrentProfileUpdateError("runtime policy changed concurrently")
        updated = connection.execute(
            """
            UPDATE personal_context_runtime_heads SET current_version_id = ?
            WHERE profile_id = ? AND scope_id = ? AND current_version_id = ?
            """,
            (version_id, profile_id, scope_id, expected_version_id),
        )
        if updated.rowcount != 1:
            raise ConcurrentProfileUpdateError("runtime policy changed concurrently")

    def set_runtime_policy(
        self,
        profile_id: str,
        scope_id: str,
        *,
        version_id: str,
        expected_version_id: str | None,
        expected_manifest_version: str,
        policy: Mapping[str, Any],
    ) -> None:
        """Persist one encrypted server-local runtime policy revision."""

        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(profile_id, connection=connection)
            self._require_writable_manifest_state(
                connection,
                profile_id,
                expected_manifest_version,
                keys,
            )
            self._insert_encrypted(
                connection,
                keys,
                profile_id=profile_id,
                object_type="runtime_policy",
                object_id=scope_id,
                version_id=version_id,
                parent_version_id=expected_version_id,
                value=policy,
            )
            self._set_runtime_head(
                connection,
                profile_id,
                scope_id,
                version_id,
                expected_version_id,
            )

    def get_runtime_policy(
        self,
        profile_id: str,
        scope_id: str,
    ) -> tuple[str, dict[str, Any]] | None:
        """Return the authenticated server-local runtime policy."""

        with self._database.transaction() as connection:
            row = connection.execute(
                """
                SELECT versions.*
                FROM personal_context_runtime_heads AS heads
                JOIN personal_context_object_versions AS versions
                  ON versions.profile_id = heads.profile_id
                 AND versions.object_type = 'runtime_policy'
                 AND versions.object_id = heads.scope_id
                 AND versions.version_id = heads.current_version_id
                WHERE heads.profile_id = ? AND heads.scope_id = ?
                """,
                (profile_id, scope_id),
            ).fetchone()
            if row is None:
                return None
            keys = self._keys.load(profile_id, connection=connection)
            plaintext = self._decrypt_row(row, keys)
        try:
            payload = json.loads(plaintext)
        except (TypeError, ValueError):
            raise ProfileIntegrityError("Runtime policy validation failed") from None
        if not isinstance(payload, dict):
            raise ProfileIntegrityError("Runtime policy validation failed")
        return str(row["version_id"]), payload
