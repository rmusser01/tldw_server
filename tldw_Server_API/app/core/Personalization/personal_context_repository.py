"""Encrypted canonical Personal Context repository in Personalization.db."""

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
    ScopeKind,
    canonical_bytes,
)
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Personalization.personal_context_crypto import (
    EncryptedEnvelope,
    EnvelopeAuthenticationError,
    EnvelopeCipher,
)
from tldw_Server_API.app.core.Personalization.personal_context_key_provider import (
    ServerProfileKeyProvider,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ConcurrentProfileUpdateError,
    ProfileAlreadyExistsError,
    ProfileIntegrityError,
    ProfileKeyMaterial,
    ProfileStorageLockedError,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_ENVELOPE_SCHEMA_VERSION = 1


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
                raise ProfileIntegrityError("Encrypted object schema version is unsupported")
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
    ) -> None:
        """Atomically create wrapped keys, manifest, and required global scope."""

        if global_scope.profile_id != manifest.profile_id or global_scope.kind is not ScopeKind.GLOBAL:
            raise ValueError("global scope must belong to the profile")
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

    def get_manifest(self, profile_id: str) -> ProfileManifest | None:
        """Return one authenticated manifest without cross-profile fallback."""

        return self._read_model(profile_id, "manifest", profile_id, ProfileManifest)

    def get_scope(self, profile_id: str, scope_id: str) -> ProfileScope | None:
        """Return one authenticated scope for the exact profile."""

        return self._read_model(profile_id, "scope", scope_id, ProfileScope)

    def commit_manifest_version(
        self,
        manifest: ProfileManifest,
        *,
        expected_version_id: str,
    ) -> None:
        """Commit the next manifest revision with an optimistic head check."""

        with self._database.transaction(immediate=True) as connection:
            row = self._head_row(
                connection,
                manifest.profile_id,
                "manifest",
                manifest.profile_id,
            )
            if row is None or str(row["version_id"]) != expected_version_id:
                raise ConcurrentProfileUpdateError("manifest head changed concurrently")
            keys = self._keys.load(manifest.profile_id, connection=connection)
            try:
                current = ProfileManifest.model_validate_json(self._decrypt_row(row, keys))
            except ValidationError:
                raise ProfileIntegrityError("Canonical object validation failed") from None
            if (
                manifest.current_version_id == expected_version_id
                or manifest.revision != current.revision + 1
                or manifest.purge_generation != current.purge_generation
                or manifest.created_at != current.created_at
                or manifest.updated_at < current.updated_at
            ):
                raise ValueError("manifest revision is invalid")
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

    def commit_scope(
        self,
        scope: ProfileScope,
        *,
        expected_version_id: str | None,
    ) -> None:
        """Commit a scope revision with an optimistic head check."""

        with self._database.transaction(immediate=True) as connection:
            current_row = self._head_row(
                connection,
                scope.profile_id,
                "scope",
                scope.scope_id,
            )
            current_version_id = None if current_row is None else str(current_row["version_id"])
            if current_version_id != expected_version_id:
                raise ConcurrentProfileUpdateError("scope head changed concurrently")
            keys = self._keys.load(scope.profile_id, connection=connection)
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

    def commit_proposal(self, proposal: ProfileProposal) -> None:
        """Persist a new pending proposal under a generated storage version."""

        if proposal.state is not ProposalState.PENDING:
            raise ValueError("only pending proposals may be committed")
        version_id = str(uuid.uuid4())
        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(proposal.profile_id, connection=connection)
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
            resolved = ProfileProposal.model_validate(
                {
                    **current.model_dump(mode="python"),
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
                parent_version_id=str(row["version_id"]),
                value=resolved,
            )
            self._set_head(
                connection,
                profile_id=profile_id,
                object_type="proposal",
                object_id=proposal_id,
                version_id=version_id,
                expected_version_id=str(row["version_id"]),
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
                    raise ProfileIntegrityError("Encrypted object schema version is unsupported") from None
                if schema_version != _ENVELOPE_SCHEMA_VERSION:
                    raise ProfileIntegrityError("Encrypted object schema version is unsupported")
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
        policy: Mapping[str, Any],
    ) -> None:
        """Persist one encrypted server-local runtime policy revision."""

        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(profile_id, connection=connection)
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
