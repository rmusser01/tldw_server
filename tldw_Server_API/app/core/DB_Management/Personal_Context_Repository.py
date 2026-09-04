"""Database-layer encrypted Personal Context repository."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import time
import uuid
from collections.abc import Mapping, Sequence
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
    SyncMode,
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
from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    CanonicalApplyReceipt,
    IngressIdentity,
    PersonalContextPublicationJournal,
    PublicationBatchReceipt,
    PublicationObject,
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
_SYNC_HISTORY_KEY_LABEL = b"tldw-personal-context-sync-history-v1"
_DIRECT_CONFIRMED_FULL_PROFILE_PURGE = object()


def _now_text() -> str:
    return _now_datetime().isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _now_datetime() -> datetime:
    """Return a profile-core-compatible millisecond UTC timestamp."""

    now = datetime.now(UTC)
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


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
            "personal_context_publication_profiles",
            "personal_context_publication_batches",
            "personal_context_publication_rows",
            "personal_context_ingress_receipts",
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

    @classmethod
    def _canonical_digest(cls, value: BaseModel | Mapping[str, Any]) -> str:
        """Return the exact canonical-byte digest used for ingress identity."""

        return "sha256:" + hashlib.sha256(cls._canonical_payload(value)).hexdigest()

    @staticmethod
    def _integrity_tag(key: bytes, plaintext: bytes) -> str:
        digest = hmac.new(key, plaintext, hashlib.sha256).hexdigest()
        return f"hmac-sha256-v1:{digest}"

    @staticmethod
    def _publication_object(
        value: BaseModel | Mapping[str, Any],
        *,
        domain: str,
        object_id: str,
        version_id: str,
        role: str = "semantic",
        operation: str = "upsert",
    ) -> PublicationObject:
        """Build an encrypted-only source-publication payload from canonical bytes."""

        return PublicationObject(
            domain=domain,
            object_id=object_id,
            version_id=version_id,
            operation=operation,  # type: ignore[arg-type]
            role=role,  # type: ignore[arg-type]
            canonical=PersonalContextRepository._canonical_payload(value),
        )

    def _append_publication(
        self,
        connection: sqlite3.Connection,
        keys: ProfileKeyMaterial,
        *,
        manifest: ProfileManifest,
        semantic: Sequence[PublicationObject] = (),
        ingress: IngressIdentity | None = None,
    ) -> PublicationBatchReceipt:
        """Append semantic objects before the exact canonical manifest when supplied."""

        objects = list(semantic)
        if semantic or ingress is not None:
            objects.append(
                self._publication_object(
                    manifest,
                    domain="personal_context.manifest",
                    object_id=manifest.profile_id,
                    version_id=manifest.current_version_id,
                    role="manifest",
                )
            )
        else:
            objects = [
                self._publication_object(
                    manifest,
                    domain="personal_context.manifest",
                    object_id=manifest.profile_id,
                    version_id=manifest.current_version_id,
                    role="manifest",
                )
            ]
        return PersonalContextPublicationJournal(keys).append_batch(
            connection,
            profile_id=manifest.profile_id,
            purge_generation=manifest.purge_generation,
            objects=objects,
            ingress=ingress,
            manifest=manifest if ingress is not None else None,
            now=_now_text(),
        )

    def _current_manifest_for_publication(
        self,
        connection: sqlite3.Connection,
        profile_id: str,
        keys: ProfileKeyMaterial,
    ) -> ProfileManifest:
        row = self._head_row(connection, profile_id, "manifest", profile_id)
        if row is None:
            raise ConcurrentProfileUpdateError("manifest head changed concurrently")
        try:
            return ProfileManifest.model_validate_json(self._decrypt_row(row, keys))
        except ValidationError:
            raise ProfileIntegrityError("Canonical object validation failed") from None

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

    def _insert_initial_profile_objects(
        self,
        connection: sqlite3.Connection,
        keys: ProfileKeyMaterial,
        manifest: ProfileManifest,
        global_scope: ProfileScope,
        *,
        runtime_policy: Mapping[str, Any] | None,
        runtime_version_id: str | None,
    ) -> None:
        """Insert the shared initial object set under caller-owned transaction."""

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
                UNION ALL SELECT 1 FROM personal_context_publication_profiles
                UNION ALL SELECT 1 FROM personal_context_publication_batches
                UNION ALL SELECT 1 FROM personal_context_publication_rows
                UNION ALL SELECT 1 FROM personal_context_ingress_receipts
                LIMIT 1
                """
            ).fetchone()
            if surviving_state is not None:
                raise ProfileStorageLockedError("existing profile key material is unavailable")
            keys = self._keys.create(manifest.profile_id, connection=connection)
            self._insert_initial_profile_objects(
                connection,
                keys,
                manifest,
                global_scope,
                runtime_policy=runtime_policy,
                runtime_version_id=runtime_version_id,
            )
            self._append_publication(
                connection,
                keys,
                manifest=manifest,
                semantic=(
                    self._publication_object(
                        global_scope,
                        domain="personal_context.scope",
                        object_id=global_scope.scope_id,
                        version_id=global_scope.version_id,
                    ),
                ),
            )

    def reserve_sync_profile(
        self,
        candidate_profile_id: str,
    ) -> tuple[str, str, str, bytes]:
        """Reserve random profile identity and wrapped keys without canonical objects.

        The key row is control-plane custody only. An interrupted or cancelled
        review therefore leaves no manifest, scope, record, or proposal replica.
        The first serialized caller wins; retries reuse its profile identity.
        """

        with self._database.transaction(immediate=True) as connection:
            object_state = connection.execute(
                """
                SELECT 1 FROM personal_context_object_versions
                UNION ALL SELECT 1 FROM personal_context_object_heads
                UNION ALL SELECT 1 FROM personal_context_runtime_heads
                UNION ALL SELECT 1 FROM personal_context_receipts
                UNION ALL SELECT 1 FROM personal_context_publication_profiles
                UNION ALL SELECT 1 FROM personal_context_publication_batches
                UNION ALL SELECT 1 FROM personal_context_publication_rows
                UNION ALL SELECT 1 FROM personal_context_ingress_receipts
                LIMIT 1
                """
            ).fetchone()
            if object_state is not None:
                raise ProfileAlreadyExistsError("a Personal Context profile exists")
            key_rows = connection.execute(
                """
                SELECT profile_id, created_at
                FROM personal_context_profile_keys
                ORDER BY profile_id
                """
            ).fetchall()
            if len(key_rows) > 1:
                raise ProfileStorageLockedError(
                    "multiple profile key reservations exist"
                )
            if key_rows:
                profile_id = str(key_rows[0]["profile_id"])
                created_at = str(key_rows[0]["created_at"])
                keys = self._keys.load(profile_id, connection=connection)
            else:
                profile_id = candidate_profile_id
                keys = self._keys.create(profile_id, connection=connection)
                row = connection.execute(
                    """
                    SELECT created_at
                    FROM personal_context_profile_keys
                    WHERE profile_id = ?
                    """,
                    (profile_id,),
                ).fetchone()
                if row is None:
                    raise ProfileStorageLockedError(
                        "profile key reservation is unavailable"
                    )
                created_at = str(row["created_at"])
            return (
                profile_id,
                created_at,
                f"personal-context-integrity-v{keys.integrity_key_version}",
                bytes(keys.integrity_key),
            )

    def materialize_sync_profile(
        self,
        manifest: ProfileManifest,
        global_scope: ProfileScope,
        *,
        runtime_policy: Mapping[str, Any] | None = None,
        runtime_version_id: str | None = None,
    ) -> None:
        """Persist one exact reviewed profile plan using its reserved keys."""

        if (
            global_scope.profile_id != manifest.profile_id
            or global_scope.kind is not ScopeKind.GLOBAL
        ):
            raise ValueError("global scope must belong to the profile")
        if (runtime_policy is None) != (runtime_version_id is None):
            raise ValueError("runtime policy and version must be provided together")
        with self._database.transaction(immediate=True) as connection:
            existing_row = self._head_row(
                connection,
                manifest.profile_id,
                "manifest",
                manifest.profile_id,
            )
            if existing_row is not None:
                keys = self._keys.load(manifest.profile_id, connection=connection)
                try:
                    existing_manifest = ProfileManifest.model_validate_json(
                        self._decrypt_row(existing_row, keys)
                    )
                except ValidationError:
                    raise ProfileIntegrityError(
                        "Canonical object validation failed"
                    ) from None
                existing_scope_row = self._head_row(
                    connection,
                    manifest.profile_id,
                    "scope",
                    global_scope.scope_id,
                )
                if existing_scope_row is None:
                    raise ProfileIntegrityError(
                        "Canonical global scope is unavailable"
                    )
                try:
                    existing_scope = ProfileScope.model_validate_json(
                        self._decrypt_row(existing_scope_row, keys)
                    )
                except ValidationError:
                    raise ProfileIntegrityError(
                        "Canonical object validation failed"
                    ) from None
                if existing_manifest != manifest or existing_scope != global_scope:
                    raise ConcurrentProfileUpdateError("reviewed profile plan changed")
                runtime_row = connection.execute(
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
                    (manifest.profile_id, global_scope.scope_id),
                ).fetchone()
                if runtime_policy is None:
                    if runtime_row is not None:
                        raise ConcurrentProfileUpdateError(
                            "reviewed profile plan changed"
                        )
                else:
                    if (
                        runtime_row is None
                        or str(runtime_row["version_id"]) != runtime_version_id
                    ):
                        raise ConcurrentProfileUpdateError(
                            "reviewed profile plan changed"
                        )
                    try:
                        existing_runtime = json.loads(
                            self._decrypt_row(runtime_row, keys)
                        )
                    except (TypeError, ValueError):
                        raise ProfileIntegrityError(
                            "Runtime policy validation failed"
                        ) from None
                    if (
                        not isinstance(existing_runtime, dict)
                        or existing_runtime != dict(runtime_policy)
                    ):
                        raise ConcurrentProfileUpdateError(
                            "reviewed profile plan changed"
                        )
                return

            other_state = connection.execute(
                """
                SELECT 1 FROM personal_context_object_versions
                UNION ALL SELECT 1 FROM personal_context_object_heads
                UNION ALL SELECT 1 FROM personal_context_runtime_heads
                UNION ALL SELECT 1 FROM personal_context_receipts
                UNION ALL SELECT 1 FROM personal_context_publication_profiles
                UNION ALL SELECT 1 FROM personal_context_publication_batches
                UNION ALL SELECT 1 FROM personal_context_publication_rows
                UNION ALL SELECT 1 FROM personal_context_ingress_receipts
                LIMIT 1
                """
            ).fetchone()
            if other_state is not None:
                raise ProfileStorageLockedError(
                    "existing profile state is unavailable"
                )
            key_rows = connection.execute(
                "SELECT profile_id FROM personal_context_profile_keys ORDER BY profile_id"
            ).fetchall()
            if [str(row["profile_id"]) for row in key_rows] != [
                manifest.profile_id
            ]:
                raise ProfileStorageLockedError(
                    "profile key reservation is unavailable"
                )
            keys = self._keys.load(manifest.profile_id, connection=connection)
            self._insert_initial_profile_objects(
                connection,
                keys,
                manifest,
                global_scope,
                runtime_policy=runtime_policy,
                runtime_version_id=runtime_version_id,
            )
            self._append_publication(
                connection,
                keys,
                manifest=manifest,
                semantic=(
                    self._publication_object(
                        global_scope,
                        domain="personal_context.scope",
                        object_id=global_scope.scope_id,
                        version_id=global_scope.version_id,
                    ),
                ),
            )

    def get_manifest(self, profile_id: str) -> ProfileManifest | None:
        """Return one authenticated manifest without cross-profile fallback."""

        return self._read_model(profile_id, "manifest", profile_id, ProfileManifest)

    def apply_ingress_and_publish(
        self,
        *,
        identity: IngressIdentity,
        domain: str,
        value: ProfileManifest | ProfileScope | ProfileRecord | ProfileProposal | Mapping[str, Any],
        base_object_hash: str | None,
    ) -> CanonicalApplyReceipt:
        """Atomically accept one ingress object and its authority publication."""

        with self._database.transaction(immediate=True) as connection:
            if domain == "personal_context.record":
                record = ProfileRecord.model_validate(value)
                if record.controls.sync_mode is SyncMode.DEVICE_ONLY:
                    raise ValueError("Device-only records cannot synchronize")
                profile_id = record.profile_id
                object_type = "record"
                object_id = record.record_id
                version_id = record.version_id
                operation = "tombstone" if record.state is RecordState.DELETED else "upsert"
            elif domain == "personal_context.scope":
                scope = ProfileScope.model_validate(value)
                profile_id = scope.profile_id
                object_type = "scope"
                object_id = scope.scope_id
                version_id = scope.version_id
                operation = "upsert"
            elif domain == "personal_context.proposal":
                proposal = ProfileProposal.model_validate(value)
                if (
                    proposal.proposed_record is not None
                    and proposal.proposed_record.controls.sync_mode is SyncMode.DEVICE_ONLY
                ):
                    raise ValueError("Device-only proposals cannot synchronize")
                profile_id = proposal.profile_id
                object_type = "proposal"
                object_id = proposal.proposal_id
                version_id = str(uuid.uuid4())
                operation = "upsert"
            elif domain == "personal_context.manifest":
                manifest = ProfileManifest.model_validate(value)
                profile_id = manifest.profile_id
                object_type = "manifest"
                object_id = manifest.profile_id
                version_id = manifest.current_version_id
                operation = "upsert"
            else:
                raise ValueError("Unsupported Personal Context Sync domain")
            canonical_input: BaseModel | Mapping[str, Any]
            if object_type == "record":
                canonical_input = record
            elif object_type == "scope":
                canonical_input = scope
            elif object_type == "proposal":
                canonical_input = proposal
            else:
                canonical_input = manifest
            if identity.canonical_payload_digest != self._canonical_digest(canonical_input):
                raise ValueError("ingress canonical payload digest is invalid")
            replay = PersonalContextPublicationJournal.read_ingress_receipt(
                connection, identity
            )
            if replay is not None:
                return replay
            keys = self._keys.load(profile_id, connection=connection)
            current = self._current_manifest_for_publication(connection, profile_id, keys)
            if identity.purge_generation != current.purge_generation:
                raise ConcurrentProfileUpdateError("ingress purge generation changed")
            if object_type == "manifest":
                if base_object_hash != self._canonical_digest(current):
                    raise ConcurrentProfileUpdateError("manifest head changed concurrently")
                self._validate_manifest_transition(
                    current, manifest, expected_version_id=current.current_version_id
                )
                self._insert_manifest_revision(
                    connection, keys, manifest, expected_version_id=current.current_version_id
                )
                batch = self._append_publication(
                    connection, keys, manifest=manifest, ingress=identity
                )
                return CanonicalApplyReceipt(
                    resulting_object_id=manifest.profile_id,
                    resulting_version_id=manifest.current_version_id,
                    manifest_revision=manifest.revision,
                    manifest_version_id=manifest.current_version_id,
                    purge_generation=manifest.purge_generation,
                    publication_batch_id=batch.publication_batch_id,
                    profile_publication_sequence=batch.profile_publication_sequence,
                    receipt_id=PersonalContextPublicationJournal._receipt_id(identity),
                    dataset_id=identity.dataset_id,
                    device_id=identity.device_id,
                    client_envelope_id=identity.client_envelope_id,
                    canonical_payload_digest=identity.canonical_payload_digest,
                )
            existing = self._head_row(connection, profile_id, object_type, object_id)
            expected_version = None if existing is None else str(existing["version_id"])
            existing_value: BaseModel | None = None
            if existing is not None:
                try:
                    model_type = {
                        "record": ProfileRecord,
                        "scope": ProfileScope,
                        "proposal": ProfileProposal,
                    }[object_type]
                    existing_value = model_type.model_validate_json(
                        self._decrypt_row(existing, keys)
                    )
                except (KeyError, ValidationError):
                    raise ProfileIntegrityError("Canonical object validation failed") from None
            expected_base_hash = (
                None
                if existing_value is None
                else self._canonical_digest(existing_value)
            )
            if base_object_hash != expected_base_hash:
                raise ConcurrentProfileUpdateError("canonical object changed concurrently")
            if object_type == "record":
                if record.parent_version_id != expected_version:
                    raise ConcurrentProfileUpdateError("record parent does not match head")
                scope_row = self._head_row(connection, profile_id, "scope", record.scope_id)
                if scope_row is None:
                    raise KeyError("Personal context scope not found")
                if existing_value is not None:
                    existing_record = ProfileRecord.model_validate(existing_value)
                    if (
                        existing_record.state is RecordState.DELETED
                        or record.scope_id != existing_record.scope_id
                        or record.kind is not existing_record.kind
                        or record.created_at != existing_record.created_at
                        or record.updated_at < existing_record.updated_at
                        or record.version_id == existing_record.version_id
                    ):
                        raise ConcurrentProfileUpdateError("record head changed concurrently")
                self._validate_new_head_quota(
                    connection, profile_id, object_type, _MAX_RECORD_HEADS,
                    expected_version_id=expected_version,
                )
                self._validate_semantic_key_available(
                    connection, keys, record,
                    excluding_record_id=None if expected_version is None else record.record_id,
                )
                canonical_value: BaseModel | Mapping[str, Any] = record
            elif object_type == "scope":
                if existing_value is not None:
                    existing_scope = ProfileScope.model_validate(existing_value)
                    if (
                        scope.kind is not existing_scope.kind
                        or scope.created_at != existing_scope.created_at
                        or scope.updated_at < existing_scope.updated_at
                        or scope.version_id == existing_scope.version_id
                    ):
                        raise ConcurrentProfileUpdateError("scope head changed concurrently")
                elif scope.kind is ScopeKind.GLOBAL:
                    scope_rows = connection.execute(
                        """
                        SELECT versions.*
                        FROM personal_context_object_heads AS heads
                        JOIN personal_context_object_versions AS versions
                          ON versions.profile_id = heads.profile_id
                         AND versions.object_type = heads.object_type
                         AND versions.object_id = heads.object_id
                         AND versions.version_id = heads.current_version_id
                        WHERE heads.profile_id = ? AND heads.object_type = 'scope'
                        """,
                        (profile_id,),
                    ).fetchall()
                    for scope_row in scope_rows:
                        candidate = ProfileScope.model_validate_json(
                            self._decrypt_row(scope_row, keys)
                        )
                        if candidate.kind is ScopeKind.GLOBAL:
                            raise ConcurrentProfileUpdateError(
                                "global scope changed concurrently"
                            )
                self._validate_new_head_quota(
                    connection, profile_id, object_type, _MAX_SCOPE_HEADS,
                    expected_version_id=expected_version,
                )
                canonical_value = scope
            else:
                scope_row = self._head_row(connection, profile_id, "scope", proposal.scope_id)
                if scope_row is None:
                    raise KeyError("Personal context scope not found")
                if proposal.state is ProposalState.PENDING:
                    if existing is not None:
                        raise ConcurrentProfileUpdateError("proposal head changed concurrently")
                    canonical_value = proposal
                else:
                    if existing_value is None:
                        raise ConcurrentProfileUpdateError("proposal head changed concurrently")
                    existing_proposal = ProfileProposal.model_validate(existing_value)
                    if existing_proposal.state is not ProposalState.PENDING:
                        raise ConcurrentProfileUpdateError("proposal head changed concurrently")
                    expected_receipt = ProfileProposal.model_validate(
                        {
                            **existing_proposal.model_dump(mode="python"),
                            "state": proposal.state,
                            "proposed_record": None,
                            "confidence": None,
                        }
                    )
                    if expected_receipt != proposal:
                        raise ConcurrentProfileUpdateError("proposal head changed concurrently")
                    canonical_value = self._replace_proposal_with_receipt(
                        connection,
                        keys,
                        existing,
                        existing_proposal,
                        proposal.state,
                        version_id=version_id,
                    )
            if object_type != "proposal" or proposal.state is ProposalState.PENDING:
                self._insert_encrypted(
                    connection, keys, profile_id=profile_id, object_type=object_type,
                    object_id=object_id, version_id=version_id,
                    parent_version_id=expected_version, value=canonical_value,
                )
                self._set_head(
                    connection, profile_id=profile_id, object_type=object_type,
                    object_id=object_id, version_id=version_id,
                    expected_version_id=expected_version,
                )
            next_manifest = ProfileManifest.model_validate(
                {
                    **current.model_dump(mode="python"),
                    "revision": current.revision + 1,
                    "updated_at": _now_datetime(),
                    "current_version_id": str(uuid.uuid4()),
                }
            )
            self._insert_manifest_revision(
                connection, keys, next_manifest,
                expected_version_id=current.current_version_id,
            )
            semantic = self._publication_object(
                canonical_value, domain=domain, object_id=object_id,
                version_id=version_id, operation=operation,
            )
            batch = self._append_publication(
                connection, keys, manifest=next_manifest,
                semantic=(semantic,), ingress=identity,
            )
            return CanonicalApplyReceipt(
                resulting_object_id=object_id,
                resulting_version_id=version_id,
                manifest_revision=next_manifest.revision,
                manifest_version_id=next_manifest.current_version_id,
                purge_generation=next_manifest.purge_generation,
                publication_batch_id=batch.publication_batch_id,
                profile_publication_sequence=batch.profile_publication_sequence,
                receipt_id=PersonalContextPublicationJournal._receipt_id(identity),
                dataset_id=identity.dataset_id,
                device_id=identity.device_id,
                client_envelope_id=identity.client_envelope_id,
                canonical_payload_digest=identity.canonical_payload_digest,
            )

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

    def compact_pre_activation(self, profile_id: str, *, through_sequence: int) -> int:
        """Mark superseded source bodies for later compaction below one watermark.

        The exact encrypted bytes remain recoverable until the activation owner
        completes its independent relay/coverage protocol.  This method only
        advances content-free row state and never touches a newer sequence.
        """

        if through_sequence < 1:
            raise ValueError("publication watermark must be positive")
        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(profile_id, connection=connection)
            profile_row = connection.execute(
                """
                SELECT next_sequence FROM personal_context_publication_profiles
                WHERE profile_id = ?
                """,
                (profile_id,),
            ).fetchone()
            if profile_row is None or through_sequence >= int(profile_row["next_sequence"]):
                raise ValueError("publication watermark is not an exact committed head")
            whole_batch = connection.execute(
                """
                SELECT 1 FROM personal_context_publication_batches
                WHERE profile_id = ? AND profile_publication_sequence = ?
                """,
                (profile_id, through_sequence),
            ).fetchone()
            if whole_batch is None:
                raise ValueError("publication watermark is not an exact committed head")
            rows = connection.execute(
                """
                SELECT * FROM personal_context_publication_rows
                WHERE profile_id = ? AND profile_publication_sequence <= ?
                  AND row_state != 'shredded'
                ORDER BY profile_publication_sequence, batch_ordinal
                """,
                (profile_id, through_sequence),
            ).fetchall()
            journal = PersonalContextPublicationJournal(keys)
            latest: dict[tuple[str, str], sqlite3.Row] = {}
            decoded: list[tuple[sqlite3.Row, tuple[str, str]]] = []
            for row in rows:
                domain, _canonical = journal.decrypt_row(row)
                identity = (domain, str(row["opaque_object_id"]))
                latest[identity] = row
                decoded.append((row, identity))
            superseded = [
                row
                for row, identity in decoded
                if latest[identity] is not row
            ]
            for row in superseded:
                journal.transition_row_state(connection, row, row_state="staged")
            return len(superseded)

    def has_sync_profile_reservation(self) -> bool:
        """Return whether the only durable state is one content-free key reservation."""

        with self._database.transaction() as connection:
            key_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM personal_context_profile_keys"
                ).fetchone()[0]
            )
            object_state = connection.execute(
                """
                SELECT 1 FROM personal_context_object_versions
                UNION ALL SELECT 1 FROM personal_context_object_heads
                UNION ALL SELECT 1 FROM personal_context_runtime_heads
                UNION ALL SELECT 1 FROM personal_context_receipts
                LIMIT 1
                """
            ).fetchone()
        return key_count == 1 and object_state is None

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

    def sync_bootstrap_snapshot(
        self, profile_id: str
    ) -> tuple[
        ProfileManifest,
        tuple[ProfileScope, ...],
        tuple[ProfileRecord, ...],
        tuple[ProfileProposal, ...],
        str,
        bytes,
    ]:
        """Read all bounded canonical Sync heads and key identity in one transaction."""

        with self._database.transaction() as connection:
            keys = self._keys.load(profile_id, connection=connection)
            manifest_row = self._head_row(connection, profile_id, "manifest", profile_id)
            if manifest_row is None:
                raise KeyError("Personal context profile not found")

            def read_heads(
                object_type: str,
                model_type: type[_ModelT],
            ) -> tuple[_ModelT, ...]:
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
                    LIMIT ?
                    """,
                    (profile_id, object_type, _MAX_LIST_ROWS + 1),
                ).fetchall()
                if len(rows) > _MAX_LIST_ROWS:
                    raise ProfileQuotaExceededError("sync bootstrap head limit exceeded")
                try:
                    return tuple(
                        model_type.model_validate_json(self._decrypt_row(row, keys))
                        for row in rows
                    )
                except ValidationError:
                    raise ProfileIntegrityError(
                        "Canonical object validation failed"
                    ) from None

            try:
                manifest = ProfileManifest.model_validate_json(
                    self._decrypt_row(manifest_row, keys)
                )
            except ValidationError:
                raise ProfileIntegrityError("Canonical object validation failed") from None
            scopes = read_heads("scope", ProfileScope)
            records = read_heads("record", ProfileRecord)
            proposals = read_heads("proposal", ProfileProposal)
            key_id = f"personal-context-integrity-v{keys.integrity_key_version}"
            return (
                manifest,
                scopes,
                records,
                proposals,
                key_id,
                bytes(keys.integrity_key),
            )

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
            self._append_publication(connection, keys, manifest=manifest)

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
            self._append_publication(
                connection,
                keys,
                manifest=manifest,
                semantic=(
                    self._publication_object(
                        scope,
                        domain="personal_context.scope",
                        object_id=scope.scope_id,
                        version_id=scope.version_id,
                    ),
                ),
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
            self._append_publication(
                connection,
                keys,
                manifest=self._current_manifest_for_publication(
                    connection, scope.profile_id, keys
                ),
                semantic=(
                    self._publication_object(
                        scope,
                        domain="personal_context.scope",
                        object_id=scope.scope_id,
                        version_id=scope.version_id,
                    ),
                ),
            )

    def commit_record_version(
        self,
        record: ProfileRecord,
        *,
        expected_version_id: str | None,
        allow_orphan_tombstone: bool = False,
    ) -> None:
        """Insert an immutable record and compare-and-set its head atomically."""

        orphan_tombstone = (
            allow_orphan_tombstone
            and expected_version_id is None
            and record.parent_version_id is not None
            and record.state is RecordState.DELETED
            and record.payload is None
        )
        if record.parent_version_id != expected_version_id and not orphan_tombstone:
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
            self._append_publication(
                connection,
                keys,
                manifest=self._current_manifest_for_publication(
                    connection, record.profile_id, keys
                ),
                semantic=(
                    self._publication_object(
                        record,
                        domain="personal_context.record",
                        object_id=record.record_id,
                        version_id=record.version_id,
                        operation=(
                            "tombstone"
                            if record.state is RecordState.DELETED
                            else "upsert"
                        ),
                    ),
                ),
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
            self._append_publication(
                connection,
                keys,
                manifest=manifest,
                semantic=(
                    self._publication_object(
                        record,
                        domain="personal_context.record",
                        object_id=record.record_id,
                        version_id=record.version_id,
                        operation=(
                            "tombstone"
                            if record.state is RecordState.DELETED
                            else "upsert"
                        ),
                    ),
                ),
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
            self._append_publication(
                connection,
                keys,
                manifest=self._current_manifest_for_publication(
                    connection, proposal.profile_id, keys
                ),
                semantic=(
                    self._publication_object(
                        proposal,
                        domain="personal_context.proposal",
                        object_id=proposal.proposal_id,
                        version_id=version_id,
                    ),
                ),
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

    def commit_synced_proposal_receipt(
        self,
        proposal: ProfileProposal,
        *,
        expected_manifest_version: str,
    ) -> None:
        """Commit one exact inbound terminal receipt without a local rewrite."""

        if proposal.state is ProposalState.PENDING:
            raise ValueError("synced proposal receipt must be terminal")
        version_id = str(uuid.uuid4())
        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(proposal.profile_id, connection=connection)
            self._require_writable_manifest_state(
                connection,
                proposal.profile_id,
                expected_manifest_version,
                keys,
            )
            row = self._head_row(
                connection,
                proposal.profile_id,
                "proposal",
                proposal.proposal_id,
            )
            if row is not None:
                current = ProfileProposal.model_validate_json(
                    self._decrypt_row(row, keys)
                )
                if current == proposal:
                    return
                if current.state is not ProposalState.PENDING:
                    raise ConcurrentProfileUpdateError(
                        "proposal head changed concurrently"
                    )
                expected = ProfileProposal.model_validate(
                    {
                        **current.model_dump(mode="python"),
                        "state": proposal.state,
                        "proposed_record": None,
                        "confidence": None,
                    }
                )
                if expected != proposal:
                    raise ConcurrentProfileUpdateError(
                        "synced proposal receipt differs from pending content"
                    )
                receipt = self._replace_proposal_with_receipt(
                    connection,
                    keys,
                    row,
                    current,
                    proposal.state,
                    version_id=version_id,
                )
                self._append_publication(
                    connection,
                    keys,
                    manifest=self._current_manifest_for_publication(
                        connection, proposal.profile_id, keys
                    ),
                    semantic=(
                        self._publication_object(
                            receipt,
                            domain="personal_context.proposal",
                            object_id=receipt.proposal_id,
                            version_id=version_id,
                        ),
                    ),
                )
                return

            self._prune_terminal_proposals_for_insert(
                connection,
                proposal.profile_id,
            )
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
            connection.execute(
                """
                INSERT INTO personal_context_receipts(
                    profile_id, receipt_id, version_id, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    proposal.profile_id,
                    proposal.proposal_id,
                    version_id,
                    _now_text(),
                ),
            )
            self._append_publication(
                connection,
                keys,
                manifest=self._current_manifest_for_publication(
                    connection, proposal.profile_id, keys
                ),
                semantic=(
                    self._publication_object(
                        proposal,
                        domain="personal_context.proposal",
                        object_id=proposal.proposal_id,
                        version_id=version_id,
                    ),
                ),
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
            self._append_publication(
                connection,
                keys,
                manifest=self._current_manifest_for_publication(
                    connection, profile_id, keys
                ),
                semantic=(
                    self._publication_object(
                        resolved,
                        domain="personal_context.proposal",
                        object_id=resolved.proposal_id,
                        version_id=version_id,
                    ),
                ),
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
                receipt = self._replace_proposal_with_receipt(
                    connection,
                    keys,
                    proposal_row,
                    proposal,
                    ProposalState.EXPIRED,
                    version_id=receipt_version,
                )
                self._append_publication(
                    connection,
                    keys,
                    manifest=self._current_manifest_for_publication(
                        connection, profile_id, keys
                    ),
                    semantic=(
                        self._publication_object(
                            receipt,
                            domain="personal_context.proposal",
                            object_id=receipt.proposal_id,
                            version_id=receipt_version,
                        ),
                    ),
                )
                return receipt
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
            self._append_publication(
                connection,
                keys,
                manifest=manifest,
                semantic=(
                    self._publication_object(
                        record,
                        domain="personal_context.record",
                        object_id=record.record_id,
                        version_id=record.version_id,
                        operation=(
                            "tombstone"
                            if record.state is RecordState.DELETED
                            else "upsert"
                        ),
                    ),
                    self._publication_object(
                        receipt,
                        domain="personal_context.proposal",
                        object_id=receipt.proposal_id,
                        version_id=receipt_version,
                    ),
                ),
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
        journal_destruction_authorization: object | None = None,
    ) -> None:
        """Advance the purge barrier and remove every readable profile body.

        Only the service's confirmed direct full-profile purge may pass the
        private authorization capability that destroys old journal DEKs.
        Replica/materializer purge application deliberately omits it.
        """

        if journal_destruction_authorization not in (
            None,
            _DIRECT_CONFIRMED_FULL_PROFILE_PURGE,
        ):
            raise PermissionError("direct purge authorization is invalid")
        destroy_journal_bodies = (
            journal_destruction_authorization is _DIRECT_CONFIRMED_FULL_PROFILE_PURGE
        )

        with self._database.transaction(immediate=True) as connection:
            lease = connection.execute(
                """
                SELECT 1 FROM personal_context_publication_relay_leases
                WHERE profile_id = ? AND expires_at_ns > ?
                """,
                (manifest.profile_id, time.time_ns()),
            ).fetchone()
            if lease is not None:
                raise ConcurrentProfileUpdateError("Personal context relay is active")
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
            self._append_publication(connection, keys, manifest=manifest)
            if destroy_journal_bodies:
                old_publication_rows = connection.execute(
                    """
                    SELECT rows.*
                    FROM personal_context_publication_rows AS rows
                    JOIN personal_context_publication_batches AS batches
                      ON batches.profile_id = rows.profile_id
                     AND batches.profile_publication_sequence = rows.profile_publication_sequence
                    WHERE rows.profile_id = ? AND batches.purge_generation < ?
                    """,
                    (manifest.profile_id, manifest.purge_generation),
                ).fetchall()
                for publication_row in old_publication_rows:
                    PersonalContextPublicationJournal.cryptographically_shred_row(
                        connection,
                        publication_row,
                    )
            connection.execute(
                """
                UPDATE personal_context_publication_batches
                SET status = 'purge_terminal', updated_at = ?
                WHERE profile_id = ? AND purge_generation < ?
                  AND status != 'purge_terminal'
                """,
                (_now_text(), manifest.profile_id, manifest.purge_generation),
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

    def sync_integrity_key(self, profile_id: str) -> tuple[str, bytes]:
        """Return the canonical profile integrity key for its Sync adapter."""

        keys = self._keys.load(profile_id)
        return (
            f"personal-context-integrity-v{keys.integrity_key_version}",
            bytes(keys.integrity_key),
        )

    def sync_encryption_key(self, profile_id: str) -> tuple[bytes, int]:
        """Return a rotation-stable profile-derived key for Sync history."""

        keys = self._keys.load(profile_id)
        storage_key = hmac.new(
            keys.integrity_key,
            _SYNC_HISTORY_KEY_LABEL,
            hashlib.sha256,
        ).digest()
        return storage_key, int(keys.integrity_key_version)

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
            publication_rows = connection.execute(
                """
                SELECT * FROM personal_context_publication_rows
                WHERE profile_id = ? AND row_state != 'shredded'
                """,
                (profile_id,),
            ).fetchall()
            for row in publication_rows:
                envelope = EncryptedEnvelope(
                    algorithm=str(row["algorithm"]),
                    nonce=bytes(row["nonce"]),
                    wrapped_dek=bytes(row["wrapped_dek"]),
                    wrapped_dek_nonce=bytes(row["wrapped_dek_nonce"]),
                    ciphertext=bytes(row["ciphertext"]),
                    key_version=int(row["key_version"]),
                )
                aad = PersonalContextPublicationJournal._aad(
                    profile_id=profile_id,
                    batch_id=str(row["publication_batch_id"]),
                    sequence=int(row["profile_publication_sequence"]),
                    ordinal=int(row["batch_ordinal"]),
                    batch_size=int(row["batch_size"]),
                    role=str(row["role"]),
                    purge_generation=int(row["purge_generation"]),
                    object_id=str(row["opaque_object_id"]),
                    version_id=str(row["opaque_version_id"]),
                    operation=str(row["operation"]),
                    deterministic_envelope_id=str(row["deterministic_envelope_id"]),
                    integrity_tag=str(row["integrity_tag"]),
                    sync_server_cursor=(
                        None
                        if row["sync_server_cursor"] is None
                        else int(row["sync_server_cursor"])
                    ),
                    row_state=str(row["row_state"]),
                )
                try:
                    rewrapped = cipher.rewrap(
                        envelope,
                        aad,
                        new_encryption_key,
                        new_key_version=new_key_version,
                    )
                except EnvelopeAuthenticationError:
                    raise ProfileIntegrityError(
                        "Encrypted publication authentication failed"
                    ) from None
                updated = connection.execute(
                    """
                    UPDATE personal_context_publication_rows
                    SET wrapped_dek = ?, wrapped_dek_nonce = ?, key_version = ?
                    WHERE profile_id = ? AND profile_publication_sequence = ?
                      AND batch_ordinal = ? AND key_version = ?
                    """,
                    (
                        rewrapped.wrapped_dek,
                        rewrapped.wrapped_dek_nonce,
                        rewrapped.key_version,
                        profile_id,
                        row["profile_publication_sequence"],
                        row["batch_ordinal"],
                        current_keys.key_version,
                    ),
                )
                if updated.rowcount != 1:
                    raise ConcurrentProfileUpdateError(
                        "encrypted publication changed concurrently"
                    )
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
