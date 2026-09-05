"""Database-layer encrypted Personal Context repository."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar

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
from tldw_Server_API.app.core.exceptions import (
    PersonalContextActivationInputError,
    PersonalContextActivationMissingError,
    PersonalContextActivationPendingError,
    PersonalContextActivationStaleError,
)
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
    PublicationRelayLease,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ConcurrentProfileUpdateError,
    PreparedPersonalContextActivation,
    ProfileAlreadyExistsError,
    ProfileIntegrityError,
    ProfileKeyMaterial,
    ProfileQuotaExceededError,
    ProfileSemanticKeyCollisionError,
    ProfileStorageLockedError,
    ProfileUnsupportedSchemaError,
)
from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import (
    PersonalContextExchangeProof,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_ENVELOPE_SCHEMA_VERSION = 1
_MAX_PENDING_PROPOSALS = 200
_MAX_PROPOSAL_HEADS = 1_000
_MAX_RECORD_HEADS = 1_000
_MAX_SCOPE_HEADS = 1_000
_MAX_LIST_ROWS = 1_000
_MAX_CONFLICT_HEADS = 1_000
_SYNC_HISTORY_KEY_LABEL = b"tldw-personal-context-sync-history-v1"
_DIRECT_CONFIRMED_FULL_PROFILE_PURGE = object()
_DIRECT_PURGE_CLEANUP_ORIGIN = "direct_confirmed_full_profile_purge"
_DIRECT_PURGE_CLAIM_SECONDS = 60
_ACTIVATION_LEASE_SECONDS = 60
_VERIFIED_DIRECT_PURGE_EXECUTION = object()
_DIRECT_PURGE_CAPABILITY_SIGNING_KEY = secrets.token_bytes(32)


@dataclass(frozen=True, slots=True)
class DirectPurgeCleanupIntent:
    """Content-free authority for one direct-purge retention cleanup."""

    intent_id: str
    profile_id: str
    old_generation_through: int
    purge_generation: int
    state: Literal["pending", "claimed", "complete"]
    owner_token: str | None


@dataclass(frozen=True, slots=True, repr=False)
class _VerifiedDirectPurgeCleanupClaim:
    """Opaque live-journal claim bound to one authenticated Sync target."""

    _repository: PersonalContextRepository
    _intent: DirectPurgeCleanupIntent
    _user_id: str
    _dataset_id: str
    _store: object
    _database: object
    _provenance: object
    _authentication_tag: bytes


@dataclass(frozen=True, slots=True)
class _DirectPurgeCleanupExecution:
    """Validated immutable target snapshot for destructive Sync SQL."""

    repository: PersonalContextRepository
    intent: DirectPurgeCleanupIntent
    user_id: str
    dataset_id: str
    store: object
    database: object

    @property
    def profile_id(self) -> str:
        return self.intent.profile_id

    @property
    def old_generation_through(self) -> int:
        return self.intent.old_generation_through

    @property
    def purge_generation(self) -> int:
        return self.intent.purge_generation


def _direct_purge_capability_tag(
    *,
    repository: PersonalContextRepository,
    intent: DirectPurgeCleanupIntent,
    user_id: str,
    dataset_id: str,
    store: object,
    database: object,
) -> bytes:
    """Authenticate every scalar and object-identity cleanup target."""

    source_database = object.__getattribute__(repository, "_database")
    payload = canonical_json_bytes(
        {
            "repository_identity": id(repository),
            "source_database_identity": id(source_database),
            "store_identity": id(store),
            "database_identity": id(database),
            "user_id": user_id,
            "dataset_id": dataset_id,
            "intent_id": intent.intent_id,
            "profile_id": intent.profile_id,
            "old_generation_through": intent.old_generation_through,
            "purge_generation": intent.purge_generation,
            "state": intent.state,
            "owner_token": intent.owner_token,
        }
    )
    return hmac.new(
        _DIRECT_PURGE_CAPABILITY_SIGNING_KEY,
        payload,
        hashlib.sha256,
    ).digest()


def _validate_direct_purge_cleanup_claim(
    claim: object,
    *,
    expected_store: object | None = None,
    expected_database: object | None = None,
) -> _DirectPurgeCleanupExecution:
    """Return a live target snapshot only for an exact untampered capability."""

    if type(claim) is not _VerifiedDirectPurgeCleanupClaim:
        raise PermissionError("direct purge execution capability type is invalid")
    repository = object.__getattribute__(claim, "_repository")
    raw_intent = object.__getattribute__(claim, "_intent")
    user_id = object.__getattribute__(claim, "_user_id")
    dataset_id = object.__getattribute__(claim, "_dataset_id")
    store = object.__getattribute__(claim, "_store")
    database = object.__getattribute__(claim, "_database")
    provenance = object.__getattribute__(claim, "_provenance")
    authentication_tag = object.__getattribute__(claim, "_authentication_tag")
    if (
        type(repository) is not PersonalContextRepository
        or type(raw_intent) is not DirectPurgeCleanupIntent
        or type(user_id) is not str
        or not user_id
        or type(dataset_id) is not str
        or not dataset_id
        or store is None
        or database is None
        or provenance is not _VERIFIED_DIRECT_PURGE_EXECUTION
        or type(authentication_tag) is not bytes
        or (expected_store is not None and store is not expected_store)
        or (expected_database is not None and database is not expected_database)
        or type(raw_intent.intent_id) is not str
        or not raw_intent.intent_id
        or type(raw_intent.profile_id) is not str
        or not raw_intent.profile_id
        or type(raw_intent.old_generation_through) is not int
        or raw_intent.old_generation_through < 0
        or type(raw_intent.purge_generation) is not int
        or raw_intent.purge_generation != raw_intent.old_generation_through + 1
        or raw_intent.state != "claimed"
        or type(raw_intent.owner_token) is not str
        or not raw_intent.owner_token
    ):
        raise PermissionError("direct purge execution target is invalid")
    intent = DirectPurgeCleanupIntent(
        intent_id=raw_intent.intent_id,
        profile_id=raw_intent.profile_id,
        old_generation_through=raw_intent.old_generation_through,
        purge_generation=raw_intent.purge_generation,
        state=raw_intent.state,
        owner_token=raw_intent.owner_token,
    )
    expected_tag = _direct_purge_capability_tag(
        repository=repository,
        intent=intent,
        user_id=user_id,
        dataset_id=dataset_id,
        store=store,
        database=database,
    )
    if not hmac.compare_digest(authentication_tag, expected_tag):
        raise PermissionError("direct purge execution capability was modified")
    PersonalContextRepository._require_live_direct_purge_cleanup_claim(
        repository,
        intent,
    )
    return _DirectPurgeCleanupExecution(
        repository=repository,
        intent=intent,
        user_id=user_id,
        dataset_id=dataset_id,
        store=store,
        database=database,
    )


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
            "personal_context_purge_cleanup_intents",
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
        if object_type in {"record", "scope", "proposal"}:
            self._require_unfrozen_object(connection, keys, object_type, object_id, value)
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

    def _sync_conflict_rows(self, connection: sqlite3.Connection, profile_id: str) -> list[sqlite3.Row]:
        rows = connection.execute(
            """SELECT versions.* FROM personal_context_object_heads AS heads
               JOIN personal_context_object_versions AS versions
                 ON versions.profile_id = heads.profile_id AND versions.object_type = heads.object_type
                AND versions.object_id = heads.object_id AND versions.version_id = heads.current_version_id
              WHERE heads.profile_id = ? AND heads.object_type = 'sync_conflict' LIMIT ?""",
            (profile_id, _MAX_CONFLICT_HEADS + 1),
        ).fetchall()
        if len(rows) > _MAX_CONFLICT_HEADS:
            raise ProfileQuotaExceededError("Personal Context conflict capacity exhausted")
        return rows

    def _require_unfrozen_object(
        self,
        connection: sqlite3.Connection,
        keys: ProfileKeyMaterial,
        object_type: str,
        object_id: str,
        value: BaseModel | Mapping[str, Any],
    ) -> None:
        """Check exact object and contested key ownership under the canonical write lock."""
        payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
        slot = self._conflict_key_slot(payload) if object_type == "record" else None
        for row in self._sync_conflict_rows(connection, str(payload["profile_id"])):
            journal = json.loads(self._decrypt_row(row, keys))
            if journal["state"] != "unresolved":
                continue
            if [object_type, object_id] in [head[:2] for head in journal["heads"]] or (
                slot is not None and slot == journal["key_slot"]
            ):
                raise ConcurrentProfileUpdateError("Personal Context object is frozen for review")

    @staticmethod
    def _conflict_key_slot(payload: Mapping[str, Any]) -> list[Any] | None:
        if payload.get("state") != "active" or payload.get("semantic_key") is None:
            return None
        return [payload["scope_id"], payload["kind"], payload["semantic_key"]]

    def _write_sync_conflict(
        self,
        connection: sqlite3.Connection,
        keys: ProfileKeyMaterial,
        journal: Mapping[str, Any],
        *,
        previous_version: str | None,
        object_type: str = "sync_conflict",
    ) -> None:
        version = str(uuid.uuid4())
        self._insert_encrypted(
            connection,
            keys,
            profile_id=journal["profile_id"],
            object_type=object_type,
            object_id=journal["conflict_id"],
            version_id=version,
            parent_version_id=previous_version,
            value=journal,
        )
        self._set_head(
            connection,
            profile_id=journal["profile_id"],
            object_type=object_type,
            object_id=journal["conflict_id"],
            version_id=version,
            expected_version_id=previous_version,
        )

    def _sync_conflict_head(
        self, connection: sqlite3.Connection, profile_id: str, conflict_id: str
    ) -> sqlite3.Row | None:
        return self._head_row(connection, profile_id, "sync_conflict", conflict_id) or self._head_row(
            connection,
            profile_id,
            "sync_conflict_receipt",
            conflict_id,
        )

    def get_sync_conflict(self, profile_id: str, conflict_id: str) -> dict[str, Any]:
        """Read the authenticated private conflict journal through canonical custody."""
        with self._database.transaction() as connection:
            keys = self._keys.load(profile_id, connection=connection)
            row = self._sync_conflict_head(connection, profile_id, conflict_id)
            if row is None:
                raise ConcurrentProfileUpdateError("Personal Context conflict is unavailable")
            return json.loads(self._decrypt_row(row, keys))

    @contextmanager
    def sync_conflict_staging_guard(
        self,
        profile_id: str,
        conflict_id: str,
        *,
        dataset_id: str,
        purge_generation: int,
    ) -> Iterator[dict[str, Any]]:
        """Keep canonical authority alive through local protected Sync attachment."""
        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(profile_id, connection=connection)
            manifest = self._current_manifest_for_publication(connection, profile_id, keys)
            self._require_current_writable_manifest_state(connection, profile_id, keys)
            row = self._sync_conflict_head(connection, profile_id, conflict_id)
            if row is None or manifest.purge_generation != purge_generation:
                raise ConcurrentProfileUpdateError("Personal Context conflict authority changed")
            journal = json.loads(self._decrypt_row(row, keys))
            if journal["dataset_id"] != dataset_id or journal["purge_generation"] != purge_generation:
                raise ConcurrentProfileUpdateError("Personal Context conflict authority changed")
            yield journal

    def capture_sync_conflict(
        self,
        *,
        profile_id: str,
        conflict_id: str,
        dataset_id: str,
        device_id: str,
        local_envelope_id: str,
        domain: str,
        object_id: str,
        local_payload: Mapping[str, Any],
        local_envelope_digest: str,
        purge_generation: int,
    ) -> dict[str, Any]:
        """Commit immutable heads and private narrow freezes before Sync reports conflict."""
        with self._database.transaction(immediate=True) as connection:
            keys = self._keys.load(profile_id, connection=connection)
            manifest = self._current_manifest_for_publication(connection, profile_id, keys)
            self._require_current_writable_manifest_state(connection, profile_id, keys)
            if manifest.purge_generation != purge_generation:
                raise ConcurrentProfileUpdateError("Personal Context conflict generation changed")
            identity = {
                "profile_id": profile_id,
                "conflict_id": conflict_id,
                "dataset_id": dataset_id,
                "device_id": device_id,
                "local_envelope_id": local_envelope_id,
                "domain": domain,
                "object_id": object_id,
                "purge_generation": purge_generation,
                "local_digest": self._canonical_digest(local_payload),
                "local_envelope_digest": local_envelope_digest,
            }
            existing = self._sync_conflict_head(connection, profile_id, conflict_id)
            if existing is not None:
                journal = json.loads(self._decrypt_row(existing, keys))
                if any(journal.get(key) != value for key, value in identity.items()):
                    raise ConcurrentProfileUpdateError("Personal Context conflict identity changed")
                return journal
            if len(self._sync_conflict_rows(connection, profile_id)) >= _MAX_CONFLICT_HEADS:
                raise ProfileQuotaExceededError("Personal Context conflict capacity exhausted")
            object_type = domain.removeprefix("personal_context.")
            current = self._head_row(connection, profile_id, object_type, object_id)
            heads = [[object_type, object_id, None if current is None else str(current["version_id"])]]
            contested_slot = None
            if object_type == "record":
                slot = self._conflict_key_slot(local_payload)
                if slot is not None:
                    record_rows = connection.execute(
                        """SELECT versions.* FROM personal_context_object_heads AS heads
                           JOIN personal_context_object_versions AS versions
                             ON versions.profile_id = heads.profile_id AND versions.object_type = heads.object_type
                            AND versions.object_id = heads.object_id AND versions.version_id = heads.current_version_id
                          WHERE heads.profile_id = ? AND heads.object_type = 'record' LIMIT ?""",
                        (profile_id, _MAX_RECORD_HEADS + 1),
                    ).fetchall()
                    if len(record_rows) > _MAX_RECORD_HEADS:
                        raise ProfileQuotaExceededError("record head quota exceeded")
                    for record_row in record_rows:
                        record = ProfileRecord.model_validate_json(self._decrypt_row(record_row, keys))
                        if (
                            record.record_id != object_id
                            and self._conflict_key_slot(record.model_dump(mode="json")) == slot
                            and (record.expires_at is None or record.expires_at > datetime.now(UTC))
                        ):
                            current = self._head_row(connection, profile_id, "record", record.record_id)
                            heads.append(["record", record.record_id, record.version_id])
                            contested_slot = slot
                            break
            if current is None:
                raise ConcurrentProfileUpdateError("Personal Context authority candidate is unavailable")
            payload = json.loads(self._decrypt_row(current, keys))
            source = connection.execute(
                """SELECT * FROM personal_context_publication_rows
                    WHERE profile_id = ? AND opaque_object_id = ? AND opaque_version_id = ?
                    ORDER BY profile_publication_sequence DESC LIMIT 1""",
                (profile_id, current["object_id"], current["version_id"]),
            ).fetchone()
            if source is None:
                raise ConcurrentProfileUpdateError("Personal Context candidate publication is unavailable")
            # A dedicated candidate identity never competes with normal publication lineage.
            candidate_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"tldw:personal-context:candidate:{conflict_id}:{current['version_id']}")
            )
            journal = {
                **identity,
                "state": "unresolved",
                "heads": heads,
                "key_slot": contested_slot,
                "remote_envelope_id": candidate_id,
                "candidate": payload,
                "candidate_object_id": str(current["object_id"]),
                "candidate_version_id": str(current["version_id"]),
                "candidate_created_at": str(current["created_at"]),
                "integrity_key_id": f"personal-context-integrity-v{keys.integrity_key_version}",
                "authority": {
                    "role": "home_authority",
                    "publication_batch_id": str(source["publication_batch_id"]),
                    "profile_publication_sequence": int(source["profile_publication_sequence"]),
                    "batch_ordinal": int(source["batch_ordinal"]),
                    "batch_size": int(source["batch_size"]),
                },
            }
            self._write_sync_conflict(connection, keys, journal, previous_version=None)
            return journal

    def resolve_sync_conflict(
        self,
        *,
        profile_id: str,
        conflict_id: str,
        dataset_id: str,
        device_id: str,
        expected_local_envelope_id: str,
        expected_remote_envelope_id: str,
        idempotency_key: str,
        action: str,
        command: Mapping[str, Any] | None,
        purge_generation: int,
        exchange: PersonalContextExchangeProof,
    ) -> dict[str, Any]:
        """Commit a reviewed choice and its exact replay receipt in one transaction."""
        with self._database.transaction(immediate=True) as connection:
            self.validate_activation_exchange(
                profile_id=profile_id,
                device_id=device_id,
                dataset_id=dataset_id,
                activation_epoch=exchange.activation_epoch,
                continuity_token=exchange.continuity_token,
                _connection=connection,
            )
            keys = self._keys.load(profile_id, connection=connection)
            row = self._sync_conflict_head(connection, profile_id, conflict_id)
            if row is None:
                raise ConcurrentProfileUpdateError("Personal Context conflict is unavailable")
            journal = json.loads(self._decrypt_row(row, keys))
            manifest = self._current_manifest_for_publication(connection, profile_id, keys)
            if (
                journal["dataset_id"] != dataset_id
                or journal["device_id"] != device_id
                or journal["local_envelope_id"] != expected_local_envelope_id
                or journal["remote_envelope_id"] != expected_remote_envelope_id
                or journal["purge_generation"] != purge_generation
                or manifest.purge_generation != purge_generation
                or action not in {"skip", "overwrite", "duplicate_rename"}
                or not idempotency_key
            ):
                raise ConcurrentProfileUpdateError("Personal Context conflict review is stale")
            digest = self._canonical_digest({"action": action, "command": command})
            if journal["state"] == "resolved":
                if journal["command_digest"] != digest or journal["idempotency_key"] != idempotency_key:
                    raise ConcurrentProfileUpdateError("Personal Context resolution command changed")
                return journal["receipt"]
            for head_type, head_id, version in journal["heads"]:
                head = self._head_row(connection, profile_id, head_type, head_id)
                if (None if head is None else str(head["version_id"])) != version:
                    raise ConcurrentProfileUpdateError("Personal Context conflict head changed")
            if (action == "skip") != (command is None):
                raise ValueError("Personal Context resolution payload is invalid")
            if command is not None:
                target = command["object_id"]
                if action == "overwrite" and target != journal["candidate_object_id"]:
                    raise ValueError("Reviewed overwrite must name the shared canonical object")
                if action == "duplicate_rename":
                    if journal["domain"] != "personal_context.record" or target in {
                        item[1] for item in journal["heads"]
                    }:
                        raise ValueError("Personal Context duplicate requires a new record identity")
                    if self._head_row(connection, profile_id, "record", target) is not None:
                        raise ConcurrentProfileUpdateError("Personal Context duplicate identity already exists")
                    if (
                        self._conflict_key_slot(command["payload"]) == journal["key_slot"]
                        and journal["key_slot"] is not None
                    ):
                        raise ValueError("Personal Context duplicate requires a free key")
            # Only this exact journal is released within the same write transaction;
            # any error rolls this back, and other conflicts remain enforced.
            deciding = {**journal, "state": "resolving"}
            self._write_sync_conflict(connection, keys, deciding, previous_version=str(row["version_id"]))
            receipt: dict[str, Any] = {
                "action": action,
                "resulting_object_id": journal["candidate_object_id"],
                "publication_batch_id": None,
            }
            if command is not None:
                result = self.apply_ingress_and_publish(
                    identity=IngressIdentity(
                        dataset_id=dataset_id,
                        device_id=device_id,
                        client_envelope_id=command["client_envelope_id"],
                        canonical_payload_digest=self._canonical_digest(command["payload"]),
                        purge_generation=purge_generation,
                        wire_entity_version=str(command["entity_version"]),
                    ),
                    domain=journal["domain"],
                    value=command["payload"],
                    base_object_hash=command["base_object_hash"],
                    _resolution_connection=connection,
                )
                receipt = {"action": action, **asdict(result)}
            self._write_sync_conflict(
                connection,
                keys,
                {
                    **journal,
                    "state": "resolved",
                    "idempotency_key": idempotency_key,
                    "command_digest": digest,
                    "receipt": receipt,
                },
                previous_version=None,
                object_type="sync_conflict_receipt",
            )
            connection.execute(
                "DELETE FROM personal_context_object_heads WHERE profile_id = ? AND object_type = 'sync_conflict' AND object_id = ?",
                (profile_id, conflict_id),
            )
            return receipt

    def apply_ingress_and_publish(
        self,
        *,
        identity: IngressIdentity,
        domain: str,
        value: ProfileManifest | ProfileScope | ProfileRecord | ProfileProposal | Mapping[str, Any],
        base_object_hash: str | None,
        _resolution_connection: sqlite3.Connection | None = None,
    ) -> CanonicalApplyReceipt:
        """Atomically accept one ingress object and its authority publication."""

        transaction = (
            self._database.transaction(immediate=True)
            if _resolution_connection is None
            else nullcontext(_resolution_connection)
        )
        with transaction as connection:
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
            expected_wire_version = (
                "sync-proposal-sha256:"
                + hashlib.sha256(canonical_bytes(canonical_input)).hexdigest()
                if object_type == "proposal"
                else version_id
            )
            if identity.wire_entity_version != expected_wire_version:
                raise ValueError("ingress wire entity version is invalid")
            keys = self._keys.load(profile_id, connection=connection)
            replay = (
                PersonalContextPublicationJournal(keys).read_ingress_receipt(connection, identity)
                if _resolution_connection is None
                else None
            )
            if replay is not None:
                return replay
            current = self._current_manifest_for_publication(connection, profile_id, keys)
            if identity.purge_generation != current.purge_generation:
                raise ConcurrentProfileUpdateError("ingress purge generation changed")
            if object_type == "manifest":
                current_hashes = {
                    self._canonical_digest(current),
                    self._integrity_tag(keys.integrity_key, self._canonical_payload(current)),
                }
                if base_object_hash not in current_hashes:
                    raise ConcurrentProfileUpdateError("manifest head changed concurrently")
                self._validate_manifest_transition(
                    current, manifest, expected_version_id=current.current_version_id
                )
                self._insert_manifest_revision(
                    connection, keys, manifest, expected_version_id=current.current_version_id
                )
                batch = self._append_publication(
                    connection,
                    keys,
                    manifest=manifest,
                    ingress=identity if _resolution_connection is None else None,
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
                    wire_entity_version=identity.wire_entity_version,
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
            expected_base_hashes = (
                {None}
                if existing_value is None
                else {
                    self._canonical_digest(existing_value),
                    self._integrity_tag(
                        keys.integrity_key,
                        self._canonical_payload(existing_value),
                    ),
                }
            )
            if base_object_hash not in expected_base_hashes:
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
                connection,
                keys,
                manifest=next_manifest,
                semantic=(semantic,),
                ingress=identity if _resolution_connection is None else None,
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
                wire_entity_version=identity.wire_entity_version,
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

    @staticmethod
    def _activation_aad(row: Mapping[str, Any] | sqlite3.Row) -> bytes:
        """Bind an encrypted baseline to its immutable journal identity."""
        return canonical_json_bytes(
            {
                "purpose": "personal-context-activation-v1",
                **{
                    name: row[name]
                    for name in (
                        "profile_id",
                        "device_id",
                        "activation_id",
                        "baseline_digest",
                        "purge_generation",
                        "publication_watermark",
                    )
                },
            }
        )

    @staticmethod
    def _require_activation_lease(
        connection: sqlite3.Connection,
        profile_id: str,
        lease: PublicationRelayLease | None,
    ) -> None:
        """Fence activation with the current owner and bound its larger snapshot work."""
        if (
            lease is None
            or lease.profile_id != profile_id
            or connection.execute(
                """SELECT 1 FROM personal_context_publication_relay_leases
               WHERE profile_id = ? AND owner_token = ? AND expires_at_ns > ?""",
                (profile_id, lease.owner_token, time.time_ns()),
            ).fetchone()
            is None
        ):
            raise PersonalContextActivationStaleError("personal_context_activation_required")
        connection.execute(
            """UPDATE personal_context_publication_relay_leases SET expires_at_ns = ?
               WHERE profile_id = ? AND owner_token = ?""",
            (time.time_ns() + _ACTIVATION_LEASE_SECONDS * 1_000_000_000, profile_id, lease.owner_token),
        )

    def _decode_activation(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> PreparedPersonalContextActivation:
        """Authenticate an exact journal snapshot before returning plaintext."""
        if row["state"] == "expired":
            raise PersonalContextActivationStaleError("personal_context_activation_required")
        keys = self._keys.load(str(row["profile_id"]), connection=connection)
        baseline = EnvelopeCipher(keys.encryption_key, key_version=keys.key_version).decrypt(
            EncryptedEnvelope(
                **{
                    name: row[name]
                    for name in (
                        "algorithm",
                        "key_version",
                        "nonce",
                        "wrapped_dek",
                        "wrapped_dek_nonce",
                        "ciphertext",
                    )
                }
            ),
            self._activation_aad(row),
        )
        if not hmac.compare_digest(hashlib.sha256(baseline).hexdigest(), row["baseline_digest"]):
            raise ProfileIntegrityError("Personal Context activation integrity failed")
        return PreparedPersonalContextActivation(
            **{
                name: row[name]
                for name in (
                    "profile_id",
                    "device_id",
                    "activation_id",
                    "baseline_digest",
                    "purge_generation",
                    "publication_watermark",
                    "state",
                    "sync_receipt_id",
                    "home_server_cursor",
                    "activation_epoch",
                    "continuity_token",
                )
            },
            baseline=baseline,
        )

    def load_activation(self, activation_id: str) -> PreparedPersonalContextActivation:
        """Read and authenticate a durable baseline after a process restart."""
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM personal_context_activations WHERE activation_id = ?",
                (activation_id,),
            ).fetchone()
            if row is None:
                raise PersonalContextActivationMissingError("personal_context_activation_required")
            return self._decode_activation(connection, row)

    def prepare_activation(
        self,
        profile_id: str,
        *,
        device_id: str,
        lease: PublicationRelayLease | None,
        fresh: bool = False,
    ) -> PreparedPersonalContextActivation:
        """Commit one device's exact eligible heads and whole-batch watermark.

        Args:
            profile_id: Canonical profile to snapshot.
            device_id: Nonempty device identifier of at most 128 characters.
            lease: Current publication lease owned by the caller for this profile.
            fresh: Replace an active baseline; prepared and installed baselines
                always replay until their installation or acknowledgment finishes.

        Returns:
            The authenticated durable preparation, including its exact baseline,
            digest, purge generation and whole-batch publication watermark.

        Raises:
            PersonalContextActivationInputError: The device identifier is invalid.
            PersonalContextActivationStaleError: The caller's lease is not current.
            PersonalContextActivationPendingError: Another preparation or an
                incomplete publication batch prevents preparing a fresh baseline.
            ProfileIntegrityError: The encrypted baseline or watermark is invalid.
            ProfileStorageLockedError: Canonical profile keys are unavailable.
        """
        if not device_id or len(device_id) > 128:
            raise PersonalContextActivationInputError("personal_context_activation_required")
        with self._database.transaction(immediate=True) as connection:
            self._require_activation_lease(connection, profile_id, lease)
            manifest, scopes, records, proposals, _key_id, _key = self.sync_bootstrap_snapshot(
                profile_id,
                connection=connection,
            )
            row = connection.execute(
                """SELECT * FROM personal_context_activations
                   WHERE profile_id = ? AND device_id = ? AND purge_generation = ?
                   ORDER BY rowid DESC LIMIT 1""",
                (profile_id, device_id, manifest.purge_generation),
            ).fetchone()
            if row is not None and row["state"] != "expired":
                previous = self._decode_activation(connection, row)
                if row["state"] == "prepared":
                    return previous
                try:
                    self._activation_current_pair(connection, row)
                except PersonalContextActivationStaleError:
                    pass  # A broken continuity proof requires a new exact-head baseline.
                else:
                    if row["state"] == "installed" or not fresh:
                        return previous
            if (
                connection.execute(
                    """SELECT 1 FROM personal_context_activations
                   WHERE profile_id = ? AND purge_generation = ? AND state = 'prepared' LIMIT 1""",
                    (profile_id, manifest.purge_generation),
                ).fetchone()
                is not None
            ):
                raise PersonalContextActivationPendingError("personal_context_activation_pending")
            if (
                connection.execute(
                    """SELECT 1 FROM personal_context_activations a
                   WHERE a.profile_id = ? AND a.purge_generation = ? AND a.home_server_cursor IS NOT NULL
                     AND EXISTS (SELECT 1 FROM personal_context_publication_batches b
                       WHERE b.profile_id = a.profile_id
                         AND b.status NOT IN ('complete','covered_by_activation','purge_terminal'))
                   LIMIT 1""",
                    (profile_id, manifest.purge_generation),
                ).fetchone()
                is not None
            ):
                raise PersonalContextActivationPendingError("personal_context_activation_pending")
            profile = connection.execute(
                "SELECT * FROM personal_context_publication_profiles WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            watermark = 0 if profile is None else int(profile["next_sequence"]) - 1
            if (
                watermark
                and connection.execute(
                    """SELECT 1 FROM personal_context_publication_batches b
                   WHERE b.profile_id = ? AND b.profile_publication_sequence = ?
                     AND b.batch_size = (SELECT COUNT(*) FROM personal_context_publication_rows r
                       WHERE r.profile_id = b.profile_id
                         AND r.profile_publication_sequence = b.profile_publication_sequence)""",
                    (profile_id, watermark),
                ).fetchone()
                is None
            ):
                raise ProfileIntegrityError("Personal Context activation watermark is invalid")
            baseline = canonical_json_bytes(
                {
                    "manifest": manifest.model_dump(mode="json"),
                    "scopes": [item.model_dump(mode="json") for item in scopes],
                    "records": [
                        item.model_dump(mode="json") for item in records if item.controls.sync_mode is SyncMode.SYNCABLE
                    ],
                    "proposals": [
                        item.model_dump(mode="json")
                        for item in proposals
                        if item.proposed_record is None or item.proposed_record.controls.sync_mode is SyncMode.SYNCABLE
                    ],
                }
            )
            identity = {
                "profile_id": profile_id,
                "device_id": device_id,
                "activation_id": str(uuid.uuid4()),
                "baseline_digest": hashlib.sha256(baseline).hexdigest(),
                "purge_generation": manifest.purge_generation,
                "publication_watermark": watermark,
            }
            keys = self._keys.load(profile_id, connection=connection)
            envelope = EnvelopeCipher(keys.encryption_key, key_version=keys.key_version).encrypt(
                baseline,
                self._activation_aad(identity),
            )
            now = _now_text()
            connection.execute(
                """INSERT INTO personal_context_activations (
                   profile_id, device_id, activation_id, baseline_digest, purge_generation,
                   publication_watermark, state, algorithm, key_version, nonce, wrapped_dek,
                   wrapped_dek_nonce, ciphertext, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 'prepared', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    *identity.values(),
                    envelope.algorithm,
                    envelope.key_version,
                    envelope.nonce,
                    envelope.wrapped_dek,
                    envelope.wrapped_dek_nonce,
                    envelope.ciphertext,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM personal_context_activations WHERE activation_id = ?",
                (identity["activation_id"],),
            ).fetchone()
            return self._decode_activation(connection, row)

    @contextmanager
    def activation_install_guard(
        self,
        activation_id: str,
        baseline_digest: str,
        *,
        lease: PublicationRelayLease | None,
    ) -> Iterator[PreparedPersonalContextActivation]:
        """Hold canonical generation and lease stable through the independent Sync commit."""
        with self._database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM personal_context_activations WHERE activation_id = ?",
                (activation_id,),
            ).fetchone()
            if row is None:
                raise PersonalContextActivationMissingError("personal_context_activation_required")
            if not hmac.compare_digest(row["baseline_digest"], baseline_digest):
                raise PersonalContextActivationStaleError("personal_context_activation_required")
            self._require_activation_lease(connection, row["profile_id"], lease)
            keys = self._keys.load(row["profile_id"], connection=connection)
            manifest = self._current_manifest_for_publication(connection, row["profile_id"], keys)
            if manifest.purge_generation != row["purge_generation"]:
                raise PersonalContextActivationStaleError("personal_context_activation_required")
            yield self._decode_activation(connection, row)

    def complete_activation_install(
        self,
        activation_id: str,
        baseline_digest: str,
        sync_receipt_id: str,
        *,
        home_server_cursor: int,
        lease: PublicationRelayLease | None,
    ) -> PreparedPersonalContextActivation:
        """CAS verified Sync installation, source coverage, and continuity in one commit."""
        if not sync_receipt_id or type(home_server_cursor) is not int or home_server_cursor < 0:
            raise PersonalContextActivationInputError("personal_context_activation_required")
        with self._database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM personal_context_activations WHERE activation_id = ?",
                (activation_id,),
            ).fetchone()
            if row is None:
                raise PersonalContextActivationMissingError("personal_context_activation_required")
            if not hmac.compare_digest(row["baseline_digest"], baseline_digest):
                raise PersonalContextActivationStaleError("personal_context_activation_required")
            activation = self._decode_activation(connection, row)
            self._require_activation_lease(connection, activation.profile_id, lease)
            profile = connection.execute(
                "SELECT * FROM personal_context_publication_profiles WHERE profile_id = ?",
                (activation.profile_id,),
            ).fetchone()
            keys = self._keys.load(activation.profile_id, connection=connection)
            manifest = self._current_manifest_for_publication(connection, activation.profile_id, keys)
            if manifest.purge_generation != activation.purge_generation:
                raise PersonalContextActivationStaleError("personal_context_activation_required")
            if row["state"] != "prepared":
                if row["sync_receipt_id"] != sync_receipt_id or row["home_server_cursor"] != home_server_cursor:
                    raise PersonalContextActivationStaleError("personal_context_activation_required")
                self._activation_current_pair(connection, row)
                return self._decode_activation(connection, row)
            epoch = profile["activation_epoch"] if profile else None
            token = profile["continuity_token"] if profile else None
            if not epoch or not token:
                epoch, token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            now = _now_text()
            connection.execute(
                """INSERT INTO personal_context_publication_profiles (
                   profile_id, next_sequence, activation_covered_through_sequence,
                   purge_generation, activation_epoch, continuity_token, updated_at)
                   VALUES (?, 1, ?, ?, ?, ?, ?)
                   ON CONFLICT(profile_id) DO UPDATE SET
                   activation_covered_through_sequence = MAX(activation_covered_through_sequence, excluded.activation_covered_through_sequence),
                   activation_epoch = excluded.activation_epoch, continuity_token = excluded.continuity_token,
                   updated_at = excluded.updated_at""",
                (
                    activation.profile_id,
                    activation.publication_watermark,
                    activation.purge_generation,
                    epoch,
                    token,
                    now,
                ),
            )
            connection.execute(
                """UPDATE personal_context_publication_batches SET status = 'covered_by_activation',
                   activation_id = ?, baseline_digest = ?, sync_receipt_id = ?, updated_at = ?
                   WHERE profile_id = ? AND profile_publication_sequence <= ?
                     AND status NOT IN ('complete','covered_by_activation','purge_terminal')""",
                (
                    activation_id,
                    baseline_digest,
                    sync_receipt_id,
                    now,
                    activation.profile_id,
                    activation.publication_watermark,
                ),
            )
            connection.execute(
                """UPDATE personal_context_activations SET state = 'installed', sync_receipt_id = ?,
                   home_server_cursor = ?, activation_epoch = ?, continuity_token = ?, updated_at = ?
                   WHERE activation_id = ? AND state = 'prepared' AND baseline_digest = ?""",
                (sync_receipt_id, home_server_cursor, epoch, token, now, activation_id, baseline_digest),
            )
            return self._decode_activation(
                connection,
                connection.execute(
                    "SELECT * FROM personal_context_activations WHERE activation_id = ?",
                    (activation_id,),
                ).fetchone(),
            )

    def expire_activation(
        self,
        activation_id: str,
        baseline_digest: str,
        *,
        sync_receipt_id: str,
        lease: PublicationRelayLease | None,
    ) -> None:
        """Retire only one device baseline after the exact Sync installation expires.

        The Sync owner verifies its receipt and expiry before calling. Source
        coverage and the shared continuity pair survive this per-device change.
        """
        if not sync_receipt_id or len(sync_receipt_id) > 128:
            raise PersonalContextActivationInputError("personal_context_activation_required")
        with self._database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM personal_context_activations WHERE activation_id = ?",
                (activation_id,),
            ).fetchone()
            if row is None:
                raise PersonalContextActivationMissingError("personal_context_activation_required")
            if not hmac.compare_digest(row["baseline_digest"], baseline_digest) or row["sync_receipt_id"] not in (
                None,
                sync_receipt_id,
            ):
                raise PersonalContextActivationStaleError("personal_context_activation_required")
            self._require_activation_lease(connection, row["profile_id"], lease)
            if row["state"] == "expired":
                return
            self._decode_activation(connection, row)
            connection.execute(
                "DELETE FROM personal_context_activation_devices WHERE activation_id = ? AND device_id = ?",
                (activation_id, row["device_id"]),
            )
            connection.execute(
                """UPDATE personal_context_activations SET state = 'expired', sync_receipt_id = ?,
                   activation_epoch = NULL, continuity_token = NULL, ciphertext = ?, wrapped_dek = ?,
                   wrapped_dek_nonce = ?, nonce = ?, updated_at = ? WHERE activation_id = ?""",
                (sync_receipt_id, b"", b"", b"", b"", _now_text(), activation_id),
            )

    def compact_activation(self, activation_id: str) -> int:
        """Erase covered source ciphertext only after its content-free proof commits."""
        with self._database.transaction(immediate=True) as connection:
            activation = connection.execute(
                "SELECT * FROM personal_context_activations WHERE activation_id = ?",
                (activation_id,),
            ).fetchone()
            if activation is None:
                raise PersonalContextActivationMissingError("personal_context_activation_required")
            if activation["state"] == "prepared" or not activation["sync_receipt_id"]:
                raise PersonalContextActivationStaleError("personal_context_activation_required")
            return connection.execute(
                """UPDATE personal_context_publication_rows SET ciphertext = ?, wrapped_dek = ?,
                   wrapped_dek_nonce = ?, nonce = ?, row_state = 'shredded'
                   WHERE profile_id = ? AND profile_publication_sequence IN (
                     SELECT profile_publication_sequence FROM personal_context_publication_batches
                     WHERE profile_id = ? AND activation_id = ? AND baseline_digest = ?
                       AND sync_receipt_id = ? AND status = 'covered_by_activation')
                     AND profile_publication_sequence <= (
                       SELECT activation_covered_through_sequence FROM personal_context_publication_profiles
                       WHERE profile_id = ?) AND row_state != 'shredded'""",
                (
                    b"",
                    b"",
                    b"",
                    b"",
                    activation["profile_id"],
                    activation["profile_id"],
                    activation_id,
                    activation["baseline_digest"],
                    activation["sync_receipt_id"],
                    activation["profile_id"],
                ),
            ).rowcount

    def confirm_activation_device(
        self,
        activation_id: str,
        baseline_digest: str,
        device_id: str,
        sync_ack_receipt_id: str,
        *,
        local_receipt_id: str,
        dataset_id: str,
    ) -> PreparedPersonalContextActivation:
        """Record the exact verified independent Sync acknowledgment idempotently."""
        if any(
            not value or len(value) > 128
            for value in (
                sync_ack_receipt_id,
                local_receipt_id,
                dataset_id,
                device_id,
            )
        ):
            raise PersonalContextActivationInputError("personal_context_activation_required")
        with self._database.transaction(immediate=True) as connection:
            row = connection.execute(
                "SELECT * FROM personal_context_activations WHERE activation_id = ?",
                (activation_id,),
            ).fetchone()
            if row is None:
                raise PersonalContextActivationMissingError("personal_context_activation_required")
            if (
                row["device_id"] != device_id
                or row["state"] == "prepared"
                or not hmac.compare_digest(row["baseline_digest"], baseline_digest)
            ):
                raise PersonalContextActivationStaleError("personal_context_activation_required")
            self._activation_current_pair(connection, row)
            expected = (baseline_digest, local_receipt_id, sync_ack_receipt_id, dataset_id)
            receipt = connection.execute(
                """SELECT baseline_digest, local_receipt_id, sync_ack_receipt_id, dataset_id
                   FROM personal_context_activation_devices WHERE activation_id = ? AND device_id = ?""",
                (activation_id, device_id),
            ).fetchone()
            if receipt is not None and tuple(receipt) != expected:
                raise PersonalContextActivationStaleError("personal_context_activation_required")
            connection.execute(
                """INSERT INTO personal_context_activation_devices (
                   activation_id, device_id, baseline_digest, local_receipt_id,
                   sync_ack_receipt_id, dataset_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(activation_id, device_id) DO NOTHING""",
                (activation_id, device_id, *expected, _now_text()),
            )
            connection.execute(
                "UPDATE personal_context_activations SET state = 'active', updated_at = ? WHERE activation_id = ?",
                (_now_text(), activation_id),
            )
            return self._decode_activation(
                connection,
                connection.execute(
                    "SELECT * FROM personal_context_activations WHERE activation_id = ?",
                    (activation_id,),
                ).fetchone(),
            )

    def _activation_current_pair(
        self,
        connection: sqlite3.Connection,
        activation: sqlite3.Row,
    ) -> None:
        """Reject a stored baseline after generation or publication continuity changes."""
        profile = connection.execute(
            "SELECT * FROM personal_context_publication_profiles WHERE profile_id = ?",
            (activation["profile_id"],),
        ).fetchone()
        keys = self._keys.load(activation["profile_id"], connection=connection)
        manifest = self._current_manifest_for_publication(connection, activation["profile_id"], keys)
        if (
            profile is None
            or manifest.purge_generation != activation["purge_generation"]
            or profile["purge_generation"] != activation["purge_generation"]
            or not activation["activation_epoch"]
            or not activation["continuity_token"]
            or not hmac.compare_digest(profile["activation_epoch"] or "", activation["activation_epoch"])
            or not hmac.compare_digest(profile["continuity_token"] or "", activation["continuity_token"])
        ):
            raise PersonalContextActivationStaleError("personal_context_activation_required")

    def validate_activation_exchange(
        self,
        *,
        profile_id: str,
        device_id: str,
        dataset_id: str,
        activation_epoch: str,
        continuity_token: str,
        _connection: sqlite3.Connection | None = None,
    ) -> PersonalContextExchangeProof:
        """Return the canonical proof only for an acknowledged current device baseline."""
        try:
            supplied = PersonalContextExchangeProof(
                ongoing_sync_version=1,
                activation_epoch=activation_epoch,
                continuity_token=continuity_token,
            )
        except (TypeError, ValueError):
            raise PersonalContextActivationInputError("personal_context_activation_required") from None
        transaction = self._database.transaction() if _connection is None else nullcontext(_connection)
        with transaction as connection:
            row = connection.execute(
                """SELECT a.* FROM personal_context_activations a
                   JOIN personal_context_activation_devices d ON d.activation_id = a.activation_id
                     AND d.device_id = a.device_id AND d.baseline_digest = a.baseline_digest
                   WHERE a.profile_id = ? AND a.device_id = ? AND d.dataset_id = ? AND a.state = 'active'
                     AND a.rowid = (SELECT MAX(latest.rowid) FROM personal_context_activations latest
                       WHERE latest.profile_id = a.profile_id AND latest.device_id = a.device_id)
                   ORDER BY a.rowid DESC LIMIT 1""",
                (profile_id, device_id, dataset_id),
            ).fetchone()
            if row is None:
                raise PersonalContextActivationMissingError("personal_context_activation_required")
            self._activation_current_pair(connection, row)
            if not (
                hmac.compare_digest(row["activation_epoch"], supplied.activation_epoch)
                and hmac.compare_digest(row["continuity_token"], supplied.continuity_token)
            ):
                raise PersonalContextActivationStaleError("personal_context_activation_required")
        return supplied

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
        self, profile_id: str, *, connection: sqlite3.Connection | None = None
    ) -> tuple[
        ProfileManifest,
        tuple[ProfileScope, ...],
        tuple[ProfileRecord, ...],
        tuple[ProfileProposal, ...],
        str,
        bytes,
    ]:
        """Read all bounded canonical Sync heads and key identity in one transaction."""

        with (self._database.transaction() if connection is None else nullcontext(connection)) as connection:
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
            if destroy_journal_bodies and not self._database.retention_prerequisites_verified(
                connection
            ):
                raise ProfileIntegrityError(
                    "Canonical SQLite retention prerequisites are not verified"
                )
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
                old_ingress_receipts = connection.execute(
                    """
                    SELECT DISTINCT receipts.dataset_id, receipts.device_id,
                           receipts.client_envelope_id, receipts.receipt_id,
                           receipts.purge_generation,
                           receipts.publication_batch_id,
                           receipts.profile_publication_sequence
                    FROM personal_context_ingress_receipts AS receipts
                    JOIN personal_context_publication_batches AS batches
                      ON batches.publication_batch_id = receipts.publication_batch_id
                     AND batches.profile_publication_sequence = receipts.profile_publication_sequence
                     AND batches.purge_generation = receipts.purge_generation
                    JOIN personal_context_publication_rows AS result
                      ON result.profile_id = batches.profile_id
                     AND result.publication_batch_id = batches.publication_batch_id
                     AND result.profile_publication_sequence = batches.profile_publication_sequence
                     AND result.opaque_object_id = receipts.resulting_object_id
                     AND result.opaque_version_id = receipts.resulting_version_id
                    JOIN personal_context_publication_rows AS published_manifest
                      ON published_manifest.profile_id = batches.profile_id
                     AND published_manifest.publication_batch_id = batches.publication_batch_id
                     AND published_manifest.profile_publication_sequence = batches.profile_publication_sequence
                     AND published_manifest.role = 'manifest'
                     AND published_manifest.opaque_version_id = receipts.resulting_manifest_version_id
                    WHERE batches.profile_id = ? AND batches.purge_generation < ?
                    """,
                    (manifest.profile_id, manifest.purge_generation),
                ).fetchall()
                for receipt in old_ingress_receipts:
                    deleted = connection.execute(
                        """
                        DELETE FROM personal_context_ingress_receipts
                        WHERE dataset_id = ? AND device_id = ?
                          AND client_envelope_id = ? AND receipt_id = ?
                          AND purge_generation = ? AND publication_batch_id = ?
                          AND profile_publication_sequence = ?
                        """,
                        (
                            receipt["dataset_id"],
                            receipt["device_id"],
                            receipt["client_envelope_id"],
                            receipt["receipt_id"],
                            receipt["purge_generation"],
                            receipt["publication_batch_id"],
                            receipt["profile_publication_sequence"],
                        ),
                    )
                    if deleted.rowcount != 1:
                        raise ProfileIntegrityError(
                            "Canonical ingress receipt changed during direct purge"
                        )
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
                now = _now_text()
                inserted = connection.execute(
                    """
                    INSERT INTO personal_context_purge_cleanup_intents(
                        intent_id, profile_id, old_generation_through,
                        purge_generation, origin, state, owner_token,
                        claim_expires_at_ns, created_at, updated_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, 'pending', NULL, NULL, ?, ?, NULL)
                    ON CONFLICT(profile_id, purge_generation) DO NOTHING
                    """,
                    (
                        str(uuid.uuid4()),
                        manifest.profile_id,
                        current.purge_generation,
                        manifest.purge_generation,
                        _DIRECT_PURGE_CLEANUP_ORIGIN,
                        now,
                        now,
                    ),
                )
                if inserted.rowcount != 1:
                    raise ProfileIntegrityError(
                        "Direct purge cleanup intent already exists"
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
            connection.execute(
                """DELETE FROM personal_context_activation_devices WHERE activation_id IN (
                   SELECT activation_id FROM personal_context_activations
                   WHERE profile_id = ? AND purge_generation < ?)""",
                (manifest.profile_id, manifest.purge_generation),
            )
            connection.execute(
                "DELETE FROM personal_context_activations WHERE profile_id = ? AND purge_generation < ?",
                (manifest.profile_id, manifest.purge_generation),
            )

    @staticmethod
    def _direct_purge_cleanup_intent(row: sqlite3.Row) -> DirectPurgeCleanupIntent:
        """Decode one content-free cleanup-intent row."""

        state = str(row["state"])
        if state not in {"pending", "claimed", "complete"}:
            raise ProfileIntegrityError("Direct purge cleanup intent state is invalid")
        return DirectPurgeCleanupIntent(
            intent_id=str(row["intent_id"]),
            profile_id=str(row["profile_id"]),
            old_generation_through=int(row["old_generation_through"]),
            purge_generation=int(row["purge_generation"]),
            state=state,
            owner_token=None if row["owner_token"] is None else str(row["owner_token"]),
        )

    def direct_purge_cleanup(
        self,
        profile_id: str,
        *,
        purge_generation: int,
    ) -> DirectPurgeCleanupIntent | None:
        """Return one already-authorized cleanup intent without claiming it."""

        with self._database.transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM personal_context_purge_cleanup_intents
                WHERE profile_id = ? AND purge_generation = ? AND origin = ?
                """,
                (profile_id, purge_generation, _DIRECT_PURGE_CLEANUP_ORIGIN),
            ).fetchone()
        return None if row is None else self._direct_purge_cleanup_intent(row)

    def completed_direct_purge_cleanup(
        self,
        profile_id: str,
        *,
        purge_generation: int,
    ) -> DirectPurgeCleanupIntent | None:
        """Return one completed cleanup intent for idempotency verification."""

        intent = self.direct_purge_cleanup(
            profile_id,
            purge_generation=purge_generation,
        )
        return intent if intent is not None and intent.state == "complete" else None

    def _require_live_direct_purge_cleanup_claim(
        self,
        intent: DirectPurgeCleanupIntent,
    ) -> None:
        """Re-read the exact claimed direct-purge journal row or fail closed."""

        if intent.state != "claimed" or intent.owner_token is None:
            raise PermissionError("direct purge cleanup claim is not live")
        with self._database.transaction() as connection:
            row = connection.execute(
                """
                SELECT 1 FROM personal_context_purge_cleanup_intents
                WHERE intent_id = ? AND profile_id = ?
                  AND old_generation_through = ? AND purge_generation = ?
                  AND origin = ? AND state = 'claimed' AND owner_token = ?
                  AND claim_expires_at_ns > ?
                """,
                (
                    intent.intent_id,
                    intent.profile_id,
                    intent.old_generation_through,
                    intent.purge_generation,
                    _DIRECT_PURGE_CLEANUP_ORIGIN,
                    intent.owner_token,
                    time.time_ns(),
                ),
            ).fetchone()
        if row is None:
            raise PermissionError("direct purge cleanup claim is not live")

    def verify_direct_purge_cleanup_claim(
        self,
        intent: DirectPurgeCleanupIntent,
        *,
        user_id: str,
        dataset_id: str,
        store: object,
        database: object,
    ) -> _VerifiedDirectPurgeCleanupClaim:
        """Issue an opaque capability for one exact live claim and Sync target."""

        if not user_id or not dataset_id or store is None or database is None:
            raise ValueError("direct purge cleanup execution target is invalid")
        self._require_live_direct_purge_cleanup_claim(intent)
        return _VerifiedDirectPurgeCleanupClaim(
            _repository=self,
            _intent=intent,
            _user_id=user_id,
            _dataset_id=dataset_id,
            _store=store,
            _database=database,
            _provenance=_VERIFIED_DIRECT_PURGE_EXECUTION,
            _authentication_tag=_direct_purge_capability_tag(
                repository=self,
                intent=intent,
                user_id=user_id,
                dataset_id=dataset_id,
                store=store,
                database=database,
            ),
        )

    def checkpoint_direct_purge_storage(self) -> bool:
        """Confirm the application-owned canonical WAL no longer holds old frames."""

        return self._database.checkpoint_retention_history()

    def claim_direct_purge_cleanup(
        self,
        *,
        owner_token: str,
        profile_id: str | None = None,
        purge_generation: int | None = None,
    ) -> DirectPurgeCleanupIntent | None:
        """Owner-fence the oldest pending or expired direct-purge cleanup intent."""

        if not owner_token or len(owner_token.encode("utf-8")) > 256:
            raise ValueError("cleanup owner token is invalid")
        if (profile_id is None) != (purge_generation is None):
            raise ValueError("cleanup profile and generation filters must be paired")
        now_ns = time.time_ns()
        expires_ns = now_ns + (_DIRECT_PURGE_CLAIM_SECONDS * 1_000_000_000)
        filters = ""
        params: list[Any] = [owner_token, now_ns, _DIRECT_PURGE_CLEANUP_ORIGIN]
        if profile_id is not None and purge_generation is not None:
            filters = " AND profile_id = ? AND purge_generation = ?"
            params.extend((profile_id, purge_generation))
        with self._database.transaction(immediate=True) as connection:
            row = connection.execute(
                f"""
                SELECT * FROM personal_context_purge_cleanup_intents
                WHERE (state = 'pending' OR (state = 'claimed' AND owner_token = ?)
                       OR (state = 'claimed' AND claim_expires_at_ns <= ?))
                  AND origin = ?{filters}
                ORDER BY created_at, intent_id
                LIMIT 1
                """,  # nosec B608 - only the fixed optional predicate is composed.
                tuple(params),
            ).fetchone()
            if row is None:
                return None
            updated = connection.execute(
                """
                UPDATE personal_context_purge_cleanup_intents
                SET state = 'claimed', owner_token = ?, claim_expires_at_ns = ?,
                    updated_at = ?
                WHERE intent_id = ? AND origin = ?
                  AND (state = 'pending' OR (state = 'claimed' AND owner_token = ?)
                       OR (state = 'claimed' AND claim_expires_at_ns <= ?))
                """,
                (
                    owner_token,
                    expires_ns,
                    _now_text(),
                    row["intent_id"],
                    _DIRECT_PURGE_CLEANUP_ORIGIN,
                    owner_token,
                    now_ns,
                ),
            )
            if updated.rowcount != 1:
                raise ProfileIntegrityError("Direct purge cleanup claim raced")
            claimed = connection.execute(
                "SELECT * FROM personal_context_purge_cleanup_intents WHERE intent_id = ?",
                (row["intent_id"],),
            ).fetchone()
        if claimed is None:
            raise ProfileIntegrityError("Direct purge cleanup claim disappeared")
        return self._direct_purge_cleanup_intent(claimed)

    def release_direct_purge_cleanup(self, intent: DirectPurgeCleanupIntent) -> None:
        """Return one failed owned claim to pending for prompt recovery."""

        if intent.state != "claimed" or intent.owner_token is None:
            raise ValueError("cleanup intent is not owner-claimed")
        with self._database.transaction(immediate=True) as connection:
            updated = connection.execute(
                """
                UPDATE personal_context_purge_cleanup_intents
                SET state = 'pending', owner_token = NULL,
                    claim_expires_at_ns = NULL, updated_at = ?
                WHERE intent_id = ? AND profile_id = ? AND purge_generation = ?
                  AND state = 'claimed' AND owner_token = ? AND origin = ?
                """,
                (
                    _now_text(),
                    intent.intent_id,
                    intent.profile_id,
                    intent.purge_generation,
                    intent.owner_token,
                    _DIRECT_PURGE_CLEANUP_ORIGIN,
                ),
            )
            if updated.rowcount != 1:
                raise ProfileIntegrityError("Direct purge cleanup release lost ownership")

    def complete_direct_purge_cleanup(self, intent: DirectPurgeCleanupIntent) -> None:
        """Complete exactly one cleanup claim under its current owner fence."""

        if intent.state != "claimed" or intent.owner_token is None:
            raise ValueError("cleanup intent is not owner-claimed")
        now = _now_text()
        with self._database.transaction(immediate=True) as connection:
            updated = connection.execute(
                """
                UPDATE personal_context_purge_cleanup_intents
                SET state = 'complete', claim_expires_at_ns = NULL,
                    updated_at = ?, completed_at = ?
                WHERE intent_id = ? AND profile_id = ? AND purge_generation = ?
                  AND state = 'claimed' AND owner_token = ? AND origin = ?
                  AND claim_expires_at_ns > ?
                """,
                (
                    now,
                    now,
                    intent.intent_id,
                    intent.profile_id,
                    intent.purge_generation,
                    intent.owner_token,
                    _DIRECT_PURGE_CLEANUP_ORIGIN,
                    time.time_ns(),
                ),
            )
            if updated.rowcount != 1:
                completed = connection.execute(
                    """
                    SELECT state, owner_token
                    FROM personal_context_purge_cleanup_intents
                    WHERE intent_id = ? AND profile_id = ? AND purge_generation = ?
                      AND origin = ?
                    """,
                    (
                        intent.intent_id,
                        intent.profile_id,
                        intent.purge_generation,
                        _DIRECT_PURGE_CLEANUP_ORIGIN,
                    ),
                ).fetchone()
                if (
                    completed is None
                    or completed["state"] != "complete"
                    or completed["owner_token"] != intent.owner_token
                ):
                    raise ProfileIntegrityError(
                        "Direct purge cleanup completion lost ownership"
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
            for row in connection.execute(
                "SELECT * FROM personal_context_activations WHERE profile_id = ? AND state != 'expired'", (profile_id,),
            ).fetchall():
                rewrapped = cipher.rewrap(
                    EncryptedEnvelope(**{name: row[name] for name in (
                        "algorithm", "key_version", "nonce", "wrapped_dek", "wrapped_dek_nonce", "ciphertext",
                    )}), self._activation_aad(row), new_encryption_key, new_key_version=new_key_version,
                )
                connection.execute(
                    """UPDATE personal_context_activations
                       SET wrapped_dek = ?, wrapped_dek_nonce = ?, key_version = ?
                       WHERE activation_id = ?""",
                    (rewrapped.wrapped_dek, rewrapped.wrapped_dek_nonce, rewrapped.key_version, row["activation_id"]),
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
