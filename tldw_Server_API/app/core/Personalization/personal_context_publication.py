"""Encrypted, transaction-owned source publications for Personal Context."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tldw_profile_core import ProfileManifest
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.Personalization.personal_context_crypto import (
    EncryptedEnvelope,
    EnvelopeAuthenticationError,
    EnvelopeCipher,
)
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ProfileKeyMaterial,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB


@dataclass(frozen=True, slots=True)
class PublicationObject:
    """One canonical object encrypted into an ordered source-publication batch."""

    domain: str
    object_id: str
    version_id: str
    operation: Literal["upsert", "tombstone"]
    role: Literal["semantic", "manifest", "purge_barrier"]
    canonical: bytes


@dataclass(frozen=True, slots=True)
class IngressIdentity:
    """Durable, content-free identity of one client-originated Sync envelope."""

    dataset_id: str
    device_id: str
    client_envelope_id: str
    canonical_payload_digest: str
    purge_generation: int
    wire_entity_version: str


@dataclass(frozen=True, slots=True)
class PublicationBatchReceipt:
    """Opaque identity of one committed source-publication batch."""

    profile_id: str
    publication_batch_id: str
    profile_publication_sequence: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class CanonicalApplyReceipt:
    """Replay-stable result of accepting one client ingress envelope."""

    resulting_object_id: str
    resulting_version_id: str
    manifest_revision: int
    manifest_version_id: str
    purge_generation: int
    publication_batch_id: str
    profile_publication_sequence: int
    receipt_id: str
    dataset_id: str
    device_id: str
    client_envelope_id: str
    canonical_payload_digest: str
    wire_entity_version: str


@dataclass(frozen=True, slots=True)
class PublicationSourceRow:
    """One authenticated, decrypted relay source row kept only in memory."""

    profile_id: str
    profile_publication_sequence: int
    publication_batch_id: str
    batch_ordinal: int
    batch_size: int
    purge_generation: int
    role: Literal["semantic", "manifest", "purge_barrier"]
    object_id: str
    version_id: str
    operation: Literal["upsert", "tombstone"]
    deterministic_envelope_id: str
    integrity_tag: str
    domain: str
    canonical: bytes
    sync_server_cursor: int | None
    row_state: str
    relay_owner_token: str | None = None


@dataclass(frozen=True, slots=True)
class PublicationSourceBatch:
    """Earliest nonterminal authority-publication batch for a profile."""

    profile_id: str
    profile_publication_sequence: int
    publication_batch_id: str
    rows: tuple[PublicationSourceRow, ...]


class PublicationRelayPoisoned(RuntimeError):
    """Content-free durable attention state for the earliest corrupt batch."""


@dataclass(frozen=True, slots=True)
class PublicationRelayLease:
    """Opaque owner-fenced claim for one bounded external relay operation."""

    profile_id: str
    owner_token: str


class PersonalContextPublicationRelayStore:
    """SQLite-backed source-journal access for the cross-database relay."""

    _locks_guard = threading.Lock()
    _profile_locks: dict[tuple[str, str], threading.RLock] = {}

    def __init__(self, database: PersonalizationDB) -> None:
        self._database = database

    @contextmanager
    def profile_lease(self, profile_id: str) -> Iterator[PublicationRelayLease | None]:
        """Acquire a recoverable SQLite lease shared by every process entry point."""

        key = (self._database.db_path, profile_id)
        with self._locks_guard:
            lock = self._profile_locks.setdefault(key, threading.RLock())
        with lock:
            owner_token = uuid.uuid4().hex
            now = time.time_ns()
            with self._database.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO personal_context_publication_relay_leases(
                        profile_id, owner_token, expires_at_ns
                    ) VALUES (?, ?, ?)
                    """,
                    (profile_id, owner_token, now),
                )
                claimed = connection.execute(
                    """
                    UPDATE personal_context_publication_relay_leases
                    SET owner_token = ?, expires_at_ns = ?
                    WHERE profile_id = ? AND expires_at_ns <= ?
                    """,
                    # Pull recovery bounds every held external stage to 100 ms;
                    # retain the durable fence long enough to cover it safely.
                    (owner_token, now + 1_000_000_000, profile_id, now),
                ).rowcount == 1
            try:
                yield (
                    PublicationRelayLease(profile_id, owner_token)
                    if claimed
                    else None
                )
            finally:
                if claimed:
                    with self._database.transaction(immediate=True) as connection:
                        connection.execute(
                            """
                            UPDATE personal_context_publication_relay_leases
                            SET expires_at_ns = ?
                            WHERE profile_id = ? AND owner_token = ?
                            """,
                            (time.time_ns(), profile_id, owner_token),
                        )

    def renew_lease(self, lease: PublicationRelayLease) -> bool:
        """CAS-renew the current owner before an external stage transition."""

        now = time.time_ns()
        with self._database.transaction(immediate=True) as connection:
            return connection.execute(
                """
                UPDATE personal_context_publication_relay_leases
                SET expires_at_ns = ?
                WHERE profile_id = ? AND owner_token = ? AND expires_at_ns > ?
                """,
                (now + 1_000_000_000, lease.profile_id, lease.owner_token, now),
            ).rowcount == 1

    def row_is_current(
        self, row: PublicationSourceRow, lease: PublicationRelayLease
    ) -> bool:
        """Recheck purge and terminal state immediately before external staging."""

        with self._database.transaction(immediate=True) as connection:
            current = connection.execute(
                """
                SELECT b.status, b.purge_generation, r.row_state
                FROM personal_context_publication_batches b
                JOIN personal_context_publication_rows r
                  ON r.profile_id = b.profile_id
                 AND r.profile_publication_sequence = b.profile_publication_sequence
                WHERE r.profile_id = ? AND r.profile_publication_sequence = ?
                  AND r.batch_ordinal = ?
                """,
                (row.profile_id, row.profile_publication_sequence, row.batch_ordinal),
            ).fetchone()
            lease_row = connection.execute(
                """
                SELECT 1 FROM personal_context_publication_relay_leases
                WHERE profile_id = ? AND owner_token = ? AND expires_at_ns > ?
                """,
                (lease.profile_id, lease.owner_token, time.time_ns()),
            ).fetchone()
        return bool(
            lease_row is not None
            and lease.profile_id == row.profile_id
            and current is not None
            and current["status"] not in {"complete", "covered_by_activation", "purge_terminal"}
            and int(current["purge_generation"]) == row.purge_generation
            and current["row_state"] in {"pending", "staged", "acknowledged"}
        )

    def earliest_nonterminal_batch(
        self,
        profile_id: str,
        *,
        row_limit: int,
    ) -> PublicationSourceBatch | None:
        """Claim and decrypt only the earliest incomplete sequence under SQLite lock."""

        if row_limit < 1:
            raise ValueError("publication row limit must be positive")

        from tldw_Server_API.app.core.DB_Management.Personal_Context_Key_Store import (
            ServerProfileKeyProvider,
        )

        with self._database.transaction(immediate=True) as connection:
            batch = connection.execute(
                """
                SELECT * FROM personal_context_publication_batches
                WHERE profile_id = ?
                  AND status NOT IN ('complete', 'covered_by_activation', 'purge_terminal')
                ORDER BY profile_publication_sequence ASC
                LIMIT 1
                """,
                (profile_id,),
            ).fetchone()
            if batch is None:
                return None
            attention = connection.execute(
                """
                SELECT 1 FROM personal_context_publication_relay_attention
                WHERE profile_id = ? AND profile_publication_sequence = ?
                """,
                (profile_id, batch["profile_publication_sequence"]),
            ).fetchone()
            if attention is not None:
                raise PublicationRelayPoisoned("Personal Context relay needs attention")
            connection.execute(
                """
                UPDATE personal_context_publication_batches
                SET status = 'relaying', updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                WHERE profile_id = ? AND profile_publication_sequence = ?
                  AND status IN ('pending', 'relaying')
                """,
                (profile_id, batch["profile_publication_sequence"]),
            )
            keys = ServerProfileKeyProvider(self._database).load(profile_id, connection=connection)
            journal = PersonalContextPublicationJournal(keys)
            rows = connection.execute(
                """
                SELECT * FROM personal_context_publication_rows
                WHERE profile_id = ? AND profile_publication_sequence = ?
                ORDER BY batch_ordinal ASC
                LIMIT ?
                """,
                (profile_id, batch["profile_publication_sequence"], row_limit),
            ).fetchall()
            sequence = int(batch["profile_publication_sequence"])
            batch_id = str(batch["publication_batch_id"])

        source_rows: list[PublicationSourceRow] = []
        for row in rows:
            try:
                domain, canonical = journal.decrypt_row(row)
            except Exception as exc:  # noqa: BLE001 - ciphertext must fail closed.
                with self._database.transaction(immediate=True) as connection:
                    self._mark_attention_in_transaction(
                        connection,
                        profile_id=profile_id,
                        sequence=sequence,
                    )
                raise PublicationRelayPoisoned(
                    "Personal Context relay needs attention"
                ) from exc
            source_rows.append(
                PublicationSourceRow(
                    profile_id=profile_id,
                    profile_publication_sequence=int(row["profile_publication_sequence"]),
                    publication_batch_id=str(row["publication_batch_id"]),
                    batch_ordinal=int(row["batch_ordinal"]),
                    batch_size=int(row["batch_size"]),
                    purge_generation=int(row["purge_generation"]),
                    role=str(row["role"]),  # type: ignore[arg-type]
                    object_id=str(row["opaque_object_id"]),
                    version_id=str(row["opaque_version_id"]),
                    operation=str(row["operation"]),  # type: ignore[arg-type]
                    deterministic_envelope_id=str(row["deterministic_envelope_id"]),
                    integrity_tag=str(row["integrity_tag"]),
                    domain=domain,
                    canonical=canonical,
                    sync_server_cursor=(
                        None
                        if row["sync_server_cursor"] is None
                        else int(row["sync_server_cursor"])
                    ),
                    row_state=str(row["row_state"]),
                )
            )
        return PublicationSourceBatch(
            profile_id=profile_id,
            profile_publication_sequence=sequence,
            publication_batch_id=batch_id,
            rows=tuple(source_rows),
        )

    @staticmethod
    def _mark_attention_in_transaction(
        connection: sqlite3.Connection,
        *,
        profile_id: str,
        sequence: int,
    ) -> None:
        connection.execute(
            """
            INSERT OR IGNORE INTO personal_context_publication_relay_attention(
                profile_id, profile_publication_sequence, error_code, created_at
            ) VALUES (?, ?, 'relay_poisoned', strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            """,
            (profile_id, sequence),
        )

    def mark_attention(
        self,
        batch: PublicationSourceBatch,
        *,
        lease: PublicationRelayLease,
    ) -> None:
        """Persist attention without preserving canonical values or exception text."""

        with self._database.transaction(immediate=True) as connection:
            owned = connection.execute(
                """SELECT 1 FROM personal_context_publication_relay_leases
                   WHERE profile_id = ? AND owner_token = ? AND expires_at_ns > ?""",
                (lease.profile_id, lease.owner_token, time.time_ns()),
            ).fetchone()
            current = connection.execute(
                """SELECT 1 FROM personal_context_publication_batches
                   WHERE profile_id = ? AND profile_publication_sequence = ?
                     AND publication_batch_id = ? AND status = 'relaying'
                     AND purge_generation = ?""",
                (
                    batch.profile_id,
                    batch.profile_publication_sequence,
                    batch.publication_batch_id,
                    batch.rows[0].purge_generation if batch.rows else -1,
                ),
            ).fetchone()
            if owned is None or current is None or lease.profile_id != batch.profile_id:
                raise RuntimeError("publication relay source claim changed")
            self._mark_attention_in_transaction(
                connection,
                profile_id=batch.profile_id,
                sequence=batch.profile_publication_sequence,
            )

    def acknowledge_row(
        self, row: PublicationSourceRow, *, server_cursor: int, lease: PublicationRelayLease
    ) -> None:
        """Record an exact durable Sync receipt without persisting canonical data."""

        from tldw_Server_API.app.core.DB_Management.Personal_Context_Key_Store import (
            ServerProfileKeyProvider,
        )

        with self._database.transaction(immediate=True) as connection:
            lease_row = connection.execute(
                """
                SELECT 1 FROM personal_context_publication_relay_leases
                WHERE profile_id = ? AND owner_token = ? AND expires_at_ns > ?
                """,
                (lease.profile_id, lease.owner_token, time.time_ns()),
            ).fetchone()
            if lease_row is None:
                raise RuntimeError("publication relay lease changed")
            current = connection.execute(
                """
                SELECT * FROM personal_context_publication_rows
                WHERE profile_id = ? AND profile_publication_sequence = ? AND batch_ordinal = ?
                """,
                (row.profile_id, row.profile_publication_sequence, row.batch_ordinal),
            ).fetchone()
            if current is None:
                raise RuntimeError("publication row is unavailable")
            profile = connection.execute(
                "SELECT purge_generation FROM personal_context_publication_profiles WHERE profile_id = ?",
                (row.profile_id,),
            ).fetchone()
            if profile is None or int(profile["purge_generation"]) != row.purge_generation:
                raise RuntimeError("publication row is stale")
            if current["row_state"] == "acknowledged":
                if int(current["sync_server_cursor"]) != server_cursor:
                    raise RuntimeError("publication receipt changed concurrently")
                return
            journal = PersonalContextPublicationJournal(
                ServerProfileKeyProvider(self._database).load(row.profile_id, connection=connection)
            )
            journal.transition_row_state(
                connection,
                current,
                row_state="acknowledged",
                sync_server_cursor=server_cursor,
            )

    def record_staged_row(
        self, row: PublicationSourceRow, *, server_cursor: int, lease: PublicationRelayLease
    ) -> None:
        """Persist an invisible Sync cursor under the exact live source claim."""

        from tldw_Server_API.app.core.DB_Management.Personal_Context_Key_Store import (
            ServerProfileKeyProvider,
        )

        with self._database.transaction(immediate=True) as connection:
            lease_row = connection.execute(
                """SELECT 1 FROM personal_context_publication_relay_leases
                   WHERE profile_id = ? AND owner_token = ? AND expires_at_ns > ?""",
                (lease.profile_id, lease.owner_token, time.time_ns()),
            ).fetchone()
            current = connection.execute(
                """SELECT * FROM personal_context_publication_rows
                   WHERE profile_id = ? AND profile_publication_sequence = ?
                     AND batch_ordinal = ?""",
                (row.profile_id, row.profile_publication_sequence, row.batch_ordinal),
            ).fetchone()
            profile = connection.execute(
                "SELECT purge_generation FROM personal_context_publication_profiles WHERE profile_id = ?",
                (row.profile_id,),
            ).fetchone()
            if (
                lease_row is None
                or lease.profile_id != row.profile_id
                or current is None
                or profile is None
                or int(profile["purge_generation"]) != row.purge_generation
                or current["row_state"] not in {"pending", "staged"}
            ):
                raise RuntimeError("publication relay source claim changed")
            if (
                current["row_state"] == "staged"
                and int(current["sync_server_cursor"]) == server_cursor
            ):
                return
            journal = PersonalContextPublicationJournal(
                ServerProfileKeyProvider(self._database).load(
                    row.profile_id, connection=connection
                )
            )
            journal.transition_row_state(
                connection,
                current,
                row_state="staged",
                sync_server_cursor=server_cursor,
            )

    def complete_if_acknowledged(
        self,
        batch: PublicationSourceBatch,
        *,
        lease: PublicationRelayLease,
    ) -> bool:
        """Advance a batch terminally only once every row has a durable receipt."""

        with self._database.transaction(immediate=True) as connection:
            lease_row = connection.execute(
                """
                SELECT 1 FROM personal_context_publication_relay_leases
                WHERE profile_id = ? AND owner_token = ? AND expires_at_ns > ?
                """,
                (lease.profile_id, lease.owner_token, time.time_ns()),
            ).fetchone()
            if lease.profile_id != batch.profile_id or lease_row is None:
                return False
            pending = connection.execute(
                """
                SELECT 1 FROM personal_context_publication_rows
                WHERE profile_id = ? AND profile_publication_sequence = ?
                  AND row_state != 'acknowledged'
                LIMIT 1
                """,
                (batch.profile_id, batch.profile_publication_sequence),
            ).fetchone()
            if pending is not None:
                return False
            updated = connection.execute(
                """
                UPDATE personal_context_publication_batches
                SET status = 'complete', updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
                WHERE profile_id = ? AND profile_publication_sequence = ?
                  AND status = 'relaying'
                """,
                (batch.profile_id, batch.profile_publication_sequence),
            )
            return updated.rowcount == 1


class PersonalContextPublicationJournal:
    """Append complete encrypted batches inside the canonical write transaction."""

    def __init__(self, keys: ProfileKeyMaterial) -> None:
        self._keys = keys

    @staticmethod
    def _aad(
        *,
        profile_id: str,
        batch_id: str,
        sequence: int,
        ordinal: int,
        batch_size: int,
        role: str,
        purge_generation: int,
        object_id: str,
        version_id: str,
        operation: str,
        deterministic_envelope_id: str,
        integrity_tag: str,
        sync_server_cursor: int | None,
        row_state: str,
    ) -> bytes:
        return canonical_json_bytes(
            {
                "batch_id": batch_id,
                "batch_size": batch_size,
                "deterministic_envelope_id": deterministic_envelope_id,
                "envelope": "tldw-personal-context-publication-v1",
                "integrity_tag": integrity_tag,
                "ordinal": ordinal,
                "opaque_object_id": object_id,
                "opaque_version_id": version_id,
                "operation": operation,
                "profile_id": profile_id,
                "purge_generation": purge_generation,
                "role": role,
                "sequence": sequence,
                "sync_server_cursor": sync_server_cursor,
                "row_state": row_state,
            }
        )

    @staticmethod
    def _batch_id(profile_id: str, sequence: int, purge_generation: int) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"tldw:personal-context:publication:{profile_id}:{sequence}:{purge_generation}",
            )
        )

    @staticmethod
    def _envelope_id(batch_id: str, ordinal: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{batch_id}:{ordinal}"))

    @staticmethod
    def _receipt_id(identity: IngressIdentity) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "tldw:personal-context:ingress:"
                f"{identity.dataset_id}:{identity.device_id}:{identity.client_envelope_id}",
            )
        )

    def _claim_next_sequence(
        self,
        connection: sqlite3.Connection,
        *,
        profile_id: str,
        purge_generation: int,
        now: str,
    ) -> int:
        connection.execute(
            """
            INSERT OR IGNORE INTO personal_context_publication_profiles(
                profile_id, next_sequence, activation_covered_through_sequence,
                purge_generation, updated_at
            ) VALUES (?, 1, 0, ?, ?)
            """,
            (profile_id, purge_generation, now),
        )
        row = connection.execute(
            """
            SELECT next_sequence FROM personal_context_publication_profiles
            WHERE profile_id = ?
            """,
            (profile_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("Personal Context publication state is unavailable")
        sequence = int(row["next_sequence"])
        updated = connection.execute(
            """
            UPDATE personal_context_publication_profiles
            SET next_sequence = ?, purge_generation = ?, updated_at = ?
            WHERE profile_id = ? AND next_sequence = ?
            """,
            (sequence + 1, purge_generation, now, profile_id, sequence),
        )
        if updated.rowcount != 1:
            raise RuntimeError("Personal Context publication sequence changed concurrently")
        return sequence

    def append_batch(
        self,
        connection: sqlite3.Connection,
        *,
        profile_id: str,
        purge_generation: int,
        objects: Sequence[PublicationObject],
        ingress: IngressIdentity | None = None,
        manifest: ProfileManifest | None = None,
        now: str,
    ) -> PublicationBatchReceipt:
        """Insert one complete, ordered, encrypted batch under caller-owned SQL state."""

        if not objects:
            raise ValueError("publication batch must contain at least one object")
        if ingress is not None and ingress.purge_generation != purge_generation:
            raise ValueError("ingress purge generation is invalid")
        sequence = self._claim_next_sequence(
            connection,
            profile_id=profile_id,
            purge_generation=purge_generation,
            now=now,
        )
        batch_id = self._batch_id(profile_id, sequence, purge_generation)
        batch_size = len(objects)
        connection.execute(
            """
            INSERT INTO personal_context_publication_batches(
                profile_id, profile_publication_sequence, publication_batch_id,
                purge_generation, batch_size, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (profile_id, sequence, batch_id, purge_generation, batch_size, now, now),
        )
        cipher = EnvelopeCipher(self._keys.encryption_key, key_version=self._keys.key_version)
        for ordinal, item in enumerate(objects):
            payload = canonical_json_bytes(
                {"canonical": item.canonical.decode("utf-8"), "domain": item.domain}
            )
            deterministic_envelope_id = self._envelope_id(batch_id, ordinal)
            integrity_tag = "hmac-sha256-v1:" + hmac.new(
                self._keys.integrity_key,
                payload,
                hashlib.sha256,
            ).hexdigest()
            aad = self._aad(
                profile_id=profile_id,
                batch_id=batch_id,
                sequence=sequence,
                ordinal=ordinal,
                batch_size=batch_size,
                role=item.role,
                purge_generation=purge_generation,
                object_id=item.object_id,
                version_id=item.version_id,
                operation=item.operation,
                deterministic_envelope_id=deterministic_envelope_id,
                integrity_tag=integrity_tag,
                sync_server_cursor=None,
                row_state="pending",
            )
            envelope = cipher.encrypt(payload, aad)
            connection.execute(
                """
                INSERT INTO personal_context_publication_rows(
                    profile_id, profile_publication_sequence, publication_batch_id,
                    batch_ordinal, batch_size, purge_generation, role,
                    opaque_object_id, opaque_version_id, operation, algorithm,
                    key_version, nonce, wrapped_dek, wrapped_dek_nonce, ciphertext,
                    integrity_tag, payload_size_bytes, deterministic_envelope_id,
                    row_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    profile_id,
                    sequence,
                    batch_id,
                    ordinal,
                    batch_size,
                    purge_generation,
                    item.role,
                    item.object_id,
                    item.version_id,
                    item.operation,
                    envelope.algorithm,
                    envelope.key_version,
                    envelope.nonce,
                    envelope.wrapped_dek,
                    envelope.wrapped_dek_nonce,
                    envelope.ciphertext,
                    integrity_tag,
                    len(payload),
                    deterministic_envelope_id,
                ),
            )
        if ingress is not None:
            if manifest is None:
                raise ValueError("ingress publication requires a manifest result")
            semantic = next((item for item in objects if item.role == "semantic"), None)
            result = semantic or next(item for item in objects if item.role == "manifest")
            connection.execute(
                """
                INSERT INTO personal_context_ingress_receipts(
                    dataset_id, device_id, client_envelope_id,
                    canonical_payload_digest, purge_generation, wire_entity_version,
                    resulting_object_id, resulting_version_id,
                    resulting_manifest_revision, resulting_manifest_version_id,
                    publication_batch_id, profile_publication_sequence, receipt_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ingress.dataset_id,
                    ingress.device_id,
                    ingress.client_envelope_id,
                    ingress.canonical_payload_digest,
                    ingress.purge_generation,
                    ingress.wire_entity_version,
                    result.object_id,
                    result.version_id,
                    manifest.revision,
                    manifest.current_version_id,
                    batch_id,
                    sequence,
                    self._receipt_id(ingress),
                    now,
                ),
            )
        return PublicationBatchReceipt(
            profile_id=profile_id,
            publication_batch_id=batch_id,
            profile_publication_sequence=sequence,
            batch_size=batch_size,
        )

    def decrypt_row(self, row: sqlite3.Row) -> tuple[str, bytes]:
        """Authenticate one journal row and return its encrypted route/body pair."""

        aad = self._aad(
            profile_id=str(row["profile_id"]),
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
        envelope = EncryptedEnvelope(
            algorithm=str(row["algorithm"]),
            nonce=bytes(row["nonce"]),
            wrapped_dek=bytes(row["wrapped_dek"]),
            wrapped_dek_nonce=bytes(row["wrapped_dek_nonce"]),
            ciphertext=bytes(row["ciphertext"]),
            key_version=int(row["key_version"]),
        )
        plaintext = EnvelopeCipher(
            self._keys.encryption_key,
            key_version=self._keys.key_version,
        ).decrypt(envelope, aad)
        expected = "hmac-sha256-v1:" + hmac.new(
            self._keys.integrity_key,
            plaintext,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, str(row["integrity_tag"])):
            raise EnvelopeAuthenticationError("publication authentication failed")
        payload = json.loads(plaintext)
        if not isinstance(payload, dict) or not isinstance(payload.get("domain"), str):
            raise EnvelopeAuthenticationError("publication authentication failed")
        canonical = payload.get("canonical")
        if not isinstance(canonical, str):
            raise EnvelopeAuthenticationError("publication authentication failed")
        return payload["domain"], canonical.encode("utf-8")

    def transition_row_state(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        row_state: Literal["pending", "staged", "acknowledged", "shredded"],
        sync_server_cursor: int | None = None,
    ) -> None:
        """Re-encrypt one row before changing authenticated relay state."""

        domain, canonical = self.decrypt_row(row)
        payload = canonical_json_bytes(
            {"canonical": canonical.decode("utf-8"), "domain": domain}
        )
        aad = self._aad(
            profile_id=str(row["profile_id"]),
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
            sync_server_cursor=sync_server_cursor if sync_server_cursor is not None else (
                None if row["sync_server_cursor"] is None else int(row["sync_server_cursor"])
            ),
            row_state=row_state,
        )
        envelope = EnvelopeCipher(
            self._keys.encryption_key,
            key_version=self._keys.key_version,
        ).encrypt(payload, aad)
        updated = connection.execute(
            """
            UPDATE personal_context_publication_rows
            SET nonce = ?, wrapped_dek = ?, wrapped_dek_nonce = ?, ciphertext = ?,
                key_version = ?, algorithm = ?, row_state = ?, sync_server_cursor = ?
            WHERE profile_id = ? AND profile_publication_sequence = ?
              AND batch_ordinal = ? AND row_state = ?
            """,
            (
                envelope.nonce,
                envelope.wrapped_dek,
                envelope.wrapped_dek_nonce,
                envelope.ciphertext,
                envelope.key_version,
                envelope.algorithm,
                row_state,
                sync_server_cursor if sync_server_cursor is not None else row["sync_server_cursor"],
                row["profile_id"],
                row["profile_publication_sequence"],
                row["batch_ordinal"],
                row["row_state"],
            ),
        )
        if updated.rowcount != 1:
            raise RuntimeError("publication row changed concurrently")

    @staticmethod
    def cryptographically_shred_row(
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> None:
        """Destroy one old-generation row DEK during an explicit profile purge."""

        if row["row_state"] == "shredded":
            return
        updated = connection.execute(
            """
            UPDATE personal_context_publication_rows
            SET wrapped_dek = ?, wrapped_dek_nonce = ?, row_state = 'shredded'
            WHERE profile_id = ? AND profile_publication_sequence = ?
              AND batch_ordinal = ?
            """,
            (
                secrets.token_bytes(len(bytes(row["wrapped_dek"]))),
                secrets.token_bytes(len(bytes(row["wrapped_dek_nonce"]))),
                row["profile_id"],
                row["profile_publication_sequence"],
                row["batch_ordinal"],
            ),
        )
        if updated.rowcount != 1:
            raise RuntimeError("publication row changed concurrently")

    def read_ingress_receipt(
        self,
        connection: sqlite3.Connection,
        identity: IngressIdentity,
    ) -> CanonicalApplyReceipt | None:
        """Return a prior replay receipt or reject identity reuse with new bytes."""

        row = connection.execute(
            """
            SELECT * FROM personal_context_ingress_receipts
            WHERE dataset_id = ? AND device_id = ? AND client_envelope_id = ?
            """,
            (identity.dataset_id, identity.device_id, identity.client_envelope_id),
        ).fetchone()
        if row is None:
            return None
        old_identity_matches = (
            str(row["dataset_id"]) == identity.dataset_id
            and str(row["device_id"]) == identity.device_id
            and str(row["client_envelope_id"]) == identity.client_envelope_id
            and str(row["canonical_payload_digest"])
            == identity.canonical_payload_digest
            and int(row["purge_generation"]) == identity.purge_generation
            and str(row["receipt_id"]) == self._receipt_id(identity)
            and bool(str(row["resulting_object_id"]))
            and bool(str(row["resulting_version_id"]))
            and int(row["resulting_manifest_revision"]) >= 0
            and bool(str(row["resulting_manifest_version_id"]))
            and bool(str(row["publication_batch_id"]))
            and int(row["profile_publication_sequence"]) >= 1
        )
        if not old_identity_matches:
            raise ValueError("ingress identity reused with a different payload")
        source_row = connection.execute(
            """
            SELECT result.*
            FROM personal_context_publication_batches AS batch
            JOIN personal_context_publication_rows AS result
              ON result.publication_batch_id = batch.publication_batch_id
             AND result.profile_publication_sequence = batch.profile_publication_sequence
            WHERE batch.publication_batch_id = ?
              AND batch.profile_publication_sequence = ?
              AND batch.purge_generation = ?
              AND result.opaque_object_id = ?
              AND result.opaque_version_id = ?
              AND EXISTS (
                    SELECT 1 FROM personal_context_publication_rows AS manifest
                    WHERE manifest.publication_batch_id = batch.publication_batch_id
                      AND manifest.profile_publication_sequence = batch.profile_publication_sequence
                      AND manifest.role = 'manifest'
                      AND manifest.opaque_version_id = ?
              )
            LIMIT 1
            """,
            (
                row["publication_batch_id"],
                row["profile_publication_sequence"],
                row["purge_generation"],
                row["resulting_object_id"],
                row["resulting_version_id"],
                row["resulting_manifest_version_id"],
            ),
        ).fetchone()
        if source_row is None:
            raise ValueError("ingress identity reused with a different payload")
        try:
            _domain, canonical = self.decrypt_row(source_row)
        except Exception as exc:  # noqa: BLE001 - legacy backfill must fail closed.
            raise ValueError(
                "ingress identity reused with a different payload"
            ) from exc
        source_digest = "sha256:" + hashlib.sha256(canonical).hexdigest()
        if not hmac.compare_digest(source_digest, identity.canonical_payload_digest):
            raise ValueError("ingress identity reused with a different payload")
        stored_wire_version = str(row["wire_entity_version"])
        if stored_wire_version == "":
            updated = connection.execute(
                """
                UPDATE personal_context_ingress_receipts
                SET wire_entity_version = ?
                WHERE dataset_id = ? AND device_id = ? AND client_envelope_id = ?
                  AND wire_entity_version = ''
                """,
                (
                    identity.wire_entity_version,
                    identity.dataset_id,
                    identity.device_id,
                    identity.client_envelope_id,
                ),
            )
            if updated.rowcount != 1:
                raise ValueError("ingress identity reused with a different payload")
            stored_wire_version = identity.wire_entity_version
        if stored_wire_version != identity.wire_entity_version:
            raise ValueError("ingress identity reused with a different payload")
        return CanonicalApplyReceipt(
            resulting_object_id=str(row["resulting_object_id"]),
            resulting_version_id=str(row["resulting_version_id"]),
            manifest_revision=int(row["resulting_manifest_revision"]),
            manifest_version_id=str(row["resulting_manifest_version_id"]),
            purge_generation=int(row["purge_generation"]),
            publication_batch_id=str(row["publication_batch_id"]),
            profile_publication_sequence=int(row["profile_publication_sequence"]),
            receipt_id=str(row["receipt_id"]),
            dataset_id=str(row["dataset_id"]),
            device_id=str(row["device_id"]),
            client_envelope_id=str(row["client_envelope_id"]),
            canonical_payload_digest=str(row["canonical_payload_digest"]),
            wire_entity_version=stored_wire_version,
        )
