"""Encrypted, transaction-owned source publications for Personal Context."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

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
                    canonical_payload_digest, purge_generation,
                    resulting_object_id, resulting_version_id,
                    resulting_manifest_revision, resulting_manifest_version_id,
                    publication_batch_id, profile_publication_sequence, receipt_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ingress.dataset_id,
                    ingress.device_id,
                    ingress.client_envelope_id,
                    ingress.canonical_payload_digest,
                    ingress.purge_generation,
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
            sync_server_cursor=(
                None
                if row["sync_server_cursor"] is None
                else int(row["sync_server_cursor"])
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
                key_version = ?, algorithm = ?, row_state = ?
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

    @classmethod
    def read_ingress_receipt(
        cls,
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
        if (
            str(row["canonical_payload_digest"]) != identity.canonical_payload_digest
            or int(row["purge_generation"]) != identity.purge_generation
        ):
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
        )
