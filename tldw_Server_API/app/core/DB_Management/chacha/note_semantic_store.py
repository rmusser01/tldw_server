"""Owner-bound SQL seam for Notes semantic-index persistence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ...Notes_Graph.semantic_endpoint import canonical_semantic_endpoint_origin
from ..ChaChaNotes_DB import BackendConnectionWrapper, BackendType
from .note_semantic_models import (
    SemanticChunkRecord,
    SemanticDesiredState,
    SemanticDimensionState,
    SemanticGeneration,
    SemanticGenerationIntegrity,
    SemanticGenerationState,
    SemanticHealthSnapshot,
    SemanticIndexConfig,
    SemanticIndexingError,
    SemanticManifestPublication,
    SemanticNoteRecord,
    SemanticNoteState,
    SemanticObsoleteVectorClaim,
    SemanticOperationReceipt,
    SemanticProjectionChunk,
    SemanticSnapshotSeed,
    SemanticWorkClaimState,
    SemanticWorkItem,
    SemanticWorkKind,
    SemanticWorkReclaimResult,
)

if TYPE_CHECKING:
    from ..ChaChaNotes_DB import CharactersRAGDB


SemanticConnection = sqlite3.Connection | BackendConnectionWrapper


class _SemanticCASMiss(Exception):
    """Abort and roll back an atomic semantic compare-and-swap."""


class NoteSemanticStore:
    """Own semantic configuration, generation, manifest, and cleanup SQL."""

    _SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    _SAFE_MODEL = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._-]*"
        r"(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?"
        r"(?::[A-Za-z0-9][A-Za-z0-9._-]*)?$"
    )
    _ERROR_CODE = re.compile(r"^[a-z][a-z0-9_:-]{0,127}$")
    _DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
    _HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
    _MAX_WORK_ATTEMPTS = 5
    _MAX_OPERATION_RECEIPT_PRUNE = 16
    _MAX_PROJECTION_VECTOR_IDS = 1_600
    _PROJECTION_READ_BATCH_SIZE = 500
    _VECTOR_SIDE_EFFECT_IN_PROGRESS = "vector_side_effect_in_progress"

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    @property
    def owner_user_id(self) -> str:
        return str(self._db.client_id)

    @property
    def is_postgres(self) -> bool:
        return self._db.backend_type == BackendType.POSTGRESQL

    def _scope(self, dataset_id: str) -> str:
        value = str(dataset_id).strip()
        if not value or len(value.encode("utf-8")) > 256:
            raise ValueError("notes_semantic_dataset_scope_invalid")
        return value

    @staticmethod
    def _iso(now: datetime) -> str:
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("notes_semantic_timestamp_invalid")
        return now.astimezone(timezone.utc).isoformat()

    def _timestamp(self, now: datetime) -> datetime | str:
        return now.astimezone(timezone.utc) if self.is_postgres else self._iso(now)

    def _set_scope(self, conn: SemanticConnection, dataset_id: str) -> None:
        if self.is_postgres:
            conn.execute("SELECT set_config('app.current_dataset_id', ?, true)", (dataset_id,))

    def _serialize_dataset_mutation(
        self,
        conn: SemanticConnection,
        dataset_id: str,
    ) -> None:
        if not self.is_postgres:
            return
        identity = f"{self.owner_user_id}\0{dataset_id}".encode()
        digest = hashlib.sha256(
            b"notes-semantic-dataset-mutation-v1\0" + identity
        ).digest()
        lock_keys = (
            int.from_bytes(digest[:4], "big", signed=True),
            int.from_bytes(digest[4:8], "big", signed=True),
        )
        conn.execute("SELECT pg_advisory_xact_lock(?, ?)", lock_keys).fetchone()

    def _has_unexpired_cancellation_intent_locked(
        self,
        conn: SemanticConnection,
        *,
        dataset_id: str,
        root_job_id: str,
        now: datetime,
    ) -> bool:
        return conn.execute(
            "SELECT 1 FROM note_semantic_operation_receipts "
            "WHERE owner_user_id=? AND dataset_id=? AND action='cancel' "
            "AND run_id=? AND expires_at>? LIMIT 1",
            (
                self.owner_user_id,
                dataset_id,
                root_job_id,
                self._timestamp(now),
            ),
        ).fetchone() is not None

    @classmethod
    def _safe_token(cls, value: str, *, field: str) -> str:
        normalized = str(value).strip()
        if cls._SAFE_TOKEN.fullmatch(normalized) is None:
            raise ValueError(f"notes_semantic_{field}_invalid")
        return normalized

    @classmethod
    def _safe_model(cls, value: str) -> str:
        normalized = str(value).strip()
        if len(normalized) > 256 or cls._SAFE_MODEL.fullmatch(normalized) is None:
            raise ValueError("notes_semantic_model_invalid")
        return normalized

    @classmethod
    def _error_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        if cls._ERROR_CODE.fullmatch(normalized) is None:
            raise ValueError("notes_semantic_error_code_invalid")
        return normalized

    @classmethod
    def _digest(cls, value: str, *, field: str) -> str:
        if not isinstance(value, str) or cls._DIGEST.fullmatch(value) is None:
            raise ValueError(f"notes_semantic_{field}_invalid")
        return value

    @staticmethod
    def _endpoint_origin_display(value: str) -> str:
        origin = canonical_semantic_endpoint_origin(value)
        if origin is None or origin != value:
            raise ValueError("notes_semantic_endpoint_origin_display_invalid")
        return origin

    @staticmethod
    def _record(row: Any) -> dict[str, Any]:
        return dict(row)

    @staticmethod
    def _read_iso(value: object | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        return str(value)

    @staticmethod
    def _manifest_digest(value: object) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _state_value(value: SemanticNoteState | str) -> str:
        state = str(getattr(value, "value", value))
        if state not in {item.value for item in SemanticNoteState}:
            raise ValueError("notes_semantic_note_state_invalid")
        return state

    def _generation_manifest_locked(
        self,
        conn: SemanticConnection,
        *,
        dataset: str,
        generation_id: str,
    ) -> tuple[str, tuple[str, ...]]:
        note_rows = conn.execute(
            "SELECT note_id,content_version,content_fingerprint,dirty_generation,state,"
            "chunk_count,manifest_hash,error_code FROM note_semantic_note_state "
            "WHERE owner_user_id=? AND dataset_id=? AND generation_id=? ORDER BY note_id",
            (self.owner_user_id, dataset, generation_id),
        ).fetchall()
        chunk_rows = conn.execute(
            "SELECT c.chunk_id,c.note_id,c.content_version,c.ordinal,c.field,c.start_offset,"
            "c.end_offset,c.chunk_fingerprint,c.normalization_version,c.chunker_version "
            "FROM note_semantic_chunks c JOIN note_semantic_note_state n "
            "ON n.owner_user_id=c.owner_user_id AND n.dataset_id=c.dataset_id "
            "AND n.generation_id=c.generation_id AND n.note_id=c.note_id "
            "WHERE c.owner_user_id=? AND c.dataset_id=? AND c.generation_id=? "
            "AND n.state='indexed' AND n.content_version=c.content_version "
            "ORDER BY c.note_id,c.ordinal,c.chunk_id",
            (self.owner_user_id, dataset, generation_id),
        ).fetchall()
        notes = [
            {
                "note_id": str(row["note_id"]),
                "content_version": int(row["content_version"]),
                "content_fingerprint": str(row["content_fingerprint"]),
                "dirty_generation": int(row["dirty_generation"]),
                "state": str(row["state"]),
                "chunk_count": int(row["chunk_count"]),
                "manifest_hash": row["manifest_hash"],
                "error_code": row["error_code"],
            }
            for row in note_rows
        ]
        chunks = [
            {
                "chunk_id": str(row["chunk_id"]),
                "note_id": str(row["note_id"]),
                "content_version": int(row["content_version"]),
                "ordinal": int(row["ordinal"]),
                "field": str(row["field"]),
                "start_offset": int(row["start_offset"]),
                "end_offset": int(row["end_offset"]),
                "chunk_fingerprint": str(row["chunk_fingerprint"]),
                "normalization_version": str(row["normalization_version"]),
                "chunker_version": str(row["chunker_version"]),
            }
            for row in chunk_rows
        ]
        vector_ids = tuple(chunk["chunk_id"] for chunk in chunks)
        return self._manifest_digest({"notes": notes, "chunks": chunks}), vector_ids

    def _refresh_generation_counts_locked(
        self,
        conn: SemanticConnection,
        *,
        dataset: str,
        generation_id: str,
    ) -> None:
        counts = conn.execute(
            "SELECT COUNT(*) AS note_count,"
            "SUM(CASE WHEN state IN ('indexed','excluded','failed','tombstoned') THEN 1 ELSE 0 END) AS terminal_count,"
            "SUM(CASE WHEN state='indexed' THEN chunk_count ELSE 0 END) AS indexed_chunks "
            "FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? AND generation_id=?",
            (self.owner_user_id, dataset, generation_id),
        ).fetchone()
        published_chunks = conn.execute(
            "SELECT COUNT(*) AS chunk_count FROM note_semantic_chunks c "
            "JOIN note_semantic_note_state n ON n.owner_user_id=c.owner_user_id "
            "AND n.dataset_id=c.dataset_id AND n.generation_id=c.generation_id "
            "AND n.note_id=c.note_id WHERE c.owner_user_id=? AND c.dataset_id=? "
            "AND c.generation_id=? AND n.state='indexed' AND n.content_version=c.content_version",
            (self.owner_user_id, dataset, generation_id),
        ).fetchone()
        conn.execute(
            "UPDATE note_semantic_generations SET published_note_count=?,published_chunk_count=? "
            "WHERE owner_user_id=? AND dataset_id=? AND id=?",
            (
                int(counts["terminal_count"] or 0),
                int(published_chunks["chunk_count"] or 0),
                self.owner_user_id,
                dataset,
                generation_id,
            ),
        )

    def _increment_generation_counts_locked(
        self,
        conn: SemanticConnection,
        *,
        dataset: str,
        generation_id: str,
        note_delta: int,
        chunk_delta: int,
    ) -> None:
        cursor = conn.execute(
            "UPDATE note_semantic_generations SET published_note_count="
            "published_note_count+?,published_chunk_count=published_chunk_count+? "
            "WHERE owner_user_id=? AND dataset_id=? AND id=? AND "
            "published_note_count+?>=0 AND published_chunk_count+?>=0",
            (
                note_delta,
                chunk_delta,
                self.owner_user_id,
                dataset,
                generation_id,
                note_delta,
                chunk_delta,
            ),
        )
        if cursor.rowcount != 1:
            raise _SemanticCASMiss

    def _note_publication_contribution_locked(
        self,
        conn: SemanticConnection,
        *,
        dataset: str,
        generation_id: str,
        note_id: str,
        lock: bool = False,
    ) -> tuple[int, int]:
        """Return this Note's terminal and visible-chunk contribution."""

        query = (
            "SELECT state,content_version FROM note_semantic_note_state WHERE "
            "owner_user_id=? AND dataset_id=? AND generation_id=? AND note_id=?"
        )
        if lock and self.is_postgres:
            query += " FOR UPDATE"
        row = conn.execute(
            query,
            (self.owner_user_id, dataset, generation_id, note_id),
        ).fetchone()
        if row is None:
            return 0, 0
        state = str(row["state"])
        terminal = int(
            state
            in {
                SemanticNoteState.INDEXED.value,
                SemanticNoteState.EXCLUDED.value,
                SemanticNoteState.FAILED.value,
                SemanticNoteState.TOMBSTONED.value,
            }
        )
        if state != SemanticNoteState.INDEXED.value:
            return terminal, 0
        chunks = conn.execute(
            "SELECT COUNT(*) AS chunk_count FROM note_semantic_chunks WHERE "
            "owner_user_id=? AND dataset_id=? AND generation_id=? AND note_id=? "
            "AND content_version=?",
            (
                self.owner_user_id,
                dataset,
                generation_id,
                note_id,
                int(row["content_version"]),
            ),
        ).fetchone()
        return terminal, int(chunks["chunk_count"] or 0)

    def _apply_note_contribution_delta_locked(
        self,
        conn: SemanticConnection,
        *,
        dataset: str,
        generation_id: str,
        note_id: str,
        before: tuple[int, int],
    ) -> None:
        after = self._note_publication_contribution_locked(
            conn,
            dataset=dataset,
            generation_id=generation_id,
            note_id=note_id,
        )
        note_delta = after[0] - before[0]
        chunk_delta = after[1] - before[1]
        if note_delta or chunk_delta:
            self._increment_generation_counts_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
                note_delta=note_delta,
                chunk_delta=chunk_delta,
            )

    def _config_from_row(self, row: Any) -> SemanticIndexConfig:
        value = self._record(row)
        return SemanticIndexConfig(
            owner_user_id=str(value["owner_user_id"]),
            dataset_id=str(value["dataset_id"]),
            desired_state=SemanticDesiredState(str(value["desired_state"])),
            configuration_revision=int(value["configuration_revision"]),
            semantic_index_revision=int(value["semantic_index_revision"]),
            capability_revision=value["capability_revision"],
            disclosure_hash=value["disclosure_hash"],
            compatibility_hash=value["compatibility_hash"],
            provider=value["provider"],
            model=value["model"],
            model_revision=value.get("model_revision"),
            endpoint_origin_revision=value["endpoint_origin_revision"],
            endpoint_origin_display=value["endpoint_origin_display"],
            data_boundary=value["data_boundary"],
            vector_backend=value["vector_backend"],
            storage_boundary=value["storage_boundary"],
            storage_label=value["storage_label"],
            metric=str(value["metric"]),
            dimension_state=SemanticDimensionState(str(value["dimension_state"])),
            dimensions=value["dimensions"],
            normalization_version=str(value["normalization_version"]),
            chunker_version=str(value["chunker_version"]),
            active_generation_id=value["active_generation_id"],
            enabled_at=self._read_iso(value["enabled_at"]),
            disabled_at=self._read_iso(value["disabled_at"]),
            consented_at=self._read_iso(value["consented_at"]),
            updated_at=self._read_iso(value["updated_at"]) or "",
        )

    def _generation_from_row(self, row: Any) -> SemanticGeneration:
        value = self._record(row)
        return SemanticGeneration(
            id=str(value["id"]),
            owner_user_id=str(value["owner_user_id"]),
            dataset_id=str(value["dataset_id"]),
            configuration_revision=int(value["configuration_revision"]),
            state=SemanticGenerationState(str(value["state"])),
            compatibility_hash=value["compatibility_hash"],
            model_revision=value.get("model_revision"),
            dimension_state=SemanticDimensionState(str(value["dimension_state"])),
            dimensions=value["dimensions"],
            root_job_id=value["root_job_id"],
            expected_note_count=int(value["expected_note_count"]),
            expected_chunk_count=int(value["expected_chunk_count"]),
            published_note_count=int(value["published_note_count"]),
            published_chunk_count=int(value["published_chunk_count"]),
            manifest_hash=value["manifest_hash"],
            publication_receipt=value["publication_receipt"],
            terminal_error_code=value["terminal_error_code"],
            created_at=self._read_iso(value["created_at"]) or "",
            published_at=self._read_iso(value["published_at"]),
            retired_at=self._read_iso(value["retired_at"]),
            deleted_at=self._read_iso(value["deleted_at"]),
        )

    def _note_from_row(self, row: Any) -> SemanticNoteRecord:
        value = self._record(row)
        return SemanticNoteRecord(
            owner_user_id=str(value["owner_user_id"]), dataset_id=str(value["dataset_id"]),
            generation_id=str(value["generation_id"]), note_id=str(value["note_id"]),
            content_version=int(value["content_version"]), content_fingerprint=str(value["content_fingerprint"]),
            dirty_generation=int(value["dirty_generation"]), state=SemanticNoteState(str(value["state"])),
            chunk_count=int(value["chunk_count"]), manifest_hash=value["manifest_hash"],
            error_code=value["error_code"], published_at=self._read_iso(value["published_at"]),
        )

    def _work_from_row(self, row: Any) -> SemanticWorkItem:
        value = self._record(row)
        return SemanticWorkItem(
            id=str(value["id"]), owner_user_id=str(value["owner_user_id"]),
            dataset_id=str(value["dataset_id"]), kind=SemanticWorkKind(str(value["kind"])),
            note_id=value["note_id"], generation_id=value["generation_id"],
            dirty_generation=value["dirty_generation"], fencing_token=str(value["fencing_token"]),
            claim_state=SemanticWorkClaimState(str(value["claim_state"])),
            attempt_count=int(value["attempt_count"]),
            next_eligible_at=self._read_iso(value["next_eligible_at"]) or "",
            claim_token=value["claim_token"], claimed_at=self._read_iso(value["claimed_at"]),
            error_code=value["error_code"], created_at=self._read_iso(value["created_at"]) or "",
            updated_at=self._read_iso(value["updated_at"]) or "",
        )

    def _operation_from_row(self, row: Any) -> SemanticOperationReceipt:
        value = self._record(row)
        return SemanticOperationReceipt(
            owner_user_id=str(value["owner_user_id"]),
            dataset_id=str(value["dataset_id"]),
            key_digest=str(value["key_digest"]),
            action=str(value["action"]),
            request_fingerprint=str(value["request_fingerprint"]),
            run_id=value["run_id"],
            expected_revision=int(value["expected_revision"]),
            state=str(value["state"]),
            response_json=value["response_json"],
            expires_at=self._read_iso(value["expires_at"]) or "",
        )

    @classmethod
    def _hex_digest(cls, value: str, *, field: str) -> str:
        if not isinstance(value, str) or cls._HEX_DIGEST.fullmatch(value) is None:
            raise ValueError(f"notes_semantic_{field}_invalid")
        return value

    def get_operation_receipt(
        self,
        *,
        dataset_id: str,
        key_digest: str,
        now: datetime,
    ) -> SemanticOperationReceipt | None:
        """Read one unexpired mutation receipt without changing durable state."""

        dataset = self._scope(dataset_id)
        key = self._hex_digest(key_digest, field="idempotency_digest")
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            row = conn.execute(
                "SELECT * FROM note_semantic_operation_receipts "
                "WHERE owner_user_id=? AND dataset_id=? AND key_digest=? "
                "AND expires_at>?",
                (self.owner_user_id, dataset, key, timestamp),
            ).fetchone()
        return None if row is None else self._operation_from_row(row)

    def begin_operation_receipt(
        self,
        *,
        dataset_id: str,
        key_digest: str,
        action: str,
        request_fingerprint: str,
        run_id: str | None,
        expected_revision: int,
        expires_at: datetime,
        now: datetime,
    ) -> tuple[SemanticOperationReceipt, bool]:
        """Create or replay one exact bounded Notes-side mutation receipt."""

        dataset = self._scope(dataset_id)
        key = self._hex_digest(key_digest, field="idempotency_digest")
        fingerprint = self._hex_digest(
            request_fingerprint,
            field="request_fingerprint",
        )
        if action not in {"enable", "cancel"}:
            raise ValueError("notes_semantic_operation_action_invalid")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("notes_semantic_configuration_revision_invalid")
        normalized_run = (
            None if run_id is None else self._safe_token(run_id, field="run_id")
        )
        timestamp = self._timestamp(now)
        expiry = self._timestamp(expires_at)
        if expires_at <= now:
            raise ValueError("notes_semantic_operation_expiry_invalid")
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            self._serialize_dataset_mutation(conn, dataset)
            conn.execute(
                "DELETE FROM note_semantic_operation_receipts WHERE owner_user_id=? "
                "AND dataset_id=? AND key_digest=? AND expires_at<=?",
                (self.owner_user_id, dataset, key, timestamp),
            )
            conn.execute(
                "DELETE FROM note_semantic_operation_receipts WHERE owner_user_id=? "
                "AND dataset_id=? AND key_digest IN ("
                "SELECT key_digest FROM note_semantic_operation_receipts WHERE "
                "owner_user_id=? AND dataset_id=? AND expires_at<=? "
                "ORDER BY expires_at,key_digest LIMIT ?)",
                (
                    self.owner_user_id,
                    dataset,
                    self.owner_user_id,
                    dataset,
                    timestamp,
                    self._MAX_OPERATION_RECEIPT_PRUNE,
                ),
            )
            predecessor = conn.execute(
                "SELECT key_digest FROM note_semantic_operation_receipts "
                "WHERE owner_user_id=? AND dataset_id=? AND action=? "
                "AND request_fingerprint=? AND expires_at>? "
                "ORDER BY created_at,key_digest LIMIT 1",
                (self.owner_user_id, dataset, action, fingerprint, timestamp),
            ).fetchone()
            if (
                predecessor is not None
                and str(self._record(predecessor)["key_digest"]) != key
            ):
                raise SemanticIndexingError("notes_semantic_idempotency_conflict")
            cursor = conn.execute(
                "INSERT INTO note_semantic_operation_receipts("
                "owner_user_id,dataset_id,key_digest,action,request_fingerprint,run_id,"
                "expected_revision,state,response_json,expires_at,created_at,updated_at"
                ") VALUES (?,?,?,?,?,?,?,'pending',NULL,?,?,?) "
                "ON CONFLICT(owner_user_id,key_digest) DO NOTHING",
                (
                    self.owner_user_id,
                    dataset,
                    key,
                    action,
                    fingerprint,
                    normalized_run,
                    expected_revision,
                    expiry,
                    timestamp,
                    timestamp,
                ),
            )
            row = conn.execute(
                "SELECT * FROM note_semantic_operation_receipts "
                "WHERE owner_user_id=? AND key_digest=?",
                (self.owner_user_id, key),
            ).fetchone()
        if row is None:
            raise SemanticIndexingError("notes_semantic_operation_receipt_unavailable")
        receipt = self._operation_from_row(row)
        if (
            receipt.dataset_id != dataset
            or receipt.action != action
            or receipt.request_fingerprint != fingerprint
            or (
                receipt.run_id != normalized_run
                and not (action == "enable" and normalized_run is None)
            )
            or receipt.expected_revision != expected_revision
        ):
            raise SemanticIndexingError("notes_semantic_idempotency_conflict")
        return receipt, cursor.rowcount == 0

    def complete_operation_receipt(
        self,
        *,
        dataset_id: str,
        key_digest: str,
        request_fingerprint: str,
        run_id: str | None,
        response: dict[str, Any],
        now: datetime,
    ) -> SemanticOperationReceipt:
        """Persist one bounded content-free mutation response behind its fingerprint."""

        dataset = self._scope(dataset_id)
        key = self._hex_digest(key_digest, field="idempotency_digest")
        fingerprint = self._hex_digest(
            request_fingerprint,
            field="request_fingerprint",
        )
        normalized_run = (
            None if run_id is None else self._safe_token(run_id, field="run_id")
        )
        payload = json.dumps(
            response,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(payload.encode("utf-8")) > 8192:
            raise ValueError("notes_semantic_operation_response_too_large")
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            conn.execute(
                "DELETE FROM note_semantic_operation_receipts WHERE owner_user_id=? "
                "AND dataset_id=? AND key_digest=? AND expires_at<=?",
                (self.owner_user_id, dataset, key, timestamp),
            )
            conn.execute(
                "UPDATE note_semantic_operation_receipts SET state='completed',run_id=?,"
                "response_json=?,updated_at=? WHERE owner_user_id=? AND dataset_id=? "
                "AND key_digest=? AND request_fingerprint=? AND state='pending' "
                "AND expires_at>? AND (run_id IS NULL OR run_id=?)",
                (
                    normalized_run,
                    payload,
                    timestamp,
                    self.owner_user_id,
                    dataset,
                    key,
                    fingerprint,
                    timestamp,
                    normalized_run,
                ),
            )
            row = conn.execute(
                "SELECT * FROM note_semantic_operation_receipts WHERE owner_user_id=? "
                "AND dataset_id=? AND key_digest=? AND request_fingerprint=?",
                (self.owner_user_id, dataset, key, fingerprint),
            ).fetchone()
        if row is None:
            raise SemanticIndexingError("notes_semantic_operation_receipt_conflict")
        receipt = self._operation_from_row(row)
        if (
            receipt.state != "completed"
            or receipt.run_id != normalized_run
            or receipt.response_json != payload
        ):
            raise SemanticIndexingError("notes_semantic_operation_receipt_conflict")
        return receipt

    def resolve_enabled_dataset_for_local_mutation(
        self,
        *,
        tx: SemanticConnection | None = None,
    ) -> str | None:
        """Return the sole canonical enabled semantic dataset for local Note writes."""

        def resolve(conn: SemanticConnection) -> str | None:
            authority = conn.execute(
                "SELECT dataset_id FROM note_task_scope_authority WHERE owner_user_id=?",
                (self.owner_user_id,),
            ).fetchone()
            if authority is None:
                return None
            dataset = str(self._record(authority)["dataset_id"])
            self._set_scope(conn, dataset)
            enabled = conn.execute(
                "SELECT 1 FROM note_semantic_index_configs WHERE owner_user_id=? "
                "AND dataset_id=? AND desired_state='enabled' LIMIT 1",
                (self.owner_user_id, dataset),
            ).fetchone()
            return dataset if enabled is not None else None

        if tx is not None:
            return resolve(tx)
        with self._db.transaction() as conn:
            return resolve(conn)

    def _stage_obsolete_vectors_locked(
        self,
        conn: SemanticConnection,
        *,
        dataset: str,
        generation_id: str,
        vector_ids: Sequence[str],
        source_kind: str,
        note_id: str | None,
        dirty_generation: int | None,
        now: datetime,
    ) -> int:
        if source_kind not in {
            "unpublished",
            "manifest_replace",
            "tombstone",
            "hard_delete",
            "note_failure",
        }:
            raise ValueError("notes_semantic_cleanup_source_invalid")
        ids = tuple(
            self._safe_token(vector_id, field="vector_id") for vector_id in vector_ids
        )
        if len(ids) != len(set(ids)):
            raise ValueError("notes_semantic_cleanup_vector_ids_duplicate")
        if dirty_generation is not None and (
            type(dirty_generation) is not int or dirty_generation < 1
        ):
            raise ValueError("notes_semantic_dirty_generation_invalid")
        timestamp = self._timestamp(now)
        staged = 0
        for vector_id in ids:
            cursor = conn.execute(
                """
                INSERT INTO note_semantic_obsolete_vectors(
                  id,owner_user_id,dataset_id,generation_id,vector_id,note_id,source_kind,
                  dirty_generation,claim_state,attempt_count,next_eligible_at,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,?,?,'pending',0,?,?,?)
                ON CONFLICT(owner_user_id,dataset_id,generation_id,vector_id) DO UPDATE SET
                  source_kind=CASE
                    WHEN CASE note_semantic_obsolete_vectors.source_kind
                      WHEN 'hard_delete' THEN 4 WHEN 'tombstone' THEN 3
                      WHEN 'manifest_replace' THEN 2 WHEN 'note_failure' THEN 2 ELSE 1 END
                    > CASE excluded.source_kind
                      WHEN 'hard_delete' THEN 4 WHEN 'tombstone' THEN 3
                      WHEN 'manifest_replace' THEN 2 WHEN 'note_failure' THEN 2 ELSE 1 END
                    THEN note_semantic_obsolete_vectors.source_kind ELSE excluded.source_kind END,
                  note_id=COALESCE(excluded.note_id,note_semantic_obsolete_vectors.note_id),
                  dirty_generation=CASE
                    WHEN excluded.dirty_generation IS NULL
                      THEN note_semantic_obsolete_vectors.dirty_generation
                    WHEN note_semantic_obsolete_vectors.dirty_generation IS NULL
                      OR excluded.dirty_generation>note_semantic_obsolete_vectors.dirty_generation
                      THEN excluded.dirty_generation
                    ELSE note_semantic_obsolete_vectors.dirty_generation END,
                  claim_state='pending',attempt_count=0,
                  next_eligible_at=excluded.next_eligible_at,claim_token=NULL,claimed_at=NULL,
                  error_code=NULL,updated_at=excluded.updated_at
                WHERE note_semantic_obsolete_vectors.claim_state<>'claimed'
                """,
                (
                    str(uuid.uuid4()),
                    self.owner_user_id,
                    dataset,
                    generation_id,
                    vector_id,
                    note_id,
                    source_kind,
                    dirty_generation,
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
            staged += cursor.rowcount
        return staged

    def _obsolete_cleanup_blocked_by_work_locked(
        self,
        conn: SemanticConnection,
        *,
        dataset: str,
        generation_id: str,
        note_id: str | None,
        dirty_generation: int | None,
    ) -> bool:
        if note_id is None or dirty_generation is None:
            return False
        return conn.execute(
            "SELECT 1 FROM note_semantic_work w JOIN note_semantic_generations g ON "
            "g.owner_user_id=w.owner_user_id AND g.dataset_id=w.dataset_id AND "
            "g.id=w.generation_id JOIN note_semantic_index_configs c ON "
            "c.owner_user_id=g.owner_user_id AND c.dataset_id=g.dataset_id WHERE "
            "w.owner_user_id=? AND w.dataset_id=? AND w.generation_id=? AND "
            "w.note_id=? AND w.dirty_generation=? AND w.kind='index_note' AND "
            "g.state IN ('staging','active') AND c.desired_state='enabled' AND "
            "c.configuration_revision=g.configuration_revision AND "
            "(w.claim_state IN ('pending','claimed') OR "
            "(w.claim_state='failed' AND w.attempt_count<?)) LIMIT 1",
            (
                self.owner_user_id,
                dataset,
                generation_id,
                note_id,
                dirty_generation,
                self._MAX_WORK_ATTEMPTS,
            ),
        ).fetchone() is not None

    def stage_obsolete_vector_cleanup(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        vector_ids: Sequence[str],
        source_kind: str,
        note_id: str | None,
        dirty_generation: int | None,
        now: datetime,
    ) -> int:
        """Durably record opaque IDs before vector or manifest authority can vanish."""

        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            return self._stage_obsolete_vectors_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
                vector_ids=vector_ids,
                source_kind=source_kind,
                note_id=note_id,
                dirty_generation=dirty_generation,
                now=now,
            )

    def list_obsolete_vector_ids(
        self,
        dataset_id: str,
        generation_id: str,
        *,
        limit: int,
    ) -> tuple[str, ...]:
        dataset = self._scope(dataset_id)
        if type(limit) is not int or not 1 <= limit <= 100_000:
            raise ValueError("notes_semantic_cleanup_limit_invalid")
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            rows = conn.execute(
                "SELECT vector_id FROM note_semantic_obsolete_vectors WHERE "
                "owner_user_id=? AND dataset_id=? AND generation_id=? "
                "ORDER BY created_at,vector_id LIMIT ?",
                (self.owner_user_id, dataset, generation_id, limit),
            ).fetchall()
        return tuple(str(row["vector_id"]) for row in rows)

    def claim_obsolete_vector_cleanup_batch(
        self,
        *,
        dataset_id: str,
        limit: int,
        now: datetime,
        generation_id: str | None = None,
    ) -> SemanticObsoleteVectorClaim | None:
        """Claim one bounded generation-homogeneous page that is not visible."""

        dataset = self._scope(dataset_id)
        if type(limit) is not int or not 1 <= limit <= 100_000:
            raise ValueError("notes_semantic_cleanup_limit_invalid")
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            first_query = (
                "SELECT o.generation_id FROM note_semantic_obsolete_vectors o WHERE "
                "o.owner_user_id=? AND o.dataset_id=? AND o.claim_state IN ('pending','failed') "
                "AND o.attempt_count<? AND o.next_eligible_at<=? AND NOT EXISTS ("
                "SELECT 1 FROM note_semantic_chunks c JOIN note_semantic_note_state n ON "
                "n.owner_user_id=c.owner_user_id AND n.dataset_id=c.dataset_id AND "
                "n.generation_id=c.generation_id AND n.note_id=c.note_id WHERE "
                "c.owner_user_id=o.owner_user_id AND c.dataset_id=o.dataset_id AND "
                "c.generation_id=o.generation_id AND c.chunk_id=o.vector_id AND "
                "n.state='indexed' AND n.content_version=c.content_version) AND NOT ("
                "o.note_id IS NOT NULL AND o.dirty_generation IS NOT NULL AND "
                "EXISTS (SELECT 1 FROM "
                "note_semantic_work w JOIN note_semantic_generations g ON "
                "g.owner_user_id=w.owner_user_id AND g.dataset_id=w.dataset_id AND "
                "g.id=w.generation_id JOIN note_semantic_index_configs cfg ON "
                "cfg.owner_user_id=g.owner_user_id AND cfg.dataset_id=g.dataset_id WHERE "
                "w.owner_user_id=o.owner_user_id AND w.dataset_id=o.dataset_id AND "
                "w.generation_id=o.generation_id AND w.note_id=o.note_id AND "
                "w.dirty_generation=o.dirty_generation AND w.kind='index_note' AND "
                "g.state IN ('staging','active') AND cfg.desired_state='enabled' AND "
                "cfg.configuration_revision=g.configuration_revision AND "
                "(w.claim_state IN ('pending','claimed') OR "
                "(w.claim_state='failed' AND w.attempt_count<?)))) "
                "ORDER BY o.next_eligible_at,o.id LIMIT 1"
            )
            first_params: tuple[Any, ...] = (
                self.owner_user_id,
                dataset,
                self._MAX_WORK_ATTEMPTS,
                timestamp,
                self._MAX_WORK_ATTEMPTS,
            )
            if generation_id is not None:
                first_query = first_query.replace(
                    "AND o.claim_state",
                    "AND o.generation_id=? AND o.claim_state",
                    1,
                )
                first_params = (
                    self.owner_user_id,
                    dataset,
                    generation_id,
                    self._MAX_WORK_ATTEMPTS,
                    timestamp,
                    self._MAX_WORK_ATTEMPTS,
                )
            if self.is_postgres:
                first_query += " FOR UPDATE SKIP LOCKED"
            first = conn.execute(
                first_query,
                first_params,
            ).fetchone()
            if first is None:
                return None
            generation_id = str(first["generation_id"])
            page_query = (
                "SELECT o.id,o.vector_id,o.attempt_count FROM note_semantic_obsolete_vectors o "
                "WHERE o.owner_user_id=? AND o.dataset_id=? AND o.generation_id=? "
                "AND o.claim_state IN ('pending','failed') AND o.attempt_count<? "
                "AND o.next_eligible_at<=? AND NOT EXISTS (SELECT 1 FROM "
                "note_semantic_chunks c JOIN note_semantic_note_state n ON "
                "n.owner_user_id=c.owner_user_id AND n.dataset_id=c.dataset_id AND "
                "n.generation_id=c.generation_id AND n.note_id=c.note_id WHERE "
                "c.owner_user_id=o.owner_user_id AND c.dataset_id=o.dataset_id AND "
                "c.generation_id=o.generation_id AND c.chunk_id=o.vector_id AND "
                "n.state='indexed' AND n.content_version=c.content_version) AND NOT ("
                "o.note_id IS NOT NULL AND o.dirty_generation IS NOT NULL AND "
                "EXISTS (SELECT 1 FROM "
                "note_semantic_work w JOIN note_semantic_generations g ON "
                "g.owner_user_id=w.owner_user_id AND g.dataset_id=w.dataset_id AND "
                "g.id=w.generation_id JOIN note_semantic_index_configs cfg ON "
                "cfg.owner_user_id=g.owner_user_id AND cfg.dataset_id=g.dataset_id WHERE "
                "w.owner_user_id=o.owner_user_id AND w.dataset_id=o.dataset_id AND "
                "w.generation_id=o.generation_id AND w.note_id=o.note_id AND "
                "w.dirty_generation=o.dirty_generation AND w.kind='index_note' AND "
                "g.state IN ('staging','active') AND cfg.desired_state='enabled' AND "
                "cfg.configuration_revision=g.configuration_revision AND "
                "(w.claim_state IN ('pending','claimed') OR "
                "(w.claim_state='failed' AND w.attempt_count<?)))) "
                "ORDER BY o.next_eligible_at,o.vector_id,o.id LIMIT ?"
            )
            if self.is_postgres:
                page_query += " FOR UPDATE SKIP LOCKED"
            rows = conn.execute(
                page_query,
                (
                    self.owner_user_id,
                    dataset,
                    generation_id,
                    self._MAX_WORK_ATTEMPTS,
                    timestamp,
                    self._MAX_WORK_ATTEMPTS,
                    limit,
                ),
            ).fetchall()
            if not rows:
                return None
            claim_token = str(uuid.uuid4())
            ledger_ids: list[str] = []
            vector_ids: list[str] = []
            attempts = 0
            for row in rows:
                ledger_id = str(row["id"])
                updated = conn.execute(
                    "UPDATE note_semantic_obsolete_vectors SET claim_state='claimed',"
                    "claim_token=?,claimed_at=?,updated_at=? WHERE owner_user_id=? AND "
                    "dataset_id=? AND id=? AND claim_state IN ('pending','failed')",
                    (
                        claim_token,
                        timestamp,
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        ledger_id,
                    ),
                )
                if updated.rowcount == 1:
                    ledger_ids.append(ledger_id)
                    vector_ids.append(str(row["vector_id"]))
                    attempts = max(attempts, int(row["attempt_count"]))
        if not ledger_ids:
            return None
        return SemanticObsoleteVectorClaim(
            owner_user_id=self.owner_user_id,
            dataset_id=dataset,
            generation_id=generation_id,
            ledger_ids=tuple(ledger_ids),
            vector_ids=tuple(vector_ids),
            claim_token=claim_token,
            attempt_count=attempts,
        )

    def retry_obsolete_vector_cleanup(
        self,
        *,
        dataset_id: str,
        ledger_ids: Sequence[str],
        claim_token: str,
        error_code: str,
        retry_at: datetime,
        now: datetime,
    ) -> int:
        dataset = self._scope(dataset_id)
        ids = tuple(ledger_ids)
        if not ids or len(ids) != len(set(ids)) or not claim_token:
            return 0
        retry_timestamp = self._timestamp(retry_at)
        now_timestamp = self._timestamp(now)
        if retry_at <= now:
            raise ValueError("notes_semantic_retry_timestamp_invalid")
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            updated = 0
            for ledger_id in ids:
                cursor = conn.execute(
                    "UPDATE note_semantic_obsolete_vectors SET claim_state='failed',"
                    "attempt_count=attempt_count+1,next_eligible_at=?,claim_token=NULL,"
                    "claimed_at=NULL,error_code=?,updated_at=? WHERE owner_user_id=? AND "
                    "dataset_id=? AND id=? AND claim_state='claimed' AND claim_token=? "
                    "AND attempt_count<?",
                    (
                        retry_timestamp,
                        self._error_code(error_code),
                        now_timestamp,
                        self.owner_user_id,
                        dataset,
                        ledger_id,
                        claim_token,
                        self._MAX_WORK_ATTEMPTS,
                    ),
                )
                updated += cursor.rowcount
        return updated

    def release_obsolete_vector_claim(
        self,
        *,
        dataset_id: str,
        ledger_ids: Sequence[str],
        claim_token: str,
        now: datetime,
    ) -> bool:
        """Release one exact cleanup claim without consuming a backend attempt."""

        dataset = self._scope(dataset_id)
        ids = tuple(ledger_ids)
        if not ids or len(ids) != len(set(ids)) or not claim_token:
            return False
        timestamp = self._timestamp(now)
        try:
            with self._db.transaction() as conn:
                self._set_scope(conn, dataset)
                for ledger_id in ids:
                    cursor = conn.execute(
                        "UPDATE note_semantic_obsolete_vectors SET claim_state='pending',"
                        "next_eligible_at=?,claim_token=NULL,claimed_at=NULL,error_code=NULL,"
                        "updated_at=? WHERE owner_user_id=? AND dataset_id=? AND id=? AND "
                        "claim_state='claimed' AND claim_token=?",
                        (
                            timestamp,
                            timestamp,
                            self.owner_user_id,
                            dataset,
                            ledger_id,
                            claim_token,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise _SemanticCASMiss
        except _SemanticCASMiss:
            return False
        return True

    def reclaim_expired_obsolete_vector_claims(
        self,
        *,
        dataset_id: str,
        expired_before: datetime,
        limit: int,
        now: datetime,
    ) -> int:
        """Boundedly recover cleanup rows retained across worker crashes."""

        dataset = self._scope(dataset_id)
        if type(limit) is not int or not 1 <= limit <= 100_000:
            raise ValueError("notes_semantic_cleanup_limit_invalid")
        expired = self._timestamp(expired_before)
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            query = (
                "SELECT id,claim_token FROM note_semantic_obsolete_vectors WHERE "
                "owner_user_id=? AND dataset_id=? AND claim_state='claimed' AND "
                "claimed_at<=? ORDER BY claimed_at,id LIMIT ?"
            )
            if self.is_postgres:
                query += " FOR UPDATE SKIP LOCKED"
            rows = conn.execute(
                query,
                (self.owner_user_id, dataset, expired, limit),
            ).fetchall()
            reclaimed = 0
            for row in rows:
                cursor = conn.execute(
                    "UPDATE note_semantic_obsolete_vectors SET claim_state='failed',"
                    "attempt_count=attempt_count+1,next_eligible_at=?,claim_token=NULL,"
                    "claimed_at=NULL,error_code='claim_lease_expired',updated_at=? WHERE "
                    "owner_user_id=? AND dataset_id=? AND id=? AND claim_state='claimed' "
                    "AND claim_token=? AND attempt_count<?",
                    (
                        timestamp,
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        str(row["id"]),
                        str(row["claim_token"]),
                        self._MAX_WORK_ATTEMPTS,
                    ),
                )
                reclaimed += cursor.rowcount
        return reclaimed

    def rearm_exhausted_obsolete_vector_cleanup(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        limit: int,
        now: datetime,
    ) -> int:
        """Grant one bounded retry to retained, eligible exhausted ledger rows."""

        dataset = self._scope(dataset_id)
        if type(limit) is not int or not 1 <= limit <= 100_000:
            raise ValueError("notes_semantic_cleanup_limit_invalid")
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            query = (
                "SELECT o.id FROM note_semantic_obsolete_vectors o WHERE "
                "o.owner_user_id=? AND o.dataset_id=? AND o.generation_id=? AND "
                "o.claim_state='failed' AND o.attempt_count>=? AND "
                "o.next_eligible_at<=? AND NOT EXISTS (SELECT 1 FROM "
                "note_semantic_chunks c JOIN note_semantic_note_state n ON "
                "n.owner_user_id=c.owner_user_id AND n.dataset_id=c.dataset_id AND "
                "n.generation_id=c.generation_id AND n.note_id=c.note_id WHERE "
                "c.owner_user_id=o.owner_user_id AND c.dataset_id=o.dataset_id AND "
                "c.generation_id=o.generation_id AND c.chunk_id=o.vector_id AND "
                "n.state='indexed' AND n.content_version=c.content_version) AND NOT ("
                "o.note_id IS NOT NULL AND o.dirty_generation IS NOT NULL AND "
                "EXISTS (SELECT 1 FROM note_semantic_work w JOIN "
                "note_semantic_generations g ON g.owner_user_id=w.owner_user_id AND "
                "g.dataset_id=w.dataset_id AND g.id=w.generation_id JOIN "
                "note_semantic_index_configs cfg ON cfg.owner_user_id=g.owner_user_id "
                "AND cfg.dataset_id=g.dataset_id WHERE w.owner_user_id=o.owner_user_id "
                "AND w.dataset_id=o.dataset_id AND w.generation_id=o.generation_id AND "
                "w.note_id=o.note_id AND w.dirty_generation=o.dirty_generation AND "
                "w.kind='index_note' AND g.state IN ('staging','active') AND "
                "cfg.desired_state='enabled' AND "
                "cfg.configuration_revision=g.configuration_revision AND "
                "(w.claim_state IN ('pending','claimed') OR (w.claim_state='failed' "
                "AND w.attempt_count<?)))) ORDER BY o.updated_at,o.id LIMIT ?"
            )
            if self.is_postgres:
                query += " FOR UPDATE SKIP LOCKED"
            rows = conn.execute(
                query,
                (
                    self.owner_user_id,
                    dataset,
                    generation_id,
                    self._MAX_WORK_ATTEMPTS,
                    timestamp,
                    self._MAX_WORK_ATTEMPTS,
                    limit,
                ),
            ).fetchall()
            rearmed = 0
            for row in rows:
                cursor = conn.execute(
                    "UPDATE note_semantic_obsolete_vectors AS o SET attempt_count=?,"
                    "next_eligible_at=?,claim_token=NULL,claimed_at=NULL,"
                    "error_code='cleanup_rearmed',updated_at=? WHERE o.owner_user_id=? "
                    "AND o.dataset_id=? AND o.generation_id=? AND o.id=? AND "
                    "o.claim_state='failed' AND o.attempt_count>=? AND "
                    "o.next_eligible_at<=? AND NOT EXISTS (SELECT 1 FROM "
                    "note_semantic_chunks c JOIN note_semantic_note_state n ON "
                    "n.owner_user_id=c.owner_user_id AND n.dataset_id=c.dataset_id AND "
                    "n.generation_id=c.generation_id AND n.note_id=c.note_id WHERE "
                    "c.owner_user_id=o.owner_user_id AND c.dataset_id=o.dataset_id AND "
                    "c.generation_id=o.generation_id AND c.chunk_id=o.vector_id AND "
                    "n.state='indexed' AND n.content_version=c.content_version) AND NOT ("
                    "o.note_id IS NOT NULL AND o.dirty_generation IS NOT NULL AND "
                    "EXISTS (SELECT 1 FROM note_semantic_work w JOIN "
                    "note_semantic_generations g ON g.owner_user_id=w.owner_user_id AND "
                    "g.dataset_id=w.dataset_id AND g.id=w.generation_id JOIN "
                    "note_semantic_index_configs cfg ON cfg.owner_user_id=g.owner_user_id "
                    "AND cfg.dataset_id=g.dataset_id WHERE w.owner_user_id=o.owner_user_id "
                    "AND w.dataset_id=o.dataset_id AND w.generation_id=o.generation_id AND "
                    "w.note_id=o.note_id AND w.dirty_generation=o.dirty_generation AND "
                    "w.kind='index_note' AND g.state IN ('staging','active') AND "
                    "cfg.desired_state='enabled' AND "
                    "cfg.configuration_revision=g.configuration_revision AND "
                    "(w.claim_state IN ('pending','claimed') OR (w.claim_state='failed' "
                    "AND w.attempt_count<?))))",
                    (
                        self._MAX_WORK_ATTEMPTS - 1,
                        timestamp,
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        generation_id,
                        str(row["id"]),
                        self._MAX_WORK_ATTEMPTS,
                        timestamp,
                        self._MAX_WORK_ATTEMPTS,
                    ),
                )
                rearmed += cursor.rowcount
        return rearmed

    def authorize_obsolete_vector_claim(
        self,
        *,
        dataset_id: str,
        ledger_ids: Sequence[str],
        claim_token: str,
    ) -> bool:
        dataset = self._scope(dataset_id)
        ids = tuple(ledger_ids)
        if not ids or len(ids) != len(set(ids)):
            return False
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            for ledger_id in ids:
                row = conn.execute(
                    "SELECT o.vector_id,o.generation_id,o.source_kind,o.note_id,"
                    "o.dirty_generation FROM note_semantic_obsolete_vectors o "
                    "WHERE o.owner_user_id=? AND o.dataset_id=? AND o.id=? AND "
                    "o.claim_state='claimed' AND o.claim_token=?",
                    (self.owner_user_id, dataset, ledger_id, claim_token),
                ).fetchone()
                if row is None:
                    return False
                if self._obsolete_cleanup_blocked_by_work_locked(
                    conn,
                    dataset=dataset,
                    generation_id=str(row["generation_id"]),
                    note_id=row["note_id"],
                    dirty_generation=row["dirty_generation"],
                ):
                    return False
                visible = conn.execute(
                    "SELECT 1 FROM note_semantic_chunks c JOIN note_semantic_note_state n ON "
                    "n.owner_user_id=c.owner_user_id AND n.dataset_id=c.dataset_id AND "
                    "n.generation_id=c.generation_id AND n.note_id=c.note_id WHERE "
                    "c.owner_user_id=? AND c.dataset_id=? AND c.generation_id=? AND "
                    "c.chunk_id=? AND n.state='indexed' AND n.content_version=c.content_version "
                    "LIMIT 1",
                    (
                        self.owner_user_id,
                        dataset,
                        str(row["generation_id"]),
                        str(row["vector_id"]),
                    ),
                ).fetchone()
                if visible is not None:
                    return False
        return True

    def complete_obsolete_vector_claim(
        self,
        *,
        dataset_id: str,
        ledger_ids: Sequence[str],
        claim_token: str,
    ) -> bool:
        dataset = self._scope(dataset_id)
        ids = tuple(ledger_ids)
        if not ids or len(ids) != len(set(ids)) or not claim_token:
            return False
        try:
            with self._db.transaction() as conn:
                self._set_scope(conn, dataset)
                for ledger_id in ids:
                    query = (
                        "SELECT vector_id,generation_id,note_id,dirty_generation FROM "
                        "note_semantic_obsolete_vectors WHERE owner_user_id=? AND "
                        "dataset_id=? AND id=? AND claim_state='claimed' AND claim_token=?"
                    )
                    if self.is_postgres:
                        query += " FOR UPDATE"
                    row = conn.execute(
                        query,
                        (self.owner_user_id, dataset, ledger_id, claim_token),
                    ).fetchone()
                    if row is None:
                        raise _SemanticCASMiss
                    generation_id = str(row["generation_id"])
                    vector_id = str(row["vector_id"])
                    if self._obsolete_cleanup_blocked_by_work_locked(
                        conn,
                        dataset=dataset,
                        generation_id=generation_id,
                        note_id=row["note_id"],
                        dirty_generation=row["dirty_generation"],
                    ):
                        raise _SemanticCASMiss
                    visible = conn.execute(
                        "SELECT 1 FROM note_semantic_chunks c JOIN "
                        "note_semantic_note_state n ON n.owner_user_id=c.owner_user_id "
                        "AND n.dataset_id=c.dataset_id AND n.generation_id=c.generation_id "
                        "AND n.note_id=c.note_id WHERE c.owner_user_id=? AND "
                        "c.dataset_id=? AND c.generation_id=? AND c.chunk_id=? AND "
                        "n.state='indexed' AND n.content_version=c.content_version LIMIT 1",
                        (self.owner_user_id, dataset, generation_id, vector_id),
                    ).fetchone()
                    if visible is not None:
                        raise _SemanticCASMiss
                    conn.execute(
                        "DELETE FROM note_semantic_chunks WHERE owner_user_id=? AND "
                        "dataset_id=? AND generation_id=? AND chunk_id=?",
                        (self.owner_user_id, dataset, generation_id, vector_id),
                    )
                    cursor = conn.execute(
                        "DELETE FROM note_semantic_obsolete_vectors WHERE owner_user_id=? "
                        "AND dataset_id=? AND id=? AND claim_state='claimed' AND "
                        "claim_token=?",
                        (self.owner_user_id, dataset, ledger_id, claim_token),
                    )
                    if cursor.rowcount != 1:
                        raise _SemanticCASMiss
        except _SemanticCASMiss:
            return False
        return True

    def list_generation_cleanup_vector_ids(
        self,
        *,
        dataset_id: str,
        work_id: str,
        generation_id: str,
        claim_token: str,
        fencing_token: str,
        limit: int,
    ) -> tuple[str, ...] | None:
        if type(limit) is not int or not 1 <= limit <= 100_000:
            raise ValueError("notes_semantic_cleanup_limit_invalid")
        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            if not self._generation_cleanup_authorized_locked(
                conn,
                dataset=dataset,
                work_id=work_id,
                generation_id=generation_id,
                claim_token=claim_token,
                fencing_token=fencing_token,
            ):
                return None
            rows = conn.execute(
                "SELECT chunk_id FROM note_semantic_chunks WHERE owner_user_id=? AND "
                "dataset_id=? AND generation_id=? ORDER BY note_id,ordinal,chunk_id LIMIT ?",
                (self.owner_user_id, dataset, generation_id, limit),
            ).fetchall()
        return tuple(str(row["chunk_id"]) for row in rows)

    def complete_generation_vector_cleanup_page(
        self,
        *,
        dataset_id: str,
        work_id: str,
        generation_id: str,
        claim_token: str,
        fencing_token: str,
        vector_ids: Sequence[str],
    ) -> bool:
        ids = tuple(vector_ids)
        if not ids or len(ids) != len(set(ids)):
            return False
        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            if not self._generation_cleanup_authorized_locked(
                conn,
                dataset=dataset,
                work_id=work_id,
                generation_id=generation_id,
                claim_token=claim_token,
                fencing_token=fencing_token,
            ):
                return False
            deleted = 0
            for vector_id in ids:
                cursor = conn.execute(
                    "DELETE FROM note_semantic_chunks WHERE owner_user_id=? AND dataset_id=? "
                    "AND generation_id=? AND chunk_id=?",
                    (self.owner_user_id, dataset, generation_id, vector_id),
                )
                deleted += cursor.rowcount
        return deleted == len(ids)

    def create_configuration(
        self, *, dataset_id: str, capability_revision: str, disclosure_hash: str, provider: str,
        model: str, endpoint_origin_revision: str, endpoint_origin_display: str, data_boundary: str,
        vector_backend: str, storage_boundary: str, storage_label: str, normalization_version: str,
        chunker_version: str, now: datetime, model_revision: str | None = None,
    ) -> SemanticIndexConfig:
        dataset = self._scope(dataset_id)
        timestamp = self._timestamp(now)
        revision = (
            None
            if model_revision is None
            else self._safe_token(model_revision, field="model_revision")
        )
        params = (
            self.owner_user_id, dataset, self._safe_token(capability_revision, field="capability_revision"),
            self._safe_token(disclosure_hash, field="disclosure_hash"), self._safe_token(provider, field="provider"),
            self._safe_model(model), revision,
            self._safe_token(endpoint_origin_revision, field="endpoint_origin_revision"),
            self._endpoint_origin_display(endpoint_origin_display), self._safe_token(data_boundary, field="data_boundary"),
            self._safe_token(vector_backend, field="vector_backend"), self._safe_token(storage_boundary, field="storage_boundary"),
            self._safe_token(storage_label.replace(" ", "_"), field="storage_label"),
            self._safe_token(normalization_version, field="normalization_version"),
            self._safe_token(chunker_version, field="chunker_version"), timestamp,
            timestamp,
        )
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            self._serialize_dataset_mutation(conn, dataset)
            authority = conn.execute(
                "SELECT dataset_id FROM note_task_scope_authority WHERE owner_user_id=?",
                (self.owner_user_id,),
            ).fetchone()
            if authority is None:
                false_value: bool | int = False if self.is_postgres else 0
                conn.execute(
                    "INSERT INTO note_task_scope_authority("
                    "owner_user_id,dataset_id,task_graph_bound,moodboard_graph_bound,studio_graph_bound"
                    ") VALUES (?,?,?,?,?)",
                    (
                        self.owner_user_id,
                        dataset,
                        false_value,
                        false_value,
                        false_value,
                    ),
                )
            elif str(self._record(authority)["dataset_id"]) != dataset:
                raise SemanticIndexingError("notes_semantic_dataset_authority_conflict")
            conn.execute(
                """
                INSERT INTO note_semantic_index_configs(
                  owner_user_id,dataset_id,desired_state,configuration_revision,semantic_index_revision,
                  capability_revision,disclosure_hash,provider,model,model_revision,endpoint_origin_revision,
                  endpoint_origin_display,data_boundary,vector_backend,storage_boundary,storage_label,
                  metric,dimension_state,normalization_version,chunker_version,consented_at,updated_at
                ) VALUES (?,?, 'disabled',1,0,?,?,?,?,?,?,?,?,?,?,?, 'cosine','pending',?,?,?,?)
                """,
                params,
            )
            row = conn.execute(
                "SELECT * FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=?",
                (self.owner_user_id, dataset),
            ).fetchone()
        return self._config_from_row(row)

    def get_configuration(self, dataset_id: str) -> SemanticIndexConfig | None:
        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            row = conn.execute(
                "SELECT * FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=?",
                (self.owner_user_id, dataset),
            ).fetchone()
        return None if row is None else self._config_from_row(row)

    def _transition_configuration(
        self, *, dataset: str, expected_configuration_revision: int, capability_revision: str | None,
        desired_state: SemanticDesiredState, now: datetime,
    ) -> SemanticIndexConfig | None:
        timestamp = self._timestamp(now)
        where = "owner_user_id=? AND dataset_id=? AND configuration_revision=?"
        params: tuple[Any, ...] = (self.owner_user_id, dataset, expected_configuration_revision)
        if capability_revision is not None:
            where += " AND capability_revision=?"
            params += (self._safe_token(capability_revision, field="capability_revision"),)
        if desired_state is SemanticDesiredState.ENABLED:
            transition = "enabled_at=?"
        else:
            transition = "disabled_at=?"
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            self._serialize_dataset_mutation(conn, dataset)
            cursor = conn.execute(
                "UPDATE note_semantic_index_configs SET desired_state=?, configuration_revision=configuration_revision+1, "  # nosec B608
                + transition + ", updated_at=? WHERE " + where,
                (desired_state.value, timestamp, timestamp, *params),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=?",
                (self.owner_user_id, dataset),
            ).fetchone()
        return self._config_from_row(row)

    def enable_configuration(
        self, *, dataset_id: str, expected_configuration_revision: int, capability_revision: str,
        now: datetime,
    ) -> SemanticIndexConfig | None:
        return self._transition_configuration(
            dataset=self._scope(dataset_id), expected_configuration_revision=expected_configuration_revision,
            capability_revision=capability_revision, desired_state=SemanticDesiredState.ENABLED, now=now,
        )

    def renew_configuration_consent(
        self,
        *,
        dataset_id: str,
        expected_configuration_revision: int,
        capability_revision: str,
        disclosure_hash: str,
        compatibility_hash: str | None,
        provider: str,
        model: str,
        model_revision: str | None,
        endpoint_origin_revision: str,
        endpoint_origin_display: str,
        data_boundary: str,
        vector_backend: str,
        storage_boundary: str,
        storage_label: str,
        resolved_dimensions: int | None,
        normalization_version: str,
        chunker_version: str,
        now: datetime,
        activate_disabled: bool = False,
    ) -> SemanticIndexConfig | None:
        """Commit one disclosed capability, optionally re-enabling after cleanup."""

        if type(expected_configuration_revision) is not int or expected_configuration_revision < 0:
            raise ValueError("notes_semantic_configuration_revision_invalid")
        pending_dimensions = resolved_dimensions is None and compatibility_hash is None
        resolved_dimension = (
            type(resolved_dimensions) is int
            and resolved_dimensions > 0
            and compatibility_hash is not None
        )
        if not (pending_dimensions or resolved_dimension):
            raise ValueError("notes_semantic_dimensions_invalid")
        dataset = self._scope(dataset_id)
        timestamp = self._timestamp(now)
        revision = (
            None
            if model_revision is None
            else self._safe_token(model_revision, field="model_revision")
        )
        values = (
            self._safe_token(capability_revision, field="capability_revision"),
            self._safe_token(disclosure_hash, field="disclosure_hash"),
            (
                None
                if compatibility_hash is None
                else self._safe_token(compatibility_hash, field="compatibility_hash")
            ),
            self._safe_token(provider, field="provider"),
            self._safe_model(model),
            revision,
            self._safe_token(
                endpoint_origin_revision,
                field="endpoint_origin_revision",
            ),
            self._endpoint_origin_display(endpoint_origin_display),
            self._safe_token(data_boundary, field="data_boundary"),
            self._safe_token(vector_backend, field="vector_backend"),
            self._safe_token(storage_boundary, field="storage_boundary"),
            self._safe_token(
                storage_label.replace(" ", "_"),
                field="storage_label",
            ),
            "pending" if pending_dimensions else "resolved",
            resolved_dimensions,
            self._safe_token(
                normalization_version,
                field="normalization_version",
            ),
            self._safe_token(chunker_version, field="chunker_version"),
            timestamp,
            timestamp,
            timestamp,
            self.owner_user_id,
            dataset,
            expected_configuration_revision,
            "disabled" if activate_disabled else "enabled",
            "disabled" if activate_disabled else "enabled",
            self.owner_user_id,
            dataset,
            self.owner_user_id,
            dataset,
            self.owner_user_id,
            dataset,
        )
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            self._serialize_dataset_mutation(conn, dataset)
            cursor = conn.execute(
                "UPDATE note_semantic_index_configs SET capability_revision=?,"
                "disclosure_hash=?,compatibility_hash=?,provider=?,model=?,"
                "model_revision=?,endpoint_origin_revision=?,endpoint_origin_display=?,"
                "data_boundary=?,vector_backend=?,storage_boundary=?,storage_label=?,"
                "metric='cosine',dimension_state=?,dimensions=?,"
                "normalization_version=?,chunker_version=?,"
                "configuration_revision=configuration_revision+1,consented_at=?,"
                "updated_at=?,desired_state='enabled',"
                "enabled_at=CASE WHEN desired_state='disabled' THEN ? ELSE enabled_at END "
                "WHERE owner_user_id=? AND dataset_id=? AND configuration_revision=? "
                "AND (active_generation_id IS NULL OR desired_state='enabled') "
                "AND desired_state=? AND (?='enabled' OR (NOT EXISTS ("
                "SELECT 1 FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                "AND claim_state IN ('pending','claimed','failed')) AND NOT EXISTS ("
                "SELECT 1 FROM note_semantic_generations WHERE owner_user_id=? "
                "AND dataset_id=? AND deleted_at IS NULL) AND NOT EXISTS ("
                "SELECT 1 FROM note_semantic_obsolete_vectors WHERE owner_user_id=? "
                "AND dataset_id=?))) ",
                values,
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM note_semantic_index_configs "
                "WHERE owner_user_id=? AND dataset_id=?",
                (self.owner_user_id, dataset),
            ).fetchone()
        return None if row is None else self._config_from_row(row)

    def disable_configuration(
        self, *, dataset_id: str, expected_configuration_revision: int, now: datetime,
    ) -> SemanticIndexConfig | None:
        return self._transition_configuration(
            dataset=self._scope(dataset_id), expected_configuration_revision=expected_configuration_revision,
            capability_revision=None, desired_state=SemanticDesiredState.DISABLED, now=now,
        )

    def disable_and_schedule_cleanup(
        self,
        *,
        dataset_id: str,
        expected_configuration_revision: int,
        now: datetime,
    ) -> SemanticIndexConfig | None:
        """Disable reads, retire every generation, and durably queue deletion."""

        dataset = self._scope(dataset_id)
        timestamp = self._timestamp(now)
        try:
            with self._db.transaction() as conn:
                self._set_scope(conn, dataset)
                self._serialize_dataset_mutation(conn, dataset)
                config_query = (
                    "SELECT configuration_revision FROM note_semantic_index_configs "
                    "WHERE owner_user_id=? AND dataset_id=?"
                )
                if self.is_postgres:
                    config_query += " FOR UPDATE"
                config = conn.execute(
                    config_query,
                    (self.owner_user_id, dataset),
                ).fetchone()
                if config is None:
                    raise _SemanticCASMiss
                record = self._record(config)
                if int(record["configuration_revision"]) != expected_configuration_revision:
                    raise _SemanticCASMiss

                generations = conn.execute(
                    "SELECT id FROM note_semantic_generations g "
                    "WHERE owner_user_id=? AND dataset_id=? AND (deleted_at IS NULL OR EXISTS ("
                    "SELECT 1 FROM note_semantic_obsolete_vectors o WHERE "
                    "o.owner_user_id=g.owner_user_id AND o.dataset_id=g.dataset_id "
                    "AND o.generation_id=g.id))",
                    (self.owner_user_id, dataset),
                ).fetchall()
                updated = conn.execute(
                    "UPDATE note_semantic_index_configs SET desired_state='disabled', "
                    "active_generation_id=NULL, configuration_revision=configuration_revision+1, "
                    "semantic_index_revision=semantic_index_revision+1, disabled_at=?, updated_at=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND configuration_revision=?",
                    (
                        timestamp,
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        expected_configuration_revision,
                    ),
                )
                if updated.rowcount != 1:
                    raise _SemanticCASMiss
                conn.execute(
                    "UPDATE note_semantic_generations SET state='retired', retired_at=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND deleted_at IS NULL "
                    "AND state IN ('staging','active','failed')",
                    (timestamp, self.owner_user_id, dataset),
                )
                conn.execute(
                    "DELETE FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                    "AND kind IN ('index_note','delete_note_vectors') "
                    "AND claim_state<>'claimed'",
                    (self.owner_user_id, dataset),
                )
                for generation_row in generations:
                    generation_id = str(self._record(generation_row)["id"])
                    self._enqueue_work(
                        conn,
                        dataset=dataset,
                        kind=SemanticWorkKind.DELETE_GENERATION,
                        note_id=None,
                        generation_id=generation_id,
                        dirty_generation=None,
                        now=now,
                    )
                row = conn.execute(
                    "SELECT * FROM note_semantic_index_configs "
                    "WHERE owner_user_id=? AND dataset_id=?",
                    (self.owner_user_id, dataset),
                ).fetchone()
        except _SemanticCASMiss:
            return None
        return self._config_from_row(row)

    def create_generation(
        self, *, dataset_id: str, configuration_revision: int, compatibility_hash: str | None,
        dimension_state: SemanticDimensionState, dimensions: int | None, root_job_id: str | None,
        now: datetime, model_revision: str | None = None,
    ) -> SemanticGeneration:
        dataset = self._scope(dataset_id)
        if dimension_state is SemanticDimensionState.RESOLVED:
            if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
                raise ValueError("notes_semantic_dimensions_invalid")
            if compatibility_hash is None:
                raise ValueError("notes_semantic_compatibility_hash_invalid")
            resolved_compatibility_hash = self._safe_token(
                compatibility_hash,
                field="compatibility_hash",
            )
        else:
            if dimensions is not None:
                raise ValueError("notes_semantic_dimensions_pending_invalid")
            if compatibility_hash is not None:
                raise ValueError("notes_semantic_pending_compatibility_hash_invalid")
            resolved_compatibility_hash = None
        generation_id = str(uuid.uuid4())
        timestamp = self._timestamp(now)
        revision = (
            None
            if model_revision is None
            else self._safe_token(model_revision, field="model_revision")
        )
        normalized_root_job_id = (
            None
            if root_job_id is None
            else self._safe_token(root_job_id, field="root_job_id")
        )
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            self._serialize_dataset_mutation(conn, dataset)
            if (
                normalized_root_job_id is not None
                and self._has_unexpired_cancellation_intent_locked(
                    conn,
                    dataset_id=dataset,
                    root_job_id=normalized_root_job_id,
                    now=now,
                )
            ):
                raise SemanticIndexingError("notes_semantic_run_cancelled")
            if self.is_postgres:
                config_query = (
                    "SELECT configuration_revision,dimension_state,dimensions,compatibility_hash,model_revision "
                    "FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=? FOR UPDATE"
                )
            else:
                config_query = (
                    "SELECT configuration_revision,dimension_state,dimensions,compatibility_hash,model_revision "
                    "FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=?"
                )
            config = conn.execute(
                config_query,
                (self.owner_user_id, dataset),
            ).fetchone()
            if config is None:
                raise ValueError("notes_semantic_configuration_revision_mismatch")
            config_record = self._record(config)
            if int(config_record["configuration_revision"]) != configuration_revision:
                raise ValueError("notes_semantic_configuration_revision_mismatch")
            config_dimensions = config_record["dimensions"]
            if config_dimensions is not None:
                config_dimensions = int(config_dimensions)
            config_hash = config_record["compatibility_hash"]
            if config_hash is not None:
                config_hash = str(config_hash)
            if (
                str(config_record["dimension_state"]) != dimension_state.value
                or config_dimensions != dimensions
                or config_hash != resolved_compatibility_hash
                or config_record.get("model_revision") != revision
            ):
                raise ValueError("notes_semantic_generation_identity_mismatch")
            conn.execute(
                """
                INSERT INTO note_semantic_generations(
                  id,owner_user_id,dataset_id,configuration_revision,state,compatibility_hash,
                  model_revision,dimension_state,dimensions,root_job_id,created_at
                ) VALUES (?,?,?,?, 'staging',?,?,?,?,?,?)
                """,
                (
                    generation_id, self.owner_user_id, dataset, configuration_revision,
                    resolved_compatibility_hash, revision, dimension_state.value,
                    dimensions, normalized_root_job_id, timestamp,
                ),
            )
            row = conn.execute(
                "SELECT * FROM note_semantic_generations WHERE owner_user_id=? AND dataset_id=? AND id=?",
                (self.owner_user_id, dataset, generation_id),
            ).fetchone()
        return self._generation_from_row(row)

    def get_generation(self, dataset_id: str, generation_id: str) -> SemanticGeneration | None:
        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            row = conn.execute(
                "SELECT * FROM note_semantic_generations WHERE owner_user_id=? AND dataset_id=? AND id=?",
                (self.owner_user_id, dataset, generation_id),
            ).fetchone()
        return None if row is None else self._generation_from_row(row)

    def get_generation_by_root_job_id(
        self,
        dataset_id: str,
        root_job_id: str,
    ) -> SemanticGeneration | None:
        """Resolve a generation by its exact root Job recovery fence."""

        dataset = self._scope(dataset_id)
        root_job = self._safe_token(root_job_id, field="root_job_id")
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            rows = conn.execute(
                "SELECT * FROM note_semantic_generations "
                "WHERE owner_user_id=? AND dataset_id=? AND root_job_id=? "
                "ORDER BY created_at,id LIMIT 2",
                (self.owner_user_id, dataset, root_job),
            ).fetchall()
        if len(rows) > 1:
            raise SemanticIndexingError("notes_semantic_root_job_ambiguous")
        return None if not rows else self._generation_from_row(rows[0])

    def get_staging_generation(self, dataset_id: str) -> SemanticGeneration | None:
        """Return the sole staging generation for an owner/dataset scope."""

        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            row = conn.execute(
                "SELECT * FROM note_semantic_generations "
                "WHERE owner_user_id=? AND dataset_id=? AND state='staging' LIMIT 1",
                (self.owner_user_id, dataset),
            ).fetchone()
        return None if row is None else self._generation_from_row(row)

    def has_pending_cleanup(self, dataset_id: str) -> bool:
        """Return whether durable generation cleanup remains unconfirmed."""

        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            row = conn.execute(
                "SELECT 1 FROM note_semantic_work "
                "WHERE owner_user_id=? AND dataset_id=? AND kind='delete_generation' "
                "AND claim_state IN ('pending','claimed','failed') LIMIT 1",
                (self.owner_user_id, dataset),
            ).fetchone()
        return row is not None

    def can_rebind_disabled_configuration(self, dataset_id: str) -> bool:
        """Return whether prior semantic storage is confirmed clean."""

        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            row = conn.execute(
                "SELECT 1 FROM note_semantic_index_configs c WHERE "
                "c.owner_user_id=? AND c.dataset_id=? AND c.desired_state='disabled' "
                "AND c.active_generation_id IS NULL AND NOT EXISTS ("
                "SELECT 1 FROM note_semantic_work w WHERE w.owner_user_id=c.owner_user_id "
                "AND w.dataset_id=c.dataset_id AND "
                "w.claim_state IN ('pending','claimed','failed')) AND NOT EXISTS ("
                "SELECT 1 FROM note_semantic_generations g WHERE "
                "g.owner_user_id=c.owner_user_id AND g.dataset_id=c.dataset_id "
                "AND g.deleted_at IS NULL) AND NOT EXISTS ("
                "SELECT 1 FROM note_semantic_obsolete_vectors o WHERE "
                "o.owner_user_id=c.owner_user_id AND o.dataset_id=c.dataset_id) LIMIT 1",
                (self.owner_user_id, dataset),
            ).fetchone()
        return row is not None

    def has_stalled_cleanup(
        self,
        dataset_id: str,
        *,
        expired_before: datetime,
    ) -> bool:
        """Return whether generation cleanup exhausted attempts or lost its lease."""

        dataset = self._scope(dataset_id)
        expired = self._timestamp(expired_before)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            row = conn.execute(
                "SELECT 1 FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                "AND kind='delete_generation' AND (attempt_count>=? OR "
                "(claim_state='claimed' AND claimed_at<=?)) LIMIT 1",
                (self.owner_user_id, dataset, self._MAX_WORK_ATTEMPTS, expired),
            ).fetchone()
        return row is not None

    def rearm_exhausted_generation_cleanup(
        self,
        *,
        dataset_id: str,
        limit: int,
        now: datetime,
    ) -> int:
        """Boundedly rearm exhausted generation cleanup for production convergence."""

        dataset = self._scope(dataset_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("notes_semantic_maintenance_limit_invalid")
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            query = (
                "SELECT id FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                "AND kind='delete_generation' AND claim_state='failed' AND attempt_count>=? "
                "ORDER BY updated_at,id LIMIT ?"
            )
            if self.is_postgres:
                query += " FOR UPDATE SKIP LOCKED"
            rows = conn.execute(
                query,
                (self.owner_user_id, dataset, self._MAX_WORK_ATTEMPTS, limit),
            ).fetchall()
            rearmed = 0
            for row in rows:
                cursor = conn.execute(
                    "UPDATE note_semantic_work SET claim_state='pending',attempt_count=0,"
                    "next_eligible_at=?,claim_token=NULL,claimed_at=NULL,error_code=NULL,"
                    "updated_at=? WHERE owner_user_id=? AND dataset_id=? AND id=? "
                    "AND kind='delete_generation' AND claim_state='failed' AND attempt_count>=?",
                    (
                        timestamp,
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        str(self._record(row)["id"]),
                        self._MAX_WORK_ATTEMPTS,
                    ),
                )
                rearmed += cursor.rowcount
        return rearmed

    def has_pending_index_work(self, dataset_id: str, generation_id: str) -> bool:
        """Return whether eligible index work remains for one generation."""

        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            row = conn.execute(
                "SELECT 1 FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                "AND generation_id=? AND kind='index_note' "
                "AND claim_state IN ('pending','claimed','failed') LIMIT 1",
                (self.owner_user_id, dataset, generation_id),
            ).fetchone()
        return row is not None

    def list_maintenance_dataset_ids(
        self,
        *,
        limit: int,
        after_dataset_id: str | None = None,
    ) -> tuple[str, ...]:
        """List owner-scoped datasets that may require bounded maintenance."""

        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("notes_semantic_maintenance_limit_invalid")
        after = "" if after_dataset_id is None else self._scope(after_dataset_id)
        with self._db.transaction() as conn:
            rows = conn.execute(
                "SELECT dataset_id FROM note_task_scope_authority "
                "WHERE owner_user_id=? AND dataset_id>? ORDER BY dataset_id LIMIT ?",
                (self.owner_user_id, after, limit),
            ).fetchall()
            datasets: list[str] = []
            authority_queries = (
                "SELECT 1 FROM note_semantic_index_configs "
                "WHERE owner_user_id=? AND dataset_id=? LIMIT 1",
                "SELECT 1 FROM note_semantic_work "
                "WHERE owner_user_id=? AND dataset_id=? LIMIT 1",
                "SELECT 1 FROM note_semantic_obsolete_vectors "
                "WHERE owner_user_id=? AND dataset_id=? LIMIT 1",
                "SELECT 1 FROM note_semantic_operation_receipts "
                "WHERE owner_user_id=? AND dataset_id=? LIMIT 1",
            )
            for row in rows:
                dataset = str(self._record(row)["dataset_id"])
                self._set_scope(conn, dataset)
                if any(
                    conn.execute(query, (self.owner_user_id, dataset)).fetchone()
                    is not None
                    for query in authority_queries
                ):
                    datasets.append(dataset)
        return tuple(datasets)

    def list_observability_dataset_ids(
        self,
        *,
        limit: int,
        after_dataset_id: str | None = None,
    ) -> tuple[str, ...]:
        """Page raw owner dataset authority for complete health aggregation."""

        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("notes_semantic_maintenance_limit_invalid")
        after = "" if after_dataset_id is None else self._scope(after_dataset_id)
        with self._db.transaction() as conn:
            rows = conn.execute(
                "SELECT dataset_id FROM note_task_scope_authority "
                "WHERE owner_user_id=? AND dataset_id>? ORDER BY dataset_id LIMIT ?",
                (self.owner_user_id, after, limit),
            ).fetchall()
        return tuple(str(self._record(row)["dataset_id"]) for row in rows)

    def get_observability_snapshot(
        self,
        dataset_id: str,
        *,
        current_capability_revision: str | None,
    ) -> SemanticHealthSnapshot:
        """Read one authoritative owner/dataset health snapshot from durable state."""

        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            config_row = conn.execute(
                "SELECT active_generation_id,capability_revision,vector_backend,"
                "configuration_revision "
                "FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=?",
                (self.owner_user_id, dataset),
            ).fetchone()
            config = None if config_row is None else self._record(config_row)
            active_generation_id = None if config is None else config["active_generation_id"]
            observed_generation_id = active_generation_id
            if config is not None:
                working_row = conn.execute(
                    "SELECT id FROM note_semantic_generations WHERE owner_user_id=? "
                    "AND dataset_id=? AND configuration_revision=? AND deleted_at IS NULL "
                    "AND state IN ('staging','failed') ORDER BY "
                    "CASE WHEN state='staging' THEN 0 ELSE 1 END,created_at DESC,id DESC LIMIT 1",
                    (
                        self.owner_user_id,
                        dataset,
                        int(config["configuration_revision"]),
                    ),
                ).fetchone()
                if working_row is not None:
                    observed_generation_id = str(self._record(working_row)["id"])

            indexed = excluded = failed = pending = dirty = 0
            if observed_generation_id is not None:
                counts_row = conn.execute(
                    "SELECT "
                    "COALESCE(SUM(CASE WHEN state='indexed' THEN 1 ELSE 0 END),0) AS indexed,"
                    "COALESCE(SUM(CASE WHEN state='excluded' THEN 1 ELSE 0 END),0) AS excluded,"
                    "COALESCE(SUM(CASE WHEN state='failed' THEN 1 ELSE 0 END),0) AS failed,"
                    "COALESCE(SUM(CASE WHEN state='pending' THEN 1 ELSE 0 END),0) AS pending "
                    "FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? "
                    "AND generation_id=?",
                    (self.owner_user_id, dataset, observed_generation_id),
                ).fetchone()
                counts = self._record(counts_row)
                indexed = int(counts["indexed"])
                excluded = int(counts["excluded"])
                failed = int(counts["failed"])
                pending = int(counts["pending"])
                dirty_row = conn.execute(
                    "SELECT COUNT(DISTINCT note_id) AS dirty FROM note_semantic_work "
                    "WHERE owner_user_id=? AND dataset_id=? AND generation_id=? "
                    "AND kind='index_note' AND claim_state IN ('pending','claimed','failed')",
                    (self.owner_user_id, dataset, observed_generation_id),
                ).fetchone()
                dirty = int(self._record(dirty_row)["dirty"])

            stale_row = conn.execute(
                "SELECT COUNT(*) AS stale FROM note_semantic_generations WHERE "
                "owner_user_id=? AND dataset_id=? AND deleted_at IS NULL AND "
                "state IN ('retired','failed','deleting')",
                (self.owner_user_id, dataset),
            ).fetchone()
            stale = int(self._record(stale_row)["stale"])
            capability_stale = bool(
                current_capability_revision is not None
                and config is not None
                and config["capability_revision"] != current_capability_revision
            )
            if capability_stale:
                current_stale_row = conn.execute(
                    "SELECT COUNT(*) AS stale FROM note_semantic_generations WHERE "
                    "owner_user_id=? AND dataset_id=? AND configuration_revision=? "
                    "AND deleted_at IS NULL AND state IN ('active','staging')",
                    (
                        self.owner_user_id,
                        dataset,
                        int(config["configuration_revision"]),
                    ),
                ).fetchone()
                stale += int(self._record(current_stale_row)["stale"])

            cleanup_row = conn.execute(
                "SELECT COUNT(*) AS backlog,COALESCE(SUM(attempt_count),0) AS retries,"
                "MIN(created_at) AS oldest_created_at FROM ("
                "SELECT attempt_count,created_at FROM note_semantic_work "
                "WHERE owner_user_id=? AND dataset_id=? "
                "AND kind IN ('delete_note_vectors','delete_generation') "
                "AND claim_state IN ('pending','claimed','failed') UNION ALL "
                "SELECT attempt_count,created_at FROM note_semantic_obsolete_vectors "
                "WHERE owner_user_id=? AND dataset_id=? "
                "AND claim_state IN ('pending','claimed','failed')"
                ") AS cleanup",
                (self.owner_user_id, dataset, self.owner_user_id, dataset),
            ).fetchone()
            cleanup = self._record(cleanup_row)

        configured_backend = None if config is None else str(config["vector_backend"] or "")
        backend = configured_backend if configured_backend in {"chromadb", "pgvector"} else "unavailable"
        return SemanticHealthSnapshot(
            backend=backend,
            indexed_notes=indexed,
            excluded_notes=excluded,
            failed_notes=failed,
            dirty_notes=dirty,
            pending_notes=pending,
            stale_generations=stale,
            cleanup_backlog=int(cleanup["backlog"]),
            cleanup_retries=int(cleanup["retries"]),
            oldest_cleanup_created_at=self._read_iso(cleanup["oldest_created_at"]),
        )

    def finalize_owner_erasure(self, *, dataset_ids: Sequence[str]) -> int:
        """Atomically remove confirmed semantic state and owner Notes.

        PostgreSQL intentionally operates only through the canonical dataset authority;
        unsupported owner-wide RLS bypass is not part of the supported erasure flow.
        """

        datasets = tuple(dict.fromkeys(self._scope(value) for value in dataset_ids))
        try:
            with self._db.transaction() as conn:
                authority = conn.execute(
                    "SELECT dataset_id FROM note_task_scope_authority WHERE owner_user_id=?",
                    (self.owner_user_id,),
                ).fetchone()
                authority_dataset = (
                    None
                    if authority is None
                    else str(self._record(authority)["dataset_id"])
                )
                if any(dataset != authority_dataset for dataset in datasets):
                    raise _SemanticCASMiss
                lock_datasets = datasets or (
                    (authority_dataset,) if authority_dataset is not None else ()
                )
                for dataset in lock_datasets:
                    self._set_scope(conn, dataset)
                    self._serialize_dataset_mutation(conn, dataset)

                if not self.is_postgres:
                    foreign_scope = conn.execute(
                        "SELECT 1 FROM ("
                        "SELECT owner_user_id FROM note_semantic_index_configs "
                        "UNION SELECT owner_user_id FROM note_semantic_generations "
                        "UNION SELECT owner_user_id FROM note_semantic_work "
                        "UNION SELECT owner_user_id FROM note_semantic_obsolete_vectors "
                        "UNION SELECT owner_user_id FROM note_semantic_operation_receipts"
                        ") semantic_owners WHERE owner_user_id<>? LIMIT 1",
                        (self.owner_user_id,),
                    ).fetchone()
                    if foreign_scope is not None:
                        raise SemanticIndexingError(
                            "notes_semantic_erasure_unknown_physical_state"
                        )
                    catalog_rows = conn.execute(
                        "SELECT DISTINCT dataset_id FROM ("
                        "SELECT dataset_id FROM note_semantic_index_configs WHERE owner_user_id=? "
                        "UNION SELECT dataset_id FROM note_semantic_generations WHERE owner_user_id=? "
                        "UNION SELECT dataset_id FROM note_semantic_work WHERE owner_user_id=? "
                        "UNION SELECT dataset_id FROM note_semantic_obsolete_vectors WHERE owner_user_id=? "
                        "UNION SELECT dataset_id FROM note_semantic_operation_receipts WHERE owner_user_id=?"
                        ") ORDER BY dataset_id",
                        (self.owner_user_id,) * 5,
                    ).fetchall()
                    if tuple(str(row[0]) for row in catalog_rows) != tuple(sorted(datasets)):
                        raise _SemanticCASMiss

                for dataset in datasets:
                    self._set_scope(conn, dataset)
                    config = conn.execute(
                        "SELECT desired_state,active_generation_id FROM "
                        "note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=?",
                        (self.owner_user_id, dataset),
                    ).fetchone()
                    if config is None:
                        unknown = any(
                            conn.execute(
                                f"SELECT 1 FROM {table} WHERE owner_user_id=? "  # nosec B608
                                "AND dataset_id=? LIMIT 1",
                                (self.owner_user_id, dataset),
                            ).fetchone()
                            is not None
                            for table in (
                                "note_semantic_generations",
                                "note_semantic_work",
                                "note_semantic_obsolete_vectors",
                                "note_semantic_operation_receipts",
                            )
                        )
                        if unknown:
                            raise SemanticIndexingError(
                                "notes_semantic_erasure_unknown_physical_state"
                            )
                    else:
                        record = self._record(config)
                        if (
                            str(record["desired_state"]) != "disabled"
                            or record["active_generation_id"] is not None
                        ):
                            raise _SemanticCASMiss
                        live = conn.execute(
                            "SELECT 1 FROM note_semantic_generations WHERE owner_user_id=? "
                            "AND dataset_id=? AND deleted_at IS NULL LIMIT 1",
                            (self.owner_user_id, dataset),
                        ).fetchone()
                        pending = conn.execute(
                            "SELECT 1 FROM note_semantic_work WHERE owner_user_id=? AND "
                            "dataset_id=? AND claim_state IN ('pending','claimed','failed') LIMIT 1",
                            (self.owner_user_id, dataset),
                        ).fetchone()
                        obsolete = conn.execute(
                            "SELECT 1 FROM note_semantic_obsolete_vectors WHERE "
                            "owner_user_id=? AND dataset_id=? LIMIT 1",
                            (self.owner_user_id, dataset),
                        ).fetchone()
                        if live is not None or pending is not None or obsolete is not None:
                            raise _SemanticCASMiss
                    conn.execute(
                        "DELETE FROM note_semantic_operation_receipts WHERE "
                        "owner_user_id=? AND dataset_id=?",
                        (self.owner_user_id, dataset),
                    )
                    conn.execute(
                        "DELETE FROM note_semantic_index_configs WHERE "
                        "owner_user_id=? AND dataset_id=?",
                        (self.owner_user_id, dataset),
                    )

                if self.is_postgres:
                    conn.execute(
                        "DELETE FROM note_edges WHERE user_id=?",
                        (self.owner_user_id,),
                    )
                    conn.execute(
                        "DELETE FROM note_wikilink_edges WHERE owner_user_id=?",
                        (self.owner_user_id,),
                    )
                    deleted = conn.execute(
                        "DELETE FROM notes WHERE client_id=?",
                        (self.owner_user_id,),
                    ).rowcount
                else:
                    conn.execute("DELETE FROM note_edges")
                    conn.execute("DELETE FROM note_wikilink_edges")
                    deleted = conn.execute("DELETE FROM notes").rowcount
        except _SemanticCASMiss:
            raise SemanticIndexingError(
                "notes_semantic_erasure_finalization_fence_lost"
            ) from None
        return int(deleted)

    def list_dirty_generation_watermarks(
        self,
        *,
        dataset_id: str,
        limit: int,
    ) -> tuple[tuple[str, int], ...]:
        """List coalesced generations with eligible dirty Note work."""

        dataset = self._scope(dataset_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("notes_semantic_maintenance_limit_invalid")
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            rows = conn.execute(
                "SELECT generation_id,MAX(dirty_generation) AS dirty_generation "
                "FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                "AND kind='index_note' AND claim_state IN ('pending','failed') "
                "AND generation_id IS NOT NULL AND dirty_generation IS NOT NULL "
                "GROUP BY generation_id ORDER BY generation_id LIMIT ?",
                (self.owner_user_id, dataset, limit),
            ).fetchall()
        return tuple(
            (
                str(self._record(row)["generation_id"]),
                int(self._record(row)["dirty_generation"]),
            )
            for row in rows
        )

    def list_failed_generations(
        self,
        *,
        dataset_id: str,
        limit: int,
    ) -> tuple[str, ...]:
        """List active generations containing Note-local terminal failures."""

        dataset = self._scope(dataset_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("notes_semantic_maintenance_limit_invalid")
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            rows = conn.execute(
                "SELECT DISTINCT n.generation_id FROM note_semantic_note_state n "
                "JOIN note_semantic_generations g ON g.owner_user_id=n.owner_user_id "
                "AND g.dataset_id=n.dataset_id AND g.id=n.generation_id "
                "WHERE n.owner_user_id=? AND n.dataset_id=? AND n.state='failed' "
                "AND g.state='active' ORDER BY n.generation_id LIMIT ?",
                (self.owner_user_id, dataset, limit),
            ).fetchall()
        return tuple(str(self._record(row)["generation_id"]) for row in rows)

    def list_obsolete_cleanup_generations(
        self,
        *,
        dataset_id: str,
        limit: int,
    ) -> tuple[str, ...]:
        """List generations with crash-durable obsolete-vector cleanup rows."""

        dataset = self._scope(dataset_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("notes_semantic_maintenance_limit_invalid")
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            rows = conn.execute(
                "SELECT DISTINCT generation_id FROM note_semantic_obsolete_vectors "
                "WHERE owner_user_id=? AND dataset_id=? AND claim_state IN ('pending','failed') "
                "ORDER BY generation_id LIMIT ?",
                (self.owner_user_id, dataset, limit),
            ).fetchall()
        return tuple(str(self._record(row)["generation_id"]) for row in rows)

    def rearm_failed_notes(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        limit: int,
        now: datetime,
    ) -> int:
        """Boundedly rearm failed Notes as coalescing index work."""

        dataset = self._scope(dataset_id)
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("notes_semantic_work_claim_limit_invalid")
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            query = (
                "SELECT note_id,dirty_generation FROM note_semantic_note_state "
                "WHERE owner_user_id=? AND dataset_id=? AND generation_id=? "
                "AND state='failed' ORDER BY note_id LIMIT ?"
            )
            if self.is_postgres:
                query += " FOR UPDATE SKIP LOCKED"
            rows = conn.execute(
                query,
                (self.owner_user_id, dataset, generation_id, limit),
            ).fetchall()
            rearmed = 0
            for row in rows:
                record = self._record(row)
                note_id = str(record["note_id"])
                dirty_generation = int(record["dirty_generation"])
                updated = conn.execute(
                    "UPDATE note_semantic_note_state SET state='pending',error_code=NULL,"
                    "published_at=NULL WHERE owner_user_id=? AND dataset_id=? "
                    "AND generation_id=? AND note_id=? AND state='failed' "
                    "AND dirty_generation=?",
                    (
                        self.owner_user_id,
                        dataset,
                        generation_id,
                        note_id,
                        dirty_generation,
                    ),
                )
                if updated.rowcount != 1:
                    continue
                self._enqueue_work(
                    conn,
                    dataset=dataset,
                    kind=SemanticWorkKind.INDEX_NOTE,
                    note_id=note_id,
                    generation_id=generation_id,
                    dirty_generation=dirty_generation,
                    now=now,
                )
                rearmed += 1
            if rearmed:
                conn.execute(
                    "UPDATE note_semantic_index_configs SET semantic_index_revision="
                    "semantic_index_revision+1,updated_at=? WHERE owner_user_id=? "
                    "AND dataset_id=? AND desired_state='enabled'",
                    (timestamp, self.owner_user_id, dataset),
                )
        return rearmed

    def reclaim_expired_dataset_work(
        self,
        *,
        dataset_id: str,
        expired_before: datetime,
        limit: int,
        now: datetime,
    ) -> SemanticWorkReclaimResult:
        """Boundedly reclaim expired semantic claims across one dataset."""

        dataset = self._scope(dataset_id)
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("notes_semantic_work_claim_limit_invalid")
        expired = self._timestamp(expired_before)
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            self._serialize_dataset_mutation(conn, dataset)
            query = (
                "SELECT id,claim_token,kind,error_code FROM note_semantic_work WHERE owner_user_id=? "
                "AND dataset_id=? AND claim_state='claimed' AND claimed_at<=? "
                "ORDER BY claimed_at,id LIMIT ?"
            )
            if self.is_postgres:
                query += " FOR UPDATE SKIP LOCKED"
            rows = conn.execute(
                query,
                (self.owner_user_id, dataset, expired, limit),
            ).fetchall()
            reclaimed = 0
            cleanup_reclaimed = 0
            for row in rows:
                record = self._record(row)
                if (
                    str(record["kind"]) == SemanticWorkKind.INDEX_NOTE.value
                    and str(record.get("error_code") or "") == self._VECTOR_SIDE_EFFECT_IN_PROGRESS
                ):
                    # Once physical publication starts, lease expiry cannot prove that
                    # the vector writer has stopped. Its worker must drain and release.
                    continue
                cursor = conn.execute(
                    "UPDATE note_semantic_work SET claim_state='failed',"
                    "attempt_count=attempt_count+1,next_eligible_at=?,claim_token=NULL,"
                    "claimed_at=NULL,error_code='claim_lease_expired',updated_at=? WHERE "
                    "owner_user_id=? AND dataset_id=? AND id=? AND claim_state='claimed' "
                    "AND claim_token=? AND attempt_count<?",
                    (
                        timestamp,
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        str(record["id"]),
                        str(record["claim_token"]),
                        self._MAX_WORK_ATTEMPTS,
                    ),
                )
                committed = int(cursor.rowcount)
                reclaimed += committed
                if str(record["kind"]) in {
                    SemanticWorkKind.DELETE_NOTE_VECTORS.value,
                    SemanticWorkKind.DELETE_GENERATION.value,
                }:
                    cleanup_reclaimed += committed
        return SemanticWorkReclaimResult(
            total_transitions=reclaimed,
            cleanup_transitions=cleanup_reclaimed,
        )

    def authorize_note_vector_upsert(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        generation_id: str,
        generation_fencing_token: str,
        expected_configuration_revision: int,
        work_id: str,
        claim_token: str,
        work_fencing_token: str,
        claimed_dirty_generation: int,
        note_id: str,
        now: datetime,
    ) -> bool:
        """Fence one exact claimed Note before its external vector write starts."""

        if owner_user_id != self.owner_user_id:
            return False
        dataset = self._scope(dataset_id)
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            self._serialize_dataset_mutation(conn, dataset)
            generation = conn.execute(
                "SELECT 1 FROM note_semantic_generations g "
                "JOIN note_semantic_index_configs c ON c.owner_user_id=g.owner_user_id "
                "AND c.dataset_id=g.dataset_id WHERE g.owner_user_id=? "
                "AND g.dataset_id=? AND g.id=? AND g.root_job_id=? "
                "AND g.configuration_revision=? AND c.configuration_revision=? "
                "AND c.desired_state='enabled' AND g.state IN ('staging','active')",
                (
                    self.owner_user_id,
                    dataset,
                    generation_id,
                    generation_fencing_token,
                    expected_configuration_revision,
                    expected_configuration_revision,
                ),
            ).fetchone()
            if generation is None:
                return False
            authorized = conn.execute(
                "UPDATE note_semantic_work SET error_code=?,updated_at=? WHERE "
                "owner_user_id=? AND dataset_id=? AND id=? AND kind='index_note' "
                "AND generation_id=? AND note_id=? AND dirty_generation=? "
                "AND fencing_token=? AND claim_state='claimed' AND claim_token=?",
                (
                    self._VECTOR_SIDE_EFFECT_IN_PROGRESS,
                    timestamp,
                    self.owner_user_id,
                    dataset,
                    work_id,
                    generation_id,
                    note_id,
                    claimed_dirty_generation,
                    work_fencing_token,
                    claim_token,
                ),
            )
        return authorized.rowcount == 1

    def claim_generation_cleanup_batch(
        self,
        *,
        dataset_id: str,
        limit: int,
        now: datetime,
    ) -> tuple[SemanticWorkItem, ...]:
        """Claim bounded generation deletion work without touching index work."""

        dataset = self._scope(dataset_id)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("notes_semantic_maintenance_limit_invalid")
        timestamp = self._timestamp(now)
        claimed: list[SemanticWorkItem] = []
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            query = (
                "SELECT cleanup.id FROM note_semantic_work cleanup WHERE "
                "cleanup.owner_user_id=? AND cleanup.dataset_id=? "
                "AND cleanup.kind='delete_generation' "
                "AND cleanup.claim_state IN ('pending','failed') "
                "AND cleanup.attempt_count<? AND cleanup.next_eligible_at<=? "
                "AND NOT EXISTS (SELECT 1 FROM note_semantic_work writer WHERE "
                "writer.owner_user_id=cleanup.owner_user_id "
                "AND writer.dataset_id=cleanup.dataset_id "
                "AND writer.generation_id=cleanup.generation_id "
                "AND writer.kind IN ('index_note','delete_note_vectors') "
                "AND writer.claim_state='claimed') "
                "ORDER BY cleanup.next_eligible_at,cleanup.id LIMIT ?"
            )
            if self.is_postgres:
                query += " FOR UPDATE SKIP LOCKED"
            rows = conn.execute(
                query,
                (self.owner_user_id, dataset, self._MAX_WORK_ATTEMPTS, timestamp, limit),
            ).fetchall()
            for row in rows:
                work_id = str(self._record(row)["id"])
                claim_token = str(uuid.uuid4())
                cursor = conn.execute(
                    "UPDATE note_semantic_work SET claim_state='claimed',claim_token=?,"
                    "claimed_at=?,updated_at=? WHERE owner_user_id=? AND dataset_id=? "
                    "AND id=? AND kind='delete_generation' AND claim_state IN ('pending','failed')",
                    (
                        claim_token,
                        timestamp,
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        work_id,
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                value = conn.execute(
                    "SELECT * FROM note_semantic_work WHERE owner_user_id=? "
                    "AND dataset_id=? AND id=? AND claim_token=?",
                    (self.owner_user_id, dataset, work_id, claim_token),
                ).fetchone()
                if value is not None:
                    claimed.append(self._work_from_row(value))
        return tuple(claimed)

    def resolve_generation_dimensions(
        self, *, dataset_id: str, generation_id: str, expected_configuration_revision: int,
        dimensions: int, compatibility_hash: str, now: datetime,
        model_revision: str | None = None,
    ) -> SemanticGeneration | None:
        dataset = self._scope(dataset_id)
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("notes_semantic_dimensions_invalid")
        final_hash = self._safe_token(compatibility_hash, field="compatibility_hash")
        revision = (
            None
            if model_revision is None
            else self._safe_token(model_revision, field="model_revision")
        )
        timestamp = self._timestamp(now)
        next_revision = expected_configuration_revision + 1
        try:
            with self._db.transaction() as conn:
                self._set_scope(conn, dataset)
                self._serialize_dataset_mutation(conn, dataset)
                config_cursor = conn.execute(
                    "UPDATE note_semantic_index_configs SET dimension_state='resolved', dimensions=?, "
                    "compatibility_hash=?, model_revision=?, configuration_revision=?, updated_at=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND configuration_revision=? "
                    "AND desired_state='enabled' AND dimension_state='pending' "
                    "AND dimensions IS NULL AND compatibility_hash IS NULL",
                    (
                        dimensions, final_hash, revision, next_revision, timestamp, self.owner_user_id,
                        dataset, expected_configuration_revision,
                    ),
                )
                if config_cursor.rowcount != 1:
                    raise _SemanticCASMiss
                generation_cursor = conn.execute(
                    "UPDATE note_semantic_generations SET dimension_state='resolved', dimensions=?, "
                    "compatibility_hash=?, model_revision=?, configuration_revision=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=? AND configuration_revision=? "
                    "AND state='staging' AND dimension_state='pending' "
                    "AND dimensions IS NULL AND compatibility_hash IS NULL",
                    (
                        dimensions, final_hash, revision, next_revision, self.owner_user_id, dataset,
                        generation_id, expected_configuration_revision,
                    ),
                )
                if generation_cursor.rowcount != 1:
                    raise _SemanticCASMiss
                row = conn.execute(
                    "SELECT * FROM note_semantic_generations "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=?",
                    (self.owner_user_id, dataset, generation_id),
                ).fetchone()
        except _SemanticCASMiss:
            return None
        return self._generation_from_row(row)

    def activate_generation(
        self, *, dataset_id: str, generation_id: str, expected_configuration_revision: int,
        publication_receipt: str, now: datetime,
    ) -> SemanticIndexConfig | None:
        dataset = self._scope(dataset_id)
        timestamp = self._timestamp(now)
        receipt = self._safe_token(publication_receipt, field="publication_receipt")
        try:
            with self._db.transaction() as conn:
                self._set_scope(conn, dataset)
                self._serialize_dataset_mutation(conn, dataset)
                config = conn.execute(
                    "SELECT * FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=? "
                    "AND configuration_revision=? AND desired_state='enabled' "
                    "AND dimension_state='resolved' AND dimensions IS NOT NULL "
                    "AND compatibility_hash IS NOT NULL",
                    (self.owner_user_id, dataset, expected_configuration_revision),
                ).fetchone()
                if config is None:
                    raise _SemanticCASMiss
                config_record = self._record(config)
                previous_generation_id = config_record["active_generation_id"]
                if previous_generation_id == generation_id:
                    raise _SemanticCASMiss
                config_dimensions = int(config_record["dimensions"])
                config_hash = str(config_record["compatibility_hash"])
                candidate = conn.execute(
                    "SELECT id,root_job_id FROM note_semantic_generations "
                    "WHERE owner_user_id=? AND dataset_id=? "
                    "AND id=? AND configuration_revision=? AND state='staging' "
                    "AND dimension_state='resolved' AND dimensions=? AND compatibility_hash=?",
                    (
                        self.owner_user_id, dataset, generation_id,
                        expected_configuration_revision, config_dimensions, config_hash,
                    ),
                ).fetchone()
                if candidate is None:
                    raise _SemanticCASMiss
                candidate_root_job_id = self._record(candidate)["root_job_id"]
                if (
                    candidate_root_job_id is not None
                    and self._has_unexpired_cancellation_intent_locked(
                        conn,
                        dataset_id=dataset,
                        root_job_id=str(candidate_root_job_id),
                        now=now,
                    )
                ):
                    raise SemanticIndexingError("notes_semantic_run_cancelled")
                if previous_generation_id is not None:
                    retired = conn.execute(
                        "UPDATE note_semantic_generations SET state='retired', retired_at=? "
                        "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state='active'",
                        (timestamp, self.owner_user_id, dataset, previous_generation_id),
                    )
                    if retired.rowcount != 1:
                        raise _SemanticCASMiss
                activated = conn.execute(
                    "UPDATE note_semantic_generations SET state='active', publication_receipt=?, published_at=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=? AND configuration_revision=? "
                    "AND state='staging' AND dimension_state='resolved' "
                    "AND dimensions=? AND compatibility_hash=?",
                    (
                        receipt, timestamp, self.owner_user_id, dataset, generation_id,
                        expected_configuration_revision, config_dimensions, config_hash,
                    ),
                )
                if activated.rowcount != 1:
                    raise _SemanticCASMiss
                updated = conn.execute(
                    "UPDATE note_semantic_index_configs SET active_generation_id=?, "
                    "configuration_revision=configuration_revision+1, "
                    "semantic_index_revision=semantic_index_revision+1, updated_at=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND configuration_revision=? "
                    "AND desired_state='enabled' AND dimension_state='resolved' "
                    "AND dimensions=? AND compatibility_hash=?",
                    (
                        generation_id, timestamp, self.owner_user_id, dataset,
                        expected_configuration_revision, config_dimensions, config_hash,
                    ),
                )
                if updated.rowcount != 1:
                    raise _SemanticCASMiss
                if previous_generation_id is not None:
                    self._enqueue_work(
                        conn,
                        dataset=dataset,
                        kind=SemanticWorkKind.DELETE_GENERATION,
                        note_id=None,
                        generation_id=str(previous_generation_id),
                        dirty_generation=None,
                        now=now,
                    )
                row = conn.execute(
                    "SELECT * FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=?",
                    (self.owner_user_id, dataset),
                ).fetchone()
        except _SemanticCASMiss:
            return None
        return self._config_from_row(row)

    def record_note_dirty(
        self, *, dataset_id: str, generation_id: str, note_id: str, content_version: int,
        content_fingerprint: str, now: datetime, tx: SemanticConnection | None = None,
    ) -> SemanticNoteRecord:
        dataset = self._scope(dataset_id)
        if not isinstance(content_version, int) or content_version < 1:
            raise ValueError("notes_semantic_content_version_invalid")
        fingerprint = self._digest(content_fingerprint, field="content_fingerprint")
        timestamp = self._timestamp(now)

        def write(conn: SemanticConnection) -> SemanticNoteRecord:
            self._set_scope(conn, dataset)
            before = self._note_publication_contribution_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
                note_id=note_id,
                lock=True,
            )
            conn.execute(
                """
                INSERT INTO note_semantic_note_state(
                  owner_user_id,dataset_id,generation_id,note_id,content_version,content_fingerprint,
                  dirty_generation,state,chunk_count
                ) VALUES (?,?,?,?,?,?,1,'pending',0)
                ON CONFLICT(owner_user_id,dataset_id,generation_id,note_id) DO UPDATE SET
                  content_version=excluded.content_version, content_fingerprint=excluded.content_fingerprint,
                  dirty_generation=note_semantic_note_state.dirty_generation+1, state='pending',
                  error_code=NULL, published_at=NULL
                """,
                (self.owner_user_id, dataset, generation_id, note_id, content_version, fingerprint),
            )
            conn.execute(
                """
                INSERT INTO note_semantic_work(
                  id,owner_user_id,dataset_id,kind,note_id,generation_id,dirty_generation,
                  fencing_token,claim_state,attempt_count,next_eligible_at,created_at,updated_at
                ) VALUES (?,?,?,?,?,?,(SELECT dirty_generation FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? AND generation_id=? AND note_id=?),
                  ?,'pending',0,?,?,?)
                ON CONFLICT(owner_user_id,dataset_id,kind,note_id) WHERE note_id IS NOT NULL DO UPDATE SET
                  generation_id=excluded.generation_id, dirty_generation=excluded.dirty_generation,
                  fencing_token=excluded.fencing_token, claim_state='pending', attempt_count=0,
                  next_eligible_at=excluded.next_eligible_at, claim_token=NULL, claimed_at=NULL,
                  error_code=NULL, updated_at=excluded.updated_at
                WHERE NOT (
                  note_semantic_work.claim_state='claimed'
                  AND COALESCE(note_semantic_work.error_code,'')=?
                )
                """,
                (
                    str(uuid.uuid4()), self.owner_user_id, dataset, SemanticWorkKind.INDEX_NOTE.value,
                    note_id, generation_id, self.owner_user_id, dataset, generation_id, note_id,
                    str(uuid.uuid4()), timestamp, timestamp, timestamp,
                    self._VECTOR_SIDE_EFFECT_IN_PROGRESS,
                ),
            )
            row = conn.execute(
                "SELECT * FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? AND generation_id=? AND note_id=?",
                (self.owner_user_id, dataset, generation_id, note_id),
            ).fetchone()
            self._apply_note_contribution_delta_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
                note_id=note_id,
                before=before,
            )
            return self._note_from_row(row)

        if tx is not None:
            return write(tx)
        with self._db.transaction() as conn:
            return write(conn)

    def mark_note_dirty(
        self,
        *,
        dataset_id: str | None,
        note_id: str,
        content_version: int,
        content_fingerprint: str,
        now: datetime,
        tx: SemanticConnection,
    ) -> SemanticNoteRecord | None:
        """Mark a Note dirty only when an active local generation owns its dataset."""

        if dataset_id is None:
            return None
        dataset = self._scope(dataset_id)
        self._set_scope(tx, dataset)
        config = tx.execute(
            "SELECT g.id AS target_generation_id FROM note_semantic_index_configs c "
            "JOIN note_semantic_generations g ON g.owner_user_id=c.owner_user_id "
            "AND g.dataset_id=c.dataset_id WHERE c.owner_user_id=? AND c.dataset_id=? "
            "AND c.desired_state='enabled' AND g.state IN ('staging','active') "
            "ORDER BY CASE WHEN g.state='staging' THEN 0 ELSE 1 END LIMIT 1",
            (self.owner_user_id, dataset),
        ).fetchone()
        if config is None:
            return None
        generation_id = str(config["target_generation_id"])
        tx.execute(
            "DELETE FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
            "AND kind='delete_note_vectors' AND note_id=?",
            (self.owner_user_id, dataset, note_id),
        )
        return self.record_note_dirty(
            dataset_id=dataset,
            generation_id=generation_id,
            note_id=note_id,
            content_version=content_version,
            content_fingerprint=content_fingerprint,
            now=now,
            tx=tx,
        )

    def mark_note_tombstoned(
        self,
        *,
        dataset_id: str | None,
        note_id: str,
        content_version: int,
        content_fingerprint: str,
        hard_delete: bool = False,
        now: datetime,
        tx: SemanticConnection,
    ) -> SemanticNoteRecord | None:
        """Tombstone one Note and queue opaque vector cleanup when locally enabled."""

        if dataset_id is None:
            return None
        dataset = self._scope(dataset_id)
        if not isinstance(content_version, int) or content_version < 1:
            raise ValueError("notes_semantic_content_version_invalid")
        fingerprint = self._digest(content_fingerprint, field="content_fingerprint")
        timestamp = self._timestamp(now)
        self._set_scope(tx, dataset)
        if hard_delete:
            rows = tx.execute(
                "SELECT generation_id,chunk_id FROM note_semantic_chunks WHERE "
                "owner_user_id=? AND dataset_id=? AND note_id=? "
                "ORDER BY generation_id,ordinal,chunk_id",
                (self.owner_user_id, dataset, note_id),
            ).fetchall()
            ids_by_generation: dict[str, list[str]] = {}
            for vector_row in rows:
                ids_by_generation.setdefault(
                    str(vector_row["generation_id"]), []
                ).append(str(vector_row["chunk_id"]))
            for source_generation_id, vector_ids in ids_by_generation.items():
                self._stage_obsolete_vectors_locked(
                    tx,
                    dataset=dataset,
                    generation_id=source_generation_id,
                    vector_ids=vector_ids,
                    source_kind="hard_delete",
                    note_id=note_id,
                    dirty_generation=None,
                    now=now,
                )
        config = tx.execute(
            "SELECT g.id AS target_generation_id FROM note_semantic_index_configs c "
            "JOIN note_semantic_generations g ON g.owner_user_id=c.owner_user_id "
            "AND g.dataset_id=c.dataset_id WHERE c.owner_user_id=? AND c.dataset_id=? "
            "AND c.desired_state='enabled' AND g.state IN ('staging','active') "
            "ORDER BY CASE WHEN g.state='staging' THEN 0 ELSE 1 END LIMIT 1",
            (self.owner_user_id, dataset),
        ).fetchone()
        if config is None:
            return None
        generation_id = str(config["target_generation_id"])
        before = self._note_publication_contribution_locked(
            tx,
            dataset=dataset,
            generation_id=generation_id,
            note_id=note_id,
            lock=True,
        )
        obsolete_ids = tuple(
            str(row["chunk_id"])
            for row in tx.execute(
                "SELECT chunk_id FROM note_semantic_chunks WHERE owner_user_id=? AND "
                "dataset_id=? AND generation_id=? AND note_id=? ORDER BY ordinal,chunk_id",
                (self.owner_user_id, dataset, generation_id, note_id),
            ).fetchall()
        )
        tx.execute(
            """
            INSERT INTO note_semantic_note_state(
              owner_user_id,dataset_id,generation_id,note_id,content_version,content_fingerprint,
              dirty_generation,state,chunk_count,published_at
            ) VALUES (?,?,?,?,?,?,1,'tombstoned',0,?)
            ON CONFLICT(owner_user_id,dataset_id,generation_id,note_id) DO UPDATE SET
              content_version=excluded.content_version, content_fingerprint=excluded.content_fingerprint,
              dirty_generation=note_semantic_note_state.dirty_generation+1, state='tombstoned',
              manifest_hash=NULL, chunk_count=0, error_code=NULL, published_at=excluded.published_at
            """,
            (
                self.owner_user_id,
                dataset,
                generation_id,
                note_id,
                content_version,
                fingerprint,
                timestamp,
            ),
        )
        row = tx.execute(
            "SELECT * FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? "
            "AND generation_id=? AND note_id=?",
            (self.owner_user_id, dataset, generation_id, note_id),
        ).fetchone()
        dirty_generation = int(row["dirty_generation"])
        self._stage_obsolete_vectors_locked(
            tx,
            dataset=dataset,
            generation_id=generation_id,
            vector_ids=obsolete_ids,
            source_kind="hard_delete" if hard_delete else "tombstone",
            note_id=note_id,
            dirty_generation=dirty_generation,
            now=now,
        )
        tx.execute(
            "DELETE FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
            "AND kind='index_note' AND note_id=? AND NOT ("
            "claim_state='claimed' AND COALESCE(error_code,'')=?)",
            (
                self.owner_user_id,
                dataset,
                note_id,
                self._VECTOR_SIDE_EFFECT_IN_PROGRESS,
            ),
        )
        self._enqueue_work(
            tx,
            dataset=dataset,
            kind=SemanticWorkKind.DELETE_NOTE_VECTORS,
            note_id=note_id,
            generation_id=generation_id,
            dirty_generation=dirty_generation,
            now=now,
        )
        tx.execute(
            "UPDATE note_semantic_index_configs SET semantic_index_revision=semantic_index_revision+1, "
            "updated_at=? WHERE owner_user_id=? AND dataset_id=?",
            (timestamp, self.owner_user_id, dataset),
        )
        self._apply_note_contribution_delta_locked(
            tx,
            dataset=dataset,
            generation_id=generation_id,
            note_id=note_id,
            before=before,
        )
        return self._note_from_row(row)

    def claim_dirty_note(
        self, *, dataset_id: str, generation_id: str, note_id: str, dirty_generation: int,
        now: datetime,
    ) -> SemanticNoteRecord | None:
        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            row = conn.execute(
                "SELECT * FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? AND generation_id=? "
                "AND note_id=? AND dirty_generation=? AND state='pending'",
                (self.owner_user_id, dataset, generation_id, note_id, dirty_generation),
            ).fetchone()
        return None if row is None else self._note_from_row(row)

    def publish_note_manifest(
        self, *, owner_user_id: str, dataset_id: str, generation_id: str, note_id: str,
        claimed_dirty_generation: int, content_version: int, manifest: dict[str, Any], now: datetime,
    ) -> bool:
        if owner_user_id != self.owner_user_id:
            return False
        dataset = self._scope(dataset_id)
        chunk_count = manifest.get("chunk_count")
        manifest_hash = manifest.get("manifest_hash")
        if not isinstance(chunk_count, int) or chunk_count < 0 or not isinstance(manifest_hash, str):
            raise ValueError("notes_semantic_manifest_invalid")
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            before = self._note_publication_contribution_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
                note_id=note_id,
                lock=True,
            )
            cursor = conn.execute(
                "UPDATE note_semantic_note_state SET state='indexed', chunk_count=?, manifest_hash=?, error_code=NULL, published_at=? "
                "WHERE owner_user_id=? AND dataset_id=? AND generation_id=? AND note_id=? "
                "AND dirty_generation=? AND content_version=? AND state='pending'",
                (chunk_count, manifest_hash, timestamp, self.owner_user_id, dataset, generation_id, note_id,
                 claimed_dirty_generation, content_version),
            )
            if cursor.rowcount != 1:
                return False
            conn.execute(
                "UPDATE note_semantic_index_configs SET semantic_index_revision=semantic_index_revision+1, updated_at=? "
                "WHERE owner_user_id=? AND dataset_id=?",
                (timestamp, self.owner_user_id, dataset),
            )
            self._apply_note_contribution_delta_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
                note_id=note_id,
                before=before,
            )
        return True

    def tombstone_note(
        self, *, dataset_id: str, generation_id: str, note_id: str, content_version: int,
        dirty_generation: int, now: datetime,
    ) -> SemanticNoteRecord | None:
        dataset = self._scope(dataset_id)
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            before = self._note_publication_contribution_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
                note_id=note_id,
                lock=True,
            )
            cursor = conn.execute(
                "UPDATE note_semantic_note_state SET state='tombstoned', content_version=?, dirty_generation=?, "
                "manifest_hash=NULL, chunk_count=0, published_at=? WHERE owner_user_id=? AND dataset_id=? "
                "AND generation_id=? AND note_id=? AND dirty_generation < ?",
                (content_version, dirty_generation, timestamp, self.owner_user_id, dataset, generation_id,
                 note_id, dirty_generation),
            )
            if cursor.rowcount != 1:
                return None
            conn.execute(
                "DELETE FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                "AND kind='index_note' AND note_id=? AND NOT ("
                "claim_state='claimed' AND COALESCE(error_code,'')=?)",
                (
                    self.owner_user_id,
                    dataset,
                    note_id,
                    self._VECTOR_SIDE_EFFECT_IN_PROGRESS,
                ),
            )
            self._enqueue_work(
                conn, dataset=dataset, kind=SemanticWorkKind.DELETE_NOTE_VECTORS, note_id=note_id,
                generation_id=generation_id, dirty_generation=dirty_generation, now=now,
            )
            conn.execute(
                "UPDATE note_semantic_index_configs SET semantic_index_revision=semantic_index_revision+1, updated_at=? "
                "WHERE owner_user_id=? AND dataset_id=?",
                (timestamp, self.owner_user_id, dataset),
            )
            row = conn.execute(
                "SELECT * FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? AND generation_id=? AND note_id=?",
                (self.owner_user_id, dataset, generation_id, note_id),
            ).fetchone()
            self._apply_note_contribution_delta_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
                note_id=note_id,
                before=before,
            )
        return self._note_from_row(row)

    def _enqueue_work(
        self, conn: SemanticConnection, *, dataset: str, kind: SemanticWorkKind, note_id: str | None,
        generation_id: str | None, dirty_generation: int | None, now: datetime,
    ) -> None:
        timestamp = self._timestamp(now)
        params = (
            str(uuid.uuid4()), self.owner_user_id, dataset, kind.value, note_id, generation_id,
            dirty_generation, str(uuid.uuid4()), timestamp, timestamp, timestamp,
        )
        if kind is SemanticWorkKind.DELETE_GENERATION:
            conn.execute(
                "INSERT INTO note_semantic_work(id,owner_user_id,dataset_id,kind,note_id,generation_id,dirty_generation,"
                "fencing_token,claim_state,attempt_count,next_eligible_at,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,'pending',0,?,?,?) "
                "ON CONFLICT(owner_user_id,dataset_id,kind,generation_id) "
                "WHERE generation_id IS NOT NULL AND kind='delete_generation' DO UPDATE SET "
                "generation_id=excluded.generation_id, dirty_generation=excluded.dirty_generation, "
                "fencing_token=excluded.fencing_token, claim_state='pending', attempt_count=0, next_eligible_at=excluded.next_eligible_at, "
                "claim_token=NULL, claimed_at=NULL, error_code=NULL, updated_at=excluded.updated_at "
                "WHERE note_semantic_work.claim_state<>'claimed'",
                params,
            )
        elif kind is SemanticWorkKind.INDEX_NOTE:
            conn.execute(
                "INSERT INTO note_semantic_work(id,owner_user_id,dataset_id,kind,note_id,generation_id,dirty_generation,"
                "fencing_token,claim_state,attempt_count,next_eligible_at,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,'pending',0,?,?,?) "
                "ON CONFLICT(owner_user_id,dataset_id,kind,note_id) WHERE note_id IS NOT NULL DO UPDATE SET "
                "generation_id=excluded.generation_id, dirty_generation=excluded.dirty_generation, "
                "fencing_token=excluded.fencing_token, claim_state='pending', attempt_count=0, next_eligible_at=excluded.next_eligible_at, "
                "claim_token=NULL, claimed_at=NULL, error_code=NULL, updated_at=excluded.updated_at "
                "WHERE NOT (note_semantic_work.claim_state='claimed' "
                "AND COALESCE(note_semantic_work.error_code,'')=?)",
                (*params, self._VECTOR_SIDE_EFFECT_IN_PROGRESS),
            )
        else:
            conn.execute(
                "INSERT INTO note_semantic_work(id,owner_user_id,dataset_id,kind,note_id,generation_id,dirty_generation,"
                "fencing_token,claim_state,attempt_count,next_eligible_at,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,'pending',0,?,?,?) "
                "ON CONFLICT(owner_user_id,dataset_id,kind,note_id) WHERE note_id IS NOT NULL DO UPDATE SET "
                "generation_id=excluded.generation_id, dirty_generation=excluded.dirty_generation, "
                "fencing_token=excluded.fencing_token, claim_state='pending', attempt_count=0, next_eligible_at=excluded.next_eligible_at, "
                "claim_token=NULL, claimed_at=NULL, error_code=NULL, updated_at=excluded.updated_at",
                params,
            )

    def claim_work(self, *, dataset_id: str, now: datetime) -> SemanticWorkItem | None:
        dataset = self._scope(dataset_id)
        timestamp = self._timestamp(now)
        claim_token = str(uuid.uuid4())
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            row = conn.execute(
                "SELECT id FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                "AND claim_state IN ('pending','failed') AND attempt_count < ? AND next_eligible_at <= ? "
                "ORDER BY next_eligible_at,id LIMIT 1",
                (self.owner_user_id, dataset, self._MAX_WORK_ATTEMPTS, timestamp),
            ).fetchone()
            if row is None:
                return None
            work_id = str(row[0])
            conn.execute(
                "UPDATE note_semantic_work SET claim_state='claimed', claim_token=?, claimed_at=?, updated_at=? "
                "WHERE owner_user_id=? AND dataset_id=? AND id=? AND claim_state IN ('pending','failed')",
                (claim_token, timestamp, timestamp, self.owner_user_id, dataset, work_id),
            )
            claimed = conn.execute(
                "SELECT * FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? AND id=? AND claim_token=?",
                (self.owner_user_id, dataset, work_id, claim_token),
            ).fetchone()
        return None if claimed is None else self._work_from_row(claimed)

    def retry_work(
        self, *, dataset_id: str, work_id: str, expected_claim_token: str | None, error_code: str,
        retry_at: datetime, now: datetime,
    ) -> SemanticWorkItem | None:
        dataset = self._scope(dataset_id)
        if not expected_claim_token:
            return None
        retry_timestamp = self._timestamp(retry_at)
        now_timestamp = self._timestamp(now)
        if retry_timestamp <= now_timestamp:
            raise ValueError("notes_semantic_retry_timestamp_invalid")
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            cursor = conn.execute(
                "UPDATE note_semantic_work SET claim_state='failed', attempt_count=attempt_count+1, next_eligible_at=?, "
                "claim_token=NULL, claimed_at=NULL, error_code=?, updated_at=? WHERE owner_user_id=? AND dataset_id=? "
                "AND id=? AND claim_state='claimed' AND claim_token=? AND attempt_count < ?",
                (retry_timestamp, self._error_code(error_code), now_timestamp, self.owner_user_id, dataset, work_id,
                 expected_claim_token, self._MAX_WORK_ATTEMPTS),
            )
            if cursor.rowcount != 1:
                return None
            row = conn.execute(
                "SELECT * FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? AND id=?",
                (self.owner_user_id, dataset, work_id),
            ).fetchone()
        return self._work_from_row(row)

    def fail_generation(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        generation_fencing_token: str,
        expected_configuration_revision: int,
        error_code: str,
        now: datetime,
    ) -> bool:
        """Mark a staging generation systemically failed behind its root fence."""

        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            cursor = conn.execute(
                "UPDATE note_semantic_generations SET state='failed',terminal_error_code=? "
                "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state='staging' "
                "AND root_job_id=? AND configuration_revision=? AND EXISTS ("
                "SELECT 1 FROM note_semantic_index_configs c WHERE "
                "c.owner_user_id=note_semantic_generations.owner_user_id AND "
                "c.dataset_id=note_semantic_generations.dataset_id AND "
                "c.configuration_revision=? AND c.desired_state='enabled')",
                (
                    self._error_code(error_code),
                    self.owner_user_id,
                    dataset,
                    generation_id,
                    generation_fencing_token,
                    expected_configuration_revision,
                    expected_configuration_revision,
                ),
            )
            if cursor.rowcount == 1:
                self._enqueue_work(
                    conn,
                    dataset=dataset,
                    kind=SemanticWorkKind.DELETE_GENERATION,
                    note_id=None,
                    generation_id=generation_id,
                    dirty_generation=None,
                    now=now,
                )
        return cursor.rowcount == 1

    def seed_generation_snapshot(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        expected_configuration_revision: int,
        generation_fencing_token: str,
        seeds: Sequence[SemanticSnapshotSeed],
        now: datetime,
    ) -> bool:
        """CAS one bounded active-Note snapshot into a staging generation."""

        dataset = self._scope(dataset_id)
        if len(seeds) > 100_000:
            raise ValueError("notes_semantic_snapshot_limit_exceeded")
        normalized: list[tuple[SemanticSnapshotSeed, str]] = []
        note_ids: set[str] = set()
        for seed in seeds:
            if seed.note_id in note_ids:
                raise ValueError("notes_semantic_snapshot_note_duplicate")
            note_ids.add(seed.note_id)
            if type(seed.content_version) is not int or seed.content_version < 1:
                raise ValueError("notes_semantic_content_version_invalid")
            if type(seed.planned_chunk_count) is not int or seed.planned_chunk_count < 0:
                raise ValueError("notes_semantic_chunk_count_invalid")
            state = self._state_value(seed.state)
            if state not in {
                SemanticNoteState.PENDING.value,
                SemanticNoteState.EXCLUDED.value,
                SemanticNoteState.FAILED.value,
            }:
                raise ValueError("notes_semantic_snapshot_state_invalid")
            if (
                state == SemanticNoteState.PENDING.value
                and seed.planned_chunk_count == 0
            ):
                raise ValueError("notes_semantic_pending_chunks_required")
            if (
                state != SemanticNoteState.PENDING.value
                and seed.planned_chunk_count != 0
            ):
                raise ValueError("notes_semantic_terminal_chunks_unexpected")
            if state in {SemanticNoteState.EXCLUDED.value, SemanticNoteState.FAILED.value}:
                if seed.error_code is None:
                    raise ValueError("notes_semantic_terminal_error_required")
            elif seed.error_code is not None:
                raise ValueError("notes_semantic_terminal_error_unexpected")
            normalized.append((seed, state))
        timestamp = self._timestamp(now)
        try:
            with self._db.transaction() as conn:
                self._set_scope(conn, dataset)
                generation = conn.execute(
                    "SELECT g.id,g.expected_note_count,g.expected_chunk_count FROM "
                    "note_semantic_generations g "
                    "JOIN note_semantic_index_configs c ON c.owner_user_id=g.owner_user_id "
                    "AND c.dataset_id=g.dataset_id WHERE g.owner_user_id=? AND g.dataset_id=? "
                    "AND g.id=? AND g.configuration_revision=? AND g.state='staging' "
                    "AND g.dimension_state='resolved' AND g.root_job_id=? "
                    "AND c.configuration_revision=? AND c.desired_state='enabled'",
                    (
                        self.owner_user_id,
                        dataset,
                        generation_id,
                        expected_configuration_revision,
                        generation_fencing_token,
                        expected_configuration_revision,
                    ),
                ).fetchone()
                if generation is None:
                    raise _SemanticCASMiss
                existing_ids = {
                    str(row["note_id"])
                    for row in conn.execute(
                        "SELECT note_id FROM note_semantic_note_state WHERE owner_user_id=? "
                        "AND dataset_id=? AND generation_id=?",
                        (self.owner_user_id, dataset, generation_id),
                    ).fetchall()
                }
                removed_ids = existing_ids - note_ids
                for note_id in sorted(removed_ids):
                    removed_state = conn.execute(
                        "SELECT dirty_generation FROM note_semantic_note_state WHERE "
                        "owner_user_id=? AND dataset_id=? AND generation_id=? AND note_id=?",
                        (self.owner_user_id, dataset, generation_id, note_id),
                    ).fetchone()
                    removed_vector_ids = tuple(
                        str(row["chunk_id"])
                        for row in conn.execute(
                            "SELECT chunk_id FROM note_semantic_chunks WHERE owner_user_id=? "
                            "AND dataset_id=? AND generation_id=? AND note_id=? "
                            "ORDER BY ordinal,chunk_id",
                            (self.owner_user_id, dataset, generation_id, note_id),
                        ).fetchall()
                    )
                    staged = self._stage_obsolete_vectors_locked(
                        conn,
                        dataset=dataset,
                        generation_id=generation_id,
                        vector_ids=removed_vector_ids,
                        source_kind="manifest_replace",
                        note_id=note_id,
                        dirty_generation=(
                            None
                            if removed_state is None
                            else int(removed_state["dirty_generation"])
                        ),
                        now=now,
                    )
                    if staged != len(removed_vector_ids):
                        raise _SemanticCASMiss
                    conn.execute(
                        "DELETE FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                        "AND generation_id=? AND note_id=? AND kind='index_note' "
                        "AND NOT (claim_state='claimed' AND COALESCE(error_code,'')=?)",
                        (
                            self.owner_user_id,
                            dataset,
                            generation_id,
                            note_id,
                            self._VECTOR_SIDE_EFFECT_IN_PROGRESS,
                        ),
                    )
                    conn.execute(
                        "DELETE FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? "
                        "AND generation_id=? AND note_id=?",
                        (self.owner_user_id, dataset, generation_id, note_id),
                    )
                for seed, state in normalized:
                    note = conn.execute(
                        "SELECT version,deleted FROM notes WHERE client_id=? AND id=?",
                        (self.owner_user_id, seed.note_id),
                    ).fetchone()
                    if (
                        note is None
                        or bool(note["deleted"])
                        or int(note["version"]) != seed.content_version
                    ):
                        raise _SemanticCASMiss
                    fingerprint = self._digest(
                        seed.content_fingerprint,
                        field="content_fingerprint",
                    )
                    error_code = self._error_code(seed.error_code)
                    current = conn.execute(
                        "SELECT content_version,content_fingerprint,dirty_generation,state,"
                        "chunk_count,error_code FROM note_semantic_note_state "
                        "WHERE owner_user_id=? AND dataset_id=? AND generation_id=? AND note_id=?",
                        (self.owner_user_id, dataset, generation_id, seed.note_id),
                    ).fetchone()
                    chunk_count = (
                        seed.planned_chunk_count
                        if state == SemanticNoteState.PENDING.value
                        else 0
                    )
                    if current is None:
                        changed = True
                        dirty_generation = 1
                        conn.execute(
                            "INSERT INTO note_semantic_note_state(owner_user_id,dataset_id,generation_id,"
                            "note_id,content_version,content_fingerprint,dirty_generation,state,chunk_count,"
                            "manifest_hash,error_code,published_at) VALUES (?,?,?,?,?,?,1,?,?,?,?,?)",
                            (
                                self.owner_user_id,
                                dataset,
                                generation_id,
                                seed.note_id,
                                seed.content_version,
                                fingerprint,
                                state,
                                chunk_count,
                                None,
                                error_code,
                                timestamp if state != SemanticNoteState.PENDING.value else None,
                            ),
                        )
                    else:
                        exact_state_match = (
                            int(current["content_version"]) == seed.content_version
                            and str(current["content_fingerprint"]) == fingerprint
                            and str(current["state"]) == state
                            and int(current["chunk_count"]) == chunk_count
                            and current["error_code"] == error_code
                        )
                        preserved_terminal = (
                            int(current["content_version"]) == seed.content_version
                            and str(current["content_fingerprint"]) == fingerprint
                            and state == SemanticNoteState.PENDING.value
                            and str(current["state"])
                            in {
                                SemanticNoteState.INDEXED.value,
                                SemanticNoteState.FAILED.value,
                            }
                            and int(current["chunk_count"])
                            == seed.planned_chunk_count
                        )
                        unchanged = exact_state_match or preserved_terminal
                        dirty_generation = int(current["dirty_generation"])
                        changed = not unchanged
                        if not unchanged:
                            dirty_generation += 1
                        if state != SemanticNoteState.PENDING.value:
                            obsolete_ids = tuple(
                                str(row["chunk_id"])
                                for row in conn.execute(
                                    "SELECT chunk_id FROM note_semantic_chunks WHERE "
                                    "owner_user_id=? AND dataset_id=? AND generation_id=? "
                                    "AND note_id=? ORDER BY ordinal,chunk_id",
                                    (
                                        self.owner_user_id,
                                        dataset,
                                        generation_id,
                                        seed.note_id,
                                    ),
                                ).fetchall()
                            )
                            staged = self._stage_obsolete_vectors_locked(
                                conn,
                                dataset=dataset,
                                generation_id=generation_id,
                                vector_ids=obsolete_ids,
                                source_kind=(
                                    "note_failure"
                                    if state == SemanticNoteState.FAILED.value
                                    else "manifest_replace"
                                ),
                                note_id=seed.note_id,
                                dirty_generation=dirty_generation,
                                now=now,
                            )
                            if staged != len(obsolete_ids):
                                raise _SemanticCASMiss
                            conn.execute(
                                "DELETE FROM note_semantic_chunks WHERE owner_user_id=? "
                                "AND dataset_id=? AND generation_id=? AND note_id=?",
                                (
                                    self.owner_user_id,
                                    dataset,
                                    generation_id,
                                    seed.note_id,
                                ),
                            )
                        if not unchanged:
                            conn.execute(
                                "UPDATE note_semantic_note_state SET content_version=?,"
                                "content_fingerprint=?,dirty_generation=?,state=?,chunk_count=?,"
                                "manifest_hash=NULL,error_code=?,published_at=? WHERE owner_user_id=? "
                                "AND dataset_id=? AND generation_id=? AND note_id=?",
                                (
                                    seed.content_version,
                                    fingerprint,
                                    dirty_generation,
                                    state,
                                    chunk_count,
                                    error_code,
                                    timestamp if state != SemanticNoteState.PENDING.value else None,
                                    self.owner_user_id,
                                    dataset,
                                    generation_id,
                                    seed.note_id,
                                ),
                            )
                    if state == SemanticNoteState.PENDING.value and changed:
                        self._enqueue_work(
                            conn,
                            dataset=dataset,
                            kind=SemanticWorkKind.INDEX_NOTE,
                            note_id=seed.note_id,
                            generation_id=generation_id,
                            dirty_generation=dirty_generation,
                            now=now,
                        )
                    else:
                        if state != SemanticNoteState.PENDING.value:
                            conn.execute(
                                "DELETE FROM note_semantic_work WHERE owner_user_id=? "
                                "AND dataset_id=? AND kind='index_note' AND generation_id=? "
                                "AND note_id=? AND NOT ("
                                "claim_state='claimed' AND COALESCE(error_code,'')=?)",
                                (
                                    self.owner_user_id,
                                    dataset,
                                    generation_id,
                                    seed.note_id,
                                    self._VECTOR_SIDE_EFFECT_IN_PROGRESS,
                                ),
                            )
                updated_generation = conn.execute(
                    "UPDATE note_semantic_generations SET expected_note_count=?,"
                    "expected_chunk_count=? WHERE owner_user_id=? AND dataset_id=? "
                    "AND id=? AND state='staging' AND root_job_id=? AND "
                    "configuration_revision=?",
                    (
                        len(normalized),
                        sum(seed.planned_chunk_count for seed, _state in normalized),
                        self.owner_user_id,
                        dataset,
                        generation_id,
                        generation_fencing_token,
                        expected_configuration_revision,
                    ),
                )
                if updated_generation.rowcount != 1:
                    raise _SemanticCASMiss
                self._refresh_generation_counts_locked(
                    conn,
                    dataset=dataset,
                    generation_id=generation_id,
                )
        except _SemanticCASMiss:
            return False
        return True

    def claim_work_batch(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        kind: SemanticWorkKind | str,
        limit: int,
        now: datetime,
    ) -> tuple[SemanticWorkItem, ...]:
        """Claim a deterministic bounded work batch with per-row CAS tokens."""

        dataset = self._scope(dataset_id)
        work_kind = SemanticWorkKind(str(getattr(kind, "value", kind)))
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("notes_semantic_work_claim_limit_invalid")
        timestamp = self._timestamp(now)
        claimed: list[SemanticWorkItem] = []
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            query = (
                "SELECT id FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                "AND generation_id=? AND kind=? AND claim_state IN ('pending','failed') "
                "AND attempt_count < ? AND next_eligible_at<=? ORDER BY next_eligible_at,id LIMIT ?"
            )
            if self.is_postgres:
                query += " FOR UPDATE SKIP LOCKED"
            rows = conn.execute(
                query,
                (
                    self.owner_user_id,
                    dataset,
                    generation_id,
                    work_kind.value,
                    self._MAX_WORK_ATTEMPTS,
                    timestamp,
                    limit,
                ),
            ).fetchall()
            for row in rows:
                claim_token = str(uuid.uuid4())
                cursor = conn.execute(
                    "UPDATE note_semantic_work SET claim_state='claimed',claim_token=?,"
                    "claimed_at=?,updated_at=? WHERE owner_user_id=? AND dataset_id=? AND id=? "
                    "AND generation_id=? AND kind=? AND claim_state IN ('pending','failed')",
                    (
                        claim_token,
                        timestamp,
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        str(row["id"]),
                        generation_id,
                        work_kind.value,
                    ),
                )
                if cursor.rowcount != 1:
                    continue
                value = conn.execute(
                    "SELECT * FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                    "AND id=? AND claim_token=?",
                    (self.owner_user_id, dataset, str(row["id"]), claim_token),
                ).fetchone()
                if value is not None:
                    claimed.append(self._work_from_row(value))
        return tuple(claimed)

    def release_work_claim(
        self,
        *,
        dataset_id: str,
        work_id: str,
        claim_token: str,
        fencing_token: str,
        now: datetime,
    ) -> bool:
        """Release only the exact unprocessed claim without consuming an attempt."""

        dataset = self._scope(dataset_id)
        if not claim_token:
            return False
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            self._serialize_dataset_mutation(conn, dataset)
            work = conn.execute(
                "SELECT kind,generation_id,note_id,dirty_generation,attempt_count,error_code "
                "FROM note_semantic_work WHERE owner_user_id=? AND "
                "dataset_id=? AND id=? AND claim_state='claimed' AND claim_token=? "
                "AND fencing_token=?",
                (
                    self.owner_user_id,
                    dataset,
                    work_id,
                    claim_token,
                    fencing_token,
                ),
            )
            work_row = work.fetchone()
            if work_row is None:
                return False
            record = self._record(work_row)
            disabled = conn.execute(
                "SELECT 1 FROM note_semantic_index_configs WHERE owner_user_id=? "
                "AND dataset_id=? AND desired_state='disabled'",
                (self.owner_user_id, dataset),
            ).fetchone()
            if disabled is not None:
                cursor = conn.execute(
                    "DELETE FROM note_semantic_work WHERE owner_user_id=? AND "
                    "dataset_id=? AND id=? AND kind IN ('index_note','delete_note_vectors') "
                    "AND claim_state='claimed' AND claim_token=? AND fencing_token=?",
                    (
                        self.owner_user_id,
                        dataset,
                        work_id,
                        claim_token,
                        fencing_token,
                    ),
                )
                if cursor.rowcount == 1:
                    return True
            if (
                str(record["kind"]) == SemanticWorkKind.INDEX_NOTE.value
                and str(record.get("error_code") or "")
                == self._VECTOR_SIDE_EFFECT_IN_PROGRESS
            ):
                target = self._preferred_pending_note_target_locked(
                    conn,
                    dataset=dataset,
                    note_id=str(record["note_id"]),
                )
                if target is None:
                    deleted = conn.execute(
                        "DELETE FROM note_semantic_work WHERE owner_user_id=? AND "
                        "dataset_id=? AND id=? AND kind='index_note' "
                        "AND claim_state='claimed' AND claim_token=? AND fencing_token=?",
                        (
                            self.owner_user_id,
                            dataset,
                            work_id,
                            claim_token,
                            fencing_token,
                        ),
                    )
                    return deleted.rowcount == 1
                latest_generation_id, latest_dirty_generation = target
                attempt_count = int(record["attempt_count"])
                if (
                    latest_generation_id != str(record["generation_id"])
                    or latest_dirty_generation != int(record["dirty_generation"])
                ):
                    attempt_count = 0
                rearmed = conn.execute(
                    "UPDATE note_semantic_work SET generation_id=?,dirty_generation=?,fencing_token=?,"
                    "claim_state='pending',attempt_count=?,next_eligible_at=?,claim_token=NULL,"
                    "claimed_at=NULL,error_code=NULL,updated_at=? WHERE owner_user_id=? AND "
                    "dataset_id=? AND id=? AND kind='index_note' AND claim_state='claimed' "
                    "AND claim_token=? AND fencing_token=?",
                    (
                        latest_generation_id,
                        latest_dirty_generation,
                        str(uuid.uuid4()),
                        attempt_count,
                        self._timestamp(now),
                        self._timestamp(now),
                        self.owner_user_id,
                        dataset,
                        work_id,
                        claim_token,
                        fencing_token,
                    ),
                )
                return rearmed.rowcount == 1
            cursor = conn.execute(
                "UPDATE note_semantic_work SET claim_state='pending',claim_token=NULL,"
                "claimed_at=NULL,error_code=NULL,updated_at=? WHERE owner_user_id=? AND "
                "dataset_id=? AND id=? AND claim_state='claimed' AND claim_token=? AND "
                "fencing_token=?",
                (
                    self._timestamp(now),
                    self.owner_user_id,
                    dataset,
                    work_id,
                    claim_token,
                    fencing_token,
                ),
            )
        return cursor.rowcount == 1

    def _preferred_pending_note_target_locked(
        self,
        conn: SemanticConnection,
        *,
        dataset: str,
        note_id: str,
    ) -> tuple[str, int] | None:
        """Return the lifecycle generation that currently owns pending Note work."""

        state = conn.execute(
            "SELECT n.generation_id,n.dirty_generation FROM "
            "note_semantic_index_configs c JOIN note_semantic_generations g ON "
            "g.owner_user_id=c.owner_user_id AND g.dataset_id=c.dataset_id "
            "JOIN note_semantic_note_state n ON n.owner_user_id=g.owner_user_id "
            "AND n.dataset_id=g.dataset_id AND n.generation_id=g.id WHERE "
            "c.owner_user_id=? AND c.dataset_id=? AND c.desired_state='enabled' "
            "AND g.state IN ('staging','active') AND n.note_id=? "
            "AND n.state='pending' ORDER BY "
            "CASE WHEN g.state='staging' THEN 0 ELSE 1 END LIMIT 1",
            (self.owner_user_id, dataset, note_id),
        ).fetchone()
        if state is None:
            return None
        record = self._record(state)
        return str(record["generation_id"]), int(record["dirty_generation"])

    def reclaim_expired_work_claims(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        kind: SemanticWorkKind | str,
        expired_before: datetime,
        limit: int,
        now: datetime,
    ) -> int:
        """Boundedly convert expired claims to deterministic retry attempts."""

        dataset = self._scope(dataset_id)
        work_kind = SemanticWorkKind(str(getattr(kind, "value", kind)))
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("notes_semantic_work_claim_limit_invalid")
        expired = self._timestamp(expired_before)
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            query = (
                "SELECT id,claim_token FROM note_semantic_work WHERE owner_user_id=? AND "
                "dataset_id=? AND generation_id=? AND kind=? AND claim_state='claimed' "
                "AND NOT (kind='index_note' AND COALESCE(error_code,'')=?) "
                "AND claimed_at<=? ORDER BY claimed_at,id LIMIT ?"
            )
            if self.is_postgres:
                query += " FOR UPDATE SKIP LOCKED"
            rows = conn.execute(
                query,
                (
                    self.owner_user_id,
                    dataset,
                    generation_id,
                    work_kind.value,
                    self._VECTOR_SIDE_EFFECT_IN_PROGRESS,
                    expired,
                    limit,
                ),
            ).fetchall()
            reclaimed = 0
            for row in rows:
                cursor = conn.execute(
                    "UPDATE note_semantic_work SET claim_state='failed',"
                    "attempt_count=attempt_count+1,next_eligible_at=?,claim_token=NULL,"
                    "claimed_at=NULL,error_code='claim_lease_expired',updated_at=? WHERE "
                    "owner_user_id=? AND dataset_id=? AND id=? AND claim_state='claimed' "
                    "AND claim_token=? AND attempt_count<?",
                    (
                        timestamp,
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        str(row["id"]),
                        str(row["claim_token"]),
                        self._MAX_WORK_ATTEMPTS,
                    ),
                )
                reclaimed += cursor.rowcount
        return reclaimed

    def fail_claimed_note(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        generation_fencing_token: str,
        expected_configuration_revision: int,
        work_id: str,
        claim_token: str,
        work_fencing_token: str,
        claimed_dirty_generation: int,
        note_id: str,
        error_code: str,
        now: datetime,
    ) -> bool:
        """Publish one documented Note-specific terminal failure by exact CAS."""

        dataset = self._scope(dataset_id)
        timestamp = self._timestamp(now)
        try:
            with self._db.transaction() as conn:
                self._set_scope(conn, dataset)
                generation = conn.execute(
                    "SELECT g.state FROM note_semantic_generations g JOIN "
                    "note_semantic_index_configs c ON c.owner_user_id=g.owner_user_id AND "
                    "c.dataset_id=g.dataset_id WHERE g.owner_user_id=? AND g.dataset_id=? "
                    "AND g.id=? AND g.root_job_id=? AND g.configuration_revision=? AND "
                    "c.configuration_revision=? AND c.desired_state='enabled' AND "
                    "g.state IN ('staging','active')",
                    (
                        self.owner_user_id,
                        dataset,
                        generation_id,
                        generation_fencing_token,
                        expected_configuration_revision,
                        expected_configuration_revision,
                    ),
                ).fetchone()
                work = conn.execute(
                    "SELECT id FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                    "AND id=? AND generation_id=? AND note_id=? AND kind='index_note' AND "
                    "dirty_generation=? AND fencing_token=? AND claim_state='claimed' AND "
                    "claim_token=?",
                    (
                        self.owner_user_id,
                        dataset,
                        work_id,
                        generation_id,
                        note_id,
                        claimed_dirty_generation,
                        work_fencing_token,
                        claim_token,
                    ),
                ).fetchone()
                state = conn.execute(
                    "SELECT state,dirty_generation FROM note_semantic_note_state WHERE "
                    "owner_user_id=? AND dataset_id=? AND generation_id=? AND note_id=?",
                    (self.owner_user_id, dataset, generation_id, note_id),
                ).fetchone()
                if (
                    generation is None
                    or work is None
                    or state is None
                    or str(state["state"]) != SemanticNoteState.PENDING.value
                    or int(state["dirty_generation"]) != claimed_dirty_generation
                ):
                    raise _SemanticCASMiss
                obsolete_ids = tuple(
                    str(row["chunk_id"])
                    for row in conn.execute(
                        "SELECT chunk_id FROM note_semantic_chunks WHERE owner_user_id=? "
                        "AND dataset_id=? AND generation_id=? AND note_id=?",
                        (self.owner_user_id, dataset, generation_id, note_id),
                    ).fetchall()
                )
                self._stage_obsolete_vectors_locked(
                    conn,
                    dataset=dataset,
                    generation_id=generation_id,
                    vector_ids=obsolete_ids,
                    source_kind="note_failure",
                    note_id=note_id,
                    dirty_generation=claimed_dirty_generation,
                    now=now,
                )
                failed = conn.execute(
                    "UPDATE note_semantic_note_state SET state='failed',manifest_hash=NULL,"
                    "error_code=?,published_at=? WHERE owner_user_id=? AND dataset_id=? AND "
                    "generation_id=? AND note_id=? AND state='pending' AND dirty_generation=?",
                    (
                        self._error_code(error_code),
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        generation_id,
                        note_id,
                        claimed_dirty_generation,
                    ),
                )
                completed = conn.execute(
                    "UPDATE note_semantic_work SET claim_state='completed',claim_token=NULL,"
                    "claimed_at=NULL,error_code=?,updated_at=? WHERE owner_user_id=? AND "
                    "dataset_id=? AND id=? AND claim_state='claimed' AND claim_token=? AND "
                    "fencing_token=?",
                    (
                        self._error_code(error_code),
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        work_id,
                        claim_token,
                        work_fencing_token,
                    ),
                )
                if failed.rowcount != 1 or completed.rowcount != 1:
                    raise _SemanticCASMiss
                self._increment_generation_counts_locked(
                    conn,
                    dataset=dataset,
                    generation_id=generation_id,
                    note_delta=1,
                    chunk_delta=0,
                )
        except _SemanticCASMiss:
            return False
        return True

    def get_note_state(
        self,
        dataset_id: str,
        generation_id: str,
        note_id: str,
    ) -> SemanticNoteRecord | None:
        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            row = conn.execute(
                "SELECT * FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? "
                "AND generation_id=? AND note_id=?",
                (self.owner_user_id, dataset, generation_id, note_id),
            ).fetchone()
        return None if row is None else self._note_from_row(row)

    def _projection_vector_ids(self, values: Sequence[str]) -> tuple[str, ...]:
        vector_ids = tuple(
            self._safe_token(value, field="projection_vector_id") for value in values
        )
        if len(vector_ids) > self._MAX_PROJECTION_VECTOR_IDS:
            raise ValueError("notes_semantic_projection_vector_limit_exceeded")
        return tuple(dict.fromkeys(vector_ids))

    def load_projection_chunks(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        vector_ids: Sequence[str],
    ) -> tuple[SemanticProjectionChunk, ...]:
        """Load current published chunks with live owner-scoped Note content."""

        dataset = self._scope(dataset_id)
        generation = self._safe_token(generation_id, field="generation_id")
        requested = self._projection_vector_ids(vector_ids)
        if not requested:
            return ()
        deleted_false: bool | int = False if self.is_postgres else 0
        records: dict[str, SemanticProjectionChunk] = {}
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            for start in range(0, len(requested), self._PROJECTION_READ_BATCH_SIZE):
                batch = requested[start : start + self._PROJECTION_READ_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    "SELECT c.chunk_id,c.owner_user_id,c.dataset_id,c.generation_id,"
                    "c.note_id,c.content_version,c.ordinal,c.field,c.start_offset,"
                    "c.end_offset,c.chunk_fingerprint,c.normalization_version,"
                    "c.chunker_version,s.content_fingerprint,n.title,n.content,"
                    "n.created_at,n.last_modified AS updated_at "
                    "FROM note_semantic_chunks c "
                    "JOIN note_semantic_note_state s ON s.owner_user_id=c.owner_user_id "
                    "AND s.dataset_id=c.dataset_id AND s.generation_id=c.generation_id "
                    "AND s.note_id=c.note_id "
                    "JOIN note_semantic_generations g ON g.owner_user_id=c.owner_user_id "
                    "AND g.dataset_id=c.dataset_id AND g.id=c.generation_id "
                    "JOIN note_semantic_index_configs cfg ON cfg.owner_user_id=c.owner_user_id "
                    "AND cfg.dataset_id=c.dataset_id "
                    "JOIN notes n ON n.id=c.note_id AND n.client_id=c.owner_user_id "
                    "WHERE c.owner_user_id=? AND c.dataset_id=? AND c.generation_id=? "
                    "AND cfg.desired_state='enabled' AND cfg.active_generation_id=c.generation_id "
                    "AND g.state='active' AND s.state='indexed' "
                    "AND s.content_version=c.content_version AND n.version=s.content_version "
                    "AND n.deleted=? AND c.chunk_id IN ("  # nosec B608
                    + placeholders
                    + ") ORDER BY c.note_id,c.ordinal,c.chunk_id",
                    (
                        self.owner_user_id,
                        dataset,
                        generation,
                        deleted_false,
                        *batch,
                    ),
                ).fetchall()
                for row in rows:
                    value = self._record(row)
                    vector_id = str(value["chunk_id"])
                    records[vector_id] = SemanticProjectionChunk(
                        owner_user_id=str(value["owner_user_id"]),
                        dataset_id=str(value["dataset_id"]),
                        generation_id=str(value["generation_id"]),
                        vector_id=vector_id,
                        note_id=str(value["note_id"]),
                        content_version=int(value["content_version"]),
                        content_fingerprint=str(value["content_fingerprint"]),
                        title=str(value["title"]),
                        content=str(value["content"]),
                        created_at=value["created_at"],
                        updated_at=value["updated_at"],
                        ordinal=int(value["ordinal"]),
                        field=str(value["field"]),
                        start_offset=int(value["start_offset"]),
                        end_offset=int(value["end_offset"]),
                        chunk_fingerprint=str(value["chunk_fingerprint"]),
                        normalization_version=str(value["normalization_version"]),
                        chunker_version=str(value["chunker_version"]),
                    )
        return tuple(records[vector_id] for vector_id in requested if vector_id in records)

    def filter_projection_note_ids(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        note_ids: Sequence[str],
        tag: str | None = None,
        source: str | None = None,
        time_range_start: datetime | str | None = None,
        time_range_end: datetime | str | None = None,
        time_range_field: str = "updated_at",
    ) -> frozenset[str]:
        """Return current indexed Notes satisfying graph request filters."""

        dataset = self._scope(dataset_id)
        generation = self._safe_token(generation_id, field="generation_id")
        normalized_ids = tuple(dict.fromkeys(str(value).strip() for value in note_ids))
        if not normalized_ids:
            return frozenset()
        if len(normalized_ids) > self._MAX_PROJECTION_VECTOR_IDS:
            raise ValueError("notes_semantic_projection_note_limit_exceeded")
        if any(not value for value in normalized_ids):
            raise ValueError("notes_semantic_projection_note_id_invalid")

        clauses = [
            "s.owner_user_id=?",
            "s.dataset_id=?",
            "s.generation_id=?",
            "s.state='indexed'",
            "cfg.desired_state='enabled'",
            "cfg.active_generation_id=s.generation_id",
            "g.state='active'",
            "n.client_id=s.owner_user_id",
            "n.version=s.content_version",
            "n.deleted=?",
        ]
        deleted_false: bool | int = False if self.is_postgres else 0
        params: list[Any] = [
            self.owner_user_id,
            dataset,
            generation,
            deleted_false,
        ]
        normalized_tag = str(tag or "").strip()
        if normalized_tag.lower().startswith("tag:"):
            normalized_tag = normalized_tag[4:].strip()
        if normalized_tag:
            clauses.append(
                "EXISTS (SELECT 1 FROM note_keywords nk JOIN keywords k "
                "ON k.id=nk.keyword_id WHERE nk.note_id=n.id "
                "AND LOWER(k.keyword)=LOWER(?) AND k.deleted=?)"
            )
            params.extend((normalized_tag, deleted_false))

        normalized_source = str(source or "").strip()
        if normalized_source.lower().startswith("source:"):
            normalized_source = normalized_source[len("source:") :]
        if normalized_source:
            source_name, separator, external_ref = normalized_source.partition(":")
            if separator and external_ref:
                clauses.append(
                    "EXISTS (SELECT 1 FROM conversations conv "
                    "WHERE conv.id=n.conversation_id AND conv.source=? "
                    "AND conv.external_ref=?)"
                )
            else:
                clauses.append(
                    "EXISTS (SELECT 1 FROM conversations conv "
                    "WHERE conv.id=n.conversation_id AND conv.source=?)"
                )
            params.append(source_name.strip())
            if separator and external_ref:
                params.append(external_ref.strip())

        timestamp_column = (
            "n.created_at" if time_range_field == "created_at" else "n.last_modified"
        )
        if time_range_field not in {"created_at", "updated_at"}:
            raise ValueError("notes_semantic_projection_time_field_invalid")
        if time_range_start is not None:
            clauses.append(f"{timestamp_column}>=?")  # nosec B608
            params.append(time_range_start)
        if time_range_end is not None:
            clauses.append(f"{timestamp_column}<=?")  # nosec B608
            params.append(time_range_end)

        admitted: set[str] = set()
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            for start in range(0, len(normalized_ids), self._PROJECTION_READ_BATCH_SIZE):
                batch = normalized_ids[start : start + self._PROJECTION_READ_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    "SELECT DISTINCT n.id FROM note_semantic_note_state s "  # nosec B608
                    "JOIN note_semantic_generations g ON g.owner_user_id=s.owner_user_id "
                    "AND g.dataset_id=s.dataset_id AND g.id=s.generation_id "
                    "JOIN note_semantic_index_configs cfg ON cfg.owner_user_id=s.owner_user_id "
                    "AND cfg.dataset_id=s.dataset_id "
                    "JOIN notes n ON n.id=s.note_id WHERE "
                    + " AND ".join(clauses)
                    + " AND n.id IN ("
                    + placeholders
                    + ") ORDER BY n.id",  # nosec B608
                    (*params, *batch),
                ).fetchall()
                admitted.update(str(row["id"]) for row in rows)
        return frozenset(admitted)

    def publish_indexed_manifest(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        generation_id: str,
        generation_fencing_token: str,
        expected_configuration_revision: int,
        work_id: str,
        claim_token: str,
        work_fencing_token: str,
        claimed_dirty_generation: int,
        content_version: int,
        content_fingerprint: str,
        chunks: Sequence[SemanticChunkRecord],
        now: datetime,
    ) -> SemanticManifestPublication | None:
        """Replace one visible manifest and complete its exact work claim atomically."""

        if owner_user_id != self.owner_user_id or not chunks:
            return None
        dataset = self._scope(dataset_id)
        fingerprint = self._digest(content_fingerprint, field="content_fingerprint")
        chunk_rows = tuple(chunks)
        vector_ids = tuple(chunk.chunk_id for chunk in chunk_rows)
        if len(vector_ids) != len(set(vector_ids)):
            raise ValueError("notes_semantic_manifest_vector_ids_duplicate")
        for ordinal, chunk in enumerate(chunk_rows):
            if (
                chunk.generation_id != generation_id
                or chunk.note_id != chunk_rows[0].note_id
                or chunk.content_version != content_version
                or chunk.ordinal != ordinal
                or chunk.field not in {"title", "content"}
                or chunk.start_offset < 0
                or chunk.end_offset <= chunk.start_offset
            ):
                raise ValueError("notes_semantic_manifest_invalid")
            self._digest(chunk.chunk_fingerprint, field="chunk_fingerprint")
        note_id = chunk_rows[0].note_id
        note_manifest_hash = self._manifest_digest(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "ordinal": chunk.ordinal,
                    "field": chunk.field,
                    "start": chunk.start_offset,
                    "end": chunk.end_offset,
                    "fingerprint": chunk.chunk_fingerprint,
                }
                for chunk in chunk_rows
            ]
        )
        timestamp = self._timestamp(now)
        try:
            with self._db.transaction() as conn:
                self._set_scope(conn, dataset)
                generation = conn.execute(
                    "SELECT g.state FROM note_semantic_generations g "
                    "JOIN note_semantic_index_configs c ON c.owner_user_id=g.owner_user_id "
                    "AND c.dataset_id=g.dataset_id WHERE g.owner_user_id=? "
                    "AND g.dataset_id=? AND g.id=? AND g.root_job_id=? "
                    "AND g.configuration_revision=? AND c.configuration_revision=? "
                    "AND c.desired_state='enabled' AND g.state IN ('staging','active')",
                    (
                        self.owner_user_id,
                        dataset,
                        generation_id,
                        generation_fencing_token,
                        expected_configuration_revision,
                        expected_configuration_revision,
                    ),
                ).fetchone()
                if generation is None:
                    raise _SemanticCASMiss
                work = conn.execute(
                    "SELECT id FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                    "AND id=? AND generation_id=? AND note_id=? AND kind='index_note' "
                    "AND dirty_generation=? AND fencing_token=? AND claim_state='claimed' "
                    "AND claim_token=?",
                    (
                        self.owner_user_id,
                        dataset,
                        work_id,
                        generation_id,
                        note_id,
                        claimed_dirty_generation,
                        work_fencing_token,
                        claim_token,
                    ),
                ).fetchone()
                if work is None:
                    raise _SemanticCASMiss
                state = conn.execute(
                    "SELECT dirty_generation,content_version,content_fingerprint,state "
                    "FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? "
                    "AND generation_id=? AND note_id=?",
                    (self.owner_user_id, dataset, generation_id, note_id),
                ).fetchone()
                if (
                    state is None
                    or int(state["dirty_generation"]) != claimed_dirty_generation
                    or int(state["content_version"]) != content_version
                    or str(state["content_fingerprint"]) != fingerprint
                    or str(state["state"]) != SemanticNoteState.PENDING.value
                ):
                    raise _SemanticCASMiss
                old_vector_ids = tuple(
                    str(row["chunk_id"])
                    for row in conn.execute(
                        "SELECT chunk_id FROM note_semantic_chunks WHERE owner_user_id=? "
                        "AND dataset_id=? AND generation_id=? AND note_id=? ORDER BY ordinal,chunk_id",
                        (self.owner_user_id, dataset, generation_id, note_id),
                    ).fetchall()
                )
                new_id_set = set(vector_ids)
                old_only = tuple(
                    vector_id
                    for vector_id in old_vector_ids
                    if vector_id not in new_id_set
                )
                self._stage_obsolete_vectors_locked(
                    conn,
                    dataset=dataset,
                    generation_id=generation_id,
                    vector_ids=old_only,
                    source_kind="manifest_replace",
                    note_id=note_id,
                    dirty_generation=claimed_dirty_generation,
                    now=now,
                )
                conn.execute(
                    "DELETE FROM note_semantic_chunks WHERE owner_user_id=? AND dataset_id=? "
                    "AND generation_id=? AND note_id=?",
                    (self.owner_user_id, dataset, generation_id, note_id),
                )
                for chunk in chunk_rows:
                    cleanup_state = conn.execute(
                        "SELECT claim_state FROM note_semantic_obsolete_vectors WHERE "
                        "owner_user_id=? AND dataset_id=? AND generation_id=? AND vector_id=?",
                        (
                            self.owner_user_id,
                            dataset,
                            generation_id,
                            chunk.chunk_id,
                        ),
                    ).fetchone()
                    if (
                        cleanup_state is not None
                        and str(cleanup_state["claim_state"]) == "claimed"
                    ):
                        raise _SemanticCASMiss
                    conn.execute(
                        "INSERT INTO note_semantic_chunks(chunk_id,owner_user_id,dataset_id,"
                        "generation_id,note_id,content_version,ordinal,field,start_offset,end_offset,"
                        "chunk_fingerprint,normalization_version,chunker_version) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            chunk.chunk_id,
                            self.owner_user_id,
                            dataset,
                            generation_id,
                            note_id,
                            content_version,
                            chunk.ordinal,
                            chunk.field,
                            chunk.start_offset,
                            chunk.end_offset,
                            chunk.chunk_fingerprint,
                            chunk.normalization_version,
                            chunk.chunker_version,
                        ),
                    )
                    conn.execute(
                        "DELETE FROM note_semantic_obsolete_vectors WHERE owner_user_id=? "
                        "AND dataset_id=? AND generation_id=? AND vector_id=?",
                        (
                            self.owner_user_id,
                            dataset,
                            generation_id,
                            chunk.chunk_id,
                        ),
                    )
                updated = conn.execute(
                    "UPDATE note_semantic_note_state SET state='indexed',chunk_count=?,"
                    "manifest_hash=?,error_code=NULL,published_at=? WHERE owner_user_id=? "
                    "AND dataset_id=? AND generation_id=? AND note_id=? AND dirty_generation=? "
                    "AND content_version=? AND state='pending'",
                    (
                        len(chunk_rows),
                        note_manifest_hash,
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        generation_id,
                        note_id,
                        claimed_dirty_generation,
                        content_version,
                    ),
                )
                if updated.rowcount != 1:
                    raise _SemanticCASMiss
                pending_target = self._preferred_pending_note_target_locked(
                    conn,
                    dataset=dataset,
                    note_id=note_id,
                )
                if pending_target is None:
                    completed = conn.execute(
                        "UPDATE note_semantic_work SET claim_state='completed',claim_token=NULL,"
                        "claimed_at=NULL,error_code=NULL,updated_at=? WHERE owner_user_id=? "
                        "AND dataset_id=? AND id=? AND fencing_token=? AND claim_token=? "
                        "AND claim_state='claimed'",
                        (
                            timestamp,
                            self.owner_user_id,
                            dataset,
                            work_id,
                            work_fencing_token,
                            claim_token,
                        ),
                    )
                else:
                    target_generation_id, target_dirty_generation = pending_target
                    completed = conn.execute(
                        "UPDATE note_semantic_work SET generation_id=?,dirty_generation=?,"
                        "fencing_token=?,claim_state='pending',attempt_count=0,"
                        "next_eligible_at=?,claim_token=NULL,claimed_at=NULL,error_code=NULL,"
                        "updated_at=? WHERE owner_user_id=? AND dataset_id=? AND id=? "
                        "AND fencing_token=? AND claim_token=? AND claim_state='claimed'",
                        (
                            target_generation_id,
                            target_dirty_generation,
                            str(uuid.uuid4()),
                            timestamp,
                            timestamp,
                            self.owner_user_id,
                            dataset,
                            work_id,
                            work_fencing_token,
                            claim_token,
                        ),
                    )
                if completed.rowcount != 1:
                    raise _SemanticCASMiss
                if str(generation["state"]) == SemanticGenerationState.ACTIVE.value:
                    conn.execute(
                        "UPDATE note_semantic_index_configs SET semantic_index_revision="
                        "semantic_index_revision+1,updated_at=? WHERE owner_user_id=? AND dataset_id=?",
                        (timestamp, self.owner_user_id, dataset),
                    )
                self._increment_generation_counts_locked(
                    conn,
                    dataset=dataset,
                    generation_id=generation_id,
                    note_delta=1,
                    chunk_delta=len(chunk_rows),
                )
        except _SemanticCASMiss:
            return None
        return SemanticManifestPublication(
            note_id=note_id,
            generation_id=generation_id,
            old_vector_ids=old_only,
            new_vector_ids=vector_ids,
            dirty_generation=claimed_dirty_generation,
            manifest_hash=note_manifest_hash,
        )

    def list_visible_vector_ids(
        self,
        dataset_id: str,
        generation_id: str,
        note_id: str | None = None,
    ) -> tuple[str, ...]:
        dataset = self._scope(dataset_id)
        query = (
            "SELECT c.chunk_id FROM note_semantic_chunks c "
            "JOIN note_semantic_note_state n "
            "ON n.owner_user_id=c.owner_user_id AND n.dataset_id=c.dataset_id "
            "AND n.generation_id=c.generation_id AND n.note_id=c.note_id "
            "WHERE c.owner_user_id=? AND c.dataset_id=? AND c.generation_id=? "
            "AND n.state='indexed' AND n.content_version=c.content_version"
        )
        params: tuple[Any, ...] = (self.owner_user_id, dataset, generation_id)
        if note_id is not None:
            query += " AND c.note_id=?"
            params += (note_id,)
        query += " ORDER BY c.note_id,c.ordinal,c.chunk_id"
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            # The only optional query fragment above is a fixed literal.
            rows = conn.execute(
                query,  # nosec B608
                params,
            ).fetchall()
        return tuple(str(row["chunk_id"]) for row in rows)

    def get_generation_integrity(
        self,
        dataset_id: str,
        generation_id: str,
    ) -> SemanticGenerationIntegrity:
        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            self._refresh_generation_counts_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
            )
            generation = conn.execute(
                "SELECT * FROM note_semantic_generations WHERE owner_user_id=? AND dataset_id=? AND id=?",
                (self.owner_user_id, dataset, generation_id),
            ).fetchone()
            if generation is None:
                raise SemanticIndexingError("notes_semantic_generation_missing")
            counts = conn.execute(
                "SELECT COUNT(*) AS total,"
                "SUM(CASE WHEN state='indexed' THEN 1 ELSE 0 END) AS indexed,"
                "SUM(CASE WHEN state='excluded' THEN 1 ELSE 0 END) AS excluded,"
                "SUM(CASE WHEN state='failed' THEN 1 ELSE 0 END) AS failed,"
                "SUM(CASE WHEN state='pending' THEN 1 ELSE 0 END) AS pending,"
                "SUM(CASE WHEN state='tombstoned' THEN 1 ELSE 0 END) AS tombstoned,"
                "SUM(CASE WHEN state IN ('failed','excluded') THEN chunk_count ELSE 0 END) "
                "AS waived_chunks "
                "FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? AND generation_id=?",
                (self.owner_user_id, dataset, generation_id),
            ).fetchone()
            if str(generation["state"]) == SemanticGenerationState.ACTIVE.value:
                vector_ids = tuple(
                    str(row["chunk_id"])
                    for row in conn.execute(
                        "SELECT c.chunk_id FROM note_semantic_chunks c JOIN "
                        "note_semantic_note_state n ON n.owner_user_id=c.owner_user_id AND "
                        "n.dataset_id=c.dataset_id AND n.generation_id=c.generation_id AND "
                        "n.note_id=c.note_id WHERE c.owner_user_id=? AND c.dataset_id=? AND "
                        "c.generation_id=? AND n.state='indexed' AND "
                        "n.content_version=c.content_version ORDER BY c.note_id,c.ordinal,c.chunk_id",
                        (self.owner_user_id, dataset, generation_id),
                    ).fetchall()
                )
                manifest_hash = str(generation["manifest_hash"])
            else:
                manifest_hash, vector_ids = self._generation_manifest_locked(
                    conn,
                    dataset=dataset,
                    generation_id=generation_id,
                )
                conn.execute(
                    "UPDATE note_semantic_generations SET manifest_hash=? WHERE owner_user_id=? "
                    "AND dataset_id=? AND id=?",
                    (manifest_hash, self.owner_user_id, dataset, generation_id),
                )
        record = self._record(generation)
        dimensions = record["dimensions"]
        compatibility_hash = record["compatibility_hash"]
        root_job_id = record["root_job_id"]
        if dimensions is None or compatibility_hash is None or root_job_id is None:
            raise SemanticIndexingError("notes_semantic_generation_dimensions_unresolved")
        indexed = int(counts["indexed"] or 0)
        excluded = int(counts["excluded"] or 0)
        failed = int(counts["failed"] or 0)
        pending = int(counts["pending"] or 0)
        tombstoned = int(counts["tombstoned"] or 0)
        waived_chunks = int(counts["waived_chunks"] or 0)
        terminal = indexed + excluded + failed + tombstoned
        return SemanticGenerationIntegrity(
            generation_id=generation_id,
            generation_fencing_token=str(root_job_id),
            expected_note_count=int(record["expected_note_count"]),
            expected_chunk_count=int(record["expected_chunk_count"]),
            published_note_count=int(record["published_note_count"]),
            published_chunk_count=int(record["published_chunk_count"]),
            terminal_note_count=terminal,
            indexed_note_count=indexed,
            excluded_note_count=excluded,
            failed_note_count=failed,
            pending_note_count=pending,
            tombstoned_note_count=tombstoned,
            eligible_note_count=indexed + failed + pending,
            waived_chunk_count=waived_chunks,
            vector_ids=vector_ids,
            manifest_hash=manifest_hash,
            dimensions=int(dimensions),
            compatibility_hash=str(compatibility_hash),
            terminal_error_code=record["terminal_error_code"],
        )

    @staticmethod
    def assert_generation_activatable(integrity: SemanticGenerationIntegrity) -> None:
        """Apply the exact initial activation coverage and integrity policy."""

        if integrity.terminal_error_code is not None:
            raise SemanticIndexingError("notes_semantic_systemic_failure")
        if integrity.pending_note_count != 0:
            raise SemanticIndexingError("notes_semantic_snapshot_incomplete")
        if (
            integrity.expected_note_count != integrity.terminal_note_count
            or integrity.published_note_count != integrity.terminal_note_count
        ):
            raise SemanticIndexingError("notes_semantic_note_count_mismatch")
        if (
            integrity.expected_chunk_count
            != integrity.published_chunk_count + integrity.waived_chunk_count
            or len(integrity.vector_ids) != integrity.published_chunk_count
        ):
            raise SemanticIndexingError("notes_semantic_chunk_count_mismatch")
        if integrity.eligible_note_count > 0 and integrity.indexed_note_count == 0:
            raise SemanticIndexingError("notes_semantic_eligible_corpus_unindexed")

    def activate_generation_verified(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        expected_configuration_revision: int,
        generation_fencing_token: str,
        expected_manifest_hash: str,
        expected_vector_ids: Sequence[str],
        expected_dimensions: int,
        expected_compatibility_hash: str,
        publication_receipt: str,
        now: datetime,
    ) -> SemanticIndexConfig | None:
        """Atomically verify and activate one complete staging generation."""

        dataset = self._scope(dataset_id)
        receipt = self._safe_token(publication_receipt, field="publication_receipt")
        expected_hash = self._digest(expected_manifest_hash, field="manifest_hash")
        expected_ids = tuple(expected_vector_ids)
        timestamp = self._timestamp(now)
        try:
            with self._db.transaction() as conn:
                self._set_scope(conn, dataset)
                self._serialize_dataset_mutation(conn, dataset)
                generation_query = (
                    "SELECT * FROM note_semantic_generations WHERE owner_user_id=? "
                    "AND dataset_id=? AND id=? AND configuration_revision=? "
                    "AND state='staging' AND root_job_id=?"
                )
                config_query = (
                    "SELECT * FROM note_semantic_index_configs WHERE owner_user_id=? "
                    "AND dataset_id=? AND configuration_revision=? AND desired_state='enabled'"
                )
                if self.is_postgres:
                    generation_query += " FOR UPDATE"
                    config_query += " FOR UPDATE"
                generation_row = conn.execute(
                    generation_query,
                    (
                        self.owner_user_id,
                        dataset,
                        generation_id,
                        expected_configuration_revision,
                        generation_fencing_token,
                    ),
                ).fetchone()
                config = conn.execute(
                    config_query,
                    (self.owner_user_id, dataset, expected_configuration_revision),
                ).fetchone()
                if generation_row is None or config is None:
                    raise _SemanticCASMiss
                generation_root_job_id = self._record(generation_row)["root_job_id"]
                if (
                    generation_root_job_id is not None
                    and self._has_unexpired_cancellation_intent_locked(
                        conn,
                        dataset_id=dataset,
                        root_job_id=str(generation_root_job_id),
                        now=now,
                    )
                ):
                    raise SemanticIndexingError("notes_semantic_run_cancelled")
                snapshot_query = (
                    "SELECT s.note_id,s.content_version,n.version,n.deleted,n.client_id "
                    "FROM note_semantic_note_state s JOIN notes n ON n.id=s.note_id "
                    "WHERE s.owner_user_id=? AND s.dataset_id=? AND s.generation_id=? "
                    "ORDER BY s.note_id"
                )
                if self.is_postgres:
                    snapshot_query += " FOR UPDATE OF s,n"
                snapshot_rows = conn.execute(
                    snapshot_query,
                    (self.owner_user_id, dataset, generation_id),
                ).fetchall()
                generation_record = self._record(generation_row)
                if len(snapshot_rows) != int(generation_record["expected_note_count"]):
                    raise _SemanticCASMiss
                if any(
                    str(row["client_id"]) != self.owner_user_id
                    or bool(row["deleted"])
                    or int(row["version"]) != int(row["content_version"])
                    for row in snapshot_rows
                ):
                    raise _SemanticCASMiss
                self._refresh_generation_counts_locked(
                    conn,
                    dataset=dataset,
                    generation_id=generation_id,
                )
                refreshed_generation = conn.execute(
                    "SELECT * FROM note_semantic_generations WHERE owner_user_id=? "
                    "AND dataset_id=? AND id=?",
                    (self.owner_user_id, dataset, generation_id),
                ).fetchone()
                if refreshed_generation is None:
                    raise _SemanticCASMiss
                generation_record = self._record(refreshed_generation)
                if (
                    int(generation_record["dimensions"] or 0) != expected_dimensions
                    or str(generation_record["compatibility_hash"] or "")
                    != expected_compatibility_hash
                    or int(config["dimensions"] or 0) != expected_dimensions
                    or str(config["compatibility_hash"] or "") != expected_compatibility_hash
                ):
                    raise SemanticIndexingError("notes_semantic_generation_identity_mismatch")
                counts = conn.execute(
                    "SELECT SUM(CASE WHEN state='indexed' THEN 1 ELSE 0 END) AS indexed,"
                    "SUM(CASE WHEN state='excluded' THEN 1 ELSE 0 END) AS excluded,"
                    "SUM(CASE WHEN state='failed' THEN 1 ELSE 0 END) AS failed,"
                    "SUM(CASE WHEN state='pending' THEN 1 ELSE 0 END) AS pending,"
                    "SUM(CASE WHEN state='tombstoned' THEN 1 ELSE 0 END) AS tombstoned,"
                    "SUM(CASE WHEN state IN ('failed','excluded') THEN chunk_count ELSE 0 END) "
                    "AS waived_chunks "
                    "FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? "
                    "AND generation_id=?",
                    (self.owner_user_id, dataset, generation_id),
                ).fetchone()
                manifest_hash, vector_ids = self._generation_manifest_locked(
                    conn,
                    dataset=dataset,
                    generation_id=generation_id,
                )
                indexed = int(counts["indexed"] or 0)
                excluded = int(counts["excluded"] or 0)
                failed = int(counts["failed"] or 0)
                pending = int(counts["pending"] or 0)
                tombstoned = int(counts["tombstoned"] or 0)
                integrity = SemanticGenerationIntegrity(
                    generation_id=generation_id,
                    generation_fencing_token=generation_fencing_token,
                    expected_note_count=int(generation_record["expected_note_count"]),
                    expected_chunk_count=int(generation_record["expected_chunk_count"]),
                    published_note_count=int(generation_record["published_note_count"]),
                    published_chunk_count=int(generation_record["published_chunk_count"]),
                    terminal_note_count=indexed + excluded + failed + tombstoned,
                    indexed_note_count=indexed,
                    excluded_note_count=excluded,
                    failed_note_count=failed,
                    pending_note_count=pending,
                    tombstoned_note_count=tombstoned,
                    eligible_note_count=indexed + failed + pending,
                    waived_chunk_count=int(counts["waived_chunks"] or 0),
                    vector_ids=vector_ids,
                    manifest_hash=manifest_hash,
                    dimensions=expected_dimensions,
                    compatibility_hash=expected_compatibility_hash,
                    terminal_error_code=generation_record["terminal_error_code"],
                )
                self.assert_generation_activatable(integrity)
                if manifest_hash != expected_hash:
                    raise SemanticIndexingError("notes_semantic_manifest_hash_mismatch")
                if vector_ids != expected_ids:
                    raise SemanticIndexingError("notes_semantic_vector_ids_mismatch")
                previous_generation_id = config["active_generation_id"]
                if previous_generation_id is not None:
                    retired = conn.execute(
                        "UPDATE note_semantic_generations SET state='retired',retired_at=? "
                        "WHERE owner_user_id=? AND dataset_id=? AND id=? AND state='active'",
                        (timestamp, self.owner_user_id, dataset, previous_generation_id),
                    )
                    if retired.rowcount != 1:
                        raise _SemanticCASMiss
                activated = conn.execute(
                    "UPDATE note_semantic_generations SET state='active',publication_receipt=?,"
                    "published_at=?,manifest_hash=?,configuration_revision=? WHERE "
                    "owner_user_id=? AND dataset_id=? "
                    "AND id=? AND state='staging' AND root_job_id=?",
                    (
                        receipt,
                        timestamp,
                        manifest_hash,
                        expected_configuration_revision + 1,
                        self.owner_user_id,
                        dataset,
                        generation_id,
                        generation_fencing_token,
                    ),
                )
                if activated.rowcount != 1:
                    raise _SemanticCASMiss
                updated = conn.execute(
                    "UPDATE note_semantic_index_configs SET active_generation_id=?,"
                    "configuration_revision=configuration_revision+1,semantic_index_revision="
                    "semantic_index_revision+1,updated_at=? WHERE owner_user_id=? AND dataset_id=? "
                    "AND configuration_revision=? AND desired_state='enabled' AND dimensions=? "
                    "AND compatibility_hash=?",
                    (
                        generation_id,
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        expected_configuration_revision,
                        expected_dimensions,
                        expected_compatibility_hash,
                    ),
                )
                if updated.rowcount != 1:
                    raise _SemanticCASMiss
                if previous_generation_id is not None:
                    self._enqueue_work(
                        conn,
                        dataset=dataset,
                        kind=SemanticWorkKind.DELETE_GENERATION,
                        note_id=None,
                        generation_id=str(previous_generation_id),
                        dirty_generation=None,
                        now=now,
                    )
                row = conn.execute(
                    "SELECT * FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=?",
                    (self.owner_user_id, dataset),
                ).fetchone()
        except _SemanticCASMiss:
            return None
        return self._config_from_row(row)

    def publish_note_tombstone(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        generation_id: str,
        generation_fencing_token: str,
        expected_configuration_revision: int,
        work_id: str,
        claim_token: str,
        work_fencing_token: str,
        claimed_dirty_generation: int,
        note_id: str,
        now: datetime,
    ) -> SemanticManifestPublication | None:
        if owner_user_id != self.owner_user_id:
            return None
        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            generation = conn.execute(
                "SELECT g.state FROM note_semantic_generations g "
                "JOIN note_semantic_index_configs c ON c.owner_user_id=g.owner_user_id "
                "AND c.dataset_id=g.dataset_id WHERE g.owner_user_id=? AND g.dataset_id=? "
                "AND g.id=? AND g.root_job_id=? AND g.configuration_revision=? "
                "AND c.configuration_revision=? AND c.desired_state='enabled' "
                "AND g.state IN ('staging','active')",
                (
                    self.owner_user_id,
                    dataset,
                    generation_id,
                    generation_fencing_token,
                    expected_configuration_revision,
                    expected_configuration_revision,
                ),
            ).fetchone()
            work = conn.execute(
                "SELECT id FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                "AND id=? AND generation_id=? AND note_id=? AND kind='delete_note_vectors' "
                "AND dirty_generation=? AND fencing_token=? AND claim_state='claimed' AND claim_token=?",
                (
                    self.owner_user_id,
                    dataset,
                    work_id,
                    generation_id,
                    note_id,
                    claimed_dirty_generation,
                    work_fencing_token,
                    claim_token,
                ),
            ).fetchone()
            state = conn.execute(
                "SELECT state,dirty_generation FROM note_semantic_note_state WHERE owner_user_id=? "
                "AND dataset_id=? AND generation_id=? AND note_id=?",
                (self.owner_user_id, dataset, generation_id, note_id),
            ).fetchone()
            if (
                generation is None
                or work is None
                or state is None
                or str(state["state"]) != SemanticNoteState.TOMBSTONED.value
                or int(state["dirty_generation"]) != claimed_dirty_generation
            ):
                return None
            old_ids = tuple(
                str(row["chunk_id"])
                for row in conn.execute(
                    "SELECT chunk_id FROM note_semantic_chunks WHERE owner_user_id=? AND dataset_id=? "
                    "AND generation_id=? AND note_id=? ORDER BY ordinal,chunk_id",
                    (self.owner_user_id, dataset, generation_id, note_id),
                ).fetchall()
            )
            self._stage_obsolete_vectors_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
                vector_ids=old_ids,
                source_kind="tombstone",
                note_id=note_id,
                dirty_generation=claimed_dirty_generation,
                now=now,
            )
            completed = conn.execute(
                "UPDATE note_semantic_work SET claim_state='completed',claim_token=NULL,"
                "claimed_at=NULL,error_code=NULL,updated_at=? WHERE owner_user_id=? "
                "AND dataset_id=? AND id=? AND fencing_token=? AND claim_token=? "
                "AND claim_state='claimed'",
                (
                    self._timestamp(now),
                    self.owner_user_id,
                    dataset,
                    work_id,
                    work_fencing_token,
                    claim_token,
                ),
            )
            if completed.rowcount != 1:
                return None
        return SemanticManifestPublication(
            note_id=note_id,
            generation_id=generation_id,
            old_vector_ids=old_ids,
            new_vector_ids=(),
            dirty_generation=claimed_dirty_generation,
            manifest_hash=None,
        )

    def authorize_obsolete_vector_cleanup(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        note_id: str,
        dirty_generation: int,
        vector_ids: Sequence[str],
    ) -> bool:
        dataset = self._scope(dataset_id)
        requested = tuple(vector_ids)
        if len(requested) != len(set(requested)):
            return False
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            state = conn.execute(
                "SELECT dirty_generation FROM note_semantic_note_state WHERE owner_user_id=? "
                "AND dataset_id=? AND generation_id=? AND note_id=?",
                (self.owner_user_id, dataset, generation_id, note_id),
            ).fetchone()
            if state is None or int(state["dirty_generation"]) < dirty_generation:
                return False
            visible = {
                str(row["chunk_id"])
                for row in conn.execute(
                    "SELECT c.chunk_id FROM note_semantic_chunks c "
                    "JOIN note_semantic_note_state n "
                    "ON n.owner_user_id=c.owner_user_id AND n.dataset_id=c.dataset_id "
                    "AND n.generation_id=c.generation_id AND n.note_id=c.note_id "
                    "WHERE c.owner_user_id=? AND c.dataset_id=? AND c.generation_id=? "
                    "AND c.note_id=? AND n.state='indexed' "
                    "AND n.content_version=c.content_version",
                    (self.owner_user_id, dataset, generation_id, note_id),
                ).fetchall()
            }
        return not visible.intersection(requested)

    def complete_obsolete_vector_cleanup(
        self,
        *,
        dataset_id: str,
        generation_id: str,
        note_id: str,
        dirty_generation: int,
        vector_ids: Sequence[str],
        now: datetime,
    ) -> bool:
        dataset = self._scope(dataset_id)
        requested = tuple(vector_ids)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            state = conn.execute(
                "SELECT dirty_generation,state FROM note_semantic_note_state WHERE owner_user_id=? "
                "AND dataset_id=? AND generation_id=? AND note_id=?",
                (self.owner_user_id, dataset, generation_id, note_id),
            ).fetchone()
            if state is None or int(state["dirty_generation"]) < dirty_generation:
                return False
            visible_rows = conn.execute(
                "SELECT c.chunk_id FROM note_semantic_chunks c JOIN note_semantic_note_state n "
                "ON n.owner_user_id=c.owner_user_id AND n.dataset_id=c.dataset_id "
                "AND n.generation_id=c.generation_id AND n.note_id=c.note_id "
                "WHERE c.owner_user_id=? AND c.dataset_id=? AND c.generation_id=? "
                "AND c.note_id=? AND n.state='indexed' AND n.content_version=c.content_version",
                (self.owner_user_id, dataset, generation_id, note_id),
            ).fetchall()
            if {str(row["chunk_id"]) for row in visible_rows}.intersection(requested):
                return False
            for vector_id in requested:
                conn.execute(
                    "DELETE FROM note_semantic_chunks WHERE owner_user_id=? AND dataset_id=? "
                    "AND generation_id=? AND note_id=? AND chunk_id=?",
                    (self.owner_user_id, dataset, generation_id, note_id, vector_id),
                )
            self._refresh_generation_counts_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
            )
        return True

    def _generation_cleanup_authorized_locked(
        self,
        conn: Any,
        *,
        dataset: str,
        work_id: str,
        generation_id: str,
        claim_token: str,
        fencing_token: str,
    ) -> bool:
        row = conn.execute(
            "SELECT w.id FROM note_semantic_work w JOIN note_semantic_generations g "
            "ON g.owner_user_id=w.owner_user_id AND g.dataset_id=w.dataset_id "
            "AND g.id=w.generation_id JOIN note_semantic_index_configs c "
            "ON c.owner_user_id=w.owner_user_id AND c.dataset_id=w.dataset_id "
            "WHERE w.owner_user_id=? AND w.dataset_id=? AND w.id=? "
            "AND w.kind='delete_generation' AND w.generation_id=? "
            "AND w.claim_state='claimed' AND w.claim_token=? AND w.fencing_token=? "
            "AND g.state IN ('retired','failed','deleting') "
            "AND (c.active_generation_id IS NULL OR c.active_generation_id<>w.generation_id) "
            "AND NOT EXISTS (SELECT 1 FROM note_semantic_work writer WHERE "
            "writer.owner_user_id=w.owner_user_id AND writer.dataset_id=w.dataset_id "
            "AND writer.generation_id=w.generation_id "
            "AND writer.kind IN ('index_note','delete_note_vectors') "
            "AND writer.claim_state='claimed')",
            (
                self.owner_user_id,
                dataset,
                work_id,
                generation_id,
                claim_token,
                fencing_token,
            ),
        ).fetchone()
        return row is not None

    def authorize_generation_cleanup(
        self,
        *,
        dataset_id: str,
        work_id: str,
        generation_id: str,
        claim_token: str,
        fencing_token: str,
    ) -> bool:
        """Fence delayed generation cleanup away from the current active generation."""

        dataset = self._scope(dataset_id)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            return self._generation_cleanup_authorized_locked(
                conn,
                dataset=dataset,
                work_id=work_id,
                generation_id=generation_id,
                claim_token=claim_token,
                fencing_token=fencing_token,
            )

    def complete_generation_cleanup(
        self,
        *,
        dataset_id: str,
        work_id: str,
        generation_id: str,
        claim_token: str,
        fencing_token: str,
        now: datetime,
    ) -> bool:
        dataset = self._scope(dataset_id)
        timestamp = self._timestamp(now)
        try:
            with self._db.transaction() as conn:
                self._set_scope(conn, dataset)
                config = conn.execute(
                    "SELECT active_generation_id FROM note_semantic_index_configs "
                    "WHERE owner_user_id=? AND dataset_id=?",
                    (self.owner_user_id, dataset),
                ).fetchone()
                if config is None or config["active_generation_id"] == generation_id:
                    raise _SemanticCASMiss
                work = conn.execute(
                    "UPDATE note_semantic_work SET claim_state='completed',claim_token=NULL,"
                    "claimed_at=NULL,error_code=NULL,updated_at=? WHERE owner_user_id=? "
                    "AND dataset_id=? AND id=? AND kind='delete_generation' AND generation_id=? "
                    "AND claim_state='claimed' AND claim_token=? AND fencing_token=?",
                    (
                        timestamp,
                        self.owner_user_id,
                        dataset,
                        work_id,
                        generation_id,
                        claim_token,
                        fencing_token,
                    ),
                )
                if work.rowcount != 1:
                    raise _SemanticCASMiss
                generation = conn.execute(
                    "UPDATE note_semantic_generations SET state='deleting',deleted_at=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=? "
                    "AND state IN ('retired','failed','deleting')",
                    (timestamp, self.owner_user_id, dataset, generation_id),
                )
                if generation.rowcount != 1:
                    raise _SemanticCASMiss
        except _SemanticCASMiss:
            return False
        return True
