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
from urllib.parse import urlsplit

from ..ChaChaNotes_DB import BackendConnectionWrapper, BackendType
from .note_semantic_models import (
    SemanticChunkRecord,
    SemanticDesiredState,
    SemanticDimensionState,
    SemanticGeneration,
    SemanticGenerationIntegrity,
    SemanticGenerationState,
    SemanticIndexConfig,
    SemanticIndexingError,
    SemanticManifestPublication,
    SemanticNoteRecord,
    SemanticNoteState,
    SemanticSnapshotSeed,
    SemanticWorkClaimState,
    SemanticWorkItem,
    SemanticWorkKind,
)

if TYPE_CHECKING:
    from ..ChaChaNotes_DB import CharactersRAGDB


SemanticConnection = sqlite3.Connection | BackendConnectionWrapper


class _SemanticCASMiss(Exception):
    """Abort and roll back an atomic semantic compare-and-swap."""


class NoteSemanticStore:
    """Own semantic configuration, generation, manifest, and cleanup SQL."""

    _SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
    _ERROR_CODE = re.compile(r"^[a-z][a-z0-9_:-]{0,127}$")
    _DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
    _MAX_WORK_ATTEMPTS = 5

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

    @classmethod
    def _safe_token(cls, value: str, *, field: str) -> str:
        normalized = str(value).strip()
        if cls._SAFE_TOKEN.fullmatch(normalized) is None:
            raise ValueError(f"notes_semantic_{field}_invalid")
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
        parsed = urlsplit(str(value))
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("notes_semantic_endpoint_origin_display_invalid")
        origin = f"{parsed.scheme}://{parsed.hostname.lower()}"
        if parsed.port is not None:
            origin = f"{origin}:{parsed.port}"
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
        manifest_hash, _ = self._generation_manifest_locked(
            conn,
            dataset=dataset,
            generation_id=generation_id,
        )
        conn.execute(
            "UPDATE note_semantic_generations SET expected_note_count=?,expected_chunk_count=?,"
            "published_note_count=?,published_chunk_count=?,manifest_hash=? "
            "WHERE owner_user_id=? AND dataset_id=? AND id=?",
            (
                int(counts["note_count"] or 0),
                int(counts["indexed_chunks"] or 0),
                int(counts["terminal_count"] or 0),
                int(published_chunks["chunk_count"] or 0),
                manifest_hash,
                self.owner_user_id,
                dataset,
                generation_id,
            ),
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

    def create_configuration(
        self, *, dataset_id: str, capability_revision: str, disclosure_hash: str, provider: str,
        model: str, endpoint_origin_revision: str, endpoint_origin_display: str, data_boundary: str,
        vector_backend: str, storage_boundary: str, storage_label: str, normalization_version: str,
        chunker_version: str, now: datetime,
    ) -> SemanticIndexConfig:
        dataset = self._scope(dataset_id)
        timestamp = self._timestamp(now)
        params = (
            self.owner_user_id, dataset, self._safe_token(capability_revision, field="capability_revision"),
            self._safe_token(disclosure_hash, field="disclosure_hash"), self._safe_token(provider, field="provider"),
            self._safe_token(model, field="model"), self._safe_token(endpoint_origin_revision, field="endpoint_origin_revision"),
            self._endpoint_origin_display(endpoint_origin_display), self._safe_token(data_boundary, field="data_boundary"),
            self._safe_token(vector_backend, field="vector_backend"), self._safe_token(storage_boundary, field="storage_boundary"),
            self._safe_token(storage_label.replace(" ", "_"), field="storage_label"),
            self._safe_token(normalization_version, field="normalization_version"),
            self._safe_token(chunker_version, field="chunker_version"), timestamp,
            timestamp,
        )
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            conn.execute(
                """
                INSERT INTO note_semantic_index_configs(
                  owner_user_id,dataset_id,desired_state,configuration_revision,semantic_index_revision,
                  capability_revision,disclosure_hash,provider,model,endpoint_origin_revision,
                  endpoint_origin_display,data_boundary,vector_backend,storage_boundary,storage_label,
                  metric,dimension_state,normalization_version,chunker_version,consented_at,updated_at
                ) VALUES (?,?, 'disabled',1,0,?,?,?,?,?,?,?,?,?,?, 'cosine','pending',?,?,?,?)
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

    def disable_configuration(
        self, *, dataset_id: str, expected_configuration_revision: int, now: datetime,
    ) -> SemanticIndexConfig | None:
        return self._transition_configuration(
            dataset=self._scope(dataset_id), expected_configuration_revision=expected_configuration_revision,
            capability_revision=None, desired_state=SemanticDesiredState.DISABLED, now=now,
        )

    def create_generation(
        self, *, dataset_id: str, configuration_revision: int, compatibility_hash: str | None,
        dimension_state: SemanticDimensionState, dimensions: int | None, root_job_id: str | None,
        now: datetime,
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
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            if self.is_postgres:
                config_query = (
                    "SELECT configuration_revision,dimension_state,dimensions,compatibility_hash "
                    "FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=? FOR UPDATE"
                )
            else:
                config_query = (
                    "SELECT configuration_revision,dimension_state,dimensions,compatibility_hash "
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
            ):
                raise ValueError("notes_semantic_generation_identity_mismatch")
            conn.execute(
                """
                INSERT INTO note_semantic_generations(
                  id,owner_user_id,dataset_id,configuration_revision,state,compatibility_hash,
                  dimension_state,dimensions,root_job_id,created_at
                ) VALUES (?,?,?,?, 'staging',?,?,?,?,?)
                """,
                (
                    generation_id, self.owner_user_id, dataset, configuration_revision,
                    resolved_compatibility_hash, dimension_state.value,
                    dimensions, None if root_job_id is None else self._safe_token(root_job_id, field="root_job_id"), timestamp,
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

    def resolve_generation_dimensions(
        self, *, dataset_id: str, generation_id: str, expected_configuration_revision: int,
        dimensions: int, compatibility_hash: str, now: datetime,
    ) -> SemanticGeneration | None:
        dataset = self._scope(dataset_id)
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("notes_semantic_dimensions_invalid")
        final_hash = self._safe_token(compatibility_hash, field="compatibility_hash")
        timestamp = self._timestamp(now)
        next_revision = expected_configuration_revision + 1
        try:
            with self._db.transaction() as conn:
                self._set_scope(conn, dataset)
                config_cursor = conn.execute(
                    "UPDATE note_semantic_index_configs SET dimension_state='resolved', dimensions=?, "
                    "compatibility_hash=?, configuration_revision=?, updated_at=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND configuration_revision=? "
                    "AND desired_state='enabled' AND dimension_state='pending' "
                    "AND dimensions IS NULL AND compatibility_hash IS NULL",
                    (
                        dimensions, final_hash, next_revision, timestamp, self.owner_user_id,
                        dataset, expected_configuration_revision,
                    ),
                )
                if config_cursor.rowcount != 1:
                    raise _SemanticCASMiss
                generation_cursor = conn.execute(
                    "UPDATE note_semantic_generations SET dimension_state='resolved', dimensions=?, "
                    "compatibility_hash=?, configuration_revision=? "
                    "WHERE owner_user_id=? AND dataset_id=? AND id=? AND configuration_revision=? "
                    "AND state='staging' AND dimension_state='pending' "
                    "AND dimensions IS NULL AND compatibility_hash IS NULL",
                    (
                        dimensions, final_hash, next_revision, self.owner_user_id, dataset,
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
                    "SELECT id FROM note_semantic_generations WHERE owner_user_id=? AND dataset_id=? "
                    "AND id=? AND configuration_revision=? AND state='staging' "
                    "AND dimension_state='resolved' AND dimensions=? AND compatibility_hash=?",
                    (
                        self.owner_user_id, dataset, generation_id,
                        expected_configuration_revision, config_dimensions, config_hash,
                    ),
                ).fetchone()
                if candidate is None:
                    raise _SemanticCASMiss
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
                """,
                (
                    str(uuid.uuid4()), self.owner_user_id, dataset, SemanticWorkKind.INDEX_NOTE.value,
                    note_id, generation_id, self.owner_user_id, dataset, generation_id, note_id,
                    str(uuid.uuid4()), timestamp, timestamp, timestamp,
                ),
            )
            row = conn.execute(
                "SELECT * FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? AND generation_id=? AND note_id=?",
                (self.owner_user_id, dataset, generation_id, note_id),
            ).fetchone()
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
        tx.execute(
            "DELETE FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
            "AND kind='index_note' AND note_id=?",
            (self.owner_user_id, dataset, note_id),
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
        return True

    def tombstone_note(
        self, *, dataset_id: str, generation_id: str, note_id: str, content_version: int,
        dirty_generation: int, now: datetime,
    ) -> SemanticNoteRecord | None:
        dataset = self._scope(dataset_id)
        timestamp = self._timestamp(now)
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
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
                "AND kind='index_note' AND note_id=?",
                (self.owner_user_id, dataset, note_id),
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
                "claim_token=NULL, claimed_at=NULL, error_code=NULL, updated_at=excluded.updated_at",
                params,
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
                    "SELECT g.id FROM note_semantic_generations g "
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
                    conn.execute(
                        "DELETE FROM note_semantic_work WHERE owner_user_id=? AND dataset_id=? "
                        "AND generation_id=? AND note_id=? AND kind='index_note'",
                        (self.owner_user_id, dataset, generation_id, note_id),
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
                        unchanged = (
                            int(current["content_version"]) == seed.content_version
                            and str(current["content_fingerprint"]) == fingerprint
                            and str(current["state"]) == state
                            and int(current["chunk_count"]) == chunk_count
                            and current["error_code"] == error_code
                        )
                        dirty_generation = int(current["dirty_generation"])
                        changed = not unchanged
                        if not unchanged:
                            dirty_generation += 1
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
                                "AND note_id=?",
                                (
                                    self.owner_user_id,
                                    dataset,
                                    generation_id,
                                    seed.note_id,
                                ),
                            )
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
                conn.execute(
                    "DELETE FROM note_semantic_chunks WHERE owner_user_id=? AND dataset_id=? "
                    "AND generation_id=? AND note_id=?",
                    (self.owner_user_id, dataset, generation_id, note_id),
                )
                for chunk in chunk_rows:
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
                if completed.rowcount != 1:
                    raise _SemanticCASMiss
                if str(generation["state"]) == SemanticGenerationState.ACTIVE.value:
                    conn.execute(
                        "UPDATE note_semantic_index_configs SET semantic_index_revision="
                        "semantic_index_revision+1,updated_at=? WHERE owner_user_id=? AND dataset_id=?",
                        (timestamp, self.owner_user_id, dataset),
                    )
                self._refresh_generation_counts_locked(
                    conn,
                    dataset=dataset,
                    generation_id=generation_id,
                )
        except _SemanticCASMiss:
            return None
        old_only = tuple(vector_id for vector_id in old_vector_ids if vector_id not in set(vector_ids))
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
                "SUM(CASE WHEN state='tombstoned' THEN 1 ELSE 0 END) AS tombstoned "
                "FROM note_semantic_note_state WHERE owner_user_id=? AND dataset_id=? AND generation_id=?",
                (self.owner_user_id, dataset, generation_id),
            ).fetchone()
            manifest_hash, vector_ids = self._generation_manifest_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
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
            integrity.expected_chunk_count != integrity.published_chunk_count
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
                self._refresh_generation_counts_locked(
                    conn,
                    dataset=dataset,
                    generation_id=generation_id,
                )
                generation_row = conn.execute(
                    "SELECT * FROM note_semantic_generations WHERE owner_user_id=? AND dataset_id=? "
                    "AND id=? AND configuration_revision=? AND state='staging' AND root_job_id=?",
                    (
                        self.owner_user_id,
                        dataset,
                        generation_id,
                        expected_configuration_revision,
                        generation_fencing_token,
                    ),
                ).fetchone()
                config = conn.execute(
                    "SELECT * FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=? "
                    "AND configuration_revision=? AND desired_state='enabled'",
                    (self.owner_user_id, dataset, expected_configuration_revision),
                ).fetchone()
                if generation_row is None or config is None:
                    raise _SemanticCASMiss
                generation_record = self._record(generation_row)
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
                    "SUM(CASE WHEN state='tombstoned' THEN 1 ELSE 0 END) AS tombstoned "
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
                    "published_at=?,manifest_hash=? WHERE owner_user_id=? AND dataset_id=? "
                    "AND id=? AND state='staging' AND root_job_id=?",
                    (
                        receipt,
                        timestamp,
                        manifest_hash,
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
            self._refresh_generation_counts_locked(
                conn,
                dataset=dataset,
                generation_id=generation_id,
            )
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
            row = conn.execute(
                "SELECT w.id FROM note_semantic_work w JOIN note_semantic_generations g "
                "ON g.owner_user_id=w.owner_user_id AND g.dataset_id=w.dataset_id "
                "AND g.id=w.generation_id JOIN note_semantic_index_configs c "
                "ON c.owner_user_id=w.owner_user_id AND c.dataset_id=w.dataset_id "
                "WHERE w.owner_user_id=? AND w.dataset_id=? AND w.id=? "
                "AND w.kind='delete_generation' AND w.generation_id=? "
                "AND w.claim_state='claimed' AND w.claim_token=? AND w.fencing_token=? "
                "AND g.state IN ('retired','failed','deleting') "
                "AND (c.active_generation_id IS NULL OR c.active_generation_id<>w.generation_id)",
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
