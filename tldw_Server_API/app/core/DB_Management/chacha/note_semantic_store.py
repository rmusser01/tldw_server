"""Owner-bound SQL seam for Notes semantic-index persistence."""

from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from ..ChaChaNotes_DB import BackendConnectionWrapper, BackendType
from .note_semantic_models import (
    SemanticDesiredState,
    SemanticDimensionState,
    SemanticGeneration,
    SemanticGenerationState,
    SemanticIndexConfig,
    SemanticNoteRecord,
    SemanticNoteState,
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
        with self._db.transaction() as conn:
            self._set_scope(conn, dataset)
            cursor = conn.execute(
                "UPDATE note_semantic_index_configs SET desired_state=?, configuration_revision=configuration_revision+1, "  # nosec B608
                "enabled_at=CASE WHEN ?='enabled' THEN ? ELSE enabled_at END, "
                "disabled_at=CASE WHEN ?='disabled' THEN ? ELSE disabled_at END, updated_at=? WHERE " + where,
                (desired_state.value, desired_state.value, timestamp, desired_state.value, timestamp, timestamp, *params),
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
            config = conn.execute(
                "SELECT configuration_revision,dimension_state,dimensions,compatibility_hash "
                "FROM note_semantic_index_configs WHERE owner_user_id=? AND dataset_id=?",
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
