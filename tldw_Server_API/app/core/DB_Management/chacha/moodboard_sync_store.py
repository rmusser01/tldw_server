"""Tenant-scoped moodboard and Studio graph binding and bootstrap storage."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendConnectionWrapper,
    BackendType,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_contract import (
    SYNC_ENVELOPE_MAX_BYTES,
    NotesMoodboardStudioContractError,
    notes_moodboard_note_object_hash,
    notes_moodboard_object_hash,
    notes_studio_document_object_hash,
    parse_notes_moodboard_note_tombstone_v1,
    parse_notes_moodboard_tombstone_v1,
    parse_notes_studio_document_tombstone_v1,
    placement_object_id,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


GraphConnection: TypeAlias = sqlite3.Connection | BackendConnectionWrapper


class MoodboardSyncStore:
    """Scoped product reads and one-way graph authority binding."""

    _LOCAL_UNBOUND = "local-unbound"
    _POSTGRES_BIND_LOCK_TABLES = (
        "note_task_scope_authority",
        "notes",
        "moodboards",
        "moodboard_notes",
        "note_studio_documents",
    )

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def _execute(
        self,
        conn: GraphConnection,
        query: str,
        params: tuple[Any, ...] = (),
    ) -> Any:
        prepared_query, prepared_params = self._db._prepare_backend_statement(query, params)
        return conn.execute(prepared_query, prepared_params or ())

    def _set_postgres_dataset_scope(
        self,
        conn: GraphConnection,
        dataset_id: str,
    ) -> None:
        if self._db.backend_type == BackendType.POSTGRESQL:
            self._execute(
                conn,
                "SELECT set_config('app.current_dataset_id', ?, true)",
                (dataset_id,),
            )

    @staticmethod
    def _validated_scope(owner_user_id: str, target_dataset_id: str) -> tuple[str, str]:
        owner = str(owner_user_id).strip()
        target = str(target_dataset_id).strip()
        if not owner or not target or target == MoodboardSyncStore._LOCAL_UNBOUND:
            raise InputError("A non-sentinel owner and target dataset are required.")  # noqa: TRY003
        return owner, target

    def _resolve_graph_dataset(
        self,
        *,
        owner_user_id: str,
        flag: str,
        conn: GraphConnection | None,
    ) -> str:
        owner = str(owner_user_id).strip()
        if not owner:
            raise InputError("Graph owner cannot be empty.")  # noqa: TRY003
        if not self._db._supports_notes_moodboard_studio_v61():
            return self._LOCAL_UNBOUND
        if self._db.backend_type == BackendType.POSTGRESQL and owner != str(self._db.client_id):
            raise ConflictError(
                "Graph compatibility scope is unavailable.",
                entity="notes",
                entity_id=owner,
            )  # noqa: TRY003
        bound_value: bool | int = (
            True if self._db.backend_type == BackendType.POSTGRESQL else 1
        )
        query = (
            "SELECT owner_user_id,dataset_id FROM note_task_scope_authority "
            f"WHERE owner_user_id = ? AND {flag} = ? LIMIT 2"  # nosec B608 - fixed flag names.
        )
        cursor = (
            self._db.execute_query(query, (owner, bound_value))
            if conn is None
            else self._execute(conn, query, (owner, bound_value))
        )
        rows = cursor.fetchall()
        if not rows:
            return self._LOCAL_UNBOUND
        if len(rows) != 1:
            raise ConflictError(
                "Graph compatibility scope is inconsistent.",
                entity="notes",
                entity_id=owner,
            )  # noqa: TRY003
        row_owner = str(rows[0]["owner_user_id"]).strip()
        dataset = str(rows[0]["dataset_id"]).strip()
        if row_owner != owner or not dataset or dataset == self._LOCAL_UNBOUND:
            raise ConflictError(
                "Graph compatibility scope is inconsistent.",
                entity="notes",
                entity_id=owner,
            )  # noqa: TRY003
        return dataset

    def resolve_moodboard_compatibility_dataset_id(
        self,
        *,
        owner_user_id: str,
        conn: GraphConnection | None = None,
    ) -> str:
        """Resolve the immutable dataset scope for the owner's moodboard graph.

        Args:
            owner_user_id: Owner whose compatibility scope should be resolved.
            conn: Optional caller-owned transaction connection.

        Returns:
            The bound dataset ID, or ``local-unbound`` before binding or on a
            schema that predates moodboard synchronization.

        Raises:
            InputError: If the owner ID is empty or invalid.
            ConflictError: If stored graph authority is inconsistent.
        """
        return self._resolve_graph_dataset(
            owner_user_id=owner_user_id,
            flag="moodboard_graph_bound",
            conn=conn,
        )

    def resolve_studio_compatibility_dataset_id(
        self,
        *,
        owner_user_id: str,
        conn: GraphConnection | None = None,
    ) -> str:
        """Resolve the immutable dataset scope for the owner's Studio graph.

        Args:
            owner_user_id: Owner whose compatibility scope should be resolved.
            conn: Optional caller-owned transaction connection.

        Returns:
            The bound dataset ID, or ``local-unbound`` before binding or on a
            schema that predates Studio synchronization.

        Raises:
            InputError: If the owner ID is empty or invalid.
            ConflictError: If stored graph authority is inconsistent.
        """
        return self._resolve_graph_dataset(
            owner_user_id=owner_user_id,
            flag="studio_graph_bound",
            conn=conn,
        )

    def _require_bootstrap_scope(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        flag: str,
    ) -> tuple[str, str]:
        owner, dataset = self._validated_scope(owner_user_id, dataset_id)
        if self._resolve_graph_dataset(
            owner_user_id=owner,
            flag=flag,
            conn=None,
        ) != dataset:
            raise ConflictError(
                "Graph is not bound to the requested bootstrap scope.",
                entity="notes",
                entity_id=dataset,
            )  # noqa: TRY003
        return owner, dataset

    def _bind_graph(
        self,
        *,
        owner_user_id: str,
        target_dataset_id: str,
        flag: str,
        tables: tuple[tuple[str, str], ...],
        rekey: Callable[[GraphConnection, str, str], None],
        prove: Callable[[GraphConnection, str, str], None],
        conn: GraphConnection | None,
    ) -> dict[str, int]:
        owner, target = self._validated_scope(owner_user_id, target_dataset_id)
        if not self._db._supports_notes_moodboard_studio_v61():
            raise ConflictError(
                "Graph binding is unavailable for this backend schema.",
                entity="notes",
                entity_id=target,
            )  # noqa: TRY003
        postgres = self._db.backend_type == BackendType.POSTGRESQL
        if postgres and owner != str(self._db.client_id):
            raise ConflictError(
                "Graph owner does not match the authenticated PostgreSQL client.",
                entity="notes",
                entity_id=owner,
            )  # noqa: TRY003

        def snapshot(transaction_conn: GraphConnection, dataset: str) -> dict[str, tuple[int, str]]:
            result: dict[str, tuple[int, str]] = {}
            for table, ordering in tables:
                lock = " FOR UPDATE" if postgres else ""
                rows = self._execute(
                    transaction_conn,
                    f"SELECT * FROM {table} WHERE owner_user_id=? AND dataset_id=? "  # nosec B608
                    f"ORDER BY {ordering}{lock}",  # nosec B608
                    (owner, dataset),
                ).fetchall()
                canonical = [
                    {key: value for key, value in dict(row).items() if key != "dataset_id"}
                    for row in rows
                ]
                result[table] = (
                    len(canonical),
                    self._db._note_task_v60_hash(self._db._note_task_v60_json_safe(canonical)),
                )
            return result

        def counts(state: dict[str, tuple[int, str]]) -> dict[str, int]:
            return {table: count for table, (count, _digest) in state.items()}

        def prepare_postgres_bind(transaction_conn: GraphConnection) -> None:
            if not postgres:
                return
            version = self._db._get_schema_version_postgres(transaction_conn, lock=True)
            if version != self._db._POSTGRES_SCHEMA_VERSION:
                raise ConflictError(
                    "Graph binding requires the current PostgreSQL schema.",
                    entity="notes",
                    entity_id=target,
                )  # noqa: TRY003
            self._execute(
                transaction_conn,
                "LOCK TABLE "
                + ",".join(self._POSTGRES_BIND_LOCK_TABLES)
                + " IN ACCESS EXCLUSIVE MODE",
            )
            self._db._verify_notes_moodboard_studio_schema_postgres(transaction_conn)
            for table, _ordering in tables:
                self._execute(
                    transaction_conn,
                    f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY",  # nosec B608 - fixed graph table names.
                )

        def finish_postgres_bind(transaction_conn: GraphConnection) -> None:
            if not postgres:
                return
            for table, _ordering in tables:
                self._execute(
                    transaction_conn,
                    f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",  # nosec B608 - fixed graph table names.
                )
            self._db._verify_notes_moodboard_studio_schema_postgres(transaction_conn)

        def bind(transaction_conn: GraphConnection) -> dict[str, int]:
            prepare_postgres_bind(transaction_conn)
            lock = " FOR UPDATE" if postgres else ""
            authority_rows = self._execute(
                transaction_conn,
                "SELECT owner_user_id,dataset_id,task_graph_bound,moodboard_graph_bound,studio_graph_bound "
                f"FROM note_task_scope_authority WHERE owner_user_id=?{lock}",  # nosec B608
                (owner,),
            ).fetchall()
            if len(authority_rows) > 1:
                raise ConflictError("Graph authority is inconsistent.", entity="notes", entity_id=target)  # noqa: TRY003
            authority = dict(authority_rows[0]) if authority_rows else None
            if authority is not None:
                authority_owner = str(authority["owner_user_id"]).strip()
                authority_dataset = str(authority["dataset_id"]).strip()
                if authority_owner != owner or not authority_dataset or authority_dataset == self._LOCAL_UNBOUND:
                    raise ConflictError("Graph authority is inconsistent.", entity="notes", entity_id=target)  # noqa: TRY003
                if authority_dataset != target:
                    raise ConflictError("Graph dataset binding is immutable.", entity="notes", entity_id=target)  # noqa: TRY003

            datasets: set[str] = set()
            for table, _ordering in tables:
                rows = self._execute(
                    transaction_conn,
                    f"SELECT DISTINCT dataset_id FROM {table} WHERE owner_user_id=?",  # nosec B608
                    (owner,),
                ).fetchall()
                datasets.update(str(row["dataset_id"]).strip() for row in rows)
            if "" in datasets or datasets - {self._LOCAL_UNBOUND, target}:
                raise ConflictError("Graph contains an incompatible dataset scope.", entity="notes", entity_id=target)  # noqa: TRY003

            source = snapshot(transaction_conn, self._LOCAL_UNBOUND)
            target_state = snapshot(transaction_conn, target)
            if authority is not None and bool(authority[flag]):
                if any(counts(source).values()) or datasets - {target}:
                    raise ConflictError("Graph authority conflicts with product scope.", entity="notes", entity_id=target)  # noqa: TRY003
                prove(transaction_conn, owner, target)
                result = counts(target_state)
                finish_postgres_bind(transaction_conn)
                return result
            if any(counts(target_state).values()):
                raise ConflictError("Graph binding target collision.", entity="notes", entity_id=target)  # noqa: TRY003

            if any(counts(source).values()):
                prove(transaction_conn, owner, self._LOCAL_UNBOUND)
                rekey(transaction_conn, owner, target)
                remaining = snapshot(transaction_conn, self._LOCAL_UNBOUND)
                rebound = snapshot(transaction_conn, target)
                if any(counts(remaining).values()) or rebound != source:
                    raise ConflictError("Graph binding failed complete-set verification.", entity="notes", entity_id=target)  # noqa: TRY003
                prove(transaction_conn, owner, target)
            else:
                rebound = source

            if authority is None:
                false_value: bool | int = False if postgres else 0
                true_value: bool | int = True if postgres else 1
                graph_values = {
                    "task_graph_bound": false_value,
                    "moodboard_graph_bound": false_value,
                    "studio_graph_bound": false_value,
                }
                graph_values[flag] = true_value
                self._execute(
                    transaction_conn,
                    "INSERT INTO note_task_scope_authority("
                    "owner_user_id,dataset_id,task_graph_bound,moodboard_graph_bound,studio_graph_bound"
                    ") VALUES (?,?,?,?,?)",
                    (
                        owner,
                        target,
                        graph_values["task_graph_bound"],
                        graph_values["moodboard_graph_bound"],
                        graph_values["studio_graph_bound"],
                    ),
                )
            else:
                false_value = False if postgres else 0
                self._execute(
                    transaction_conn,
                    f"UPDATE note_task_scope_authority SET {flag}=? "  # nosec B608 - fixed flag names.
                    f"WHERE owner_user_id=? AND dataset_id=? AND {flag}=?",  # nosec B608
                    (True if postgres else 1, owner, target, false_value),
                )
            result = counts(rebound)
            finish_postgres_bind(transaction_conn)
            return result

        def bind_with_savepoint(transaction_conn: GraphConnection) -> dict[str, int]:
            if not postgres:
                return bind(transaction_conn)
            self._execute(transaction_conn, "SAVEPOINT bind_local_moodboard_studio_graph")
            try:
                result = bind(transaction_conn)
            except Exception:  # noqa: BLE001 - rollback must restore FORCE RLS/catalog on every failed bind.
                self._execute(transaction_conn, "ROLLBACK TO SAVEPOINT bind_local_moodboard_studio_graph")
                self._execute(transaction_conn, "RELEASE SAVEPOINT bind_local_moodboard_studio_graph")
                self._db._verify_notes_moodboard_studio_schema_postgres(transaction_conn)
                raise
            self._execute(transaction_conn, "RELEASE SAVEPOINT bind_local_moodboard_studio_graph")
            return result

        if conn is not None:
            return bind_with_savepoint(conn)
        with self._db.transaction() as transaction_conn:
            return bind_with_savepoint(transaction_conn)

    def _prove_moodboard_graph(
        self,
        conn: GraphConnection,
        *,
        owner: str,
        dataset: str,
    ) -> None:
        try:
            boards = self._execute(
                conn,
                "SELECT * FROM moodboards WHERE owner_user_id=? AND dataset_id=? ORDER BY id",
                (owner, dataset),
            ).fetchall()
            sync_ids = self._validate_moodboard_rows(
                conn,
                rows=boards,
                owner=owner,
                dataset=dataset,
            )
            placements = self._execute(
                conn,
                "SELECT * FROM moodboard_notes WHERE owner_user_id=? AND dataset_id=? "
                "ORDER BY moodboard_id,note_id",
                (owner, dataset),
            ).fetchall()
            self._validate_moodboard_placement_rows(
                placements,
                owner=owner,
                dataset=dataset,
                sync_ids=sync_ids,
            )
        except (NotesMoodboardStudioContractError, TypeError, ValueError) as exc:
            raise ConflictError(
                "Moodboard graph binding failed canonical readiness proof.",
                entity="moodboards",
            ) from exc  # noqa: TRY003

    def _validate_moodboard_rows(
        self,
        conn: GraphConnection,
        *,
        rows: list[Any],
        owner: str,
        dataset: str,
    ) -> dict[int, str]:
        sync_ids: dict[int, str] = {}
        collection_sync_ids: list[str] = []
        for raw in rows:
            row = dict(raw)
            if row["owner_user_id"] != owner or row["dataset_id"] != dataset:
                raise ValueError("moodboard scope mismatch")
            payload = parse_notes_moodboard_tombstone_v1(
                {
                    "moodboard_id": row["sync_id"],
                    "name": row["name"],
                    "description": row["description"],
                    "smart_rule": (
                        None
                        if row["smart_rule_json"] is None
                        else json.loads(row["smart_rule_json"])
                    ),
                    "canvas": json.loads(row["canvas_json"]),
                }
            )
            expected = notes_moodboard_object_hash(
                payload,
                revision=int(row["canonical_revision"]),
                deleted=bool(row["deleted"]),
            )
            if row["source_diagnostic_code"] is not None or row["canonical_hash"] != expected:
                raise ValueError("moodboard lineage mismatch")
            if payload.smart_rule is not None:
                collection_sync_ids.extend(payload.smart_rule.collection_sync_ids)
            sync_ids[int(row["id"])] = str(row["sync_id"])
        self._db._prove_moodboard_collection_sync_ids_v61(
            conn,
            owner_user_id=owner,
            collection_sync_ids=collection_sync_ids,
        )
        return sync_ids

    @staticmethod
    def _validate_moodboard_placement_rows(
        rows: list[Any],
        *,
        owner: str,
        dataset: str,
        sync_ids: dict[int, str] | None = None,
    ) -> None:
        for raw in rows:
            row = dict(raw)
            if row["owner_user_id"] != owner or row["dataset_id"] != dataset:
                raise ValueError("placement scope mismatch")
            if sync_ids is None and row.get("_note_exists") != 1:
                raise ValueError("placement note missing")
            moodboard_sync_id = (
                None if sync_ids is None else sync_ids.get(int(row["moodboard_id"]))
            )
            if moodboard_sync_id is None:
                moodboard_sync_id = row.get("moodboard_sync_id")
            if not moodboard_sync_id:
                raise ValueError("placement parent missing")
            payload = parse_notes_moodboard_note_tombstone_v1(
                {
                    "moodboard_id": moodboard_sync_id,
                    "note_id": row["note_id"],
                    "x": row["x"],
                    "y": row["y"],
                    "width": row["width"],
                    "height": row["height"],
                    "order_index": row["order_index"],
                    "display": json.loads(row["display_json"]),
                }
            )
            expected = notes_moodboard_note_object_hash(
                payload,
                revision=int(row["canonical_revision"]),
                deleted=bool(row["deleted"]),
            )
            if (
                row["source_diagnostic_code"] is not None
                or row["placement_id"] != placement_object_id(payload)
                or row["canonical_hash"] != expected
            ):
                raise ValueError("placement lineage mismatch")

    def _prove_studio_graph(
        self,
        conn: GraphConnection,
        *,
        owner: str,
        dataset: str,
    ) -> None:
        try:
            rows = self._execute(
                conn,
                "SELECT * FROM note_studio_documents "
                "WHERE owner_user_id=? AND dataset_id=? ORDER BY note_id",
                (owner, dataset),
            ).fetchall()
            self._validate_studio_rows(
                conn,
                rows=rows,
                owner=owner,
                dataset=dataset,
            )
        except (NotesMoodboardStudioContractError, TypeError, ValueError) as exc:
            raise ConflictError(
                "Studio graph binding failed canonical readiness proof.",
                entity="note_studio_documents",
            ) from exc  # noqa: TRY003

    def _fetch_notes_for_studio_rows(
        self,
        conn: GraphConnection,
        *,
        owner: str,
        note_ids: set[str],
    ) -> dict[str, dict[str, Any]]:
        notes: dict[str, dict[str, Any]] = {}
        ordered = sorted(note_ids)
        for offset in range(0, len(ordered), 400):
            batch = ordered[offset : offset + 400]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            rows = self._execute(
                conn,
                "SELECT * FROM notes WHERE client_id=? "
                f"AND id IN ({placeholders})",  # nosec B608 - placeholders only.
                (owner, *batch),
            ).fetchall()
            notes.update((str(row["id"]), dict(row)) for row in rows)
        return notes

    def _validate_studio_rows(
        self,
        conn: GraphConnection,
        *,
        rows: list[Any],
        owner: str,
        dataset: str,
    ) -> None:
        deserialized: list[tuple[dict[str, Any], Any]] = []
        note_ids: set[str] = set()
        for raw in rows:
            row = dict(raw)
            if row["owner_user_id"] != owner or row["dataset_id"] != dataset:
                raise ValueError("Studio scope mismatch")
            if row["source_diagnostic_code"] is not None:
                raise ValueError("Studio diagnostic blocker")
            document = self._db._deserialize_row_fields(
                row,
                ["payload_json", "diagram_manifest_json", "accepted_provenance_json"],
            )
            if document is None:
                raise ValueError("Studio row unavailable")
            parsed = parse_notes_studio_document_tombstone_v1(
                self._db.note_store._studio_document_mapping(document)
            )
            note_ids.add(str(row["note_id"]))
            if parsed.source_note_id is not None:
                note_ids.add(str(parsed.source_note_id))
            deserialized.append((row, parsed))
        notes = self._fetch_notes_for_studio_rows(
            conn,
            owner=owner,
            note_ids=note_ids,
        )
        for row, parsed in deserialized:
            parent = notes.get(str(row["note_id"]))
            if parent is None:
                raise ValueError("Studio parent unavailable")
            parent_revision, parent_hash = self._db.note_store._notes_note_head(parent)
            if parsed.note_revision > parent_revision:
                raise ValueError("Studio parent head mismatch")
            if parsed.note_revision == parent_revision and (
                parsed.note_hash != parent_hash
                or parsed.companion_content_hash
                != self._db.note_store._normalized_text_hash(parent["content"])
            ):
                raise ValueError("Studio current parent head mismatch")
            provenance = parsed.accepted_provenance
            if parsed.source_note_id is None:
                if provenance.source_revision is not None or provenance.source_hash is not None:
                    raise ValueError("Studio source lineage mismatch")
            else:
                source = notes.get(str(parsed.source_note_id))
                if source is None:
                    raise ValueError("Studio source unavailable")
                source_revision, source_hash = self._db.note_store._notes_note_head(source)
                normalized_source = str(source["content"] or "").replace(
                    "\r\n", "\n"
                ).replace("\r", "\n")
                if provenance.source_revision is None or provenance.source_revision > source_revision:
                    raise ValueError("Studio source head mismatch")
                if provenance.source_revision == source_revision and (
                    provenance.source_hash != source_hash
                    or (
                        parsed.excerpt_snapshot is not None
                        and parsed.excerpt_snapshot not in normalized_source
                    )
                ):
                    raise ValueError("Studio current source head mismatch")
            deleted = bool(row["deleted"])
            if deleted != bool(parent["deleted"]):
                raise ValueError("Studio lifecycle differs from parent note")
            expected = notes_studio_document_object_hash(
                parsed,
                revision=int(row["canonical_revision"]),
                deleted=deleted,
            )
            if row["canonical_hash"] != expected:
                raise ValueError("Studio lineage mismatch")
            if self._db.note_store._studio_envelope_size(
                parsed.model_dump(mode="json"),
                revision=int(row["canonical_revision"]),
                deleted=deleted,
                object_hash=expected,
            ) > SYNC_ENVELOPE_MAX_BYTES:
                raise ValueError("Studio envelope is oversized")

    def bind_local_moodboard_graph_to_dataset(
        self,
        *,
        owner_user_id: str,
        target_dataset_id: str,
        conn: GraphConnection | None = None,
    ) -> dict[str, int]:
        """Bind and verify the complete local moodboard graph atomically.

        The operation rekeys only ``local-unbound`` moodboards and explicit
        placements. When ``conn`` is supplied, the caller owns the surrounding
        transaction; otherwise this method opens and commits one transaction.

        Args:
            owner_user_id: Owner of the local moodboard graph.
            target_dataset_id: Permanent personal dataset scope to bind.
            conn: Optional caller-owned transaction connection.

        Returns:
            Row counts keyed by each moodboard graph table name.

        Raises:
            InputError: If either scope identifier is invalid.
            ConflictError: If binding is unavailable, unauthorized, collides
                with existing scope, or fails canonical completeness proofs.
        """
        def prove(
            transaction_conn: GraphConnection,
            owner: str,
            dataset: str,
        ) -> None:
            blocked = self._execute(
                transaction_conn,
                "SELECT 1 FROM moodboards WHERE owner_user_id=? AND dataset_id=? "
                "AND source_diagnostic_code IS NOT NULL "
                "UNION ALL "
                "SELECT 1 FROM moodboard_notes WHERE owner_user_id=? AND dataset_id=? "
                "AND source_diagnostic_code IS NOT NULL LIMIT 1",
                (owner, dataset, owner, dataset),
            ).fetchone()
            if blocked is not None:
                raise ConflictError(
                    "Moodboard graph binding failed canonical readiness proof.",
                    entity="moodboards",
                )  # noqa: TRY003
            invalid = self._execute(
                transaction_conn,
                "SELECT 1 FROM moodboard_notes p "
                "LEFT JOIN moodboards b ON b.owner_user_id=p.owner_user_id "
                " AND b.dataset_id=p.dataset_id AND b.id=p.moodboard_id "
                "LEFT JOIN notes n ON n.client_id=p.owner_user_id AND n.id=p.note_id "
                "WHERE p.owner_user_id=? AND p.dataset_id=? "
                "AND (b.id IS NULL OR n.id IS NULL) LIMIT 1",
                (owner, dataset),
            ).fetchone()
            if invalid is not None:
                raise ConflictError("Moodboard graph binding failed parent proof.", entity="moodboards")  # noqa: TRY003
            self._prove_moodboard_graph(
                transaction_conn,
                owner=owner,
                dataset=dataset,
            )

        def rekey(transaction_conn: GraphConnection, owner: str, target: str) -> None:
            self._execute(
                transaction_conn,
                "UPDATE moodboards SET dataset_id=? WHERE owner_user_id=? AND dataset_id=?",
                (target, owner, self._LOCAL_UNBOUND),
            )

        return self._bind_graph(
            owner_user_id=owner_user_id,
            target_dataset_id=target_dataset_id,
            flag="moodboard_graph_bound",
            tables=(("moodboards", "sync_id"), ("moodboard_notes", "moodboard_id,note_id")),
            rekey=rekey,
            prove=prove,
            conn=conn,
        )

    def bind_local_studio_graph_to_dataset(
        self,
        *,
        owner_user_id: str,
        target_dataset_id: str,
        conn: GraphConnection | None = None,
    ) -> dict[str, int]:
        """Bind and verify the complete local Studio graph atomically.

        The operation rekeys only ``local-unbound`` Studio documents. When
        ``conn`` is supplied, the caller owns the surrounding transaction;
        otherwise this method opens and commits one transaction.

        Args:
            owner_user_id: Owner of the local Studio graph.
            target_dataset_id: Permanent personal dataset scope to bind.
            conn: Optional caller-owned transaction connection.

        Returns:
            Row counts keyed by the Studio graph table name.

        Raises:
            InputError: If either scope identifier is invalid.
            ConflictError: If binding is unavailable, unauthorized, collides
                with existing scope, or fails canonical completeness proofs.
        """
        def prove(
            transaction_conn: GraphConnection,
            owner: str,
            dataset: str,
        ) -> None:
            invalid = self._execute(
                transaction_conn,
                "SELECT 1 FROM note_studio_documents s "
                "LEFT JOIN notes n ON n.client_id=s.owner_user_id AND n.id=s.note_id "
                "LEFT JOIN notes source ON source.client_id=s.owner_user_id "
                "AND source.id=s.source_note_id "
                "WHERE s.owner_user_id=? AND s.dataset_id=? "
                "AND (n.id IS NULL OR s.source_diagnostic_code IS NOT NULL "
                "OR (s.source_note_id IS NOT NULL AND source.id IS NULL)) LIMIT 1",
                (owner, dataset),
            ).fetchone()
            if invalid is not None:
                raise ConflictError(
                    "Studio graph binding failed canonical readiness proof.",
                    entity="note_studio_documents",
                )  # noqa: TRY003
            self._prove_studio_graph(
                transaction_conn,
                owner=owner,
                dataset=dataset,
            )

        def rekey(transaction_conn: GraphConnection, owner: str, target: str) -> None:
            self._execute(
                transaction_conn,
                "UPDATE note_studio_documents SET dataset_id=? WHERE owner_user_id=? AND dataset_id=?",
                (target, owner, self._LOCAL_UNBOUND),
            )

        return self._bind_graph(
            owner_user_id=owner_user_id,
            target_dataset_id=target_dataset_id,
            flag="studio_graph_bound",
            tables=(("note_studio_documents", "note_id"),),
            rekey=rekey,
            prove=prove,
            conn=conn,
        )

    def page_moodboards_for_sync_bootstrap(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        after_sync_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Page canonically verified moodboards from an already-bound scope.

        Args:
            owner_user_id: Owner of the bound moodboard graph.
            dataset_id: Dataset to expose for bootstrap.
            after_sync_id: Exclusive moodboard sync-ID cursor.
            limit: Requested page size, clamped to the range 1 through 500.

        Returns:
            Moodboard rows ordered by sync ID.

        Raises:
            InputError: If a scope identifier is invalid.
            ConflictError: If the graph is unbound or canonical validation
                detects corrupt or inconsistent rows.
        """
        owner, dataset = self._require_bootstrap_scope(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            flag="moodboard_graph_bound",
        )
        with self._db.transaction() as conn:
            self._set_postgres_dataset_scope(conn, dataset)
            rows = self._execute(
                conn,
                "SELECT * FROM moodboards WHERE owner_user_id=? AND dataset_id=? "
                "AND sync_id>? ORDER BY sync_id LIMIT ?",
                (owner, dataset, after_sync_id or "", max(1, min(int(limit), 500))),
            ).fetchall()
            try:
                self._validate_moodboard_rows(
                    conn,
                    rows=rows,
                    owner=owner,
                    dataset=dataset,
                )
            except (NotesMoodboardStudioContractError, TypeError, ValueError) as exc:
                raise ConflictError(
                    "Moodboard graph binding failed canonical readiness proof.",
                    entity="moodboards",
                ) from exc  # noqa: TRY003
        return [dict(row) for row in rows]

    def page_moodboard_placements_for_sync_bootstrap(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        after_moodboard_sync_id: str | None = None,
        after_note_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Page verified explicit moodboard placements from a bound scope.

        Args:
            owner_user_id: Owner of the bound moodboard graph.
            dataset_id: Dataset to expose for bootstrap.
            after_moodboard_sync_id: Exclusive primary placement cursor.
            after_note_id: Exclusive secondary cursor within a moodboard.
            limit: Requested page size, clamped to the range 1 through 500.

        Returns:
            Placement rows ordered by moodboard sync ID and note ID.

        Raises:
            InputError: If a scope identifier is invalid.
            ConflictError: If the graph is unbound or canonical validation
                detects missing parents or inconsistent rows.
        """
        owner, dataset = self._require_bootstrap_scope(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            flag="moodboard_graph_bound",
        )
        with self._db.transaction() as conn:
            self._set_postgres_dataset_scope(conn, dataset)
            rows = self._execute(
                conn,
                "SELECT p.*,b.sync_id AS moodboard_sync_id,"
                "CASE WHEN n.id IS NULL THEN 0 ELSE 1 END AS _note_exists "
                "FROM moodboard_notes p "
                "JOIN moodboards b ON b.owner_user_id=p.owner_user_id AND b.dataset_id=p.dataset_id "
                "AND b.id=p.moodboard_id "
                "LEFT JOIN notes n ON n.client_id=p.owner_user_id AND n.id=p.note_id "
                "WHERE p.owner_user_id=? AND p.dataset_id=? "
                "AND (b.sync_id>? OR (b.sync_id=? AND p.note_id>?)) "
                "ORDER BY b.sync_id,p.note_id LIMIT ?",
                (
                    owner,
                    dataset,
                    after_moodboard_sync_id or "",
                    after_moodboard_sync_id or "",
                    after_note_id or "",
                    max(1, min(int(limit), 500)),
                ),
            ).fetchall()
            try:
                self._validate_moodboard_placement_rows(
                    rows,
                    owner=owner,
                    dataset=dataset,
                )
            except (NotesMoodboardStudioContractError, TypeError, ValueError) as exc:
                raise ConflictError(
                    "Moodboard graph binding failed canonical readiness proof.",
                    entity="moodboard_notes",
                ) from exc  # noqa: TRY003
        result = [dict(row) for row in rows]
        for row in result:
            row.pop("_note_exists", None)
        return result

    def page_studio_documents_for_sync_bootstrap(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        after_note_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Page canonically verified Studio documents from a bound scope.

        Args:
            owner_user_id: Owner of the bound Studio graph.
            dataset_id: Dataset to expose for bootstrap.
            after_note_id: Exclusive note-ID cursor.
            limit: Requested page size, clamped to the range 1 through 500.

        Returns:
            Studio document rows ordered by note ID.

        Raises:
            InputError: If a scope identifier is invalid.
            ConflictError: If the graph is unbound or canonical validation
                detects corrupt or inconsistent rows.
        """
        owner, dataset = self._require_bootstrap_scope(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            flag="studio_graph_bound",
        )
        with self._db.transaction() as conn:
            self._set_postgres_dataset_scope(conn, dataset)
            rows = self._execute(
                conn,
                "SELECT * FROM note_studio_documents WHERE owner_user_id=? AND dataset_id=? "
                "AND note_id>? ORDER BY note_id LIMIT ?",
                (owner, dataset, after_note_id or "", max(1, min(int(limit), 500))),
            ).fetchall()
            try:
                self._validate_studio_rows(
                    conn,
                    rows=rows,
                    owner=owner,
                    dataset=dataset,
                )
            except (NotesMoodboardStudioContractError, TypeError, ValueError) as exc:
                raise ConflictError(
                    "Studio graph binding failed canonical readiness proof.",
                    entity="note_studio_documents",
                ) from exc  # noqa: TRY003
        return [dict(row) for row in rows]
