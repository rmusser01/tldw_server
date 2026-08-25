from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendConnectionWrapper,
    BackendType,
    ConflictError,
    InputError,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


GraphConnection: TypeAlias = sqlite3.Connection | BackendConnectionWrapper


class MoodboardSyncStore:
    """Scoped product reads and one-way graph authority binding."""

    _LOCAL_UNBOUND = "local-unbound"

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
        if self._db.backend_type == BackendType.POSTGRESQL and owner != str(self._db.client_id):
            raise ConflictError(
                "Graph compatibility scope is unavailable.",
                entity="notes",
                entity_id=owner,
            )  # noqa: TRY003
        query = (
            "SELECT owner_user_id,dataset_id FROM note_task_scope_authority "
            f"WHERE owner_user_id = ? AND {flag} = 1 LIMIT 2"  # nosec B608 - fixed flag names.
        )
        cursor = self._db.execute_query(query, (owner,)) if conn is None else self._execute(conn, query, (owner,))
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
        prove: Callable[[GraphConnection, str], None],
        conn: GraphConnection | None,
    ) -> dict[str, int]:
        owner, target = self._validated_scope(owner_user_id, target_dataset_id)
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

        def bind(transaction_conn: GraphConnection) -> dict[str, int]:
            if postgres:
                self._execute(transaction_conn, "LOCK TABLE note_task_scope_authority IN EXCLUSIVE MODE")
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
                return counts(target_state)
            if any(counts(target_state).values()):
                raise ConflictError("Graph binding target collision.", entity="notes", entity_id=target)  # noqa: TRY003

            if any(counts(source).values()):
                prove(transaction_conn, owner)
                rekey(transaction_conn, owner, target)
                remaining = snapshot(transaction_conn, self._LOCAL_UNBOUND)
                rebound = snapshot(transaction_conn, target)
                if any(counts(remaining).values()) or rebound != source:
                    raise ConflictError("Graph binding failed complete-set verification.", entity="notes", entity_id=target)  # noqa: TRY003
            else:
                rebound = source

            if authority is None:
                graph_values = {
                    "task_graph_bound": 0,
                    "moodboard_graph_bound": 0,
                    "studio_graph_bound": 0,
                }
                graph_values[flag] = 1
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
                self._execute(
                    transaction_conn,
                    f"UPDATE note_task_scope_authority SET {flag}=1 "  # nosec B608 - fixed flag names.
                    f"WHERE owner_user_id=? AND dataset_id=? AND {flag}=0",  # nosec B608
                    (owner, target),
                )
            return counts(rebound)

        if conn is not None:
            return bind(conn)
        with self._db.transaction() as transaction_conn:
            return bind(transaction_conn)

    def bind_local_moodboard_graph_to_dataset(
        self,
        *,
        owner_user_id: str,
        target_dataset_id: str,
        conn: GraphConnection | None = None,
    ) -> dict[str, int]:
        def prove(transaction_conn: GraphConnection, owner: str) -> None:
            blocked = self._execute(
                transaction_conn,
                "SELECT 1 FROM moodboards WHERE owner_user_id=? AND dataset_id=? "
                "AND source_diagnostic_code IS NOT NULL "
                "UNION ALL "
                "SELECT 1 FROM moodboard_notes WHERE owner_user_id=? AND dataset_id=? "
                "AND source_diagnostic_code IS NOT NULL LIMIT 1",
                (owner, self._LOCAL_UNBOUND, owner, self._LOCAL_UNBOUND),
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
                (owner, self._LOCAL_UNBOUND),
            ).fetchone()
            if invalid is not None:
                raise ConflictError("Moodboard graph binding failed parent proof.", entity="moodboards")  # noqa: TRY003

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
        def prove(transaction_conn: GraphConnection, owner: str) -> None:
            invalid = self._execute(
                transaction_conn,
                "SELECT 1 FROM note_studio_documents s "
                "LEFT JOIN notes n ON n.client_id=s.owner_user_id AND n.id=s.note_id "
                "LEFT JOIN notes source ON source.client_id=s.owner_user_id "
                "AND source.id=s.source_note_id "
                "WHERE s.owner_user_id=? AND s.dataset_id=? "
                "AND (n.id IS NULL OR s.source_diagnostic_code IS NOT NULL "
                "OR (s.source_note_id IS NOT NULL AND source.id IS NULL)) LIMIT 1",
                (owner, self._LOCAL_UNBOUND),
            ).fetchone()
            if invalid is not None:
                raise ConflictError(
                    "Studio graph binding failed canonical readiness proof.",
                    entity="note_studio_documents",
                )  # noqa: TRY003

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
        owner, dataset = self._require_bootstrap_scope(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            flag="moodboard_graph_bound",
        )
        rows = self._db.execute_query(
            "SELECT * FROM moodboards WHERE owner_user_id=? AND dataset_id=? "
            "AND sync_id>? ORDER BY sync_id LIMIT ?",
            (owner, dataset, after_sync_id or "", max(1, min(int(limit), 500))),
        ).fetchall()
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
        owner, dataset = self._require_bootstrap_scope(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            flag="moodboard_graph_bound",
        )
        rows = self._db.execute_query(
            "SELECT p.*,b.sync_id AS moodboard_sync_id FROM moodboard_notes p "
            "JOIN moodboards b ON b.owner_user_id=p.owner_user_id AND b.dataset_id=p.dataset_id "
            "AND b.id=p.moodboard_id WHERE p.owner_user_id=? AND p.dataset_id=? "
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
        return [dict(row) for row in rows]

    def page_studio_documents_for_sync_bootstrap(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        after_note_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        owner, dataset = self._require_bootstrap_scope(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            flag="studio_graph_bound",
        )
        rows = self._db.execute_query(
            "SELECT * FROM note_studio_documents WHERE owner_user_id=? AND dataset_id=? "
            "AND note_id>? ORDER BY note_id LIMIT ?",
            (owner, dataset, after_note_id or "", max(1, min(int(limit), 500))),
        ).fetchall()
        return [dict(row) for row in rows]
