"""Repeatable source reads for cloneable Media rows and child collections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    MediaDbLike,
    require_media_database_like,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Sharing.clone_models import MediaCloneSnapshot

_SnapshotResult = TypeVar("_SnapshotResult")


class CloneSnapshotRepository:
    """Materialize active Media clone inputs through one repeatable transaction."""

    def __init__(self, session: MediaDbLike):
        self.session = session

    @classmethod
    def from_legacy_db(cls, db: MediaDbLike) -> CloneSnapshotRepository:
        return cls(
            session=require_media_database_like(
                db,
                error_message="db_instance must be a Media database object.",
            )
        )

    @staticmethod
    def _validate_media_ids(media_ids: Sequence[int]) -> tuple[int, ...]:
        from tldw_Server_API.app.core.Sharing.clone_models import CloneSnapshotUnavailable

        if isinstance(media_ids, (str, bytes, bytearray)) or not isinstance(
            media_ids,
            Sequence,
        ):
            raise CloneSnapshotUnavailable(cleanup_state="complete")
        normalized = tuple(media_ids)
        if any(
            isinstance(media_id, bool)
            or not isinstance(media_id, int)
            or media_id <= 0
            for media_id in normalized
        ):
            raise CloneSnapshotUnavailable(cleanup_state="complete")
        if len(set(normalized)) != len(normalized):
            raise CloneSnapshotUnavailable(cleanup_state="complete")
        return normalized

    def _run_snapshot(self, reader: Callable[[Any, Any], _SnapshotResult]) -> _SnapshotResult:
        from tldw_Server_API.app.core.Sharing.clone_models import CloneSnapshotUnavailable

        backend = self.session.backend  # type: ignore[attr-defined]
        connection: Any | None = None
        pool: Any | None = None
        committed = False
        primary_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        result: _SnapshotResult | None = None

        try:
            if backend.backend_type == BackendType.SQLITE:
                sqlite_path = str(getattr(backend.config, "sqlite_path", "") or "").strip()
                lowered_path = sqlite_path.lower()
                private_memory = sqlite_path == ":memory:" or (
                    "mode=memory" in lowered_path and "cache=shared" not in lowered_path
                )
                if not sqlite_path or private_memory:
                    raise CloneSnapshotUnavailable(cleanup_state="complete")
                connection = backend.connect()
                connection.execute("PRAGMA query_only = ON")
                query_only_row = connection.execute("PRAGMA query_only").fetchone()
                if query_only_row is None or int(query_only_row[0]) != 1:
                    raise RuntimeError("SQLite query-only mode unavailable")
                connection.execute("BEGIN")
                if not bool(getattr(connection, "in_transaction", False)):
                    raise RuntimeError("SQLite snapshot transaction unavailable")
            else:
                pool = backend.get_pool()
                connection = pool.get_connection()
                connection.rollback()
                backend.apply_and_verify_scope(
                    connection,
                    fallback_user_id=self.session.client_id,
                )
                with connection.cursor() as cursor:
                    cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
                isolation_rows = backend.execute(
                    "SHOW transaction_isolation",
                    connection=connection,
                    log_errors=False,
                ).rows
                read_only_rows = backend.execute(
                    "SHOW transaction_read_only",
                    connection=connection,
                    log_errors=False,
                ).rows
                isolation = next(iter(isolation_rows[0].values()), None) if isolation_rows else None
                read_only = next(iter(read_only_rows[0].values()), None) if read_only_rows else None
                if str(isolation).lower() != "repeatable read" or str(read_only).lower() not in {
                    "on",
                    "true",
                }:
                    raise RuntimeError("PostgreSQL repeatable read unavailable")

            result = reader(backend, connection)
            connection.commit()
            committed = True
        except BaseException as exc:  # noqa: BLE001 - cleanup must run for every path
            primary_error = exc

        if connection is not None:
            if not committed:
                try:
                    connection.rollback()
                except BaseException as exc:  # noqa: BLE001 - preserve primary failure
                    cleanup_error = exc
            try:
                if backend.backend_type == BackendType.SQLITE:
                    backend.disconnect(connection)
                else:
                    (pool or backend.get_pool()).return_connection(connection)
            except BaseException as exc:  # noqa: BLE001 - convert cleanup failures below
                cleanup_error = cleanup_error or exc

        if primary_error is not None and not isinstance(primary_error, Exception):
            raise primary_error
        if primary_error is not None or cleanup_error is not None or result is None:
            failure = primary_error or cleanup_error
            logger.bind(
                backend=backend.backend_type.value,
                exception_type=type(failure).__name__ if failure is not None else "Unknown",
            ).warning("Media clone snapshot read failed")
            raise CloneSnapshotUnavailable(cleanup_state="complete") from None
        return result

    def read(self, media_ids: Sequence[int]) -> dict[int, MediaCloneSnapshot]:
        """Return active Media rows and child collections in requested ID order."""
        from tldw_Server_API.app.core.Sharing.clone_models import (
            CloneSnapshotUnavailable,
            MediaCloneSnapshot,
        )

        requested_ids = self._validate_media_ids(media_ids)
        if not requested_ids:
            return {}

        def materialize(backend: Any, connection: Any) -> dict[int, MediaCloneSnapshot]:
            placeholders = ", ".join("?" for _ in requested_ids)
            active_value = False if backend.backend_type == BackendType.POSTGRESQL else 0

            def read_rows(query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
                query_result = backend.execute(
                    query,
                    params,
                    connection=connection,
                    log_errors=False,
                )
                return [dict(row) for row in query_result.rows]

            media_rows = read_rows(
                f"SELECT * FROM Media WHERE id IN ({placeholders}) "  # nosec B608
                "AND deleted = ? AND is_trash = ?",
                (*requested_ids, active_value, active_value),
            )
            media_by_id = {int(row["id"]): row for row in media_rows}
            if len(media_by_id) != len(requested_ids) or any(
                media_id not in media_by_id for media_id in requested_ids
            ):
                raise CloneSnapshotUnavailable(cleanup_state="complete")

            chunk_rows = read_rows(
                f"SELECT * FROM UnvectorizedMediaChunks "  # nosec B608
                f"WHERE media_id IN ({placeholders}) AND deleted = ? "
                "ORDER BY media_id, chunk_index, id",
                (*requested_ids, active_value),
            )
            transcript_rows = read_rows(
                f"SELECT t.* FROM Transcripts t JOIN Media m ON m.id = t.media_id "  # nosec B608
                f"WHERE t.media_id IN ({placeholders}) AND t.deleted = ? "
                "AND m.deleted = ? AND m.is_trash = ? "
                "ORDER BY t.media_id, "
                "CASE WHEN m.latest_transcription_run_id IS NOT NULL "
                "AND t.transcription_run_id = m.latest_transcription_run_id THEN 0 ELSE 1 END, "
                "CASE WHEN t.transcription_run_id IS NULL THEN 1 ELSE 0 END, "
                "t.transcription_run_id DESC, t.created_at DESC, t.id DESC",
                (*requested_ids, active_value, active_value, active_value),
            )
            keyword_order = (
                "LOWER(k.keyword), k.keyword"
                if backend.backend_type == BackendType.POSTGRESQL
                else "k.keyword COLLATE NOCASE"
            )
            keyword_rows = read_rows(
                f"SELECT mk.media_id, k.keyword FROM MediaKeywords mk "  # nosec B608
                "JOIN Keywords k ON k.id = mk.keyword_id "
                "JOIN Media m ON m.id = mk.media_id "
                f"WHERE mk.media_id IN ({placeholders}) AND k.deleted = ? "
                "AND m.deleted = ? AND m.is_trash = ? "
                f"ORDER BY mk.media_id, {keyword_order}, k.id",  # nosec B608
                (*requested_ids, active_value, active_value, active_value),
            )

            chunks_by_media = {media_id: [] for media_id in requested_ids}
            transcripts_by_media = {media_id: [] for media_id in requested_ids}
            keywords_by_media = {media_id: [] for media_id in requested_ids}
            for row in chunk_rows:
                media_id = int(row["media_id"])
                if media_id not in chunks_by_media:
                    raise CloneSnapshotUnavailable(cleanup_state="complete")
                chunks_by_media[media_id].append(row)
            for row in transcript_rows:
                media_id = int(row["media_id"])
                if media_id not in transcripts_by_media:
                    raise CloneSnapshotUnavailable(cleanup_state="complete")
                transcripts_by_media[media_id].append(row)
            for row in keyword_rows:
                media_id = int(row["media_id"])
                if media_id not in keywords_by_media:
                    raise CloneSnapshotUnavailable(cleanup_state="complete")
                keywords_by_media[media_id].append(str(row["keyword"]))

            snapshots: dict[int, MediaCloneSnapshot] = {}
            for media_id in requested_ids:
                media_row = dict(media_by_id[media_id])
                media_row["keywords"] = tuple(keywords_by_media[media_id])
                snapshots[media_id] = MediaCloneSnapshot.from_rows(
                    media=media_row,
                    chunks=chunks_by_media[media_id],
                    transcripts=transcripts_by_media[media_id],
                )
            return snapshots

        return self._run_snapshot(materialize)


def read_media_clone_snapshots(
    self: MediaDbLike,
    media_ids: Sequence[int],
) -> dict[int, MediaCloneSnapshot]:
    """MediaDatabase binding for repeatable clone source reads."""
    return CloneSnapshotRepository.from_legacy_db(self).read(media_ids)


__all__ = ["CloneSnapshotRepository", "read_media_clone_snapshots"]
