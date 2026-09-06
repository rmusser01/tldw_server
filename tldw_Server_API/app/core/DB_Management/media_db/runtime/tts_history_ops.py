"""Package-owned TTS history CRUD helpers."""

from __future__ import annotations

import json
from contextlib import nullcontext, suppress
from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def create_tts_history_entry(
    self,
    *,
    user_id: str,
    text_hash: str,
    created_at: str | None = None,
    text: str | None = None,
    text_length: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    voice_id: str | None = None,
    voice_name: str | None = None,
    voice_info: dict[str, Any] | None = None,
    format: str | None = None,
    duration_ms: int | None = None,
    generation_time_ms: int | None = None,
    params_json: dict[str, Any] | None = None,
    status: str | None = None,
    segments_json: dict[str, Any] | None = None,
    favorite: bool = False,
    job_id: int | None = None,
    output_id: int | None = None,
    artifact_ids: list[Any] | None = None,
    artifact_deleted_at: str | None = None,
    error_message: str | None = None,
    deleted: bool = False,
    deleted_at: str | None = None,
    output_incarnation: str | None = None,
    conn: Any | None = None,
) -> int | None:
    """Insert history without committing a supplied or enclosing transaction."""
    now = created_at or self._get_current_utc_timestamp_str()
    insert_sql = (
        "INSERT INTO tts_history "
        "(user_id, created_at, text, text_hash, text_length, provider, model, voice_id, voice_name, "
        "voice_info, format, duration_ms, generation_time_ms, params_json, status, segments_json, "
        "favorite, job_id, output_id, artifact_ids, artifact_deleted_at, error_message, deleted, deleted_at, output_incarnation) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    if self.backend_type == BackendType.POSTGRESQL:
        insert_sql += " RETURNING id"

    voice_info_str = json.dumps(voice_info, separators=(",", ":"), ensure_ascii=True) if voice_info else None
    params_str = json.dumps(params_json, separators=(",", ":"), ensure_ascii=True) if params_json else None
    segments_str = json.dumps(segments_json, separators=(",", ":"), ensure_ascii=True) if segments_json else None
    artifacts_str = json.dumps(artifact_ids, separators=(",", ":"), ensure_ascii=True) if artifact_ids else None

    if output_incarnation is not None:
        _validate_output_history_identity(user_id, output_incarnation)
    context = self.transaction() if conn is None else nullcontext(conn)
    with context as history_conn:
        if output_incarnation is not None:
            instance = _lock_output_history_instance(self, str(user_id), output_incarnation, history_conn)
            if instance["state"] == "disposed":
                output_id, artifacts_str, artifact_deleted_at = None, None, instance["disposed_at"]
        cursor = self.execute_query(
            insert_sql,
            (
                str(user_id),
                now,
                text,
                str(text_hash),
                int(text_length) if text_length is not None else None,
                provider,
                model,
                voice_id,
                voice_name,
                voice_info_str,
                format,
                int(duration_ms) if duration_ms is not None else None,
                int(generation_time_ms) if generation_time_ms is not None else None,
                params_str,
                status,
                segments_str,
                bool(favorite),
                int(job_id) if job_id is not None else None,
                int(output_id) if output_id is not None else None,
                artifacts_str,
                artifact_deleted_at,
                error_message,
                bool(deleted),
                deleted_at,
                output_incarnation,
            ),
            commit=False,
            connection=history_conn,
        )
        if self.backend_type == BackendType.POSTGRESQL:
            row = cursor.fetchone()
            return int(row["id"]) if row and row.get("id") is not None else None
        return cursor.lastrowid


def _validate_output_history_identity(user_id: str, incarnation: str) -> None:
    if (
        user_id is None
        or not str(user_id).strip()
        or not isinstance(incarnation, str)
        or len(incarnation) != 32
        or any(char not in "0123456789abcdef" for char in incarnation)
    ):
        raise ValueError("output_history_invalid")


def _lock_output_history_instance(self, user_id: str, incarnation: str, conn: Any) -> dict[str, Any]:
    """Acquire the shared insert/dispose fence inside the caller's transaction."""
    self.execute_query(
        "INSERT INTO tts_output_instances (user_id, output_incarnation, state) VALUES (?, ?, 'live') "
        "ON CONFLICT (user_id, output_incarnation) DO NOTHING",
        (user_id, incarnation),
        connection=conn,
        log_errors=False,
    )
    query = "SELECT state, disposal_token, disposed_at FROM tts_output_instances WHERE user_id = ? AND output_incarnation = ?"
    if self.backend_type == BackendType.POSTGRESQL:
        query += " FOR UPDATE"
    row = self.execute_query(query, (user_id, incarnation), connection=conn, log_errors=False).fetchone()
    if row is None:
        raise RuntimeError("output_history_unavailable")
    return dict(row)


def dispose_tts_output_instance(
    self,
    *,
    user_id: str,
    output_incarnation: str,
    disposal_token: str,
    deleted_at: str,
    conn: Any | None = None,
) -> int:
    """Permanently clear this user's original-instance links, including late inserts.

    A supplied connection must belong to a caller-owned transaction. Replays
    preserve the first disposal token/timestamp; numeric output IDs are not used.
    """
    _validate_output_history_identity(user_id, output_incarnation)
    _validate_output_history_identity(user_id, disposal_token)
    try:
        timestamp = datetime.fromisoformat(deleted_at)
        if timestamp.tzinfo is None:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("output_history_invalid") from None
    with self.transaction() if conn is None else nullcontext(conn) as history_conn:
        instance = _lock_output_history_instance(self, str(user_id), output_incarnation, history_conn)
        if instance["state"] == "live":
            self.execute_query(
                "UPDATE tts_output_instances SET state = 'disposed', disposal_token = ?, disposed_at = ? "
                "WHERE user_id = ? AND output_incarnation = ? AND state = 'live'",
                (disposal_token, deleted_at, str(user_id), output_incarnation),
                connection=history_conn,
                log_errors=False,
            )
        else:
            deleted_at = instance["disposed_at"]
        cursor = self.execute_query(
            "UPDATE tts_history SET artifact_deleted_at = ?, output_id = NULL, artifact_ids = NULL "
            "WHERE user_id = ? AND output_incarnation = ? "
            "AND (output_id IS NOT NULL OR artifact_ids IS NOT NULL OR artifact_deleted_at IS NULL)",
            (deleted_at, str(user_id), output_incarnation),
            connection=history_conn,
            log_errors=False,
        )
        return int(cursor.rowcount or 0)


def _build_tts_history_filters(
    self,
    *,
    user_id: str,
    q: str | None = None,
    text_hash: str | None = None,
    favorite: bool | None = None,
    provider: str | None = None,
    model: str | None = None,
    voice_id: str | None = None,
    voice_name: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    cursor_created_at: str | None = None,
    cursor_id: int | None = None,
    include_deleted: bool = False,
) -> tuple[list[str], list[Any]]:
    """Build shared SQL conditions and params for TTS history reads."""
    conditions: list[str] = ["user_id = ?"]
    params: list[Any] = [str(user_id)]

    if not include_deleted:
        conditions.append("deleted = 0")
    if favorite is not None:
        conditions.append("favorite = ?")
        params.append(1 if favorite else 0)
    if provider:
        conditions.append("provider = ?")
        params.append(str(provider))
    if model:
        conditions.append("model = ?")
        params.append(str(model))
    if voice_id:
        conditions.append("voice_id = ?")
        params.append(str(voice_id))
    if voice_name:
        conditions.append("voice_name = ?")
        params.append(str(voice_name))
    if text_hash:
        conditions.append("text_hash = ?")
        params.append(str(text_hash))
    if created_from:
        conditions.append("created_at >= ?")
        params.append(str(created_from))
    if created_to:
        conditions.append("created_at <= ?")
        params.append(str(created_to))
    if q:
        pattern = f"%{q}%"
        self._append_case_insensitive_like(conditions, params, "text", pattern)
    if cursor_created_at and cursor_id is not None:
        conditions.append("(created_at < ? OR (created_at = ? AND id < ?))")
        params.extend([str(cursor_created_at), str(cursor_created_at), int(cursor_id)])

    return conditions, params


def list_tts_history(
    self,
    *,
    user_id: str,
    q: str | None = None,
    text_hash: str | None = None,
    favorite: bool | None = None,
    provider: str | None = None,
    model: str | None = None,
    voice_id: str | None = None,
    voice_name: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    cursor_created_at: str | None = None,
    cursor_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List TTS history rows for a user."""
    try:
        limit = int(limit)
        offset = int(offset)
    except (TypeError, ValueError):
        limit, offset = 50, 0

    limit = max(1, min(201, limit))
    offset = max(0, offset)

    conditions, params = _build_tts_history_filters(
        self,
        user_id=user_id,
        q=q,
        text_hash=text_hash,
        favorite=favorite,
        provider=provider,
        model=model,
        voice_id=voice_id,
        voice_name=voice_name,
        created_from=created_from,
        created_to=created_to,
        cursor_created_at=cursor_created_at,
        cursor_id=cursor_id,
        include_deleted=False,
    )

    query = (
        "SELECT id, user_id, created_at, text, provider, model, voice_id, voice_name, "  # nosec B608
        "voice_info, format, duration_ms, status, favorite, job_id, output_id, "
        "artifact_deleted_at "
        "FROM tts_history WHERE "
        + " AND ".join(conditions)
        + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    rows = self.execute_query(query, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def count_tts_history(
    self,
    *,
    user_id: str,
    q: str | None = None,
    text_hash: str | None = None,
    favorite: bool | None = None,
    provider: str | None = None,
    model: str | None = None,
    voice_id: str | None = None,
    voice_name: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
) -> int:
    """Count matching TTS history rows for a user."""
    conditions, params = _build_tts_history_filters(
        self,
        user_id=user_id,
        q=q,
        text_hash=text_hash,
        favorite=favorite,
        provider=provider,
        model=model,
        voice_id=voice_id,
        voice_name=voice_name,
        created_from=created_from,
        created_to=created_to,
        include_deleted=False,
    )
    query = "SELECT COUNT(*) AS count FROM tts_history WHERE " + " AND ".join(conditions)  # nosec B608
    row = self.execute_query(query, tuple(params)).fetchone()
    if not row:
        return 0
    try:
        return int(row["count"])
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        return int(list(row)[0])


def get_tts_history_entry(
    self,
    *,
    user_id: str,
    history_id: int,
    include_deleted: bool = False,
) -> dict[str, Any] | None:
    """Fetch a single TTS history row."""
    conditions = ["id = ?", "user_id = ?"]
    params: list[Any] = [int(history_id), str(user_id)]
    if not include_deleted:
        conditions.append("deleted = 0")
    query = (
        "SELECT id, user_id, created_at, text, text_hash, text_length, provider, model, "  # nosec B608
        "voice_id, voice_name, voice_info, format, duration_ms, generation_time_ms, "
        "params_json, status, segments_json, favorite, job_id, output_id, artifact_ids, "
        "artifact_deleted_at, error_message, deleted, deleted_at "
        "FROM tts_history WHERE "
        + " AND ".join(conditions)
        + " LIMIT 1"
    )
    row = self.execute_query(query, tuple(params)).fetchone()
    return dict(row) if row else None


def update_tts_history_favorite(
    self,
    *,
    user_id: str,
    history_id: int,
    favorite: bool,
) -> bool:
    """Set or clear the favorite bit for a TTS history row."""
    cursor = self.execute_query(
        "UPDATE tts_history SET favorite = ? WHERE id = ? AND user_id = ? AND deleted = 0",
        (1 if favorite else 0, int(history_id), str(user_id)),
        commit=True,
    )
    try:
        return cursor.rowcount > 0
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        return False


def soft_delete_tts_history_entry(
    self,
    *,
    user_id: str,
    history_id: int,
    deleted_at: str | None = None,
) -> bool:
    """Soft-delete one TTS history row."""
    ts = deleted_at or self._get_current_utc_timestamp_str()
    cursor = self.execute_query(
        "UPDATE tts_history SET deleted = 1, deleted_at = ? WHERE id = ? AND user_id = ? AND deleted = 0",
        (ts, int(history_id), str(user_id)),
        commit=True,
    )
    try:
        return cursor.rowcount > 0
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        return False


def mark_tts_history_artifacts_deleted_for_output(
    self,
    *,
    user_id: str,
    output_id: int,
    deleted_at: str | None = None,
) -> int:
    """Clear artifact linkage for rows attached to the given output artifact."""
    ts = deleted_at or self._get_current_utc_timestamp_str()
    cursor = self.execute_query(
        (
            "UPDATE tts_history "
            "SET artifact_deleted_at = ?, output_id = NULL, artifact_ids = NULL "
            "WHERE user_id = ? AND output_id = ? AND deleted = 0"
        ),
        (ts, str(user_id), int(output_id)),
        commit=True,
    )
    try:
        return int(cursor.rowcount or 0)
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        return 0


def mark_tts_history_artifacts_deleted_for_file_id(
    self,
    *,
    user_id: str,
    file_id: int,
    deleted_at: str | None = None,
) -> int:
    """Clear artifact linkage for rows whose artifact_ids contain the file id."""
    ts = deleted_at or self._get_current_utc_timestamp_str()
    rows = self.execute_query(
        (
            "SELECT id, artifact_ids FROM tts_history "
            "WHERE user_id = ? AND artifact_ids IS NOT NULL AND deleted = 0"
        ),
        (str(user_id),),
    ).fetchall()
    if not rows:
        return 0

    matched_ids: list[int] = []
    for row in rows:
        raw = row["artifact_ids"]
        if raw is None:
            continue
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            parsed = None
        if isinstance(parsed, list) and file_id in parsed:
            matched_ids.append(int(row["id"]))

    if not matched_ids:
        return 0

    placeholders = ",".join(["?"] * len(matched_ids))
    params: list[Any] = [ts, str(user_id)] + matched_ids
    cursor = self.execute_query(
        (
            "UPDATE tts_history "  # nosec B608
            "SET artifact_deleted_at = ?, output_id = NULL, artifact_ids = NULL "
            f"WHERE user_id = ? AND id IN ({placeholders}) AND deleted = 0"
        ),
        tuple(params),
        commit=True,
    )
    try:
        return int(cursor.rowcount or 0)
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        return len(matched_ids)


def purge_tts_history_for_user(
    self,
    *,
    user_id: str,
    retention_days: int,
    max_rows: int,
) -> int:
    """Purge old and excess TTS history rows for a user."""
    removed = 0
    if retention_days and retention_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=int(retention_days))
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        cursor = self.execute_query(
            "DELETE FROM tts_history WHERE user_id = ? AND created_at < ?",
            (str(user_id), cutoff_str),
            commit=True,
        )
        with suppress(_MEDIA_NONCRITICAL_EXCEPTIONS):
            removed += int(cursor.rowcount or 0)

    if max_rows and max_rows > 0:
        row = self.execute_query(
            "SELECT COUNT(*) AS count FROM tts_history WHERE user_id = ?",
            (str(user_id),),
        ).fetchone()
        if row:
            try:
                total = int(row["count"])
            except _MEDIA_NONCRITICAL_EXCEPTIONS:
                total = int(list(row)[0])
            if total > max_rows:
                to_remove = total - int(max_rows)
                cursor = self.execute_query(
                    (
                        "DELETE FROM tts_history WHERE user_id = ? AND id IN ("
                        "SELECT id FROM tts_history WHERE user_id = ? "
                        "ORDER BY created_at ASC, id ASC LIMIT ?"
                        ")"
                    ),
                    (str(user_id), str(user_id), int(to_remove)),
                    commit=True,
                )
                try:
                    removed += int(cursor.rowcount or 0)
                except _MEDIA_NONCRITICAL_EXCEPTIONS:
                    removed += max(0, to_remove)

    return removed


def list_tts_history_user_ids(self) -> list[str]:
    """List distinct TTS history user ids."""
    rows = self.execute_query(
        "SELECT DISTINCT user_id FROM tts_history",
        None,
    ).fetchall()
    user_ids: list[str] = []
    for row in rows:
        try:
            user_ids.append(str(row["user_id"]))
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            try:
                user_ids.append(str(row[0]))
            except _MEDIA_NONCRITICAL_EXCEPTIONS:
                continue
    return user_ids


__all__ = [
    "create_tts_history_entry",
    "dispose_tts_output_instance",
    "_build_tts_history_filters",
    "list_tts_history",
    "count_tts_history",
    "get_tts_history_entry",
    "update_tts_history_favorite",
    "soft_delete_tts_history_entry",
    "mark_tts_history_artifacts_deleted_for_output",
    "mark_tts_history_artifacts_deleted_for_file_id",
    "purge_tts_history_for_user",
    "list_tts_history_user_ids",
]
