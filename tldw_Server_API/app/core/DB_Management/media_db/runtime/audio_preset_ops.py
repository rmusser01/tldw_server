"""Package-owned audio preset CRUD helpers."""

from __future__ import annotations

import json
import uuid
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def _db_bool(self: Any, value: bool) -> bool | int:
    return bool(value) if self.backend_type == BackendType.POSTGRESQL else (1 if value else 0)


def _json_dumps(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, separators=(",", ":"), ensure_ascii=True)


def _json_loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _row_to_audio_preset(row: Any) -> dict[str, Any] | None:
    if not row:
        return None
    data = dict(row)
    data["favorite"] = bool(data.get("favorite"))
    data["is_default"] = bool(data.get("is_default"))
    data["config"] = _json_loads(data.pop("config_json", None))
    data["capability_assumptions"] = _json_loads(data.pop("capability_assumptions_json", None))
    return data


def _check_audio_preset_name_available(
    self: Any,
    *,
    user_id: str,
    kind: str,
    name: str,
    exclude_id: str | None = None,
    conn: Any | None = None,
) -> None:
    excluded_preset_id = str(exclude_id) if exclude_id else None
    row = self.execute_query(
        (
            "SELECT id FROM audio_presets "
            "WHERE user_id = ? AND kind = ? AND deleted = ? AND LOWER(name) = LOWER(?) "
            "AND (? IS NULL OR id <> ?) LIMIT 1"
        ),
        (
            str(user_id),
            str(kind),
            _db_bool(self, False),
            str(name),
            excluded_preset_id,
            excluded_preset_id,
        ),
        connection=conn,
    ).fetchone()
    if row:
        raise ConflictError("An audio preset with this name already exists for this kind.")


def _get_audio_preset_row(
    self: Any,
    *,
    user_id: str,
    preset_id: str,
    include_deleted: bool = False,
    conn: Any | None = None,
) -> dict[str, Any] | None:
    deleted_filter = None if include_deleted else _db_bool(self, False)
    row = self.execute_query(
        (
            "SELECT id, user_id AS owner_user_id, kind, name, description, favorite, is_default, "
            "config_json, capability_assumptions_json, created_at, updated_at "
            "FROM audio_presets WHERE id = ? AND user_id = ? "
            "AND (? IS NULL OR deleted = ?) LIMIT 1"
        ),
        (str(preset_id), str(user_id), deleted_filter, deleted_filter),
        connection=conn,
    ).fetchone()
    return _row_to_audio_preset(row)


def create_audio_preset(
    self: Any,
    *,
    user_id: str,
    kind: str,
    name: str,
    description: str | None = None,
    favorite: bool = False,
    is_default: bool = False,
    config: dict[str, Any] | None = None,
    capability_assumptions: dict[str, Any] | None = None,
    preset_id: str | None = None,
) -> dict[str, Any]:
    """Create an audio preset for one user."""
    new_id = preset_id or str(uuid.uuid4())
    now = self._get_current_utc_timestamp_str()
    user_id_str = str(user_id)
    kind_str = str(kind)

    with self.transaction() as conn:
        _check_audio_preset_name_available(
            self,
            user_id=user_id_str,
            kind=kind_str,
            name=name,
            conn=conn,
        )
        if is_default:
            self.execute_query(
                "UPDATE audio_presets SET is_default = ? WHERE user_id = ? AND kind = ? AND deleted = ?",
                (_db_bool(self, False), user_id_str, kind_str, _db_bool(self, False)),
                connection=conn,
            )
        self.execute_query(
            (
                "INSERT INTO audio_presets "
                "(id, user_id, kind, name, description, favorite, is_default, config_json, "
                "capability_assumptions_json, created_at, updated_at, deleted, version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                new_id,
                user_id_str,
                kind_str,
                name,
                description,
                _db_bool(self, favorite),
                _db_bool(self, is_default),
                _json_dumps(config),
                _json_dumps(capability_assumptions),
                now,
                now,
                _db_bool(self, False),
                1,
            ),
            connection=conn,
        )
        created = _get_audio_preset_row(self, user_id=user_id_str, preset_id=new_id, conn=conn)
    if created is None:
        raise ConflictError("Audio preset could not be created.")
    return created


def list_audio_presets(
    self: Any,
    *,
    user_id: str,
    kind: str | None = None,
    favorite: bool | None = None,
    is_default: bool | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List active audio presets for one user."""
    try:
        limit = int(limit)
        offset = int(offset)
    except (TypeError, ValueError):
        limit, offset = 100, 0
    limit = max(1, min(200, limit))
    offset = max(0, offset)

    kind_filter = str(kind) if kind else None
    favorite_filter = _db_bool(self, bool(favorite)) if favorite is not None else None
    is_default_filter = _db_bool(self, bool(is_default)) if is_default is not None else None

    rows = self.execute_query(
        (
            "SELECT id, user_id AS owner_user_id, kind, name, description, favorite, is_default, "
            "config_json, capability_assumptions_json, created_at, updated_at "
            "FROM audio_presets WHERE user_id = ? AND deleted = ? "
            "AND (? IS NULL OR kind = ?) "
            "AND (? IS NULL OR favorite = ?) "
            "AND (? IS NULL OR is_default = ?) "
            "ORDER BY updated_at DESC, created_at DESC, name ASC LIMIT ? OFFSET ?"
        ),
        (
            str(user_id),
            _db_bool(self, False),
            kind_filter,
            kind_filter,
            favorite_filter,
            favorite_filter,
            is_default_filter,
            is_default_filter,
            limit,
            offset,
        ),
    ).fetchall()
    return [preset for row in rows if (preset := _row_to_audio_preset(row)) is not None]


def count_audio_presets(
    self: Any,
    *,
    user_id: str,
    kind: str | None = None,
    favorite: bool | None = None,
    is_default: bool | None = None,
) -> int:
    """Count active audio presets for one user."""
    kind_filter = str(kind) if kind else None
    favorite_filter = _db_bool(self, bool(favorite)) if favorite is not None else None
    is_default_filter = _db_bool(self, bool(is_default)) if is_default is not None else None
    row = self.execute_query(
        (
            "SELECT COUNT(*) AS count FROM audio_presets "
            "WHERE user_id = ? AND deleted = ? "
            "AND (? IS NULL OR kind = ?) "
            "AND (? IS NULL OR favorite = ?) "
            "AND (? IS NULL OR is_default = ?)"
        ),
        (
            str(user_id),
            _db_bool(self, False),
            kind_filter,
            kind_filter,
            favorite_filter,
            favorite_filter,
            is_default_filter,
            is_default_filter,
        ),
    ).fetchone()
    if not row:
        return 0
    try:
        return int(row["count"])
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        return int(list(row)[0])


def get_audio_preset(
    self: Any,
    *,
    user_id: str,
    preset_id: str,
    include_deleted: bool = False,
) -> dict[str, Any] | None:
    """Fetch one audio preset by id for one user."""
    return _get_audio_preset_row(
        self,
        user_id=str(user_id),
        preset_id=str(preset_id),
        include_deleted=include_deleted,
    )


def update_audio_preset(
    self: Any,
    *,
    user_id: str,
    preset_id: str,
    updates: dict[str, Any],
) -> dict[str, Any] | None:
    """Update mutable fields on an active audio preset."""
    allowed = {
        "name",
        "description",
        "favorite",
        "is_default",
        "config",
        "capability_assumptions",
    }
    sanitized = {key: value for key, value in updates.items() if key in allowed}
    if not sanitized:
        return get_audio_preset(self, user_id=user_id, preset_id=preset_id)

    user_id_str = str(user_id)
    preset_id_str = str(preset_id)
    now = self._get_current_utc_timestamp_str()

    with self.transaction() as conn:
        current = _get_audio_preset_row(self, user_id=user_id_str, preset_id=preset_id_str, conn=conn)
        if current is None:
            return None
        kind = str(current["kind"])
        if "name" in sanitized:
            _check_audio_preset_name_available(
                self,
                user_id=user_id_str,
                kind=kind,
                name=str(sanitized["name"]),
                exclude_id=preset_id_str,
                conn=conn,
            )
        if sanitized.get("is_default") is True:
            self.execute_query(
                "UPDATE audio_presets SET is_default = ? WHERE user_id = ? AND kind = ? AND deleted = ?",
                (_db_bool(self, False), user_id_str, kind, _db_bool(self, False)),
                connection=conn,
            )

        set_clauses: list[str] = ["updated_at = ?"]
        params: list[Any] = [now]
        if "name" in sanitized:
            set_clauses.append("name = ?")
            params.append(str(sanitized["name"]))
        if "description" in sanitized:
            set_clauses.append("description = ?")
            params.append(sanitized["description"])
        if "favorite" in sanitized:
            set_clauses.append("favorite = ?")
            params.append(_db_bool(self, bool(sanitized["favorite"])))
        if "is_default" in sanitized:
            set_clauses.append("is_default = ?")
            params.append(_db_bool(self, bool(sanitized["is_default"])))
        if "config" in sanitized:
            set_clauses.append("config_json = ?")
            params.append(_json_dumps(sanitized["config"]))
        if "capability_assumptions" in sanitized:
            set_clauses.append("capability_assumptions_json = ?")
            params.append(_json_dumps(sanitized["capability_assumptions"]))
        params.extend([preset_id_str, user_id_str, _db_bool(self, False)])

        self.execute_query(
            (
                "UPDATE audio_presets SET "  # nosec B608
                + ", ".join(set_clauses)
                + " WHERE id = ? AND user_id = ? AND deleted = ?"
            ),
            tuple(params),
            connection=conn,
        )
        return _get_audio_preset_row(self, user_id=user_id_str, preset_id=preset_id_str, conn=conn)


def soft_delete_audio_preset(
    self: Any,
    *,
    user_id: str,
    preset_id: str,
    deleted_at: str | None = None,
) -> bool:
    """Soft-delete one audio preset for one user."""
    ts = deleted_at or self._get_current_utc_timestamp_str()
    cursor = self.execute_query(
        (
            "UPDATE audio_presets SET deleted = ?, deleted_at = ?, is_default = ? "
            "WHERE id = ? AND user_id = ? AND deleted = ?"
        ),
        (
            _db_bool(self, True),
            ts,
            _db_bool(self, False),
            str(preset_id),
            str(user_id),
            _db_bool(self, False),
        ),
        commit=True,
    )
    try:
        return bool(cursor.rowcount and cursor.rowcount > 0)
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        return False


__all__ = [
    "create_audio_preset",
    "list_audio_presets",
    "count_audio_presets",
    "get_audio_preset",
    "update_audio_preset",
    "soft_delete_audio_preset",
]
