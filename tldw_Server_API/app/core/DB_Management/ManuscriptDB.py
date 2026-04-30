# ManuscriptDB.py
# Description: Helper class wrapping CRUD operations for manuscript tables
#   (projects, parts, chapters, scenes) stored in the ChaChaNotes DB.
#
from __future__ import annotations

"""
ManuscriptDB.py
---------------

Thin helper that receives a :class:`CharactersRAGDB` instance and exposes
ergonomic CRUD methods for the four manuscript tables introduced in schema V41:

- ``manuscript_projects``
- ``manuscript_parts``
- ``manuscript_chapters``
- ``manuscript_scenes``

All public methods use the underlying DB's ``transaction()`` context manager
and follow the existing optimistic-locking / soft-delete conventions.
"""

import json  # noqa: E402
import sqlite3  # noqa: E402
import uuid  # noqa: E402
from typing import Any  # noqa: E402

from loguru import logger  # noqa: E402

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (  # noqa: E402
    CharactersRAGDB,
    ConflictError,
    InputError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_PROJECT_STATUSES = frozenset(
    {"draft", "outlining", "writing", "revising", "complete", "archived"}
)
_VALID_CHAPTER_STATUSES = frozenset({"outline", "draft", "revising", "final"})
_VALID_SCENE_STATUSES = frozenset({"outline", "draft", "revising", "final"})

_REORDER_ENTITY_TABLES = {
    "part": "manuscript_parts",
    "chapter": "manuscript_chapters",
    "scene": "manuscript_scenes",
}

_MANUSCRIPT_ENTITY_TABLES = {
    "project": ("manuscript_projects", "project"),
    "manuscript": ("manuscript_parts", "manuscript"),
    "part": ("manuscript_parts", "manuscript"),
    "chapter": ("manuscript_chapters", "chapter"),
    "scene": ("manuscript_scenes", "scene"),
}

_MANUSCRIPT_VERSION_TABLES = {
    "manuscript": ("manuscript_parts", "manuscript"),
    "part": ("manuscript_parts", "manuscript"),
    "chapter": ("manuscript_chapters", "chapter"),
    "scene": ("manuscript_scenes", "scene"),
}

# Column whitelists for dynamic UPDATE statements — keys are the *caller*
# names (before any JSON-column mapping performed inside the method).
_UPDATABLE_PROJECT_COLS = frozenset({
    "title", "subtitle", "author", "genre", "status",
    "synopsis", "target_word_count", "word_count",
    "settings",  # mapped to settings_json by update_project()
})
_UPDATABLE_PART_COLS = frozenset({"title", "sort_order", "synopsis", "word_count"})
_UPDATABLE_CHAPTER_COLS = frozenset({
    "title", "status", "sort_order", "synopsis", "pov_character_id", "word_count", "part_id",
})
_UPDATABLE_SCENE_COLS = frozenset({
    "title", "content_json", "content_plain", "status",
    "sort_order", "synopsis", "word_count", "pov_character_id",
})
_UPDATABLE_CHARACTER_COLS = frozenset({
    "name", "role", "cast_group", "full_name", "age", "gender",
    "appearance", "personality", "backstory", "motivation",
    "arc_summary", "notes", "sort_order",
    "custom_fields",  # mapped to custom_fields_json by update_character()
})
_UPDATABLE_WORLD_INFO_COLS = frozenset({
    "kind", "name", "description", "parent_id", "sort_order",
    "properties", "tags",  # mapped to *_json by update_world_info()
})
_UPDATABLE_PLOT_LINE_COLS = frozenset({
    "title", "description", "status", "color", "sort_order",
})
_UPDATABLE_PLOT_EVENT_COLS = frozenset({
    "title", "description", "plot_line_id", "event_type", "sort_order",
    "scene_id", "chapter_id",
})
_UPDATABLE_PLOT_HOLE_COLS = frozenset({
    "title", "description", "severity", "status", "resolution",
    "scene_id", "chapter_id", "plot_line_id", "detected_by",
})
_UPDATABLE_CITATION_COLS = frozenset({
    "source_type", "source_id", "source_title", "excerpt", "query_used", "anchor_offset",
})


def _word_count(text: str | None) -> int:
    """Return the number of whitespace-delimited words in *text*."""
    if text and text.strip():
        return len(text.split())
    return 0


# ---------------------------------------------------------------------------
# ManuscriptDBHelper
# ---------------------------------------------------------------------------


class ManuscriptDBHelper:
    """High-level CRUD facade for manuscript tables.

    Parameters
    ----------
    db:
        A fully-initialised :class:`CharactersRAGDB` whose schema already
        includes the V41 manuscript tables.
    """

    # Column allowlists for update methods (defense-in-depth against injection)
    _UPDATABLE_PROJECT_COLS = frozenset({
        "title", "subtitle", "author", "genre", "status", "synopsis",
        "target_word_count", "settings_json",
    })
    _UPDATABLE_PART_COLS = frozenset({"title", "sort_order", "synopsis"})
    _UPDATABLE_CHAPTER_COLS = frozenset({
        "title", "part_id", "sort_order", "synopsis", "pov_character_id", "status",
    })
    _UPDATABLE_SCENE_COLS = frozenset({
        "title", "chapter_id", "sort_order", "content_json", "content_plain",
        "synopsis", "pov_character_id", "status", "word_count",
    })
    _UPDATABLE_CHARACTER_COLS = frozenset({
        "name", "role", "cast_group", "full_name", "age", "gender",
        "appearance", "personality", "backstory", "motivation", "arc_summary",
        "notes", "custom_fields_json", "sort_order",
    })
    _UPDATABLE_WORLD_INFO_COLS = frozenset({
        "name", "description", "parent_id", "properties_json", "tags_json", "sort_order",
    })
    _UPDATABLE_PLOT_LINE_COLS = frozenset({
        "title", "description", "status", "color", "sort_order",
    })
    _UPDATABLE_PLOT_EVENT_COLS = frozenset({
        "title", "description", "plot_line_id", "scene_id", "chapter_id", "event_type", "sort_order",
    })
    _UPDATABLE_PLOT_HOLE_COLS = frozenset({
        "title", "description", "severity", "status", "scene_id",
        "chapter_id", "plot_line_id", "resolution", "detected_by",
    })

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db
        self._ensure_version_schema()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now(self) -> str:
        return self.db._get_current_utc_timestamp_iso()

    def _uuid(self) -> str:
        return str(uuid.uuid4())

    def _ensure_version_schema(self) -> None:
        """Ensure manuscript manual snapshot storage exists."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS manuscript_versions (
                    id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL CHECK(entity_type IN ('manuscript','chapter','scene')),
                    entity_id TEXT NOT NULL,
                    project_id TEXT NOT NULL REFERENCES manuscript_projects(id) ON DELETE CASCADE,
                    version_number INTEGER NOT NULL,
                    label TEXT,
                    payload_json TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    client_id TEXT NOT NULL DEFAULT 'unknown',
                    UNIQUE(entity_type, entity_id, version_number)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_manuscript_versions_entity "
                "ON manuscript_versions(entity_type, entity_id, version_number DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_manuscript_versions_project "
                "ON manuscript_versions(project_id, created_at DESC)"
            )

    @property
    def _client_id(self) -> str:
        return self.db.client_id

    @staticmethod
    def _scene_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw scene DB row into API-friendly dict.

        Preserves ``content_json`` for API responses and mirrors it into a
        parsed ``content`` key for legacy helper callers.
        """
        d = dict(row)
        raw = d.get("content_json")
        if raw is None:
            d["content"] = None
            return d
        try:
            d["content"] = json.loads(raw)
        except (ValueError, TypeError):
            d["content"] = None
        return d

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        return dict(row)

    def _entity_row_to_dict(self, entity_type: str, row: Any) -> dict[str, Any]:
        data = dict(row)
        normalized_type = self._normalize_entity_type(entity_type)
        if normalized_type == "project":
            data = self._project_row_to_dict(data)
        elif normalized_type == "scene":
            data = self._scene_row_to_dict(data)
        data["deleted"] = bool(data.get("deleted"))
        return data

    @staticmethod
    def _normalize_entity_type(entity_type: str) -> str:
        return "manuscript" if entity_type == "part" else entity_type

    @classmethod
    def _validate_entity_type(cls, entity_type: str) -> tuple[str, str]:
        try:
            return _MANUSCRIPT_ENTITY_TABLES[entity_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported manuscript entity type: {entity_type}") from exc

    @classmethod
    def _validate_version_entity_type(cls, entity_type: str) -> tuple[str, str]:
        try:
            return _MANUSCRIPT_VERSION_TABLES[entity_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported manuscript version entity type: {entity_type}") from exc

    def _fetch_entity_row(
        self,
        conn: Any,
        entity_type: str,
        entity_id: str,
        *,
        deleted: bool | None = False,
    ) -> dict[str, Any]:
        table, label = self._validate_entity_type(entity_type)
        if deleted is None:
            deleted_clause = ""
            params: tuple[Any, ...] = (entity_id,)
        else:
            deleted_clause = " AND deleted = ?"
            params = (entity_id, 1 if deleted else 0)
        row = conn.execute(
            f"SELECT * FROM {table} WHERE id = ?{deleted_clause}",  # nosec B608
            params,
        ).fetchone()
        if row is None:
            deleted_label = "deleted " if deleted else ""
            raise InputError(f"{deleted_label}{label} '{entity_id}' not found")
        return self._entity_row_to_dict(entity_type, row)

    def _version_payload_for(self, entity_type: str, entity_id: str) -> dict[str, Any]:
        table, label = self._validate_version_entity_type(entity_type)
        with self.db.transaction() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE id = ? AND deleted = 0",  # nosec B608
                (entity_id,),
            ).fetchone()
            if row is None:
                raise InputError(f"{label} '{entity_id}' not found")
            data = self._entity_row_to_dict(entity_type, row)

        normalized_type = self._normalize_entity_type(entity_type)
        if normalized_type == "scene":
            return {
                "title": data["title"],
                "chapter_id": data["chapter_id"],
                "project_id": data["project_id"],
                "sort_order": data["sort_order"],
                "content_json": data.get("content_json"),
                "content": data.get("content"),
                "content_plain": data.get("content_plain") or "",
                "synopsis": data.get("synopsis"),
                "word_count": data.get("word_count") or 0,
                "status": data.get("status") or "draft",
            }
        if normalized_type == "chapter":
            scenes = self.list_scenes(entity_id)
            return {
                "title": data["title"],
                "project_id": data["project_id"],
                "part_id": data.get("part_id"),
                "sort_order": data["sort_order"],
                "synopsis": data.get("synopsis"),
                "word_count": data.get("word_count") or 0,
                "status": data.get("status") or "draft",
                "scene_ids": [scene["id"] for scene in scenes],
                "rendered_plain": "\n\n".join(scene.get("content_plain") or "" for scene in scenes).strip(),
            }

        chapters = self.list_chapters(data["project_id"], part_id=entity_id)
        rendered_parts: list[str] = []
        scene_ids: list[str] = []
        for chapter in chapters:
            scenes = self.list_scenes(chapter["id"])
            scene_ids.extend(scene["id"] for scene in scenes)
            rendered_parts.extend(scene.get("content_plain") or "" for scene in scenes)
        return {
            "title": data["title"],
            "project_id": data["project_id"],
            "sort_order": data["sort_order"],
            "synopsis": data.get("synopsis"),
            "word_count": data.get("word_count") or 0,
            "chapter_ids": [chapter["id"] for chapter in chapters],
            "scene_ids": scene_ids,
            "rendered_plain": "\n\n".join(rendered_parts).strip(),
        }

    def _next_version_number(self, conn: Any, entity_type: str, entity_id: str) -> int:
        row = conn.execute(
            """
            SELECT MAX(version_number) AS max_version
              FROM manuscript_versions
             WHERE entity_type = ? AND entity_id = ?
            """,
            (self._normalize_entity_type(entity_type), entity_id),
        ).fetchone()
        return int(row["max_version"] or 0) + 1

    @staticmethod
    def _version_row_to_dict(row: Any) -> dict[str, Any]:
        data = dict(row)
        data["payload"] = json.loads(data.pop("payload_json"))
        return data

    @staticmethod
    def _project_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw project DB row into API-friendly dict."""
        d = dict(row)
        raw = d.pop("settings_json", None) or "{}"
        try:
            d["settings"] = json.loads(raw)
        except (ValueError, TypeError):
            d["settings"] = {}
        return d

    # Alias used by dev-branch callers (kept for compatibility).
    _deserialize_project_row = _project_row_to_dict

    # Tables eligible for cross-project ownership checks.
    _PROJECT_CHECK_TABLES: frozenset[str] = frozenset({
        "manuscript_parts",
        "manuscript_characters",
        "manuscript_scenes",
        "manuscript_chapters",
        "manuscript_world_info",
        "manuscript_plot_lines",
    })

    def _project_is_active(self, conn: Any, project_id: str) -> bool:
        """Return ``True`` when the project exists and is not soft-deleted."""
        row = conn.execute(
            "SELECT 1 FROM manuscript_projects WHERE id = ? AND deleted = 0",
            (project_id,),
        ).fetchone()
        return row is not None

    def _assert_active_project(self, conn: Any, project_id: str, label: str = "project") -> None:
        """Raise when a project is missing or soft-deleted."""
        if not self._project_is_active(conn, project_id):
            raise ConflictError(f"{label.title()} {project_id!r} not found or soft-deleted")

    def _fetch_active_project_owned_row(
        self,
        conn: Any,
        table: str,
        entity_id: str,
        *,
        project_column: str = "project_id",
    ) -> dict[str, Any] | None:
        """Fetch a row only when its owning project is active."""
        row = conn.execute(
            f"SELECT * FROM {table} WHERE id = ? AND deleted = 0",  # nosec B608
            (entity_id,),
        ).fetchone()
        if row is None:
            return None
        if not self._project_is_active(conn, row[project_column]):
            return None
        return row

    def _require_active_project_owned_row(
        self,
        conn: Any,
        table: str,
        entity_id: str,
        *,
        entity_label: str,
        action: str,
        project_column: str = "project_id",
    ) -> dict[str, Any]:
        """Fetch an active descendant row or raise a conflict for writes."""
        row = self._fetch_active_project_owned_row(
            conn,
            table,
            entity_id,
            project_column=project_column,
        )
        if row is None:
            raise ConflictError(
                f"{entity_label} {entity_id!r} {action} failed (version conflict or not found).",
                entity=table,
                entity_id=entity_id,
            )
        return row

    def _assert_same_project(
        self,
        conn: Any,
        table: str,
        entity_id: str,
        expected_project_id: str,
        label: str = "entity",
    ) -> None:
        """Verify an entity belongs to the expected project.

        Raises :class:`InputError` on missing row, :class:`ConflictError` on mismatch.
        """
        if table not in self._PROJECT_CHECK_TABLES:
            raise InputError(f"Internal error: unknown table '{table}'")
        self._assert_active_project(conn, expected_project_id)
        row = conn.execute(
            f"SELECT project_id FROM {table} WHERE id = ? AND deleted = 0",  # nosec B608
            (entity_id,),
        ).fetchone()
        if row is None:
            raise InputError(f"{label} '{entity_id}' not found or deleted")
        if row["project_id"] != expected_project_id:
            raise ConflictError(f"{label} '{entity_id}' belongs to a different project")

    def _validate_plot_refs(
        self,
        conn: Any,
        project_id: str,
        *,
        plot_line_id: str | None = None,
        scene_id: str | None = None,
        chapter_id: str | None = None,
    ) -> None:
        """Validate that plot-related references share the same project
        and that scene belongs to chapter when both are supplied."""
        self._assert_active_project(conn, project_id)
        if plot_line_id:
            self._assert_same_project(conn, "manuscript_plot_lines", plot_line_id, project_id, "plot_line")
        if scene_id:
            self._assert_same_project(conn, "manuscript_scenes", scene_id, project_id, "scene")
        if chapter_id:
            self._assert_same_project(conn, "manuscript_chapters", chapter_id, project_id, "chapter")
        if scene_id and chapter_id:
            row = conn.execute(
                "SELECT chapter_id FROM manuscript_scenes WHERE id = ? AND deleted = 0",
                (scene_id,),
            ).fetchone()
            if row and row["chapter_id"] != chapter_id:
                raise ValueError(f"Scene '{scene_id}' does not belong to chapter '{chapter_id}'")

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    def create_project(
        self,
        title: str,
        *,
        subtitle: str | None = None,
        author: str | None = None,
        genre: str | None = None,
        status: str = "draft",
        synopsis: str | None = None,
        target_word_count: int | None = None,
        settings: dict[str, Any] | None = None,
        project_id: str | None = None,
    ) -> str:
        """Insert a new manuscript project and return its ID."""
        pid = project_id or self._uuid()
        now = self._now()
        settings_json = json.dumps(settings) if settings else "{}"

        if status not in _VALID_PROJECT_STATUSES:
            raise ValueError(f"Invalid project status: {status!r}")  # noqa: TRY003

        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO manuscript_projects
                    (id, title, subtitle, author, genre, status, synopsis,
                     target_word_count, word_count, settings_json,
                     created_at, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, 0, ?, 1)
                """,
                (
                    pid, title, subtitle, author, genre, status, synopsis,
                    target_word_count, settings_json,
                    now, now, self._client_id,
                ),
            )
        logger.debug("Created manuscript project {}", pid)
        return pid

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Fetch a single project by ID; returns *None* if missing or deleted."""
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM manuscript_projects WHERE id = ? AND deleted = 0",
                (project_id,),
            ).fetchone()
        return self._project_row_to_dict(row) if row else None

    def list_projects(
        self,
        *,
        status_filter: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Return ``(projects, total_count)`` with optional status filter."""
        with self.db.transaction() as conn:
            total_row = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                  FROM manuscript_projects
                 WHERE deleted = 0
                   AND (? IS NULL OR status = ?)
                """,
                (status_filter, status_filter),
            ).fetchone()
            total = total_row["cnt"] if total_row else 0

            rows = conn.execute(
                """
                SELECT *
                  FROM manuscript_projects
                 WHERE deleted = 0
                   AND (? IS NULL OR status = ?)
                 ORDER BY last_modified DESC
                 LIMIT ? OFFSET ?
                """,
                (status_filter, status_filter, limit, offset),
            ).fetchall()

        return [self._project_row_to_dict(r) for r in rows], int(total)

    def update_project(
        self,
        project_id: str,
        updates: dict[str, Any],
        expected_version: int,
    ) -> None:
        """Update a project with optimistic locking."""
        if not updates:
            return

        unknown = set(updates) - _UPDATABLE_PROJECT_COLS
        if unknown:
            raise ValueError(f"Unknown update column(s) for project: {unknown}")

        now = self._now()
        next_version = expected_version + 1

        set_parts: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            if key == "settings":
                col = "settings_json"
            else:
                col = key
            if col not in self._UPDATABLE_PROJECT_COLS:
                raise ValueError(f"Invalid update column for project: {key!r}")  # noqa: TRY003
            if key == "settings":
                set_parts.append("settings_json = ?")
                params.append(json.dumps(value))
            else:
                set_parts.append(f"{col} = ?")
                params.append(value)

        set_parts.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        params.extend([now, next_version, self._client_id])
        params.extend([project_id, expected_version])

        with self.db.transaction() as conn:
            cur = conn.execute(
                f"UPDATE manuscript_projects SET {', '.join(set_parts)} "  # nosec B608
                "WHERE id = ? AND version = ? AND deleted = 0",
                params,
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Project {project_id!r} update failed (version conflict or not found).",
                    entity="manuscript_projects",
                    entity_id=project_id,
                )

    def soft_delete_project(self, project_id: str, expected_version: int) -> None:
        """Soft-delete a project with optimistic locking."""
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            cur = conn.execute(
                "UPDATE manuscript_projects "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, project_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Project {project_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_projects",
                    entity_id=project_id,
                )

    # ------------------------------------------------------------------
    # Parts
    # ------------------------------------------------------------------

    def create_part(
        self,
        project_id: str,
        title: str,
        *,
        sort_order: float = 0,
        synopsis: str | None = None,
        part_id: str | None = None,
    ) -> str:
        """Insert a new part within a project; returns the part ID."""
        pid = part_id or self._uuid()
        now = self._now()

        with self.db.transaction() as conn:
            self._assert_active_project(conn, project_id)
            conn.execute(
                """
                INSERT INTO manuscript_parts
                    (id, project_id, title, sort_order, synopsis, word_count,
                     created_at, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?, 0, ?, 1)
                """,
                (pid, project_id, title, sort_order, synopsis, now, now, self._client_id),
            )
        logger.debug("Created manuscript part {} in project {}", pid, project_id)
        return pid

    def get_part(self, part_id: str) -> dict[str, Any] | None:
        """Fetch a part by ID; returns *None* if missing or deleted."""
        with self.db.transaction() as conn:
            row = self._fetch_active_project_owned_row(conn, "manuscript_parts", part_id)
        return dict(row) if row else None

    def list_parts(self, project_id: str) -> list[dict[str, Any]]:
        """List all non-deleted parts for a project ordered by sort_order."""
        with self.db.transaction() as conn:
            if not self._project_is_active(conn, project_id):
                return []
            rows = conn.execute(
                "SELECT * FROM manuscript_parts "
                "WHERE project_id = ? AND deleted = 0 ORDER BY sort_order",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_part(
        self,
        part_id: str,
        updates: dict[str, Any],
        expected_version: int,
    ) -> None:
        """Update a part with optimistic locking."""
        if not updates:
            return

        unknown = set(updates) - _UPDATABLE_PART_COLS
        if unknown:
            raise ValueError(f"Unknown update column(s) for part: {unknown}")

        now = self._now()
        next_version = expected_version + 1

        set_parts: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        set_parts.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        params.extend([now, next_version, self._client_id])
        params.extend([part_id, expected_version])

        with self.db.transaction() as conn:
            self._require_active_project_owned_row(
                conn,
                "manuscript_parts",
                part_id,
                entity_label="Part",
                action="update",
            )
            cur = conn.execute(
                f"UPDATE manuscript_parts SET {', '.join(set_parts)} "  # nosec B608
                "WHERE id = ? AND version = ? AND deleted = 0",
                params,
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Part {part_id!r} update failed (version conflict or not found).",
                    entity="manuscript_parts",
                    entity_id=part_id,
                )

    def soft_delete_part(self, part_id: str, expected_version: int) -> None:
        """Soft-delete a part with optimistic locking.

        Cascades the soft-delete to all child chapters and their scenes.
        """
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            part_row = self._require_active_project_owned_row(
                conn,
                "manuscript_parts",
                part_id,
                entity_label="Part",
                action="delete",
            )
            chapter_rows = conn.execute(
                "SELECT id FROM manuscript_chapters WHERE part_id = ? AND deleted = 0",
                (part_id,),
            ).fetchall()
            chapter_ids = [row["id"] for row in chapter_rows]

            cur = conn.execute(
                "UPDATE manuscript_parts "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, part_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Part {part_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_parts",
                    entity_id=part_id,
                )

            # Cascade to child chapters
            conn.execute(
                "UPDATE manuscript_chapters SET deleted = 1, last_modified = ?, client_id = ? "
                "WHERE part_id = ? AND deleted = 0",
                (now, self._client_id, part_id),
            )
            if chapter_ids:
                # Process in chunks to avoid hitting SQLite's SQLITE_MAX_VARIABLE_NUMBER limit (999).
                _BATCH = 900
                for i in range(0, len(chapter_ids), _BATCH):
                    batch = chapter_ids[i:i + _BATCH]
                    placeholders = ", ".join("?" for _ in batch)
                    conn.execute(
                        "UPDATE manuscript_scenes SET deleted = 1, last_modified = ?, client_id = ? "
                        f"WHERE chapter_id IN ({placeholders}) AND deleted = 0",  # nosec B608
                        (now, self._client_id, *batch),
                    )
                for chapter_id in chapter_ids:
                    self._mark_analyses_stale_in_txn(conn, "chapter", chapter_id)

            if part_row is not None:
                self._mark_analyses_stale_in_txn(conn, "project", part_row["project_id"])

    # ------------------------------------------------------------------
    # Chapters
    # ------------------------------------------------------------------

    def create_chapter(
        self,
        project_id: str,
        title: str,
        *,
        part_id: str | None = None,
        sort_order: float = 0,
        synopsis: str | None = None,
        status: str = "draft",
        chapter_id: str | None = None,
    ) -> str:
        """Insert a new chapter; returns its ID."""
        cid = chapter_id or self._uuid()
        now = self._now()

        if status not in _VALID_CHAPTER_STATUSES:
            raise ValueError(f"Invalid chapter status: {status!r}")  # noqa: TRY003

        with self.db.transaction() as conn:
            self._assert_active_project(conn, project_id)
            if part_id is not None:
                self._assert_same_project(conn, "manuscript_parts", part_id, project_id, "part")
            conn.execute(
                """
                INSERT INTO manuscript_chapters
                    (id, project_id, part_id, title, sort_order, synopsis,
                     pov_character_id, word_count, status,
                     created_at, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, ?, 0, ?, 1)
                """,
                (
                    cid, project_id, part_id, title, sort_order, synopsis,
                    status, now, now, self._client_id,
                ),
            )
        logger.debug("Created manuscript chapter {} in project {}", cid, project_id)
        return cid

    def get_chapter(self, chapter_id: str) -> dict[str, Any] | None:
        """Fetch a chapter by ID; returns *None* if missing or deleted."""
        with self.db.transaction() as conn:
            row = self._fetch_active_project_owned_row(conn, "manuscript_chapters", chapter_id)
        return dict(row) if row else None

    def list_chapters(
        self,
        project_id: str,
        *,
        part_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List non-deleted chapters, optionally filtered by part_id."""
        if part_id is not None:
            sql = (
                "SELECT * FROM manuscript_chapters "
                "WHERE project_id = ? AND part_id = ? AND deleted = 0 "
                "ORDER BY sort_order"
            )
            params: tuple[Any, ...] = (project_id, part_id)
        else:
            sql = (
                "SELECT * FROM manuscript_chapters "
                "WHERE project_id = ? AND deleted = 0 ORDER BY sort_order"
            )
            params = (project_id,)

        with self.db.transaction() as conn:
            if not self._project_is_active(conn, project_id):
                return []
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def update_chapter(
        self,
        chapter_id: str,
        updates: dict[str, Any],
        expected_version: int,
    ) -> None:
        """Update a chapter with optimistic locking."""
        if not updates:
            return

        unknown = set(updates) - _UPDATABLE_CHAPTER_COLS
        if unknown:
            raise ValueError(f"Unknown update column(s) for chapter: {unknown}")

        now = self._now()
        next_version = expected_version + 1

        set_parts: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        set_parts.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        params.extend([now, next_version, self._client_id])
        params.extend([chapter_id, expected_version])

        with self.db.transaction() as conn:
            current_row = self._require_active_project_owned_row(
                conn,
                "manuscript_chapters",
                chapter_id,
                entity_label="Chapter",
                action="update",
            )
            if "part_id" in updates and updates["part_id"] is not None:
                self._assert_same_project(
                    conn,
                    "manuscript_parts",
                    updates["part_id"],
                    current_row["project_id"],
                    "part",
                )
            cur = conn.execute(
                f"UPDATE manuscript_chapters SET {', '.join(set_parts)} "  # nosec B608
                "WHERE id = ? AND version = ? AND deleted = 0",
                params,
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Chapter {chapter_id!r} update failed (version conflict or not found).",
                    entity="manuscript_chapters",
                    entity_id=chapter_id,
                )

    def soft_delete_chapter(self, chapter_id: str, expected_version: int) -> None:
        """Soft-delete a chapter with optimistic locking.

        Cascades the soft-delete to all child scenes.
        """
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            chapter_row = self._require_active_project_owned_row(
                conn,
                "manuscript_chapters",
                chapter_id,
                entity_label="Chapter",
                action="delete",
            )
            cur = conn.execute(
                "UPDATE manuscript_chapters "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, chapter_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Chapter {chapter_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_chapters",
                    entity_id=chapter_id,
                )

            # Cascade to child scenes
            conn.execute(
                "UPDATE manuscript_scenes SET deleted = 1, last_modified = ?, client_id = ? "
                "WHERE chapter_id = ? AND deleted = 0",
                (now, self._client_id, chapter_id),
            )

            self._mark_analyses_stale_in_txn(conn, "chapter", chapter_id)
            if chapter_row is not None:
                self._mark_analyses_stale_in_txn(conn, "project", chapter_row["project_id"])

    # ------------------------------------------------------------------
    # Scenes
    # ------------------------------------------------------------------

    def create_scene(
        self,
        chapter_id: str,
        project_id: str,
        *,
        title: str = "Untitled Scene",
        content_json: str | None = None,
        content_plain: str = "",
        synopsis: str | None = None,
        sort_order: float = 0,
        status: str = "draft",
        scene_id: str | None = None,
    ) -> str:
        """Insert a new scene; returns its ID.

        After insertion the word counts for the chapter (and part / project)
        are propagated.
        """
        sid = scene_id or self._uuid()
        now = self._now()
        wc = _word_count(content_plain)

        if status not in _VALID_SCENE_STATUSES:
            raise ValueError(f"Invalid scene status: {status!r}")  # noqa: TRY003

        with self.db.transaction() as conn:
            self._assert_same_project(conn, "manuscript_chapters", chapter_id, project_id, "chapter")
            conn.execute(
                """
                INSERT INTO manuscript_scenes
                    (id, chapter_id, project_id, title, sort_order,
                     content_json, content_plain, synopsis, word_count,
                     pov_character_id, status,
                     created_at, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, 0, ?, 1)
                """,
                (
                    sid, chapter_id, project_id, title, sort_order,
                    content_json, content_plain, synopsis, wc,
                    status, now, now, self._client_id,
                ),
            )
            self._propagate_word_counts(conn, chapter_id, project_id)
            self._mark_scene_family_analyses_stale_in_txn(
                conn,
                chapter_id=chapter_id,
                project_id=project_id,
            )

        logger.debug("Created manuscript scene {} in chapter {}", sid, chapter_id)
        return sid

    def get_scene(self, scene_id: str) -> dict[str, Any] | None:
        """Fetch a scene by ID; returns *None* if missing or deleted.

        The returned dict has ``content_json`` deserialized into ``content``.
        """
        with self.db.transaction() as conn:
            row = self._fetch_active_project_owned_row(conn, "manuscript_scenes", scene_id)
        return self._scene_row_to_dict(dict(row)) if row else None

    def list_scenes(self, chapter_id: str) -> list[dict[str, Any]]:
        """List non-deleted scenes for a chapter ordered by sort_order."""
        with self.db.transaction() as conn:
            chapter_row = conn.execute(
                "SELECT project_id FROM manuscript_chapters WHERE id = ? AND deleted = 0",
                (chapter_id,),
            ).fetchone()
            if chapter_row is None or not self._project_is_active(conn, chapter_row["project_id"]):
                return []
            rows = conn.execute(
                "SELECT * FROM manuscript_scenes "
                "WHERE chapter_id = ? AND deleted = 0 ORDER BY sort_order",
                (chapter_id,),
            ).fetchall()
        return [self._scene_row_to_dict(dict(r)) for r in rows]

    def update_scene(
        self,
        scene_id: str,
        updates: dict[str, Any],
        expected_version: int,
    ) -> None:
        """Update a scene with optimistic locking.

        If ``content_plain`` is among the updates the ``word_count`` is
        recomputed automatically and word counts are propagated to parent
        entities.
        """
        if not updates:
            return

        unknown = set(updates) - _UPDATABLE_SCENE_COLS
        if unknown:
            raise ValueError(f"Unknown update column(s) for scene: {unknown}")

        now = self._now()
        next_version = expected_version + 1

        # If content_plain changed, recompute word count
        if "content_plain" in updates:
            updates["word_count"] = _word_count(updates["content_plain"])

        set_parts: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        set_parts.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        params.extend([now, next_version, self._client_id])
        params.extend([scene_id, expected_version])

        with self.db.transaction() as conn:
            current_row = self._require_active_project_owned_row(
                conn,
                "manuscript_scenes",
                scene_id,
                entity_label="Scene",
                action="update",
            )
            cur = conn.execute(
                f"UPDATE manuscript_scenes SET {', '.join(set_parts)} "  # nosec B608
                "WHERE id = ? AND version = ? AND deleted = 0",
                params,
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Scene {scene_id!r} update failed (version conflict or not found).",
                    entity="manuscript_scenes",
                    entity_id=scene_id,
                )

            # Propagate if the scene body changed.
            if "content_plain" in updates or "content_json" in updates:
                self._propagate_word_counts(conn, current_row["chapter_id"], current_row["project_id"])
                self._mark_scene_family_analyses_stale_in_txn(
                    conn,
                    scene_id=scene_id,
                    chapter_id=current_row["chapter_id"],
                    project_id=current_row["project_id"],
                )

    def soft_delete_scene(self, scene_id: str, expected_version: int) -> None:
        """Soft-delete a scene with optimistic locking; propagates word counts."""
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            row = self._require_active_project_owned_row(
                conn,
                "manuscript_scenes",
                scene_id,
                entity_label="Scene",
                action="delete",
            )

            cur = conn.execute(
                "UPDATE manuscript_scenes "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, scene_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Scene {scene_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_scenes",
                    entity_id=scene_id,
            )

            if row:
                self._propagate_word_counts(conn, row["chapter_id"], row["project_id"])
                self._mark_scene_family_analyses_stale_in_txn(
                    conn,
                    scene_id=scene_id,
                    chapter_id=row["chapter_id"],
                    project_id=row["project_id"],
                )

    def create_version(self, entity_type: str, entity_id: str, *, label: str | None = None) -> dict[str, Any]:
        """Create a manual manuscript/chapter/scene snapshot."""
        normalized_type = self._normalize_entity_type(entity_type)
        self._validate_version_entity_type(normalized_type)
        payload = self._version_payload_for(normalized_type, entity_id)
        version_id = self._uuid()
        now = self._now()
        with self.db.transaction() as conn:
            version_number = self._next_version_number(conn, normalized_type, entity_id)
            conn.execute(
                """
                INSERT INTO manuscript_versions (
                    id, entity_type, entity_id, project_id, version_number,
                    label, payload_json, created_at, client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    normalized_type,
                    entity_id,
                    payload["project_id"],
                    version_number,
                    label,
                    json.dumps(payload, sort_keys=True),
                    now,
                    self._client_id,
                ),
            )
        return self.get_version(normalized_type, entity_id, version_number)

    def list_versions(self, entity_type: str, entity_id: str) -> list[dict[str, Any]]:
        """List manual snapshots for a manuscript/chapter/scene."""
        normalized_type = self._normalize_entity_type(entity_type)
        self._validate_version_entity_type(normalized_type)
        with self.db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM manuscript_versions
                 WHERE entity_type = ? AND entity_id = ?
                 ORDER BY version_number DESC
                """,
                (normalized_type, entity_id),
            ).fetchall()
        return [self._version_row_to_dict(row) for row in rows]

    def get_version(self, entity_type: str, entity_id: str, version_number: int) -> dict[str, Any]:
        """Fetch a single manual snapshot."""
        normalized_type = self._normalize_entity_type(entity_type)
        self._validate_version_entity_type(normalized_type)
        with self.db.transaction() as conn:
            row = conn.execute(
                """
                SELECT * FROM manuscript_versions
                 WHERE entity_type = ? AND entity_id = ? AND version_number = ?
                """,
                (normalized_type, entity_id, version_number),
            ).fetchone()
        if row is None:
            raise InputError("manuscript version not found")
        return self._version_row_to_dict(row)

    def restore_version(
        self,
        entity_type: str,
        entity_id: str,
        version_number: int,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        """Restore a manual snapshot into the active working record."""
        normalized_type = self._normalize_entity_type(entity_type)
        version = self.get_version(normalized_type, entity_id, version_number)
        payload = version["payload"]
        with self.db.transaction() as conn:
            current = self._fetch_entity_row(conn, normalized_type, entity_id, deleted=False)
        resolved_expected = int(current["version"] if expected_version is None else expected_version)

        if normalized_type == "scene":
            if payload.get("chapter_id") != current.get("chapter_id"):
                raise ValueError("cannot restore scene version across chapters")
            updates = {
                "title": payload["title"],
                "sort_order": payload["sort_order"],
                "content_json": payload.get("content_json"),
                "content_plain": payload.get("content_plain") or "",
                "synopsis": payload.get("synopsis"),
                "status": payload.get("status") or "draft",
            }
            self.update_scene(entity_id, updates, resolved_expected)
            restored = self.get_scene(entity_id)
            if restored is None:
                raise InputError(f"scene '{entity_id}' not found after restore")
            return restored

        if normalized_type == "chapter":
            updates = {
                "title": payload["title"],
                "part_id": payload.get("part_id"),
                "sort_order": payload["sort_order"],
                "synopsis": payload.get("synopsis"),
                "status": payload.get("status") or "draft",
            }
            self.update_chapter(entity_id, updates, resolved_expected)
            restored = self.get_chapter(entity_id)
            if restored is None:
                raise InputError(f"chapter '{entity_id}' not found after restore")
            return restored

        updates = {
            "title": payload["title"],
            "sort_order": payload["sort_order"],
            "synopsis": payload.get("synopsis"),
        }
        self.update_part(entity_id, updates, resolved_expected)
        restored = self.get_part(entity_id)
        if restored is None:
            raise InputError(f"manuscript '{entity_id}' not found after restore")
        return restored

    def list_trash(self, *, entity_type: str | None = None) -> list[dict[str, Any]]:
        """List soft-deleted project/manuscript/chapter/scene records."""
        entities = (
            [(self._normalize_entity_type(entity_type), *self._validate_entity_type(entity_type))]
            if entity_type is not None
            else [
                (kind, *table_info)
                for kind, table_info in (
                    ("project", _MANUSCRIPT_ENTITY_TABLES["project"]),
                    ("manuscript", _MANUSCRIPT_ENTITY_TABLES["manuscript"]),
                    ("chapter", _MANUSCRIPT_ENTITY_TABLES["chapter"]),
                    ("scene", _MANUSCRIPT_ENTITY_TABLES["scene"]),
                )
            ]
        )
        records: list[dict[str, Any]] = []
        with self.db.transaction() as conn:
            for kind, table, _label in entities:
                rows = conn.execute(
                    f"SELECT * FROM {table} WHERE deleted = 1 ORDER BY last_modified DESC",  # nosec B608
                ).fetchall()
                for row in rows:
                    record = self._entity_row_to_dict(kind, row)
                    record["entity_type"] = kind
                    records.append(record)
        return records

    def restore_trash(
        self,
        entity_type: str,
        entity_id: str,
        *,
        expected_version: int | None = None,
    ) -> dict[str, Any]:
        """Restore a soft-deleted project/manuscript/chapter/scene record."""
        normalized_type = self._normalize_entity_type(entity_type)
        table, label = self._validate_entity_type(normalized_type)
        now = self._now()
        with self.db.transaction() as conn:
            row = self._fetch_entity_row(conn, normalized_type, entity_id, deleted=True)
            if expected_version is not None and int(row["version"]) != int(expected_version):
                raise ConflictError(
                    f"{label.title()} {entity_id!r} restore failed (version conflict).",
                    entity=table,
                    entity_id=entity_id,
                )
            try:
                if normalized_type in {"manuscript", "part"}:
                    self._fetch_entity_row(conn, "project", row["project_id"], deleted=False)
                elif normalized_type == "chapter":
                    self._fetch_entity_row(conn, "project", row["project_id"], deleted=False)
                    if row.get("part_id"):
                        parent = self._fetch_entity_row(conn, "part", row["part_id"], deleted=False)
                        if parent["project_id"] != row["project_id"]:
                            raise ConflictError(
                                f"{label.title()} {entity_id!r} restore failed (parent project mismatch).",
                                entity=table,
                                entity_id=entity_id,
                            )
                elif normalized_type == "scene":
                    parent = self._fetch_entity_row(conn, "chapter", row["chapter_id"], deleted=False)
                    if parent["project_id"] != row["project_id"]:
                        raise ConflictError(
                            f"{label.title()} {entity_id!r} restore failed (parent project mismatch).",
                            entity=table,
                            entity_id=entity_id,
                        )
            except InputError as exc:
                raise ConflictError(
                    f"{label.title()} {entity_id!r} restore failed (parent missing or deleted).",
                    entity=table,
                    entity_id=entity_id,
                ) from exc
            next_version = int(row["version"]) + 1
            cur = conn.execute(
                f"UPDATE {table} "  # nosec B608
                "SET deleted = 0, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND deleted = 1",
                (now, next_version, self._client_id, entity_id),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"{label.title()} {entity_id!r} restore failed (not found).",
                    entity=table,
                    entity_id=entity_id,
                )
            if normalized_type == "scene":
                self._propagate_word_counts(conn, row["chapter_id"], row["project_id"])
                self._mark_scene_family_analyses_stale_in_txn(
                    conn,
                    scene_id=entity_id,
                    chapter_id=row["chapter_id"],
                    project_id=row["project_id"],
                )
            restored = self._fetch_entity_row(conn, normalized_type, entity_id, deleted=False)
        return restored

    def get_all_scene_texts(self, project_id: str) -> list[str]:
        """Get all scene plain texts for a project in narrative order (single query)."""
        with self.db.transaction() as conn:
            if not self._project_is_active(conn, project_id):
                return []
            cur = conn.execute(
                "SELECT s.content_plain "
                "FROM manuscript_scenes s "
                "JOIN manuscript_chapters c ON c.id = s.chapter_id AND c.deleted = 0 "
                "LEFT JOIN manuscript_parts p ON p.id = c.part_id AND p.deleted = 0 "
                "WHERE s.project_id = ? AND s.deleted = 0 "
                "ORDER BY COALESCE(p.sort_order, -1), c.sort_order, s.sort_order",
                (project_id,),
            )
            return [row["content_plain"] for row in cur.fetchall() if row["content_plain"]]

    # ------------------------------------------------------------------
    # Word-count propagation
    # ------------------------------------------------------------------

    def _propagate_word_counts(self, conn: sqlite3.Connection, chapter_id: str, project_id: str) -> None:
        """Cascade word counts: scenes -> chapter -> part (if any) -> project.

        Must be called inside an existing transaction (receives the connection).
        Bumps ``version`` and ``client_id`` on each parent so that optimistic
        locking and sync triggers stay consistent with other mutations.
        """
        now = self._now()

        # 1. Chapter word count = SUM of its non-deleted scenes
        ch_wc_row = conn.execute(
            "SELECT COALESCE(SUM(word_count), 0) AS wc "
            "FROM manuscript_scenes WHERE chapter_id = ? AND deleted = 0",
            (chapter_id,),
        ).fetchone()
        ch_wc = int(ch_wc_row["wc"]) if ch_wc_row else 0

        conn.execute(
            "UPDATE manuscript_chapters "
            "SET word_count = ?, last_modified = ?, version = version + 1, client_id = ? "
            "WHERE id = ?",
            (ch_wc, now, self._client_id, chapter_id),
        )

        # 2. Determine if the chapter belongs to a part
        ch_row = conn.execute(
            "SELECT part_id FROM manuscript_chapters WHERE id = ?",
            (chapter_id,),
        ).fetchone()
        part_id = ch_row["part_id"] if ch_row else None

        if part_id:
            # Part word count = SUM of its non-deleted chapters
            part_wc_row = conn.execute(
                "SELECT COALESCE(SUM(word_count), 0) AS wc "
                "FROM manuscript_chapters WHERE part_id = ? AND deleted = 0",
                (part_id,),
            ).fetchone()
            part_wc = int(part_wc_row["wc"]) if part_wc_row else 0

            conn.execute(
                "UPDATE manuscript_parts "
                "SET word_count = ?, last_modified = ?, version = version + 1, client_id = ? "
                "WHERE id = ?",
                (part_wc, now, self._client_id, part_id),
            )

        # 3. Project word count = SUM of its non-deleted scenes
        #    (authoritative count from scenes, not double-counting via chapters/parts)
        proj_wc_row = conn.execute(
            "SELECT COALESCE(SUM(word_count), 0) AS wc "
            "FROM manuscript_scenes WHERE project_id = ? AND deleted = 0",
            (project_id,),
        ).fetchone()
        proj_wc = int(proj_wc_row["wc"]) if proj_wc_row else 0

        conn.execute(
            "UPDATE manuscript_projects "
            "SET word_count = ?, last_modified = ?, version = version + 1, client_id = ? "
            "WHERE id = ?",
            (proj_wc, now, self._client_id, project_id),
        )

    # ------------------------------------------------------------------
    # Project structure
    # ------------------------------------------------------------------

    def get_project_structure(self, project_id: str) -> dict[str, Any]:
        """Build a hierarchical view of the project.

        Returns::

            {
                "project_id": "...",
                "parts": [
                    {"id": ..., "title": ..., "chapters": [
                        {"id": ..., "title": ..., "scenes": [...]},
                    ]},
                ],
                "unassigned_chapters": [
                    {"id": ..., "title": ..., "scenes": [...]},
                ],
            }
        """
        with self.db.transaction() as conn:
            parts = conn.execute(
                "SELECT * FROM manuscript_parts "
                "WHERE project_id = ? AND deleted = 0 ORDER BY sort_order",
                (project_id,),
            ).fetchall()

            chapters = conn.execute(
                "SELECT * FROM manuscript_chapters "
                "WHERE project_id = ? AND deleted = 0 ORDER BY sort_order",
                (project_id,),
            ).fetchall()

            scenes = conn.execute(
                "SELECT * FROM manuscript_scenes "
                "WHERE project_id = ? AND deleted = 0 ORDER BY sort_order",
                (project_id,),
            ).fetchall()

        # Index scenes by chapter_id
        scenes_by_chapter: dict[str, list[dict[str, Any]]] = {}
        for s in scenes:
            sd = dict(s)
            scenes_by_chapter.setdefault(sd["chapter_id"], []).append(sd)

        # Index chapters by part_id
        chapters_by_part: dict[str | None, list[dict[str, Any]]] = {}
        for c in chapters:
            cd = dict(c)
            cd["scenes"] = scenes_by_chapter.get(cd["id"], [])
            chapters_by_part.setdefault(cd["part_id"], []).append(cd)

        result_parts = []
        for p in parts:
            pd = dict(p)
            pd["chapters"] = chapters_by_part.get(pd["id"], [])
            result_parts.append(pd)

        return {
            "project_id": project_id,
            "parts": result_parts,
            "unassigned_chapters": chapters_by_part.get(None, []),
        }

    # ------------------------------------------------------------------
    # FTS search
    # ------------------------------------------------------------------

    def search_scenes(
        self,
        project_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Full-text search across scene titles, plain content, and synopses.

        Returns matching scenes with FTS5 ``snippet()`` highlights.

        .. note::
            FTS5 is SQLite-only.  A ``NotImplementedError`` is raised if
            the underlying database does not have the expected FTS table.
        """
        # Escape FTS5 special characters by wrapping each term in double quotes
        escaped_query = " ".join(
            f'"{term.replace(chr(34), chr(34)*2)}"' for term in query.split() if term
        )
        if not escaped_query:
            return []

        with self.db.transaction() as conn:
            if not self._project_is_active(conn, project_id):
                return []
            try:
                rows = conn.execute(
                    """
                    SELECT s.*,
                           snippet(manuscript_scenes_fts, 1, '<b>', '</b>', '...', 32) AS snippet
                    FROM manuscript_scenes_fts AS fts
                    JOIN manuscript_scenes AS s ON s.rowid = fts.rowid
                    WHERE manuscript_scenes_fts MATCH ?
                      AND s.project_id = ?
                      AND s.deleted = 0
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (escaped_query, project_id, limit),
                ).fetchall()
            except Exception as exc:
                if "no such table" in str(exc).lower():
                    raise NotImplementedError(
                        "Full-text search is only supported on SQLite (FTS5)"
                    ) from exc
                raise
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Reorder
    # ------------------------------------------------------------------

    def reorder_items(
        self,
        entity_type: str,
        items: list[dict[str, Any]],
        *,
        project_id: str | None = None,
    ) -> None:
        """Batch-update ``sort_order`` (and optionally ``part_id`` for chapters).

        Parameters
        ----------
        entity_type:
            One of ``"part"``, ``"chapter"``, ``"scene"``.
        items:
            A list of dicts, each containing at minimum ``"id"``,
            ``"sort_order"``, and ``"version"``.  For chapters, an optional
            ``"part_id"`` can be included to reparent.  The ``"version"``
            field enables optimistic locking per item.
        project_id:
            When provided, every item is validated to belong to this project
            before updating.
        """
        table = _REORDER_ENTITY_TABLES.get(entity_type)
        if table is None:
            raise ValueError(  # noqa: TRY003
                f"Invalid entity_type {entity_type!r}; "
                f"must be one of {sorted(_REORDER_ENTITY_TABLES)}"
            )

        now = self._now()

        with self.db.transaction() as conn:
            if project_id is not None:
                self._assert_active_project(conn, project_id)
            current_rows: list[dict[str, Any]] = []
            for item in items:
                row = conn.execute(
                    f"SELECT * FROM {table} WHERE id = ? AND deleted = 0",  # nosec B608
                    (item["id"],),
                ).fetchone()
                if row is None:
                    raise ValueError(f"{entity_type} {item['id']!r} not found")
                if project_id is not None and row["project_id"] != project_id:
                    raise ValueError(
                        f"{entity_type} {item['id']!r} does not belong to project {project_id!r}"
                    )
                if (
                    entity_type == "chapter"
                    and "part_id" in item
                    and item["part_id"] is not None
                ):
                    effective_project_id = project_id or row["project_id"]
                    self._assert_same_project(
                        conn,
                        "manuscript_parts",
                        item["part_id"],
                        effective_project_id,
                        "part",
                    )
                current_rows.append(dict(row))

            if entity_type == "chapter":
                for item, row in zip(items, current_rows, strict=True):
                    if "part_id" not in item or item["part_id"] is None:
                        continue
                    self._assert_same_project(
                        conn,
                        "manuscript_parts",
                        item["part_id"],
                        row["project_id"],
                        "part",
                    )

            stale_project_ids: set[str] = set()
            stale_chapter_ids: set[str] = set()

            for item, row in zip(items, current_rows, strict=True):
                item_id = item["id"]
                sort_order = item["sort_order"]
                expected_version = item.get("version")

                version_clause = " AND version = ?" if expected_version is not None else ""
                version_params = (expected_version,) if expected_version is not None else ()

                if entity_type == "chapter" and "part_id" in item:
                    cur = conn.execute(
                        f"UPDATE {table} SET sort_order = ?, part_id = ?, "  # nosec B608
                        "last_modified = ?, version = version + 1 "
                        f"WHERE id = ? AND deleted = 0{version_clause}",
                        (sort_order, item["part_id"], now, item_id) + version_params,
                    )
                else:
                    cur = conn.execute(
                        f"UPDATE {table} SET sort_order = ?, "  # nosec B608
                        "last_modified = ?, version = version + 1 "
                        f"WHERE id = ? AND deleted = 0{version_clause}",
                        (sort_order, now, item_id) + version_params,
                    )

                if expected_version is not None and cur.rowcount == 0:
                    raise ConflictError(
                        f"{entity_type.title()} {item_id!r} reorder failed "
                        f"(version conflict or not found).",
                        entity=table,
                        entity_id=item_id,
                    )

                if entity_type == "scene":
                    stale_chapter_ids.add(row["chapter_id"])
                    stale_project_ids.add(row["project_id"])
                else:
                    stale_project_ids.add(row["project_id"])

            for chapter_id in stale_chapter_ids:
                self._mark_analyses_stale_in_txn(conn, "chapter", chapter_id)
            for project_id in stale_project_ids:
                self._mark_analyses_stale_in_txn(conn, "project", project_id)

    # ==================================================================
    # Characters
    # ==================================================================

    def create_character(
        self,
        project_id: str,
        name: str,
        *,
        role: str = "supporting",
        cast_group: str | None = None,
        full_name: str | None = None,
        age: str | None = None,
        gender: str | None = None,
        appearance: str | None = None,
        personality: str | None = None,
        backstory: str | None = None,
        motivation: str | None = None,
        arc_summary: str | None = None,
        notes: str | None = None,
        custom_fields: dict[str, Any] | None = None,
        sort_order: float = 0,
        character_id: str | None = None,
    ) -> str:
        """Insert a new character and return its ID."""
        cid = character_id or self._uuid()
        now = self._now()
        cf_json = json.dumps(custom_fields) if custom_fields else "{}"

        with self.db.transaction() as conn:
            self._assert_active_project(conn, project_id)
            conn.execute(
                """
                INSERT INTO manuscript_characters
                    (id, project_id, name, role, cast_group, full_name, age, gender,
                     appearance, personality, backstory, motivation, arc_summary,
                     notes, custom_fields_json, sort_order,
                     created_at, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1)
                """,
                (
                    cid, project_id, name, role, cast_group, full_name, age, gender,
                    appearance, personality, backstory, motivation, arc_summary,
                    notes, cf_json, sort_order,
                    now, now, self._client_id,
                ),
            )
            self._mark_analyses_stale_in_txn(conn, "project", project_id)
        logger.debug("Created manuscript character {} in project {}", cid, project_id)
        return cid

    def get_character(self, character_id: str) -> dict[str, Any] | None:
        """Fetch a character by ID; returns *None* if missing or deleted."""
        with self.db.transaction() as conn:
            row = self._fetch_active_project_owned_row(conn, "manuscript_characters", character_id)
        if not row:
            return None
        d = dict(row)
        d["custom_fields"] = json.loads(d.pop("custom_fields_json", "{}"))
        return d

    def list_characters(
        self,
        project_id: str,
        *,
        role_filter: str | None = None,
        cast_group_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """List non-deleted characters for a project, optionally filtered."""
        with self.db.transaction() as conn:
            if not self._project_is_active(conn, project_id):
                return []
            cur = conn.execute(
                """
                SELECT *
                  FROM manuscript_characters
                 WHERE project_id = ?
                   AND deleted = 0
                   AND (? IS NULL OR role = ?)
                   AND (? IS NULL OR cast_group = ?)
                 ORDER BY sort_order
                """,
                (
                    project_id,
                    role_filter,
                    role_filter,
                    cast_group_filter,
                    cast_group_filter,
                ),
            )
            rows = cur.fetchall()

        results = []
        for r in rows:
            d = dict(r)
            d["custom_fields"] = json.loads(d.pop("custom_fields_json", "{}"))
            results.append(d)
        return results

    def update_character(
        self,
        character_id: str,
        updates: dict[str, Any],
        expected_version: int,
    ) -> None:
        """Update a character with optimistic locking."""
        if not updates:
            return

        unknown = set(updates) - _UPDATABLE_CHARACTER_COLS
        if unknown:
            raise ValueError(f"Unknown update column(s) for character: {unknown}")

        now = self._now()
        next_version = expected_version + 1
        should_stale = "name" in updates or "role" in updates

        if "custom_fields" in updates:
            updates["custom_fields_json"] = json.dumps(updates.pop("custom_fields"))

        with self.db.transaction() as conn:
            current_row = self._require_active_project_owned_row(
                conn,
                "manuscript_characters",
                character_id,
                entity_label="Character",
                action="update",
            )

            current = dict(current_row)
            current.update(updates)
            cur = conn.execute(
                """
                UPDATE manuscript_characters
                   SET name = ?,
                       role = ?,
                       cast_group = ?,
                       full_name = ?,
                       age = ?,
                       gender = ?,
                       appearance = ?,
                       personality = ?,
                       backstory = ?,
                       motivation = ?,
                       arc_summary = ?,
                       notes = ?,
                       custom_fields_json = ?,
                       sort_order = ?,
                       last_modified = ?,
                       version = ?,
                       client_id = ?
                 WHERE id = ? AND version = ? AND deleted = 0
                """,
                (
                    current["name"],
                    current["role"],
                    current["cast_group"],
                    current["full_name"],
                    current["age"],
                    current["gender"],
                    current["appearance"],
                    current["personality"],
                    current["backstory"],
                    current["motivation"],
                    current["arc_summary"],
                    current["notes"],
                    current["custom_fields_json"],
                    current["sort_order"],
                    now,
                    next_version,
                    self._client_id,
                    character_id,
                    expected_version,
                ),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Character {character_id!r} update failed (version conflict or not found).",
                    entity="manuscript_characters",
                    entity_id=character_id,
                )
            if should_stale:
                self._mark_analyses_stale_in_txn(conn, "project", current_row["project_id"])

    def soft_delete_character(self, character_id: str, expected_version: int) -> None:
        """Soft-delete a character with optimistic locking."""
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            row = self._require_active_project_owned_row(
                conn,
                "manuscript_characters",
                character_id,
                entity_label="Character",
                action="delete",
            )
            cur = conn.execute(
                "UPDATE manuscript_characters "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, character_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Character {character_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_characters",
                    entity_id=character_id,
                )
            if row is not None:
                self._mark_analyses_stale_in_txn(conn, "project", row["project_id"])

    # ==================================================================
    # Character Relationships
    # ==================================================================

    def create_relationship(
        self,
        project_id: str,
        from_character_id: str,
        to_character_id: str,
        relationship_type: str,
        *,
        description: str | None = None,
        bidirectional: bool = True,
        relationship_id: str | None = None,
    ) -> str:
        """Insert a character relationship and return its ID."""
        rid = relationship_id or self._uuid()
        now = self._now()

        with self.db.transaction() as conn:
            self._assert_active_project(conn, project_id)
            self._assert_same_project(
                conn, "manuscript_characters", from_character_id, project_id, "from_character"
            )
            self._assert_same_project(
                conn, "manuscript_characters", to_character_id, project_id, "to_character"
            )
            conn.execute(
                """
                INSERT INTO manuscript_character_relationships
                    (id, project_id, from_character_id, to_character_id,
                     relationship_type, description, bidirectional,
                     created_at, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1)
                """,
                (
                    rid, project_id, from_character_id, to_character_id,
                    relationship_type, description, int(bidirectional),
                    now, now, self._client_id,
                ),
            )
        logger.debug("Created relationship {} in project {}", rid, project_id)
        return rid

    def get_relationship(self, relationship_id: str) -> dict[str, Any] | None:
        """Fetch a relationship by ID; returns *None* if missing or deleted."""
        with self.db.transaction() as conn:
            row = self._fetch_active_project_owned_row(
                conn, "manuscript_character_relationships", relationship_id
            )
        return dict(row) if row else None

    def list_relationships(self, project_id: str) -> list[dict[str, Any]]:
        """List non-deleted relationships for a project."""
        with self.db.transaction() as conn:
            if not self._project_is_active(conn, project_id):
                return []
            rows = conn.execute(
                "SELECT * FROM manuscript_character_relationships "
                "WHERE project_id = ? AND deleted = 0",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def soft_delete_relationship(self, relationship_id: str, expected_version: int) -> None:
        """Soft-delete a relationship with optimistic locking."""
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            self._require_active_project_owned_row(
                conn,
                "manuscript_character_relationships",
                relationship_id,
                entity_label="Relationship",
                action="delete",
            )
            cur = conn.execute(
                "UPDATE manuscript_character_relationships "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, relationship_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Relationship {relationship_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_character_relationships",
                    entity_id=relationship_id,
                )

    # ==================================================================
    # Scene-Character Linking
    # ==================================================================

    def link_scene_character(
        self,
        scene_id: str,
        character_id: str,
        *,
        is_pov: bool = False,
    ) -> None:
        """Link a character to a scene (upsert — updates ``is_pov`` on conflict)."""
        with self.db.transaction() as conn:
            # Verify both entities belong to the same project.
            scene_row = conn.execute(
                "SELECT project_id FROM manuscript_scenes WHERE id = ? AND deleted = 0",
                (scene_id,),
            ).fetchone()
            if scene_row is None:
                raise ValueError(f"Scene '{scene_id}' not found or deleted")
            self._assert_active_project(conn, scene_row["project_id"])
            self._assert_same_project(
                conn, "manuscript_characters", character_id, scene_row["project_id"], "character"
            )
            conn.execute(
                "INSERT INTO manuscript_scene_characters "
                "(scene_id, character_id, is_pov, last_modified, client_id, version) "
                "VALUES (?, ?, ?, ?, ?, 1) "
                "ON CONFLICT(scene_id, character_id) DO UPDATE SET "
                "is_pov = excluded.is_pov, last_modified = excluded.last_modified, "
                "client_id = excluded.client_id, version = version + 1",
                (scene_id, character_id, int(is_pov), self._now(), self._client_id),
            )

    def unlink_scene_character(self, scene_id: str, character_id: str) -> None:
        """Remove a character-scene link."""
        with self.db.transaction() as conn:
            scene_row = conn.execute(
                "SELECT project_id FROM manuscript_scenes WHERE id = ? AND deleted = 0",
                (scene_id,),
            ).fetchone()
            if scene_row is not None and not self._project_is_active(conn, scene_row["project_id"]):
                raise ConflictError(f"Project {scene_row['project_id']!r} not found or soft-deleted")
            conn.execute(
                "DELETE FROM manuscript_scene_characters "
                "WHERE scene_id = ? AND character_id = ?",
                (scene_id, character_id),
            )

    def list_scene_characters(self, scene_id: str) -> list[dict[str, Any]]:
        """List characters linked to a scene, including name and role."""
        with self.db.transaction() as conn:
            scene_row = conn.execute(
                "SELECT project_id FROM manuscript_scenes WHERE id = ? AND deleted = 0",
                (scene_id,),
            ).fetchone()
            if scene_row is None or not self._project_is_active(conn, scene_row["project_id"]):
                return []
            rows = conn.execute(
                "SELECT sc.scene_id, sc.character_id, sc.is_pov, "
                "       c.name, c.role "
                "FROM manuscript_scene_characters sc "
                "JOIN manuscript_characters c ON c.id = sc.character_id AND c.deleted = 0 "
                "WHERE sc.scene_id = ?",
                (scene_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ==================================================================
    # World Info
    # ==================================================================

    def create_world_info(
        self,
        project_id: str,
        kind: str,
        name: str,
        *,
        description: str | None = None,
        parent_id: str | None = None,
        properties: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        sort_order: float = 0,
        world_info_id: str | None = None,
    ) -> str:
        """Insert a world-info entry and return its ID."""
        wid = world_info_id or self._uuid()
        now = self._now()
        props_json = json.dumps(properties) if properties else "{}"
        tags_json = json.dumps(tags) if tags else "[]"

        with self.db.transaction() as conn:
            self._assert_active_project(conn, project_id)
            if parent_id:
                self._assert_same_project(
                    conn, "manuscript_world_info", parent_id, project_id, "parent_world_info"
                )
            conn.execute(
                """
                INSERT INTO manuscript_world_info
                    (id, project_id, kind, name, description, parent_id,
                     properties_json, tags_json, sort_order,
                     created_at, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1)
                """,
                (
                    wid, project_id, kind, name, description, parent_id,
                    props_json, tags_json, sort_order,
                    now, now, self._client_id,
                ),
            )
            self._mark_analyses_stale_in_txn(conn, "project", project_id)
        logger.debug("Created world info {} in project {}", wid, project_id)
        return wid

    def get_world_info(self, world_info_id: str) -> dict[str, Any] | None:
        """Fetch a world-info entry by ID; returns *None* if missing or deleted."""
        with self.db.transaction() as conn:
            row = self._fetch_active_project_owned_row(
                conn, "manuscript_world_info", world_info_id
            )
        if not row:
            return None
        d = dict(row)
        d["properties"] = json.loads(d.pop("properties_json", "{}"))
        d["tags"] = json.loads(d.pop("tags_json", "[]"))
        return d

    def list_world_info(
        self,
        project_id: str,
        *,
        kind_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """List non-deleted world-info entries for a project."""
        where = "project_id = ? AND deleted = 0"
        params: list[Any] = [project_id]
        if kind_filter:
            where += " AND kind = ?"
            params.append(kind_filter)

        with self.db.transaction() as conn:
            if not self._project_is_active(conn, project_id):
                return []
            rows = conn.execute(
                f"SELECT * FROM manuscript_world_info WHERE {where} ORDER BY sort_order",  # nosec B608
                params,
            ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            d["properties"] = json.loads(d.pop("properties_json", "{}"))
            d["tags"] = json.loads(d.pop("tags_json", "[]"))
            results.append(d)
        return results

    def update_world_info(
        self,
        world_info_id: str,
        updates: dict[str, Any],
        expected_version: int,
    ) -> None:
        """Update a world-info entry with optimistic locking."""
        if not updates:
            return

        unknown = set(updates) - _UPDATABLE_WORLD_INFO_COLS
        if unknown:
            raise ValueError(f"Unknown update column(s) for world_info: {unknown}")

        now = self._now()
        next_version = expected_version + 1
        should_stale = "name" in updates or "kind" in updates

        if "properties" in updates:
            updates["properties_json"] = json.dumps(updates.pop("properties"))
        if "tags" in updates:
            updates["tags_json"] = json.dumps(updates.pop("tags"))

        set_parts: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        set_parts.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        params.extend([now, next_version, self._client_id])
        params.extend([world_info_id, expected_version])

        with self.db.transaction() as conn:
            current_row = self._require_active_project_owned_row(
                conn,
                "manuscript_world_info",
                world_info_id,
                entity_label="WorldInfo",
                action="update",
            )
            if "parent_id" in updates and updates["parent_id"] is not None:
                self._assert_same_project(
                    conn,
                    "manuscript_world_info",
                    updates["parent_id"],
                    current_row["project_id"],
                    "parent_world_info",
                )
            cur = conn.execute(
                f"UPDATE manuscript_world_info SET {', '.join(set_parts)} "  # nosec B608
                "WHERE id = ? AND version = ? AND deleted = 0",
                params,
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"WorldInfo {world_info_id!r} update failed (version conflict or not found).",
                    entity="manuscript_world_info",
                    entity_id=world_info_id,
                )
            if should_stale:
                self._mark_analyses_stale_in_txn(conn, "project", current_row["project_id"])

    def soft_delete_world_info(self, world_info_id: str, expected_version: int) -> None:
        """Soft-delete a world-info entry with optimistic locking."""
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            row = self._require_active_project_owned_row(
                conn,
                "manuscript_world_info",
                world_info_id,
                entity_label="WorldInfo",
                action="delete",
            )
            cur = conn.execute(
                "UPDATE manuscript_world_info "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, world_info_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"WorldInfo {world_info_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_world_info",
                    entity_id=world_info_id,
                )
            if row is not None:
                self._mark_analyses_stale_in_txn(conn, "project", row["project_id"])

    # ==================================================================
    # Scene-World Info Linking
    # ==================================================================

    def link_scene_world_info(self, scene_id: str, world_info_id: str) -> None:
        """Link a world-info entry to a scene (INSERT OR IGNORE)."""
        with self.db.transaction() as conn:
            # Verify both entities belong to the same project.
            scene_row = conn.execute(
                "SELECT project_id FROM manuscript_scenes WHERE id = ? AND deleted = 0",
                (scene_id,),
            ).fetchone()
            if scene_row is None:
                raise ValueError(f"Scene '{scene_id}' not found or deleted")
            self._assert_active_project(conn, scene_row["project_id"])
            self._assert_same_project(
                conn, "manuscript_world_info", world_info_id, scene_row["project_id"], "world_info"
            )
            conn.execute(
                "INSERT INTO manuscript_scene_world_info "
                "(scene_id, world_info_id, last_modified, client_id, version) "
                "VALUES (?, ?, ?, ?, 1) "
                "ON CONFLICT(scene_id, world_info_id) DO UPDATE SET "
                "last_modified = excluded.last_modified, "
                "client_id = excluded.client_id, version = version + 1",
                (scene_id, world_info_id, self._now(), self._client_id),
            )

    def unlink_scene_world_info(self, scene_id: str, world_info_id: str) -> None:
        """Remove a world-info-scene link."""
        with self.db.transaction() as conn:
            scene_row = conn.execute(
                "SELECT project_id FROM manuscript_scenes WHERE id = ? AND deleted = 0",
                (scene_id,),
            ).fetchone()
            if scene_row is not None and not self._project_is_active(conn, scene_row["project_id"]):
                raise ConflictError(f"Project {scene_row['project_id']!r} not found or soft-deleted")
            conn.execute(
                "DELETE FROM manuscript_scene_world_info "
                "WHERE scene_id = ? AND world_info_id = ?",
                (scene_id, world_info_id),
            )

    def list_scene_world_info(self, scene_id: str) -> list[dict[str, Any]]:
        """List world-info entries linked to a scene, including name and kind."""
        with self.db.transaction() as conn:
            scene_row = conn.execute(
                "SELECT project_id FROM manuscript_scenes WHERE id = ? AND deleted = 0",
                (scene_id,),
            ).fetchone()
            if scene_row is None or not self._project_is_active(conn, scene_row["project_id"]):
                return []
            rows = conn.execute(
                "SELECT sw.scene_id, sw.world_info_id, "
                "       w.name, w.kind "
                "FROM manuscript_scene_world_info sw "
                "JOIN manuscript_world_info w ON w.id = sw.world_info_id AND w.deleted = 0 "
                "WHERE sw.scene_id = ?",
                (scene_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ==================================================================
    # Plot Lines
    # ==================================================================

    def create_plot_line(
        self,
        project_id: str,
        title: str,
        *,
        description: str | None = None,
        status: str = "active",
        color: str | None = None,
        sort_order: float = 0,
        plot_line_id: str | None = None,
    ) -> str:
        """Insert a new plot line and return its ID."""
        pid = plot_line_id or self._uuid()
        now = self._now()

        with self.db.transaction() as conn:
            self._assert_active_project(conn, project_id)
            conn.execute(
                """
                INSERT INTO manuscript_plot_lines
                    (id, project_id, title, description, status, color, sort_order,
                     created_at, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1)
                """,
                (
                    pid, project_id, title, description, status, color, sort_order,
                    now, now, self._client_id,
                ),
            )
        logger.debug("Created plot line {} in project {}", pid, project_id)
        return pid

    def get_plot_line(self, plot_line_id: str) -> dict[str, Any] | None:
        """Fetch a plot line by ID; returns *None* if missing or deleted."""
        with self.db.transaction() as conn:
            row = self._fetch_active_project_owned_row(
                conn, "manuscript_plot_lines", plot_line_id
            )
        return dict(row) if row else None

    def list_plot_lines(self, project_id: str) -> list[dict[str, Any]]:
        """List non-deleted plot lines for a project ordered by sort_order."""
        with self.db.transaction() as conn:
            if not self._project_is_active(conn, project_id):
                return []
            rows = conn.execute(
                "SELECT * FROM manuscript_plot_lines "
                "WHERE project_id = ? AND deleted = 0 ORDER BY sort_order",
                (project_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_plot_line(
        self,
        plot_line_id: str,
        updates: dict[str, Any],
        expected_version: int,
    ) -> None:
        """Update a plot line with optimistic locking."""
        if not updates:
            return

        unknown = set(updates) - _UPDATABLE_PLOT_LINE_COLS
        if unknown:
            raise ValueError(f"Unknown update column(s) for plot_line: {unknown}")

        now = self._now()
        next_version = expected_version + 1

        set_parts: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        set_parts.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        params.extend([now, next_version, self._client_id])
        params.extend([plot_line_id, expected_version])

        with self.db.transaction() as conn:
            self._require_active_project_owned_row(
                conn,
                "manuscript_plot_lines",
                plot_line_id,
                entity_label="PlotLine",
                action="update",
            )
            cur = conn.execute(
                f"UPDATE manuscript_plot_lines SET {', '.join(set_parts)} "  # nosec B608
                "WHERE id = ? AND version = ? AND deleted = 0",
                params,
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"PlotLine {plot_line_id!r} update failed (version conflict or not found).",
                    entity="manuscript_plot_lines",
                    entity_id=plot_line_id,
                )

    def soft_delete_plot_line(self, plot_line_id: str, expected_version: int) -> None:
        """Soft-delete a plot line with optimistic locking."""
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            self._require_active_project_owned_row(
                conn,
                "manuscript_plot_lines",
                plot_line_id,
                entity_label="PlotLine",
                action="delete",
            )
            cur = conn.execute(
                "UPDATE manuscript_plot_lines "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, plot_line_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"PlotLine {plot_line_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_plot_lines",
                    entity_id=plot_line_id,
                )

    # ==================================================================
    # Plot Events
    # ==================================================================

    def create_plot_event(
        self,
        project_id: str,
        plot_line_id: str,
        title: str,
        *,
        description: str | None = None,
        scene_id: str | None = None,
        chapter_id: str | None = None,
        event_type: str = "plot",
        sort_order: float = 0,
        event_id: str | None = None,
    ) -> str:
        """Insert a new plot event and return its ID."""
        eid = event_id or self._uuid()
        now = self._now()

        with self.db.transaction() as conn:
            self._validate_plot_refs(
                conn, project_id,
                plot_line_id=plot_line_id, scene_id=scene_id, chapter_id=chapter_id,
            )
            conn.execute(
                """
                INSERT INTO manuscript_plot_events
                    (id, project_id, plot_line_id, scene_id, chapter_id,
                     title, description, event_type, sort_order,
                     created_at, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1)
                """,
                (
                    eid, project_id, plot_line_id, scene_id, chapter_id,
                    title, description, event_type, sort_order,
                    now, now, self._client_id,
                ),
            )
        logger.debug("Created plot event {} for plot line {}", eid, plot_line_id)
        return eid

    def get_plot_event(self, event_id: str) -> dict[str, Any] | None:
        """Fetch a plot event by ID; returns *None* if missing or deleted."""
        with self.db.transaction() as conn:
            row = self._fetch_active_project_owned_row(
                conn, "manuscript_plot_events", event_id
            )
        return dict(row) if row else None

    def list_plot_events(self, plot_line_id: str) -> list[dict[str, Any]]:
        """List non-deleted plot events for a plot line ordered by sort_order."""
        with self.db.transaction() as conn:
            plot_line_row = self._fetch_active_project_owned_row(
                conn,
                "manuscript_plot_lines",
                plot_line_id,
            )
            if plot_line_row is None:
                return []
            rows = conn.execute(
                "SELECT * FROM manuscript_plot_events "
                "WHERE plot_line_id = ? AND deleted = 0 ORDER BY sort_order",
                (plot_line_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_plot_event(
        self,
        event_id: str,
        updates: dict[str, Any],
        expected_version: int,
    ) -> None:
        """Update a plot event with optimistic locking."""
        if not updates:
            return

        unknown = set(updates) - _UPDATABLE_PLOT_EVENT_COLS
        if unknown:
            raise ValueError(f"Unknown update column(s) for plot_event: {unknown}")

        now = self._now()
        next_version = expected_version + 1

        set_parts: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        set_parts.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        params.extend([now, next_version, self._client_id])
        params.extend([event_id, expected_version])

        with self.db.transaction() as conn:
            row = self._require_active_project_owned_row(
                conn,
                "manuscript_plot_events",
                event_id,
                entity_label="PlotEvent",
                action="update",
            )

            ref_cols = {"plot_line_id", "scene_id", "chapter_id"} & updates.keys()
            if ref_cols:
                self._validate_plot_refs(
                    conn,
                    row["project_id"],
                    plot_line_id=updates.get("plot_line_id", row["plot_line_id"]),
                    scene_id=updates.get("scene_id", row["scene_id"]),
                    chapter_id=updates.get("chapter_id", row["chapter_id"]),
                )
            cur = conn.execute(
                f"UPDATE manuscript_plot_events SET {', '.join(set_parts)} "  # nosec B608
                "WHERE id = ? AND version = ? AND deleted = 0",
                params,
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"PlotEvent {event_id!r} update failed (version conflict or not found).",
                    entity="manuscript_plot_events",
                    entity_id=event_id,
                )

    def soft_delete_plot_event(self, event_id: str, expected_version: int) -> None:
        """Soft-delete a plot event with optimistic locking."""
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            self._require_active_project_owned_row(
                conn,
                "manuscript_plot_events",
                event_id,
                entity_label="PlotEvent",
                action="delete",
            )
            cur = conn.execute(
                "UPDATE manuscript_plot_events "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, event_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"PlotEvent {event_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_plot_events",
                    entity_id=event_id,
                )

    # ==================================================================
    # Plot Holes
    # ==================================================================

    def create_plot_hole(
        self,
        project_id: str,
        title: str,
        *,
        description: str | None = None,
        severity: str = "medium",
        scene_id: str | None = None,
        chapter_id: str | None = None,
        plot_line_id: str | None = None,
        detected_by: str = "manual",
        plot_hole_id: str | None = None,
    ) -> str:
        """Insert a new plot hole and return its ID."""
        phid = plot_hole_id or self._uuid()
        now = self._now()

        with self.db.transaction() as conn:
            self._validate_plot_refs(
                conn, project_id,
                plot_line_id=plot_line_id, scene_id=scene_id, chapter_id=chapter_id,
            )
            conn.execute(
                """
                INSERT INTO manuscript_plot_holes
                    (id, project_id, title, description, severity, status,
                     scene_id, chapter_id, plot_line_id, resolution, detected_by,
                     created_at, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, NULL, ?, ?, ?, 0, ?, 1)
                """,
                (
                    phid, project_id, title, description, severity,
                    scene_id, chapter_id, plot_line_id, detected_by,
                    now, now, self._client_id,
                ),
            )
        logger.debug("Created plot hole {} in project {}", phid, project_id)
        return phid

    def get_plot_hole(self, plot_hole_id: str) -> dict[str, Any] | None:
        """Fetch a plot hole by ID; returns *None* if missing or deleted."""
        with self.db.transaction() as conn:
            row = self._fetch_active_project_owned_row(
                conn, "manuscript_plot_holes", plot_hole_id
            )
        return dict(row) if row else None

    def list_plot_holes(
        self,
        project_id: str,
        *,
        status_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """List non-deleted plot holes for a project."""
        where = "project_id = ? AND deleted = 0"
        params: list[Any] = [project_id]
        if status_filter:
            where += " AND status = ?"
            params.append(status_filter)

        with self.db.transaction() as conn:
            if not self._project_is_active(conn, project_id):
                return []
            rows = conn.execute(
                f"SELECT * FROM manuscript_plot_holes WHERE {where}",  # nosec B608
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def update_plot_hole(
        self,
        plot_hole_id: str,
        updates: dict[str, Any],
        expected_version: int,
    ) -> None:
        """Update a plot hole with optimistic locking."""
        if not updates:
            return

        unknown = set(updates) - _UPDATABLE_PLOT_HOLE_COLS
        if unknown:
            raise ValueError(f"Unknown update column(s) for plot_hole: {unknown}")

        now = self._now()
        next_version = expected_version + 1

        set_parts: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        set_parts.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        params.extend([now, next_version, self._client_id])
        params.extend([plot_hole_id, expected_version])

        with self.db.transaction() as conn:
            row = self._require_active_project_owned_row(
                conn,
                "manuscript_plot_holes",
                plot_hole_id,
                entity_label="PlotHole",
                action="update",
            )

            ref_cols = {"scene_id", "chapter_id", "plot_line_id"} & updates.keys()
            if ref_cols:
                self._validate_plot_refs(
                    conn,
                    row["project_id"],
                    plot_line_id=updates.get("plot_line_id", row["plot_line_id"]),
                    scene_id=updates.get("scene_id", row["scene_id"]),
                    chapter_id=updates.get("chapter_id", row["chapter_id"]),
                )
            cur = conn.execute(
                f"UPDATE manuscript_plot_holes SET {', '.join(set_parts)} "  # nosec B608
                "WHERE id = ? AND version = ? AND deleted = 0",
                params,
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"PlotHole {plot_hole_id!r} update failed (version conflict or not found).",
                    entity="manuscript_plot_holes",
                    entity_id=plot_hole_id,
                )

    def soft_delete_plot_hole(self, plot_hole_id: str, expected_version: int) -> None:
        """Soft-delete a plot hole with optimistic locking."""
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            self._require_active_project_owned_row(
                conn,
                "manuscript_plot_holes",
                plot_hole_id,
                entity_label="PlotHole",
                action="delete",
            )
            cur = conn.execute(
                "UPDATE manuscript_plot_holes "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, plot_hole_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"PlotHole {plot_hole_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_plot_holes",
                    entity_id=plot_hole_id,
                )

    # ==================================================================
    # Citations
    # ==================================================================

    def create_citation(
        self,
        project_id: str,
        scene_id: str,
        source_type: str,
        *,
        source_id: str | None = None,
        source_title: str | None = None,
        excerpt: str | None = None,
        query_used: str | None = None,
        anchor_offset: int | None = None,
        citation_id: str | None = None,
    ) -> str:
        """Insert a new citation and return its ID."""
        cid = citation_id or self._uuid()
        now = self._now()

        with self.db.transaction() as conn:
            self._assert_same_project(conn, "manuscript_scenes", scene_id, project_id, "scene")
            conn.execute(
                """
                INSERT INTO manuscript_citations
                    (id, project_id, scene_id, source_type, source_id,
                     source_title, excerpt, query_used, anchor_offset,
                     created_at, last_modified, deleted, client_id, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1)
                """,
                (
                    cid, project_id, scene_id, source_type, source_id,
                    source_title, excerpt, query_used, anchor_offset,
                    now, now, self._client_id,
                ),
            )
        logger.debug("Created citation {} for scene {}", cid, scene_id)
        return cid

    def get_citation(self, citation_id: str) -> dict[str, Any] | None:
        """Fetch a citation by ID; returns *None* if missing or deleted."""
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM manuscript_citations WHERE id = ? AND deleted = 0",
                (citation_id,),
            ).fetchone()
            if row is None:
                return None
            if not self._project_is_active(conn, row["project_id"]):
                return None
            scene_row = self._fetch_active_project_owned_row(
                conn, "manuscript_scenes", row["scene_id"]
            )
            if scene_row is None:
                return None
        return dict(row)

    def list_citations(self, scene_id: str) -> list[dict[str, Any]]:
        """List non-deleted citations for a scene."""
        with self.db.transaction() as conn:
            scene_row = conn.execute(
                "SELECT project_id FROM manuscript_scenes WHERE id = ? AND deleted = 0",
                (scene_id,),
            ).fetchone()
            if scene_row is None or not self._project_is_active(conn, scene_row["project_id"]):
                return []
            rows = conn.execute(
                "SELECT * FROM manuscript_citations "
                "WHERE scene_id = ? AND deleted = 0",
                (scene_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def soft_delete_citation(self, citation_id: str, expected_version: int) -> None:
        """Soft-delete a citation with optimistic locking."""
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            row = self._require_active_project_owned_row(
                conn,
                "manuscript_citations",
                citation_id,
                entity_label="Citation",
                action="delete",
            )
            if self._fetch_active_project_owned_row(conn, "manuscript_scenes", row["scene_id"]) is None:
                raise ConflictError(
                    f"Citation {citation_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_citations",
                    entity_id=citation_id,
                )
            cur = conn.execute(
                "UPDATE manuscript_citations "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, citation_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Citation {citation_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_citations",
                    entity_id=citation_id,
                )

    def update_citation(
        self,
        citation_id: str,
        updates: dict[str, Any],
        expected_version: int,
    ) -> None:
        """Update a citation with optimistic locking."""
        if not updates:
            return

        unknown = set(updates) - _UPDATABLE_CITATION_COLS
        if unknown:
            raise ValueError(f"Unknown update column(s) for citation: {unknown}")

        now = self._now()
        next_version = expected_version + 1

        set_parts: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            set_parts.append(f"{key} = ?")
            params.append(value)

        set_parts.extend(["last_modified = ?", "version = ?", "client_id = ?"])
        params.extend([now, next_version, self._client_id])
        params.extend([citation_id, expected_version])

        with self.db.transaction() as conn:
            row = self._require_active_project_owned_row(
                conn,
                "manuscript_citations",
                citation_id,
                entity_label="Citation",
                action="update",
            )
            if self._fetch_active_project_owned_row(conn, "manuscript_scenes", row["scene_id"]) is None:
                raise ConflictError(
                    f"Citation {citation_id!r} update failed (version conflict or not found).",
                    entity="manuscript_citations",
                    entity_id=citation_id,
                )
            cur = conn.execute(
                f"UPDATE manuscript_citations SET {', '.join(set_parts)} "  # nosec B608
                "WHERE id = ? AND version = ? AND deleted = 0",
                params,
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Citation {citation_id!r} update failed (version conflict or not found).",
                    entity="manuscript_citations",
                    entity_id=citation_id,
                )

    # ------------------------------------------------------------------
    # AI Analyses
    # ------------------------------------------------------------------

    def create_analysis(
        self,
        project_id: str,
        scope_type: str,
        scope_id: str,
        analysis_type: str,
        result: dict,
        *,
        score: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        analysis_id: str | None = None,
    ) -> str:
        """Insert a new AI analysis row and return its ID."""
        aid = analysis_id or self._uuid()
        now = self._now()

        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO manuscript_ai_analyses
                   (id, project_id, scope_type, scope_id, analysis_type, provider, model,
                    result_json, score, created_at, last_modified, client_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    aid,
                    project_id,
                    scope_type,
                    scope_id,
                    analysis_type,
                    provider,
                    model,
                    json.dumps(result),
                    score,
                    now,
                    now,
                    self._client_id,
                ),
            )
        logger.debug("Created analysis {} ({}) for {} {}", aid, analysis_type, scope_type, scope_id)
        return aid

    def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        """Return a single analysis by ID, or None if deleted/missing.

        The ``result_json`` column is deserialized into a ``result`` key.
        """
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM manuscript_ai_analyses WHERE id = ? AND deleted = 0",
                (analysis_id,),
            ).fetchone()
            if not self._analysis_row_is_visible(conn, row):
                return None
        if row is None:
            return None
        d = dict(row)
        d["result"] = json.loads(d.pop("result_json", "{}"))
        return d

    def list_analyses(
        self,
        project_id: str,
        *,
        scope_type: str | None = None,
        scope_id: str | None = None,
        analysis_type: str | None = None,
        include_stale: bool = False,
    ) -> list[dict[str, Any]]:
        """List non-deleted analyses for a project with optional filters.

        By default stale analyses are excluded unless *include_stale* is True.
        """
        with self.db.transaction() as conn:
            if not self._project_is_active(conn, project_id):
                return []
            rows = conn.execute(
                """
                SELECT *
                  FROM manuscript_ai_analyses
                 WHERE project_id = ?
                   AND deleted = 0
                   AND (? = 1 OR stale = 0)
                   AND (? IS NULL OR scope_type = ?)
                   AND (? IS NULL OR scope_id = ?)
                   AND (? IS NULL OR analysis_type = ?)
                   AND (
                        scope_type = 'project'
                        OR (
                            scope_type = 'part'
                            AND EXISTS (
                                SELECT 1
                                  FROM manuscript_parts p
                                 WHERE p.id = manuscript_ai_analyses.scope_id
                                   AND p.project_id = manuscript_ai_analyses.project_id
                                   AND p.deleted = 0
                            )
                        )
                        OR (
                            scope_type = 'chapter'
                            AND EXISTS (
                                SELECT 1
                                  FROM manuscript_chapters c
                                 WHERE c.id = manuscript_ai_analyses.scope_id
                                   AND c.project_id = manuscript_ai_analyses.project_id
                                   AND c.deleted = 0
                            )
                        )
                        OR (
                            scope_type = 'scene'
                            AND EXISTS (
                                SELECT 1
                                  FROM manuscript_scenes s
                                  JOIN manuscript_chapters c
                                    ON c.id = s.chapter_id
                                   AND c.project_id = s.project_id
                                   AND c.deleted = 0
                                 WHERE s.id = manuscript_ai_analyses.scope_id
                                   AND s.project_id = manuscript_ai_analyses.project_id
                                   AND s.deleted = 0
                            )
                        )
                   )
                 ORDER BY created_at DESC
                """,
                (
                    project_id,
                    1 if include_stale else 0,
                    scope_type,
                    scope_type,
                    scope_id,
                    scope_id,
                    analysis_type,
                    analysis_type,
                ),
            ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                d = dict(row)
                d["result"] = json.loads(d.pop("result_json", "{}"))
                results.append(d)
            return results

    def mark_analyses_stale(self, scope_type: str, scope_id: str) -> int:
        """Mark all non-deleted analyses for a scope as stale.

        Returns the count of rows updated.
        """
        now = self._now()
        with self.db.transaction() as conn:
            cur = conn.execute(
                "UPDATE manuscript_ai_analyses "
                "SET stale = 1, last_modified = ?, version = version + 1, client_id = ? "
                "WHERE scope_type = ? AND scope_id = ? AND stale = 0 AND deleted = 0",
                (now, self._client_id, scope_type, scope_id),
            )
            return cur.rowcount

    def _mark_analyses_stale_in_txn(self, conn: Any, scope_type: str, scope_id: str) -> int:
        """Mark analyses stale within an existing transaction (no new txn opened)."""
        now = self._now()
        cur = conn.execute(
            "UPDATE manuscript_ai_analyses "
            "SET stale = 1, last_modified = ?, version = version + 1, client_id = ? "
            "WHERE scope_type = ? AND scope_id = ? AND stale = 0 AND deleted = 0",
            (now, self._client_id, scope_type, scope_id),
        )
        return cur.rowcount

    def _mark_scene_family_analyses_stale_in_txn(
        self,
        conn: Any,
        *,
        scene_id: str | None = None,
        chapter_id: str | None = None,
        project_id: str | None = None,
    ) -> int:
        """Mark analysis rows for the affected scene family as stale.

        Returns the total number of analysis rows updated across all supplied
        scopes.
        """
        total = 0
        if scene_id is not None:
            total += self._mark_analyses_stale_in_txn(conn, "scene", scene_id)
        if chapter_id is not None:
            total += self._mark_analyses_stale_in_txn(conn, "chapter", chapter_id)
        if project_id is not None:
            total += self._mark_analyses_stale_in_txn(conn, "project", project_id)
        return total

    def _analysis_scope_is_active(
        self,
        conn: Any,
        *,
        scope_type: str,
        scope_id: str,
        project_id: str,
    ) -> bool:
        """Return ``True`` when the analysis scope is still readable."""
        if not self._project_is_active(conn, project_id):
            return False
        if scope_type == "project":
            return True
        if scope_type == "part":
            row = conn.execute(
                "SELECT 1 FROM manuscript_parts "
                "WHERE id = ? AND project_id = ? AND deleted = 0",
                (scope_id, project_id),
            ).fetchone()
            return row is not None
        if scope_type == "chapter":
            row = conn.execute(
                "SELECT 1 FROM manuscript_chapters "
                "WHERE id = ? AND project_id = ? AND deleted = 0",
                (scope_id, project_id),
            ).fetchone()
            return row is not None
        if scope_type == "scene":
            row = conn.execute(
                "SELECT 1 "
                "FROM manuscript_scenes s "
                "JOIN manuscript_chapters c "
                "ON c.id = s.chapter_id AND c.project_id = s.project_id AND c.deleted = 0 "
                "WHERE s.id = ? AND s.project_id = ? AND s.deleted = 0",
                (scope_id, project_id),
            ).fetchone()
            return row is not None
        return False

    def _analysis_row_is_visible(self, conn: Any, row: Any) -> bool:
        """Return ``True`` when a cached analysis should be exposed to callers."""
        if row is None or row["deleted"] != 0:
            return False
        return self._analysis_scope_is_active(
            conn,
            scope_type=row["scope_type"],
            scope_id=row["scope_id"],
            project_id=row["project_id"],
        )

    def soft_delete_analysis(self, analysis_id: str, expected_version: int) -> None:
        """Soft-delete an analysis with optimistic locking."""
        now = self._now()
        next_version = expected_version + 1

        with self.db.transaction() as conn:
            cur = conn.execute(
                "UPDATE manuscript_ai_analyses "
                "SET deleted = 1, last_modified = ?, version = ?, client_id = ? "
                "WHERE id = ? AND version = ? AND deleted = 0",
                (now, next_version, self._client_id, analysis_id, expected_version),
            )
            if cur.rowcount == 0:
                raise ConflictError(
                    f"Analysis {analysis_id!r} delete failed (version conflict or not found).",
                    entity="manuscript_ai_analyses",
                    entity_id=analysis_id,
                )
