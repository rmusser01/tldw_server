"""Workspace domain store: workspaces, clone targets, sources, saved views, operations, inventory, memberships, artifacts,
activity, runtime bindings and source-review plans (TASK-13950.22 PR I2)."""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import time
import unicodedata
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Literal, overload

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    SavedViewNameUniqueConstraintError,
    UniqueConstraintError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.chacha.workspace_migration_records import (
    parse_workspace_migration_json,
    serialize_workspace_migration_json,
    workspace_migration_session_from_row,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    _CHACHA_NONCRITICAL_EXCEPTIONS,
    SOURCE_VIEW_LIMIT_REACHED,
    SOURCE_VIEW_MAX_COUNT,
    SOURCE_VIEW_MAX_INTEGER,
    SOURCE_VIEW_MAX_NAME_LENGTH,
    SOURCE_VIEW_MAX_STATE_BYTES,
    SOURCE_VIEW_MIN_NAME_LENGTH,
    SOURCE_VIEW_NAME_EXISTS,
    SOURCE_VIEW_VERSION_CONFLICT,
    BackendConnectionWrapper,
    CharactersRAGDBError,
    ConflictError,
    InputError,
    WorkspaceSourceSavedViewConflictError,
    WorkspaceSourceSavedViewNotFoundError,
)
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError
from tldw_Server_API.app.core.Flashcards.source_review import build_source_review_launch_metadata
from tldw_Server_API.app.core.Sharing.clone_models import WorkspaceCloneSnapshot
from tldw_Server_API.app.core.Workspaces.file_inventory_models import (
    bounded_inventory_diagnostics,
    decode_inventory_cursor,
    encode_inventory_cursor,
    normalize_durable_inventory_state,
    normalize_inventory_counts,
    sort_inventory_relative_paths,
)
from tldw_Server_API.app.core.Workspaces.membership_models import (
    WORKSPACE_MEMBERSHIP_MAX_METADATA_BYTES,
    WORKSPACE_MEMBERSHIP_MAX_PROVENANCE_BYTES,
    dump_membership_json_object,
)
from tldw_Server_API.app.core.Workspaces.operations import (
    WORKSPACE_OPERATION_ACTIVE_STATUSES,
    WORKSPACE_OPERATION_STATUSES,
    redact_operation_diagnostics,
)

_WORKSPACE_PROFILES = frozenset({"research", "project"})
_WORKSPACE_PROJECT_ROOT_BACKENDS = frozenset({"host_local", "sandbox_volume"})

_WORKSPACE_ACTIVITY_MAX_METADATA_BYTES = 16 * 1024

_WORKSPACE_CLONE_OPERATION_KIND = "shared_workspace_clone"

_WORKSPACE_CLONE_RECONCILIATION_MAX = 100

_WORKSPACE_ACTIVITY_SECRET_KEY_RE = re.compile(
    r"(^|[_\-.])("
    r"api[_\-.]?key|access[_\-.]?key|secret|token|password|passwd|pwd|"
    r"credential|credentials|private[_\-.]?key|client[_\-.]?secret|"
    r"bearer|authorization|auth[_\-.]?header"
    r")($|[_\-.])",
    re.IGNORECASE,
)

_WORKSPACE_ACTIVITY_SECRET_VALUE_RE = re.compile(
    r"(-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\bsk-[A-Za-z0-9_\-]{8,}|"
    r"\b(?:ghp|github_pat|xox[baprs])_[A-Za-z0-9_\-]{8,})",
    re.IGNORECASE,
)

_WORKSPACE_ACTIVITY_PATH_KEY_RE = re.compile(
    r"(^|[_\-.])("
    r"absolute[_\-.]?root|root|paths?|mount[_\-.]?path|workspace[_\-.]?path|"
    r"repo[_\-.]?path|local[_\-.]?path|project[_\-.]?root|dir|directory"
    r")($|[_\-.])",
    re.IGNORECASE,
)

_ALLOWED_WORKSPACE_ARTIFACT_REVIEW_STATES: tuple[str, ...] = (
    "draft",
    "reviewing",
    "accepted",
    "needs_revision",
    "rejected",
    "exported",
    "assigned",
    "archived",
)

_ALLOWED_WORKSPACE_SOURCE_REVIEW_STATES: tuple[str, ...] = (
    "unset",
    "needs_review",
    "reviewed",
)

_ALLOWED_WORKSPACE_SOURCE_CREATE_REVIEW_STATES: tuple[str, ...] = (
    "unset",
    "needs_review",
)

_WORKSPACE_ARTIFACT_JSON_FIELDS: tuple[tuple[str, str, Any], ...] = (
    ("producer_metadata_json", "producer_metadata", {}),
    ("source_lineage_json", "source_lineage", {}),
    ("review_metadata_json", "review_metadata", {}),
    ("version_metadata_json", "version_metadata", {}),
    ("export_refs_json", "export_refs", []),
    ("redaction_json", "redaction", {"support_safe": True, "redacted": False}),
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.chacha.context import ChaChaContext


class WorkspaceStore:
    def __init__(self, ctx: ChaChaContext) -> None:
        self._ctx = ctx

    def list_note_clipper_workspace_placements(self, clip_id: str) -> list[dict[str, Any]]:
        """Return active workspace placements for a clip."""
        params: tuple[str, str, bool] | tuple[str, int]
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            query = (
                "SELECT clip_id, workspace_id, workspace_note_id, source_note_id, source_note_version "
                "FROM note_clipper_workspace_placements "
                "WHERE client_id = ? AND clip_id = ? AND deleted = ? ORDER BY workspace_id"
            )
            params = (self._ctx.client_id, clip_id, False)
        else:
            query = (
                "SELECT clip_id, workspace_id, workspace_note_id, source_note_id, source_note_version "
                "FROM note_clipper_workspace_placements "
                "WHERE clip_id = ? AND deleted = ? ORDER BY workspace_id"
            )
            params = (clip_id, 0)
        cursor = self._ctx.execute_query(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def upsert_note_clipper_workspace_placement(
        self,
        *,
        clip_id: str,
        workspace_id: str,
        workspace_note_id: int | None = None,
        source_note_id: str | None = None,
        source_note_version: int | None = None,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> dict[str, Any]:
        normalized_clip_id = str(clip_id).strip()
        normalized_workspace_id = str(workspace_id).strip()
        if not normalized_clip_id or not normalized_workspace_id:
            raise InputError("clip_id and workspace_id are required.")  # noqa: TRY003
        if source_note_version is not None and (not isinstance(source_note_version, int) or source_note_version < 1):
            raise InputError("source_note_version must be an integer >= 1 when provided.")  # noqa: TRY003

        now = self._ctx.utc_now_iso()
        deleted_value = False if self._ctx.backend_type == BackendType.POSTGRESQL else 0
        postgres = self._ctx.backend_type == BackendType.POSTGRESQL
        if postgres:
            query = (
                "INSERT INTO note_clipper_workspace_placements ("
                "client_id, clip_id, workspace_id, workspace_note_id, source_note_id, source_note_version, "
                "created_at, last_modified, deleted"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(client_id, clip_id, workspace_id) DO UPDATE SET "
                "workspace_note_id = excluded.workspace_note_id, "
                "source_note_id = excluded.source_note_id, "
                "source_note_version = excluded.source_note_version, "
                "last_modified = excluded.last_modified, "
                "deleted = excluded.deleted"
            )
        else:
            query = (
                "INSERT INTO note_clipper_workspace_placements ("
                "clip_id, workspace_id, workspace_note_id, source_note_id, source_note_version, "
                "created_at, last_modified, deleted"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(clip_id, workspace_id) DO UPDATE SET "
                "workspace_note_id = excluded.workspace_note_id, "
                "source_note_id = excluded.source_note_id, "
                "source_note_version = excluded.source_note_version, "
                "last_modified = excluded.last_modified, "
                "deleted = excluded.deleted"
            )
        def _execute(inner_conn: sqlite3.Connection | BackendConnectionWrapper) -> dict[str, Any]:
            owner_clause = "client_id = ? AND " if postgres else ""
            owner_params: tuple[Any, ...] = (self._ctx.client_id,) if postgres else ()
            workspace_row = inner_conn.execute(
                f"SELECT id FROM workspaces WHERE {owner_clause}id = ? AND deleted = ?",  # nosec B608
                (
                    *owner_params,
                    normalized_workspace_id,
                    False if self._ctx.backend_type == BackendType.POSTGRESQL else 0,
                ),
            ).fetchone()
            if workspace_row is None:
                raise ConflictError(
                    "Workspace not found or deleted.",
                    entity="workspaces",
                    entity_id=normalized_workspace_id,
                )
            clip_row = inner_conn.execute(
                f"SELECT clip_id, note_id FROM note_clipper_documents "  # nosec B608
                f"WHERE {owner_clause}clip_id = ? AND deleted = ?",
                (
                    *owner_params,
                    normalized_clip_id,
                    False if self._ctx.backend_type == BackendType.POSTGRESQL else 0,
                ),
            ).fetchone()
            if clip_row is None:
                raise ConflictError(
                    "Canonical clip document not found or deleted.",
                    entity="note_clipper_documents",
                    entity_id=normalized_clip_id,
                )
            canonical_note_id = str(clip_row["note_id"])
            normalized_source_note_id = (
                str(source_note_id).strip()
                if source_note_id is not None
                else canonical_note_id
            )
            if normalized_source_note_id != canonical_note_id:
                raise InputError(
                    "source_note_id must match the clip document note_id."
                )  # noqa: TRY003
            params = (
                *owner_params,
                normalized_clip_id,
                normalized_workspace_id,
                workspace_note_id,
                normalized_source_note_id,
                source_note_version,
                now,
                now,
                deleted_value,
            )
            prepared_query, prepared_params = self._ctx.prepare_statement(query, params)
            inner_conn.execute(prepared_query, prepared_params or ())
            result = inner_conn.execute(
                "SELECT clip_id, workspace_id, workspace_note_id, source_note_id, source_note_version "
                f"FROM note_clipper_workspace_placements WHERE {owner_clause}clip_id = ? "  # nosec B608
                "AND workspace_id = ? AND deleted = ?",
                (
                    *owner_params,
                    normalized_clip_id,
                    normalized_workspace_id,
                    False if self._ctx.backend_type == BackendType.POSTGRESQL else 0,
                ),
            ).fetchone()
            if result is None:
                raise CharactersRAGDBError(
                    f"Failed to read workspace placement for clip ID '{normalized_clip_id}' and workspace '{normalized_workspace_id}'."
                )
            return dict(result)

        if conn is None:
            with self._ctx.transaction() as transaction_conn:
                return _execute(transaction_conn)
        return _execute(conn)

    @staticmethod
    def _validate_workspace_profile(workspace_profile: str) -> str:
        """Validate a persisted workspace profile value."""
        normalized = str(workspace_profile or "").strip().lower()
        if normalized not in _WORKSPACE_PROFILES:
            raise InputError("workspace_profile must be 'research' or 'project'")  # noqa: TRY003
        return normalized

    @staticmethod
    def _validate_workspace_clone_identifier(value: Any, field_name: str) -> str:
        """Validate one clone lifecycle identifier using the clone request contract."""
        if not isinstance(value, str) or not value:
            raise InputError(f"{field_name} must be a non-empty ASCII string")  # noqa: TRY003
        if len(value) > 255:
            raise InputError(f"{field_name} must be at most 255 characters")  # noqa: TRY003
        if not value.isascii() or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
            raise InputError(f"{field_name} must contain only printable ASCII characters")  # noqa: TRY003
        return value

    @staticmethod
    def _normalize_workspace_clone_name(value: Any) -> str:
        """Normalize a clone target name using the immutable clone request contract."""
        if not isinstance(value, str):
            raise InputError("name must be a non-empty string")  # noqa: TRY003
        normalized = " ".join(value.split())
        if not normalized:
            raise InputError("name must be non-empty after normalization")  # noqa: TRY003
        if len(normalized) > 255:
            raise InputError("name must be at most 255 characters")  # noqa: TRY003
        return normalized

    @staticmethod
    def _validate_workspace_project_root_backend(backend: Any) -> str:
        """Validate a Workspace-owned project-root backend."""
        normalized = str(backend or "").strip().lower()
        if normalized not in _WORKSPACE_PROJECT_ROOT_BACKENDS:
            raise InputError("workspace project root backend must be 'host_local' or 'sandbox_volume'")  # noqa: TRY003
        return normalized

    @staticmethod
    def _serialize_workspace_project_root_metadata(value: Any) -> str:
        """Serialize project-root metadata while rejecting malformed JSON strings."""
        if value is None:
            return "{}"
        if isinstance(value, str):
            try:
                json.loads(value)
            except json.JSONDecodeError as exc:
                raise InputError("metadata_json must be valid JSON when provided as a string.") from exc  # noqa: TRY003
            return value
        try:
            return json.dumps(value, ensure_ascii=True)
        except TypeError as exc:
            raise InputError("metadata_json must be JSON-serializable.") from exc  # noqa: TRY003

    def _workspace_active_deleted_value(self) -> bool | int:
        return False if self._ctx.backend_type == BackendType.POSTGRESQL else 0

    def _workspace_primary_root_value(self) -> bool | int:
        return True if self._ctx.backend_type == BackendType.POSTGRESQL else 1

    def _workspace_project_root_row_by_id(
        self,
        conn: sqlite3.Connection | BackendConnectionWrapper,
        workspace_id: str,
        root_id: str,
    ) -> dict[str, Any] | None:
        row = conn.execute(
            """
            SELECT r.*
              FROM workspace_project_roots r
              JOIN workspaces w ON w.id = r.workspace_id
             WHERE r.workspace_id = ?
               AND r.root_id = ?
               AND w.deleted = ?
            """,
            (workspace_id, root_id, self._workspace_active_deleted_value()),
        ).fetchone()
        return dict(row) if row else None

    def _workspace_project_root_integrity_error(self, exc: Exception) -> CharactersRAGDBError:
        message = str(exc).lower()
        if "unique" in message or "duplicate" in message or "foreign key" in message or "constraint" in message:
            return ConflictError(
                f"Workspace project root write conflicted: {exc}",
                entity="workspace_project_roots",
            )
        return CharactersRAGDBError(f"Workspace project root write failed: {exc}")  # noqa: TRY003

    @classmethod
    def _serialize_workspace_assistant_defaults_json(cls, value: Any) -> str | None:
        """Serialize Workspace Assistant Defaults for DB storage."""
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                parsed = json.loads(stripped)
            except (json.JSONDecodeError, ValueError) as exc:
                raise InputError("assistant_defaults_json must be valid JSON when provided as a string.") from exc
            if not isinstance(parsed, Mapping):
                raise InputError("assistant_defaults_json must be a JSON object or null.")  # noqa: TRY003
            return stripped
        if not isinstance(value, Mapping):
            raise InputError("assistant_defaults_json must be a mapping, JSON object string, or null.")  # noqa: TRY003
        try:
            return json.dumps(dict(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise InputError("assistant_defaults_json must be JSON serializable.") from exc

    @classmethod
    def _load_workspace_assistant_defaults_json(cls, value: Any) -> dict[str, Any] | None:
        """Load Workspace Assistant Defaults from DB storage."""
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Ignoring malformed workspace assistant_defaults_json stored in the database.")
            return None
        if not isinstance(parsed, Mapping):
            logger.warning("Ignoring non-object workspace assistant_defaults_json stored in the database.")
            return None
        return dict(parsed)

    @classmethod
    def _workspace_row_to_dict(cls, row: Any) -> dict[str, Any]:
        """Normalize stored defaults, retaining only a computed private corruption flag.

        SQL NULL is unset or opted out according to the stored choice bit. A
        non-object default or a non-null default paired with opt-out is invalid.
        The corruption flag is computed, read-only, and never a storage field.
        """
        workspace = dict(row)
        stored_defaults = workspace.get("assistant_defaults_json")
        if "assistant_defaults_json" in workspace:
            workspace["assistant_defaults_json"] = cls._load_workspace_assistant_defaults_json(stored_defaults)
        workspace["assistant_defaults_explicit_none"] = bool(workspace.get("assistant_defaults_explicit_none", False))
        workspace["_assistant_defaults_invalid"] = (
            stored_defaults is not None
            and (workspace.get("assistant_defaults_json") is None or workspace["assistant_defaults_explicit_none"])
        )
        return workspace

    def _get_workspace_internal(
        self,
        workspace_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Read a Workspace without public lifecycle-marker filtering."""
        query = "SELECT * FROM workspaces WHERE id = ?"
        if not include_deleted:
            query += " AND deleted = 0"
        row = self._ctx.execute_query(query, (workspace_id,)).fetchone()
        return self._workspace_row_to_dict(row) if row else None

    def _get_workspace_internal_with_conn(
        self,
        conn: Any,
        workspace_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Read a Workspace through an existing lifecycle transaction."""
        query = "SELECT * FROM workspaces WHERE id = ?"
        if not include_deleted:
            query += " AND deleted = 0"
        row = conn.execute(query, (workspace_id,)).fetchone()
        return self._workspace_row_to_dict(row) if row else None

    @classmethod
    def _clone_reservation_matches(
        cls,
        row: Mapping[str, Any],
        *,
        operation_id: str,
        request_fingerprint: str,
        name: str,
        description: str | None,
        workspace_profile: str,
    ) -> bool:
        """Return whether an existing row is the exact reusable reservation."""
        return (
            not bool(row.get("deleted"))
            and row.get("system_operation_kind") == _WORKSPACE_CLONE_OPERATION_KIND
            and row.get("system_operation_state") in {"staged", "publication_pending"}
            and row.get("system_operation_id") == operation_id
            and row.get("system_request_fingerprint") == request_fingerprint
            and row.get("name") == name
            and row.get("description") == description
            and row.get("workspace_profile") == workspace_profile
        )

    @staticmethod
    def _clone_target_conflict(workspace_id: str) -> ConflictError:
        """Build the controlled collision error for a deterministic target id."""
        return ConflictError(
            f"Workspace clone target '{workspace_id}' is already reserved.",
            entity="workspaces",
            entity_id=workspace_id,
        )

    def reserve_clone_target(
        self,
        *,
        workspace_id: str,
        operation_id: str,
        request_fingerprint: str,
        name: str,
        description: str | None,
        workspace_profile: str,
    ) -> dict[str, Any]:
        """Reserve a hidden clone target, opting out of unrepresented source Persona choices."""
        validated_workspace_id = self._validate_workspace_clone_identifier(workspace_id, "workspace_id")
        validated_operation_id = self._validate_workspace_clone_identifier(operation_id, "operation_id")
        validated_fingerprint = self._validate_workspace_clone_identifier(
            request_fingerprint,
            "request_fingerprint",
        )
        normalized_name = self._normalize_workspace_clone_name(name)
        normalized_profile = self._validate_workspace_profile(workspace_profile)
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO workspaces (
                        id, name, description, metadata_json, study_materials_policy,
                        workspace_profile, archived, created_at, last_modified, deleted,
                        client_id, version, system_operation_id, system_operation_kind,
                        system_operation_state, system_request_fingerprint, assistant_defaults_explicit_none
                    ) VALUES (?, ?, ?, '{}', 'general', ?, ?, ?, ?, ?, ?, 1, ?, ?, 'staged', ?, ?)
                    ON CONFLICT(id) DO NOTHING
                    """,
                    (
                        validated_workspace_id,
                        normalized_name,
                        description,
                        normalized_profile,
                        True,
                        now,
                        now,
                        False,
                        self._ctx.client_id or "unknown",
                        validated_operation_id,
                        _WORKSPACE_CLONE_OPERATION_KIND,
                        validated_fingerprint,
                        True,
                    ),
                )
                workspace = self._get_workspace_internal_with_conn(
                    conn,
                    validated_workspace_id,
                    include_deleted=True,
                )
                if workspace is None:  # pragma: no cover - defensive database invariant
                    raise CharactersRAGDBError("Failed to load reserved Workspace clone target.")  # noqa: TRY003
                if not self._clone_reservation_matches(
                    workspace,
                    operation_id=validated_operation_id,
                    request_fingerprint=validated_fingerprint,
                    name=normalized_name,
                    description=description,
                    workspace_profile=normalized_profile,
                ):
                    raise self._clone_target_conflict(validated_workspace_id)
                return workspace
        except ConflictError:
            raise
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            raise CharactersRAGDBError("Workspace clone target reservation failed.") from exc  # noqa: TRY003

    def _transition_clone_target(
        self,
        *,
        workspace_id: str,
        operation_id: str,
        expected_state: str,
        set_clause: str,
        set_params: tuple[Any, ...],
    ) -> dict[str, Any]:
        """Apply one exact operation-owned clone lifecycle transition."""
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                cursor = conn.execute(
                    f"""
                    UPDATE workspaces
                       SET {set_clause}, last_modified = ?, version = version + 1
                     WHERE id = ?
                       AND deleted = ?
                       AND system_operation_id = ?
                       AND system_operation_kind = ?
                       AND system_operation_state = ?
                    """,  # nosec B608 - set_clause is fixed at internal call sites.
                    (
                        *set_params,
                        now,
                        workspace_id,
                        False,
                        operation_id,
                        _WORKSPACE_CLONE_OPERATION_KIND,
                        expected_state,
                    ),
                )
                if cursor.rowcount != 1:
                    raise self._clone_target_conflict(workspace_id)
                workspace = self._get_workspace_internal_with_conn(conn, workspace_id)
                if workspace is None:  # pragma: no cover - defensive database invariant
                    raise CharactersRAGDBError("Failed to load transitioned Workspace clone target.")  # noqa: TRY003
                return workspace
        except ConflictError:
            raise
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            raise CharactersRAGDBError("Workspace clone target transition failed.") from exc  # noqa: TRY003

    def publish_clone_target(
        self,
        *,
        workspace_id: str,
        operation_id: str,
    ) -> dict[str, Any]:
        """Move an exact owned staged target into hidden publication-pending state."""
        validated_workspace_id = self._validate_workspace_clone_identifier(workspace_id, "workspace_id")
        validated_operation_id = self._validate_workspace_clone_identifier(operation_id, "operation_id")
        return self._transition_clone_target(
            workspace_id=validated_workspace_id,
            operation_id=validated_operation_id,
            expected_state="staged",
            set_clause="archived = ?, system_operation_state = ?",
            set_params=(False, "publication_pending"),
        )

    def confirm_clone_target_publication(
        self,
        *,
        workspace_id: str,
        operation_id: str,
    ) -> dict[str, Any]:
        """Expose an exact owned publication-pending target by clearing all markers."""
        validated_workspace_id = self._validate_workspace_clone_identifier(workspace_id, "workspace_id")
        validated_operation_id = self._validate_workspace_clone_identifier(operation_id, "operation_id")
        return self._transition_clone_target(
            workspace_id=validated_workspace_id,
            operation_id=validated_operation_id,
            expected_state="publication_pending",
            set_clause=(
                "system_operation_id = NULL, system_operation_kind = NULL, "
                "system_operation_state = NULL, system_request_fingerprint = NULL"
            ),
            set_params=(),
        )

    def discard_clone_target(
        self,
        *,
        workspace_id: str,
        operation_id: str,
    ) -> bool:
        """Soft-delete only an exact operation-owned staged or pending clone target."""
        validated_workspace_id = self._validate_workspace_clone_identifier(workspace_id, "workspace_id")
        validated_operation_id = self._validate_workspace_clone_identifier(operation_id, "operation_id")
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE workspaces
                       SET deleted = ?, last_modified = ?, version = version + 1
                     WHERE id = ?
                       AND deleted = ?
                       AND system_operation_id = ?
                       AND system_operation_kind = ?
                       AND system_operation_state IN ('staged', 'publication_pending')
                    """,
                    (
                        True,
                        now,
                        validated_workspace_id,
                        False,
                        validated_operation_id,
                        _WORKSPACE_CLONE_OPERATION_KIND,
                    ),
                )
                return cursor.rowcount == 1
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            raise CharactersRAGDBError("Workspace clone target discard failed.") from exc  # noqa: TRY003

    def list_clone_targets_for_reconciliation(
        self,
        *,
        operation_ids: Sequence[str],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return a bounded set of caller-correlated staged clone targets."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise InputError("limit must be an integer between 1 and 100")  # noqa: TRY003
        if isinstance(operation_ids, (str, bytes)) or not isinstance(operation_ids, Sequence):
            raise InputError("operation_ids must be a sequence of identifiers")  # noqa: TRY003
        if len(operation_ids) > _WORKSPACE_CLONE_RECONCILIATION_MAX:
            raise InputError("operation_ids may contain at most 100 identifiers")  # noqa: TRY003
        if not operation_ids:
            return []
        validated_ids = [
            self._validate_workspace_clone_identifier(operation_id, "operation_id")
            for operation_id in operation_ids
        ]
        placeholders = ", ".join("?" for _ in validated_ids)
        cursor = self._ctx.execute_query(
            f"""
            SELECT *
              FROM workspaces
             WHERE deleted = ?
               AND system_operation_kind = ?
               AND system_operation_state IN ('staged', 'publication_pending')
               AND system_operation_id IN ({placeholders})
             ORDER BY system_operation_id ASC, id ASC
             LIMIT ?
            """,  # nosec B608 - placeholders are generated from bounded validated inputs.
            (
                False,
                _WORKSPACE_CLONE_OPERATION_KIND,
                *validated_ids,
                limit,
            ),
        )
        return [self._workspace_row_to_dict(row) for row in cursor.fetchall()]

    def upsert_workspace(
        self,
        workspace_id: str,
        name: str,
        *,
        description: str | None = None,
        metadata_json: str | None = None,
        study_materials_policy: str = "general",
        workspace_profile: str | None = None,
        initial_assistant_opt_out: bool = False,
    ) -> dict[str, Any]:
        """Create a workspace or update the existing row in place.

        If the workspace already exists (and is not deleted), the provided
        mutable fields are applied to the existing row using optimistic locking.
        Internal import callers may opt out on creation when the source cannot
        represent a Persona choice; this never changes an existing row's choice.

        Returns:
            A dict representing the workspace row.

        Raises:
            InputError: If *workspace_id* or *name* are empty.
            CharactersRAGDBError: On database errors.
        """
        if not workspace_id or not name:
            raise InputError("workspace_id and name are required.")  # noqa: TRY003

        normalized_policy = study_materials_policy or "general"
        normalized_profile = (
            self._validate_workspace_profile(workspace_profile)
            if workspace_profile is not None
            else "research"
        )
        existing = self.get_workspace(workspace_id)
        if existing is not None:
            updates: dict[str, Any] = {}
            if existing.get("name") != name:
                updates["name"] = name
            if description is not None and existing.get("description") != description:
                updates["description"] = description
            if metadata_json is not None and existing.get("metadata_json") != metadata_json:
                updates["metadata_json"] = metadata_json
            if existing.get("study_materials_policy") != normalized_policy:
                updates["study_materials_policy"] = normalized_policy
            if (
                workspace_profile is not None
                and existing.get("workspace_profile", "research") != normalized_profile
            ):
                updates["workspace_profile"] = normalized_profile
            if not updates:
                return existing
            return self.update_workspace(workspace_id, updates, expected_version=int(existing["version"]))

        now = self._ctx.utc_now_iso()
        client_id = self._ctx.client_id or "unknown"
        query = (
            "INSERT INTO workspaces "
            "(id, name, description, metadata_json, study_materials_policy, workspace_profile, "
            "created_at, last_modified, deleted, client_id, version, assistant_defaults_explicit_none) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 1, ?)"
        )
        params = (
            workspace_id,
            name,
            description,
            metadata_json or "{}",
            normalized_policy,
            normalized_profile,
            now,
            now,
            client_id,
            initial_assistant_opt_out,
        )
        try:
            with self._ctx.transaction() as conn:
                conn.execute(query, params)
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            # Race condition: another thread inserted between our check and insert.
            existing = self.get_workspace(workspace_id)
            if existing is not None:
                return existing
            if self._get_workspace_internal(workspace_id, include_deleted=True) is not None:
                raise ConflictError(  # noqa: TRY003
                    f"Workspace '{workspace_id}' already exists.",
                    entity="workspaces",
                    entity_id=workspace_id,
                ) from exc
            if isinstance(exc, UniqueConstraintError):
                # RLS can hide a conflicting ID; do not disclose its owner or row.
                raise NotFoundError("Workspace not found") from exc  # noqa: TRY003
            raise CharactersRAGDBError("Workspace creation failed.") from exc  # noqa: TRY003
        return self.get_workspace(workspace_id)  # type: ignore[return-value]

    def get_workspace(
        self,
        workspace_id: str,
        *,
        include_deleted: bool = False,
        conn: Any = None,
        for_update: bool = False,
    ) -> dict[str, Any] | None:
        """Read a visible Workspace, optionally locking it in an existing transaction."""
        if for_update and (
            conn is None or not getattr(conn, "in_transaction", False)
            or (self._ctx.backend_type == BackendType.POSTGRESQL and not getattr(self._ctx.connection_state(), "tx_depth", 0))
        ):
            raise InputError("Workspace locking reads require a transaction connection.")
        query = "SELECT * FROM workspaces WHERE id = ? AND system_operation_state IS NULL"
        if not include_deleted:
            query += " AND deleted = 0"
        if for_update and self._ctx.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE"
        cursor = (
            conn.execute(query, (workspace_id,))
            if conn is not None else self._ctx.execute_query(query, (workspace_id,), read_only=True)
        )
        row = cursor.fetchone()
        return self._workspace_row_to_dict(row) if row else None

    def list_workspaces(self) -> list[dict[str, Any]]:
        """Return all non-deleted workspaces ordered by name."""
        query = (
            "SELECT * FROM workspaces "
            "WHERE deleted = 0 AND system_operation_state IS NULL ORDER BY name"
        )
        cursor = self._ctx.execute_query(query, (), read_only=True)
        return [self._workspace_row_to_dict(row) for row in cursor.fetchall()]

    def read_workspace_clone_snapshot(self, workspace_id: str) -> WorkspaceCloneSnapshot:
        """Read one active Workspace and its cloneable rows from one source snapshot."""
        from tldw_Server_API.app.core.Sharing.clone_models import (
            CloneSnapshotUnavailable,
            WorkspaceCloneSnapshot,
        )

        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise CloneSnapshotUnavailable(cleanup_state="complete")

        backend = self._ctx.backend
        connection: Any | None = None
        pool: Any | None = None
        committed = False
        primary_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        snapshot: WorkspaceCloneSnapshot | None = None

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
                # Keep the required native capability lazy for duck-typed backends.
                apply_and_verify_scope: Callable[..., None] = getattr(backend, "apply_and_verify_scope")  # noqa: B009
                apply_and_verify_scope(
                    connection,
                    fallback_user_id=self._ctx.client_id,
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

            active_value = False if backend.backend_type == BackendType.POSTGRESQL else 0

            def read_rows(query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
                result = backend.execute(
                    query,
                    params,
                    connection=connection,
                    log_errors=False,
                )
                return [dict(row) for row in result.rows]

            workspace_rows = read_rows(
                "SELECT * FROM workspaces "
                "WHERE id = ? AND deleted = ? AND system_operation_state IS NULL",
                (workspace_id, active_value),
            )
            if len(workspace_rows) != 1:
                raise CloneSnapshotUnavailable(cleanup_state="complete")

            membership_rows = read_rows(
                "SELECT * FROM workspace_resource_memberships "
                "WHERE workspace_id = ? AND deleted = ? "
                "ORDER BY updated_at DESC, resource_type ASC, resource_id ASC",
                (workspace_id, active_value),
            )
            source_rows = read_rows(
                "SELECT * FROM workspace_sources WHERE workspace_id = ? "
                "ORDER BY position, added_at",
                (workspace_id,),
            )
            note_rows = read_rows(
                "SELECT * FROM workspace_notes WHERE workspace_id = ? AND deleted = ? "
                "ORDER BY last_modified DESC",
                (workspace_id, active_value),
            )
            artifact_rows = read_rows(
                "SELECT * FROM workspace_artifacts WHERE workspace_id = ? ORDER BY created_at DESC",
                (workspace_id,),
            )

            normalized_memberships = []
            for row in membership_rows:
                normalized = self._normalize_workspace_membership_row(row)
                if normalized is None:
                    raise RuntimeError("Workspace membership row unavailable")
                normalized_memberships.append(normalized)
            normalized_artifacts = []
            for row in artifact_rows:
                normalized = self._normalize_workspace_artifact_row(row)
                if normalized is None:
                    raise RuntimeError("Workspace artifact row unavailable")
                normalized_artifacts.append(normalized)

            snapshot = WorkspaceCloneSnapshot.from_rows(
                workspace=self._workspace_row_to_dict(workspace_rows[0]),
                memberships=normalized_memberships,
                sources=source_rows,
                notes=note_rows,
                artifacts=normalized_artifacts,
            )
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
        if primary_error is not None or cleanup_error is not None or snapshot is None:
            failure = primary_error or cleanup_error
            logger.bind(
                backend=backend.backend_type.value,
                exception_type=type(failure).__name__ if failure is not None else "Unknown",
            ).warning("Workspace clone snapshot read failed")
            raise CloneSnapshotUnavailable(cleanup_state="complete") from None
        return snapshot

    def update_workspace(
        self,
        workspace_id: str,
        update_data: dict[str, Any],
        expected_version: int,
    ) -> dict[str, Any]:
        """Update a workspace with optimistic locking.

        Returns:
            The updated workspace row dict.

        Raises:
            ConflictError: If the workspace is not found or version mismatch.
        """
        existing = self.get_workspace(workspace_id)
        if existing is None:
            raise ConflictError(  # noqa: TRY003
                f"Workspace '{workspace_id}' not found.",
                entity="workspaces",
                entity_id=workspace_id,
            )
        if existing["version"] != expected_version:
            raise ConflictError(  # noqa: TRY003
                f"Workspace '{workspace_id}' version mismatch (db={existing['version']}, expected={expected_version}).",
                entity="workspaces",
                entity_id=workspace_id,
            )

        now = self._ctx.utc_now_iso()
        workspace_update_columns = (
            "name", "description", "metadata_json", "study_materials_policy", "workspace_profile", "archived", "tag",
            "banner_title", "banner_subtitle", "banner_color",
            "audio_provider", "audio_model", "audio_voice", "audio_speed",
            "assistant_defaults_json",
        )
        if not any(col in update_data for col in workspace_update_columns):
            raise InputError("No recognized workspace fields to update.")  # noqa: TRY003

        set_clauses = ["last_modified = ?", "version = ?"]
        params: list[Any] = [now, expected_version + 1]

        for col in workspace_update_columns:
            if col in update_data:
                set_clauses.append(f"{col} = ?")
                if col == "workspace_profile":
                    params.append(self._validate_workspace_profile(update_data[col]))
                elif col == "assistant_defaults_json":
                    stored_defaults = self._serialize_workspace_assistant_defaults_json(update_data[col])
                    params.append(stored_defaults)
                    set_clauses.append("assistant_defaults_explicit_none = ?")
                    params.append(stored_defaults is None)
                else:
                    params.append(update_data[col])

        query = f"UPDATE workspaces SET {', '.join(set_clauses)} WHERE id = ? AND version = ?"  # nosec B608
        params.extend([workspace_id, expected_version])

        with self._ctx.transaction() as conn:
            cursor = conn.execute(query, tuple(params))
            if cursor.rowcount == 0:
                raise ConflictError(  # noqa: TRY003
                    f"Workspace '{workspace_id}' concurrent update detected.",
                    entity="workspaces",
                    entity_id=workspace_id,
                )
        return self.get_workspace(workspace_id)  # type: ignore[return-value]

    def _require_outermost_workspace_delete(self, workspace_id: str) -> None:
        """A staged delete must be able to commit admission closure on its own."""
        if self._ctx.backend_type == BackendType.SQLITE:
            nested = bool(self._ctx.get_connection().in_transaction)
        else:
            connection = self._ctx.get_connection()
            raw = connection._connection
            transaction_status = getattr(getattr(raw, "info", None), "transaction_status", None)
            nested = bool(
                getattr(self._ctx.connection_state(), "tx_depth", 0)
                or connection._backend._tx_depth(raw)
                or getattr(transaction_status, "name", None) != "IDLE"
            )
        if nested:
            raise ConflictError(  # noqa: TRY003
                "Workspace deletion requires an outermost transaction boundary.",
                entity="workspaces",
                entity_id=workspace_id,
            )

    def _lock_native_workspace_delete(self, conn: Any, workspace_id: str) -> Any:
        """Lock only the owned, ordinary workspace during a lifecycle transition."""
        lock = " FOR UPDATE" if self._ctx.backend_type == BackendType.POSTGRESQL else ""
        return conn.execute(
            "SELECT version, deleted, native_chat_admission_closed FROM workspaces "
            "WHERE id = ? AND client_id = ? AND system_operation_state IS NULL" + lock,  # nosec B608 - lock is a fixed backend-only suffix.
            (workspace_id, self._ctx.client_id),
        ).fetchone()

    def _begin_native_workspace_delete(
        self, workspace_id: str, *, expected_version: int | None, hard: bool
    ) -> bool:
        """Durably fence new native admissions before enumerating conversations."""
        self._require_outermost_workspace_delete(workspace_id)
        with self._ctx.transaction() as conn:
            workspace = self._lock_native_workspace_delete(conn, workspace_id)
            if workspace is None and hard:
                return False
            if workspace is None or (not hard and workspace["deleted"]):
                raise ConflictError(  # noqa: TRY003
                    f"Workspace '{workspace_id}' not found.", entity="workspaces", entity_id=workspace_id
                )
            if expected_version is not None and workspace["version"] != expected_version:
                raise ConflictError(  # noqa: TRY003
                    f"Workspace '{workspace_id}' version mismatch.", entity="workspaces", entity_id=workspace_id
                )
            if not workspace["native_chat_admission_closed"]:
                conn.execute(
                    "UPDATE workspaces SET native_chat_admission_closed = ? "
                    "WHERE id = ? AND client_id = ?",
                    (True, workspace_id, self._ctx.client_id),
                )
        return True

    def _finish_native_workspace_delete(
        self, conn: Any, workspace_id: str, *, expected_version: int | None, hard: bool, now: str | None = None
    ) -> None:
        """Stop a final transition if a protected scoped chat escaped the cascade."""
        workspace = self._lock_native_workspace_delete(conn, workspace_id)
        if workspace is None or not workspace["native_chat_admission_closed"]:
            raise self._workspace_delete_incomplete(workspace_id)
        if expected_version is not None and workspace["version"] != expected_version:
            raise self._workspace_delete_incomplete(workspace_id)
        if not hard and workspace["deleted"]:
            raise self._workspace_delete_incomplete(workspace_id)
        deleted_clause = "" if hard else " AND deleted = ?"
        params: tuple[Any, ...] = (workspace_id,)
        if not hard:
            params += (False,)
        residual = conn.execute(
            "SELECT id FROM conversations WHERE workspace_id = ? "  # nosec B608 - only a fixed deleted predicate is appended.
            "AND scope_type = 'workspace'" + deleted_clause + " "
            "AND ((client_id = ? AND (required_projection_version IS NOT NULL OR native_bundle_json IS NOT NULL "
            "OR native_creation_operation_kind IS NOT NULL OR native_creation_operation_id IS NOT NULL)) "
            "OR EXISTS (SELECT 1 FROM workspace_chat_startup_receipts r "
            "WHERE r.conversation_id = conversations.id AND r.owner_user_id = ?)) "
            "LIMIT 1",
            (*params, self._ctx.client_id, self._ctx.owner_user_id),
        ).fetchone()
        if residual is not None:
            raise self._workspace_delete_incomplete(workspace_id)
        if not hard:
            if expected_version is None:
                raise self._workspace_delete_incomplete(workspace_id)
            cursor = conn.execute(
                "UPDATE workspaces SET deleted = 1, last_modified = ?, version = ? "
                "WHERE id = ? AND client_id = ? AND version = ? AND native_chat_admission_closed = ?",
                (now, expected_version + 1, workspace_id, self._ctx.client_id, expected_version, True),
            )
            if cursor.rowcount == 0:
                raise self._workspace_delete_incomplete(workspace_id)

    @staticmethod
    def _workspace_delete_incomplete(workspace_id: str) -> ConflictError:
        """Return the retryable conflict for a fenced but unfinished cascade."""
        return ConflictError("workspace_delete_incomplete", entity="workspaces", entity_id=workspace_id)

    def delete_workspace(
        self,
        workspace_id: str,
        expected_version: int,
    ) -> bool:
        """Soft-delete a workspace and cascade soft-delete its scoped conversations.

        Returns:
            True on success.

        Raises:
            ConflictError: If not found or version mismatch.
        """
        self._begin_native_workspace_delete(workspace_id, expected_version=expected_version, hard=False)

        try:
            return self._cascade_and_finish_soft_workspace_delete(workspace_id, expected_version)
        except Exception as exc:
            raise self._workspace_delete_incomplete(workspace_id) from exc

    def _cascade_and_finish_soft_workspace_delete(self, workspace_id: str, expected_version: int) -> bool:
        """Run the existing per-chat cascade after durable admission closure."""
        # read_only: settle this read's own PostgreSQL transaction, so a failed
        # cascade does not leave the connection INTRANS and have the retry
        # rejected by _require_outermost_workspace_delete.
        conversations = self._ctx.execute_query(
            "SELECT id, version FROM conversations WHERE workspace_id = ? AND scope_type = ? AND deleted = 0",
            (workspace_id, "workspace"),
            read_only=True,
        ).fetchall()
        for conversation in conversations:
            conversation_id = conversation["id"] if isinstance(conversation, dict) else conversation[0]
            conversation_version = conversation["version"] if isinstance(conversation, dict) else conversation[1]
            failed_message_ids: set[str] = set()
            while True:
                # Settle this page's reads before invoking independently owned deletes.
                with self._ctx.transaction():
                    batch = self._ctx.db.get_messages_for_conversation(conversation_id, limit=100, offset=0)
                batch = [m for m in batch if m.get("id") not in failed_message_ids]
                if not batch:
                    break
                deleted_this_batch = 0
                for message in batch:
                    message_id = message.get("id")
                    if not message_id:
                        continue
                    try:
                        self._ctx.db.soft_delete_message(message_id, message.get("version", 1))
                        deleted_this_batch += 1
                    except _CHACHA_NONCRITICAL_EXCEPTIONS:
                        logger.warning(
                            "Failed to soft-delete message {} during workspace delete {}.",
                            message_id,
                            workspace_id,
                        )
                        failed_message_ids.add(str(message_id))
                if deleted_this_batch == 0:
                    break
            self._ctx.db.soft_delete_conversation(conversation_id, int(conversation_version))

        now = self._ctx.utc_now_iso()
        with self._ctx.transaction() as conn:
            conn.execute(
                "UPDATE quizzes "
                "SET workspace_id = NULL, last_modified = ?, version = version + 1, client_id = ? "
                "WHERE workspace_id = ? AND deleted = 0",
                (now, self._ctx.client_id, workspace_id),
            )
            conn.execute(
                "UPDATE decks "
                "SET workspace_id = NULL, last_modified = ?, version = version + 1, client_id = ? "
                "WHERE workspace_id = ? AND deleted = 0",
                (now, self._ctx.client_id, workspace_id),
            )
            self._finish_native_workspace_delete(
                conn, workspace_id, expected_version=expected_version, hard=False, now=now
            )
        return True

    def hard_delete_workspace(self, workspace_id: str) -> None:
        """Permanently delete a workspace row. FK CASCADE handles sub-resources."""
        if not self._begin_native_workspace_delete(workspace_id, expected_version=None, hard=True):
            return
        try:
            self._cascade_and_finish_hard_workspace_delete(workspace_id)
        except Exception as exc:
            raise self._workspace_delete_incomplete(workspace_id) from exc

    def _cascade_and_finish_hard_workspace_delete(self, workspace_id: str) -> None:
        """Purge scoped chats before deleting their workspace row."""
        conversations = self._ctx.execute_query(
            "SELECT id FROM conversations WHERE workspace_id = ? AND scope_type = ?",
            (workspace_id, "workspace"),
            read_only=True,
        ).fetchall()
        for conversation in conversations:
            conversation_id = conversation["id"] if isinstance(conversation, dict) else conversation[0]
            self._ctx.db.hard_delete_conversation(str(conversation_id))
        with self._ctx.transaction() as conn:
            self._finish_native_workspace_delete(conn, workspace_id, expected_version=None, hard=True)
            conn.execute("DELETE FROM workspace_resource_memberships WHERE workspace_id = ?", (workspace_id,))
            conn.execute("DELETE FROM workspace_project_roots WHERE workspace_id = ?", (workspace_id,))
            conn.execute("DELETE FROM workspaces WHERE id = ? AND client_id = ?", (workspace_id, self._ctx.client_id))

    @staticmethod
    def _normalize_workspace_source_saved_view_name(name: str) -> tuple[str, str]:
        """Return the trimmed display name and Python NFKC/casefold key."""
        if not isinstance(name, str):
            raise InputError("Saved view name must be a string.")  # noqa: TRY003
        WorkspaceStore._validate_workspace_source_saved_view_utf8_text(name, "name")
        display_name = name.strip()
        if not SOURCE_VIEW_MIN_NAME_LENGTH <= len(display_name) <= SOURCE_VIEW_MAX_NAME_LENGTH:
            raise InputError(  # noqa: TRY003
                f"Saved view name must contain {SOURCE_VIEW_MIN_NAME_LENGTH} to "
                f"{SOURCE_VIEW_MAX_NAME_LENGTH} characters after trimming."
            )
        return display_name, unicodedata.normalize("NFKC", display_name).casefold()

    @staticmethod
    def _validate_workspace_source_saved_view_state_json(state_json: str) -> str:
        """Validate only the raw UTF-8 storage bound; parsing belongs to the API."""
        if not isinstance(state_json, str):
            raise InputError("Saved view state_json must be a string.")  # noqa: TRY003
        encoded = WorkspaceStore._validate_workspace_source_saved_view_utf8_text(
            state_json,
            "state_json",
        )
        if len(encoded) > SOURCE_VIEW_MAX_STATE_BYTES:
            raise InputError(  # noqa: TRY003
                f"Saved view state_json must not exceed {SOURCE_VIEW_MAX_STATE_BYTES} UTF-8 bytes."
            )
        return state_json

    @staticmethod
    def _validate_workspace_source_saved_view_utf8_text(value: str, field_name: str) -> bytes:
        """Return strict UTF-8 bytes after rejecting database-incompatible NUL text."""
        if "\x00" in value:
            raise InputError(f"Saved view {field_name} must not contain NUL characters.")  # noqa: TRY003
        try:
            return value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise InputError(f"Saved view {field_name} must contain valid UTF-8 text.") from exc  # noqa: TRY003

    @staticmethod
    def _validate_workspace_source_saved_view_schema_version(schema_version: int) -> int:
        """Require a positive portable integer without interpreting its schema."""
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or not 1 <= schema_version <= SOURCE_VIEW_MAX_INTEGER
        ):
            raise InputError(  # noqa: TRY003
                f"Saved view schema_version must be an integer from 1 to {SOURCE_VIEW_MAX_INTEGER}."
            )
        return schema_version

    @staticmethod
    def _validate_workspace_source_saved_view_expected_version(expected_version: int) -> int:
        """Require a positive optimistic-lock version."""
        if (
            isinstance(expected_version, bool)
            or not isinstance(expected_version, int)
            or not 1 <= expected_version <= SOURCE_VIEW_MAX_INTEGER
        ):
            raise InputError(  # noqa: TRY003
                f"Saved view expected_version must be an integer from 1 to {SOURCE_VIEW_MAX_INTEGER}."
            )
        return expected_version

    def _require_workspace_source_saved_view_workspace(
        self,
        conn: Any,
        owner_user_id: str,
        workspace_id: str,
        *,
        lock: bool = False,
    ) -> None:
        """Require an active workspace owned by the explicitly supplied principal."""
        query = "SELECT id FROM workspaces WHERE id = ? AND client_id = ? AND deleted = ?"
        if lock and self._ctx.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE"
        row = conn.execute(
            query,
            (workspace_id, owner_user_id, self._workspace_active_deleted_value()),
        ).fetchone()
        if row is None:
            raise WorkspaceSourceSavedViewNotFoundError

    @staticmethod
    def _workspace_source_saved_view_row(row: Any) -> dict[str, Any]:
        """Convert a saved-view database row without parsing state_json."""
        return dict(row)

    def _get_workspace_source_saved_view_with_conn(
        self,
        conn: Any,
        owner_user_id: str,
        workspace_id: str,
        view_id: str,
    ) -> dict[str, Any] | None:
        """Load one row through the complete owner/workspace/view predicate."""
        row = conn.execute(
            """
            SELECT id, workspace_id, owner_user_id, name, name_key, schema_version,
                   state_json, version, created_at, updated_at
              FROM workspace_source_saved_views
             WHERE owner_user_id = ? AND workspace_id = ? AND id = ?
            """,
            (owner_user_id, workspace_id, view_id),
        ).fetchone()
        return self._workspace_source_saved_view_row(row) if row is not None else None

    def _find_workspace_source_saved_view_name_with_conn(
        self,
        conn: Any,
        owner_user_id: str,
        workspace_id: str,
        name_key: str,
        *,
        exclude_view_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Find an owned name conflict and return only safe response metadata."""
        query = (
            "SELECT id, version FROM workspace_source_saved_views "
            "WHERE owner_user_id = ? AND workspace_id = ? AND name_key = ?"
        )
        params: tuple[Any, ...] = (owner_user_id, workspace_id, name_key)
        if exclude_view_id is not None:
            query += " AND id <> ?"
            params += (exclude_view_id,)
        row = conn.execute(query, params).fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def _raise_workspace_source_saved_view_name_conflict(row: Mapping[str, Any]) -> None:
        """Raise the stable duplicate-name conflict using owned row metadata."""
        raise WorkspaceSourceSavedViewConflictError(
            SOURCE_VIEW_NAME_EXISTS,
            {"view_id": str(row["id"]), "version": int(row["version"])},
        )

    @staticmethod
    def _is_workspace_source_saved_view_sqlite_unique_error(exc: sqlite3.IntegrityError) -> bool:
        """Match only the saved-view owner/workspace/name SQLite unique key."""
        expected_columns = (
            "workspace_source_saved_views.owner_user_id, "
            "workspace_source_saved_views.workspace_id, "
            "workspace_source_saved_views.name_key"
        )
        return (
            getattr(exc, "sqlite_errorname", None) == "SQLITE_CONSTRAINT_UNIQUE"
            and expected_columns in str(exc)
        )

    @staticmethod
    def _is_workspace_source_saved_view_postgres_unique_error(exc: Exception) -> bool:
        """Match the exact safe category or legacy named driver uniqueness error."""
        current: BaseException | None = exc
        while current is not None:
            if isinstance(current, SavedViewNameUniqueConstraintError):
                return True
            sqlstate = getattr(current, "sqlstate", None) or getattr(current, "pgcode", None)
            diagnostics = getattr(current, "diag", None)
            constraint_name = getattr(diagnostics, "constraint_name", None)
            if (
                sqlstate == "23505"
                and constraint_name == "uq_workspace_source_saved_views_owner_name"
            ):
                return True
            current = current.__cause__
        return False

    def _is_workspace_source_saved_view_unique_error(self, exc: Exception) -> bool:
        """Dispatch precise unique-constraint detection by active backend."""
        if isinstance(exc, sqlite3.IntegrityError):
            return self._is_workspace_source_saved_view_sqlite_unique_error(exc)
        return self._is_workspace_source_saved_view_postgres_unique_error(exc)

    def _raise_workspace_source_saved_view_duplicate_from_fresh_transaction(
        self,
        owner_user_id: str,
        workspace_id: str,
        name_key: str,
    ) -> None:
        """Resolve duplicate metadata through a backend-appropriate fresh transaction."""
        def find_scoped_conflict(conn: Any) -> dict[str, Any]:
            self._require_workspace_source_saved_view_workspace(conn, owner_user_id, workspace_id)
            conflict = self._find_workspace_source_saved_view_name_with_conn(
                conn,
                owner_user_id,
                workspace_id,
                name_key,
            )
            if conflict is None:
                raise WorkspaceSourceSavedViewNotFoundError
            return conflict

        if self._ctx.backend_type == BackendType.POSTGRESQL:
            backend = self._ctx.backend
            raw_conn = backend.connect()
            conflict: dict[str, Any] | None = None
            workspace_found = False
            try:
                fresh_pg_conn = BackendConnectionWrapper(self._ctx.db, raw_conn, backend)
                fresh_pg_conn.execute(
                    "SELECT set_config('app.current_user_id', ?, true)",
                    (str(owner_user_id),),
                )
                workspace_found = fresh_pg_conn.execute(
                    "SELECT id FROM workspaces WHERE id = ? AND client_id = ? AND deleted = ?",
                    (workspace_id, owner_user_id, self._workspace_active_deleted_value()),
                ).fetchone() is not None
                if workspace_found:
                    conflict = self._find_workspace_source_saved_view_name_with_conn(
                        fresh_pg_conn,
                        owner_user_id,
                        workspace_id,
                        name_key,
                    )
                raw_conn.commit()
            except BaseException:
                with contextlib.suppress(*_CHACHA_NONCRITICAL_EXCEPTIONS):
                    raw_conn.rollback()
                raise
            finally:
                backend.disconnect(raw_conn)
            if not workspace_found or conflict is None:
                raise WorkspaceSourceSavedViewNotFoundError
            self._raise_workspace_source_saved_view_name_conflict(conflict)

        with self._ctx.transaction() as transaction_conn:
            conflict = find_scoped_conflict(transaction_conn)
        self._raise_workspace_source_saved_view_name_conflict(conflict)

    def create_workspace_source_saved_view(
        self,
        owner_user_id: str,
        workspace_id: str,
        *,
        name: str,
        schema_version: int,
        state_json: str,
    ) -> dict[str, Any]:
        """Create one raw saved view for an active owned workspace."""
        display_name, name_key = self._normalize_workspace_source_saved_view_name(name)
        stored_schema_version = self._validate_workspace_source_saved_view_schema_version(schema_version)
        stored_state_json = self._validate_workspace_source_saved_view_state_json(state_json)
        view_id = str(uuid.uuid4())
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                self._require_workspace_source_saved_view_workspace(
                    conn,
                    owner_user_id,
                    workspace_id,
                    lock=True,
                )
                conflict = self._find_workspace_source_saved_view_name_with_conn(
                    conn,
                    owner_user_id,
                    workspace_id,
                    name_key,
                )
                if conflict is not None:
                    self._raise_workspace_source_saved_view_name_conflict(conflict)
                count_row = conn.execute(
                    "SELECT COUNT(*) AS total FROM workspace_source_saved_views "
                    "WHERE owner_user_id = ? AND workspace_id = ?",
                    (owner_user_id, workspace_id),
                ).fetchone()
                if count_row is not None and int(count_row["total"]) >= SOURCE_VIEW_MAX_COUNT:
                    raise WorkspaceSourceSavedViewConflictError(
                        SOURCE_VIEW_LIMIT_REACHED,
                        {"limit": SOURCE_VIEW_MAX_COUNT},
                    )
                conn.execute(
                    """
                    INSERT INTO workspace_source_saved_views (
                        id, workspace_id, owner_user_id, name, name_key, schema_version,
                        state_json, version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        view_id,
                        workspace_id,
                        owner_user_id,
                        display_name,
                        name_key,
                        stored_schema_version,
                        stored_state_json,
                        now,
                        now,
                    ),
                )
                created = self._get_workspace_source_saved_view_with_conn(
                    conn,
                    owner_user_id,
                    workspace_id,
                    view_id,
                )
                if created is None:  # pragma: no cover - defensive database invariant
                    raise CharactersRAGDBError("Failed to load created workspace source saved view.")
                return created
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            if not self._is_workspace_source_saved_view_unique_error(exc):
                raise
            self._raise_workspace_source_saved_view_duplicate_from_fresh_transaction(
                owner_user_id,
                workspace_id,
                name_key,
            )
            raise AssertionError("unreachable") from exc  # pragma: no cover

    def list_workspace_source_saved_views(
        self,
        owner_user_id: str,
        workspace_id: str,
    ) -> list[dict[str, Any]]:
        """List raw owned saved views in stable newest-first order."""
        with self._ctx.transaction() as conn:
            self._require_workspace_source_saved_view_workspace(conn, owner_user_id, workspace_id)
            rows = conn.execute(
                """
                SELECT id, workspace_id, owner_user_id, name, name_key, schema_version,
                       state_json, version, created_at, updated_at
                  FROM workspace_source_saved_views
                 WHERE owner_user_id = ? AND workspace_id = ?
                 ORDER BY updated_at DESC, name_key ASC, id ASC
                """,
                (owner_user_id, workspace_id),
            ).fetchall()
            return [self._workspace_source_saved_view_row(row) for row in rows]

    def get_workspace_source_saved_view(
        self,
        owner_user_id: str,
        workspace_id: str,
        view_id: str,
    ) -> dict[str, Any]:
        """Get one raw owned saved view or raise the focused not-found error."""
        with self._ctx.transaction() as conn:
            self._require_workspace_source_saved_view_workspace(conn, owner_user_id, workspace_id)
            row = self._get_workspace_source_saved_view_with_conn(
                conn,
                owner_user_id,
                workspace_id,
                view_id,
            )
            if row is None:
                raise WorkspaceSourceSavedViewNotFoundError
            return row

    def update_workspace_source_saved_view(
        self,
        owner_user_id: str,
        workspace_id: str,
        view_id: str,
        *,
        expected_version: int,
        name: str | None = None,
        schema_version: int | None = None,
        state_json: str | None = None,
    ) -> dict[str, Any]:
        """Version-update the supplied raw saved-view fields in one transaction."""
        stored_expected_version = self._validate_workspace_source_saved_view_expected_version(expected_version)
        if name is None and schema_version is None and state_json is None:
            raise InputError("Saved view update requires name, schema_version, or state_json.")  # noqa: TRY003
        display_name: str | None = None
        name_key: str | None = None
        if name is not None:
            display_name, name_key = self._normalize_workspace_source_saved_view_name(name)
        stored_schema_version = (
            self._validate_workspace_source_saved_view_schema_version(schema_version)
            if schema_version is not None
            else None
        )
        stored_state_json = (
            self._validate_workspace_source_saved_view_state_json(state_json)
            if state_json is not None
            else None
        )
        try:
            with self._ctx.transaction() as conn:
                self._require_workspace_source_saved_view_workspace(
                    conn,
                    owner_user_id,
                    workspace_id,
                    lock=True,
                )
                current = self._get_workspace_source_saved_view_with_conn(
                    conn,
                    owner_user_id,
                    workspace_id,
                    view_id,
                )
                if current is None:
                    raise WorkspaceSourceSavedViewNotFoundError
                current_version = int(current["version"])
                if current_version != stored_expected_version:
                    raise WorkspaceSourceSavedViewConflictError(
                        SOURCE_VIEW_VERSION_CONFLICT,
                        {"view_id": view_id, "current_version": current_version},
                    )
                if current_version >= SOURCE_VIEW_MAX_INTEGER:
                    raise InputError(  # noqa: TRY003
                        "Saved view version has reached the maximum supported value."
                    )
                if name_key is not None:
                    conflict = self._find_workspace_source_saved_view_name_with_conn(
                        conn,
                        owner_user_id,
                        workspace_id,
                        name_key,
                        exclude_view_id=view_id,
                    )
                    if conflict is not None:
                        self._raise_workspace_source_saved_view_name_conflict(conflict)

                assignments = ["updated_at = ?", "version = version + 1"]
                params: list[Any] = [self._ctx.utc_now_iso()]
                if display_name is not None and name_key is not None:
                    assignments.extend(("name = ?", "name_key = ?"))
                    params.extend((display_name, name_key))
                if stored_schema_version is not None:
                    assignments.append("schema_version = ?")
                    params.append(stored_schema_version)
                if stored_state_json is not None:
                    assignments.append("state_json = ?")
                    params.append(stored_state_json)
                params.extend((owner_user_id, workspace_id, view_id, stored_expected_version))
                cursor = conn.execute(
                    f"UPDATE workspace_source_saved_views SET {', '.join(assignments)} "  # nosec B608
                    "WHERE owner_user_id = ? AND workspace_id = ? AND id = ? AND version = ?",
                    tuple(params),
                )
                if cursor.rowcount != 1:
                    latest = self._get_workspace_source_saved_view_with_conn(
                        conn,
                        owner_user_id,
                        workspace_id,
                        view_id,
                    )
                    if latest is None:
                        raise WorkspaceSourceSavedViewNotFoundError
                    raise WorkspaceSourceSavedViewConflictError(
                        SOURCE_VIEW_VERSION_CONFLICT,
                        {"view_id": view_id, "current_version": int(latest["version"])},
                    )
                updated = self._get_workspace_source_saved_view_with_conn(
                    conn,
                    owner_user_id,
                    workspace_id,
                    view_id,
                )
                if updated is None:  # pragma: no cover - defensive database invariant
                    raise WorkspaceSourceSavedViewNotFoundError
                return updated
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            if name_key is None or not self._is_workspace_source_saved_view_unique_error(exc):
                raise
            self._raise_workspace_source_saved_view_duplicate_from_fresh_transaction(
                owner_user_id,
                workspace_id,
                name_key,
            )
            raise AssertionError("unreachable") from exc  # pragma: no cover

    def delete_workspace_source_saved_view(
        self,
        owner_user_id: str,
        workspace_id: str,
        view_id: str,
    ) -> None:
        """Unconditionally delete one owned view after active-workspace validation."""
        with self._ctx.transaction() as conn:
            self._require_workspace_source_saved_view_workspace(
                conn,
                owner_user_id,
                workspace_id,
                lock=True,
            )
            cursor = conn.execute(
                "DELETE FROM workspace_source_saved_views "
                "WHERE owner_user_id = ? AND workspace_id = ? AND id = ?",
                (owner_user_id, workspace_id, view_id),
            )
            if cursor.rowcount != 1:
                raise WorkspaceSourceSavedViewNotFoundError

    def upsert_workspace_primary_root(self, workspace_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create or replace the primary project root for a workspace."""
        if not workspace_id:
            raise InputError("workspace_id is required.")  # noqa: TRY003
        root_id = str(data.get("root_id") or "").strip()
        if not root_id:
            raise InputError("root_id is required.")  # noqa: TRY003
        backend = self._validate_workspace_project_root_backend(data.get("backend"))
        expected_workspace_version_raw = data.get("expected_workspace_version")
        expected_workspace_version: int | None = None
        if expected_workspace_version_raw is not None:
            if isinstance(expected_workspace_version_raw, bool):
                raise InputError("expected_workspace_version must be an integer when provided.")  # noqa: TRY003
            if isinstance(expected_workspace_version_raw, int):
                expected_workspace_version = expected_workspace_version_raw
            elif isinstance(expected_workspace_version_raw, str):
                normalized_expected_version = expected_workspace_version_raw.strip()
                if not normalized_expected_version or not normalized_expected_version.isdecimal():
                    raise InputError("expected_workspace_version must be an integer when provided.")  # noqa: TRY003
                expected_workspace_version = int(normalized_expected_version)
            else:
                raise InputError("expected_workspace_version must be an integer when provided.")  # noqa: TRY003
        replace_existing = bool(data.get("replace_existing", False))
        metadata_json = (
            self._serialize_workspace_project_root_metadata(data.get("metadata_json"))
            if "metadata_json" in data
            else None
        )
        now = self._ctx.utc_now_iso()
        primary_value = self._workspace_primary_root_value()
        deleted_value = self._workspace_active_deleted_value()

        values = {
            "root_id": root_id,
            "workspace_id": workspace_id,
            "is_primary": primary_value,
            "backend": backend,
            "absolute_root": data.get("absolute_root"),
            "sandbox_volume_id": data.get("sandbox_volume_id"),
            "display_name": data.get("display_name"),
            "root_state": data.get("root_state") or "not_configured",
            "git_state": data.get("git_state") or "absent",
            "file_inventory_state": data.get("file_inventory_state") or "not_started",
            "indexing_state": data.get("indexing_state") or "disabled",
            "sandbox_mount_state": data.get("sandbox_mount_state") or "not_configured",
            "mcp_trust_state": data.get("mcp_trust_state") or "not_configured",
            "metadata_json": metadata_json or "{}",
            "created_at": now,
            "updated_at": now,
            "version": 1,
        }

        try:
            with self._ctx.transaction() as conn:
                def update_workspace_for_root_attach() -> None:
                    if expected_workspace_version is None:
                        conn.execute(
                            """
                            UPDATE workspaces
                               SET workspace_profile = 'project',
                                   last_modified = ?,
                                   version = version + 1
                             WHERE id = ? AND deleted = ?
                            """,
                            (now, workspace_id, deleted_value),
                        )
                        return

                    cursor = conn.execute(
                        """
                        UPDATE workspaces
                           SET workspace_profile = 'project',
                               last_modified = ?,
                               version = version + 1
                         WHERE id = ? AND deleted = ? AND version = ?
                        """,
                        (now, workspace_id, deleted_value, expected_workspace_version),
                    )
                    if cursor.rowcount == 0:
                        raise ConflictError(
                            f"Workspace '{workspace_id}' concurrent root attach detected.",
                            entity="workspaces",
                            entity_id=workspace_id,
                        )

                workspace_row = conn.execute(
                    "SELECT id, workspace_profile, version FROM workspaces WHERE id = ? AND deleted = ?",
                    (workspace_id, deleted_value),
                ).fetchone()
                if workspace_row is None:
                    raise ConflictError(
                        f"Workspace '{workspace_id}' not found.",
                        entity="workspaces",
                        entity_id=workspace_id,
                    )
                workspace = dict(workspace_row)
                if (
                    expected_workspace_version is not None
                    and int(workspace["version"]) != expected_workspace_version
                ):
                    raise ConflictError(
                        f"Workspace '{workspace_id}' version mismatch "
                        f"(db={workspace['version']}, expected={expected_workspace_version}).",
                        entity="workspaces",
                        entity_id=workspace_id,
                    )
                existing_primary_row = conn.execute(
                    """
                    SELECT *
                      FROM workspace_project_roots
                     WHERE workspace_id = ?
                       AND is_primary = ?
                    """,
                    (workspace_id, primary_value),
                ).fetchone()
                existing_primary = dict(existing_primary_row) if existing_primary_row else None
                root_binding_changed = existing_primary is None or existing_primary.get("root_id") != root_id

                if existing_primary is not None and existing_primary.get("root_id") == root_id:
                    if existing_primary.get("backend") != backend:
                        if not replace_existing:
                            raise ConflictError(
                                f"Workspace project root '{root_id}' backend mismatch.",
                                entity="workspace_project_roots",
                                entity_id=root_id,
                            )
                        root_binding_changed = True
                    else:
                        binding_columns = (
                            "absolute_root",
                            "sandbox_volume_id",
                            "display_name",
                            "metadata_json",
                        )
                        operational_columns = (
                            "root_state",
                            "git_state",
                            "file_inventory_state",
                            "indexing_state",
                            "sandbox_mount_state",
                            "mcp_trust_state",
                        )
                        root_updates: dict[str, Any] = {}
                        for column in binding_columns:
                            if column not in data:
                                continue
                            new_value = metadata_json if column == "metadata_json" else values[column]
                            if existing_primary.get(column) != new_value:
                                root_updates[column] = new_value
                        for column in operational_columns:
                            if column not in data:
                                continue
                            new_value = values[column]
                            if existing_primary.get(column) != new_value:
                                root_updates[column] = new_value
                        if root_updates:
                            root_binding_changed = True
                            set_clauses = ["updated_at = ?", "version = ?"]
                            params: list[Any] = [now, int(existing_primary["version"]) + 1]
                            for column, value in root_updates.items():
                                set_clauses.append(f"{column} = ?")
                                params.append(value)
                            params.extend([workspace_id, root_id])
                            conn.execute(
                                f"""
                                UPDATE workspace_project_roots
                                   SET {', '.join(set_clauses)}
                                 WHERE workspace_id = ?
                                   AND root_id = ?
                                """,  # nosec B608
                                tuple(params),
                            )
                        root = self._workspace_project_root_row_by_id(conn, workspace_id, root_id)
                        if root is None:
                            raise CharactersRAGDBError(  # noqa: TRY003
                                f"Failed to reload workspace project root '{root_id}'."
                            )
                        if root_binding_changed or workspace.get("workspace_profile") != "project":
                            update_workspace_for_root_attach()
                        return root
                if existing_primary is None or existing_primary.get("root_id") != root_id or replace_existing:
                    conn.execute(
                        "DELETE FROM workspace_project_roots WHERE workspace_id = ? AND is_primary = ?",
                        (workspace_id, primary_value),
                    )
                    conn.execute(
                        """
                        INSERT INTO workspace_project_roots (
                            root_id, workspace_id, is_primary, backend, absolute_root, sandbox_volume_id,
                            display_name, root_state, git_state, file_inventory_state, indexing_state,
                            sandbox_mount_state, mcp_trust_state, metadata_json, created_at, updated_at, version
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            values["root_id"],
                            values["workspace_id"],
                            values["is_primary"],
                            values["backend"],
                            values["absolute_root"],
                            values["sandbox_volume_id"],
                            values["display_name"],
                            values["root_state"],
                            values["git_state"],
                            values["file_inventory_state"],
                            values["indexing_state"],
                            values["sandbox_mount_state"],
                            values["mcp_trust_state"],
                            values["metadata_json"],
                            values["created_at"],
                            values["updated_at"],
                            values["version"],
                        ),
                    )
                if root_binding_changed or workspace.get("workspace_profile") != "project":
                    update_workspace_for_root_attach()
                root = self._workspace_project_root_row_by_id(conn, workspace_id, root_id)
                if root is None:
                    raise CharactersRAGDBError(  # noqa: TRY003
                        f"Failed to reload workspace project root '{root_id}'."
                    )
                return root
        except ConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            raise self._workspace_project_root_integrity_error(exc) from exc
        except BackendDatabaseError as exc:
            raise self._workspace_project_root_integrity_error(exc) from exc
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Workspace project root write failed: {exc}") from exc  # noqa: TRY003

    def get_workspace_primary_root(self, workspace_id: str) -> dict[str, Any] | None:
        """Return the active primary project root for a non-deleted workspace."""
        if not workspace_id:
            return None
        cursor = self._ctx.execute_query(
            """
            SELECT r.*
              FROM workspace_project_roots r
              JOIN workspaces w ON w.id = r.workspace_id
             WHERE r.workspace_id = ?
               AND r.is_primary = ?
               AND w.deleted = ?
             ORDER BY r.created_at DESC
             LIMIT 1
            """,
            (workspace_id, self._workspace_primary_root_value(), self._workspace_active_deleted_value()),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_workspace_project_roots(self, workspace_id: str) -> list[dict[str, Any]]:
        """Return project roots for a non-deleted workspace."""
        if not workspace_id:
            return []
        cursor = self._ctx.execute_query(
            """
            SELECT r.*
              FROM workspace_project_roots r
              JOIN workspaces w ON w.id = r.workspace_id
             WHERE r.workspace_id = ?
               AND w.deleted = ?
             ORDER BY r.is_primary DESC, r.created_at ASC, r.root_id ASC
            """,
            (workspace_id, self._workspace_active_deleted_value()),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_workspace_project_root_state(
        self,
        workspace_id: str,
        root_id: str,
        updates: dict[str, Any],
        expected_version: int,
    ) -> dict[str, Any]:
        """Update mutable project-root state with optimistic locking."""
        if not workspace_id or not root_id:
            raise InputError("workspace_id and root_id are required.")  # noqa: TRY003
        operational_updates = {
            "root_state",
            "git_state",
            "file_inventory_state",
            "indexing_state",
            "sandbox_mount_state",
            "mcp_trust_state",
        }
        clean_updates: dict[str, Any] = {}
        for key, value in (updates or {}).items():
            if key not in operational_updates:
                raise InputError(f"Unsupported workspace project root state field: {key}")  # noqa: TRY003
            clean_updates[key] = value
        existing = self.get_workspace_primary_root(workspace_id)
        if existing is None or existing.get("root_id") != root_id:
            raise ConflictError(
                f"Workspace project root '{root_id}' not found.",
                entity="workspace_project_roots",
                entity_id=root_id,
            )
        if int(existing["version"]) != int(expected_version):
            raise ConflictError(
                f"Workspace project root '{root_id}' version mismatch.",
                entity="workspace_project_roots",
                entity_id=root_id,
            )
        if not clean_updates:
            return existing

        now = self._ctx.utc_now_iso()
        set_clauses = ["updated_at = ?", "version = ?"]
        params: list[Any] = [now, int(expected_version) + 1]
        for key, value in clean_updates.items():
            set_clauses.append(f"{key} = ?")
            params.append(value)
        params.extend([workspace_id, root_id, int(expected_version)])

        try:
            with self._ctx.transaction() as conn:
                cursor = conn.execute(
                    f"""
                    UPDATE workspace_project_roots
                       SET {', '.join(set_clauses)}
                     WHERE workspace_id = ?
                       AND root_id = ?
                       AND version = ?
                    """,  # nosec B608
                    tuple(params),
                )
                if cursor.rowcount == 0:
                    raise ConflictError(
                        f"Workspace project root '{root_id}' concurrent update detected.",
                        entity="workspace_project_roots",
                        entity_id=root_id,
                    )
                root = self._workspace_project_root_row_by_id(conn, workspace_id, root_id)
                if root is None:
                    raise CharactersRAGDBError(  # noqa: TRY003
                        f"Failed to reload workspace project root '{root_id}'."
                    )
                return root
        except ConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            raise self._workspace_project_root_integrity_error(exc) from exc
        except BackendDatabaseError as exc:
            raise self._workspace_project_root_integrity_error(exc) from exc
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Workspace project root update failed: {exc}") from exc  # noqa: TRY003

    @staticmethod
    def _validate_workspace_operation_status(status: Any) -> str:
        normalized = str(status or "").strip().lower()
        if normalized not in WORKSPACE_OPERATION_STATUSES:
            raise InputError("workspace operation status is invalid.")  # noqa: TRY003
        return normalized

    @staticmethod
    def _workspace_operation_json(value: Any, *, diagnostics: bool = False) -> str:
        payload = redact_operation_diagnostics(value) if diagnostics else (value or {})
        if isinstance(payload, str):
            try:
                json.loads(payload)
            except json.JSONDecodeError as exc:
                raise InputError("workspace operation JSON fields must contain valid JSON.") from exc  # noqa: TRY003
            return payload
        try:
            return json.dumps(payload, ensure_ascii=True)
        except TypeError as exc:
            raise InputError("workspace operation values must be JSON serializable.") from exc  # noqa: TRY003

    @staticmethod
    def _workspace_operation_json_load(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if value is None:
            return {}
        try:
            loaded = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return dict(loaded) if isinstance(loaded, dict) else {}

    def _workspace_operation_row(self, row: Any) -> dict[str, Any] | None:
        if row is None:
            return None
        payload = dict(row)
        payload["result_ref"] = self._workspace_operation_json_load(payload.pop("result_ref_json", None))
        payload["diagnostics"] = redact_operation_diagnostics(
            self._workspace_operation_json_load(payload.pop("diagnostics_json", None))
        )
        return payload

    @staticmethod
    def _workspace_operation_integrity_error(exc: Exception) -> CharactersRAGDBError:
        message = str(exc).lower()
        if "unique" in message or "duplicate" in message or "foreign key" in message or "constraint" in message:
            return ConflictError(
                f"Workspace operation write conflicted: {exc}",
                entity="workspace_operations",
            )
        return CharactersRAGDBError(f"Workspace operation write failed: {exc}")  # noqa: TRY003

    @staticmethod
    def _workspace_operation_timestamp(value: datetime | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc).isoformat()
        text = str(value).strip()
        return text or None

    def create_workspace_operation(
        self,
        *,
        workspace_id: str,
        user_id: str,
        command: str,
        idempotency_key: str,
        request_fingerprint: str,
        linked_idempotency_key: str | None = None,
        status: str = "queued",
        result_ref: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        expires_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        """Create an idempotent Workspace operation or return the matching existing row."""
        workspace_key = str(workspace_id or "").strip()
        user_key = str(user_id or "").strip()
        command_key = str(command or "").strip()
        key = str(idempotency_key or "").strip()
        fingerprint = str(request_fingerprint or "").strip()
        if not workspace_key or not user_key or not command_key or not key or not fingerprint:
            raise InputError(
                "workspace_id, user_id, command, idempotency_key, and request_fingerprint are required."
            )  # noqa: TRY003
        normalized_status = self._validate_workspace_operation_status(status)
        self._delete_expired_workspace_operation_idempotency(
            workspace_id=workspace_key,
            user_id=user_key,
            command=command_key,
            idempotency_key=key,
        )
        existing = self.get_workspace_operation_by_idempotency(
            workspace_id=workspace_key,
            user_id=user_key,
            command=command_key,
            idempotency_key=key,
        )
        if existing is not None:
            if existing.get("request_fingerprint") != fingerprint:
                raise ConflictError(
                    "Workspace operation idempotency key was reused with a different request.",
                    entity="workspace_operations",
                    entity_id=str(existing.get("id") or ""),
                )
            return existing
        if self.get_workspace(workspace_key) is None:
            raise ConflictError(
                f"Workspace '{workspace_key}' not found.",
                entity="workspaces",
                entity_id=workspace_key,
            )

        now = self._ctx.utc_now_iso()
        operation_id = f"wop_{uuid.uuid4().hex}"
        expiry = self._workspace_operation_timestamp(expires_at)
        if expiry is None:
            expiry = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        params = (
            operation_id,
            workspace_key,
            user_key,
            command_key,
            key,
            fingerprint,
            linked_idempotency_key,
            normalized_status,
            self._workspace_operation_json(result_ref),
            self._workspace_operation_json(diagnostics, diagnostics=True),
            now,
            now,
            expiry,
        )
        try:
            with self._ctx.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO workspace_operations (
                        id, workspace_id, user_id, command, idempotency_key,
                        request_fingerprint, linked_idempotency_key, status,
                        result_ref_json, diagnostics_json, created_at, updated_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                )
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            raced = self.get_workspace_operation_by_idempotency(
                workspace_id=workspace_key,
                user_id=user_key,
                command=command_key,
                idempotency_key=key,
            )
            if raced is not None:
                if raced.get("request_fingerprint") == fingerprint:
                    return raced
                raise ConflictError(
                    "Workspace operation idempotency key was reused with a different request.",
                    entity="workspace_operations",
                    entity_id=str(raced.get("id") or ""),
                ) from exc
            raise self._workspace_operation_integrity_error(exc) from exc
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Workspace operation create failed: {exc}") from exc  # noqa: TRY003
        created = self.get_workspace_operation(workspace_key, operation_id)
        if created is None:
            raise CharactersRAGDBError("Failed to reload created workspace operation.")  # noqa: TRY003
        return created

    def _delete_expired_workspace_operation_idempotency(
        self,
        *,
        workspace_id: str,
        user_id: str,
        command: str,
        idempotency_key: str,
    ) -> int:
        """Remove an expired operation envelope before reusing an idempotency tuple."""
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM workspace_operations
                     WHERE workspace_id = ?
                       AND user_id = ?
                       AND command = ?
                       AND idempotency_key = ?
                       AND expires_at IS NOT NULL
                       AND expires_at < ?
                    """,
                    (workspace_id, user_id, command, idempotency_key, now),
                )
                return int(getattr(cursor, "rowcount", 0) or 0)
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise self._workspace_operation_integrity_error(exc) from exc

    def get_workspace_operation(self, workspace_id: str, operation_id: str) -> dict[str, Any] | None:
        """Return a Workspace operation by id."""
        workspace_key = str(workspace_id or "").strip()
        operation_key = str(operation_id or "").strip()
        if not workspace_key or not operation_key:
            return None
        cursor = self._ctx.execute_query(
            """
            SELECT *
              FROM workspace_operations
             WHERE workspace_id = ?
               AND id = ?
            """,
            (workspace_key, operation_key),
        )
        return self._workspace_operation_row(cursor.fetchone())

    def get_workspace_operation_by_idempotency(
        self,
        *,
        workspace_id: str,
        user_id: str,
        command: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        """Return a Workspace operation by command idempotency key."""
        workspace_key = str(workspace_id or "").strip()
        user_key = str(user_id or "").strip()
        command_key = str(command or "").strip()
        key = str(idempotency_key or "").strip()
        if not workspace_key or not user_key or not command_key or not key:
            return None
        cursor = self._ctx.execute_query(
            """
            SELECT *
              FROM workspace_operations
             WHERE workspace_id = ?
               AND user_id = ?
               AND command = ?
               AND idempotency_key = ?
            """,
            (workspace_key, user_key, command_key, key),
        )
        return self._workspace_operation_row(cursor.fetchone())

    def update_workspace_operation(
        self,
        workspace_id: str,
        operation_id: str,
        *,
        status: str | None = None,
        result_ref: Mapping[str, Any] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        expires_at: datetime | str | None = None,
    ) -> dict[str, Any]:
        """Update mutable Workspace operation fields."""
        existing = self.get_workspace_operation(workspace_id, operation_id)
        if existing is None:
            raise ConflictError(
                f"Workspace operation '{operation_id}' not found.",
                entity="workspace_operations",
                entity_id=operation_id,
            )
        set_clauses = ["updated_at = ?"]
        params: list[Any] = [self._ctx.utc_now_iso()]
        if status is not None:
            set_clauses.append("status = ?")
            params.append(self._validate_workspace_operation_status(status))
        if result_ref is not None:
            set_clauses.append("result_ref_json = ?")
            params.append(self._workspace_operation_json(result_ref))
        if diagnostics is not None:
            set_clauses.append("diagnostics_json = ?")
            params.append(self._workspace_operation_json(diagnostics, diagnostics=True))
        if expires_at is not None:
            set_clauses.append("expires_at = ?")
            params.append(self._workspace_operation_timestamp(expires_at))
        params.extend([workspace_id, operation_id])
        try:
            with self._ctx.transaction() as conn:
                cursor = conn.execute(
                    f"""
                    UPDATE workspace_operations
                       SET {', '.join(set_clauses)}
                     WHERE workspace_id = ?
                       AND id = ?
                    """,  # nosec B608
                    tuple(params),
                )
                if cursor.rowcount == 0:
                    raise ConflictError(
                        f"Workspace operation '{operation_id}' concurrent update detected.",
                        entity="workspace_operations",
                        entity_id=operation_id,
                    )
        except ConflictError:
            raise
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Workspace operation update failed: {exc}") from exc  # noqa: TRY003
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Workspace operation update failed: {exc}") from exc  # noqa: TRY003
        updated = self.get_workspace_operation(workspace_id, operation_id)
        if updated is None:
            raise CharactersRAGDBError("Failed to reload updated workspace operation.")  # noqa: TRY003
        return updated

    def cleanup_expired_workspace_operations(self, *, now: datetime | str | None = None) -> int:
        """Delete expired operation envelopes without touching linked Workspace roots."""
        now_iso = self._workspace_operation_timestamp(now) or datetime.now(timezone.utc).isoformat()
        try:
            with self._ctx.transaction() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM workspace_operations
                     WHERE expires_at IS NOT NULL
                       AND expires_at <= ?
                    """,
                    (now_iso,),
                )
                return int(getattr(cursor, "rowcount", 0) or 0)
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Workspace operation cleanup failed: {exc}") from exc  # noqa: TRY003
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Workspace operation cleanup failed: {exc}") from exc  # noqa: TRY003

    def list_active_workspace_operations(self, workspace_id: str) -> list[dict[str, Any]]:
        """Return non-expired active operations for a workspace."""
        workspace_key = str(workspace_id or "").strip()
        if not workspace_key:
            return []
        now_iso = datetime.now(timezone.utc).isoformat()
        active_statuses = sorted(WORKSPACE_OPERATION_ACTIVE_STATUSES)
        cursor = self._ctx.execute_query(
            """
            SELECT *
              FROM workspace_operations
             WHERE workspace_id = ?
               AND status IN (?, ?)
               AND (expires_at IS NULL OR expires_at > ?)
             ORDER BY created_at ASC, id ASC
            """,
            (workspace_key, active_statuses[0], active_statuses[1], now_iso),
        )
        return [
            operation
            for row in cursor.fetchall()
            if (operation := self._workspace_operation_row(row)) is not None
        ]

    def _workspace_file_inventory_bool_value(self, value: bool) -> bool | int:
        return bool(value) if self._ctx.backend_type == BackendType.POSTGRESQL else int(bool(value))

    @staticmethod
    def _workspace_file_inventory_integrity_error(exc: Exception) -> CharactersRAGDBError:
        message = str(exc).lower()
        if "unique" in message or "duplicate" in message or "foreign key" in message or "constraint" in message:
            return ConflictError(
                f"Workspace file inventory write conflicted: {exc}",
                entity="workspace_file_inventory",
            )
        return CharactersRAGDBError(f"Workspace file inventory write failed: {exc}")  # noqa: TRY003

    @staticmethod
    @overload
    def _workspace_file_inventory_int(value: Any, *, field_name: str, required: Literal[True]) -> int: ...
    @staticmethod
    @overload
    def _workspace_file_inventory_int(value: Any, *, field_name: str, required: bool = False) -> int | None: ...
    @staticmethod
    def _workspace_file_inventory_int(value: Any, *, field_name: str, required: bool = False) -> int | None:
        if value is None:
            if required:
                raise InputError(f"{field_name} is required.")  # noqa: TRY003
            return None
        if isinstance(value, bool):
            raise InputError(f"{field_name} must be an integer.")  # noqa: TRY003
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and (stripped.isdecimal() or (stripped.startswith("-") and stripped[1:].isdecimal())):
                return int(stripped)
        raise InputError(f"{field_name} must be an integer.")  # noqa: TRY003

    @staticmethod
    def _workspace_file_inventory_json(value: Any, *, default: Any) -> str:
        if value is None:
            value = default
        if isinstance(value, str):
            try:
                json.loads(value)
            except json.JSONDecodeError as exc:
                raise InputError("workspace file inventory JSON fields must contain valid JSON.") from exc  # noqa: TRY003
            return value
        try:
            return json.dumps(value, ensure_ascii=True)
        except TypeError as exc:
            raise InputError("workspace file inventory values must be JSON serializable.") from exc  # noqa: TRY003

    @staticmethod
    def _workspace_file_inventory_json_load(value: Any, *, fallback: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str) or not value.strip():
            return fallback
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback

    def _workspace_file_inventory_counts_json(self, value: Any) -> str:
        return json.dumps(normalize_inventory_counts(value), ensure_ascii=True)

    def _workspace_file_inventory_diagnostics_json(self, value: Any) -> str:
        return json.dumps(bounded_inventory_diagnostics(value), ensure_ascii=True)

    def _workspace_file_inventory_scan_by_id(
        self,
        scan_id: str,
        *,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM workspace_file_inventory_scans WHERE scan_id = ?"
        if conn is None:
            row = self._ctx.execute_query(query, (scan_id,)).fetchone()
        else:
            row = conn.execute(query, (scan_id,)).fetchone()
        return dict(row) if row else None

    def _workspace_file_inventory_require_scan(
        self,
        scan_id: str,
        *,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> dict[str, Any]:
        scan = self._workspace_file_inventory_scan_by_id(scan_id, conn=conn)
        if scan is None:
            raise ConflictError(
                f"Workspace file inventory scan '{scan_id}' not found.",
                entity="workspace_file_inventory_scans",
                entity_id=scan_id,
            )
        return scan

    def _workspace_file_inventory_update_root_state(
        self,
        conn: sqlite3.Connection | BackendConnectionWrapper,
        *,
        workspace_id: str,
        root_id: str,
        state: str,
        updated_at: str,
    ) -> None:
        conn.execute(
            """
            UPDATE workspace_project_roots
               SET file_inventory_state = ?,
                   updated_at = ?
             WHERE workspace_id = ?
               AND root_id = ?
            """,
            (state, updated_at, workspace_id, root_id),
        )

    @staticmethod
    def _workspace_file_inventory_relative_path(value: Any) -> str:
        try:
            return sort_inventory_relative_paths([str(value or "")])[0]
        except ValueError as exc:
            raise InputError("relative_path must be a safe project-root-relative path.") from exc  # noqa: TRY003

    @staticmethod
    def _workspace_file_inventory_prefix(value: Any) -> str | None:
        if value is None:
            return None
        raw = str(value).strip().replace("\\", "/")
        if not raw:
            return None
        if raw.startswith("/") or raw.startswith("~"):
            raise InputError("prefix must be project-root-relative.")  # noqa: TRY003
        had_trailing_slash = raw.endswith("/")
        parts: list[str] = []
        for part in raw.split("/"):
            if not part or part == ".":
                continue
            if part == "..":
                raise InputError("prefix must not contain parent directory traversal.")  # noqa: TRY003
            parts.append(part)
        if not parts:
            return None
        normalized = "/".join(parts)
        return f"{normalized}/" if had_trailing_slash else normalized

    def begin_workspace_file_inventory_scan(
        self,
        workspace_id: str,
        root_id: str,
        root_version: int,
        policy_fingerprint: str,
        *,
        requested_by: str | None = None,
    ) -> dict[str, Any]:
        """Create a queued Workspace file inventory scan or reuse an active scan with an attached job."""
        normalized_workspace_id = str(workspace_id or "").strip()
        normalized_root_id = str(root_id or "").strip()
        normalized_policy = str(policy_fingerprint or "").strip()
        normalized_root_version = self._workspace_file_inventory_int(
            root_version,
            field_name="root_version",
            required=True,
        )
        if not normalized_workspace_id or not normalized_root_id or not normalized_policy:
            raise InputError("workspace_id, root_id, and policy_fingerprint are required.")  # noqa: TRY003

        now = self._ctx.utc_now_iso()
        scan_id = f"ws-file-scan-{uuid.uuid4().hex}"
        try:
            with self._ctx.transaction() as conn:
                root = self._workspace_project_root_row_by_id(conn, normalized_workspace_id, normalized_root_id)
                if root is None:
                    raise ConflictError(
                        f"Workspace project root '{normalized_root_id}' not found.",
                        entity="workspace_project_roots",
                        entity_id=normalized_root_id,
                    )
                if int(root["version"]) != int(normalized_root_version):
                    raise ConflictError(
                        f"Workspace project root '{normalized_root_id}' version mismatch.",
                        entity="workspace_project_roots",
                        entity_id=normalized_root_id,
                    )
                active = conn.execute(
                    """
                    SELECT *
                      FROM workspace_file_inventory_scans
                     WHERE workspace_id = ?
                       AND root_id = ?
                       AND root_version = ?
                       AND ignore_policy_fingerprint = ?
                       AND state IN ('queued', 'scanning')
                       AND job_id IS NOT NULL
                     ORDER BY created_at DESC, scan_id DESC
                     LIMIT 1
                    """,
                    (
                        normalized_workspace_id,
                        normalized_root_id,
                        normalized_root_version,
                        normalized_policy,
                    ),
                ).fetchone()
                if active is not None:
                    return dict(active)

                conn.execute(
                    """
                    INSERT INTO workspace_file_inventory_scans (
                        scan_id, workspace_id, root_id, root_version, state, requested_by,
                        ignore_policy_fingerprint, counts_json, diagnostics_json, created_at, updated_at, version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        scan_id,
                        normalized_workspace_id,
                        normalized_root_id,
                        normalized_root_version,
                        "queued",
                        str(requested_by).strip() if requested_by is not None else None,
                        normalized_policy,
                        self._workspace_file_inventory_counts_json({}),
                        self._workspace_file_inventory_diagnostics_json([]),
                        now,
                        now,
                    ),
                )
                self._workspace_file_inventory_update_root_state(
                    conn,
                    workspace_id=normalized_workspace_id,
                    root_id=normalized_root_id,
                    state="queued",
                    updated_at=now,
                )
                created = self._workspace_file_inventory_scan_by_id(scan_id, conn=conn)
                if created is None:
                    raise CharactersRAGDBError(  # noqa: TRY003
                        f"Failed to read workspace file inventory scan '{scan_id}' after create."
                    )
                return created
        except ConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            raise self._workspace_file_inventory_integrity_error(exc) from exc
        except BackendDatabaseError as exc:
            raise self._workspace_file_inventory_integrity_error(exc) from exc
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Workspace file inventory scan create failed: {exc}") from exc  # noqa: TRY003

    def attach_workspace_file_inventory_job(self, scan_id: str, job_row: Mapping[str, Any]) -> dict[str, Any]:
        """Attach a Jobs row to a queued Workspace file inventory scan idempotently."""
        normalized_scan_id = str(scan_id or "").strip()
        if not normalized_scan_id:
            raise InputError("scan_id is required.")  # noqa: TRY003
        if not isinstance(job_row, Mapping):
            raise InputError("job_row must be a mapping.")  # noqa: TRY003
        job_id = self._workspace_file_inventory_int(
            job_row.get("job_id", job_row.get("id")),
            field_name="job_id",
            required=True,
        )
        job_uuid = str(job_row.get("job_uuid") or job_row.get("uuid") or job_id).strip()
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                scan = self._workspace_file_inventory_require_scan(normalized_scan_id, conn=conn)
                if scan.get("job_id") is not None:
                    if int(scan["job_id"]) == int(job_id) and str(scan.get("job_uuid") or "") == job_uuid:
                        return scan
                    raise ConflictError(
                        f"Workspace file inventory scan '{normalized_scan_id}' already has a different job.",
                        entity="workspace_file_inventory_scans",
                        entity_id=normalized_scan_id,
                    )
                conn.execute(
                    """
                    UPDATE workspace_file_inventory_scans
                       SET job_id = ?,
                           job_uuid = ?,
                           updated_at = ?,
                           version = version + 1
                     WHERE scan_id = ?
                    """,
                    (job_id, job_uuid, now, normalized_scan_id),
                )
                updated = self._workspace_file_inventory_scan_by_id(normalized_scan_id, conn=conn)
                if updated is None:
                    raise CharactersRAGDBError(  # noqa: TRY003
                        f"Failed to read workspace file inventory scan '{normalized_scan_id}' after job attach."
                    )
                return updated
        except ConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            raise self._workspace_file_inventory_integrity_error(exc) from exc
        except BackendDatabaseError as exc:
            raise self._workspace_file_inventory_integrity_error(exc) from exc
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Workspace file inventory job attach failed: {exc}") from exc  # noqa: TRY003

    def mark_workspace_file_inventory_enqueue_failed(
        self,
        scan_id: str,
        diagnostics: Any,
    ) -> dict[str, Any]:
        """Mark a queued scan failed when Jobs enqueue fails after scan-row creation."""
        return self._transition_workspace_file_inventory_scan(
            scan_id,
            state="failed",
            diagnostics=diagnostics,
            set_completed=True,
            set_started=False,
        )

    def mark_workspace_file_inventory_scanning(self, scan_id: str) -> dict[str, Any]:
        """Mark a Workspace file inventory scan as acquired by a worker."""
        return self._transition_workspace_file_inventory_scan(
            scan_id,
            state="scanning",
            diagnostics=None,
            set_completed=False,
            set_started=True,
        )

    def _transition_workspace_file_inventory_scan(
        self,
        scan_id: str,
        *,
        state: str,
        diagnostics: Any,
        set_completed: bool,
        set_started: bool,
    ) -> dict[str, Any]:
        normalized_scan_id = str(scan_id or "").strip()
        normalized_state = normalize_durable_inventory_state(state)
        if not normalized_scan_id:
            raise InputError("scan_id is required.")  # noqa: TRY003
        now = self._ctx.utc_now_iso()
        diagnostics_json = self._workspace_file_inventory_diagnostics_json(diagnostics)
        try:
            with self._ctx.transaction() as conn:
                scan = self._workspace_file_inventory_require_scan(normalized_scan_id, conn=conn)
                set_clauses = ["state = ?", "updated_at = ?", "version = version + 1"]
                params: list[Any] = [normalized_state, now]
                if diagnostics is not None:
                    set_clauses.append("diagnostics_json = ?")
                    params.append(diagnostics_json)
                if set_started:
                    set_clauses.append("started_at = COALESCE(started_at, ?)")
                    params.append(now)
                if set_completed:
                    set_clauses.append("completed_at = ?")
                    params.append(now)
                params.append(normalized_scan_id)
                conn.execute(
                    f"""
                    UPDATE workspace_file_inventory_scans
                       SET {', '.join(set_clauses)}
                     WHERE scan_id = ?
                    """,  # nosec B608
                    tuple(params),
                )
                self._workspace_file_inventory_update_root_state(
                    conn,
                    workspace_id=str(scan["workspace_id"]),
                    root_id=str(scan["root_id"]),
                    state=normalized_state,
                    updated_at=now,
                )
                updated = self._workspace_file_inventory_scan_by_id(normalized_scan_id, conn=conn)
                if updated is None:
                    raise CharactersRAGDBError(  # noqa: TRY003
                        f"Failed to read workspace file inventory scan '{normalized_scan_id}' after update."
                    )
                return updated
        except ConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            raise self._workspace_file_inventory_integrity_error(exc) from exc
        except BackendDatabaseError as exc:
            raise self._workspace_file_inventory_integrity_error(exc) from exc
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Workspace file inventory scan update failed: {exc}") from exc  # noqa: TRY003

    def complete_workspace_file_inventory_scan(
        self,
        scan_id: str,
        state: str,
        counts: Any,
        diagnostics: Any,
        *,
        root_snapshot_token: str | None = None,
    ) -> dict[str, Any]:
        """Complete a Workspace file inventory scan with durable counts and diagnostics."""
        normalized_state = str(state or "").strip().lower()
        if normalized_state not in {"current", "partial", "failed"}:
            raise InputError("completed inventory state must be current, partial, or failed.")  # noqa: TRY003
        normalized_scan_id = str(scan_id or "").strip()
        if not normalized_scan_id:
            raise InputError("scan_id is required.")  # noqa: TRY003
        now = self._ctx.utc_now_iso()
        counts_json = self._workspace_file_inventory_counts_json(counts)
        diagnostics_json = self._workspace_file_inventory_diagnostics_json(diagnostics)
        try:
            with self._ctx.transaction() as conn:
                scan = self._workspace_file_inventory_require_scan(normalized_scan_id, conn=conn)
                conn.execute(
                    """
                    UPDATE workspace_file_inventory_scans
                       SET state = ?,
                           root_snapshot_token = ?,
                           completed_at = ?,
                           counts_json = ?,
                           diagnostics_json = ?,
                           updated_at = ?,
                           version = version + 1
                     WHERE scan_id = ?
                    """,
                    (
                        normalized_state,
                        root_snapshot_token,
                        now,
                        counts_json,
                        diagnostics_json,
                        now,
                        normalized_scan_id,
                    ),
                )
                self._workspace_file_inventory_update_root_state(
                    conn,
                    workspace_id=str(scan["workspace_id"]),
                    root_id=str(scan["root_id"]),
                    state=normalized_state,
                    updated_at=now,
                )
                updated = self._workspace_file_inventory_scan_by_id(normalized_scan_id, conn=conn)
                if updated is None:
                    raise CharactersRAGDBError(  # noqa: TRY003
                        f"Failed to read workspace file inventory scan '{normalized_scan_id}' after completion."
                    )
                return updated
        except ConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            raise self._workspace_file_inventory_integrity_error(exc) from exc
        except BackendDatabaseError as exc:
            raise self._workspace_file_inventory_integrity_error(exc) from exc
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Workspace file inventory scan completion failed: {exc}") from exc  # noqa: TRY003

    def replace_workspace_file_inventory_items(
        self,
        workspace_id: str,
        root_id: str,
        scan_id: str,
        items: list[Mapping[str, Any]],
        *,
        scan_coverage_complete: bool,
    ) -> int:
        """Replace the current item projection for seen paths from one scan."""
        normalized_workspace_id = str(workspace_id or "").strip()
        normalized_root_id = str(root_id or "").strip()
        normalized_scan_id = str(scan_id or "").strip()
        if not normalized_workspace_id or not normalized_root_id or not normalized_scan_id:
            raise InputError("workspace_id, root_id, and scan_id are required.")  # noqa: TRY003
        now = self._ctx.utc_now_iso()
        false_value = self._workspace_file_inventory_bool_value(False)
        true_value = self._workspace_file_inventory_bool_value(True)
        normalized_items: list[dict[str, Any]] = []
        for item in items or []:
            if not isinstance(item, Mapping):
                raise InputError("inventory items must be mappings.")  # noqa: TRY003
            relative_path = self._workspace_file_inventory_relative_path(item.get("relative_path"))
            entry_kind = str(item.get("entry_kind") or "file").strip() or "file"
            normalized_items.append(
                {
                    "relative_path": relative_path,
                    "entry_kind": entry_kind,
                    "size_bytes": self._workspace_file_inventory_int(item.get("size_bytes"), field_name="size_bytes"),
                    "mtime_ns": self._workspace_file_inventory_int(item.get("mtime_ns"), field_name="mtime_ns"),
                    "mode_bits": self._workspace_file_inventory_int(item.get("mode_bits"), field_name="mode_bits"),
                    "extension": item.get("extension"),
                    "mime_hint": item.get("mime_hint"),
                    "language_hint": item.get("language_hint"),
                    "ignored": self._workspace_file_inventory_bool_value(bool(item.get("ignored", False))),
                    "ignore_reason": item.get("ignore_reason"),
                    "indexing_candidate": self._workspace_file_inventory_bool_value(
                        bool(item.get("indexing_candidate", False))
                    ),
                    "metadata_json": self._workspace_file_inventory_json(item.get("metadata"), default={}),
                }
            )
        try:
            with self._ctx.transaction() as conn:
                scan = self._workspace_file_inventory_require_scan(normalized_scan_id, conn=conn)
                if scan["workspace_id"] != normalized_workspace_id or scan["root_id"] != normalized_root_id:
                    raise ConflictError(
                        "Workspace file inventory scan does not match workspace/root.",
                        entity="workspace_file_inventory_scans",
                        entity_id=normalized_scan_id,
                    )
                for item in normalized_items:
                    conn.execute(
                        """
                        INSERT INTO workspace_file_inventory_items (
                            workspace_id, root_id, relative_path, scan_id, entry_kind, size_bytes,
                            mtime_ns, mode_bits, extension, mime_hint, language_hint, ignored,
                            ignore_reason, indexing_candidate, coverage_state, last_seen_at, deleted,
                            metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'current', ?, ?, ?)
                        ON CONFLICT(workspace_id, root_id, relative_path) DO UPDATE SET
                            scan_id = excluded.scan_id,
                            entry_kind = excluded.entry_kind,
                            size_bytes = excluded.size_bytes,
                            mtime_ns = excluded.mtime_ns,
                            mode_bits = excluded.mode_bits,
                            extension = excluded.extension,
                            mime_hint = excluded.mime_hint,
                            language_hint = excluded.language_hint,
                            ignored = excluded.ignored,
                            ignore_reason = excluded.ignore_reason,
                            indexing_candidate = excluded.indexing_candidate,
                            coverage_state = 'current',
                            last_seen_at = excluded.last_seen_at,
                            deleted = excluded.deleted,
                            metadata_json = excluded.metadata_json
                        """,
                        (
                            normalized_workspace_id,
                            normalized_root_id,
                            item["relative_path"],
                            normalized_scan_id,
                            item["entry_kind"],
                            item["size_bytes"],
                            item["mtime_ns"],
                            item["mode_bits"],
                            item["extension"],
                            item["mime_hint"],
                            item["language_hint"],
                            item["ignored"],
                            item["ignore_reason"],
                            item["indexing_candidate"],
                            now,
                            false_value,
                            item["metadata_json"],
                        ),
                    )
                if scan_coverage_complete:
                    conn.execute(
                        """
                        UPDATE workspace_file_inventory_items
                           SET coverage_state = 'previous',
                               deleted = ?,
                               metadata_json = metadata_json
                         WHERE workspace_id = ?
                           AND root_id = ?
                           AND scan_id <> ?
                        """,
                        (true_value, normalized_workspace_id, normalized_root_id, normalized_scan_id),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE workspace_file_inventory_items
                           SET coverage_state = 'previous'
                         WHERE workspace_id = ?
                           AND root_id = ?
                           AND scan_id <> ?
                           AND deleted = ?
                        """,
                        (normalized_workspace_id, normalized_root_id, normalized_scan_id, false_value),
                    )
                return len(normalized_items)
        except ConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            raise self._workspace_file_inventory_integrity_error(exc) from exc
        except BackendDatabaseError as exc:
            raise self._workspace_file_inventory_integrity_error(exc) from exc
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Workspace file inventory item replacement failed: {exc}") from exc  # noqa: TRY003

    def get_workspace_file_inventory_status(
        self,
        workspace_id: str,
        *,
        policy_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        """Return the latest Workspace file inventory status projection."""
        normalized_workspace_id = str(workspace_id or "").strip()
        if not normalized_workspace_id:
            raise InputError("workspace_id is required.")  # noqa: TRY003
        root = self.get_workspace_primary_root(normalized_workspace_id)
        zero_counts = normalize_inventory_counts({})
        if root is None:
            return {
                "workspace_id": normalized_workspace_id,
                "root_id": None,
                "scan_id": None,
                "state": "not_started",
                "durable_state": None,
                "stale": False,
                "counts": zero_counts,
                "diagnostics": [],
            }
        cursor = self._ctx.execute_query(
            """
            SELECT *
              FROM workspace_file_inventory_scans
             WHERE workspace_id = ?
               AND root_id = ?
             ORDER BY created_at DESC, scan_id DESC
             LIMIT 1
            """,
            (normalized_workspace_id, root["root_id"]),
        )
        row = cursor.fetchone()
        if row is None:
            return {
                "workspace_id": normalized_workspace_id,
                "root_id": root["root_id"],
                "scan_id": None,
                "state": "not_started",
                "durable_state": None,
                "stale": False,
                "counts": zero_counts,
                "diagnostics": [],
            }

        scan = dict(row)
        latest_diagnostics = bounded_inventory_diagnostics(
            self._workspace_file_inventory_json_load(scan.get("diagnostics_json"), fallback=[])
        )
        if normalize_durable_inventory_state(scan.get("state")) == "failed" and any(
            diagnostic.get("code") == "root_version_mismatch"
            for diagnostic in latest_diagnostics
        ):
            completed_row = self._ctx.execute_query(
                """
                SELECT *
                  FROM workspace_file_inventory_scans
                 WHERE workspace_id = ?
                   AND root_id = ?
                   AND state IN ('current', 'partial')
                 ORDER BY completed_at DESC, created_at DESC, scan_id DESC
                 LIMIT 1
                """,
                (normalized_workspace_id, root["root_id"]),
            ).fetchone()
            if completed_row is not None:
                scan = dict(completed_row)

        durable_state = normalize_durable_inventory_state(scan.get("state"))
        counts = normalize_inventory_counts(
            self._workspace_file_inventory_json_load(scan.get("counts_json"), fallback={})
        )
        diagnostics = bounded_inventory_diagnostics(
            self._workspace_file_inventory_json_load(scan.get("diagnostics_json"), fallback=[])
        )
        policy_mismatch = (
            policy_fingerprint is not None
            and str(scan.get("ignore_policy_fingerprint") or "") != str(policy_fingerprint)
        )
        stale = durable_state in {"current", "partial"} and (
            int(scan["root_version"]) != int(root["version"]) or policy_mismatch
        )
        return {
            "workspace_id": normalized_workspace_id,
            "root_id": root["root_id"],
            "scan_id": scan["scan_id"],
            "state": "stale" if stale else durable_state,
            "durable_state": durable_state,
            "stale": stale,
            "counts": counts,
            "diagnostics": diagnostics,
            "root_version": root["version"],
            "scan_root_version": scan["root_version"],
            "ignore_policy_fingerprint": scan.get("ignore_policy_fingerprint"),
            "root_snapshot_token": scan.get("root_snapshot_token"),
            "job_id": scan.get("job_id"),
            "job_uuid": scan.get("job_uuid"),
            "started_at": scan.get("started_at"),
            "completed_at": scan.get("completed_at"),
            "updated_at": scan.get("updated_at"),
        }

    def list_workspace_file_inventory_items(
        self,
        workspace_id: str,
        *,
        prefix: str | None = None,
        cursor: str | None = None,
        limit: int = 100,
        include_ignored: bool = False,
        entry_kind: str | None = None,
    ) -> dict[str, Any]:
        """List non-deleted inventory items sorted by project-root-relative path."""
        normalized_workspace_id = str(workspace_id or "").strip()
        if not normalized_workspace_id:
            raise InputError("workspace_id is required.")  # noqa: TRY003
        bounded_limit = max(1, min(int(limit), 500))
        root = self.get_workspace_primary_root(normalized_workspace_id)
        if root is None:
            return {"items": [], "next_cursor": None}

        normalized_prefix = self._workspace_file_inventory_prefix(prefix)
        normalized_entry_kind = str(entry_kind or "").strip().lower() or None
        try:
            cursor_path = decode_inventory_cursor(cursor) if cursor else None
        except ValueError as exc:
            raise InputError("cursor must be a valid workspace file inventory cursor.") from exc  # noqa: TRY003
        false_value = self._workspace_file_inventory_bool_value(False)
        clauses = ["workspace_id = ?", "root_id = ?", "deleted = ?"]
        params: list[Any] = [normalized_workspace_id, root["root_id"], false_value]
        if not include_ignored:
            clauses.append("ignored = ?")
            params.append(false_value)
        if normalized_prefix is not None:
            clauses.append("relative_path LIKE ?")
            params.append(f"{normalized_prefix}%")
        if normalized_entry_kind is not None:
            clauses.append("entry_kind = ?")
            params.append(normalized_entry_kind)
        if cursor_path is not None:
            clauses.append("relative_path > ?")
            params.append(cursor_path)
        params.append(bounded_limit + 1)
        query = f"""
            SELECT workspace_id, root_id, relative_path, scan_id, entry_kind, size_bytes,
                   mtime_ns, mode_bits, extension, mime_hint, language_hint, ignored,
                   ignore_reason, indexing_candidate, coverage_state, last_seen_at, deleted,
                   metadata_json
              FROM workspace_file_inventory_items
             WHERE {' AND '.join(clauses)}
             ORDER BY relative_path ASC
             LIMIT ?
        """  # nosec B608
        rows = [dict(row) for row in self._ctx.execute_query(query, tuple(params)).fetchall()]
        has_more = len(rows) > bounded_limit
        visible_rows = rows[:bounded_limit]
        next_cursor = encode_inventory_cursor(visible_rows[-1]["relative_path"]) if has_more and visible_rows else None
        for item in visible_rows:
            item["ignored"] = bool(item.get("ignored"))
            item["indexing_candidate"] = bool(item.get("indexing_candidate"))
            item["deleted"] = bool(item.get("deleted"))
            item["metadata"] = self._workspace_file_inventory_json_load(item.get("metadata_json"), fallback={})
        return {"items": visible_rows, "next_cursor": next_cursor}

    def _get_workspace_migration_session_by_idempotency_key(self, idempotency_key: str) -> dict[str, Any] | None:
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_migration_sessions WHERE idempotency_key = ?",
            (idempotency_key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return workspace_migration_session_from_row(row, chunks=self._list_workspace_migration_chunks(str(row["id"])))

    def _list_workspace_migration_chunks(self, migration_id: str) -> list[dict[str, Any]]:
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_migration_chunks WHERE migration_id = ? ORDER BY accepted_at, id",
            (migration_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_workspace_migration_session(self, migration_id: str) -> dict[str, Any] | None:
        """Return a Research Workspace migration session and accepted chunk receipts."""
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_migration_sessions WHERE id = ?",
            (migration_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return workspace_migration_session_from_row(row, chunks=self._list_workspace_migration_chunks(str(row["id"])))

    def list_workspace_migration_sessions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent Research Workspace migration sessions with accepted chunk receipts."""
        bounded_limit = max(1, min(int(limit), 500))
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_migration_sessions ORDER BY updated_at DESC, created_at DESC LIMIT ?",
            (bounded_limit,),
        )
        sessions: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            sessions.append(workspace_migration_session_from_row(row, chunks=self._list_workspace_migration_chunks(str(row["id"]))))
        return sessions

    def _get_workspace_migration_chunk(self, migration_id: str, chunk_id: str) -> dict[str, Any] | None:
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_migration_chunks WHERE migration_id = ? AND id = ?",
            (migration_id, chunk_id),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def upsert_workspace_migration_session(self, data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Create a workspace migration session or return the existing idempotent session."""
        migration_id = str(data.get("id") or "").strip()
        idempotency_key = str(data.get("idempotency_key") or "").strip()
        target_workspace_id = str(data.get("target_workspace_id") or "").strip()
        target_workspace_name = str(data.get("target_workspace_name") or "").strip()
        manifest_hash = str(data.get("manifest_hash") or "").strip().lower()
        source_product = str(data.get("source_product") or "research-workspace-webui").strip()
        if not all((migration_id, idempotency_key, target_workspace_id, target_workspace_name, manifest_hash)):
            raise InputError(  # noqa: TRY003
                "id, idempotency_key, target_workspace_id, target_workspace_name, and manifest_hash are required."
            )

        existing = self.get_workspace_migration_session(migration_id)
        existing_by_key = self._get_workspace_migration_session_by_idempotency_key(idempotency_key)
        if existing_by_key is not None and existing_by_key["id"] != migration_id:
            raise ConflictError(  # noqa: TRY003
                "Workspace migration idempotency key is already used by another migration.",
                entity="workspace_migration_sessions",
                entity_id=idempotency_key,
            )
        if existing is not None:
            if existing["idempotency_key"] != idempotency_key or existing["manifest_hash"] != manifest_hash:
                raise ConflictError(  # noqa: TRY003
                    "Workspace migration already exists with different idempotency data.",
                    entity="workspace_migration_sessions",
                    entity_id=migration_id,
                )
            return existing, False

        self.upsert_workspace(
            target_workspace_id,
            target_workspace_name,
            study_materials_policy="workspace",
            initial_assistant_opt_out=True,
        )

        now = self._ctx.utc_now_iso()
        manifest_json = serialize_workspace_migration_json(data.get("manifest"), "manifest", default={})
        diagnostics_json = serialize_workspace_migration_json(
            data.get("diagnostics"),
            "diagnostics",
            default={},
        )
        declared_chunks_json = serialize_workspace_migration_json(
            data.get("declared_chunks"),
            "declared_chunks",
            default=[],
        )
        false_value = False if self._ctx.backend_type == BackendType.POSTGRESQL else 0
        params = (
            migration_id,
            target_workspace_id,
            target_workspace_name,
            source_product,
            idempotency_key,
            manifest_hash,
            manifest_json,
            diagnostics_json,
            declared_chunks_json,
            "created",
            false_value,
            now,
            now,
            1,
        )
        with self._ctx.transaction() as conn:
            try:
                conn.execute(
                    "INSERT INTO workspace_migration_sessions ("
                    "id, target_workspace_id, target_workspace_name, source_product, idempotency_key, "
                    "manifest_hash, manifest_json, diagnostics_json, declared_chunks_json, status, "
                    "client_delete_eligible, created_at, updated_at, version"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    params,
                )
            except sqlite3.IntegrityError as exc:
                raced = self.get_workspace_migration_session(migration_id)
                if raced is not None and raced["idempotency_key"] == idempotency_key and raced["manifest_hash"] == manifest_hash:
                    return raced, False
                raise ConflictError(  # noqa: TRY003
                    "Workspace migration could not be created due to a uniqueness conflict.",
                    entity="workspace_migration_sessions",
                    entity_id=migration_id,
                ) from exc
        created = self.get_workspace_migration_session(migration_id)
        if created is None:
            raise CharactersRAGDBError(f"Failed to read workspace migration '{migration_id}' after create.")  # noqa: TRY003
        return created, True

    def _declared_workspace_migration_chunks(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        parsed = parse_workspace_migration_json(session.get("declared_chunks_json"), fallback=[])
        return parsed if isinstance(parsed, list) else []

    def _workspace_migration_missing_chunk_ids(self, session: dict[str, Any]) -> list[str]:
        accepted = {str(chunk.get("id")) for chunk in session.get("_chunks", [])}
        return [
            str(chunk.get("id"))
            for chunk in self._declared_workspace_migration_chunks(session)
            if str(chunk.get("id")) not in accepted
        ]

    def _verify_workspace_migration_delete_eligibility(self, session: dict[str, Any]) -> dict[str, Any]:
        """Check receipt consistency without treating hashes as migrated content."""
        declared = self._declared_workspace_migration_chunks(session)
        accepted_chunks = session.get("_chunks", [])
        accepted = {str(chunk.get("id")): chunk for chunk in accepted_chunks}
        if not declared:
            return {
                "eligible": False,
                "status": "no_declared_chunks",
                "missing_chunk_ids": [],
                "mismatch_chunk_ids": [],
                "undeclared_chunk_ids": sorted(accepted.keys()),
            }

        declared_by_id = {str(chunk.get("id")): chunk for chunk in declared}
        missing = [
            chunk_id
            for chunk_id in declared_by_id
            if chunk_id not in accepted
        ]
        undeclared = [
            chunk_id
            for chunk_id in accepted
            if chunk_id not in declared_by_id
        ]
        mismatched: list[str] = []
        for chunk_id, expected in declared_by_id.items():
            receipt = accepted.get(chunk_id)
            if receipt is None:
                continue
            try:
                expected_byte_count_value = expected.get("byte_count")
                receipt_byte_count_value = receipt.get("byte_count")
                if expected_byte_count_value is None or receipt_byte_count_value is None:
                    raise TypeError
                expected_byte_count = int(expected_byte_count_value)
                receipt_byte_count = int(receipt_byte_count_value)
            except (TypeError, ValueError):
                mismatched.append(chunk_id)
                continue
            expected_kind = str(expected.get("chunk_kind") or "workspace_bundle")
            receipt_kind = str(receipt.get("chunk_kind") or "workspace_bundle")
            expected_hash = str(expected.get("sha256") or "").lower()
            receipt_hash = str(receipt.get("sha256") or "").lower()
            if (
                expected_hash != receipt_hash
                or expected_byte_count != receipt_byte_count
                or expected_kind != receipt_kind
            ):
                mismatched.append(chunk_id)

        receipts_match = not missing and not mismatched and not undeclared
        if receipts_match:
            status = "content_not_migrated"
        elif missing:
            status = "missing_chunks"
        else:
            status = "verification_failed"
        return {
            "eligible": False,
            "status": status,
            "missing_chunk_ids": sorted(missing),
            "mismatch_chunk_ids": sorted(mismatched),
            "undeclared_chunk_ids": sorted(undeclared),
        }

    def add_workspace_migration_chunk(
        self,
        migration_id: str,
        chunk_id: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Record an idempotent chunk receipt for a workspace migration."""
        normalized_migration_id = str(migration_id or "").strip()
        normalized_chunk_id = str(chunk_id or "").strip()
        sha256 = str(data.get("sha256") or "").strip().lower()
        chunk_kind = str(data.get("chunk_kind") or "workspace_bundle").strip()
        try:
            byte_count_value = data.get("byte_count")
            if byte_count_value is None:
                raise TypeError
            byte_count = int(byte_count_value)
        except (TypeError, ValueError) as exc:
            raise InputError("byte_count must be an integer.") from exc  # noqa: TRY003
        if not all((normalized_migration_id, normalized_chunk_id, sha256, chunk_kind)):
            raise InputError("migration_id, chunk_id, sha256, and chunk_kind are required.")  # noqa: TRY003

        session = self.get_workspace_migration_session(normalized_migration_id)
        if session is None:
            raise ConflictError(  # noqa: TRY003
                "Workspace migration not found.",
                entity="workspace_migration_sessions",
                entity_id=normalized_migration_id,
            )
        if session.get("status") == "finalized":
            raise ConflictError(  # noqa: TRY003
                "Workspace migration is already finalized.",
                entity="workspace_migration_sessions",
                entity_id=normalized_migration_id,
            )

        declared = self._declared_workspace_migration_chunks(session)
        declared_by_id = {str(chunk.get("id")): chunk for chunk in declared}
        if declared_by_id:
            expected = declared_by_id.get(normalized_chunk_id)
            if expected is None:
                raise ConflictError(  # noqa: TRY003
                    "Workspace migration chunk was not declared in the manifest.",
                    entity="workspace_migration_chunks",
                    entity_id=normalized_chunk_id,
                )
            try:
                expected_byte_count_value = expected.get("byte_count")
                if expected_byte_count_value is None:
                    raise TypeError
                expected_byte_count = int(expected_byte_count_value)
            except (TypeError, ValueError) as exc:
                raise ConflictError(  # noqa: TRY003
                    "Workspace migration chunk declaration has an invalid byte count.",
                    entity="workspace_migration_chunks",
                    entity_id=normalized_chunk_id,
                ) from exc
            if (
                str(expected.get("sha256") or "").lower() != sha256
                or expected_byte_count != byte_count
                or str(expected.get("chunk_kind") or "workspace_bundle") != chunk_kind
            ):
                raise ConflictError(  # noqa: TRY003
                    "Workspace migration chunk receipt does not match the declared manifest.",
                    entity="workspace_migration_chunks",
                    entity_id=normalized_chunk_id,
                )

        existing = self._get_workspace_migration_chunk(normalized_migration_id, normalized_chunk_id)
        if existing is not None:
            if existing["sha256"] == sha256 and int(existing["byte_count"]) == byte_count:
                return existing
            raise ConflictError(  # noqa: TRY003
                "Workspace migration chunk already exists with different content.",
                entity="workspace_migration_chunks",
                entity_id=normalized_chunk_id,
            )

        metadata_json = serialize_workspace_migration_json(data.get("metadata"), "metadata", default={})
        now = self._ctx.utc_now_iso()
        with self._ctx.transaction() as conn:
            try:
                conn.execute(
                    "INSERT INTO workspace_migration_chunks ("
                    "migration_id, id, sha256, byte_count, chunk_kind, metadata_json, accepted_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        normalized_migration_id,
                        normalized_chunk_id,
                        sha256,
                        byte_count,
                        chunk_kind,
                        metadata_json,
                        now,
                    ),
                )
                conn.execute(
                    "UPDATE workspace_migration_sessions SET updated_at = ?, version = version + 1 WHERE id = ?",
                    (now, normalized_migration_id),
                )
            except sqlite3.IntegrityError as exc:
                raced = self._get_workspace_migration_chunk(normalized_migration_id, normalized_chunk_id)
                if raced is not None and raced["sha256"] == sha256 and int(raced["byte_count"]) == byte_count:
                    return raced
                raise ConflictError(  # noqa: TRY003
                    "Workspace migration chunk could not be accepted due to a uniqueness conflict.",
                    entity="workspace_migration_chunks",
                    entity_id=normalized_chunk_id,
                ) from exc
        accepted = self._get_workspace_migration_chunk(normalized_migration_id, normalized_chunk_id)
        if accepted is None:
            raise CharactersRAGDBError(  # noqa: TRY003
                f"Failed to read workspace migration chunk '{normalized_chunk_id}' after insert."
            )
        return accepted

    def finalize_workspace_migration(
        self,
        migration_id: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        """Finalize a workspace migration after all declared chunks have receipts."""
        normalized_migration_id = str(migration_id or "").strip()
        manifest_hash = str(data.get("manifest_hash") or "").strip().lower()
        session = self.get_workspace_migration_session(normalized_migration_id)
        if session is None:
            raise ConflictError(  # noqa: TRY003
                "Workspace migration not found.",
                entity="workspace_migration_sessions",
                entity_id=normalized_migration_id,
            )
        if session["manifest_hash"] != manifest_hash:
            raise ConflictError(  # noqa: TRY003
                "Workspace migration manifest hash mismatch.",
                entity="workspace_migration_sessions",
                entity_id=normalized_migration_id,
            )
        if session.get("status") == "finalized":
            return session

        missing = self._workspace_migration_missing_chunk_ids(session)
        if missing:
            raise ConflictError(  # noqa: TRY003
                f"Workspace migration is missing declared chunks: {', '.join(missing)}",
                entity="workspace_migration_sessions",
                entity_id=normalized_migration_id,
            )

        now = self._ctx.utc_now_iso()
        declared_count = len(self._declared_workspace_migration_chunks(session))
        accepted_count = len(session.get("_chunks", []))
        verification = self._verify_workspace_migration_delete_eligibility(session)
        client_delete_eligible = bool(verification["eligible"])
        recovery_manifest = {
            "migration_id": normalized_migration_id,
            "target_workspace_id": session["target_workspace_id"],
            "source_product": session["source_product"],
            "manifest_hash": session["manifest_hash"],
            "status": "finalized",
            "declared_chunk_count": declared_count,
            "accepted_chunk_count": accepted_count,
            "missing_chunk_ids": verification["missing_chunk_ids"],
            "client_delete_eligible": client_delete_eligible,
            "can_delete_legacy_storage": client_delete_eligible,
            "server_readback_verified": client_delete_eligible,
            "verification_status": verification["status"],
            "mismatch_chunk_ids": verification["mismatch_chunk_ids"],
            "undeclared_chunk_ids": verification["undeclared_chunk_ids"],
            "next_step": "enqueue_workspace_source_ingestion_jobs",
            "finalized_at": now,
        }
        recovery_json = serialize_workspace_migration_json(
            recovery_manifest,
            "recovery_manifest",
            default={},
        )
        eligible_value = client_delete_eligible if self._ctx.backend_type == BackendType.POSTGRESQL else int(client_delete_eligible)
        with self._ctx.transaction() as conn:
            conn.execute(
                "UPDATE workspace_migration_sessions "
                "SET status = ?, finalized_at = ?, recovery_manifest_json = ?, "
                "client_delete_eligible = ?, updated_at = ?, version = version + 1 "
                "WHERE id = ?",
                ("finalized", now, recovery_json, eligible_value, now, normalized_migration_id),
            )
        finalized = self.get_workspace_migration_session(normalized_migration_id)
        if finalized is None:
            raise CharactersRAGDBError(  # noqa: TRY003
                f"Failed to read workspace migration '{normalized_migration_id}' after finalize."
            )
        return finalized

    def record_workspace_migration_client_delete_ack(
        self,
        migration_id: str,
        data: dict[str, Any],
    ) -> bool:
        """Record client deletion acknowledgement only after deletion eligibility is explicitly enabled."""
        normalized_migration_id = str(migration_id or "").strip()
        manifest_hash = str(data.get("acknowledged_manifest_hash") or "").strip().lower()
        session = self.get_workspace_migration_session(normalized_migration_id)
        if session is None:
            raise ConflictError(  # noqa: TRY003
                "Workspace migration not found.",
                entity="workspace_migration_sessions",
                entity_id=normalized_migration_id,
            )
        if session["manifest_hash"] != manifest_hash:
            raise ConflictError(  # noqa: TRY003
                "Workspace migration manifest hash mismatch.",
                entity="workspace_migration_sessions",
                entity_id=normalized_migration_id,
            )
        if session.get("status") != "finalized":
            raise ConflictError(  # noqa: TRY003
                "Workspace migration must be finalized before client deletion can be acknowledged.",
                entity="workspace_migration_sessions",
                entity_id=normalized_migration_id,
            )
        if not bool(session.get("client_delete_eligible")):
            raise ConflictError(  # noqa: TRY003
                "Client deletion is not enabled for Research Workspace migrations.",
                entity="workspace_migration_sessions",
                entity_id=normalized_migration_id,
            )

        now = self._ctx.utc_now_iso()
        with self._ctx.transaction() as conn:
            conn.execute(
                "UPDATE workspace_migration_sessions "
                "SET client_delete_ack_at = ?, updated_at = ?, version = version + 1 "
                "WHERE id = ?",
                (now, now, normalized_migration_id),
            )
        return True

    def _normalize_workspace_source_review_state(self, value: Any, *, for_create: bool) -> str:
        if value is None:
            state = ""
        elif isinstance(value, str):
            state = value.strip().lower()
        else:
            raise InputError("Workspace source review state must be a string.")  # noqa: TRY003
        if for_create and not state:
            state = "unset"
        allowed_states = (
            _ALLOWED_WORKSPACE_SOURCE_CREATE_REVIEW_STATES
            if for_create
            else _ALLOWED_WORKSPACE_SOURCE_REVIEW_STATES
        )
        if state not in allowed_states:
            raise InputError(f"Unsupported workspace source review state '{state}'.")  # noqa: TRY003
        return state

    def _build_workspace_source_review_transition(
        self,
        review_state: Any,
        *,
        actor_user_id: str | None = None,
        for_create: bool = False,
    ) -> dict[str, Any]:
        state = self._normalize_workspace_source_review_state(review_state, for_create=for_create)
        actor = None
        if state == "reviewed":
            if not isinstance(actor_user_id, str) or not actor_user_id.strip():
                raise InputError("A non-empty actor_user_id is required to review a workspace source.")  # noqa: TRY003
            actor = actor_user_id.strip()
        now = self._ctx.utc_now_iso()
        return {
            "review_state": state,
            "review_state_updated_at": now,
            "reviewed_at": now if state == "reviewed" else None,
            "reviewed_by_user_id": actor if state == "reviewed" else None,
        }

    def add_workspace_source(self, workspace_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Add a source to a workspace."""
        source_id = data.get("id")
        if not source_id:
            raise InputError("Source id is required.")  # noqa: TRY003
        now = self._ctx.utc_now_iso()
        review_transition = self._build_workspace_source_review_transition(
            data.get("review_state"),
            for_create=True,
        )
        query = (
            "INSERT INTO workspace_sources "
            "(id, workspace_id, media_id, title, source_type, url, position, selected, added_at, "
            "review_state, review_state_updated_at, reviewed_at, reviewed_by_user_id, version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1) "
            "ON CONFLICT(workspace_id, id) DO NOTHING"
        )
        params = (
            source_id,
            workspace_id,
            data.get("media_id", 0),
            data.get("title", ""),
            data.get("source_type", ""),
            data.get("url"),
            data.get("position", 0),
            bool(data.get("selected", True)),
            now,
            review_transition["review_state"],
            review_transition["review_state_updated_at"],
            review_transition["reviewed_at"],
            review_transition["reviewed_by_user_id"],
        )
        with self._ctx.transaction() as conn:
            try:
                conn.execute(query, params)
            except sqlite3.IntegrityError as exc:
                raise ConflictError(  # noqa: TRY003
                    f"Workspace source '{source_id}' could not be added.",
                    entity="workspace_sources",
                    entity_id=source_id,
                ) from exc
        return self._get_workspace_source(workspace_id, source_id)  # type: ignore[return-value]

    def _get_workspace_source(
        self,
        workspace_id: str,
        source_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_sources WHERE workspace_id = ? AND id = ?",
            (workspace_id, source_id),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_workspace_source(
        self,
        workspace_id: str,
        source_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a workspace source by id."""
        return self._get_workspace_source(workspace_id, source_id, include_deleted=include_deleted)

    def list_workspace_sources(self, workspace_id: str) -> list[dict[str, Any]]:
        """List all sources for a workspace ordered by position."""
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_sources WHERE workspace_id = ? ORDER BY position, added_at",
            (workspace_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def _redact_workspace_activity_metadata(cls, value: Any, *, key: str | None = None) -> Any:
        if isinstance(value, Mapping):
            redacted: dict[str, Any] = {}
            for raw_key, raw_value in value.items():
                item_key = str(raw_key)
                if _WORKSPACE_ACTIVITY_SECRET_KEY_RE.search(item_key):
                    continue
                redacted[item_key] = cls._redact_workspace_activity_metadata(raw_value, key=item_key)
            return redacted
        if isinstance(value, list):
            return [cls._redact_workspace_activity_metadata(item, key=key) for item in value]
        if isinstance(value, tuple):
            return [cls._redact_workspace_activity_metadata(item, key=key) for item in value]
        if isinstance(value, str):
            if _WORKSPACE_ACTIVITY_SECRET_VALUE_RE.search(value):
                return "[redacted]"
            if key and _WORKSPACE_ACTIVITY_PATH_KEY_RE.search(key):
                from tldw_Server_API.app.core.Workspaces.runtime_bindings import redacted_path_hint

                return redacted_path_hint(value)
        return value

    @classmethod
    def _workspace_activity_metadata_json(cls, value: Any) -> tuple[dict[str, Any], str]:
        if value is None:
            normalized: dict[str, Any] = {}
        elif isinstance(value, str):
            if not value.strip():
                normalized = {}
            else:
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise InputError("metadata must be valid JSON.") from exc  # noqa: TRY003
                if not isinstance(parsed, Mapping):
                    raise InputError("metadata must be a JSON object.")  # noqa: TRY003
                normalized = dict(parsed)
        elif isinstance(value, Mapping):
            normalized = dict(value)
        else:
            raise InputError("metadata must be a JSON object.")  # noqa: TRY003

        redacted = cls._redact_workspace_activity_metadata(normalized)
        try:
            dumped = json.dumps(
                redacted,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise InputError("metadata must be JSON serializable.") from exc  # noqa: TRY003
        if len(dumped.encode("utf-8")) > _WORKSPACE_ACTIVITY_MAX_METADATA_BYTES:
            raise InputError(
                f"metadata exceeds {_WORKSPACE_ACTIVITY_MAX_METADATA_BYTES} bytes."
            ) from None  # noqa: TRY003
        return redacted, dumped

    @staticmethod
    def _normalize_workspace_activity_required_string(value: Any, field_name: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise InputError(f"{field_name} is required.")  # noqa: TRY003
        return normalized

    @staticmethod
    def _normalize_workspace_activity_optional_string(value: Any, *, max_length: int) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            return None
        return normalized[:max_length]

    @staticmethod
    def _normalize_workspace_activity_limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise InputError("limit must be an integer.")  # noqa: TRY003
        if limit < 1 or limit > 100:
            raise InputError("limit must be between 1 and 100.")  # noqa: TRY003
        return limit

    def _normalize_workspace_activity_row(self, row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        raw_metadata = item.pop("metadata_json", None)
        if isinstance(raw_metadata, Mapping):
            metadata = dict(raw_metadata)
        elif isinstance(raw_metadata, str) and raw_metadata.strip():
            try:
                parsed = json.loads(raw_metadata)
                metadata = dict(parsed) if isinstance(parsed, Mapping) else {}
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode workspace activity metadata_json for workspace={} event={} ({} chars); raw value omitted.",
                    item.get("workspace_id"),
                    item.get("event_id"),
                    len(raw_metadata),
                )
                metadata = {}
        else:
            metadata = {}
        item["metadata"] = metadata
        item["version"] = int(item.get("version") or 1)
        return item

    def record_workspace_activity_event(
        self,
        workspace_id: str,
        data: Mapping[str, Any],
        *,
        user_id: str | None = None,
        return_row: bool = True,
    ) -> dict[str, Any] | None:
        """Append a bounded, secret-safe activity event for a workspace."""
        normalized_workspace_id = self._normalize_workspace_activity_required_string(workspace_id, "workspace_id")
        event_type = self._normalize_workspace_activity_required_string(data.get("event_type"), "event_type")
        category = self._normalize_workspace_activity_required_string(data.get("category"), "category")
        event_id = self._normalize_workspace_activity_optional_string(data.get("event_id"), max_length=128)
        if event_id is None:
            event_id = f"wae_{time.time_ns():020d}_{uuid.uuid4().hex}"
        actor = self._normalize_workspace_membership_actor(user_id)
        resource_type = self._normalize_workspace_activity_optional_string(data.get("resource_type"), max_length=80)
        resource_id = self._normalize_workspace_activity_optional_string(data.get("resource_id"), max_length=512)
        summary = self._normalize_workspace_activity_optional_string(data.get("summary"), max_length=512)
        metadata, metadata_json = self._workspace_activity_metadata_json(data.get("metadata"))
        _ = metadata
        now = self._ctx.utc_now_iso()

        try:
            with self._ctx.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO workspace_activity_events (
                        workspace_id,
                        event_id,
                        event_type,
                        category,
                        actor_user_id,
                        resource_type,
                        resource_id,
                        summary,
                        metadata_json,
                        created_at,
                        client_id,
                        version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        normalized_workspace_id,
                        event_id,
                        event_type,
                        category,
                        actor,
                        resource_type,
                        resource_id,
                        summary,
                        metadata_json,
                        now,
                        self._ctx.client_id or "unknown",
                    ),
                )
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Workspace activity event write failed: {exc}") from exc  # noqa: TRY003

        if not return_row:
            return None

        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_activity_events WHERE workspace_id = ? AND event_id = ?",
            (normalized_workspace_id, event_id),
        )
        row = self._normalize_workspace_activity_row(cursor.fetchone())
        if row is not None:
            return row
        raise CharactersRAGDBError("Created workspace activity event could not be reloaded.")  # noqa: TRY003

    def list_workspace_activity_events(
        self,
        workspace_id: str,
        *,
        limit: int = 50,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent workspace activity events in deterministic newest-first order."""
        normalized_workspace_id = self._normalize_workspace_activity_required_string(workspace_id, "workspace_id")
        normalized_limit = self._normalize_workspace_activity_limit(limit)
        if category is None:
            cursor = self._ctx.execute_query(
                """
                SELECT *
                  FROM workspace_activity_events
                 WHERE workspace_id = ?
                 ORDER BY created_at DESC, event_id DESC
                 LIMIT ?
                """,
                (normalized_workspace_id, normalized_limit),
            )
        else:
            normalized_category = self._normalize_workspace_activity_required_string(category, "category")
            cursor = self._ctx.execute_query(
                """
                SELECT *
                  FROM workspace_activity_events
                 WHERE workspace_id = ?
                   AND category = ?
                 ORDER BY created_at DESC, event_id DESC
                 LIMIT ?
                """,
                (normalized_workspace_id, normalized_category, normalized_limit),
            )
        return [
            item
            for row in cursor.fetchall()
            if (item := self._normalize_workspace_activity_row(row)) is not None
        ]

    def update_workspace_source(
        self,
        workspace_id: str,
        source_id: str,
        updates: dict[str, Any],
        *,
        expected_version: int,
        actor_user_id: str | None = None,
    ) -> dict[str, Any]:
        """Update a workspace source with optimistic locking."""
        existing = self._get_workspace_source(workspace_id, source_id)
        if existing is None:
            raise ConflictError(  # noqa: TRY003
                f"Workspace source '{source_id}' not found.",
                entity="workspace_sources",
                entity_id=source_id,
            )
        if existing["version"] != expected_version:
            raise ConflictError(  # noqa: TRY003
                f"Source '{source_id}' version mismatch (db={existing['version']}, expected={expected_version}).",
                entity="workspace_sources",
                entity_id=source_id,
            )
        set_clauses = ["version = ?"]
        params: list[Any] = [expected_version + 1]
        for col in ("title", "source_type", "url", "position", "selected"):
            if col in updates:
                set_clauses.append(f"{col} = ?")
                params.append(updates[col])
        if "review_state" in updates:
            review_transition = self._build_workspace_source_review_transition(
                updates["review_state"],
                actor_user_id=actor_user_id,
            )
            for col in (
                "review_state",
                "review_state_updated_at",
                "reviewed_at",
                "reviewed_by_user_id",
            ):
                set_clauses.append(f"{col} = ?")
                params.append(review_transition[col])
        query = f"UPDATE workspace_sources SET {', '.join(set_clauses)} WHERE workspace_id = ? AND id = ? AND version = ?"  # nosec B608
        params.extend([workspace_id, source_id, expected_version])
        with self._ctx.transaction() as conn:
            cursor = conn.execute(query, tuple(params))
            if cursor.rowcount == 0:
                raise ConflictError(  # noqa: TRY003
                    f"Source '{source_id}' concurrent update detected.",
                    entity="workspace_sources",
                    entity_id=source_id,
                )
        return self._get_workspace_source(workspace_id, source_id)  # type: ignore[return-value]

    def update_workspace_source_review_states(
        self,
        workspace_id: str,
        source_ids: list[str],
        review_state: Any,
        actor_user_id: str | None,
    ) -> list[dict[str, Any]]:
        """Atomically transition workspace sources to one review state."""
        normalized_ids: list[str] = []
        seen_ids: set[str] = set()
        for source_id in source_ids:
            if not isinstance(source_id, str) or not source_id.strip():
                raise InputError("source_ids entries must be non-empty strings.")  # noqa: TRY003
            if source_id not in seen_ids:
                seen_ids.add(source_id)
                normalized_ids.append(source_id)
        review_transition = self._build_workspace_source_review_transition(
            review_state,
            actor_user_id=actor_user_id,
        )
        if not normalized_ids:
            return []

        placeholders = ", ".join("?" for _ in normalized_ids)
        with self._ctx.transaction() as conn:
            rows = conn.execute(
                f"SELECT id FROM workspace_sources WHERE workspace_id = ? AND id IN ({placeholders})",  # nosec B608  # Placeholders are generated "?" tokens; IDs remain bound parameters.
                (workspace_id, *normalized_ids),
            ).fetchall()
            found_ids = {str(row["id"]) for row in rows}
            missing_ids = [source_id for source_id in normalized_ids if source_id not in found_ids]
            if missing_ids:
                raise ConflictError(  # noqa: TRY003
                    f"Workspace sources not found: {', '.join(repr(source_id) for source_id in missing_ids)}.",
                    entity="workspace_sources",
                    entity_id=missing_ids[0],
                )

            cursor = conn.execute(
                "UPDATE workspace_sources "
                "SET review_state = ?, review_state_updated_at = ?, reviewed_at = ?, "
                f"reviewed_by_user_id = ?, version = version + 1 WHERE workspace_id = ? AND id IN ({placeholders})",  # nosec B608  # Placeholders are generated "?" tokens; IDs remain bound parameters.
                (
                    review_transition["review_state"],
                    review_transition["review_state_updated_at"],
                    review_transition["reviewed_at"],
                    review_transition["reviewed_by_user_id"],
                    workspace_id,
                    *normalized_ids,
                ),
            )
            if cursor.rowcount != len(normalized_ids):
                raise ConflictError(  # noqa: TRY003
                    "Workspace source review-state update was incomplete.",
                    entity="workspace_sources",
                    entity_id=workspace_id,
                )

            updated_rows = conn.execute(
                f"SELECT * FROM workspace_sources WHERE workspace_id = ? AND id IN ({placeholders})",  # nosec B608  # Placeholders are generated "?" tokens; IDs remain bound parameters.
                (workspace_id, *normalized_ids),
            ).fetchall()

        updated_by_id = {str(row["id"]): dict(row) for row in updated_rows}
        return [updated_by_id[source_id] for source_id in normalized_ids]

    def delete_workspace_source(self, workspace_id: str, source_id: str) -> None:
        """Hard-delete a workspace source."""
        with self._ctx.transaction() as conn:
            conn.execute(
                "DELETE FROM workspace_sources WHERE workspace_id = ? AND id = ?",
                (workspace_id, source_id),
            )

    def update_workspace_source_selection(self, workspace_id: str, *, selected_ids: list[str]) -> None:
        """Batch-update selection: mark listed ids as selected, all others as unselected.

        Bumps the version of every affected row so clients holding stale versions
        will see a conflict on their next individual update.
        """
        with self._ctx.transaction() as conn:
            conn.execute(
                "UPDATE workspace_sources SET selected = ?, version = version + 1 WHERE workspace_id = ?",
                (False, workspace_id),
            )
            if selected_ids:
                placeholders = ", ".join("?" for _ in selected_ids)
                conn.execute(
                    f"UPDATE workspace_sources SET selected = ?, version = version + 1 WHERE workspace_id = ? AND id IN ({placeholders})",  # nosec B608
                    (True, workspace_id, *selected_ids),
                )

    def reorder_workspace_sources(self, workspace_id: str, ordered_ids: list[str]) -> None:
        """Set position of sources based on list order.

        Bumps the version of every reordered row for conflict detection.
        """
        with self._ctx.transaction() as conn:
            for idx, source_id in enumerate(ordered_ids):
                conn.execute(
                    "UPDATE workspace_sources SET position = ?, version = version + 1 WHERE workspace_id = ? AND id = ?",
                    (idx, workspace_id, source_id),
                )

    @staticmethod
    def _normalize_runtime_binding_limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise InputError("limit must be an integer.")  # noqa: TRY003
        if limit < 1 or limit > 1000:
            raise InputError("limit must be between 1 and 1000.")  # noqa: TRY003
        return limit

    @staticmethod
    def _normalize_workspace_runtime_binding_actor(user_id: str | None) -> str | None:
        if user_id is None:
            return None
        normalized = str(user_id).strip()
        return normalized or None

    @staticmethod
    def _workspace_runtime_binding_is_constraint_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "unique",
                "duplicate",
                "constraint",
                "foreign key",
            )
        )

    def _workspace_runtime_binding_write_error(self, exc: Exception) -> CharactersRAGDBError:
        if self._workspace_runtime_binding_is_constraint_error(exc):
            return ConflictError(
                f"Workspace runtime binding write conflicted: {exc}",
                entity="workspace_runtime_bindings",
            )
        return CharactersRAGDBError(f"Workspace runtime binding write failed: {exc}")  # noqa: TRY003

    @staticmethod
    def _normalize_workspace_runtime_binding_required_string(value: Any, field_name: str) -> str:
        normalized = str(value if value is not None else "").strip()
        if not normalized:
            raise InputError(f"{field_name} is required.")  # noqa: TRY003
        return normalized

    @staticmethod
    def _workspace_runtime_binding_filter_value(value: Any, field_name: str) -> str:
        normalized = str(value if value is not None else "").strip().lower()
        if not normalized:
            raise InputError(f"{field_name} is required.")  # noqa: TRY003
        return normalized

    def _normalize_workspace_runtime_binding_input(self, data: Mapping[str, Any]) -> dict[str, Any]:
        from tldw_Server_API.app.core.Workspaces.runtime_bindings import (
            normalize_runtime_binding_payload,
        )

        return normalize_runtime_binding_payload(data)

    def _normalize_workspace_runtime_binding_row(self, row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        from tldw_Server_API.app.core.Workspaces.runtime_bindings import (
            runtime_binding_response_payload,
        )

        return runtime_binding_response_payload(row)

    @staticmethod
    def _workspace_runtime_binding_fields_match(
        existing: Mapping[str, Any],
        requested: Mapping[str, Any],
    ) -> bool:
        return all(
            existing.get(field_name) == requested.get(field_name)
            for field_name in (
                "binding_kind",
                "owner_domain",
                "locator_ref",
                "label",
                "status",
                "path_hint",
                "portability",
                "metadata",
                "redaction_report",
            )
        )

    def _require_workspace_runtime_binding_workspace(
        self,
        conn: sqlite3.Connection | BackendConnectionWrapper,
        workspace_id: str,
    ) -> None:
        row = conn.execute(
            "SELECT id FROM workspaces WHERE id = ? AND deleted = ?",
            (workspace_id, self._workspace_active_deleted_value()),
        ).fetchone()
        if row is None:
            raise ConflictError(
                "Workspace not found or deleted.",
                entity="workspaces",
                entity_id=workspace_id,
            )

    def _get_workspace_runtime_binding_with_conn(
        self,
        conn: sqlite3.Connection | BackendConnectionWrapper,
        workspace_id: str,
        binding_id: str,
        *,
        include_deleted: bool,
    ) -> dict[str, Any] | None:
        include_deleted_flag = 1 if include_deleted else 0
        row = conn.execute(
            """
            SELECT *
              FROM workspace_runtime_bindings
             WHERE workspace_id = ?
               AND binding_id = ?
               AND (? = 1 OR deleted = ?)
            """,
            (
                workspace_id,
                binding_id,
                include_deleted_flag,
                self._workspace_active_deleted_value(),
            ),
        ).fetchone()
        return self._normalize_workspace_runtime_binding_row(row)

    def upsert_workspace_runtime_binding(
        self,
        workspace_id: str,
        data: Mapping[str, Any],
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Create, update, or restore a Workspace runtime binding descriptor."""
        normalized_workspace_id = self._normalize_workspace_runtime_binding_required_string(
            workspace_id,
            "workspace_id",
        )
        values = self._normalize_workspace_runtime_binding_input(data)
        actor = self._normalize_workspace_runtime_binding_actor(user_id)
        now = self._ctx.utc_now_iso()
        active_deleted_value = self._workspace_active_deleted_value()

        try:
            with self._ctx.transaction() as conn:
                self._require_workspace_runtime_binding_workspace(conn, normalized_workspace_id)
                existing = self._get_workspace_runtime_binding_with_conn(
                    conn,
                    normalized_workspace_id,
                    values["binding_id"],
                    include_deleted=True,
                )
                if existing is not None:
                    if (
                        existing.get("deleted") not in (True, 1)
                        and self._workspace_runtime_binding_fields_match(existing, values)
                    ):
                        return existing
                    cursor = conn.execute(
                        """
                        UPDATE workspace_runtime_bindings
                           SET binding_kind = ?,
                               owner_domain = ?,
                               locator_ref = ?,
                               label = ?,
                               status = ?,
                               path_hint = ?,
                               portability = ?,
                               metadata_json = ?,
                               redaction_report_json = ?,
                               updated_by_user_id = ?,
                               updated_at = ?,
                               deleted = ?,
                               client_id = ?,
                               version = version + 1
                         WHERE workspace_id = ?
                           AND binding_id = ?
                        """,
                        (
                            values["binding_kind"],
                            values["owner_domain"],
                            values["locator_ref"],
                            values["label"],
                            values["status"],
                            values["path_hint"],
                            values["portability"],
                            values["metadata_json"],
                            values["redaction_report_json"],
                            actor,
                            now,
                            active_deleted_value,
                            self._ctx.client_id or "unknown",
                            normalized_workspace_id,
                            values["binding_id"],
                        ),
                    )
                    if cursor.rowcount == 1:
                        updated = self._get_workspace_runtime_binding_with_conn(
                            conn,
                            normalized_workspace_id,
                            values["binding_id"],
                            include_deleted=False,
                        )
                        if updated is None:
                            raise CharactersRAGDBError("Updated runtime binding row could not be reloaded.")  # noqa: TRY003
                        return updated
                    raise CharactersRAGDBError("Runtime binding update changed no rows.")  # noqa: TRY003

                conn.execute(
                    """
                    INSERT INTO workspace_runtime_bindings (
                        workspace_id,
                        binding_id,
                        binding_kind,
                        owner_domain,
                        locator_ref,
                        label,
                        status,
                        path_hint,
                        portability,
                        metadata_json,
                        redaction_report_json,
                        created_by_user_id,
                        updated_by_user_id,
                        created_at,
                        updated_at,
                        deleted,
                        client_id,
                        version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        normalized_workspace_id,
                        values["binding_id"],
                        values["binding_kind"],
                        values["owner_domain"],
                        values["locator_ref"],
                        values["label"],
                        values["status"],
                        values["path_hint"],
                        values["portability"],
                        values["metadata_json"],
                        values["redaction_report_json"],
                        actor,
                        actor,
                        now,
                        now,
                        active_deleted_value,
                        self._ctx.client_id or "unknown",
                    ),
                )
                created = self._get_workspace_runtime_binding_with_conn(
                    conn,
                    normalized_workspace_id,
                    values["binding_id"],
                    include_deleted=False,
                )
                if created is None:
                    raise CharactersRAGDBError("Created runtime binding row could not be reloaded.")  # noqa: TRY003
                return created
        except ConflictError:
            raise
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            raise self._workspace_runtime_binding_write_error(exc) from exc

    def get_workspace_runtime_binding(
        self,
        workspace_id: str,
        binding_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch one Workspace runtime binding descriptor."""
        normalized_workspace_id = self._normalize_workspace_runtime_binding_required_string(
            workspace_id,
            "workspace_id",
        )
        normalized_binding_id = self._normalize_workspace_runtime_binding_required_string(
            binding_id,
            "binding_id",
        )
        try:
            if include_deleted:
                cursor = self._ctx.execute_query(
                    "SELECT * FROM workspace_runtime_bindings WHERE workspace_id = ? AND binding_id = ?",
                    (normalized_workspace_id, normalized_binding_id),
                )
                return self._normalize_workspace_runtime_binding_row(cursor.fetchone())
            cursor = self._ctx.execute_query(
                "SELECT * FROM workspace_runtime_bindings WHERE workspace_id = ? AND binding_id = ? AND deleted = ?",
                (
                    normalized_workspace_id,
                    normalized_binding_id,
                    self._workspace_active_deleted_value(),
                ),
            )
            return self._normalize_workspace_runtime_binding_row(cursor.fetchone())
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed fetching workspace runtime binding: {exc}") from exc  # noqa: TRY003

    def list_workspace_runtime_bindings(
        self,
        workspace_id: str,
        *,
        binding_kind: str | None = None,
        owner_domain: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List runtime binding descriptors for one Workspace."""
        normalized_workspace_id = self._normalize_workspace_runtime_binding_required_string(
            workspace_id,
            "workspace_id",
        )
        normalized_limit = self._normalize_runtime_binding_limit(limit)
        normalized_kind = (
            self._workspace_runtime_binding_filter_value(binding_kind, "binding_kind")
            if binding_kind is not None
            else None
        )
        normalized_owner = (
            self._workspace_runtime_binding_filter_value(owner_domain, "owner_domain")
            if owner_domain is not None
            else None
        )
        include_deleted_flag = 1 if include_deleted else 0
        cursor = self._ctx.execute_query(
            """
            SELECT *
              FROM workspace_runtime_bindings
             WHERE workspace_id = ?
               AND (? = 1 OR deleted = ?)
               AND (? IS NULL OR binding_kind = ?)
               AND (? IS NULL OR owner_domain = ?)
             ORDER BY updated_at DESC, binding_id ASC
             LIMIT ?
            """,
            (
                normalized_workspace_id,
                include_deleted_flag,
                self._workspace_active_deleted_value(),
                normalized_kind,
                normalized_kind,
                normalized_owner,
                normalized_owner,
                normalized_limit,
            ),
        )
        return [
            normalized
            for row in cursor.fetchall()
            if (normalized := self._normalize_workspace_runtime_binding_row(row)) is not None
        ]

    def workspace_runtime_binding_summary(
        self,
        workspace_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        """Return unpaginated runtime binding counts for one Workspace."""
        normalized_workspace_id = self._normalize_workspace_runtime_binding_required_string(
            workspace_id,
            "workspace_id",
        )
        include_deleted_flag = 1 if include_deleted else 0
        cursor = self._ctx.execute_query(
            """
            SELECT binding_kind, status, COUNT(*) AS count
              FROM workspace_runtime_bindings
             WHERE workspace_id = ?
               AND (? = 1 OR deleted = ?)
             GROUP BY binding_kind, status
            """,
            (
                normalized_workspace_id,
                include_deleted_flag,
                self._workspace_active_deleted_value(),
            ),
        )
        by_kind: Counter[str] = Counter()
        by_status: Counter[str] = Counter()
        total = 0
        for row in cursor.fetchall():
            count = int(row["count"] if isinstance(row, Mapping) else row[2])
            binding_kind = str((row["binding_kind"] if isinstance(row, Mapping) else row[0]) or "")
            status = str((row["status"] if isinstance(row, Mapping) else row[1]) or "")
            total += count
            if binding_kind:
                by_kind[binding_kind] += count
            if status:
                by_status[status] += count
        return {
            "total": total,
            "by_kind": dict(sorted(by_kind.items())),
            "by_status": dict(sorted(by_status.items())),
        }

    def archive_workspace_runtime_binding(
        self,
        workspace_id: str,
        binding_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Soft-delete one Workspace runtime binding descriptor."""
        normalized_workspace_id = self._normalize_workspace_runtime_binding_required_string(
            workspace_id,
            "workspace_id",
        )
        normalized_binding_id = self._normalize_workspace_runtime_binding_required_string(
            binding_id,
            "binding_id",
        )
        actor = self._normalize_workspace_runtime_binding_actor(user_id)
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                existing = self._get_workspace_runtime_binding_with_conn(
                    conn,
                    normalized_workspace_id,
                    normalized_binding_id,
                    include_deleted=True,
                )
                if existing is None or existing.get("deleted") in (True, 1):
                    return None
                deleted_value = True if self._ctx.backend_type == BackendType.POSTGRESQL else 1
                cursor = conn.execute(
                    """
                    UPDATE workspace_runtime_bindings
                       SET deleted = ?,
                           status = ?,
                           updated_at = ?,
                           updated_by_user_id = ?,
                           client_id = ?,
                           version = version + 1
                     WHERE workspace_id = ?
                       AND binding_id = ?
                       AND deleted = ?
                    """,
                    (
                        deleted_value,
                        "archived",
                        now,
                        actor,
                        self._ctx.client_id or "unknown",
                        normalized_workspace_id,
                        normalized_binding_id,
                        self._workspace_active_deleted_value(),
                    ),
                )
                if cursor.rowcount == 1:
                    return self._get_workspace_runtime_binding_with_conn(
                        conn,
                        normalized_workspace_id,
                        normalized_binding_id,
                        include_deleted=True,
                    )
                raced = self._get_workspace_runtime_binding_with_conn(
                    conn,
                    normalized_workspace_id,
                    normalized_binding_id,
                    include_deleted=True,
                )
                if raced is None or raced.get("deleted") in (True, 1):
                    return raced
                raise CharactersRAGDBError("Archive race left runtime binding active.")  # noqa: TRY003
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            raise self._workspace_runtime_binding_write_error(exc) from exc

    @staticmethod
    def _normalize_membership_limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise InputError("limit must be an integer.")  # noqa: TRY003
        if limit < 1 or limit > 1000:
            raise InputError("limit must be between 1 and 1000.")  # noqa: TRY003
        return limit

    @staticmethod
    def _normalize_workspace_membership_required_string(value: Any, field_name: str) -> str:
        normalized = str(value if value is not None else "").strip()
        if not normalized:
            raise InputError(f"{field_name} is required.")  # noqa: TRY003
        return normalized

    @staticmethod
    def _normalize_workspace_membership_optional_string(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @classmethod
    def _normalize_workspace_membership_json_object(cls, value: Any, field_name: str) -> tuple[dict[str, Any], str]:
        max_bytes = (
            WORKSPACE_MEMBERSHIP_MAX_PROVENANCE_BYTES
            if field_name == "provenance"
            else WORKSPACE_MEMBERSHIP_MAX_METADATA_BYTES
        )
        try:
            return dump_membership_json_object(value, field_name=field_name, max_bytes=max_bytes)
        except ValueError as exc:
            raise InputError(str(exc)) from exc  # noqa: TRY003

    @classmethod
    def _load_workspace_membership_json_object(cls, value: Any, *, field_name: str) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str):
            if not value.strip():
                return {}
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to decode workspace resource membership JSON field {} ({} chars)",
                    field_name,
                    len(value),
                )
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _normalize_workspace_membership_input(self, data: Mapping[str, Any]) -> dict[str, Any]:
        resource_type = self._normalize_workspace_membership_required_string(
            data.get("resource_type"),
            "resource_type",
        )
        resource_id = self._normalize_workspace_membership_required_string(
            data.get("resource_id"),
            "resource_id",
        )
        role = self._normalize_workspace_membership_optional_string(data.get("role")) or "member"
        transfer_policy = (
            self._normalize_workspace_membership_optional_string(data.get("transfer_policy"))
            or "link"
        )
        label = self._normalize_workspace_membership_optional_string(data.get("label"))
        provenance, provenance_json = self._normalize_workspace_membership_json_object(
            data.get("provenance", data.get("provenance_json")),
            "provenance",
        )
        metadata, metadata_json = self._normalize_workspace_membership_json_object(
            data.get("metadata", data.get("metadata_json")),
            "metadata",
        )
        return {
            "resource_type": resource_type,
            "resource_id": resource_id,
            "role": role,
            "label": label,
            "transfer_policy": transfer_policy,
            "provenance": provenance,
            "provenance_json": provenance_json,
            "metadata": metadata,
            "metadata_json": metadata_json,
        }

    def _normalize_workspace_membership_actor(self, user_id: str | None) -> str | None:
        if user_id is None:
            return None
        normalized = str(user_id).strip()
        return normalized or None

    def _normalize_workspace_membership_row(self, row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        item["provenance"] = self._load_workspace_membership_json_object(
            item.get("provenance_json"),
            field_name="provenance_json",
        )
        item["metadata"] = self._load_workspace_membership_json_object(
            item.get("metadata_json"),
            field_name="metadata_json",
        )
        item["version"] = int(item.get("version") or 1)
        return item

    @staticmethod
    def _workspace_membership_fields_match(
        existing: Mapping[str, Any],
        requested: Mapping[str, Any],
    ) -> bool:
        return (
            existing.get("role") == requested.get("role")
            and existing.get("label") == requested.get("label")
            and existing.get("transfer_policy") == requested.get("transfer_policy")
            and existing.get("provenance") == requested.get("provenance")
            and existing.get("metadata") == requested.get("metadata")
        )

    @staticmethod
    def _workspace_membership_is_constraint_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "unique",
                "duplicate",
                "constraint",
                "foreign key",
            )
        )

    def _workspace_membership_write_error(self, exc: Exception) -> CharactersRAGDBError:
        if self._workspace_membership_is_constraint_error(exc):
            return ConflictError(
                f"Workspace resource membership write conflicted: {exc}",
                entity="workspace_resource_memberships",
            )
        return CharactersRAGDBError(f"Workspace resource membership write failed: {exc}")  # noqa: TRY003

    def _get_workspace_resource_membership_with_conn(
        self,
        conn: sqlite3.Connection | BackendConnectionWrapper,
        workspace_id: str,
        resource_type: str,
        resource_id: str,
        *,
        include_deleted: bool,
    ) -> dict[str, Any] | None:
        clauses = [
            "workspace_id = ?",
            "resource_type = ?",
            "resource_id = ?",
        ]
        params: list[Any] = [workspace_id, resource_type, resource_id]
        if not include_deleted:
            clauses.append("deleted = ?")
            params.append(self._workspace_active_deleted_value())
        row = conn.execute(
            f"SELECT * FROM workspace_resource_memberships WHERE {' AND '.join(clauses)}",  # nosec B608
            tuple(params),
        ).fetchone()
        return self._normalize_workspace_membership_row(row)

    def add_workspace_resource_membership(
        self,
        workspace_id: str,
        data: dict[str, Any],
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Create or restore a Workspace resource membership."""
        normalized_workspace_id = self._normalize_workspace_membership_required_string(
            workspace_id,
            "workspace_id",
        )
        values = self._normalize_workspace_membership_input(data)
        actor = self._normalize_workspace_membership_actor(user_id)
        restore_deleted = bool(data.get("restore_deleted") is True)
        now = self._ctx.utc_now_iso()
        active_deleted_value = self._workspace_active_deleted_value()

        try:
            with self._ctx.transaction() as conn:
                existing = self._get_workspace_resource_membership_with_conn(
                    conn,
                    normalized_workspace_id,
                    values["resource_type"],
                    values["resource_id"],
                    include_deleted=True,
                )
                if existing is not None:
                    if existing.get("deleted") in (True, 1):
                        if not restore_deleted:
                            raise ConflictError(
                                "Workspace resource membership exists but is deleted.",
                                entity="workspace_resource_memberships",
                                entity_id=values["resource_id"],
                            )
                        deleted_value = True if self._ctx.backend_type == BackendType.POSTGRESQL else 1
                        cursor = conn.execute(
                            """
                            UPDATE workspace_resource_memberships
                               SET role = ?,
                                   label = ?,
                                   transfer_policy = ?,
                                   provenance_json = ?,
                                   metadata_json = ?,
                                   updated_by_user_id = ?,
                                   updated_at = ?,
                                   deleted = ?,
                                   client_id = ?,
                                   version = version + 1
                             WHERE workspace_id = ?
                               AND resource_type = ?
                               AND resource_id = ?
                               AND deleted = ?
                            """,
                            (
                                values["role"],
                                values["label"],
                                values["transfer_policy"],
                                values["provenance_json"],
                                values["metadata_json"],
                                actor,
                                now,
                                active_deleted_value,
                                self._ctx.client_id or "unknown",
                                normalized_workspace_id,
                                values["resource_type"],
                                values["resource_id"],
                                deleted_value,
                            ),
                        )
                        if cursor.rowcount == 1:
                            restored = self._get_workspace_resource_membership_with_conn(
                                conn,
                                normalized_workspace_id,
                                values["resource_type"],
                                values["resource_id"],
                                include_deleted=False,
                            )
                            if restored is None:
                                raise CharactersRAGDBError("Restored membership row could not be reloaded.")  # noqa: TRY003
                            return restored

                        raced = self._get_workspace_resource_membership_with_conn(
                            conn,
                            normalized_workspace_id,
                            values["resource_type"],
                            values["resource_id"],
                            include_deleted=True,
                        )
                        if raced is None or raced.get("deleted") in (True, 1):
                            raise CharactersRAGDBError("Restore race left membership unavailable.")  # noqa: TRY003
                        if self._workspace_membership_fields_match(raced, values):
                            return raced
                        raise ConflictError(
                            "Workspace resource membership already restored with different fields.",
                            entity="workspace_resource_memberships",
                            entity_id=values["resource_id"],
                        )

                    if self._workspace_membership_fields_match(existing, values):
                        return existing
                    raise ConflictError(
                        "Workspace resource membership already exists with different fields.",
                        entity="workspace_resource_memberships",
                        entity_id=values["resource_id"],
                    )

                conn.execute(
                    """
                    INSERT INTO workspace_resource_memberships (
                        workspace_id,
                        resource_type,
                        resource_id,
                        role,
                        label,
                        transfer_policy,
                        provenance_json,
                        metadata_json,
                        created_by_user_id,
                        updated_by_user_id,
                        created_at,
                        updated_at,
                        deleted,
                        client_id,
                        version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        normalized_workspace_id,
                        values["resource_type"],
                        values["resource_id"],
                        values["role"],
                        values["label"],
                        values["transfer_policy"],
                        values["provenance_json"],
                        values["metadata_json"],
                        actor,
                        actor,
                        now,
                        now,
                        active_deleted_value,
                        self._ctx.client_id or "unknown",
                    ),
                )
                created = self._get_workspace_resource_membership_with_conn(
                    conn,
                    normalized_workspace_id,
                    values["resource_type"],
                    values["resource_id"],
                    include_deleted=False,
                )
                if created is None:
                    raise CharactersRAGDBError("Created membership row could not be reloaded.")  # noqa: TRY003
                return created
        except ConflictError:
            raise
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            existing = self.get_workspace_resource_membership(
                normalized_workspace_id,
                values["resource_type"],
                values["resource_id"],
                include_deleted=True,
            )
            if existing is not None and not existing.get("deleted") and self._workspace_membership_fields_match(
                existing,
                values,
            ):
                return existing
            raise self._workspace_membership_write_error(exc) from exc

    def get_workspace_resource_membership(
        self,
        workspace_id: str,
        resource_type: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch one Workspace resource membership by composite id."""
        normalized_workspace_id = self._normalize_workspace_membership_required_string(
            workspace_id,
            "workspace_id",
        )
        normalized_resource_type = self._normalize_workspace_membership_required_string(
            resource_type,
            "resource_type",
        )
        normalized_resource_id = self._normalize_workspace_membership_required_string(
            resource_id,
            "resource_id",
        )
        try:
            if include_deleted:
                cursor = self._ctx.execute_query(
                    "SELECT * FROM workspace_resource_memberships "
                    "WHERE workspace_id = ? AND resource_type = ? AND resource_id = ?",
                    (
                        normalized_workspace_id,
                        normalized_resource_type,
                        normalized_resource_id,
                    ),
                )
                return self._normalize_workspace_membership_row(cursor.fetchone())
            cursor = self._ctx.execute_query(
                "SELECT * FROM workspace_resource_memberships "
                "WHERE workspace_id = ? AND resource_type = ? AND resource_id = ? AND deleted = ?",
                (
                    normalized_workspace_id,
                    normalized_resource_type,
                    normalized_resource_id,
                    self._workspace_active_deleted_value(),
                ),
            )
            return self._normalize_workspace_membership_row(cursor.fetchone())
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed fetching workspace resource membership: {exc}") from exc  # noqa: TRY003

    def list_workspace_resource_memberships(
        self,
        workspace_id: str,
        *,
        resource_type: str | None = None,
        role: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        cursor: tuple[str, str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """List resource memberships for a workspace in deterministic order."""
        normalized_workspace_id = self._normalize_workspace_membership_required_string(
            workspace_id,
            "workspace_id",
        )
        normalized_limit = self._normalize_membership_limit(limit)
        clauses = ["workspace_id = ?"]
        params: list[Any] = [normalized_workspace_id]
        if resource_type is not None:
            clauses.append("resource_type = ?")
            params.append(
                self._normalize_workspace_membership_required_string(
                    resource_type,
                    "resource_type",
                )
            )
        if role is not None:
            clauses.append("role = ?")
            params.append(self._normalize_workspace_membership_required_string(role, "role"))
        if not include_deleted:
            clauses.append("deleted = ?")
            params.append(self._workspace_active_deleted_value())
        if cursor is not None:
            if len(cursor) != 3:
                raise InputError("cursor must be a tuple of (updated_at, resource_type, resource_id).")  # noqa: TRY003
            cursor_updated_at, cursor_resource_type, cursor_resource_id = cursor
            clauses.append(
                "(updated_at < ? "
                "OR (updated_at = ? AND resource_type > ?) "
                "OR (updated_at = ? AND resource_type = ? AND resource_id > ?))"
            )
            params.extend(
                [
                    cursor_updated_at,
                    cursor_updated_at,
                    cursor_resource_type,
                    cursor_updated_at,
                    cursor_resource_type,
                    cursor_resource_id,
                ]
            )
        params.append(normalized_limit + 1)
        cursor_obj = self._ctx.execute_query(
            f"SELECT * FROM workspace_resource_memberships WHERE {' AND '.join(clauses)} "  # nosec B608
            "ORDER BY updated_at DESC, resource_type ASC, resource_id ASC LIMIT ?",
            tuple(params),
        )
        return [
            normalized
            for row in cursor_obj.fetchall()
            if (normalized := self._normalize_workspace_membership_row(row)) is not None
        ]

    def list_resource_workspace_memberships(
        self,
        resource_type: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
        limit: int = 100,
        cursor: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """List all workspace memberships for one resource."""
        normalized_resource_type = self._normalize_workspace_membership_required_string(
            resource_type,
            "resource_type",
        )
        normalized_resource_id = self._normalize_workspace_membership_required_string(
            resource_id,
            "resource_id",
        )
        normalized_limit = self._normalize_membership_limit(limit)
        clauses = ["resource_type = ?", "resource_id = ?"]
        params: list[Any] = [normalized_resource_type, normalized_resource_id]
        if not include_deleted:
            clauses.append("deleted = ?")
            params.append(self._workspace_active_deleted_value())
        if cursor is not None:
            if len(cursor) != 2:
                raise InputError("cursor must be a tuple of (updated_at, workspace_id).")  # noqa: TRY003
            cursor_updated_at, cursor_workspace_id = cursor
            clauses.append(
                "(updated_at < ? OR (updated_at = ? AND workspace_id > ?))"
            )
            params.extend([cursor_updated_at, cursor_updated_at, cursor_workspace_id])
        params.append(normalized_limit + 1)
        cursor_obj = self._ctx.execute_query(
            f"SELECT * FROM workspace_resource_memberships WHERE {' AND '.join(clauses)} "  # nosec B608
            "ORDER BY updated_at DESC, workspace_id ASC LIMIT ?",
            tuple(params),
        )
        return [
            normalized
            for row in cursor_obj.fetchall()
            if (normalized := self._normalize_workspace_membership_row(row)) is not None
        ]

    def workspace_resource_membership_summary(self, workspace_id: str) -> dict[str, Any]:
        """Return compact active membership counts for one Workspace."""
        normalized_workspace_id = self._normalize_workspace_membership_required_string(
            workspace_id,
            "workspace_id",
        )
        active_deleted_value = self._workspace_active_deleted_value()

        def _row_value(row: Any, key: str, index: int, default: Any = 0) -> Any:
            if row is None:
                return default
            try:
                return row[key]
            except (KeyError, IndexError, TypeError):
                try:
                    return row[index]
                except (IndexError, TypeError):
                    return default

        try:
            total_cursor = self._ctx.execute_query(
                """
                SELECT COUNT(*) AS total
                  FROM workspace_resource_memberships
                 WHERE workspace_id = ? AND deleted = ?
                """,
                (normalized_workspace_id, active_deleted_value),
            )
            total_row = total_cursor.fetchone()
            resource_cursor = self._ctx.execute_query(
                """
                SELECT resource_type, COUNT(*) AS count
                  FROM workspace_resource_memberships
                 WHERE workspace_id = ? AND deleted = ?
                 GROUP BY resource_type
                 ORDER BY resource_type ASC
                """,
                (normalized_workspace_id, active_deleted_value),
            )
            role_cursor = self._ctx.execute_query(
                """
                SELECT role, COUNT(*) AS count
                  FROM workspace_resource_memberships
                 WHERE workspace_id = ? AND deleted = ?
                 GROUP BY role
                 ORDER BY role ASC
                """,
                (normalized_workspace_id, active_deleted_value),
            )
            by_resource_type = {
                str(_row_value(row, "resource_type", 0)): int(_row_value(row, "count", 1))
                for row in resource_cursor.fetchall()
                if _row_value(row, "resource_type", 0, None) is not None
            }
            by_role = {
                str(_row_value(row, "role", 0)): int(_row_value(row, "count", 1))
                for row in role_cursor.fetchall()
                if _row_value(row, "role", 0, None) is not None
            }
            return {
                "total": int(_row_value(total_row, "total", 0)),
                "by_resource_type": by_resource_type,
                "by_role": by_role,
            }
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(
                f"Failed summarizing workspace resource memberships: {exc}"
            ) from exc  # noqa: TRY003

    def delete_workspace_resource_membership(
        self,
        workspace_id: str,
        resource_type: str,
        resource_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Soft-delete one Workspace resource membership.

        Missing rows and already deleted rows are idempotent no-ops.
        """
        normalized_workspace_id = self._normalize_workspace_membership_required_string(
            workspace_id,
            "workspace_id",
        )
        normalized_resource_type = self._normalize_workspace_membership_required_string(
            resource_type,
            "resource_type",
        )
        normalized_resource_id = self._normalize_workspace_membership_required_string(
            resource_id,
            "resource_id",
        )
        actor = self._normalize_workspace_membership_actor(user_id)
        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                existing = self._get_workspace_resource_membership_with_conn(
                    conn,
                    normalized_workspace_id,
                    normalized_resource_type,
                    normalized_resource_id,
                    include_deleted=True,
                )
                if existing is None or existing.get("deleted") in (True, 1):
                    return None
                deleted_value = True if self._ctx.backend_type == BackendType.POSTGRESQL else 1
                active_deleted_value = self._workspace_active_deleted_value()
                cursor = conn.execute(
                    """
                    UPDATE workspace_resource_memberships
                       SET deleted = ?,
                           updated_at = ?,
                           updated_by_user_id = ?,
                           client_id = ?,
                           version = version + 1
                     WHERE workspace_id = ?
                       AND resource_type = ?
                       AND resource_id = ?
                       AND deleted = ?
                    """,
                    (
                        deleted_value,
                        now,
                        actor,
                        self._ctx.client_id or "unknown",
                        normalized_workspace_id,
                        normalized_resource_type,
                        normalized_resource_id,
                        active_deleted_value,
                    ),
                )
                if cursor.rowcount == 1:
                    return self._get_workspace_resource_membership_with_conn(
                        conn,
                        normalized_workspace_id,
                        normalized_resource_type,
                        normalized_resource_id,
                        include_deleted=True,
                    )

                raced = self._get_workspace_resource_membership_with_conn(
                    conn,
                    normalized_workspace_id,
                    normalized_resource_type,
                    normalized_resource_id,
                    include_deleted=True,
                )
                if raced is None or raced.get("deleted") in (True, 1):
                    return None
                raise CharactersRAGDBError("Delete race left membership active.")  # noqa: TRY003
        except (sqlite3.IntegrityError, BackendDatabaseError) as exc:
            raise self._workspace_membership_write_error(exc) from exc

    def get_conversation_for_workspace_membership(
        self,
        conversation_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch the compact conversation row needed by Workspace membership adapters."""
        normalized_conversation_id = self._normalize_workspace_membership_required_string(
            conversation_id,
            "conversation_id",
        )
        deleted_clause = ""
        params: list[Any] = [normalized_conversation_id]
        if not include_deleted:
            deleted_clause = " AND deleted = ?"
            params.append(self._workspace_active_deleted_value())
        cursor = self._ctx.execute_query(
            "SELECT id, title, workspace_id, scope_type, last_modified, deleted, version "
            f"FROM conversations WHERE id = ?{deleted_clause}",  # nosec B608
            tuple(params),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    @staticmethod
    def _copy_workspace_artifact_default(default: Any) -> Any:
        if isinstance(default, (dict, list)):
            return json.loads(json.dumps(default))
        return default

    @classmethod
    def _load_workspace_artifact_json(
        cls,
        raw: Any,
        default: Any,
        *,
        field_name: str = "workspace_artifact_json",
    ) -> Any:
        if raw is None:
            return cls._copy_workspace_artifact_default(default)
        if isinstance(raw, (dict, list)):
            return raw
        if isinstance(raw, str):
            if not raw.strip():
                return cls._copy_workspace_artifact_default(default)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                preview = raw[:120].replace("\n", "\\n")
                logger.warning(
                    "Failed to decode workspace artifact JSON field {} ({} chars): {}",
                    field_name,
                    len(raw),
                    preview,
                )
                return cls._copy_workspace_artifact_default(default)
        return cls._copy_workspace_artifact_default(default)

    @classmethod
    def _dump_workspace_artifact_json(cls, value: Any, default: Any) -> str:
        if value is None:
            value = cls._copy_workspace_artifact_default(default)
        return json.dumps(value, ensure_ascii=True)

    @staticmethod
    def _workspace_artifact_version_id(artifact_id: str, version: int) -> str:
        return f"{artifact_id}:v{version}"

    def _normalize_workspace_artifact_review_state(self, value: Any) -> str:
        state = str(value or "draft").strip().lower()
        if state not in _ALLOWED_WORKSPACE_ARTIFACT_REVIEW_STATES:
            raise InputError(f"Unsupported workspace artifact review state '{state}'.")  # noqa: TRY003
        return state

    def _normalize_workspace_artifact_row(self, row: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not row:
            return None
        item = dict(row)
        artifact_id = str(item.get("id") or "")
        version = int(item.get("version") or 1)
        item["content_type"] = item.get("content_type") or "text/markdown"
        item["review_state"] = item.get("review_state") or "draft"
        item["owner_scope"] = item.get("owner_scope") or "user"
        item["root_artifact_id"] = item.get("root_artifact_id") or artifact_id
        item["artifact_version_id"] = item.get("artifact_version_id") or self._workspace_artifact_version_id(artifact_id, version)
        item["previous_version_id"] = item.get("previous_version_id")
        item["schema_version"] = int(item.get("schema_version") or 1)
        for column_name, response_name, default in _WORKSPACE_ARTIFACT_JSON_FIELDS:
            item[response_name] = self._load_workspace_artifact_json(
                item.get(column_name),
                default,
                field_name=column_name,
            )
        return item

    def _insert_workspace_artifact_version(
        self,
        conn: Any,
        *,
        workspace_id: str,
        artifact_id: str,
        artifact_version_id: str,
        root_artifact_id: str,
        previous_version_id: str | None,
        version: int,
        review_state: str,
        title: str,
        content: str | None,
        producer_metadata_json: str,
        source_lineage_json: str,
        review_metadata_json: str,
        version_metadata_json: str,
        export_refs_json: str,
        redaction_json: str,
        created_at: str,
    ) -> None:
        query = (
            "INSERT INTO workspace_artifact_versions (workspace_id, artifact_id, artifact_version_id, "
            "root_artifact_id, previous_version_id, version, review_state, title, content, "
            "producer_metadata_json, source_lineage_json, review_metadata_json, version_metadata_json, "
            "export_refs_json, redaction_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        conn.execute(
            query,
            (
                workspace_id,
                artifact_id,
                artifact_version_id,
                root_artifact_id,
                previous_version_id,
                version,
                review_state,
                title,
                content,
                producer_metadata_json,
                source_lineage_json,
                review_metadata_json,
                version_metadata_json,
                export_refs_json,
                redaction_json,
                created_at,
            ),
        )

    def add_workspace_artifact(self, workspace_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Add an artifact to a workspace."""
        artifact_id = data.get("id")
        if not artifact_id:
            raise InputError("Artifact id is required.")  # noqa: TRY003
        artifact_id = str(artifact_id)
        now = self._ctx.utc_now_iso()
        review_state = self._normalize_workspace_artifact_review_state(data.get("review_state"))
        root_artifact_id = artifact_id
        artifact_version_id = self._workspace_artifact_version_id(artifact_id, 1)
        previous_version_id = None
        content_type = str(data.get("content_type") or "text/markdown")
        producer_metadata_json = self._dump_workspace_artifact_json(data.get("producer_metadata"), {})
        source_lineage_json = self._dump_workspace_artifact_json(data.get("source_lineage"), {})
        review_metadata_json = self._dump_workspace_artifact_json(data.get("review_metadata"), {})
        version_metadata_json = self._dump_workspace_artifact_json(data.get("version_metadata"), {})
        export_refs_json = self._dump_workspace_artifact_json(data.get("export_refs"), [])
        redaction_json = self._dump_workspace_artifact_json(data.get("redaction"), {"support_safe": True, "redacted": False})
        schema_version = int(data.get("schema_version") or 1)
        query = (
            "INSERT INTO workspace_artifacts (id, workspace_id, artifact_type, title, status, content, "
            "total_tokens, total_cost_usd, created_at, completed_at, version, content_type, preview_text, summary, "
            "review_state, owner_scope, owner_id, project_id, task_id, source_collection_id, root_artifact_id, "
            "artifact_version_id, previous_version_id, producer_metadata_json, source_lineage_json, review_metadata_json, "
            "version_metadata_json, export_refs_json, redaction_json, schema_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            artifact_id,
            workspace_id,
            data.get("artifact_type", ""),
            data.get("title", ""),
            data.get("status", "pending"),
            data.get("content"),
            data.get("total_tokens"),
            data.get("total_cost_usd"),
            now,
            data.get("completed_at"),
            content_type,
            data.get("preview_text"),
            data.get("summary"),
            review_state,
            data.get("owner_scope") or "user",
            data.get("owner_id"),
            data.get("project_id"),
            data.get("task_id"),
            data.get("source_collection_id"),
            root_artifact_id,
            artifact_version_id,
            previous_version_id,
            producer_metadata_json,
            source_lineage_json,
            review_metadata_json,
            version_metadata_json,
            export_refs_json,
            redaction_json,
            schema_version,
        )
        with self._ctx.transaction() as conn:
            conn.execute(query, params)
            self._insert_workspace_artifact_version(
                conn,
                workspace_id=workspace_id,
                artifact_id=artifact_id,
                artifact_version_id=artifact_version_id,
                root_artifact_id=root_artifact_id,
                previous_version_id=previous_version_id,
                version=1,
                review_state=review_state,
                title=str(data.get("title", "")),
                content=data.get("content"),
                producer_metadata_json=producer_metadata_json,
                source_lineage_json=source_lineage_json,
                review_metadata_json=review_metadata_json,
                version_metadata_json=version_metadata_json,
                export_refs_json=export_refs_json,
                redaction_json=redaction_json,
                created_at=now,
            )
            row = conn.execute(
                "SELECT * FROM workspace_artifacts WHERE workspace_id = ? AND id = ?",
                (workspace_id, artifact_id),
            ).fetchone()
            artifact = self._normalize_workspace_artifact_row(row)
            if artifact is None:
                raise CharactersRAGDBError("Workspace artifact insert could not be verified.")  # noqa: TRY003
        return artifact

    def _get_workspace_artifact(
        self,
        workspace_id: str,
        artifact_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_artifacts WHERE workspace_id = ? AND id = ?",
            (workspace_id, artifact_id),
        )
        row = cursor.fetchone()
        return self._normalize_workspace_artifact_row(row)

    def get_workspace_artifact(
        self,
        workspace_id: str,
        artifact_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a workspace artifact by id."""
        return self._get_workspace_artifact(workspace_id, artifact_id, include_deleted=include_deleted)

    def list_workspace_artifacts(self, workspace_id: str) -> list[dict[str, Any]]:
        """List all artifacts for a workspace."""
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_artifacts WHERE workspace_id = ? ORDER BY created_at DESC",
            (workspace_id,),
        )
        return [
            normalized
            for row in cursor.fetchall()
            if (normalized := self._normalize_workspace_artifact_row(row)) is not None
        ]

    def list_workspace_artifact_versions(self, workspace_id: str, artifact_id: str) -> list[dict[str, Any]]:
        """List version history for a workspace artifact."""
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_artifact_versions WHERE workspace_id = ? AND artifact_id = ? ORDER BY version ASC",
            (workspace_id, artifact_id),
        )
        versions: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            item = dict(row)
            for column_name, response_name, default in _WORKSPACE_ARTIFACT_JSON_FIELDS:
                item[response_name] = self._load_workspace_artifact_json(
                    item.get(column_name),
                    default,
                    field_name=column_name,
                )
            versions.append(item)
        return versions

    def get_workspace_artifact_version(
        self,
        workspace_id: str,
        artifact_id: str,
        artifact_version_id: str,
    ) -> dict[str, Any] | None:
        """Fetch one workspace artifact version snapshot by stable version id."""
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_artifact_versions "
            "WHERE workspace_id = ? AND artifact_id = ? AND artifact_version_id = ?",
            (workspace_id, artifact_id, artifact_version_id),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        item = dict(row)
        for column_name, response_name, default in _WORKSPACE_ARTIFACT_JSON_FIELDS:
            item[response_name] = self._load_workspace_artifact_json(
                item.get(column_name),
                default,
                field_name=column_name,
            )
        return item

    def append_workspace_artifact_export_ref(
        self,
        workspace_id: str,
        artifact_id: str,
        export_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append an export reference without creating a new artifact content version."""
        for _attempt in range(3):
            with self._ctx.transaction() as conn:
                artifact_row = conn.execute(
                    "SELECT artifact_version_id, export_refs_json FROM workspace_artifacts "
                    "WHERE workspace_id = ? AND id = ?",
                    (workspace_id, artifact_id),
                ).fetchone()
                if artifact_row is None:
                    raise ConflictError(  # noqa: TRY003
                        f"Workspace artifact '{artifact_id}' not found.",
                        entity="workspace_artifacts",
                        entity_id=artifact_id,
                    )

                artifact_export_refs_json = artifact_row["export_refs_json"]
                artifact_version_id = str(
                    export_ref.get("artifact_version_id") or artifact_row["artifact_version_id"] or ""
                )
                if not artifact_version_id:
                    raise InputError("Workspace artifact export_ref artifact_version_id is required.")  # noqa: TRY003

                artifact_export_refs = self._load_workspace_artifact_json(
                    artifact_export_refs_json,
                    [],
                    field_name="workspace_artifacts.export_refs_json",
                )
                if not isinstance(artifact_export_refs, list):
                    raise InputError("Workspace artifact export_refs must be a list.")  # noqa: TRY003
                artifact_export_refs = list(artifact_export_refs)
                artifact_export_refs.append(dict(export_ref))
                updated_artifact_export_refs_json = self._dump_workspace_artifact_json(artifact_export_refs, [])

                if artifact_export_refs_json is None:
                    artifact_cursor = conn.execute(
                        "UPDATE workspace_artifacts SET export_refs_json = ? "
                        "WHERE workspace_id = ? AND id = ? AND export_refs_json IS NULL",
                        (updated_artifact_export_refs_json, workspace_id, artifact_id),
                    )
                else:
                    artifact_cursor = conn.execute(
                        "UPDATE workspace_artifacts SET export_refs_json = ? "
                        "WHERE workspace_id = ? AND id = ? AND export_refs_json = ?",
                        (updated_artifact_export_refs_json, workspace_id, artifact_id, artifact_export_refs_json),
                    )
                if artifact_cursor.rowcount == 0:
                    continue

                version_row = conn.execute(
                    "SELECT export_refs_json FROM workspace_artifact_versions "
                    "WHERE workspace_id = ? AND artifact_id = ? AND artifact_version_id = ?",
                    (workspace_id, artifact_id, artifact_version_id),
                ).fetchone()
                if version_row is None:
                    raise ConflictError(  # noqa: TRY003
                        f"Workspace artifact version '{artifact_version_id}' missing for artifact '{artifact_id}'.",
                        entity="workspace_artifact_versions",
                        entity_id=artifact_version_id,
                    )

                version_export_refs = self._load_workspace_artifact_json(
                    version_row["export_refs_json"],
                    [],
                    field_name="workspace_artifact_versions.export_refs_json",
                )
                if not isinstance(version_export_refs, list):
                    raise InputError("Workspace artifact version export_refs must be a list.")  # noqa: TRY003
                version_export_refs = list(version_export_refs)
                version_export_refs.append(dict(export_ref))
                version_export_refs_json = self._dump_workspace_artifact_json(version_export_refs, [])
                version_cursor = conn.execute(
                    "UPDATE workspace_artifact_versions SET export_refs_json = ? "
                    "WHERE workspace_id = ? AND artifact_id = ? AND artifact_version_id = ?",
                    (version_export_refs_json, workspace_id, artifact_id, artifact_version_id),
                )
                if version_cursor.rowcount == 0:
                    raise ConflictError(  # noqa: TRY003
                        f"Artifact version '{artifact_version_id}' export ref update failed.",
                        entity="workspace_artifact_versions",
                        entity_id=artifact_version_id,
                    )
                return self._get_workspace_artifact(workspace_id, artifact_id)  # type: ignore[return-value]

        raise ConflictError(  # noqa: TRY003
            f"Artifact '{artifact_id}' export ref update conflicted with another writer.",
            entity="workspace_artifacts",
            entity_id=artifact_id,
        )

    def update_workspace_artifact(
        self,
        workspace_id: str,
        artifact_id: str,
        updates: dict[str, Any],
        *,
        expected_version: int,
    ) -> dict[str, Any]:
        """Update a workspace artifact with optimistic locking."""
        existing = self._get_workspace_artifact(workspace_id, artifact_id)
        if existing is None:
            raise ConflictError(  # noqa: TRY003
                f"Workspace artifact '{artifact_id}' not found.",
                entity="workspace_artifacts",
                entity_id=artifact_id,
            )
        if existing["version"] != expected_version:
            raise ConflictError(  # noqa: TRY003
                f"Artifact '{artifact_id}' version mismatch (db={existing['version']}, expected={expected_version}).",
                entity="workspace_artifacts",
                entity_id=artifact_id,
            )
        next_version = expected_version + 1
        next_root_artifact_id = str(existing.get("root_artifact_id") or artifact_id)
        previous_version_id = self._workspace_artifact_version_id(artifact_id, expected_version)
        next_artifact_version_id = self._workspace_artifact_version_id(artifact_id, next_version)
        review_state = self._normalize_workspace_artifact_review_state(
            updates.get("review_state") if "review_state" in updates else existing.get("review_state")
        )
        set_clauses = ["version = ?"]
        params: list[Any] = [next_version]
        forced_updates: dict[str, Any] = {
            "root_artifact_id": next_root_artifact_id,
            "artifact_version_id": next_artifact_version_id,
            "previous_version_id": previous_version_id,
            "review_state": review_state,
        }
        for col, value in forced_updates.items():
            set_clauses.append(f"{col} = ?")
            params.append(value)
        for col in (
            "title",
            "status",
            "content",
            "content_type",
            "preview_text",
            "summary",
            "owner_scope",
            "owner_id",
            "project_id",
            "task_id",
            "source_collection_id",
            "total_tokens",
            "total_cost_usd",
            "completed_at",
            "schema_version",
        ):
            if col in updates:
                set_clauses.append(f"{col} = ?")
                params.append(updates[col])
        json_updates: dict[str, tuple[str, Any]] = {
            "producer_metadata": ("producer_metadata_json", {}),
            "source_lineage": ("source_lineage_json", {}),
            "review_metadata": ("review_metadata_json", {}),
            "version_metadata": ("version_metadata_json", {}),
            "export_refs": ("export_refs_json", []),
            "redaction": ("redaction_json", {"support_safe": True, "redacted": False}),
        }
        next_json_values: dict[str, str] = {}
        for field_name, (column_name, default) in json_updates.items():
            value = updates[field_name] if field_name in updates else existing.get(field_name)
            dumped = self._dump_workspace_artifact_json(value, default)
            next_json_values[column_name] = dumped
            if field_name in updates:
                set_clauses.append(f"{column_name} = ?")
                params.append(dumped)
        query = f"UPDATE workspace_artifacts SET {', '.join(set_clauses)} WHERE workspace_id = ? AND id = ? AND version = ?"  # nosec B608
        params.extend([workspace_id, artifact_id, expected_version])
        with self._ctx.transaction() as conn:
            cursor = conn.execute(query, tuple(params))
            if cursor.rowcount == 0:
                raise ConflictError(  # noqa: TRY003
                    f"Artifact '{artifact_id}' concurrent update detected.",
                    entity="workspace_artifacts",
                    entity_id=artifact_id,
                )
            self._insert_workspace_artifact_version(
                conn,
                workspace_id=workspace_id,
                artifact_id=artifact_id,
                artifact_version_id=next_artifact_version_id,
                root_artifact_id=next_root_artifact_id,
                previous_version_id=previous_version_id,
                version=next_version,
                review_state=review_state,
                title=str(updates.get("title") if "title" in updates else existing.get("title") or ""),
                content=updates.get("content") if "content" in updates else existing.get("content"),
                producer_metadata_json=next_json_values["producer_metadata_json"],
                source_lineage_json=next_json_values["source_lineage_json"],
                review_metadata_json=next_json_values["review_metadata_json"],
                version_metadata_json=next_json_values["version_metadata_json"],
                export_refs_json=next_json_values["export_refs_json"],
                redaction_json=next_json_values["redaction_json"],
                created_at=self._ctx.utc_now_iso(),
            )
        return self._get_workspace_artifact(workspace_id, artifact_id)  # type: ignore[return-value]

    def delete_workspace_artifact(self, workspace_id: str, artifact_id: str) -> None:
        """Hard-delete a workspace artifact."""
        with self._ctx.transaction() as conn:
            conn.execute(
                "DELETE FROM workspace_artifact_versions WHERE workspace_id = ? AND artifact_id = ?",
                (workspace_id, artifact_id),
            )
            conn.execute(
                "DELETE FROM workspace_artifacts WHERE workspace_id = ? AND id = ?",
                (workspace_id, artifact_id),
            )

    def add_workspace_note(self, workspace_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Add a note to a workspace."""
        now = self._ctx.utc_now_iso()
        import json as _json
        keywords_json = _json.dumps(data.get("keywords", []))
        query = (
            "INSERT INTO workspace_notes (workspace_id, title, content, keywords_json, created_at, last_modified, deleted, version) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, 1)"
        )
        params = (
            workspace_id,
            data.get("title", ""),
            data.get("content", ""),
            keywords_json,
            now,
            now,
        )
        with self._ctx.transaction() as conn:
            if self._ctx.backend_type == BackendType.POSTGRESQL:
                cursor = conn.execute(query + " RETURNING id", params)
                inserted = cursor.fetchone()
                note_id = int(inserted["id"]) if inserted else None
            else:
                cursor = conn.execute(query, params)
                note_id = cursor.lastrowid
            if note_id is None:
                raise CharactersRAGDBError("Workspace note insert ID was unavailable.")  # noqa: TRY003
            row = conn.execute(
                "SELECT * FROM workspace_notes WHERE workspace_id = ? AND id = ?",
                (workspace_id, note_id),
            ).fetchone()
            if row is None:
                raise CharactersRAGDBError("Workspace note insert could not be verified.")  # noqa: TRY003
            note = dict(row)
        return note

    def _get_workspace_note(
        self,
        workspace_id: str,
        note_id: int,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        deleted_clause = "" if include_deleted else " AND deleted = 0"
        cursor = self._ctx.execute_query(
            f"SELECT * FROM workspace_notes WHERE workspace_id = ? AND id = ?{deleted_clause}",  # nosec B608
            (workspace_id, note_id),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_workspace_note(
        self,
        workspace_id: str,
        note_id: int,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Fetch a workspace note by id."""
        return self._get_workspace_note(workspace_id, note_id, include_deleted=include_deleted)

    def list_workspace_notes(self, workspace_id: str) -> list[dict[str, Any]]:
        """List all non-deleted notes for a workspace."""
        cursor = self._ctx.execute_query(
            "SELECT * FROM workspace_notes WHERE workspace_id = ? AND deleted = 0 ORDER BY last_modified DESC",
            (workspace_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def update_workspace_note(
        self,
        workspace_id: str,
        note_id: int,
        updates: dict[str, Any],
        *,
        expected_version: int,
    ) -> dict[str, Any]:
        """Update a workspace note with optimistic locking."""
        existing = self._get_workspace_note(workspace_id, note_id)
        if existing is None:
            raise ConflictError(  # noqa: TRY003
                f"Workspace note '{note_id}' not found.",
                entity="workspace_notes",
                entity_id=str(note_id),
            )
        if existing["version"] != expected_version:
            raise ConflictError(  # noqa: TRY003
                f"Note '{note_id}' version mismatch (db={existing['version']}, expected={expected_version}).",
                entity="workspace_notes",
                entity_id=str(note_id),
            )
        now = self._ctx.utc_now_iso()
        set_clauses = ["last_modified = ?", "version = ?"]
        params: list[Any] = [now, expected_version + 1]
        for col in ("title", "content", "keywords_json"):
            if col in updates:
                set_clauses.append(f"{col} = ?")
                params.append(updates[col])
        query = f"UPDATE workspace_notes SET {', '.join(set_clauses)} WHERE workspace_id = ? AND id = ? AND version = ?"  # nosec B608
        params.extend([workspace_id, note_id, expected_version])
        with self._ctx.transaction() as conn:
            cursor = conn.execute(query, tuple(params))
            if cursor.rowcount == 0:
                raise ConflictError(  # noqa: TRY003
                    f"Note '{note_id}' concurrent update detected.",
                    entity="workspace_notes",
                    entity_id=str(note_id),
                )
        return self._get_workspace_note(workspace_id, note_id)  # type: ignore[return-value]

    def delete_workspace_note(self, workspace_id: str, note_id: int) -> None:
        """Soft-delete a workspace note."""
        now = self._ctx.utc_now_iso()
        with self._ctx.transaction() as conn:
            conn.execute(
                "UPDATE workspace_notes SET deleted = 1, last_modified = ? WHERE workspace_id = ? AND id = ?",
                (now, workspace_id, note_id),
            )

    def sync_note_source_folders(
        self,
        note_id: str,
        source_id: int,
        folder_paths: list[str] | tuple[str, ...],
    ) -> list[dict[str, Any]]:
        if int(source_id) <= 0:
            raise InputError("source_id must be a positive integer.")  # noqa: TRY003

        desired_paths = self._ctx.db.expand_note_folder_paths(list(folder_paths or []))
        folder_scope = ""
        folder_params: tuple[str, ...] = ()
        if self._ctx.backend_type == BackendType.POSTGRESQL:
            folder_scope = " AND EXISTS (SELECT 1 FROM note_folders owner_folder WHERE owner_folder.id = note_folder_source_memberships.folder_id AND owner_folder.client_id = ?)"
            folder_params = (self._ctx.client_id,)
        key_scope = ""
        if folder_params:
            key_scope = " AND EXISTS (SELECT 1 FROM note_folders owner_folder WHERE owner_folder.id = note_folder_source_keys.folder_id AND owner_folder.client_id = ?)"
        try:
            with self._ctx.transaction() as conn:
                self._ctx.db.ensure_note_exists_for_folder_sync(conn, note_id)
                desired_pairs: list[tuple[str, int]] = []
                for folder_path in desired_paths:
                    folder_id = self._ctx.db.ensure_note_folder_path_locked(conn, folder_path)
                    desired_pairs.append((folder_path, folder_id))

                existing_rows = conn.execute(
                    f"""
                    SELECT folder_id
                      FROM note_folder_source_memberships
                     WHERE note_id = ? AND source_id = ?{folder_scope}
                    """,  # nosec B608 - fixed owner SQL fragments; values stay bound.
                    (note_id, int(source_id)) + folder_params,
                ).fetchall()
                existing_ids: set[int] = set()
                for row in existing_rows:
                    record = self._ctx.db.coerce_mapping_row(row) or {}
                    existing_folder_id = record.get("folder_id")
                    if existing_folder_id is None:
                        continue
                    existing_ids.add(int(existing_folder_id))
                desired_ids = {folder_id for _, folder_id in desired_pairs}
                desired_keys = {
                    canonical_key
                    for folder_path, _ in desired_pairs
                    for canonical_key in [self._ctx.db.note_folder_path_key(folder_path)]
                    if canonical_key is not None
                }

                for folder_id in existing_ids - desired_ids:
                    conn.execute(
                        """
                        DELETE FROM note_folder_source_memberships
                         WHERE note_id = ? AND source_id = ? AND folder_id = ?
                        """,
                        (note_id, int(source_id), folder_id),
                    )

                existing_key_rows = conn.execute(
                    f"""
                    SELECT folder_key
                      FROM note_folder_source_keys
                     WHERE source_id = ?{key_scope}
                    """,  # nosec B608 - fixed owner SQL fragments; values stay bound.
                    (int(source_id),) + folder_params,
                ).fetchall()
                existing_keys: set[str] = set()
                for row in existing_key_rows:
                    record = self._ctx.db.coerce_mapping_row(row) or {}
                    folder_key = record.get("folder_key")
                    if folder_key is None:
                        continue
                    existing_keys.add(str(folder_key))
                for folder_key in existing_keys - desired_keys:
                    conn.execute(
                        f"""
                        DELETE FROM note_folder_source_keys
                         WHERE source_id = ? AND folder_key = ?{key_scope}
                        """,  # nosec B608 - fixed owner SQL fragments; values stay bound.
                        (int(source_id), folder_key) + folder_params,
                    )

                now = self._ctx.utc_now_iso()
                for folder_path, folder_id in desired_pairs:
                    folder_key = self._ctx.db.note_folder_path_key(folder_path)
                    if not folder_key:
                        continue
                    conn.execute(
                        f"""
                        DELETE FROM note_folder_source_keys
                         WHERE source_id = ? AND folder_key = ?{key_scope}
                        """,  # nosec B608 - fixed owner SQL fragments; values stay bound.
                        (int(source_id), folder_key) + folder_params,
                    )
                    conn.execute(
                        """
                        INSERT INTO note_folder_source_keys(source_id, folder_key, folder_id, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (int(source_id), folder_key, folder_id, now),
                    )
                    if folder_id in existing_ids:
                        continue
                    conn.execute(
                        """
                        INSERT INTO note_folder_source_memberships(note_id, source_id, folder_id, created_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (note_id, int(source_id), folder_id, now),
                    )
        except CharactersRAGDBError:
            raise
        except (sqlite3.Error, BackendDatabaseError) as exc:
            raise CharactersRAGDBError(f"Failed syncing source-managed note folders: {exc}") from exc  # noqa: TRY003
        return self._ctx.db.get_note_folders_for_note(note_id)

    # ==========================
    # Source Review Plans
    # ==========================
    @staticmethod
    def _source_review_timestamp_value(value: Any) -> str:
        """Serialize computed due timestamps consistently for both backends."""
        if isinstance(value, datetime):
            timestamp: datetime = value
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return str(value)

    def create_source_review_plan(
        self,
        *,
        title: str,
        starts_on: str,
        timezone_name: str,
        source_bundle_json: dict[str, Any],
        schedule: list[dict[str, Any]],
    ) -> int:
        """Create a source review plan and all of its occurrences atomically."""
        source_bundle = self._ctx.db.ensure_json_string_from_mixed(source_bundle_json)
        if source_bundle is None:
            raise InputError("Source review source_bundle_json is required")  # noqa: TRY003

        now = self._ctx.utc_now_iso()
        try:
            with self._ctx.transaction() as conn:
                insert_plan_sql = (
                    "INSERT INTO source_review_plans("
                    "title, starts_on, timezone, source_bundle_json, created_at, last_modified, "
                    "deleted, client_id, version"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)"
                )
                plan_params = (
                    title,
                    starts_on,
                    timezone_name,
                    source_bundle,
                    now,
                    now,
                    False,
                    self._ctx.client_id,
                    1,
                )
                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    cursor = conn.execute(insert_plan_sql + " RETURNING id", plan_params)
                    row = cursor.fetchone()
                    plan_id = int(row["id"]) if row else None
                else:
                    cursor = conn.execute(insert_plan_sql, plan_params)
                    plan_id = int(cursor.lastrowid) if cursor.lastrowid is not None else None
                if plan_id is None:
                    raise CharactersRAGDBError("Failed to determine source review plan ID after insert")  # noqa: TRY003

                insert_occurrence_sql = """
                    INSERT INTO source_review_occurrences(
                      plan_id, offset_value, offset_unit, activity_type, due_at, status,
                      launch_state_json, started_at, completed_at, completion_source,
                      created_at, last_modified, deleted, client_id, version
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                for schedule_row in schedule:
                    conn.execute(
                        insert_occurrence_sql,
                        (
                            plan_id,
                            int(schedule_row["offset_value"]),
                            schedule_row["offset_unit"],
                            schedule_row["activity_type"],
                            self._source_review_timestamp_value(schedule_row["due_at"]),
                            "pending",
                            None,
                            None,
                            None,
                            None,
                            now,
                            now,
                            False,
                            self._ctx.client_id,
                            1,
                        ),
                    )
                return plan_id
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to create source review plan: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to create source review plan: {exc}") from exc  # noqa: TRY003

    def _source_review_occurrences_by_plan(
        self,
        plan_ids: list[int],
    ) -> dict[int, list[dict[str, Any]]]:
        """Load active occurrence summaries for source review plan responses."""
        occurrences: dict[int, list[dict[str, Any]]] = {plan_id: [] for plan_id in plan_ids}
        if not plan_ids:
            return occurrences

        placeholders = ", ".join("?" for _ in plan_ids)
        rows = self._ctx.execute_query(
            f"""
            SELECT id, plan_id, offset_value, offset_unit, activity_type, due_at,
                   status, launch_state_json, started_at, completed_at,
                   completion_source, created_at, last_modified, deleted,
                   client_id, version
              FROM source_review_occurrences
             WHERE deleted = ? AND plan_id IN ({placeholders})
             ORDER BY plan_id ASC, due_at ASC, id ASC
            """,  # nosec B608 - placeholders are generated from an internal integer list.
            (False, *plan_ids),
        ).fetchall()
        for row in rows:
            item = self._ctx.deserialize_row_fields(row, ["launch_state_json"])
            if item is not None:
                occurrences[int(item["plan_id"])].append(item)
        return occurrences

    def get_source_review_plan(self, plan_id: int) -> dict[str, Any] | None:
        """Return one active source review plan with occurrence summaries."""
        row = self._ctx.execute_query(
            """
            SELECT id, title, starts_on, timezone, source_bundle_json,
                   created_at, last_modified, deleted, client_id, version
              FROM source_review_plans
             WHERE id = ? AND deleted = ?
            """,
            (plan_id, False),
        ).fetchone()
        plan = self._ctx.deserialize_row_fields(row, ["source_bundle_json"])
        if plan is None:
            return None
        plan["occurrences"] = self._source_review_occurrences_by_plan([int(plan["id"])])[int(plan["id"])]
        return plan

    def list_source_review_plans(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List active source review plans in stable newest-first order."""
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        count_row = self._ctx.execute_query(
            "SELECT COUNT(*) AS total FROM source_review_plans WHERE deleted = ?",
            (False,),
        ).fetchone()
        total = int(count_row["total"]) if count_row else 0
        rows = self._ctx.execute_query(
            """
            SELECT id, title, starts_on, timezone, source_bundle_json,
                   created_at, last_modified, deleted, client_id, version
              FROM source_review_plans
             WHERE deleted = ?
             ORDER BY created_at DESC, id DESC
             LIMIT ? OFFSET ?
            """,
            (False, limit, offset),
        ).fetchall()
        plans = [
            item
            for row in rows
            if (item := self._ctx.deserialize_row_fields(row, ["source_bundle_json"])) is not None
        ]
        occurrences = self._source_review_occurrences_by_plan([int(plan["id"]) for plan in plans])
        for plan in plans:
            plan["occurrences"] = occurrences[int(plan["id"])]
        return plans, total

    def list_due_source_review_occurrences(
        self,
        *,
        now_utc: str,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List due pending or in-progress occurrences with their active plan snapshot."""
        limit = max(0, int(limit))
        offset = max(0, int(offset))
        where_params = (False, False, now_utc)
        count_row = self._ctx.execute_query(
            """
            SELECT COUNT(*) AS total
              FROM source_review_occurrences o
              JOIN source_review_plans p ON p.id = o.plan_id
             WHERE o.deleted = ? AND p.deleted = ?
               AND o.status IN ('pending', 'in_progress')
               AND o.due_at <= ?
            """,
            where_params,
        ).fetchone()
        total = int(count_row["total"]) if count_row else 0
        rows = self._ctx.execute_query(
            """
            SELECT o.id, o.plan_id, o.offset_value, o.offset_unit, o.activity_type,
                   o.due_at, o.status, o.launch_state_json, o.started_at,
                   o.completed_at, o.completion_source, o.created_at, o.last_modified,
                   o.deleted, o.client_id, o.version,
                   p.title AS plan_title, p.starts_on, p.timezone, p.source_bundle_json
              FROM source_review_occurrences o
              JOIN source_review_plans p ON p.id = o.plan_id
             WHERE o.deleted = ? AND p.deleted = ?
               AND o.status IN ('pending', 'in_progress')
               AND o.due_at <= ?
             ORDER BY o.due_at ASC, o.id ASC
             LIMIT ? OFFSET ?
            """,
            (*where_params, limit, offset),
        ).fetchall()
        return (
            [
                item
                for row in rows
                if (
                    item := self._ctx.deserialize_row_fields(
                        row,
                        ["launch_state_json", "source_bundle_json"],
                    )
                )
                is not None
            ],
            total,
        )

    def _get_source_review_occurrence_row(
        self,
        conn: Any,
        occurrence_id: int,
        *,
        for_update: bool = False,
    ) -> dict[str, Any] | None:
        """Load one active occurrence and its plan snapshot within a transaction."""
        query = """
            SELECT o.id, o.plan_id, o.offset_value, o.offset_unit, o.activity_type,
                   o.due_at, o.status, o.launch_state_json, o.started_at,
                   o.completed_at, o.completion_source, o.created_at, o.last_modified,
                   o.deleted, o.client_id, o.version,
                   p.title AS plan_title, p.starts_on, p.timezone, p.source_bundle_json
              FROM source_review_occurrences o
              JOIN source_review_plans p ON p.id = o.plan_id
             WHERE o.id = ? AND o.deleted = ? AND p.deleted = ?
        """
        if for_update and self._ctx.backend_type == BackendType.POSTGRESQL:
            query += " FOR UPDATE OF o"
        row = conn.execute(query, (occurrence_id, False, False)).fetchone()
        return self._ctx.deserialize_row_fields(row, ["launch_state_json", "source_bundle_json"])

    def _require_source_review_occurrence(
        self,
        conn: Any,
        occurrence_id: int,
    ) -> dict[str, Any]:
        """Lock and return an occurrence or raise the domain not-found error."""
        occurrence = self._get_source_review_occurrence_row(conn, occurrence_id, for_update=True)
        if occurrence is None:
            raise NotFoundError(f"Source review occurrence {occurrence_id} not found")
        return occurrence

    def start_source_review_occurrence(self, occurrence_id: int) -> dict[str, Any]:
        """Start a pending occurrence or idempotently return an active/completed one."""
        try:
            with self._ctx.transaction() as conn:
                occurrence = self._require_source_review_occurrence(conn, occurrence_id)
                status = str(occurrence["status"])
                if status in {"in_progress", "completed"}:
                    return occurrence
                if status == "skipped":
                    raise ConflictError(
                        "Skipped source review occurrence cannot be started",
                        entity="source_review_occurrences",
                        identifier=occurrence_id,
                    )

                now = self._ctx.utc_now_iso()
                launch_state = build_source_review_launch_metadata(
                    activity_type=occurrence["activity_type"],
                    plan_id=int(occurrence["plan_id"]),
                    occurrence_id=occurrence_id,
                    created_at=now,
                )
                cursor = conn.execute(
                    """
                    UPDATE source_review_occurrences
                       SET status = 'in_progress', launch_state_json = ?, started_at = ?,
                           last_modified = ?, client_id = ?, version = version + 1
                     WHERE id = ? AND deleted = ? AND status = 'pending'
                    """,
                    (json.dumps(launch_state), now, now, self._ctx.client_id, occurrence_id, False),
                )
                if cursor.rowcount != 1:
                    raise ConflictError(
                        "Concurrent update starting source review occurrence",
                        entity="source_review_occurrences",
                        identifier=occurrence_id,
                    )
                started = self._get_source_review_occurrence_row(conn, occurrence_id)
                if started is None:
                    raise NotFoundError(
                        f"Source review occurrence {occurrence_id} not found"
                    )
                return started
        except ValueError as exc:
            raise CharactersRAGDBError(
                "Stored source review launch metadata is invalid"
            ) from exc
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to start source review occurrence: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to start source review occurrence: {exc}") from exc  # noqa: TRY003

    def complete_source_review_occurrence(
        self,
        occurrence_id: int,
        *,
        completion_source: str = "manual",
    ) -> dict[str, Any]:
        """Complete an in-progress occurrence idempotently."""
        try:
            with self._ctx.transaction() as conn:
                occurrence = self._require_source_review_occurrence(conn, occurrence_id)
                status = str(occurrence["status"])
                if status == "completed":
                    return occurrence
                if status == "pending":
                    message = "Source review occurrence must be started before completion"
                elif status == "skipped":
                    message = "Skipped source review occurrence cannot be completed"
                else:
                    message = ""
                if message:
                    raise ConflictError(
                        message,
                        entity="source_review_occurrences",
                        identifier=occurrence_id,
                    )

                now = self._ctx.utc_now_iso()
                cursor = conn.execute(
                    """
                    UPDATE source_review_occurrences
                       SET status = 'completed', completed_at = ?, completion_source = ?,
                           last_modified = ?, client_id = ?, version = version + 1
                     WHERE id = ? AND deleted = ? AND status = 'in_progress'
                    """,
                    (now, completion_source, now, self._ctx.client_id, occurrence_id, False),
                )
                if cursor.rowcount != 1:
                    raise ConflictError(
                        "Concurrent update completing source review occurrence",
                        entity="source_review_occurrences",
                        identifier=occurrence_id,
                    )
                completed = self._get_source_review_occurrence_row(conn, occurrence_id)
                if completed is None:
                    raise NotFoundError(
                        f"Source review occurrence {occurrence_id} not found"
                    )
                return completed
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to complete source review occurrence: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to complete source review occurrence: {exc}") from exc  # noqa: TRY003

    def skip_source_review_occurrence(self, occurrence_id: int) -> dict[str, Any]:
        """Skip a pending/in-progress occurrence or return an already skipped one."""
        try:
            with self._ctx.transaction() as conn:
                occurrence = self._require_source_review_occurrence(conn, occurrence_id)
                status = str(occurrence["status"])
                if status == "skipped":
                    return occurrence
                if status == "completed":
                    raise ConflictError(
                        "Completed source review occurrence cannot be skipped",
                        entity="source_review_occurrences",
                        identifier=occurrence_id,
                    )

                now = self._ctx.utc_now_iso()
                cursor = conn.execute(
                    """
                    UPDATE source_review_occurrences
                       SET status = 'skipped', last_modified = ?, client_id = ?, version = version + 1
                     WHERE id = ? AND deleted = ? AND status IN ('pending', 'in_progress')
                    """,
                    (now, self._ctx.client_id, occurrence_id, False),
                )
                if cursor.rowcount != 1:
                    raise ConflictError(
                        "Concurrent update skipping source review occurrence",
                        entity="source_review_occurrences",
                        identifier=occurrence_id,
                    )
                skipped = self._get_source_review_occurrence_row(conn, occurrence_id)
                if skipped is None:
                    raise NotFoundError(
                        f"Source review occurrence {occurrence_id} not found"
                    )
                return skipped
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to skip source review occurrence: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to skip source review occurrence: {exc}") from exc  # noqa: TRY003

    def soft_delete_source_review_plan(self, plan_id: int) -> bool:
        """Soft-delete a plan and its active occurrences in one transaction."""
        try:
            with self._ctx.transaction() as conn:
                select_query = "SELECT deleted FROM source_review_plans WHERE id = ?"
                if self._ctx.backend_type == BackendType.POSTGRESQL:
                    select_query += " FOR UPDATE"
                plan = conn.execute(select_query, (plan_id,)).fetchone()
                if plan is None:
                    raise NotFoundError(f"Source review plan {plan_id} not found")
                if bool(plan["deleted"]):
                    return False

                now = self._ctx.utc_now_iso()
                conn.execute(
                    """
                    UPDATE source_review_plans
                       SET deleted = ?, last_modified = ?, client_id = ?, version = version + 1
                     WHERE id = ? AND deleted = ?
                    """,
                    (True, now, self._ctx.client_id, plan_id, False),
                )
                conn.execute(
                    """
                    UPDATE source_review_occurrences
                       SET deleted = ?, last_modified = ?, client_id = ?, version = version + 1
                     WHERE plan_id = ? AND deleted = ?
                    """,
                    (True, now, self._ctx.client_id, plan_id, False),
                )
                return True
        except sqlite3.Error as exc:
            raise CharactersRAGDBError(f"Failed to delete source review plan: {exc}") from exc  # noqa: TRY003
        except BackendDatabaseError as exc:
            raise CharactersRAGDBError(f"Failed to delete source review plan: {exc}") from exc  # noqa: TRY003
