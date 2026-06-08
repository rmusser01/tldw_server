"""Workspace resource membership adapter registry and pilot adapters."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.Workspaces.membership_models import WorkspaceResourceRef


@dataclass(frozen=True)
class WorkspaceMembershipContext:
    """Context shared with resource adapters during membership operations."""

    workspace_id: str
    user_id: str | None
    chacha_db: Any
    media_db: Any | None = None
    request_metadata: Mapping[str, Any] = field(default_factory=dict)


class WorkspaceMembershipAdapterError(Exception):
    """Fail-closed adapter error that the service maps onto API-facing errors."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 404,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = dict(details or {})


class WorkspaceMembershipAdapter(Protocol):
    resource_type: str

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        """Validate access to a resource and return its canonical reference."""

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        """Return a display summary for a resource."""

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        """Optional hook after a membership row is created or restored."""

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        """Optional hook after a membership row is soft-deleted."""


class WorkspaceNoteMembershipAdapter:
    resource_type = "workspace_note"

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        note_id = _parse_int_resource_id(resource_id, resource_type=self.resource_type)
        row = context.chacha_db.get_workspace_note(context.workspace_id, note_id)
        if not row or _is_deleted(row) or _row_workspace_mismatch(row, context.workspace_id):
            raise _resource_not_found(self.resource_type, str(note_id))
        title = _first_non_empty(row, "title") or f"Workspace note {note_id}"
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=str(row.get("id") or note_id),
            title=title,
            updated_at=_first_non_empty(row, "last_modified", "updated_at", "created_at"),
        )

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        return self.validate_access(resource_id, context)

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None


class WorkspaceSourceMembershipAdapter:
    resource_type = "workspace_source"

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        canonical_id = _require_resource_id(resource_id)
        row = context.chacha_db.get_workspace_source(context.workspace_id, canonical_id)
        if not row or _is_deleted(row) or _row_workspace_mismatch(row, context.workspace_id):
            raise _resource_not_found(self.resource_type, canonical_id)
        source_type = _first_non_empty(row, "source_type")
        metadata = _compact_metadata(row, "source_type", "media_id")
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=str(row.get("id") or canonical_id),
            title=_first_non_empty(row, "title") or f"Workspace source {canonical_id}",
            subtitle=source_type,
            updated_at=_first_non_empty(row, "last_modified", "updated_at", "added_at", "created_at"),
            metadata=metadata,
        )

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        return self.validate_access(resource_id, context)

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None


class WorkspaceArtifactMembershipAdapter:
    resource_type = "workspace_artifact"

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        canonical_id = _require_resource_id(resource_id)
        row = context.chacha_db.get_workspace_artifact(context.workspace_id, canonical_id)
        if not row or _is_deleted(row) or _row_workspace_mismatch(row, context.workspace_id):
            raise _resource_not_found(self.resource_type, canonical_id)
        artifact_type = _first_non_empty(row, "artifact_type")
        metadata = _compact_metadata(row, "artifact_type", "review_state", "status")
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=str(row.get("id") or canonical_id),
            title=_first_non_empty(row, "title") or f"Workspace artifact {canonical_id}",
            subtitle=artifact_type,
            updated_at=_first_non_empty(row, "last_modified", "updated_at", "completed_at", "created_at"),
            metadata=metadata,
        )

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        return self.validate_access(resource_id, context)

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None


class MediaMembershipAdapter:
    resource_type = "media"

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        media_id = _parse_int_resource_id(resource_id, resource_type=self.resource_type)
        if context.media_db is None:
            raise WorkspaceMembershipAdapterError(
                "media_db_unavailable",
                "Media DB is required to validate media workspace membership.",
                status_code=503,
            )
        try:
            row = media_db_api.get_media_by_id(
                context.media_db,
                media_id,
                include_deleted=False,
                include_trash=False,
            )
        except Exception as exc:  # pragma: no cover - defensive adapter boundary.
            raise WorkspaceMembershipAdapterError(
                "media_lookup_failed",
                "Media lookup failed while validating workspace membership.",
                status_code=503,
            ) from exc
        if not row:
            raise _resource_not_found(self.resource_type, str(media_id))
        canonical_id = str(row.get("id") or media_id)
        subtitle = _first_non_empty(row, "media_type", "type", "content_type")
        metadata = _compact_metadata(row, "media_type", "type", "content_type", "url")
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=canonical_id,
            title=_first_non_empty(row, "title", "name", "url") or f"Media {canonical_id}",
            subtitle=subtitle,
            href=_media_href(canonical_id),
            updated_at=_first_non_empty(row, "last_modified", "updated_at", "created_at"),
            metadata=metadata,
        )

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        return self.validate_access(resource_id, context)

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None


class ChatMembershipAdapter:
    resource_type = "chat"

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        canonical_id = _require_resource_id(resource_id)
        row = context.chacha_db.get_conversation_for_workspace_membership(canonical_id)
        if not row or _is_deleted(row):
            raise _resource_not_found(self.resource_type, canonical_id)

        scope_type = str(row.get("scope_type") or "").strip().lower()
        row_workspace_id = row.get("workspace_id")
        if scope_type == "workspace" and str(row_workspace_id or "") != context.workspace_id:
            raise _resource_not_found(self.resource_type, canonical_id)
        if scope_type not in ("", "global", "workspace"):
            raise _resource_not_found(self.resource_type, canonical_id)

        normalized_scope = scope_type or "global"
        metadata: dict[str, Any] = {"scope_type": normalized_scope}
        if row_workspace_id:
            metadata["workspace_id"] = row_workspace_id
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=str(row.get("id") or canonical_id),
            title=_first_non_empty(row, "title") or f"Chat {canonical_id}",
            subtitle="chat",
            updated_at=_first_non_empty(row, "last_modified", "updated_at", "created_at"),
            metadata=metadata,
        )

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        return self.validate_access(resource_id, context)

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None


def default_workspace_membership_adapters() -> dict[str, WorkspaceMembershipAdapter]:
    adapters: tuple[WorkspaceMembershipAdapter, ...] = (
        WorkspaceNoteMembershipAdapter(),
        MediaMembershipAdapter(),
        WorkspaceSourceMembershipAdapter(),
        WorkspaceArtifactMembershipAdapter(),
        ChatMembershipAdapter(),
    )
    return {adapter.resource_type: adapter for adapter in adapters}


def get_workspace_membership_adapter(
    resource_type: str,
    adapters: Mapping[str, WorkspaceMembershipAdapter] | None = None,
) -> WorkspaceMembershipAdapter:
    normalized = _require_resource_id(resource_type)
    registry = adapters if adapters is not None else default_workspace_membership_adapters()
    adapter = registry.get(normalized)
    if adapter is None:
        raise WorkspaceMembershipAdapterError(
            "unsupported_resource_type",
            f"Workspace resource type '{normalized}' is not supported.",
            status_code=400,
            details={"resource_type": normalized},
        )
    return adapter


def _require_resource_id(value: Any) -> str:
    if value is None:
        raise WorkspaceMembershipAdapterError(
            "invalid_resource_id",
            "Workspace membership resource_id is required.",
            status_code=400,
        )
    normalized = str(value).strip()
    if not normalized:
        raise WorkspaceMembershipAdapterError(
            "invalid_resource_id",
            "Workspace membership resource_id is required.",
            status_code=400,
        )
    return normalized


def _parse_int_resource_id(value: Any, *, resource_type: str) -> int:
    raw = _require_resource_id(value)
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise WorkspaceMembershipAdapterError(
            "invalid_resource_id",
            f"Workspace membership resource_id for '{resource_type}' must be an integer.",
            status_code=400,
        ) from exc
    if parsed < 0:
        raise WorkspaceMembershipAdapterError(
            "invalid_resource_id",
            f"Workspace membership resource_id for '{resource_type}' must be non-negative.",
            status_code=400,
        )
    return parsed


def _resource_not_found(resource_type: str, resource_id: str) -> WorkspaceMembershipAdapterError:
    return WorkspaceMembershipAdapterError(
        "resource_not_found",
        f"Workspace resource '{resource_type}:{resource_id}' was not found or is not visible.",
        status_code=404,
        details={"resource_type": resource_type, "resource_id": resource_id},
    )


def _is_deleted(row: Mapping[str, Any]) -> bool:
    return row.get("deleted") in (True, 1, "1", "true", "True")


def _row_workspace_mismatch(row: Mapping[str, Any], workspace_id: str) -> bool:
    row_workspace_id = row.get("workspace_id")
    return row_workspace_id is not None and str(row_workspace_id) != workspace_id


def _first_non_empty(row: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _compact_metadata(row: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: row[key] for key in keys if row.get(key) is not None}


def _media_href(media_id: str) -> str:
    return f"/media/{media_id}"
