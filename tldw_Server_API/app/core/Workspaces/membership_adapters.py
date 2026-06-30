"""Workspace resource membership adapter registry and pilot adapters."""
from __future__ import annotations

import inspect
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError as MediaDatabaseError
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    ConflictError as PromptsConflictError,
    DatabaseError as PromptsDatabaseError,
    InputError as PromptsInputError,
    SchemaError as PromptsSchemaError,
)
from tldw_Server_API.app.core.Workspaces.membership_models import WorkspaceResourceRef
from tldw_Server_API.app.core.exceptions import WorkspaceMembershipAdapterError

_LOOKUP_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    sqlite3.Error,
    BackendDatabaseError,
)
_MEDIA_LOOKUP_EXCEPTIONS = _LOOKUP_EXCEPTIONS + (MediaDatabaseError,)
_PROMPT_LOOKUP_EXCEPTIONS = _LOOKUP_EXCEPTIONS + (
    PromptsConflictError,
    PromptsDatabaseError,
    PromptsInputError,
    PromptsSchemaError,
)
_RUNTIME_BINDING_LOOKUP_EXCEPTIONS = _LOOKUP_EXCEPTIONS + (CharactersRAGDBError,)


@dataclass(frozen=True)
class WorkspaceMembershipContext:
    """Context shared with resource adapters during membership operations."""

    workspace_id: str
    user_id: str | None
    chacha_db: Any
    media_db: Any | None = None
    prompts_db: Any | None = None
    workflows_db: Any | None = None
    watchlists_db: Any | None = None
    request_metadata: Mapping[str, Any] = field(default_factory=dict)


class WorkspaceMembershipAdapter(Protocol):
    resource_type: str

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        """Validate access to a resource and return its canonical reference."""

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        """Return a display summary for a resource."""

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        """Reserved hook for future transition-aware link side effects.

        The first-slice service does not invoke this until membership writes can
        report insert/restore transitions or use an outbox.
        """

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        """Optional hook after a membership row is soft-deleted."""


class WorkspaceNoteMembershipAdapter:
    resource_type = "workspace_note"

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        note_id = _parse_int_resource_id(resource_id, resource_type=self.resource_type)
        row = context.chacha_db.get_workspace_note(context.workspace_id, note_id)
        if not row or _is_deleted(row) or _row_workspace_mismatch(row, context.workspace_id):
            raise _resource_not_found(self.resource_type, str(note_id))
        return self._ref(row, note_id)

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        note_id = _parse_int_resource_id(resource_id, resource_type=self.resource_type)
        row = _call_with_optional_include_deleted(
            context.chacha_db.get_workspace_note,
            context.workspace_id,
            note_id,
            include_deleted=True,
        )
        if not row or _row_workspace_mismatch(row, context.workspace_id):
            raise _resource_not_found(self.resource_type, str(note_id))
        return self._ref(row, note_id, state=_resource_state(row))

    def _ref(self, row: Mapping[str, Any], note_id: int, *, state: str = "available") -> WorkspaceResourceRef:
        title = _first_non_empty(row, "title") or f"Workspace note {note_id}"
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=str(row.get("id") or note_id),
            title=title,
            updated_at=_first_non_empty(row, "last_modified", "updated_at", "created_at"),
            state=state,
        )

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
        return self._ref(row, canonical_id)

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        canonical_id = _require_resource_id(resource_id)
        row = _call_with_optional_include_deleted(
            context.chacha_db.get_workspace_source,
            context.workspace_id,
            canonical_id,
            include_deleted=True,
        )
        if not row or _row_workspace_mismatch(row, context.workspace_id):
            raise _resource_not_found(self.resource_type, canonical_id)
        return self._ref(row, canonical_id, state=_resource_state(row))

    def _ref(self, row: Mapping[str, Any], canonical_id: str, *, state: str = "available") -> WorkspaceResourceRef:
        source_type = _first_non_empty(row, "source_type")
        metadata = _compact_metadata(row, "source_type", "media_id")
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=str(row.get("id") or canonical_id),
            title=_first_non_empty(row, "title") or f"Workspace source {canonical_id}",
            subtitle=source_type,
            updated_at=_first_non_empty(row, "last_modified", "updated_at", "added_at", "created_at"),
            metadata=metadata,
            state=state,
        )

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
        return self._ref(row, canonical_id)

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        canonical_id = _require_resource_id(resource_id)
        row = _call_with_optional_include_deleted(
            context.chacha_db.get_workspace_artifact,
            context.workspace_id,
            canonical_id,
            include_deleted=True,
        )
        if not row or _row_workspace_mismatch(row, context.workspace_id):
            raise _resource_not_found(self.resource_type, canonical_id)
        return self._ref(row, canonical_id, state=_resource_state(row))

    def _ref(self, row: Mapping[str, Any], canonical_id: str, *, state: str = "available") -> WorkspaceResourceRef:
        artifact_type = _first_non_empty(row, "artifact_type")
        metadata = _compact_metadata(row, "artifact_type", "review_state", "status")
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=str(row.get("id") or canonical_id),
            title=_first_non_empty(row, "title") or f"Workspace artifact {canonical_id}",
            subtitle=artifact_type,
            updated_at=_first_non_empty(row, "last_modified", "updated_at", "completed_at", "created_at"),
            metadata=metadata,
            state=state,
        )

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
        except _MEDIA_LOOKUP_EXCEPTIONS as exc:  # pragma: no cover - defensive adapter boundary.
            raise WorkspaceMembershipAdapterError(
                "media_lookup_failed",
                "Media lookup failed while validating workspace membership.",
                status_code=503,
            ) from exc
        if not row:
            raise _resource_not_found(self.resource_type, str(media_id))
        return self._ref(row, media_id)

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        media_id = _parse_int_resource_id(resource_id, resource_type=self.resource_type)
        if context.media_db is None:
            raise WorkspaceMembershipAdapterError(
                "media_db_unavailable",
                "Media DB is required to summarize media workspace membership.",
                status_code=503,
            )
        try:
            row = media_db_api.get_media_by_id(
                context.media_db,
                media_id,
                include_deleted=True,
                include_trash=True,
            )
        except _MEDIA_LOOKUP_EXCEPTIONS as exc:  # pragma: no cover - defensive adapter boundary.
            raise WorkspaceMembershipAdapterError(
                "media_lookup_failed",
                "Media lookup failed while summarizing workspace membership.",
                status_code=503,
            ) from exc
        if not row:
            raise _resource_not_found(self.resource_type, str(media_id))
        return self._ref(row, media_id, state=_resource_state(row))

    def _ref(self, row: Mapping[str, Any], media_id: int, *, state: str = "available") -> WorkspaceResourceRef:
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
            state=state,
        )

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
        return self._ref(row, canonical_id, context)

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        canonical_id = _require_resource_id(resource_id)
        row = _call_with_optional_include_deleted(
            context.chacha_db.get_conversation_for_workspace_membership,
            canonical_id,
            include_deleted=True,
        )
        if not row:
            raise _resource_not_found(self.resource_type, canonical_id)
        return self._ref(row, canonical_id, context, state=_resource_state(row))

    def _ref(
        self,
        row: Mapping[str, Any],
        canonical_id: str,
        context: WorkspaceMembershipContext,
        *,
        state: str = "available",
    ) -> WorkspaceResourceRef:
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
            state=state,
        )

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None


class PromptMembershipAdapter:
    resource_type = "prompt"

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        row = self._get_prompt(resource_id, context, include_deleted=False)
        if not row or _is_deleted(row):
            raise _resource_not_found(self.resource_type, _require_resource_id(resource_id))
        return self._ref(row)

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        row = self._get_prompt(resource_id, context, include_deleted=True)
        if not row:
            raise _resource_not_found(self.resource_type, _require_resource_id(resource_id))
        return self._ref(row, state=_resource_state(row))

    def _get_prompt(
        self,
        resource_id: str,
        context: WorkspaceMembershipContext,
        *,
        include_deleted: bool,
    ) -> Mapping[str, Any] | None:
        if context.prompts_db is None:
            raise WorkspaceMembershipAdapterError(
                "prompts_db_unavailable",
                "Prompts DB is required to validate prompt workspace membership.",
                status_code=503,
            )
        try:
            row = context.prompts_db.fetch_prompt_details(resource_id, include_deleted=include_deleted)
        except _PROMPT_LOOKUP_EXCEPTIONS as exc:  # pragma: no cover - defensive adapter boundary.
            raise WorkspaceMembershipAdapterError(
                "prompt_lookup_failed",
                "Prompt lookup failed while validating workspace membership.",
                status_code=503,
            ) from exc
        return _as_mapping(row)

    def _ref(self, row: Mapping[str, Any], *, state: str = "available") -> WorkspaceResourceRef:
        prompt_id = str(row.get("id") or "")
        if not prompt_id:
            raise _resource_not_found(self.resource_type, "unknown")
        metadata = _compact_metadata(row, "author", "prompt_format", "prompt_schema_version")
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=prompt_id,
            title=_first_non_empty(row, "name") or f"Prompt {prompt_id}",
            subtitle=_first_non_empty(row, "prompt_format") or "prompt",
            updated_at=_first_non_empty(row, "last_modified", "updated_at", "created_at"),
            metadata=metadata,
            state=state,
        )

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None


class WorkflowMembershipAdapter:
    resource_type = "workflow"

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        workflow_id = _parse_int_resource_id(resource_id, resource_type=self.resource_type)
        row = self._get_workflow(workflow_id, context)
        if not row or not self._is_visible(row, context) or not _workflow_is_active(row):
            raise _resource_not_found(self.resource_type, str(workflow_id))
        return self._ref(row, workflow_id)

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        workflow_id = _parse_int_resource_id(resource_id, resource_type=self.resource_type)
        row = self._get_workflow(workflow_id, context)
        if not row or not self._is_visible(row, context):
            raise _resource_not_found(self.resource_type, str(workflow_id))
        state = "available" if _workflow_is_active(row) else "archived"
        return self._ref(row, workflow_id, state=state)

    def _get_workflow(self, workflow_id: int, context: WorkspaceMembershipContext) -> Mapping[str, Any] | None:
        if context.workflows_db is None:
            raise WorkspaceMembershipAdapterError(
                "workflows_db_unavailable",
                "Workflows DB is required to validate workflow workspace membership.",
                status_code=503,
            )
        try:
            return _as_mapping(context.workflows_db.get_definition(workflow_id))
        except _LOOKUP_EXCEPTIONS as exc:  # pragma: no cover - defensive adapter boundary.
            raise WorkspaceMembershipAdapterError(
                "workflow_lookup_failed",
                "Workflow lookup failed while validating workspace membership.",
                status_code=503,
            ) from exc

    def _is_visible(self, row: Mapping[str, Any], context: WorkspaceMembershipContext) -> bool:
        tenant_id = str(context.request_metadata.get("tenant_id") or "").strip()
        if tenant_id and str(row.get("tenant_id") or "") != tenant_id:
            return False
        if context.request_metadata.get("is_workflows_admin") is True:
            return True
        return bool(context.user_id) and str(row.get("owner_id") or "") == str(context.user_id)

    def _ref(
        self,
        row: Mapping[str, Any],
        workflow_id: int,
        *,
        state: str = "available",
    ) -> WorkspaceResourceRef:
        metadata = {
            "version": row.get("version"),
            "tags": _metadata_tags(row.get("tags")),
        }
        metadata = {key: value for key, value in metadata.items() if value not in (None, [], "")}
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=str(row.get("id") or workflow_id),
            title=_first_non_empty(row, "name") or f"Workflow {workflow_id}",
            subtitle="workflow",
            updated_at=_first_non_empty(row, "updated_at", "created_at"),
            metadata=metadata,
            state=state,
        )

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None


class WatchlistMembershipAdapter:
    resource_type = "watchlist"

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        watchlist_id = _parse_int_resource_id(resource_id, resource_type=self.resource_type)
        row = self._get_watchlist(watchlist_id, context, include_deleted=False)
        if not row or _is_deleted(row):
            raise _resource_not_found(self.resource_type, str(watchlist_id))
        return self._ref(row, watchlist_id)

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        watchlist_id = _parse_int_resource_id(resource_id, resource_type=self.resource_type)
        row = self._get_watchlist(watchlist_id, context, include_deleted=True)
        if not row:
            raise _resource_not_found(self.resource_type, str(watchlist_id))
        return self._ref(row, watchlist_id, state=_resource_state(row))

    def _get_watchlist(
        self,
        watchlist_id: int,
        context: WorkspaceMembershipContext,
        *,
        include_deleted: bool,
    ) -> Mapping[str, Any] | None:
        if context.watchlists_db is None:
            raise WorkspaceMembershipAdapterError(
                "watchlists_db_unavailable",
                "Watchlists DB is required to validate watchlist workspace membership.",
                status_code=503,
            )
        try:
            return _as_mapping(context.watchlists_db.get_watchlist(watchlist_id, include_deleted=include_deleted))
        except KeyError:
            return None
        except _LOOKUP_EXCEPTIONS as exc:  # pragma: no cover - defensive adapter boundary.
            raise WorkspaceMembershipAdapterError(
                "watchlist_lookup_failed",
                "Watchlist lookup failed while validating workspace membership.",
                status_code=503,
            ) from exc

    def _ref(
        self,
        row: Mapping[str, Any],
        watchlist_id: int,
        *,
        state: str = "available",
    ) -> WorkspaceResourceRef:
        metadata = _compact_metadata(row, "domain", "status", "priority")
        tags = _metadata_tags(row.get("tags", row.get("tags_json")))
        if tags:
            metadata["tags"] = tags
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=str(row.get("id") or watchlist_id),
            title=_first_non_empty(row, "name") or f"Watchlist {watchlist_id}",
            subtitle=_first_non_empty(row, "domain") or "watchlist",
            updated_at=_first_non_empty(row, "updated_at", "created_at"),
            metadata=metadata,
            state=state,
        )

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None


@dataclass(frozen=True)
class _RuntimeBindingSessionResolver:
    resource_type: str
    binding_kind: str
    owner_domain: str

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        binding_id = _require_resource_id(resource_id)
        row = self._get_binding(binding_id, context, include_deleted=False)
        if not row or self._binding_mismatch(row) or _is_deleted(row) or row.get("status") == "archived":
            raise _resource_not_found(self.resource_type, binding_id)
        return self._ref(row, binding_id)

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        binding_id = _require_resource_id(resource_id)
        row = self._get_binding(binding_id, context, include_deleted=True)
        if not row or self._binding_mismatch(row):
            raise _resource_not_found(self.resource_type, binding_id)
        state = "archived" if _is_deleted(row) or row.get("status") == "archived" else "available"
        return self._ref(row, binding_id, state=state)

    def _get_binding(
        self,
        binding_id: str,
        context: WorkspaceMembershipContext,
        *,
        include_deleted: bool,
    ) -> Mapping[str, Any] | None:
        try:
            row = context.chacha_db.get_workspace_runtime_binding(
                context.workspace_id,
                binding_id,
                include_deleted=include_deleted,
            )
        except _RUNTIME_BINDING_LOOKUP_EXCEPTIONS as exc:  # pragma: no cover - defensive adapter boundary.
            raise WorkspaceMembershipAdapterError(
                "runtime_binding_lookup_failed",
                "Runtime binding lookup failed while validating workspace membership.",
                status_code=503,
            ) from exc
        return _as_mapping(row)

    def _binding_mismatch(self, row: Mapping[str, Any]) -> bool:
        return row.get("binding_kind") != self.binding_kind or row.get("owner_domain") != self.owner_domain

    def _ref(
        self,
        row: Mapping[str, Any],
        binding_id: str,
        *,
        state: str = "available",
    ) -> WorkspaceResourceRef:
        metadata = _compact_metadata(
            row,
            "binding_kind",
            "owner_domain",
            "status",
            "portability",
            "path_hint",
            "redaction_report",
        )
        descriptor_metadata = row.get("metadata")
        if isinstance(descriptor_metadata, Mapping):
            metadata["descriptor"] = dict(descriptor_metadata)
        return WorkspaceResourceRef(
            resource_type=self.resource_type,
            resource_id=str(row.get("binding_id") or binding_id),
            title=_first_non_empty(row, "label") or f"{self.resource_type} {binding_id}",
            subtitle=self.binding_kind,
            updated_at=_first_non_empty(row, "updated_at", "created_at"),
            metadata=metadata,
            state=state,
        )


class AcpSessionMembershipAdapter:
    resource_type = "acp_session"

    def __init__(self, resolver: _RuntimeBindingSessionResolver | None = None) -> None:
        self._resolver = resolver or _RuntimeBindingSessionResolver(
            resource_type=self.resource_type,
            binding_kind="acp_session",
            owner_domain="acp",
        )

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        return self._resolver.validate_access(resource_id, context)

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        return self._resolver.summarize(resource_id, context)

    def on_link(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None

    def on_unlink(self, membership: Mapping[str, Any], context: WorkspaceMembershipContext) -> None:
        return None


class SandboxSessionMembershipAdapter:
    resource_type = "sandbox_session"

    def __init__(self, resolver: _RuntimeBindingSessionResolver | None = None) -> None:
        self._resolver = resolver or _RuntimeBindingSessionResolver(
            resource_type=self.resource_type,
            binding_kind="sandbox_session",
            owner_domain="sandbox",
        )

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        return self._resolver.validate_access(resource_id, context)

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        return self._resolver.summarize(resource_id, context)

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
        PromptMembershipAdapter(),
        WorkflowMembershipAdapter(),
        WatchlistMembershipAdapter(),
        AcpSessionMembershipAdapter(),
        SandboxSessionMembershipAdapter(),
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
    return row.get("deleted") in (True, 1, "1", "true", "True") or bool(row.get("deleted_at"))


def _row_workspace_mismatch(row: Mapping[str, Any], workspace_id: str) -> bool:
    row_workspace_id = row.get("workspace_id")
    return row_workspace_id is not None and str(row_workspace_id) != workspace_id


def _is_archived(row: Mapping[str, Any]) -> bool:
    return row.get("archived") in (True, 1, "1", "true", "True") or bool(row.get("archived_at"))


def _is_trash(row: Mapping[str, Any]) -> bool:
    return row.get("is_trash") in (True, 1, "1", "true", "True")


def _resource_state(row: Mapping[str, Any]) -> str:
    if _is_deleted(row) or _is_trash(row):
        return "deleted"
    if _is_archived(row):
        return "archived"
    return "available"


def _call_with_optional_include_deleted(func: Any, *args: Any, include_deleted: bool) -> Any:
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        try:
            return func(*args, include_deleted=include_deleted)
        except TypeError:
            return func(*args)
    if "include_deleted" in parameters:
        return func(*args, include_deleted=include_deleted)
    return func(*args)


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


def _as_mapping(row: Any) -> Mapping[str, Any] | None:
    if row is None:
        return None
    if isinstance(row, Mapping):
        return row
    if hasattr(row, "to_dict"):
        mapped = row.to_dict()
        return mapped if isinstance(mapped, Mapping) else None
    try:
        return vars(row)
    except TypeError:
        return None


def _workflow_is_active(row: Mapping[str, Any]) -> bool:
    return row.get("is_active") not in (False, 0, "0", "false", "False")


def _metadata_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return [raw]
        value = parsed
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            normalized = str(item or "").strip()
            if normalized and normalized not in result:
                result.append(normalized)
        return result
    return []


def _media_href(media_id: str) -> str:
    return f"/media/{media_id}"
