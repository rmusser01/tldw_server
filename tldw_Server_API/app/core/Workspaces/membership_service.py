"""Workspace resource membership service orchestration."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.Workspaces.membership_adapters import (
    WorkspaceMembershipAdapter,
    WorkspaceMembershipAdapterError,
    WorkspaceMembershipContext,
    default_workspace_membership_adapters,
    get_workspace_membership_adapter,
)
from tldw_Server_API.app.core.Workspaces.membership_models import (
    WORKSPACE_MEMBERSHIP_ROLES,
    WORKSPACE_MEMBERSHIP_TRANSFER_POLICIES,
    WorkspaceMembershipCursor,
    WorkspaceResourceMembershipCursor,
    WorkspaceResourceRef,
    decode_membership_cursor,
    decode_resource_membership_cursor,
    encode_membership_cursor,
    encode_resource_membership_cursor,
)


_SUMMARY_UNAVAILABLE_MESSAGE = "Workspace resource summary is unavailable."


class WorkspaceMembershipServiceError(Exception):
    """Service-level Workspace membership error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 409,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = dict(details or {})


class WorkspaceMembershipService:
    """Validate and persist Workspace cross-resource memberships."""

    def __init__(
        self,
        chacha_db: Any,
        *,
        adapters: Mapping[str, WorkspaceMembershipAdapter] | None = None,
    ) -> None:
        self.chacha_db = chacha_db
        self.adapters = dict(adapters) if adapters is not None else default_workspace_membership_adapters()

    def link_membership(
        self,
        workspace_id: str,
        request: Any,
        *,
        user_id: str | None = None,
        media_db: Any | None = None,
        request_metadata: Mapping[str, Any] | None = None,
        resolve: bool = True,
    ) -> dict[str, Any]:
        """Validate a resource and create or restore its Workspace membership."""
        workspace = self._require_workspace(workspace_id)
        self._require_writable_workspace(workspace)
        data = self._membership_request_data(request)
        adapter = self._get_adapter(data["resource_type"])
        context = self._context(
            workspace_id,
            user_id=user_id,
            media_db=media_db,
            request_metadata=request_metadata,
        )
        ref = self._validate_access(adapter, data["resource_id"], context)
        existing = self.chacha_db.get_workspace_resource_membership(
            workspace_id,
            ref.resource_type,
            ref.resource_id,
            include_deleted=True,
        )
        restore_deleted = existing is not None and self._row_is_deleted(existing)
        should_call_on_link = existing is None or restore_deleted
        row = self.chacha_db.add_workspace_resource_membership(
            workspace_id,
            {
                "resource_type": ref.resource_type,
                "resource_id": ref.resource_id,
                "role": data["role"],
                "label": data.get("label"),
                "transfer_policy": data["transfer_policy"],
                "provenance": data["provenance"],
                "metadata": data["metadata"],
                "restore_deleted": restore_deleted,
            },
            user_id=user_id,
        )
        if should_call_on_link:
            adapter.on_link(row, context)
        return self._serialize_membership(row, context=context, summary_ref=ref if resolve else None, resolve=resolve)

    def get_membership(
        self,
        workspace_id: str,
        resource_type: str,
        resource_id: str,
        *,
        user_id: str | None = None,
        media_db: Any | None = None,
        resolve: bool = True,
    ) -> dict[str, Any] | None:
        """Fetch one active membership row for a Workspace."""
        self._require_workspace(workspace_id)
        adapter = self._get_adapter(resource_type)
        canonical_resource_type = adapter.resource_type
        canonical_resource_id = self._canonical_resource_id(
            adapter,
            canonical_resource_type,
            resource_id,
            workspace_id=workspace_id,
            user_id=user_id,
            media_db=media_db,
            request_metadata=None,
            validate=False,
        )
        row = self.chacha_db.get_workspace_resource_membership(
            workspace_id,
            canonical_resource_type,
            canonical_resource_id,
            include_deleted=False,
        )
        if row is None:
            return None
        context = self._context(workspace_id, user_id=user_id, media_db=media_db)
        return self._serialize_membership(row, context=context, resolve=resolve)

    def list_workspace_memberships(
        self,
        workspace_id: str,
        *,
        resource_type: str | None = None,
        role: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        cursor: str | WorkspaceMembershipCursor | tuple[str, str, str] | None = None,
        user_id: str | None = None,
        media_db: Any | None = None,
        resolve: bool = True,
    ) -> dict[str, Any]:
        """List memberships for one Workspace."""
        self._require_workspace(workspace_id)
        canonical_resource_type = None
        if resource_type is not None:
            canonical_resource_type = self._get_adapter(resource_type).resource_type
        normalized_limit = self._normalize_limit(limit)
        normalized_cursor = self._workspace_cursor_tuple(cursor)
        rows = self.chacha_db.list_workspace_resource_memberships(
            workspace_id,
            resource_type=canonical_resource_type,
            role=role,
            include_deleted=include_deleted,
            limit=normalized_limit,
            cursor=normalized_cursor,
        )
        page_rows, next_cursor = self._trim_workspace_page(rows, normalized_limit)
        context = self._context(workspace_id, user_id=user_id, media_db=media_db)
        items = [self._serialize_membership(row, context=context, resolve=resolve) for row in page_rows]
        return {
            "workspace_id": workspace_id,
            "items": items,
            "total": len(items),
            "next_cursor": next_cursor,
            "summary": self._summary_from_rows(page_rows),
        }

    def list_resource_memberships(
        self,
        resource_type: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
        limit: int = 100,
        cursor: str | WorkspaceResourceMembershipCursor | tuple[str, str] | None = None,
        user_id: str | None = None,
        media_db: Any | None = None,
        resolve: bool = True,
    ) -> dict[str, Any]:
        """List Workspace memberships for one canonical resource."""
        adapter = self._get_adapter(resource_type)
        canonical_resource_type = adapter.resource_type
        canonical_resource_id = self._canonical_resource_id(
            adapter,
            canonical_resource_type,
            resource_id,
            workspace_id="",
            user_id=user_id,
            media_db=media_db,
            request_metadata=None,
            validate=canonical_resource_type == "media",
        )
        normalized_limit = self._normalize_limit(limit)
        normalized_cursor = self._resource_cursor_tuple(cursor)
        rows = self.chacha_db.list_resource_workspace_memberships(
            canonical_resource_type,
            canonical_resource_id,
            include_deleted=include_deleted,
            limit=normalized_limit,
            cursor=normalized_cursor,
        )
        page_rows, next_cursor = self._trim_resource_page(rows, normalized_limit)
        items = [
            self._serialize_membership(
                row,
                context=self._context(str(row.get("workspace_id") or ""), user_id=user_id, media_db=media_db),
                resolve=resolve,
            )
            for row in page_rows
        ]
        return {
            "resource_type": canonical_resource_type,
            "resource_id": canonical_resource_id,
            "items": items,
            "total": len(items),
            "next_cursor": next_cursor,
            "summary": self._summary_from_rows(page_rows),
        }

    def unlink_membership(
        self,
        workspace_id: str,
        resource_type: str,
        resource_id: str,
        *,
        user_id: str | None = None,
        media_db: Any | None = None,
    ) -> dict[str, Any] | None:
        """Soft-delete one Workspace membership."""
        workspace = self._require_workspace(workspace_id)
        self._require_writable_workspace(workspace)
        adapter = self._get_adapter(resource_type)
        context = self._context(workspace_id, user_id=user_id, media_db=media_db)
        canonical_resource_id = self._canonical_resource_id(
            adapter,
            adapter.resource_type,
            resource_id,
            workspace_id=workspace_id,
            user_id=user_id,
            media_db=media_db,
            request_metadata=None,
            validate=False,
        )
        row = self.chacha_db.delete_workspace_resource_membership(
            workspace_id,
            adapter.resource_type,
            canonical_resource_id,
            user_id=user_id,
        )
        if row is None:
            return None
        adapter.on_unlink(row, context)
        return self._serialize_membership(row, context=context, resolve=False)

    def workspace_membership_summary(self, workspace_id: str) -> dict[str, Any]:
        """Return compact active membership totals for Workspace context."""
        self._require_workspace(workspace_id)
        rows: list[Mapping[str, Any]] = []
        cursor: tuple[str, str, str] | None = None
        page_size = 1000
        while True:
            page = self.chacha_db.list_workspace_resource_memberships(
                workspace_id,
                include_deleted=False,
                limit=page_size,
                cursor=cursor,
            )
            active_page = page[:page_size]
            rows.extend(active_page)
            if len(page) <= page_size or not active_page:
                break
            last = active_page[-1]
            cursor = (
                str(last.get("updated_at") or ""),
                str(last.get("resource_type") or ""),
                str(last.get("resource_id") or ""),
            )
        return self._summary_from_rows(rows)

    def backfill_workspace_memberships(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Task 5 owns broad backfill; this slice only exposes an explicit stub."""
        return {
            "status": "not_implemented",
            "message": "Workspace membership backfill is owned by Task 5.",
        }

    def _require_workspace(self, workspace_id: str) -> Mapping[str, Any]:
        workspace = self.chacha_db.get_workspace(workspace_id)
        if workspace is None:
            raise WorkspaceMembershipServiceError(
                "workspace_not_found",
                f"Workspace '{workspace_id}' was not found.",
                status_code=404,
            )
        return workspace

    def _require_writable_workspace(self, workspace: Mapping[str, Any]) -> None:
        if workspace.get("archived") in (True, 1, "1", "true", "True"):
            raise WorkspaceMembershipServiceError(
                "workspace_archived",
                "Archived workspaces cannot be modified.",
                status_code=409,
            )

    def _context(
        self,
        workspace_id: str,
        *,
        user_id: str | None,
        media_db: Any | None,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> WorkspaceMembershipContext:
        return WorkspaceMembershipContext(
            workspace_id=workspace_id,
            user_id=user_id,
            chacha_db=self.chacha_db,
            media_db=media_db,
            request_metadata=dict(request_metadata or {}),
        )

    def _get_adapter(self, resource_type: str) -> WorkspaceMembershipAdapter:
        try:
            return get_workspace_membership_adapter(resource_type, self.adapters)
        except WorkspaceMembershipAdapterError as exc:
            raise self._service_error_from_adapter_error(exc) from exc

    def _validate_access(
        self,
        adapter: WorkspaceMembershipAdapter,
        resource_id: str,
        context: WorkspaceMembershipContext,
    ) -> WorkspaceResourceRef:
        try:
            return adapter.validate_access(resource_id, context)
        except WorkspaceMembershipAdapterError as exc:
            raise self._service_error_from_adapter_error(exc) from exc

    def _canonical_resource_id(
        self,
        adapter: WorkspaceMembershipAdapter,
        resource_type: str,
        resource_id: str,
        *,
        workspace_id: str,
        user_id: str | None,
        media_db: Any | None,
        request_metadata: Mapping[str, Any] | None,
        validate: bool,
    ) -> str:
        if validate:
            context = self._context(
                workspace_id,
                user_id=user_id,
                media_db=media_db,
                request_metadata=request_metadata,
            )
            return self._validate_access(adapter, resource_id, context).resource_id
        normalized_resource_id = self._non_empty_string(resource_id, "resource_id")
        if resource_type in {"media", "workspace_note"}:
            try:
                parsed = int(normalized_resource_id)
            except (TypeError, ValueError) as exc:
                raise WorkspaceMembershipServiceError(
                    "invalid_resource_id",
                    f"Workspace membership resource_id for '{resource_type}' must be an integer.",
                    status_code=400,
                ) from exc
            if parsed < 0:
                raise WorkspaceMembershipServiceError(
                    "invalid_resource_id",
                    f"Workspace membership resource_id for '{resource_type}' must be non-negative.",
                    status_code=400,
                )
            return str(parsed)
        return normalized_resource_id

    def _serialize_membership(
        self,
        row: Mapping[str, Any],
        *,
        context: WorkspaceMembershipContext,
        summary_ref: WorkspaceResourceRef | None = None,
        resolve: bool = True,
    ) -> dict[str, Any]:
        item = {
            "workspace_id": str(row.get("workspace_id") or context.workspace_id),
            "resource_type": str(row.get("resource_type") or ""),
            "resource_id": str(row.get("resource_id") or ""),
            "role": str(row.get("role") or "member"),
            "label": row.get("label"),
            "transfer_policy": str(row.get("transfer_policy") or "link"),
            "provenance": self._mapping_value(row.get("provenance")),
            "metadata": self._mapping_value(row.get("metadata")),
            "summary": None,
            "created_at": str(row.get("created_at") or ""),
            "updated_at": str(row.get("updated_at") or ""),
            "version": int(row.get("version") or 1),
            "deleted": self._row_is_deleted(row),
        }
        if resolve:
            if summary_ref is None:
                summary_ref = self._resolve_summary(row, context)
            item["summary"] = self._summary_to_dict(summary_ref)
        return item

    def _resolve_summary(
        self,
        row: Mapping[str, Any],
        context: WorkspaceMembershipContext,
    ) -> WorkspaceResourceRef:
        resource_type = str(row.get("resource_type") or "")
        resource_id = str(row.get("resource_id") or "")
        try:
            adapter = get_workspace_membership_adapter(resource_type, self.adapters)
            return adapter.summarize(resource_id, context)
        except WorkspaceMembershipAdapterError as exc:
            return self._unresolved_summary(resource_type, resource_id, exc.code, exc.message)
        except Exception:  # pragma: no cover - protects list calls from adapter bugs.
            return self._unresolved_summary(
                resource_type,
                resource_id,
                "summary_unavailable",
                _SUMMARY_UNAVAILABLE_MESSAGE,
            )

    @staticmethod
    def _service_error_from_adapter_error(exc: WorkspaceMembershipAdapterError) -> WorkspaceMembershipServiceError:
        return WorkspaceMembershipServiceError(
            exc.code,
            exc.message,
            status_code=exc.status_code,
            details=exc.details,
        )

    @staticmethod
    def _summary_to_dict(ref: WorkspaceResourceRef) -> dict[str, Any]:
        return {
            "title": ref.title,
            "subtitle": ref.subtitle,
            "href": ref.href,
            "updated_at": ref.updated_at,
            "state": ref.state,
            "metadata": dict(ref.metadata),
        }

    @staticmethod
    def _unresolved_summary(
        resource_type: str,
        resource_id: str,
        code: str,
        message: str,
    ) -> WorkspaceResourceRef:
        bounded_message = message[:240]
        return WorkspaceResourceRef(
            resource_type=resource_type,
            resource_id=resource_id,
            state="unresolved",
            metadata={"code": code, "message": bounded_message},
        )

    @staticmethod
    def _row_is_deleted(row: Mapping[str, Any]) -> bool:
        return row.get("deleted") in (True, 1, "1", "true", "True")

    @staticmethod
    def _mapping_value(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, Mapping) else {}

    def _membership_request_data(self, request: Any) -> dict[str, Any]:
        if hasattr(request, "model_dump"):
            raw = request.model_dump()
        elif isinstance(request, Mapping):
            raw = dict(request)
        else:
            raise WorkspaceMembershipServiceError(
                "invalid_membership_request",
                "Workspace membership request must be a mapping.",
                status_code=400,
            )
        data = {
            "resource_type": self._non_empty_string(raw.get("resource_type"), "resource_type"),
            "resource_id": self._non_empty_string(raw.get("resource_id"), "resource_id"),
            "role": str(raw.get("role") or "member"),
            "label": raw.get("label"),
            "transfer_policy": str(raw.get("transfer_policy") or "link"),
            "provenance": self._mapping_value(raw.get("provenance")),
            "metadata": self._mapping_value(raw.get("metadata")),
        }
        if data["role"] not in WORKSPACE_MEMBERSHIP_ROLES:
            raise WorkspaceMembershipServiceError(
                "unsupported_membership_role",
                f"Workspace membership role '{data['role']}' is not supported.",
                status_code=400,
            )
        if data["transfer_policy"] not in WORKSPACE_MEMBERSHIP_TRANSFER_POLICIES:
            raise WorkspaceMembershipServiceError(
                "unsupported_transfer_policy",
                f"Workspace membership transfer policy '{data['transfer_policy']}' is not supported.",
                status_code=400,
            )
        if data["label"] is not None:
            data["label"] = str(data["label"])
        return data

    @staticmethod
    def _non_empty_string(value: Any, field_name: str) -> str:
        if value is None:
            raise WorkspaceMembershipServiceError(
                "invalid_membership_request",
                f"Workspace membership {field_name} is required.",
                status_code=400,
            )
        normalized = str(value).strip()
        if not normalized:
            raise WorkspaceMembershipServiceError(
                "invalid_membership_request",
                f"Workspace membership {field_name} is required.",
                status_code=400,
            )
        return normalized

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise WorkspaceMembershipServiceError("invalid_limit", "limit must be an integer.", status_code=400)
        if limit < 1 or limit > 1000:
            raise WorkspaceMembershipServiceError("invalid_limit", "limit must be between 1 and 1000.", status_code=400)
        return limit

    @staticmethod
    def _workspace_cursor_tuple(
        cursor: str | WorkspaceMembershipCursor | tuple[str, str, str] | None,
    ) -> tuple[str, str, str] | None:
        if cursor is None:
            return None
        if isinstance(cursor, WorkspaceMembershipCursor):
            return (cursor.updated_at, cursor.resource_type, cursor.resource_id)
        if isinstance(cursor, str):
            try:
                decoded = decode_membership_cursor(cursor)
            except ValueError as exc:
                raise WorkspaceMembershipServiceError(
                    "invalid_cursor",
                    "Workspace membership cursor is invalid.",
                    status_code=400,
                ) from exc
            return (decoded.updated_at, decoded.resource_type, decoded.resource_id)
        if len(cursor) != 3:
            raise WorkspaceMembershipServiceError("invalid_cursor", "Workspace membership cursor is invalid.", status_code=400)
        return cursor

    @staticmethod
    def _resource_cursor_tuple(
        cursor: str | WorkspaceResourceMembershipCursor | tuple[str, str] | None,
    ) -> tuple[str, str] | None:
        if cursor is None:
            return None
        if isinstance(cursor, WorkspaceResourceMembershipCursor):
            return (cursor.updated_at, cursor.workspace_id)
        if isinstance(cursor, str):
            try:
                decoded = decode_resource_membership_cursor(cursor)
            except ValueError as exc:
                raise WorkspaceMembershipServiceError(
                    "invalid_cursor",
                    "Workspace resource membership cursor is invalid.",
                    status_code=400,
                ) from exc
            return (decoded.updated_at, decoded.workspace_id)
        if len(cursor) != 2:
            raise WorkspaceMembershipServiceError("invalid_cursor", "Workspace resource membership cursor is invalid.", status_code=400)
        return cursor

    @staticmethod
    def _trim_workspace_page(rows: list[Mapping[str, Any]], limit: int) -> tuple[list[Mapping[str, Any]], str | None]:
        page_rows = rows[:limit]
        if len(rows) <= limit or not page_rows:
            return page_rows, None
        last = page_rows[-1]
        return page_rows, encode_membership_cursor(
            WorkspaceMembershipCursor(
                updated_at=str(last.get("updated_at") or ""),
                resource_type=str(last.get("resource_type") or ""),
                resource_id=str(last.get("resource_id") or ""),
            )
        )

    @staticmethod
    def _trim_resource_page(rows: list[Mapping[str, Any]], limit: int) -> tuple[list[Mapping[str, Any]], str | None]:
        page_rows = rows[:limit]
        if len(rows) <= limit or not page_rows:
            return page_rows, None
        last = page_rows[-1]
        return page_rows, encode_resource_membership_cursor(
            WorkspaceResourceMembershipCursor(
                updated_at=str(last.get("updated_at") or ""),
                workspace_id=str(last.get("workspace_id") or ""),
            )
        )

    @staticmethod
    def _summary_from_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        active_rows = [
            row
            for row in rows
            if row.get("deleted") not in (True, 1, "1", "true", "True")
        ]
        by_resource_type = Counter(str(row.get("resource_type") or "") for row in active_rows)
        by_role = Counter(str(row.get("role") or "member") for row in active_rows)
        by_resource_type.pop("", None)
        by_role.pop("", None)
        return {
            "total": len(active_rows),
            "by_resource_type": dict(sorted(by_resource_type.items())),
            "by_role": dict(sorted(by_role.items())),
        }
