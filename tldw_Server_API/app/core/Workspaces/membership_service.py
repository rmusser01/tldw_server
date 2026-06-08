"""Workspace resource membership service orchestration."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Workspaces.membership_adapters import (
    WorkspaceMembershipAdapter,
    WorkspaceMembershipContext,
    default_workspace_membership_adapters,
    get_workspace_membership_adapter,
)
from tldw_Server_API.app.core.Workspaces.membership_models import (
    WORKSPACE_MEMBERSHIP_MAX_METADATA_BYTES,
    WORKSPACE_MEMBERSHIP_MAX_PROVENANCE_BYTES,
    WORKSPACE_MEMBERSHIP_ROLES,
    WORKSPACE_MEMBERSHIP_TRANSFER_POLICIES,
    WorkspaceMembershipCursor,
    WorkspaceResourceMembershipCursor,
    WorkspaceResourceRef,
    decode_membership_cursor,
    decode_resource_membership_cursor,
    encode_membership_cursor,
    encode_resource_membership_cursor,
    normalize_membership_json_object,
)
from tldw_Server_API.app.core.exceptions import (
    WorkspaceMembershipAdapterError,
    WorkspaceMembershipServiceError,
)


_SUMMARY_UNAVAILABLE_MESSAGE = "Workspace resource summary is unavailable."
_BACKFILL_ERROR_LIMIT = 25
_BACKFILL_SAFE_ERROR_MESSAGE = "Workspace membership backfill could not link this resource."


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
        try:
            adapter.on_unlink(row, context)
        except Exception:  # noqa: BLE001 - unlink hooks are post-delete best-effort cleanup.
            logger.opt(exception=True).warning(
                "Workspace membership unlink hook failed for workspace={} resource_type={} resource_id={}",
                workspace_id,
                adapter.resource_type,
                canonical_resource_id,
            )
        return self._serialize_membership(row, context=context, resolve=False)

    def workspace_membership_summary(self, workspace_id: str) -> dict[str, Any]:
        """Return compact active membership totals for Workspace context."""
        self._require_workspace(workspace_id)
        summary = self.chacha_db.workspace_resource_membership_summary(workspace_id)
        return {
            "total": int(summary.get("total") or 0),
            "by_resource_type": dict(summary.get("by_resource_type") or {}),
            "by_role": dict(summary.get("by_role") or {}),
        }

    def backfill_workspace_memberships(
        self,
        workspace_id: str,
        *,
        user_id: str | None = None,
        media_db: Any | None = None,
    ) -> dict[str, Any]:
        """Explicitly link existing Workspace-scoped rows into generic memberships."""
        workspace = self._require_workspace(workspace_id)
        self._require_writable_workspace(workspace)
        created = 0
        existing = 0
        restored = 0
        skipped = 0
        errors: list[dict[str, str]] = []

        for candidate in self._workspace_backfill_candidates(workspace_id, errors):
            before = self.chacha_db.get_workspace_resource_membership(
                workspace_id,
                candidate["resource_type"],
                candidate["resource_id"],
                include_deleted=True,
            )
            was_deleted = before is not None and self._row_is_deleted(before)
            try:
                self.link_membership(
                    workspace_id,
                    {
                        "resource_type": candidate["resource_type"],
                        "resource_id": candidate["resource_id"],
                        "role": candidate["role"],
                        "label": candidate.get("label"),
                        "transfer_policy": "link",
                        "provenance": {
                            "source_surface": "workspace_backfill",
                            "source_table": candidate["source_table"],
                        },
                        "metadata": {},
                    },
                    user_id=user_id,
                    media_db=media_db,
                    resolve=False,
                )
            except Exception as exc:  # noqa: BLE001 - explicit backfill records bounded diagnostics and continues.
                self._append_backfill_error(
                    errors,
                    resource_type=candidate["resource_type"],
                    resource_id=candidate["resource_id"],
                    exc=exc,
                )
                continue
            if before is None:
                created += 1
            elif was_deleted:
                restored += 1
            else:
                existing += 1

        summary = self.workspace_membership_summary(workspace_id)
        status = "partial" if errors or skipped else "complete"
        return {
            "status": status,
            "created": created,
            "existing": existing,
            "restored": restored,
            "skipped": skipped,
            "errors": errors,
            "summary": summary,
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

    def _workspace_backfill_candidates(
        self,
        workspace_id: str,
        errors: list[dict[str, str]],
    ) -> list[dict[str, str | None]]:
        candidates: list[dict[str, str | None]] = []
        for source in self._safe_backfill_rows(
            "workspace_source",
            "",
            errors,
            self.chacha_db.list_workspace_sources,
            workspace_id,
        ):
            source_id = self._backfill_row_id(source)
            if source_id is not None:
                candidates.append(
                    {
                        "resource_type": "workspace_source",
                        "resource_id": source_id,
                        "role": "source",
                        "label": self._backfill_label(source),
                        "source_table": "workspace_sources",
                    }
                )
            media_id = self._positive_backfill_int(source.get("media_id"))
            if media_id is not None:
                candidates.append(
                    {
                        "resource_type": "media",
                        "resource_id": str(media_id),
                        "role": "source",
                        "label": self._backfill_label(source),
                        "source_table": "workspace_sources",
                    }
                )

        for artifact in self._safe_backfill_rows(
            "workspace_artifact",
            "",
            errors,
            self.chacha_db.list_workspace_artifacts,
            workspace_id,
        ):
            artifact_id = self._backfill_row_id(artifact)
            if artifact_id is not None:
                candidates.append(
                    {
                        "resource_type": "workspace_artifact",
                        "resource_id": artifact_id,
                        "role": "artifact",
                        "label": self._backfill_label(artifact),
                        "source_table": "workspace_artifacts",
                    }
                )

        for note in self._safe_backfill_rows(
            "workspace_note",
            "",
            errors,
            self.chacha_db.list_workspace_notes,
            workspace_id,
        ):
            note_id = self._backfill_row_id(note)
            if note_id is not None:
                candidates.append(
                    {
                        "resource_type": "workspace_note",
                        "resource_id": note_id,
                        "role": "reference",
                        "label": self._backfill_label(note),
                        "source_table": "workspace_notes",
                    }
                )

        for conversation in self._workspace_backfill_conversations(workspace_id, errors):
            conversation_id = self._backfill_row_id(conversation)
            if conversation_id is not None:
                candidates.append(
                    {
                        "resource_type": "chat",
                        "resource_id": conversation_id,
                        "role": "conversation",
                        "label": self._backfill_label(conversation),
                        "source_table": "conversations",
                    }
                )
        return candidates

    def _workspace_backfill_conversations(
        self,
        workspace_id: str,
        errors: list[dict[str, str]],
    ) -> list[Mapping[str, Any]]:
        list_workspace_conversations = getattr(self.chacha_db, "list_workspace_conversations", None)
        if callable(list_workspace_conversations):
            return self._safe_backfill_rows("chat", "", errors, list_workspace_conversations, workspace_id)
        search_conversations = getattr(self.chacha_db, "search_conversations", None)
        if callable(search_conversations):
            return self._safe_backfill_rows(
                "chat",
                "",
                errors,
                search_conversations,
                None,
                scope_type="workspace",
                workspace_id=workspace_id,
            )
        self._append_backfill_error(
            errors,
            resource_type="chat",
            resource_id="",
            code="chat_listing_unavailable",
            message="Workspace-scoped chat listing is unavailable.",
        )
        return []

    def _safe_backfill_rows(
        self,
        resource_type: str,
        resource_id: str,
        errors: list[dict[str, str]],
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> list[Mapping[str, Any]]:
        try:
            rows = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - bounded diagnostics are the backfill contract.
            self._append_backfill_error(
                errors,
                resource_type=resource_type,
                resource_id=resource_id,
                exc=exc,
            )
            return []
        if not isinstance(rows, list):
            self._append_backfill_error(
                errors,
                resource_type=resource_type,
                resource_id=resource_id,
                code="invalid_backfill_rows",
                message="Workspace backfill row listing returned an invalid payload.",
            )
            return []
        return [row for row in rows if isinstance(row, Mapping)]

    @staticmethod
    def _backfill_row_id(row: Mapping[str, Any]) -> str | None:
        raw_id = row.get("id")
        if raw_id is None:
            return None
        normalized = str(raw_id).strip()
        return normalized or None

    @staticmethod
    def _backfill_label(row: Mapping[str, Any]) -> str | None:
        for key in ("title", "name"):
            raw = row.get(key)
            if raw is not None and str(raw).strip():
                return str(raw).strip()[:512]
        return None

    @staticmethod
    def _positive_backfill_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def _append_backfill_error(
        cls,
        errors: list[dict[str, str]],
        *,
        resource_type: str,
        resource_id: str,
        exc: Exception | None = None,
        code: str | None = None,
        message: str | None = None,
    ) -> None:
        if len(errors) >= _BACKFILL_ERROR_LIMIT:
            return
        if isinstance(exc, WorkspaceMembershipServiceError):
            code = exc.code
            message = exc.message
        elif exc is not None:
            code = code or "backfill_link_failed"
            message = message or _BACKFILL_SAFE_ERROR_MESSAGE
        errors.append(
            {
                "resource_type": resource_type,
                "resource_id": resource_id,
                "code": code or "backfill_link_failed",
                "message": cls._safe_backfill_message(message),
            }
        )

    @staticmethod
    def _safe_backfill_message(message: str | None) -> str:
        raw_message = str(message or _BACKFILL_SAFE_ERROR_MESSAGE).replace("\n", " ").strip()
        if "/" in raw_message or "\\" in raw_message:
            return _BACKFILL_SAFE_ERROR_MESSAGE
        return raw_message[:240]

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
            "provenance": self._membership_json_value(
                raw.get("provenance"),
                field_name="provenance",
                max_bytes=WORKSPACE_MEMBERSHIP_MAX_PROVENANCE_BYTES,
            ),
            "metadata": self._membership_json_value(
                raw.get("metadata"),
                field_name="metadata",
                max_bytes=WORKSPACE_MEMBERSHIP_MAX_METADATA_BYTES,
            ),
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
    def _membership_json_value(value: Any, *, field_name: str, max_bytes: int) -> dict[str, Any]:
        try:
            return normalize_membership_json_object(value, field_name=field_name, max_bytes=max_bytes)
        except ValueError as exc:
            raise WorkspaceMembershipServiceError(
                "invalid_membership_request",
                str(exc),
                status_code=400,
            ) from exc

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
        if (
            not isinstance(cursor, (tuple, list))
            or len(cursor) != 3
            or not all(isinstance(part, str) and part for part in cursor)
        ):
            raise WorkspaceMembershipServiceError("invalid_cursor", "Workspace membership cursor is invalid.", status_code=400)
        return (cursor[0], cursor[1], cursor[2])

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
        if (
            not isinstance(cursor, (tuple, list))
            or len(cursor) != 2
            or not all(isinstance(part, str) and part for part in cursor)
        ):
            raise WorkspaceMembershipServiceError("invalid_cursor", "Workspace resource membership cursor is invalid.", status_code=400)
        return (cursor[0], cursor[1])

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
