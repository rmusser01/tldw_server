from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
    get_or_create_audit_service_for_user_id_optional,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    AuditEventCategory,
    AuditEventType,
)
from tldw_Server_API.app.core.exceptions import (
    BadRequestError,
    ResourceNotFoundError,
)
from tldw_Server_API.app.core.AuthNZ.repos.managed_secret_refs_repo import (
    ManagedSecretRefsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import (
    McpHubRepo,
    encode_managed_secret_credential_ref,
)
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
    build_secret_payload,
    dumps_envelope,
    encrypt_byok_payload,
    key_hint_for_api_key,
)
from tldw_Server_API.app.services.mcp_hub_external_legacy_inventory import (
    McpHubExternalLegacyInventoryService,
)
from tldw_Server_API.app.services.mcp_hub_external_access_resolver import (
    McpHubExternalAccessResolver,
)
from tldw_Server_API.app.services.mcp_hub_external_auth_service import (
    ManagedExternalAuthBridge,
)
from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
    McpHubWorkspaceRootResolver,
)


async def _await_if_needed(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def emit_mcp_hub_audit(
    *,
    action: str,
    actor_id: int | None,
    resource_type: str,
    resource_id: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Best-effort audit emitter for MCP Hub mutations."""
    try:
        svc = await get_or_create_audit_service_for_user_id_optional(actor_id)
        ctx = AuditContext(
            user_id=str(actor_id) if actor_id is not None else None,
            endpoint="/api/v1/mcp/hub",
            method="INTERNAL",
        )
        await svc.log_event(
            event_type=AuditEventType.CONFIG_CHANGED,
            category=AuditEventCategory.SYSTEM,
            context=ctx,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            metadata={
                "actor_id": actor_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
                **(metadata or {}),
            },
        )
        await svc.flush(raise_on_failure=False)
    except Exception as exc:
        logger.warning("MCP hub audit emission failed for action={}: {}", action, exc)


class McpHubConflictError(BadRequestError):
    """Raised when creating an MCP Hub resource would overwrite an existing one."""


class McpHubValidationError(BadRequestError):
    """Structured MCP Hub validation error with machine-readable detail payload."""

    def __init__(self, detail: dict[str, Any]) -> None:
        self.detail = dict(detail)
        super().__init__(str(self.detail.get("message") or self.detail.get("code") or "Invalid MCP Hub request"))


_SCOPE_RANK = {"global": 0, "org": 1, "team": 2, "user": 3}
_CREDENTIAL_SLOT_PRIVILEGE_CLASSES = {"read", "write", "admin"}
_CREDENTIAL_SLOT_PRIVILEGE_RANK = {"read": 0, "write": 1, "admin": 2}
_CREDENTIAL_SLOT_REQUIRED_PERMISSIONS = {
    "read": "grant.credentials.read",
    "write": "grant.credentials.write",
    "admin": "grant.credentials.admin",
}
_SHARED_WORKSPACE_SCOPE_TYPES = {"global", "org", "team"}
_UNION_POLICY_KEYS = {"allowed_tools", "denied_tools", "tool_names", "tool_patterns", "capabilities"}
_PATH_SCOPE_MULTI_ROOT_MODE = "workspace_root"


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, (list, tuple, set)):
        return []
    out: list[str] = []
    for entry in value:
        cleaned = str(entry or "").strip()
        if cleaned:
            out.append(cleaned)
    return out


def _allowed_shared_workspace_roots() -> tuple[Path, ...]:
    from tldw_Server_API.app.core.config import get_config_value

    raw_values: list[str] = []
    raw_values.extend(
        entry.strip()
        for entry in str(get_config_value("ACP-WORKSPACE", "allowed_base_paths", "") or "").replace(
            os.pathsep,
            ",",
        ).split(",")
        if entry.strip()
    )
    raw_values.extend(
        entry.strip()
        for entry in str(os.getenv("ACP_WORKSPACE_ALLOWED_BASE_PATHS", "") or "").replace(
            os.pathsep,
            ",",
        ).split(",")
        if entry.strip()
    )

    roots: list[Path] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            logger.warning("Ignoring non-absolute MCP shared workspace base path: {}", raw_value)
            continue
        normalized = Path(os.path.normpath(str(candidate)))
        marker = str(normalized)
        if marker in seen:
            continue
        seen.add(marker)
        roots.append(normalized)
    return tuple(roots)


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _merge_policy_documents(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in dict(overlay or {}).items():
        if key in _UNION_POLICY_KEYS:
            merged[key] = _unique(_as_str_list(merged.get(key)) + _as_str_list(value))
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_policy_documents(_as_dict(merged.get(key)), value)
            continue
        merged[key] = value
    return merged


def _normalize_path_scope_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"workspace_root", "cwd_descendants"}:
        return normalized
    return "none"


def _roots_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _workspace_source_readiness_summary(
    *,
    warning_codes: list[str] | None = None,
    warning_message: str | None = None,
    conflicting_workspace_ids: list[str] | None = None,
    conflicting_workspace_roots: list[str] | None = None,
    unresolved_workspace_ids: list[str] | None = None,
) -> dict[str, Any]:
    codes = _unique(_as_str_list(warning_codes or []))
    conflict_ids = _unique(_as_str_list(conflicting_workspace_ids or []))
    conflict_roots = _unique(_as_str_list(conflicting_workspace_roots or []))
    unresolved = _unique(_as_str_list(unresolved_workspace_ids or []))
    return {
        "is_multi_root_ready": not (codes or conflict_ids or conflict_roots or unresolved),
        "warning_codes": codes,
        "warning_message": str(warning_message or "").strip() or None,
        "conflicting_workspace_ids": conflict_ids,
        "conflicting_workspace_roots": conflict_roots,
        "unresolved_workspace_ids": unresolved,
    }


class McpHubService:
    """Business logic for MCP Hub profile and external server management."""

    def __init__(
        self,
        repo: McpHubRepo,
        *,
        legacy_inventory_service: McpHubExternalLegacyInventoryService | None = None,
        workspace_root_resolver: McpHubWorkspaceRootResolver | None = None,
    ):
        self.repo = repo
        self.legacy_inventory_service = legacy_inventory_service
        self.workspace_root_resolver = workspace_root_resolver or McpHubWorkspaceRootResolver(repo=repo)
        self.external_access_resolver = McpHubExternalAccessResolver(repo=repo)

    @staticmethod
    def _normalize_credential_slot_privilege_class(value: str) -> str:
        normalized = str(value or "").strip().lower()
        if normalized not in _CREDENTIAL_SLOT_PRIVILEGE_CLASSES:
            raise BadRequestError(
                "Credential slot privilege_class must be one of: read, write, admin"
            )
        return normalized

    @staticmethod
    def _normalize_shared_workspace_scope(
        owner_scope_type: str,
        owner_scope_id: int | None,
    ) -> tuple[str, int | None]:
        scope_type = str(owner_scope_type or "").strip().lower()
        if scope_type not in _SHARED_WORKSPACE_SCOPE_TYPES:
            raise BadRequestError("Shared workspaces must use owner_scope_type of: global, org, team")
        if scope_type == "global":
            return scope_type, None
        if owner_scope_id is None:
            raise BadRequestError(f"{scope_type} shared workspaces require owner_scope_id")
        return scope_type, int(owner_scope_id)

    @staticmethod
    def _normalize_shared_workspace_root(absolute_root: str) -> str:
        raw_value = str(absolute_root or "").strip()
        if not raw_value:
            raise BadRequestError("absolute_root is required")
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            raise BadRequestError("absolute_root must be an absolute path")
        normalized = Path(os.path.normpath(str(candidate)))
        if str(normalized) in {normalized.anchor, "/"}:
            raise BadRequestError("absolute_root must not be the filesystem root")
        allowed_roots = _allowed_shared_workspace_roots()
        if allowed_roots and not any(
            normalized == root or normalized.is_relative_to(root)
            for root in allowed_roots
        ):
            raise BadRequestError(
                "absolute_root must stay under the configured allowed base paths"
            )
        return str(normalized)

    @staticmethod
    def _scope_reference_allowed(
        *,
        object_scope_type: str,
        object_scope_id: int | None,
        target_scope_type: str,
        target_scope_id: int | None,
        resource_name: str,
    ) -> None:
        object_rank = _SCOPE_RANK.get(object_scope_type, -1)
        target_rank = _SCOPE_RANK.get(target_scope_type, -1)
        if object_rank < 0 or target_rank < 0:
            raise BadRequestError("Invalid MCP Hub owner scope")
        if object_rank > target_rank:
            raise BadRequestError(
                f"Cannot reference a narrower-scope {resource_name} from a broader target"
            )
        if object_rank == target_rank and object_scope_type != "global":
            if object_scope_id != target_scope_id:
                raise BadRequestError(
                    f"Cannot reference a {resource_name} from a different owner scope"
                )

    @staticmethod
    def _scope_reference_reason(
        *,
        object_scope_type: str,
        object_scope_id: int | None,
        target_scope_type: str,
        target_scope_id: int | None,
    ) -> str | None:
        object_rank = _SCOPE_RANK.get(object_scope_type, -1)
        target_rank = _SCOPE_RANK.get(target_scope_type, -1)
        if object_rank < 0 or target_rank < 0:
            return "scope_incompatible_reference"
        if object_rank > target_rank:
            return "scope_incompatible_reference"
        if object_rank == target_rank and object_scope_type != "global":
            if object_scope_id != target_scope_id:
                return "scope_incompatible_reference"
        return None

    @staticmethod
    def _shared_workspace_scope_compatible(
        *,
        target_scope_type: str,
        target_scope_id: int | None,
        entry_scope_type: str,
        entry_scope_id: int | None,
    ) -> bool:
        target_scope = str(target_scope_type or "").strip().lower()
        entry_scope = str(entry_scope_type or "").strip().lower()
        if entry_scope == "global":
            return True
        return entry_scope == target_scope and entry_scope_id == target_scope_id

    async def validate_path_scope_object_reference(
        self,
        *,
        path_scope_object_id: int | None,
        target_scope_type: str,
        target_scope_id: int | None,
    ) -> dict[str, Any] | None:
        if path_scope_object_id is None:
            return None
        row = await self.repo.get_path_scope_object(int(path_scope_object_id))
        if not row:
            raise ResourceNotFoundError("mcp_path_scope_object", identifier=str(path_scope_object_id))
        if not bool(row.get("is_active", True)):
            raise BadRequestError("Referenced path scope object is inactive")

        self._scope_reference_allowed(
            object_scope_type=str(row.get("owner_scope_type") or "global"),
            object_scope_id=row.get("owner_scope_id"),
            target_scope_type=str(target_scope_type or "global"),
            target_scope_id=target_scope_id,
            resource_name="path scope object",
        )
        return row

    async def validate_workspace_set_object_reference(
        self,
        *,
        workspace_set_object_id: int | None,
        target_scope_type: str,
        target_scope_id: int | None,
    ) -> dict[str, Any] | None:
        if workspace_set_object_id is None:
            return None
        row = await self.repo.get_workspace_set_object(int(workspace_set_object_id))
        if not row:
            raise ResourceNotFoundError("mcp_workspace_set_object", identifier=str(workspace_set_object_id))
        if not bool(row.get("is_active", True)):
            raise BadRequestError("Referenced workspace set object is inactive")
        self._scope_reference_allowed(
            object_scope_type=str(row.get("owner_scope_type") or "global"),
            object_scope_id=row.get("owner_scope_id"),
            target_scope_type=str(target_scope_type or "global"),
            target_scope_id=target_scope_id,
            resource_name="workspace set object",
        )
        return row

    async def inspect_path_scope_object_reference(
        self,
        *,
        path_scope_object_id: int | None,
        target_scope_type: str,
        target_scope_id: int | None,
    ) -> dict[str, Any] | None:
        if path_scope_object_id is None:
            return None
        row = await self.repo.get_path_scope_object(int(path_scope_object_id))
        if not row:
            return {
                "reference_field": "path_scope_object_id",
                "reference_object_kind": "path_scope_object",
                "reference_object_id": str(path_scope_object_id),
                "reference_reason": "missing_reference",
                "reference_scope_type": None,
                "reference_scope_id": None,
                "reference_label": None,
            }
        if not bool(row.get("is_active", True)):
            return {
                "reference_field": "path_scope_object_id",
                "reference_object_kind": "path_scope_object",
                "reference_object_id": str(row.get("id") or path_scope_object_id),
                "reference_reason": "inactive_reference",
                "reference_scope_type": row.get("owner_scope_type"),
                "reference_scope_id": row.get("owner_scope_id"),
                "reference_label": row.get("name"),
            }
        scope_reason = self._scope_reference_reason(
            object_scope_type=str(row.get("owner_scope_type") or "global"),
            object_scope_id=row.get("owner_scope_id"),
            target_scope_type=str(target_scope_type or "global"),
            target_scope_id=target_scope_id,
        )
        if scope_reason:
            return {
                "reference_field": "path_scope_object_id",
                "reference_object_kind": "path_scope_object",
                "reference_object_id": str(row.get("id") or path_scope_object_id),
                "reference_reason": scope_reason,
                "reference_scope_type": row.get("owner_scope_type"),
                "reference_scope_id": row.get("owner_scope_id"),
                "reference_label": row.get("name"),
            }
        return None

    async def inspect_workspace_set_object_reference(
        self,
        *,
        workspace_set_object_id: int | None,
        target_scope_type: str,
        target_scope_id: int | None,
    ) -> dict[str, Any] | None:
        if workspace_set_object_id is None:
            return None
        row = await self.repo.get_workspace_set_object(int(workspace_set_object_id))
        if not row:
            return {
                "reference_field": "workspace_set_object_id",
                "reference_object_kind": "workspace_set_object",
                "reference_object_id": str(workspace_set_object_id),
                "reference_reason": "missing_reference",
                "reference_scope_type": None,
                "reference_scope_id": None,
                "reference_label": None,
            }
        if not bool(row.get("is_active", True)):
            return {
                "reference_field": "workspace_set_object_id",
                "reference_object_kind": "workspace_set_object",
                "reference_object_id": str(row.get("id") or workspace_set_object_id),
                "reference_reason": "inactive_reference",
                "reference_scope_type": row.get("owner_scope_type"),
                "reference_scope_id": row.get("owner_scope_id"),
                "reference_label": row.get("name"),
            }
        scope_reason = self._scope_reference_reason(
            object_scope_type=str(row.get("owner_scope_type") or "global"),
            object_scope_id=row.get("owner_scope_id"),
            target_scope_type=str(target_scope_type or "global"),
            target_scope_id=target_scope_id,
        )
        if scope_reason:
            return {
                "reference_field": "workspace_set_object_id",
                "reference_object_kind": "workspace_set_object",
                "reference_object_id": str(row.get("id") or workspace_set_object_id),
                "reference_reason": scope_reason,
                "reference_scope_type": row.get("owner_scope_type"),
                "reference_scope_id": row.get("owner_scope_id"),
                "reference_label": row.get("name"),
            }
        return None

    async def inspect_permission_profile_reference(
        self,
        *,
        profile_id: int | None,
        target_scope_type: str,
        target_scope_id: int | None,
    ) -> dict[str, Any] | None:
        if profile_id is None:
            return None
        row = await self.repo.get_permission_profile(int(profile_id))
        if not row:
            return {
                "reference_field": "profile_id",
                "reference_object_kind": "permission_profile",
                "reference_object_id": str(profile_id),
                "reference_reason": "missing_reference",
                "reference_scope_type": None,
                "reference_scope_id": None,
                "reference_label": None,
            }
        if not bool(row.get("is_active", True)):
            return {
                "reference_field": "profile_id",
                "reference_object_kind": "permission_profile",
                "reference_object_id": str(row.get("id") or profile_id),
                "reference_reason": "inactive_reference",
                "reference_scope_type": row.get("owner_scope_type"),
                "reference_scope_id": row.get("owner_scope_id"),
                "reference_label": row.get("name"),
            }
        scope_reason = self._scope_reference_reason(
            object_scope_type=str(row.get("owner_scope_type") or "global"),
            object_scope_id=row.get("owner_scope_id"),
            target_scope_type=str(target_scope_type or "global"),
            target_scope_id=target_scope_id,
        )
        if scope_reason:
            return {
                "reference_field": "profile_id",
                "reference_object_kind": "permission_profile",
                "reference_object_id": str(row.get("id") or profile_id),
                "reference_reason": scope_reason,
                "reference_scope_type": row.get("owner_scope_type"),
                "reference_scope_id": row.get("owner_scope_id"),
                "reference_label": row.get("name"),
            }
        return None

    @staticmethod
    def _multi_root_validation_error(
        *,
        code: str,
        message: str,
        **detail: Any,
    ) -> McpHubValidationError:
        payload = {"code": code, "message": message}
        payload.update({key: value for key, value in detail.items() if value is not None})
        return McpHubValidationError(payload)

    async def _resolve_effective_assignment_path_policy(
        self,
        *,
        assignment_id: int | None,
        profile_id: int | None,
        path_scope_object_id: int | None,
        inline_policy_document: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if profile_id is not None:
            profile_row = await self.repo.get_permission_profile(int(profile_id))
            if profile_row and bool(profile_row.get("is_active", True)):
                profile_path_scope_object_id = profile_row.get("path_scope_object_id")
                if profile_path_scope_object_id is not None:
                    path_scope_row = await self.repo.get_path_scope_object(int(profile_path_scope_object_id))
                    if path_scope_row and bool(path_scope_row.get("is_active", True)):
                        merged = _merge_policy_documents(
                            merged,
                            _as_dict(path_scope_row.get("path_scope_document")),
                        )
                merged = _merge_policy_documents(
                    merged,
                    _as_dict(profile_row.get("policy_document")),
                )

        if path_scope_object_id is not None:
            path_scope_row = await self.repo.get_path_scope_object(int(path_scope_object_id))
            if path_scope_row and bool(path_scope_row.get("is_active", True)):
                merged = _merge_policy_documents(
                    merged,
                    _as_dict(path_scope_row.get("path_scope_document")),
                )

        merged = _merge_policy_documents(merged, _as_dict(inline_policy_document))

        if assignment_id is not None:
            override_row = await self.repo.get_policy_override_by_assignment(int(assignment_id))
            if override_row and bool(override_row.get("is_active", True)):
                merged = _merge_policy_documents(
                    merged,
                    _as_dict(override_row.get("override_policy_document")),
                )

        return merged

    async def _resolve_assignment_workspace_source(
        self,
        *,
        assignment_id: int | None,
        owner_scope_type: str,
        owner_scope_id: int | None,
        workspace_source_mode: str | None,
        workspace_set_object_id: int | None,
        inline_workspace_ids: list[str] | None,
    ) -> tuple[str, str, list[str]]:
        source_mode = str(workspace_source_mode or "").strip().lower() or "inline"
        if source_mode == "named" and workspace_set_object_id is not None:
            workspace_set_row = await self.repo.get_workspace_set_object(int(workspace_set_object_id))
            if workspace_set_row and bool(workspace_set_row.get("is_active", True)):
                workspace_ids = _unique(
                    _as_str_list(
                        [
                            row.get("workspace_id")
                            for row in await self.repo.list_workspace_set_members(int(workspace_set_object_id))
                        ]
                    )
                )
                trust_source = (
                    "shared_registry"
                    if str(workspace_set_row.get("owner_scope_type") or "").strip().lower() != "user"
                    else "user_local"
                )
                return source_mode, trust_source, workspace_ids
            return source_mode, "user_local", []

        if inline_workspace_ids is None and assignment_id is not None:
            inline_workspace_ids = [
                row.get("workspace_id")
                for row in await self.repo.list_policy_assignment_workspaces(int(assignment_id))
            ]
        workspace_ids = _unique(_as_str_list(inline_workspace_ids or []))
        # Inline membership still uses the acting user's trusted local workspaces in v1.
        _ = owner_scope_type
        _ = owner_scope_id
        return source_mode, "user_local", workspace_ids

    async def validate_multi_root_assignment_readiness(
        self,
        *,
        actor_id: int | None,
        assignment_id: int | None,
        owner_scope_type: str,
        owner_scope_id: int | None,
        profile_id: int | None,
        path_scope_object_id: int | None,
        inline_policy_document: dict[str, Any] | None,
        workspace_source_mode: str | None,
        workspace_set_object_id: int | None,
        inline_workspace_ids: list[str] | None,
    ) -> None:
        effective_policy_document = await self._resolve_effective_assignment_path_policy(
            assignment_id=assignment_id,
            profile_id=profile_id,
            path_scope_object_id=path_scope_object_id,
            inline_policy_document=inline_policy_document,
        )
        effective_path_scope_mode = _normalize_path_scope_mode(
            effective_policy_document.get("path_scope_mode")
        )
        if effective_path_scope_mode != _PATH_SCOPE_MULTI_ROOT_MODE:
            return

        source_mode, trust_source, workspace_ids = await self._resolve_assignment_workspace_source(
            assignment_id=assignment_id,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            workspace_source_mode=workspace_source_mode,
            workspace_set_object_id=workspace_set_object_id,
            inline_workspace_ids=inline_workspace_ids,
        )
        if len(workspace_ids) <= 1:
            return

        user_id = str(actor_id if actor_id is not None else owner_scope_id) if trust_source == "user_local" else None
        resolved_roots: list[tuple[str, Path]] = []
        unresolved_workspace_ids: list[str] = []
        for workspace_id in workspace_ids:
            resolved = await self.workspace_root_resolver.resolve_for_context(
                session_id=None,
                user_id=user_id,
                workspace_id=workspace_id,
                workspace_trust_source=trust_source,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
            workspace_root = str(resolved.get("workspace_root") or "").strip()
            if not workspace_root:
                unresolved_workspace_ids.append(workspace_id)
                continue
            resolved_roots.append((workspace_id, Path(workspace_root).expanduser().resolve(strict=False)))

        if unresolved_workspace_ids:
            raise self._multi_root_validation_error(
                code="assignment_workspace_unresolvable",
                message="Workspace source cannot resolve every workspace for multi-root execution.",
                unresolved_workspace_ids=sorted(unresolved_workspace_ids),
                workspace_source_mode=source_mode,
                workspace_trust_source=trust_source,
            )

        for index, (left_workspace_id, left_root) in enumerate(resolved_roots):
            for right_workspace_id, right_root in resolved_roots[index + 1 :]:
                if not _roots_overlap(left_root, right_root):
                    continue
                raise self._multi_root_validation_error(
                    code="assignment_multi_root_overlap",
                    message=(
                        "Named workspace source contains overlapping roots for multi-root execution."
                        if source_mode == "named"
                        else "Inline workspace source contains overlapping roots for multi-root execution."
                    ),
                    conflicting_workspace_ids=[left_workspace_id, right_workspace_id],
                    conflicting_workspace_roots=[str(left_root), str(right_root)],
                    workspace_source_mode=source_mode,
                    workspace_trust_source=trust_source,
                )

    async def inspect_multi_root_assignment_readiness(
        self,
        *,
        actor_id: int | None,
        assignment_row: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            await self.validate_multi_root_assignment_readiness(
                actor_id=actor_id,
                assignment_id=int(assignment_row.get("id")) if assignment_row.get("id") is not None else None,
                owner_scope_type=str(assignment_row.get("owner_scope_type") or "global"),
                owner_scope_id=assignment_row.get("owner_scope_id"),
                profile_id=assignment_row.get("profile_id"),
                path_scope_object_id=assignment_row.get("path_scope_object_id"),
                inline_policy_document=_as_dict(assignment_row.get("inline_policy_document")),
                workspace_source_mode=str(assignment_row.get("workspace_source_mode") or "").strip().lower() or None,
                workspace_set_object_id=assignment_row.get("workspace_set_object_id"),
                inline_workspace_ids=None,
            )
        except McpHubValidationError as exc:
            return dict(exc.detail)
        return None

    @staticmethod
    def _make_governance_audit_finding(
        *,
        finding_type: str,
        severity: str,
        scope_type: str,
        scope_id: int | None,
        object_kind: str,
        object_id: str,
        object_label: str,
        message: str,
        details: dict[str, Any] | None = None,
        navigate_to: dict[str, Any] | None = None,
        related_object_kind: str | None = None,
        related_object_id: str | None = None,
        related_object_label: str | None = None,
    ) -> dict[str, Any]:
        return {
            "finding_type": finding_type,
            "severity": severity,
            "scope_type": str(scope_type or "global"),
            "scope_id": scope_id,
            "object_kind": object_kind,
            "object_id": object_id,
            "object_label": object_label,
            "message": message,
            "details": dict(details or {}),
            "navigate_to": dict(
                navigate_to
                or {
                    "tab": "profiles",
                    "object_kind": object_kind,
                    "object_id": object_id,
                }
            ),
            "related_object_kind": related_object_kind,
            "related_object_id": related_object_id,
            "related_object_label": related_object_label,
        }

    @staticmethod
    def _sort_governance_audit_findings(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        severity_rank = {"error": 0, "warning": 1}
        return sorted(
            items,
            key=lambda item: (
                severity_rank.get(str(item.get("severity") or "warning"), 9),
                str(item.get("scope_type") or ""),
                str(item.get("object_kind") or ""),
                str(item.get("object_label") or ""),
                str(item.get("finding_type") or ""),
            ),
        )

    async def list_governance_audit_findings(
        self,
        *,
        actor_id: int | None,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
        severity: str | None = None,
        finding_type: str | None = None,
        object_kind: str | None = None,
        scope_type: str | None = None,
    ) -> dict[str, Any]:
        findings: list[dict[str, Any]] = []

        workspace_sets = await self.repo.list_workspace_set_objects(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        for workspace_set in workspace_sets:
            summary = await self.get_workspace_set_readiness_summary(
                workspace_set_object_id=int(workspace_set.get("id")),
            )
            if summary.get("is_multi_root_ready", True):
                continue
            findings.append(
                self._make_governance_audit_finding(
                    finding_type="workspace_source_readiness_warning",
                    severity="warning",
                    scope_type=str(workspace_set.get("owner_scope_type") or "user"),
                    scope_id=workspace_set.get("owner_scope_id"),
                    object_kind="workspace_set_object",
                    object_id=str(workspace_set.get("id") or ""),
                    object_label=str(workspace_set.get("name") or ""),
                    message=str(summary.get("warning_message") or "Multi-root readiness warning"),
                    details={
                        "warning_codes": list(summary.get("warning_codes") or []),
                        "conflicting_workspace_ids": list(summary.get("conflicting_workspace_ids") or []),
                        "conflicting_workspace_roots": list(summary.get("conflicting_workspace_roots") or []),
                        "unresolved_workspace_ids": list(summary.get("unresolved_workspace_ids") or []),
                        "advisory_kind": "multi_root_readiness",
                    },
                    navigate_to={
                        "tab": "workspace-sets",
                        "object_kind": "workspace_set_object",
                        "object_id": str(workspace_set.get("id") or ""),
                    },
                )
            )

        shared_workspaces = await self.repo.list_shared_workspace_entries(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        for shared_workspace in shared_workspaces:
            summary = await self.get_shared_workspace_readiness_summary(
                shared_workspace_id=int(shared_workspace.get("id")),
            )
            if summary.get("is_multi_root_ready", True):
                continue
            findings.append(
                self._make_governance_audit_finding(
                    finding_type="shared_workspace_overlap_warning",
                    severity="warning",
                    scope_type=str(shared_workspace.get("owner_scope_type") or "team"),
                    scope_id=shared_workspace.get("owner_scope_id"),
                    object_kind="shared_workspace",
                    object_id=str(shared_workspace.get("id") or ""),
                    object_label=str(
                        shared_workspace.get("display_name")
                        or shared_workspace.get("workspace_id")
                        or shared_workspace.get("id")
                        or ""
                    ),
                    message=str(
                        summary.get("warning_message")
                        or "May conflict with other visible shared roots in multi-root assignments."
                    ),
                    details={
                        "warning_codes": list(summary.get("warning_codes") or []),
                        "conflicting_workspace_ids": list(summary.get("conflicting_workspace_ids") or []),
                        "conflicting_workspace_roots": list(summary.get("conflicting_workspace_roots") or []),
                    },
                    navigate_to={
                        "tab": "shared-workspaces",
                        "object_kind": "shared_workspace",
                        "object_id": str(shared_workspace.get("id") or ""),
                    },
                )
            )

        permission_profiles = await self.repo.list_permission_profiles(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        for profile in permission_profiles:
            broken_reference = await self.inspect_path_scope_object_reference(
                path_scope_object_id=profile.get("path_scope_object_id"),
                target_scope_type=str(profile.get("owner_scope_type") or "global"),
                target_scope_id=profile.get("owner_scope_id"),
            )
            if not broken_reference:
                continue
            reference_label = str(
                broken_reference.get("reference_label")
                or broken_reference.get("reference_object_id")
                or ""
            )
            findings.append(
                self._make_governance_audit_finding(
                    finding_type="broken_object_reference",
                    severity="error",
                    scope_type=str(profile.get("owner_scope_type") or "global"),
                    scope_id=profile.get("owner_scope_id"),
                    object_kind="permission_profile",
                    object_id=str(profile.get("id") or ""),
                    object_label=str(profile.get("name") or profile.get("id") or ""),
                    message=f"Permission profile references a {str(broken_reference.get('reference_reason') or '').replace('_', ' ')} path scope object.",
                    details=broken_reference,
                    navigate_to={
                        "tab": "profiles",
                        "object_kind": "permission_profile",
                        "object_id": str(profile.get("id") or ""),
                    },
                    related_object_kind=str(broken_reference.get("reference_object_kind") or "path_scope_object"),
                    related_object_id=str(broken_reference.get("reference_object_id") or ""),
                    related_object_label=reference_label or None,
                )
            )

        assignments = await self.repo.list_policy_assignments(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        for assignment in assignments:
            assignment_label = str(
                assignment.get("target_id")
                or assignment.get("target_type")
                or assignment.get("id")
                or ""
            )
            broken_profile_reference = await self.inspect_permission_profile_reference(
                profile_id=assignment.get("profile_id"),
                target_scope_type=str(assignment.get("owner_scope_type") or "global"),
                target_scope_id=assignment.get("owner_scope_id"),
            )
            if broken_profile_reference:
                reference_label = str(
                    broken_profile_reference.get("reference_label")
                    or broken_profile_reference.get("reference_object_id")
                    or ""
                )
                findings.append(
                    self._make_governance_audit_finding(
                        finding_type="broken_object_reference",
                        severity="error",
                        scope_type=str(assignment.get("owner_scope_type") or "global"),
                        scope_id=assignment.get("owner_scope_id"),
                        object_kind="policy_assignment",
                        object_id=str(assignment.get("id") or ""),
                        object_label=assignment_label,
                        message=f"Assignment references a {str(broken_profile_reference.get('reference_reason') or '').replace('_', ' ')} permission profile.",
                        details=broken_profile_reference,
                        navigate_to={
                            "tab": "assignments",
                            "object_kind": "policy_assignment",
                            "object_id": str(assignment.get("id") or ""),
                        },
                        related_object_kind=str(
                            broken_profile_reference.get("reference_object_kind") or "permission_profile"
                        ),
                        related_object_id=str(broken_profile_reference.get("reference_object_id") or ""),
                        related_object_label=reference_label or None,
                    )
                )

            broken_path_scope_reference = await self.inspect_path_scope_object_reference(
                path_scope_object_id=assignment.get("path_scope_object_id"),
                target_scope_type=str(assignment.get("owner_scope_type") or "global"),
                target_scope_id=assignment.get("owner_scope_id"),
            )
            if broken_path_scope_reference:
                reference_label = str(
                    broken_path_scope_reference.get("reference_label")
                    or broken_path_scope_reference.get("reference_object_id")
                    or ""
                )
                findings.append(
                    self._make_governance_audit_finding(
                        finding_type="broken_object_reference",
                        severity="error",
                        scope_type=str(assignment.get("owner_scope_type") or "global"),
                        scope_id=assignment.get("owner_scope_id"),
                        object_kind="policy_assignment",
                        object_id=str(assignment.get("id") or ""),
                        object_label=assignment_label,
                        message=f"Assignment references a {str(broken_path_scope_reference.get('reference_reason') or '').replace('_', ' ')} path scope object.",
                        details=broken_path_scope_reference,
                        navigate_to={
                            "tab": "assignments",
                            "object_kind": "policy_assignment",
                            "object_id": str(assignment.get("id") or ""),
                        },
                        related_object_kind=str(
                            broken_path_scope_reference.get("reference_object_kind") or "path_scope_object"
                        ),
                        related_object_id=str(broken_path_scope_reference.get("reference_object_id") or ""),
                        related_object_label=reference_label or None,
                    )
                )

            broken_workspace_set_reference = await self.inspect_workspace_set_object_reference(
                workspace_set_object_id=assignment.get("workspace_set_object_id"),
                target_scope_type=str(assignment.get("owner_scope_type") or "global"),
                target_scope_id=assignment.get("owner_scope_id"),
            )
            if broken_workspace_set_reference:
                reference_label = str(
                    broken_workspace_set_reference.get("reference_label")
                    or broken_workspace_set_reference.get("reference_object_id")
                    or ""
                )
                findings.append(
                    self._make_governance_audit_finding(
                        finding_type="broken_object_reference",
                        severity="error",
                        scope_type=str(assignment.get("owner_scope_type") or "global"),
                        scope_id=assignment.get("owner_scope_id"),
                        object_kind="policy_assignment",
                        object_id=str(assignment.get("id") or ""),
                        object_label=assignment_label,
                        message=f"Assignment references a {str(broken_workspace_set_reference.get('reference_reason') or '').replace('_', ' ')} workspace set object.",
                        details=broken_workspace_set_reference,
                        navigate_to={
                            "tab": "assignments",
                            "object_kind": "policy_assignment",
                            "object_id": str(assignment.get("id") or ""),
                        },
                        related_object_kind=str(
                            broken_workspace_set_reference.get("reference_object_kind") or "workspace_set_object"
                        ),
                        related_object_id=str(broken_workspace_set_reference.get("reference_object_id") or ""),
                        related_object_label=reference_label or None,
                    )
                )

            blocker = await self.inspect_multi_root_assignment_readiness(
                actor_id=actor_id,
                assignment_row=assignment,
            )
            if blocker:
                findings.append(
                    self._make_governance_audit_finding(
                        finding_type="assignment_validation_blocker",
                        severity="error",
                        scope_type=str(assignment.get("owner_scope_type") or "global"),
                        scope_id=assignment.get("owner_scope_id"),
                        object_kind="policy_assignment",
                        object_id=str(assignment.get("id") or ""),
                        object_label=assignment_label,
                        message=str(
                            blocker.get("message")
                            or "Assignment is not ready for its current governance configuration."
                        ),
                        details=blocker,
                        navigate_to={
                            "tab": "assignments",
                            "object_kind": "policy_assignment",
                            "object_id": str(assignment.get("id") or ""),
                        },
                        related_object_kind=(
                            "workspace_set_object"
                            if assignment.get("workspace_source_mode") == "named"
                            and assignment.get("workspace_set_object_id") is not None
                            else None
                        ),
                        related_object_id=(
                            str(assignment.get("workspace_set_object_id"))
                            if assignment.get("workspace_source_mode") == "named"
                            and assignment.get("workspace_set_object_id") is not None
                            else None
                        ),
                    )
                )

            resolver_fn = getattr(self.external_access_resolver, "resolve", None)
            if callable(resolver_fn):
                external_access = await resolver_fn(
                    assignment_id=int(assignment.get("id")),
                    effective_policy={
                        "capabilities": list(
                            _as_dict(assignment.get("inline_policy_document")).get("capabilities") or []
                        )
                    },
                )
            else:
                fallback_resolver_fn = getattr(
                    self.external_access_resolver,
                    "resolve_assignment_external_access",
                    None,
                )
                if not callable(fallback_resolver_fn):
                    external_access = {"servers": []}
                else:
                    external_access = await fallback_resolver_fn(
                        assignment_id=int(assignment.get("id")),
                        owner_scope_type=str(assignment.get("owner_scope_type") or "global"),
                        owner_scope_id=assignment.get("owner_scope_id"),
                        profile_id=assignment.get("profile_id"),
                    )
            for server in list(external_access.get("servers") or []):
                blocked_reason = str(server.get("blocked_reason") or "").strip().lower()
                if blocked_reason not in {
                    "required_slot_not_granted",
                    "required_slot_secret_missing",
                    "disabled_by_assignment",
                    "missing_secret",
                }:
                    continue
                findings.append(
                    self._make_governance_audit_finding(
                        finding_type="external_binding_issue",
                        severity="error",
                        scope_type=str(assignment.get("owner_scope_type") or "global"),
                        scope_id=assignment.get("owner_scope_id"),
                        object_kind="policy_assignment",
                        object_id=str(assignment.get("id") or ""),
                        object_label=str(
                            assignment.get("target_id")
                            or assignment.get("target_type")
                            or assignment.get("id")
                            or ""
                        ),
                        message=str(
                            server.get("blocked_reason")
                            or "External binding issue detected for this assignment."
                        ),
                        details={
                            "server_id": server.get("server_id"),
                            "server_name": server.get("server_name"),
                            "requested_slots": list(server.get("requested_slots") or []),
                            "bound_slots": list(server.get("bound_slots") or []),
                            "missing_bound_slots": list(server.get("missing_bound_slots") or []),
                            "missing_secret_slots": list(server.get("missing_secret_slots") or []),
                            "blocked_reason": server.get("blocked_reason"),
                        },
                        navigate_to={
                            "tab": "assignments",
                            "object_kind": "policy_assignment",
                            "object_id": str(assignment.get("id") or ""),
                        },
                        related_object_kind="external_server",
                        related_object_id=str(server.get("server_id") or ""),
                        related_object_label=(
                            str(server.get("server_name") or "") or str(server.get("server_id") or "")
                        ),
                    )
                )

        external_servers = await self.list_external_servers(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        for server in external_servers:
            if str(server.get("server_source") or "managed") != "managed":
                continue
            blocked_reason = str(server.get("auth_template_blocked_reason") or "").strip()
            runtime_executable = bool(server.get("runtime_executable"))
            if runtime_executable and not blocked_reason:
                continue
            findings.append(
                self._make_governance_audit_finding(
                    finding_type="external_server_configuration_issue",
                    severity="error",
                    scope_type=str(server.get("owner_scope_type") or "global"),
                    scope_id=server.get("owner_scope_id"),
                    object_kind="external_server",
                    object_id=str(server.get("id") or ""),
                    object_label=str(server.get("name") or server.get("id") or ""),
                    message=blocked_reason or "Managed external server is not runtime executable.",
                    details={
                        "runtime_executable": runtime_executable,
                        "auth_template_present": bool(server.get("auth_template_present")),
                        "auth_template_valid": bool(server.get("auth_template_valid")),
                        "auth_template_blocked_reason": blocked_reason or None,
                    },
                    navigate_to={
                        "tab": "credentials",
                        "object_kind": "external_server",
                        "object_id": str(server.get("id") or ""),
                    },
                )
            )

        filtered = self._sort_governance_audit_findings(findings)
        if severity:
            filtered = [
                item for item in filtered if str(item.get("severity") or "").strip().lower() == str(severity).strip().lower()
            ]
        if finding_type:
            filtered = [
                item
                for item in filtered
                if str(item.get("finding_type") or "").strip().lower() == str(finding_type).strip().lower()
            ]
        if object_kind:
            filtered = [
                item
                for item in filtered
                if str(item.get("object_kind") or "").strip().lower() == str(object_kind).strip().lower()
            ]
        if scope_type:
            filtered = [
                item
                for item in filtered
                if str(item.get("scope_type") or "").strip().lower() == str(scope_type).strip().lower()
            ]

        counts = {"error": 0, "warning": 0}
        for item in filtered:
            severity_key = str(item.get("severity") or "warning").strip().lower()
            if severity_key in counts:
                counts[severity_key] += 1

        return {
            "items": filtered,
            "total": len(filtered),
            "counts": counts,
        }

    async def _validate_workspace_id_for_user(self, *, owner_scope_id: int | None, workspace_id: str) -> None:
        if owner_scope_id is None:
            raise BadRequestError("Workspace set objects require owner_scope_id")
        result = await self.workspace_root_resolver.resolve_for_context(
            session_id=None,
            user_id=str(owner_scope_id),
            workspace_id=str(workspace_id or "").strip(),
        )
        if not result.get("workspace_root"):
            raise BadRequestError(
                f"Workspace id '{workspace_id}' is not a trusted workspace for user {owner_scope_id}"
            )

    async def _validate_shared_workspace_id(
        self,
        *,
        owner_scope_type: str,
        owner_scope_id: int | None,
        workspace_id: str,
    ) -> None:
        entries = await self.repo.list_shared_workspace_entries(workspace_id=str(workspace_id or "").strip())
        compatible = [
            row
            for row in entries
            if bool(row.get("is_active", True))
            and self._shared_workspace_scope_compatible(
                target_scope_type=owner_scope_type,
                target_scope_id=owner_scope_id,
                entry_scope_type=str(row.get("owner_scope_type") or "global"),
                entry_scope_id=row.get("owner_scope_id"),
            )
        ]
        if not compatible:
            raise BadRequestError(
                f"Workspace id '{workspace_id}' is not a trusted shared workspace for scope "
                f"{owner_scope_type}:{owner_scope_id}"
            )

    async def _shared_workspace_is_referenced(self, row: dict[str, Any]) -> bool:
        workspace_id = str(row.get("workspace_id") or "").strip()
        if not workspace_id:
            return False
        workspace_sets = await self.repo.list_workspace_set_objects()
        for workspace_set in workspace_sets:
            scope_type = str(workspace_set.get("owner_scope_type") or "").strip().lower()
            if scope_type == "user":
                continue
            if not self._shared_workspace_scope_compatible(
                target_scope_type=scope_type,
                target_scope_id=workspace_set.get("owner_scope_id"),
                entry_scope_type=str(row.get("owner_scope_type") or "global"),
                entry_scope_id=row.get("owner_scope_id"),
            ):
                continue
            members = await self.repo.list_workspace_set_members(int(workspace_set.get("id") or 0))
            if any(str(member.get("workspace_id") or "").strip() == workspace_id for member in members):
                return True
        return False

    async def get_workspace_set_readiness_summary(
        self,
        *,
        workspace_set_object_id: int,
    ) -> dict[str, Any]:
        workspace_set = await self.repo.get_workspace_set_object(workspace_set_object_id)
        if not workspace_set:
            return _workspace_source_readiness_summary()
        owner_scope_type = str(workspace_set.get("owner_scope_type") or "user").strip().lower()
        owner_scope_id = workspace_set.get("owner_scope_id")
        trust_source = "shared_registry" if owner_scope_type != "user" else "user_local"
        workspace_ids = _unique(
            _as_str_list(
                [
                    row.get("workspace_id")
                    for row in await self.repo.list_workspace_set_members(workspace_set_object_id)
                ]
            )
        )
        resolved_roots: list[tuple[str, Path]] = []
        unresolved_workspace_ids: list[str] = []
        for workspace_id in workspace_ids:
            resolved = await self.workspace_root_resolver.resolve_for_context(
                session_id=None,
                user_id=str(owner_scope_id) if trust_source == "user_local" and owner_scope_id is not None else None,
                workspace_id=workspace_id,
                workspace_trust_source=trust_source,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
            )
            workspace_root = str(resolved.get("workspace_root") or "").strip()
            if not workspace_root:
                unresolved_workspace_ids.append(workspace_id)
                continue
            resolved_roots.append((workspace_id, Path(workspace_root).expanduser().resolve(strict=False)))

        for index, (left_workspace_id, left_root) in enumerate(resolved_roots):
            for right_workspace_id, right_root in resolved_roots[index + 1 :]:
                if not _roots_overlap(left_root, right_root):
                    continue
                return _workspace_source_readiness_summary(
                    warning_codes=["multi_root_overlap_warning"],
                    warning_message="May overlap with another trusted root in multi-root assignments.",
                    conflicting_workspace_ids=[left_workspace_id, right_workspace_id],
                    conflicting_workspace_roots=[str(left_root), str(right_root)],
                    unresolved_workspace_ids=unresolved_workspace_ids,
                )

        if unresolved_workspace_ids:
            return _workspace_source_readiness_summary(
                warning_codes=["workspace_unresolvable_warning"],
                warning_message="Contains workspace ids that do not currently resolve through the expected trust source.",
                unresolved_workspace_ids=unresolved_workspace_ids,
            )

        return _workspace_source_readiness_summary()

    async def get_shared_workspace_readiness_summary(
        self,
        *,
        shared_workspace_id: int,
    ) -> dict[str, Any]:
        row = await self.repo.get_shared_workspace_entry(shared_workspace_id)
        if not row:
            return _workspace_source_readiness_summary()
        current_root = str(row.get("absolute_root") or "").strip()
        if not current_root:
            return _workspace_source_readiness_summary(
                warning_codes=["workspace_unresolvable_warning"],
                warning_message="Shared workspace root is unavailable.",
                unresolved_workspace_ids=[str(row.get("workspace_id") or "")],
            )
        current_path = Path(current_root).expanduser().resolve(strict=False)
        scope_type = str(row.get("owner_scope_type") or "team").strip().lower()
        scope_id = row.get("owner_scope_id")
        visible_entries: list[dict[str, Any]] = []
        visible_entries.extend(
            await self.repo.list_shared_workspace_entries(
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
            )
        )
        if scope_type != "global":
            visible_entries.extend(
                await self.repo.list_shared_workspace_entries(
                    owner_scope_type="global",
                    owner_scope_id=None,
                )
            )
        seen: set[int] = set()
        for entry in visible_entries:
            entry_id = int(entry.get("id") or 0)
            if entry_id == int(shared_workspace_id) or entry_id in seen:
                continue
            seen.add(entry_id)
            candidate_root = str(entry.get("absolute_root") or "").strip()
            if not candidate_root:
                continue
            candidate_path = Path(candidate_root).expanduser().resolve(strict=False)
            if not _roots_overlap(current_path, candidate_path):
                continue
            return _workspace_source_readiness_summary(
                warning_codes=["multi_root_overlap_warning"],
                warning_message="May conflict with other visible shared roots in multi-root assignments.",
                conflicting_workspace_ids=[
                    str(row.get("workspace_id") or ""),
                    str(entry.get("workspace_id") or ""),
                ],
                conflicting_workspace_roots=[str(current_path), str(candidate_path)],
            )
        return _workspace_source_readiness_summary()

    @classmethod
    def _credential_slot_required_permission(cls, privilege_class: str) -> str:
        normalized = cls._normalize_credential_slot_privilege_class(privilege_class)
        return _CREDENTIAL_SLOT_REQUIRED_PERMISSIONS[normalized]

    @classmethod
    def _credential_slot_broadens(
        cls,
        previous_privilege_class: str | None,
        next_privilege_class: str,
    ) -> bool:
        if previous_privilege_class is None:
            return True
        previous = cls._normalize_credential_slot_privilege_class(previous_privilege_class)
        next_value = cls._normalize_credential_slot_privilege_class(next_privilege_class)
        return _CREDENTIAL_SLOT_PRIVILEGE_RANK[next_value] > _CREDENTIAL_SLOT_PRIVILEGE_RANK[previous]

    async def _binding_audit_privilege_metadata(
        self,
        *,
        external_server_id: str,
        slot_name: str | None,
    ) -> dict[str, Any]:
        slot: dict[str, Any] | None
        if slot_name:
            slot = await self.repo.get_external_server_credential_slot(
                server_id=external_server_id,
                slot_name=slot_name,
            )
        else:
            slot = await self.repo.get_external_server_default_slot(server_id=external_server_id)
        if not slot:
            return {}
        try:
            privilege_class = self._normalize_credential_slot_privilege_class(
                str(slot.get("privilege_class") or "")
            )
        except BadRequestError:
            return {"slot_name": str(slot.get("slot_name") or slot_name or "")}
        return {
            "slot_name": str(slot.get("slot_name") or slot_name or ""),
            "privilege_class": privilege_class,
            "required_permission": self._credential_slot_required_permission(privilege_class),
        }

    async def _list_legacy_inventory(self) -> list[dict[str, Any]]:
        if self.legacy_inventory_service is not None:
            return await self.legacy_inventory_service.list_inventory()
        return list(getattr(self, "_legacy_external_inventory", []) or [])

    async def _get_legacy_inventory_item(self, server_id: str) -> dict[str, Any] | None:
        if self.legacy_inventory_service is not None:
            return await self.legacy_inventory_service.get_inventory_item(server_id)
        inventory = await self._list_legacy_inventory()
        target = str(server_id or "").strip()
        return next((item for item in inventory if str(item.get("id") or "") == target), None)

    @staticmethod
    def _supported_auth_template_target_for_transport(transport: str) -> str | None:
        normalized = str(transport or "").strip().lower()
        if normalized == "websocket":
            return "header"
        if normalized == "stdio":
            return "env"
        return None

    async def _validate_managed_external_auth_template_config(
        self,
        *,
        server_id: str,
        transport: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        out = dict(config or {})
        auth = dict(out.get("auth") or {})
        mode = str(auth.get("mode") or "").strip().lower()
        raw_mappings = auth.get("mappings")

        if raw_mappings is None:
            return out
        if mode not in {"template"}:
            raise BadRequestError("Managed auth templates require auth.mode='template'")
        if not isinstance(raw_mappings, list) or not raw_mappings:
            raise BadRequestError("Managed auth template requires at least one mapping")

        expected_target = self._supported_auth_template_target_for_transport(transport)
        if expected_target is None:
            raise BadRequestError("Managed auth templates are supported only for websocket and stdio transports")

        slot_rows = await self.repo.list_external_server_credential_slots(server_id=server_id)
        valid_slots = {
            str(row.get("slot_name") or "").strip().lower()
            for row in slot_rows
            if str(row.get("slot_name") or "").strip()
        }
        seen_targets: set[tuple[str, str]] = set()
        normalized_mappings: list[dict[str, Any]] = []

        for raw_mapping in raw_mappings:
            if not isinstance(raw_mapping, dict):
                raise BadRequestError("Managed auth template mappings must be objects")

            slot_name = str(raw_mapping.get("slot_name") or "").strip().lower()
            if not slot_name:
                raise BadRequestError("Managed auth template mapping requires slot_name")
            if slot_name not in valid_slots:
                raise BadRequestError(f"Managed auth template references unknown slot: {slot_name}")

            target_type = str(raw_mapping.get("target_type") or "").strip().lower()
            if target_type not in {"header", "env"}:
                raise BadRequestError(f"Unsupported auth template target_type: {target_type or 'missing'}")
            if target_type != expected_target:
                raise BadRequestError("Managed auth template target_type is invalid for the server transport")

            target_name = str(raw_mapping.get("target_name") or "").strip()
            if not target_name:
                raise BadRequestError("Managed auth template mapping requires target_name")
            target_key = (target_type, target_name.lower())
            if target_key in seen_targets:
                raise BadRequestError("Managed auth template contains duplicate target mappings")
            seen_targets.add(target_key)

            normalized_mappings.append(
                {
                    "slot_name": slot_name,
                    "target_type": target_type,
                    "target_name": target_name,
                    "prefix": str(raw_mapping.get("prefix") or ""),
                    "suffix": str(raw_mapping.get("suffix") or ""),
                    "required": bool(raw_mapping.get("required", True)),
                }
            )

        auth["mode"] = "template"
        auth["mappings"] = normalized_mappings
        auth.pop("required_slots", None)
        auth.pop("slot_bindings", None)
        out["auth"] = auth
        return out

    async def _build_auth_template_status(
        self,
        *,
        row: dict[str, Any],
        slots: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result = {
            "auth_template_present": False,
            "auth_template_valid": False,
            "auth_template_blocked_reason": "no_auth_template",
        }
        if str(row.get("server_source") or "managed") != "managed":
            return result

        config = dict(row.get("config") or {})
        auth = dict(config.get("auth") or {})
        template_mappings = ManagedExternalAuthBridge._extract_template_mappings(auth)
        if not template_mappings:
            return result

        result["auth_template_present"] = True

        expected_target = self._supported_auth_template_target_for_transport(str(row.get("transport") or ""))
        if expected_target is None:
            result["auth_template_blocked_reason"] = "unsupported_template_transport_target"
            return result

        slot_lookup = {
            str(slot.get("slot_name") or "").strip().lower(): dict(slot)
            for slot in slots
            if str(slot.get("slot_name") or "").strip()
        }
        seen_targets: set[tuple[str, str]] = set()

        for raw_mapping in template_mappings:
            slot_name = str(raw_mapping.get("slot_name") or "").strip().lower()
            target_type = str(raw_mapping.get("target_type") or "").strip().lower()
            target_name = str(raw_mapping.get("target_name") or "").strip()
            if not slot_name or slot_name not in slot_lookup or not target_name:
                result["auth_template_blocked_reason"] = "auth_template_invalid"
                return result
            if target_type not in {"header", "env"}:
                result["auth_template_blocked_reason"] = "auth_template_invalid"
                return result
            if target_type != expected_target:
                result["auth_template_blocked_reason"] = "unsupported_template_transport_target"
                return result
            target_key = (target_type, target_name.lower())
            if target_key in seen_targets:
                result["auth_template_blocked_reason"] = "auth_template_invalid"
                return result
            seen_targets.add(target_key)

            if bool(raw_mapping.get("required", True)) and not bool(slot_lookup[slot_name].get("secret_configured")):
                result["auth_template_blocked_reason"] = "required_slot_secret_missing"
                return result

        result["auth_template_valid"] = True
        result["auth_template_blocked_reason"] = None
        return result

    async def _resolve_binding_target(
        self,
        *,
        binding_target_type: str,
        binding_target_id: int,
    ) -> dict[str, Any]:
        if binding_target_type == "profile":
            row = await self.repo.get_permission_profile(int(binding_target_id))
            if not row:
                raise ResourceNotFoundError("mcp_permission_profile", identifier=str(binding_target_id))
            return row
        if binding_target_type == "assignment":
            row = await self.repo.get_policy_assignment(int(binding_target_id))
            if not row:
                raise ResourceNotFoundError("mcp_policy_assignment", identifier=str(binding_target_id))
            return row
        raise BadRequestError("Invalid credential binding target")

    @staticmethod
    def _validate_binding_scope(
        *,
        server_row: dict[str, Any],
        target_row: dict[str, Any],
    ) -> None:
        server_scope_type = str(server_row.get("owner_scope_type") or "global")
        target_scope_type = str(target_row.get("owner_scope_type") or "global")
        server_rank = _SCOPE_RANK.get(server_scope_type, -1)
        target_rank = _SCOPE_RANK.get(target_scope_type, -1)
        if server_rank < 0 or target_rank < 0:
            raise BadRequestError("Invalid MCP Hub owner scope")
        if server_rank > target_rank:
            raise BadRequestError("Cannot bind a narrower-scope external server to a broader target")
        if server_rank == target_rank and server_scope_type != "global":
            if server_row.get("owner_scope_id") != target_row.get("owner_scope_id"):
                raise BadRequestError("Cannot bind an external server from a different owner scope")

    @staticmethod
    def _normalize_imported_external_config(config: dict[str, Any]) -> dict[str, Any]:
        out = dict(config or {})
        auth = dict(out.get("auth") or {})
        mode = str(auth.get("mode") or "").strip().lower()
        if mode == "bearer_env":
            auth = {"mode": "bearer_token"}
        elif mode == "api_key_env":
            auth = {
                "mode": "api_key_header",
                "api_key_header": str(auth.get("api_key_header") or "X-API-KEY"),
            }
        if auth:
            out["auth"] = auth
        return out

    async def _attach_slot_summary(self, row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        if str(out.get("server_source") or "managed") == "managed":
            out["credential_slots"] = await self.repo.list_external_server_credential_slots(
                server_id=str(out.get("id") or "")
            )
        else:
            out["credential_slots"] = []
        out.update(
            await self._build_auth_template_status(
                row=out,
                slots=list(out.get("credential_slots") or []),
            )
        )
        return out

    async def create_permission_profile(
        self,
        *,
        name: str,
        owner_scope_type: str,
        owner_scope_id: int | None,
        mode: str,
        path_scope_object_id: int | None,
        policy_document: dict[str, Any],
        actor_id: int | None,
        description: str | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        row = await self.repo.create_permission_profile(
            name=name,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            mode=mode,
            path_scope_object_id=path_scope_object_id,
            policy_document=policy_document,
            actor_id=actor_id,
            description=description,
            is_active=is_active,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.permission_profile.create",
                actor_id=actor_id,
                resource_type="mcp_permission_profile",
                resource_id=str(row.get("id") or ""),
                metadata={"name": row.get("name"), "owner_scope_type": row.get("owner_scope_type")},
            )
        )
        return row

    async def create_path_scope_object(
        self,
        *,
        name: str,
        owner_scope_type: str,
        owner_scope_id: int | None,
        path_scope_document: dict[str, Any],
        actor_id: int | None,
        description: str | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        row = await self.repo.create_path_scope_object(
            name=name,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            path_scope_document=path_scope_document,
            actor_id=actor_id,
            description=description,
            is_active=is_active,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.path_scope_object.create",
                actor_id=actor_id,
                resource_type="mcp_path_scope_object",
                resource_id=str(row.get("id") or ""),
                metadata={"name": row.get("name"), "owner_scope_type": row.get("owner_scope_type")},
            )
        )
        return row

    async def create_workspace_set_object(
        self,
        *,
        name: str,
        owner_scope_type: str,
        owner_scope_id: int | None,
        actor_id: int | None,
        description: str | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        scope_type = str(owner_scope_type or "").strip().lower()
        if scope_type == "user" and owner_scope_id is None:
            raise BadRequestError("Workspace set objects require owner_scope_id")
        if scope_type in {"org", "team"} and owner_scope_id is None:
            raise BadRequestError(f"{scope_type} workspace set objects require owner_scope_id")
        if scope_type not in _SCOPE_RANK:
            raise BadRequestError("Invalid workspace set object owner_scope_type")
        row = await self.repo.create_workspace_set_object(
            name=name,
            owner_scope_type=scope_type,
            owner_scope_id=owner_scope_id,
            actor_id=actor_id,
            description=description,
            is_active=is_active,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.workspace_set_object.create",
                actor_id=actor_id,
                resource_type="mcp_workspace_set_object",
                resource_id=str(row.get("id") or ""),
                metadata={"name": row.get("name"), "owner_scope_type": row.get("owner_scope_type")},
            )
        )
        return row

    async def list_workspace_set_objects(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self.repo.list_workspace_set_objects(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )

    async def get_workspace_set_object(self, workspace_set_object_id: int) -> dict[str, Any] | None:
        return await self.repo.get_workspace_set_object(workspace_set_object_id)

    async def update_workspace_set_object(
        self,
        workspace_set_object_id: int,
        *,
        actor_id: int | None = None,
        **update_fields: Any,
    ) -> dict[str, Any] | None:
        next_scope_type = update_fields.get("owner_scope_type")
        next_scope_id = update_fields.get("owner_scope_id")
        if next_scope_type is not None:
            normalized_scope = str(next_scope_type or "").strip().lower()
            if normalized_scope not in _SCOPE_RANK:
                raise BadRequestError("Invalid workspace set object owner_scope_type")
            if normalized_scope == "user" and next_scope_id is None:
                raise BadRequestError("Workspace set objects require owner_scope_id")
            if normalized_scope in {"org", "team"} and next_scope_id is None:
                raise BadRequestError(f"{normalized_scope} workspace set objects require owner_scope_id")
            update_fields["owner_scope_type"] = normalized_scope
        row = await self.repo.update_workspace_set_object(
            workspace_set_object_id,
            actor_id=actor_id,
            **update_fields,
        )
        if row:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.workspace_set_object.update",
                    actor_id=actor_id,
                    resource_type="mcp_workspace_set_object",
                    resource_id=str(row.get("id") or workspace_set_object_id),
                    metadata={"name": row.get("name"), "owner_scope_type": row.get("owner_scope_type")},
                )
            )
        return row

    async def delete_workspace_set_object(self, workspace_set_object_id: int, *, actor_id: int | None) -> bool:
        row = await self.repo.get_workspace_set_object(workspace_set_object_id)
        if not row:
            return False
        assignments = await self.repo.list_policy_assignments()
        if any(int(assignment.get("workspace_set_object_id") or 0) == int(workspace_set_object_id) for assignment in assignments):
            raise BadRequestError("Cannot delete a workspace set object while it is still referenced")
        deleted = await self.repo.delete_workspace_set_object(workspace_set_object_id)
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.workspace_set_object.delete",
                    actor_id=actor_id,
                    resource_type="mcp_workspace_set_object",
                    resource_id=str(workspace_set_object_id),
                )
            )
        return deleted

    async def list_workspace_set_members(self, workspace_set_object_id: int) -> list[dict[str, Any]]:
        return await self.repo.list_workspace_set_members(workspace_set_object_id)

    async def add_workspace_set_member(
        self,
        workspace_set_object_id: int,
        *,
        workspace_id: str,
        actor_id: int | None,
    ) -> dict[str, Any]:
        owner = await self.repo.get_workspace_set_object(workspace_set_object_id)
        if not owner:
            raise ResourceNotFoundError("mcp_workspace_set_object", identifier=str(workspace_set_object_id))
        owner_scope_type = str(owner.get("owner_scope_type") or "user")
        owner_scope_id = owner.get("owner_scope_id")
        if owner_scope_type == "user":
            await self._validate_workspace_id_for_user(
                owner_scope_id=owner_scope_id,
                workspace_id=str(workspace_id or "").strip(),
            )
        else:
            await self._validate_shared_workspace_id(
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
                workspace_id=str(workspace_id or "").strip(),
            )
        row = await self.repo.add_workspace_set_member(
            workspace_set_object_id,
            workspace_id=workspace_id,
            actor_id=actor_id,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.workspace_set_object.member.add",
                actor_id=actor_id,
                resource_type="mcp_workspace_set_object",
                resource_id=str(workspace_set_object_id),
                metadata={"workspace_id": workspace_id},
            )
        )
        return row

    async def delete_workspace_set_member(
        self,
        workspace_set_object_id: int,
        workspace_id: str,
        *,
        actor_id: int | None,
    ) -> bool:
        deleted = await self.repo.delete_workspace_set_member(workspace_set_object_id, workspace_id)
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.workspace_set_object.member.delete",
                    actor_id=actor_id,
                    resource_type="mcp_workspace_set_object",
                    resource_id=str(workspace_set_object_id),
                    metadata={"workspace_id": workspace_id},
                )
            )
        return deleted

    async def create_shared_workspace_entry(
        self,
        *,
        workspace_id: str,
        display_name: str,
        absolute_root: str,
        owner_scope_type: str,
        owner_scope_id: int | None,
        actor_id: int | None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        scope_type, scope_id = self._normalize_shared_workspace_scope(owner_scope_type, owner_scope_id)
        row = await self.repo.create_shared_workspace_entry(
            workspace_id=str(workspace_id or "").strip(),
            display_name=str(display_name or "").strip(),
            absolute_root=self._normalize_shared_workspace_root(absolute_root),
            owner_scope_type=scope_type,
            owner_scope_id=scope_id,
            actor_id=actor_id,
            is_active=is_active,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.shared_workspace.create",
                actor_id=actor_id,
                resource_type="mcp_shared_workspace",
                resource_id=str(row.get("id") or ""),
                metadata={"workspace_id": row.get("workspace_id"), "owner_scope_type": row.get("owner_scope_type")},
            )
        )
        return row

    async def list_shared_workspace_entries(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self.repo.list_shared_workspace_entries(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            workspace_id=workspace_id,
        )

    async def get_shared_workspace_entry(self, shared_workspace_id: int) -> dict[str, Any] | None:
        return await self.repo.get_shared_workspace_entry(shared_workspace_id)

    async def update_shared_workspace_entry(
        self,
        shared_workspace_id: int,
        *,
        actor_id: int | None = None,
        **update_fields: Any,
    ) -> dict[str, Any] | None:
        existing = await self.repo.get_shared_workspace_entry(shared_workspace_id)
        if not existing:
            return None
        scope_type = update_fields.get("owner_scope_type", existing.get("owner_scope_type"))
        scope_id = update_fields.get("owner_scope_id", existing.get("owner_scope_id"))
        normalized_scope_type, normalized_scope_id = self._normalize_shared_workspace_scope(
            str(scope_type or ""),
            scope_id,
        )
        update_fields["owner_scope_type"] = normalized_scope_type
        update_fields["owner_scope_id"] = normalized_scope_id
        if "absolute_root" in update_fields and update_fields.get("absolute_root") is not None:
            update_fields["absolute_root"] = self._normalize_shared_workspace_root(
                str(update_fields.get("absolute_root") or "")
            )
        referenced = await self._shared_workspace_is_referenced(existing)
        if referenced:
            if "workspace_id" in update_fields and str(update_fields.get("workspace_id") or "").strip() != str(existing.get("workspace_id") or "").strip():
                raise BadRequestError("Cannot change workspace_id while the shared workspace is still referenced")
            if normalized_scope_type != str(existing.get("owner_scope_type") or "") or normalized_scope_id != existing.get("owner_scope_id"):
                raise BadRequestError("Cannot change shared workspace scope while it is still referenced")
        row = await self.repo.update_shared_workspace_entry(
            shared_workspace_id,
            actor_id=actor_id,
            **update_fields,
        )
        if row:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.shared_workspace.update",
                    actor_id=actor_id,
                    resource_type="mcp_shared_workspace",
                    resource_id=str(row.get("id") or shared_workspace_id),
                    metadata={"workspace_id": row.get("workspace_id"), "owner_scope_type": row.get("owner_scope_type")},
                )
            )
        return row

    async def delete_shared_workspace_entry(self, shared_workspace_id: int, *, actor_id: int | None) -> bool:
        row = await self.repo.get_shared_workspace_entry(shared_workspace_id)
        if not row:
            return False
        if await self._shared_workspace_is_referenced(row):
            raise BadRequestError("Cannot delete a shared workspace while it is still referenced")
        deleted = await self.repo.delete_shared_workspace_entry(shared_workspace_id)
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.shared_workspace.delete",
                    actor_id=actor_id,
                    resource_type="mcp_shared_workspace",
                    resource_id=str(shared_workspace_id),
                )
            )
        return deleted

    async def list_path_scope_objects(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self.repo.list_path_scope_objects(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )

    async def get_path_scope_object(self, path_scope_object_id: int) -> dict[str, Any] | None:
        return await self.repo.get_path_scope_object(path_scope_object_id)

    async def update_path_scope_object(
        self,
        path_scope_object_id: int,
        *,
        actor_id: int | None = None,
        **update_fields: Any,
    ) -> dict[str, Any] | None:
        row = await self.repo.update_path_scope_object(
            path_scope_object_id,
            actor_id=actor_id,
            **update_fields,
        )
        if row:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.path_scope_object.update",
                    actor_id=actor_id,
                    resource_type="mcp_path_scope_object",
                    resource_id=str(row.get("id") or path_scope_object_id),
                    metadata={"name": row.get("name"), "owner_scope_type": row.get("owner_scope_type")},
                )
            )
        return row

    async def delete_path_scope_object(self, path_scope_object_id: int, *, actor_id: int | None) -> bool:
        deleted = await self.repo.delete_path_scope_object(path_scope_object_id)
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.path_scope_object.delete",
                    actor_id=actor_id,
                    resource_type="mcp_path_scope_object",
                    resource_id=str(path_scope_object_id),
                )
            )
        return deleted

    async def list_permission_profiles(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self.repo.list_permission_profiles(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )

    async def get_permission_profile(self, profile_id: int) -> dict[str, Any] | None:
        """Fetch a single permission profile by id."""
        return await self.repo.get_permission_profile(profile_id)

    async def update_permission_profile(
        self,
        profile_id: int,
        *,
        actor_id: int | None = None,
        **update_fields: Any,
    ) -> dict[str, Any] | None:
        row = await self.repo.update_permission_profile(
            profile_id,
            actor_id=actor_id,
            **update_fields,
        )
        if row:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.permission_profile.update",
                    actor_id=actor_id,
                    resource_type="mcp_permission_profile",
                    resource_id=str(row.get("id") or profile_id),
                    metadata={"name": row.get("name"), "owner_scope_type": row.get("owner_scope_type")},
                )
            )
        return row

    async def delete_permission_profile(self, profile_id: int, *, actor_id: int | None) -> bool:
        deleted = await self.repo.delete_permission_profile(profile_id)
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.permission_profile.delete",
                    actor_id=actor_id,
                    resource_type="mcp_permission_profile",
                    resource_id=str(profile_id),
                    metadata=None,
                )
            )
        return deleted

    async def create_policy_assignment(
        self,
        *,
        target_type: str,
        target_id: str | None,
        owner_scope_type: str,
        owner_scope_id: int | None,
        profile_id: int | None,
        path_scope_object_id: int | None,
        workspace_source_mode: str | None,
        workspace_set_object_id: int | None,
        inline_policy_document: dict[str, Any],
        approval_policy_id: int | None,
        actor_id: int | None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        await self.validate_multi_root_assignment_readiness(
            actor_id=actor_id,
            assignment_id=None,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            profile_id=profile_id,
            path_scope_object_id=path_scope_object_id,
            inline_policy_document=inline_policy_document,
            workspace_source_mode=workspace_source_mode,
            workspace_set_object_id=workspace_set_object_id,
            inline_workspace_ids=None,
        )
        row = await self.repo.create_policy_assignment(
            target_type=target_type,
            target_id=target_id,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            profile_id=profile_id,
            path_scope_object_id=path_scope_object_id,
            workspace_source_mode=workspace_source_mode,
            workspace_set_object_id=workspace_set_object_id,
            inline_policy_document=inline_policy_document,
            approval_policy_id=approval_policy_id,
            actor_id=actor_id,
            is_active=is_active,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.policy_assignment.create",
                actor_id=actor_id,
                resource_type="mcp_policy_assignment",
                resource_id=str(row.get("id") or ""),
                metadata={"target_type": row.get("target_type"), "target_id": row.get("target_id")},
            )
        )
        return row

    async def list_policy_assignments(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return await self.repo.list_policy_assignments(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            target_type=target_type,
            target_id=target_id,
        )

    async def get_policy_assignment(self, assignment_id: int) -> dict[str, Any] | None:
        """Fetch a single policy assignment by id."""
        return await self.repo.get_policy_assignment(assignment_id)

    async def update_policy_assignment(
        self,
        assignment_id: int,
        *,
        actor_id: int | None = None,
        **update_fields: Any,
    ) -> dict[str, Any] | None:
        existing = await self.repo.get_policy_assignment(assignment_id)
        if existing is None:
            return None
        await self.validate_multi_root_assignment_readiness(
            actor_id=actor_id,
            assignment_id=assignment_id,
            owner_scope_type=str(update_fields.get("owner_scope_type") or existing.get("owner_scope_type") or "global"),
            owner_scope_id=update_fields.get("owner_scope_id", existing.get("owner_scope_id")),
            profile_id=update_fields.get("profile_id", existing.get("profile_id")),
            path_scope_object_id=update_fields.get("path_scope_object_id", existing.get("path_scope_object_id")),
            inline_policy_document=update_fields.get(
                "inline_policy_document",
                existing.get("inline_policy_document") or {},
            ),
            workspace_source_mode=update_fields.get(
                "workspace_source_mode",
                existing.get("workspace_source_mode"),
            ),
            workspace_set_object_id=update_fields.get(
                "workspace_set_object_id",
                existing.get("workspace_set_object_id"),
            ),
            inline_workspace_ids=None,
        )
        row = await self.repo.update_policy_assignment(
            assignment_id,
            actor_id=actor_id,
            **update_fields,
        )
        if row:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.policy_assignment.update",
                    actor_id=actor_id,
                    resource_type="mcp_policy_assignment",
                    resource_id=str(row.get("id") or assignment_id),
                    metadata={"target_type": row.get("target_type"), "target_id": row.get("target_id")},
                )
            )
        return row

    async def delete_policy_assignment(self, assignment_id: int, *, actor_id: int | None) -> bool:
        existing_override = await self.repo.get_policy_override_by_assignment(assignment_id)
        if existing_override:
            await self.repo.delete_policy_override_by_assignment(assignment_id)
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.policy_override.delete",
                    actor_id=actor_id,
                    resource_type="mcp_policy_override",
                    resource_id=str(existing_override.get("id") or ""),
                    metadata={"assignment_id": assignment_id},
                )
            )
        deleted = await self.repo.delete_policy_assignment(assignment_id)
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.policy_assignment.delete",
                    actor_id=actor_id,
                    resource_type="mcp_policy_assignment",
                    resource_id=str(assignment_id),
                    metadata=None,
                )
            )
        return deleted

    async def list_policy_assignment_workspaces(self, assignment_id: int) -> list[dict[str, Any]]:
        return await self.repo.list_policy_assignment_workspaces(assignment_id)

    async def add_policy_assignment_workspace(
        self,
        assignment_id: int,
        *,
        workspace_id: str,
        actor_id: int | None,
    ) -> dict[str, Any]:
        workspace_value = str(workspace_id or "").strip()
        assignment = await self.repo.get_policy_assignment(assignment_id)
        if not assignment:
            raise ResourceNotFoundError("mcp_policy_assignment", identifier=str(assignment_id))
        existing = await self.repo.list_policy_assignment_workspaces(assignment_id)
        if any(str(row.get("workspace_id") or "").strip() == workspace_value for row in existing):
            raise McpHubConflictError("Workspace already attached to assignment")
        row = await self.repo.add_policy_assignment_workspace(
            assignment_id,
            workspace_id=workspace_value,
            actor_id=actor_id,
        )
        try:
            await self.validate_multi_root_assignment_readiness(
                actor_id=actor_id,
                assignment_id=assignment_id,
                owner_scope_type=str(assignment.get("owner_scope_type") or "global"),
                owner_scope_id=assignment.get("owner_scope_id"),
                profile_id=assignment.get("profile_id"),
                path_scope_object_id=assignment.get("path_scope_object_id"),
                inline_policy_document=assignment.get("inline_policy_document") or {},
                workspace_source_mode=assignment.get("workspace_source_mode"),
                workspace_set_object_id=assignment.get("workspace_set_object_id"),
                inline_workspace_ids=[
                    str(existing_row.get("workspace_id") or "").strip()
                    for existing_row in existing
                ]
                + [workspace_value],
            )
        except BadRequestError:
            await self.repo.delete_policy_assignment_workspace(assignment_id, workspace_value)
            raise
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.policy_assignment_workspace.create",
                actor_id=actor_id,
                resource_type="mcp_policy_assignment_workspace",
                resource_id=f"{assignment_id}:{workspace_value}",
                metadata={"assignment_id": assignment_id, "workspace_id": workspace_value},
            )
        )
        return row

    async def delete_policy_assignment_workspace(
        self,
        assignment_id: int,
        *,
        workspace_id: str,
        actor_id: int | None,
    ) -> bool:
        assignment = await self.repo.get_policy_assignment(assignment_id)
        if not assignment:
            raise ResourceNotFoundError("mcp_policy_assignment", identifier=str(assignment_id))
        existing = await self.repo.list_policy_assignment_workspaces(assignment_id)
        next_workspace_ids = [
            str(row.get("workspace_id") or "").strip()
            for row in existing
            if str(row.get("workspace_id") or "").strip() != str(workspace_id or "").strip()
        ]
        await self.validate_multi_root_assignment_readiness(
            actor_id=actor_id,
            assignment_id=assignment_id,
            owner_scope_type=str(assignment.get("owner_scope_type") or "global"),
            owner_scope_id=assignment.get("owner_scope_id"),
            profile_id=assignment.get("profile_id"),
            path_scope_object_id=assignment.get("path_scope_object_id"),
            inline_policy_document=assignment.get("inline_policy_document") or {},
            workspace_source_mode=assignment.get("workspace_source_mode"),
            workspace_set_object_id=assignment.get("workspace_set_object_id"),
            inline_workspace_ids=next_workspace_ids,
        )
        deleted = await self.repo.delete_policy_assignment_workspace(assignment_id, workspace_id)
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.policy_assignment_workspace.delete",
                    actor_id=actor_id,
                    resource_type="mcp_policy_assignment_workspace",
                    resource_id=f"{assignment_id}:{workspace_id}",
                    metadata={"assignment_id": assignment_id, "workspace_id": workspace_id},
                )
            )
        return deleted

    async def get_policy_override(self, assignment_id: int) -> dict[str, Any] | None:
        """Fetch a single assignment-bound override by assignment id."""
        return await self.repo.get_policy_override_by_assignment(assignment_id)

    async def upsert_policy_override(
        self,
        assignment_id: int,
        *,
        override_policy_document: dict[str, Any],
        is_active: bool,
        broadens_access: bool,
        grant_authority_snapshot: dict[str, Any],
        actor_id: int | None,
    ) -> dict[str, Any] | None:
        row = await self.repo.upsert_policy_override(
            assignment_id,
            override_policy_document=override_policy_document,
            is_active=is_active,
            broadens_access=broadens_access,
            grant_authority_snapshot=grant_authority_snapshot,
            actor_id=actor_id,
        )
        if row:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.policy_override.upsert",
                    actor_id=actor_id,
                    resource_type="mcp_policy_override",
                    resource_id=str(row.get("id") or ""),
                    metadata={
                        "assignment_id": assignment_id,
                        "broadens_access": bool(row.get("broadens_access")),
                        "is_active": bool(row.get("is_active")),
                    },
                )
            )
        return row

    async def delete_policy_override(self, assignment_id: int, *, actor_id: int | None) -> bool:
        existing = await self.repo.get_policy_override_by_assignment(assignment_id)
        deleted = await self.repo.delete_policy_override_by_assignment(assignment_id)
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.policy_override.delete",
                    actor_id=actor_id,
                    resource_type="mcp_policy_override",
                    resource_id=str((existing or {}).get("id") or ""),
                    metadata={"assignment_id": assignment_id},
                )
            )
        return deleted

    async def create_approval_policy(
        self,
        *,
        name: str,
        owner_scope_type: str,
        owner_scope_id: int | None,
        mode: str,
        rules: dict[str, Any],
        actor_id: int | None,
        description: str | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        row = await self.repo.create_approval_policy(
            name=name,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            mode=mode,
            rules=rules,
            actor_id=actor_id,
            description=description,
            is_active=is_active,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.approval_policy.create",
                actor_id=actor_id,
                resource_type="mcp_approval_policy",
                resource_id=str(row.get("id") or ""),
                metadata={"name": row.get("name"), "mode": row.get("mode")},
            )
        )
        return row

    async def list_approval_policies(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self.repo.list_approval_policies(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )

    async def get_approval_policy(self, approval_policy_id: int) -> dict[str, Any] | None:
        """Fetch a single approval policy by id."""
        return await self.repo.get_approval_policy(approval_policy_id)

    async def update_approval_policy(
        self,
        approval_policy_id: int,
        *,
        actor_id: int | None = None,
        **update_fields: Any,
    ) -> dict[str, Any] | None:
        row = await self.repo.update_approval_policy(
            approval_policy_id,
            actor_id=actor_id,
            **update_fields,
        )
        if row:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.approval_policy.update",
                    actor_id=actor_id,
                    resource_type="mcp_approval_policy",
                    resource_id=str(row.get("id") or approval_policy_id),
                    metadata={"name": row.get("name"), "mode": row.get("mode")},
                )
            )
        return row

    async def delete_approval_policy(self, approval_policy_id: int, *, actor_id: int | None) -> bool:
        deleted = await self.repo.delete_approval_policy(approval_policy_id)
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.approval_policy.delete",
                    actor_id=actor_id,
                    resource_type="mcp_approval_policy",
                    resource_id=str(approval_policy_id),
                    metadata=None,
                )
            )
        return deleted

    async def record_approval_decision(
        self,
        *,
        approval_policy_id: int | None,
        context_key: str,
        conversation_id: str | None,
        tool_name: str,
        scope_key: str,
        decision: str,
        consume_on_match: bool = False,
        expires_at: Any = None,
        actor_id: int | None = None,
    ) -> dict[str, Any]:
        row = await self.repo.create_approval_decision(
            approval_policy_id=approval_policy_id,
            context_key=context_key,
            conversation_id=conversation_id,
            tool_name=tool_name,
            scope_key=scope_key,
            decision=decision,
            consume_on_match=consume_on_match,
            expires_at=expires_at,
            actor_id=actor_id,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.approval_decision.create",
                actor_id=actor_id,
                resource_type="mcp_approval_decision",
                resource_id=str(row.get("id") or ""),
                metadata={
                    "approval_policy_id": approval_policy_id,
                    "tool_name": tool_name,
                    "decision": decision,
                },
            )
        )
        return row

    async def create_acp_profile(
        self,
        *,
        name: str,
        owner_scope_type: str,
        owner_scope_id: int | None,
        profile: dict[str, Any],
        actor_id: int | None,
        description: str | None = None,
        is_active: bool = True,
    ) -> dict[str, Any]:
        row = await self.repo.create_acp_profile(
            name=name,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            profile_json=json.dumps(profile or {}),
            actor_id=actor_id,
            description=description,
            is_active=is_active,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.acp_profile.create",
                actor_id=actor_id,
                resource_type="mcp_acp_profile",
                resource_id=str(row.get("id") or ""),
                metadata={"name": row.get("name"), "owner_scope_type": row.get("owner_scope_type")},
            )
        )
        return row

    async def list_acp_profiles(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self.repo.list_acp_profiles(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )

    async def get_acp_profile(self, profile_id: int) -> dict[str, Any] | None:
        row = await self.repo.get_acp_profile(int(profile_id))
        if not row:
            return None
        normalized = dict(row)
        try:
            normalized["profile"] = json.loads(str(normalized.get("profile_json") or "{}"))
        except json.JSONDecodeError:
            normalized["profile"] = {}
        return normalized

    async def update_acp_profile(
        self,
        profile_id: int,
        *,
        name: str | None = None,
        description: str | None = None,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
        profile: dict[str, Any] | None = None,
        is_active: bool | None = None,
        actor_id: int | None = None,
    ) -> dict[str, Any] | None:
        row = await self.repo.update_acp_profile(
            profile_id,
            name=name,
            description=description,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            profile_json=json.dumps(profile) if profile is not None else None,
            is_active=is_active,
            actor_id=actor_id,
        )
        if row:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.acp_profile.update",
                    actor_id=actor_id,
                    resource_type="mcp_acp_profile",
                    resource_id=str(row.get("id") or profile_id),
                    metadata={"name": row.get("name"), "owner_scope_type": row.get("owner_scope_type")},
                )
            )
        return row

    async def delete_acp_profile(self, profile_id: int, *, actor_id: int | None) -> bool:
        deleted = await self.repo.delete_acp_profile(profile_id)
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.acp_profile.delete",
                    actor_id=actor_id,
                    resource_type="mcp_acp_profile",
                    resource_id=str(profile_id),
                    metadata=None,
                )
            )
        return deleted

    async def create_external_server(
        self,
        *,
        server_id: str,
        name: str,
        transport: str,
        config: dict[str, Any],
        owner_scope_type: str,
        owner_scope_id: int | None,
        enabled: bool,
        actor_id: int | None,
        allow_existing: bool = False,
    ) -> dict[str, Any]:
        """Create an external server definition, optionally allowing idempotent update behavior."""
        previous = await self.repo.get_external_server(server_id)
        if previous is not None and not allow_existing:
            raise McpHubConflictError(f"External server already exists: {server_id}")
        normalized_config = await self._validate_managed_external_auth_template_config(
            server_id=server_id,
            transport=transport,
            config=dict(config or {}),
        )
        row = await self.repo.upsert_external_server(
            server_id=server_id,
            name=name,
            transport=transport,
            config_json=json.dumps(normalized_config),
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            enabled=enabled,
            server_source="managed",
            actor_id=actor_id,
        )
        action = (
            "mcp_hub.external_server.create"
            if previous is None
            else "mcp_hub.external_server.update"
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action=action,
                actor_id=actor_id,
                resource_type="mcp_external_server",
                resource_id=server_id,
                metadata={
                    "name": row.get("name"),
                    "transport": row.get("transport"),
                    "enabled": row.get("enabled"),
                },
            )
        )
        return row

    async def update_external_server(
        self,
        server_id: str,
        *,
        name: str | None = None,
        transport: str | None = None,
        config: dict[str, Any] | None = None,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
        enabled: bool | None = None,
        actor_id: int | None,
    ) -> dict[str, Any]:
        """Update an existing external server definition."""
        existing = await self.repo.get_external_server(server_id)
        if not existing:
            raise ResourceNotFoundError("mcp_external_server", identifier=server_id)
        existing_config: dict[str, Any] = {}
        raw_config = existing.get("config_json")
        if isinstance(raw_config, dict):
            existing_config = dict(raw_config)
        elif isinstance(raw_config, str) and raw_config.strip():
            try:
                parsed = json.loads(raw_config)
                if isinstance(parsed, dict):
                    existing_config = parsed
            except (TypeError, ValueError):
                existing_config = {}
        next_transport = transport if transport is not None else str(existing.get("transport") or "")
        next_config = await self._validate_managed_external_auth_template_config(
            server_id=server_id,
            transport=next_transport,
            config=dict(config if config is not None else existing_config),
        )
        row = await self.repo.update_external_server(
            server_id,
            name=name if name is not None else str(existing.get("name") or ""),
            transport=next_transport,
            config_json=json.dumps(next_config),
            owner_scope_type=(
                owner_scope_type
                if owner_scope_type is not None
                else str(existing.get("owner_scope_type") or "global")
            ),
            owner_scope_id=owner_scope_id if owner_scope_id is not None else existing.get("owner_scope_id"),
            enabled=enabled if enabled is not None else bool(existing.get("enabled")),
            server_source=str(existing.get("server_source") or "managed"),
            legacy_source_ref=existing.get("legacy_source_ref"),
            superseded_by_server_id=existing.get("superseded_by_server_id"),
            actor_id=actor_id,
        )
        if not row:
            raise ResourceNotFoundError("mcp_external_server", identifier=server_id)
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.external_server.update",
                actor_id=actor_id,
                resource_type="mcp_external_server",
                resource_id=server_id,
                metadata={
                    "name": row.get("name"),
                    "transport": row.get("transport"),
                    "enabled": row.get("enabled"),
                },
            )
        )
        return row

    async def list_external_servers(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        rows = await self.repo.list_external_servers(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        managed_ids = {str(row.get("id") or "") for row in rows}
        legacy_rows = [
            {
                "id": str(item.get("id") or ""),
                "name": str(item.get("name") or ""),
                "enabled": bool(item.get("enabled", True)),
                "owner_scope_type": str(item.get("owner_scope_type") or owner_scope_type or "global"),
                "owner_scope_id": item.get("owner_scope_id", owner_scope_id),
                "transport": str(item.get("transport") or ""),
                "config_json": json.dumps(item.get("config") or {}),
                "config": dict(item.get("config") or {}),
                "secret_configured": bool(),
                "key_hint": None,
                "server_source": "legacy",
                "legacy_source_ref": item.get("legacy_source_ref"),
                "superseded_by_server_id": item.get("superseded_by_server_id"),
                "binding_count": 0,
                "runtime_executable": False,
                "auth_template_present": False,
                "auth_template_valid": False,
                "auth_template_blocked_reason": "no_auth_template",
                "credential_slots": [],
                "created_by": None,
                "updated_by": None,
                "created_at": None,
                "updated_at": None,
            }
            for item in await self._list_legacy_inventory()
            if str(item.get("id") or "") not in managed_ids
        ]
        managed_rows = [await self._attach_slot_summary(row) for row in rows]
        return [*managed_rows, *legacy_rows]

    async def list_external_server_credential_slots(
        self,
        *,
        server_id: str,
    ) -> list[dict[str, Any]]:
        server = await self.repo.get_external_server(server_id)
        if not server:
            raise ResourceNotFoundError("mcp_external_server", identifier=server_id)
        return await self.repo.list_external_server_credential_slots(server_id=server_id)

    async def create_external_server_credential_slot(
        self,
        *,
        server_id: str,
        slot_name: str,
        display_name: str,
        secret_kind: str,
        privilege_class: str,
        is_required: bool,
        actor_id: int | None,
    ) -> dict[str, Any]:
        server = await self.repo.get_external_server(server_id)
        if not server:
            raise ResourceNotFoundError("mcp_external_server", identifier=server_id)
        normalized_privilege_class = self._normalize_credential_slot_privilege_class(
            privilege_class
        )
        row = await self.repo.create_external_server_credential_slot(
            server_id=server_id,
            slot_name=slot_name,
            display_name=display_name,
            secret_kind=secret_kind,
            privilege_class=normalized_privilege_class,
            is_required=is_required,
            actor_id=actor_id,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.external_server_slot.create",
                actor_id=actor_id,
                resource_type="mcp_external_server",
                resource_id=server_id,
                metadata={
                    "slot_name": row.get("slot_name"),
                    "privilege_class": normalized_privilege_class,
                    "required_permission": self._credential_slot_required_permission(
                        normalized_privilege_class
                    ),
                },
            )
        )
        return row

    async def update_external_server_credential_slot(
        self,
        *,
        server_id: str,
        slot_name: str,
        display_name: str | None = None,
        secret_kind: str | None = None,
        privilege_class: str | None = None,
        is_required: bool | None = None,
        actor_id: int | None,
    ) -> dict[str, Any]:
        existing = await self.repo.get_external_server_credential_slot(
            server_id=server_id,
            slot_name=slot_name,
        )
        previous_privilege_class: str | None = None
        if existing is not None and existing.get("privilege_class") is not None:
            previous_privilege_class = str(existing.get("privilege_class") or "").strip().lower() or None
        kwargs: dict[str, Any] = {
            "server_id": server_id,
            "slot_name": slot_name,
            "actor_id": actor_id,
        }
        next_privilege_class = previous_privilege_class
        if display_name is not None:
            kwargs["display_name"] = display_name
        if secret_kind is not None:
            kwargs["secret_kind"] = secret_kind
        if privilege_class is not None:
            next_privilege_class = self._normalize_credential_slot_privilege_class(privilege_class)
            kwargs["privilege_class"] = next_privilege_class
        if is_required is not None:
            kwargs["is_required"] = is_required
        row = await self.repo.update_external_server_credential_slot(**kwargs)
        if not row:
            raise ResourceNotFoundError("mcp_external_server_slot", identifier=f"{server_id}/{slot_name}")
        metadata: dict[str, Any] = {
            "slot_name": row.get("slot_name"),
            "privilege_class": str(row.get("privilege_class") or next_privilege_class or ""),
        }
        if next_privilege_class is not None and self._credential_slot_broadens(
            previous_privilege_class,
            next_privilege_class,
        ):
            metadata["required_permission"] = self._credential_slot_required_permission(
                next_privilege_class
            )
        if previous_privilege_class is not None and next_privilege_class is not None:
            metadata["previous_privilege_class"] = previous_privilege_class
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.external_server_slot.update",
                actor_id=actor_id,
                resource_type="mcp_external_server",
                resource_id=server_id,
                metadata=metadata,
            )
        )
        return row

    async def delete_external_server_credential_slot(
        self,
        *,
        server_id: str,
        slot_name: str,
        actor_id: int | None,
    ) -> bool:
        deleted = await self.repo.delete_external_server_credential_slot(
            server_id=server_id,
            slot_name=slot_name,
        )
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.external_server_slot.delete",
                    actor_id=actor_id,
                    resource_type="mcp_external_server",
                    resource_id=server_id,
                    metadata={"slot_name": slot_name},
                )
            )
        return deleted

    async def set_external_server_slot_secret(
        self,
        *,
        server_id: str,
        slot_name: str,
        secret_value: str,
        actor_id: int | None,
    ) -> dict[str, Any]:
        slot = await self.repo.get_external_server_credential_slot(server_id=server_id, slot_name=slot_name)
        if not slot:
            raise ResourceNotFoundError("mcp_external_server_slot", identifier=f"{server_id}/{slot_name}")
        secret = (secret_value or "").strip()
        if not secret:
            raise BadRequestError("Secret value is required")
        secret_payload = build_secret_payload(secret)
        envelope = encrypt_byok_payload(secret_payload)
        stored = await self.repo.upsert_external_server_slot_secret(
            server_id=server_id,
            slot_name=slot_name,
            encrypted_blob=dumps_envelope(envelope),
            key_hint=key_hint_for_api_key(secret),
            actor_id=actor_id,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.external_server_slot_secret.update",
                actor_id=actor_id,
                resource_type="mcp_external_server",
                resource_id=server_id,
                metadata={"slot_name": slot_name, "key_hint": stored.get("key_hint")},
            )
        )
        return {
            "server_id": server_id,
            "slot_name": slot_name,
            "secret_configured": bool(stored),
            "key_hint": stored.get("key_hint"),
            "updated_at": stored.get("updated_at"),
        }

    async def clear_external_server_slot_secret(
        self,
        *,
        server_id: str,
        slot_name: str,
        actor_id: int | None,
    ) -> bool:
        cleared = await self.repo.clear_external_server_slot_secret(server_id=server_id, slot_name=slot_name)
        if cleared:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.external_server_slot_secret.clear",
                    actor_id=actor_id,
                    resource_type="mcp_external_server",
                    resource_id=server_id,
                    metadata={"slot_name": slot_name},
                )
            )
        return cleared

    async def get_external_server_auth_template(
        self,
        *,
        server_id: str,
    ) -> dict[str, Any]:
        server = await self.repo.get_external_server(server_id)
        if not server:
            raise ResourceNotFoundError("mcp_external_server", identifier=server_id)
        config = dict(server.get("config") or {})
        auth = dict(config.get("auth") or {})
        try:
            mappings = ManagedExternalAuthBridge._extract_template_mappings(auth) or []
        except ValueError as exc:
            raise BadRequestError(str(exc)) from exc
        return {
            "mode": "template",
            "mappings": mappings,
        }

    async def update_external_server_auth_template(
        self,
        *,
        server_id: str,
        auth_template: dict[str, Any],
        actor_id: int | None,
    ) -> dict[str, Any]:
        server = await self.repo.get_external_server(server_id)
        if not server:
            raise ResourceNotFoundError("mcp_external_server", identifier=server_id)
        existing_config: dict[str, Any] = dict(server.get("config") or {})
        next_config = dict(existing_config)
        next_config["auth"] = {
            "mode": "template",
            "mappings": list(auth_template.get("mappings") or []),
        }
        await self.update_external_server(
            server_id,
            config=next_config,
            actor_id=actor_id,
        )
        return await self.get_external_server_auth_template(server_id=server_id)

    async def import_legacy_external_server(
        self,
        *,
        server_id: str,
        actor_id: int | None,
    ) -> dict[str, Any]:
        legacy = await self._get_legacy_inventory_item(server_id)
        if legacy is None:
            raise ResourceNotFoundError("mcp_external_server_legacy", identifier=server_id)

        row = await self.create_external_server(
            server_id=server_id,
            name=str(legacy.get("name") or server_id),
            transport=str(legacy.get("transport") or "websocket"),
            config=self._normalize_imported_external_config(dict(legacy.get("config") or {})),
            owner_scope_type=str(legacy.get("owner_scope_type") or "global"),
            owner_scope_id=legacy.get("owner_scope_id"),
            enabled=bool(legacy.get("enabled", True)),
            actor_id=actor_id,
            allow_existing=True,
        )
        updated = await self.repo.update_external_server(
            server_id,
            name=str(row.get("name") or server_id),
            transport=str(row.get("transport") or legacy.get("transport") or "websocket"),
            config_json=json.dumps(dict(row.get("config") or {})),
            owner_scope_type=str(row.get("owner_scope_type") or "global"),
            owner_scope_id=row.get("owner_scope_id"),
            enabled=bool(row.get("enabled")),
            server_source="managed",
            legacy_source_ref=legacy.get("legacy_source_ref"),
            superseded_by_server_id=None,
            actor_id=actor_id,
        )
        return updated or row

    async def _resolve_binding_credential_ref(
        self,
        *,
        target_row: dict[str, Any],
        slot_name: str | None,
        managed_secret_ref_id: int | None,
    ) -> str:
        if managed_secret_ref_id is None:
            return "slot" if slot_name else "server"

        managed_refs_repo = ManagedSecretRefsRepo(self.repo.db_pool)
        await managed_refs_repo.ensure_tables()
        ref = await managed_refs_repo.get_ref(int(managed_secret_ref_id))
        if not ref:
            raise BadRequestError("Managed secret ref not found")
        self._scope_reference_allowed(
            object_scope_type=str(ref.get("owner_scope_type") or "global"),
            object_scope_id=ref.get("owner_scope_id"),
            target_scope_type=str(target_row.get("owner_scope_type") or "global"),
            target_scope_id=target_row.get("owner_scope_id"),
            resource_name="managed secret ref",
        )
        return encode_managed_secret_credential_ref(int(ref["id"]))

    async def list_profile_credential_bindings(
        self,
        *,
        profile_id: int,
    ) -> list[dict[str, Any]]:
        await self._resolve_binding_target(binding_target_type="profile", binding_target_id=profile_id)
        return await self.repo.list_credential_bindings(
            binding_target_type="profile",
            binding_target_id=str(profile_id),
        )

    async def upsert_profile_credential_binding(
        self,
        *,
        profile_id: int,
        external_server_id: str,
        slot_name: str | None = None,
        managed_secret_ref_id: int | None = None,
        actor_id: int | None,
    ) -> dict[str, Any]:
        target_row = await self._resolve_binding_target(binding_target_type="profile", binding_target_id=profile_id)
        server_row = await self.repo.get_external_server(external_server_id)
        if not server_row:
            raise ResourceNotFoundError("mcp_external_server", identifier=external_server_id)
        self._validate_binding_scope(server_row=server_row, target_row=target_row)
        credential_ref = await self._resolve_binding_credential_ref(
            target_row=target_row,
            slot_name=slot_name,
            managed_secret_ref_id=managed_secret_ref_id,
        )
        row = await self.repo.upsert_credential_binding(
            binding_target_type="profile",
            binding_target_id=str(profile_id),
            external_server_id=external_server_id,
            slot_name=slot_name,
            credential_ref=credential_ref,
            binding_mode="grant",
            usage_rules={},
            actor_id=actor_id,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.profile_credential_binding.upsert",
                actor_id=actor_id,
                resource_type="mcp_permission_profile",
                resource_id=str(profile_id),
                metadata={
                    "external_server_id": external_server_id,
                    "slot_name": slot_name,
                    "binding_mode": "grant",
                    "managed_secret_ref_id": managed_secret_ref_id,
                    **(
                        await self._binding_audit_privilege_metadata(
                            external_server_id=external_server_id,
                            slot_name=slot_name,
                        )
                    ),
                },
            )
        )
        return row

    async def delete_profile_credential_binding(
        self,
        *,
        profile_id: int,
        external_server_id: str,
        slot_name: str | None = None,
        actor_id: int | None,
    ) -> bool:
        await self._resolve_binding_target(binding_target_type="profile", binding_target_id=profile_id)
        deleted = await self.repo.delete_credential_binding(
            binding_target_type="profile",
            binding_target_id=str(profile_id),
            external_server_id=external_server_id,
            slot_name=slot_name,
        )
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                action="mcp_hub.profile_credential_binding.delete",
                actor_id=actor_id,
                resource_type="mcp_permission_profile",
                resource_id=str(profile_id),
                metadata={"external_server_id": external_server_id, "slot_name": slot_name},
            )
        )
        return deleted

    async def list_assignment_credential_bindings(
        self,
        *,
        assignment_id: int,
    ) -> list[dict[str, Any]]:
        await self._resolve_binding_target(binding_target_type="assignment", binding_target_id=assignment_id)
        return await self.repo.list_credential_bindings(
            binding_target_type="assignment",
            binding_target_id=str(assignment_id),
        )

    async def upsert_assignment_credential_binding(
        self,
        *,
        assignment_id: int,
        external_server_id: str,
        slot_name: str | None = None,
        binding_mode: str,
        managed_secret_ref_id: int | None = None,
        actor_id: int | None,
    ) -> dict[str, Any]:
        target_row = await self._resolve_binding_target(binding_target_type="assignment", binding_target_id=assignment_id)
        server_row = await self.repo.get_external_server(external_server_id)
        if not server_row:
            raise ResourceNotFoundError("mcp_external_server", identifier=external_server_id)
        self._validate_binding_scope(server_row=server_row, target_row=target_row)
        credential_ref = await self._resolve_binding_credential_ref(
            target_row=target_row,
            slot_name=slot_name,
            managed_secret_ref_id=managed_secret_ref_id if binding_mode == "grant" else None,
        )
        row = await self.repo.upsert_credential_binding(
            binding_target_type="assignment",
            binding_target_id=str(assignment_id),
            external_server_id=external_server_id,
            slot_name=slot_name,
            credential_ref=credential_ref,
            binding_mode=binding_mode,
            usage_rules={},
            actor_id=actor_id,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.assignment_credential_binding.upsert",
                actor_id=actor_id,
                resource_type="mcp_policy_assignment",
                resource_id=str(assignment_id),
                metadata={
                    "external_server_id": external_server_id,
                    "slot_name": slot_name,
                    "binding_mode": binding_mode,
                    "managed_secret_ref_id": managed_secret_ref_id if binding_mode == "grant" else None,
                    **(
                        await self._binding_audit_privilege_metadata(
                            external_server_id=external_server_id,
                            slot_name=slot_name if binding_mode == "grant" else None,
                        )
                        if binding_mode == "grant"
                        else {}
                    ),
                },
            )
        )
        return row

    async def delete_assignment_credential_binding(
        self,
        *,
        assignment_id: int,
        external_server_id: str,
        slot_name: str | None = None,
        actor_id: int | None,
    ) -> bool:
        await self._resolve_binding_target(binding_target_type="assignment", binding_target_id=assignment_id)
        deleted = await self.repo.delete_credential_binding(
            binding_target_type="assignment",
            binding_target_id=str(assignment_id),
            external_server_id=external_server_id,
            slot_name=slot_name,
        )
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                action="mcp_hub.assignment_credential_binding.delete",
                actor_id=actor_id,
                resource_type="mcp_policy_assignment",
                resource_id=str(assignment_id),
                metadata={"external_server_id": external_server_id, "slot_name": slot_name},
            )
        )
        return deleted

    async def resolve_effective_external_access(
        self,
        *,
        assignment_id: int,
        actor_id: int | None,
    ) -> dict[str, Any]:
        resolver = McpHubExternalAccessResolver(repo=self.repo)
        assignment = await self.repo.get_policy_assignment(int(assignment_id))
        policy_document: dict[str, Any] = {}
        if assignment:
            profile_id = assignment.get("profile_id")
            if profile_id is not None:
                profile = await self.repo.get_permission_profile(int(profile_id))
                if profile and bool(profile.get("is_active", True)):
                    policy_document = dict(profile.get("policy_document") or {})
            inline_policy = dict((assignment or {}).get("inline_policy_document") or {})
            policy_document.update(inline_policy)
            override_row = await self.repo.get_policy_override_by_assignment(int(assignment_id))
            if override_row and bool(override_row.get("is_active", True)):
                policy_document.update(dict(override_row.get("override_policy_document") or {}))
        effective_policy = {"capabilities": list(policy_document.get("capabilities") or [])}
        summary = await resolver.resolve(
            assignment_id=int(assignment_id),
            effective_policy=effective_policy,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.external_access.resolve",
                actor_id=actor_id,
                resource_type="mcp_policy_assignment",
                resource_id=str(assignment_id),
                metadata={"server_count": len(summary.get("servers") or [])},
            )
        )
        return summary

    async def delete_external_server(self, server_id: str, *, actor_id: int | None) -> bool:
        deleted = await self.repo.delete_external_server(server_id)
        if deleted:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.external_server.delete",
                    actor_id=actor_id,
                    resource_type="mcp_external_server",
                    resource_id=server_id,
                    metadata=None,
                )
            )
        return deleted

    async def set_external_server_secret(
        self,
        *,
        server_id: str,
        secret_value: str,
        actor_id: int | None,
    ) -> dict[str, Any]:
        server = await self.repo.get_external_server(server_id)
        if not server:
            raise ResourceNotFoundError("mcp_external_server", identifier=server_id)

        secret = (secret_value or "").strip()
        if not secret:
            raise BadRequestError("Secret value is required")

        slots = await self.repo.list_external_server_credential_slots(server_id=server_id)
        if slots:
            default_slot = await self.repo.get_external_server_default_slot(server_id=server_id)
            if default_slot is None:
                raise BadRequestError("Server-level secret alias is only valid for default-slot servers")
            slot_row = await self.set_external_server_slot_secret(
                server_id=server_id,
                slot_name=str(default_slot.get("slot_name") or ""),
                secret_value=secret,
                actor_id=actor_id,
            )
            stored = await self.repo.upsert_external_secret(
                server_id=server_id,
                encrypted_blob=dumps_envelope(encrypt_byok_payload(build_secret_payload(secret))),
                key_hint=key_hint_for_api_key(secret),
                actor_id=actor_id,
            )
            return {
                "server_id": server_id,
                "slot_name": slot_row.get("slot_name"),
                "secret_configured": bool(stored),
                "key_hint": stored.get("key_hint"),
                "updated_at": stored.get("updated_at"),
            }

        secret_payload = build_secret_payload(secret)
        envelope = encrypt_byok_payload(secret_payload)
        stored = await self.repo.upsert_external_secret(
            server_id=server_id,
            encrypted_blob=dumps_envelope(envelope),
            key_hint=key_hint_for_api_key(secret),
            actor_id=actor_id,
        )
        await _await_if_needed(
            emit_mcp_hub_audit(
                action="mcp_hub.external_secret.update",
                actor_id=actor_id,
                resource_type="mcp_external_server",
                resource_id=server_id,
                metadata={"key_hint": stored.get("key_hint")},
            )
        )
        secret_configured = bool(stored)
        return {
            "server_id": server_id,
            "secret_configured": secret_configured,
            "key_hint": stored.get("key_hint"),
            "updated_at": stored.get("updated_at"),
        }

    async def clear_external_server_secret(
        self,
        *,
        server_id: str,
        actor_id: int | None,
    ) -> bool:
        cleared = await self.repo.clear_external_secret(server_id)
        if cleared:
            await _await_if_needed(
                emit_mcp_hub_audit(
                    action="mcp_hub.external_secret.clear",
                    actor_id=actor_id,
                    resource_type="mcp_external_server",
                    resource_id=server_id,
                    metadata=None,
                )
            )
        return cleared
