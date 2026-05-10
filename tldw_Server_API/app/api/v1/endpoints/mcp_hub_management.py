from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal
from tldw_Server_API.app.api.v1.schemas.mcp_hub_schemas import (
    ACPProfileCreateRequest,
    ACPProfileResponse,
    ACPProfileUpdateRequest,
    ApprovalDecisionCreateRequest,
    ApprovalDecisionResponse,
    ApprovalPolicyCreateRequest,
    ApprovalPolicyResponse,
    ApprovalPolicyUpdateRequest,
    AssignmentCredentialBindingUpsertRequest,
    CapabilityAdapterMappingCreateRequest,
    CapabilityAdapterMappingPreviewResponse,
    CapabilityAdapterMappingResponse,
    CapabilityAdapterMappingUpdateRequest,
    CredentialBindingResponse,
    EffectivePolicyResponse,
    EffectiveExternalAccessResponse,
    ExternalSecretSetRequest,
    ExternalSecretSetResponse,
    ExternalServerDiscoveryRefreshRequest,
    ExternalServerDiscoveryRefreshResponse,
    ExternalServerAuthTemplateResponse,
    ExternalServerAuthTemplateUpdateRequest,
    ExternalServerCredentialSlotCreateRequest,
    ExternalServerCredentialSlotResponse,
    ExternalServerCredentialSlotUpdateRequest,
    ExternalServerSlotSecretSetResponse,
    ExternalServerCreateRequest,
    ExternalServerResponse,
    ExternalServerUpdateRequest,
    GovernancePackDetailResponse,
    GovernancePackDryRunRequest,
    GovernancePackDryRunResponse,
    GovernancePackImportRequest,
    GovernancePackImportResponse,
    GovernancePackSourceDryRunRequest,
    GovernancePackSourceImportRequest,
    GovernancePackSourcePrepareRequest,
    GovernancePackSourcePrepareResponse,
    GovernancePackSourceUpdateCheckResponse,
    GovernancePackSourceUpgradeDryRunRequest,
    GovernancePackSourceUpgradeExecuteRequest,
    GovernancePackSourceUpgradePrepareResponse,
    GovernancePackUpgradeDryRunRequest,
    GovernancePackUpgradeDryRunResponse,
    GovernancePackUpgradeExecuteRequest,
    GovernancePackUpgradeExecutionResponse,
    GovernancePackUpgradeHistoryEntryResponse,
    GovernancePackSummaryResponse,
    GovernancePackTrustPolicyRequest,
    GovernancePackTrustPolicyResponse,
    GovernanceAuditFindingListResponse,
    MCPHubDeleteResponse,
    McpCredentialSlotStatusResponse,
    PathScopeObjectCreateRequest,
    PathScopeObjectResponse,
    PathScopeObjectUpdateRequest,
    PermissionProfileCreateRequest,
    PermissionProfileResponse,
    PermissionProfileUpdateRequest,
    PolicyAssignmentCreateRequest,
    PolicyAssignmentResponse,
    PolicyAssignmentWorkspaceCreateRequest,
    PolicyAssignmentWorkspaceResponse,
    PolicyAssignmentUpdateRequest,
    PolicyOverrideResponse,
    PolicyOverrideUpsertRequest,
    ProfileCredentialBindingUpsertRequest,
    SharedWorkspaceCreateRequest,
    SharedWorkspaceResponse,
    SharedWorkspaceUpdateRequest,
    ToolRegistryEntryResponse,
    ToolRegistryModuleResponse,
    ToolRegistrySummaryResponse,
    WorkspaceSetObjectCreateRequest,
    WorkspaceSetObjectMemberCreateRequest,
    WorkspaceSetObjectMemberResponse,
    WorkspaceSetObjectResponse,
    WorkspaceSetObjectUpdateRequest,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.core.config import config
from tldw_Server_API.app.core.exceptions import BadRequestError, ResourceNotFoundError
from tldw_Server_API.app.core.MCP_unified.server import get_mcp_server
from tldw_Server_API.app.services.mcp_hub_external_legacy_inventory import (
    McpHubExternalLegacyInventoryService,
)
from tldw_Server_API.app.services.mcp_hub_capability_adapter_service import (
    McpHubCapabilityAdapterService,
)
from tldw_Server_API.app.services.mcp_hub_governance_pack_service import (
    GovernancePackAlreadyExistsError,
    GovernancePackUpgradeConflictError,
    GovernancePackUpgradeStaleError,
    McpHubGovernancePackService,
)
from tldw_Server_API.app.services.mcp_hub_governance_pack_trust_service import (
    McpHubGovernancePackTrustService,
    GovernancePackTrustPolicyStaleError,
)
from tldw_Server_API.app.services.mcp_hub_governance_pack_distribution_service import (
    McpHubGovernancePackDistributionService,
)
from tldw_Server_API.app.services.mcp_credential_broker_service import (
    McpCredentialBrokerService,
)
from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver, get_mcp_hub_policy_resolver
from tldw_Server_API.app.services.mcp_hub_service import (
    McpHubConflictError,
    McpHubService,
    get_mcp_hub_event_bus,
    replay_mcp_hub_audit_events,
)
from tldw_Server_API.app.services.mcp_hub_tool_registry import McpHubToolRegistryService

router = APIRouter(prefix="/mcp/hub", tags=["mcp-hub"], dependencies=[Depends(check_rate_limit)])

_MCP_HUB_ADMIN_PERMISSIONS = frozenset({SYSTEM_CONFIGURE, "*"})
_VALID_SCOPE_TYPES = frozenset({"global", "org", "team", "user"})
_CAPABILITY_GRANT_PERMISSIONS = {
    "credentials.use": "grant.credentials.use",
    "filesystem.delete": "grant.filesystem.delete",
    "filesystem.read": "grant.filesystem.read",
    "filesystem.write": "grant.filesystem.write",
    "mcp.server.connect": "grant.mcp.server.connect",
    "network.external": "grant.network.external",
    "process.execute": "grant.process.execute",
    "tool.invoke": "grant.tool.invoke",
}
_CREDENTIAL_GRANT_PERMISSIONS = {
    "read": "grant.credentials.read",
    "write": "grant.credentials.write",
    "admin": "grant.credentials.admin",
}
_CREDENTIAL_PRIVILEGE_RANK = {"read": 0, "write": 1, "admin": 2}
_DEFAULT_CREDENTIAL_SLOT_NAMES = frozenset({"bearer_token", "api_key"})
_TOOL_GRANT_KEYS = ("allowed_tools", "tool_patterns", "tool_names")
_UNION_POLICY_KEYS = frozenset({"allowed_tools", "denied_tools", "tool_names", "tool_patterns", "capabilities"})
_FILESYSTEM_GRANT_CAPABILITIES = frozenset({"filesystem.read", "filesystem.write", "filesystem.delete"})
_SUPPORTED_APPROVAL_DURATIONS = frozenset({"once", "session", "conversation"})
_DEFAULT_SCOPED_APPROVAL_TTL_MINUTES = 480
_PATH_SCOPE_BREADTH_RANK = {"cwd_descendants": 0, "workspace_root": 1, "none": 2}
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_PATH_SCOPE_DOCUMENT_KEYS = {"path_scope_mode", "path_scope_enforcement", "path_allowlist_prefixes"}


async def get_mcp_hub_service() -> McpHubService:
    """Resolve MCP Hub service with storage bootstrap checks."""
    pool = await get_db_pool()
    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    cfg = config
    inventory_service = McpHubExternalLegacyInventoryService(
        config_path=str(
            getattr(cfg, "external_servers_config_path", None)
            or "tldw_Server_API/Config_Files/mcp_external_servers.yaml"
        )
    )
    return McpHubService(repo, legacy_inventory_service=inventory_service)


async def get_mcp_hub_governance_pack_service() -> McpHubGovernancePackService:
    """Resolve governance-pack service with MCP Hub storage bootstrap checks."""
    pool = await get_db_pool()
    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    return McpHubGovernancePackService(repo=repo)


async def get_mcp_hub_governance_pack_trust_service() -> McpHubGovernancePackTrustService:
    """Resolve governance-pack trust service with MCP Hub storage bootstrap checks."""
    pool = await get_db_pool()
    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    return McpHubGovernancePackTrustService(repo=repo)


async def get_mcp_hub_governance_pack_distribution_service() -> McpHubGovernancePackDistributionService:
    """Resolve governance-pack distribution service with MCP Hub storage bootstrap checks."""
    pool = await get_db_pool()
    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    trust_service = McpHubGovernancePackTrustService(repo=repo)
    return McpHubGovernancePackDistributionService(trust_service=trust_service, repo=repo)


async def get_mcp_hub_capability_adapter_service() -> McpHubCapabilityAdapterService:
    """Resolve capability-adapter service with storage bootstrap checks."""
    pool = await get_db_pool()
    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    return McpHubCapabilityAdapterService(
        repo=repo,
        tool_registry=McpHubToolRegistryService(),
    )


async def get_mcp_hub_policy_resolver_dep() -> McpHubPolicyResolver:
    """Resolve MCP Hub policy resolver for effective policy previews."""
    return await get_mcp_hub_policy_resolver()


async def get_mcp_hub_tool_registry_dep() -> McpHubToolRegistryService:
    """Resolve the derived MCP Hub tool registry service."""
    return McpHubToolRegistryService()


async def get_mcp_credential_broker_service() -> McpCredentialBrokerService:
    """Resolve the MCP credential broker service backed by the current MCP Hub repo."""
    pool = await get_db_pool()
    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    return McpCredentialBrokerService(repo=repo)


def _load_json_object(raw: Any) -> dict[str, Any]:
    """Parse JSON-like values into a dict, returning empty dict on decode failures."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError):
            logger.debug("MCP hub payload JSON decode failed")
            return {}
    return {}


def _is_mutation_allowed(principal: AuthPrincipal) -> bool:
    """Return True when the principal is allowed to mutate MCP Hub configuration."""
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    if "admin" in roles:
        return True
    permissions = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    required = {perm.lower() for perm in _MCP_HUB_ADMIN_PERMISSIONS}
    return bool(permissions & required)


def _is_admin_principal(principal: AuthPrincipal) -> bool:
    """Return True only for explicit admin principals."""
    if bool(getattr(principal, "is_admin", False)):
        return True
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    return "admin" in roles


def _require_mutation_permission(principal: AuthPrincipal) -> None:
    """Require mutation permission for MCP Hub write operations."""
    if _is_mutation_allowed(principal):
        return
    raise HTTPException(status_code=403, detail=f"{SYSTEM_CONFIGURE} permission required")


def _runtime_unavailable(message: str) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "ok": False,
            "message": message,
            "requires_restart": True,
        },
    )


def _normalize_refresh_server_id(server_id: str | None) -> str | None:
    if server_id is None:
        return None
    normalized = server_id.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail="server_id must be a non-empty string")
    return normalized


async def _resolve_external_federation_runtime() -> tuple[Any, Any, str]:
    """Resolve the live MCP external federation module and its registry."""

    try:
        server = get_mcp_server()
        if not getattr(server, "initialized", False):
            await server.initialize()
    except Exception as exc:
        logger.warning("MCP runtime unavailable for external discovery refresh: {}", type(exc).__name__)
        raise _runtime_unavailable("MCP runtime is unavailable; restart or retry after startup completes") from exc

    registry = getattr(server, "module_registry", None)
    if registry is None:
        raise _runtime_unavailable("MCP module registry is unavailable")

    try:
        module = await registry.get_module("external_federation")
        if module is not None:
            return module, registry, "external_federation"

        modules = await registry.get_all_modules()
        for module_id, candidate in modules.items():
            if candidate.__class__.__name__ == "ExternalFederationModule":
                return candidate, registry, str(module_id)
    except Exception as exc:
        logger.warning("MCP external federation module lookup failed: {}", type(exc).__name__)
        raise _runtime_unavailable("External federation module is unavailable") from exc

    raise _runtime_unavailable("External federation module is unavailable")


async def _refresh_external_server_discovery_runtime(
    *,
    server_id: str | None,
) -> ExternalServerDiscoveryRefreshResponse:
    module, registry, module_id = await _resolve_external_federation_runtime()
    manager = getattr(module, "_manager", None)
    if manager is None or not hasattr(manager, "reconcile_servers"):
        raise _runtime_unavailable("External federation manager is unavailable")

    try:
        result = await manager.reconcile_servers(server_id=server_id)
    except Exception as exc:
        logger.warning("External server discovery reconciliation failed: {}", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={
                "ok": False,
                "server_id": server_id,
                "message": "External server discovery reconciliation failed",
                "requires_restart": False,
            },
        ) from exc

    errors = dict(result.get("errors") or {})
    if hasattr(module, "invalidate_capability_caches"):
        module.invalidate_capability_caches()
    refresh_registries = getattr(registry, "refresh_module_registries", None)
    if callable(refresh_registries):
        try:
            registry_refreshed = await refresh_registries(module_id)
        except Exception as exc:
            logger.warning("MCP external federation registry refresh failed: {}", type(exc).__name__)
            registry_refreshed = False
        if registry_refreshed is False:
            errors["module_registry"] = "module_registry_refresh_failed"
    return ExternalServerDiscoveryRefreshResponse(
        ok=not errors,
        server_id=result.get("server_id") or server_id,
        reconciled_servers=int(result.get("reconciled_servers") or 0),
        refreshed_servers=int(result.get("refreshed_servers") or 0),
        total_servers=int(result.get("total_servers") or 0),
        virtual_tools=int(result.get("virtual_tools") or 0),
        errors=errors,
        requires_restart=False,
        message="Discovery refreshed" if not errors else "Discovery refreshed with errors",
    )


def _normalize_event_type_filter(event_type: list[str] | None) -> set[str] | None:
    values: set[str] = set()
    for raw in event_type or []:
        for part in str(raw or "").split(","):
            cleaned = part.strip()
            if cleaned:
                values.add(cleaned)
    return values or None


def _format_mcp_hub_sse(event: dict[str, Any]) -> str:
    event_id = str(event.get("event_id") or "")
    event_name = str(event.get("event_type") or "mcp_hub.update")
    payload = json.dumps(event, default=str)
    return f"id: {event_id}\nevent: {event_name}\ndata: {payload}\n\n"


def _event_matches_visible_scope(
    event: dict[str, Any],
    visible_scopes: list[tuple[str | None, int | None]],
) -> bool:
    if any(scope_type is None for scope_type, _scope_id in visible_scopes):
        return True
    metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
    raw_scope_type = metadata.get("owner_scope_type") or event.get("owner_scope_type") or "global"
    scope_type = str(raw_scope_type or "global").strip().lower()
    if scope_type == "global":
        scope_id = None
    else:
        raw_scope_id = metadata.get("owner_scope_id") or event.get("owner_scope_id")
        try:
            scope_id = int(raw_scope_id)
        except (TypeError, ValueError):
            return False
    return (scope_type, scope_id) in visible_scopes


def _ensure_mutable_governance_object(row: dict[str, Any] | None, object_label: str) -> None:
    """Reject direct mutation of imported governance-pack managed base objects."""
    if row and bool(row.get("is_immutable")):
        raise HTTPException(
            status_code=400,
            detail=f"{object_label} is immutable because it is managed by an imported governance pack",
        )


def _require_grant_authority(principal: AuthPrincipal, policy_document: dict[str, Any]) -> None:
    """Require capability grant authority for the requested policy document."""
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    if "admin" in roles:
        return

    granted = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    if "*" in granted:
        return

    raw_capabilities = _as_str_list(policy_document.get("capabilities"))
    capabilities = {
        str(capability).strip().lower()
        for capability in raw_capabilities
        if str(capability).strip()
    }
    if any(
        str(entry).strip()
        for key in _TOOL_GRANT_KEYS
        for entry in _as_str_list(policy_document.get(key))
    ):
        capabilities.add("tool.invoke")
    missing = [
        required
        for capability in sorted(capabilities)
        if (required := _CAPABILITY_GRANT_PERMISSIONS.get(capability)) and required.lower() not in granted
    ]
    if missing:
        raise HTTPException(
            status_code=403,
            detail=f"Grant authority required: {', '.join(missing)}",
        )


def _as_str_list(value: Any) -> list[str]:
    """Normalize a loose scalar-or-sequence value into a cleaned string list."""
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


def _unique(items: list[str]) -> list[str]:
    """Preserve order while removing duplicate string values."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_path_scope_mode(value: Any) -> str:
    """Normalize a path scope mode, defaulting missing values to `none`."""
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _PATH_SCOPE_BREADTH_RANK else "none"


def _normalize_path_allowlist_prefixes(
    value: Any,
    *,
    reject_invalid: bool,
) -> list[str]:
    """Normalize workspace-relative path allowlist prefixes."""
    raw_entries = _as_str_list(value)
    normalized_entries: list[str] = []
    seen: set[str] = set()
    for raw_entry in raw_entries:
        candidate = str(raw_entry or "").strip().replace("\\", "/")
        while candidate.startswith("./"):
            candidate = candidate[2:]
        candidate = re.sub(r"/+", "/", candidate)
        if candidate.startswith("/") or _WINDOWS_ABSOLUTE_PATH_RE.match(candidate):
            if reject_invalid:
                raise HTTPException(status_code=400, detail="path_allowlist_prefixes must be workspace-relative")
            continue
        parts: list[str] = []
        invalid = False
        for part in candidate.split("/"):
            cleaned = str(part or "").strip()
            if not cleaned or cleaned == ".":
                continue
            if cleaned == "..":
                invalid = True
                break
            parts.append(cleaned)
        if invalid or not parts:
            if reject_invalid:
                raise HTTPException(
                    status_code=400,
                    detail="path_allowlist_prefixes cannot contain traversal or empty entries",
                )
            continue
        normalized = "/".join(parts)
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_entries.append(normalized)
    return sorted(normalized_entries)


def _normalized_path_allowlist_for_comparison(policy_document: dict[str, Any]) -> set[str]:
    """Return a permissively normalized allowlist set for broadened-access comparisons."""
    return set(
        _normalize_path_allowlist_prefixes(
            policy_document.get("path_allowlist_prefixes"),
            reject_invalid=False,
        )
    )


def _normalize_policy_document_for_storage(
    policy_document: dict[str, Any],
    *,
    allow_inherited_scope: bool,
) -> dict[str, Any]:
    """Normalize path policy fields before persisting MCP Hub documents."""
    normalized = deepcopy(policy_document)
    if "path_allowlist_prefixes" not in normalized:
        return normalized

    allowlist = _normalize_path_allowlist_prefixes(
        normalized.get("path_allowlist_prefixes"),
        reject_invalid=True,
    )
    path_scope_mode = _normalize_path_scope_mode(normalized.get("path_scope_mode"))
    if path_scope_mode == "none" and not allow_inherited_scope:
        raise HTTPException(
            status_code=400,
            detail="path_allowlist_prefixes requires path_scope_mode to be workspace_root or cwd_descendants",
        )
    if "path_scope_mode" in normalized and path_scope_mode == "none":
        raise HTTPException(
            status_code=400,
            detail="path_allowlist_prefixes cannot be combined with path_scope_mode='none'",
        )
    if not allowlist:
        normalized.pop("path_allowlist_prefixes", None)
    else:
        normalized["path_allowlist_prefixes"] = allowlist
    return normalized


def _normalize_path_scope_document_for_storage(path_scope_document: dict[str, Any]) -> dict[str, Any]:
    """Normalize and limit a reusable path-scope object document to path-governance keys."""
    filtered = {
        key: deepcopy(value)
        for key, value in dict(path_scope_document or {}).items()
        if key in _PATH_SCOPE_DOCUMENT_KEYS
    }
    return _normalize_policy_document_for_storage(filtered, allow_inherited_scope=False)


def _merge_policy_documents(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge overlay policy fields into a base document using union semantics for allowlists."""
    merged = deepcopy(base)
    for key, value in overlay.items():
        if key in _UNION_POLICY_KEYS:
            merged[key] = _unique(_as_str_list(merged.get(key)) + _as_str_list(value))
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_policy_documents(
                dict(merged.get(key) or {}),
                dict(value or {}),
            )
            continue
        merged[key] = deepcopy(value)
    return merged


def _tool_grant_entries(policy_document: dict[str, Any]) -> set[str]:
    """Collect every tool-grant entry across the supported allowlist keys."""
    entries: set[str] = set()
    for key in _TOOL_GRANT_KEYS:
        entries.update(_as_str_list(policy_document.get(key)))
    return entries


def _grant_authority_delta(base_policy_document: dict[str, Any], merged_policy_document: dict[str, Any]) -> dict[str, Any]:
    """Extract the broadened-access delta that requires explicit grant authority."""
    base_capabilities = {entry.lower() for entry in _as_str_list(base_policy_document.get("capabilities"))}
    merged_capabilities = {entry.lower() for entry in _as_str_list(merged_policy_document.get("capabilities"))}
    next_document: dict[str, Any] = {}

    extra_capabilities = sorted(merged_capabilities - base_capabilities)
    if extra_capabilities:
        next_document["capabilities"] = extra_capabilities

    base_tools = _tool_grant_entries(base_policy_document)
    merged_tools = _tool_grant_entries(merged_policy_document)
    extra_tools = sorted(merged_tools - base_tools)
    if extra_tools:
        next_document["allowed_tools"] = extra_tools

    base_scope_rank = _PATH_SCOPE_BREADTH_RANK[_normalize_path_scope_mode(base_policy_document.get("path_scope_mode"))]
    merged_scope_rank = _PATH_SCOPE_BREADTH_RANK[_normalize_path_scope_mode(merged_policy_document.get("path_scope_mode"))]
    base_allowlist = _normalized_path_allowlist_for_comparison(base_policy_document)
    merged_allowlist = _normalized_path_allowlist_for_comparison(merged_policy_document)
    broadened_path_scope = merged_scope_rank > base_scope_rank
    broadened_allowlist = bool(base_allowlist) and (
        (not merged_allowlist) or bool(merged_allowlist - base_allowlist)
    )
    if broadened_path_scope or broadened_allowlist:
        existing_caps = {
            entry.lower()
            for entry in _as_str_list(next_document.get("capabilities"))
            if str(entry).strip()
        }
        filesystem_caps = {
            capability
            for capability in merged_capabilities
            if capability in _FILESYSTEM_GRANT_CAPABILITIES
        }
        existing_caps.update(filesystem_caps or {"filesystem.read"})
        next_document["capabilities"] = sorted(existing_caps)

    return next_document


def _grant_authority_snapshot(principal: AuthPrincipal, delta_document: dict[str, Any]) -> dict[str, Any]:
    """Capture granted and required permissions for an override broadening audit record."""
    granted_permissions = sorted(
        {
            str(permission).strip().lower()
            for permission in (principal.permissions or [])
            if str(permission).strip()
        }
    )
    required_permissions = sorted(
        {
            required
            for capability in {
                str(capability).strip().lower()
                for capability in _as_str_list(delta_document.get("capabilities"))
            }
            if (required := _CAPABILITY_GRANT_PERMISSIONS.get(capability))
        }
    )
    if _tool_grant_entries(delta_document):
        required_permissions.append(_CAPABILITY_GRANT_PERMISSIONS["tool.invoke"])
    return {
        "required_permissions": _unique(required_permissions),
        "granted_permissions": granted_permissions,
    }


def _normalize_credential_slot_privilege_class(value: Any) -> str:
    """Normalize a credential slot privilege class or raise an HTTP 400."""
    normalized = str(value or "").strip().lower()
    if normalized not in _CREDENTIAL_GRANT_PERMISSIONS:
        raise HTTPException(
            status_code=400,
            detail="Credential slot privilege_class must be one of: read, write, admin",
        )
    return normalized


def _credential_required_permission(privilege_class: Any) -> str:
    """Return the required grant-authority permission for a credential privilege class."""
    return _CREDENTIAL_GRANT_PERMISSIONS[_normalize_credential_slot_privilege_class(privilege_class)]


def _principal_satisfies_credential_grant(principal: AuthPrincipal, privilege_class: Any) -> bool:
    """Return True when the principal may grant the requested credential privilege class."""
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    if "admin" in roles:
        return True

    granted = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    if "*" in granted:
        return True

    normalized = _normalize_credential_slot_privilege_class(privilege_class)
    required_rank = _CREDENTIAL_PRIVILEGE_RANK[normalized]
    for candidate, candidate_permission in _CREDENTIAL_GRANT_PERMISSIONS.items():
        if candidate_permission in granted and _CREDENTIAL_PRIVILEGE_RANK[candidate] >= required_rank:
            return True
    return False


def _require_credential_grant_authority(principal: AuthPrincipal, privilege_class: Any) -> None:
    """Require the principal to hold the matching credential grant-authority permission."""
    if _principal_satisfies_credential_grant(principal, privilege_class):
        return
    raise HTTPException(
        status_code=403,
        detail=f"Grant authority required: {_credential_required_permission(privilege_class)}",
    )


def _credential_privilege_broadens(previous_privilege_class: Any, next_privilege_class: Any) -> bool:
    """Return True when a credential slot privilege class broadens access."""
    previous = _normalize_credential_slot_privilege_class(previous_privilege_class)
    next_value = _normalize_credential_slot_privilege_class(next_privilege_class)
    return _CREDENTIAL_PRIVILEGE_RANK[next_value] > _CREDENTIAL_PRIVILEGE_RANK[previous]


async def _resolve_binding_slot_for_grant_authority(
    *,
    svc: McpHubService,
    server_id: str,
    slot_name: str | None,
) -> dict[str, Any] | None:
    """Resolve the effective credential slot used for grant-authority checks."""
    try:
        slots = await svc.list_external_server_credential_slots(server_id=server_id)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if slot_name is not None:
        normalized_slot = str(slot_name or "").strip()
        return next(
            (
                dict(slot)
                for slot in slots
                if str(slot.get("slot_name") or "").strip() == normalized_slot
            ),
            None,
        )
    if len(slots) != 1:
        return None
    slot = dict(slots[0])
    if str(slot.get("slot_name") or "").strip() not in _DEFAULT_CREDENTIAL_SLOT_NAMES:
        return None
    return slot


def _collect_scope_ids(values: list[int] | None, active_id: int | None) -> list[int]:
    out: set[int] = set()
    for raw in values or []:
        try:
            out.add(int(raw))
        except (TypeError, ValueError):
            continue
    if active_id is not None:
        try:
            out.add(int(active_id))
        except (TypeError, ValueError):
            pass
    return sorted(out)


def _resolve_visible_scope_filters(
    *,
    principal: AuthPrincipal,
    owner_scope_type: str | None,
    owner_scope_id: int | None,
) -> list[tuple[str | None, int | None]]:
    """
    Resolve list query filters constrained to scopes visible to the authenticated principal.

    Returns one or more `(scope_type, scope_id)` filters. A `scope_type` of `None` means
    unrestricted query (admin-like contexts only).
    """
    scope_type = owner_scope_type.strip().lower() if owner_scope_type else None
    if scope_type is not None and scope_type not in _VALID_SCOPE_TYPES:
        raise HTTPException(status_code=422, detail="Invalid owner_scope_type")

    if _is_mutation_allowed(principal):
        if scope_type is None:
            if owner_scope_id is not None:
                raise HTTPException(status_code=422, detail="owner_scope_id requires owner_scope_type")
            return [(None, None)]
        if scope_type == "global":
            if owner_scope_id is not None:
                raise HTTPException(status_code=422, detail="global scope cannot include owner_scope_id")
            return [("global", None)]
        if owner_scope_id is None:
            raise HTTPException(status_code=422, detail=f"{scope_type} scope requires owner_scope_id")
        return [(scope_type, int(owner_scope_id))]

    user_id = int(principal.user_id) if principal.user_id is not None else None
    org_ids = _collect_scope_ids(principal.org_ids, principal.active_org_id)
    team_ids = _collect_scope_ids(principal.team_ids, principal.active_team_id)

    if scope_type is None:
        if owner_scope_id is not None:
            raise HTTPException(status_code=422, detail="owner_scope_id requires owner_scope_type")
        filters: list[tuple[str | None, int | None]] = [("global", None)]
        if user_id is not None:
            filters.append(("user", user_id))
        filters.extend(("org", org_id) for org_id in org_ids)
        filters.extend(("team", team_id) for team_id in team_ids)
        return filters

    if scope_type == "global":
        if owner_scope_id is not None:
            raise HTTPException(status_code=422, detail="global scope cannot include owner_scope_id")
        return [("global", None)]

    if scope_type == "user":
        target_user_id = int(owner_scope_id) if owner_scope_id is not None else user_id
        if target_user_id is None or user_id is None or target_user_id != user_id:
            raise HTTPException(status_code=403, detail="Forbidden scope filter")
        return [("user", user_id)]

    if scope_type == "org":
        if owner_scope_id is None:
            return [("org", org_id) for org_id in org_ids]
        if int(owner_scope_id) not in set(org_ids):
            raise HTTPException(status_code=403, detail="Forbidden scope filter")
        return [("org", int(owner_scope_id))]

    if scope_type == "team":
        if owner_scope_id is None:
            return [("team", team_id) for team_id in team_ids]
        if int(owner_scope_id) not in set(team_ids):
            raise HTTPException(status_code=403, detail="Forbidden scope filter")
        return [("team", int(owner_scope_id))]

    raise HTTPException(status_code=422, detail="Invalid owner_scope_type")


def _assert_principal_can_access_scope(
    principal: AuthPrincipal,
    *,
    owner_scope_type: str,
    owner_scope_id: int | None,
) -> None:
    """Assert that the principal can access a concrete owner scope."""
    _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )


def _resolve_governance_pack_mutation_scope(
    *,
    principal: AuthPrincipal,
    owner_scope_type: str,
    owner_scope_id: int | None,
) -> tuple[str, int | None]:
    """Resolve governance-pack mutation scope, defaulting user scope to the active principal."""
    scope_type = str(owner_scope_type or "").strip().lower()
    if scope_type not in _VALID_SCOPE_TYPES:
        raise HTTPException(status_code=422, detail="Invalid owner_scope_type")

    if scope_type == "global":
        if owner_scope_id is not None:
            raise HTTPException(status_code=422, detail="global scope cannot include owner_scope_id")
        return ("global", None)

    if scope_type == "user":
        if owner_scope_id is not None:
            return ("user", int(owner_scope_id))
        if principal.user_id is None:
            raise HTTPException(status_code=422, detail="user scope requires owner_scope_id")
        return ("user", int(principal.user_id))

    if owner_scope_id is None:
        raise HTTPException(status_code=422, detail=f"{scope_type} scope requires owner_scope_id")
    return (scope_type, int(owner_scope_id))


def _resolve_capability_adapter_mutation_scope(
    *,
    owner_scope_type: str,
    owner_scope_id: int | None,
) -> tuple[str, int | None]:
    """Resolve capability-adapter mapping scope, rejecting unsupported user scope."""
    scope_type = str(owner_scope_type or "").strip().lower()
    if scope_type not in {"global", "org", "team"}:
        raise HTTPException(status_code=422, detail="Invalid owner_scope_type")
    if scope_type == "global":
        if owner_scope_id is not None:
            raise HTTPException(status_code=422, detail="global scope cannot include owner_scope_id")
        return ("global", None)
    if owner_scope_id is None:
        raise HTTPException(status_code=422, detail=f"{scope_type} scope requires owner_scope_id")
    return (scope_type, int(owner_scope_id))


def _resolve_visible_capability_adapter_scope_filters(
    *,
    principal: AuthPrincipal,
    owner_scope_type: str | None,
    owner_scope_id: int | None,
) -> list[tuple[str | None, int | None]]:
    """Resolve visible filters for capability mappings, excluding unsupported user scope."""
    if owner_scope_type is not None and str(owner_scope_type or "").strip().lower() == "user":
        raise HTTPException(status_code=422, detail="Invalid owner_scope_type")
    filters = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    return [
        (scope_type, scope_id)
        for scope_type, scope_id in filters
        if scope_type is None or scope_type in {"global", "org", "team"}
    ]


def _portable_grant_capability(capability: Any) -> str | None:
    """Map a portable governance-pack capability to the runtime grant-authority root capability."""
    normalized = str(capability or "").strip().lower()
    if not normalized:
        return None
    if normalized in _CAPABILITY_GRANT_PERMISSIONS:
        return normalized
    if normalized.startswith("tool.invoke."):
        return "tool.invoke"
    if normalized.startswith("network.external."):
        return "network.external"
    if normalized.startswith("process.execute."):
        return "process.execute"
    if normalized.startswith("mcp.server.connect"):
        return "mcp.server.connect"
    if normalized.startswith("filesystem."):
        return normalized if normalized in _CAPABILITY_GRANT_PERMISSIONS else None
    return None


def _governance_pack_grant_document(pack_document: dict[str, Any]) -> dict[str, Any]:
    """Build a synthetic policy document for grant-authority evaluation of portable capabilities."""
    capabilities: list[str] = []
    raw_profiles = pack_document.get("profiles")
    if not isinstance(raw_profiles, list):
        return {"capabilities": capabilities}
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            continue
        raw_capabilities = raw_profile.get("capabilities")
        if not isinstance(raw_capabilities, dict):
            continue
        for capability in _as_str_list(raw_capabilities.get("allow")):
            mapped = _portable_grant_capability(capability)
            if mapped:
                capabilities.append(mapped)
    return {"capabilities": _unique(capabilities)}


def _governance_pack_manifest_summary(pack_document: dict[str, Any]) -> tuple[str, str]:
    """Return safe manifest identifiers for audit logging without logging the full pack payload."""
    manifest = pack_document.get("manifest")
    if not isinstance(manifest, dict):
        return ("<unknown>", "<unknown>")
    pack_id = str(manifest.get("pack_id") or "").strip() or "<unknown>"
    pack_version = str(manifest.get("pack_version") or "").strip() or "<unknown>"
    return (pack_id, pack_version)


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate row dictionaries by `id` while preserving final-write wins semantics."""
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id") or "")
        if not row_id:
            continue
        deduped[row_id] = row
    return list(deduped.values())


def _dedupe_rows_by_id(
    rows: list[dict[str, Any]],
    *,
    id_getter: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    """Deduplicate row dictionaries using a caller-provided stable identity key."""
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        row_id = id_getter(row)
        if not row_id:
            continue
        deduped[tuple(row_id)] = row
    return list(deduped.values())


def _profile_row_to_response(row: dict[str, Any]) -> ACPProfileResponse:
    return ACPProfileResponse(
        id=int(row.get("id")),
        name=str(row.get("name") or ""),
        description=row.get("description"),
        owner_scope_type=str(row.get("owner_scope_type") or "global"),
        owner_scope_id=row.get("owner_scope_id"),
        profile=_load_json_object(row.get("profile_json")),
        is_active=bool(row.get("is_active")),
        created_by=row.get("created_by"),
        updated_by=row.get("updated_by"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _external_row_to_response(row: dict[str, Any]) -> ExternalServerResponse:
    slots = [
        ExternalServerCredentialSlotResponse.model_validate(slot)
        for slot in (row.get("credential_slots") or [])
        if isinstance(slot, dict)
    ]
    return ExternalServerResponse(
        id=str(row.get("id") or ""),
        name=str(row.get("name") or ""),
        enabled=bool(row.get("enabled")),
        owner_scope_type=str(row.get("owner_scope_type") or "global"),
        owner_scope_id=row.get("owner_scope_id"),
        transport=str(row.get("transport") or ""),
        config=_load_json_object(row.get("config_json") if row.get("config_json") is not None else row.get("config")),
        secret_configured=bool(row.get("secret_configured")),
        key_hint=row.get("key_hint"),
        server_source=str(row.get("server_source") or "managed"),
        legacy_source_ref=row.get("legacy_source_ref"),
        superseded_by_server_id=row.get("superseded_by_server_id"),
        binding_count=int(row.get("binding_count") or 0),
        runtime_executable=bool(
            row.get("runtime_executable")
            if row.get("runtime_executable") is not None
            else True
        ),
        auth_template_present=bool(row.get("auth_template_present")),
        auth_template_valid=bool(row.get("auth_template_valid")),
        auth_template_blocked_reason=row.get("auth_template_blocked_reason"),
        credential_slots=slots,
        created_by=row.get("created_by"),
        updated_by=row.get("updated_by"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _permission_profile_row_to_response(row: dict[str, Any]) -> PermissionProfileResponse:
    return PermissionProfileResponse(
        id=int(row.get("id")),
        name=str(row.get("name") or ""),
        description=row.get("description"),
        owner_scope_type=str(row.get("owner_scope_type") or "global"),
        owner_scope_id=row.get("owner_scope_id"),
        mode=str(row.get("mode") or "custom"),
        path_scope_object_id=row.get("path_scope_object_id"),
        policy_document=_load_json_object(row.get("policy_document")),
        is_active=bool(row.get("is_active")),
        created_by=row.get("created_by"),
        updated_by=row.get("updated_by"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _policy_assignment_row_to_response(row: dict[str, Any]) -> PolicyAssignmentResponse:
    return PolicyAssignmentResponse(
        id=int(row.get("id")),
        target_type=str(row.get("target_type") or "default"),
        target_id=row.get("target_id"),
        owner_scope_type=str(row.get("owner_scope_type") or "global"),
        owner_scope_id=row.get("owner_scope_id"),
        profile_id=row.get("profile_id"),
        path_scope_object_id=row.get("path_scope_object_id"),
        workspace_source_mode=(
            str(row.get("workspace_source_mode") or "").strip().lower() or None
        ),
        workspace_set_object_id=row.get("workspace_set_object_id"),
        inline_policy_document=_load_json_object(row.get("inline_policy_document")),
        approval_policy_id=row.get("approval_policy_id"),
        is_active=bool(row.get("is_active")),
        has_override=bool(row.get("has_override")),
        override_id=row.get("override_id"),
        override_active=bool(row.get("override_active")),
        override_updated_at=row.get("override_updated_at"),
        created_by=row.get("created_by"),
        updated_by=row.get("updated_by"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _path_scope_object_row_to_response(row: dict[str, Any]) -> PathScopeObjectResponse:
    return PathScopeObjectResponse(
        id=int(row.get("id")),
        name=str(row.get("name") or ""),
        description=row.get("description"),
        owner_scope_type=str(row.get("owner_scope_type") or "global"),
        owner_scope_id=row.get("owner_scope_id"),
        path_scope_document=_load_json_object(row.get("path_scope_document")),
        is_active=bool(row.get("is_active")),
        created_by=row.get("created_by"),
        updated_by=row.get("updated_by"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _workspace_set_object_row_to_response(row: dict[str, Any]) -> WorkspaceSetObjectResponse:
    return WorkspaceSetObjectResponse(
        id=int(row.get("id")),
        name=str(row.get("name") or ""),
        description=row.get("description"),
        owner_scope_type=str(row.get("owner_scope_type") or "user"),
        owner_scope_id=row.get("owner_scope_id"),
        is_active=bool(row.get("is_active")),
        readiness_summary=row.get("readiness_summary"),
        created_by=row.get("created_by"),
        updated_by=row.get("updated_by"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _workspace_set_member_row_to_response(
    row: dict[str, Any],
) -> WorkspaceSetObjectMemberResponse:
    return WorkspaceSetObjectMemberResponse(
        workspace_set_object_id=int(row.get("workspace_set_object_id")),
        workspace_id=str(row.get("workspace_id") or ""),
        created_by=row.get("created_by"),
        created_at=row.get("created_at"),
    )


def _shared_workspace_row_to_response(row: dict[str, Any]) -> SharedWorkspaceResponse:
    return SharedWorkspaceResponse(
        id=int(row.get("id")),
        workspace_id=str(row.get("workspace_id") or ""),
        display_name=str(row.get("display_name") or ""),
        absolute_root=str(row.get("absolute_root") or ""),
        owner_scope_type=str(row.get("owner_scope_type") or "team"),
        owner_scope_id=row.get("owner_scope_id"),
        is_active=bool(row.get("is_active")),
        readiness_summary=row.get("readiness_summary"),
        created_by=row.get("created_by"),
        updated_by=row.get("updated_by"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _policy_assignment_workspace_row_to_response(
    row: dict[str, Any],
) -> PolicyAssignmentWorkspaceResponse:
    return PolicyAssignmentWorkspaceResponse(
        assignment_id=int(row.get("assignment_id")),
        workspace_id=str(row.get("workspace_id") or ""),
        created_by=row.get("created_by"),
        created_at=row.get("created_at"),
    )


def _policy_override_row_to_response(row: dict[str, Any]) -> PolicyOverrideResponse:
    return PolicyOverrideResponse(
        id=int(row.get("id")),
        assignment_id=int(row.get("assignment_id")),
        override_policy_document=_load_json_object(row.get("override_policy_document")),
        is_active=bool(row.get("is_active")),
        created_by=row.get("created_by"),
        updated_by=row.get("updated_by"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _approval_policy_row_to_response(row: dict[str, Any]) -> ApprovalPolicyResponse:
    return ApprovalPolicyResponse(
        id=int(row.get("id")),
        name=str(row.get("name") or ""),
        description=row.get("description"),
        owner_scope_type=str(row.get("owner_scope_type") or "global"),
        owner_scope_id=row.get("owner_scope_id"),
        mode=str(row.get("mode") or "allow_silently"),
        rules=_load_json_object(row.get("rules")),
        is_active=bool(row.get("is_active")),
        created_by=row.get("created_by"),
        updated_by=row.get("updated_by"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _approval_decision_row_to_response(row: dict[str, Any]) -> ApprovalDecisionResponse:
    return ApprovalDecisionResponse(
        id=int(row.get("id")),
        approval_policy_id=row.get("approval_policy_id"),
        context_key=str(row.get("context_key") or ""),
        conversation_id=row.get("conversation_id"),
        tool_name=str(row.get("tool_name") or ""),
        scope_key=str(row.get("scope_key") or ""),
        decision=str(row.get("decision") or "denied"),
        consume_on_match=bool(row.get("consume_on_match")),
        expires_at=row.get("expires_at"),
        consumed_at=row.get("consumed_at"),
        created_by=row.get("created_by"),
        created_at=row.get("created_at"),
    )


def _extract_context_key_user_id(context_key: str) -> int | None:
    match = re.search(r"(?:^|\|)user:(\d+)(?:\||$)", str(context_key))
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _validated_duration_options(rules: dict[str, Any] | None) -> list[str]:
    """Return supported approval durations from policy rules or raise for unsupported entries."""
    raw = (rules or {}).get("duration_options")
    if raw is None:
        return ["once", "session"]
    if not isinstance(raw, list):
        raise HTTPException(status_code=422, detail="rules.duration_options must be a list")
    out: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        value = str(entry or "").strip().lower()
        if not value:
            continue
        if value not in _SUPPORTED_APPROVAL_DURATIONS:
            raise HTTPException(
                status_code=422,
                detail=f"rules.duration_options contains unsupported duration '{value}'",
            )
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out or ["once", "session"]


def _normalized_approval_rules(rules: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize approval policy rules to the server-supported shape."""
    normalized = dict(rules or {})
    normalized["duration_options"] = _validated_duration_options(normalized)
    return normalized


def _approval_ttl_minutes(rules: dict[str, Any], duration: str) -> int:
    """Resolve a bounded TTL for scoped approvals."""
    duration_key = str(duration or "").strip().lower()
    raw = rules.get(f"{duration_key}_ttl_minutes")
    if raw is None:
        raw = rules.get("scoped_ttl_minutes")
    try:
        value = int(raw) if raw is not None else _DEFAULT_SCOPED_APPROVAL_TTL_MINUTES
    except (TypeError, ValueError):
        value = _DEFAULT_SCOPED_APPROVAL_TTL_MINUTES
    return max(1, min(value, 24 * 60))


def _approval_decision_lifetime(
    *,
    decision: str,
    duration: str,
    conversation_id: str | None,
    rules: dict[str, Any],
) -> tuple[bool, datetime | None]:
    """Compute server-side approval persistence from a validated duration selection."""
    normalized_decision = str(decision or "").strip().lower()
    normalized_duration = str(duration or "").strip().lower()
    if normalized_decision != "approved":
        return False, datetime.now(timezone.utc)
    if normalized_duration == "once":
        return True, None
    if normalized_duration in {"session", "conversation"}:
        if not str(conversation_id or "").strip():
            raise HTTPException(status_code=422, detail="conversation_id is required for scoped approvals")
        ttl_minutes = _approval_ttl_minutes(rules, normalized_duration)
        return False, datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
    raise HTTPException(status_code=422, detail=f"Unsupported approval duration '{normalized_duration}'")


async def _get_visible_policy_assignment_or_404(
    *,
    assignment_id: int,
    principal: AuthPrincipal,
    svc: McpHubService,
) -> dict[str, Any]:
    """Fetch a policy assignment and ensure the principal can access its owner scope."""
    assignment = await svc.get_policy_assignment(assignment_id)
    if assignment is None:
        raise ResourceNotFoundError("mcp_policy_assignment", identifier=str(assignment_id))
    _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=str(assignment.get("owner_scope_type") or "global"),
        owner_scope_id=assignment.get("owner_scope_id"),
    )
    return assignment


async def _get_visible_permission_profile_or_404(
    *,
    profile_id: int,
    principal: AuthPrincipal,
    svc: McpHubService,
) -> dict[str, Any]:
    """Fetch a permission profile and ensure the principal can access its owner scope."""
    profile = await svc.get_permission_profile(profile_id)
    if profile is None:
        raise ResourceNotFoundError("mcp_permission_profile", identifier=str(profile_id))
    _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=str(profile.get("owner_scope_type") or "global"),
        owner_scope_id=profile.get("owner_scope_id"),
    )
    return profile


async def _get_visible_workspace_set_object_or_404(
    *,
    workspace_set_object_id: int,
    principal: AuthPrincipal,
    svc: McpHubService,
) -> dict[str, Any]:
    """Fetch a workspace-set object and ensure the principal can access its owner scope."""
    row = await svc.get_workspace_set_object(workspace_set_object_id)
    if row is None:
        raise ResourceNotFoundError("mcp_workspace_set_object", identifier=str(workspace_set_object_id))
    _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=str(row.get("owner_scope_type") or "user"),
        owner_scope_id=row.get("owner_scope_id"),
    )
    return row


async def _get_visible_shared_workspace_or_404(
    *,
    shared_workspace_id: int,
    principal: AuthPrincipal,
    svc: McpHubService,
) -> dict[str, Any]:
    """Fetch a shared-workspace entry and ensure the principal can access its owner scope."""
    row = await svc.get_shared_workspace_entry(shared_workspace_id)
    if row is None:
        raise ResourceNotFoundError("mcp_shared_workspace", identifier=str(shared_workspace_id))
    _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=str(row.get("owner_scope_type") or "team"),
        owner_scope_id=row.get("owner_scope_id"),
    )
    return row


async def _validate_assignment_workspace_source(
    *,
    svc: McpHubService,
    workspace_source_mode: str | None,
    workspace_set_object_id: int | None,
    owner_scope_type: str,
    owner_scope_id: int | None,
) -> None:
    """Validate named-vs-inline workspace source selection for an assignment."""
    normalized_mode = str(workspace_source_mode or "").strip().lower()
    if not normalized_mode:
        return
    if normalized_mode not in {"inline", "named"}:
        raise HTTPException(status_code=422, detail="workspace_source_mode must be one of: inline, named")
    if normalized_mode == "named":
        if workspace_set_object_id is None:
            raise HTTPException(status_code=422, detail="workspace_set_object_id is required when workspace_source_mode is named")
        try:
            await svc.validate_workspace_set_object_reference(
                workspace_set_object_id=workspace_set_object_id,
                target_scope_type=owner_scope_type,
                target_scope_id=owner_scope_id,
            )
        except ResourceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except BadRequestError as exc:
            raise HTTPException(status_code=400, detail=_bad_request_detail(exc)) from exc
        return
    if workspace_set_object_id is not None:
        raise HTTPException(status_code=422, detail="workspace_set_object_id may only be set when workspace_source_mode is named")


def _bad_request_detail(exc: BadRequestError) -> Any:
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict):
        return detail
    return str(exc)


async def _assignment_base_policy_document(
    *,
    svc: McpHubService,
    assignment: dict[str, Any],
) -> tuple[int | None, dict[str, Any]]:
    """Resolve the merged profile-plus-inline base policy for an assignment."""
    base_document: dict[str, Any] = {}
    profile_id = assignment.get("profile_id")
    if profile_id is not None:
        profile_row = await svc.get_permission_profile(int(profile_id))
        if profile_row and bool(profile_row.get("is_active", True)):
            base_document = _merge_policy_documents(
                base_document,
                _load_json_object(profile_row.get("policy_document")),
            )
    base_document = _merge_policy_documents(
        base_document,
        _load_json_object(assignment.get("inline_policy_document")),
    )
    return (int(profile_id) if profile_id is not None else None, base_document)


def _binding_row_to_response(row: dict[str, Any]) -> CredentialBindingResponse:
    return CredentialBindingResponse(
        id=int(row.get("id")),
        binding_target_type=str(row.get("binding_target_type") or ""),
        binding_target_id=str(row.get("binding_target_id") or ""),
        external_server_id=str(row.get("external_server_id") or ""),
        slot_name=(
            str(row.get("slot_name") or "").strip()
            if str(row.get("slot_name") or "").strip()
            else None
        ),
        credential_ref=str(row.get("credential_ref") or "server"),
        managed_secret_ref_id=(
            int(row.get("managed_secret_ref_id"))
            if row.get("managed_secret_ref_id") is not None
            else None
        ),
        binding_mode=str(row.get("binding_mode") or "grant"),
        usage_rules=_load_json_object(row.get("usage_rules")),
        created_by=row.get("created_by"),
        updated_by=row.get("updated_by"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


@router.get("/tool-registry", response_model=list[ToolRegistryEntryResponse])
async def list_tool_registry(
    _principal: AuthPrincipal = Depends(get_auth_principal),
    registry: McpHubToolRegistryService = Depends(get_mcp_hub_tool_registry_dep),
) -> list[ToolRegistryEntryResponse]:
    """List normalized MCP tool metadata derived from the live module registry."""
    return [ToolRegistryEntryResponse.model_validate(row) for row in await registry.list_entries()]


@router.get("/tool-registry/modules", response_model=list[ToolRegistryModuleResponse])
async def list_tool_registry_modules(
    _principal: AuthPrincipal = Depends(get_auth_principal),
    registry: McpHubToolRegistryService = Depends(get_mcp_hub_tool_registry_dep),
) -> list[ToolRegistryModuleResponse]:
    """List grouped MCP tool metadata summaries by module."""
    return [ToolRegistryModuleResponse.model_validate(row) for row in await registry.list_modules()]


@router.get("/tool-registry/summary", response_model=ToolRegistrySummaryResponse)
async def get_tool_registry_summary(
    _principal: AuthPrincipal = Depends(get_auth_principal),
    registry: McpHubToolRegistryService = Depends(get_mcp_hub_tool_registry_dep),
) -> ToolRegistrySummaryResponse:
    """Return tool entries and module summaries from a single registry enumeration."""
    summary = await registry.get_summary()
    return ToolRegistrySummaryResponse(
        entries=[ToolRegistryEntryResponse.model_validate(row) for row in summary.get("entries", [])],
        modules=[ToolRegistryModuleResponse.model_validate(row) for row in summary.get("modules", [])],
    )


@router.post("/capability-mappings/preview", response_model=CapabilityAdapterMappingPreviewResponse)
async def preview_capability_mapping(
    payload: CapabilityAdapterMappingCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubCapabilityAdapterService = Depends(get_mcp_hub_capability_adapter_service),
) -> CapabilityAdapterMappingPreviewResponse:
    """Validate and preview a capability adapter mapping without persisting it."""
    _require_mutation_permission(principal)
    resolved_scope_type, resolved_scope_id = _resolve_capability_adapter_mutation_scope(
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
    )
    try:
        preview = await svc.preview_mapping(
            mapping_id=payload.mapping_id,
            title=payload.title,
            description=payload.description,
            owner_scope_type=resolved_scope_type,
            owner_scope_id=resolved_scope_id,
            capability_name=payload.capability_name,
            adapter_contract_version=payload.adapter_contract_version,
            resolved_policy_document=payload.resolved_policy_document,
            supported_environment_requirements=payload.supported_environment_requirements,
            is_active=payload.is_active,
        )
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=_bad_request_detail(exc)) from exc
    return CapabilityAdapterMappingPreviewResponse.model_validate(preview)


@router.get("/capability-mappings", response_model=list[CapabilityAdapterMappingResponse])
async def list_capability_mappings(
    owner_scope_type: str | None = None,
    owner_scope_id: int | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubCapabilityAdapterService = Depends(get_mcp_hub_capability_adapter_service),
) -> list[CapabilityAdapterMappingResponse]:
    """List capability adapter mappings visible to the current principal."""
    filters = _resolve_visible_capability_adapter_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_id in filters:
        rows.extend(
            await svc.list_capability_adapter_mappings(
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
            )
        )
    rows = _dedupe_rows(rows)
    return [CapabilityAdapterMappingResponse.model_validate(row) for row in rows]


@router.post("/capability-mappings", response_model=CapabilityAdapterMappingResponse, status_code=201)
async def create_capability_mapping(
    payload: CapabilityAdapterMappingCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubCapabilityAdapterService = Depends(get_mcp_hub_capability_adapter_service),
) -> CapabilityAdapterMappingResponse:
    """Create a capability adapter mapping after validation and grant-authority checks."""
    _require_mutation_permission(principal)
    resolved_scope_type, resolved_scope_id = _resolve_capability_adapter_mutation_scope(
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
    )
    try:
        preview = await svc.preview_mapping(
            mapping_id=payload.mapping_id,
            title=payload.title,
            description=payload.description,
            owner_scope_type=resolved_scope_type,
            owner_scope_id=resolved_scope_id,
            capability_name=payload.capability_name,
            adapter_contract_version=payload.adapter_contract_version,
            resolved_policy_document=payload.resolved_policy_document,
            supported_environment_requirements=payload.supported_environment_requirements,
            is_active=payload.is_active,
        )
        _require_grant_authority(
            principal,
            dict(preview["normalized_mapping"].get("resolved_policy_document") or {}),
        )
        created = await svc.create_mapping(
            mapping_id=payload.mapping_id,
            title=payload.title,
            description=payload.description,
            owner_scope_type=resolved_scope_type,
            owner_scope_id=resolved_scope_id,
            capability_name=payload.capability_name,
            adapter_contract_version=payload.adapter_contract_version,
            resolved_policy_document=payload.resolved_policy_document,
            supported_environment_requirements=payload.supported_environment_requirements,
            actor_id=principal.user_id,
            is_active=payload.is_active,
        )
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=_bad_request_detail(exc)) from exc
    return CapabilityAdapterMappingResponse.model_validate(created)


@router.put("/capability-mappings/{capability_adapter_mapping_id}", response_model=CapabilityAdapterMappingResponse)
async def update_capability_mapping(
    capability_adapter_mapping_id: int,
    payload: CapabilityAdapterMappingUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubCapabilityAdapterService = Depends(get_mcp_hub_capability_adapter_service),
) -> CapabilityAdapterMappingResponse:
    """Update a capability adapter mapping after validation and grant-authority checks."""
    _require_mutation_permission(principal)
    update_fields = payload.model_dump(exclude_unset=True)
    try:
        preview = await svc.preview_update(
            capability_adapter_mapping_id,
            **update_fields,
        )
        _require_grant_authority(
            principal,
            dict(preview["normalized_mapping"].get("resolved_policy_document") or {}),
        )
        updated = await svc.update_mapping(
            capability_adapter_mapping_id,
            **update_fields,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=_bad_request_detail(exc)) from exc
    return CapabilityAdapterMappingResponse.model_validate(updated)


@router.delete("/capability-mappings/{capability_adapter_mapping_id}", response_model=MCPHubDeleteResponse)
async def delete_capability_mapping(
    capability_adapter_mapping_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubCapabilityAdapterService = Depends(get_mcp_hub_capability_adapter_service),
) -> MCPHubDeleteResponse:
    """Delete a capability adapter mapping."""
    _require_mutation_permission(principal)
    try:
        await svc.delete_capability_adapter_mapping(capability_adapter_mapping_id)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MCPHubDeleteResponse(ok=True)


@router.post("/governance-packs/dry-run", response_model=GovernancePackDryRunResponse)
async def dry_run_governance_pack(
    payload: GovernancePackDryRunRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackService = Depends(get_mcp_hub_governance_pack_service),
) -> GovernancePackDryRunResponse:
    """Validate a governance-pack document and report dry-run compatibility details."""
    _require_mutation_permission(principal)
    resolved_scope_type, resolved_scope_id = _resolve_governance_pack_mutation_scope(
        principal=principal,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
    )
    pack_document = payload.pack.model_dump()
    pack_id, pack_version = _governance_pack_manifest_summary(pack_document)
    try:
        report = await svc.dry_run_pack_document(
            document=pack_document,
            owner_scope_type=resolved_scope_type,
            owner_scope_id=resolved_scope_id,
        )
    except ValueError as exc:
        logger.warning(
            "Governance pack dry-run rejected for pack {}@{} in scope {}:{}: {}",
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    report_payload = report.model_dump() if hasattr(report, "model_dump") else dict(report)
    return GovernancePackDryRunResponse.model_validate({"report": report_payload})


@router.post("/governance-packs/source/prepare", response_model=GovernancePackSourcePrepareResponse, status_code=201)
async def prepare_governance_pack_source(
    payload: GovernancePackSourcePrepareRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackDistributionService = Depends(get_mcp_hub_governance_pack_distribution_service),
) -> GovernancePackSourcePrepareResponse:
    """Resolve a trusted governance-pack source into a pinned prepared candidate."""
    _require_mutation_permission(principal)
    try:
        prepared = await svc.prepare_source_candidate(
            source=payload.source.model_dump(),
            actor_id=principal.user_id,
        )
    except ValueError as exc:
        logger.warning("Governance pack source prepare rejected for actor {}: {}", principal.user_id, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GovernancePackSourcePrepareResponse.model_validate(prepared)


@router.post("/governance-packs/source/dry-run", response_model=GovernancePackDryRunResponse)
async def dry_run_governance_pack_source_candidate(
    payload: GovernancePackSourceDryRunRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    distribution_svc: McpHubGovernancePackDistributionService = Depends(get_mcp_hub_governance_pack_distribution_service),
    governance_svc: McpHubGovernancePackService = Depends(get_mcp_hub_governance_pack_service),
) -> GovernancePackDryRunResponse:
    """Dry-run a previously prepared governance-pack source candidate in the target scope."""
    _require_mutation_permission(principal)
    resolved_scope_type, resolved_scope_id = _resolve_governance_pack_mutation_scope(
        principal=principal,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
    )
    try:
        prepared = await distribution_svc.load_prepared_candidate(
            payload.candidate_id,
            actor_id=principal.user_id,
            revalidate_trust=True,
        )
    except ValueError as exc:
        logger.warning(
            "Governance pack source dry-run rejected for candidate {} in scope {}:{}: {}",
            payload.candidate_id,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pack_document = dict(prepared.get("pack_document") or {})
    pack_id, pack_version = _governance_pack_manifest_summary(pack_document)
    try:
        report = await governance_svc.dry_run_pack_document(
            document=pack_document,
            owner_scope_type=resolved_scope_type,
            owner_scope_id=resolved_scope_id,
        )
    except ValueError as exc:
        logger.warning(
            "Governance pack source dry-run rejected for candidate {} pack {}@{} in scope {}:{}: {}",
            payload.candidate_id,
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    report_payload = report.model_dump() if hasattr(report, "model_dump") else dict(report)
    return GovernancePackDryRunResponse.model_validate({"report": report_payload})


@router.post(
    "/governance-packs/{governance_pack_id}/check-updates",
    response_model=GovernancePackSourceUpdateCheckResponse,
)
async def check_governance_pack_updates(
    governance_pack_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackDistributionService = Depends(get_mcp_hub_governance_pack_distribution_service),
) -> GovernancePackSourceUpdateCheckResponse:
    """Check whether a Git-backed governance-pack install has a newer trusted source candidate."""
    _require_mutation_permission(principal)
    try:
        result = await svc.check_for_updates(governance_pack_id)
    except ValueError as exc:
        logger.warning(
            "Governance pack update check rejected for pack {} and actor {}: {}",
            governance_pack_id,
            principal.user_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GovernancePackSourceUpdateCheckResponse.model_validate(result)


@router.post(
    "/governance-packs/{governance_pack_id}/prepare-upgrade-candidate",
    response_model=GovernancePackSourceUpgradePrepareResponse,
    status_code=201,
)
async def prepare_governance_pack_upgrade_candidate(
    governance_pack_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackDistributionService = Depends(get_mcp_hub_governance_pack_distribution_service),
) -> GovernancePackSourceUpgradePrepareResponse:
    """Prepare a pinned source candidate for a newer Git-backed governance-pack update."""
    _require_mutation_permission(principal)
    try:
        result = await svc.prepare_upgrade_candidate(
            governance_pack_id=governance_pack_id,
            actor_id=principal.user_id,
        )
    except ValueError as exc:
        logger.warning(
            "Governance pack upgrade candidate prepare rejected for pack {} and actor {}: {}",
            governance_pack_id,
            principal.user_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GovernancePackSourceUpgradePrepareResponse.model_validate(result)


@router.post("/governance-packs/dry-run-upgrade", response_model=GovernancePackUpgradeDryRunResponse)
async def dry_run_governance_pack_upgrade(
    payload: GovernancePackUpgradeDryRunRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackService = Depends(get_mcp_hub_governance_pack_service),
) -> GovernancePackUpgradeDryRunResponse:
    """Validate a governance-pack upgrade plan for an installed pack in the same scope."""
    _require_mutation_permission(principal)
    resolved_scope_type, resolved_scope_id = _resolve_governance_pack_mutation_scope(
        principal=principal,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
    )
    pack_document = payload.pack.model_dump()
    pack_id, pack_version = _governance_pack_manifest_summary(pack_document)
    try:
        plan = await svc.dry_run_upgrade_document(
            source_governance_pack_id=payload.source_governance_pack_id,
            document=pack_document,
            owner_scope_type=resolved_scope_type,
            owner_scope_id=resolved_scope_id,
        )
    except ValueError as exc:
        logger.warning(
            "Governance pack upgrade dry-run rejected for source {} target {}@{} in scope {}:{}: {}",
            payload.source_governance_pack_id,
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    plan_payload = plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)
    return GovernancePackUpgradeDryRunResponse.model_validate({"plan": plan_payload})


@router.post("/governance-packs/source/dry-run-upgrade", response_model=GovernancePackUpgradeDryRunResponse)
async def dry_run_governance_pack_source_upgrade(
    payload: GovernancePackSourceUpgradeDryRunRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    distribution_svc: McpHubGovernancePackDistributionService = Depends(get_mcp_hub_governance_pack_distribution_service),
    governance_svc: McpHubGovernancePackService = Depends(get_mcp_hub_governance_pack_service),
) -> GovernancePackUpgradeDryRunResponse:
    """Dry-run a governance-pack upgrade using a prepared source candidate."""
    _require_mutation_permission(principal)
    resolved_scope_type, resolved_scope_id = _resolve_governance_pack_mutation_scope(
        principal=principal,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
    )
    try:
        prepared = await distribution_svc.load_prepared_candidate(
            payload.candidate_id,
            actor_id=principal.user_id,
            revalidate_trust=True,
        )
    except ValueError as exc:
        logger.warning(
            "Governance pack source upgrade dry-run rejected for source {} candidate {} in scope {}:{}: {}",
            payload.source_governance_pack_id,
            payload.candidate_id,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pack_document = dict(prepared.get("pack_document") or {})
    _require_grant_authority(principal, _governance_pack_grant_document(pack_document))
    pack_id, pack_version = _governance_pack_manifest_summary(pack_document)
    try:
        plan = await governance_svc.dry_run_upgrade_document(
            source_governance_pack_id=payload.source_governance_pack_id,
            document=pack_document,
            owner_scope_type=resolved_scope_type,
            owner_scope_id=resolved_scope_id,
        )
    except ValueError as exc:
        logger.warning(
            "Governance pack source upgrade dry-run rejected for source {} candidate {} target {}@{} in scope {}:{}: {}",
            payload.source_governance_pack_id,
            payload.candidate_id,
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    plan_payload = plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)
    return GovernancePackUpgradeDryRunResponse.model_validate({"plan": plan_payload})


@router.post("/governance-packs/import", response_model=GovernancePackImportResponse, status_code=201)
async def import_governance_pack(
    payload: GovernancePackImportRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackService = Depends(get_mcp_hub_governance_pack_service),
) -> GovernancePackImportResponse:
    """Persist an importable governance-pack document into immutable MCP Hub base objects."""
    _require_mutation_permission(principal)
    resolved_scope_type, resolved_scope_id = _resolve_governance_pack_mutation_scope(
        principal=principal,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
    )
    pack_document = payload.pack.model_dump()
    _require_grant_authority(principal, _governance_pack_grant_document(pack_document))
    pack_id, pack_version = _governance_pack_manifest_summary(pack_document)
    try:
        result = await svc.import_pack_document(
            document=pack_document,
            owner_scope_type=resolved_scope_type,
            owner_scope_id=resolved_scope_id,
            actor_id=principal.user_id,
        )
    except GovernancePackAlreadyExistsError as exc:
        logger.warning(
            "Governance pack import conflict for pack {}@{} in scope {}:{}: {}",
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning(
            "Governance pack import rejected for pack {}@{} in scope {}:{}: {}",
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GovernancePackImportResponse.model_validate(result)


@router.post("/governance-packs/source/execute-upgrade", response_model=GovernancePackUpgradeExecutionResponse)
async def execute_governance_pack_source_upgrade(
    payload: GovernancePackSourceUpgradeExecuteRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    distribution_svc: McpHubGovernancePackDistributionService = Depends(get_mcp_hub_governance_pack_distribution_service),
    governance_svc: McpHubGovernancePackService = Depends(get_mcp_hub_governance_pack_service),
) -> GovernancePackUpgradeExecutionResponse:
    """Execute a governance-pack upgrade using a validated prepared source candidate."""
    _require_mutation_permission(principal)
    resolved_scope_type, resolved_scope_id = _resolve_governance_pack_mutation_scope(
        principal=principal,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
    )
    try:
        prepared = await distribution_svc.validate_prepared_upgrade_candidate(
            governance_pack_id=payload.source_governance_pack_id,
            candidate_id=payload.candidate_id,
            actor_id=principal.user_id,
        )
    except ValueError as exc:
        logger.warning(
            "Governance pack source upgrade execute rejected for source {} candidate {} in scope {}:{}: {}",
            payload.source_governance_pack_id,
            payload.candidate_id,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pack_document = dict(prepared.get("pack_document") or {})
    _require_grant_authority(principal, _governance_pack_grant_document(pack_document))
    pack_id, pack_version = _governance_pack_manifest_summary(pack_document)
    try:
        result = await governance_svc.execute_upgrade_document(
            source_governance_pack_id=payload.source_governance_pack_id,
            document=pack_document,
            owner_scope_type=resolved_scope_type,
            owner_scope_id=resolved_scope_id,
            actor_id=principal.user_id,
            planner_inputs_fingerprint=payload.planner_inputs_fingerprint,
            adapter_state_fingerprint=payload.adapter_state_fingerprint,
        )
    except (GovernancePackAlreadyExistsError, GovernancePackUpgradeConflictError, GovernancePackUpgradeStaleError) as exc:
        logger.warning(
            "Governance pack source upgrade execute conflict for source {} candidate {} target {}@{} in scope {}:{}: {}",
            payload.source_governance_pack_id,
            payload.candidate_id,
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning(
            "Governance pack source upgrade execute rejected for source {} candidate {} target {}@{} in scope {}:{}: {}",
            payload.source_governance_pack_id,
            payload.candidate_id,
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result_payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    return GovernancePackUpgradeExecutionResponse.model_validate(result_payload)


@router.post("/governance-packs/source/import", response_model=GovernancePackImportResponse, status_code=201)
async def import_governance_pack_source_candidate(
    payload: GovernancePackSourceImportRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    distribution_svc: McpHubGovernancePackDistributionService = Depends(get_mcp_hub_governance_pack_distribution_service),
    governance_svc: McpHubGovernancePackService = Depends(get_mcp_hub_governance_pack_service),
) -> GovernancePackImportResponse:
    """Import a prepared governance-pack source candidate into immutable MCP Hub base objects."""
    _require_mutation_permission(principal)
    resolved_scope_type, resolved_scope_id = _resolve_governance_pack_mutation_scope(
        principal=principal,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
    )
    try:
        prepared = await distribution_svc.load_prepared_candidate(
            payload.candidate_id,
            actor_id=principal.user_id,
            revalidate_trust=True,
        )
    except ValueError as exc:
        logger.warning(
            "Governance pack source import rejected for candidate {} in scope {}:{}: {}",
            payload.candidate_id,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    pack_document = dict(prepared.get("pack_document") or {})
    source_metadata = dict(prepared.get("candidate") or {})
    _require_grant_authority(principal, _governance_pack_grant_document(pack_document))
    pack_id, pack_version = _governance_pack_manifest_summary(pack_document)
    try:
        result = await governance_svc.import_pack_document(
            document=pack_document,
            owner_scope_type=resolved_scope_type,
            owner_scope_id=resolved_scope_id,
            actor_id=principal.user_id,
            source_metadata=source_metadata,
        )
    except GovernancePackAlreadyExistsError as exc:
        logger.warning(
            "Governance pack source import conflict for candidate {} pack {}@{} in scope {}:{}: {}",
            payload.candidate_id,
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning(
            "Governance pack source import rejected for candidate {} pack {}@{} in scope {}:{}: {}",
            payload.candidate_id,
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GovernancePackImportResponse.model_validate(result)


@router.post("/governance-packs/execute-upgrade", response_model=GovernancePackUpgradeExecutionResponse)
async def execute_governance_pack_upgrade(
    payload: GovernancePackUpgradeExecuteRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackService = Depends(get_mcp_hub_governance_pack_service),
) -> GovernancePackUpgradeExecutionResponse:
    """Execute a previously dry-run governance-pack upgrade when the planner state is unchanged."""
    _require_mutation_permission(principal)
    resolved_scope_type, resolved_scope_id = _resolve_governance_pack_mutation_scope(
        principal=principal,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
    )
    pack_document = payload.pack.model_dump()
    _require_grant_authority(principal, _governance_pack_grant_document(pack_document))
    pack_id, pack_version = _governance_pack_manifest_summary(pack_document)
    try:
        result = await svc.execute_upgrade_document(
            source_governance_pack_id=payload.source_governance_pack_id,
            document=pack_document,
            owner_scope_type=resolved_scope_type,
            owner_scope_id=resolved_scope_id,
            actor_id=principal.user_id,
            planner_inputs_fingerprint=payload.planner_inputs_fingerprint,
            adapter_state_fingerprint=payload.adapter_state_fingerprint,
        )
    except (GovernancePackAlreadyExistsError, GovernancePackUpgradeConflictError, GovernancePackUpgradeStaleError) as exc:
        logger.warning(
            "Governance pack upgrade execute conflict for source {} target {}@{} in scope {}:{}: {}",
            payload.source_governance_pack_id,
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning(
            "Governance pack upgrade execute rejected for source {} target {}@{} in scope {}:{}: {}",
            payload.source_governance_pack_id,
            pack_id,
            pack_version,
            resolved_scope_type,
            resolved_scope_id,
            exc,
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result_payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    return GovernancePackUpgradeExecutionResponse.model_validate(result_payload)


@router.get("/governance-packs", response_model=list[GovernancePackSummaryResponse])
async def list_governance_packs(
    owner_scope_type: str | None = None,
    owner_scope_id: int | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackService = Depends(get_mcp_hub_governance_pack_service),
) -> list[GovernancePackSummaryResponse]:
    """List governance packs visible to the current principal with scope-constrained filtering."""
    filters = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_id in filters:
        rows.extend(
            await svc.list_governance_packs(
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
            )
        )
    rows = _dedupe_rows(rows)
    return [GovernancePackSummaryResponse.model_validate(row) for row in rows]


@router.get(
    "/governance-packs/trust-policy",
    response_model=GovernancePackTrustPolicyResponse,
)
async def get_governance_pack_trust_policy(
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackTrustService = Depends(get_mcp_hub_governance_pack_trust_service),
) -> GovernancePackTrustPolicyResponse:
    """Return the deployment-wide governance-pack trust policy."""
    _require_mutation_permission(principal)
    try:
        return GovernancePackTrustPolicyResponse.model_validate(await svc.get_policy())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put(
    "/governance-packs/trust-policy",
    response_model=GovernancePackTrustPolicyResponse,
)
async def update_governance_pack_trust_policy(
    payload: GovernancePackTrustPolicyRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackTrustService = Depends(get_mcp_hub_governance_pack_trust_service),
) -> GovernancePackTrustPolicyResponse:
    """Update the deployment-wide governance-pack trust policy."""
    _require_mutation_permission(principal)
    try:
        updated = await svc.update_policy(payload.model_dump(), actor_id=principal.user_id)
    except GovernancePackTrustPolicyStaleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GovernancePackTrustPolicyResponse.model_validate(updated)


@router.get(
    "/governance-packs/{governance_pack_id}/upgrade-history",
    response_model=list[GovernancePackUpgradeHistoryEntryResponse],
)
async def list_governance_pack_upgrade_history(
    governance_pack_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackService = Depends(get_mcp_hub_governance_pack_service),
) -> list[GovernancePackUpgradeHistoryEntryResponse]:
    """List the recorded upgrade lineage for the governance-pack identity installed in this scope."""
    row = await svc.get_governance_pack_detail(governance_pack_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"mcp_governance_pack '{governance_pack_id}' was not found")
    _assert_principal_can_access_scope(
        principal,
        owner_scope_type=str(row.get("owner_scope_type") or "global"),
        owner_scope_id=row.get("owner_scope_id"),
    )
    history = await svc.list_governance_pack_upgrade_history(governance_pack_id)
    return [GovernancePackUpgradeHistoryEntryResponse.model_validate(item) for item in history]


@router.get("/governance-packs/{governance_pack_id}", response_model=GovernancePackDetailResponse)
async def get_governance_pack_detail(
    governance_pack_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubGovernancePackService = Depends(get_mcp_hub_governance_pack_service),
) -> GovernancePackDetailResponse:
    """Return the persisted governance pack manifest plus imported-object provenance links."""
    row = await svc.get_governance_pack_detail(governance_pack_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"mcp_governance_pack '{governance_pack_id}' was not found")
    _assert_principal_can_access_scope(
        principal,
        owner_scope_type=str(row.get("owner_scope_type") or "global"),
        owner_scope_id=row.get("owner_scope_id"),
    )
    return GovernancePackDetailResponse.model_validate(row)


@router.post("/permission-profiles", response_model=PermissionProfileResponse, status_code=201)
async def create_permission_profile(
    payload: PermissionProfileCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> PermissionProfileResponse:
    """Create a new permission profile within the provided owner scope."""
    _require_mutation_permission(principal)
    policy_document = _normalize_policy_document_for_storage(
        payload.policy_document,
        allow_inherited_scope=False,
    )
    _require_grant_authority(principal, policy_document)
    try:
        await svc.validate_path_scope_object_reference(
            path_scope_object_id=payload.path_scope_object_id,
            target_scope_type=payload.owner_scope_type,
            target_scope_id=payload.owner_scope_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    row = await svc.create_permission_profile(
        name=payload.name,
        description=payload.description,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
        mode=payload.mode,
        path_scope_object_id=payload.path_scope_object_id,
        policy_document=policy_document,
        is_active=payload.is_active,
        actor_id=principal.user_id,
    )
    return _permission_profile_row_to_response(row)


@router.get("/permission-profiles", response_model=list[PermissionProfileResponse])
async def list_permission_profiles(
    owner_scope_type: str | None = None,
    owner_scope_id: int | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[PermissionProfileResponse]:
    """List permission profiles visible to the current principal with scope-constrained filtering."""
    filters = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_id in filters:
        rows.extend(
            await svc.list_permission_profiles(
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
            )
        )
    rows = _dedupe_rows(rows)
    return [_permission_profile_row_to_response(row) for row in rows]


@router.put("/permission-profiles/{profile_id}", response_model=PermissionProfileResponse)
async def update_permission_profile(
    profile_id: int,
    payload: PermissionProfileUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> PermissionProfileResponse:
    """Update an existing permission profile by id."""
    _require_mutation_permission(principal)
    update_fields = payload.model_dump(exclude_unset=True)
    if "policy_document" in update_fields:
        update_fields["policy_document"] = _normalize_policy_document_for_storage(
            update_fields.get("policy_document") or {},
            allow_inherited_scope=False,
        )
        _require_grant_authority(principal, update_fields.get("policy_document") or {})
    try:
        existing = await svc.get_permission_profile(profile_id)
        if not existing:
            raise ResourceNotFoundError("mcp_permission_profile", identifier=str(profile_id))
        _ensure_mutable_governance_object(existing, "Permission profile")
        if (
            "path_scope_object_id" in update_fields
            or "workspace_source_mode" in update_fields
            or "workspace_set_object_id" in update_fields
            or "owner_scope_type" in update_fields
            or "owner_scope_id" in update_fields
        ):
            await svc.validate_path_scope_object_reference(
                path_scope_object_id=update_fields.get(
                    "path_scope_object_id",
                    existing.get("path_scope_object_id"),
                ),
                target_scope_type=str(update_fields.get("owner_scope_type") or existing.get("owner_scope_type") or "global"),
                target_scope_id=update_fields.get("owner_scope_id", existing.get("owner_scope_id")),
            )
        row = await svc.update_permission_profile(
            profile_id,
            actor_id=principal.user_id,
            **update_fields,
        )
        if not row:
            raise ResourceNotFoundError("mcp_permission_profile", identifier=str(profile_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _permission_profile_row_to_response(row)


@router.delete("/permission-profiles/{profile_id}", response_model=MCPHubDeleteResponse)
async def delete_permission_profile(
    profile_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Delete a permission profile by id."""
    _require_mutation_permission(principal)
    try:
        existing = await svc.get_permission_profile(profile_id)
        if not existing:
            raise ResourceNotFoundError("mcp_permission_profile", identifier=str(profile_id))
        _ensure_mutable_governance_object(existing, "Permission profile")
        deleted = await svc.delete_permission_profile(profile_id, actor_id=principal.user_id)
        if not deleted:
            raise ResourceNotFoundError("mcp_permission_profile", identifier=str(profile_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MCPHubDeleteResponse(ok=True)


@router.post("/path-scope-objects", response_model=PathScopeObjectResponse, status_code=201)
async def create_path_scope_object(
    payload: PathScopeObjectCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> PathScopeObjectResponse:
    _require_mutation_permission(principal)
    row = await svc.create_path_scope_object(
        name=payload.name,
        description=payload.description,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
        path_scope_document=_normalize_path_scope_document_for_storage(payload.path_scope_document),
        is_active=payload.is_active,
        actor_id=principal.user_id,
    )
    return _path_scope_object_row_to_response(row)


@router.get("/path-scope-objects", response_model=list[PathScopeObjectResponse])
async def list_path_scope_objects(
    owner_scope_type: str | None = None,
    owner_scope_id: int | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[PathScopeObjectResponse]:
    filters = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_id in filters:
        rows.extend(
            await svc.list_path_scope_objects(
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
            )
        )
    rows = _dedupe_rows(rows)
    return [_path_scope_object_row_to_response(row) for row in rows]


@router.put("/path-scope-objects/{path_scope_object_id}", response_model=PathScopeObjectResponse)
async def update_path_scope_object(
    path_scope_object_id: int,
    payload: PathScopeObjectUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> PathScopeObjectResponse:
    _require_mutation_permission(principal)
    update_fields = payload.model_dump(exclude_unset=True)
    if "path_scope_document" in update_fields:
        update_fields["path_scope_document"] = _normalize_path_scope_document_for_storage(
            update_fields.get("path_scope_document") or {}
        )
    try:
        row = await svc.update_path_scope_object(
            path_scope_object_id,
            actor_id=principal.user_id,
            **update_fields,
        )
        if not row:
            raise ResourceNotFoundError("mcp_path_scope_object", identifier=str(path_scope_object_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _path_scope_object_row_to_response(row)


@router.delete("/path-scope-objects/{path_scope_object_id}", response_model=MCPHubDeleteResponse)
async def delete_path_scope_object(
    path_scope_object_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    _require_mutation_permission(principal)
    try:
        deleted = await svc.delete_path_scope_object(path_scope_object_id, actor_id=principal.user_id)
        if not deleted:
            raise ResourceNotFoundError("mcp_path_scope_object", identifier=str(path_scope_object_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MCPHubDeleteResponse(ok=True)


@router.post("/workspace-set-objects", response_model=WorkspaceSetObjectResponse, status_code=201)
async def create_workspace_set_object(
    payload: WorkspaceSetObjectCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> WorkspaceSetObjectResponse:
    _require_mutation_permission(principal)
    try:
        row = await svc.create_workspace_set_object(
            name=payload.name,
            description=payload.description,
            owner_scope_type=payload.owner_scope_type,
            owner_scope_id=payload.owner_scope_id
            if payload.owner_scope_id is not None
            else principal.user_id,
            is_active=payload.is_active,
            actor_id=principal.user_id,
        )
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    row = dict(row)
    row["readiness_summary"] = await svc.get_workspace_set_readiness_summary(
        workspace_set_object_id=int(row.get("id") or 0)
    )
    return _workspace_set_object_row_to_response(row)


@router.get("/workspace-set-objects", response_model=list[WorkspaceSetObjectResponse])
async def list_workspace_set_objects(
    owner_scope_type: str | None = None,
    owner_scope_id: int | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[WorkspaceSetObjectResponse]:
    filters = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_id in filters:
        rows.extend(
            await svc.list_workspace_set_objects(
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
            )
        )
    rows = _dedupe_rows(rows)
    rows_with_readiness: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["readiness_summary"] = await svc.get_workspace_set_readiness_summary(
            workspace_set_object_id=int(row.get("id") or 0)
        )
        rows_with_readiness.append(enriched)
    return [_workspace_set_object_row_to_response(row) for row in rows_with_readiness]


@router.put("/workspace-set-objects/{workspace_set_object_id}", response_model=WorkspaceSetObjectResponse)
async def update_workspace_set_object(
    workspace_set_object_id: int,
    payload: WorkspaceSetObjectUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> WorkspaceSetObjectResponse:
    _require_mutation_permission(principal)
    try:
        await _get_visible_workspace_set_object_or_404(
            workspace_set_object_id=workspace_set_object_id,
            principal=principal,
            svc=svc,
        )
        row = await svc.update_workspace_set_object(
            workspace_set_object_id,
            actor_id=principal.user_id,
            **payload.model_dump(exclude_unset=True),
        )
        if not row:
            raise ResourceNotFoundError("mcp_workspace_set_object", identifier=str(workspace_set_object_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _workspace_set_object_row_to_response(row)


@router.delete("/workspace-set-objects/{workspace_set_object_id}", response_model=MCPHubDeleteResponse)
async def delete_workspace_set_object(
    workspace_set_object_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    _require_mutation_permission(principal)
    try:
        await _get_visible_workspace_set_object_or_404(
            workspace_set_object_id=workspace_set_object_id,
            principal=principal,
            svc=svc,
        )
        deleted = await svc.delete_workspace_set_object(workspace_set_object_id, actor_id=principal.user_id)
        if not deleted:
            raise ResourceNotFoundError("mcp_workspace_set_object", identifier=str(workspace_set_object_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MCPHubDeleteResponse(ok=True)


@router.get(
    "/workspace-set-objects/{workspace_set_object_id}/members",
    response_model=list[WorkspaceSetObjectMemberResponse],
)
async def list_workspace_set_members(
    workspace_set_object_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[WorkspaceSetObjectMemberResponse]:
    await _get_visible_workspace_set_object_or_404(
        workspace_set_object_id=workspace_set_object_id,
        principal=principal,
        svc=svc,
    )
    rows = await svc.list_workspace_set_members(workspace_set_object_id)
    return [_workspace_set_member_row_to_response(row) for row in rows]


@router.post(
    "/workspace-set-objects/{workspace_set_object_id}/members",
    response_model=WorkspaceSetObjectMemberResponse,
    status_code=201,
)
async def add_workspace_set_member(
    workspace_set_object_id: int,
    payload: WorkspaceSetObjectMemberCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> WorkspaceSetObjectMemberResponse:
    _require_mutation_permission(principal)
    try:
        await _get_visible_workspace_set_object_or_404(
            workspace_set_object_id=workspace_set_object_id,
            principal=principal,
            svc=svc,
        )
        row = await svc.add_workspace_set_member(
            workspace_set_object_id,
            workspace_id=payload.workspace_id,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (BadRequestError, McpHubConflictError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _workspace_set_member_row_to_response(row)


@router.delete(
    "/workspace-set-objects/{workspace_set_object_id}/members/{workspace_id}",
    response_model=MCPHubDeleteResponse,
)
async def delete_workspace_set_member(
    workspace_set_object_id: int,
    workspace_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    _require_mutation_permission(principal)
    try:
        await _get_visible_workspace_set_object_or_404(
            workspace_set_object_id=workspace_set_object_id,
            principal=principal,
            svc=svc,
        )
        deleted = await svc.delete_workspace_set_member(
            workspace_set_object_id,
            workspace_id,
            actor_id=principal.user_id,
        )
        if not deleted:
            raise ResourceNotFoundError(
                "mcp_workspace_set_object_member",
                identifier=f"{workspace_set_object_id}:{workspace_id}",
            )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MCPHubDeleteResponse(ok=True)


@router.post("/shared-workspaces", response_model=SharedWorkspaceResponse, status_code=201)
async def create_shared_workspace(
    payload: SharedWorkspaceCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> SharedWorkspaceResponse:
    _require_mutation_permission(principal)
    if str(payload.owner_scope_type or "").strip().lower() == "user":
        raise HTTPException(status_code=400, detail="Shared workspaces must use owner_scope_type of: global, org, team")
    try:
        row = await svc.create_shared_workspace_entry(
            workspace_id=payload.workspace_id,
            display_name=payload.display_name,
            absolute_root=payload.absolute_root,
            owner_scope_type=payload.owner_scope_type,
            owner_scope_id=payload.owner_scope_id,
            actor_id=principal.user_id,
            is_active=payload.is_active,
        )
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    row = dict(row)
    row["readiness_summary"] = await svc.get_shared_workspace_readiness_summary(
        shared_workspace_id=int(row.get("id") or 0)
    )
    return _shared_workspace_row_to_response(row)


@router.get("/shared-workspaces", response_model=list[SharedWorkspaceResponse])
async def list_shared_workspaces(
    owner_scope_type: str | None = None,
    owner_scope_id: int | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[SharedWorkspaceResponse]:
    filters = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_id in filters:
        if scope_type == "user":
            continue
        rows.extend(
            await svc.list_shared_workspace_entries(
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
            )
        )
    rows = _dedupe_rows(rows)
    rows_with_readiness: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["readiness_summary"] = await svc.get_shared_workspace_readiness_summary(
            shared_workspace_id=int(row.get("id") or 0)
        )
        rows_with_readiness.append(enriched)
    return [_shared_workspace_row_to_response(row) for row in rows_with_readiness]


@router.get("/audit/findings", response_model=GovernanceAuditFindingListResponse)
async def list_governance_audit_findings(
    owner_scope_type: str | None = None,
    owner_scope_id: int | None = None,
    severity: str | None = None,
    finding_type: str | None = None,
    object_kind: str | None = None,
    scope_type: str | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> GovernanceAuditFindingListResponse:
    filters = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    all_items: list[dict[str, Any]] = []
    for visible_scope_type, visible_scope_id in filters:
        result = await svc.list_governance_audit_findings(
            actor_id=principal.user_id,
            owner_scope_type=visible_scope_type,
            owner_scope_id=visible_scope_id,
            severity=severity,
            finding_type=finding_type,
            object_kind=object_kind,
            scope_type=scope_type,
        )
        all_items.extend(list(result.get("items") or []))
    deduped_items = _dedupe_rows_by_id(
        all_items,
        id_getter=lambda row: (
            str(row.get("finding_type") or ""),
            str(row.get("object_kind") or ""),
            str(row.get("object_id") or ""),
            str(row.get("scope_type") or ""),
            str(row.get("scope_id") or ""),
            str(row.get("message") or ""),
        ),
    )
    counts = {"error": 0, "warning": 0}
    for item in deduped_items:
        severity_key = str(item.get("severity") or "warning").strip().lower()
        if severity_key in counts:
            counts[severity_key] += 1
    return GovernanceAuditFindingListResponse(
        items=deduped_items,
        total=len(deduped_items),
        counts=counts,
    )


@router.get(
    "/events/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "MCP Hub governance event stream.",
            "content": {"text/event-stream": {}},
        },
    },
)
async def stream_mcp_hub_events(
    request: Request,
    after_event_id: str | None = Query(
        default=None,
        description="Replay only events after this SSE event id when retained in the MCP Hub event ring.",
    ),
    event_type: list[str] | None = Query(
        default=None,
        description="Optional repeated or comma-separated MCP Hub event type filters.",
    ),
    owner_scope_type: str | None = Query(
        default=None,
        description="Optional scope filter constrained by the authenticated principal's visible MCP Hub scopes.",
    ),
    owner_scope_id: int | None = Query(
        default=None,
        description="Optional scope id used with owner_scope_type.",
    ),
    replay: bool = Query(
        default=True,
        description="Replay retained matching events before following the live stream.",
    ),
    limit: int | None = Query(
        default=None,
        ge=1,
        le=1000,
        description="Optional maximum number of matching events to emit before closing.",
    ),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> StreamingResponse:
    """Stream MCP Hub governance mutation events over SSE.

    Events are emitted from the same mutation path that writes durable audit
    records. Replay reads unified audit first and then falls back to the bounded
    in-process ring for process-local events that were not audit-backed.
    """
    bus = await get_mcp_hub_event_bus()
    event_types = _normalize_event_type_filter(event_type)
    visible_scopes = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    allow_cross_tenant_replay = _is_admin_principal(principal)
    replay_user_id = (
        None
        if allow_cross_tenant_replay or principal.user_id is None
        else str(principal.user_id)
    )

    async def event_generator():
        sent = 0
        queue = None
        sent_event_ids: set[str] = set()
        try:
            queue = await bus.subscribe()
            if replay:
                durable_replayed = await replay_mcp_hub_audit_events(
                    principal_user_id=principal.user_id,
                    user_id=replay_user_id,
                    after_event_id=after_event_id,
                    event_types=event_types,
                    limit=None,
                    allow_cross_tenant=allow_cross_tenant_replay,
                )
                for event in durable_replayed:
                    if not _event_matches_visible_scope(event, visible_scopes):
                        continue
                    event_id = str(event.get("event_id") or "")
                    if event_id:
                        sent_event_ids.add(event_id)
                    yield _format_mcp_hub_sse(event)
                    sent += 1
                    if limit is not None and sent >= limit:
                        return

                replayed = await bus.replay(
                    after_event_id=after_event_id,
                    event_types=event_types,
                )
                for event in replayed:
                    event_id = str(event.get("event_id") or "")
                    if event_id and event_id in sent_event_ids:
                        continue
                    if not _event_matches_visible_scope(event, visible_scopes):
                        continue
                    if event_id:
                        sent_event_ids.add(event_id)
                    yield _format_mcp_hub_sse(event)
                    sent += 1
                    if limit is not None and sent >= limit:
                        return

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if event is None:
                    break
                if event_types and str(event.get("event_type") or "") not in event_types:
                    continue
                event_id = str(event.get("event_id") or "")
                if event_id and event_id in sent_event_ids:
                    continue
                if not _event_matches_visible_scope(event, visible_scopes):
                    continue
                if event_id:
                    sent_event_ids.add(event_id)
                yield _format_mcp_hub_sse(event)
                sent += 1
                if limit is not None and sent >= limit:
                    return
        except (asyncio.CancelledError, GeneratorExit):
            return
        finally:
            if queue is not None:
                await bus.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.put("/shared-workspaces/{shared_workspace_id}", response_model=SharedWorkspaceResponse)
async def update_shared_workspace(
    shared_workspace_id: int,
    payload: SharedWorkspaceUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> SharedWorkspaceResponse:
    _require_mutation_permission(principal)
    if "owner_scope_type" in payload.model_fields_set and str(payload.owner_scope_type or "").strip().lower() == "user":
        raise HTTPException(status_code=400, detail="Shared workspaces must use owner_scope_type of: global, org, team")
    try:
        await _get_visible_shared_workspace_or_404(
            shared_workspace_id=shared_workspace_id,
            principal=principal,
            svc=svc,
        )
        row = await svc.update_shared_workspace_entry(
            shared_workspace_id,
            actor_id=principal.user_id,
            **payload.model_dump(exclude_unset=True),
        )
        if not row:
            raise ResourceNotFoundError("mcp_shared_workspace", identifier=str(shared_workspace_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _shared_workspace_row_to_response(row)


@router.delete("/shared-workspaces/{shared_workspace_id}", response_model=MCPHubDeleteResponse)
async def delete_shared_workspace(
    shared_workspace_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    _require_mutation_permission(principal)
    try:
        await _get_visible_shared_workspace_or_404(
            shared_workspace_id=shared_workspace_id,
            principal=principal,
            svc=svc,
        )
        deleted = await svc.delete_shared_workspace_entry(shared_workspace_id, actor_id=principal.user_id)
        if not deleted:
            raise ResourceNotFoundError("mcp_shared_workspace", identifier=str(shared_workspace_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MCPHubDeleteResponse(ok=True)


@router.post("/policy-assignments", response_model=PolicyAssignmentResponse, status_code=201)
async def create_policy_assignment(
    payload: PolicyAssignmentCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> PolicyAssignmentResponse:
    """Create a new policy assignment for a default, group, or persona target."""
    _require_mutation_permission(principal)
    inline_policy_document = _normalize_policy_document_for_storage(
        payload.inline_policy_document,
        allow_inherited_scope=False,
    )
    _require_grant_authority(principal, inline_policy_document)
    try:
        await svc.validate_path_scope_object_reference(
            path_scope_object_id=payload.path_scope_object_id,
            target_scope_type=payload.owner_scope_type,
            target_scope_id=payload.owner_scope_id,
        )
        await _validate_assignment_workspace_source(
            svc=svc,
            workspace_source_mode=payload.workspace_source_mode,
            workspace_set_object_id=payload.workspace_set_object_id,
            owner_scope_type=payload.owner_scope_type,
            owner_scope_id=payload.owner_scope_id,
        )
        row = await svc.create_policy_assignment(
            target_type=payload.target_type,
            target_id=payload.target_id,
            owner_scope_type=payload.owner_scope_type,
            owner_scope_id=payload.owner_scope_id,
            profile_id=payload.profile_id,
            path_scope_object_id=payload.path_scope_object_id,
            workspace_source_mode=payload.workspace_source_mode,
            workspace_set_object_id=payload.workspace_set_object_id,
            inline_policy_document=inline_policy_document,
            approval_policy_id=payload.approval_policy_id,
            is_active=payload.is_active,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=_bad_request_detail(exc)) from exc
    return _policy_assignment_row_to_response(row)


@router.get("/policy-assignments", response_model=list[PolicyAssignmentResponse])
async def list_policy_assignments(
    owner_scope_type: str | None = None,
    owner_scope_id: int | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[PolicyAssignmentResponse]:
    """List policy assignments visible to the current principal with scope-constrained filtering."""
    filters = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_id in filters:
        rows.extend(
            await svc.list_policy_assignments(
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
                target_type=target_type,
                target_id=target_id,
            )
        )
    rows = _dedupe_rows(rows)
    return [_policy_assignment_row_to_response(row) for row in rows]


@router.put("/policy-assignments/{assignment_id}", response_model=PolicyAssignmentResponse)
async def update_policy_assignment(
    assignment_id: int,
    payload: PolicyAssignmentUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> PolicyAssignmentResponse:
    """Update an existing policy assignment by id."""
    _require_mutation_permission(principal)
    update_fields = payload.model_dump(exclude_unset=True)
    if "inline_policy_document" in update_fields:
        update_fields["inline_policy_document"] = _normalize_policy_document_for_storage(
            update_fields.get("inline_policy_document") or {},
            allow_inherited_scope=False,
        )
        _require_grant_authority(principal, update_fields.get("inline_policy_document") or {})
    try:
        existing = await svc.get_policy_assignment(assignment_id)
        if not existing:
            raise ResourceNotFoundError("mcp_policy_assignment", identifier=str(assignment_id))
        _ensure_mutable_governance_object(existing, "Policy assignment")
        if (
            "path_scope_object_id" in update_fields
            or "owner_scope_type" in update_fields
            or "owner_scope_id" in update_fields
        ):
            await svc.validate_path_scope_object_reference(
                path_scope_object_id=update_fields.get(
                    "path_scope_object_id",
                    existing.get("path_scope_object_id"),
                ),
                target_scope_type=str(update_fields.get("owner_scope_type") or existing.get("owner_scope_type") or "global"),
                target_scope_id=update_fields.get("owner_scope_id", existing.get("owner_scope_id")),
            )
            await _validate_assignment_workspace_source(
                svc=svc,
                workspace_source_mode=update_fields.get(
                    "workspace_source_mode",
                    existing.get("workspace_source_mode"),
                ),
                workspace_set_object_id=update_fields.get(
                    "workspace_set_object_id",
                    existing.get("workspace_set_object_id"),
                ),
                owner_scope_type=str(update_fields.get("owner_scope_type") or existing.get("owner_scope_type") or "global"),
                owner_scope_id=update_fields.get("owner_scope_id", existing.get("owner_scope_id")),
            )
        row = await svc.update_policy_assignment(
            assignment_id,
            actor_id=principal.user_id,
            **update_fields,
        )
        if not row:
            raise ResourceNotFoundError("mcp_policy_assignment", identifier=str(assignment_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=_bad_request_detail(exc)) from exc
    return _policy_assignment_row_to_response(row)


@router.delete("/policy-assignments/{assignment_id}", response_model=MCPHubDeleteResponse)
async def delete_policy_assignment(
    assignment_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Delete a policy assignment by id."""
    _require_mutation_permission(principal)
    try:
        existing = await svc.get_policy_assignment(assignment_id)
        if not existing:
            raise ResourceNotFoundError("mcp_policy_assignment", identifier=str(assignment_id))
        _ensure_mutable_governance_object(existing, "Policy assignment")
        deleted = await svc.delete_policy_assignment(assignment_id, actor_id=principal.user_id)
        if not deleted:
            raise ResourceNotFoundError("mcp_policy_assignment", identifier=str(assignment_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MCPHubDeleteResponse(ok=True)


@router.get(
    "/policy-assignments/{assignment_id}/workspaces",
    response_model=list[PolicyAssignmentWorkspaceResponse],
)
async def list_policy_assignment_workspaces(
    assignment_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[PolicyAssignmentWorkspaceResponse]:
    await _get_visible_policy_assignment_or_404(assignment_id=assignment_id, principal=principal, svc=svc)
    rows = await svc.list_policy_assignment_workspaces(assignment_id)
    return [_policy_assignment_workspace_row_to_response(row) for row in rows]


@router.post(
    "/policy-assignments/{assignment_id}/workspaces",
    response_model=PolicyAssignmentWorkspaceResponse,
    status_code=201,
)
async def add_policy_assignment_workspace(
    assignment_id: int,
    payload: PolicyAssignmentWorkspaceCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> PolicyAssignmentWorkspaceResponse:
    _require_mutation_permission(principal)
    try:
        await _get_visible_policy_assignment_or_404(assignment_id=assignment_id, principal=principal, svc=svc)
        row = await svc.add_policy_assignment_workspace(
            assignment_id,
            workspace_id=payload.workspace_id,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except McpHubConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=_bad_request_detail(exc)) from exc
    return _policy_assignment_workspace_row_to_response(row)


@router.delete(
    "/policy-assignments/{assignment_id}/workspaces/{workspace_id}",
    response_model=MCPHubDeleteResponse,
)
async def delete_policy_assignment_workspace(
    assignment_id: int,
    workspace_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    _require_mutation_permission(principal)
    try:
        await _get_visible_policy_assignment_or_404(assignment_id=assignment_id, principal=principal, svc=svc)
        deleted = await svc.delete_policy_assignment_workspace(
            assignment_id,
            workspace_id=workspace_id,
            actor_id=principal.user_id,
        )
        if not deleted:
            raise ResourceNotFoundError(
                "mcp_policy_assignment_workspace",
                identifier=f"{assignment_id}:{workspace_id}",
            )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MCPHubDeleteResponse(ok=True)


@router.get(
    "/policy-assignments/{assignment_id}/override",
    response_model=PolicyOverrideResponse,
)
async def get_policy_assignment_override(
    assignment_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> PolicyOverrideResponse:
    """Fetch the single override attached to a policy assignment."""
    try:
        await _get_visible_policy_assignment_or_404(
            assignment_id=assignment_id,
            principal=principal,
            svc=svc,
        )
        row = await svc.get_policy_override(assignment_id)
        if not row:
            raise ResourceNotFoundError("mcp_policy_override", identifier=str(assignment_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _policy_override_row_to_response(row)


@router.put(
    "/policy-assignments/{assignment_id}/override",
    response_model=PolicyOverrideResponse,
)
async def upsert_policy_assignment_override(
    assignment_id: int,
    payload: PolicyOverrideUpsertRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> PolicyOverrideResponse:
    """Create or replace the single override attached to a policy assignment."""
    _require_mutation_permission(principal)
    try:
        override_policy_document = _normalize_policy_document_for_storage(
            payload.override_policy_document,
            allow_inherited_scope=True,
        )
        assignment = await _get_visible_policy_assignment_or_404(
            assignment_id=assignment_id,
            principal=principal,
            svc=svc,
        )
        profile_id, base_document = await _assignment_base_policy_document(
            svc=svc,
            assignment=assignment,
        )
        merged_document = _merge_policy_documents(base_document, override_policy_document)
        delta_document = _grant_authority_delta(base_document, merged_document)
        if delta_document:
            _require_grant_authority(principal, delta_document)
        row = await svc.upsert_policy_override(
            assignment_id,
            override_policy_document=override_policy_document,
            is_active=payload.is_active,
            broadens_access=bool(delta_document),
            grant_authority_snapshot={
                **_grant_authority_snapshot(principal, delta_document),
                "assignment_id": assignment_id,
                "profile_id": profile_id,
            },
            actor_id=principal.user_id,
        )
        if not row:
            raise ResourceNotFoundError("mcp_policy_assignment", identifier=str(assignment_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _policy_override_row_to_response(row)


@router.delete(
    "/policy-assignments/{assignment_id}/override",
    response_model=MCPHubDeleteResponse,
)
async def delete_policy_assignment_override(
    assignment_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Delete the override attached to a policy assignment."""
    _require_mutation_permission(principal)
    try:
        await _get_visible_policy_assignment_or_404(
            assignment_id=assignment_id,
            principal=principal,
            svc=svc,
        )
        deleted = await svc.delete_policy_override(assignment_id, actor_id=principal.user_id)
        if not deleted:
            raise ResourceNotFoundError("mcp_policy_override", identifier=str(assignment_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MCPHubDeleteResponse(ok=True)


@router.post("/approval-policies", response_model=ApprovalPolicyResponse, status_code=201)
async def create_approval_policy(
    payload: ApprovalPolicyCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ApprovalPolicyResponse:
    """Create a new runtime approval policy within the provided owner scope."""
    _require_mutation_permission(principal)
    normalized_rules = _normalized_approval_rules(payload.rules)
    row = await svc.create_approval_policy(
        name=payload.name,
        description=payload.description,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
        mode=payload.mode,
        rules=normalized_rules,
        is_active=payload.is_active,
        actor_id=principal.user_id,
    )
    return _approval_policy_row_to_response(row)


@router.get("/approval-policies", response_model=list[ApprovalPolicyResponse])
async def list_approval_policies(
    owner_scope_type: str | None = None,
    owner_scope_id: int | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[ApprovalPolicyResponse]:
    """List approval policies visible to the current principal with scope-constrained filtering."""
    filters = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_id in filters:
        rows.extend(
            await svc.list_approval_policies(
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
            )
        )
    rows = _dedupe_rows(rows)
    return [_approval_policy_row_to_response(row) for row in rows]


@router.put("/approval-policies/{approval_policy_id}", response_model=ApprovalPolicyResponse)
async def update_approval_policy(
    approval_policy_id: int,
    payload: ApprovalPolicyUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ApprovalPolicyResponse:
    """Update an approval policy by id."""
    _require_mutation_permission(principal)
    update_fields = payload.model_dump(exclude_unset=True)
    if "rules" in update_fields:
        update_fields["rules"] = _normalized_approval_rules(update_fields.get("rules"))
    try:
        existing = await svc.get_approval_policy(approval_policy_id)
        if not existing:
            raise ResourceNotFoundError("mcp_approval_policy", identifier=str(approval_policy_id))
        _ensure_mutable_governance_object(existing, "Approval policy")
        row = await svc.update_approval_policy(
            approval_policy_id,
            actor_id=principal.user_id,
            **update_fields,
        )
        if not row:
            raise ResourceNotFoundError("mcp_approval_policy", identifier=str(approval_policy_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _approval_policy_row_to_response(row)


@router.delete("/approval-policies/{approval_policy_id}", response_model=MCPHubDeleteResponse)
async def delete_approval_policy(
    approval_policy_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Delete an approval policy by id."""
    _require_mutation_permission(principal)
    try:
        existing = await svc.get_approval_policy(approval_policy_id)
        if not existing:
            raise ResourceNotFoundError("mcp_approval_policy", identifier=str(approval_policy_id))
        _ensure_mutable_governance_object(existing, "Approval policy")
        deleted = await svc.delete_approval_policy(approval_policy_id, actor_id=principal.user_id)
        if not deleted:
            raise ResourceNotFoundError("mcp_approval_policy", identifier=str(approval_policy_id))
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return MCPHubDeleteResponse(ok=True)


@router.post("/approval-decisions", response_model=ApprovalDecisionResponse, status_code=201)
async def create_approval_decision(
    payload: ApprovalDecisionCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ApprovalDecisionResponse:
    """Persist a user approval or denial decision for a pending MCP Hub runtime approval."""
    actor_id = int(principal.user_id) if principal.user_id is not None else None
    context_user_id = _extract_context_key_user_id(payload.context_key)
    if actor_id is None:
        raise HTTPException(status_code=403, detail="Authenticated user required")
    if context_user_id is None:
        raise HTTPException(status_code=422, detail="context_key must include a numeric user id")
    if context_user_id != actor_id and not _is_mutation_allowed(principal):
        raise HTTPException(status_code=403, detail="Forbidden approval context")

    approval_rules: dict[str, Any] = {"duration_options": ["once", "session"]}
    if payload.approval_policy_id is not None:
        policy_row = await svc.get_approval_policy(int(payload.approval_policy_id))
        if policy_row is None:
            raise HTTPException(status_code=404, detail="Approval policy not found")
        approval_rules = _normalized_approval_rules(policy_row.get("rules") if isinstance(policy_row, dict) else {})

    if payload.duration not in _validated_duration_options(approval_rules):
        raise HTTPException(
            status_code=422,
            detail=f"duration '{payload.duration}' is not allowed by the approval policy",
        )

    consume_on_match, expires_at = _approval_decision_lifetime(
        decision=payload.decision,
        duration=payload.duration,
        conversation_id=payload.conversation_id,
        rules=approval_rules,
    )

    row = await svc.record_approval_decision(
        approval_policy_id=payload.approval_policy_id,
        context_key=payload.context_key,
        conversation_id=payload.conversation_id,
        tool_name=payload.tool_name,
        scope_key=payload.scope_key,
        decision=payload.decision,
        consume_on_match=consume_on_match,
        expires_at=expires_at,
        actor_id=actor_id,
    )
    return _approval_decision_row_to_response(row)


@router.get("/effective-policy", response_model=EffectivePolicyResponse)
async def get_effective_policy(
    persona_id: str | None = None,
    group_id: str | None = None,
    org_id: int | None = None,
    team_id: int | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    resolver: McpHubPolicyResolver = Depends(get_mcp_hub_policy_resolver_dep),
) -> EffectivePolicyResponse:
    """Resolve the effective MCP Hub policy for the authenticated user and optional target context."""
    if principal.user_id is None:
        raise HTTPException(status_code=403, detail="Authenticated user required")

    metadata: dict[str, Any] = {"mcp_policy_context_enabled": True}
    if persona_id:
        metadata["persona_id"] = persona_id
    if group_id:
        metadata["group_id"] = group_id
    if org_id is not None:
        metadata["org_id"] = org_id
    if team_id is not None:
        metadata["team_id"] = team_id

    return await resolver.resolve_for_context(
        user_id=principal.user_id,
        metadata=metadata,
    )


@router.get("/acp-profiles", response_model=list[ACPProfileResponse])
async def list_acp_profiles(
    owner_scope_type: str | None = None,
    owner_scope_id: int | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[ACPProfileResponse]:
    """List ACP profiles visible to the current principal with scope-constrained filtering."""
    filters = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_id in filters:
        rows.extend(
            await svc.list_acp_profiles(
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
            )
        )
    rows = _dedupe_rows(rows)
    return [_profile_row_to_response(row) for row in rows]


@router.post("/acp-profiles", response_model=ACPProfileResponse, status_code=201)
async def create_acp_profile(
    payload: ACPProfileCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ACPProfileResponse:
    """Create a new ACP profile within the provided owner scope."""
    _require_mutation_permission(principal)
    row = await svc.create_acp_profile(
        name=payload.name,
        description=payload.description,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
        profile=payload.profile,
        is_active=payload.is_active,
        actor_id=principal.user_id,
    )
    return _profile_row_to_response(row)


@router.put("/acp-profiles/{profile_id}", response_model=ACPProfileResponse)
async def update_acp_profile(
    profile_id: int,
    payload: ACPProfileUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ACPProfileResponse:
    """Update an existing ACP profile by id."""
    _require_mutation_permission(principal)
    row = await svc.update_acp_profile(
        profile_id,
        name=payload.name,
        description=payload.description,
        owner_scope_type=payload.owner_scope_type,
        owner_scope_id=payload.owner_scope_id,
        profile=payload.profile,
        is_active=payload.is_active,
        actor_id=principal.user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="ACP profile not found")
    return _profile_row_to_response(row)


@router.delete("/acp-profiles/{profile_id}", response_model=MCPHubDeleteResponse)
async def delete_acp_profile(
    profile_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Delete an ACP profile by id."""
    _require_mutation_permission(principal)
    deleted = await svc.delete_acp_profile(profile_id, actor_id=principal.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="ACP profile not found")
    return MCPHubDeleteResponse(ok=True)


@router.get("/external-servers", response_model=list[ExternalServerResponse])
async def list_external_servers(
    owner_scope_type: str | None = None,
    owner_scope_id: int | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[ExternalServerResponse]:
    """List external MCP servers visible to the current principal with scope-constrained filtering."""
    filters = _resolve_visible_scope_filters(
        principal=principal,
        owner_scope_type=owner_scope_type,
        owner_scope_id=owner_scope_id,
    )
    rows: list[dict[str, Any]] = []
    for scope_type, scope_id in filters:
        rows.extend(
            await svc.list_external_servers(
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
            )
        )
    rows = _dedupe_rows(rows)
    return [_external_row_to_response(row) for row in rows]


@router.post("/external-servers", response_model=ExternalServerResponse, status_code=201)
async def create_external_server(
    payload: ExternalServerCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ExternalServerResponse:
    """Create a new external MCP server definition."""
    _require_mutation_permission(principal)
    try:
        row = await svc.create_external_server(
            server_id=payload.server_id,
            name=payload.name,
            transport=payload.transport,
            config=payload.config,
            owner_scope_type=payload.owner_scope_type,
            owner_scope_id=payload.owner_scope_id,
            enabled=payload.enabled,
            actor_id=principal.user_id,
            allow_existing=False,
        )
    except McpHubConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _external_row_to_response(row)


@router.post(
    "/external-servers/refresh-discovery",
    response_model=ExternalServerDiscoveryRefreshResponse,
)
async def refresh_external_server_discovery(
    payload: ExternalServerDiscoveryRefreshRequest | None = Body(default=None),
    server_id: str | None = Query(default=None),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> ExternalServerDiscoveryRefreshResponse:
    """Reconcile live MCP external federation discovery with managed storage."""
    _require_mutation_permission(principal)
    body_server_id = _normalize_refresh_server_id(payload.server_id if payload is not None else None)
    query_server_id = _normalize_refresh_server_id(server_id)
    if body_server_id is not None and query_server_id is not None and body_server_id != query_server_id:
        raise HTTPException(
            status_code=422,
            detail="server_id query parameter conflicts with request body server_id",
        )
    effective_server_id = query_server_id if query_server_id is not None else body_server_id
    return await _refresh_external_server_discovery_runtime(server_id=effective_server_id)


@router.post("/external-servers/{server_id}/import", response_model=ExternalServerResponse, status_code=201)
async def import_external_server(
    server_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ExternalServerResponse:
    """Import a legacy external server definition into MCP Hub managed storage."""
    _require_mutation_permission(principal)
    try:
        row = await svc.import_legacy_external_server(
            server_id=server_id,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except McpHubConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _external_row_to_response(row)


@router.get(
    "/external-servers/{server_id}/auth-template",
    response_model=ExternalServerAuthTemplateResponse,
)
async def get_external_server_auth_template(
    server_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ExternalServerAuthTemplateResponse:
    """Fetch the managed auth template for an external server."""
    _ = principal
    try:
        row = await svc.get_external_server_auth_template(server_id=server_id)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExternalServerAuthTemplateResponse.model_validate(row)


@router.put(
    "/external-servers/{server_id}/auth-template",
    response_model=ExternalServerAuthTemplateResponse,
)
async def update_external_server_auth_template(
    server_id: str,
    payload: ExternalServerAuthTemplateUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ExternalServerAuthTemplateResponse:
    """Create or update the managed auth template for an external server."""
    _require_mutation_permission(principal)
    try:
        row = await svc.update_external_server_auth_template(
            server_id=server_id,
            auth_template=payload.model_dump(),
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExternalServerAuthTemplateResponse.model_validate(row)


@router.get(
    "/external-servers/{server_id}/credential-slots",
    response_model=list[ExternalServerCredentialSlotResponse],
)
async def list_external_server_credential_slots(
    server_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[ExternalServerCredentialSlotResponse]:
    """List credential slots configured on a managed external server."""
    _ = principal
    try:
        rows = await svc.list_external_server_credential_slots(server_id=server_id)
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [ExternalServerCredentialSlotResponse.model_validate(row) for row in rows]


@router.post(
    "/external-servers/{server_id}/credential-slots",
    response_model=ExternalServerCredentialSlotResponse,
    status_code=201,
)
async def create_external_server_credential_slot(
    server_id: str,
    payload: ExternalServerCredentialSlotCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ExternalServerCredentialSlotResponse:
    """Create a credential slot on a managed external server."""
    _require_mutation_permission(principal)
    _require_credential_grant_authority(principal, payload.privilege_class)
    try:
        row = await svc.create_external_server_credential_slot(
            server_id=server_id,
            slot_name=payload.slot_name,
            display_name=payload.display_name,
            secret_kind=payload.secret_kind,
            privilege_class=payload.privilege_class,
            is_required=payload.is_required,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExternalServerCredentialSlotResponse.model_validate(row)


@router.put(
    "/external-servers/{server_id}/credential-slots/{slot_name}",
    response_model=ExternalServerCredentialSlotResponse,
)
async def update_external_server_credential_slot(
    server_id: str,
    slot_name: str,
    payload: ExternalServerCredentialSlotUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ExternalServerCredentialSlotResponse:
    """Update credential slot metadata on a managed external server."""
    _require_mutation_permission(principal)
    if payload.privilege_class is not None:
        slot_row = await _resolve_binding_slot_for_grant_authority(
            svc=svc,
            server_id=server_id,
            slot_name=slot_name,
        )
        if slot_row is not None and _credential_privilege_broadens(
            slot_row.get("privilege_class"),
            payload.privilege_class,
        ):
            _require_credential_grant_authority(principal, payload.privilege_class)
    try:
        row = await svc.update_external_server_credential_slot(
            server_id=server_id,
            slot_name=slot_name,
            display_name=payload.display_name,
            secret_kind=payload.secret_kind,
            privilege_class=payload.privilege_class,
            is_required=payload.is_required,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExternalServerCredentialSlotResponse.model_validate(row)


@router.delete(
    "/external-servers/{server_id}/credential-slots/{slot_name}",
    response_model=MCPHubDeleteResponse,
)
async def delete_external_server_credential_slot(
    server_id: str,
    slot_name: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Delete a credential slot from a managed external server."""
    _require_mutation_permission(principal)
    deleted = await svc.delete_external_server_credential_slot(
        server_id=server_id,
        slot_name=slot_name,
        actor_id=principal.user_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Credential slot not found")
    return MCPHubDeleteResponse(ok=True)


@router.post(
    "/external-servers/{server_id}/credential-slots/{slot_name}/secret",
    response_model=ExternalServerSlotSecretSetResponse,
)
async def set_external_server_slot_secret(
    server_id: str,
    slot_name: str,
    payload: ExternalSecretSetRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ExternalServerSlotSecretSetResponse:
    """Set or rotate the secret value for an external server credential slot."""
    _require_mutation_permission(principal)
    try:
        out = await svc.set_external_server_slot_secret(
            server_id=server_id,
            slot_name=slot_name,
            secret_value=payload.secret,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExternalServerSlotSecretSetResponse(**out)


@router.delete(
    "/external-servers/{server_id}/credential-slots/{slot_name}/secret",
    response_model=MCPHubDeleteResponse,
)
async def clear_external_server_slot_secret(
    server_id: str,
    slot_name: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Clear the stored secret for an external server credential slot."""
    _require_mutation_permission(principal)
    deleted = await svc.clear_external_server_slot_secret(
        server_id=server_id,
        slot_name=slot_name,
        actor_id=principal.user_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Credential slot secret not found")
    return MCPHubDeleteResponse(ok=True)


@router.get("/permission-profiles/{profile_id}/credential-bindings", response_model=list[CredentialBindingResponse])
async def list_profile_credential_bindings(
    profile_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[CredentialBindingResponse]:
    """List external server bindings attached to a permission profile."""
    await _get_visible_permission_profile_or_404(profile_id=profile_id, principal=principal, svc=svc)
    rows = await svc.list_profile_credential_bindings(profile_id=profile_id)
    return [_binding_row_to_response(row) for row in rows]


@router.put(
    "/permission-profiles/{profile_id}/credential-bindings/{server_id}",
    response_model=CredentialBindingResponse,
)
async def upsert_profile_credential_binding(
    profile_id: int,
    server_id: str,
    payload: ProfileCredentialBindingUpsertRequest | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> CredentialBindingResponse:
    """Grant a managed external server to a permission profile."""
    _require_mutation_permission(principal)
    await _get_visible_permission_profile_or_404(profile_id=profile_id, principal=principal, svc=svc)
    slot_row = await _resolve_binding_slot_for_grant_authority(
        svc=svc,
        server_id=server_id,
        slot_name=None,
    )
    if slot_row is not None:
        _require_credential_grant_authority(principal, slot_row.get("privilege_class"))
    try:
        row = await svc.upsert_profile_credential_binding(
            profile_id=profile_id,
            external_server_id=server_id,
            managed_secret_ref_id=(payload.managed_secret_ref_id if payload else None),
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _binding_row_to_response(row)


@router.put(
    "/permission-profiles/{profile_id}/credential-bindings/{server_id}/{slot_name}",
    response_model=CredentialBindingResponse,
)
async def upsert_profile_slot_credential_binding(
    profile_id: int,
    server_id: str,
    slot_name: str,
    payload: ProfileCredentialBindingUpsertRequest | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> CredentialBindingResponse:
    """Grant a managed external server slot to a permission profile."""
    _require_mutation_permission(principal)
    await _get_visible_permission_profile_or_404(profile_id=profile_id, principal=principal, svc=svc)
    slot_row = await _resolve_binding_slot_for_grant_authority(
        svc=svc,
        server_id=server_id,
        slot_name=slot_name,
    )
    if slot_row is not None:
        _require_credential_grant_authority(principal, slot_row.get("privilege_class"))
    try:
        row = await svc.upsert_profile_credential_binding(
            profile_id=profile_id,
            external_server_id=server_id,
            slot_name=slot_name,
            managed_secret_ref_id=(payload.managed_secret_ref_id if payload else None),
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _binding_row_to_response(row)


@router.delete(
    "/permission-profiles/{profile_id}/credential-bindings/{server_id}",
    response_model=MCPHubDeleteResponse,
)
async def delete_profile_credential_binding(
    profile_id: int,
    server_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Delete an external server binding from a permission profile."""
    _require_mutation_permission(principal)
    await _get_visible_permission_profile_or_404(profile_id=profile_id, principal=principal, svc=svc)
    deleted = await svc.delete_profile_credential_binding(
        profile_id=profile_id,
        external_server_id=server_id,
        actor_id=principal.user_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Credential binding not found")
    return MCPHubDeleteResponse(ok=True)


@router.delete(
    "/permission-profiles/{profile_id}/credential-bindings/{server_id}/{slot_name}",
    response_model=MCPHubDeleteResponse,
)
async def delete_profile_slot_credential_binding(
    profile_id: int,
    server_id: str,
    slot_name: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Delete an external server slot binding from a permission profile."""
    _require_mutation_permission(principal)
    await _get_visible_permission_profile_or_404(profile_id=profile_id, principal=principal, svc=svc)
    deleted = await svc.delete_profile_credential_binding(
        profile_id=profile_id,
        external_server_id=server_id,
        slot_name=slot_name,
        actor_id=principal.user_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Credential binding not found")
    return MCPHubDeleteResponse(ok=True)


@router.get("/policy-assignments/{assignment_id}/credential-bindings", response_model=list[CredentialBindingResponse])
async def list_assignment_credential_bindings(
    assignment_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> list[CredentialBindingResponse]:
    """List external server bindings attached to a policy assignment."""
    await _get_visible_policy_assignment_or_404(assignment_id=assignment_id, principal=principal, svc=svc)
    rows = await svc.list_assignment_credential_bindings(assignment_id=assignment_id)
    return [_binding_row_to_response(row) for row in rows]


@router.put(
    "/policy-assignments/{assignment_id}/credential-bindings/{server_id}",
    response_model=CredentialBindingResponse,
)
async def upsert_assignment_credential_binding(
    assignment_id: int,
    server_id: str,
    payload: AssignmentCredentialBindingUpsertRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> CredentialBindingResponse:
    """Create or update an external server binding on a policy assignment."""
    _require_mutation_permission(principal)
    await _get_visible_policy_assignment_or_404(assignment_id=assignment_id, principal=principal, svc=svc)
    if payload.binding_mode == "grant":
        slot_row = await _resolve_binding_slot_for_grant_authority(
            svc=svc,
            server_id=server_id,
            slot_name=None,
        )
        if slot_row is not None:
            _require_credential_grant_authority(principal, slot_row.get("privilege_class"))
    try:
        row = await svc.upsert_assignment_credential_binding(
            assignment_id=assignment_id,
            external_server_id=server_id,
            binding_mode=payload.binding_mode,
            managed_secret_ref_id=payload.managed_secret_ref_id,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _binding_row_to_response(row)


@router.put(
    "/policy-assignments/{assignment_id}/credential-bindings/{server_id}/{slot_name}",
    response_model=CredentialBindingResponse,
)
async def upsert_assignment_slot_credential_binding(
    assignment_id: int,
    server_id: str,
    slot_name: str,
    payload: AssignmentCredentialBindingUpsertRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> CredentialBindingResponse:
    """Create or update an external server slot binding on a policy assignment."""
    _require_mutation_permission(principal)
    await _get_visible_policy_assignment_or_404(assignment_id=assignment_id, principal=principal, svc=svc)
    if payload.binding_mode == "grant":
        slot_row = await _resolve_binding_slot_for_grant_authority(
            svc=svc,
            server_id=server_id,
            slot_name=slot_name,
        )
        if slot_row is not None:
            _require_credential_grant_authority(principal, slot_row.get("privilege_class"))
    try:
        row = await svc.upsert_assignment_credential_binding(
            assignment_id=assignment_id,
            external_server_id=server_id,
            slot_name=slot_name,
            binding_mode=payload.binding_mode,
            managed_secret_ref_id=payload.managed_secret_ref_id,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _binding_row_to_response(row)


@router.delete(
    "/policy-assignments/{assignment_id}/credential-bindings/{server_id}",
    response_model=MCPHubDeleteResponse,
)
async def delete_assignment_credential_binding(
    assignment_id: int,
    server_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Delete an external server binding from a policy assignment."""
    _require_mutation_permission(principal)
    await _get_visible_policy_assignment_or_404(assignment_id=assignment_id, principal=principal, svc=svc)
    deleted = await svc.delete_assignment_credential_binding(
        assignment_id=assignment_id,
        external_server_id=server_id,
        actor_id=principal.user_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Credential binding not found")
    return MCPHubDeleteResponse(ok=True)


@router.delete(
    "/policy-assignments/{assignment_id}/credential-bindings/{server_id}/{slot_name}",
    response_model=MCPHubDeleteResponse,
)
async def delete_assignment_slot_credential_binding(
    assignment_id: int,
    server_id: str,
    slot_name: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Delete an external server slot binding from a policy assignment."""
    _require_mutation_permission(principal)
    await _get_visible_policy_assignment_or_404(assignment_id=assignment_id, principal=principal, svc=svc)
    deleted = await svc.delete_assignment_credential_binding(
        assignment_id=assignment_id,
        external_server_id=server_id,
        slot_name=slot_name,
        actor_id=principal.user_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Credential binding not found")
    return MCPHubDeleteResponse(ok=True)


@router.get(
    "/permission-profiles/{profile_id}/credential-bindings/status/{server_id}/{slot_name}",
    response_model=McpCredentialSlotStatusResponse,
)
@router.get(
    "/permission-profiles/{profile_id}/credential-bindings/{server_id}/{slot_name}/status",
    response_model=McpCredentialSlotStatusResponse,
    include_in_schema=False,
    deprecated=True,
)
async def get_profile_slot_credential_status(
    profile_id: int,
    server_id: str,
    slot_name: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
    broker: McpCredentialBrokerService = Depends(get_mcp_credential_broker_service),
) -> McpCredentialSlotStatusResponse:
    """Return the remediation status for a profile-scoped external credential slot."""
    await _get_visible_permission_profile_or_404(profile_id=profile_id, principal=principal, svc=svc)
    try:
        status = await broker.get_slot_status(
            server_id=server_id,
            slot_name=slot_name,
            profile_id=profile_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return McpCredentialSlotStatusResponse.model_validate(status)


@router.get(
    "/policy-assignments/{assignment_id}/credential-bindings/status/{server_id}/{slot_name}",
    response_model=McpCredentialSlotStatusResponse,
)
@router.get(
    "/policy-assignments/{assignment_id}/credential-bindings/{server_id}/{slot_name}/status",
    response_model=McpCredentialSlotStatusResponse,
    include_in_schema=False,
    deprecated=True,
)
async def get_assignment_slot_credential_status(
    assignment_id: int,
    server_id: str,
    slot_name: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
    broker: McpCredentialBrokerService = Depends(get_mcp_credential_broker_service),
) -> McpCredentialSlotStatusResponse:
    """Return the remediation status for an assignment-scoped external credential slot."""
    await _get_visible_policy_assignment_or_404(assignment_id=assignment_id, principal=principal, svc=svc)
    try:
        status = await broker.get_slot_status(
            server_id=server_id,
            slot_name=slot_name,
            assignment_id=assignment_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return McpCredentialSlotStatusResponse.model_validate(status)


@router.get("/policy-assignments/{assignment_id}/external-access", response_model=EffectiveExternalAccessResponse)
async def get_assignment_external_access(
    assignment_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> EffectiveExternalAccessResponse:
    """Preview effective external server access for a policy assignment."""
    await _get_visible_policy_assignment_or_404(assignment_id=assignment_id, principal=principal, svc=svc)
    summary = await svc.resolve_effective_external_access(
        assignment_id=assignment_id,
        actor_id=principal.user_id,
    )
    return EffectiveExternalAccessResponse.model_validate(summary)


@router.put("/external-servers/{server_id}", response_model=ExternalServerResponse)
async def update_external_server(
    server_id: str,
    payload: ExternalServerUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ExternalServerResponse:
    """Update an existing external MCP server definition."""
    _require_mutation_permission(principal)
    try:
        row = await svc.update_external_server(
            server_id,
            name=payload.name,
            transport=payload.transport,
            config=payload.config,
            owner_scope_type=payload.owner_scope_type,
            owner_scope_id=payload.owner_scope_id,
            enabled=payload.enabled,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _external_row_to_response(row)


@router.delete("/external-servers/{server_id}", response_model=MCPHubDeleteResponse)
async def delete_external_server(
    server_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> MCPHubDeleteResponse:
    """Delete an external MCP server definition by id."""
    _require_mutation_permission(principal)
    deleted = await svc.delete_external_server(server_id, actor_id=principal.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="External server not found")
    return MCPHubDeleteResponse(ok=True)


@router.post("/external-servers/{server_id}/secret", response_model=ExternalSecretSetResponse)
async def set_external_secret(
    server_id: str,
    payload: ExternalSecretSetRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    svc: McpHubService = Depends(get_mcp_hub_service),
) -> ExternalSecretSetResponse:
    """Set or rotate an external MCP server secret using encrypted-at-rest storage."""
    _require_mutation_permission(principal)
    try:
        out = await svc.set_external_server_secret(
            server_id=server_id,
            secret_value=payload.secret,
            actor_id=principal.user_id,
        )
    except ResourceNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except BadRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExternalSecretSetResponse(**out)
