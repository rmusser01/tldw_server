"""First-run MCP tools setup service."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
from typing import Any, Literal

from loguru import logger

from tldw_Server_API.app.core.Setup.first_run_mcp_tools import (
    CATALOG_VERSION,
    PROFILE_DISPLAY_NAME,
    SETUP_ORIGIN,
    compute_first_run_policy_hash,
    generate_first_run_policy,
    is_safe_external_validation_candidate,
)

ConflictResolution = Literal["keep_existing", "replace_existing"]
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]
_GLOBAL_SCOPE = "global"
_DEFAULT_TARGET = "default"
_BUILT_IN_SAMPLE_TOOL = "mcp.tools.list"
_GENERATED_POLICY_KEYS = ("allowed_tools", "capabilities", "first_run_mcp_tools")
_STEP_PAYLOAD_KEYS = {
    "acknowledged",
    "selected_pack_ids",
    "selected_addon_ids",
    "confirmed_addon_ids",
    "confirmation_version",
    "validation_state",
    "profile_id",
    "assignment_id",
    "catalog_version",
    "effective_tool_count",
    "validated_at",
    "validation_message",
    "last_validation_run_id",
    "sample_tool_name",
    "external_status",
}
_MESSAGE_BUILT_IN_PASSED = "Built-in MCP tool check passed."
_MESSAGE_BUILT_IN_FAILED = "Built-in MCP tool check failed."
_MESSAGE_EXTERNAL_DISCOVERY_INCOMPLETE = "External discovery did not complete."
_MESSAGE_NO_SAFE_EXTERNAL_TOOL = "No safe no-argument external read-only tool was available."
_MESSAGE_EXTERNAL_TOOL_FAILED = "External MCP validation tool check failed."
_MESSAGE_EXTERNAL_TOOL_PASSED = "External MCP validation tool check passed."
_MESSAGE_PROFILE_MANUALLY_CHANGED = "Generated MCP profile was changed in MCP Hub."


@dataclass(frozen=True)
class McpToolsApplyRequest:
    """Service-local apply request used before API schemas are wired."""

    selected_pack_ids: Sequence[str]
    selected_addon_ids: Sequence[str]
    confirmed_addon_ids: Sequence[str] = field(default_factory=list)
    confirmation_version: str | None = None
    conflict_resolution: ConflictResolution | None = None
    profile_id: int | None = None


@dataclass(frozen=True)
class McpToolsApplyResult:
    """Result returned by first-run MCP tools apply."""

    status: str
    profile_id: int | None
    assignment_id: int | None
    catalog_version: str
    selected_pack_ids: list[str]
    selected_addon_ids: list[str]
    effective_tool_count: int
    effective_tools: list[str]
    disabled_addons: list[str]
    validation_state: str
    step_payload: dict[str, Any]
    conflict: dict[str, Any] | None = None


@dataclass(frozen=True)
class McpToolsValidationResult:
    """Safe result returned by first-run MCP tool validation."""

    status: str
    validation_state: str
    profile_id: int | None
    assignment_id: int | None
    catalog_version: str | None
    selected_pack_ids: list[str]
    selected_addon_ids: list[str]
    effective_tool_count: int | None
    validated_at: str | None
    validation_message: str | None
    last_validation_run_id: str | None
    sample_tool_name: str | None
    external_status: str | None
    step_payload: dict[str, Any]


class SetupMcpToolsService:
    """Apply first-run MCP tool pack selections to MCP Hub profiles."""

    def __init__(
        self,
        *,
        hub: Any,
        tool_registry: Any | None = None,
        tool_executor: ToolExecutor | None = None,
    ) -> None:
        self.hub = hub
        if tool_registry is None:
            from tldw_Server_API.app.services.mcp_hub_tool_registry import (
                McpHubToolRegistryService,
            )

            tool_registry = McpHubToolRegistryService()
        self.tool_registry = tool_registry
        self.tool_executor = tool_executor or _default_tool_executor

    async def apply_selection(
        self,
        *,
        state: Any,
        request: McpToolsApplyRequest,
        actor_id: int | None = None,
    ) -> McpToolsApplyResult:
        """Apply a first-run MCP selection to a generated global default policy."""

        if request.conflict_resolution not in {None, "keep_existing", "replace_existing"}:
            raise ValueError("Unsupported first-run MCP conflict resolution")

        tool_entries = await self.tool_registry.list_entries()
        generated_policy = generate_first_run_policy(
            selected_pack_ids=request.selected_pack_ids,
            selected_addon_ids=request.selected_addon_ids,
            confirmed_addon_ids=request.confirmed_addon_ids,
            confirmation_version=request.confirmation_version,
            setup_instance_id=_setup_instance_id(state),
            tool_entries=tool_entries,
        )
        profile = await self._find_existing_profile(setup_instance_id=_setup_instance_id(state))

        if request.conflict_resolution is not None:
            if profile is None or request.profile_id != int(profile["id"]):
                raise ValueError("First-run MCP profile id mismatch")

        if profile is None:
            profile = await self.hub.create_permission_profile(
                name=PROFILE_DISPLAY_NAME,
                owner_scope_type=_GLOBAL_SCOPE,
                owner_scope_id=None,
                mode="custom",
                path_scope_object_id=None,
                policy_document=generated_policy,
                actor_id=actor_id,
            )
            policy_document = generated_policy
        else:
            policy_document = _dict(profile.get("policy_document"))
            conflict = _profile_conflict(profile, policy_document)
            if conflict and request.conflict_resolution == "keep_existing":
                assignment = await self._ensure_default_assignment(
                    profile_id=int(profile["id"]),
                    actor_id=actor_id,
                )
                return _applied_result(
                    policy_document=policy_document,
                    profile_id=int(profile["id"]),
                    assignment_id=int(assignment["id"]),
                    request=request,
                )
            if conflict and request.conflict_resolution != "replace_existing":
                return _conflict_result(
                    conflict=conflict,
                    policy_document=policy_document,
                    request=request,
                )
            merged_policy = _merge_generated_policy(policy_document, generated_policy)
            updated = await self.hub.update_permission_profile(
                int(profile["id"]),
                actor_id=actor_id,
                policy_document=merged_policy,
            )
            profile = updated or profile
            policy_document = merged_policy

        assignment = await self._ensure_default_assignment(
            profile_id=int(profile["id"]),
            actor_id=actor_id,
        )
        return _applied_result(
            policy_document=policy_document,
            profile_id=int(profile["id"]),
            assignment_id=int(assignment["id"]),
            request=request,
        )

    async def validate_selection(
        self,
        *,
        saved_state: Mapping[str, Any],
    ) -> McpToolsValidationResult:
        """Safely validate the saved first-run MCP tool selection."""

        saved_state = _dict(saved_state)
        policy_document = await self._validation_policy_document(saved_state)
        allowed_tools = _str_list(policy_document.get("allowed_tools"))
        if _BUILT_IN_SAMPLE_TOOL not in allowed_tools:
            return _validation_result(
                saved_state=saved_state,
                validation_state="failed",
                validation_message=_MESSAGE_BUILT_IN_FAILED,
                sample_tool_name=_BUILT_IN_SAMPLE_TOOL,
                external_status="not_checked",
            )

        try:
            tools_payload = await self.tool_executor(_BUILT_IN_SAMPLE_TOOL, {})
        except Exception:
            return _validation_result(
                saved_state=saved_state,
                validation_state="failed",
                validation_message=_MESSAGE_BUILT_IN_FAILED,
                sample_tool_name=_BUILT_IN_SAMPLE_TOOL,
                external_status="not_checked",
            )

        if not isinstance(tools_payload, Mapping) or not isinstance(tools_payload.get("tools"), list):
            return _validation_result(
                saved_state=saved_state,
                validation_state="failed",
                validation_message=_MESSAGE_BUILT_IN_FAILED,
                sample_tool_name=_BUILT_IN_SAMPLE_TOOL,
                external_status="not_checked",
            )

        external_servers = await self._enabled_external_servers()
        if not external_servers:
            return _validation_result(
                saved_state=saved_state,
                validation_state="built_in_passed",
                validation_message=_MESSAGE_BUILT_IN_PASSED,
                sample_tool_name=_BUILT_IN_SAMPLE_TOOL,
                external_status="not_configured",
            )

        refresh_succeeded = False
        for server in external_servers:
            server_id = str(server.get("id") or "").strip()
            try:
                refresh_payload = await self.tool_executor(
                    "external.tools.refresh",
                    {"server_id": server_id} if server_id else {},
                )
                refresh_succeeded = refresh_succeeded or _refresh_succeeded(refresh_payload)
            except Exception as exc:
                logger.debug("First-run MCP external discovery refresh failed: {}", type(exc).__name__)

        if not refresh_succeeded:
            return _validation_result(
                saved_state=saved_state,
                validation_state="external_discovery_incomplete",
                validation_message=_MESSAGE_EXTERNAL_DISCOVERY_INCOMPLETE,
                sample_tool_name=_BUILT_IN_SAMPLE_TOOL,
                external_status="discovery_incomplete",
            )

        try:
            refreshed_entries = await self.tool_registry.list_entries()
            refreshed_tools_payload = await self.tool_executor(
                _BUILT_IN_SAMPLE_TOOL,
                {"module": "external_federation"},
            )
        except Exception:
            return _validation_result(
                saved_state=saved_state,
                validation_state="external_discovery_incomplete",
                validation_message=_MESSAGE_EXTERNAL_DISCOVERY_INCOMPLETE,
                sample_tool_name=_BUILT_IN_SAMPLE_TOOL,
                external_status="discovery_incomplete",
            )

        tool_defs = _tool_defs_by_name(refreshed_tools_payload)
        for entry in refreshed_entries:
            entry = _dict(entry)
            tool_name = str(entry.get("tool_name") or "").strip()
            if not tool_name:
                continue
            tool_def = tool_defs.get(tool_name)
            if tool_def is None:
                continue
            if not is_safe_external_validation_candidate(entry, tool_def):
                continue
            try:
                await self.tool_executor(tool_name, {})
            except Exception:
                return _validation_result(
                    saved_state=saved_state,
                    validation_state="failed",
                    validation_message=_MESSAGE_EXTERNAL_TOOL_FAILED,
                    sample_tool_name=tool_name,
                    external_status="tool_failed",
                )
            return _validation_result(
                saved_state=saved_state,
                validation_state="external_tool_passed",
                validation_message=_MESSAGE_EXTERNAL_TOOL_PASSED,
                sample_tool_name=tool_name,
                external_status="tool_passed",
            )

        return _validation_result(
            saved_state=saved_state,
            validation_state="no_safe_external_tool",
            validation_message=_MESSAGE_NO_SAFE_EXTERNAL_TOOL,
            sample_tool_name=_BUILT_IN_SAMPLE_TOOL,
            external_status="no_safe_tool",
        )

    async def recovery_status(
        self,
        *,
        saved_state: Mapping[str, Any],
    ) -> McpToolsValidationResult | None:
        """Return follow-up status that requires MCP Hub data, if any."""

        saved_state = _dict(saved_state)
        profile = await self._saved_profile(saved_state)
        if profile is None:
            return None
        policy_document = _dict(profile.get("policy_document"))
        if _dict(policy_document.get("first_run_mcp_tools")).get("setup_origin") != SETUP_ORIGIN:
            return None
        conflict = _profile_conflict(profile, policy_document)
        if conflict is None:
            return None
        return _status_result(
            saved_state=saved_state,
            status="profile_manually_changed",
            validation_state=str(saved_state.get("validation_state") or "not_run"),
            validation_message=_MESSAGE_PROFILE_MANUALLY_CHANGED,
        )

    async def _find_existing_profile(self, *, setup_instance_id: str) -> dict[str, Any] | None:
        profiles = await self.hub.list_permission_profiles(
            owner_scope_type=_GLOBAL_SCOPE,
            owner_scope_id=None,
        )
        for profile in profiles:
            if profile.get("owner_scope_type") != _GLOBAL_SCOPE or profile.get("owner_scope_id") is not None:
                continue
            policy_document = _dict(profile.get("policy_document"))
            provenance = _dict(policy_document.get("first_run_mcp_tools"))
            if (
                provenance.get("setup_origin") == SETUP_ORIGIN
                and provenance.get("setup_instance_id") == setup_instance_id
            ):
                return _dict(profile)
        return None

    async def _ensure_default_assignment(self, *, profile_id: int, actor_id: int | None) -> dict[str, Any]:
        assignments = await self.hub.list_policy_assignments(
            owner_scope_type=_GLOBAL_SCOPE,
            owner_scope_id=None,
            target_type=_DEFAULT_TARGET,
            target_id=None,
        )
        assignments = [
            assignment
            for assignment in assignments
            if assignment.get("owner_scope_type") == _GLOBAL_SCOPE
            and assignment.get("owner_scope_id") is None
            and assignment.get("target_type") == _DEFAULT_TARGET
            and assignment.get("target_id") is None
        ]
        if not assignments:
            return await self.hub.create_policy_assignment(
                target_type=_DEFAULT_TARGET,
                target_id=None,
                owner_scope_type=_GLOBAL_SCOPE,
                owner_scope_id=None,
                profile_id=profile_id,
                path_scope_object_id=None,
                workspace_source_mode=None,
                workspace_set_object_id=None,
                inline_policy_document={},
                approval_policy_id=None,
                actor_id=actor_id,
                is_active=True,
            )

        assignment = _dict(assignments[0])
        if assignment.get("profile_id") == profile_id:
            return assignment
        updated = await self.hub.update_policy_assignment(
            int(assignment["id"]),
            actor_id=actor_id,
            profile_id=profile_id,
        )
        return _dict(updated or assignment)

    async def _validation_policy_document(self, saved_state: Mapping[str, Any]) -> dict[str, Any]:
        profile = await self._saved_profile(saved_state)
        return _dict(profile.get("policy_document")) if profile else {}

    async def _saved_profile(self, saved_state: Mapping[str, Any]) -> dict[str, Any] | None:
        profile_id = _safe_int(saved_state.get("profile_id"))
        if profile_id is None:
            return None
        profiles = await self.hub.list_permission_profiles(
            owner_scope_type=_GLOBAL_SCOPE,
            owner_scope_id=None,
        )
        for profile in profiles:
            if _safe_int(_dict(profile).get("id")) == profile_id:
                return _dict(profile)
        return None

    async def _enabled_external_servers(self) -> list[dict[str, Any]]:
        list_external_servers = getattr(self.hub, "list_external_servers", None)
        if not callable(list_external_servers):
            return []
        servers = await list_external_servers()
        return [_dict(server) for server in servers if bool(_dict(server).get("enabled", True))]


def _setup_instance_id(state: Any) -> str:
    return f"first_run:{state.created_at.isoformat()}"


def _profile_conflict(profile: Mapping[str, Any], policy_document: Mapping[str, Any]) -> dict[str, Any] | None:
    provenance = _dict(policy_document.get("first_run_mcp_tools"))
    current_hash = compute_first_run_policy_hash(policy_document)
    expected_hash = str(provenance.get("last_generated_hash") or "")
    if current_hash == expected_hash:
        return None
    return {
        "reason": "profile_manually_changed",
        "profile_id": int(profile["id"]),
        "current_hash": current_hash,
        "expected_hash": expected_hash,
    }


def _merge_generated_policy(
    existing_policy: Mapping[str, Any],
    generated_policy: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(existing_policy)
    for key in _GENERATED_POLICY_KEYS:
        if key in generated_policy:
            merged[key] = generated_policy[key]
        else:
            merged.pop(key, None)
    return merged


def _applied_result(
    *,
    policy_document: Mapping[str, Any],
    profile_id: int,
    assignment_id: int,
    request: McpToolsApplyRequest,
) -> McpToolsApplyResult:
    provenance = _dict(policy_document.get("first_run_mcp_tools"))
    effective_tools = _str_list(policy_document.get("allowed_tools"))
    selected_pack_ids = _str_list(provenance.get("selected_pack_ids")) or _str_list(request.selected_pack_ids)
    selected_addon_ids = _str_list(provenance.get("selected_addon_ids")) or _str_list(request.selected_addon_ids)
    step_payload = _step_payload(
        request=request,
        profile_id=profile_id,
        assignment_id=assignment_id,
        selected_pack_ids=selected_pack_ids,
        selected_addon_ids=selected_addon_ids,
        effective_tool_count=len(effective_tools),
    )
    return McpToolsApplyResult(
        status="applied",
        profile_id=profile_id,
        assignment_id=assignment_id,
        catalog_version=CATALOG_VERSION,
        selected_pack_ids=selected_pack_ids,
        selected_addon_ids=selected_addon_ids,
        effective_tool_count=len(effective_tools),
        effective_tools=effective_tools,
        disabled_addons=[],
        validation_state="not_run",
        step_payload=step_payload,
    )


def _conflict_result(
    *,
    conflict: dict[str, Any],
    policy_document: Mapping[str, Any],
    request: McpToolsApplyRequest,
) -> McpToolsApplyResult:
    provenance = _dict(policy_document.get("first_run_mcp_tools"))
    effective_tools = _str_list(policy_document.get("allowed_tools"))
    return McpToolsApplyResult(
        status="conflict",
        profile_id=int(conflict["profile_id"]),
        assignment_id=None,
        catalog_version=CATALOG_VERSION,
        selected_pack_ids=_str_list(provenance.get("selected_pack_ids")) or _str_list(request.selected_pack_ids),
        selected_addon_ids=_str_list(provenance.get("selected_addon_ids")) or _str_list(request.selected_addon_ids),
        effective_tool_count=len(effective_tools),
        effective_tools=effective_tools,
        disabled_addons=[],
        validation_state="not_run",
        step_payload={},
        conflict=conflict,
    )


def _step_payload(
    *,
    request: McpToolsApplyRequest,
    profile_id: int,
    assignment_id: int,
    selected_pack_ids: list[str],
    selected_addon_ids: list[str],
    effective_tool_count: int,
) -> dict[str, Any]:
    selected_addon_set = set(selected_addon_ids)
    payload = {
        "acknowledged": True,
        "selected_pack_ids": selected_pack_ids,
        "selected_addon_ids": selected_addon_ids,
        "confirmed_addon_ids": [
            addon_id
            for addon_id in _str_list(request.confirmed_addon_ids)
            if addon_id in selected_addon_set
        ],
        "confirmation_version": request.confirmation_version,
        "validation_state": "not_run",
        "profile_id": profile_id,
        "assignment_id": assignment_id,
        "catalog_version": CATALOG_VERSION,
        "effective_tool_count": effective_tool_count,
        "validated_at": None,
        "validation_message": None,
        "last_validation_run_id": None,
        "sample_tool_name": None,
        "external_status": None,
    }
    return {key: payload[key] for key in _STEP_PAYLOAD_KEYS}


def _validation_result(
    *,
    saved_state: Mapping[str, Any],
    validation_state: str,
    validation_message: str,
    sample_tool_name: str | None,
    external_status: str,
) -> McpToolsValidationResult:
    validated_at = datetime.now(timezone.utc).isoformat()
    last_validation_run_id = f"mcp-tools-validation:{uuid.uuid4().hex}"
    profile_id = _safe_int(saved_state.get("profile_id"))
    assignment_id = _safe_int(saved_state.get("assignment_id"))
    catalog_version = str(saved_state["catalog_version"]) if saved_state.get("catalog_version") else CATALOG_VERSION
    selected_pack_ids = _str_list(saved_state.get("selected_pack_ids"))
    selected_addon_ids = _str_list(saved_state.get("selected_addon_ids"))
    effective_tool_count = _safe_int(saved_state.get("effective_tool_count"))
    payload = {
        "acknowledged": True,
        "selected_pack_ids": selected_pack_ids,
        "selected_addon_ids": selected_addon_ids,
        "confirmed_addon_ids": _str_list(saved_state.get("confirmed_addon_ids")),
        "confirmation_version": saved_state.get("confirmation_version"),
        "validation_state": validation_state,
        "profile_id": profile_id,
        "assignment_id": assignment_id,
        "catalog_version": catalog_version,
        "effective_tool_count": effective_tool_count,
        "validated_at": validated_at,
        "validation_message": validation_message,
        "last_validation_run_id": last_validation_run_id,
        "sample_tool_name": sample_tool_name,
        "external_status": external_status,
    }
    return McpToolsValidationResult(
        status="failed" if validation_state == "failed" else "validated",
        validation_state=validation_state,
        profile_id=profile_id,
        assignment_id=assignment_id,
        catalog_version=catalog_version,
        selected_pack_ids=selected_pack_ids,
        selected_addon_ids=selected_addon_ids,
        effective_tool_count=effective_tool_count,
        validated_at=validated_at,
        validation_message=validation_message,
        last_validation_run_id=last_validation_run_id,
        sample_tool_name=sample_tool_name,
        external_status=external_status,
        step_payload={key: payload[key] for key in _STEP_PAYLOAD_KEYS},
    )


def _status_result(
    *,
    saved_state: Mapping[str, Any],
    status: str,
    validation_state: str,
    validation_message: str,
) -> McpToolsValidationResult:
    profile_id = _safe_int(saved_state.get("profile_id"))
    assignment_id = _safe_int(saved_state.get("assignment_id"))
    return McpToolsValidationResult(
        status=status,
        validation_state=validation_state,
        profile_id=profile_id,
        assignment_id=assignment_id,
        catalog_version=str(saved_state["catalog_version"]) if saved_state.get("catalog_version") else CATALOG_VERSION,
        selected_pack_ids=_str_list(saved_state.get("selected_pack_ids")),
        selected_addon_ids=_str_list(saved_state.get("selected_addon_ids")),
        effective_tool_count=_safe_int(saved_state.get("effective_tool_count")),
        validated_at=str(saved_state["validated_at"]) if saved_state.get("validated_at") else None,
        validation_message=validation_message,
        last_validation_run_id=(
            str(saved_state["last_validation_run_id"]) if saved_state.get("last_validation_run_id") else None
        ),
        sample_tool_name=str(saved_state["sample_tool_name"]) if saved_state.get("sample_tool_name") else None,
        external_status=str(saved_state["external_status"]) if saved_state.get("external_status") else None,
        step_payload={},
    )


def _tool_defs_by_name(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tools"), list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for tool_def in payload["tools"]:
        tool_def = _dict(tool_def)
        name = str(tool_def.get("name") or tool_def.get("tool_name") or "").strip()
        if name:
            out[name] = tool_def
    return out


def _refresh_succeeded(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if "refreshed_servers" in payload:
        return _safe_int(payload.get("refreshed_servers")) not in {None, 0}
    errors = payload.get("errors")
    if isinstance(errors, Mapping) and errors:
        return False
    return payload.get("ok") is True or payload.get("success") is True


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _default_tool_executor(tool_name: str, arguments: dict[str, Any]) -> Any:
    args = dict(arguments or {})
    if tool_name == _BUILT_IN_SAMPLE_TOOL:
        from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol
        from tldw_Server_API.app.core.MCP_unified.protocol_types import RequestContext

        context = RequestContext(
            request_id=f"first-run-mcp-tools-validation:{uuid.uuid4().hex}",
            client_id="setup",
            metadata={},
        )
        return await MCPProtocol()._handle_tools_list(args, context)

    from tldw_Server_API.app.core.MCP_unified.server import get_mcp_server

    server = get_mcp_server()
    if not getattr(server, "initialized", False):
        await server.initialize()
    module = await server.module_registry.find_module_for_tool(tool_name)
    if module is None:
        raise RuntimeError("MCP validation tool unavailable")
    return await module.execute_with_circuit_breaker(module.execute_tool, tool_name, args)


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, Sequence):
        return []
    out: list[str] = []
    for item in value:
        cleaned = str(item or "").strip()
        if cleaned:
            out.append(cleaned)
    return out
