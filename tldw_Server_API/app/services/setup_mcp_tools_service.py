"""First-run MCP tools setup service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from tldw_Server_API.app.core.Setup.first_run_mcp_tools import (
    CATALOG_VERSION,
    PROFILE_DISPLAY_NAME,
    SETUP_ORIGIN,
    compute_first_run_policy_hash,
    generate_first_run_policy,
)

ConflictResolution = Literal["keep_existing", "replace_existing"]
_GLOBAL_SCOPE = "global"
_DEFAULT_TARGET = "default"
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
}


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


class SetupMcpToolsService:
    """Apply first-run MCP tool pack selections to MCP Hub profiles."""

    def __init__(self, *, hub: Any, tool_registry: Any | None = None) -> None:
        self.hub = hub
        if tool_registry is None:
            from tldw_Server_API.app.services.mcp_hub_tool_registry import (
                McpHubToolRegistryService,
            )

            tool_registry = McpHubToolRegistryService()
        self.tool_registry = tool_registry

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
    }
    return {key: payload[key] for key in _STEP_PAYLOAD_KEYS}


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
