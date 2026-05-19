from __future__ import annotations

from copy import deepcopy
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.merge_utils import (
    UNION_LIST_KEYS as _UNION_LIST_KEYS,
    _as_dict,
    _as_str_list,
    _unique,
    merge_config,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.services.mcp_hub_capability_resolution_service import (
    CapabilityResolutionResult,
    McpHubCapabilityResolutionService,
)

_TARGET_ORDER = {"default": 0, "group": 1, "persona": 2}
_SCOPE_ORDER = {"global": 0, "org": 1, "team": 2, "user": 3}


def _collect_scope_ids(metadata: dict[str, Any], singular_key: str, plural_key: str) -> list[int]:
    """Collect integer scope ids from singular and plural metadata keys."""
    raw_values: list[Any] = []
    if singular_key in metadata:
        raw_values.append(metadata.get(singular_key))
    plural_value = metadata.get(plural_key)
    if isinstance(plural_value, (list, tuple, set)):
        raw_values.extend(list(plural_value))

    out: set[int] = set()
    for raw in raw_values:
        try:
            out.add(int(raw))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _normalize_scope_id(value: Any) -> int | None:
    """Return an integer scope id when present and parseable."""
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _governance_pack_reference(document: dict[str, Any]) -> tuple[str, str] | None:
    """Extract governance-pack identity metadata from a stored policy document."""
    metadata = _as_dict(document.get("governance_pack"))
    pack_id = str(metadata.get("pack_id") or "").strip()
    pack_version = str(metadata.get("pack_version") or "").strip()
    if not pack_id or not pack_version:
        return None
    return pack_id, pack_version


def _extract_targets(metadata: dict[str, Any]) -> list[tuple[str, str | None]]:
    """Return the ordered assignment targets applicable to the runtime metadata."""
    targets: list[tuple[str, str | None]] = [("default", None)]
    group_id = str(metadata.get("group_id") or "").strip()
    if group_id:
        targets.append(("group", group_id))
    persona_id = str(metadata.get("persona_id") or "").strip()
    if persona_id:
        targets.append(("persona", persona_id))
    return targets


def _candidate_scope_filters(user_id: int | None, metadata: dict[str, Any]) -> list[tuple[str, int | None]]:
    """Build candidate owner-scope filters for assignment lookup."""
    filters: list[tuple[str, int | None]] = [("global", None)]
    filters.extend(("org", org_id) for org_id in _collect_scope_ids(metadata, "org_id", "org_ids"))
    filters.extend(("team", team_id) for team_id in _collect_scope_ids(metadata, "team_id", "team_ids"))
    if user_id is not None:
        filters.append(("user", user_id))
    return filters


def _merge_policy_documents(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge policy documents with union semantics and explicit-null overrides."""
    return merge_config(base, overlay, skip_none=False)


def _has_explicit_scalar_value(value: Any) -> bool:
    """Return whether a scalar value should override mapping-provided hints."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _merge_resolved_policy_documents(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge mapping effects into authored policy while preserving explicit authored scalars."""
    merged = deepcopy(base)
    for key, value in overlay.items():
        if key in _UNION_LIST_KEYS:
            merged[key] = _unique(_as_str_list(merged.get(key)) + _as_str_list(value))
            continue
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_resolved_policy_documents(_as_dict(merged.get(key)), value)
            continue
        if key not in merged or not _has_explicit_scalar_value(merged.get(key)):
            merged[key] = deepcopy(value)
    return merged


def _allowed_tool_patterns(policy_document: dict[str, Any]) -> list[str]:
    """Return the effective allowed-tool pattern list from a policy document."""
    patterns = (
        _as_str_list(policy_document.get("allowed_tools"))
        + _as_str_list(policy_document.get("tool_patterns"))
        + _as_str_list(policy_document.get("tool_names"))
    )
    return _unique(patterns)


def _provenance_entries(
    *,
    layer_document: dict[str, Any],
    source_kind: str,
    assignment_id: int,
    profile_id: int | None,
    override_id: int | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for field, value in layer_document.items():
        entries.append(
            {
                "field": str(field),
                "value": deepcopy(value),
                "source_kind": source_kind,
                "assignment_id": assignment_id,
                "profile_id": profile_id,
                "override_id": override_id,
                "effect": "merged" if field in _UNION_LIST_KEYS else "replaced",
            }
        )
    return entries


def _capability_mapping_provenance_entries(
    resolution: CapabilityResolutionResult,
    *,
    resolution_intent: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for summary in resolution.mapping_summaries:
        is_deny = str(resolution_intent or "").strip().lower() == "deny"
        entries.append(
            {
                "field": "capabilities",
                "value": summary.get("capability_name"),
                "source_kind": "capability_mapping",
                "assignment_id": None,
                "profile_id": None,
                "override_id": None,
                "capability_name": summary.get("capability_name"),
                "mapping_id": summary.get("mapping_id"),
                "mapping_scope_type": summary.get("mapping_scope_type"),
                "mapping_scope_id": summary.get("mapping_scope_id"),
                "resolved_effects": deepcopy(_as_dict(summary.get("resolved_effects"))),
                "resolution_intent": "deny" if is_deny else "allow",
                "effect": "narrowed" if is_deny else "merged",
            }
        )
    for capability_name in resolution.unresolved_capabilities:
        is_deny = str(resolution_intent or "").strip().lower() == "deny"
        entries.append(
            {
                "field": "capabilities",
                "value": capability_name,
                "source_kind": "capability_mapping",
                "assignment_id": None,
                "profile_id": None,
                "override_id": None,
                "capability_name": capability_name,
                "mapping_id": None,
                "mapping_scope_type": None,
                "mapping_scope_id": None,
                "resolved_effects": {},
                "resolution_intent": "deny" if is_deny else "allow",
                "effect": "blocked",
            }
        )
    return entries


def _apply_denied_capability_resolution(
    base: dict[str, Any],
    deny_overlay: dict[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(base)
    denied_patterns = _unique(
        _as_str_list(deny_overlay.get("denied_tools"))
        + _as_str_list(deny_overlay.get("allowed_tools"))
        + _as_str_list(deny_overlay.get("tool_patterns"))
        + _as_str_list(deny_overlay.get("tool_names"))
    )
    if denied_patterns:
        merged["denied_tools"] = _unique(
            _as_str_list(merged.get("denied_tools")) + denied_patterns
        )
    return merged


class McpHubPolicyResolver:
    """Resolve effective MCP Hub policy for a runtime request context."""

    def __init__(
        self,
        repo: McpHubRepo,
        capability_resolution_service: McpHubCapabilityResolutionService | None = None,
    ) -> None:
        self.repo = repo
        self.capability_resolution_service = capability_resolution_service or McpHubCapabilityResolutionService(
            repo=repo
        )

    async def _document_uses_active_governance_pack(
        self,
        *,
        document: dict[str, Any],
        owner_scope_type: str | None,
        owner_scope_id: Any,
        cache: dict[tuple[str, str, str, int | None], bool],
    ) -> bool:
        pack_ref = _governance_pack_reference(document)
        if pack_ref is None:
            return True
        scope_type = str(owner_scope_type or "global").strip().lower() or "global"
        scope_id = _normalize_scope_id(owner_scope_id)
        cache_key = (pack_ref[0], pack_ref[1], scope_type, scope_id)
        if cache_key not in cache:
            pack_row = await self.repo.get_governance_pack_by_identity(
                pack_id=pack_ref[0],
                pack_version=pack_ref[1],
                owner_scope_type=scope_type,
                owner_scope_id=scope_id,
            )
            cache[cache_key] = bool(pack_row and pack_row.get("is_active_install"))
        return cache[cache_key]

    async def _row_uses_active_governance_pack(
        self,
        *,
        row: dict[str, Any],
        document_field: str,
        cache: dict[tuple[str, str, str, int | None], bool],
    ) -> bool:
        return await self._document_uses_active_governance_pack(
            document=_as_dict(row.get(document_field)),
            owner_scope_type=row.get("owner_scope_type"),
            owner_scope_id=row.get("owner_scope_id"),
            cache=cache,
        )

    async def resolve_for_context(
        self,
        *,
        user_id: int | str | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metadata_map = dict(metadata or {})
        if not metadata_map.get("mcp_policy_context_enabled"):
            return self._disabled_policy()

        user_id_int: int | None = None
        try:
            if user_id is not None:
                user_id_int = int(str(user_id))
        except (TypeError, ValueError):
            user_id_int = None

        assignments = await self._load_applicable_assignments(
            user_id=user_id_int,
            metadata=metadata_map,
        )
        if not assignments:
            return self._disabled_policy()

        merged_policy_document: dict[str, Any] = {}
        sources: list[dict[str, Any]] = []
        profile_cache: dict[int, dict[str, Any] | None] = {}
        resolved_approval_policy_id: int | None = None
        provenance: list[dict[str, Any]] = []
        selected_assignment_id: int | None = None
        selected_assignment_workspace_ids: list[str] = []
        selected_workspace_source_mode: str | None = None
        selected_workspace_set_object_id: int | None = None
        selected_workspace_set_object_name: str | None = None
        selected_workspace_trust_source: str | None = None
        selected_workspace_scope_type: str | None = None
        selected_workspace_scope_id: int | None = None
        path_scope_object_cache: dict[int, dict[str, Any] | None] = {}
        workspace_set_object_cache: dict[int, dict[str, Any] | None] = {}
        approval_policy_cache: dict[int, dict[str, Any] | None] = {}
        governance_pack_activity_cache: dict[tuple[str, str, str, int | None], bool] = {}

        for assignment in assignments:
            if not await self._row_uses_active_governance_pack(
                row=assignment,
                document_field="inline_policy_document",
                cache=governance_pack_activity_cache,
            ):
                continue
            assignment_document: dict[str, Any] = {}
            profile_id = assignment.get("profile_id")
            profile_key: int | None = int(profile_id) if profile_id is not None else None
            assignment_id = int(assignment.get("id"))
            if profile_id is not None:
                if profile_key not in profile_cache:
                    profile_cache[profile_key] = await self.repo.get_permission_profile(profile_key)
                profile_row = profile_cache.get(profile_key) or {}
                if bool(profile_row.get("is_active", True)) and await self._row_uses_active_governance_pack(
                    row=profile_row,
                    document_field="policy_document",
                    cache=governance_pack_activity_cache,
                ):
                    profile_path_scope_object_id = profile_row.get("path_scope_object_id")
                    profile_path_scope_key = (
                        int(profile_path_scope_object_id)
                        if profile_path_scope_object_id is not None
                        else None
                    )
                    if profile_path_scope_key is not None:
                        if profile_path_scope_key not in path_scope_object_cache:
                            path_scope_object_cache[profile_path_scope_key] = await self.repo.get_path_scope_object(
                                profile_path_scope_key
                            )
                        path_scope_row = path_scope_object_cache.get(profile_path_scope_key) or {}
                        if bool(path_scope_row.get("is_active", True)):
                            path_scope_document = _as_dict(path_scope_row.get("path_scope_document"))
                            assignment_document = _merge_policy_documents(
                                assignment_document,
                                path_scope_document,
                            )
                            provenance.extend(
                                _provenance_entries(
                                    layer_document=path_scope_document,
                                    source_kind="profile_path_scope_object",
                                    assignment_id=assignment_id,
                                    profile_id=profile_key,
                                    override_id=None,
                                )
                            )
                    profile_document = _as_dict(profile_row.get("policy_document"))
                    assignment_document = _merge_policy_documents(
                        assignment_document,
                        profile_document,
                    )
                    provenance.extend(
                        _provenance_entries(
                            layer_document=profile_document,
                            source_kind="profile",
                            assignment_id=assignment_id,
                            profile_id=profile_key,
                            override_id=None,
                        )
                    )

            assignment_path_scope_object_id = assignment.get("path_scope_object_id")
            assignment_path_scope_key = (
                int(assignment_path_scope_object_id)
                if assignment_path_scope_object_id is not None
                else None
            )
            if assignment_path_scope_key is not None:
                if assignment_path_scope_key not in path_scope_object_cache:
                    path_scope_object_cache[assignment_path_scope_key] = await self.repo.get_path_scope_object(
                        assignment_path_scope_key
                    )
                assignment_path_scope_row = path_scope_object_cache.get(assignment_path_scope_key) or {}
                if bool(assignment_path_scope_row.get("is_active", True)):
                    assignment_path_scope_document = _as_dict(
                        assignment_path_scope_row.get("path_scope_document")
                    )
                    assignment_document = _merge_policy_documents(
                        assignment_document,
                        assignment_path_scope_document,
                    )
                    provenance.extend(
                        _provenance_entries(
                            layer_document=assignment_path_scope_document,
                            source_kind="assignment_path_scope_object",
                            assignment_id=assignment_id,
                            profile_id=profile_key,
                            override_id=None,
                        )
                    )

            inline_policy_document = _as_dict(assignment.get("inline_policy_document"))
            assignment_document = _merge_policy_documents(
                assignment_document,
                inline_policy_document,
            )
            provenance.extend(
                _provenance_entries(
                    layer_document=inline_policy_document,
                    source_kind="assignment_inline",
                    assignment_id=assignment_id,
                    profile_id=profile_key,
                    override_id=None,
                )
            )
            override_row = await self.repo.get_policy_override_by_assignment(assignment_id)
            if override_row and bool(override_row.get("is_active", True)):
                override_document = _as_dict(override_row.get("override_policy_document"))
                assignment_document = _merge_policy_documents(
                    assignment_document,
                    override_document,
                )
                provenance.extend(
                    _provenance_entries(
                        layer_document=override_document,
                        source_kind="assignment_override",
                        assignment_id=assignment_id,
                        profile_id=profile_key,
                        override_id=int(override_row.get("id")),
                    )
                )
            merged_policy_document = _merge_policy_documents(merged_policy_document, assignment_document)
            approval_policy_id = assignment.get("approval_policy_id")
            if approval_policy_id is not None:
                approval_policy_key = int(approval_policy_id)
                if approval_policy_key not in approval_policy_cache:
                    approval_policy_cache[approval_policy_key] = await self.repo.get_approval_policy(
                        approval_policy_key
                    )
                approval_policy_row = approval_policy_cache.get(approval_policy_key) or {}
                if bool(approval_policy_row.get("is_active", True)) and await self._row_uses_active_governance_pack(
                    row=approval_policy_row,
                    document_field="rules",
                    cache=governance_pack_activity_cache,
                ):
                    resolved_approval_policy_id = approval_policy_key
            selected_assignment_id = assignment_id
            selected_workspace_scope_type = str(assignment.get("owner_scope_type") or "global")
            selected_workspace_scope_id = assignment.get("owner_scope_id")
            selected_workspace_source_mode = (
                str(assignment.get("workspace_source_mode") or "").strip().lower() or "inline"
            )
            selected_workspace_set_object_id = None
            selected_workspace_set_object_name = None
            selected_workspace_trust_source = "user_local"
            if selected_workspace_source_mode == "named" and assignment.get("workspace_set_object_id") is not None:
                selected_workspace_set_object_id = int(assignment.get("workspace_set_object_id"))
                if selected_workspace_set_object_id not in workspace_set_object_cache:
                    workspace_set_object_cache[selected_workspace_set_object_id] = await self.repo.get_workspace_set_object(
                        selected_workspace_set_object_id
                    )
                workspace_set_object_row = workspace_set_object_cache.get(selected_workspace_set_object_id) or {}
                if bool(workspace_set_object_row.get("is_active", True)):
                    selected_workspace_set_object_name = str(workspace_set_object_row.get("name") or "").strip() or None
                    selected_workspace_trust_source = (
                        "shared_registry"
                        if str(workspace_set_object_row.get("owner_scope_type") or "").strip().lower() != "user"
                        else "user_local"
                    )
                    selected_assignment_workspace_ids = _unique(
                        _as_str_list(
                            [
                                row.get("workspace_id")
                                for row in await self.repo.list_workspace_set_members(selected_workspace_set_object_id)
                            ]
                        )
                    )
                else:
                    selected_assignment_workspace_ids = []
            else:
                selected_assignment_workspace_ids = _unique(
                    _as_str_list(
                        [
                            row.get("workspace_id")
                            for row in await self.repo.list_policy_assignment_workspaces(assignment_id)
                        ]
                    )
                )
            sources.append(
                {
                    "assignment_id": assignment_id,
                    "target_type": str(assignment.get("target_type") or "default"),
                    "target_id": assignment.get("target_id"),
                    "owner_scope_type": str(assignment.get("owner_scope_type") or "global"),
                    "owner_scope_id": assignment.get("owner_scope_id"),
                    "profile_id": assignment.get("profile_id"),
                    "path_scope_object_id": assignment.get("path_scope_object_id"),
                }
            )

        if not sources:
            return self._disabled_policy()

        authored_policy_document = deepcopy(merged_policy_document)
        allow_capability_resolution = await self.capability_resolution_service.resolve_capabilities(
            capability_names=_as_str_list(authored_policy_document.get("capabilities")),
            metadata=metadata_map,
            resolution_intent="allow",
        )
        deny_capability_resolution = await self.capability_resolution_service.resolve_capabilities(
            capability_names=_as_str_list(authored_policy_document.get("denied_capabilities")),
            metadata=metadata_map,
            resolution_intent="deny",
        )
        resolved_policy_document = _merge_resolved_policy_documents(
            authored_policy_document,
            allow_capability_resolution.resolved_policy_document,
        )
        resolved_policy_document = _apply_denied_capability_resolution(
            resolved_policy_document,
            deny_capability_resolution.resolved_policy_document,
        )
        provenance.extend(
            _capability_mapping_provenance_entries(
                allow_capability_resolution,
                resolution_intent="allow",
            )
        )
        provenance.extend(
            _capability_mapping_provenance_entries(
                deny_capability_resolution,
                resolution_intent="deny",
            )
        )
        capability_mapping_summary = [
            *allow_capability_resolution.mapping_summaries,
            *deny_capability_resolution.mapping_summaries,
        ]
        capability_warnings = [
            *allow_capability_resolution.warnings,
            *deny_capability_resolution.warnings,
        ]
        resolved_capabilities = _unique(
            allow_capability_resolution.resolved_capabilities
            + deny_capability_resolution.resolved_capabilities
        )
        unresolved_capabilities = _unique(
            allow_capability_resolution.unresolved_capabilities
            + deny_capability_resolution.unresolved_capabilities
        )
        supported_environment_requirements = _unique(
            allow_capability_resolution.supported_environment_requirements
            + deny_capability_resolution.supported_environment_requirements
        )
        unsupported_environment_requirements = _unique(
            allow_capability_resolution.unsupported_environment_requirements
            + deny_capability_resolution.unsupported_environment_requirements
        )

        # Ensure tool_tier_overrides is present in the resolved document so
        # downstream consumers (GovernanceFilter, runner_client) can rely on it.
        resolved_policy_document.setdefault("tool_tier_overrides", {})

        # Ensure conditions key is present so GovernanceFilter can always read it.
        resolved_policy_document.setdefault("conditions", {})

        return {
            "enabled": True,
            "allowed_tools": _allowed_tool_patterns(resolved_policy_document),
            "denied_tools": _unique(_as_str_list(resolved_policy_document.get("denied_tools"))),
            "capabilities": _unique(_as_str_list(resolved_policy_document.get("capabilities"))),
            "approval_policy_id": resolved_approval_policy_id,
            "approval_mode": str(resolved_policy_document.get("approval_mode") or "").strip() or None,
            "policy_document": resolved_policy_document,
            "authored_policy_document": authored_policy_document,
            "resolved_policy_document": resolved_policy_document,
            "resolved_capabilities": resolved_capabilities,
            "unresolved_capabilities": unresolved_capabilities,
            "capability_mapping_summary": capability_mapping_summary,
            "capability_warnings": capability_warnings,
            "supported_environment_requirements": supported_environment_requirements,
            "unsupported_environment_requirements": unsupported_environment_requirements,
            "selected_assignment_id": selected_assignment_id,
            "selected_workspace_source_mode": selected_workspace_source_mode,
            "selected_workspace_set_object_id": selected_workspace_set_object_id,
            "selected_workspace_set_object_name": selected_workspace_set_object_name,
            "selected_workspace_trust_source": selected_workspace_trust_source,
            "selected_workspace_scope_type": selected_workspace_scope_type,
            "selected_workspace_scope_id": selected_workspace_scope_id,
            "selected_assignment_workspace_ids": selected_assignment_workspace_ids,
            "sources": sources,
            "provenance": provenance,
        }

    async def _load_applicable_assignments(
        self,
        *,
        user_id: int | None,
        metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for scope_type, scope_id in _candidate_scope_filters(user_id, metadata):
            for target_type, target_id in _extract_targets(metadata):
                rows.extend(
                    await self.repo.list_policy_assignments(
                        owner_scope_type=scope_type,
                        owner_scope_id=scope_id,
                        target_type=target_type,
                        target_id=target_id,
                    )
                )

        filtered = [
            row
            for row in rows
            if bool(row.get("is_active", True))
            and str(row.get("target_type") or "").strip().lower() in _TARGET_ORDER
            and str(row.get("owner_scope_type") or "").strip().lower() in _SCOPE_ORDER
        ]
        deduped: dict[int, dict[str, Any]] = {}
        for row in filtered:
            deduped[int(row.get("id"))] = row

        return sorted(
            deduped.values(),
            key=lambda row: (
                _TARGET_ORDER.get(str(row.get("target_type") or "").strip().lower(), 99),
                _SCOPE_ORDER.get(str(row.get("owner_scope_type") or "").strip().lower(), 99),
                int(row.get("id") or 0),
            ),
        )

    @staticmethod
    def _disabled_policy() -> dict[str, Any]:
        return {
            "enabled": False,
            "allowed_tools": [],
            "denied_tools": [],
            "capabilities": [],
            "approval_policy_id": None,
            "approval_mode": None,
            "policy_document": {},
            "authored_policy_document": {},
            "resolved_policy_document": {"tool_tier_overrides": {}},
            "resolved_capabilities": [],
            "unresolved_capabilities": [],
            "capability_mapping_summary": [],
            "capability_warnings": [],
            "supported_environment_requirements": [],
            "unsupported_environment_requirements": [],
            "selected_assignment_id": None,
            "selected_workspace_source_mode": None,
            "selected_workspace_set_object_id": None,
            "selected_workspace_set_object_name": None,
            "selected_workspace_trust_source": None,
            "selected_workspace_scope_type": None,
            "selected_workspace_scope_id": None,
            "selected_assignment_workspace_ids": [],
            "sources": [],
            "provenance": [],
        }


async def get_mcp_hub_policy_resolver() -> McpHubPolicyResolver:
    """Create a policy resolver backed by the current AuthNZ database."""
    pool = await get_db_pool()
    repo = McpHubRepo(pool)
    try:
        await repo.ensure_tables()
    except Exception as exc:
        logger.debug("MCP Hub policy resolver table ensure failed: {}", exc)
        raise
    return McpHubPolicyResolver(repo=repo)
