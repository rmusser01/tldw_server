from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field

from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.exceptions import TransactionPassthroughError
from tldw_Server_API.app.core.MCP_unified.governance_packs import (
    ApprovalTemplate,
    AssignmentTemplate,
    CapabilityProfile,
    GovernancePack,
    GovernancePackManifest,
    PersonaTemplate,
    build_opa_bundle,
    normalize_governance_pack,
    validate_governance_pack,
)
from tldw_Server_API.app.services.mcp_hub_capability_resolution_service import (
    McpHubCapabilityResolutionService,
)

_RUNTIME_APPROVAL_MODE_MAP = {
    "allow": "allow_silently",
    "ask": "ask_every_time",
}
_SUPPORTED_ENVIRONMENT_REQUIREMENTS = {
    "local_mapping_required",
    "no_external_secrets",
    "workspace_bounded_read",
    "workspace_bounded_write",
}


def _dump_model(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


def _imported_name(display_name: str, *, pack_id: str, source_object_id: str) -> str:
    return f"{display_name} [{pack_id}:{source_object_id}]"


def _imported_upgrade_name(
    display_name: str,
    *,
    pack_id: str,
    pack_version: str,
    source_object_id: str,
) -> str:
    return f"{display_name} [{pack_id}:{source_object_id}@{pack_version}]"


def _normalized_source_metadata(source_metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize prepared source-candidate metadata for governance-pack install persistence."""
    if not isinstance(source_metadata, dict):
        return {}
    return {
        "source_type": str(source_metadata.get("source_type") or "").strip().lower() or None,
        "source_location": str(source_metadata.get("source_location") or "").strip() or None,
        "source_ref_requested": str(source_metadata.get("source_ref_requested") or "").strip() or None,
        "source_ref_kind": str(source_metadata.get("source_ref_kind") or "").strip().lower() or None,
        "source_subpath": str(source_metadata.get("source_subpath") or "").strip() or None,
        "source_commit_resolved": str(source_metadata.get("source_commit_resolved") or "").strip() or None,
        "pack_content_digest": str(source_metadata.get("pack_content_digest") or "").strip() or None,
        "source_verified": source_metadata.get("source_verified"),
        "source_verification_mode": str(source_metadata.get("source_verification_mode") or "").strip() or None,
        "signer_fingerprint": str(source_metadata.get("signer_fingerprint") or "").strip() or None,
        "signer_identity": str(source_metadata.get("signer_identity") or "").strip() or None,
        "verified_object_type": str(source_metadata.get("verified_object_type") or "").strip() or None,
        "verification_result_code": str(source_metadata.get("verification_result_code") or "").strip() or None,
        "verification_warning_code": str(source_metadata.get("verification_warning_code") or "").strip() or None,
        "source_fetched_at": source_metadata.get("source_fetched_at"),
        "fetched_by": source_metadata.get("fetched_by"),
    }


class GovernancePackImportResult(BaseModel):
    governance_pack_id: int
    imported_object_ids: dict[str, list[int]] = Field(default_factory=dict)
    imported_object_counts: dict[str, int] = Field(default_factory=dict)
    blocked_objects: list[str] = Field(default_factory=list)


class GovernancePackDryRunReport(BaseModel):
    manifest: dict[str, Any] = Field(default_factory=dict)
    digest: str
    resolved_capabilities: list[str] = Field(default_factory=list)
    unresolved_capabilities: list[str] = Field(default_factory=list)
    capability_mapping_summary: list[dict[str, Any]] = Field(default_factory=list)
    supported_environment_requirements: list[str] = Field(default_factory=list)
    unsupported_environment_requirements: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    blocked_objects: list[str] = Field(default_factory=list)
    verdict: str


class GovernancePackUpgradePlan(BaseModel):
    source_governance_pack_id: int
    source_manifest: dict[str, Any] = Field(default_factory=dict)
    target_manifest: dict[str, Any] = Field(default_factory=dict)
    object_diff: list[dict[str, Any]] = Field(default_factory=list)
    dependency_impact: list[dict[str, Any]] = Field(default_factory=list)
    structural_conflicts: list[str] = Field(default_factory=list)
    behavioral_conflicts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    planner_inputs_fingerprint: str
    adapter_state_fingerprint: str
    upgradeable: bool


class GovernancePackUpgradeExecutionResult(BaseModel):
    upgrade_id: int
    source_governance_pack_id: int
    target_governance_pack_id: int
    from_pack_version: str
    to_pack_version: str
    planner_inputs_fingerprint: str
    adapter_state_fingerprint: str
    imported_object_ids: dict[str, list[int]] = Field(default_factory=dict)
    imported_object_counts: dict[str, int] = Field(default_factory=dict)


class GovernancePackAlreadyExistsError(TransactionPassthroughError, ValueError):
    """Raised when a governance pack identity already exists in the target scope."""

    def __init__(
        self,
        pack_id: str,
        pack_version: str,
        owner_scope_type: str,
        owner_scope_id: int | None,
    ) -> None:
        scope_label = (
            f"{owner_scope_type}:{owner_scope_id}"
            if owner_scope_id is not None
            else str(owner_scope_type or "global")
        )
        super().__init__(
            f"Governance pack '{pack_id}@{pack_version}' already exists for scope {scope_label}"
        )
        self.pack_id = pack_id
        self.pack_version = pack_version
        self.owner_scope_type = owner_scope_type
        self.owner_scope_id = owner_scope_id


class GovernancePackUpgradeConflictError(ValueError):
    """Raised when an upgrade plan has blocking conflicts."""


class GovernancePackUpgradeStaleError(TransactionPassthroughError, ValueError):
    """Raised when execute-upgrade inputs no longer match current planner state.

    Passthrough so it survives the AuthNZ transaction boundary, which otherwise
    sanitizes to ``TransactionError`` without a ``__cause__`` (d7bab343d3).
    """


def _is_duplicate_governance_pack_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "uq_mcp_governance_packs_scope_version" in message
        or "unique constraint failed: mcp_governance_packs" in message
        or "duplicate key value violates unique constraint" in message
    )


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _governance_pack_ref(document: dict[str, Any]) -> tuple[str, str] | None:
    metadata = dict(document.get("governance_pack") or {})
    pack_id = str(metadata.get("pack_id") or "").strip()
    pack_version = str(metadata.get("pack_version") or "").strip()
    if not pack_id or not pack_version:
        return None
    return pack_id, pack_version


def _source_object_key(object_type: str, item: dict[str, Any]) -> str:
    if object_type == "approval_policy":
        return str(item.get("approval_template_id") or "").strip()
    if object_type == "permission_profile":
        return str(item.get("profile_id") or "").strip()
    if object_type == "policy_assignment":
        return str(item.get("assignment_template_id") or "").strip()
    return ""


def _normalized_runtime_object_maps(normalized_ir: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    data = dict(normalized_ir.get("data") or {})
    object_map: dict[str, dict[str, dict[str, Any]]] = {
        "approval_policy": {},
        "permission_profile": {},
        "policy_assignment": {},
    }
    for object_type, items in (
        ("approval_policy", data.get("approvals") or []),
        ("permission_profile", data.get("profiles") or []),
        ("policy_assignment", data.get("assignments") or []),
    ):
        for raw_item in items:
            item = dict(raw_item or {})
            source_object_id = _source_object_key(object_type, item)
            if source_object_id:
                object_map[object_type][source_object_id] = item
    return object_map


def _normalize_string_values(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple, set)):
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalized.append(cleaned)
    return normalized


class McpHubGovernancePackService:
    """Materialize schema-first governance packs into immutable MCP Hub base objects."""

    def __init__(
        self,
        repo: McpHubRepo,
        capability_resolution_service: McpHubCapabilityResolutionService | None = None,
    ):
        self.repo = repo
        self.capability_resolution_service = capability_resolution_service or McpHubCapabilityResolutionService(
            repo=repo
        )

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            cleaned = str(item or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            out.append(cleaned)
        return out

    @staticmethod
    def _resolution_metadata(
        *,
        owner_scope_type: str,
        owner_scope_id: int | None,
    ) -> dict[str, Any]:
        if owner_scope_type == "team" and owner_scope_id is not None:
            return {"team_id": owner_scope_id}
        if owner_scope_type == "org" and owner_scope_id is not None:
            return {"org_id": owner_scope_id}
        return {}

    @staticmethod
    def _semantic_runtime_object(object_type: str, item: dict[str, Any]) -> dict[str, Any]:
        payload = dict(item or {})
        if object_type == "permission_profile":
            capabilities = dict(payload.get("capabilities") or {})
            return {
                "capabilities": {
                    "allow": _normalize_string_values(capabilities.get("allow")),
                    "deny": _normalize_string_values(capabilities.get("deny")),
                },
                "approval_intent": str(payload.get("approval_intent") or "").strip(),
                "environment_requirements": _normalize_string_values(
                    payload.get("environment_requirements")
                ),
            }
        if object_type == "approval_policy":
            return {
                "mode": str(payload.get("mode") or "").strip().lower(),
            }
        if object_type == "policy_assignment":
            return {
                "target_type": str(payload.get("target_type") or "").strip().lower(),
                "capability_profile_id": str(payload.get("capability_profile_id") or "").strip(),
                "persona_template_id": str(payload.get("persona_template_id") or "").strip(),
                "approval_template_id": str(payload.get("approval_template_id") or "").strip(),
            }
        return payload

    async def _collect_upgrade_dependencies(
        self,
        *,
        governance_pack_id: int,
        owner_scope_type: str,
        owner_scope_id: int | None,
    ) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
        imported_objects = await self.repo.list_governance_pack_objects(governance_pack_id)
        assignments = await self.repo.list_policy_assignments(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        assignment_rows_by_id = {
            str(item.get("id")): item
            for item in assignments
            if item.get("id") is not None
        }
        mutable_assignments = [
            item for item in assignments if not bool(item.get("is_immutable"))
        ]
        mutable_assignments_by_profile: dict[str, list[dict[str, Any]]] = {}
        mutable_assignments_by_approval: dict[str, list[dict[str, Any]]] = {}
        for assignment in mutable_assignments:
            profile_id = assignment.get("profile_id")
            if profile_id is not None:
                mutable_assignments_by_profile.setdefault(str(profile_id), []).append(assignment)
            approval_policy_id = assignment.get("approval_policy_id")
            if approval_policy_id is not None:
                mutable_assignments_by_approval.setdefault(
                    str(approval_policy_id),
                    [],
                ).append(assignment)

        dependencies: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for imported_object in imported_objects:
            object_type = str(imported_object.get("object_type") or "").strip().lower()
            source_object_id = str(imported_object.get("source_object_id") or "").strip()
            object_id = str(imported_object.get("object_id") or "").strip()
            dependents: list[dict[str, Any]] = []
            if object_type == "permission_profile":
                for assignment in mutable_assignments_by_profile.get(object_id, []):
                    dependents.append(
                        {
                            "dependent_type": "policy_assignment",
                            "dependent_id": int(assignment["id"]),
                            "reference_field": "profile_id",
                            "target_type": assignment.get("target_type"),
                            "target_id": assignment.get("target_id"),
                            "conflict_on_removed": True,
                            "conflict_on_modified": True,
                        }
                    )
            elif object_type == "approval_policy":
                for assignment in mutable_assignments_by_approval.get(object_id, []):
                    dependents.append(
                        {
                            "dependent_type": "policy_assignment",
                            "dependent_id": int(assignment["id"]),
                            "reference_field": "approval_policy_id",
                            "target_type": assignment.get("target_type"),
                            "target_id": assignment.get("target_id"),
                            "conflict_on_removed": True,
                            "conflict_on_modified": True,
                        }
                    )
            elif object_type == "policy_assignment":
                assignment = assignment_rows_by_id.get(object_id)
                if assignment and assignment.get("has_override") and assignment.get("override_active"):
                    dependents.append(
                        {
                            "dependent_type": "policy_override",
                            "dependent_id": int(assignment["override_id"]),
                            "reference_field": "assignment_id",
                            "target_type": assignment.get("target_type"),
                            "target_id": assignment.get("target_id"),
                            "conflict_on_removed": True,
                            "conflict_on_modified": True,
                        }
                    )
                if object_id.isdigit():
                    for workspace in await self.repo.list_policy_assignment_workspaces(int(object_id)):
                        dependents.append(
                            {
                                "dependent_type": "policy_assignment_workspace",
                                "dependent_id": int(object_id),
                                "reference_field": "assignment_id",
                                "target_type": assignment.get("target_type") if assignment else None,
                                "target_id": workspace.get("workspace_id"),
                                "conflict_on_removed": True,
                                "conflict_on_modified": False,
                            }
                        )
            dependencies[(object_type, source_object_id)] = dependents
        return imported_objects, dependencies

    @staticmethod
    def _version_compare(source_version: str, target_version: str) -> int | None:
        try:
            source = Version(str(source_version or "").strip())
            target = Version(str(target_version or "").strip())
        except InvalidVersion:
            return None
        if target > source:
            return 1
        if target < source:
            return -1
        return 0

    async def _rollback_import(
        self,
        *,
        governance_pack_id: int | None,
        imported_object_ids: dict[str, list[int]],
    ) -> None:
        for assignment_id in reversed(imported_object_ids.get("policy_assignments", [])):
            try:
                await self.repo.delete_policy_assignment(assignment_id)
            except Exception as exc:
                logger.warning(
                    "Governance pack rollback could not delete policy_assignment {}: {}",
                    assignment_id,
                    exc,
                )
        for profile_id in reversed(imported_object_ids.get("permission_profiles", [])):
            try:
                await self.repo.delete_permission_profile(profile_id)
            except Exception as exc:
                logger.warning(
                    "Governance pack rollback could not delete permission_profile {}: {}",
                    profile_id,
                    exc,
                )
        for approval_policy_id in reversed(imported_object_ids.get("approval_policies", [])):
            try:
                await self.repo.delete_approval_policy(approval_policy_id)
            except Exception as exc:
                logger.warning(
                    "Governance pack rollback could not delete approval_policy {}: {}",
                    approval_policy_id,
                    exc,
                )
        if governance_pack_id is not None:
            try:
                await self.repo.delete_governance_pack(governance_pack_id)
            except Exception as exc:
                logger.warning(
                    "Governance pack rollback could not delete governance_pack {}: {}",
                    governance_pack_id,
                    exc,
                )

    @staticmethod
    def build_pack_from_document(document: dict[str, Any]) -> GovernancePack:
        payload = dict(document or {})
        return GovernancePack(
            source_path=Path("<api>"),
            manifest=GovernancePackManifest.model_validate(payload.get("manifest") or {}),
            profiles=[
                CapabilityProfile.model_validate(item)
                for item in payload.get("profiles") or []
            ],
            approvals=[
                ApprovalTemplate.model_validate(item)
                for item in payload.get("approvals") or []
            ],
            personas=[
                PersonaTemplate.model_validate(item)
                for item in payload.get("personas") or []
            ],
            assignments=[
                AssignmentTemplate.model_validate(item)
                for item in payload.get("assignments") or []
            ],
            raw_profiles=[dict(item) for item in payload.get("profiles") or []],
            raw_approvals=[dict(item) for item in payload.get("approvals") or []],
            raw_personas=[dict(item) for item in payload.get("personas") or []],
            raw_assignments=[dict(item) for item in payload.get("assignments") or []],
        )

    async def dry_run_pack(
        self,
        *,
        pack: GovernancePack,
        owner_scope_type: str,
        owner_scope_id: int | None,
    ) -> GovernancePackDryRunReport:
        validation = validate_governance_pack(pack)
        if validation.errors:
            raise ValueError("; ".join(validation.errors))

        bundle = build_opa_bundle(pack)
        resolved_capabilities: list[str] = []
        unresolved_capabilities: list[str] = []
        capability_mapping_summary: list[dict[str, Any]] = []
        supported_environment_requirements: list[str] = []
        unsupported_environment_requirements: list[str] = []
        warnings: list[str] = []
        blocked_objects: list[str] = []
        seen_mapping_ids: set[str] = set()
        resolution_metadata = self._resolution_metadata(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )

        for profile in pack.profiles:
            allow_resolution = await self.capability_resolution_service.resolve_capabilities(
                capability_names=self._unique(list(profile.capabilities.allow)),
                metadata=resolution_metadata,
                resolution_intent="allow",
            )
            deny_resolution = await self.capability_resolution_service.resolve_capabilities(
                capability_names=self._unique(list(profile.capabilities.deny)),
                metadata=resolution_metadata,
                resolution_intent="deny",
            )
            resolved_capabilities = self._unique(
                resolved_capabilities
                + allow_resolution.resolved_capabilities
                + deny_resolution.resolved_capabilities
            )
            unresolved_capabilities = self._unique(
                unresolved_capabilities
                + allow_resolution.unresolved_capabilities
                + deny_resolution.unresolved_capabilities
            )
            supported_environment_requirements = self._unique(
                supported_environment_requirements
                + allow_resolution.supported_environment_requirements
                + deny_resolution.supported_environment_requirements
            )
            unsupported_environment_requirements = self._unique(
                unsupported_environment_requirements
                + allow_resolution.unsupported_environment_requirements
                + deny_resolution.unsupported_environment_requirements
            )
            for summary in [*allow_resolution.mapping_summaries, *deny_resolution.mapping_summaries]:
                mapping_key = str(summary.get("mapping_id") or summary.get("capability_name") or "").strip()
                if mapping_key and mapping_key in seen_mapping_ids:
                    continue
                if mapping_key:
                    seen_mapping_ids.add(mapping_key)
                capability_mapping_summary.append(dict(summary))
            for warning in [*allow_resolution.warnings, *deny_resolution.warnings]:
                warnings.append(f"profile:{profile.profile_id}: {warning}")
            for requirement in profile.environment_requirements:
                requirement_value = str(requirement or "").strip()
                if not requirement_value:
                    continue
                if requirement_value not in _SUPPORTED_ENVIRONMENT_REQUIREMENTS:
                    warnings.append(
                        f"profile:{profile.profile_id} uses unsupported environment requirement '{requirement_value}'"
                    )
                    continue
                profile_supported_environment_requirements = self._unique(
                    allow_resolution.supported_environment_requirements
                    + deny_resolution.supported_environment_requirements
                )
                if requirement_value not in profile_supported_environment_requirements:
                    warnings.append(
                        f"profile:{profile.profile_id} requires environment requirement "
                        f"'{requirement_value}' but current capability mappings do not guarantee it"
                    )

        for approval in pack.approvals:
            if str(approval.mode or "").strip().lower() not in _RUNTIME_APPROVAL_MODE_MAP:
                blocked_objects.append(f"approval_template:{approval.approval_template_id}")
                warnings.append(
                    f"approval template '{approval.approval_template_id}' cannot map to a local runtime approval mode"
                )

        verdict = "importable"
        if unresolved_capabilities or blocked_objects:
            verdict = "blocked"

        return GovernancePackDryRunReport(
            manifest={
                "pack_id": pack.manifest.pack_id,
                "pack_version": pack.manifest.pack_version,
                "title": pack.manifest.title,
                "description": pack.manifest.description,
            },
            digest=bundle.digest,
            resolved_capabilities=resolved_capabilities,
            unresolved_capabilities=unresolved_capabilities,
            capability_mapping_summary=capability_mapping_summary,
            supported_environment_requirements=supported_environment_requirements,
            unsupported_environment_requirements=unsupported_environment_requirements,
            warnings=self._unique(warnings),
            blocked_objects=blocked_objects,
            verdict=verdict,
        )

    async def dry_run_pack_document(
        self,
        *,
        document: dict[str, Any],
        owner_scope_type: str,
        owner_scope_id: int | None,
    ) -> GovernancePackDryRunReport:
        pack = self.build_pack_from_document(document)
        return await self.dry_run_pack(
            pack=pack,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )

    async def dry_run_upgrade_pack(
        self,
        *,
        source_governance_pack_id: int,
        pack: GovernancePack,
        owner_scope_type: str,
        owner_scope_id: int | None,
    ) -> GovernancePackUpgradePlan:
        validation = validate_governance_pack(pack)
        if validation.errors:
            raise ValueError("; ".join(validation.errors))

        source_pack = await self.repo.get_governance_pack(source_governance_pack_id)
        if source_pack is None:
            raise ValueError(
                f"mcp_governance_pack '{source_governance_pack_id}' was not found"
            )

        source_scope_type = str(source_pack.get("owner_scope_type") or "").strip().lower()
        source_scope_id = source_pack.get("owner_scope_id")
        structural_conflicts: list[str] = []
        behavioral_conflicts: list[str] = []
        warnings: list[str] = []
        object_diff: list[dict[str, Any]] = []
        dependency_impact: list[dict[str, Any]] = []

        if not bool(source_pack.get("is_active_install")):
            structural_conflicts.append(
                "Only active governance pack installs can be upgraded in place"
            )
        if (
            str(owner_scope_type or "").strip().lower() != source_scope_type
            or owner_scope_id != source_scope_id
        ):
            structural_conflicts.append(
                "Target upgrade must use the same owner scope as the installed governance pack"
            )

        target_manifest = {
            "pack_id": pack.manifest.pack_id,
            "pack_version": pack.manifest.pack_version,
            "title": pack.manifest.title,
            "description": pack.manifest.description,
        }
        source_manifest = dict(source_pack.get("manifest") or {})
        if str(pack.manifest.pack_id or "").strip() != str(source_pack.get("pack_id") or "").strip():
            structural_conflicts.append(
                "Target upgrade must keep the same pack_id as the installed governance pack"
            )

        version_compare = self._version_compare(
            str(source_pack.get("pack_version") or ""),
            str(pack.manifest.pack_version or ""),
        )
        if version_compare is None:
            structural_conflicts.append(
                "Governance pack upgrades require semantic versions for both installed and target versions"
            )
        elif version_compare <= 0:
            structural_conflicts.append(
                "Target governance pack version must be newer than the installed version"
            )

        existing_target = await self.repo.get_governance_pack_by_identity(
            pack_id=pack.manifest.pack_id,
            pack_version=pack.manifest.pack_version,
            owner_scope_type=source_scope_type,
            owner_scope_id=source_scope_id,
        )
        if existing_target is not None and int(existing_target["id"]) != int(source_governance_pack_id):
            structural_conflicts.append(
                f"Governance pack '{pack.manifest.pack_id}@{pack.manifest.pack_version}' is already installed in this scope"
            )

        report = await self.dry_run_pack(
            pack=pack,
            owner_scope_type=str(owner_scope_type or "").strip().lower(),
            owner_scope_id=owner_scope_id,
        )
        if report.verdict != "importable":
            structural_conflicts.append(
                "Target governance pack is not importable under the current adapter mapping state"
            )
        warnings = self._unique(
            warnings
            + list(report.warnings)
            + [
                f"unresolved capability:{name}"
                for name in report.unresolved_capabilities
            ]
            + [f"blocked object:{name}" for name in report.blocked_objects]
        )

        source_object_maps = _normalized_runtime_object_maps(
            dict(source_pack.get("normalized_ir") or {})
        )
        target_ir = normalize_governance_pack(pack).to_dict()
        target_object_maps = _normalized_runtime_object_maps(target_ir)

        _, dependencies_by_object = await self._collect_upgrade_dependencies(
            governance_pack_id=source_governance_pack_id,
            owner_scope_type=source_scope_type,
            owner_scope_id=source_scope_id,
        )

        for object_type in ("approval_policy", "permission_profile", "policy_assignment"):
            source_items = source_object_maps.get(object_type, {})
            target_items = target_object_maps.get(object_type, {})
            for source_object_id in sorted(set(source_items) | set(target_items)):
                source_item = source_items.get(source_object_id)
                target_item = target_items.get(source_object_id)
                change_type = "unchanged"
                if source_item is None and target_item is not None:
                    change_type = "added"
                elif source_item is not None and target_item is None:
                    change_type = "removed"
                elif (
                    source_item is not None
                    and target_item is not None
                    and self._semantic_runtime_object(object_type, source_item)
                    != self._semantic_runtime_object(object_type, target_item)
                ):
                    change_type = "modified"

                if change_type == "unchanged":
                    continue

                object_diff.append(
                    {
                        "object_type": object_type,
                        "source_object_id": source_object_id,
                        "change_type": change_type,
                        "previous_digest": (
                            _stable_digest(self._semantic_runtime_object(object_type, source_item))
                            if source_item is not None
                            else None
                        ),
                        "next_digest": (
                            _stable_digest(self._semantic_runtime_object(object_type, target_item))
                            if target_item is not None
                            else None
                        ),
                    }
                )

                dependents = dependencies_by_object.get((object_type, source_object_id), [])
                if not dependents:
                    continue
                for dependent in dependents:
                    public_dependent = {
                        key: value
                        for key, value in dependent.items()
                        if key not in {"conflict_on_removed", "conflict_on_modified"}
                    }
                    conflict_on_removed = bool(dependent.get("conflict_on_removed", True))
                    conflict_on_modified = bool(dependent.get("conflict_on_modified", True))
                    if change_type == "removed":
                        impact = "structural_conflict" if conflict_on_removed else "rebind_required"
                    else:
                        impact = "behavioral_conflict" if conflict_on_modified else "rebind_required"
                    dependency_impact.append(
                        {
                            "object_type": object_type,
                            "source_object_id": source_object_id,
                            "change_type": change_type,
                            "impact": impact,
                            **public_dependent,
                        }
                    )
                    if change_type == "removed" and conflict_on_removed:
                        structural_conflicts.append(
                            f"{object_type}:{source_object_id} is removed but dependent "
                            f"{dependent['dependent_type'].replace('_', ' ')} {dependent['dependent_id']} "
                            f"still references it via {dependent['reference_field']}"
                        )
                    elif change_type != "removed" and conflict_on_modified:
                        behavioral_conflicts.append(
                            f"{object_type}:{source_object_id} materially changes while dependent "
                            f"{dependent['dependent_type'].replace('_', ' ')} {dependent['dependent_id']} "
                            f"still references it via {dependent['reference_field']}"
                        )

        planner_inputs_fingerprint = _stable_digest(
            {
                "source_governance_pack_id": int(source_governance_pack_id),
                "source_manifest": source_manifest,
                "target_manifest": target_manifest,
                "source_scope_type": source_scope_type,
                "source_scope_id": source_scope_id,
                "requested_scope_type": str(owner_scope_type or "").strip().lower(),
                "requested_scope_id": owner_scope_id,
                "object_diff": object_diff,
                "dependency_impact": dependency_impact,
                "dependency_snapshot": {
                    f"{object_type}:{source_object_id}": dependents
                    for (object_type, source_object_id), dependents in sorted(
                        dependencies_by_object.items()
                    )
                },
            }
        )
        adapter_state_fingerprint = _stable_digest(
            {
                "digest": report.digest,
                "resolved_capabilities": report.resolved_capabilities,
                "unresolved_capabilities": report.unresolved_capabilities,
                "capability_mapping_summary": report.capability_mapping_summary,
                "supported_environment_requirements": report.supported_environment_requirements,
                "unsupported_environment_requirements": report.unsupported_environment_requirements,
                "warnings": report.warnings,
                "blocked_objects": report.blocked_objects,
                "verdict": report.verdict,
            }
        )

        return GovernancePackUpgradePlan(
            source_governance_pack_id=int(source_governance_pack_id),
            source_manifest=source_manifest,
            target_manifest=target_manifest,
            object_diff=object_diff,
            dependency_impact=dependency_impact,
            structural_conflicts=self._unique(structural_conflicts),
            behavioral_conflicts=self._unique(behavioral_conflicts),
            warnings=self._unique(warnings),
            planner_inputs_fingerprint=planner_inputs_fingerprint,
            adapter_state_fingerprint=adapter_state_fingerprint,
            upgradeable=not structural_conflicts and not behavioral_conflicts,
        )

    async def dry_run_upgrade_document(
        self,
        *,
        source_governance_pack_id: int,
        document: dict[str, Any],
        owner_scope_type: str,
        owner_scope_id: int | None,
    ) -> GovernancePackUpgradePlan:
        pack = self.build_pack_from_document(document)
        return await self.dry_run_upgrade_pack(
            source_governance_pack_id=source_governance_pack_id,
            pack=pack,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )

    async def _stage_upgrade_import(
        self,
        *,
        conn: Any,
        pack: GovernancePack,
        owner_scope_type: str,
        owner_scope_id: int | None,
        actor_id: int | None,
    ) -> tuple[GovernancePackImportResult, dict[str, dict[str, int]]]:
        normalized_ir = normalize_governance_pack(pack)
        bundle = build_opa_bundle(pack)
        manifest = pack.manifest
        pack_row = await self.repo.create_governance_pack(
            pack_id=manifest.pack_id,
            pack_version=manifest.pack_version,
            pack_schema_version=manifest.pack_schema_version,
            capability_taxonomy_version=manifest.capability_taxonomy_version,
            adapter_contract_version=manifest.adapter_contract_version,
            title=manifest.title,
            description=manifest.description,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            bundle_digest=bundle.digest,
            manifest=_dump_model(manifest),
            normalized_ir=normalized_ir.to_dict(),
            actor_id=actor_id,
            is_active_install=False,
            source_type=str(getattr(pack, "source_type", "") or "").strip().lower() or None,
            source_location=str(getattr(pack, "source_location", "") or "").strip() or None,
            source_ref_requested=str(getattr(pack, "source_ref_requested", "") or "").strip() or None,
            source_ref_kind=str(getattr(pack, "source_ref_kind", "") or "").strip().lower() or None,
            source_subpath=str(getattr(pack, "source_subpath", "") or "").strip() or None,
            source_commit_resolved=str(getattr(pack, "source_commit_resolved", "") or "").strip() or None,
            pack_content_digest=str(getattr(pack, "pack_content_digest", "") or "").strip() or None,
            source_verified=getattr(pack, "source_verified", None),
            source_verification_mode=str(getattr(pack, "source_verification_mode", "") or "").strip() or None,
            signer_fingerprint=str(getattr(pack, "signer_fingerprint", "") or "").strip() or None,
            signer_identity=str(getattr(pack, "signer_identity", "") or "").strip() or None,
            verified_object_type=str(getattr(pack, "verified_object_type", "") or "").strip() or None,
            verification_result_code=str(getattr(pack, "verification_result_code", "") or "").strip() or None,
            verification_warning_code=str(getattr(pack, "verification_warning_code", "") or "").strip() or None,
            conn=conn,
        )
        if not pack_row:
            raise RuntimeError("Failed to stage governance pack upgrade manifest")

        governance_pack_id = int(pack_row["id"])
        imported_object_ids: dict[str, list[int]] = {
            "approval_policies": [],
            "permission_profiles": [],
            "policy_assignments": [],
        }
        runtime_id_map: dict[str, dict[str, int]] = {
            "approval_policy": {},
            "permission_profile": {},
            "policy_assignment": {},
        }
        approval_policy_ids_by_template: dict[str, int] = {}
        profile_ids_by_template: dict[str, int] = {}
        persona_ids_by_template: dict[str, str] = {
            persona.persona_template_id: persona.persona_template_id
            for persona in pack.personas
        }
        persona_approval_ids: dict[str, str] = {
            persona.persona_template_id: persona.approval_template_id
            for persona in pack.personas
        }

        for approval in pack.approvals:
            runtime_mode = _RUNTIME_APPROVAL_MODE_MAP.get(str(approval.mode or "").strip().lower())
            if runtime_mode is None:
                raise RuntimeError(
                    f"approval template '{approval.approval_template_id}' cannot map to a local runtime approval mode"
                )
            created = await self.repo.create_approval_policy(
                name=_imported_upgrade_name(
                    approval.name,
                    pack_id=manifest.pack_id,
                    pack_version=manifest.pack_version,
                    source_object_id=approval.approval_template_id,
                ),
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
                mode=runtime_mode,
                rules={
                    "portable_mode": approval.mode,
                    "approval_template_id": approval.approval_template_id,
                    "governance_pack": {
                        "pack_id": manifest.pack_id,
                        "pack_version": manifest.pack_version,
                        "source_object_id": approval.approval_template_id,
                    },
                },
                actor_id=actor_id,
                description=approval.name,
                is_active=True,
                is_immutable=True,
                conn=conn,
            )
            approval_id = int(created["id"])
            approval_policy_ids_by_template[approval.approval_template_id] = approval_id
            runtime_id_map["approval_policy"][approval.approval_template_id] = approval_id
            imported_object_ids["approval_policies"].append(approval_id)
            await self.repo.create_governance_pack_object(
                governance_pack_id=governance_pack_id,
                object_type="approval_policy",
                object_id=approval_id,
                source_object_id=approval.approval_template_id,
                conn=conn,
            )

        for profile in pack.profiles:
            created = await self.repo.create_permission_profile(
                name=_imported_upgrade_name(
                    profile.name,
                    pack_id=manifest.pack_id,
                    pack_version=manifest.pack_version,
                    source_object_id=profile.profile_id,
                ),
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
                mode="preset",
                policy_document={
                    "capabilities": list(profile.capabilities.allow),
                    "denied_capabilities": list(profile.capabilities.deny),
                    "environment_requirements": list(profile.environment_requirements),
                    "approval_intent": profile.approval_intent,
                    "governance_pack": {
                        "pack_id": manifest.pack_id,
                        "pack_version": manifest.pack_version,
                        "source_object_id": profile.profile_id,
                    },
                },
                actor_id=actor_id,
                description=profile.description,
                is_active=True,
                is_immutable=True,
                conn=conn,
            )
            profile_id = int(created["id"])
            profile_ids_by_template[profile.profile_id] = profile_id
            runtime_id_map["permission_profile"][profile.profile_id] = profile_id
            imported_object_ids["permission_profiles"].append(profile_id)
            await self.repo.create_governance_pack_object(
                governance_pack_id=governance_pack_id,
                object_type="permission_profile",
                object_id=profile_id,
                source_object_id=profile.profile_id,
                conn=conn,
            )

        for assignment in pack.assignments:
            profile_id = profile_ids_by_template.get(assignment.capability_profile_id)
            if profile_id is None:
                raise RuntimeError(
                    f"assignment template '{assignment.assignment_template_id}' has no staged capability profile"
                )

            approval_template_id = assignment.approval_template_id
            if approval_template_id is None and assignment.persona_template_id is not None:
                approval_template_id = persona_approval_ids.get(assignment.persona_template_id)
            approval_policy_id = (
                approval_policy_ids_by_template.get(approval_template_id)
                if approval_template_id is not None
                else None
            )
            if approval_template_id is not None and approval_policy_id is None:
                raise RuntimeError(
                    f"assignment template '{assignment.assignment_template_id}' has no staged approval policy"
                )

            target_id = None
            if assignment.target_type == "persona" and assignment.persona_template_id is not None:
                target_id = persona_ids_by_template.get(assignment.persona_template_id)

            created = await self.repo.create_policy_assignment(
                target_type=assignment.target_type,
                target_id=target_id,
                owner_scope_type=owner_scope_type,
                owner_scope_id=owner_scope_id,
                profile_id=profile_id,
                inline_policy_document={
                    "persona_template_id": assignment.persona_template_id,
                    "approval_template_id": approval_template_id,
                    "capability_profile_id": assignment.capability_profile_id,
                    "governance_pack": {
                        "pack_id": manifest.pack_id,
                        "pack_version": manifest.pack_version,
                        "source_object_id": assignment.assignment_template_id,
                    },
                },
                approval_policy_id=approval_policy_id,
                actor_id=actor_id,
                is_active=True,
                is_immutable=True,
                conn=conn,
            )
            assignment_id = int(created["id"])
            runtime_id_map["policy_assignment"][assignment.assignment_template_id] = assignment_id
            imported_object_ids["policy_assignments"].append(assignment_id)
            await self.repo.create_governance_pack_object(
                governance_pack_id=governance_pack_id,
                object_type="policy_assignment",
                object_id=assignment_id,
                source_object_id=assignment.assignment_template_id,
                conn=conn,
            )

        return (
            GovernancePackImportResult(
                governance_pack_id=governance_pack_id,
                imported_object_ids=imported_object_ids,
                imported_object_counts={
                    key: len(value) for key, value in imported_object_ids.items()
                },
                blocked_objects=[],
            ),
            runtime_id_map,
        )

    async def execute_upgrade_pack(
        self,
        *,
        source_governance_pack_id: int,
        pack: GovernancePack,
        owner_scope_type: str,
        owner_scope_id: int | None,
        actor_id: int | None,
        planner_inputs_fingerprint: str,
        adapter_state_fingerprint: str,
    ) -> GovernancePackUpgradeExecutionResult:
        plan = await self.dry_run_upgrade_pack(
            source_governance_pack_id=source_governance_pack_id,
            pack=pack,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        if (
            str(planner_inputs_fingerprint).strip() != plan.planner_inputs_fingerprint
            or str(adapter_state_fingerprint).strip() != plan.adapter_state_fingerprint
        ):
            raise GovernancePackUpgradeStaleError(
                "Governance pack upgrade plan is stale; rerun dry-run-upgrade"
            )
        if not plan.upgradeable:
            raise GovernancePackUpgradeConflictError(
                "Governance pack upgrade has blocking conflicts"
            )

        source_pack = await self.repo.get_governance_pack(source_governance_pack_id)
        if source_pack is None:
            raise ValueError(
                f"mcp_governance_pack '{source_governance_pack_id}' was not found"
            )

        source_scope_type = str(source_pack.get("owner_scope_type") or "").strip().lower()
        source_scope_id = source_pack.get("owner_scope_id")
        imported_objects, dependencies_by_object = await self._collect_upgrade_dependencies(
            governance_pack_id=source_governance_pack_id,
            owner_scope_type=source_scope_type,
            owner_scope_id=source_scope_id,
        )

        try:
            async with self.repo.db_pool.transaction() as conn:
                source_pack_tx = await self.repo.get_governance_pack(
                    source_governance_pack_id,
                    conn=conn,
                )
                if source_pack_tx is None or not bool(source_pack_tx.get("is_active_install")):
                    raise GovernancePackUpgradeStaleError(
                        "Governance pack upgrade source install is no longer active"
                    )
                existing_target = await self.repo.get_governance_pack_by_identity(
                    pack_id=pack.manifest.pack_id,
                    pack_version=pack.manifest.pack_version,
                    owner_scope_type=source_scope_type,
                    owner_scope_id=source_scope_id,
                    conn=conn,
                )
                if existing_target is not None:
                    raise GovernancePackAlreadyExistsError(
                        pack.manifest.pack_id,
                        pack.manifest.pack_version,
                        source_scope_type,
                        source_scope_id,
                    )

                staged_import, runtime_id_map = await self._stage_upgrade_import(
                    conn=conn,
                    pack=pack,
                    owner_scope_type=source_scope_type,
                    owner_scope_id=source_scope_id,
                    actor_id=actor_id,
                )
                for imported_object in imported_objects:
                    object_type = str(imported_object.get("object_type") or "").strip().lower()
                    source_object_id = str(imported_object.get("source_object_id") or "").strip()
                    old_object_id = int(imported_object["object_id"])
                    new_object_id = runtime_id_map.get(object_type, {}).get(source_object_id)
                    if new_object_id is None:
                        continue
                    dependents = dependencies_by_object.get((object_type, source_object_id), [])
                    if object_type == "permission_profile":
                        for dependent in dependents:
                            if dependent["dependent_type"] != "policy_assignment":
                                continue
                            await self.repo.update_policy_assignment(
                                int(dependent["dependent_id"]),
                                profile_id=new_object_id,
                                actor_id=actor_id,
                                conn=conn,
                            )
                    elif object_type == "approval_policy":
                        for dependent in dependents:
                            if dependent["dependent_type"] != "policy_assignment":
                                continue
                            await self.repo.update_policy_assignment(
                                int(dependent["dependent_id"]),
                                approval_policy_id=new_object_id,
                                actor_id=actor_id,
                                conn=conn,
                            )
                    elif object_type == "policy_assignment":
                        await self.repo.rebind_policy_assignment_workspaces(
                            old_assignment_id=old_object_id,
                            new_assignment_id=new_object_id,
                            conn=conn,
                        )
                        for dependent in dependents:
                            if dependent["dependent_type"] != "policy_override":
                                continue
                            await self.repo.rebind_policy_override_assignment(
                                old_assignment_id=old_object_id,
                                new_assignment_id=new_object_id,
                                actor_id=actor_id,
                                conn=conn,
                            )

                executed_at = datetime.now(timezone.utc)
                upgrade_row = await self.repo.create_governance_pack_upgrade(
                    pack_id=str(source_pack_tx.get("pack_id") or ""),
                    owner_scope_type=source_scope_type,
                    owner_scope_id=source_scope_id,
                    from_governance_pack_id=int(source_governance_pack_id),
                    to_governance_pack_id=int(staged_import.governance_pack_id),
                    from_pack_version=str(source_pack_tx.get("pack_version") or ""),
                    to_pack_version=pack.manifest.pack_version,
                    status="executed",
                    planned_by=actor_id,
                    executed_by=actor_id,
                    planner_inputs_fingerprint=plan.planner_inputs_fingerprint,
                    adapter_state_fingerprint=plan.adapter_state_fingerprint,
                    plan_summary={
                        "object_diff_count": len(plan.object_diff),
                        "dependency_impact_count": len(plan.dependency_impact),
                    },
                    accepted_resolutions={},
                    executed_at=executed_at,
                    conn=conn,
                )
                if not upgrade_row:
                    raise RuntimeError("Failed to persist governance pack upgrade lineage")

                await self.repo.update_governance_pack_install_state(
                    int(source_governance_pack_id),
                    is_active_install=False,
                    superseded_by_governance_pack_id=int(staged_import.governance_pack_id),
                    actor_id=actor_id,
                    conn=conn,
                )
                await self.repo.update_governance_pack_install_state(
                    int(staged_import.governance_pack_id),
                    is_active_install=True,
                    installed_from_upgrade_id=int(upgrade_row["id"]),
                    actor_id=actor_id,
                    conn=conn,
                )
        except TransactionError as exc:
            if exc.__cause__ is not None:
                raise exc.__cause__
            raise

        return GovernancePackUpgradeExecutionResult(
            upgrade_id=int(upgrade_row["id"]),
            source_governance_pack_id=int(source_governance_pack_id),
            target_governance_pack_id=int(staged_import.governance_pack_id),
            from_pack_version=str(source_pack.get("pack_version") or ""),
            to_pack_version=pack.manifest.pack_version,
            planner_inputs_fingerprint=plan.planner_inputs_fingerprint,
            adapter_state_fingerprint=plan.adapter_state_fingerprint,
            imported_object_ids=staged_import.imported_object_ids,
            imported_object_counts=staged_import.imported_object_counts,
        )

    async def execute_upgrade_document(
        self,
        *,
        source_governance_pack_id: int,
        document: dict[str, Any],
        owner_scope_type: str,
        owner_scope_id: int | None,
        actor_id: int | None,
        planner_inputs_fingerprint: str,
        adapter_state_fingerprint: str,
    ) -> GovernancePackUpgradeExecutionResult:
        pack = self.build_pack_from_document(document)
        return await self.execute_upgrade_pack(
            source_governance_pack_id=source_governance_pack_id,
            pack=pack,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            actor_id=actor_id,
            planner_inputs_fingerprint=planner_inputs_fingerprint,
            adapter_state_fingerprint=adapter_state_fingerprint,
        )

    async def import_pack_document(
        self,
        *,
        document: dict[str, Any],
        owner_scope_type: str,
        owner_scope_id: int | None,
        actor_id: int | None,
        source_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pack = self.build_pack_from_document(document)
        report = await self.dry_run_pack(
            pack=pack,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        if report.verdict != "importable":
            raise ValueError("Governance pack dry-run did not pass")
        imported = await self.import_pack(
            pack=pack,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
            actor_id=actor_id,
            source_metadata=source_metadata,
        )
        return {
            "governance_pack_id": imported.governance_pack_id,
            "imported_object_counts": dict(imported.imported_object_counts),
            "blocked_objects": list(imported.blocked_objects),
            "report": report.model_dump(),
        }

    async def list_governance_packs(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict[str, Any]]:
        return await self.repo.list_governance_packs(
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )

    async def get_governance_pack_detail(self, governance_pack_id: int) -> dict[str, Any] | None:
        pack_row = await self.repo.get_governance_pack(governance_pack_id)
        if pack_row is None:
            return None
        return {
            **pack_row,
            "imported_objects": await self.repo.list_governance_pack_objects(governance_pack_id),
        }

    async def list_governance_pack_upgrade_history(
        self,
        governance_pack_id: int,
    ) -> list[dict[str, Any]]:
        pack_row = await self.repo.get_governance_pack(governance_pack_id)
        if pack_row is None:
            return []
        return await self.repo.list_governance_pack_upgrades(
            pack_id=str(pack_row.get("pack_id") or ""),
            owner_scope_type=str(pack_row.get("owner_scope_type") or "global"),
            owner_scope_id=pack_row.get("owner_scope_id"),
        )

    async def import_pack(
        self,
        *,
        pack: GovernancePack,
        owner_scope_type: str,
        owner_scope_id: int | None,
        actor_id: int | None,
        source_metadata: dict[str, Any] | None = None,
    ) -> GovernancePackImportResult:
        validation = validate_governance_pack(pack)
        if validation.errors:
            raise ValueError("; ".join(validation.errors))

        normalized_ir = normalize_governance_pack(pack)
        bundle = build_opa_bundle(pack)
        manifest = pack.manifest
        existing = await self.repo.get_governance_pack_by_identity(
            pack_id=manifest.pack_id,
            pack_version=manifest.pack_version,
            owner_scope_type=owner_scope_type,
            owner_scope_id=owner_scope_id,
        )
        if existing is not None:
            raise GovernancePackAlreadyExistsError(
                manifest.pack_id,
                manifest.pack_version,
                owner_scope_type,
                owner_scope_id,
            )

        blocked_objects: list[str] = []
        imported_object_ids: dict[str, list[int]] = {
            "approval_policies": [],
            "permission_profiles": [],
            "policy_assignments": [],
        }
        governance_pack_id: int | None = None
        normalized_source_metadata = _normalized_source_metadata(source_metadata)
        approval_policy_ids_by_template: dict[str, int] = {}
        profile_ids_by_template: dict[str, int] = {}
        persona_ids_by_template: dict[str, str] = {
            persona.persona_template_id: persona.persona_template_id
            for persona in pack.personas
        }
        persona_approval_ids: dict[str, str] = {
            persona.persona_template_id: persona.approval_template_id
            for persona in pack.personas
        }

        try:
            try:
                pack_row = await self.repo.create_governance_pack(
                    pack_id=manifest.pack_id,
                    pack_version=manifest.pack_version,
                    pack_schema_version=manifest.pack_schema_version,
                    capability_taxonomy_version=manifest.capability_taxonomy_version,
                    adapter_contract_version=manifest.adapter_contract_version,
                    title=manifest.title,
                    description=manifest.description,
                    owner_scope_type=owner_scope_type,
                    owner_scope_id=owner_scope_id,
                    bundle_digest=bundle.digest,
                    manifest=_dump_model(manifest),
                    normalized_ir=normalized_ir.to_dict(),
                    actor_id=actor_id,
                    source_type=normalized_source_metadata.get("source_type"),
                    source_location=normalized_source_metadata.get("source_location"),
                    source_ref_requested=normalized_source_metadata.get("source_ref_requested"),
                    source_ref_kind=normalized_source_metadata.get("source_ref_kind"),
                    source_subpath=normalized_source_metadata.get("source_subpath"),
                    source_commit_resolved=normalized_source_metadata.get("source_commit_resolved"),
                    pack_content_digest=normalized_source_metadata.get("pack_content_digest"),
                    source_verified=normalized_source_metadata.get("source_verified"),
                    source_verification_mode=normalized_source_metadata.get("source_verification_mode"),
                    signer_fingerprint=normalized_source_metadata.get("signer_fingerprint"),
                    signer_identity=normalized_source_metadata.get("signer_identity"),
                    verified_object_type=normalized_source_metadata.get("verified_object_type"),
                    verification_result_code=normalized_source_metadata.get("verification_result_code"),
                    verification_warning_code=normalized_source_metadata.get("verification_warning_code"),
                    source_fetched_at=normalized_source_metadata.get("source_fetched_at"),
                    fetched_by=normalized_source_metadata.get("fetched_by"),
                )
            except Exception as exc:
                if _is_duplicate_governance_pack_error(exc):
                    raise GovernancePackAlreadyExistsError(
                        manifest.pack_id,
                        manifest.pack_version,
                        owner_scope_type,
                        owner_scope_id,
                    ) from exc
                raise
            if not pack_row:
                raise RuntimeError("Failed to persist governance pack manifest")
            governance_pack_id = int(pack_row["id"])

            for approval in pack.approvals:
                runtime_mode = _RUNTIME_APPROVAL_MODE_MAP.get(str(approval.mode or "").strip().lower())
                if runtime_mode is None:
                    blocked_objects.append(f"approval_template:{approval.approval_template_id}")
                    continue
                created = await self.repo.create_approval_policy(
                    name=_imported_name(
                        approval.name,
                        pack_id=manifest.pack_id,
                        source_object_id=approval.approval_template_id,
                    ),
                    owner_scope_type=owner_scope_type,
                    owner_scope_id=owner_scope_id,
                    mode=runtime_mode,
                    rules={
                        "portable_mode": approval.mode,
                        "approval_template_id": approval.approval_template_id,
                        "governance_pack": {
                            "pack_id": manifest.pack_id,
                            "pack_version": manifest.pack_version,
                            "source_object_id": approval.approval_template_id,
                        },
                    },
                    actor_id=actor_id,
                    description=approval.name,
                    is_active=True,
                    is_immutable=True,
                )
                approval_id = int(created["id"])
                approval_policy_ids_by_template[approval.approval_template_id] = approval_id
                imported_object_ids["approval_policies"].append(approval_id)
                await self.repo.create_governance_pack_object(
                    governance_pack_id=governance_pack_id,
                    object_type="approval_policy",
                    object_id=approval_id,
                    source_object_id=approval.approval_template_id,
                )

            for profile in pack.profiles:
                created = await self.repo.create_permission_profile(
                    name=_imported_name(
                        profile.name,
                        pack_id=manifest.pack_id,
                        source_object_id=profile.profile_id,
                    ),
                    owner_scope_type=owner_scope_type,
                    owner_scope_id=owner_scope_id,
                    mode="preset",
                    policy_document={
                        "capabilities": list(profile.capabilities.allow),
                        "denied_capabilities": list(profile.capabilities.deny),
                        "environment_requirements": list(profile.environment_requirements),
                        "approval_intent": profile.approval_intent,
                        "governance_pack": {
                            "pack_id": manifest.pack_id,
                            "pack_version": manifest.pack_version,
                            "source_object_id": profile.profile_id,
                        },
                    },
                    actor_id=actor_id,
                    description=profile.description,
                    is_active=True,
                    is_immutable=True,
                )
                profile_id = int(created["id"])
                profile_ids_by_template[profile.profile_id] = profile_id
                imported_object_ids["permission_profiles"].append(profile_id)
                await self.repo.create_governance_pack_object(
                    governance_pack_id=governance_pack_id,
                    object_type="permission_profile",
                    object_id=profile_id,
                    source_object_id=profile.profile_id,
                )

            for assignment in pack.assignments:
                profile_id = profile_ids_by_template.get(assignment.capability_profile_id)
                if profile_id is None:
                    blocked_objects.append(f"assignment_template:{assignment.assignment_template_id}")
                    continue

                approval_template_id = assignment.approval_template_id
                if approval_template_id is None and assignment.persona_template_id is not None:
                    approval_template_id = persona_approval_ids.get(assignment.persona_template_id)
                approval_policy_id = (
                    approval_policy_ids_by_template.get(approval_template_id)
                    if approval_template_id is not None
                    else None
                )
                if approval_template_id is not None and approval_policy_id is None:
                    blocked_objects.append(f"assignment_template:{assignment.assignment_template_id}")
                    continue

                target_id = None
                if assignment.target_type == "persona" and assignment.persona_template_id is not None:
                    target_id = persona_ids_by_template.get(assignment.persona_template_id)

                created = await self.repo.create_policy_assignment(
                    target_type=assignment.target_type,
                    target_id=target_id,
                    owner_scope_type=owner_scope_type,
                    owner_scope_id=owner_scope_id,
                    profile_id=profile_id,
                    inline_policy_document={
                        "persona_template_id": assignment.persona_template_id,
                        "approval_template_id": approval_template_id,
                        "capability_profile_id": assignment.capability_profile_id,
                        "governance_pack": {
                            "pack_id": manifest.pack_id,
                            "pack_version": manifest.pack_version,
                            "source_object_id": assignment.assignment_template_id,
                        },
                    },
                    approval_policy_id=approval_policy_id,
                    actor_id=actor_id,
                    is_active=True,
                    is_immutable=True,
                )
                assignment_id = int(created["id"])
                imported_object_ids["policy_assignments"].append(assignment_id)
                await self.repo.create_governance_pack_object(
                    governance_pack_id=governance_pack_id,
                    object_type="policy_assignment",
                    object_id=assignment_id,
                    source_object_id=assignment.assignment_template_id,
                )
        except Exception:
            await self._rollback_import(
                governance_pack_id=governance_pack_id,
                imported_object_ids=imported_object_ids,
            )
            raise

        return GovernancePackImportResult(
            governance_pack_id=governance_pack_id,
            imported_object_ids=imported_object_ids,
            imported_object_counts={
                key: len(value) for key, value in imported_object_ids.items()
            },
            blocked_objects=blocked_objects,
        )
