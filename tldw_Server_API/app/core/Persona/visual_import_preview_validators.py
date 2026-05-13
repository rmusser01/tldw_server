"""Fixture-level import-preview validation for Persona Visual renderer metadata.

The helpers in this module operate on already-normalized manifest and asset
metadata. They do not parse archives, write asset rows, activate packs, or load
renderer runtimes. The goal is to give future Manifest V2 archive and UI slices
a deterministic diagnostics contract that is backed by the backend renderer
capability registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tldw_Server_API.app.core.Persona.visual_renderer_capabilities import (
    PersonaVisualRendererCapability,
    get_persona_visual_renderer_capability,
)


@dataclass(frozen=True)
class PersonaVisualImportPreviewAsset:
    """Normalized asset metadata available to renderer import-preview checks."""

    source_asset_id: str
    asset_role: str
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class PersonaVisualRendererImportPreviewResult:
    """Structured diagnostics for a renderer import-preview validation pass."""

    status: str
    renderer_type: str
    manifest_version: int | None
    renderer_contract_version: int | None
    can_commit: bool
    activation_eligible: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    normalized_role_categories: dict[str, list[str]] = field(default_factory=dict)
    setup_status: str | None = None
    setup_blockers: list[str] = field(default_factory=list)
    disabled_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the preview result."""

        return {
            "status": self.status,
            "renderer_type": self.renderer_type,
            "manifest_version": self.manifest_version,
            "renderer_contract_version": self.renderer_contract_version,
            "can_commit": self.can_commit,
            "activation_eligible": self.activation_eligible,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "normalized_role_categories": {
                category: list(source_asset_ids)
                for category, source_asset_ids in self.normalized_role_categories.items()
            },
            "setup_status": self.setup_status,
            "setup_blockers": list(self.setup_blockers),
            "disabled_reason": self.disabled_reason,
        }


def preview_renderer_import(
    *,
    manifest: Mapping[str, Any],
    assets: Sequence[PersonaVisualImportPreviewAsset | Mapping[str, Any]],
) -> PersonaVisualRendererImportPreviewResult:
    """Validate renderer metadata for an import preview without committing assets."""

    renderer_type = str(manifest.get("renderer_type") or "")
    manifest_version = _coerce_int(manifest.get("manifest_version"))
    renderer_contract_version = _coerce_int(manifest.get("renderer_contract_version"))
    capability = get_persona_visual_renderer_capability(renderer_type)
    if capability is None:
        return PersonaVisualRendererImportPreviewResult(
            status="unsupported_renderer",
            renderer_type=renderer_type,
            manifest_version=manifest_version,
            renderer_contract_version=renderer_contract_version,
            can_commit=False,
            activation_eligible=False,
            blockers=[f"unknown_renderer:{renderer_type or 'missing'}"],
            normalized_role_categories=_normalize_role_categories(None, assets),
        )

    normalized_role_categories = _normalize_role_categories(capability, assets)
    blockers = _capability_blockers(
        capability=capability,
        manifest_version=manifest_version,
        renderer_contract_version=renderer_contract_version,
        normalized_role_categories=normalized_role_categories,
    )
    status = _preview_status(capability, blockers)
    can_commit = capability.import_supported and status == "supported" and not blockers
    activation_eligible = (
        capability.can_activate
        and capability.buddy_runtime_supported
        and status == "supported"
        and not blockers
    )
    return PersonaVisualRendererImportPreviewResult(
        status=status,
        renderer_type=renderer_type,
        manifest_version=manifest_version,
        renderer_contract_version=renderer_contract_version,
        can_commit=can_commit,
        activation_eligible=activation_eligible,
        blockers=blockers,
        normalized_role_categories=normalized_role_categories,
        setup_status=capability.setup_status,
        setup_blockers=list(capability.setup_blockers),
        disabled_reason=capability.disabled_reason,
    )


def _capability_blockers(
    *,
    capability: PersonaVisualRendererCapability,
    manifest_version: int | None,
    renderer_contract_version: int | None,
    normalized_role_categories: Mapping[str, Sequence[str]],
) -> list[str]:
    blockers: list[str] = []
    for blocker in capability.setup_blockers:
        _append_unique(blockers, blocker)
    if capability.disabled_reason:
        _append_unique(blockers, capability.disabled_reason)
    if manifest_version is not None and manifest_version not in capability.manifest_versions:
        _append_unique(blockers, f"unsupported_manifest_version:{manifest_version}")
    if (
        renderer_contract_version is not None
        and capability.renderer_contract_versions
        and renderer_contract_version not in capability.renderer_contract_versions
    ):
        _append_unique(
            blockers,
            f"unsupported_renderer_contract_version:{renderer_contract_version}",
        )
    for category in capability.required_role_categories:
        if not normalized_role_categories.get(category):
            _append_unique(blockers, f"missing_required_role_category:{category}")
    return blockers


def _preview_status(
    capability: PersonaVisualRendererCapability,
    blockers: Sequence[str],
) -> str:
    if capability.setup_status != "supported":
        return capability.setup_status
    if blockers:
        return "invalid_renderer_assets"
    return "supported"


def _normalize_role_categories(
    capability: PersonaVisualRendererCapability | None,
    assets: Sequence[PersonaVisualImportPreviewAsset | Mapping[str, Any]],
) -> dict[str, list[str]]:
    role_categories = _role_categories_by_role(capability)
    normalized: dict[str, list[str]] = {
        category: []
        for category in (capability.required_role_categories if capability else ())
    }

    for asset in assets:
        source_asset_id = _asset_text(asset, "source_asset_id")
        asset_role = _asset_text(asset, "asset_role")
        if not source_asset_id or not asset_role:
            continue
        categories = role_categories.get(asset_role) or (asset_role,)
        for category in categories:
            normalized.setdefault(category, []).append(source_asset_id)
    return normalized


def _role_categories_by_role(
    capability: PersonaVisualRendererCapability | None,
) -> dict[str, tuple[str, ...]]:
    if capability is None:
        return {}

    categories_by_role: dict[str, list[str]] = {}
    for category, roles in capability.role_category_map.items():
        categories_by_role.setdefault(category, []).append(category)
        for role in roles:
            categories_by_role.setdefault(role, []).append(category)
    return {
        role: tuple(dict.fromkeys(categories))
        for role, categories in categories_by_role.items()
    }


def _asset_text(
    asset: PersonaVisualImportPreviewAsset | Mapping[str, Any],
    field_name: str,
) -> str:
    if isinstance(asset, Mapping):
        return str(asset.get(field_name) or "")
    return str(getattr(asset, field_name) or "")


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


__all__ = [
    "PersonaVisualImportPreviewAsset",
    "PersonaVisualRendererImportPreviewResult",
    "preview_renderer_import",
]
