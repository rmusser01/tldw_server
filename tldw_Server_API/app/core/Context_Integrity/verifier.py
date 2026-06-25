"""Verification of current context assets against approved manifest entries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

from tldw_Server_API.app.core.Context_Integrity.models import (
    ContextAssetDescriptor,
    ContextAssetSource,
    ContextIntegrityFinding,
)


def _entry_by_asset_id(entries: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(entry["asset_id"]): entry for entry in entries}


def _entry_source_type(entry: Mapping[str, Any]) -> ContextAssetSource:
    return cast(ContextAssetSource, str(entry.get("source_type", "skill_file")))


def verify_inventory(
    *,
    current_assets: Iterable[ContextAssetDescriptor],
    approved_entries: Iterable[Mapping[str, Any]],
) -> tuple[ContextIntegrityFinding, ...]:
    """Compare current inventory against approved manifest entries."""
    approved = _entry_by_asset_id(approved_entries)
    current = {asset.asset_id: asset for asset in current_assets}
    findings: list[ContextIntegrityFinding] = []

    for asset_id, asset in current.items():
        entry = approved.get(asset_id)
        if entry is None:
            findings.append(
                ContextIntegrityFinding(
                    asset_id=asset_id,
                    state="new_unapproved",
                    severity="warning",
                    summary=f"Unapproved context asset detected: {asset.display_name}",
                    remediation="Review and approve the asset before model use.",
                    source_type=asset.source_type,
                    current_digest=asset.digest,
                )
            )
            continue

        approved_digest = str(entry.get("digest"))
        if approved_digest == asset.digest:
            continue

        approved_executable = bool(entry.get("executable", False))
        state = "changed_approved_executable" if approved_executable else "changed_approved_non_executable"
        findings.append(
            ContextIntegrityFinding(
                asset_id=asset_id,
                state=state,
                severity="error" if approved_executable else "warning",
                summary=f"Approved context asset changed: {asset.display_name}",
                remediation="Review the diff and approve a new manifest version or restore the asset.",
                source_type=asset.source_type,
                current_digest=asset.digest,
                approved_digest=approved_digest,
            )
        )

    for asset_id, entry in approved.items():
        if asset_id in current:
            continue
        required = bool(entry.get("required", False))
        findings.append(
            ContextIntegrityFinding(
                asset_id=asset_id,
                state="missing_required" if required else "missing_optional",
                severity="error" if required else "warning",
                summary=f"Approved context asset is missing: {entry.get('display_name') or asset_id}",
                remediation="Restore the asset or approve a manifest that removes it.",
                source_type=_entry_source_type(entry),
                approved_digest=str(entry.get("digest")),
            )
        )

    return tuple(findings)
