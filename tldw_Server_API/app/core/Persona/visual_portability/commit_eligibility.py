"""Shared commit eligibility checks for persona visual import previews.

These helpers keep the REST enqueue path and background import worker aligned so
stored preview metadata and archive revalidation results both fail closed before
any visual pack records are created.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def is_import_preview_plan_committable(proposed_plan: Any) -> bool:
    """Return whether a preview plan can be committed into a visual pack."""
    return not import_preview_commit_blockers(proposed_plan)


def is_import_preview_result_committable(preview_result: Mapping[str, Any]) -> bool:
    """Return whether a revalidated preview result is safe to commit."""
    if str(preview_result.get("status") or "") != "completed":
        return False
    return is_import_preview_plan_committable(preview_result.get("proposed_plan"))


def import_preview_commit_blockers(proposed_plan: Any) -> list[str]:
    """Normalize reasons that an import preview plan must not be committed."""
    if not isinstance(proposed_plan, Mapping):
        return []

    blockers = _string_list(proposed_plan.get("commit_blockers"))
    if "commit_eligible" in proposed_plan and proposed_plan.get("commit_eligible") is not True:
        blockers.append("commit_eligible_not_true")

    renderer_preview = proposed_plan.get("renderer_import_preview")
    if isinstance(renderer_preview, Mapping) and renderer_preview.get("can_commit") is not True:
        blockers.append("renderer_import_preview_not_committable")

    return blockers


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]
