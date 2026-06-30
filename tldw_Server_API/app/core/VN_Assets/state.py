"""Derived VN asset pack state and readiness helpers."""

from __future__ import annotations

from collections.abc import Iterable

from tldw_Server_API.app.core.VN_Assets.constants import (
    ITEM_REVIEW_STATUS_APPROVED,
    ITEM_REVIEW_STATUS_DRAFT,
    SLOT_STATUS_APPROVED,
    SLOT_STATUS_CANCELLED,
    SLOT_STATUS_FAILED,
    SLOT_STATUS_GENERATING,
    SLOT_STATUS_PLANNED,
    SLOT_STATUS_QUEUED,
    SLOT_STATUS_REVIEWING,
    SLOT_STATUS_SKIPPED,
    PACK_STATUS_GENERATING,
    PACK_STATUS_NOT_READY,
    PACK_STATUS_READY,
)
from tldw_Server_API.app.core.VN_Assets.models import PackReadiness, SlotReadiness


def derive_slot_status(
    *,
    has_active_job: bool,
    has_queued_job: bool,
    is_skipped: bool,
    is_cancelled: bool = False,
    requested_variants: int,
    failed_variants: int,
    review_statuses: Iterable[str],
    required_for_runtime: bool,
) -> str:
    """Derive a slot status from jobs, failures, and review statuses."""
    if has_active_job:
        return SLOT_STATUS_GENERATING
    if has_queued_job:
        return SLOT_STATUS_QUEUED
    if is_skipped:
        return SLOT_STATUS_SKIPPED

    statuses = list(review_statuses)
    approved_count = statuses.count(ITEM_REVIEW_STATUS_APPROVED)
    draft_count = statuses.count(ITEM_REVIEW_STATUS_DRAFT)

    if requested_variants > 0 and failed_variants >= requested_variants and approved_count == 0:
        return SLOT_STATUS_FAILED
    if draft_count > 0:
        return SLOT_STATUS_REVIEWING
    if required_for_runtime and approved_count == 0 and (statuses or failed_variants > 0):
        return SLOT_STATUS_REVIEWING
    if approved_count > 0:
        return SLOT_STATUS_APPROVED
    if is_cancelled:
        return SLOT_STATUS_CANCELLED
    if statuses:
        return SLOT_STATUS_REVIEWING
    return SLOT_STATUS_PLANNED


def derive_pack_readiness(
    *,
    required_slots: Iterable[SlotReadiness],
    optional_slots: Iterable[SlotReadiness],
    active_jobs: int,
    approved_item_errors: Iterable[str],
) -> PackReadiness:
    """Derive whether a pack can be used at runtime."""
    warnings: list[str] = []
    errors = list(approved_item_errors)

    required = list(required_slots)
    optional = list(optional_slots)

    for slot in required:
        warnings.extend(slot.warnings)
        if slot.status != SLOT_STATUS_APPROVED:
            errors.append(f"required_slot_not_ready:{slot.slot_id}")

    for slot in optional:
        warnings.extend(slot.warnings)

    if active_jobs > 0:
        return PackReadiness(
            ready=False,
            status=PACK_STATUS_GENERATING,
            warnings=_deduplicate(warnings),
            errors=errors,
        )

    ready = not errors
    return PackReadiness(
        ready=ready,
        status=PACK_STATUS_READY if ready else PACK_STATUS_NOT_READY,
        warnings=_deduplicate(warnings),
        errors=errors,
    )


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
