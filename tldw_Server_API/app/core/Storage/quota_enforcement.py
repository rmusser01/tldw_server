"""
Storage quota enforcement utilities.

Provides a lightweight check that can be called before file uploads to verify
that a user (or their org/team) has sufficient storage quota remaining.
"""
from __future__ import annotations

import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import (
    AuthnzStorageQuotasRepo,
)


def _quota_fail_open_enabled() -> bool:
    """Return True when quota check errors should allow writes."""
    value = os.getenv("STORAGE_QUOTA_FAIL_OPEN", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


async def check_storage_quota(
    user_id: int,
    file_size_bytes: int,
    db_pool: Any,
    *,
    org_id: int | None = None,
    team_id: int | None = None,
) -> dict[str, Any]:
    """Check if a user has enough storage quota for a file.

    Queries the storage_quotas repo via the user's org or team membership.
    If no quota is configured, the upload is allowed by default.

    Args:
        user_id: The user attempting the upload.
        file_size_bytes: Size of the file to be uploaded, in bytes.
        db_pool: AuthNZ DatabasePool instance.
        org_id: Optional organization ID for org-level quota checks.
        team_id: Optional team ID for team-level quota checks.

    Returns:
        Dict with keys:
            allowed (bool): Whether the upload should proceed.
            used_mb (float): Current storage usage in MB.
            quota_mb (int | None): Configured quota in MB, or None if unlimited.
            remaining_mb (float | None): Remaining quota in MB, or None if unlimited.
            reason (str): Human-readable explanation.
    """
    file_size_mb = file_size_bytes / (1024 * 1024)

    repo = AuthnzStorageQuotasRepo(db_pool=db_pool)

    # Check org-level quota first, then team-level
    can_alloc = True
    reason = "No quota limit set"
    status: dict[str, Any] = {
        "quota_mb": None,
        "used_mb": 0.0,
        "remaining_mb": None,
        "has_quota": False,
    }

    try:
        if org_id is not None:
            status = await repo.check_quota_status(org_id=org_id)
        elif team_id is not None:
            status = await repo.check_quota_status(team_id=team_id)
    except Exception as exc:
        fail_open = _quota_fail_open_enabled()
        logger.warning(
            "Storage quota check failed for user_id={}: {}",
            user_id,
            type(exc).__name__,
        )
        return {
            "allowed": fail_open,
            "used_mb": 0.0,
            "quota_mb": None,
            "remaining_mb": None,
            "reason": (
                "Quota check unavailable (fail-open)"
                if fail_open
                else "Quota check unavailable"
            ),
        }

    if not status.get("has_quota", False):
        return {
            "allowed": True,
            "used_mb": status.get("used_mb", 0.0),
            "quota_mb": None,
            "remaining_mb": None,
            "reason": "No quota limit set",
        }

    quota_mb = status.get("quota_mb", 0)
    used_mb = status.get("used_mb", 0.0)
    remaining_mb = status.get("remaining_mb", 0.0)

    if status.get("at_hard_limit", False):
        can_alloc = False
        reason = "Storage quota exceeded (at hard limit)"
    elif file_size_mb > remaining_mb:
        can_alloc = False
        reason = (
            f"Insufficient storage quota. "
            f"Need {file_size_mb:.2f} MB, only {remaining_mb:.2f} MB available"
        )
    elif status.get("at_soft_limit", False):
        can_alloc = True
        reason = "Warning: Approaching storage limit (soft limit reached)"
    else:
        can_alloc = True
        reason = "Quota check passed"

    return {
        "allowed": can_alloc,
        "used_mb": used_mb,
        "quota_mb": quota_mb,
        "remaining_mb": remaining_mb,
        "reason": reason,
    }
