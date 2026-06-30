"""
Helpers for mapping profile update skip reasons to structured error responses.
"""

from __future__ import annotations

from collections.abc import Iterable

from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileErrorDetail,
)
from tldw_Server_API.app.core.UserProfiles.error_mapping import (
    classify_legacy_profile_update_skips,
)


def classify_profile_update_skips(
    skipped: Iterable[dict[str, str]],
) -> tuple[int, str, str, list[UserProfileErrorDetail]] | None:
    """Map per-key skip reasons into a single structured error response."""
    skipped_list = list(skipped)
    if not skipped_list:
        return None

    errors = [
        UserProfileErrorDetail(
            key=str(item.get("key") or ""),
            message=str(item.get("message") or ""),
        )
        for item in skipped_list
    ]
    mapped = classify_legacy_profile_update_skips(skipped_list)
    if mapped is None:
        return None
    return (
        mapped.status_code,
        mapped.error_code,
        mapped.detail,
        errors,
    )


__all__ = ["classify_profile_update_skips"]
