"""Entitlement checks for ingestion source creation."""

from __future__ import annotations

import hashlib
from typing import Any

from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_mode
from tldw_Server_API.app.services.admin_system_ops_service import list_feature_flags

LOCAL_DIRECTORY_INGESTION_SOURCE_FLAG_KEY = "ingestion_sources.local_directory"


def can_create_local_directory_ingestion_source(current_user: Any) -> bool:
    """Return whether the current user may create local-directory sources."""
    if is_single_user_mode():
        return True
    user_id = _user_id(current_user)
    if user_id is None:
        return False
    org_ids = _org_ids(current_user)
    for flag in list_feature_flags():
        if flag.get("key") != LOCAL_DIRECTORY_INGESTION_SOURCE_FLAG_KEY:
            continue
        if _enabled_flag_applies(flag, user_id=user_id, org_ids=org_ids):
            return True
    return False


def _enabled_flag_applies(flag: dict[str, Any], *, user_id: int, org_ids: set[int]) -> bool:
    if not bool(flag.get("enabled")):
        return False
    target_user_ids = {_coerce_int(value) for value in flag.get("target_user_ids") or []}
    target_user_ids.discard(None)
    if target_user_ids and user_id not in target_user_ids:
        return False
    rollout_percent = _rollout_percent(flag.get("rollout_percent"))
    if rollout_percent <= 0:
        return False
    if rollout_percent < 100 and not _is_in_rollout(
        key=str(flag.get("key") or ""),
        user_id=user_id,
        percent=rollout_percent,
    ):
        return False

    scope = str(flag.get("scope") or "global").strip().lower()
    if scope == "global":
        return True
    if scope == "user":
        return _coerce_int(flag.get("user_id")) == user_id
    if scope == "org":
        org_id = _coerce_int(flag.get("org_id"))
        return org_id is not None and org_id in org_ids
    return False


def _user_id(current_user: Any) -> int | None:
    value = getattr(current_user, "id", None)
    coerced = _coerce_int(value)
    if coerced is None or coerced <= 0:
        return None
    return coerced


def _org_ids(current_user: Any) -> set[int]:
    raw_values = [getattr(current_user, "active_org_id", None)]
    org_ids = getattr(current_user, "org_ids", None)
    if isinstance(org_ids, (list, tuple, set)):
        raw_values.extend(org_ids)
    cleaned = {_coerce_int(value) for value in raw_values}
    cleaned.discard(None)
    return {value for value in cleaned if value > 0}


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rollout_percent(value: Any) -> int:
    try:
        percent = int(value)
    except (TypeError, ValueError):
        return 100
    return min(100, max(0, percent))


def _is_in_rollout(*, key: str, user_id: int, percent: int) -> bool:
    digest = hashlib.sha256(f"{key}:{user_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return bucket < percent
