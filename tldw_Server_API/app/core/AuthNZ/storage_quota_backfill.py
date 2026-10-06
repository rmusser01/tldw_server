"""Shared values for the users.storage_quota_mb -> limits.storage_quota_mb backfill (spec 2 §8)."""

from loguru import logger

LEGACY_DEFAULT_STORAGE_QUOTA_MB = 5120
STORAGE_QUOTA_KEY = "limits.storage_quota_mb"
PG_BACKFILL_MARKER = "storage_quota_mb_to_user_overrides_v1"


def skip_values() -> list[int]:
    """Column values that can't be told apart from "never set": 5120 and the configured default."""
    values = {LEGACY_DEFAULT_STORAGE_QUOTA_MB}
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings

        values.add(int(get_settings().DEFAULT_STORAGE_QUOTA_MB))
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        logger.warning(
            f"Could not read DEFAULT_STORAGE_QUOTA_MB ({type(exc).__name__}); "
            f"only the legacy default {LEGACY_DEFAULT_STORAGE_QUOTA_MB} will be skipped"
        )
    return sorted(values)
