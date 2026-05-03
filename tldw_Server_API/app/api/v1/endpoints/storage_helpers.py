"""Helper functions for storage endpoint record conversion and authorization."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path as PathlibPath

from tldw_Server_API.app.api.v1.schemas.storage_schemas import GeneratedFile, QuotaStatus
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import FILE_CATEGORY_VOICE_CLONE
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


def _principal_is_storage_admin(principal: AuthPrincipal) -> bool:
    """Check storage-admin compatibility claims."""
    roles = {str(role).strip().lower() for role in (principal.roles or []) if str(role).strip()}
    permissions = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    if bool(getattr(principal, "is_admin", False)):
        return True
    if "admin" in roles:
        return True
    return bool(permissions & {"*", "system.configure"})


def _parse_datetime(value: object) -> datetime | None:
    """Parse datetime from API/database values."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def _to_generated_file(record: dict) -> GeneratedFile:
    """Convert a generated-files database record to an API schema."""
    return GeneratedFile(
        id=record.get("id", 0),
        uuid=record.get("uuid", ""),
        user_id=record.get("user_id", 0),
        org_id=record.get("org_id"),
        team_id=record.get("team_id"),
        filename=record.get("filename", ""),
        original_filename=record.get("original_filename"),
        storage_path=record.get("storage_path", ""),
        mime_type=record.get("mime_type"),
        file_size_bytes=record.get("file_size_bytes", 0),
        checksum=record.get("checksum"),
        file_category=record.get("file_category", "image"),
        source_feature=record.get("source_feature", "export"),
        source_ref=record.get("source_ref"),
        folder_tag=record.get("folder_tag"),
        tags=record.get("tags", []),
        is_transient=record.get("is_transient", False),
        expires_at=_parse_datetime(record.get("expires_at")),
        retention_policy=record.get("retention_policy", "user_default"),
        is_deleted=record.get("is_deleted", False),
        deleted_at=_parse_datetime(record.get("deleted_at")),
        created_at=_parse_datetime(record.get("created_at")) or datetime.now(timezone.utc),
        updated_at=_parse_datetime(record.get("updated_at")) or datetime.now(timezone.utc),
        accessed_at=_parse_datetime(record.get("accessed_at")),
    )


def _resolve_storage_base_dir(user_id: int, record: dict) -> PathlibPath:
    """Resolve the base directory for a stored file based on category."""
    if record.get("file_category") == FILE_CATEGORY_VOICE_CLONE:
        return DatabasePaths.get_user_voices_dir(user_id)
    return DatabasePaths.get_user_outputs_dir(user_id)


def _normalize_folder_tag(folder_tag: str | None) -> str | None:
    """Normalize optional virtual folder tags using the storage folder rules."""
    if folder_tag is None:
        return None
    name = folder_tag.strip()
    if not name or "/" in name or "\\" in name:
        raise ValueError("Invalid folder name")
    return name


def _to_quota_status(data: dict) -> QuotaStatus:
    """Convert quota service data to an API schema."""
    return QuotaStatus(
        quota_mb=data.get("quota_mb"),
        used_mb=data.get("used_mb", 0.0),
        remaining_mb=data.get("remaining_mb"),
        usage_pct=data.get("usage_pct", 0.0),
        at_soft_limit=data.get("at_soft_limit", False),
        at_hard_limit=data.get("at_hard_limit", False),
        has_quota=data.get("has_quota", False),
    )
