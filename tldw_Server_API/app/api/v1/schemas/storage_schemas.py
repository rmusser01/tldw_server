"""Pydantic schemas for storage management APIs.

Handles generated files tracking, storage quotas, and file browser operations.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta, default_offset_pagination_aliases

# File categories
FileCategory = Literal["tts_audio", "stt_audio", "image", "voice_clone", "mindmap", "spreadsheet"]

# Source features
SourceFeature = Literal["tts", "stt", "image_gen", "voice_studio", "mindmap", "data_tables", "export", "vn_assets"]

# Retention policies
RetentionPolicy = Literal["user_default", "permanent", "transient", "custom"]


# =========================================================================
# Generated File Schemas
# =========================================================================

class GeneratedFileBase(BaseModel):
    """Base fields for generated files."""
    filename: str = Field(description="Stored filename")
    original_filename: str | None = Field(default=None, description="Original user-provided filename")
    mime_type: str | None = Field(default=None, description="MIME type")
    file_size_bytes: int = Field(default=0, ge=0, description="File size in bytes")
    file_category: FileCategory = Field(description="File category")
    source_feature: SourceFeature = Field(description="Feature that generated the file")
    source_ref: str | None = Field(default=None, description="Reference to source entity")
    folder_tag: str | None = Field(default=None, description="Virtual folder tag")
    tags: list[str] | None = Field(default=None, description="Additional tags")


class GeneratedFileCreate(GeneratedFileBase):
    """Request to create a generated file record."""
    storage_path: str = Field(description="Relative path to file in storage")
    checksum: str | None = Field(default=None, description="SHA-256 checksum")
    org_id: int | None = Field(default=None, description="Organization ID")
    team_id: int | None = Field(default=None, description="Team ID")
    is_transient: bool = Field(default=False, description="Whether file is temporary")
    expires_at: datetime | None = Field(default=None, description="Expiration timestamp")
    retention_policy: RetentionPolicy = Field(default="user_default")


class GeneratedFileUpdate(BaseModel):
    """Request to update a generated file."""
    folder_tag: str | None = Field(default=None, description="Move to virtual folder")
    tags: list[str] | None = Field(default=None, description="Update tags")
    retention_policy: RetentionPolicy | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)


class GeneratedFile(GeneratedFileBase):
    """Generated file response."""
    id: int
    uuid: str
    user_id: int
    org_id: int | None = None
    team_id: int | None = None
    storage_path: str
    checksum: str | None = None
    is_transient: bool = False
    expires_at: datetime | None = None
    retention_policy: str = "user_default"
    is_deleted: bool = False
    deleted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    accessed_at: datetime | None = None

    class Config:
        from_attributes = True


class GeneratedFileResponse(BaseModel):
    """Single file response wrapper."""
    file: GeneratedFile


class GeneratedFilesListResponse(BaseModel):
    """Paginated list of files."""
    files: list[GeneratedFile]
    total: int
    offset: int
    limit: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return default_offset_pagination_aliases(self)


# =========================================================================
# Folder Schemas
# =========================================================================

class FolderInfo(BaseModel):
    """Virtual folder information."""
    folder_tag: str = Field(description="Folder tag name")
    file_count: int = Field(description="Number of files in folder")
    total_bytes: int = Field(description="Total size in bytes")
    total_mb: float = Field(default=0.0, description="Total size in MB")


class FolderListResponse(BaseModel):
    """List of virtual folders."""
    folders: list[FolderInfo]


class FolderCreateRequest(BaseModel):
    """Create a virtual folder (tag)."""
    name: str = Field(min_length=1, max_length=100, description="Folder name")


# =========================================================================
# Bulk Operation Schemas
# =========================================================================

class BulkDeleteRequest(BaseModel):
    """Request for bulk delete."""
    file_ids: list[int] = Field(min_length=1, max_length=100, description="File IDs to delete")
    hard_delete: bool = Field(default=False, description="Permanently delete if True")


class BulkDeleteResponse(BaseModel):
    """Response for bulk delete."""
    deleted_count: int
    file_ids: list[int]


class StorageDeleteResponse(BaseModel):
    """Response for deleting a generated storage file."""
    success: bool
    file_id: int
    hard_delete: bool


class BulkMoveRequest(BaseModel):
    """Request for bulk move to folder."""
    file_ids: list[int] = Field(min_length=1, max_length=100, description="File IDs to move")
    folder_tag: str | None = Field(default=None, description="Target folder (None to remove from folder)")


class BulkMoveResponse(BaseModel):
    """Response for bulk move."""
    moved_count: int
    file_ids: list[int]
    folder_tag: str | None


# =========================================================================
# Storage Usage Schemas
# =========================================================================

class CategoryUsage(BaseModel):
    """Usage breakdown for a category."""
    file_count: int = Field(default=0)
    total_bytes: int = Field(default=0)
    total_mb: float = Field(default=0.0)


class StorageUsage(BaseModel):
    """Storage usage summary."""
    total_bytes: int = Field(description="Total bytes used")
    total_mb: float = Field(description="Total MB used")
    by_category: dict[str, CategoryUsage] = Field(default_factory=dict)
    trash_bytes: int = Field(default=0, description="Bytes in trash")
    trash_mb: float = Field(default=0.0, description="MB in trash")


class StorageUsageResponse(BaseModel):
    """Storage usage response with quota info."""
    usage: StorageUsage
    quota_mb: int | None = Field(default=None, description="User quota in MB")
    quota_used_mb: float | None = Field(default=None, description="Total quota used")
    available_mb: float | None = Field(default=None, description="Available quota")
    usage_percentage: float | None = Field(default=None, description="Usage percentage")
    at_soft_limit: bool = Field(default=False, description="True if at/above 80% soft limit")
    at_hard_limit: bool = Field(default=False, description="True if at/above 100% hard limit")
    warning: str | None = Field(default=None, description="Warning message if approaching limit")


class UsageBreakdownResponse(BaseModel):
    """Detailed usage breakdown."""
    user_id: int
    by_category: dict[str, CategoryUsage]
    by_folder: list[FolderInfo]
    total_bytes: int
    total_mb: float
    quota_mb: int
    available_mb: float
    usage_percentage: float


# =========================================================================
# Quota Schemas
# =========================================================================

class QuotaStatus(BaseModel):
    """Quota status for user/team/org."""
    quota_mb: int | None = Field(default=None, description="Quota limit in MB")
    used_mb: float = Field(default=0.0, description="Used storage in MB")
    remaining_mb: float | None = Field(default=None, description="Remaining storage")
    usage_pct: float = Field(default=0.0, description="Usage percentage")
    at_soft_limit: bool = Field(default=False, description="At soft limit (warning)")
    at_hard_limit: bool = Field(default=False, description="At hard limit (blocked)")
    has_quota: bool = Field(default=False, description="Whether a quota is set")


class UserQuotaResponse(BaseModel):
    """User quota status."""
    user_id: int
    quota: QuotaStatus


class TeamQuotaResponse(BaseModel):
    """Team quota status."""
    team_id: int
    quota: QuotaStatus


class OrgQuotaResponse(BaseModel):
    """Organization quota status."""
    org_id: int
    quota: QuotaStatus


class CombinedQuotaResponse(BaseModel):
    """Combined quota status across user/team/org."""
    user_id: int
    has_quota: bool
    blocking_level: str | None = Field(default=None, description="Which level is blocking (user/team/org)")
    user: QuotaStatus
    team: QuotaStatus | None = None
    org: QuotaStatus | None = None


class SetQuotaRequest(BaseModel):
    """Request to set a quota."""
    quota_mb: int = Field(ge=100, description="Quota in MB (minimum 100)")
    soft_limit_pct: int = Field(default=80, ge=0, le=100, description="Soft limit percentage")
    hard_limit_pct: int = Field(default=100, ge=0, le=100, description="Hard limit percentage")


class SetQuotaResponse(BaseModel):
    """Response after setting quota."""
    success: bool
    quota: QuotaStatus


# =========================================================================
# Trash Schemas
# =========================================================================

class TrashListResponse(BaseModel):
    """List of trashed files."""
    files: list[GeneratedFile]
    total: int
    offset: int
    limit: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return default_offset_pagination_aliases(self)


class RestoreResponse(BaseModel):
    """Response for restore operation."""
    success: bool
    file: GeneratedFile | None = None


class PermanentDeleteResponse(BaseModel):
    """Response for permanent delete."""
    success: bool
    file_id: int


# =========================================================================
# Query Parameters (as models for documentation)
# =========================================================================

class FileListQuery(BaseModel):
    """Query parameters for listing files."""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)
    file_category: FileCategory | None = None
    source_feature: SourceFeature | None = None
    folder_tag: str | None = None
    search: str | None = None
    include_deleted: bool = False


class TrashQuery(BaseModel):
    """Query parameters for trash listing."""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)


# =========================================================================
# Admin Quota Schemas
# =========================================================================

class AdminQuotaListItem(BaseModel):
    """Quota item in admin list."""
    id: int
    org_id: int | None = None
    team_id: int | None = None
    org_name: str | None = None
    team_name: str | None = None
    quota_mb: int
    used_mb: float
    soft_limit_pct: int
    hard_limit_pct: int
    usage_pct: float
    created_at: datetime
    updated_at: datetime


class AdminQuotaListResponse(BaseModel):
    """Admin quota list response."""
    quotas: list[AdminQuotaListItem]
    total: int
    offset: int
    limit: int
