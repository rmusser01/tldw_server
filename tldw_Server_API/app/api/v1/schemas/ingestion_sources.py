"""Pydantic request and response models for ingestion source endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IngestionSourceCreateRequest(BaseModel):
    """Payload used to create a new ingestion source."""

    source_type: Literal["local_directory", "archive_snapshot", "git_repository"]
    sink_type: Literal["media", "notes"]
    policy: Literal["canonical", "import_only"] = "canonical"
    enabled: bool = True
    schedule_enabled: bool = False
    schedule: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class IngestionSourcePatchRequest(BaseModel):
    """Mutable fields that can be updated for an existing ingestion source."""

    model_config = ConfigDict(extra="forbid")

    source_type: Literal["local_directory", "archive_snapshot", "git_repository"] | None = None
    sink_type: Literal["media", "notes"] | None = None
    policy: Literal["canonical", "import_only"] | None = None
    enabled: bool | None = None
    schedule_enabled: bool | None = None
    schedule: dict[str, Any] | None = None
    config: dict[str, Any] | None = None


class IngestionSourceResponse(BaseModel):
    """Serialized ingestion source state returned by list and detail endpoints."""

    model_config = ConfigDict(extra="ignore")

    id: int
    user_id: int
    source_type: str
    sink_type: str
    policy: str
    enabled: bool
    schedule_enabled: bool = False
    schedule_config: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    active_job_id: str | None = None
    last_successful_snapshot_id: int | None = None
    last_sync_started_at: str | None = None
    last_sync_completed_at: str | None = None
    last_sync_status: str | None = None
    last_error: str | None = None
    last_successful_sync_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class IngestionSourceCapabilitiesResponse(BaseModel):
    """Current-user ingestion source capability flags."""

    can_create_local_directory: bool


class IngestionSourceDirectoryEntryResponse(BaseModel):
    """One directory entry exposed by the ingestion source path browser."""

    name: str
    path: str
    is_root: bool = False


class IngestionSourceDirectoryBrowseResponse(BaseModel):
    """Directory browser response scoped to ingestion source allowed roots."""

    roots: list[IngestionSourceDirectoryEntryResponse] = Field(default_factory=list)
    current_path: str | None = None
    parent_path: str | None = None
    entries: list[IngestionSourceDirectoryEntryResponse] = Field(default_factory=list)
    error: str | None = None


class IngestionSourceItemResponse(BaseModel):
    """Serialized tracked-item state for a single ingestion source item."""

    model_config = ConfigDict(extra="ignore")

    id: int
    source_id: int
    normalized_relative_path: str
    content_hash: str | None = None
    sync_status: str
    binding: dict[str, Any] = Field(default_factory=dict)
    present_in_source: bool = True
    created_at: str | None = None
    updated_at: str | None = None


class IngestionSourceSyncTriggerResponse(BaseModel):
    """Acknowledgement returned when a sync-related job is queued."""

    status: str
    source_id: int
    job_id: int | str | None = None
    snapshot_status: str | None = None
