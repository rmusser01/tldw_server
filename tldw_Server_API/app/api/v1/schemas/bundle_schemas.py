# bundle_schemas.py
# Description: Pydantic schemas for admin backup bundle endpoints.
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


class BundleCreateRequest(BaseModel):
    """Request to create a backup bundle."""

    datasets: list[str] | None = Field(
        None,
        description="Datasets to include (defaults to all if omitted).",
    )
    user_id: int | None = Field(None, description="User ID for per-user datasets.")
    include_vector_store: bool = Field(
        False,
        description="Include vector store data (not yet supported).",
    )
    max_backups: int | None = Field(None, ge=1, le=1000)
    retention_hours: int | None = Field(
        None,
        ge=1,
        description="Auto-delete older bundles in the same user scope after creation.",
    )
    notes: str | None = Field(None, max_length=2000)

    model_config = ConfigDict(from_attributes=True)


class BundleItem(BaseModel):
    """Metadata for a single bundle artifact."""

    bundle_id: str
    user_id: int | None = None
    created_at: datetime
    size_bytes: int
    datasets: list[str]
    schema_versions: dict[str, int | None] = Field(default_factory=dict)
    app_version: str | None = None
    manifest_version: int
    notes: str | None = None

    model_config = ConfigDict(from_attributes=True)


class BundleCreateResponse(BaseModel):
    """Response for bundle creation."""

    item: BundleItem
    status: str = "created"
    message: str = "Bundle created successfully"

    model_config = ConfigDict(from_attributes=True)


class BundleListResponse(BaseModel):
    """Response for bundle listing."""

    items: list[BundleItem]
    total: int
    limit: int
    offset: int
    pagination: OffsetPaginationMeta
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class BundleMetadataResponse(BaseModel):
    """Response for single bundle metadata."""

    item: BundleItem

    model_config = ConfigDict(from_attributes=True)


class BundleImportValidation(BaseModel):
    """Compatibility check for a single dataset inside a bundle."""

    dataset: str
    manifest_version: int | None = None
    current_version: int | None = None
    compatible: bool = True
    message: str = "ok"

    model_config = ConfigDict(from_attributes=True)


class BundleImportResponse(BaseModel):
    """Response for bundle import."""

    status: str
    datasets_restored: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    safety_snapshots: dict[str, str] = Field(default_factory=dict)
    validations: list[BundleImportValidation] = Field(default_factory=list)
    rollback_failures: list[str] = Field(
        default_factory=list,
        description="Datasets whose rollback failed during an import error.",
    )

    model_config = ConfigDict(from_attributes=True)


class BundleDeleteResponse(BaseModel):
    """Response for bundle deletion."""

    status: str = "deleted"
    bundle_id: str

    model_config = ConfigDict(from_attributes=True)
