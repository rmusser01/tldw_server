from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from tldw_Server_API.app.api.v1.schemas.pagination import PagePaginationMeta


class PrivilegeBucket(BaseModel):
    key: str
    users: int
    endpoints: int
    scopes: int
    metadata: dict[str, Any] | None = None


class PrivilegeSummaryResponse(BaseModel):
    catalog_version: str
    generated_at: datetime
    group_by: str
    buckets: list[PrivilegeBucket]
    trends: list[PrivilegeTrend] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrivilegeDependency(BaseModel):
    id: str
    type: str = "dependency"
    module: str | None = None


class PrivilegeDetailItem(BaseModel):
    user_id: str
    user_name: str
    role: str
    endpoint: str
    method: str
    privilege_scope_id: str
    feature_flag_id: str | None = None
    sensitivity_tier: str
    ownership_predicates: list[str] = Field(default_factory=list)
    status: Literal["allowed", "blocked"]
    blocked_reason: str | None = None
    dependencies: list[PrivilegeDependency] = Field(default_factory=list)
    dependency_sources: list[str] = Field(default_factory=list)
    rate_limit_class: str | None = None
    rate_limit_resources: list[str] = Field(default_factory=list)
    source_module: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)


class PrivilegeDetailResponse(BaseModel):
    catalog_version: str
    generated_at: datetime
    page: int
    page_size: int
    total_items: int
    pagination: PagePaginationMeta
    items: list[PrivilegeDetailItem]
    recommended_actions: list[PrivilegeRecommendedAction] | None = None


PrivilegeOrgResponse = Union[PrivilegeSummaryResponse, PrivilegeDetailResponse]


class PrivilegeSnapshotSummary(BaseModel):
    users: int
    scopes: int
    endpoints: int | None = None
    sensitivity_breakdown: dict[str, int] = Field(default_factory=dict)
    scope_ids: list[str] = Field(default_factory=list)
    endpoint_paths: list[str] = Field(default_factory=list)


class PrivilegeSnapshotListItem(BaseModel):
    snapshot_id: str
    generated_at: datetime
    generated_by: str
    org_id: str | None = None
    team_id: str | None = None
    catalog_version: str
    summary: PrivilegeSnapshotSummary | None = None
    target_scope: Literal["org", "team", "user"] | None = None


class PrivilegeSnapshotListResponse(BaseModel):
    page: int
    page_size: int
    total_items: int
    pagination: PagePaginationMeta
    items: list[PrivilegeSnapshotListItem]
    filters: dict[str, Any] = Field(default_factory=dict)


class PrivilegeTrendWindow(BaseModel):
    start: datetime
    end: datetime


class PrivilegeTrend(BaseModel):
    key: str
    window: PrivilegeTrendWindow
    delta_users: int = 0
    delta_endpoints: int = 0
    delta_scopes: int = 0


class PrivilegeRecommendedAction(BaseModel):
    privilege_scope_id: str | None = None
    action: str
    reason: str | None = None


class PrivilegeSelfItem(BaseModel):
    endpoint: str
    method: str
    privilege_scope_id: str
    feature_flag_id: str | None = None
    sensitivity_tier: str | None = None
    ownership_predicates: list[str] = Field(default_factory=list)
    status: Literal["allowed", "blocked"]
    blocked_reason: str | None = None
    dependencies: list[PrivilegeDependency] = Field(default_factory=list)
    dependency_sources: list[str] = Field(default_factory=list)
    rate_limit_class: str | None = None
    rate_limit_resources: list[str] = Field(default_factory=list)
    source_module: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)


class PrivilegeSelfResponse(BaseModel):
    catalog_version: str
    generated_at: datetime
    items: list[PrivilegeSelfItem]
    recommended_actions: list[PrivilegeRecommendedAction] = Field(default_factory=list)


class PrivilegeSnapshotCreateRequest(BaseModel):
    target_scope: Literal["org", "team", "user"]
    org_id: str | None = None
    team_id: str | None = None
    user_ids: list[str] | None = None
    catalog_version: str | None = None
    notes: str | None = None
    async_job: bool = Field(False, alias="async")
    model_config = ConfigDict(populate_by_name=True)


class PrivilegeSnapshotAcceptedResponse(BaseModel):
    request_id: str
    status: Literal["queued", "processing"]
    estimated_ready_at: datetime | None = None


class PrivilegeSnapshotRecord(BaseModel):
    snapshot_id: str
    generated_at: datetime
    generated_by: str
    target_scope: Literal["org", "team", "user"]
    org_id: str | None = None
    team_id: str | None = None
    catalog_version: str
    summary: PrivilegeSnapshotSummary | None = None


class PrivilegeSnapshotDetailItem(BaseModel):
    user_id: str | None = None
    endpoint: str
    method: str
    privilege_scope_id: str
    feature_flag_id: str | None = None
    sensitivity_tier: str | None = None
    ownership_predicates: list[str] = Field(default_factory=list)
    status: Literal["allowed", "blocked"]
    blocked_reason: str | None = None
    dependencies: list[PrivilegeDependency] = Field(default_factory=list)
    dependency_sources: list[str] = Field(default_factory=list)
    rate_limit_class: str | None = None
    rate_limit_resources: list[str] = Field(default_factory=list)
    source_module: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)


class PrivilegeSnapshotDetailMatrix(BaseModel):
    page: int
    page_size: int
    total_items: int
    pagination: PagePaginationMeta
    items: list[PrivilegeSnapshotDetailItem]


class PrivilegeSnapshotDetailResponse(BaseModel):
    snapshot_id: str
    catalog_version: str
    generated_at: datetime
    generated_by: str
    target_scope: Literal["org", "team", "user"]
    org_id: str | None = None
    team_id: str | None = None
    summary: PrivilegeSnapshotSummary | None = None
    detail: PrivilegeSnapshotDetailMatrix | None = None
    etag: str | None = None
