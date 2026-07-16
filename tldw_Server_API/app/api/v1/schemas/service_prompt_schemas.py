"""Wire schemas for the curated Service Prompts API."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ServicePromptPartResponse(BaseModel):
    """Public metadata for one editable prompt part."""

    key: str
    label: str
    mode: Literal["literal", "template"]
    required_variables: list[str]


class ServicePromptWorkflowResponse(BaseModel):
    """Public metadata for one affected workflow."""

    id: str
    label: str


class ServicePromptCatalogItemResponse(BaseModel):
    """Catalog metadata that deliberately excludes prompt bodies."""

    id: str
    label: str
    description: str
    parts: list[ServicePromptPartResponse]
    affected_workflows: list[ServicePromptWorkflowResponse]


class ServicePromptDetailResponse(ServicePromptCatalogItemResponse):
    """Current packaged, saved, and effective state for one prompt."""

    default_parts: dict[str, str]
    saved_parts: dict[str, str] | None
    effective_parts: dict[str, str]
    source: Literal["user", "packaged"]
    revision: UUID | None


class ServicePromptUpdateRequest(BaseModel):
    """Complete compare-and-swap payload for one prompt override."""

    model_config = ConfigDict(extra="forbid")

    parts: dict[str, object]
    expected_revision: UUID | None


__all__ = [
    "ServicePromptCatalogItemResponse",
    "ServicePromptDetailResponse",
    "ServicePromptPartResponse",
    "ServicePromptUpdateRequest",
    "ServicePromptWorkflowResponse",
]
