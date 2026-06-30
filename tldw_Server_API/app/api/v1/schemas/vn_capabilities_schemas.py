"""Schemas for VN platform capabilities discovery."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class VNProfileSummary(BaseModel):
    """Visible VN policy or generation profile summary."""

    id: str
    name: str
    visible: bool = True


class VNSupportedMediaTypes(BaseModel):
    """Media types that VN content endpoints can produce."""

    image: list[str] = Field(default_factory=list)
    audio: list[str] = Field(default_factory=list)


class VNScriptedGenerationCapabilities(BaseModel):
    """Stable capabilities for backend-owned scripted generation."""

    enabled: bool
    output_schemas: list[str] = Field(default_factory=list)
    confirmation_supported: bool
    revision_activation_supported: bool
    history_supported: bool
    debug_detail_supported: bool
    dynamic_choice_supported: bool
    scene_update_supported: bool
    max_automatic_generation_batch_count: int
    moderation_blocked_raw_reveal_supported: bool


class VNRouteMigration(BaseModel):
    """Canonical route metadata for clients migrating to the VN namespace."""

    canonical: str
    supersedes: list[str] = Field(default_factory=list)


class VNCapabilitiesResponse(BaseModel):
    """Response model for ``GET /api/v1/vn/vn-capabilities``."""

    schema_version: Literal["vn_capabilities.v1"]
    generated_at: datetime
    base_path: str
    resources: dict[str, str]
    enabled_modules: dict[str, bool]
    features: dict[str, bool]
    limits: dict[str, int]
    supported_content_ratings: list[str]
    visible_policy_profiles: list[VNProfileSummary]
    visible_generation_profiles: list[VNProfileSummary]
    supported_media_types: VNSupportedMediaTypes
    scripted_generation: VNScriptedGenerationCapabilities
    route_migration: VNRouteMigration
    docs: dict[str, str]
    openapi: str
