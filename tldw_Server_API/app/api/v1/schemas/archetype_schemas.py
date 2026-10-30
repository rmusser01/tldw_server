"""
Pydantic schemas for Archetype Templates.

Archetypes are pre-built assistant profiles that users can select during
first-run onboarding to bootstrap persona, MCP, policy, buddy, and
voice configuration.
"""
from __future__ import annotations

from typing import Any, Literal

from mcp_unified.federation.models import MCPAuthType, MCPCatalogEntry
from pydantic import BaseModel, Field

# Core-owned since TASK-13950.15; re-exported for the API layer.
from tldw_Server_API.app.core.Persona.archetype_schemas import (  # noqa: F401 - re-exported (TASK-13950.15)
    ArchetypeBuddyDefaults,
    ArchetypeMCPConfig,
    ArchetypePersonaDefaults,
    ArchetypePolicyDefaults,
    ArchetypeStarterCommand,
    ArchetypeSummary,
    ArchetypeTemplate,
    ArchetypeToolOverride,
)


class ArchetypePreviewSetupState(BaseModel):
    """Minimal setup state seeded by the archetype preview endpoint."""

    status: Literal["not_started"] = "not_started"
    current_step: Literal["archetype"] = "archetype"


class ArchetypePreviewResponse(BaseModel):
    """Preview payload used to pre-fill the assistant setup wizard."""

    name: str
    system_prompt: str | None = None
    archetype_key: str
    voice_defaults: dict[str, Any] = Field(default_factory=dict)
    setup: ArchetypePreviewSetupState = Field(default_factory=ArchetypePreviewSetupState)


__all__ = [
    "ArchetypeBuddyDefaults",
    "ArchetypeMCPConfig",
    "ArchetypePersonaDefaults",
    "ArchetypePolicyDefaults",
    "ArchetypePreviewResponse",
    "ArchetypePreviewSetupState",
    "ArchetypeStarterCommand",
    "ArchetypeSummary",
    "ArchetypeTemplate",
    "ArchetypeToolOverride",
    "MCPCatalogEntry",
    "MCPAuthType",
]
