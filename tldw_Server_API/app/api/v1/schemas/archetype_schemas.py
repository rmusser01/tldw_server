"""
Pydantic schemas for Archetype Templates.

Archetypes are pre-built assistant profiles that users can select during
first-run onboarding to bootstrap persona, MCP, policy, buddy, and
voice configuration.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.persona import PersonaConfirmationMode

MCPAuthType = Literal["none", "bearer", "api_key"]


class ArchetypePersonaDefaults(BaseModel):
    """Default persona settings seeded by an archetype."""

    name: str
    system_prompt: str | None = None
    personality_traits: list[str] = Field(default_factory=list)


class ArchetypeMCPConfig(BaseModel):
    """Which MCP modules an archetype enables or disables by default."""

    enabled: list[str] = Field(default_factory=list)
    disabled: list[str] = Field(default_factory=list)


class ArchetypeToolOverride(BaseModel):
    """Per-tool policy override within an archetype."""

    tool: str
    requires_confirmation: bool = False


class ArchetypePolicyDefaults(BaseModel):
    """Safety / confirmation policy defaults for an archetype."""

    confirmation_mode: PersonaConfirmationMode = "destructive_only"
    tool_overrides: list[ArchetypeToolOverride] = Field(default_factory=list)


class ArchetypeBuddyDefaults(BaseModel):
    """Visual buddy seed values for an archetype."""

    species: str | None = None
    palette: str | None = None
    silhouette: str | None = None


class ArchetypeStarterCommand(BaseModel):
    """A starter command: either a reference to a built-in template or a custom definition.

    Exactly one of ``template_key`` or ``custom`` must be provided.
    """

    template_key: str | None = None
    custom: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ArchetypeStarterCommand:
        has_template = self.template_key is not None
        has_custom = self.custom is not None
        if has_template == has_custom:
            raise ValueError("Exactly one of 'template_key' or 'custom' must be provided")
        return self


class ArchetypeSummary(BaseModel):
    """Compact representation shown in the archetype picker."""

    key: str
    label: str
    tagline: str
    icon: str


class ArchetypeTemplate(ArchetypeSummary):
    """Full archetype definition with all default configuration sections."""

    persona: ArchetypePersonaDefaults
    mcp_modules: ArchetypeMCPConfig = Field(default_factory=ArchetypeMCPConfig)
    suggested_external_servers: list[str] = Field(default_factory=list)
    policy: ArchetypePolicyDefaults = Field(default_factory=ArchetypePolicyDefaults)
    voice_defaults: dict[str, Any] = Field(default_factory=dict)
    scope_rules: list[dict[str, Any]] = Field(default_factory=list)
    buddy: ArchetypeBuddyDefaults = Field(default_factory=ArchetypeBuddyDefaults)
    starter_commands: list[ArchetypeStarterCommand] = Field(default_factory=list)


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


class MCPCatalogEntry(BaseModel):
    """One entry in the external MCP server catalog shown during setup."""

    key: str
    name: str
    description: str
    url_template: str
    auth_type: MCPAuthType = "none"
    category: str
    logo_key: str | None = None
    suggested_for: list[str] = Field(default_factory=list)


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
