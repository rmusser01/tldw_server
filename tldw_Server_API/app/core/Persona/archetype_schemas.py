"""Core-owned archetype template models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PersonaConfirmationMode = Literal["always", "destructive_only", "never"]


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
