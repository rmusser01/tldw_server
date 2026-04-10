"""Pydantic schemas for persona archetype templates."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArchetypePersona(BaseModel):
    name: str
    system_prompt: str | None = None
    personality_traits: list[str] = Field(default_factory=list)


class ArchetypeTemplate(BaseModel):
    key: str
    label: str
    tagline: str | None = None
    icon: str | None = None
    persona: ArchetypePersona | None = None


class ArchetypeSummary(BaseModel):
    key: str
    label: str
    tagline: str | None = None
    icon: str | None = None


__all__ = ["ArchetypePersona", "ArchetypeTemplate", "ArchetypeSummary"]
