"""
Archetype API Endpoints.

Provides list, detail, and preview access to persona archetype templates
loaded from YAML configuration files.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from tldw_Server_API.app.api.v1.schemas.archetype_schemas import (
    ArchetypePreviewResponse,
    ArchetypeSummary,
    ArchetypeTemplate,
)
from tldw_Server_API.app.core.MCP_unified.security.request_guards import enforce_http_security
from tldw_Server_API.app.core.Persona.archetype_loader import (
    get_archetype,
    list_archetypes,
)

router = APIRouter()


@router.get("", response_model=list[ArchetypeSummary])
async def list_persona_archetypes(
    _guard: None = Depends(enforce_http_security),
) -> list[ArchetypeSummary]:
    """Return summaries of all available archetypes."""
    return list_archetypes()


@router.get("/{key}", response_model=ArchetypeTemplate)
async def get_persona_archetype(
    key: str,
    _guard: None = Depends(enforce_http_security),
) -> ArchetypeTemplate:
    """Return full template for an archetype. 404 if not found."""
    tmpl = get_archetype(key)
    if tmpl is None:
        raise HTTPException(status_code=404, detail=f"Archetype '{key}' not found")
    return tmpl


@router.get("/{key}/preview", response_model=ArchetypePreviewResponse)
async def get_archetype_preview(
    key: str,
    _guard: None = Depends(enforce_http_security),
) -> ArchetypePreviewResponse:
    """Return a pre-filled dict from archetype for seeding wizard state.

    Returns: { name, system_prompt, archetype_key, voice_defaults, setup }
    404 if not found.
    """
    tmpl = get_archetype(key)
    if tmpl is None:
        raise HTTPException(status_code=404, detail=f"Archetype '{key}' not found")
    return ArchetypePreviewResponse(
        name=tmpl.persona.name,
        system_prompt=tmpl.persona.system_prompt,
        archetype_key=tmpl.key,
        voice_defaults=tmpl.voice_defaults,
    )
