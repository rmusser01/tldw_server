"""End-to-end test: list archetypes -> get preview -> verify analytics events."""
from __future__ import annotations

import pathlib

import pytest

import tldw_Server_API.app.core.Persona.archetype_loader as _loader_mod
import tldw_Server_API.app.core.MCP_unified.catalog_loader as _catalog_mod
from tldw_Server_API.app.core.Persona.archetype_loader import (
    load_archetypes_from_directory,
    get_archetype,
)
from tldw_Server_API.app.core.MCP_unified.catalog_loader import (
    load_mcp_catalog,
    list_catalog_entries,
)
from tldw_Server_API.app.api.v1.endpoints.archetype_endpoints import (
    list_persona_archetypes,
    get_persona_archetype,
    get_archetype_preview,
)
from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import (
    list_mcp_catalog,
)
from tldw_Server_API.app.api.v1.schemas.persona import PersonaSetupEventCreate

ARCHETYPES_DIR = pathlib.Path("tldw_Server_API/Config_Files/persona_archetypes")
CATALOG_PATH = pathlib.Path("tldw_Server_API/Config_Files/mcp_server_catalog.yaml")

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _load_data():
    _loader_mod._CACHE = {}
    _catalog_mod._CATALOG_CACHE = []
    load_archetypes_from_directory(ARCHETYPES_DIR)
    load_mcp_catalog(CATALOG_PATH)
    yield
    _loader_mod._CACHE = {}
    _catalog_mod._CATALOG_CACHE = []


@pytest.mark.asyncio
async def test_list_archetypes_returns_all_six():
    result = await list_persona_archetypes()
    assert len(result) == 6
    keys = {a.key for a in result}
    assert keys == {
        "research_assistant",
        "study_buddy",
        "writing_coach",
        "project_manager",
        "roleplayer",
        "blank_canvas",
    }


@pytest.mark.asyncio
async def test_get_archetype_detail():
    result = await get_persona_archetype("research_assistant")
    assert result.key == "research_assistant"
    assert result.persona.name == "Research Assistant"
    assert "media" in result.mcp_modules.enabled


@pytest.mark.asyncio
async def test_get_archetype_preview_shape():
    result = await get_archetype_preview("research_assistant")
    assert result["name"] == "Research Assistant"
    assert result["archetype_key"] == "research_assistant"
    assert "voice_defaults" in result
    assert result["setup"]["current_step"] == "archetype"


@pytest.mark.asyncio
async def test_archetype_not_found():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        await get_persona_archetype("nonexistent")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_mcp_catalog_available():
    entries = await list_mcp_catalog()
    assert len(entries) >= 5
    keys = {e.key for e in entries}
    assert "github" in keys
    assert "arxiv" in keys


def test_analytics_event_types_accepted():
    """Verify all new setup event types can be used in PersonaSetupEventCreate."""
    event_types = [
        "archetype_selected",
        "archetype_changed",
        "external_server_connected",
        "external_server_failed",
        "connection_test_initiated",
        "setup_skipped",
        "setup_resumed",
    ]
    for et in event_types:
        event = PersonaSetupEventCreate(
            event_id=f"e2e-{et}",
            run_id="e2e-run-001",
            event_type=et,
            step="archetype",
            metadata={"archetype_key": "research_assistant"},
        )
        assert event.event_type == et


def test_blank_canvas_archetype_is_empty():
    blank = get_archetype("blank_canvas")
    assert blank is not None
    assert blank.mcp_modules.enabled == []
    assert blank.starter_commands == []
