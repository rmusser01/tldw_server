"""Tests that verify all starter archetype YAML files load correctly."""
from __future__ import annotations

import pathlib

import pytest

from tldw_Server_API.app.core.Persona.archetype_loader import (
    _CACHE,
    load_archetypes_from_directory,
)

pytestmark = pytest.mark.unit

ARCHETYPES_DIR = pathlib.Path("tldw_Server_API/Config_Files/persona_archetypes")

EXPECTED_KEYS = {
    "research_assistant",
    "study_buddy",
    "writing_coach",
    "project_manager",
    "roleplayer",
    "blank_canvas",
}


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure the module-level cache is empty before and after each test."""
    _CACHE.clear()
    yield
    _CACHE.clear()


@pytest.fixture()
def loaded_archetypes():
    """Load archetypes from the real config directory and return the result dict."""
    return load_archetypes_from_directory(ARCHETYPES_DIR)


class TestArchetypeYAMLFiles:
    def test_all_archetypes_load(self, loaded_archetypes):
        """All 6 expected archetype keys must be present after loading."""
        assert set(loaded_archetypes.keys()) == EXPECTED_KEYS

    @pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
    def test_archetype_has_required_persona_name(self, key, loaded_archetypes):
        """Each archetype must have a non-empty persona.name."""
        template = loaded_archetypes[key]
        assert template.persona.name, f"Archetype '{key}' has empty persona.name"

    def test_blank_canvas_has_empty_modules(self, loaded_archetypes):
        """blank_canvas must have no enabled modules and no starter commands."""
        bc = loaded_archetypes["blank_canvas"]
        assert bc.mcp_modules.enabled == []
        assert bc.starter_commands == []
