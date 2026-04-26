"""Tests for the archetype YAML loader service."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.schemas.archetype_schemas import (
    ArchetypeSummary,
    ArchetypeTemplate,
)
import tldw_Server_API.app.core.Persona.archetype_loader as _loader_mod
from tldw_Server_API.app.core.Persona.archetype_loader import (
    get_archetype,
    list_archetypes,
    load_archetypes_from_directory,
)

pytestmark = pytest.mark.unit

# -- Valid YAML content used across tests ------------------------------------

_VALID_YAML = textwrap.dedent("""\
    archetype:
      key: researcher
      label: Researcher
      tagline: Deep-dive into any topic
      icon: microscope
      persona:
        name: Research Assistant
        system_prompt: You are a research assistant.
        personality_traits:
          - analytical
          - thorough
""")

_VALID_YAML_2 = textwrap.dedent("""\
    archetype:
      key: creative
      label: Creative Writer
      tagline: Craft stories and ideas
      icon: pen
      persona:
        name: Creative Writer
""")


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure the module-level cache is empty before and after each test."""
    _loader_mod._CACHE = {}
    yield
    _loader_mod._CACHE = {}


# -- Tests -------------------------------------------------------------------


class TestLoadArchetypesFromDirectory:
    def test_single_valid_yaml(self, tmp_path: Path):
        (tmp_path / "researcher.yaml").write_text(_VALID_YAML, encoding="utf-8")

        result = load_archetypes_from_directory(tmp_path)

        assert len(result) == 1
        assert "researcher" in result
        tpl = result["researcher"]
        assert isinstance(tpl, ArchetypeTemplate)
        assert tpl.label == "Researcher"
        assert tpl.persona.name == "Research Assistant"

    def test_multiple_valid_yaml(self, tmp_path: Path):
        (tmp_path / "researcher.yaml").write_text(_VALID_YAML, encoding="utf-8")
        (tmp_path / "creative.yaml").write_text(_VALID_YAML_2, encoding="utf-8")

        result = load_archetypes_from_directory(tmp_path)

        assert len(result) == 2
        assert "researcher" in result
        assert "creative" in result

    def test_malformed_yaml_skipped(self, tmp_path: Path):
        # Write a valid file and a malformed one (invalid YAML syntax).
        (tmp_path / "good.yaml").write_text(_VALID_YAML, encoding="utf-8")
        (tmp_path / "bad.yaml").write_text("archetype: {{{invalid", encoding="utf-8")

        result = load_archetypes_from_directory(tmp_path)

        assert len(result) == 1
        assert "researcher" in result

    def test_yaml_missing_archetype_key_skipped(self, tmp_path: Path):
        (tmp_path / "nokey.yaml").write_text("something_else:\n  foo: bar\n", encoding="utf-8")

        result = load_archetypes_from_directory(tmp_path)

        assert len(result) == 0

    def test_yaml_with_validation_error_skipped(self, tmp_path: Path):
        # Missing required fields (label, tagline, icon).
        bad_content = textwrap.dedent("""\
            archetype:
              key: broken
        """)
        (tmp_path / "broken.yaml").write_text(bad_content, encoding="utf-8")

        result = load_archetypes_from_directory(tmp_path)

        assert len(result) == 0

    def test_empty_directory(self, tmp_path: Path):
        result = load_archetypes_from_directory(tmp_path)

        assert result == {}

    def test_nonexistent_directory(self, tmp_path: Path):
        result = load_archetypes_from_directory(tmp_path / "does_not_exist")

        assert result == {}

    def test_cache_is_cleared_on_reload(self, tmp_path: Path):
        (tmp_path / "researcher.yaml").write_text(_VALID_YAML, encoding="utf-8")
        load_archetypes_from_directory(tmp_path)
        assert len(_loader_mod._CACHE) == 1

        # Remove the file and reload -- cache should now be empty.
        (tmp_path / "researcher.yaml").unlink()
        load_archetypes_from_directory(tmp_path)
        assert len(_loader_mod._CACHE) == 0

    def test_non_yaml_files_ignored(self, tmp_path: Path):
        (tmp_path / "notes.txt").write_text("not yaml", encoding="utf-8")
        (tmp_path / "researcher.yaml").write_text(_VALID_YAML, encoding="utf-8")

        result = load_archetypes_from_directory(tmp_path)

        assert len(result) == 1

    def test_duplicate_keys_keep_first_loaded_entry(self, tmp_path: Path):
        (tmp_path / "01_researcher.yaml").write_text(_VALID_YAML, encoding="utf-8")
        duplicate = _VALID_YAML.replace("Researcher", "Duplicate Researcher", 1)
        (tmp_path / "02_duplicate.yaml").write_text(duplicate, encoding="utf-8")

        result = load_archetypes_from_directory(tmp_path)

        assert len(result) == 1
        assert result["researcher"].label == "Researcher"

    def test_returns_a_copy_of_the_cache(self, tmp_path: Path):
        (tmp_path / "researcher.yaml").write_text(_VALID_YAML, encoding="utf-8")

        result = load_archetypes_from_directory(tmp_path)
        result.pop("researcher")

        assert "researcher" in _loader_mod._CACHE

    def test_unexpected_exceptions_propagate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        (tmp_path / "researcher.yaml").write_text(_VALID_YAML, encoding="utf-8")

        def boom(_raw: str):
            raise RuntimeError("unexpected parser failure")

        monkeypatch.setattr(_loader_mod.yaml, "safe_load", boom)

        with pytest.raises(RuntimeError, match="unexpected parser failure"):
            load_archetypes_from_directory(tmp_path)


class TestListArchetypes:
    def test_returns_summaries(self, tmp_path: Path):
        (tmp_path / "researcher.yaml").write_text(_VALID_YAML, encoding="utf-8")
        (tmp_path / "creative.yaml").write_text(_VALID_YAML_2, encoding="utf-8")
        load_archetypes_from_directory(tmp_path)

        summaries = list_archetypes()

        assert len(summaries) == 2
        for s in summaries:
            assert isinstance(s, ArchetypeSummary)
        keys = {s.key for s in summaries}
        assert keys == {"researcher", "creative"}

    def test_empty_cache(self):
        assert list_archetypes() == []


class TestGetArchetype:
    def test_found(self, tmp_path: Path):
        (tmp_path / "researcher.yaml").write_text(_VALID_YAML, encoding="utf-8")
        load_archetypes_from_directory(tmp_path)

        result = get_archetype("researcher")

        assert result is not None
        assert result.key == "researcher"
        assert isinstance(result, ArchetypeTemplate)

    def test_not_found(self):
        assert get_archetype("nonexistent") is None
