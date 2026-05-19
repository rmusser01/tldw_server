"""Tests for the MCP server catalog YAML loader service."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.schemas.archetype_schemas import MCPCatalogEntry
import tldw_Server_API.app.core.MCP_unified.catalog_loader as _catalog_mod
from tldw_Server_API.app.core.MCP_unified.catalog_loader import (
    get_catalog_entry,
    list_catalog_entries,
    load_mcp_catalog,
)

pytestmark = pytest.mark.unit

# -- Valid YAML content used across tests ------------------------------------

_TWO_ENTRY_YAML = textwrap.dedent("""\
    catalog:
      - key: github
        name: GitHub
        description: Repositories, issues, PRs, and code search
        url_template: https://api.github.com
        auth_type: bearer
        category: development
        logo_key: github
        suggested_for:
          - research_assistant
          - project_manager

      - key: arxiv
        name: arXiv
        description: Academic paper search and retrieval
        url_template: https://export.arxiv.org/api
        auth_type: none
        category: research
        logo_key: arxiv
        suggested_for:
          - research_assistant
          - study_buddy
""")

_ONE_GOOD_ONE_BAD_YAML = textwrap.dedent("""\
    catalog:
      - key: github
        name: GitHub
        description: Repositories, issues, PRs, and code search
        url_template: https://api.github.com
        auth_type: bearer
        category: development

      - key: broken
        name: 12345
""")


# -- Fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure the module-level cache is empty before and after each test."""
    _catalog_mod._CATALOG_CACHE = []
    yield
    _catalog_mod._CATALOG_CACHE = []


@pytest.fixture()
def catalog_file(tmp_path: Path) -> Path:
    """Create a temporary catalog YAML file with two valid entries."""
    p = tmp_path / "mcp_server_catalog.yaml"
    p.write_text(_TWO_ENTRY_YAML, encoding="utf-8")
    return p


# -- Tests -------------------------------------------------------------------


class TestLoadMcpCatalog:
    def test_load_catalog(self, catalog_file: Path):
        result = load_mcp_catalog(catalog_file)

        assert len(result) == 2
        assert all(isinstance(e, MCPCatalogEntry) for e in result)
        keys = {e.key for e in result}
        assert keys == {"github", "arxiv"}

    def test_cache_is_populated(self, catalog_file: Path):
        load_mcp_catalog(catalog_file)

        assert len(_catalog_mod._CATALOG_CACHE) == 2

    def test_cache_is_cleared_on_reload(self, catalog_file: Path):
        load_mcp_catalog(catalog_file)
        assert len(_catalog_mod._CATALOG_CACHE) == 2

        # Overwrite with a single-entry file and reload.
        catalog_file.write_text(
            textwrap.dedent("""\
                catalog:
                  - key: solo
                    name: Solo
                    description: Only entry
                    url_template: https://example.com
                    auth_type: none
                    category: test
            """),
            encoding="utf-8",
        )
        load_mcp_catalog(catalog_file)
        assert len(_catalog_mod._CATALOG_CACHE) == 1

    def test_returns_a_copy_of_the_cache(self, catalog_file: Path):
        result = load_mcp_catalog(catalog_file)

        result.pop()

        assert len(_catalog_mod._CATALOG_CACHE) == 2


class TestListCatalogEntries:
    def test_returns_all_entries(self, catalog_file: Path):
        load_mcp_catalog(catalog_file)

        entries = list_catalog_entries()

        assert len(entries) == 2

    def test_filter_by_archetype_key(self, catalog_file: Path):
        load_mcp_catalog(catalog_file)

        # Both github and arxiv suggest "research_assistant"
        entries = list_catalog_entries(archetype_key="research_assistant")
        assert len(entries) == 2

        # Only github suggests "project_manager"
        entries = list_catalog_entries(archetype_key="project_manager")
        assert len(entries) == 1
        assert entries[0].key == "github"

    def test_filter_returns_empty_for_unknown_archetype(self, catalog_file: Path):
        load_mcp_catalog(catalog_file)

        entries = list_catalog_entries(archetype_key="nonexistent_archetype")
        assert entries == []

    def test_empty_cache(self):
        assert list_catalog_entries() == []


class TestGetCatalogEntry:
    def test_found(self, catalog_file: Path):
        load_mcp_catalog(catalog_file)

        entry = get_catalog_entry("github")

        assert entry is not None
        assert entry.key == "github"
        assert entry.name == "GitHub"
        assert isinstance(entry, MCPCatalogEntry)

    def test_not_found(self, catalog_file: Path):
        load_mcp_catalog(catalog_file)

        assert get_catalog_entry("nonexistent") is None

    def test_not_found_empty_cache(self):
        assert get_catalog_entry("anything") is None


class TestMalformedEntries:
    def test_malformed_entry_skipped(self, tmp_path: Path):
        """One good entry + one entry missing required fields => only good loads."""
        p = tmp_path / "catalog.yaml"
        p.write_text(_ONE_GOOD_ONE_BAD_YAML, encoding="utf-8")

        result = load_mcp_catalog(p)

        assert len(result) == 1
        assert result[0].key == "github"

    def test_missing_file(self, tmp_path: Path):
        result = load_mcp_catalog(tmp_path / "does_not_exist.yaml")

        assert result == []

    def test_invalid_yaml_syntax(self, tmp_path: Path):
        p = tmp_path / "bad.yaml"
        p.write_text("catalog: {{{invalid", encoding="utf-8")

        result = load_mcp_catalog(p)

        assert result == []

    def test_unexpected_exceptions_propagate(self, catalog_file: Path, monkeypatch: pytest.MonkeyPatch):
        def boom(_raw: str):
            raise RuntimeError("unexpected parser failure")

        monkeypatch.setattr(_catalog_mod.yaml, "safe_load", boom)

        with pytest.raises(RuntimeError, match="unexpected parser failure"):
            load_mcp_catalog(catalog_file)

    def test_missing_catalog_key(self, tmp_path: Path):
        p = tmp_path / "nokey.yaml"
        p.write_text("something_else:\n  foo: bar\n", encoding="utf-8")

        result = load_mcp_catalog(p)

        assert result == []
