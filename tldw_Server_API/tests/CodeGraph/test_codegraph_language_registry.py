from __future__ import annotations

from tldw_Server_API.app.core.CodeGraph.dependencies import DependencyHealth
from tldw_Server_API.app.core.CodeGraph.language_registry import CodeGraphLanguageRegistry
from tldw_Server_API.app.core.CodeGraph.models import make_edge_id, make_node_id


def test_node_id_is_deterministic_for_same_identity() -> None:
    first = make_node_id("ws", "python", "app/main.py", "function", "main", 10)
    second = make_node_id("ws", "python", "app/main.py", "function", "main", 10)

    assert first == second
    assert first.startswith("node_")


def test_edge_id_changes_when_target_changes() -> None:
    first = make_edge_id("node_a", "calls", "node_b", "app/main.py", 12, 4)
    second = make_edge_id("node_a", "calls", "node_c", "app/main.py", 12, 4)

    assert first != second


def test_node_id_uses_unambiguous_identity_encoding() -> None:
    first = make_node_id("ws", "python:extra", "app/main.py", "function", "main", 10)
    second = make_node_id("ws:python", "extra", "app/main.py", "function", "main", 10)

    assert first != second


def test_registry_reports_foundation_languages_and_planned_languages() -> None:
    registry = CodeGraphLanguageRegistry(
        dependency_health=DependencyHealth(available=False, missing=("tree_sitter",), present=())
    )

    by_id = {language.language_id: language for language in registry.list_languages()}

    assert by_id["python"].stage == "foundation"
    assert by_id["javascript"].stage == "foundation"
    assert by_id["typescript"].stage == "foundation"
    assert by_id["c"].stage == "planned"
    assert by_id["cpp"].stage == "planned"
    assert by_id["csharp"].stage == "planned"
    assert by_id["java"].stage == "planned"
    assert by_id["kotlin"].stage == "planned"


def test_registry_maps_extensions_without_claiming_symbol_extraction() -> None:
    registry = CodeGraphLanguageRegistry()

    assert registry.language_for_path("api/server.py").language_id == "python"
    assert registry.language_for_path("apps/ui/page.tsx").language_id == "typescript"
    assert registry.language_for_path("apps/ui/component.jsx").language_id == "javascript"
    assert registry.language_for_path("src/main.cc").stage == "planned"
    assert registry.language_for_path("README.md") is None
