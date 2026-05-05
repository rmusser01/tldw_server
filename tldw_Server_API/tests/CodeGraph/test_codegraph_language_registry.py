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
    assert by_id["java"].stage == "foundation"
    assert by_id["kotlin"].stage == "foundation"
    assert by_id["csharp"].stage == "foundation"
    assert by_id["c"].stage == "foundation"
    assert by_id["cpp"].stage == "foundation"


def test_registry_maps_extensions_and_reports_symbol_extraction_support() -> None:
    registry = CodeGraphLanguageRegistry(
        dependency_health=DependencyHealth(
            available=True,
            missing=(),
            present=(
                "tree_sitter",
                "tree_sitter_python",
                "tree_sitter_javascript",
                "tree_sitter_typescript",
                "tree_sitter_java",
                "tree_sitter_kotlin",
                "tree_sitter_c_sharp",
                "tree_sitter_c",
                "tree_sitter_cpp",
            ),
        )
    )

    assert registry.language_for_path("api/server.py").language_id == "python"
    assert registry.language_for_path("apps/ui/page.tsx").language_id == "typescript"
    assert registry.language_for_path("apps/ui/component.jsx").language_id == "javascript"
    assert registry.language_for_path("src/main/java/com/example/App.java").language_id == "java"
    assert registry.language_for_path("src/main/kotlin/com/example/App.kt").language_id == "kotlin"
    assert registry.language_for_path("src/main/csharp/Example/App.cs").language_id == "csharp"
    assert registry.language_for_path("src/main.c").language_id == "c"
    assert registry.language_for_path("src/main.cc").language_id == "cpp"
    assert registry.language_for_path("include/main.hpp").language_id == "cpp"
    assert registry.language_for_path("README.md") is None

    by_id = {language.language_id: language for language in registry.list_languages()}

    assert by_id["python"].symbol_extraction is True
    assert by_id["javascript"].symbol_extraction is True
    assert by_id["typescript"].symbol_extraction is True
    assert by_id["java"].symbol_extraction is True
    assert by_id["kotlin"].symbol_extraction is True
    assert by_id["csharp"].symbol_extraction is True
    assert by_id["c"].symbol_extraction is True
    assert by_id["cpp"].symbol_extraction is True


def test_registry_reports_missing_parser_dependencies_per_language() -> None:
    registry = CodeGraphLanguageRegistry(
        dependency_health=DependencyHealth(
            available=False,
            missing=(
                "tree_sitter_javascript",
                "tree_sitter_java",
                "tree_sitter_kotlin",
                "tree_sitter_c_sharp",
                "tree_sitter_c",
                "tree_sitter_cpp",
            ),
            present=("tree_sitter", "tree_sitter_typescript"),
        )
    )

    by_id = {language.language_id: language for language in registry.list_languages()}

    assert by_id["python"].symbol_extraction is True
    assert by_id["javascript"].symbol_extraction is False
    assert by_id["javascript"].dependency_missing == ("tree_sitter_javascript",)
    assert by_id["typescript"].symbol_extraction is True
    assert by_id["typescript"].dependency_missing == ()
    assert by_id["java"].symbol_extraction is False
    assert by_id["java"].dependency_missing == ("tree_sitter_java",)
    assert by_id["kotlin"].symbol_extraction is False
    assert by_id["kotlin"].dependency_missing == ("tree_sitter_kotlin",)
    assert by_id["csharp"].symbol_extraction is False
    assert by_id["csharp"].dependency_missing == ("tree_sitter_c_sharp",)
    assert by_id["c"].symbol_extraction is False
    assert by_id["c"].dependency_missing == ("tree_sitter_c",)
    assert by_id["cpp"].symbol_extraction is False
    assert by_id["cpp"].dependency_missing == ("tree_sitter_cpp",)
