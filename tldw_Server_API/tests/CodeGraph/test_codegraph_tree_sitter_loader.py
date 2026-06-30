from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def _require_parser(loader: Any, language_id: str) -> Any:
    result = loader.load_parser(language_id)
    if result.available:
        return result
    if result.missing:
        pytest.skip(f"{language_id} parser dependencies are not available: {', '.join(result.missing)}")
    pytest.skip(f"{language_id} parser is not available: {result.error or 'unknown error'}")


def test_require_parser_skips_when_optional_dependency_is_missing() -> None:
    loader = SimpleNamespace(
        load_parser=lambda language_id: SimpleNamespace(
            available=False,
            missing=(f"tree_sitter_{language_id}",),
            error=None,
        )
    )

    with pytest.raises(pytest.skip.Exception):
        _require_parser(loader, "java")


def test_loader_module_imports_without_eager_parser_imports() -> None:
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    assert isinstance(loader, ModuleType)


def test_load_parser_reports_missing_optional_dependency(monkeypatch) -> None:
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )
    real_import_module = loader.importlib.import_module

    def fake_import_module(name: str) -> ModuleType:
        if name == "tree_sitter_javascript":
            raise ModuleNotFoundError("No module named 'tree_sitter_javascript'")
        return real_import_module(name)

    monkeypatch.setattr(loader.importlib, "import_module", fake_import_module)

    result = loader.load_parser("javascript")

    assert result.parser is None
    assert result.missing == ("tree_sitter_javascript",)
    assert result.error is None


def test_load_parser_reports_optional_import_errors(monkeypatch) -> None:
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )
    real_import_module = loader.importlib.import_module

    def fake_import_module(name: str) -> ModuleType:
        if name == "tree_sitter_javascript":
            raise OSError("bad native extension")
        return real_import_module(name)

    monkeypatch.setattr(loader.importlib, "import_module", fake_import_module)

    result = loader.load_parser("javascript")

    assert result.parser is None
    assert result.missing == ()
    assert result.error == "Failed to import tree_sitter_javascript: bad native extension"


def test_javascript_parser_can_parse_exported_function() -> None:
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    result = _require_parser(loader, "javascript")
    tree = result.parser.parse(b"export function helper() { return 1; }")

    assert result.missing == ()
    assert result.error is None
    assert tree.root_node.type == "program"
    assert not tree.root_node.has_error


def test_typescript_parser_can_parse_interface() -> None:
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    result = _require_parser(loader, "typescript")
    tree = result.parser.parse(b"interface User { id: string }")

    assert result.missing == ()
    assert result.error is None
    assert tree.root_node.type == "program"
    assert not tree.root_node.has_error


def test_tsx_parser_can_parse_component() -> None:
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    result = _require_parser(loader, "tsx")
    tree = result.parser.parse(b"export function Card() { return <div />; }")

    assert result.missing == ()
    assert result.error is None
    assert tree.root_node.type == "program"
    assert not tree.root_node.has_error


def test_java_parser_can_parse_class() -> None:
    """Load the optional Java parser and parse a compact class fixture."""
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    result = _require_parser(loader, "java")
    tree = result.parser.parse(b"class Greeter { String greet() { return helper(); } }")

    assert result.missing == ()
    assert result.error is None
    assert tree.root_node.type == "program"
    assert not tree.root_node.has_error


def test_kotlin_parser_can_parse_class() -> None:
    """Load the optional Kotlin parser and parse a compact class fixture."""
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    result = _require_parser(loader, "kotlin")
    tree = result.parser.parse(b"class Greeter {\n fun greet(): String { return helper() }\n}")

    assert result.missing == ()
    assert result.error is None
    assert tree.root_node.type == "source_file"
    assert not tree.root_node.has_error


def test_csharp_parser_can_parse_class() -> None:
    """Load the optional C# parser and parse a compact class fixture."""
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    result = _require_parser(loader, "csharp")
    tree = result.parser.parse(b"class Greeter { string Greet() { return Helper(); } }")

    assert result.missing == ()
    assert result.error is None
    assert tree.root_node.type == "compilation_unit"
    assert not tree.root_node.has_error


def test_c_parser_can_parse_function() -> None:
    """Load the optional C parser and parse a compact function fixture."""
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    result = _require_parser(loader, "c")
    tree = result.parser.parse(b"int helper(int value) { return value + 1; }")

    assert result.missing == ()
    assert result.error is None
    assert tree.root_node.type == "translation_unit"
    assert not tree.root_node.has_error


def test_cpp_parser_can_parse_class() -> None:
    """Load the optional C++ parser and parse a compact class fixture."""
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    result = _require_parser(loader, "cpp")
    tree = result.parser.parse(b"class Greeter { int helper() { return 1; } };")

    assert result.missing == ()
    assert result.error is None
    assert tree.root_node.type == "translation_unit"
    assert not tree.root_node.has_error


def test_load_parser_reports_missing_java_kotlin_csharp_c_cpp_optional_dependencies(monkeypatch) -> None:
    """Report missing JVM/.NET/C-family parser packages without raising import errors."""
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )
    real_import_module = loader.importlib.import_module

    def fake_import_module(name: str) -> ModuleType:
        if name in {
            "tree_sitter_java",
            "tree_sitter_kotlin",
            "tree_sitter_c_sharp",
            "tree_sitter_c",
            "tree_sitter_cpp",
        }:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import_module(name)

    monkeypatch.setattr(loader.importlib, "import_module", fake_import_module)

    java = loader.load_parser("java")
    kotlin = loader.load_parser("kotlin")
    csharp = loader.load_parser("csharp")
    c = loader.load_parser("c")
    cpp = loader.load_parser("cpp")

    assert java.parser is None
    assert java.missing == ("tree_sitter_java",)
    assert java.error is None
    assert kotlin.parser is None
    assert kotlin.missing == ("tree_sitter_kotlin",)
    assert kotlin.error is None
    assert csharp.parser is None
    assert csharp.missing == ("tree_sitter_c_sharp",)
    assert csharp.error is None
    assert c.parser is None
    assert c.missing == ("tree_sitter_c",)
    assert c.error is None
    assert cpp.parser is None
    assert cpp.missing == ("tree_sitter_cpp",)
    assert cpp.error is None
