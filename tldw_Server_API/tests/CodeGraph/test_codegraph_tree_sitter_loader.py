from __future__ import annotations

import importlib
from types import ModuleType


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

    def fake_import_module(name: str):
        if name == "tree_sitter_javascript":
            raise ModuleNotFoundError("No module named 'tree_sitter_javascript'")
        return real_import_module(name)

    monkeypatch.setattr(loader.importlib, "import_module", fake_import_module)

    result = loader.load_parser("javascript")

    assert result.parser is None
    assert result.missing == ("tree_sitter_javascript",)
    assert result.error is None


def test_javascript_parser_can_parse_exported_function() -> None:
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    result = loader.load_parser("javascript")
    tree = result.parser.parse(b"export function helper() { return 1; }")

    assert result.missing == ()
    assert result.error is None
    assert tree.root_node.type == "program"
    assert not tree.root_node.has_error


def test_typescript_parser_can_parse_interface() -> None:
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    result = loader.load_parser("typescript")
    tree = result.parser.parse(b"interface User { id: string }")

    assert result.missing == ()
    assert result.error is None
    assert tree.root_node.type == "program"
    assert not tree.root_node.has_error


def test_tsx_parser_can_parse_component() -> None:
    loader = importlib.import_module(
        "tldw_Server_API.app.core.CodeGraph.extractors.tree_sitter_loader"
    )

    result = loader.load_parser("tsx")
    tree = result.parser.parse(b"export function Card() { return <div />; }")

    assert result.missing == ()
    assert result.error is None
    assert tree.root_node.type == "program"
    assert not tree.root_node.has_error
