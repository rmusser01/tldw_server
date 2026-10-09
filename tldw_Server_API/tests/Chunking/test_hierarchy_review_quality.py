"""Documentation, typing, and category contracts for extracted hierarchy code."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

HIERARCHY = Path(__file__).parents[2] / "app" / "core" / "Chunking" / "hierarchical"
HIERARCHY_TESTS = (
    "test_hierarchical_builder.py",
    "test_hierarchical_flatten.py",
    "test_hierarchical_grouping.py",
    "test_hierarchical_leaves.py",
    "test_hierarchical_spans.py",
    "test_hierarchy_malformed_contracts.py",
    "test_hierarchy_refactor_contracts.py",
)


def parse_source(path: Path) -> ast.Module:
    """Parse a component or contract module without importing its dependencies."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_hierarchy_models_document_context_and_value_contracts() -> None:
    """Every model and protocol member must expose a nonempty docstring."""
    source = parse_source(HIERARCHY / "models.py")
    missing = [
        getattr(node, "name", "module")
        for node in ast.walk(source)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and not ast.get_docstring(node)
    ]
    assert not missing, f"Undocumented hierarchy contracts: {missing}"


def test_service_documents_build_arguments_result_and_input_error() -> None:
    """The service contract must describe its option inputs and return envelope."""
    source = parse_source(HIERARCHY / "service.py")
    build_tree = next(
        node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == "build_tree"
    )
    documentation = ast.get_docstring(build_tree) or ""
    required = ["Args:", "Returns:", "Raises:", "InvalidInputError", "root", "schema_version"]
    required.extend(f"{argument.arg}:" for argument in build_tree.args.args if argument.arg != "self")
    assert all(term in documentation for term in required)


@pytest.mark.parametrize("helper", ["_append_with_titles", "walk"])
def test_flatten_side_effect_helpers_declare_none_return(helper: str) -> None:
    """Nested output-mutating helpers must declare their no-result contract."""
    source = parse_source(HIERARCHY / "flatten.py")
    function = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef) and node.name == helper)
    assert isinstance(function.returns, ast.Constant) and function.returns.value is None


def test_grouping_contract_tests_have_complete_annotations() -> None:
    """Grouping test inputs, including patched callbacks, must be typed."""
    source = parse_source(Path(__file__).with_name("test_hierarchical_grouping.py"))
    missing: list[str] = []
    for function in (node for node in ast.walk(source) if isinstance(node, ast.FunctionDef)):
        arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        arguments.extend(argument for argument in (function.args.vararg, function.args.kwarg) if argument is not None)
        missing.extend(f"{function.name}.{argument.arg}" for argument in arguments if argument.annotation is None)
        if function.returns is None:
            missing.append(f"{function.name}.return")
    assert not missing, f"Untyped grouping contracts: {missing}"


@pytest.mark.parametrize("filename", HIERARCHY_TESTS)
def test_hierarchy_component_contracts_are_unit_tests(filename: str) -> None:
    """Pure hierarchy contracts must participate in unit-category selection."""
    source = parse_source(Path(__file__).with_name(filename))
    assignments = [
        node.value
        for node in source.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
    ]
    assert any(ast.unparse(value) == "pytest.mark.unit" for value in assignments)
