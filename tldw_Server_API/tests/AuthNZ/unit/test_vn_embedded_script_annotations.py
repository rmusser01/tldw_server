"""Guard executable VN test scripts' annotation and docstring contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _missing_helper_contracts(source: str) -> list[str]:
    """Find missing signatures or docstrings, including nested async callbacks."""
    missing: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.returns is None:
            missing.append(f"{node.name}: return annotation")
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        for argument in arguments:
            if argument.annotation is None:
                missing.append(f"{node.name}: {argument.arg} annotation")
        if not (ast.get_docstring(node) or "").strip():
            missing.append(f"{node.name}: docstring")
    return missing


@pytest.mark.parametrize("script_name", ["REGISTRATION_SCRIPT", "STORAGE_SCRIPT"])
def test_vn_embedded_helpers_have_complete_annotations_and_docstrings(script_name: str) -> None:
    """Parse and compile the actual scripts without running their DB lifecycle."""
    path = Path(__file__).resolve().parents[1] / "integration" / "test_vn_generated_file_idempotency.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    scripts = {
        target.id: ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {"REGISTRATION_SCRIPT", "STORAGE_SCRIPT"}
    }
    source = scripts[script_name]
    compile(source, f"{path}:{script_name}", "exec")
    assert _missing_helper_contracts(source) == []


def test_contract_check_detects_all_argument_kinds_in_nested_async_helpers() -> None:
    """Ensure positional-only, keyword-only and variadic gaps cannot escape."""
    source = """
def outer():
    async def callback(positional, /, normal, *args, keyword, **kwargs):
        pass
"""
    assert _missing_helper_contracts(source) == [
        "outer: return annotation",
        "outer: docstring",
        "callback: return annotation",
        "callback: positional annotation",
        "callback: normal annotation",
        "callback: keyword annotation",
        "callback: args annotation",
        "callback: kwargs annotation",
        "callback: docstring",
    ]


def test_contract_check_accepts_complete_nested_sync_and_async_helpers() -> None:
    """Accept fully documented signatures without imposing callback value types."""
    source = '''
async def outer() -> None:
    """Own the callback lifecycle."""
    def callback(positional: int, /, normal: str, *args: object, keyword: bool, **kwargs: object) -> None:
        """Accept dynamically supplied callback arguments."""
        pass
'''
    assert _missing_helper_contracts(source) == []
