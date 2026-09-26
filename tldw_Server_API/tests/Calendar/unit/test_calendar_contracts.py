"""Static documentation and typing contracts for Calendar schemas, persistence, and sync."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

API_ROOT = Path(__file__).resolve().parents[3]
DOCUMENTED_PATHS = (
    API_ROOT / "app/api/v1/schemas/calendar_schemas.py",
    API_ROOT / "app/core/DB_Management/Calendar_DB.py",
    API_ROOT / "app/core/Calendar/calendar_sync_worker.py",
)
DB_TEST_PATH = Path(__file__).with_name("test_calendar_db.py")


def _parse_module(path: Path) -> ast.Module:
    """Parse owned source without importing service dependencies or opening databases."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", DOCUMENTED_PATHS, ids=lambda path: path.name)
def test_calendar_owned_modules_have_docstrings(path: Path) -> None:
    """Require each owned production module to explain its domain purpose."""
    assert ast.get_docstring(_parse_module(path)), f"Missing module docstring: {path.name}"


@pytest.mark.parametrize("path", DOCUMENTED_PATHS, ids=lambda path: path.name)
def test_calendar_owned_definitions_have_docstrings(path: Path) -> None:
    """Keep documentation coverage scoped to reviewed definitions, including private helpers."""
    missing = [
        f"{node.name}:{node.lineno}"
        for node in ast.walk(_parse_module(path))
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and not ast.get_docstring(node)
    ]
    assert not missing, f"Undocumented definitions in {path.name}: {missing}"


@pytest.mark.parametrize(("path", "function_name", "parameters"), [
    (API_ROOT / "app/core/Calendar/recurrence.py", "_validate_timezone_rule", ("properties",)),
    (API_ROOT / "app/core/Calendar/providers/caldav.py", "_component_dates", (
        "component", "name", "resolved_timezones",
    )),
])
def test_reviewed_recurrence_helpers_document_parameters_returns_and_validation(
    path: Path, function_name: str, parameters: tuple[str, ...],
) -> None:
    """Keep the two reviewed private-helper contracts complete without importing providers."""
    function = next(
        node for node in ast.walk(_parse_module(path))
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    documentation = ast.get_docstring(function) or ""
    assert "Args:" in documentation and "Returns:" in documentation and "Raises:" in documentation
    arguments = documentation.split("Args:", 1)[1].split("Returns:", 1)[0]
    assert all(f"{parameter}:" in arguments for parameter in parameters)
    assert documentation.split("Returns:", 1)[1].split("Raises:", 1)[0].strip()
    assert "CalendarValidationError:" in documentation.split("Raises:", 1)[1]


def test_calendar_db_test_signatures_have_concrete_annotations() -> None:
    """Require fixture/helper types and complete test signatures rather than Any placeholders."""
    expected_parameters = {
        "tmp_path": "Path",
        "calendar_db": "CalendarDatabase",
        "db": "CalendarDatabase",
        "monkeypatch": "MonkeyPatch",
    }
    expected_returns = {"calendar_db": "CalendarDatabase", "_create_calendar": "CalendarRow"}
    violations: list[str] = []
    for node in ast.walk(_parse_module(DB_TEST_PATH)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments.extend(arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None)
        for argument in arguments:
            annotation = ast.unparse(argument.annotation) if argument.annotation is not None else None
            if annotation is None or any(
                isinstance(part, ast.Name) and part.id == "Any" for part in ast.walk(argument.annotation)
            ):
                violations.append(f"{node.name}.{argument.arg}: missing concrete annotation")
            elif argument.arg in expected_parameters and annotation != expected_parameters[argument.arg]:
                violations.append(f"{node.name}.{argument.arg}: expected {expected_parameters[argument.arg]}")
        returned = ast.unparse(node.returns) if node.returns is not None else None
        expected = "None" if node.name.startswith("test_") else expected_returns.get(node.name)
        if returned is None or (expected is not None and returned != expected):
            violations.append(f"{node.name}: expected return {expected or 'annotation'}")
    assert not violations, f"DB test signature violations: {violations}"


def test_calendar_db_tests_are_classified_unit_exactly_once() -> None:
    """Keep a single module unit marker so marker selection includes every DB regression."""
    module = _parse_module(DB_TEST_PATH)
    markers = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Attribute) and ast.unparse(node) == "pytest.mark.unit"
    ]
    assignments = [
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets)
    ]
    assert len(markers) == 1 and len(assignments) == 1 and ast.unparse(assignments[0]) == "pytest.mark.unit"
