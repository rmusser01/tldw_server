"""
Guardrail tests for AuthNZ admin CLI module docstrings.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_TARGET_MODULES = (
    ("create_admin.py", "--username myadmin --password 'S3cureP@ss!' [--email admin@example.com]"),
    ("reset_admin_password.py", "--username admin --new-password 'N3wS3cure!'"),
)


@pytest.mark.unit
def test_admin_cli_modules_define_module_docstrings() -> None:
    project_root = Path(__file__).resolve().parents[3]

    missing_docstrings: list[str] = []

    for module_name, _ in _TARGET_MODULES:
        path = project_root / "app" / "core" / "AuthNZ" / module_name
        module_ast = ast.parse(path.read_text(encoding="utf-8"))
        module_docstring = ast.get_docstring(module_ast, clean=False)
        if not module_docstring or not module_docstring.strip():
            missing_docstrings.append(path.relative_to(project_root).as_posix())

    assert missing_docstrings == [], (  # nosec B101
        "Expected AuthNZ admin CLI modules to define non-empty module docstrings: "
        f"{missing_docstrings}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(("module_name", "expected_args"), _TARGET_MODULES)
def test_admin_cli_docstrings_preserve_multiline_usage_examples(
    module_name: str,
    expected_args: str,
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    path = project_root / "app" / "core" / "AuthNZ" / module_name

    module_ast = ast.parse(path.read_text(encoding="utf-8"))
    module_docstring = ast.get_docstring(module_ast, clean=False)
    assert module_docstring is not None  # nosec B101

    doc_lines = module_docstring.splitlines()
    usage_index = doc_lines.index("Usage:")

    assert doc_lines[usage_index + 1].rstrip().endswith("\\")  # nosec B101
    assert doc_lines[usage_index + 2].strip() == expected_args  # nosec B101
