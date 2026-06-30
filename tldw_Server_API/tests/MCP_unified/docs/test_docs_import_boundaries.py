from __future__ import annotations

import ast
import importlib
from pathlib import Path


DOCS_PACKAGE_ROOT = Path("mcp_unified/docs")
FORBIDDEN_IMPORT_PREFIX = "tldw_Server_API"


def _import_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_docs_package_imports_without_host_or_web_dependencies() -> None:
    module = importlib.import_module("mcp_unified.docs")

    assert hasattr(module, "DocsSettings")  # nosec B101
    assert hasattr(module, "AccessScope")  # nosec B101


def test_docs_core_does_not_import_tldw_server_modules() -> None:
    violations: list[tuple[str, str]] = []
    for path in DOCS_PACKAGE_ROOT.rglob("*.py"):
        for name in _import_names(path):
            if name == FORBIDDEN_IMPORT_PREFIX or name.startswith(f"{FORBIDDEN_IMPORT_PREFIX}."):
                violations.append((str(path), name))

    assert violations == []  # nosec B101


def test_docs_package_does_not_import_optional_web_acquisition_dependencies() -> None:
    forbidden = {"playwright", "trafilatura", "requests", "aiohttp"}
    violations: list[tuple[str, str]] = []
    for path in DOCS_PACKAGE_ROOT.rglob("*.py"):
        for name in _import_names(path):
            root = name.split(".", 1)[0]
            if root in forbidden:
                violations.append((str(path), name))

    assert violations == []  # nosec B101
