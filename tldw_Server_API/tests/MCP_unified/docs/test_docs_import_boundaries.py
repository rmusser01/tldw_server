from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
DOCS_PACKAGE_ROOT = REPO_ROOT / "apps/mcp-unified/src/mcp_unified/docs"
FORBIDDEN_IMPORT_PREFIX = "tldw_Server_API"
pytestmark = pytest.mark.unit


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


def test_standalone_mount_imports_without_host_dependencies() -> None:
    module = importlib.import_module("mcp_unified.docs.standalone")

    assert hasattr(module, "create_standalone_docs_mount")  # nosec B101
    assert hasattr(module, "StandaloneDocsProfile")  # nosec B101


def test_docs_public_exports_include_standalone_mount() -> None:
    module = importlib.import_module("mcp_unified.docs")

    assert hasattr(module, "create_standalone_docs_mount")  # nosec B101
    assert hasattr(module, "StandaloneDocsProfile")  # nosec B101


def test_docs_core_does_not_import_tldw_server_modules() -> None:
    violations: list[tuple[str, str]] = []
    for path in DOCS_PACKAGE_ROOT.rglob("*.py"):
        for name in _import_names(path):
            if name == FORBIDDEN_IMPORT_PREFIX or name.startswith(f"{FORBIDDEN_IMPORT_PREFIX}."):
                violations.append((str(path), name))

    assert violations == []  # nosec B101


def test_docs_package_does_not_import_optional_web_acquisition_dependencies() -> None:
    forbidden = {"playwright", "trafilatura", "requests", "aiohttp", "httpx", "bs4"}
    violations: list[tuple[str, str]] = []
    for path in DOCS_PACKAGE_ROOT.rglob("*.py"):
        for name in _import_names(path):
            root = name.split(".", 1)[0]
            if root in forbidden:
                violations.append((str(path), name))

    assert violations == []  # nosec B101


def test_docs_package_import_does_not_load_rich_extractors() -> None:
    for name in list(sys.modules):
        if name == "mcp_unified.docs" or name.startswith("mcp_unified.docs."):
            sys.modules.pop(name, None)
    for name in ("trafilatura", "bs4"):
        sys.modules.pop(name, None)

    importlib.import_module("mcp_unified.docs")

    assert "trafilatura" not in sys.modules  # nosec B101
    assert "bs4" not in sys.modules  # nosec B101
