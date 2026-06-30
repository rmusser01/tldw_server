from __future__ import annotations

import ast
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_generation_service_does_not_import_workflows_adapter_common():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "core"
        / "StudyPacks"
        / "generation_service.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "tldw_Server_API.app.core.Workflows.adapters._common" not in imported_modules  # nosec B101
