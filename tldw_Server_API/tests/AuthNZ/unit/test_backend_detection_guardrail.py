"""
Guardrail audit for backend routing in AuthNZ core modules.

This test ensures AuthNZ backend routing does not regress to connection
capability probing (for example, ``hasattr(conn, "fetchrow")``), which can
misclassify shim/wrapper connections. Backend selection should use
``DatabasePool`` state instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


@pytest.mark.unit
def test_authnz_core_avoids_connection_capability_backend_probing():
    project_root = Path(__file__).resolve().parents[3]
    roots = [project_root / "app" / "core" / name for name in ("AuthNZ", "Provider_Credentials")]
    for root in roots:
        assert root.exists(), f"Expected core path not found: {root}"
    files = [path for root in roots for path in root.rglob("*.py")]
    files += [project_root / "app" / "core" / "Usage" / f"llm_budget_{name}.py" for name in ("guard", "middleware")]
    assert project_root / "app" / "core" / "Provider_Credentials" / "byok_runtime.py" in files
    assert project_root / "app" / "core" / "Usage" / "llm_budget_guard.py" in files

    forbidden_patterns = (
        re.compile(r"hasattr\(\s*conn\s*,\s*['\"]fetch['\"]\s*\)"),
        re.compile(r"hasattr\(\s*conn\s*,\s*['\"]fetchrow['\"]\s*\)"),
        re.compile(r"hasattr\(\s*conn\s*,\s*['\"]fetchval['\"]\s*\)"),
        re.compile(r"hasattr\(\s*conn\s*,\s*['\"]execute['\"]\s*\)"),
    )

    offending: list[str] = []

    for path in files:
        rel = path.relative_to(project_root).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, start=1):
            for pattern in forbidden_patterns:
                if pattern.search(line):
                    offending.append(f"{rel}:{lineno}:{line.strip()}")

    assert offending == [], (
        "Found forbidden connection-capability backend probes in AuthNZ core: "
        f"{offending}"
    )
