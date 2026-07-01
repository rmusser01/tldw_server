from __future__ import annotations

from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
MCP_UNIFIED_PYPROJECT = REPO_ROOT / "apps" / "mcp-unified" / "pyproject.toml"


def test_docs_web_extra_installs_rich_html_extractors() -> None:
    data = tomllib.loads(MCP_UNIFIED_PYPROJECT.read_text(encoding="utf-8"))
    optional_dependencies = data["project"]["optional-dependencies"]

    docs_web = set(optional_dependencies["docs-web"])

    assert "beautifulsoup4>=4.12.0" in docs_web  # nosec B101
    assert "trafilatura>=1.6.0" in docs_web  # nosec B101
