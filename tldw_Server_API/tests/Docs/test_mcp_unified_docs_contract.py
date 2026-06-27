"""Documentation contracts for the Unified MCP user journey."""

from pathlib import Path

import pytest


def _read(path: str) -> str:
    """Read a repository-relative documentation file."""
    return Path(path).read_text(encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message."""
    if not condition:
        pytest.fail(message)


def test_unified_mcp_docs_state_embedded_today_standalone_planned() -> None:
    """Primary Unified MCP docs should state what ships today versus later."""
    docs = "\n\n".join(
        [
            _read("Docs/MCP/Unified/README.md"),
            _read("Docs/MCP/Unified/User_Guide.md"),
            _read("tldw_Server_API/app/core/MCP_unified/README.md"),
        ]
    ).lower()

    _require(
        "embedded in tldw server today" in docs,
        "Unified MCP docs should clearly say the current implementation is embedded in TLDW Server today.",
    )
    _require(
        "standalone gateway" in docs,
        "Unified MCP docs should mention the standalone gateway mental model explicitly.",
    )
    _require(
        "planned" in docs or "not yet shipped" in docs,
        "Unified MCP docs should distinguish planned standalone work from shipped behavior.",
    )
