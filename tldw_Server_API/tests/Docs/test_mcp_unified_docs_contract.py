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


def test_unified_mcp_quickstart_reaches_authenticated_tool_list() -> None:
    """The user guide should lead users to one verified first success."""
    guide = _read("Docs/MCP/Unified/User_Guide.md").lower()

    _require("golden path" in guide, "User guide should name a golden path quickstart.")
    _require(
        "authorization: bearer" in guide or "x-api-key" in guide,
        "Golden path should show a default supported auth header.",
    )
    _require("tools/list" in guide, "Golden path should include a tools/list request.")
    _require("tools/call" in guide, "Golden path should include a tools/call request.")
    _require(
        "expected response" in guide,
        "Golden path should include expected response shape so users can recognize success.",
    )


def test_unified_mcp_auth_matrix_exists() -> None:
    """Auth docs should classify methods by use, status, and transport."""
    guide = _read("Docs/MCP/Unified/User_Guide.md").lower()

    for snippet in (
        "auth method",
        "best for",
        "default status",
        "recommended",
        "disabled by default",
        "query",
    ):
        _require(snippet in guide, f"Auth matrix should mention {snippet}.")


def test_unified_mcp_catalog_miss_recovery_is_documented() -> None:
    """Catalog docs should explain misspelled or invisible catalog recovery."""
    docs = "\n\n".join(
        [
            _read("Docs/MCP/Unified/User_Guide.md"),
            _read("Docs/MCP/Unified/Developer_Guide.md"),
            _read("Docs/MCP/Unified/Client_Snippets.md"),
        ]
    ).lower()

    _require(
        "_meta.catalog.status" in docs,
        "Catalog docs should tell clients where catalog resolution status is returned.",
    )
    _require(
        "unresolved" in docs,
        "Catalog docs should name the unresolved catalog state.",
    )
    _require(
        "catalog_fail_open" in docs,
        "Catalog docs should document the explicit diagnostic fail-open escape hatch.",
    )


def test_unified_mcp_diagnostics_recovery_is_documented() -> None:
    """Troubleshooting docs should point users to status diagnostics."""
    guide = _read("Docs/MCP/Unified/User_Guide.md").lower()

    for snippet in (
        "problem_modules",
        "config_warnings",
        "invalid_safe_config",
        "query auth",
        "empty tool list",
    ):
        _require(snippet in guide, f"Diagnostics docs should mention {snippet}.")


def test_unified_mcp_operator_cheatsheet_covers_power_user_workflows() -> None:
    """The operator cheatsheet should cover common repeat-use MCP workflows."""
    text = _read("Docs/MCP/Unified/Operator_Cheatsheet.md").lower()

    for phrase in (
        "tools/list",
        "tools/call",
        "batch",
        "session",
        "catalog",
        "metrics",
        "status",
        "websocket",
    ):
        _require(phrase in text, f"Operator cheatsheet should mention {phrase}.")
