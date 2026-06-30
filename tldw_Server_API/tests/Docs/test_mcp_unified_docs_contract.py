"""Documentation contracts for the Unified MCP user journey."""

from pathlib import Path
import re

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
        "standalone package/gateway is planned but not shipped" in docs,
        "Unified MCP docs should explicitly tie the planned/not-shipped status to the standalone gateway.",
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


def test_unified_mcp_docs_use_supported_embedded_paths_for_smoke_and_snippets() -> None:
    """Client-facing quickstarts should default to the embedded TLDW MCP path."""
    docs = "\n\n".join(
        [
            _read("Docs/MCP/Unified/Client_Snippets.md"),
            _read("Docs/MCP/Unified/Smoke_Client.md"),
            _read("Docs/MCP/Unified/User_Guide.md"),
        ]
    )

    _require("/api/v1/mcp/status" in docs, "Docs should show embedded status path.")
    _require("/api/v1/mcp/request" in docs, "Docs should show embedded request path.")
    _require(
        "mcp_unified[gateway]" not in docs,
        "Docs should not install a non-root package path.",
    )


def test_unified_mcp_docs_label_every_package_local_request_example() -> None:
    """Package-local request examples must be framed as host-mounted/package-local."""
    smoke = _read("Docs/MCP/Unified/Smoke_Client.md")

    for line_no, line in enumerate(smoke.splitlines(), start=1):
        if "http://127.0.0.1:8000/mcp/request" not in line:
            continue
        window = "\n".join(smoke.splitlines()[max(0, line_no - 5) : line_no + 4]).lower()
        _require(
            "package-local" in window or "host-mounted" in window,
            f"Unlabeled package-local /mcp/request example near Smoke_Client.md:{line_no}",
        )


def test_unified_mcp_docs_have_path_decision_table() -> None:
    """Primary docs should distinguish embedded, package-local, and future paths."""
    docs = "\n\n".join(
        [
            _read("Docs/MCP/Unified/README.md"),
            _read("Docs/MCP/Unified/User_Guide.md"),
        ]
    ).lower()

    for snippet in (
        "which path should i use",
        "/api/v1/mcp/status",
        "/api/v1/mcp/request",
        "/mcp/status",
        "package-local",
        "standalone gateway",
        "planned but not shipped",
    ):
        _require(snippet in docs, f"Path decision docs should mention {snippet}.")


def test_unified_mcp_docs_do_not_normalize_query_token_auth() -> None:
    """Normal examples should keep query-string auth framed as legacy only."""
    docs = "\n\n".join(
        [
            _read("Docs/MCP/Unified/User_Guide.md"),
            _read("Docs/MCP/Unified/Client_Snippets.md"),
            _read("Docs/MCP/Unified/Smoke_Client.md"),
        ]
    ).lower()

    _require(
        "?token=jwt-token" not in docs,
        "Docs should not show query token auth as a normal example.",
    )
    _require(
        "disabled by default" in docs and "query auth" in docs,
        "Docs should frame query auth as legacy/disabled.",
    )


def test_unified_mcp_package_docs_do_not_promise_serve_or_publishing() -> None:
    """Package docs should not promise an unsupported standalone server product."""
    docs = "\n\n".join(
        [
            _read("apps/mcp-unified/README.md"),
            _read("apps/mcp-unified/USER_GUIDE.md"),
            _read("apps/mcp-unified/src/mcp_unified/README.md"),
        ]
    ).lower()

    forbidden = [
        "mcp-unified-gateway serve",
        "pip install mcp-unified",
        "published to pypi",
        "production standalone gateway",
    ]
    found = [phrase for phrase in forbidden if phrase in docs]
    _require(not found, f"Package docs imply unsupported standalone/published flows: {found}")
    _require("not published" in docs, "Package docs should state the package is not published.")
    _require(
        "internal" in docs and "experimental" in docs,
        "Package docs should state internal/experimental status.",
    )


def test_unified_mcp_docs_reference_existing_local_targets() -> None:
    """Relative markdown links in the high-traffic MCP docs should stay valid."""
    docs_to_check = [
        "Docs/MCP/Unified/README.md",
        "Docs/MCP/Unified/User_Guide.md",
        "Docs/MCP/Unified/Smoke_Client.md",
        "Docs/MCP/Unified/Client_Snippets.md",
        "apps/mcp-unified/README.md",
        "apps/mcp-unified/USER_GUIDE.md",
    ]
    missing: list[str] = []

    for doc_path in docs_to_check:
        text = _read(doc_path)
        for target in re.findall(r"\]\(([^)#][^)]+)\)", text):
            if (
                "://" in target
                or target.startswith("#")
                or target.startswith("mailto:")
                or target.startswith("/")
            ):
                continue
            normalized = target.split("#", 1)[0]
            if not normalized:
                continue
            candidate = Path(doc_path).parent / normalized
            if not candidate.exists():
                missing.append(f"{doc_path} -> {target}")

    _require(not missing, "Docs reference missing local targets: " + ", ".join(missing))


def test_unified_mcp_admin_env_docs_do_not_include_known_stale_mcp_vars() -> None:
    """Admin docs should not list stale MCP-only env vars as live config."""
    guide = _read("Docs/MCP/Unified/System_Admin_Guide.md")
    stale = {
        "MCP_HOST",
        "MCP_PORT",
        "MCP_AUTH_MODE",
        "MCP_MODULES_ENABLED",
        "MCP_DATABASE_MAX_OVERFLOW",
        "MCP_TRUSTED_PROXIES",
        "MCP_MAX_REQUEST_SIZE",
        "MCP_REQUEST_TIMEOUT",
    }
    found = {name for name in stale if name in guide}

    _require(not found, f"System admin guide documents stale MCP env vars: {sorted(found)}")
