from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.browser_cdp_module import (
    BrowserCDPModule,
)


def _module(settings: dict[str, Any] | None = None) -> BrowserCDPModule:
    return BrowserCDPModule(
        ModuleConfig(
            name="browser_cdp",
            settings=settings
            or {
                "debugger_url": "http://127.0.0.1:9222",
                "max_snapshot_nodes": 10,
                "max_events": 5,
                "max_observation_window_ms": 1_000,
            },
        )
    )


@pytest.mark.asyncio
async def test_browser_cdp_tools_include_read_only_metadata() -> None:
    module = _module()

    tools = await module.get_tools()
    by_name = {tool["name"]: tool for tool in tools}

    expected_tools = {
        "browser.status",
        "browser.pages.list",
        "browser.snapshot",
        "browser.page_state",
        "browser.screenshot",
        "browser.console",
        "browser.network",
    }
    assert expected_tools <= set(by_name)  # nosec B101

    for tool_name in expected_tools:
        tool = by_name[tool_name]
        schema = tool["inputSchema"]
        metadata = tool["metadata"]
        assert schema["additionalProperties"] is False  # nosec B101
        assert metadata["category"] == "browser"  # nosec B101
        assert metadata["readOnlyHint"] is True  # nosec B101
        assert metadata["uses_browser"] is True  # nosec B101
        assert metadata["cdp_backend"] is True  # nosec B101
        assert any(  # nosec B101
            capability in metadata["capabilities"]
            for capability in {
                "browser.inspect",
                "browser.debug",
                "screenshots.capture",
                "app_state.read",
            }
        )


def test_browser_cdp_validates_read_only_tool_arguments() -> None:
    module = _module()

    valid_cases = [
        ("browser.status", {}),
        ("browser.pages.list", {}),
        ("browser.snapshot", {"target_id": "page-1", "limit": 10}),
        ("browser.page_state", {"target_id": "page-1"}),
        ("browser.screenshot", {"target_id": "page-1", "format": "jpeg", "quality": 80}),
        ("browser.console", {"target_id": "page-1", "window_ms": 100, "max_events": 3}),
        ("browser.network", {"target_id": "page-1", "window_ms": 100, "max_events": 3}),
    ]
    for tool_name, args in valid_cases:
        module.validate_tool_arguments(tool_name, args)

    invalid_cases = [
        ("browser.missing", {}, "Unknown browser CDP tool"),
        ("browser.status", {"target_id": "page-1"}, "unknown arguments"),
        ("browser.snapshot", {"target_id": ""}, "target_id must be a non-empty string"),
        ("browser.snapshot", {"target_id": 7}, "target_id must be a non-empty string"),
        ("browser.snapshot", {"limit": 0}, "limit must be a positive integer"),
        ("browser.snapshot", {"limit": 11}, "limit exceeds maximum"),
        ("browser.snapshot", {"url": "http://example.com"}, "unknown arguments"),
        ("browser.snapshot", {"script": "document.body.remove()"}, "unknown arguments"),
        ("browser.page_state", {"expression": "location.href"}, "unknown arguments"),
        ("browser.screenshot", {"format": "webp"}, "format must be one of"),
        ("browser.screenshot", {"quality": 101}, "quality must be between"),
        ("browser.console", {"window_ms": 0}, "window_ms must be a positive integer"),
        ("browser.console", {"window_ms": 1_001}, "window_ms exceeds maximum"),
        ("browser.console", {"max_events": 0}, "max_events must be a positive integer"),
        ("browser.console", {"max_events": 6}, "max_events exceeds maximum"),
        ("browser.network", {"selector": "button"}, "unknown arguments"),
    ]

    for tool_name, args, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            module.validate_tool_arguments(tool_name, args)
