from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.browser_cdp import (
    CDPClientConfig,
    CDPClientError,
    CDPPageTarget,
)
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.browser_cdp_module import (
    BrowserCDPModule,
)


class _FakeCDPClient:
    def __init__(
        self,
        config: CDPClientConfig,
        *,
        pages: list[CDPPageTarget] | None = None,
        screenshot_data: str = "ZmFrZQ==",
        events: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.pages = pages or [
            CDPPageTarget(
                target_id="page-1",
                title="Test page",
                url="https://example.test/app",
                type="page",
                websocket_url="ws://127.0.0.1/devtools/page/page-1",
            )
        ]
        self.screenshot_data = screenshot_data
        self.events = events or {
            "events": [{"method": "Runtime.consoleAPICalled", "params": {"type": "log"}}],
            "truncated": False,
            "observed_for_ms": 25,
        }
        self.commands: list[tuple[str, dict[str, Any]]] = []
        self.observations: list[dict[str, Any]] = []

    async def get_version(self) -> dict[str, Any]:
        return {"browser": "Chrome/125.0.0", "protocol_version": "1.3"}

    async def list_pages(self) -> list[CDPPageTarget]:
        return self.pages

    async def send_command(
        self,
        page: CDPPageTarget,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.commands.append((method, params or {}))
        if method == "Accessibility.getFullAXTree":
            return {
                "nodes": [
                    {"nodeId": "1", "role": {"value": "RootWebArea"}},
                    {"nodeId": "2", "role": {"value": "button"}},
                    {"nodeId": "3", "role": {"value": "textbox"}},
                ]
            }
        if method == "Runtime.evaluate":
            return {
                "result": {
                    "type": "object",
                    "value": {
                        "url": page.url,
                        "title": page.title,
                        "ready_state": "complete",
                        "viewport": {
                            "width": 1280,
                            "height": 720,
                            "device_scale_factor": 2,
                        },
                    },
                }
            }
        if method == "Page.captureScreenshot":
            return {"data": self.screenshot_data}
        raise AssertionError(f"Unexpected CDP method: {method}")

    async def observe_events(
        self,
        page: CDPPageTarget,
        *,
        enable_methods: list[str],
        event_names: set[str],
        window_ms: int,
        max_events: int,
    ) -> dict[str, Any]:
        self.observations.append(
            {
                "page": page.target_id,
                "enable_methods": enable_methods,
                "event_names": event_names,
                "window_ms": window_ms,
                "max_events": max_events,
            }
        )
        return self.events


def _module(
    settings: dict[str, Any] | None = None,
    *,
    client: _FakeCDPClient | None = None,
) -> BrowserCDPModule:
    resolved_settings = settings if settings is not None else {
        "debugger_url": "http://127.0.0.1:9222",
        "max_snapshot_nodes": 10,
        "max_events": 5,
        "max_observation_window_ms": 1_000,
    }
    return BrowserCDPModule(
        ModuleConfig(
            name="browser_cdp",
            settings=resolved_settings,
        ),
        client_factory=(lambda config: client or _FakeCDPClient(config)),
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


def test_browser_cdp_config_parses_allow_non_loopback_safely() -> None:
    disabled = _module(
        settings={
            "debugger_url": "http://127.0.0.1:9222",
            "allow_non_loopback": "false",
        }
    )
    enabled = _module(
        settings={
            "debugger_url": "http://127.0.0.1:9222",
            "allow_non_loopback": "true",
        }
    )

    assert disabled._client_config().allow_non_loopback is False  # noqa: SLF001  # nosec B101
    assert enabled._client_config().allow_non_loopback is True  # noqa: SLF001  # nosec B101


def test_browser_cdp_config_clamps_default_observation_window() -> None:
    module = _module(
        settings={
            "debugger_url": "http://127.0.0.1:9222",
            "observation_window_ms": 10_000,
            "max_observation_window_ms": 750,
        }
    )

    assert module._client_config().observation_window_ms == 750  # noqa: SLF001  # nosec B101


@pytest.mark.asyncio
async def test_browser_status_reports_availability_and_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_module = _module()

    configured = await configured_module.execute_tool("browser.status", {})

    assert configured["available"] is True  # nosec B101
    assert configured["configured"] is True  # nosec B101
    assert configured["page_count"] == 1  # nosec B101
    assert configured["version"]["browser"] == "Chrome/125.0.0"  # nosec B101

    monkeypatch.delenv("MCP_BROWSER_CDP_URL", raising=False)
    unconfigured = await _module(settings={}).execute_tool("browser.status", {})

    assert unconfigured == {  # nosec B101
        "available": False,
        "configured": False,
        "reason_code": "cdp_not_configured",
        "message": "CDP debugger URL is not configured",
    }


@pytest.mark.asyncio
async def test_browser_pages_list_returns_page_targets() -> None:
    module = _module()

    result = await module.execute_tool("browser.pages.list", {})

    assert result["count"] == 1  # nosec B101
    assert result["pages"] == [  # nosec B101
        {
            "target_id": "page-1",
            "title": "Test page",
            "url": "https://example.test/app",
            "type": "page",
            "attached": None,
        }
    ]


@pytest.mark.asyncio
async def test_browser_snapshot_bounds_nodes_and_reports_truncation() -> None:
    client = _FakeCDPClient(CDPClientConfig(debugger_url="http://127.0.0.1:9222"))
    module = _module(client=client)

    result = await module.execute_tool("browser.snapshot", {"target_id": "page-1", "limit": 2})

    assert result["target"]["target_id"] == "page-1"  # nosec B101
    assert len(result["nodes"]) == 2  # nosec B101
    assert result["truncated"] is True  # nosec B101
    assert result["total_nodes"] == 3  # nosec B101
    assert client.commands[0][0] == "Accessibility.getFullAXTree"  # nosec B101


@pytest.mark.asyncio
async def test_browser_page_state_uses_fixed_read_only_evaluation() -> None:
    client = _FakeCDPClient(CDPClientConfig(debugger_url="http://127.0.0.1:9222"))
    module = _module(client=client)

    result = await module.execute_tool("browser.page_state", {})

    assert result["target"]["target_id"] == "page-1"  # nosec B101
    assert result["state"]["url"] == "https://example.test/app"  # nosec B101
    assert result["state"]["ready_state"] == "complete"  # nosec B101
    method, params = client.commands[0]
    assert method == "Runtime.evaluate"  # nosec B101
    assert params["returnByValue"] is True  # nosec B101
    assert "expression" in params  # nosec B101


@pytest.mark.asyncio
async def test_browser_screenshot_returns_bounded_base64_payload() -> None:
    client = _FakeCDPClient(CDPClientConfig(debugger_url="http://127.0.0.1:9222"))
    module = _module(client=client)

    result = await module.execute_tool("browser.screenshot", {"format": "jpeg", "quality": 80})

    assert result["target"]["target_id"] == "page-1"  # nosec B101
    assert result["mime_type"] == "image/jpeg"  # nosec B101
    assert result["data"] == "ZmFrZQ=="  # nosec B101
    assert result["byte_estimate"] == 4  # nosec B101
    assert client.commands[0] == (  # nosec B101
        "Page.captureScreenshot",
        {"format": "jpeg", "quality": 80},
    )

    oversized_client = _FakeCDPClient(
        CDPClientConfig(debugger_url="http://127.0.0.1:9222"),
        screenshot_data="ZmFrZQ==",
    )
    oversized_module = _module(
        settings={
            "debugger_url": "http://127.0.0.1:9222",
            "screenshot_max_bytes": 3,
        },
        client=oversized_client,
    )

    with pytest.raises(CDPClientError) as exc_info:
        await oversized_module.execute_tool("browser.screenshot", {})

    assert exc_info.value.reason_code == "payload_too_large"  # nosec B101


@pytest.mark.asyncio
async def test_browser_screenshot_rejects_oversized_payload_before_full_decode() -> None:
    oversized_client = _FakeCDPClient(
        CDPClientConfig(debugger_url="http://127.0.0.1:9222"),
        screenshot_data="!" * 128,
    )
    module = _module(
        settings={
            "debugger_url": "http://127.0.0.1:9222",
            "screenshot_max_bytes": 1,
        },
        client=oversized_client,
    )

    with pytest.raises(CDPClientError) as exc_info:
        await module.execute_tool("browser.screenshot", {})

    assert exc_info.value.reason_code == "payload_too_large"  # nosec B101


@pytest.mark.asyncio
async def test_browser_console_and_network_return_bounded_observed_events() -> None:
    client = _FakeCDPClient(
        CDPClientConfig(debugger_url="http://127.0.0.1:9222"),
        events={
            "events": [{"method": "Network.responseReceived", "params": {"status": 200}}],
            "truncated": True,
            "observed_for_ms": 50,
        },
    )
    module = _module(client=client)

    console = await module.execute_tool("browser.console", {"window_ms": 50, "max_events": 2})
    network = await module.execute_tool("browser.network", {"window_ms": 75, "max_events": 3})

    assert console["events"] == [{"method": "Network.responseReceived", "params": {"status": 200}}]  # nosec B101
    assert console["truncated"] is True  # nosec B101
    assert console["observed_for_ms"] == 50  # nosec B101
    assert client.observations[0]["enable_methods"] == ["Runtime.enable", "Log.enable"]  # nosec B101
    assert client.observations[0]["event_names"] == {  # nosec B101
        "Runtime.consoleAPICalled",
        "Log.entryAdded",
    }
    assert network["events"] == [{"method": "Network.responseReceived", "params": {"status": 200}}]  # nosec B101
    assert client.observations[1]["enable_methods"] == ["Network.enable"]  # nosec B101
    assert client.observations[1]["event_names"] == {  # nosec B101
        "Network.requestWillBeSent",
        "Network.responseReceived",
        "Network.loadingFailed",
    }


@pytest.mark.asyncio
async def test_browser_tool_execution_rejects_missing_target_id() -> None:
    module = _module()

    with pytest.raises(CDPClientError) as exc_info:
        await module.execute_tool("browser.snapshot", {"target_id": "missing"})

    assert exc_info.value.reason_code == "target_not_found"  # nosec B101
