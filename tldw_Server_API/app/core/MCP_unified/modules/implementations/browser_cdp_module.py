"""Read-only browser inspection MCP tools backed by Chrome DevTools Protocol."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from tldw_Server_API.app.core.MCP_unified.browser_cdp import (
    CDPBrowserClient,
    CDPClientConfig,
)

from ..base import BaseModule, ModuleConfig, create_tool_definition

_TOOL_STATUS = "browser.status"
_TOOL_PAGES_LIST = "browser.pages.list"
_TOOL_SNAPSHOT = "browser.snapshot"
_TOOL_PAGE_STATE = "browser.page_state"
_TOOL_SCREENSHOT = "browser.screenshot"
_TOOL_CONSOLE = "browser.console"
_TOOL_NETWORK = "browser.network"

_ALL_TOOLS = {
    _TOOL_STATUS,
    _TOOL_PAGES_LIST,
    _TOOL_SNAPSHOT,
    _TOOL_PAGE_STATE,
    _TOOL_SCREENSHOT,
    _TOOL_CONSOLE,
    _TOOL_NETWORK,
}


class BrowserCDPModule(BaseModule):
    """Optional read-only CDP-backed browser inspection module."""

    def __init__(
        self,
        config: ModuleConfig,
        *,
        client_factory: Callable[[CDPClientConfig], CDPBrowserClient] | None = None,
    ) -> None:
        super().__init__(config)
        self._client_factory = client_factory or CDPBrowserClient

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"configured": bool(self._client_config().debugger_url)}

    async def get_tools(self) -> list[dict[str, Any]]:
        shared_metadata = {
            "category": "browser",
            "readOnlyHint": True,
            "uses_browser": True,
            "cdp_backend": True,
        }

        tools = [
            self._tool(
                name=_TOOL_STATUS,
                description="Report configured CDP browser inspection availability.",
                properties={},
                required=[],
                capabilities=["browser.inspect"],
                metadata=shared_metadata,
            ),
            self._tool(
                name=_TOOL_PAGES_LIST,
                description="List inspectable CDP page targets.",
                properties={},
                required=[],
                capabilities=["browser.inspect"],
                metadata=shared_metadata,
            ),
            self._tool(
                name=_TOOL_SNAPSHOT,
                description="Capture a bounded read-only snapshot of the current browser page.",
                properties={
                    "target_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                },
                required=[],
                capabilities=["browser.inspect", "browser.debug"],
                metadata=shared_metadata,
            ),
            self._tool(
                name=_TOOL_PAGE_STATE,
                description="Read fixed page state such as URL, title, viewport, and readiness.",
                properties={"target_id": {"type": "string"}},
                required=[],
                capabilities=["browser.inspect", "app_state.read"],
                metadata=shared_metadata,
            ),
            self._tool(
                name=_TOOL_SCREENSHOT,
                description="Capture a bounded in-memory page screenshot through CDP.",
                properties={
                    "target_id": {"type": "string"},
                    "format": {"type": "string", "enum": ["png", "jpeg"], "default": "png"},
                    "quality": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                required=[],
                capabilities=["browser.inspect", "screenshots.capture"],
                metadata=shared_metadata,
            ),
            self._tool(
                name=_TOOL_CONSOLE,
                description="Observe console/log events during a bounded CDP window.",
                properties={
                    "target_id": {"type": "string"},
                    "window_ms": {"type": "integer", "minimum": 1},
                    "max_events": {"type": "integer", "minimum": 1},
                },
                required=[],
                capabilities=["browser.inspect", "browser.debug"],
                metadata=shared_metadata,
            ),
            self._tool(
                name=_TOOL_NETWORK,
                description="Observe network events during a bounded CDP window.",
                properties={
                    "target_id": {"type": "string"},
                    "window_ms": {"type": "integer", "minimum": 1},
                    "max_events": {"type": "integer", "minimum": 1},
                },
                required=[],
                capabilities=["browser.inspect", "browser.debug"],
                metadata=shared_metadata,
            ),
        ]
        for tool in tools:
            tool["inputSchema"]["additionalProperties"] = False
        return tools

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        args = arguments or {}
        if tool_name not in _ALL_TOOLS:
            raise ValueError(f"Unknown browser CDP tool: {tool_name}")

        if tool_name in {_TOOL_STATUS, _TOOL_PAGES_LIST}:
            self._reject_unknown(args, set())
            return

        if tool_name == _TOOL_SNAPSHOT:
            self._reject_unknown(args, {"target_id", "limit"})
            self._validate_target_id(args)
            self._positive_int(
                args,
                "limit",
                maximum=self._setting_positive_int("max_snapshot_nodes", 200),
            )
            return

        if tool_name == _TOOL_PAGE_STATE:
            self._reject_unknown(args, {"target_id"})
            self._validate_target_id(args)
            return

        if tool_name == _TOOL_SCREENSHOT:
            self._reject_unknown(args, {"target_id", "format", "quality"})
            self._validate_target_id(args)
            fmt = args.get("format")
            if fmt is not None and fmt not in {"png", "jpeg"}:
                raise ValueError("format must be one of: png, jpeg")
            quality = args.get("quality")
            if quality is not None and (
                not isinstance(quality, int) or isinstance(quality, bool) or quality < 1 or quality > 100
            ):
                raise ValueError("quality must be between 1 and 100")
            return

        if tool_name in {_TOOL_CONSOLE, _TOOL_NETWORK}:
            self._reject_unknown(args, {"target_id", "window_ms", "max_events"})
            self._validate_target_id(args)
            self._positive_int(
                args,
                "window_ms",
                maximum=self._setting_positive_int("max_observation_window_ms", 5_000),
            )
            self._positive_int(
                args,
                "max_events",
                maximum=self._setting_positive_int("max_events", 100),
            )

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        self.validate_tool_arguments(tool_name, self.sanitize_input(arguments or {}))
        raise NotImplementedError("Browser CDP tool execution is implemented in the execution slice")

    def _client_config(self) -> CDPClientConfig:
        settings = self.config.settings if isinstance(self.config.settings, dict) else {}
        debugger_url = str(
            settings.get("debugger_url")
            or os.getenv("MCP_BROWSER_CDP_URL", "")
        ).strip() or None
        return CDPClientConfig(
            debugger_url=debugger_url,
            request_timeout_seconds=float(settings.get("request_timeout_seconds", 3.0)),
            observation_window_ms=self._setting_positive_int("observation_window_ms", 250),
            max_events=self._setting_positive_int("max_events", 100),
            max_snapshot_nodes=self._setting_positive_int("max_snapshot_nodes", 200),
            screenshot_max_bytes=self._setting_positive_int("screenshot_max_bytes", 2_000_000),
            allow_non_loopback=bool(settings.get("allow_non_loopback", False)),
        )

    def _tool(
        self,
        *,
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
        capabilities: list[str],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return create_tool_definition(
            name=name,
            description=description,
            parameters={"properties": properties, "required": required},
            metadata={**metadata, "capabilities": capabilities},
        )

    def _reject_unknown(self, args: dict[str, Any], allowed: set[str]) -> None:
        unknown = sorted(set(args) - allowed)
        if unknown:
            raise ValueError(f"unknown arguments for browser CDP tool: {', '.join(unknown)}")

    def _validate_target_id(self, args: dict[str, Any]) -> None:
        target_id = args.get("target_id")
        if target_id is not None and (not isinstance(target_id, str) or not target_id.strip()):
            raise ValueError("target_id must be a non-empty string")

    def _positive_int(self, args: dict[str, Any], name: str, *, maximum: int) -> int | None:
        value = args.get(name)
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        if value > maximum:
            raise ValueError(f"{name} exceeds maximum {maximum}")
        return value

    def _setting_positive_int(self, name: str, default: int) -> int:
        settings = self.config.settings if isinstance(self.config.settings, dict) else {}
        value = settings.get(name, default)
        try:
            result = int(value)
        except (TypeError, ValueError):
            return default
        return result if result > 0 else default
