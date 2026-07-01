from __future__ import annotations

from typing import Any

from loguru import logger

from mcp_unified.docs import DocsMCPToolProvider
from tldw_Server_API.app.core.MCP_unified.adapters.docs import (
    docs_scope_from_context,
    docs_settings_from_module_config,
)
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule


class DocsModule(BaseModule):
    """Adapter from the host MCP runtime to the standalone docs corpus provider."""

    def _ensure_provider(self) -> DocsMCPToolProvider:
        provider = getattr(self, "_provider", None)
        if provider is None:
            settings = docs_settings_from_module_config(self.config)
            provider = DocsMCPToolProvider(settings=settings)
            self._settings = settings
            self._provider = provider
        return provider

    async def on_initialize(self) -> None:
        provider = self._ensure_provider()
        logger.info("Initialized Docs MCP module with db_path={}", provider.settings.db_path)

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": getattr(self, "_provider", None) is not None}

    async def get_tools(self) -> list[dict[str, Any]]:
        return self._ensure_provider().tool_definitions()

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        args = self.sanitize_input(arguments or {})
        try:
            self.validate_tool_arguments(tool_name, args)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid arguments for {tool_name}: {exc}") from exc
        return self._ensure_provider().execute(tool_name, args, scope=docs_scope_from_context(context))

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name == "docs.import_path" and not str(arguments.get("path") or "").strip():
            raise ValueError("path is required")
        if tool_name == "docs.ingest_url" and not str(arguments.get("url") or "").strip():
            raise ValueError("url is required")
        if tool_name in {"docs.search", "docs.context"} and not str(arguments.get("query") or "").strip():
            raise ValueError("query is required")
