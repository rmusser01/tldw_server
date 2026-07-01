from __future__ import annotations

import asyncio
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
        args = dict(arguments or {})
        try:
            self.validate_tool_arguments(tool_name, args)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid arguments for {tool_name}: {exc}") from exc
        provider = self._ensure_provider()
        scope = docs_scope_from_context(context)
        return await asyncio.to_thread(provider.execute, tool_name, args, scope=scope)

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        required_strings = {
            "docs.import_path": ("path",),
            "docs.ingest_url": ("url",),
            "docs.search": ("query",),
            "docs.context": ("query",),
            "docs.resolve": ("name",),
            "docs.get": ("id",),
            "docs.list": ("kind",),
            "docs.collections.create": ("name",),
            "docs.collections.update": ("name",),
            "docs.collections.set_membership": ("collection", "action"),
            "resolve-library-id": ("libraryName",),
            "get-library-docs": ("context7CompatibleLibraryID",),
        }
        for field_name in required_strings.get(tool_name, ()):
            if not str(arguments.get(field_name) or "").strip():
                raise ValueError(f"{field_name} is required")
        if (
            tool_name in {"docs.collections.set_membership", "docs.keywords.apply"}
            and not str(arguments.get("document_id") or "").strip()
        ):
            raise ValueError("document_id is required")
        if tool_name == "docs.keywords.apply" and "keywords" not in arguments:
            raise ValueError("keywords is required")
