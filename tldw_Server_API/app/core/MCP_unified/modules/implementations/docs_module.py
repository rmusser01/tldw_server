from __future__ import annotations

from typing import Any

from loguru import logger

from mcp_unified.docs import AccessScope, DocsMCPToolProvider, DocsSettings
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule


class DocsModule(BaseModule):
    """Adapter from the host MCP runtime to the standalone docs corpus provider."""

    def _ensure_provider(self) -> DocsMCPToolProvider:
        provider = getattr(self, "_provider", None)
        if provider is None:
            settings = DocsSettings.from_mapping(self.config.settings or {})
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
        return self._ensure_provider().execute(tool_name, args, scope=self._scope_from_context(context))

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name == "docs.import_path" and not str(arguments.get("path") or "").strip():
            raise ValueError("path is required")
        if tool_name in {"docs.search", "docs.context"} and not str(arguments.get("query") or "").strip():
            raise ValueError("query is required")

    @staticmethod
    def _scope_from_context(context: Any | None) -> AccessScope:
        metadata = getattr(context, "metadata", None) if context is not None else None
        profile_scope = metadata.get("profile_scope") if isinstance(metadata, dict) else None
        user_id = getattr(context, "user_id", None) if context is not None else None
        return AccessScope(
            owner_scope=str(user_id) if user_id is not None else None,
            profile_scope=str(profile_scope) if profile_scope else None,
        )
