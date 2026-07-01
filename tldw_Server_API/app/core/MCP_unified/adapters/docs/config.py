from __future__ import annotations

from typing import Any

from mcp_unified.docs import AccessScope, DocsSettings
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig


def docs_settings_from_module_config(config: ModuleConfig) -> DocsSettings:
    return DocsSettings.from_mapping(config.settings or {})


def docs_scope_from_context(context: Any | None) -> AccessScope:
    metadata = getattr(context, "metadata", None) if context is not None else None
    profile_scope = metadata.get("profile_scope") if isinstance(metadata, dict) else None
    user_id = getattr(context, "user_id", None) if context is not None else None
    return AccessScope(
        owner_scope=str(user_id) if user_id is not None else None,
        profile_scope=str(profile_scope) if profile_scope else None,
    )
