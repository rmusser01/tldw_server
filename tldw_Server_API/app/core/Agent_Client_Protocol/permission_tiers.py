from __future__ import annotations

from loguru import logger


def resolve_policy_permission_tier(tool_name: str) -> str | None:
    """Consult admin-configured ACP permission policies, if available."""
    try:
        import tldw_Server_API.app.services.admin_acp_sessions_service as store_src

        store = getattr(store_src, "_store", None)
        if store is None:
            return None
        resolver = getattr(store, "resolve_permission_tier", None)
        if not callable(resolver):
            return None
        return resolver(tool_name)
    except Exception as policy_error:
        logger.bind(error_type=type(policy_error).__name__).debug(
            "Failed to resolve ACP permission tier from admin policy store"
        )
        return None


def determine_permission_tier(tool_name: str) -> str:
    """Resolve ACP permission tier from admin policy first, then heuristics."""
    policy_tier = resolve_policy_permission_tier(tool_name)
    if policy_tier is not None:
        return policy_tier

    tool_lower = tool_name.lower()

    auto_patterns = ["read", "get", "list", "search", "find", "view", "show", "glob", "grep", "status"]
    if any(pattern in tool_lower for pattern in auto_patterns):
        return "auto"

    individual_patterns = ["delete", "remove", "exec", "run", "shell", "bash", "terminal", "push", "force"]
    if any(pattern in tool_lower for pattern in individual_patterns):
        return "individual"

    return "batch"
