"""
Rate limiting for unified MCP module.

Resource Governor is the single enforcement path for MCP requests.
"""

from __future__ import annotations

import asyncio
import os
import time

from loguru import logger

from ..config import get_config

try:
    from tldw_Server_API.app.core.config import rg_enabled  # type: ignore
    from tldw_Server_API.app.core.Resource_Governance import (  # type: ignore
        MemoryResourceGovernor,
        RedisResourceGovernor,
        RGRequest,
    )
    from tldw_Server_API.app.core.Resource_Governance.policy_loader import (  # type: ignore
        PolicyLoader,
        PolicyReloadConfig,
        default_policy_loader,
    )
except Exception:  # pragma: no cover - RG optional
    MemoryResourceGovernor = None  # type: ignore
    RedisResourceGovernor = None  # type: ignore
    RGRequest = None  # type: ignore
    PolicyLoader = None  # type: ignore
    PolicyReloadConfig = None  # type: ignore
    default_policy_loader = None  # type: ignore
    rg_enabled = None  # type: ignore


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds")


class RateLimiter:
    """MCP rate limiter backed by Resource Governor only."""

    def __init__(self) -> None:
        self.config = get_config()

    async def check_rate_limit(self, key: str, *, category: str = "default") -> None:
        """
        Enforce MCP rate limits via Resource Governor when enabled.

        Args:
            key: Unique identifier for rate limiting
            category: Category label (default, ingestion, read)

        Raises:
            RateLimitExceeded: If Resource Governor denies the request
        """
        if not getattr(self.config, "rate_limit_enabled", True):
            return

        if not _rg_mcp_enabled():
            return

        rg_decision = await _maybe_enforce_with_rg_mcp(key=key, category=category)
        if rg_decision is None:
            _log_rg_mcp_fallback("rg_decision_unavailable")
            return

        if not rg_decision.get("allowed"):
            raise RateLimitExceeded(int(rg_decision.get("retry_after") or 1))

    async def get_usage(self, key: str) -> dict[str, object]:
        """Return best-effort usage info (RG does not currently expose per-key usage here)."""
        return {}

    async def shutdown(self) -> None:
        """No background tasks to shutdown (kept for compatibility)."""
        return None


_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create rate limiter singleton."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


async def shutdown_rate_limiter() -> None:
    """Shutdown the singleton rate limiter (no-op cleanup)."""
    global _rate_limiter
    if _rate_limiter is not None:
        await _rate_limiter.shutdown()
        _rate_limiter = None


# --- Resource Governor plumbing (optional) ---
_rg_mcp_governor = None
_rg_mcp_loader = None
_rg_mcp_lock = asyncio.Lock()
_rg_mcp_init_error: str | None = None
_rg_mcp_init_error_logged = False
_rg_mcp_fallback_logged = False


def _rg_mcp_context() -> dict[str, str]:
    policy_path = os.getenv(
        "RG_POLICY_PATH",
        "tldw_Server_API/Config_Files/resource_governor_policies.yaml",
    )
    try:
        policy_path_resolved = os.path.abspath(policy_path)
    except Exception:
        policy_path_resolved = policy_path
    return {
        "backend": os.getenv("RG_BACKEND", "memory"),
        "policy_path": policy_path,
        "policy_path_resolved": policy_path_resolved,
        "policy_store": os.getenv("RG_POLICY_STORE", ""),
        "policy_reload_enabled": os.getenv("RG_POLICY_RELOAD_ENABLED", ""),
        "policy_reload_interval": os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", ""),
        "cwd": os.getcwd(),
    }


def _log_rg_mcp_init_failure(exc: Exception) -> None:
    global _rg_mcp_init_error, _rg_mcp_init_error_logged
    _rg_mcp_init_error = repr(exc)
    if _rg_mcp_init_error_logged:
        return
    _rg_mcp_init_error_logged = True
    ctx = _rg_mcp_context()
    logger.exception(
        "MCP ResourceGovernor init failed; enforcement disabled. "
        "backend={backend} policy_path={policy_path} policy_path_resolved={policy_path_resolved} "
        "policy_store={policy_store} reload_enabled={policy_reload_enabled} "
        "reload_interval={policy_reload_interval} cwd={cwd}",
        **ctx,
    )


def _log_rg_mcp_fallback(reason: str) -> None:
    global _rg_mcp_fallback_logged
    if _rg_mcp_fallback_logged:
        return
    _rg_mcp_fallback_logged = True
    ctx = _rg_mcp_context()
    logger.error(
        "MCP ResourceGovernor unavailable; skipping MCP rate limits. "
        "reason={} init_error={} backend={backend} policy_path={policy_path} "
        "policy_path_resolved={policy_path_resolved} policy_store={policy_store} "
        "reload_enabled={policy_reload_enabled} reload_interval={policy_reload_interval} cwd={cwd}",
        reason,
        _rg_mcp_init_error,
        **ctx,
    )


def _rg_mcp_entity_from_key(key: str) -> str:
    if not key:
        return "entity:unknown"
    try:
        scope = key.split(":", 1)[0]
        if scope in {"user", "client", "api_key", "ip", "service", "entity"}:
            return key
    except Exception as exc:
        logger.debug("Failed to parse entity key '{}': {}", key, exc)
    return f"client:{key}"


def _rg_mcp_enabled() -> bool:
    if rg_enabled:
        try:
            return bool(rg_enabled(True))  # type: ignore[func-returns-value]
        except Exception:
            return False
    return False


async def _get_mcp_rg_governor():
    global _rg_mcp_governor, _rg_mcp_loader
    if not _rg_mcp_enabled():
        return None
    if RGRequest is None or PolicyLoader is None:
        _log_rg_mcp_fallback("rg_components_unavailable")
        return None
    if _rg_mcp_governor is not None:
        return _rg_mcp_governor
    async with _rg_mcp_lock:
        if _rg_mcp_governor is not None:
            return _rg_mcp_governor
        try:
            loader = (
                default_policy_loader()
                if default_policy_loader
                else PolicyLoader(
                    os.getenv("RG_POLICY_PATH", "tldw_Server_API/Config_Files/resource_governor_policies.yaml"),
                    PolicyReloadConfig(
                        enabled=True,
                        interval_sec=int(os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10") or "10"),
                    ),
                )
            )
            await loader.load_once()
            _rg_mcp_loader = loader
            backend = os.getenv("RG_BACKEND", "memory").lower()
            if backend == "redis" and RedisResourceGovernor is not None:
                gov = RedisResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            else:
                gov = MemoryResourceGovernor(policy_loader=loader)  # type: ignore[call-arg]
            _rg_mcp_governor = gov
            return gov
        except Exception as exc:  # pragma: no cover - optional
            _log_rg_mcp_init_failure(exc)
            return None


async def _maybe_enforce_with_rg_mcp(*, key: str, category: str) -> dict[str, object] | None:
    gov = await _get_mcp_rg_governor()
    if gov is None:
        return None
    policy_id = f"mcp.{category}"
    try:
        if _rg_mcp_loader is not None and not _rg_mcp_loader.get_policy(policy_id):
            policy_id = "mcp.default"  # categories without their own policy share mcp.default
    except Exception as exc:  # noqa: BLE001 - a lookup failure must not deny the tool call
        logger.warning("MCP RG policy lookup for category {!r} failed ({!r}); using mcp.default", category, exc)
        policy_id = "mcp.default"
    op_id = f"mcp-{category}-{key}-{time.time_ns()}"
    try:
        decision, handle = await gov.reserve(
            RGRequest(
                entity=_rg_mcp_entity_from_key(key),
                categories={"requests": {"units": 1}},
                tags={"policy_id": policy_id, "module": "mcp", "category": category},
            ),
            op_id=op_id,
        )
        if decision.allowed:
            if handle:
                try:
                    await gov.commit(handle, None, op_id=op_id)
                except Exception:
                    logger.debug("MCP RG commit failed", exc_info=True)
            return {"allowed": True, "retry_after": None, "policy_id": policy_id}
        return {
            "allowed": False,
            "retry_after": decision.retry_after or 1,
            "policy_id": policy_id,
        }
    except Exception as exc:
        logger.debug("MCP RG reserve failed: {}", exc)
        return None
