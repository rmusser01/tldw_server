"""
ResourceGovernor startup initialization extracted from the application lifespan.
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy as _shared_is_truthy

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_IMPORT_EXCEPTIONS = (
    AssertionError,
    ImportError,
    ModuleNotFoundError,
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


async def init_resource_governor(app: Any) -> None:
    """Initialize the ResourceGovernor policy loader, governor, and app state."""
    try:
        from tldw_Server_API.app.core.config import (
            rg_backend as _rg_backend_sel,
        )
        from tldw_Server_API.app.core.config import (
            rg_policy_path as _rg_policy_path,
        )
        from tldw_Server_API.app.core.config import (
            rg_policy_reload_enabled as _rg_reload_enabled,
        )
        from tldw_Server_API.app.core.config import (
            rg_policy_reload_interval_sec as _rg_reload_interval,
        )
        from tldw_Server_API.app.core.config import (
            rg_policy_store as _rg_store_sel,
        )
        from tldw_Server_API.app.core.Resource_Governance import (
            MemoryResourceGovernor,
            RedisResourceGovernor,
        )
        from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
            PolicyLoader as _RGPolicyLoader,
        )
        from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
            PolicyReloadConfig as _RGReloadCfg,
        )
        from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
            db_policy_loader as _rg_db_loader,
        )
        from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
            default_policy_loader as _rg_default_loader,
        )

        _store_mode = _rg_store_sel()
        if _store_mode == "db":
            try:
                from tldw_Server_API.app.core.Resource_Governance.authnz_policy_store import (
                    AuthNZPolicyStore as _RGDBStore,
                )

                _store = _RGDBStore()
                _interval = _rg_reload_interval()
                rg_loader = _rg_db_loader(_store, _RGReloadCfg(enabled=True, interval_sec=_interval))
                logger.info("ResourceGovernor policy loader configured for AuthNZ DB store")
            except _STARTUP_GUARD_EXCEPTIONS as _rg_db_err:
                logger.warning(f"Failed to configure DB-backed policy store, falling back to file: {_rg_db_err}")
                rg_loader = _rg_default_loader()
                _store_mode = "file"
        else:
            _enabled = _rg_reload_enabled()
            _interval = _rg_reload_interval()
            _path = _rg_policy_path()
            rg_loader = _RGPolicyLoader(_path, _RGReloadCfg(enabled=_enabled, interval_sec=_interval))

        await rg_loader.load_once()
        try:
            if _rg_reload_enabled():
                await rg_loader.start_auto_reload()
        except _STARTUP_GUARD_EXCEPTIONS as _rg_reload_err:
            logger.debug(f"Policy auto-reload not started: {_rg_reload_err}")
        app.state.rg_policy_loader = rg_loader
        app.state.rg_policy_store = _store_mode

        try:
            _backend = _rg_backend_sel()
            if _backend == "redis":
                await _ensure_redis_boot_health_if_required()
                app.state.rg_governor = RedisResourceGovernor(policy_loader=rg_loader)
                logger.info("ResourceGovernor initialized (redis backend)")
            else:
                app.state.rg_governor = MemoryResourceGovernor(policy_loader=rg_loader)
                logger.info("ResourceGovernor initialized (memory backend)")
        except _STARTUP_GUARD_EXCEPTIONS as _rg_gov_err:
            logger.warning(f"ResourceGovernor initialization failed/skipped: {_rg_gov_err}")

        _update_policy_snapshot_state(app, rg_loader)
        _register_policy_snapshot_callback(app, rg_loader)
        _audit_route_map_coverage(app, rg_loader)
    except _IMPORT_EXCEPTIONS as _rg_err:
        logger.warning(f"ResourceGovernor policy loader initialization skipped: {_rg_err}")

    _warn_if_enabled_without_governor(app)


async def _ensure_redis_boot_health_if_required() -> None:
    """Fail startup when Redis fail-closed mode is enabled and Redis is unreachable."""
    try:
        from tldw_Server_API.app.core.config import rg_redis_fail_mode as _rg_fail_mode

        if str(_rg_fail_mode() or "").strip().lower() != "fail_closed":
            return

        from contextlib import suppress

        from tldw_Server_API.app.core.Infrastructure.redis_factory import (
            create_async_redis_client as _create_async_redis_client,
        )
        from tldw_Server_API.app.core.Infrastructure.redis_factory import (
            ensure_async_client_closed as _ensure_async_client_closed,
        )

        _start = logger.bind(component="rg_boot_health")
        with suppress(_STARTUP_GUARD_EXCEPTIONS):
            _start.info("RG boot health: verifying Redis connectivity (fail_closed mode)")
        _rc = await _create_async_redis_client(fallback_to_fake=False, context="rg_boot_health")
        try:
            res = getattr(_rc, "ping", None)
            if res:
                pr = res()
                if hasattr(pr, "__await__"):
                    await pr
        finally:
            with suppress(_STARTUP_GUARD_EXCEPTIONS):
                await _ensure_async_client_closed(_rc)
    except _STARTUP_GUARD_EXCEPTIONS as _rg_boot_err:
        logger.exception(
            f"ResourceGovernor boot health failed (Redis unreachable, fail_closed): {_rg_boot_err}"
        )
        raise RuntimeError(
            "Redis backend selected with fail_closed, but Redis is unreachable; refusing to start"
        ) from _rg_boot_err


def _update_policy_snapshot_state(app: Any, rg_loader: Any) -> None:
    """Copy the current policy snapshot summary into app state."""
    try:
        snap = rg_loader.get_snapshot()
        app.state.rg_policy_version = int(getattr(snap, "version", 0) or 0)
        app.state.rg_policy_count = len(getattr(snap, "policies", {}) or {})
    except _STARTUP_GUARD_EXCEPTIONS:
        app.state.rg_policy_version = 0
        app.state.rg_policy_count = 0


def _register_policy_snapshot_callback(app: Any, rg_loader: Any) -> None:
    """Keep policy version and count synchronized across loader reloads."""
    try:

        def _on_rg_change(snap: Any) -> None:
            try:
                app.state.rg_policy_version = int(getattr(snap, "version", 0) or 0)
                app.state.rg_policy_count = len(getattr(snap, "policies", {}) or {})
            except _STARTUP_GUARD_EXCEPTIONS:
                pass

        rg_loader.add_on_change(_on_rg_change)
    except _STARTUP_GUARD_EXCEPTIONS:
        pass


def _audit_route_map_coverage(app: Any, rg_loader: Any) -> None:
    """Best-effort audit for routes lacking ResourceGovernor route-map coverage."""
    try:
        if not _shared_is_truthy(os.getenv("RG_ROUTE_MAP_AUDIT", "true")):
            return

        snap = rg_loader.get_snapshot()
        route_map = getattr(snap, "route_map", {}) or {}
        by_path = dict(route_map.get("by_path") or {})
        by_tag = dict(route_map.get("by_tag") or {})
        if not (by_path or by_tag):
            return

        skip_prefixes = ("/docs", "/openapi.json", "/redoc", "/static", "/favicon.ico")
        missing: list[tuple[str, list[str]]] = []
        seen_paths: set[str] = set()
        for route in getattr(app, "routes", []):
            path = getattr(route, "path", None)
            if not path or path in seen_paths:
                continue
            if path.startswith(skip_prefixes):
                continue
            if not (
                path.startswith("/api/")
                or path.startswith("/v1/")
                or path.startswith("/health")
                or path.startswith("/readyz")
                or path.startswith("/metrics")
                or path.startswith("/setup")
            ):
                continue
            if _route_map_matches(path, by_path):
                seen_paths.add(path)
                continue
            tags = list(getattr(route, "tags", []) or [])
            if tags and any(tag in by_tag for tag in tags):
                seen_paths.add(path)
                continue
            missing.append((path, tags))
            seen_paths.add(path)
        if missing:
            sample = ", ".join(f"{path} (tags={tags})" for path, tags in missing[:10])
            logger.warning(
                f"RG route_map missing coverage for {len(missing)} routes; sample: {sample}"
            )
    except _IMPORT_EXCEPTIONS as _rg_audit_err:
        logger.debug(f"RG route_map audit skipped: {_rg_audit_err}")


def _route_map_matches(path: str, by_path: dict[str, Any]) -> bool:
    """Return whether a concrete route path matches a route-map entry."""
    for raw_pattern in by_path:
        pattern = str(raw_pattern)
        if pattern.endswith("*"):
            if path.startswith(pattern[:-1]):
                return True
        elif path == pattern:
            return True
    return False


def _warn_if_enabled_without_governor(app: Any) -> None:
    """Emit the existing fail-closed startup warning when RG is enabled but unavailable."""
    try:
        from tldw_Server_API.app.core.config import (
            rg_backend as _rg_backend_sel,
        )
        from tldw_Server_API.app.core.config import (
            rg_enabled as _rg_enabled_flag,
        )
        from tldw_Server_API.app.core.config import (
            rg_policy_path as _rg_policy_path,
        )
        from tldw_Server_API.app.core.config import (
            rg_policy_store as _rg_store_sel,
        )

        if bool(_rg_enabled_flag(False)) and getattr(app.state, "rg_governor", None) is None:
            logger.warning(
                "ResourceGovernor enabled but not initialized; rate limiting will fail closed. "
                f"policy_path={_rg_policy_path()} backend={_rg_backend_sel()} "
                f"store={_rg_store_sel()} cwd={os.getcwd()}"
            )
    except _IMPORT_EXCEPTIONS as _rg_warn_err:
        logger.debug(f"ResourceGovernor init warning skipped: {_rg_warn_err}")
