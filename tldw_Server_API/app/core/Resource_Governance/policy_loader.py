from __future__ import annotations

import asyncio
import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import yaml
from loguru import logger


@dataclass
class PolicyReloadConfig:
    enabled: bool = True
    interval_sec: int = 10


@dataclass(frozen=True)
class PolicySnapshot:
    version: int
    policies: dict[str, Any]
    tenant: dict[str, Any]
    route_map: dict[str, Any]
    source_path: Path
    loaded_at_monotonic: float
    mtime: float


class PolicyLoader:
    """
    Minimal policy loader with optional hot‑reload.

    - Reads YAML from `path`.
    - Exposes a fast, thread‑safe snapshot via `get_snapshot()`.
    - Optional async reload loop (`start_auto_reload`) that polls mtime.
    - Designed to be integrated into FastAPI lifespan startup/shutdown.
    """

    def __init__(
        self,
        path: str | Path,
        reload: PolicyReloadConfig | None = None,
        *,
        time_source: Callable[[], float] = time.monotonic,
        store: PolicyStoreProtocol | None = None,
    ) -> None:
        self._path = Path(path)
        self._reload_cfg = reload or PolicyReloadConfig(enabled=True, interval_sec=10)
        self._time_source = time_source

        self._snapshot: PolicySnapshot | None = None
        self._lock = asyncio.Lock()
        self._reload_task: asyncio.Task | None = None
        self._on_change: list[Callable[[PolicySnapshot], None]] = []
        self._store: PolicyStoreProtocol | None = store

    def add_on_change(self, func: Callable[[PolicySnapshot], None]) -> None:
        self._on_change.append(func)

    async def load_once(self) -> PolicySnapshot:
        """Load the policy file once and update the in‑memory snapshot."""
        if self._store is not None:
            # DB-backed: fetch latest policy snapshot
            res = await self._store.get_latest_policy()
            version = int(res[0])
            policies = dict(res[1] or {})
            tenant = dict(res[2] or {})
            # Tuple shapes supported:
            #  - (version, policies, tenant, updated_at)
            #  - (version, policies, tenant, route_map, updated_at)
            if len(res) == 5:
                db_route_map = dict(res[3] or {})
                updated_at = float(res[4])
            else:
                db_route_map = {}
                updated_at = float(res[3]) if len(res) >= 4 else self._time_source()
            mtime = float(updated_at)
            # Merge route_map from file and DB consistently (DB overrides file)
            file_route_map: dict[str, Any] = {}
            try:
                if self._path.exists():
                    with self._path.open("r", encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    file_route_map = dict(data.get("route_map") or {})
                    mtime = max(mtime, self._path.stat().st_mtime)
            except Exception as e:
                logger.debug("PolicyLoader: failed to read route_map from file: {}", e)

            def _merge_route_map(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
                out: dict[str, Any] = {}
                base = dict(base or {})
                override = dict(override or {})
                # Merge known nested maps with override precedence
                by_path = dict(base.get("by_path") or {})
                by_path.update(dict(override.get("by_path") or {}))
                by_tag = dict(base.get("by_tag") or {})
                by_tag.update(dict(override.get("by_tag") or {}))
                # Start with base and overlay override keys
                out.update(base)
                out.update(override)
                if by_path:
                    out["by_path"] = by_path
                if by_tag:
                    out["by_tag"] = by_tag
                return out

            # Precedence: file route_map overrides DB route_map on conflicts (per README)
            route_map = _merge_route_map(db_route_map, file_route_map)
        else:
            if not self._path.exists():
                raise FileNotFoundError(f"Policy file not found: {self._path}")
            mtime = self._path.stat().st_mtime
            with self._path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            version = int(data.get("version") or 1)
            policies = dict(data.get("policies") or {})
            tenant = dict(data.get("tenant") or {})
            route_map = dict(data.get("route_map") or {})

        snap = PolicySnapshot(
            version=version,
            policies=policies,
            tenant=tenant,
            route_map=route_map,
            source_path=self._path,
            loaded_at_monotonic=self._time_source(),
            mtime=mtime,
        )

        self._snapshot = snap
        logger.info(
            "ResourceGovernor policy loaded: version={}, policies={}, path={}",
            version,
            len(policies),
            str(self._path),
        )
        for cb in self._on_change:
            try:
                cb(snap)
            except Exception as e:  # noqa: BLE001
                logger.warning("Policy on_change callback error: {}", e)
        return snap

    def get_snapshot(self) -> PolicySnapshot:
        snap = self._snapshot
        if snap is None:
            raise RuntimeError("PolicyLoader not initialized. Call load_once() first.")
        return snap

    def get_policy(self, policy_id: str) -> dict[str, Any] | None:
        snap = self.get_snapshot()
        return snap.policies.get(policy_id)

    async def start_auto_reload(self) -> None:
        if not self._reload_cfg.enabled:
            return
        if self._reload_task and not self._reload_task.done():
            return
        self._reload_task = asyncio.create_task(self._reload_loop(), name="rg_policy_reload")

    async def _reload_loop(self) -> None:
        interval = max(1, int(self._reload_cfg.interval_sec))
        # Ensure one initial snapshot exists
        if self._snapshot is None:
            try:
                await self.load_once()
            except Exception as e:  # noqa: BLE001
                logger.error("Initial policy load failed: {}", e)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._maybe_reload()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("Policy reload tick failed: {}", e)

    async def _maybe_reload(self) -> None:
        # Use a lock to avoid concurrent loads
        async with self._lock:
            snap = self._snapshot
            cur_version: int | None = None
            if self._store is not None:
                try:
                    _res = await self._store.get_latest_policy()
                    # Back-compat: stores may return (version, policies, tenant, ts)
                    if isinstance(_res, tuple) and len(_res) >= 4:
                        if len(_res) == 4:
                            _v, _p, _t, updated_at = _res
                        else:
                            _v, _p, _t, _rm, updated_at = _res[0], _res[1], _res[2], _res[3], _res[4]
                        cur_version = int(_v)
                        cur_mtime = float(updated_at)
                    else:
                        cur_mtime = time.time()
                    # Also consider route_map file mtime
                    try:
                        if self._path.exists():
                            cur_mtime = max(cur_mtime, self._path.stat().st_mtime)
                    except Exception as path_stat_error:
                        logger.debug("Policy loader failed to stat policy file mtime", exc_info=path_stat_error)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to poll policy store: {}", e)
                    return
            else:
                cur_mtime = self._path.stat().st_mtime if self._path.exists() else 0
            version_changed = cur_version is not None and snap is not None and cur_version != snap.version
            if snap is None or version_changed or cur_mtime > snap.mtime:
                await self.load_once()

    async def shutdown(self) -> None:
        task = self._reload_task
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


# Optional helper to construct from default location & env
def default_policy_loader() -> PolicyLoader:
    try:
        from tldw_Server_API.app.core.config import (
            resolve_repo_relative_path,
            rg_policy_path,  # type: ignore
            rg_policy_path_default,
            rg_repo_root,
        )
    except Exception:
        rg_policy_path = None  # type: ignore
        rg_policy_path_default = None  # type: ignore
        resolve_repo_relative_path = None  # type: ignore
        rg_repo_root = None  # type: ignore
    if rg_policy_path:
        raw_path = rg_policy_path()
    else:
        if rg_policy_path_default:
            default_path = rg_policy_path_default()
        else:
            default_path = "Config_Files/resource_governor_policies.yaml"
        raw_path = os.getenv("RG_POLICY_PATH", default_path)
    try:
        if resolve_repo_relative_path:
            cfg_path = Path(resolve_repo_relative_path(raw_path))
        else:
            cfg_path = Path(raw_path).expanduser()
            if not cfg_path.is_absolute():
                base = rg_repo_root() if rg_repo_root else Path(__file__).resolve().parents[4]
                cfg_path = (base / cfg_path).resolve()
        if not cfg_path.exists():
            logger.warning("ResourceGovernor policy file not found at {}", str(cfg_path))
    except (OSError, RuntimeError, ValueError) as exc:
        logger.debug("Failed to resolve RG policy path '{}': {}", raw_path, exc)
        cfg_path = Path(raw_path)
    reload_enabled = os.getenv("RG_POLICY_RELOAD_ENABLED", "true").lower() in {"1", "true", "yes"}
    reload_interval = int(os.getenv("RG_POLICY_RELOAD_INTERVAL_SEC", "10"))
    return PolicyLoader(str(cfg_path), PolicyReloadConfig(reload_enabled, reload_interval))


@runtime_checkable
class PolicyStoreProtocol(Protocol):
    async def get_latest_policy(self) -> tuple[int, dict[str, Any], dict[str, Any], float]:
        """
        Returns a tuple: (version, policies, tenant, updated_at_epoch_seconds)
        """
        ...


def db_policy_loader(store: PolicyStoreProtocol, reload: PolicyReloadConfig | None = None) -> PolicyLoader:
    try:
        from tldw_Server_API.app.core.config import rg_repo_root  # type: ignore
    except Exception:
        rg_repo_root = None  # type: ignore
    base = rg_repo_root() if rg_repo_root else Path(__file__).resolve().parents[4]
    cfg_path = base / "Config_Files" / "resource_governor_policies.yaml"
    # path is required by constructor but unused when store is provided
    return PolicyLoader(str(cfg_path), reload or PolicyReloadConfig(), store=store)
