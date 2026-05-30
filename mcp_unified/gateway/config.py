"""Configuration bootstrap helpers for standalone MCP gateway runtimes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mcp_unified.interfaces.storage import ProfileStore
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.store import InMemoryProfileStore

from .bootstrap import GatewayProfileBootstrap, bootstrap_profile_gateway
from .runtime import GatewayRuntime

GatewayProfileStoreKind = Literal["memory", "sqlite"]


@dataclass(frozen=True, slots=True)
class GatewayProfileStoreConfig:
    """Profile-store selection for standalone gateway bootstrap."""

    kind: GatewayProfileStoreKind = "memory"
    sqlite_path: str | Path | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the configured profile store kind."""

        normalized_kind = str(self.kind).strip().lower()
        if normalized_kind not in {"memory", "sqlite"}:
            raise ValueError(
                f"Unsupported gateway profile store kind: {self.kind!r}"
            )
        object.__setattr__(self, "kind", normalized_kind)
        if normalized_kind == "sqlite":
            if self.sqlite_path is None:
                raise ValueError("sqlite_path is required for sqlite profile store")
            if not str(self.sqlite_path).strip():
                raise ValueError("sqlite_path cannot be empty")


@dataclass(frozen=True, slots=True)
class GatewayProfileBootstrapConfig:
    """Profile bootstrap configuration for a standalone gateway runtime."""

    store: GatewayProfileStoreConfig | Mapping[str, Any] = field(
        default_factory=GatewayProfileStoreConfig
    )
    profiles: Iterable[MCPProfile | Mapping[str, Any]] = field(default_factory=tuple)
    default_profile_id: str | None = None
    default_preset_id: str | None = None

    def __post_init__(self) -> None:
        """Normalize nested config values into copy-isolated profile data."""

        store = self.store
        if isinstance(store, Mapping):
            store = GatewayProfileStoreConfig(**store)
        elif not isinstance(store, GatewayProfileStoreConfig):
            raise TypeError("store must be a GatewayProfileStoreConfig or mapping")

        object.__setattr__(self, "store", store)
        object.__setattr__(
            self,
            "profiles",
            tuple(_copy_profile(profile) for profile in self.profiles or ()),
        )


async def bootstrap_profile_gateway_from_config(
    backend: GatewayRuntime,
    config: GatewayProfileBootstrapConfig | Mapping[str, Any] | None = None,
    *,
    profile_store: ProfileStore | None = None,
) -> GatewayProfileBootstrap:
    """Build a profile-aware gateway runtime from explicit gateway config."""

    resolved_config = _validate_bootstrap_config(config)
    store = profile_store
    if store is None:
        store_config = resolved_config.store
        if isinstance(store_config, Mapping):
            store_config = GatewayProfileStoreConfig(**store_config)
        store = _build_profile_store(store_config)

    return await bootstrap_profile_gateway(
        backend,
        profile_store=store,
        profiles=resolved_config.profiles,
        default_profile_id=resolved_config.default_profile_id,
        default_preset_id=resolved_config.default_preset_id,
    )


def _validate_bootstrap_config(
    config: GatewayProfileBootstrapConfig | Mapping[str, Any] | None,
) -> GatewayProfileBootstrapConfig:
    """Return a validated bootstrap config model."""

    if config is None:
        return GatewayProfileBootstrapConfig()
    if isinstance(config, GatewayProfileBootstrapConfig):
        return config
    return GatewayProfileBootstrapConfig(**config)


def _build_profile_store(store_config: GatewayProfileStoreConfig) -> ProfileStore:
    """Create the configured profile store without importing optional stores early."""

    if store_config.kind == "memory":
        return InMemoryProfileStore()
    if store_config.kind == "sqlite":
        from mcp_unified.storage.sqlite import SQLiteMCPStore

        sqlite_path = store_config.sqlite_path
        if sqlite_path is None:
            raise ValueError("sqlite_path is required for sqlite profile store")
        return SQLiteMCPStore(sqlite_path)
    raise ValueError(
        f"Unsupported gateway profile store kind: {store_config.kind!r}"
    )


def _copy_profile(profile: MCPProfile | Mapping[str, Any]) -> MCPProfile:
    """Return a validated, copy-isolated profile config value."""

    if isinstance(profile, MCPProfile):
        return profile.model_copy(deep=True)
    return MCPProfile.model_validate(profile).model_copy(deep=True)


__all__ = [
    "GatewayProfileBootstrapConfig",
    "GatewayProfileStoreConfig",
    "GatewayProfileStoreKind",
    "bootstrap_profile_gateway_from_config",
]
