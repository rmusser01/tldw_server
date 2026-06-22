"""Lifecycle configuration for standalone MCP gateway runtime integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GatewayExternalRuntimeLifecycleConfig:
    """Opt-in startup and shutdown behavior for external runtime managers."""

    reconcile_on_startup: bool = False
    stop_on_shutdown: bool = False

    def __post_init__(self) -> None:
        """Validate lifecycle toggles as explicit booleans."""

        if not isinstance(self.reconcile_on_startup, bool):
            raise ValueError("external runtime reconcile_on_startup must be a boolean")
        if not isinstance(self.stop_on_shutdown, bool):
            raise ValueError("external runtime stop_on_shutdown must be a boolean")

    @property
    def enabled(self) -> bool:
        """Return whether any lifecycle behavior is enabled."""

        return self.reconcile_on_startup or self.stop_on_shutdown


def normalize_external_runtime_lifecycle_config(
    config: GatewayExternalRuntimeLifecycleConfig | Mapping[str, Any] | None,
) -> GatewayExternalRuntimeLifecycleConfig:
    """Return a validated external runtime lifecycle config."""

    if config is None:
        return GatewayExternalRuntimeLifecycleConfig()
    if isinstance(config, GatewayExternalRuntimeLifecycleConfig):
        return config
    if isinstance(config, Mapping):
        return GatewayExternalRuntimeLifecycleConfig(**config)
    raise TypeError(
        "external_runtime_lifecycle must be a "
        "GatewayExternalRuntimeLifecycleConfig, mapping, or None"
    )


__all__ = [
    "GatewayExternalRuntimeLifecycleConfig",
    "normalize_external_runtime_lifecycle_config",
]
