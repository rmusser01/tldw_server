"""Profile bootstrap helpers for standalone MCP gateway runtimes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from mcp_unified.interfaces.storage import ProfileStore
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.presets import duplicate_builtin_preset
from mcp_unified.profiles.store import InMemoryProfileStore

from .profile_runtime import ProfileAwareGatewayRuntime
from .runtime import GatewayRuntime


@dataclass(frozen=True, slots=True)
class GatewayProfileBootstrap:
    """Result of preparing a profile-aware standalone gateway runtime."""

    runtime: ProfileAwareGatewayRuntime
    profile_store: ProfileStore
    default_profile_id: str | None
    seeded_profile_ids: tuple[str, ...]


async def bootstrap_profile_gateway(
    backend: GatewayRuntime,
    *,
    profile_store: ProfileStore | None = None,
    profiles: Iterable[MCPProfile | Mapping[str, Any]] | None = None,
    default_profile_id: str | None = None,
    default_preset_id: str | None = None,
) -> GatewayProfileBootstrap:
    """Seed profile data and return a profile-aware gateway runtime."""

    store = profile_store if profile_store is not None else InMemoryProfileStore()
    for profile in profiles or ():
        await store.upsert_profile(_validate_profile(profile))

    seeded_profile_ids: tuple[str, ...] = ()
    resolved_default_profile_id = default_profile_id
    if default_preset_id is not None:
        resolved_default_profile_id = default_profile_id or default_preset_id
        if await store.get_profile(default_preset_id) is not None:
            raise ValueError(
                f"Cannot seed MCP profile preset '{default_preset_id}': "
                f"profile id '{default_preset_id}' already exists"
            )
        preset_profile = duplicate_builtin_preset(
            default_preset_id,
            profile_id=default_preset_id,
        )
        await store.upsert_profile(preset_profile)
        seeded_profile_ids = (preset_profile.id,)

    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=store,
        default_profile_id=resolved_default_profile_id,
    )
    return GatewayProfileBootstrap(
        runtime=runtime,
        profile_store=store,
        default_profile_id=resolved_default_profile_id,
        seeded_profile_ids=seeded_profile_ids,
    )


async def build_profile_gateway_runtime(
    backend: GatewayRuntime,
    *,
    profile_store: ProfileStore | None = None,
    profiles: Iterable[MCPProfile | Mapping[str, Any]] | None = None,
    default_profile_id: str | None = None,
    default_preset_id: str | None = None,
) -> ProfileAwareGatewayRuntime:
    """Return only the profile-aware runtime for simple gateway callers."""

    bootstrap = await bootstrap_profile_gateway(
        backend,
        profile_store=profile_store,
        profiles=profiles,
        default_profile_id=default_profile_id,
        default_preset_id=default_preset_id,
    )
    return bootstrap.runtime


def _validate_profile(profile: MCPProfile | Mapping[str, Any]) -> MCPProfile:
    """Return a validated caller-owned profile model for storage."""

    if isinstance(profile, MCPProfile):
        return profile.model_copy(deep=True)
    return MCPProfile.model_validate(profile)


__all__ = [
    "GatewayProfileBootstrap",
    "bootstrap_profile_gateway",
    "build_profile_gateway_runtime",
]
