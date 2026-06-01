"""Profile bootstrap helpers for standalone MCP gateway runtimes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp_unified.gateway.profiles import GatewayProfileManager, GatewayProfileStoreMetadata
from mcp_unified.interfaces.storage import AuditStore, ProfileAssignmentStore, ProfileStore
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.presets import duplicate_builtin_preset
from mcp_unified.profiles.resolver import AssignmentBackedProfileResolver
from mcp_unified.profiles.store import InMemoryProfileAssignmentStore, InMemoryProfileStore

from .profile_runtime import ProfileAwareGatewayRuntime
from .runtime import GatewayRuntime

if TYPE_CHECKING:
    from mcp_unified.gateway.external_registry import GatewayExternalRegistryManager
    from mcp_unified.gateway.external_runtime import GatewayExternalRuntimeManager


@dataclass(frozen=True, slots=True)
class GatewayProfileBootstrap:
    """Result of preparing a profile-aware standalone gateway runtime."""

    runtime: ProfileAwareGatewayRuntime
    profile_store: ProfileStore
    assignment_store: ProfileAssignmentStore
    audit_store: AuditStore | None
    profile_manager: GatewayProfileManager
    store_metadata: GatewayProfileStoreMetadata
    default_profile_id: str | None
    seeded_profile_ids: tuple[str, ...]
    external_registry_manager: GatewayExternalRegistryManager | None = None
    external_runtime_manager: GatewayExternalRuntimeManager | None = None


async def bootstrap_profile_gateway(
    backend: GatewayRuntime,
    *,
    profile_store: ProfileStore | None = None,
    assignment_store: ProfileAssignmentStore | None = None,
    audit_store: AuditStore | None = None,
    store_metadata: GatewayProfileStoreMetadata | None = None,
    profiles: Iterable[MCPProfile | Mapping[str, Any]] | None = None,
    default_profile_id: str | None = None,
    default_preset_id: str | None = None,
    external_registry_manager: GatewayExternalRegistryManager | None = None,
    external_runtime_manager: GatewayExternalRuntimeManager | None = None,
) -> GatewayProfileBootstrap:
    """Seed profile data and return a profile-aware gateway runtime."""

    store = profile_store if profile_store is not None else InMemoryProfileStore()
    assignments = (
        assignment_store
        if assignment_store is not None
        else InMemoryProfileAssignmentStore()
    )
    metadata = (
        store_metadata
        if store_metadata is not None
        else GatewayProfileStoreMetadata(kind="memory", persistent=False)
    )
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

    resolver = AssignmentBackedProfileResolver(
        store,
        assignment_store=assignments,
        fallback_default_profile_id=resolved_default_profile_id,
    )
    runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_resolver=resolver,
    )
    profile_manager = GatewayProfileManager(
        profile_store=store,
        assignment_store=assignments,
        audit_store=audit_store,
        fallback_default_profile_id=resolved_default_profile_id,
        store_metadata=metadata,
    )
    return GatewayProfileBootstrap(
        runtime=runtime,
        profile_store=store,
        assignment_store=assignments,
        audit_store=audit_store,
        profile_manager=profile_manager,
        store_metadata=metadata,
        default_profile_id=resolved_default_profile_id,
        seeded_profile_ids=seeded_profile_ids,
        external_registry_manager=external_registry_manager,
        external_runtime_manager=external_runtime_manager,
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
