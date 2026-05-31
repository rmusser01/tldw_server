"""Configuration bootstrap helpers for standalone MCP gateway runtimes."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Literal, cast

from mcp_unified.interfaces.storage import (
    AuditStore,
    CredentialGrantStore,
    ExternalRegistryStore,
    ProfileAssignmentStore,
    ProfileStore,
)
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.store import (
    InMemoryProfileAssignmentStore,
    InMemoryProfileStore,
)

from .bootstrap import GatewayProfileBootstrap, bootstrap_profile_gateway
from .external_registry import GatewayExternalRegistryManager, GatewayStoreMetadata
from .profiles import GatewayProfileStoreMetadata
from .runtime import GatewayRuntime

try:  # pragma: no cover - Python <3.11 fallback is exercised only on older runtimes.
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - defensive for unsupported runtimes.
    _tomllib = None

GatewayConfigFormat = Literal["json", "toml"]
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


@dataclass(frozen=True, slots=True)
class GatewayProfileStorageBundle:
    """Resolved profile, assignment, audit stores and persistence metadata."""

    profile_store: ProfileStore
    assignment_store: ProfileAssignmentStore
    audit_store: AuditStore | None
    metadata: GatewayProfileStoreMetadata


@dataclass(frozen=True, slots=True)
class GatewayExternalRegistryStorageBundle:
    """Resolved external registry, credential grant, audit stores and metadata."""

    external_registry_store: ExternalRegistryStore
    credential_grant_store: CredentialGrantStore | None
    audit_store: AuditStore | None
    metadata: GatewayStoreMetadata


async def bootstrap_profile_gateway_from_config(
    backend: GatewayRuntime,
    config: GatewayProfileBootstrapConfig | Mapping[str, Any] | None = None,
    *,
    profile_store: ProfileStore | None = None,
    assignment_store: ProfileAssignmentStore | None = None,
    audit_store: AuditStore | None = None,
) -> GatewayProfileBootstrap:
    """Build a profile-aware gateway runtime from explicit gateway config."""

    resolved_config = _validate_bootstrap_config(config)
    storage = build_gateway_profile_storage(
        resolved_config.store,
        profile_store=profile_store,
        assignment_store=assignment_store,
        audit_store=audit_store,
    )

    return await bootstrap_profile_gateway(
        backend,
        profile_store=storage.profile_store,
        assignment_store=storage.assignment_store,
        audit_store=storage.audit_store,
        store_metadata=storage.metadata,
        profiles=resolved_config.profiles,
        default_profile_id=resolved_config.default_profile_id,
        default_preset_id=resolved_config.default_preset_id,
    )


def build_gateway_profile_storage(
    store_config: GatewayProfileStoreConfig | Mapping[str, Any],
    *,
    profile_store: ProfileStore | None = None,
    assignment_store: ProfileAssignmentStore | None = None,
    audit_store: AuditStore | None = None,
) -> GatewayProfileStorageBundle:
    """Resolve configured gateway profile storage dependencies."""

    if isinstance(store_config, Mapping):
        store_config = GatewayProfileStoreConfig(**store_config)

    if profile_store is not None:
        return GatewayProfileStorageBundle(
            profile_store=profile_store,
            assignment_store=_resolve_injected_assignment_store(
                store_config,
                profile_store,
                assignment_store,
            ),
            audit_store=_resolve_injected_audit_store(
                store_config,
                profile_store,
                audit_store,
            ),
            metadata=_metadata_for_store_config(store_config),
        )

    if store_config.kind == "memory":
        return GatewayProfileStorageBundle(
            profile_store=InMemoryProfileStore(),
            assignment_store=assignment_store
            if assignment_store is not None
            else InMemoryProfileAssignmentStore(),
            audit_store=audit_store,
            metadata=GatewayProfileStoreMetadata(kind="memory", persistent=False),
        )

    if store_config.kind == "sqlite":
        store = _build_profile_store(store_config)
        return GatewayProfileStorageBundle(
            profile_store=store,
            assignment_store=assignment_store
            if assignment_store is not None
            else cast(ProfileAssignmentStore, store),
            audit_store=audit_store if audit_store is not None else cast(AuditStore, store),
            metadata=GatewayProfileStoreMetadata(kind="sqlite", persistent=True),
        )

    raise ValueError(
        f"Unsupported gateway profile store kind: {store_config.kind!r}"
    )


def build_gateway_external_registry_storage(
    store_config: GatewayProfileStoreConfig | Mapping[str, Any],
    *,
    external_registry_store: ExternalRegistryStore | None = None,
    credential_grant_store: CredentialGrantStore | None = None,
    audit_store: AuditStore | None = None,
) -> GatewayExternalRegistryStorageBundle:
    """Resolve configured gateway external registry storage dependencies."""

    if isinstance(store_config, Mapping):
        store_config = GatewayProfileStoreConfig(**store_config)

    if external_registry_store is not None:
        return GatewayExternalRegistryStorageBundle(
            external_registry_store=external_registry_store,
            credential_grant_store=credential_grant_store,
            audit_store=audit_store
            if audit_store is not None
            else (
                cast(AuditStore, external_registry_store)
                if _supports_audit_store(external_registry_store)
                else None
            ),
            metadata=GatewayStoreMetadata(
                kind=store_config.kind,
                persistent=store_config.kind == "sqlite",
            ),
        )

    if store_config.kind == "memory":
        raise ValueError("external registry management requires sqlite store")

    if store_config.kind == "sqlite":
        store = _build_external_registry_store(store_config)
        return GatewayExternalRegistryStorageBundle(
            external_registry_store=store,
            credential_grant_store=credential_grant_store
            if credential_grant_store is not None
            else cast(CredentialGrantStore, store),
            audit_store=audit_store if audit_store is not None else cast(AuditStore, store),
            metadata=GatewayStoreMetadata(kind="sqlite", persistent=True),
        )

    raise ValueError(
        f"Unsupported gateway profile store kind: {store_config.kind!r}"
    )


def external_registry_manager_from_storage(
    bundle: GatewayExternalRegistryStorageBundle,
) -> GatewayExternalRegistryManager:
    """Build an external registry manager from resolved storage dependencies."""

    return GatewayExternalRegistryManager(
        external_registry_store=bundle.external_registry_store,
        credential_grant_store=bundle.credential_grant_store,
        audit_store=bundle.audit_store,
        store_metadata=bundle.metadata,
    )


def _metadata_for_store_config(
    store_config: GatewayProfileStoreConfig,
) -> GatewayProfileStoreMetadata:
    """Return user-facing persistence metadata for a validated store config."""

    return GatewayProfileStoreMetadata(
        kind=store_config.kind,
        persistent=store_config.kind == "sqlite",
    )


def _resolve_injected_assignment_store(
    store_config: GatewayProfileStoreConfig,
    profile_store: ProfileStore,
    assignment_store: ProfileAssignmentStore | None,
) -> ProfileAssignmentStore:
    """Resolve assignment storage for caller-injected profile stores."""

    if assignment_store is not None:
        return assignment_store
    if _supports_assignment_store(profile_store):
        return cast(ProfileAssignmentStore, profile_store)
    if store_config.kind == "memory":
        return InMemoryProfileAssignmentStore()
    raise ValueError(
        "assignment_store is required when injecting a profile_store for "
        "sqlite gateway profile storage unless profile_store implements "
        "profile assignment methods"
    )


def _resolve_injected_audit_store(
    store_config: GatewayProfileStoreConfig,
    profile_store: ProfileStore,
    audit_store: AuditStore | None,
) -> AuditStore | None:
    """Reuse audit-capable injected stores when no audit store is supplied."""

    if audit_store is not None:
        return audit_store
    if _supports_audit_store(profile_store):
        return cast(AuditStore, profile_store)
    if store_config.kind == "sqlite":
        raise ValueError(
            "audit_store is required when injecting a profile_store for "
            "sqlite gateway profile storage unless profile_store implements "
            "audit methods"
        )
    return None


def _supports_assignment_store(candidate: object) -> bool:
    """Return whether an object provides the profile-assignment store API."""

    return all(
        callable(getattr(candidate, method_name, None))
        for method_name in (
            "get_assignment",
            "list_assignments",
            "upsert_assignment",
            "delete_assignment",
        )
    )


def _supports_audit_store(candidate: object) -> bool:
    """Return whether an object provides the audit store API."""

    return all(
        callable(getattr(candidate, method_name, None))
        for method_name in ("append_event", "query_events")
    )


def load_gateway_profile_bootstrap_config(
    path: str | Path,
    *,
    format: GatewayConfigFormat | str | None = None,
) -> GatewayProfileBootstrapConfig:
    """Load gateway profile bootstrap config from a JSON or TOML file.

    The file format is inferred from `.json` or `.toml` suffixes unless an
    explicit `format` is supplied. The parsed payload must be a top-level
    object accepted by `GatewayProfileBootstrapConfig`. Invalid formats,
    unreadable files, malformed payloads, non-object payloads, and config
    schema/type errors raise `ValueError` with user-facing context.
    """

    config_path = Path(path)
    config_format = _detect_config_format(config_path, format)
    try:
        raw_payload = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Unable to read gateway config file: {config_path}") from exc

    payload = _parse_config_payload(raw_payload, config_format)
    if not isinstance(payload, Mapping):
        raise ValueError("Gateway config file must contain an object")
    try:
        return GatewayProfileBootstrapConfig(**payload)
    except TypeError as exc:
        raise ValueError(f"Invalid gateway config schema or types: {exc}") from exc


def _validate_bootstrap_config(
    config: GatewayProfileBootstrapConfig | Mapping[str, Any] | None,
) -> GatewayProfileBootstrapConfig:
    """Return a validated bootstrap config model."""

    if config is None:
        return GatewayProfileBootstrapConfig()
    if isinstance(config, GatewayProfileBootstrapConfig):
        return config
    return GatewayProfileBootstrapConfig(**config)


def _detect_config_format(
    path: Path,
    explicit_format: GatewayConfigFormat | str | None,
) -> GatewayConfigFormat:
    """Infer or validate the gateway config file format."""

    if explicit_format is not None:
        normalized_format = str(explicit_format).strip().lower()
        if normalized_format in {"json", "toml"}:
            return cast(GatewayConfigFormat, normalized_format)
        raise ValueError(
            f"Unsupported gateway config format: {explicit_format!r}"
        )

    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".toml":
        return "toml"
    raise ValueError(
        f"Unsupported gateway config format for path: {path}"
    )


def _parse_config_payload(raw_payload: str, config_format: GatewayConfigFormat) -> Any:
    """Parse one gateway config payload by validated format."""

    if config_format == "json":
        try:
            return json.loads(raw_payload)
        except JSONDecodeError as exc:
            raise ValueError(f"Invalid gateway config JSON: {exc}") from exc

    if config_format == "toml":
        if _tomllib is None:
            raise ValueError("Gateway TOML config loading requires Python 3.11 tomllib")
        try:
            return _tomllib.loads(raw_payload)
        except ValueError as exc:
            raise ValueError(f"Invalid gateway config TOML: {exc}") from exc

    raise ValueError(f"Unsupported gateway config format: {config_format!r}")


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


def _build_external_registry_store(
    store_config: GatewayProfileStoreConfig,
) -> ExternalRegistryStore:
    """Create the configured external registry store without early imports."""

    if store_config.kind == "sqlite":
        from mcp_unified.storage.sqlite import SQLiteMCPStore

        sqlite_path = store_config.sqlite_path
        if sqlite_path is None:
            raise ValueError("sqlite_path is required for sqlite profile store")
        return SQLiteMCPStore(sqlite_path)
    raise ValueError("external registry management requires sqlite store")


def _copy_profile(profile: MCPProfile | Mapping[str, Any]) -> MCPProfile:
    """Return a validated, copy-isolated profile config value."""

    if isinstance(profile, MCPProfile):
        return profile.model_copy(deep=True)
    return MCPProfile.model_validate(profile).model_copy(deep=True)


__all__ = [
    "GatewayConfigFormat",
    "GatewayExternalRegistryStorageBundle",
    "GatewayProfileBootstrapConfig",
    "GatewayProfileStoreConfig",
    "GatewayProfileStoreKind",
    "GatewayProfileStorageBundle",
    "bootstrap_profile_gateway_from_config",
    "build_gateway_external_registry_storage",
    "build_gateway_profile_storage",
    "external_registry_manager_from_storage",
    "load_gateway_profile_bootstrap_config",
]
