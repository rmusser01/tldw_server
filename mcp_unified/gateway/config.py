"""Configuration bootstrap helpers for standalone MCP gateway runtimes."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from json import JSONDecodeError
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from mcp_unified.federation.process_policy import (
    StdioProcessPolicy,
    coerce_stdio_process_policy,
)
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
from .admin_auth import GatewayAdminAuthConfig
from .credential_grants import GatewayCredentialGrantManager
from .external_registry import GatewayExternalRegistryManager, GatewayStoreMetadata
from .lifecycle import GatewayExternalRuntimeLifecycleConfig
from .profiles import GatewayProfileStoreMetadata
from .runtime import GatewayRuntime
from .snapshots import GatewayConfigSnapshotManager

if TYPE_CHECKING:
    from mcp_unified.federation.installers import ExternalServerInstaller
    from mcp_unified.federation.transports import ExternalFederationTransport
    from mcp_unified.gateway.external_runtime import (
        ExternalCredentialBroker,
        GatewayExternalRuntimeManager,
    )
    from mcp_unified.storage.models import ExternalServerDefinition
    from mcp_unified.tool_use_reporting import ToolUseEventStore, ToolUseRecorder

try:  # pragma: no cover - Python <3.11 fallback is exercised only on older runtimes.
    import tomllib as _tomllib
except ModuleNotFoundError:  # pragma: no cover - defensive for unsupported runtimes.
    _tomllib = None

GatewayConfigFormat = Literal["json", "toml"]
GatewayExternalRuntimeFactoryKind = Literal["stdio"]
GatewayProfileStoreKind = Literal["memory", "sqlite"]
GatewayToolUseReportingStoreKind = Literal["memory", "sqlite"]


@dataclass(frozen=True, slots=True)
class GatewayAdminAuthBootstrapConfig:
    """File-backed admin-auth bootstrap settings without persisted secrets."""

    enabled: bool = False
    header_name: str = "X-MCP-Gateway-Admin-Key"
    api_key_env_var: str = "MCP_UNIFIED_GATEWAY_ADMIN_KEY"

    def __post_init__(self) -> None:
        """Validate admin auth config values that may appear in files."""

        if not isinstance(self.enabled, bool):
            raise ValueError("admin_auth.enabled must be a boolean")
        header_name = str(self.header_name).strip()
        if not header_name:
            raise ValueError("admin_auth.header_name must be non-blank")
        api_key_env_var = str(self.api_key_env_var).strip()
        if not api_key_env_var:
            raise ValueError("admin_auth.api_key_env_var must be non-blank")
        object.__setattr__(self, "header_name", header_name)
        object.__setattr__(self, "api_key_env_var", api_key_env_var)

    def runtime_config(
        self,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> GatewayAdminAuthConfig:
        """Resolve the runtime admin auth config from the configured env var."""

        if not self.enabled:
            return GatewayAdminAuthConfig(
                enabled=False,
                header_name=self.header_name,
            )
        source = os.environ if environ is None else environ
        return GatewayAdminAuthConfig(
            enabled=True,
            header_name=self.header_name,
            api_key=source.get(self.api_key_env_var),
        )


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
class GatewayToolUseReportingStoreConfig:
    """Tool-use event-store selection for standalone gateway reporting."""

    kind: GatewayToolUseReportingStoreKind = "memory"
    sqlite_path: str | Path | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the configured tool-use event store kind."""

        normalized_kind = str(self.kind).strip().lower()
        if normalized_kind not in {"memory", "sqlite"}:
            raise ValueError(
                f"Unsupported gateway tool-use reporting store kind: {self.kind!r}"
            )
        object.__setattr__(
            self,
            "kind",
            cast(GatewayToolUseReportingStoreKind, normalized_kind),
        )
        if normalized_kind == "sqlite" and self.sqlite_path is not None:
            if not str(self.sqlite_path).strip():
                raise ValueError("tool-use reporting sqlite_path cannot be empty")


@dataclass(frozen=True, slots=True)
class GatewayToolUseReportingConfig:
    """Tool-use reporting bootstrap settings for standalone gateways.

    The memory store is process-local for tests and embedders. Standalone
    report, export, and cleanup commands require a configured SQLite store.
    """

    enabled: bool = False
    store: GatewayToolUseReportingStoreConfig | Mapping[str, Any] = field(
        default_factory=GatewayToolUseReportingStoreConfig
    )
    write_timeout_seconds: float | None = 2.0
    retention_max_age_days: int | None = None
    retention_max_events: int | None = None
    export_default_limit: int = 1000
    report_default_window: str = "24h"

    def __post_init__(self) -> None:
        """Normalize nested config and validate bounded reporting defaults."""

        if not isinstance(self.enabled, bool):
            raise ValueError("tool_use_reporting.enabled must be a boolean")
        store = self.store
        if isinstance(store, Mapping):
            store = GatewayToolUseReportingStoreConfig(**store)
        elif not isinstance(store, GatewayToolUseReportingStoreConfig):
            raise TypeError(
                "tool_use_reporting.store must be a "
                "GatewayToolUseReportingStoreConfig or mapping"
            )
        if self.enabled and store.kind == "sqlite" and store.sqlite_path is None:
            raise ValueError(
                "sqlite_path is required for sqlite tool-use reporting store"
            )

        write_timeout = self.write_timeout_seconds
        if write_timeout is not None:
            try:
                write_timeout = float(write_timeout)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "tool_use_reporting.write_timeout_seconds must be numeric or null"
                ) from exc
            if write_timeout < 0:
                raise ValueError(
                    "tool_use_reporting.write_timeout_seconds must be non-negative"
                )

        export_limit = int(self.export_default_limit)
        if export_limit <= 0:
            raise ValueError("tool_use_reporting.export_default_limit must be positive")

        object.__setattr__(self, "store", store)
        object.__setattr__(self, "write_timeout_seconds", write_timeout)
        object.__setattr__(self, "export_default_limit", export_limit)
        object.__setattr__(
            self,
            "retention_max_age_days",
            _positive_optional_int(
                self.retention_max_age_days,
                field_name="tool_use_reporting.retention_max_age_days",
            ),
        )
        object.__setattr__(
            self,
            "retention_max_events",
            _positive_optional_int(
                self.retention_max_events,
                field_name="tool_use_reporting.retention_max_events",
            ),
        )
        report_window = str(self.report_default_window).strip()
        if not report_window:
            raise ValueError(
                "tool_use_reporting.report_default_window must be non-blank"
            )
        object.__setattr__(self, "report_default_window", report_window)


@dataclass(frozen=True, slots=True)
class GatewayExternalRuntimeBootstrapConfig:
    """External runtime manager bootstrap selection for standalone gateways."""

    enabled: bool = False
    transport_factory: GatewayExternalRuntimeFactoryKind = "stdio"
    reconcile_on_startup: bool = False
    stop_on_shutdown: bool = False
    process_policy: StdioProcessPolicy | Mapping[str, Any] | None = None
    process_policy_configured: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Validate and normalize the configured runtime factory selector."""

        if not isinstance(self.enabled, bool):
            raise ValueError("external_runtime.enabled must be a boolean")
        if not isinstance(self.reconcile_on_startup, bool):
            raise ValueError("external_runtime.reconcile_on_startup must be a boolean")
        if not isinstance(self.stop_on_shutdown, bool):
            raise ValueError("external_runtime.stop_on_shutdown must be a boolean")
        if not self.enabled and (self.reconcile_on_startup or self.stop_on_shutdown):
            raise ValueError(
                "external_runtime.enabled must be true when lifecycle hooks are enabled"
            )

        normalized_factory = str(self.transport_factory).strip().lower()
        if normalized_factory != "stdio":
            raise ValueError(
                "Unsupported gateway external runtime factory: "
                f"{self.transport_factory!r}"
            )
        object.__setattr__(
            self,
            "transport_factory",
            cast(GatewayExternalRuntimeFactoryKind, normalized_factory),
        )
        process_policy_configured = self.process_policy is not None
        object.__setattr__(
            self,
            "process_policy",
            coerce_stdio_process_policy(self.process_policy),
        )
        object.__setattr__(
            self,
            "process_policy_configured",
            process_policy_configured,
        )

    def lifecycle_config(self) -> GatewayExternalRuntimeLifecycleConfig:
        """Return lifecycle preferences as the shared app lifecycle config."""

        return GatewayExternalRuntimeLifecycleConfig(
            reconcile_on_startup=self.reconcile_on_startup,
            stop_on_shutdown=self.stop_on_shutdown,
        )


@dataclass(frozen=True, slots=True)
class GatewayProfileBootstrapConfig:
    """Profile bootstrap configuration for a standalone gateway runtime."""

    store: GatewayProfileStoreConfig | Mapping[str, Any] = field(
        default_factory=GatewayProfileStoreConfig
    )
    profiles: Iterable[MCPProfile | Mapping[str, Any]] = field(default_factory=tuple)
    default_profile_id: str | None = None
    default_preset_id: str | None = None
    external_runtime: GatewayExternalRuntimeBootstrapConfig | Mapping[str, Any] = field(
        default_factory=GatewayExternalRuntimeBootstrapConfig
    )
    admin_auth: GatewayAdminAuthBootstrapConfig | Mapping[str, Any] = field(
        default_factory=GatewayAdminAuthBootstrapConfig
    )
    tool_use_reporting: GatewayToolUseReportingConfig | Mapping[str, Any] = field(
        default_factory=GatewayToolUseReportingConfig
    )

    def __post_init__(self) -> None:
        """Normalize nested config values into copy-isolated profile data."""

        store = self.store
        if isinstance(store, Mapping):
            store = GatewayProfileStoreConfig(**store)
        elif not isinstance(store, GatewayProfileStoreConfig):
            raise TypeError("store must be a GatewayProfileStoreConfig or mapping")

        external_runtime = self.external_runtime
        if isinstance(external_runtime, Mapping):
            external_runtime = GatewayExternalRuntimeBootstrapConfig(**external_runtime)
        elif not isinstance(external_runtime, GatewayExternalRuntimeBootstrapConfig):
            raise TypeError(
                "external_runtime must be a "
                "GatewayExternalRuntimeBootstrapConfig or mapping"
            )

        admin_auth = self.admin_auth
        if isinstance(admin_auth, Mapping):
            if "api_key" in admin_auth:
                raise ValueError(
                    "admin_auth.api_key must not be stored in gateway config; "
                    "use admin_auth.api_key_env_var instead"
                )
            admin_auth = GatewayAdminAuthBootstrapConfig(**admin_auth)
        elif not isinstance(admin_auth, GatewayAdminAuthBootstrapConfig):
            raise TypeError(
                "admin_auth must be a GatewayAdminAuthBootstrapConfig or mapping"
            )

        tool_use_reporting = self.tool_use_reporting
        if isinstance(tool_use_reporting, Mapping):
            tool_use_reporting = GatewayToolUseReportingConfig(**tool_use_reporting)
        elif not isinstance(tool_use_reporting, GatewayToolUseReportingConfig):
            raise TypeError(
                "tool_use_reporting must be a "
                "GatewayToolUseReportingConfig or mapping"
            )

        object.__setattr__(self, "store", store)
        object.__setattr__(self, "external_runtime", external_runtime)
        object.__setattr__(self, "admin_auth", admin_auth)
        object.__setattr__(self, "tool_use_reporting", tool_use_reporting)
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


class ExternalRegistryStorageConfigurationError(ValueError):
    """Raised when configured storage cannot support external registry management."""

    reason_code = "external_registry_store_unavailable"


class ExternalRuntimeConfigurationError(ValueError):
    """Raised when configured storage cannot support external runtime management."""

    reason_code = "external_runtime_store_unavailable"


async def bootstrap_profile_gateway_from_config(
    backend: GatewayRuntime,
    config: GatewayProfileBootstrapConfig | Mapping[str, Any] | None = None,
    *,
    profile_store: ProfileStore | None = None,
    assignment_store: ProfileAssignmentStore | None = None,
    audit_store: AuditStore | None = None,
    external_runtime_manager: GatewayExternalRuntimeManager | None = None,
    external_transport_factory: Callable[
        [ExternalServerDefinition],
        ExternalFederationTransport,
    ]
    | None = None,
    credential_broker: ExternalCredentialBroker | Callable[..., Any] | None = None,
    external_installer: ExternalServerInstaller | None = None,
) -> GatewayProfileBootstrap:
    """Build a profile-aware gateway runtime from explicit gateway config."""

    resolved_config = _validate_bootstrap_config(config)
    storage = build_gateway_profile_storage(
        resolved_config.store,
        profile_store=profile_store,
        assignment_store=assignment_store,
        audit_store=audit_store,
    )
    external_registry_manager: GatewayExternalRegistryManager | None = None
    external_storage: GatewayExternalRegistryStorageBundle | None = None
    credential_grant_manager: GatewayCredentialGrantManager | None = None
    if _supports_external_registry_store(storage.profile_store):
        external_storage = build_gateway_external_registry_storage(
            resolved_config.store,
            external_registry_store=cast(ExternalRegistryStore, storage.profile_store),
            audit_store=storage.audit_store,
        )
        external_registry_manager = external_registry_manager_from_storage(
            external_storage,
        )
        if external_storage.credential_grant_store is not None:
            credential_grant_manager = credential_grant_manager_from_storage(
                external_storage,
                profile_storage=storage,
            )

    resolved_external_runtime_manager = external_runtime_manager
    if (
        resolved_external_runtime_manager is None
        and resolved_config.external_runtime.enabled
    ):
        if external_storage is None:
            raise ExternalRuntimeConfigurationError(
                "external runtime management requires sqlite or an injected "
                "external registry-capable profile store"
            )
        resolved_external_runtime_manager = external_runtime_manager_from_storage(
            external_storage,
            transport_factory=external_transport_factory,
            process_policy=resolved_config.external_runtime.process_policy,
            process_policy_configured=resolved_config.external_runtime.process_policy_configured,
            credential_broker=credential_broker,
            installer=external_installer,
        )

    bootstrap = await bootstrap_profile_gateway(
        backend,
        profile_store=storage.profile_store,
        assignment_store=storage.assignment_store,
        audit_store=storage.audit_store,
        store_metadata=storage.metadata,
        profiles=resolved_config.profiles,
        default_profile_id=resolved_config.default_profile_id,
        default_preset_id=resolved_config.default_preset_id,
        external_registry_manager=external_registry_manager,
        external_runtime_manager=resolved_external_runtime_manager,
        external_runtime_lifecycle=resolved_config.external_runtime.lifecycle_config(),
        credential_grant_manager=credential_grant_manager,
        admin_auth=resolved_config.admin_auth.runtime_config(),
    )
    return _wrap_bootstrap_tool_use_reporting(
        bootstrap,
        resolved_config.tool_use_reporting,
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
            credential_grant_store=credential_grant_store
            if credential_grant_store is not None
            else (
                cast(CredentialGrantStore, external_registry_store)
                if _supports_credential_grant_store(external_registry_store)
                else None
            ),
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
        raise ExternalRegistryStorageConfigurationError(
            "external registry management requires sqlite or an injected "
            "equivalent external registry store"
        )

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
        f"Unsupported gateway external registry store kind: {store_config.kind!r}"
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


def external_runtime_manager_from_storage(
    bundle: GatewayExternalRegistryStorageBundle,
    *,
    transport_factory: Callable[
        [ExternalServerDefinition],
        ExternalFederationTransport,
    ]
    | None = None,
    process_policy: StdioProcessPolicy | None = None,
    process_policy_configured: bool = False,
    credential_broker: ExternalCredentialBroker | Callable[..., Any] | None = None,
    installer: ExternalServerInstaller | None = None,
) -> GatewayExternalRuntimeManager:
    """Build an external runtime manager from resolved storage dependencies."""

    from mcp_unified.federation import create_external_transport

    from .external_runtime import GatewayExternalRuntimeManager

    resolved_transport_factory = (
        create_external_transport if transport_factory is None else transport_factory
    )
    if (
        process_policy_configured
        and resolved_transport_factory is create_external_transport
    ):
        def _policy_transport_factory(
            server: ExternalServerDefinition,
        ) -> ExternalFederationTransport:
            return create_external_transport(
                server,
                process_policy=process_policy,
            )

        resolved_transport_factory = _policy_transport_factory

    return GatewayExternalRuntimeManager(
        external_registry_store=bundle.external_registry_store,
        transport_factory=resolved_transport_factory,
        audit_store=bundle.audit_store,
        credential_broker=credential_broker,
        installer=installer,
    )


def credential_grant_manager_from_storage(
    external_registry_storage: GatewayExternalRegistryStorageBundle,
    *,
    profile_storage: GatewayProfileStorageBundle | None = None,
    credential_grant_store: CredentialGrantStore | None = None,
    profile_store: ProfileStore | None = None,
    external_registry_store: ExternalRegistryStore | None = None,
    audit_store: AuditStore | None = None,
) -> GatewayCredentialGrantManager:
    """Build a credential-grant manager from resolved storage dependencies."""

    resolved_credential_store = (
        credential_grant_store
        if credential_grant_store is not None
        else external_registry_storage.credential_grant_store
    )
    if resolved_credential_store is None:
        raise ExternalRegistryStorageConfigurationError(
            "credential grant management requires a credential grant store"
        )

    resolved_profile_store = profile_store
    if resolved_profile_store is None and profile_storage is not None:
        resolved_profile_store = profile_storage.profile_store

    return GatewayCredentialGrantManager(
        credential_grant_store=resolved_credential_store,
        profile_store=resolved_profile_store,
        external_registry_store=external_registry_store
        if external_registry_store is not None
        else external_registry_storage.external_registry_store,
        audit_store=audit_store
        if audit_store is not None
        else external_registry_storage.audit_store,
        store_metadata=external_registry_storage.metadata,
    )


def gateway_config_snapshot_manager_from_storage(
    profile_storage: GatewayProfileStorageBundle,
    external_registry_storage: GatewayExternalRegistryStorageBundle,
    *,
    credential_grant_store: CredentialGrantStore | None = None,
    audit_store: AuditStore | None = None,
) -> GatewayConfigSnapshotManager:
    """Build a config snapshot manager from resolved storage dependencies."""

    resolved_credential_store = (
        credential_grant_store
        if credential_grant_store is not None
        else external_registry_storage.credential_grant_store
    )
    if resolved_credential_store is None:
        raise ExternalRegistryStorageConfigurationError(
            "config snapshots require a credential grant store"
        )

    return GatewayConfigSnapshotManager(
        profile_store=profile_storage.profile_store,
        assignment_store=profile_storage.assignment_store,
        external_registry_store=external_registry_storage.external_registry_store,
        credential_grant_store=resolved_credential_store,
        audit_store=audit_store
        if audit_store is not None
        else profile_storage.audit_store or external_registry_storage.audit_store,
    )


def build_gateway_tool_use_recorder(
    config: GatewayToolUseReportingConfig | Mapping[str, Any],
) -> ToolUseRecorder:
    """Build the configured gateway tool-use recorder."""

    if isinstance(config, Mapping):
        config = GatewayToolUseReportingConfig(**config)
    from mcp_unified.tool_use_reporting import StoreBackedToolUseRecorder

    return StoreBackedToolUseRecorder(
        _build_tool_use_event_store(config.store),
        timeout_seconds=config.write_timeout_seconds,
    )


def _wrap_bootstrap_tool_use_reporting(
    bootstrap: GatewayProfileBootstrap,
    config: GatewayToolUseReportingConfig,
) -> GatewayProfileBootstrap:
    """Wrap the runtime when gateway tool-use reporting is enabled."""

    if not config.enabled:
        return bootstrap
    from .tool_use_reporting import ToolUseReportingGatewayRuntime

    return replace(
        bootstrap,
        runtime=ToolUseReportingGatewayRuntime(
            bootstrap.runtime,
            recorder=build_gateway_tool_use_recorder(config),
            write_timeout_seconds=config.write_timeout_seconds,
        ),
    )


def _metadata_for_store_config(
    store_config: GatewayProfileStoreConfig,
) -> GatewayProfileStoreMetadata:
    """Return user-facing persistence metadata for a validated store config."""

    return GatewayProfileStoreMetadata(
        kind=store_config.kind,
        persistent=store_config.kind == "sqlite",
    )


def _positive_optional_int(value: Any, *, field_name: str) -> int | None:
    """Return a positive optional integer config value."""

    if value is None:
        return None
    resolved_value = int(value)
    if resolved_value <= 0:
        raise ValueError(f"{field_name} must be positive when set")
    return resolved_value


def _build_tool_use_event_store(
    store_config: GatewayToolUseReportingStoreConfig,
) -> ToolUseEventStore:
    """Create the configured tool-use event store without importing SQLite early."""

    if store_config.kind == "memory":
        from mcp_unified.tool_use_reporting import InMemoryToolUseEventStore

        return InMemoryToolUseEventStore()
    if store_config.kind == "sqlite":
        from mcp_unified.tool_use_reporting.sqlite import SQLiteToolUseEventStore

        sqlite_path = store_config.sqlite_path
        if sqlite_path is None:
            raise ValueError(
                "sqlite_path is required for sqlite tool-use reporting store"
            )
        return SQLiteToolUseEventStore(sqlite_path)
    raise ValueError(
        f"Unsupported gateway tool-use reporting store kind: {store_config.kind!r}"
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


def _supports_credential_grant_store(candidate: object) -> bool:
    """Return whether an object provides the credential-grant store API."""

    return all(
        callable(getattr(candidate, method_name, None))
        for method_name in (
            "get_grant",
            "list_grants",
            "create_grant",
            "upsert_grant",
            "delete_grant",
        )
    )


def _supports_external_registry_store(candidate: object) -> bool:
    """Return whether an object provides the external-registry store API."""

    return all(
        callable(getattr(candidate, method_name, None))
        for method_name in (
            "get_server",
            "list_servers",
            "list_server_definitions",
            "create_server",
            "upsert_server",
            "update_server",
            "delete_server",
        )
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
            raise ValueError(
                "sqlite_path is required for sqlite external registry store"
            )
        return SQLiteMCPStore(sqlite_path)
    raise ValueError(
        "external registry management requires sqlite or an injected equivalent "
        "external registry store"
    )


def _copy_profile(profile: MCPProfile | Mapping[str, Any]) -> MCPProfile:
    """Return a validated, copy-isolated profile config value."""

    if isinstance(profile, MCPProfile):
        return profile.model_copy(deep=True)
    return MCPProfile.model_validate(profile).model_copy(deep=True)


__all__ = [
    "GatewayAdminAuthBootstrapConfig",
    "GatewayConfigFormat",
    "GatewayExternalRuntimeBootstrapConfig",
    "GatewayExternalRuntimeFactoryKind",
    "GatewayExternalRegistryStorageBundle",
    "GatewayProfileBootstrapConfig",
    "GatewayProfileStoreConfig",
    "GatewayProfileStoreKind",
    "GatewayProfileStorageBundle",
    "GatewayToolUseReportingConfig",
    "GatewayToolUseReportingStoreConfig",
    "GatewayToolUseReportingStoreKind",
    "bootstrap_profile_gateway_from_config",
    "build_gateway_external_registry_storage",
    "build_gateway_profile_storage",
    "build_gateway_tool_use_recorder",
    "credential_grant_manager_from_storage",
    "external_registry_manager_from_storage",
    "external_runtime_manager_from_storage",
    "load_gateway_profile_bootstrap_config",
]
