from __future__ import annotations

"""
Shared provider registry foundation.

This module provides a reusable registry for provider-backed adapter domains
such as LLM, TTS, and STT. It is intentionally domain-agnostic: wrappers can
enforce adapter interfaces and domain-specific capability payloads.
"""

import importlib
import math
import asyncio
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Generic, TypeVar

from loguru import logger

AdapterT = TypeVar("AdapterT")


class ProviderStatus(str, Enum):
    """Canonical availability state for a provider entry."""

    ENABLED = "enabled"
    FAILED = "failed"
    DISABLED = "disabled"
    UNKNOWN = "unknown"

    @classmethod
    def from_value(cls, value: Any) -> "ProviderStatus":
        """Normalize raw domain status values into canonical provider status."""
        if isinstance(value, cls):
            return value

        # Accept foreign Enums by preferring .value, then .name
        if isinstance(value, Enum):
            raw = getattr(value, "value", None)
            if raw is None:
                raw = getattr(value, "name", None)
        else:
            raw = value

        normalized = str(raw or "").strip().lower().replace("_", "-")
        if not normalized:
            return cls.UNKNOWN

        enabled_aliases = {"enabled", "enable", "available", "ready", "healthy", "ok"}
        failed_aliases = {"failed", "fail", "error", "unhealthy", "unavailable", "circuit-open"}
        disabled_aliases = {"disabled", "disable", "not-configured", "off", "inactive"}
        unknown_aliases = {"unknown", "initializing", "pending", "not-initialized"}

        if normalized in enabled_aliases:
            return cls.ENABLED
        if normalized in failed_aliases:
            return cls.FAILED
        if normalized in disabled_aliases:
            return cls.DISABLED
        if normalized in unknown_aliases:
            return cls.UNKNOWN
        return cls.UNKNOWN


@dataclass
class ProviderRegistryConfig:
    """
    Base registry behavior toggles.

    Attributes:
        failure_retry_seconds:
            Retry window after initialization failure. `None` or non-positive
            values disable retries for failed providers (infinite backoff).
        normalize_names:
            When true, provider names are normalized using `normalize_name`
            (or the default normalizer when no callback is set).
        normalize_name:
            Optional callback to normalize provider names. Wrappers can inject
            domain-specific name canonicalization.
        provider_enabled_callback:
            Optional callback to determine config-driven provider enablement.
            Return `False` to force-disabled a provider, `True` to keep it
            enabled, and `None` to defer to the registry record state.
    """

    failure_retry_seconds: float | None = None
    normalize_names: bool = True
    normalize_name: Callable[[str], str] | None = None
    provider_enabled_callback: Callable[[str], bool | None] | None = None


@dataclass
class _ProviderRecord(Generic[AdapterT]):
    spec: Any
    enabled: bool = True
    generation: int = 0


class ProviderRegistryBase(Generic[AdapterT]):
    """
    Shared, domain-agnostic provider registry.

    Public API:
        - register_adapter
        - register_alias
        - get_adapter
        - list_providers
        - resolve_provider_name
        - get_status / get_status_map
        - set_provider_enabled / enable_provider / disable_provider
    """

    def __init__(
        self,
        *,
        config: ProviderRegistryConfig | None = None,
        adapter_spec_validator: Callable[[Any], bool] | None = None,
        adapter_validator: Callable[[Any], bool] | None = None,
        adapter_materializer: Callable[[str, Any], AdapterT] | None = None,
        adapter_materializer_async: Callable[[str, Any], Awaitable[AdapterT]] | None = None,
        adapter_disposer: Callable[[str, AdapterT], None] | None = None,
        adapter_disposer_async: Callable[[str, AdapterT], Awaitable[None]] | None = None,
        status_mapper: Callable[[Any], Any] | None = None,
        provider_enabled_callback: Callable[[str], bool | None] | None = None,
        aliases: dict[str, str] | None = None,
        normalize_name: Callable[[str], str] | None = None,
    ) -> None:
        self._config = config or ProviderRegistryConfig()
        self._adapter_spec_validator = adapter_spec_validator
        self._adapter_validator = adapter_validator
        self._adapter_materializer = adapter_materializer
        self._adapter_materializer_async = adapter_materializer_async
        self._adapter_disposer = adapter_disposer
        self._adapter_disposer_async = adapter_disposer_async
        self._status_mapper = status_mapper
        self._provider_enabled_callback = (
            provider_enabled_callback or self._config.provider_enabled_callback
        )
        self._normalizer = (
            normalize_name
            or self._config.normalize_name
            or self._default_normalize_name
        )

        self._providers: dict[str, _ProviderRecord[AdapterT]] = {}
        self._aliases: dict[str, str] = {}
        self._adapter_cache: dict[str, AdapterT] = {}
        self._failed_providers: dict[str, float] = {}
        self._sync_init_locks: dict[str, threading.Lock] = {}
        self._async_init_locks: dict[str, asyncio.Lock] = {}
        self._lock = threading.RLock()

        if aliases:
            for alias, provider_name in aliases.items():
                self.register_alias(alias, provider_name)

    @property
    def config(self) -> ProviderRegistryConfig:
        return self._config

    @staticmethod
    def _default_normalize_name(name: str) -> str:
        return str(name).strip().lower().replace("_", "-")

    def _normalize_name(self, name: str | None) -> str:
        raw = str(name or "").strip()
        if not raw:
            return ""
        if not self._config.normalize_names:
            return raw
        normalized = self._normalizer(raw)
        return str(normalized).strip()

    def normalize_provider_name(self, name: str | None) -> str:
        """Return the canonical provider key before alias resolution."""
        return self._normalize_name(name)

    def resolve_provider_name(self, name: str | None) -> str:
        normalized = self._normalize_name(name)
        if not normalized:
            return ""
        return self._aliases.get(normalized, normalized)

    def register_alias(self, alias: str, provider_name: str) -> None:
        alias_key = self._normalize_name(alias)
        provider_key = self._normalize_name(provider_name)
        if not alias_key or not provider_key:
            raise ValueError("Alias and provider_name must be non-empty")
        with self._lock:
            self._aliases[alias_key] = provider_key

    def register_adapter(
        self,
        name: str,
        adapter: Any,
        *,
        aliases: list[str] | tuple[str, ...] | set[str] | None = None,
        enabled: bool = True,
    ) -> None:
        """
        Register a provider adapter spec.

        Supported adapter specs:
            - Adapter class (instantiated lazily)
            - Adapter instance (reused as-is)
            - Dotted path string ("package.module.ClassName")

        Validation behavior:
            - `adapter_spec_validator` is applied at registration time.
              For dotted paths, the class is resolved before validation.
            - `adapter_validator` is applied at registration time for
              instance adapters, and again after lazy materialization in
              `get_adapter`.
        """

        provider_key = self._normalize_name(name)
        if not provider_key:
            raise ValueError("Provider name must be non-empty")

        self._validate_registration_spec(provider_key=provider_key, spec=adapter)

        with self._lock:
            previous = self._providers.get(provider_key)
            generation = previous.generation + 1 if previous is not None else 0
            self._providers[provider_key] = _ProviderRecord(
                spec=adapter,
                enabled=bool(enabled),
                generation=generation,
            )
            self._invalidate_provider_state_locked(provider_key)

        if aliases:
            for alias in aliases:
                self.register_alias(alias, provider_key)

    def set_provider_enabled_callback(
        self,
        callback: Callable[[str], bool | None] | None,
    ) -> None:
        """
        Set or replace the config-driven provider enablement callback.
        """
        with self._lock:
            self._provider_enabled_callback = callback

    def _resolve_adapter_class(self, spec: Any) -> type[Any]:
        if isinstance(spec, str):
            module_path, _, class_name = spec.rpartition(".")
            if not module_path:
                raise ImportError(f"Invalid adapter spec '{spec}'")
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            if not isinstance(cls, type):
                raise TypeError(f"Resolved adapter spec '{spec}' is not a class")
            return cls
        if isinstance(spec, type):
            return spec
        raise TypeError(f"Adapter spec must be class or dotted path string, got {type(spec)!r}")

    def _materialize_adapter(self, provider_key: str, spec: Any) -> AdapterT:
        if self._adapter_materializer is not None:
            return self._adapter_materializer(provider_key, spec)
        if isinstance(spec, str) or isinstance(spec, type):
            adapter_cls = self._resolve_adapter_class(spec)
            return adapter_cls()  # type: ignore[return-value,call-arg]
        return spec

    def _invalidate_provider_state_locked(self, provider_key: str) -> None:
        """Clear cached adapter and failure markers for a provider key."""
        self._adapter_cache.pop(provider_key, None)
        self._failed_providers.pop(provider_key, None)

    def _dispose_stale_adapter(self, provider_key: str, adapter: AdapterT) -> None:
        """Dispose one stale materialization without masking its discarded result."""
        if self._adapter_disposer is None:
            return
        try:
            self._adapter_disposer(provider_key, adapter)
        except Exception as exc:
            logger.warning(
                "Failed to dispose stale adapter for provider '{}' ({})",
                provider_key,
                type(exc).__name__,
            )

    async def _dispose_stale_adapter_async(
        self,
        provider_key: str,
        adapter: AdapterT,
    ) -> None:
        """Dispose one stale async materialization outside registry locks."""
        if self._adapter_disposer_async is None:
            self._dispose_stale_adapter(provider_key, adapter)
            return
        try:
            await self._adapter_disposer_async(provider_key, adapter)
        except Exception as exc:
            logger.warning(
                "Failed to dispose stale adapter for provider '{}' ({})",
                provider_key,
                type(exc).__name__,
            )

    def invalidate_provider(self, name: str) -> None:
        """Invalidate one provider and supersede any in-flight materialization."""
        provider_key = self.resolve_provider_name(name)
        if not provider_key:
            return
        with self._lock:
            record = self._providers.get(provider_key)
            if record is None:
                return
            record.generation += 1
            self._invalidate_provider_state_locked(provider_key)

    def _validate_registration_spec(self, *, provider_key: str, spec: Any) -> None:
        candidate_for_spec = spec
        if self._adapter_spec_validator is not None and isinstance(spec, (str, type)):
            candidate_for_spec = self._resolve_adapter_class(spec)

        if self._adapter_spec_validator is not None:
            try:
                valid_spec = bool(self._adapter_spec_validator(candidate_for_spec))
            except Exception as exc:
                raise TypeError(
                    f"Adapter spec validation failed for provider '{provider_key}'"
                ) from exc
            if not valid_spec:
                raise TypeError(
                    f"Adapter spec is not valid for provider '{provider_key}'"
                )

        if self._adapter_validator is not None and not isinstance(spec, (str, type)):
            try:
                valid_instance = bool(self._adapter_validator(spec))
            except Exception as exc:
                raise TypeError(
                    f"Adapter validation failed for provider '{provider_key}'"
                ) from exc
            if not valid_instance:
                raise TypeError(
                    f"Adapter instance is not valid for provider '{provider_key}'"
                )

    def _mark_failure_locked(self, provider_key: str) -> None:
        retry_seconds = self._config.failure_retry_seconds
        if retry_seconds is None or retry_seconds <= 0:
            self._failed_providers[provider_key] = math.inf
        else:
            self._failed_providers[provider_key] = time.time() + retry_seconds

    def _evaluate_config_enabled_locked(self, provider_key: str) -> bool | None:
        callback = self._provider_enabled_callback
        if callback is None:
            return None
        try:
            configured = callback(provider_key)
        except Exception as exc:
            logger.warning(
                "Provider enablement callback failed for provider '{}': {}",
                provider_key,
                exc,
            )
            return None
        if configured is None:
            return None
        return bool(configured)

    def _is_effectively_enabled_locked(
        self,
        provider_key: str,
        record: _ProviderRecord[AdapterT],
    ) -> bool:
        if not record.enabled:
            return False
        configured_enabled = self._evaluate_config_enabled_locked(provider_key)
        if configured_enabled is None:
            return True
        return configured_enabled

    def _has_active_failure_locked(self, provider_key: str) -> bool:
        retry_after = self._failed_providers.get(provider_key)
        if retry_after is None:
            return False
        if math.isinf(retry_after):
            return True
        if retry_after > time.time():
            return True
        # Retry window has elapsed.
        self._failed_providers.pop(provider_key, None)
        return False

    def set_provider_enabled(self, name: str, enabled: bool) -> None:
        provider_key = self.resolve_provider_name(name)
        if not provider_key:
            raise ValueError("Provider name must be non-empty")
        with self._lock:
            record = self._providers.get(provider_key)
            if record is None:
                raise KeyError(f"Provider '{name}' is not registered")
            record.enabled = bool(enabled)
            if not enabled:
                self._invalidate_provider_state_locked(provider_key)

    def enable_provider(self, name: str) -> None:
        self.set_provider_enabled(name, True)

    def disable_provider(self, name: str) -> None:
        self.set_provider_enabled(name, False)

    def get_adapter(self, name: str | None) -> AdapterT | None:
        provider_key = self.resolve_provider_name(name)
        if not provider_key:
            return None

        with self._lock:
            record = self._providers.get(provider_key)
            if record is None:
                return None
            if not self._is_effectively_enabled_locked(provider_key, record):
                self._invalidate_provider_state_locked(provider_key)
                return None
            if self._has_active_failure_locked(provider_key):
                return None
            cached = self._adapter_cache.get(provider_key)
            if cached is not None:
                return cached

        init_lock = self._get_or_create_sync_lock(provider_key)
        with init_lock:
            with self._lock:
                record = self._providers.get(provider_key)
                if record is None:
                    return None
                if not self._is_effectively_enabled_locked(provider_key, record):
                    self._invalidate_provider_state_locked(provider_key)
                    return None
                if self._has_active_failure_locked(provider_key):
                    return None
                cached = self._adapter_cache.get(provider_key)
                if cached is not None:
                    return cached
                spec = record.spec
                generation = record.generation

            try:
                adapter = self._materialize_adapter(provider_key, spec)
                if self._adapter_validator and not self._adapter_validator(adapter):
                    raise TypeError(f"Adapter for provider '{provider_key}' failed validation")
            except Exception as exc:
                with self._lock:
                    current = self._providers.get(provider_key)
                    if current is record and current.generation == generation:
                        self._mark_failure_locked(provider_key)
                logger.error("Failed to initialize adapter for provider '{}': {}", provider_key, exc)
                return None

            stale = False
            with self._lock:
                current = self._providers.get(provider_key)
                if (
                    current is not record
                    or current.generation != generation
                ):
                    stale = True
                elif not self._is_effectively_enabled_locked(provider_key, current):
                    self._invalidate_provider_state_locked(provider_key)
                    return None
                else:
                    cached = self._adapter_cache.get(provider_key)
                    if cached is not None:
                        return cached
                    self._adapter_cache[provider_key] = adapter
                    self._failed_providers.pop(provider_key, None)
                    return adapter

            if stale:
                self._dispose_stale_adapter(provider_key, adapter)
            return None

    def _get_or_create_sync_lock(self, provider_key: str) -> threading.Lock:
        with self._lock:
            lock = self._sync_init_locks.get(provider_key)
            if lock is None:
                lock = threading.Lock()
                self._sync_init_locks[provider_key] = lock
            return lock

    def _get_or_create_async_lock(self, provider_key: str) -> asyncio.Lock:
        with self._lock:
            lock = self._async_init_locks.get(provider_key)
            if lock is None:
                lock = asyncio.Lock()
                self._async_init_locks[provider_key] = lock
            return lock

    async def get_adapter_async(self, name: str | None) -> AdapterT | None:
        """
        Async variant of `get_adapter` for wrappers that require async
        initialization during materialization.
        """
        provider_key = self.resolve_provider_name(name)
        if not provider_key:
            return None

        with self._lock:
            record = self._providers.get(provider_key)
            if record is None:
                return None
            if not record.enabled:
                return None
            if self._has_active_failure_locked(provider_key):
                return None
            cached = self._adapter_cache.get(provider_key)
            if cached is not None:
                return cached

        init_lock = self._get_or_create_async_lock(provider_key)
        async with init_lock:
            with self._lock:
                record = self._providers.get(provider_key)
                if record is None:
                    return None
                if not self._is_effectively_enabled_locked(provider_key, record):
                    self._invalidate_provider_state_locked(provider_key)
                    return None
                if self._has_active_failure_locked(provider_key):
                    return None
                cached = self._adapter_cache.get(provider_key)
                if cached is not None:
                    return cached
                spec = record.spec
                generation = record.generation

            try:
                if self._adapter_materializer_async is not None:
                    adapter = await self._adapter_materializer_async(provider_key, spec)
                else:
                    adapter = self._materialize_adapter(provider_key, spec)
                if self._adapter_validator and not self._adapter_validator(adapter):
                    raise TypeError(f"Adapter for provider '{provider_key}' failed validation")
            except Exception as exc:
                with self._lock:
                    current = self._providers.get(provider_key)
                    if current is record and current.generation == generation:
                        self._mark_failure_locked(provider_key)
                logger.error("Failed to initialize adapter for provider '{}': {}", provider_key, exc)
                return None

            stale = False
            with self._lock:
                current = self._providers.get(provider_key)
                if (
                    current is not record
                    or current.generation != generation
                ):
                    stale = True
                else:
                    self._adapter_cache[provider_key] = adapter
                    self._failed_providers.pop(provider_key, None)
            if stale:
                await self._dispose_stale_adapter_async(provider_key, adapter)
                return None
            return adapter

    def list_providers(self, *, include_disabled: bool = True) -> list[str]:
        with self._lock:
            providers = []
            for provider_key, record in self._providers.items():
                if include_disabled or self._is_effectively_enabled_locked(provider_key, record):
                    providers.append(provider_key)
        return sorted(providers)

    def get_status(self, name: str | None) -> ProviderStatus:
        provider_key = self.resolve_provider_name(name)
        if not provider_key:
            return ProviderStatus.UNKNOWN
        with self._lock:
            record = self._providers.get(provider_key)
            if record is None:
                return ProviderStatus.UNKNOWN
            if not self._is_effectively_enabled_locked(provider_key, record):
                self._invalidate_provider_state_locked(provider_key)
                return ProviderStatus.DISABLED
            if self._has_active_failure_locked(provider_key):
                return ProviderStatus.FAILED
            return ProviderStatus.ENABLED

    def get_status_map(self) -> dict[str, ProviderStatus]:
        status_map: dict[str, ProviderStatus] = {}
        for provider_name in self.list_providers(include_disabled=True):
            status_map[provider_name] = self.get_status(provider_name)
        return status_map

    def map_status(self, raw_status: Any) -> ProviderStatus:
        """
        Convert a domain-specific status value to canonical `ProviderStatus`.

        Wrappers can provide `status_mapper` to convert custom status objects
        before canonical normalization.
        """
        candidate = raw_status
        if self._status_mapper is not None:
            try:
                mapped = self._status_mapper(raw_status)
            except Exception:
                return ProviderStatus.UNKNOWN
            if mapped is not None:
                candidate = mapped
        return ProviderStatus.from_value(candidate)

    @staticmethod
    def _default_capability_getter(adapter: Any) -> Any:
        maybe = getattr(adapter, "capabilities", None)
        if callable(maybe):
            return maybe()
        return maybe

    def list_capabilities(
        self,
        *,
        capability_getter: Callable[[AdapterT], Any] | None = None,
        include_disabled: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Return provider capabilities with availability metadata.

        Envelope shape:
            {"provider": str, "availability": str, "capabilities": Any}
        """
        getter = capability_getter or self._default_capability_getter
        output: list[dict[str, Any]] = []
        provider_names = self.list_providers(include_disabled=include_disabled)

        for provider_name in provider_names:
            status = self.get_status(provider_name)
            capabilities: Any = None

            if status == ProviderStatus.ENABLED:
                adapter = self.get_adapter(provider_name)
                status = self.get_status(provider_name)
                if adapter is not None and status == ProviderStatus.ENABLED:
                    try:
                        capabilities = getter(adapter)
                    except Exception as exc:
                        logger.warning(
                            "Capability discovery failed for provider '{}': {}",
                            provider_name,
                            exc,
                        )

            output.append(
                {
                    "provider": provider_name,
                    "availability": status.value,
                    "capabilities": capabilities,
                }
            )
        return output

    def clear_cache(self) -> None:
        with self._lock:
            self._adapter_cache.clear()

    def reset_failures(self) -> None:
        with self._lock:
            self._failed_providers.clear()

    def mark_failure(self, name: str | None) -> None:
        provider_key = self.resolve_provider_name(name)
        if not provider_key:
            return
        with self._lock:
            if provider_key not in self._providers:
                return
            self._mark_failure_locked(provider_key)

    def get_cached_adapters(self) -> dict[str, AdapterT]:
        with self._lock:
            return dict(self._adapter_cache)


__all__ = [
    "ProviderRegistryBase",
    "ProviderRegistryConfig",
    "ProviderStatus",
]
