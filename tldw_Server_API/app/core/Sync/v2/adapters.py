from __future__ import annotations

"""Sync v2 domain adapter contracts and registry."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .models import SyncDataset, SyncDomain, SyncEnvelope, SyncEnvelopeCreate

KNOWN_SYNC_DOMAINS: frozenset[str] = frozenset(
    {"notes", "chat", "workspaces", "source_cache", "media"}
)


@dataclass(frozen=True, slots=True)
class AdapterAccepted:
    """Adapter outcome indicating the envelope can be persisted."""

    client_envelope_id: str


@dataclass(frozen=True, slots=True)
class AdapterRejected:
    """Adapter outcome indicating the envelope failed domain validation."""

    client_envelope_id: str
    error_code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class AdapterConflict:
    """Adapter outcome indicating the envelope should become a conflict."""

    client_envelope_id: str
    domain: SyncDomain
    entity_id: str
    conflict_type: str
    message: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdapterDeferred:
    """Adapter outcome indicating the envelope should be retried later."""

    client_envelope_id: str
    message: str
    retry_after_seconds: int | None = None


SyncAdapterOutcome = AdapterAccepted | AdapterRejected | AdapterConflict | AdapterDeferred


@dataclass(frozen=True, slots=True)
class SyncAdapterContext:
    """Read-only Sync state provided to adapters during envelope evaluation."""

    prior_envelopes: Sequence[SyncEnvelope] = field(default_factory=tuple)


class SyncDomainAdapter(Protocol):
    """Protocol implemented by concrete Sync v2 domain adapters."""

    domain: SyncDomain
    supported_adapter_versions: set[int]

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
    ) -> SyncAdapterOutcome:
        """Validate an envelope before the service persists or conflicts it.

        Implementations may also accept an optional ``context`` keyword. The
        service detects support before passing it so old adapters remain valid.
        """


@dataclass(slots=True)
class StaticSyncAdapter:
    """Small adapter implementation for tests and protocol defaults."""

    domain: SyncDomain
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})
    outcomes: Mapping[str, SyncAdapterOutcome] = field(default_factory=dict)

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Return a configured test outcome or accept the envelope."""

        del context
        return self.outcomes.get(
            envelope.client_envelope_id,
            AdapterAccepted(client_envelope_id=envelope.client_envelope_id),
        )


class SyncAdapterRegistry:
    """Minimal registry for Sync v2 domain adapters."""

    def __init__(self, adapters: list[SyncDomainAdapter] | None = None) -> None:
        self._adapters: dict[SyncDomain, SyncDomainAdapter] = {}
        for adapter in adapters or []:
            self.register(adapter)

    def register(self, adapter: SyncDomainAdapter) -> None:
        if adapter.domain not in KNOWN_SYNC_DOMAINS:
            raise ValueError(f"Unknown Sync adapter domain: {adapter.domain}")
        if not adapter.supported_adapter_versions:
            raise ValueError(f"Sync adapter for {adapter.domain} has no supported versions")
        if any(version < 1 for version in adapter.supported_adapter_versions):
            raise ValueError(f"Sync adapter for {adapter.domain} has invalid versions")
        self._adapters[adapter.domain] = adapter

    def get(self, domain: SyncDomain) -> SyncDomainAdapter:
        adapter = self._adapters.get(domain)
        if adapter is None:
            raise KeyError(domain)
        return adapter

    def has_domain(self, domain: SyncDomain) -> bool:
        return domain in self._adapters

    def supports_version(self, domain: SyncDomain, adapter_version: int) -> bool:
        return adapter_version in self.get(domain).supported_adapter_versions

    @property
    def supported_domains(self) -> list[SyncDomain]:
        return sorted(self._adapters)


__all__ = [
    "AdapterAccepted",
    "AdapterConflict",
    "AdapterDeferred",
    "AdapterRejected",
    "KNOWN_SYNC_DOMAINS",
    "StaticSyncAdapter",
    "SyncAdapterContext",
    "SyncAdapterOutcome",
    "SyncAdapterRegistry",
    "SyncDomainAdapter",
]
