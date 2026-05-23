from __future__ import annotations

"""Sync v2 domain adapter contracts and registry."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from .models import (
    M1_SYNC_DOMAINS,
    SYNC_V2_SUPPORTED_DOMAINS,
    SyncDataset,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
)

ATTACHMENT_REF_REQUIRED_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "attachment_id",
        "parent_domain",
        "parent_object_id",
        "content_type",
        "size_bytes",
        "payload_hash",
        "availability",
    }
)
ATTACHMENT_REF_PARENT_DOMAINS: frozenset[str] = frozenset(
    domain for domain in M1_SYNC_DOMAINS if domain != "attachment.ref"
)
ATTACHMENT_REF_SERVER_AVAILABILITY: frozenset[str] = frozenset({"server", "server_available", "available"})

KNOWN_SYNC_DOMAINS: frozenset[str] = frozenset(
    {
        *SYNC_V2_SUPPORTED_DOMAINS,
        "notes",
        "chat",
        "workspaces",
        "source_cache",
        "media",
    }
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
class AttachmentRefMetadata:
    """Validated metadata carried by an `attachment.ref` envelope."""

    attachment_id: str
    parent_domain: SyncDomain
    parent_object_id: str
    content_type: str
    size_bytes: int
    payload_hash: str
    availability: str


class AttachmentRefValidationError(ValueError):
    """Validation failure with a stable sync error code."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


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


@dataclass(slots=True)
class AttachmentRefAdapter:
    """Validate metadata-only attachment refs and conflict divergent stable IDs."""

    domain: SyncDomain = "attachment.ref"
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1})

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Accept same-payload duplicates and conflict divergent payload hashes."""

        del dataset
        try:
            metadata = extract_attachment_ref_metadata(envelope)
        except AttachmentRefValidationError as exc:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code=exc.error_code,
                message=str(exc),
            )

        prior = context.prior_envelopes if context is not None else ()
        conflicting = next(
            (
                item
                for item in prior
                if item.operation != "tombstone"
                and _same_attachment_ref_identity(item, envelope, metadata)
                and _attachment_ref_hash(item) != metadata.payload_hash
            ),
            None,
        )
        if conflicting is not None:
            return AdapterConflict(
                client_envelope_id=envelope.client_envelope_id,
                domain=self.domain,
                entity_id=envelope.entity_id,
                conflict_type="attachment_ref_hash_mismatch",
                message=("attachment.ref stable attachment ID was reused with a " "different payload hash"),
                metadata={
                    "attachment_id": metadata.attachment_id,
                    "incoming_payload_hash": metadata.payload_hash,
                    "conflicting_payload_hash": _attachment_ref_hash(conflicting),
                    "conflicting_envelope_id": conflicting.client_envelope_id,
                },
            )

        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


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


def extract_attachment_ref_metadata(
    envelope: SyncEnvelope | SyncEnvelopeCreate,
) -> AttachmentRefMetadata:
    """Return validated attachment-ref metadata from a Sync envelope."""

    payload = envelope.payload or envelope.payload_clear
    missing = ATTACHMENT_REF_REQUIRED_PAYLOAD_KEYS.difference(payload)
    if missing:
        raise AttachmentRefValidationError(
            "attachment_ref_metadata_missing",
            "attachment.ref envelopes require payload metadata fields: " + ", ".join(sorted(missing)),
        )

    attachment_id = _required_string(payload, "attachment_id")
    parent_domain = _required_string(payload, "parent_domain")
    parent_object_id = _required_string(payload, "parent_object_id")
    content_type = _required_string(payload, "content_type")
    payload_hash = _required_string(payload, "payload_hash")
    availability = _required_string(payload, "availability").strip().lower()
    size_bytes = _required_non_negative_int(payload, "size_bytes")

    if parent_domain not in ATTACHMENT_REF_PARENT_DOMAINS:
        raise AttachmentRefValidationError(
            "attachment_ref_parent_domain_invalid",
            "attachment.ref parent_domain must reference an M1 object domain",
        )
    if envelope.object_id != attachment_id:
        raise AttachmentRefValidationError(
            "attachment_ref_object_id_mismatch",
            "attachment.ref object_id must match payload attachment_id",
        )
    if envelope.payload_hash and envelope.payload_hash != payload_hash:
        raise AttachmentRefValidationError(
            "attachment_ref_payload_hash_mismatch",
            "attachment.ref payload_hash must match the envelope payload_hash",
        )

    return AttachmentRefMetadata(
        attachment_id=attachment_id,
        parent_domain=cast(SyncDomain, parent_domain),
        parent_object_id=parent_object_id,
        content_type=content_type,
        size_bytes=size_bytes,
        payload_hash=payload_hash,
        availability=availability,
    )


def _same_attachment_ref_identity(
    prior: SyncEnvelope,
    incoming: SyncEnvelopeCreate,
    incoming_metadata: AttachmentRefMetadata,
) -> bool:
    prior_payload = prior.payload or prior.payload_clear
    prior_attachment_id = prior_payload.get("attachment_id")
    if isinstance(prior_attachment_id, str) and prior_attachment_id.strip():
        return prior_attachment_id.strip() == incoming_metadata.attachment_id
    if prior.stable_key and incoming.stable_key and prior.stable_key == incoming.stable_key:
        return True
    return prior.entity_id == incoming.entity_id


def _attachment_ref_hash(envelope: SyncEnvelope | SyncEnvelopeCreate) -> str:
    payload = envelope.payload or envelope.payload_clear
    payload_hash = payload.get("payload_hash")
    if isinstance(payload_hash, str) and payload_hash.strip():
        return payload_hash.strip()
    return envelope.payload_hash or ""


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AttachmentRefValidationError(
            "attachment_ref_metadata_invalid",
            f"attachment.ref metadata field {key} must be a non-empty string",
        )
    return value.strip()


def _required_non_negative_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AttachmentRefValidationError(
            "attachment_ref_metadata_invalid",
            f"attachment.ref metadata field {key} must be a non-negative integer",
        )
    return value


__all__ = [
    "AdapterAccepted",
    "AdapterConflict",
    "AdapterDeferred",
    "AdapterRejected",
    "AttachmentRefAdapter",
    "AttachmentRefMetadata",
    "AttachmentRefValidationError",
    "ATTACHMENT_REF_PARENT_DOMAINS",
    "ATTACHMENT_REF_REQUIRED_PAYLOAD_KEYS",
    "ATTACHMENT_REF_SERVER_AVAILABILITY",
    "KNOWN_SYNC_DOMAINS",
    "StaticSyncAdapter",
    "SyncAdapterContext",
    "SyncAdapterOutcome",
    "SyncAdapterRegistry",
    "SyncDomainAdapter",
    "extract_attachment_ref_metadata",
]
