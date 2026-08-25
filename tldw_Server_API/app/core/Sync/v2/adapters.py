from __future__ import annotations

"""Sync v2 domain adapter contracts and registry."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from .attachment_refs_v2 import (
    AttachmentRefV2ValidationError,
    attachment_ref_v2_object_hash,
    parse_attachment_ref_v2_payload,
    validate_attachment_ref_v2,
    validate_attachment_ref_v2_object_id,
    validate_attachment_ref_v2_routing_metadata,
)
from .models import (
    M1_SYNC_DOMAINS,
    SYNC_V2_SUPPORTED_DOMAINS,
    SyncDataset,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    sync_v2_attachment_ref_v2_is_writable,
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
        "notes.task",
        "notes.task_activity",
        "notes",
        "chat",
        "workspaces",
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
SyncHead = SyncEnvelope | SyncEnvelopeCreate
SyncHeadLookup = Callable[[SyncDomain, str], SyncHead | None]
AuthorizedNoteLookup = Callable[[str], SyncHead | None]
AuthorizedTaskLookup = Callable[[str], SyncHead | None]
SyncDomainHeadLoader = Callable[[SyncDomain], Sequence[SyncHead]]
BootstrapRelationshipVerifier = Callable[
    [SyncDomain, str, Mapping[str, object]], bool
]


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

    prior_envelopes: Sequence[SyncHead] = field(default_factory=tuple)
    get_head: SyncHeadLookup | None = None
    get_authorized_note: AuthorizedNoteLookup | None = None
    get_authorized_task: AuthorizedTaskLookup | None = None
    list_heads: SyncDomainHeadLoader | None = None
    trusted_server_origin: bool = False
    authenticated_actor_type: str | None = None
    authenticated_actor_id: object = None
    authenticated_device_id: str | None = None
    coordinator_derived_task_activity: bool = False
    coordinator_derived_task_projection: bool = False
    trusted_notes_task_prebootstrap_capture: bool = False
    organization_group_state: str | None = None
    organization_bootstrap_id: str | None = None
    notes_link_bootstrap_id: str | None = None
    attachment_ref_bootstrap_id: str | None = None
    notes_task_bootstrap_id: str | None = None
    notes_task_activity_bootstrap_id: str | None = None
    bootstrap_relationship_verifier: BootstrapRelationshipVerifier | None = None
    bootstrap_relationship_absence_verifier: BootstrapRelationshipVerifier | None = None
    supports_attachments: bool = False


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
    """Validate immutable v1 and strict whole-object v2 attachment refs."""

    domain: SyncDomain = "attachment.ref"
    supported_adapter_versions: set[int] = field(default_factory=lambda: {1, 2})
    v2_writes_enabled: bool = False

    def evaluate_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        dataset: SyncDataset,
        context: SyncAdapterContext | None = None,
    ) -> SyncAdapterOutcome:
        """Reject v1 writes and validate gated, ready adapter-v2 mutations."""

        if envelope.adapter_version == 1:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="attachment_ref_v1_immutable",
                message="attachment.ref adapter version 1 is immutable",
            )
        if envelope.adapter_version != 2 or envelope.schema_version != 2:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="unsupported_adapter_version",
                message="attachment.ref adapter version is not supported",
            )

        trusted_bootstrap = _trusted_attachment_ref_bootstrap(
            dataset,
            envelope,
            context,
        )
        has_bootstrap_routing = any(
            field_name in envelope.routing_metadata
            for field_name in ("bootstrap_capture", "bootstrap_id")
        )
        if has_bootstrap_routing and not trusted_bootstrap:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="attachment_ref_v2_payload_invalid",
                message="attachment.ref v2 payload validation failed",
            )
        if not trusted_bootstrap and not sync_v2_attachment_ref_v2_is_writable(
            dataset,
            notes_attachment_sync_enabled=self.v2_writes_enabled,
            supports_attachments=(
                context.supports_attachments if context is not None else False
            ),
        ):
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="attachment_ref_v2_not_writable",
                message="attachment.ref adapter version 2 is not writable for this dataset",
            )

        prior = context.prior_envelopes if context is not None else ()
        collision = next(
            (
                item
                for item in prior
                if item.object_id == envelope.object_id and item.adapter_version == 1
            ),
            None,
        )
        if collision is not None:
            return AdapterConflict(
                client_envelope_id=envelope.client_envelope_id,
                domain=self.domain,
                entity_id=envelope.entity_id,
                conflict_type="attachment_ref_immutable_version_collision",
                message="attachment.ref object identity already exists under adapter version 1",
            )

        head = next(
            (
                item
                for item in reversed(tuple(prior))
                if item.object_id == envelope.object_id and item.adapter_version == 2
            ),
            None,
        )
        try:
            payload = parse_attachment_ref_v2_payload(
                envelope.operation,
                envelope.payload or envelope.payload_clear
            )
            validate_attachment_ref_v2_object_id(envelope.object_id)
            validate_attachment_ref_v2_routing_metadata(
                envelope.operation,
                envelope.routing_metadata,
            )
            if str(payload.attachment_id) != envelope.object_id:
                raise AttachmentRefV2ValidationError(
                    "attachment.ref v2 object_id must match attachment_id"
                )
            canonical_hash = attachment_ref_v2_object_hash(
                envelope.operation,
                payload,
                object_revision=envelope.object_revision,
            )
            if envelope.payload_hash != canonical_hash:
                raise AttachmentRefV2ValidationError(
                    "attachment.ref v2 payload_hash must match the canonical object hash"
                )
            validate_attachment_ref_v2(
                envelope.operation,
                payload,
                envelope_created_at_client=envelope.created_at_client or "",
                authenticated_device_id=envelope.device_id or "",
                prior_payload=(
                    (head.payload or head.payload_clear) if head is not None else None
                ),
                prior_operation=head.operation if head is not None else None,
                trusted_server_origin=(
                    context.trusted_server_origin if context is not None else False
                ),
                verified_bootstrap=trusted_bootstrap,
            )
        except AttachmentRefV2ValidationError:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="attachment_ref_v2_payload_invalid",
                message="attachment.ref v2 payload validation failed",
            )

        restore_intent = envelope.routing_metadata.get("restore_intent") is True
        if restore_intent and (head is None or head.operation != "tombstone"):
            return AdapterConflict(
                client_envelope_id=envelope.client_envelope_id,
                domain=self.domain,
                entity_id=envelope.entity_id,
                conflict_type="attachment_ref_restore_base_conflict",
                message="attachment.ref restore requires the current tombstone head",
            )
        if head is not None and head.operation == "tombstone" and not restore_intent:
            return AdapterConflict(
                client_envelope_id=envelope.client_envelope_id,
                domain=self.domain,
                entity_id=envelope.entity_id,
                conflict_type="attachment_ref_tombstoned",
                message="attachment.ref restore requires explicit restore intent",
            )

        return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)


def _trusted_attachment_ref_bootstrap(
    dataset: SyncDataset,
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext | None,
) -> bool:
    state = dataset.metadata.get("notes_attachment_v2")
    bootstrap_id = state.get("bootstrap_id") if isinstance(state, Mapping) else None
    return bool(
        context is not None
        and context.trusted_server_origin
        and isinstance(bootstrap_id, str)
        and context.attachment_ref_bootstrap_id == bootstrap_id
        and envelope.routing_metadata.get("bootstrap_capture") is True
        and envelope.routing_metadata.get("bootstrap_id") == bootstrap_id
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


def extract_attachment_ref_metadata(
    envelope: SyncEnvelope | SyncEnvelopeCreate,
) -> AttachmentRefMetadata:
    """Return validated attachment-ref metadata from a Sync envelope."""

    payload = envelope.payload or envelope.payload_clear
    if envelope.adapter_version == 2:
        try:
            parsed = parse_attachment_ref_v2_payload(envelope.operation, payload)
        except AttachmentRefV2ValidationError as exc:
            raise AttachmentRefValidationError(
                "attachment_ref_v2_payload_invalid",
                "attachment.ref v2 payload validation failed",
            ) from exc
        return AttachmentRefMetadata(
            attachment_id=str(parsed.attachment_id),
            parent_domain="notes.note",
            parent_object_id=str(parsed.parent_object_id),
            content_type=parsed.content_type,
            size_bytes=parsed.size_bytes,
            payload_hash=parsed.blob_hash,
            availability="metadata_only",
        )

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
    "SyncDomainHeadLoader",
    "AuthorizedTaskLookup",
    "SyncHead",
    "SyncHeadLookup",
    "SyncAdapterOutcome",
    "SyncAdapterRegistry",
    "SyncDomainAdapter",
    "extract_attachment_ref_metadata",
]
