from __future__ import annotations

"""Business service for Sync v2 protocol operations."""

import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from uuid import uuid4

from .adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterDeferred,
    AdapterRejected,
    SyncAdapterContext,
    SyncDomainAdapter,
    SyncAdapterRegistry,
)
from .errors import (
    SyncIdempotencyConflictError,
    SyncInvalidDomainError,
    SyncStoreError,
)
from .models import (
    ConflictStatus,
    EncryptionPolicy,
    SyncAttachment,
    SyncAttachmentCreate,
    SyncConflict,
    SyncConflictCreate,
    SyncDataset,
    SyncDatasetCreate,
    SyncDevice,
    SyncDeviceCursor,
    SyncDeviceUpsert,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncKeyRecord,
    SyncKeyRecordCreate,
)
from .security import PrivatePayloadValidationError, validate_private_payload
from .store import SyncV2Store


@dataclass(frozen=True, slots=True)
class SyncV2Settings:
    """Server settings surfaced through Sync v2 capabilities."""

    protocol_version: int = 2
    min_supported_protocol_version: int = 2
    max_batch_size: int = 100
    max_pull_page_size: int = 100
    max_envelope_payload_bytes: int = 262_144
    max_attachment_bytes: int = 1_048_576
    supports_attachments: bool = True
    encryption_policies: list[EncryptionPolicy] = field(
        default_factory=lambda: [
            "client_private_v1",
            "server_trusted",
            "shared_workspace_v1",
        ]
    )
    restore_manifest_scan_limit: int = 10_000


@dataclass(frozen=True, slots=True)
class SyncV2Capabilities:
    protocol_version: int
    min_supported_protocol_version: int
    supported_domains: list[SyncDomain]
    encryption_policies: list[EncryptionPolicy]
    max_batch_size: int
    max_envelope_payload_bytes: int
    max_attachment_bytes: int
    supports_restore_manifest: bool = True
    supports_conflicts: bool = True
    supports_attachments: bool = True
    server_time: str | None = None


@dataclass(frozen=True, slots=True)
class SyncDeviceRegistration:
    device: SyncDevice
    server_capabilities: SyncV2Capabilities
    required_actions: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SyncDatasetEnrollment:
    dataset: SyncDataset
    cursors: dict[str, str] = field(default_factory=dict)
    key_setup_required: bool = False


@dataclass(frozen=True, slots=True)
class SyncPushAccepted:
    client_envelope_id: str
    server_sequence: int
    domain: SyncDomain
    entity_id: str


@dataclass(frozen=True, slots=True)
class SyncPushRejected:
    client_envelope_id: str
    error_code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True, slots=True)
class SyncPushConflict:
    conflict_id: str
    client_envelope_id: str
    domain: SyncDomain
    entity_id: str
    server_sequence: int | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class SyncPushResult:
    dataset_id: str
    accepted: list[SyncPushAccepted] = field(default_factory=list)
    rejected: list[SyncPushRejected] = field(default_factory=list)
    conflicts: list[SyncPushConflict] = field(default_factory=list)
    next_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class SyncPullResult:
    dataset_id: str
    encryption_policy: EncryptionPolicy = "client_private_v1"
    envelopes: list[SyncEnvelope] = field(default_factory=list)
    next_cursor: str | None = None
    has_more: bool = False


@dataclass(frozen=True, slots=True)
class SyncRestoreManifestDevice:
    device_id: str
    display_name: str | None
    client_type: str | None
    client_version: str | None
    last_seen_at: str | None
    revoked_at: str | None


@dataclass(frozen=True, slots=True)
class SyncRestoreManifestDataset:
    dataset_id: str
    scope_type: str
    encryption_policy: EncryptionPolicy
    domains: list[SyncDomain]
    workspace_id: str | None
    approximate_counts: dict[str, int] = field(default_factory=dict)
    byte_estimates: dict[str, int] = field(default_factory=dict)
    last_updated_at: str | None = None
    unresolved_conflicts: int = 0
    attachment_availability: dict[str, int] = field(default_factory=dict)
    attachment_size_classes: dict[str, int] = field(default_factory=dict)
    key_recovery_available: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SyncRestoreManifest:
    datasets: list[SyncRestoreManifestDataset] = field(default_factory=list)
    devices: list[SyncRestoreManifestDevice] = field(default_factory=list)
    generated_at: str | None = None
    filters_applied: dict[str, object] = field(default_factory=dict)


class SyncV2Service:
    """Core Sync v2 service with injected persistence and adapter dependencies."""

    def __init__(
        self,
        *,
        store: SyncV2Store,
        adapters: SyncAdapterRegistry,
        clock: Callable[[], str] | None = None,
        id_factory: Callable[[str], str] | None = None,
        settings: SyncV2Settings | None = None,
    ) -> None:
        self.store = store
        self.adapters = adapters
        self.clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self.id_factory = id_factory or (lambda prefix: f"{prefix}-{uuid4().hex}")
        self.settings = settings or SyncV2Settings()

    def capabilities(self) -> SyncV2Capabilities:
        return SyncV2Capabilities(
            protocol_version=self.settings.protocol_version,
            min_supported_protocol_version=self.settings.min_supported_protocol_version,
            supported_domains=self.adapters.supported_domains,
            encryption_policies=list(self.settings.encryption_policies),
            max_batch_size=self.settings.max_batch_size,
            max_envelope_payload_bytes=self.settings.max_envelope_payload_bytes,
            max_attachment_bytes=self.settings.max_attachment_bytes,
            supports_attachments=self.settings.supports_attachments,
            server_time=self.clock() or None,
        )

    def register_device(
        self,
        *,
        user_id: str,
        display_name: str,
        client_type: str,
        device_id: str | None = None,
        client_version: str | None = None,
        capabilities: dict[str, object] | None = None,
    ) -> SyncDeviceRegistration:
        device = self.store.upsert_device(
            SyncDeviceUpsert(
                device_id=device_id or self.id_factory("device"),
                user_id=user_id,
                display_name=display_name,
                client_type=client_type,
                client_version=client_version,
                capabilities=dict(capabilities or {}),
            )
        )
        return SyncDeviceRegistration(device=device, server_capabilities=self.capabilities())

    def enroll_dataset(
        self,
        *,
        user_id: str,
        dataset_id: str | None = None,
        scope_type: str = "personal",
        domains: Sequence[SyncDomain] | None = None,
        encryption_policy: EncryptionPolicy = "client_private_v1",
        workspace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SyncDatasetEnrollment:
        enrolled_domains = list(domains or self.adapters.supported_domains)
        dataset = self.store.enroll_dataset(
            SyncDatasetCreate(
                dataset_id=dataset_id or self.id_factory("dataset"),
                owner_user_id=user_id,
                scope_type=scope_type,
                encryption_policy=encryption_policy,
                domains=enrolled_domains,
                workspace_id=workspace_id,
                metadata=dict(metadata or {}),
            )
        )
        return SyncDatasetEnrollment(
            dataset=dataset,
            cursors={domain: "0" for domain in dataset.domains},
            key_setup_required=dataset.encryption_policy == "client_private_v1",
        )

    def push(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        envelopes: Sequence[SyncEnvelopeCreate],
    ) -> SyncPushResult:
        self._require_registered_device(user_id, device_id)
        dataset = self.store.get_dataset(dataset_id, owner_user_id=user_id)
        if dataset is None:
            return SyncPushResult(
                dataset_id=dataset_id,
                rejected=[
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="dataset_not_found_or_forbidden",
                        message="Sync dataset was not found or is not accessible",
                    )
                    for envelope in envelopes
                ],
            )

        accepted: list[SyncPushAccepted] = []
        rejected: list[SyncPushRejected] = []
        conflicts: list[SyncPushConflict] = []

        for index, envelope in enumerate(envelopes):
            if index >= self.settings.max_batch_size:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="batch_limit_exceeded",
                        message="Sync push batch exceeded the server envelope limit",
                    )
                )
                continue
            if envelope.dataset_id != dataset_id:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="dataset_mismatch",
                        message="Envelope dataset_id must match the push dataset_id",
                    )
                )
                continue
            if envelope.device_id is not None and envelope.device_id != device_id:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="device_mismatch",
                        message="Envelope device_id must match the authenticated push device_id",
                    )
                )
                continue
            if envelope.domain not in dataset.domains:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="domain_not_enrolled",
                        message=f"Sync domain is not enrolled for this dataset: {envelope.domain}",
                    )
                )
                continue
            if self._payload_exceeds_size_limit(envelope):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="payload_too_large",
                        message="Sync envelope payload exceeds the server size limit",
                    )
                )
                continue
            envelope = replace(envelope, device_id=envelope.device_id or device_id)
            try:
                outcome = self._evaluate_envelope(dataset, envelope)
            except KeyError:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="unknown_domain",
                        message=f"Sync domain is not registered: {envelope.domain}",
                    )
                )
                continue
            except PrivatePayloadValidationError as exc:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="private_payload_validation_failed",
                        message=str(exc),
                    )
                )
                continue

            if isinstance(outcome, AdapterRejected):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=outcome.client_envelope_id,
                        error_code=outcome.error_code,
                        message=outcome.message,
                        retryable=outcome.retryable,
                    )
                )
                continue
            if isinstance(outcome, AdapterDeferred):
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=outcome.client_envelope_id,
                        error_code="adapter_deferred",
                        message=outcome.message,
                        retryable=True,
                    )
                )
                continue
            if isinstance(outcome, AdapterConflict):
                try:
                    conflicts.append(self._store_conflict(dataset, envelope, outcome))
                except SyncIdempotencyConflictError:
                    rejected.append(
                        SyncPushRejected(
                            client_envelope_id=envelope.client_envelope_id,
                            error_code="idempotency_conflict",
                            message="Sync envelope ID was reused with different content",
                        )
                    )
                continue

            try:
                inserted = self.store.insert_envelope(replace(envelope, status="accepted"))
            except SyncInvalidDomainError:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="domain_not_enrolled",
                        message=f"Sync domain is not enrolled for this dataset: {envelope.domain}",
                    )
                )
                continue
            except SyncIdempotencyConflictError:
                rejected.append(
                    SyncPushRejected(
                        client_envelope_id=envelope.client_envelope_id,
                        error_code="idempotency_conflict",
                        message="Sync envelope ID was reused with different content",
                    )
                )
                continue
            accepted.append(
                SyncPushAccepted(
                    client_envelope_id=inserted.client_envelope_id,
                    server_sequence=inserted.server_sequence,
                    domain=inserted.domain,
                    entity_id=inserted.entity_id,
                )
            )

        sequences = [item.server_sequence for item in accepted]
        sequences.extend(
            item.server_sequence
            for item in conflicts
            if item.server_sequence is not None
        )
        next_sequence = max(sequences, default=None)
        return SyncPushResult(
            dataset_id=dataset_id,
            accepted=accepted,
            rejected=rejected,
            conflicts=conflicts,
            next_cursor=str(next_sequence) if next_sequence is not None else None,
        )

    def pull(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        cursor: str | int | None = None,
        domains: Sequence[SyncDomain] | None = None,
        page_size: int | None = None,
        include_own_changes: bool = False,
    ) -> SyncPullResult:
        self._require_registered_device(user_id, device_id)
        if page_size is not None and page_size < 1:
            raise SyncStoreError("Sync pull page_size must be greater than zero")
        dataset = self.store.get_dataset(dataset_id, owner_user_id=user_id)
        if dataset is None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")

        selected_domains = self._selected_domains(dataset, domains)
        since_sequence = self._resolve_cursor(dataset_id, device_id, cursor, selected_domains)
        page_limit = min(page_size or self.settings.max_pull_page_size, self.settings.max_pull_page_size)
        raw_envelopes, visible = self._scan_pull_page(
            dataset_id=dataset_id,
            device_id=device_id,
            since_sequence=since_sequence,
            domains=selected_domains,
            page_limit=page_limit,
            include_own_changes=include_own_changes,
        )

        page = visible[:page_limit]
        has_more = len(visible) > page_limit
        next_sequence = (
            page[-1].server_sequence
            if page
            else max((envelope.server_sequence for envelope in raw_envelopes), default=since_sequence)
        )
        if cursor is None and raw_envelopes:
            self._update_cursors(dataset_id, device_id, selected_domains, next_sequence)
        return SyncPullResult(
            dataset_id=dataset_id,
            encryption_policy=dataset.encryption_policy,
            envelopes=page,
            next_cursor=str(next_sequence),
            has_more=has_more,
        )

    def restore_manifest(
        self,
        *,
        user_id: str,
        dataset_ids: Sequence[str] | None = None,
        domains: Sequence[SyncDomain] | None = None,
    ) -> SyncRestoreManifest:
        allowed_dataset_ids = set(dataset_ids or [])
        selected_domains = set(domains or [])
        datasets = [
            dataset
            for dataset in self.store.list_datasets_for_user(user_id)
            if not allowed_dataset_ids or dataset.dataset_id in allowed_dataset_ids
        ]
        devices = [
            SyncRestoreManifestDevice(
                device_id=device.device_id,
                display_name=device.display_name,
                client_type=device.client_type,
                client_version=device.client_version,
                last_seen_at=device.last_seen_at,
                revoked_at=device.revoked_at,
            )
            for device in self.store.list_devices_for_user(user_id)
        ]

        manifest_datasets = [
            self._manifest_dataset(dataset, user_id=user_id, domains=selected_domains)
            for dataset in datasets
        ]
        return SyncRestoreManifest(
            datasets=manifest_datasets,
            devices=devices,
            generated_at=self.clock() or None,
            filters_applied={
                "dataset_ids": list(dataset_ids or []),
                "domains": list(domains or []),
            },
        )

    def store_attachment(
        self,
        *,
        user_id: str,
        dataset_id: str,
        domain: SyncDomain,
        entity_id: str,
        attachment_id: str,
        content_type: str,
        size_bytes: int,
        payload_ciphertext: str,
        payload_hash: str,
        encryption_policy: EncryptionPolicy = "client_private_v1",
        metadata: dict[str, object] | None = None,
    ) -> SyncAttachment:
        """Persist a small encrypted attachment payload for later restore."""

        if not self.settings.supports_attachments:
            raise SyncStoreError("Sync v2 attachment persistence is not enabled")
        if encryption_policy != "client_private_v1":
            raise SyncStoreError(
                "Sync attachment persistence requires client_private_v1 encryption"
            )
        if (
            size_bytes > self.settings.max_attachment_bytes
            or _ciphertext_exceeds_attachment_limit(
                payload_ciphertext,
                self.settings.max_attachment_bytes,
            )
        ):
            raise SyncStoreError("Sync attachment payload exceeds the server size limit")
        if not payload_ciphertext:
            raise SyncStoreError("Sync attachment payload_ciphertext is required")
        dataset = self.store.get_dataset(dataset_id, owner_user_id=user_id)
        if dataset is None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        if domain not in dataset.domains:
            raise SyncInvalidDomainError(
                f"Sync domain is not enrolled for this dataset: {domain}"
            )
        return self.store.store_attachment(
            SyncAttachmentCreate(
                attachment_id=attachment_id,
                dataset_id=dataset_id,
                domain=domain,
                entity_id=entity_id,
                content_type=content_type,
                size_bytes=size_bytes,
                payload_ciphertext=payload_ciphertext,
                payload_hash=payload_hash,
                encryption_policy=encryption_policy,
                metadata=dict(metadata or {}),
            )
        )

    def list_conflicts(
        self,
        *,
        user_id: str,
        dataset_id: str,
        status: ConflictStatus | None = None,
    ) -> list[SyncConflict]:
        dataset = self.store.get_dataset(dataset_id, owner_user_id=user_id)
        if dataset is None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        return self.store.list_conflicts(dataset_id, status=status)

    def resolve_conflict(
        self,
        *,
        user_id: str,
        conflict_id: str,
        action: str,
        resolution_envelope: SyncEnvelopeCreate | None = None,
        resolved_by_envelope_id: str | None = None,
        resolved_by_device_id: str | None = None,
        notes: str | None = None,
    ) -> SyncConflict:
        conflict = self.store.get_conflict(conflict_id)
        if conflict is None:
            raise SyncStoreError("Sync conflict was not found or is not accessible")
        dataset = self.store.get_dataset(conflict.dataset_id, owner_user_id=user_id)
        if dataset is None:
            raise SyncStoreError("Sync conflict was not found or is not accessible")
        if resolved_by_device_id is not None:
            self._require_registered_device(user_id, resolved_by_device_id)
        if resolution_envelope is not None:
            resolution_device_id = resolved_by_device_id or resolution_envelope.device_id
            self._require_registered_device(user_id, resolution_device_id or "")
            if resolution_envelope.dataset_id != dataset.dataset_id:
                raise SyncStoreError(
                    "Sync resolution envelope dataset_id must match the conflict dataset"
                )
            if (
                resolution_envelope.domain != conflict.domain
                or resolution_envelope.entity_id != conflict.entity_id
            ):
                raise SyncStoreError(
                    "Sync resolution envelope must target the conflict domain and entity"
                )
            if (
                resolved_by_device_id is not None
                and resolution_envelope.device_id is not None
                and resolution_envelope.device_id != resolved_by_device_id
            ):
                raise SyncStoreError(
                    "Sync resolution envelope device_id must match resolved_by_device_id"
                )
            if self._payload_exceeds_size_limit(resolution_envelope):
                raise SyncStoreError(
                    "Sync resolution envelope payload exceeds the server size limit"
                )
            try:
                outcome = self._evaluate_envelope(dataset, resolution_envelope)
            except PrivatePayloadValidationError as exc:
                raise SyncStoreError(
                    "Sync resolution envelope private payload validation failed"
                ) from exc
            if not isinstance(outcome, AdapterAccepted):
                raise SyncStoreError("Sync resolution envelope was not accepted")
            inserted = self.store.insert_envelope(
                replace(
                    resolution_envelope,
                    device_id=resolution_device_id,
                    status="accepted",
                )
            )
            resolved_by_envelope_id = inserted.client_envelope_id
            resolved_by_device_id = resolution_device_id
        return self.store.resolve_conflict(
            conflict_id,
            status="dismissed" if action == "dismiss" else "resolved",
            resolved_by_envelope_id=resolved_by_envelope_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=action,
            resolution_notes=notes,
        )

    def store_key_recovery_bundle(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None,
        key_purpose: str,
        wrapped_key_blob: str,
        kdf_metadata: dict[str, object] | None = None,
        recovery_hint: str | None = None,
        rotation_of_key_record_id: str | None = None,
    ) -> SyncKeyRecord:
        dataset = self.store.get_dataset(dataset_id, owner_user_id=user_id)
        if dataset is None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        if device_id is not None:
            self._require_registered_device(user_id, device_id)
        return self.store.store_key_record(
            SyncKeyRecordCreate(
                key_record_id=self.id_factory("key"),
                dataset_id=dataset.dataset_id,
                user_id=user_id,
                device_id=device_id,
                key_purpose=key_purpose,
                wrapped_key_blob=wrapped_key_blob,
                kdf_metadata=dict(kdf_metadata or {}),
                recovery_hint=recovery_hint,
                rotation_of_key_record_id=rotation_of_key_record_id,
            )
        )

    def list_key_recovery_bundles(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str | None = None,
        key_purpose: str | None = "dataset_recovery",
    ) -> list[SyncKeyRecord]:
        dataset = self.store.get_dataset(dataset_id, owner_user_id=user_id)
        if dataset is None:
            raise SyncStoreError("Sync dataset was not found or is not accessible")
        return [
            record
            for record in self.store.list_key_records(
                dataset.dataset_id,
                user_id=user_id,
                device_id=device_id,
                key_purpose=key_purpose,
            )
            if record.revoked_at is None
        ]

    def _evaluate_envelope(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelopeCreate,
    ) -> AdapterAccepted | AdapterRejected | AdapterConflict | AdapterDeferred:
        if envelope.adapter_version not in self.adapters.get(envelope.domain).supported_adapter_versions:
            return AdapterRejected(
                client_envelope_id=envelope.client_envelope_id,
                error_code="unsupported_adapter_version",
                message=f"Adapter version is not supported for {envelope.domain}",
            )
        if dataset.encryption_policy == "client_private_v1":
            validate_private_payload(
                payload_ciphertext=envelope.payload_ciphertext,
                payload_clear=envelope.payload_clear,
            )
        context = SyncAdapterContext(
            prior_envelopes=self.store.list_envelopes_for_entity(
                dataset.dataset_id,
                envelope.domain,
                entity_id=envelope.entity_id,
                stable_key=envelope.stable_key,
                limit=100,
            )
        )
        adapter = self.adapters.get(envelope.domain)
        return _call_adapter_evaluate(adapter, envelope, dataset=dataset, context=context)

    def _require_registered_device(self, user_id: str, device_id: str) -> SyncDevice:
        if not device_id:
            raise SyncStoreError("Sync device was not found or is not accessible")
        for device in self.store.list_devices_for_user(user_id):
            if device.device_id == device_id and device.revoked_at is None:
                return device
        raise SyncStoreError("Sync device was not found or is not accessible")

    def _payload_exceeds_size_limit(self, envelope: SyncEnvelopeCreate) -> bool:
        max_bytes = self.settings.max_envelope_payload_bytes
        if envelope.payload_size_bytes is not None and envelope.payload_size_bytes > max_bytes:
            return True
        actual_size = 0
        if envelope.payload_ciphertext is not None:
            actual_size += len(envelope.payload_ciphertext.encode("utf-8"))
        actual_size += _compact_json_size(envelope.payload_clear)
        actual_size += _compact_json_size(envelope.routing_metadata)
        actual_size += _compact_json_size(envelope.dependencies)
        return actual_size > max_bytes

    def _store_conflict(
        self,
        dataset: SyncDataset,
        envelope: SyncEnvelopeCreate,
        outcome: AdapterConflict,
    ) -> SyncPushConflict:
        inserted = self.store.insert_envelope(replace(envelope, status="conflict"))
        existing = self.store.get_unresolved_conflict_for_envelope(
            dataset.dataset_id,
            local_envelope_id=envelope.client_envelope_id,
            server_sequence=inserted.server_sequence,
        )
        if existing is not None:
            return SyncPushConflict(
                conflict_id=existing.conflict_id,
                client_envelope_id=outcome.client_envelope_id,
                domain=existing.domain,
                entity_id=existing.entity_id,
                server_sequence=existing.server_sequence,
                message=outcome.message,
            )
        conflict = self.store.insert_conflict(
            SyncConflictCreate(
                conflict_id=self.id_factory("conflict"),
                dataset_id=dataset.dataset_id,
                domain=outcome.domain,
                entity_id=outcome.entity_id,
                conflict_type=outcome.conflict_type,
                local_envelope_id=envelope.client_envelope_id,
                server_sequence=inserted.server_sequence,
                metadata=dict(outcome.metadata),
            )
        )
        return SyncPushConflict(
            conflict_id=conflict.conflict_id,
            client_envelope_id=outcome.client_envelope_id,
            domain=outcome.domain,
            entity_id=outcome.entity_id,
            server_sequence=inserted.server_sequence,
            message=outcome.message,
        )

    def _resolve_cursor(
        self,
        dataset_id: str,
        device_id: str,
        cursor: str | int | None,
        domains: Sequence[SyncDomain] | None,
    ) -> int:
        if cursor is not None:
            return self._parse_cursor(cursor)
        cursor_domains = list(domains or self.adapters.supported_domains)
        cursors: list[int] = []
        for domain in cursor_domains:
            stored = self.store.get_device_cursor(dataset_id, device_id, domain)
            cursors.append(stored.last_pulled_sequence if stored is not None else 0)
        return min(cursors, default=0)

    def _parse_cursor(self, cursor: str | int | None) -> int:
        if cursor is None:
            return 0
        try:
            parsed = int(cursor)
        except (TypeError, ValueError) as exc:
            raise SyncStoreError(f"Invalid sync cursor: {cursor!r}") from exc
        if parsed < 0:
            raise SyncStoreError(f"Invalid sync cursor: {cursor!r}")
        return parsed

    def _selected_domains(
        self,
        dataset: SyncDataset,
        domains: Sequence[SyncDomain] | None,
    ) -> list[SyncDomain]:
        allowed = set(dataset.domains)
        requested = list(domains or dataset.domains)
        return [domain for domain in requested if domain in allowed and self.adapters.has_domain(domain)]

    def _update_cursors(
        self,
        dataset_id: str,
        device_id: str,
        domains: Sequence[SyncDomain],
        sequence: int,
    ) -> None:
        for domain in domains:
            self.store.update_device_cursor(
                SyncDeviceCursor(
                    dataset_id=dataset_id,
                    device_id=device_id,
                    domain=domain,
                    last_pulled_sequence=sequence,
                )
            )

    def _scan_pull_page(
        self,
        *,
        dataset_id: str,
        device_id: str,
        since_sequence: int,
        domains: Sequence[SyncDomain],
        page_limit: int,
        include_own_changes: bool,
    ) -> tuple[list[SyncEnvelope], list[SyncEnvelope]]:
        visible = self.store.list_envelopes_after(
            dataset_id,
            since_sequence,
            limit=page_limit + 1,
            domains=domains,
            status="accepted",
            exclude_device_id=None if include_own_changes else device_id,
        )
        return visible, visible

    def _manifest_dataset(
        self,
        dataset: SyncDataset,
        *,
        user_id: str,
        domains: set[SyncDomain],
    ) -> SyncRestoreManifestDataset:
        selected_domains = [domain for domain in dataset.domains if not domains or domain in domains]
        stats = self.store.summarize_restore_manifest_dataset(
            dataset.dataset_id,
            user_id=user_id,
            domains=selected_domains,
        )
        last_updated_at = stats.last_updated_at or dataset.updated_at
        if last_updated_at < dataset.updated_at:
            last_updated_at = dataset.updated_at
        metadata: dict[str, object] = (
            {} if dataset.encryption_policy == "client_private_v1" else dict(dataset.metadata)
        )
        return SyncRestoreManifestDataset(
            dataset_id=dataset.dataset_id,
            scope_type=dataset.scope_type,
            encryption_policy=dataset.encryption_policy,
            domains=selected_domains,
            workspace_id=dataset.workspace_id,
            approximate_counts=stats.approximate_counts,
            byte_estimates=stats.byte_estimates,
            last_updated_at=last_updated_at,
            unresolved_conflicts=stats.unresolved_conflicts,
            attachment_availability=stats.attachment_availability,
            attachment_size_classes=stats.attachment_size_classes,
            key_recovery_available=stats.key_recovery_available,
            metadata=metadata,
        )


def _compact_json_size(value: object) -> int:
    return len(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )


def _ciphertext_exceeds_attachment_limit(
    payload_ciphertext: str,
    max_attachment_bytes: int,
) -> bool:
    """Return whether textual ciphertext is implausibly large for the binary cap."""

    return len(payload_ciphertext.encode("utf-8")) > max_attachment_bytes * 2


def _call_adapter_evaluate(
    adapter: SyncDomainAdapter,
    envelope: SyncEnvelopeCreate,
    *,
    dataset: SyncDataset,
    context: SyncAdapterContext,
) -> AdapterAccepted | AdapterRejected | AdapterConflict | AdapterDeferred:
    evaluate = adapter.evaluate_envelope
    if _evaluate_accepts_context(evaluate):
        return evaluate(envelope, dataset=dataset, context=context)
    return evaluate(envelope, dataset=dataset)


def _evaluate_accepts_context(evaluate: Callable[..., object]) -> bool:
    parameters = inspect.signature(evaluate).parameters.values()
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "context"
            and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
        )
        for parameter in parameters
    )


__all__ = [
    "SyncDatasetEnrollment",
    "SyncDeviceRegistration",
    "SyncPullResult",
    "SyncPushAccepted",
    "SyncPushConflict",
    "SyncPushRejected",
    "SyncPushResult",
    "SyncRestoreManifest",
    "SyncRestoreManifestDataset",
    "SyncRestoreManifestDevice",
    "SyncV2Capabilities",
    "SyncV2Service",
    "SyncV2Settings",
]
