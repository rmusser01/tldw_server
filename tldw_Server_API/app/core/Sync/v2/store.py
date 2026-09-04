"""Core-facing Sync v2 store facade."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from time import monotonic_ns
from typing import Any, Literal

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase

from .errors import SyncStoreError
from .models import (
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION,
    ConflictStatus,
    SyncApplyStatus,
    SyncAttachment,
    SyncAttachmentCreate,
    SyncAttachmentRevisionBinding,
    SyncBackgroundDomainStatus,
    SyncBackgroundLease,
    SyncBackgroundLeaseCreate,
    SyncBackgroundPolicy,
    SyncBackgroundPolicyUpsert,
    SyncBlobAvailabilityStatus,
    SyncBlobChunk,
    SyncBlobChunkCreate,
    SyncBlobObject,
    SyncBlobObjectCreate,
    SyncBlobQuotaUsage,
    SyncBlobUploadSession,
    SyncBlobUploadSessionCreate,
    SyncConflict,
    SyncConflictCreate,
    SyncDataset,
    SyncDatasetCreate,
    SyncDatasetStorageNamespace,
    SyncDevice,
    SyncDeviceAcknowledgmentSummary,
    SyncDeviceAuthorization,
    SyncDeviceAuthorizationCreate,
    SyncDeviceBlobAck,
    SyncDeviceBlobAckCreate,
    SyncDeviceBlobIdAck,
    SyncDeviceBlobIdAckCreate,
    SyncDeviceCursor,
    SyncDeviceDomainAck,
    SyncDeviceDomainAckCreate,
    SyncDeviceUpsert,
    SyncDomain,
    SyncDomainEnvelopeSummary,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncKeyRecord,
    SyncKeyRecordCreate,
    SyncKeyRotationEnvelopeRange,
    SyncNotesAttachmentCleanupCandidate,
    SyncNotesAttachmentSourceMap,
    SyncObjectState,
    SyncRestoreManifestStats,
)


@dataclass(frozen=True, slots=True)
class PersonalContextAuthorityScan:
    """Filtered egress with its independent raw scan checkpoint."""

    raw_scan_watermark: int
    visible_envelopes: list[SyncEnvelope]
    has_visible_lookahead: bool
    source_exhausted: bool = False
    raw_rows_scanned: int = 0


class SyncV2Store:
    """Core Sync v2 persistence interface backed by DB_Management."""

    def __init__(self, db: SyncDatabase, *, connection: Any | None = None) -> None:
        self.db = db
        self._connection = connection
        self._trusted_notes_task_bootstrap_id: str | None = None
        self._trusted_notes_task_coordinator = False

    @contextmanager
    def materialization_guard(
        self,
        envelopes: Sequence[SyncEnvelope | SyncEnvelopeCreate],
        *,
        require_predecessors: bool = True,
        trusted_notes_task_bootstrap_id: str | None = None,
        trusted_notes_task_coordinator: bool = False,
    ) -> Iterator[SyncV2Store]:
        """Hold the durable dataset lock and one Sync transaction for projection."""

        keys = [
            (envelope.dataset_id, envelope.domain, envelope.object_id)
            for envelope in envelopes
        ]
        with self.db.materialization_transaction(
            keys,
            trusted_notes_task_bootstrap_id=trusted_notes_task_bootstrap_id,
            trusted_notes_task_coordinator=trusted_notes_task_coordinator,
        ) as connection:
            guarded = copy(self)
            guarded._connection = connection
            guarded._trusted_notes_task_bootstrap_id = (
                trusted_notes_task_bootstrap_id
            )
            guarded._trusted_notes_task_coordinator = trusted_notes_task_coordinator
            if require_predecessors:
                self.db.require_materialization_predecessors_applied(
                    envelopes,
                    connection=connection,
                )
            yield guarded

    @contextmanager
    def personal_context_authority_guard(
        self,
        dataset_id: str,
        profile_id: str,
    ) -> Iterator[SyncV2Store]:
        """Hold the dataset transaction before entering an authority source guard."""

        with self.db.materialization_transaction(
            [(dataset_id, "personal_context.manifest", profile_id)]
        ) as connection:
            guarded = copy(self)
            guarded._connection = connection
            yield guarded

    def commit_personal_context_authority(self) -> None:
        """Commit the authority transaction while its external source guard is held."""

        if self._connection is None:
            raise SyncStoreError("Personal Context authority guard is required")
        self.db.commit_personal_context_authority_transaction(
            connection=self._connection
        )

    @contextmanager
    def retention_guard(self, dataset_id: str, blob_id: str) -> Iterator[SyncV2Store]:
        """Hold the dataset ordering fence and one transaction for blob GC."""

        with self.db.materialization_transaction(
            [(dataset_id, "attachment.ref", blob_id)]
        ) as connection:
            guarded = copy(self)
            guarded._connection = connection
            yield guarded

    @contextmanager
    def retention_domain_guard(
        self,
        dataset_id: str,
        domain: SyncDomain,
        object_ids: Sequence[str],
    ) -> Iterator[SyncV2Store]:
        """Hold one dataset fence while revalidating a domain checkpoint."""

        keys = [(dataset_id, domain, object_id) for object_id in object_ids]
        with self.db.materialization_transaction(keys) as connection:
            guarded = copy(self)
            guarded._connection = connection
            yield guarded

    @contextmanager
    def blob_write_guard(
        self,
        dataset_id: str,
        domain: SyncDomain,
        object_id: str,
    ) -> Iterator[SyncV2Store]:
        """Hold the dataset ordering fence through blob publication and commit."""

        with self.db.materialization_transaction(
            [(dataset_id, domain, object_id)]
        ) as connection:
            guarded = copy(self)
            guarded._connection = connection
            yield guarded

    def upsert_device(
        self,
        device: SyncDeviceUpsert,
        *,
        capabilities_resolver: Callable[
            [SyncDevice | None], dict[str, object]
        ]
        | None = None,
    ) -> SyncDevice:
        return self.db.upsert_device(
            device,
            capabilities_resolver=capabilities_resolver,
        )

    def get_device(self, user_id: str, device_id: str) -> SyncDevice | None:
        return self.db.get_device(user_id, device_id)

    def enroll_dataset(self, dataset: SyncDatasetCreate) -> SyncDataset:
        return self.db.enroll_dataset(dataset)

    def bind_personal_context_dataset(
        self,
        *,
        dataset_id: str,
        user_id: str,
        expected_binding: Mapping[str, object] | None,
        profile_id: str,
        authority_id: str,
        integrity_key_id: str,
        purge_generation: int,
        link_state: str,
    ) -> SyncDataset:
        """Merge the server-authoritative binding without rewriting other state."""

        return self.db.bind_personal_context_dataset(
            dataset_id=dataset_id,
            user_id=user_id,
            expected_binding=expected_binding,
            profile_id=profile_id,
            authority_id=authority_id,
            integrity_key_id=integrity_key_id,
            purge_generation=purge_generation,
            link_state=link_state,
        )

    def ensure_personal_context_transport_domains(
        self,
        *,
        dataset_id: str,
        user_id: str,
    ) -> SyncDataset:
        """Enroll content-free PC streams before snapshot fencing."""

        return self.db.ensure_personal_context_transport_domains(
            dataset_id=dataset_id,
            user_id=user_id,
        )

    def get_dataset(
        self,
        dataset_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> SyncDataset | None:
        return self.db.get_dataset(
            dataset_id,
            owner_user_id=owner_user_id,
            connection=self._connection,
        )

    @contextmanager
    def personal_context_transport_snapshot(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        streams: Sequence[tuple[SyncDomain, int]],
    ) -> Iterator[dict[tuple[SyncDomain, int], int]]:
        """Hold the dataset insert fence while canonical bootstrap state is read."""

        with self.db.personal_context_transport_snapshot(
            dataset_id,
            owner_user_id=owner_user_id,
            streams=streams,
        ) as watermarks:
            yield watermarks

    def complete_personal_context_link_receipt(
        self,
        *,
        user_id: str,
        dataset_id: str,
        device_id: str,
        profile_id: str,
        integrity_key_id: str,
        purge_generation: int,
        bootstrap_cursor: str,
    ) -> None:
        """Atomically persist one device-bound Personal Context link receipt."""

        self.db.complete_personal_context_link_receipt(
            user_id=user_id,
            dataset_id=dataset_id,
            device_id=device_id,
            profile_id=profile_id,
            integrity_key_id=integrity_key_id,
            purge_generation=purge_generation,
            bootstrap_cursor=bootstrap_cursor,
        )

    def has_personal_context_link_receipt(
        self, *, user_id: str, dataset_id: str, device_id: str, profile_id: str,
        integrity_key_id: str, purge_generation: int,
    ) -> bool:
        """Return whether this exact device has the current server-owned receipt."""

        return self.db.has_personal_context_link_receipt(
            user_id=user_id,
            dataset_id=dataset_id,
            device_id=device_id,
            profile_id=profile_id,
            integrity_key_id=integrity_key_id,
            purge_generation=purge_generation,
        )

    def list_datasets_for_user(self, user_id: str) -> list[SyncDataset]:
        return self.db.list_datasets_for_user(user_id)

    def list_devices_for_user(
        self,
        user_id: str,
        *,
        include_revoked: bool = False,
    ) -> list[SyncDevice]:
        return self.db.list_devices_for_user(
            user_id,
            include_revoked=include_revoked,
            connection=self._connection,
        )

    def create_device_authorization(
        self,
        authorization: SyncDeviceAuthorizationCreate,
    ) -> SyncDeviceAuthorization:
        return self.db.create_device_authorization(authorization)

    def approve_device_authorization(
        self,
        authorization_id: str,
        *,
        user_id: str,
        dataset_id: str,
        approving_device_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> SyncDeviceAuthorization:
        return self.db.approve_device_authorization(
            authorization_id,
            user_id=user_id,
            dataset_id=dataset_id,
            approving_device_id=approving_device_id,
            idempotency_key=idempotency_key,
        )

    def revoke_device(
        self,
        *,
        user_id: str,
        device_id: str,
        reason: str | None = None,
        revoke_key_records: bool = False,
    ) -> SyncDevice:
        return self.db.revoke_device(
            user_id=user_id,
            device_id=device_id,
            reason=reason,
            revoke_key_records=revoke_key_records,
        )

    def upsert_device_domain_ack(
        self,
        acknowledgment: SyncDeviceDomainAckCreate,
    ) -> SyncDeviceDomainAck:
        return self.db.upsert_device_domain_ack(
            acknowledgment,
            connection=self._connection,
        )

    def get_device_domain_ack(
        self,
        dataset_id: str,
        device_id: str,
        domain: SyncDomain,
        *,
        adapter_version: int = 1,
    ) -> SyncDeviceDomainAck | None:
        return self.db.get_device_domain_ack(
            dataset_id,
            device_id,
            domain,
            adapter_version=adapter_version,
            connection=self._connection,
        )

    def upsert_device_blob_ack(
        self,
        acknowledgment: SyncDeviceBlobAckCreate,
    ) -> SyncDeviceBlobAck:
        return self.db.upsert_device_blob_ack(
            acknowledgment,
            connection=self._connection,
        )

    def upsert_device_blob_id_ack(
        self,
        acknowledgment: SyncDeviceBlobIdAckCreate,
    ) -> SyncDeviceBlobIdAck:
        return self.db.upsert_device_blob_id_ack(
            acknowledgment,
            connection=self._connection,
        )

    def list_device_acknowledgments(
        self,
        dataset_id: str,
        device_id: str,
    ) -> SyncDeviceAcknowledgmentSummary:
        return self.db.list_device_acknowledgments(
            dataset_id,
            device_id,
            connection=self._connection,
        )

    def acknowledge_device_state_atomic(
        self,
        dataset_id: str,
        device_id: str,
        *,
        domain_acks: Sequence[SyncDeviceDomainAckCreate] = (),
        blob_acks: Sequence[SyncDeviceBlobAckCreate] = (),
        blob_id_acks: Sequence[SyncDeviceBlobIdAckCreate] = (),
    ) -> SyncDeviceAcknowledgmentSummary:
        return self.db.acknowledge_device_state_atomic(
            dataset_id,
            device_id,
            domain_acks=domain_acks,
            blob_acks=blob_acks,
            blob_id_acks=blob_id_acks,
        )

    def get_background_policy(
        self,
        dataset_id: str,
        device_id: str,
    ) -> SyncBackgroundPolicy | None:
        return self.db.get_background_policy(dataset_id, device_id)

    def upsert_background_policy(
        self,
        policy: SyncBackgroundPolicyUpsert,
    ) -> SyncBackgroundPolicy:
        return self.db.upsert_background_policy(policy)

    def get_background_lease(
        self,
        dataset_id: str,
        device_id: str,
    ) -> SyncBackgroundLease | None:
        return self.db.get_background_lease(dataset_id, device_id)

    def acquire_background_lease(
        self,
        lease: SyncBackgroundLeaseCreate,
    ) -> SyncBackgroundLease:
        return self.db.acquire_background_lease(lease)

    def summarize_background_domains(
        self,
        dataset_id: str,
        device_id: str,
        *,
        domains: Sequence[SyncDomain] | None = None,
    ) -> list[SyncBackgroundDomainStatus]:
        return self.db.summarize_background_domains(
            dataset_id,
            device_id,
            domains=domains,
        )

    def get_or_create_default_personal_dataset(self, user_id: str) -> SyncDataset:
        return self.db.get_or_create_default_personal_dataset(user_id)

    def transition_notes_task_readiness(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        expected_state: str,
        state: str,
        source_dataset_id: str,
        source_cursor: str | None,
        source_count: int,
        source_fingerprint: str | None,
        reason_code: str | None = None,
        task_activity_capture_enabled: bool | None = None,
        captured_source_rebase: bool = False,
    ) -> SyncDataset:
        """Delegate one dormant notes.task readiness transition."""

        return self.db.transition_notes_task_domain_readiness(
            dataset_id,
            owner_user_id=owner_user_id,
            readiness_key="notes_task_v1",
            expected_state=expected_state,
            state=state,
            source_dataset_id=source_dataset_id,
            source_cursor=source_cursor,
            source_count=source_count,
            source_fingerprint=source_fingerprint,
            reason_code=reason_code,
            task_activity_capture_enabled=task_activity_capture_enabled,
            captured_source_rebase=captured_source_rebase,
        )

    def transition_notes_task_activity_readiness(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        expected_state: str,
        state: str,
        source_dataset_id: str,
        source_cursor: str | None,
        source_count: int,
        source_fingerprint: str | None,
        reason_code: str | None = None,
        task_activity_capture_enabled: bool | None = None,
        captured_source_rebase: bool = False,
    ) -> SyncDataset:
        """Delegate one dormant notes.task_activity readiness transition."""

        return self.db.transition_notes_task_domain_readiness(
            dataset_id,
            owner_user_id=owner_user_id,
            readiness_key="notes_task_activity_v1",
            expected_state=expected_state,
            state=state,
            source_dataset_id=source_dataset_id,
            source_cursor=source_cursor,
            source_count=source_count,
            source_fingerprint=source_fingerprint,
            reason_code=reason_code,
            task_activity_capture_enabled=task_activity_capture_enabled,
            captured_source_rebase=captured_source_rebase,
        )

    def transition_notes_moodboard_graph_readiness(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        expected_state: str,
        state: str,
        source_dataset_id: str,
        moodboard_source_cursor: str | None,
        moodboard_source_count: int,
        moodboard_source_fingerprint: str | None,
        placement_source_cursor: str | None,
        placement_source_count: int,
        placement_source_fingerprint: str | None,
        moodboard_reason_code: str | None = None,
        placement_reason_code: str | None = None,
        moodboard_capture_enabled: bool | None = None,
    ) -> SyncDataset:
        """Delegate coupled dormant moodboard/placement readiness transition."""

        return self.db.transition_notes_moodboard_graph_readiness(
            dataset_id,
            owner_user_id=owner_user_id,
            expected_state=expected_state,
            state=state,
            source_dataset_id=source_dataset_id,
            moodboard_source_cursor=moodboard_source_cursor,
            moodboard_source_count=moodboard_source_count,
            moodboard_source_fingerprint=moodboard_source_fingerprint,
            placement_source_cursor=placement_source_cursor,
            placement_source_count=placement_source_count,
            placement_source_fingerprint=placement_source_fingerprint,
            moodboard_reason_code=moodboard_reason_code,
            placement_reason_code=placement_reason_code,
            moodboard_capture_enabled=moodboard_capture_enabled,
        )

    def transition_notes_studio_document_readiness(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        expected_state: str,
        state: str,
        source_dataset_id: str,
        source_cursor: str | None,
        source_count: int,
        source_fingerprint: str | None,
        reason_code: str | None = None,
        studio_document_capture_enabled: bool | None = None,
    ) -> SyncDataset:
        """Delegate independent dormant Studio readiness transition."""

        return self.db.transition_notes_studio_document_readiness(
            dataset_id,
            owner_user_id=owner_user_id,
            expected_state=expected_state,
            state=state,
            source_dataset_id=source_dataset_id,
            source_cursor=source_cursor,
            source_count=source_count,
            source_fingerprint=source_fingerprint,
            reason_code=reason_code,
            studio_document_capture_enabled=studio_document_capture_enabled,
        )

    def begin_notes_task_activation(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
    ) -> SyncDataset:
        """Enable coupled task/activity capture before bootstrap scans."""

        return self.db.begin_notes_task_activation(
            dataset_id,
            owner_user_id=owner_user_id,
        )

    def activate_notes_task_domains(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
    ) -> SyncDataset:
        """Publish both ready task domains atomically."""

        return self.db.activate_notes_task_domains(
            dataset_id,
            owner_user_id=owner_user_id,
        )

    def begin_notes_organization_bootstrap(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
    ) -> SyncDataset:
        return self.db.begin_notes_organization_bootstrap(
            dataset_id,
            owner_user_id=owner_user_id,
            bootstrap_id=bootstrap_id,
        )

    def transition_notes_organization_bootstrap(
        self,
        dataset_id: str,
        *,
        bootstrap_id: str,
        expected_state: str,
        state: str,
        captured_count: int,
        expected_count: int,
        error_code: str | None = None,
        ready_verifier: Callable[[], bool] | None = None,
    ) -> SyncDataset:
        return self.db.transition_notes_organization_bootstrap(
            dataset_id,
            bootstrap_id=bootstrap_id,
            expected_state=expected_state,
            state=state,
            captured_count=captured_count,
            expected_count=expected_count,
            error_code=error_code,
            ready_verifier=ready_verifier,
        )

    def begin_notes_link_bootstrap(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
    ) -> SyncDataset:
        return self.db.begin_notes_link_bootstrap(
            dataset_id,
            owner_user_id=owner_user_id,
            bootstrap_id=bootstrap_id,
        )

    def transition_notes_link_bootstrap(
        self,
        dataset_id: str,
        *,
        bootstrap_id: str,
        expected_state: str,
        state: str,
        captured_count: int,
        expected_count: int,
        source_hash: str | None,
        error_code: str | None = None,
        ready_verifier: Callable[[], bool] | None = None,
    ) -> SyncDataset:
        return self.db.transition_notes_link_bootstrap(
            dataset_id,
            bootstrap_id=bootstrap_id,
            expected_state=expected_state,
            state=state,
            captured_count=captured_count,
            expected_count=expected_count,
            source_hash=source_hash,
            error_code=error_code,
            ready_verifier=ready_verifier,
        )

    def begin_notes_attachment_bootstrap(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
    ) -> SyncDataset:
        return self.db.begin_notes_attachment_bootstrap(
            dataset_id,
            owner_user_id=owner_user_id,
            bootstrap_id=bootstrap_id,
        )

    def transition_notes_attachment_bootstrap(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
        expected_state: str,
        state: str,
        captured_count: int,
        expected_count: int,
        source_hash: str | None,
        source_cursor: str | None,
        error_code: str | None = None,
        ready_verifier: Callable[[], bool] | None = None,
    ) -> SyncDataset:
        return self.db.transition_notes_attachment_bootstrap(
            dataset_id,
            owner_user_id=owner_user_id,
            bootstrap_id=bootstrap_id,
            expected_state=expected_state,
            state=state,
            captured_count=captured_count,
            expected_count=expected_count,
            source_hash=source_hash,
            source_cursor=source_cursor,
            error_code=error_code,
            ready_verifier=ready_verifier,
        )

    def resolve_notes_attachment_source_map(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
        note_id: str,
        source_key: str,
    ) -> SyncNotesAttachmentSourceMap:
        return self.db.resolve_notes_attachment_source_map(
            dataset_id,
            owner_user_id=owner_user_id,
            bootstrap_id=bootstrap_id,
            note_id=note_id,
            source_key=source_key,
        )

    def record_notes_attachment_cleanup_candidate(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
        source_key: str,
        source_relative_path: str,
        source_blob_hash: str,
        source_size_bytes: int,
        source_modified_ns: int,
    ) -> SyncNotesAttachmentCleanupCandidate:
        return self.db.record_notes_attachment_cleanup_candidate(
            dataset_id,
            owner_user_id=owner_user_id,
            bootstrap_id=bootstrap_id,
            source_key=source_key,
            source_relative_path=source_relative_path,
            source_blob_hash=source_blob_hash,
            source_size_bytes=source_size_bytes,
            source_modified_ns=source_modified_ns,
        )

    def list_notes_attachment_cleanup_candidates(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
        after_source_key_hash: str | None = None,
        limit: int = 1_000,
    ) -> tuple[SyncNotesAttachmentCleanupCandidate, ...]:
        return self.db.list_notes_attachment_cleanup_candidates(
            dataset_id,
            owner_user_id=owner_user_id,
            bootstrap_id=bootstrap_id,
            after_source_key_hash=after_source_key_hash,
            limit=limit,
        )

    def get_notes_attachment_bootstrap_source_by_hash(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        bootstrap_id: str,
        source_key_hash: str,
    ) -> tuple[
        SyncNotesAttachmentSourceMap,
        SyncNotesAttachmentCleanupCandidate,
    ] | None:
        """Resolve one internal bootstrap source by its public-safe hash."""

        return self.db.get_notes_attachment_bootstrap_source_by_hash(
            dataset_id,
            owner_user_id=owner_user_id,
            bootstrap_id=bootstrap_id,
            source_key_hash=source_key_hash,
        )

    def insert_envelope(self, envelope: SyncEnvelopeCreate) -> SyncEnvelope:
        return self.db.insert_envelope(envelope, connection=self._connection)

    def insert_claimed_conflict_resolution_envelope(
        self,
        envelope: SyncEnvelopeCreate,
        *,
        conflict_id: str,
        dataset_id: str,
        resolved_by_device_id: str | None,
        resolution_action: str,
        resolution_notes: str | None,
    ) -> SyncEnvelope:
        if self._connection is None:
            raise SyncStoreError("Sync conflict resolution requires a dataset guard")
        return self.db.insert_claimed_conflict_resolution_envelope(
            envelope,
            conflict_id=conflict_id,
            dataset_id=dataset_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            connection=self._connection,
        )

    def get_latest_applied_predecessor(
        self,
        envelope: SyncEnvelope,
    ) -> SyncEnvelope | None:
        if self._connection is None:
            raise SyncStoreError("Sync projected-base lookup requires a dataset guard")
        return self.db.get_latest_applied_predecessor(
            envelope,
            connection=self._connection,
        )

    def list_latest_applied_heads(
        self,
        dataset_id: str,
        *,
        through_server_cursor: int | None = None,
    ) -> list[SyncEnvelope]:
        if self._connection is None:
            raise SyncStoreError("Sync projected-head lookup requires a dataset guard")
        return self.db.list_latest_applied_heads(
            dataset_id,
            through_server_cursor=through_server_cursor,
            connection=self._connection,
        )

    def insert_envelopes_atomic(
        self,
        envelopes: Sequence[SyncEnvelopeCreate],
        *,
        trusted_notes_organization_bootstrap_id: str | None = None,
        trusted_notes_task_bootstrap_id: str | None = None,
        trusted_notes_task_coordinator: bool = False,
    ) -> list[SyncEnvelope]:
        """Insert one complete validated group or return its exact stored replay."""

        return self.db.insert_envelopes_atomic(
            envelopes,
            trusted_notes_organization_bootstrap_id=trusted_notes_organization_bootstrap_id,
            trusted_notes_task_bootstrap_id=trusted_notes_task_bootstrap_id,
            trusted_notes_task_coordinator=trusted_notes_task_coordinator,
        )

    def list_mutation_group(
        self,
        dataset_id: str,
        mutation_group_id: str,
    ) -> list[SyncEnvelope]:
        """Return a complete mutation group ordered by zero-based step."""

        return self.db.list_mutation_group(
            dataset_id,
            mutation_group_id,
            connection=self._connection,
        )

    def get_existing_envelope_for_idempotency(
        self,
        envelope: SyncEnvelopeCreate,
    ) -> SyncEnvelope | None:
        return self.db.get_existing_envelope_for_idempotency(envelope)

    def list_envelopes_after(
        self,
        dataset_id: str,
        since_sequence: int,
        *,
        limit: int = 100,
        domains: Sequence[SyncDomain] | None = None,
        adapter_versions: Sequence[int] | None = None,
        status: str | Sequence[str] | None = None,
        exclude_device_id: str | None = None,
    ) -> list[SyncEnvelope]:
        return self.db.list_envelopes_after(
            dataset_id,
            since_sequence,
            limit=limit,
            domains=domains,
            adapter_versions=adapter_versions,
            status=status,
            exclude_device_id=exclude_device_id,
            connection=self._connection,
        )

    def scan_personal_context_authority(
        self,
        dataset_id: str,
        *,
        after_server_cursor: int,
        limit: int,
        row_budget: int = 100,
        wall_time_ms: int = 100,
        deadline_ns: int | None = None,
        domains: Sequence[SyncDomain] | None = None,
        adapter_versions: Sequence[int] | None = None,
        exclude_device_id: str | None = None,
        profile_id: str | None = None,
        integrity_key_id: str | None = None,
        purge_generation: int | None = None,
    ) -> PersonalContextAuthorityScan:
        """Scan a mixed page without exposing or advancing past unsafe PC rows."""

        if row_budget < 1 or wall_time_ms < 1:
            raise ValueError("Personal Context scan limits must be positive")
        raw_cursor = after_server_cursor
        raw_seen = 0
        visible: list[SyncEnvelope] = []
        source_exhausted = False
        deadline_ns = deadline_ns or monotonic_ns() + wall_time_ms * 1_000_000
        selected_domains = tuple(domains or PERSONAL_CONTEXT_SYNC_DOMAINS)
        conflict = self.get_unresolved_materialization_conflict(dataset_id)
        conflict_cursor = (
            conflict.server_sequence
            if conflict is not None
            and conflict.conflict_type
            != SYNC_REBASE_REQUIRED_AFTER_CONFLICT_RESOLUTION
            else None
        )
        while raw_seen < row_budget and monotonic_ns() < deadline_ns and len(visible) <= limit:
            chunk_limit = min(row_budget - raw_seen, max(1, limit + 1))
            raw = self.list_envelopes_after(
                dataset_id,
                raw_cursor,
                limit=chunk_limit,
                domains=selected_domains,
                adapter_versions=adapter_versions,
                status="accepted",
                exclude_device_id=exclude_device_id,
            )
            if not raw:
                source_exhausted = True
                break
            barrier = False
            for envelope in raw:
                raw_seen += 1
                if (
                    conflict_cursor is not None
                    and envelope.server_sequence >= conflict_cursor
                ):
                    barrier = True
                    break
                authority = envelope.authority
                if (
                    envelope.domain in PERSONAL_CONTEXT_SYNC_DOMAINS
                    and (
                        envelope.routing_metadata.get("profile_id") != profile_id
                        or envelope.routing_metadata.get("integrity_key_id")
                        != integrity_key_id
                        or envelope.routing_metadata.get("purge_generation")
                        != purge_generation
                    )
                ):
                    raw_cursor = envelope.server_cursor or raw_cursor
                    continue
                if (
                    envelope.domain in PERSONAL_CONTEXT_SYNC_DOMAINS
                    and authority is not None
                    and authority.role == "home_authority"
                    and envelope.apply_status != "applied"
                ):
                    barrier = True
                    break
                raw_cursor = envelope.server_cursor or raw_cursor
                if envelope.domain not in PERSONAL_CONTEXT_SYNC_DOMAINS:
                    if envelope.apply_status not in {"conflict", "superseded"}:
                        visible.append(envelope)
                elif (
                    envelope.apply_status == "applied"
                    and authority is not None
                    and authority.role == "home_authority"
                ):
                    visible.append(envelope)
            if barrier:
                break
            if len(raw) < chunk_limit:
                source_exhausted = True
                break
        return PersonalContextAuthorityScan(
            raw_scan_watermark=raw_cursor,
            visible_envelopes=visible[:limit],
            has_visible_lookahead=len(visible) > limit,
            source_exhausted=source_exhausted,
            raw_rows_scanned=raw_seen,
        )

    def mark_personal_context_ingress_applied(
        self,
        *,
        server_cursor: int,
        receipt: Any,
    ) -> SyncEnvelope:
        """Terminalize only the exact ingress whose canonical receipt was verified."""

        envelope = self.get_envelope_by_server_cursor(server_cursor)
        if (
            envelope is None
            or envelope.client_envelope_id != receipt.client_envelope_id
            or not receipt.receipt_id.strip()
            or envelope.authority is None
            or envelope.authority.role != "client_ingress"
        ):
            raise SyncStoreError("personal_context_ingress_receipt_mismatch")
        return self.db.mark_personal_context_ingress_applied(
            server_cursor=server_cursor,
            receipt={
                "dataset_id": receipt.dataset_id,
                "device_id": receipt.device_id,
                "client_envelope_id": receipt.client_envelope_id,
                "canonical_payload_digest": receipt.canonical_payload_digest,
                "purge_generation": receipt.purge_generation,
                "resulting_object_id": receipt.resulting_object_id,
                "resulting_version_id": receipt.resulting_version_id,
                "manifest_revision": receipt.manifest_revision,
                "manifest_version_id": receipt.manifest_version_id,
                "publication_batch_id": receipt.publication_batch_id,
                "profile_publication_sequence": receipt.profile_publication_sequence,
                "receipt_id": receipt.receipt_id,
                "wire_entity_version": receipt.wire_entity_version,
            },
            connection=self._connection,
        )

    def get_personal_context_ingress_receipt(
        self,
        server_cursor: int,
    ) -> Mapping[str, Any] | None:
        """Read the canonical apply receipt bound to one exact ingress cursor."""

        return self.db.get_personal_context_ingress_receipt(
            server_cursor,
            connection=self._connection,
        )

    def summarize_domain_envelopes(
        self,
        dataset_id: str,
        domain: SyncDomain,
    ) -> SyncDomainEnvelopeSummary:
        return self.db.summarize_domain_envelopes(dataset_id, domain)

    def list_envelopes_for_entity(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        entity_id: str | None = None,
        stable_key: str | None = None,
        limit: int = 100,
    ) -> list[SyncEnvelope]:
        return self.db.list_envelopes_for_entity(
            dataset_id,
            domain,
            entity_id=entity_id,
            stable_key=stable_key,
            limit=limit,
            connection=self._connection,
        )

    def get_historical_task_envelope(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        task_id: str,
        object_revision: int,
        object_hash: str,
        envelope_id: str | None = None,
    ) -> SyncEnvelope | None:
        """Resolve one exact applied historical Notes task envelope."""
        return self.db.get_historical_task_envelope(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            task_id=task_id,
            envelope_id=envelope_id,
            object_revision=object_revision,
            object_hash=object_hash,
            connection=self._connection,
        )

    def get_projection_note_envelope(
        self,
        *,
        owner_user_id: str,
        dataset_id: str,
        note_id: str,
        envelope_id: str,
        object_hash: str,
    ) -> SyncEnvelope | None:
        """Resolve one exact applied note envelope for projection proof."""
        return self.db.get_projection_note_envelope(
            owner_user_id=owner_user_id,
            dataset_id=dataset_id,
            note_id=note_id,
            envelope_id=envelope_id,
            object_hash=object_hash,
            connection=self._connection,
        )

    def get_envelope_for_entity_at_or_before(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        entity_id: str,
        server_sequence: int,
    ) -> SyncEnvelope | None:
        """Return one accepted entity envelope at a durable sequence boundary."""

        return self.db.get_envelope_for_entity_at_or_before(
            dataset_id,
            domain,
            entity_id=entity_id,
            server_sequence=server_sequence,
        )

    def get_envelope_by_server_cursor(self, server_cursor: int) -> SyncEnvelope | None:
        return self.db.get_envelope_by_server_cursor(
            server_cursor,
            connection=self._connection,
        )

    def get_envelope_by_client_id(
        self,
        dataset_id: str,
        client_envelope_id: str,
    ) -> SyncEnvelope | None:
        """Resolve a deterministic envelope ID before reconstructing its original CAS base."""

        return self.db.get_envelope_by_client_id(
            dataset_id,
            client_envelope_id,
            connection=self._connection,
        )

    def get_object_state(
        self,
        dataset_id: str,
        domain: SyncDomain,
        object_id: str,
    ) -> SyncObjectState | None:
        return self.db.get_object_state(
            dataset_id,
            domain,
            object_id,
            connection=self._connection,
        )

    def get_current_head(
        self,
        dataset_id: str,
        domain: SyncDomain,
        object_id: str,
    ) -> SyncEnvelope | None:
        """Return the canonical current head for one dataset-scoped object."""

        return self.db.get_current_head(
            dataset_id,
            domain,
            object_id,
            connection=self._connection,
        )

    def list_current_heads(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        limit: int,
        offset: int,
    ) -> list[SyncEnvelope]:
        """Return a bounded page of canonical heads for one dataset domain."""

        return self.db.list_current_heads(
            dataset_id,
            domain,
            limit=limit,
            offset=offset,
            connection=self._connection,
        )

    def upsert_object_state(self, state: SyncObjectState) -> SyncObjectState:
        return self.db.upsert_object_state(
            state,
            connection=self._connection,
            trusted_notes_task_bootstrap_id=(
                self._trusted_notes_task_bootstrap_id
            ),
            trusted_notes_task_coordinator=self._trusted_notes_task_coordinator,
        )

    def mark_envelope_apply_status(
        self,
        server_cursor: int,
        *,
        apply_status: SyncApplyStatus,
        apply_error_code: str | None = None,
        apply_error_message: str | None = None,
    ) -> SyncEnvelope:
        return self.db.mark_envelope_apply_status(
            server_cursor,
            apply_status=apply_status,
            apply_error_code=apply_error_code,
            apply_error_message=apply_error_message,
            connection=self._connection,
        )

    def discard_pending_personal_context_authority(
        self,
        **identity: Any,
    ) -> Literal["removed", "absent", "applied", "mismatch"]:
        """Classify or remove one exact invisible authority row."""

        return self.db.discard_pending_personal_context_authority(
            **identity,
            connection=self._connection,
        )

    def mark_personal_context_authority_applied(
        self,
        server_cursor: int,
        **identity: Any,
    ) -> SyncEnvelope:
        """Apply one verified authority row inside its existing Sync guard."""

        if self._connection is None:
            raise SyncStoreError("Personal Context authority finalize requires a guard")
        return self.db.mark_personal_context_authority_applied(
            server_cursor,
            **identity,
            connection=self._connection,
        )

    def mark_bootstrap_envelope_verified(
        self,
        server_cursor: int,
        *,
        bootstrap_id: str,
        notes_task_bootstrap: bool = False,
    ) -> SyncEnvelope:
        """Record a verified bootstrap step as applied without product replay."""

        return self.db.mark_bootstrap_envelope_verified(
            server_cursor,
            bootstrap_id=bootstrap_id,
            notes_task_bootstrap=notes_task_bootstrap,
            connection=self._connection,
        )

    def reconcile_bootstrap_envelope_superseded(
        self,
        server_cursor: int,
        *,
        bootstrap_id: str,
        superseded_by_cursor: int,
    ) -> SyncEnvelope:
        """Mark stale bootstrap history applied without regressing object state."""

        return self.db.reconcile_bootstrap_envelope_superseded(
            server_cursor,
            bootstrap_id=bootstrap_id,
            superseded_by_cursor=superseded_by_cursor,
            connection=self._connection,
        )

    def list_failed_applies(
        self,
        dataset_id: str,
        *,
        limit: int = 100,
    ) -> list[SyncEnvelope]:
        return self.db.list_failed_applies(dataset_id, limit=limit)

    def list_accepted_envelopes_for_replay(
        self,
        dataset_id: str,
        *,
        since_cursor: int = 0,
        limit: int = 1000,
    ) -> list[SyncEnvelope]:
        return self.db.list_accepted_envelopes_for_replay(
            dataset_id,
            since_cursor=since_cursor,
            limit=limit,
        )

    def update_device_cursor(self, cursor: SyncDeviceCursor) -> SyncDeviceCursor:
        return self.db.update_device_cursor(cursor)

    def get_device_cursor(
        self,
        dataset_id: str,
        device_id: str,
        domain: SyncDomain,
        *,
        adapter_version: int = 1,
    ) -> SyncDeviceCursor | None:
        return self.db.get_device_cursor(
            dataset_id,
            device_id,
            domain,
            adapter_version=adapter_version,
        )

    def insert_conflict(self, conflict: SyncConflictCreate) -> SyncConflict:
        return self.db.insert_conflict(conflict, connection=self._connection)

    def list_conflicts(
        self,
        dataset_id: str,
        *,
        status: ConflictStatus | None = None,
    ) -> list[SyncConflict]:
        return self.db.list_conflicts(dataset_id, status=status)

    def get_conflict(self, conflict_id: str) -> SyncConflict | None:
        return self.db.get_conflict(conflict_id, connection=self._connection)

    def get_unresolved_conflict_for_envelope(
        self,
        dataset_id: str,
        *,
        local_envelope_id: str,
        server_sequence: int | None = None,
    ) -> SyncConflict | None:
        return self.db.get_unresolved_conflict_for_envelope(
            dataset_id,
            local_envelope_id=local_envelope_id,
            server_sequence=server_sequence,
            connection=self._connection,
        )

    def get_unresolved_materialization_conflict(
        self,
        dataset_id: str,
    ) -> SyncConflict | None:
        return self.db.get_unresolved_materialization_conflict(
            dataset_id,
            connection=self._connection,
        )

    def claim_conflict_resolution(
        self,
        conflict_id: str,
        *,
        dataset_id: str | None = None,
        resolved_by_device_id: str | None = None,
        resolution_action: str | None = None,
        resolution_notes: str | None = None,
    ) -> SyncConflict:
        return self.db.claim_conflict_resolution(
            conflict_id,
            dataset_id=dataset_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            connection=self._connection,
        )

    def release_conflict_resolution_claim(
        self,
        conflict_id: str,
        *,
        dataset_id: str | None = None,
        resolved_by_device_id: str | None = None,
        resolution_action: str | None = None,
        resolution_notes: str | None = None,
    ) -> SyncConflict:
        return self.db.release_conflict_resolution_claim(
            conflict_id,
            dataset_id=dataset_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            connection=self._connection,
        )

    def require_conflict_resolution_predecessors_applied(
        self,
        conflict_id: str,
        *,
        dataset_id: str,
        resolved_by_device_id: str | None,
        resolution_action: str,
        resolution_notes: str | None,
    ) -> SyncEnvelope:
        if self._connection is None:
            raise SyncStoreError("Sync conflict resolution requires a dataset guard")
        return self.db.require_conflict_resolution_predecessors_applied(
            conflict_id,
            dataset_id=dataset_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            connection=self._connection,
        )

    def terminalize_claimed_conflict_envelope(
        self,
        conflict_id: str,
        *,
        dataset_id: str,
        resolved_by_device_id: str | None,
        resolution_action: str,
        resolution_notes: str | None,
        apply_error_code: str,
    ) -> SyncEnvelope:
        if self._connection is None:
            raise SyncStoreError("Sync conflict terminalization requires a dataset guard")
        return self.db.terminalize_claimed_conflict_envelope(
            conflict_id,
            dataset_id=dataset_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            apply_error_code=apply_error_code,
            connection=self._connection,
        )

    def rebase_later_claimed_conflict_envelopes(
        self,
        conflict_id: str,
        *,
        dataset_id: str,
        resolved_by_device_id: str | None,
        resolution_action: str,
        resolution_notes: str | None,
        expected_server_cursors: Sequence[int] | None = None,
    ) -> list[SyncConflict]:
        if self._connection is None:
            raise SyncStoreError("Sync conflict rebasing requires a dataset guard")
        return self.db.rebase_later_claimed_conflict_envelopes(
            conflict_id,
            dataset_id=dataset_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            expected_server_cursors=expected_server_cursors,
            connection=self._connection,
        )

    def stage_later_claimed_conflict_rebase_plan(
        self,
        conflict_id: str,
        *,
        dataset_id: str,
        resolved_by_device_id: str | None,
        resolution_action: str,
        resolution_notes: str | None,
    ) -> tuple[int, ...]:
        """Validate and freeze later-row rebase work before product projection."""

        if self._connection is None:
            raise SyncStoreError("Sync conflict rebasing requires a dataset guard")
        return self.db.stage_later_claimed_conflict_rebase_plan(
            conflict_id,
            dataset_id=dataset_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            connection=self._connection,
        )

    def resolve_conflict(
        self,
        conflict_id: str,
        *,
        dataset_id: str | None = None,
        server_cursor: int | None = None,
        status: ConflictStatus = "resolved",
        resolved_by_envelope_id: str | None = None,
        resolved_by_device_id: str | None = None,
        resolution_action: str | None = None,
        resolution_notes: str | None = None,
    ) -> SyncConflict:
        return self.db.resolve_conflict(
            conflict_id,
            dataset_id=dataset_id,
            server_cursor=server_cursor,
            status=status,
            resolved_by_envelope_id=resolved_by_envelope_id,
            resolved_by_device_id=resolved_by_device_id,
            resolution_action=resolution_action,
            resolution_notes=resolution_notes,
            connection=self._connection,
        )

    def store_key_record(self, record: SyncKeyRecordCreate) -> SyncKeyRecord:
        return self.db.store_key_record(record)

    def list_key_records(
        self,
        dataset_id: str,
        *,
        user_id: str,
        device_id: str | None = None,
        key_purpose: str | None = None,
    ) -> list[SyncKeyRecord]:
        return self.db.list_key_records(
            dataset_id,
            user_id=user_id,
            device_id=device_id,
            key_purpose=key_purpose,
        )

    def revoke_key_record(self, *, user_id: str, key_record_id: str) -> SyncKeyRecord:
        """Revoke one key record after a registered wrapping-key rotation."""

        return self.db.revoke_key_record(user_id=user_id, key_record_id=key_record_id)

    def get_dataset_envelope_range(self, dataset_id: str) -> SyncKeyRotationEnvelopeRange:
        return self.db.get_dataset_envelope_range(dataset_id)

    def commit_key_rotation(
        self,
        record: SyncKeyRecordCreate,
        *,
        source_key_record_ids: Sequence[str],
        superseded_at: str,
    ) -> tuple[SyncKeyRecord, list[SyncKeyRecord], SyncKeyRotationEnvelopeRange]:
        return self.db.commit_key_rotation(
            record,
            source_key_record_ids=source_key_record_ids,
            superseded_at=superseded_at,
        )

    def store_attachment(self, attachment: SyncAttachmentCreate) -> SyncAttachment:
        """Store or deduplicate an encrypted attachment through the DB layer."""

        return self.db.store_attachment(attachment)

    def get_attachment_revision_binding(
        self,
        dataset_id: str,
        attachment_id: str,
        attachment_revision: int,
        *,
        owner_user_id: str,
    ) -> SyncAttachmentRevisionBinding | None:
        return self.db.get_attachment_revision_binding(
            dataset_id,
            attachment_id,
            attachment_revision,
            owner_user_id=owner_user_id,
            connection=self._connection,
        )

    def get_attachment_revision_binding_for_blob(
        self,
        dataset_id: str,
        blob_id: str,
        *,
        owner_user_id: str,
    ) -> SyncAttachmentRevisionBinding | None:
        """Return the latest revision binding resolved to a blob for its owner."""

        return self.db.get_attachment_revision_binding_for_blob(
            dataset_id,
            blob_id,
            owner_user_id=owner_user_id,
            connection=self._connection,
        )

    def list_attachment_revision_bindings_for_blob(
        self,
        dataset_id: str,
        blob_id: str,
        *,
        owner_user_id: str,
        after_establishing_server_cursor: int = 0,
        after_attachment_id: str = "",
        after_attachment_revision: int = 0,
        limit: int = 1000,
    ) -> list[SyncAttachmentRevisionBinding]:
        """List one bounded compound-keyset page of unreleased blob bindings."""

        return self.db.list_attachment_revision_bindings_for_blob(
            dataset_id,
            blob_id,
            owner_user_id=owner_user_id,
            after_establishing_server_cursor=after_establishing_server_cursor,
            after_attachment_id=after_attachment_id,
            after_attachment_revision=after_attachment_revision,
            limit=limit,
            connection=self._connection,
        )

    def has_attachment_ref_v2_history(
        self,
        dataset_id: str,
        attachment_id: str,
        *,
        owner_user_id: str,
    ) -> bool:
        """Return whether an owned attachment has accepted adapter-v2 history."""

        return self.db.has_attachment_ref_v2_history(
            dataset_id,
            attachment_id,
            owner_user_id=owner_user_id,
            connection=self._connection,
        )

    def list_unreleased_attachment_revision_bindings(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        after_establishing_server_cursor: int = 0,
        after_attachment_id: str = "",
        after_attachment_revision: int = 0,
        limit: int = 1000,
    ) -> list[SyncAttachmentRevisionBinding]:
        """List one bounded compound-keyset page of unreleased bindings."""

        return self.db.list_unreleased_attachment_revision_bindings(
            dataset_id,
            owner_user_id=owner_user_id,
            after_establishing_server_cursor=after_establishing_server_cursor,
            after_attachment_id=after_attachment_id,
            after_attachment_revision=after_attachment_revision,
            limit=limit,
            connection=self._connection,
        )

    def list_unresolved_attachment_revision_bindings(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
        after_establishing_server_cursor: int = 0,
        limit: int = 1000,
    ) -> list[SyncAttachmentRevisionBinding]:
        """List one bounded cursor page of unresolved attachment bindings."""

        return self.db.list_unresolved_attachment_revision_bindings(
            dataset_id,
            owner_user_id=owner_user_id,
            after_establishing_server_cursor=after_establishing_server_cursor,
            limit=limit,
        )

    def resolve_attachment_revision_binding(
        self,
        dataset_id: str,
        attachment_id: str,
        attachment_revision: int,
        *,
        blob_id: str,
        owner_user_id: str,
    ) -> SyncAttachmentRevisionBinding:
        return self.db.resolve_attachment_revision_binding(
            dataset_id,
            attachment_id,
            attachment_revision,
            blob_id=blob_id,
            owner_user_id=owner_user_id,
        )

    def release_attachment_revision_binding(
        self,
        dataset_id: str,
        attachment_id: str,
        attachment_revision: int,
        *,
        released_at: str,
        owner_user_id: str,
    ) -> SyncAttachmentRevisionBinding:
        return self.db.release_attachment_revision_binding(
            dataset_id,
            attachment_id,
            attachment_revision,
            released_at=released_at,
            owner_user_id=owner_user_id,
            connection=self._connection,
        )

    def get_or_create_storage_namespace(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
    ) -> SyncDatasetStorageNamespace:
        return self.db.get_or_create_storage_namespace(
            dataset_id,
            owner_user_id=owner_user_id,
        )

    def get_storage_namespace(
        self,
        dataset_id: str,
        *,
        owner_user_id: str,
    ) -> SyncDatasetStorageNamespace | None:
        return self.db.get_storage_namespace(
            dataset_id,
            owner_user_id=owner_user_id,
            connection=self._connection,
        )

    def relocate_legacy_blob(
        self,
        blob_store: Any,
        *,
        dataset_id: str,
        owner_user_id: str,
        blob_id: str,
    ) -> SyncBlobObject:
        return self.db.relocate_legacy_blob(
            blob_store,
            dataset_id=dataset_id,
            owner_user_id=owner_user_id,
            blob_id=blob_id,
        )

    def create_blob_upload_session(
        self,
        session: SyncBlobUploadSessionCreate,
    ) -> SyncBlobUploadSession:
        return self.db.create_blob_upload_session(session)

    def get_blob_upload_session(
        self,
        upload_id: str,
        *,
        dataset_id: str | None = None,
    ) -> SyncBlobUploadSession | None:
        return self.db.get_blob_upload_session(upload_id, dataset_id=dataset_id)

    def cancel_blob_upload_session(
        self,
        upload_id: str,
        *,
        dataset_id: str | None = None,
    ) -> SyncBlobUploadSession:
        return self.db.cancel_blob_upload_session(upload_id, dataset_id=dataset_id)

    def record_blob_chunk(self, chunk: SyncBlobChunkCreate) -> SyncBlobChunk:
        return self.db.record_blob_chunk(chunk)

    def get_blob_chunk(
        self,
        upload_id: str,
        chunk_index: int,
        *,
        dataset_id: str | None = None,
    ) -> SyncBlobChunk | None:
        return self.db.get_blob_chunk(
            upload_id,
            chunk_index,
            dataset_id=dataset_id,
        )

    def complete_blob_upload(self, blob: SyncBlobObjectCreate) -> SyncBlobObject:
        return self.db.complete_blob_upload(blob, connection=self._connection)

    def require_blob_upload_completion_allowed(
        self,
        blob: SyncBlobObjectCreate,
    ) -> None:
        if self._connection is None:
            raise SyncStoreError("Sync blob completion requires a dataset guard")
        self.db.require_blob_upload_completion_allowed(
            blob,
            connection=self._connection,
        )

    def get_blob_object(
        self,
        dataset_id: str,
        *,
        attachment_id: str | None = None,
        blob_id: str | None = None,
        payload_hash: str | None = None,
        owner_user_id: str | None = None,
        include_unavailable: bool = False,
    ) -> SyncBlobObject | None:
        return self.db.get_blob_object(
            dataset_id,
            attachment_id=attachment_id,
            blob_id=blob_id,
            payload_hash=payload_hash,
            owner_user_id=owner_user_id,
            include_unavailable=include_unavailable,
            connection=self._connection,
        )

    def list_blob_availability_by_hashes(
        self,
        dataset_id: str,
        payload_hashes: Sequence[str],
        *,
        owner_user_id: str,
    ) -> dict[str, SyncBlobAvailabilityStatus]:
        """Return bounded owner-authorized blob states keyed by payload hash."""

        return self.db.list_blob_availability_by_hashes(
            dataset_id,
            payload_hashes,
            owner_user_id=owner_user_id,
            connection=self._connection,
        )

    def lock_blob_object_for_retention(
        self,
        dataset_id: str,
        blob_id: str,
        *,
        owner_user_id: str,
    ) -> SyncBlobObject | None:
        if self._connection is None:
            raise SyncStoreError("Sync blob retention requires a dataset guard")
        return self.db.get_blob_object(
            dataset_id,
            blob_id=blob_id,
            owner_user_id=owner_user_id,
            include_unavailable=True,
            connection=self._connection,
            for_update=True,
        )

    def list_blob_objects_for_dataset(
        self,
        dataset_id: str,
        *,
        status: str | None = "available",
    ) -> list[SyncBlobObject]:
        """Return committed blob metadata for retention and diagnostics scans."""

        return self.db.list_blob_objects_for_dataset(dataset_id, status=status)

    def list_blob_objects_for_dataset_page(
        self,
        dataset_id: str,
        *,
        status: str = "available",
        after_updated_at: str | None = None,
        after_blob_id: str | None = None,
        limit: int = 1000,
    ) -> list[SyncBlobObject]:
        return self.db.list_blob_objects_for_dataset_page(
            dataset_id,
            status=status,
            after_updated_at=after_updated_at,
            after_blob_id=after_blob_id,
            limit=limit,
            connection=self._connection,
        )

    def fence_blob_object_deleting(
        self,
        dataset_id: str,
        blob_id: str,
    ) -> SyncBlobObject | None:
        if self._connection is None:
            raise SyncStoreError("Sync blob deletion requires a dataset guard")
        return self.db.fence_blob_object_deleting(
            dataset_id,
            blob_id,
            connection=self._connection,
        )

    def finalize_blob_object_deleted(
        self,
        dataset_id: str,
        blob_id: str,
    ) -> SyncBlobObject | None:
        if self._connection is None:
            raise SyncStoreError("Sync blob deletion requires a dataset guard")
        return self.db.finalize_blob_object_deleted(
            dataset_id,
            blob_id,
            connection=self._connection,
        )

    def get_domain_compaction_sequence(
        self,
        dataset_id: str,
        domain: SyncDomain,
    ) -> int:
        return self.db.get_domain_compaction_sequence(dataset_id, domain)

    def record_domain_compaction(
        self,
        dataset_id: str,
        domain: SyncDomain,
        *,
        through_server_sequence: int,
        state: dict[str, object],
    ) -> int:
        return self.db.record_domain_compaction(
            dataset_id,
            domain,
            through_server_sequence=through_server_sequence,
            state=state,
            connection=self._connection,
        )

    def summarize_blob_quota(
        self,
        owner_user_id: str,
        *,
        dataset_id: str | None = None,
    ) -> SyncBlobQuotaUsage:
        return self.db.summarize_blob_quota(owner_user_id, dataset_id=dataset_id)

    def summarize_restore_manifest_dataset(
        self,
        dataset_id: str,
        *,
        user_id: str,
        domains: Sequence[SyncDomain] | None = None,
    ) -> SyncRestoreManifestStats:
        return self.db.summarize_restore_manifest_dataset(
            dataset_id,
            user_id=user_id,
            domains=domains,
        )


__all__ = ["SyncV2Store"]
