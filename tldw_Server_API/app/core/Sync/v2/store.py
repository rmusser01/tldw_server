from __future__ import annotations

"""Core-facing Sync v2 store facade."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from copy import copy
from typing import Any

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase

from .errors import SyncStoreError
from .models import (
    ConflictStatus,
    SyncApplyStatus,
    SyncAttachment,
    SyncAttachmentCreate,
    SyncBackgroundDomainStatus,
    SyncBackgroundLease,
    SyncBackgroundLeaseCreate,
    SyncBackgroundPolicy,
    SyncBackgroundPolicyUpsert,
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
    SyncDevice,
    SyncDeviceAcknowledgmentSummary,
    SyncDeviceAuthorization,
    SyncDeviceAuthorizationCreate,
    SyncDeviceBlobAck,
    SyncDeviceBlobAckCreate,
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
    SyncObjectState,
    SyncRestoreManifestStats,
)


class SyncV2Store:
    """Core Sync v2 persistence interface backed by DB_Management."""

    def __init__(self, db: SyncDatabase, *, connection: Any | None = None) -> None:
        self.db = db
        self._connection = connection

    @contextmanager
    def materialization_guard(
        self,
        envelopes: Sequence[SyncEnvelope | SyncEnvelopeCreate],
        *,
        require_predecessors: bool = True,
    ) -> Iterator[SyncV2Store]:
        """Hold the durable dataset lock and one Sync transaction for projection."""

        keys = [
            (envelope.dataset_id, envelope.domain, envelope.object_id)
            for envelope in envelopes
        ]
        with self.db.materialization_transaction(keys) as connection:
            guarded = copy(self)
            guarded._connection = connection
            if require_predecessors:
                self.db.require_materialization_predecessors_applied(
                    envelopes,
                    connection=connection,
                )
            yield guarded

    def upsert_device(self, device: SyncDeviceUpsert) -> SyncDevice:
        return self.db.upsert_device(device)

    def get_device(self, user_id: str, device_id: str) -> SyncDevice | None:
        return self.db.get_device(user_id, device_id)

    def enroll_dataset(self, dataset: SyncDatasetCreate) -> SyncDataset:
        return self.db.enroll_dataset(dataset)

    def get_dataset(
        self,
        dataset_id: str,
        *,
        owner_user_id: str | None = None,
    ) -> SyncDataset | None:
        return self.db.get_dataset(dataset_id, owner_user_id=owner_user_id)

    def list_datasets_for_user(self, user_id: str) -> list[SyncDataset]:
        return self.db.list_datasets_for_user(user_id)

    def list_devices_for_user(
        self,
        user_id: str,
        *,
        include_revoked: bool = False,
    ) -> list[SyncDevice]:
        return self.db.list_devices_for_user(user_id, include_revoked=include_revoked)

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
        return self.db.upsert_device_domain_ack(acknowledgment)

    def upsert_device_blob_ack(
        self,
        acknowledgment: SyncDeviceBlobAckCreate,
    ) -> SyncDeviceBlobAck:
        return self.db.upsert_device_blob_ack(acknowledgment)

    def list_device_acknowledgments(
        self,
        dataset_id: str,
        device_id: str,
    ) -> SyncDeviceAcknowledgmentSummary:
        return self.db.list_device_acknowledgments(dataset_id, device_id)

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
    ) -> list[SyncEnvelope]:
        """Insert one complete validated group or return its exact stored replay."""

        return self.db.insert_envelopes_atomic(
            envelopes,
            trusted_notes_organization_bootstrap_id=trusted_notes_organization_bootstrap_id,
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
        status: str | Sequence[str] | None = None,
        exclude_device_id: str | None = None,
    ) -> list[SyncEnvelope]:
        return self.db.list_envelopes_after(
            dataset_id,
            since_sequence,
            limit=limit,
            domains=domains,
            status=status,
            exclude_device_id=exclude_device_id,
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
        return self.db.upsert_object_state(state, connection=self._connection)

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

    def mark_bootstrap_envelope_verified(
        self,
        server_cursor: int,
        *,
        bootstrap_id: str,
    ) -> SyncEnvelope:
        """Record a verified bootstrap step as applied without product replay."""

        return self.db.mark_bootstrap_envelope_verified(
            server_cursor,
            bootstrap_id=bootstrap_id,
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
    ) -> SyncDeviceCursor | None:
        return self.db.get_device_cursor(dataset_id, device_id, domain)

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
        return self.db.complete_blob_upload(blob)

    def get_blob_object(
        self,
        dataset_id: str,
        *,
        attachment_id: str | None = None,
        blob_id: str | None = None,
        payload_hash: str | None = None,
        owner_user_id: str | None = None,
    ) -> SyncBlobObject | None:
        return self.db.get_blob_object(
            dataset_id,
            attachment_id=attachment_id,
            blob_id=blob_id,
            payload_hash=payload_hash,
            owner_user_id=owner_user_id,
        )

    def list_blob_objects_for_dataset(
        self,
        dataset_id: str,
        *,
        status: str | None = "available",
    ) -> list[SyncBlobObject]:
        """Return committed blob metadata for retention and diagnostics scans."""

        return self.db.list_blob_objects_for_dataset(dataset_id, status=status)

    def mark_blob_object_deleted(
        self,
        dataset_id: str,
        blob_id: str,
    ) -> SyncBlobObject | None:
        return self.db.mark_blob_object_deleted(dataset_id, blob_id)

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
