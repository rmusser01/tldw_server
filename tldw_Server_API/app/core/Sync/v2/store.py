from __future__ import annotations

"""Core-facing Sync v2 store facade."""

from collections.abc import Sequence

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase

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
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncKeyRecord,
    SyncKeyRecordCreate,
    SyncObjectState,
    SyncRestoreManifestStats,
)


class SyncV2Store:
    """Core Sync v2 persistence interface backed by DB_Management."""

    def __init__(self, db: SyncDatabase) -> None:
        self.db = db

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

    def insert_envelope(self, envelope: SyncEnvelopeCreate) -> SyncEnvelope:
        return self.db.insert_envelope(envelope)

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
        )

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

    def get_object_state(
        self,
        dataset_id: str,
        domain: SyncDomain,
        object_id: str,
    ) -> SyncObjectState | None:
        return self.db.get_object_state(dataset_id, domain, object_id)

    def upsert_object_state(self, state: SyncObjectState) -> SyncObjectState:
        return self.db.upsert_object_state(state)

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
        return self.db.insert_conflict(conflict)

    def list_conflicts(
        self,
        dataset_id: str,
        *,
        status: ConflictStatus | None = None,
    ) -> list[SyncConflict]:
        return self.db.list_conflicts(dataset_id, status=status)

    def get_conflict(self, conflict_id: str) -> SyncConflict | None:
        return self.db.get_conflict(conflict_id)

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
