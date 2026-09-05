"""Journaled Personal Context candidate delivery and explicit batched choices."""

from __future__ import annotations

import hashlib
import hmac
import uuid
from dataclasses import asdict
from typing import Any

from tldw_profile_core.canonical import canonical_json_bytes

from .adapters import AdapterAccepted
from .errors import SyncStoreError
from .models import SyncConflictCreate, SyncEnvelopeCreate, normalize_sync_timestamp


class PersonalContextConflictService:
    """Bridge Sync transport decisions to user-bound canonical authority."""

    def __init__(self, sync: Any, store: Any) -> None:
        self.sync = sync
        self.store = store

    @staticmethod
    def conflict_id(dataset_id: str, device_id: str, envelope_id: str) -> str:
        return str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"tldw:personal-context:conflict:{dataset_id}:{device_id}:{envelope_id}")
        )

    def ensure_authority_candidate(self, *, dataset: Any, source: Any, exchange: Any) -> Any:
        """Attach an authenticated immutable candidate before a terminal response."""
        canonical = self.sync._personal_context_service_for_user(dataset.owner_user_id)
        clear = self.sync._restore_personal_context_from_storage(dataset, source)
        state = dataset.metadata["personal_context"]
        if canonical.get_manifest().profile_id != state["profile_id"]:
            raise SyncStoreError("Personal Context authority binding changed")
        conflict_id = self.conflict_id(dataset.dataset_id, str(source.device_id), source.client_envelope_id)
        canonical.capture_sync_conflict(
            conflict_id=conflict_id,
            dataset_id=dataset.dataset_id,
            device_id=source.device_id,
            local_envelope_id=source.client_envelope_id,
            domain=source.domain,
            object_id=source.object_id,
            local_payload=clear.payload,
            local_envelope_digest=self._local_envelope_digest(clear),
            purge_generation=state["purge_generation"],
            exchange=exchange,
        )
        with canonical.sync_conflict_staging_guard(
            conflict_id,
            dataset_id=dataset.dataset_id,
            device_id=source.device_id,
            purge_generation=state["purge_generation"],
            exchange=exchange,
        ) as journal:
            return self._attach_candidate(dataset, source, canonical, journal)

    def _candidate_envelope(self, dataset: Any, canonical: Any, journal: Any) -> SyncEnvelopeCreate:
        """Reconstruct every immutable review field from canonical custody."""
        state = dataset.metadata["personal_context"]
        key_id, key = canonical.sync_integrity_key(journal["profile_id"])
        if (
            dataset.dataset_id != journal["dataset_id"]
            or state["profile_id"] != journal["profile_id"]
            or state["purge_generation"] != journal["purge_generation"]
            or key_id != journal["integrity_key_id"]
        ):
            raise SyncStoreError("Personal Context conflict authority changed")
        payload = journal["candidate"]
        encoded = canonical_json_bytes(payload)
        version = journal["candidate_version_id"]
        if journal["domain"] == "personal_context.proposal":
            version = "sync-proposal-sha256:" + hashlib.sha256(encoded).hexdigest()
        return SyncEnvelopeCreate(
            dataset_id=journal["dataset_id"],
            device_id="server-origin",
            client_envelope_id=journal["remote_envelope_id"],
            domain=journal["domain"],
            object_id=journal["candidate_object_id"],
            entity_version=version,
            parent_id=payload.get("profile_id")
            if journal["domain"] == "personal_context.scope"
            else payload.get("scope_id"),
            operation="tombstone" if payload.get("state") == "deleted" else "upsert",
            payload=payload,
            payload_size_bytes=len(encoded),
            payload_hash="hmac-sha256-v1:" + hmac.new(key, encoded, hashlib.sha256).hexdigest(),
            created_at_client=journal["candidate_created_at"],
            received_at_server=normalize_sync_timestamp(journal["candidate_created_at"]),
            # Candidates are delivered only by conflict review. They must never
            # advance current heads or masquerade as a new publication batch.
            status="conflict",
            apply_status="applied",
            routing_metadata={
                "profile_id": journal["profile_id"],
                "purge_generation": journal["purge_generation"],
                "integrity_key_id": key_id,
                "personal_context_conflict_candidate": journal["conflict_id"],
                "personal_context_authority": journal["authority"],
            },
        )

    def _validate_candidate(self, dataset: Any, candidate: Any, expected: SyncEnvelopeCreate) -> Any:
        clear_candidate = self.sync._restore_personal_context_from_storage(dataset, candidate)
        # Cursors and their derived server envelope ID belong to Sync allocation;
        # every other returned field is fixed by the encrypted canonical journal.
        immutable = asdict(expected)
        immutable.pop("server_cursor")
        immutable.pop("server_sequence")
        if any(getattr(clear_candidate, field) != value for field, value in immutable.items()):
            raise SyncStoreError("Personal Context conflict candidate authentication failed")
        return clear_candidate

    @staticmethod
    def _local_envelope_digest(source: Any) -> str:
        immutable = asdict(source)
        for field in (
            "server_cursor",
            "server_sequence",
            "envelope_id",
            "received_at_server",
            "server_timestamp",
            "apply_status",
            "apply_error_code",
            "apply_error_message",
            "applied_at",
        ):
            immutable.pop(field, None)
        return "sha256:" + hashlib.sha256(canonical_json_bytes(immutable)).hexdigest()

    def _attach_candidate(self, dataset: Any, source: Any, canonical: Any, journal: Any) -> Any:
        from .service import SyncPushConflict

        conflict_id = journal["conflict_id"]
        envelope = self._candidate_envelope(dataset, canonical, journal)
        candidate = self.store.get_envelope_by_client_id(dataset.dataset_id, journal["remote_envelope_id"])
        if candidate is None:
            candidate = self.store.insert_envelope(self.sync._protect_personal_context_for_storage(dataset, envelope))
        clear_candidate = self._validate_candidate(dataset, candidate, envelope)
        conflict = self.store.insert_conflict(
            SyncConflictCreate(
                conflict_id=conflict_id,
                dataset_id=dataset.dataset_id,
                domain=source.domain,
                object_id=source.object_id,
                conflict_type="personal_context_key_collision"
                if journal["key_slot"] is not None
                else "personal_context_base_conflict",
                local_envelope_id=source.client_envelope_id,
                remote_envelope_id=candidate.client_envelope_id,
                server_cursor=source.server_cursor,
            )
        )
        return SyncPushConflict(
            conflict_id=conflict.conflict_id,
            client_envelope_id=source.client_envelope_id,
            domain=source.domain,
            entity_id=source.object_id,
            server_sequence=source.server_sequence,
            message="Personal Context requires review",
            expected_local_envelope_id=source.client_envelope_id,
            expected_remote_envelope_id=candidate.client_envelope_id,
            authority_candidate=clear_candidate,
        )

    def resolve_batch_item(
        self,
        *,
        user_id: str,
        dataset: Any,
        device_id: str,
        conflict: Any,
        action: str,
        resolution_envelope: Any,
        expected_local_envelope_id: str,
        expected_remote_envelope_id: str,
        idempotency_key: str,
        exchange: Any,
    ) -> Any:
        """Validate transport identity and finalize only after the canonical receipt."""
        self.sync._require_registered_device(user_id, device_id, store=self.store)
        if (
            conflict.local_envelope_id != expected_local_envelope_id
            or conflict.remote_envelope_id != expected_remote_envelope_id
            or not idempotency_key
            or action not in {"skip", "overwrite", "duplicate_rename"}
            or (action == "skip") != (resolution_envelope is None)
        ):
            raise SyncStoreError("Personal Context conflict review is stale")
        source = self.store.get_envelope_by_server_cursor(conflict.server_sequence)
        remote = self.store.get_envelope_by_client_id(dataset.dataset_id, expected_remote_envelope_id)
        if (
            source is None
            or remote is None
            or source.dataset_id != dataset.dataset_id
            or source.device_id != device_id
            or source.client_envelope_id != expected_local_envelope_id
            or source.object_id != conflict.object_id
            or source.domain != conflict.domain
        ):
            raise SyncStoreError("Personal Context conflict candidate is unavailable")
        canonical = self.sync._personal_context_service_for_user(user_id)
        journal = canonical.get_sync_conflict(conflict.conflict_id)
        self._validate_candidate(dataset, remote, self._candidate_envelope(dataset, canonical, journal))
        local = self.sync._restore_personal_context_from_storage(dataset, source)
        local_identity = {
            "dataset_id": local.dataset_id,
            "device_id": local.device_id,
            "local_envelope_id": local.client_envelope_id,
            "domain": local.domain,
            "object_id": local.object_id,
        }
        if (
            any(journal[field] != value for field, value in local_identity.items())
            or journal["local_digest"] != "sha256:" + hashlib.sha256(canonical_json_bytes(local.payload)).hexdigest()
            or journal["local_envelope_digest"] != self._local_envelope_digest(local)
        ):
            raise SyncStoreError("Personal Context local candidate authentication failed")
        command = None
        if resolution_envelope is not None:
            if (
                resolution_envelope.dataset_id != dataset.dataset_id
                or resolution_envelope.device_id != device_id
                or resolution_envelope.domain != conflict.domain
                or set(resolution_envelope.routing_metadata) - {"profile_id", "purge_generation", "integrity_key_id"}
                or resolution_envelope.routing_metadata.get("purge_generation") != journal["purge_generation"]
                or self.sync._payload_exceeds_size_limit(resolution_envelope)
            ):
                raise SyncStoreError("Personal Context resolution envelope identity is invalid")
            adapter = self.sync.adapters.get(conflict.domain)
            # Canonical storage checks exact current heads inside its write transaction.
            if not isinstance(adapter.evaluate_envelope(resolution_envelope, dataset=dataset), AdapterAccepted):
                raise SyncStoreError("Personal Context resolution envelope is invalid")
            command = asdict(resolution_envelope)
        receipt = canonical.resolve_sync_conflict(
            conflict_id=conflict.conflict_id,
            dataset_id=dataset.dataset_id,
            device_id=device_id,
            expected_local_envelope_id=expected_local_envelope_id,
            expected_remote_envelope_id=expected_remote_envelope_id,
            idempotency_key=idempotency_key,
            action=action,
            command=command,
            purge_generation=dataset.metadata["personal_context"]["purge_generation"],
            exchange=exchange,
        )
        if conflict.status == "unresolved":
            self.store.claim_conflict_resolution(
                conflict.conflict_id,
                dataset_id=dataset.dataset_id,
                resolved_by_device_id=device_id,
                resolution_action=action,
                resolution_notes=None,
            )
            self.store.terminalize_claimed_conflict_envelope(
                conflict.conflict_id,
                dataset_id=dataset.dataset_id,
                resolved_by_device_id=device_id,
                resolution_action=action,
                resolution_notes=None,
                apply_error_code="personal_context_conflict_resolved",
            )
        return self.store.resolve_conflict(
            conflict.conflict_id,
            dataset_id=dataset.dataset_id,
            status="dismissed" if action == "skip" else "resolved",
            resolved_by_device_id=device_id,
            resolution_action=action,
            resolved_by_envelope_id=receipt.get("receipt_id"),
        )
