from __future__ import annotations

"""Materialize accepted Personal Context envelopes through the owner service."""

import hashlib
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from tldw_profile_core import ProfileManifest, ProfileProposal, ProfileRecord, ProfileScope, canonical_bytes

from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    CanonicalApplyReceipt,
    IngressIdentity,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    ProfileConflictError,
)

from ..models import (
    SyncEnvelope,
    SyncObjectState,
    resolve_personal_context_ingress_result_revision,
)
from .base import MaterializationResult

_MODELS = {
    "personal_context.manifest": ProfileManifest,
    "personal_context.scope": ProfileScope,
    "personal_context.record": ProfileRecord,
    "personal_context.proposal": ProfileProposal,
}

ServiceResolver = Callable[[str], Any]


@dataclass(slots=True)
class PersonalContextMaterializer:
    """Apply one Personal Context whole object through a user-bound service."""

    domain: str
    service_resolver: ServiceResolver

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: Any,
        guarded_mutation: Any = None,
    ) -> MaterializationResult:
        """Materialize or return a stable content-free failure."""

        del guarded_mutation
        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code="personal_context_projection_failed",
                message="Stored envelope has no server cursor",
            )
        dataset = store.get_dataset(envelope.dataset_id)
        if dataset is None or not str(getattr(dataset, "owner_user_id", "")).strip():
            return self._fail(
                envelope,
                store,
                "personal_context_authorization_unavailable",
            )
        ingress_receipt_applied = False
        try:
            value = _parse_value(self.domain, envelope.payload or envelope.payload_clear)
            current_state = store.get_object_state(
                envelope.dataset_id,
                envelope.domain,
                envelope.object_id,
            )
            object_revision = resolve_personal_context_ingress_result_revision(
                object_revision=envelope.object_revision,
                base_server_cursor=envelope.base_server_cursor,
                base_object_revision=envelope.base_object_revision,
                base_object_hash=envelope.base_object_hash,
                base_version=envelope.base_version,
            )
            if object_revision is None:
                raise ValueError("Personal Context ingress lineage is invalid")
            if envelope.object_revision is None and not _predecessor_matches(
                envelope,
                current_state,
                result_revision=object_revision,
            ):
                raise ProfileConflictError("Personal Context predecessor changed")
            service = self.service_resolver(str(dataset.owner_user_id))
            if service is None:
                raise RuntimeError("service unavailable")
            authority = envelope.authority
            if authority is not None and authority.role == "home_authority":
                return MaterializationResult(status="skipped")
            if authority is not None and authority.role == "client_ingress":
                purge_generation = _purge_generation(dataset)
                identity = IngressIdentity(
                        dataset_id=envelope.dataset_id,
                        device_id=str(envelope.device_id or ""),
                        client_envelope_id=envelope.client_envelope_id,
                        canonical_payload_digest=(
                            "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()
                        ),
                        purge_generation=purge_generation,
                        wire_entity_version=str(envelope.entity_version),
                    )
                receipt = service.apply_sync_ingress(
                    identity=identity,
                    domain=self.domain,
                    value=value,
                    base_object_hash=envelope.base_object_hash,
                )
                if not _valid_ingress_receipt(
                    receipt,
                    envelope=envelope,
                    identity=identity,
                    purge_generation=purge_generation,
                ):
                    raise ValueError("Personal Context ingress receipt is invalid")
                store.mark_personal_context_ingress_applied(
                    server_cursor=envelope.server_cursor,
                    receipt=receipt,
                )
                ingress_receipt_applied = True
            else:
                service.apply_sync_object(
                    domain=self.domain,
                    value=value,
                    actor_type="sync",
                    actor_id=envelope.device_id,
                    base_object_hash=envelope.base_object_hash,
                )
        except ProfileConflictError:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="conflict",
                apply_error_code="personal_context_base_conflict",
                apply_error_message="Personal Context base state changed",
            )
            return MaterializationResult(
                status="conflict",
                conflict_type="personal_context_base_conflict",
                message="Personal Context base state changed",
            )
        except (ValidationError, ValueError, TypeError):
            return self._fail(
                envelope,
                store,
                "personal_context_payload_invalid",
            )
        except Exception:  # noqa: BLE001 - projection boundary must fail closed
            return self._fail(
                envelope,
                store,
                "personal_context_projection_failed",
            )
        store.upsert_object_state(
            SyncObjectState(
                dataset_id=envelope.dataset_id,
                domain=envelope.domain,
                object_id=envelope.object_id,
                object_revision=object_revision,
                object_hash=envelope.payload_hash,
                latest_server_cursor=envelope.server_cursor,
                deleted=envelope.operation == "tombstone",
            )
        )
        if not ingress_receipt_applied:
            store.mark_envelope_apply_status(envelope.server_cursor, apply_status="applied")
        return MaterializationResult(status="applied")

    @staticmethod
    def _fail(
        envelope: SyncEnvelope,
        store: Any,
        error_code: str,
    ) -> MaterializationResult:
        if envelope.server_cursor is not None:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code=error_code,
                apply_error_message="Personal Context projection failed",
            )
        return MaterializationResult(
            status="failed",
            error_code=error_code,
            message="Personal Context projection failed",
        )


def _predecessor_matches(
    envelope: SyncEnvelope,
    state: SyncObjectState | None,
    *,
    result_revision: int,
) -> bool:
    """Verify mutable projection only as the referenced predecessor snapshot."""

    if result_revision == 1:
        return state is None
    return bool(
        state is not None
        and state.dataset_id == envelope.dataset_id
        and state.domain == envelope.domain
        and state.object_id == envelope.object_id
        and state.latest_server_cursor == envelope.base_server_cursor
        and state.object_revision == envelope.base_object_revision
        and state.object_hash == envelope.base_object_hash
        and state.deleted is False
    )


def _parse_value(domain: str, payload: Mapping[str, Any]) -> Any:
    """Parse a materializable canonical payload for the selected Sync domain."""

    model = _MODELS.get(domain)
    if model is not None:
        return model.model_validate(payload)
    if domain == "personal_context.purge" and set(payload) == {
        "schema_version",
        "profile_id",
        "purge_generation",
    }:
        return dict(payload)
    raise ValueError("Unsupported Personal Context materializer payload")


def _purge_generation(dataset: Any) -> int:
    """Read only the content-free profile purge generation from enrollment."""

    metadata = getattr(dataset, "metadata", {})
    personal_context = metadata.get("personal_context") if isinstance(metadata, Mapping) else None
    value = personal_context.get("purge_generation") if isinstance(personal_context, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Personal Context purge generation is unavailable")
    return value


def _valid_ingress_receipt(
    receipt: object,
    *,
    envelope: SyncEnvelope,
    identity: IngressIdentity,
    purge_generation: int,
) -> bool:
    """Bind a canonical receipt to this exact encrypted ingress envelope."""

    if not isinstance(receipt, CanonicalApplyReceipt):
        return False
    expected_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "tldw:personal-context:ingress:"
            f"{envelope.dataset_id}:{envelope.device_id or ''}:"
            f"{envelope.client_envelope_id}",
        )
    )
    return (
        receipt.receipt_id == expected_id
        and receipt.dataset_id == identity.dataset_id
        and receipt.device_id == identity.device_id
        and receipt.client_envelope_id == identity.client_envelope_id
        and receipt.canonical_payload_digest == identity.canonical_payload_digest
        and receipt.purge_generation == purge_generation
        and receipt.resulting_object_id == envelope.object_id
        and receipt.wire_entity_version == str(envelope.entity_version)
        and bool(receipt.publication_batch_id)
        and receipt.profile_publication_sequence > 0
        and receipt.manifest_revision >= 0
        and bool(receipt.manifest_version_id)
    )


__all__ = ["PersonalContextMaterializer"]
