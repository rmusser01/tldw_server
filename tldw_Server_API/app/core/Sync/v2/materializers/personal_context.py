from __future__ import annotations

"""Materialize accepted Personal Context envelopes through the owner service."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from tldw_profile_core import ProfileManifest, ProfileProposal, ProfileRecord, ProfileScope

from tldw_Server_API.app.core.Personalization.personal_context_service import (
    ProfileConflictError,
)

from ..models import SyncEnvelope, SyncObjectState
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
        try:
            value = _parse_value(self.domain, envelope.payload or envelope.payload_clear)
            current_state = store.get_object_state(
                envelope.dataset_id,
                envelope.domain,
                envelope.object_id,
            )
            service = self.service_resolver(str(dataset.owner_user_id))
            if service is None:
                raise RuntimeError("service unavailable")
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
        object_revision = envelope.object_revision
        if object_revision is None:
            object_revision = (
                1 if current_state is None else current_state.object_revision + 1
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


def _parse_value(domain: str, payload: Mapping[str, Any]) -> Any:
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


__all__ = ["PersonalContextMaterializer"]
