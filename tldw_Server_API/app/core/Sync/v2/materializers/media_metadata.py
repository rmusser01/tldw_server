from __future__ import annotations

"""Materialize metadata-only media Sync domains into restoreable object state."""

from dataclasses import dataclass

from ..models import SyncDomain, SyncEnvelope, SyncObjectState
from ..store import SyncV2Store
from .base import MaterializationResult


@dataclass(slots=True)
class MediaMetadataMaterializer:
    """Record accepted media metadata envelopes as Sync object state."""

    domain: SyncDomain = "media.item"

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        """Project one accepted media metadata envelope into object state."""

        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code="media_metadata_projection_failed",
                message="Stored Sync envelope is missing a server cursor",
            )
        if not envelope.payload_hash:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code="media_metadata_projection_failed",
                apply_error_message="Media metadata envelopes require payload_hash",
            )
            return MaterializationResult(
                status="failed",
                error_code="media_metadata_projection_failed",
                message="Media metadata envelopes require payload_hash",
            )

        current_state = store.get_object_state(
            envelope.dataset_id,
            envelope.domain,
            envelope.object_id,
        )
        if envelope.operation == "upsert":
            if current_state is not None:
                if current_state.deleted:
                    store.mark_envelope_apply_status(
                        envelope.server_cursor,
                        apply_status="conflict",
                        apply_error_code="media_metadata_tombstoned",
                        apply_error_message=(
                            "Media metadata upsert cannot resurrect a tombstoned object"
                        ),
                    )
                    return _tombstone_conflict_result(envelope, current_state=current_state)
                if current_state.object_hash == envelope.payload_hash:
                    store.mark_envelope_apply_status(
                        envelope.server_cursor,
                        apply_status="applied",
                    )
                    return MaterializationResult(status="applied")
                store.mark_envelope_apply_status(
                    envelope.server_cursor,
                    apply_status="conflict",
                    apply_error_code="media_metadata_hash_mismatch",
                    apply_error_message=(
                        "Media metadata stable object ID was reused with a "
                        "different payload hash"
                    ),
                )
                return _hash_conflict_result(envelope, current_state=current_state)

            _record_media_metadata_state(
                store=store,
                envelope=envelope,
                object_revision=_next_object_revision(envelope, current_state),
                object_hash=envelope.payload_hash,
                deleted=False,
            )
            return MaterializationResult(status="applied")

        if envelope.operation == "tombstone":
            object_hash = current_state.object_hash if current_state is not None else envelope.payload_hash
            _record_media_metadata_state(
                store=store,
                envelope=envelope,
                object_revision=_next_object_revision(envelope, current_state),
                object_hash=object_hash,
                deleted=True,
            )
            return MaterializationResult(status="applied")

        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="failed",
            apply_error_code="media_metadata_projection_failed",
            apply_error_message=f"Unsupported media metadata operation: {envelope.operation}",
        )
        return MaterializationResult(
            status="failed",
            error_code="media_metadata_projection_failed",
            message=f"Unsupported media metadata operation: {envelope.operation}",
        )


def _record_media_metadata_state(
    *,
    store: SyncV2Store,
    envelope: SyncEnvelope,
    object_revision: int,
    object_hash: str,
    deleted: bool,
) -> None:
    store.upsert_object_state(
        SyncObjectState(
            dataset_id=envelope.dataset_id,
            domain=envelope.domain,
            object_id=envelope.object_id,
            object_revision=object_revision,
            object_hash=object_hash,
            latest_server_cursor=envelope.server_cursor or 0,
            deleted=deleted,
        )
    )
    store.mark_envelope_apply_status(envelope.server_cursor or 0, apply_status="applied")


def _next_object_revision(
    envelope: SyncEnvelope,
    current_state: SyncObjectState | None,
) -> int:
    if envelope.object_revision is not None:
        return envelope.object_revision
    if current_state is None:
        return 1
    return current_state.object_revision + 1


def _hash_conflict_result(
    envelope: SyncEnvelope,
    *,
    current_state: SyncObjectState,
) -> MaterializationResult:
    return MaterializationResult(
        status="conflict",
        conflict_type="media_metadata_hash_mismatch",
        message="Media metadata stable object ID was reused with a different payload hash",
        metadata={
            "media_metadata_object_id": envelope.object_id,
            "incoming_payload_hash": envelope.payload_hash or "",
            "server_object_hash": current_state.object_hash,
            "server_object_revision": current_state.object_revision,
            "server_cursor": current_state.latest_server_cursor,
            "server_deleted": current_state.deleted,
        },
    )


def _tombstone_conflict_result(
    envelope: SyncEnvelope,
    *,
    current_state: SyncObjectState,
) -> MaterializationResult:
    return MaterializationResult(
        status="conflict",
        conflict_type="media_metadata_tombstoned",
        message="Media metadata upsert cannot resurrect a tombstoned object",
        metadata={
            "media_metadata_object_id": envelope.object_id,
            "incoming_payload_hash": envelope.payload_hash or "",
            "server_object_hash": current_state.object_hash,
            "server_object_revision": current_state.object_revision,
            "server_cursor": current_state.latest_server_cursor,
            "server_deleted": current_state.deleted,
        },
    )


__all__ = ["MediaMetadataMaterializer"]
