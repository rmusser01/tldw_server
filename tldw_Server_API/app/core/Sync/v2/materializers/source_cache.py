from __future__ import annotations

"""Materialize Sync v2 source-cache entries into restoreable object state."""

from dataclasses import dataclass

from ..models import SyncDomain, SyncEnvelope, SyncObjectState
from ..store import SyncV2Store
from .base import MaterializationResult


@dataclass(slots=True)
class SourceCacheMaterializer:
    """Record accepted `source_cache.entry` envelopes as Sync object state."""

    domain: SyncDomain = "source_cache.entry"

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        """Project one source-cache envelope into normal Sync object state."""

        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code="source_cache_projection_failed",
                message="Stored Sync envelope is missing a server cursor",
            )
        if not envelope.payload_hash:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code="source_cache_projection_failed",
                apply_error_message="source_cache.entry envelopes require payload_hash",
            )
            return MaterializationResult(
                status="failed",
                error_code="source_cache_projection_failed",
                message="source_cache.entry envelopes require payload_hash",
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
                        apply_error_code="source_cache_tombstoned",
                        apply_error_message=(
                            "source_cache.entry upsert cannot resurrect a tombstoned entry"
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
                    apply_error_code="source_cache_hash_mismatch",
                    apply_error_message=(
                        "source_cache.entry stable object ID was reused with a "
                        "different payload hash"
                    ),
                )
                return _hash_conflict_result(envelope, current_state=current_state)

            _record_source_cache_state(
                store=store,
                envelope=envelope,
                object_revision=_next_object_revision(envelope, current_state),
                object_hash=envelope.payload_hash,
                deleted=False,
            )
            return MaterializationResult(status="applied")

        if envelope.operation == "tombstone":
            object_hash = current_state.object_hash if current_state is not None else envelope.payload_hash
            _record_source_cache_state(
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
            apply_error_code="source_cache_projection_failed",
            apply_error_message=f"Unsupported source_cache.entry operation: {envelope.operation}",
        )
        return MaterializationResult(
            status="failed",
            error_code="source_cache_projection_failed",
            message=f"Unsupported source_cache.entry operation: {envelope.operation}",
        )


def _record_source_cache_state(
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
        conflict_type="source_cache_hash_mismatch",
        message="source_cache.entry stable object ID was reused with a different payload hash",
        metadata={
            "source_cache_object_id": envelope.object_id,
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
        conflict_type="source_cache_tombstoned",
        message="source_cache.entry upsert cannot resurrect a tombstoned entry",
        metadata={
            "source_cache_object_id": envelope.object_id,
            "incoming_payload_hash": envelope.payload_hash or "",
            "server_object_hash": current_state.object_hash,
            "server_object_revision": current_state.object_revision,
            "server_cursor": current_state.latest_server_cursor,
            "server_deleted": current_state.deleted,
        },
    )


__all__ = ["SourceCacheMaterializer"]
