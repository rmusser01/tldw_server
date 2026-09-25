from __future__ import annotations

"""Project a metadata-only Sync domain into restoreable object state.

``media_metadata.py`` and ``source_cache.py`` were the same 187 lines: every difference
was a docstring, an error-code prefix, or a message string. The five projection rules
they each stated -- tombstoned objects may not be resurrected by upsert, a reused stable
object ID with a different payload hash is a conflict rather than an overwrite,
``payload_hash`` is mandatory, object revision advances by one, and the tombstone path
preserves the prior object hash -- now live here once.

The domain-scoped error codes, metadata keys and messages are client-visible, so they are
parameterised rather than unified: every string this module emits is byte-identical to
what the two originals emitted.
"""

from dataclasses import dataclass

from ..models import SyncDomain, SyncEnvelope, SyncObjectState
from ..store import SyncV2Store
from .base import MaterializationResult


@dataclass(slots=True)
class MetadataOnlyMaterializer:
    """Record accepted metadata-only envelopes as Sync object state.

    Args:
        domain: the Sync domain this instance serves.
        code_prefix: prefix for error codes and the conflict metadata key,
            e.g. ``media_metadata`` -> ``media_metadata_projection_failed``.
        label: sentence-initial name used in messages, e.g. ``Media metadata``.
        lower_label: mid-sentence form, e.g. ``media metadata`` in
            "Unsupported media metadata operation".
        noun: what a stored item is called, ``object`` or ``entry``.
    """

    domain: SyncDomain
    code_prefix: str
    label: str
    lower_label: str
    noun: str

    def apply(
        self,
        envelope: SyncEnvelope,
        *,
        store: SyncV2Store,
    ) -> MaterializationResult:
        """Project one accepted envelope into object state."""

        if envelope.domain != self.domain:
            return MaterializationResult(status="skipped")

        projection_failed = f"{self.code_prefix}_projection_failed"

        # Defensive: SyncEnvelope.__post_init__ rejects a missing cursor, so this is only
        # reachable for an envelope built bypassing that validation (e.g. from a raw row).
        if envelope.server_cursor is None:
            return MaterializationResult(
                status="failed",
                error_code=projection_failed,
                message="Stored Sync envelope is missing a server cursor",
            )

        if not envelope.payload_hash:
            requires_hash = f"{self.label} envelopes require payload_hash"
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code=projection_failed,
                apply_error_message=requires_hash,
            )
            return MaterializationResult(
                status="failed",
                error_code=projection_failed,
                message=requires_hash,
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
                        apply_error_code=f"{self.code_prefix}_tombstoned",
                        apply_error_message=(
                            f"{self.label} upsert cannot resurrect a tombstoned {self.noun}"
                        ),
                    )
                    return self._tombstone_conflict_result(envelope, current_state=current_state)
                if current_state.object_hash == envelope.payload_hash:
                    store.mark_envelope_apply_status(
                        envelope.server_cursor,
                        apply_status="applied",
                    )
                    return MaterializationResult(status="applied")
                store.mark_envelope_apply_status(
                    envelope.server_cursor,
                    apply_status="conflict",
                    apply_error_code=f"{self.code_prefix}_hash_mismatch",
                    apply_error_message=(
                        f"{self.label} stable object ID was reused with a "
                        "different payload hash"
                    ),
                )
                return self._hash_conflict_result(envelope, current_state=current_state)

            _record_state(
                store=store,
                envelope=envelope,
                object_revision=_next_object_revision(envelope, current_state),
                object_hash=envelope.payload_hash,
                deleted=False,
            )
            return MaterializationResult(status="applied")

        if envelope.operation == "tombstone":
            object_hash = (
                current_state.object_hash if current_state is not None else envelope.payload_hash
            )
            _record_state(
                store=store,
                envelope=envelope,
                object_revision=_next_object_revision(envelope, current_state),
                object_hash=object_hash,
                deleted=True,
            )
            return MaterializationResult(status="applied")

        unsupported = f"Unsupported {self.lower_label} operation: {envelope.operation}"
        store.mark_envelope_apply_status(
            envelope.server_cursor,
            apply_status="failed",
            apply_error_code=projection_failed,
            apply_error_message=unsupported,
        )
        return MaterializationResult(
            status="failed",
            error_code=projection_failed,
            message=unsupported,
        )

    def _conflict_metadata(
        self,
        envelope: SyncEnvelope,
        current_state: SyncObjectState,
    ) -> dict[str, object]:
        return {
            f"{self.code_prefix}_object_id": envelope.object_id,
            "incoming_payload_hash": envelope.payload_hash or "",
            "server_object_hash": current_state.object_hash,
            "server_object_revision": current_state.object_revision,
            "server_cursor": current_state.latest_server_cursor,
            "server_deleted": current_state.deleted,
        }

    def _hash_conflict_result(
        self,
        envelope: SyncEnvelope,
        *,
        current_state: SyncObjectState,
    ) -> MaterializationResult:
        return MaterializationResult(
            status="conflict",
            conflict_type=f"{self.code_prefix}_hash_mismatch",
            message=f"{self.label} stable object ID was reused with a different payload hash",
            metadata=self._conflict_metadata(envelope, current_state),
        )

    def _tombstone_conflict_result(
        self,
        envelope: SyncEnvelope,
        *,
        current_state: SyncObjectState,
    ) -> MaterializationResult:
        return MaterializationResult(
            status="conflict",
            conflict_type=f"{self.code_prefix}_tombstoned",
            message=f"{self.label} upsert cannot resurrect a tombstoned {self.noun}",
            metadata=self._conflict_metadata(envelope, current_state),
        )


def _record_state(
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


__all__ = ["MetadataOnlyMaterializer"]
