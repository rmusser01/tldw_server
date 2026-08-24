from __future__ import annotations

"""Helpers for Sync v2 adapter head-version conflict checks."""

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from ..adapters import AdapterConflict, SyncAdapterContext, SyncHead
from ..models import SyncEnvelopeCreate

SyncHeadT = TypeVar("SyncHeadT", bound=SyncHead)


def prior_envelopes(
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext | None,
) -> list[SyncHead]:
    """Return accepted prior envelopes excluding an idempotent copy of the incoming one."""

    return [
        item
        for item in (context.prior_envelopes if context is not None else [])
        if item.client_envelope_id != envelope.client_envelope_id
    ]


def current_head(prior: Iterable[SyncHeadT]) -> SyncHeadT | None:
    """Return the latest planned head or highest-sequence stored head."""

    indexed = list(enumerate(prior))
    if not indexed:
        return None
    return max(
        indexed,
        key=lambda pair: (
            pair[1].server_sequence is None,
            pair[0] if pair[1].server_sequence is None else pair[1].server_sequence,
        ),
    )[1]


def incoming_references_head(
    envelope: SyncEnvelopeCreate,
    head: SyncHead | None,
) -> bool:
    """Return whether the incoming envelope is explicitly based on the current head."""

    if head is None:
        return True
    if envelope.base_server_cursor is not None and _same_token(
        envelope.base_server_cursor,
        head.server_cursor,
    ):
        return True
    if (
        envelope.base_object_revision is not None
        and envelope.base_object_hash is not None
        and _same_token(envelope.base_object_revision, head.object_revision)
        and _same_token(envelope.base_object_hash, head.payload_hash)
    ):
        return True
    if envelope.base_version is not None and _same_token(envelope.base_version, head.entity_version):
        return True
    return any(_dependency_references_head(dependency, head) for dependency in envelope.dependencies)


def incoming_references_exact_head(
    envelope: SyncEnvelopeCreate,
    head: SyncHead,
) -> bool:
    """Return whether every canonical base token matches the current head."""

    return bool(
        _same_optional_token(envelope.base_server_cursor, head.server_cursor)
        and _same_optional_token(
            envelope.base_object_revision,
            head.object_revision,
        )
        and _same_optional_token(envelope.base_object_hash, head.payload_hash)
    )


def delete_update_conflict(
    envelope: SyncEnvelopeCreate,
    prior: list[SyncHead],
    *,
    is_delete: Callable[[SyncHead], bool],
    conflict_factory: Callable[[SyncEnvelopeCreate], AdapterConflict],
) -> AdapterConflict | None:
    """Conflict delete-vs-update only when the incoming change is not based on head."""

    head = current_head(prior)
    if head is None or is_delete(envelope) == is_delete(head):
        return None
    if incoming_references_head(envelope, head):
        return None
    return conflict_factory(envelope)


def _dependency_references_head(dependency: dict[str, Any], head: SyncHead) -> bool:
    if _same_token(dependency.get("client_envelope_id"), head.client_envelope_id):
        return True
    if _same_token(dependency.get("envelope_id"), head.client_envelope_id):
        return True
    if _same_token(dependency.get("base_envelope_id"), head.client_envelope_id):
        return True
    if _same_token(dependency.get("server_sequence"), head.server_sequence):
        return True
    if not _dependency_matches_entity(dependency, head):
        return False
    return (
        _same_token(dependency.get("entity_version"), head.entity_version)
        or _same_token(dependency.get("base_version"), head.entity_version)
        or _same_token(dependency.get("version"), head.entity_version)
    )


def _dependency_matches_entity(dependency: dict[str, Any], head: SyncHead) -> bool:
    entity_id = dependency.get("entity_id")
    stable_key = dependency.get("stable_key")
    if entity_id is not None and not _same_token(entity_id, head.entity_id):
        return False
    if stable_key is not None and not _same_token(stable_key, head.stable_key):
        return False
    return entity_id is not None or stable_key is not None


def _same_token(left: object, right: object) -> bool:
    if left is None or right is None:
        return False
    return str(left) == str(right)


def _same_optional_token(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    return str(left) == str(right)


__all__ = [
    "current_head",
    "delete_update_conflict",
    "incoming_references_head",
    "incoming_references_exact_head",
    "prior_envelopes",
]
