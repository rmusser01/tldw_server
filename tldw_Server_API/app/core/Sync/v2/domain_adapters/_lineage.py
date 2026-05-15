from __future__ import annotations

"""Helpers for Sync v2 adapter head-version conflict checks."""

from collections.abc import Callable, Iterable
from typing import Any

from ..adapters import AdapterConflict, SyncAdapterContext
from ..models import SyncEnvelope, SyncEnvelopeCreate


def prior_envelopes(
    envelope: SyncEnvelopeCreate,
    context: SyncAdapterContext | None,
) -> list[SyncEnvelope]:
    """Return accepted prior envelopes excluding an idempotent copy of the incoming one."""

    return [
        item
        for item in (context.prior_envelopes if context is not None else [])
        if item.client_envelope_id != envelope.client_envelope_id
    ]


def current_head(prior: Iterable[SyncEnvelope]) -> SyncEnvelope | None:
    """Return the highest-sequence accepted envelope for the entity identity."""

    return max(prior, key=lambda item: item.server_sequence, default=None)


def incoming_references_head(
    envelope: SyncEnvelopeCreate,
    head: SyncEnvelope | None,
) -> bool:
    """Return whether the incoming envelope is explicitly based on the current head."""

    if head is None:
        return True
    if envelope.base_version is not None and _same_token(envelope.base_version, head.entity_version):
        return True
    return any(_dependency_references_head(dependency, head) for dependency in envelope.dependencies)


def delete_update_conflict(
    envelope: SyncEnvelopeCreate,
    prior: list[SyncEnvelope],
    *,
    is_delete: Callable[[SyncEnvelope | SyncEnvelopeCreate], bool],
    conflict_factory: Callable[[SyncEnvelopeCreate], AdapterConflict],
) -> AdapterConflict | None:
    """Conflict delete-vs-update only when the incoming change is not based on head."""

    head = current_head(prior)
    if head is None or is_delete(envelope) == is_delete(head):
        return None
    if incoming_references_head(envelope, head):
        return None
    return conflict_factory(envelope)


def _dependency_references_head(dependency: dict[str, Any], head: SyncEnvelope) -> bool:
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


def _dependency_matches_entity(dependency: dict[str, Any], head: SyncEnvelope) -> bool:
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


__all__ = [
    "current_head",
    "delete_update_conflict",
    "incoming_references_head",
    "prior_envelopes",
]
