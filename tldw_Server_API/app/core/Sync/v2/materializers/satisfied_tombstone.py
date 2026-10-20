from __future__ import annotations

"""The satisfied-tombstone rule for chats and notes the dataset never held (§5.4).

A tombstone asks for one end state: the object is gone. When the dataset holds
no object state for the object, that already holds, so the conversation and
notes materializers apply such a tombstone instead of rejecting it. They record
the object as tombstoned and soft-delete a copy the owner kept outside Sync.
``chat.message`` has the same rule in its own materializer (#3198), which can
tell a row with other Sync history from the row's own metadata. Chats and notes
keep no Sync metadata on the row, so here the owner's other datasets are asked.
"""

from collections.abc import Callable
from typing import Literal

from ..models import SyncDomain, SyncEnvelope
from ..store import SyncV2Store
from .base import MaterializationResult

UnheldTombstoneOutcome = Literal["absent", "deleted", "foreign"]

# Recorded on a conflict that is closed because its tombstone turned out to be satisfied.
SATISFIED_OBJECT_TOMBSTONE_NOTE = "tombstone_of_object_the_dataset_never_held"
_SERVER_ORIGIN_DEVICE_ID = "server-origin"


def unheld_tombstone_outcome(
    store: SyncV2Store,
    envelope: SyncEnvelope,
    *,
    delete_copy: Callable[[], UnheldTombstoneOutcome],
) -> UnheldTombstoneOutcome:
    """Decide what a tombstone of an object the dataset holds no state for does.

    A device's tombstone does not reach an object that another dataset of the
    owner holds Sync history for: that row was projected from history this
    dataset does not have, so the device cannot have meant it. The result is
    ``"foreign"`` and nothing changes. The server's own tombstone is exempt: it
    comes from the owner deleting that very row through the API.

    Otherwise ``delete_copy`` removes the owner's copy kept outside Sync, if any,
    and reports ``"absent"`` or ``"deleted"`` (or ``"foreign"`` for a row Sync
    cannot carry).

    Args:
        store: The guarded Sync store of the projection.
        envelope: The tombstone, for an object with no state in its dataset.
        delete_copy: The product delete of the owner's copy, owner-filtered.

    Returns:
        ``"absent"``, ``"deleted"`` or ``"foreign"``.
    """

    if envelope.device_id != _SERVER_ORIGIN_DEVICE_ID and held_by_another_dataset(
        store, dataset_id=envelope.dataset_id, domain=envelope.domain, object_id=envelope.object_id
    ):
        return "foreign"
    return delete_copy()


def satisfied_result(envelope: SyncEnvelope) -> MaterializationResult:
    """Return the applied result, asking the service to close a conflict the old rule recorded."""

    if envelope.apply_status == "conflict":
        return MaterializationResult(status="applied", metadata={"settles_conflict": SATISFIED_OBJECT_TOMBSTONE_NOTE})
    return MaterializationResult(status="applied")


def held_by_another_dataset(store: SyncV2Store, *, dataset_id: str, domain: SyncDomain, object_id: str) -> bool:
    """Return whether another live dataset of the owner holds a head or state for the object.

    Chats and notes keep no Sync metadata on the row, so this is how a row
    projected from other Sync history is told from a copy kept outside Sync.

    Args:
        store: The Sync store, inside the caller's transaction.
        dataset_id: The dataset the object has no history in.
        domain: The object's domain.
        object_id: The object's id.

    Returns:
        ``True`` when another dataset of the same owner has a current head or
        object state for the object.
    """

    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        return False
    for other in store.list_datasets_for_user(dataset.owner_user_id):
        if other.dataset_id == dataset.dataset_id or domain not in other.domains:
            continue
        if (
            store.get_current_head(other.dataset_id, domain, object_id) is not None
            or store.get_object_state(other.dataset_id, domain, object_id) is not None
        ):
            return True
    return False


__all__ = [
    "SATISFIED_OBJECT_TOMBSTONE_NOTE",
    "UnheldTombstoneOutcome",
    "held_by_another_dataset",
    "satisfied_result",
    "unheld_tombstone_outcome",
]
