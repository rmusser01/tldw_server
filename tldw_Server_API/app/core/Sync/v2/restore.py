from __future__ import annotations

"""Helpers for building Sync v2 restore preview plans."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .models import SyncDomain

WHOLE_OBJECT_RESTORE_DOMAINS: frozenset[SyncDomain] = frozenset(
    {"notes.note", "chat.conversation"}
)
OBJECT_RESTORE_DOMAINS: frozenset[SyncDomain] = frozenset(
    {
        "notes.note",
        "chat.conversation",
        "chat.message",
        "source_cache.entry",
        "media.item",
        "media.keyword",
        "media.keyword_link",
    }
)


@dataclass(frozen=True, slots=True)
class LocalRestoreInventoryItem:
    """One local object fingerprint supplied by a restoring client."""

    dataset_id: str | None
    domain: SyncDomain
    object_id: str
    object_revision: int | None
    object_hash: str | None
    deleted: bool


LocalInventoryIndex = dict[tuple[str | None, SyncDomain, str], LocalRestoreInventoryItem]


def build_local_inventory_index(
    entries: Sequence[Mapping[str, object]] | None,
) -> LocalInventoryIndex:
    """Normalize client inventory entries into lookup keys."""

    index: LocalInventoryIndex = {}
    for raw in entries or []:
        domain = _string_value(raw.get("domain"))
        object_id = _string_value(raw.get("object_id") or raw.get("entity_id"))
        if not domain or not object_id:
            continue
        if domain not in OBJECT_RESTORE_DOMAINS:
            continue
        item = LocalRestoreInventoryItem(
            dataset_id=_string_value(raw.get("dataset_id")),
            domain=cast(SyncDomain, domain),
            object_id=object_id,
            object_revision=_int_value(raw.get("object_revision") or raw.get("entity_version")),
            object_hash=_string_value(raw.get("object_hash") or raw.get("payload_hash")),
            deleted=_bool_value(raw.get("deleted", False)),
        )
        index[(item.dataset_id, item.domain, item.object_id)] = item
    return index


def find_local_inventory_item(
    index: LocalInventoryIndex,
    *,
    dataset_id: str,
    domain: SyncDomain,
    object_id: str,
) -> LocalRestoreInventoryItem | None:
    """Return dataset-specific local inventory, falling back to dataset-agnostic entries."""

    return index.get((dataset_id, domain, object_id)) or index.get((None, domain, object_id))


def local_inventory_matches(
    item: LocalRestoreInventoryItem,
    *,
    object_revision: int | None,
    object_hash: str | None,
    deleted: bool,
) -> bool:
    """Return whether a local fingerprint already matches the server object."""

    return (
        item.object_revision == object_revision
        and item.object_hash == object_hash
        and item.deleted is deleted
    )


def restore_action_for_domain(domain: SyncDomain, *, deleted: bool, local_present: bool) -> str:
    """Return the client action label for a previewed object."""

    if deleted:
        return "hide" if domain == "chat.message" else "delete"
    if local_present:
        return "noop"
    if domain == "chat.message":
        return "append"
    return "apply"


def attachment_available_locally(
    attachment_availability: Mapping[str, str] | None,
    *,
    attachment_id: str,
    payload_hash: str,
) -> bool:
    """Return whether the client reports having a blob needed by an attachment ref."""

    status = attachment_restore_status(
        attachment_availability,
        attachment_id=attachment_id,
        payload_hash=payload_hash,
    )
    return status in {"available", "present", "stored", "server", "verified", "verified_complete"}


def attachment_verified_locally(
    attachment_availability: Mapping[str, str] | None,
    *,
    attachment_id: str,
    payload_hash: str,
) -> bool:
    """Return whether the client reports a locally verified blob checksum."""

    status = attachment_restore_status(
        attachment_availability,
        attachment_id=attachment_id,
        payload_hash=payload_hash,
    )
    return status in {"verified", "verified_complete"}


def attachment_restore_status(
    attachment_availability: Mapping[str, str] | None,
    *,
    attachment_id: str,
    payload_hash: str,
) -> str | None:
    """Return the normalized client-reported restore status for an attachment blob."""

    if not attachment_availability:
        return None
    candidates = (
        attachment_availability.get(attachment_id),
        attachment_availability.get(payload_hash),
        attachment_availability.get(f"{attachment_id}:{payload_hash}"),
    )
    for value in candidates:
        if value is not None:
            normalized = str(value).strip().lower()
            if normalized:
                return normalized
    return None


def _string_value(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_value(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "deleted"}
    return bool(value)


__all__ = [
    "LocalRestoreInventoryItem",
    "OBJECT_RESTORE_DOMAINS",
    "WHOLE_OBJECT_RESTORE_DOMAINS",
    "attachment_available_locally",
    "attachment_restore_status",
    "attachment_verified_locally",
    "build_local_inventory_index",
    "find_local_inventory_item",
    "local_inventory_matches",
    "restore_action_for_domain",
]
