from __future__ import annotations

"""Helpers for building Sync v2 restore preview plans."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .models import NOTES_ORGANIZATION_DOMAINS, SyncDomain, SyncEnvelope

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
        *NOTES_ORGANIZATION_DOMAINS,
    }
)


class RestorePlanningError(ValueError):
    """Raised when restore candidates cannot form one safe ordered plan."""


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


def order_restore_envelopes(envelopes: Sequence[SyncEnvelope]) -> list[SyncEnvelope]:
    """Order restore candidates without splitting complete mutation groups."""

    units = _restore_units(envelopes)
    units_by_identity: dict[tuple[SyncDomain, str], list[tuple[int, int, SyncEnvelope]]] = {}
    for unit_index, unit in enumerate(units):
        for position, envelope in enumerate(unit):
            units_by_identity.setdefault((envelope.domain, envelope.object_id), []).append(
                (unit_index, position, envelope)
            )
    edges: dict[int, set[int]] = {index: set() for index in range(len(units))}

    for occurrences in units_by_identity.values():
        ordered = sorted(occurrences, key=lambda item: item[2].server_cursor or 0)
        for earlier, later in zip(ordered, ordered[1:]):
            if earlier[0] != later[0]:
                edges[earlier[0]].add(later[0])

    for unit_index, unit in enumerate(units):
        for position, envelope in enumerate(unit):
            for dependency in _restore_dependencies(envelope):
                occurrences = units_by_identity.get(dependency)
                if not occurrences:
                    raise RestorePlanningError("Restore dependency is missing")
                live = [item for item in occurrences if item[2].operation != "tombstone"]
                if live:
                    for dependency_unit, dependency_position, _ in live:
                        if dependency_unit == unit_index:
                            if dependency_position >= position:
                                raise RestorePlanningError(
                                    "Mutation group ordering conflicts with restore dependencies"
                                )
                            continue
                        edges[dependency_unit].add(unit_index)
                    continue
                for tombstone_unit, tombstone_position, _ in occurrences:
                    if tombstone_unit == unit_index:
                        if tombstone_position <= position:
                            raise RestorePlanningError(
                                "Mutation group tombstone ordering conflicts with restore dependencies"
                            )
                        continue
                    edges[unit_index].add(tombstone_unit)

    live_units = {
        index
        for index, unit in enumerate(units)
        if any(envelope.operation != "tombstone" for envelope in unit)
    }
    tombstone_units = {
        index
        for index, unit in enumerate(units)
        if all(envelope.operation == "tombstone" for envelope in unit)
    }
    for live_unit in live_units:
        edges[live_unit].update(tombstone_units)

    ordered_units: list[int] = []
    remaining = set(range(len(units)))
    while remaining:
        ready = [
            index
            for index in remaining
            if not any(index in targets for source, targets in edges.items() if source in remaining)
        ]
        if not ready:
            raise RestorePlanningError("Restore dependencies contain a cycle")
        selected = min(
            ready,
            key=lambda index: min(envelope.server_cursor or 0 for envelope in units[index]),
        )
        ordered_units.append(selected)
        remaining.remove(selected)

    return [envelope for index in ordered_units for envelope in units[index]]


def _restore_units(envelopes: Sequence[SyncEnvelope]) -> list[list[SyncEnvelope]]:
    grouped: dict[str, list[SyncEnvelope]] = {}
    for envelope in envelopes:
        if envelope.mutation_group_id:
            grouped.setdefault(envelope.mutation_group_id, []).append(envelope)
    incomplete = [group_id for group_id, group in grouped.items() if not _is_complete_restore_group(group)]
    if incomplete:
        raise RestorePlanningError("Restore mutation group is incomplete")

    emitted: set[str] = set()
    units: list[list[SyncEnvelope]] = []
    for envelope in sorted(envelopes, key=lambda item: item.server_cursor or 0):
        group_id = envelope.mutation_group_id
        if group_id is None:
            units.append([envelope])
            continue
        if group_id in emitted:
            continue
        units.append(sorted(grouped[group_id], key=lambda item: item.mutation_step or 0))
        emitted.add(group_id)
    return units


def _is_complete_restore_group(group: Sequence[SyncEnvelope]) -> bool:
    if not group or group[0].mutation_step_count != len(group):
        return False
    expected_count = group[0].mutation_step_count
    return expected_count is not None and {
        envelope.mutation_step for envelope in group
    } == set(range(expected_count)) and all(
        envelope.mutation_step_count == expected_count for envelope in group
    )


def _restore_dependencies(envelope: SyncEnvelope) -> list[tuple[SyncDomain, str]]:
    if envelope.operation == "tombstone":
        return []
    payload = envelope.payload
    dependencies: list[tuple[SyncDomain, str]] = []
    if envelope.domain in {"notes.keyword_collection", "notes.folder"}:
        parent_id = payload.get("parent_sync_id")
        if isinstance(parent_id, str) and parent_id:
            dependencies.append((envelope.domain, parent_id))
    elif envelope.domain == "notes.keyword_link":
        subject_domain: SyncDomain = (
            "notes.note"
            if payload.get("subject_type") == "note"
            else "chat.conversation"
        )
        dependencies.extend(
            [
                (subject_domain, str(payload.get("subject_id"))),
                ("notes.keyword", str(payload.get("keyword_sync_id"))),
            ]
        )
    elif envelope.domain == "notes.keyword_collection_link":
        dependencies.extend(
            [
                ("notes.keyword_collection", str(payload.get("collection_sync_id"))),
                ("notes.keyword", str(payload.get("keyword_sync_id"))),
            ]
        )
    elif envelope.domain == "notes.folder_link":
        dependencies.extend(
            [
                ("notes.note", str(payload.get("note_id"))),
                ("notes.folder", str(payload.get("folder_sync_id"))),
            ]
        )
    return dependencies


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
    "order_restore_envelopes",
    "restore_action_for_domain",
]
