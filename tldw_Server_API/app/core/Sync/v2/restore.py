from __future__ import annotations

"""Helpers for building Sync v2 restore preview plans."""

import heapq
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .models import NOTES_ORGANIZATION_DOMAINS, SyncDomain, SyncEnvelope

WHOLE_OBJECT_RESTORE_DOMAINS: frozenset[SyncDomain] = frozenset(
    {"notes.note", "notes.task", "chat.conversation"}
)
OBJECT_RESTORE_DOMAINS: frozenset[SyncDomain] = frozenset(
    {
        "notes.note",
        "notes.task",
        "notes.task_activity",
        "chat.conversation",
        "chat.message",
        "source_cache.entry",
        "media.item",
        "media.keyword",
        "media.keyword_link",
        "notes.link",
        "attachment.ref",
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
    adapter_version: int
    object_revision: int | None
    object_hash: str | None
    deleted: bool


LocalInventoryIndex = dict[
    tuple[str | None, SyncDomain, str, int],
    LocalRestoreInventoryItem,
]


def order_restore_envelopes(
    envelopes: Sequence[SyncEnvelope],
    *,
    max_actions: int = 10_000,
) -> list[SyncEnvelope]:
    """Order restore candidates without splitting complete mutation groups."""

    if len(envelopes) > max_actions:
        raise RestorePlanningError("sync_restore_action_limit_exceeded")
    units = _restore_units(envelopes)
    units_by_identity: dict[tuple[SyncDomain, str], list[tuple[int, int, SyncEnvelope]]] = {}
    for unit_index, unit in enumerate(units):
        for position, envelope in enumerate(unit):
            units_by_identity.setdefault((envelope.domain, envelope.object_id), []).append(
                (unit_index, position, envelope)
            )
    for occurrences in units_by_identity.values():
        occurrences.sort(key=lambda item: item[2].server_cursor or 0)
    live_occurrences_by_identity = {
        identity: [item for item in occurrences if item[2].operation != "tombstone"]
        for identity, occurrences in units_by_identity.items()
    }
    live_cursors_by_identity = {
        identity: [item[2].server_cursor or 0 for item in occurrences]
        for identity, occurrences in live_occurrences_by_identity.items()
    }
    live_positions_by_identity_unit: dict[
        tuple[SyncDomain, str], dict[int, list[int]]
    ] = {}
    for identity, occurrences in live_occurrences_by_identity.items():
        positions_by_unit: dict[int, list[int]] = {}
        for unit_index, position, _ in occurrences:
            positions_by_unit.setdefault(unit_index, []).append(position)
        live_positions_by_identity_unit[identity] = positions_by_unit
    edges: dict[int, set[int]] = {index: set() for index in range(len(units))}

    for occurrences in units_by_identity.values():
        for earlier, later in zip(occurrences, occurrences[1:]):
            if earlier[0] != later[0]:
                edges[earlier[0]].add(later[0])

    for unit_index, unit in enumerate(units):
        for position, envelope in enumerate(unit):
            for dependency in _restore_dependencies(envelope):
                occurrences = units_by_identity.get(dependency)
                if not occurrences:
                    raise RestorePlanningError("Restore dependency is missing")
                live = live_occurrences_by_identity[dependency]
                if live:
                    unit_positions = live_positions_by_identity_unit[dependency].get(
                        unit_index, []
                    )
                    if bisect_left(unit_positions, position):
                        continue
                    cursors = live_cursors_by_identity[dependency]
                    insertion = bisect_left(cursors, envelope.server_cursor or 0)
                    provider_index = insertion - 1
                    while provider_index >= 0 and live[provider_index][0] == unit_index:
                        provider_index -= 1
                    if provider_index < 0:
                        provider_index = insertion
                        while (
                            provider_index < len(live)
                            and live[provider_index][0] == unit_index
                        ):
                            provider_index += 1
                    if provider_index >= len(live):
                        raise RestorePlanningError(
                            "Mutation group ordering conflicts with restore dependencies"
                        )
                    provider = live[provider_index]
                    edges[provider[0]].add(unit_index)
                    continue
                for tombstone_unit, tombstone_position, _ in occurrences:
                    if tombstone_unit == unit_index:
                        if tombstone_position <= position:
                            raise RestorePlanningError(
                                "Mutation group tombstone ordering conflicts with restore dependencies"
                            )
                        continue
                    edges[unit_index].add(tombstone_unit)

    indegree = [0] * len(units)
    for targets in edges.values():
        for target in targets:
            indegree[target] += 1
    ready = [
        (
            all(envelope.operation == "tombstone" for envelope in units[index]),
            min(envelope.server_cursor or 0 for envelope in units[index]),
            index,
        )
        for index, degree in enumerate(indegree)
        if degree == 0
    ]
    heapq.heapify(ready)
    ordered_units: list[int] = []
    while ready:
        _, _, selected = heapq.heappop(ready)
        ordered_units.append(selected)
        for target in sorted(edges[selected]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(
                    ready,
                    (
                        all(
                            envelope.operation == "tombstone"
                            for envelope in units[target]
                        ),
                        min(
                            envelope.server_cursor or 0
                            for envelope in units[target]
                        ),
                        target,
                    ),
                )
    if len(ordered_units) != len(units):
        raise RestorePlanningError("Restore dependencies contain a cycle")

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
    elif envelope.domain == "notes.link":
        dependencies.extend(
            [
                ("notes.note", str(payload.get("source_note_id"))),
                ("notes.note", str(payload.get("target_note_id"))),
            ]
        )
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
    elif envelope.domain == "notes.task":
        note_id = payload.get("note_id")
        if not isinstance(note_id, str) or not note_id:
            raise RestorePlanningError("notes.task restore dependency is invalid")
        dependencies.append(("notes.note", note_id))
    elif envelope.domain == "notes.task_activity":
        note_id = payload.get("note_id")
        task_id = payload.get("task_id")
        if not isinstance(note_id, str) or not note_id:
            raise RestorePlanningError(
                "notes.task_activity restore dependency is invalid"
            )
        dependencies.append(("notes.note", note_id))
        if task_id is not None:
            if not isinstance(task_id, str) or not task_id:
                raise RestorePlanningError(
                    "notes.task_activity restore dependency is invalid"
                )
            dependencies.append(("notes.task", task_id))
    elif envelope.domain == "attachment.ref" and envelope.adapter_version == 2:
        parent_id = payload.get("parent_object_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise RestorePlanningError(
                "attachment.ref v2 restore dependency is invalid"
            )
        dependencies.append(("notes.note", parent_id))
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
            adapter_version=_positive_int_value(raw.get("adapter_version"), default=1),
            object_revision=_int_value(raw.get("object_revision") or raw.get("entity_version")),
            object_hash=_string_value(raw.get("object_hash") or raw.get("payload_hash")),
            deleted=_bool_value(raw.get("deleted", False)),
        )
        index[
            (item.dataset_id, item.domain, item.object_id, item.adapter_version)
        ] = item
    return index


def find_local_inventory_item(
    index: LocalInventoryIndex,
    *,
    dataset_id: str,
    domain: SyncDomain,
    object_id: str,
    adapter_version: int = 1,
) -> LocalRestoreInventoryItem | None:
    """Return dataset-specific local inventory, falling back to dataset-agnostic entries."""

    return index.get(
        (dataset_id, domain, object_id, adapter_version)
    ) or index.get((None, domain, object_id, adapter_version))


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


def _positive_int_value(value: object, *, default: int) -> int:
    parsed = _int_value(value)
    return parsed if parsed is not None and parsed >= 1 else default


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
