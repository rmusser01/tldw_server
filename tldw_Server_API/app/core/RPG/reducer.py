from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from tldw_Server_API.app.core.RPG.events import validate_event_envelope
from tldw_Server_API.app.core.RPG.models import RPGSnapshotState


def initial_snapshot() -> RPGSnapshotState:
    return RPGSnapshotState()


def _merged_mapping(
    current: dict[str, dict[str, Any]],
    stable_id: str,
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    updated = {key: deepcopy(value) for key, value in current.items()}
    updated[stable_id] = {**updated.get(stable_id, {}), **deepcopy(payload)}
    return updated


def _appended(
    current: list[dict[str, Any]],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [*[deepcopy(item) for item in current], deepcopy(payload)]


def reduce_event(snapshot: RPGSnapshotState, event: dict[str, Any]) -> RPGSnapshotState:
    normalized = validate_event_envelope(event)
    event_type = normalized["event_type"]
    payload = normalized["event_payload"]

    if event_type == "scene.updated":
        return replace(snapshot, scene={**deepcopy(snapshot.scene), **deepcopy(payload)})
    if event_type == "actor.upserted":
        return replace(
            snapshot,
            actors=_merged_mapping(snapshot.actors, payload["actor_id"], payload),
        )
    if event_type == "npc.upserted":
        return replace(snapshot, npcs=_merged_mapping(snapshot.npcs, payload["npc_id"], payload))
    if event_type == "quest.upserted":
        return replace(
            snapshot,
            quests=_merged_mapping(snapshot.quests, payload["quest_id"], payload),
        )
    if event_type == "inventory.item.upserted":
        return replace(
            snapshot,
            inventory=_merged_mapping(snapshot.inventory, payload["item_id"], payload),
        )
    if event_type == "location.upserted":
        return replace(
            snapshot,
            locations=_merged_mapping(snapshot.locations, payload["location_id"], payload),
        )
    if event_type == "faction.upserted":
        return replace(
            snapshot,
            factions=_merged_mapping(snapshot.factions, payload["faction_id"], payload),
        )
    if event_type == "clock.updated":
        return replace(snapshot, clocks=_merged_mapping(snapshot.clocks, payload["clock_id"], payload))
    if event_type == "roll.recorded":
        return replace(snapshot, rolls=_appended(snapshot.rolls, payload))
    if event_type == "note.added":
        return replace(snapshot, notes=_appended(snapshot.notes, payload))
    if event_type == "rule.reference.added":
        return replace(snapshot, rules_references=_appended(snapshot.rules_references, payload))
    if event_type == "ruling.added":
        return replace(
            snapshot,
            unresolved_rulings=_merged_mapping(
                snapshot.unresolved_rulings,
                payload["ruling_id"],
                payload,
            ),
        )
    raise ValueError(f"Unsupported RPG event type: {event_type}")


def reduce_events(snapshot: RPGSnapshotState, events: list[dict[str, Any]]) -> RPGSnapshotState:
    current = snapshot
    for event in events:
        current = reduce_event(current, event)
    return current
