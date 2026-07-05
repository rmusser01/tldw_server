import pytest

from tldw_Server_API.app.core.RPG.constants import MAX_RPG_EVENT_PAYLOAD_BYTES
from tldw_Server_API.app.core.RPG.events import (
    SUPPORTED_EVENT_TYPES,
    canonical_request_hash,
    validate_event_envelope,
)
from tldw_Server_API.app.core.RPG.reducer import initial_snapshot, reduce_events

pytestmark = pytest.mark.unit


def _event(event_type, payload, source_type="user"):
    return {
        "event_type": event_type,
        "event_payload": payload,
        "source_type": source_type,
    }


def test_canonical_request_hash_is_stable_for_key_order():
    left = {"events": [_event("note.added", {"text": "A", "note_id": "n1"})]}
    right = {
        "events": [
            {
                "event_payload": {"note_id": "n1", "text": "A"},
                "source_type": "user",
                "event_type": "note.added",
            }
        ]
    }

    assert canonical_request_hash(left) == canonical_request_hash(right)  # nosec B101


def test_validate_event_envelope_rejects_missing_stable_ids():
    event = _event("npc.upserted", {"name": "Ada"})

    with pytest.raises(ValueError, match="npc_id"):
        validate_event_envelope(event)


def test_validate_event_envelope_rejects_invalid_source_type():
    event = _event("note.added", {"note_id": "n1", "text": "A"}, source_type="browser")

    with pytest.raises(ValueError, match="source_type"):
        validate_event_envelope(event)


def test_validate_event_envelope_rejects_oversized_payload():
    event = _event(
        "note.added",
        {"note_id": "n1", "text": "x" * (MAX_RPG_EVENT_PAYLOAD_BYTES + 1)},
    )

    with pytest.raises(ValueError, match="too large"):
        validate_event_envelope(event)


def test_validate_event_envelope_rejects_unknown_event_type():
    event = _event("homebrew.mutates_state", {"id": "x"})

    with pytest.raises(ValueError, match="Unsupported RPG event type"):
        validate_event_envelope(event)


def test_supported_event_registry_exposes_v1_core_event_types():
    assert frozenset(  # nosec B101
        {
            "actor.upserted",
            "clock.updated",
            "faction.upserted",
            "inventory.item.upserted",
            "location.upserted",
            "note.added",
            "npc.upserted",
            "quest.upserted",
            "roll.recorded",
            "rule.reference.added",
            "ruling.added",
            "scene.updated",
        }
    ) == SUPPORTED_EVENT_TYPES


def test_reducer_rebuilds_same_snapshot_from_all_v1_core_events():
    events = [
        _event("scene.updated", {"scene_id": "s1", "summary": "Rainy docks"}),
        _event("actor.upserted", {"actor_id": "pc-1", "name": "Marin"}),
        _event("npc.upserted", {"npc_id": "npc-ada", "name": "Ada"}),
        _event("quest.upserted", {"quest_id": "q1", "title": "Find the map"}),
        _event("inventory.item.upserted", {"item_id": "map", "name": "Wet map"}),
        _event("location.upserted", {"location_id": "docks", "name": "The docks"}),
        _event("faction.upserted", {"faction_id": "guild", "name": "Harbor Guild"}),
        _event("clock.updated", {"clock_id": "storm", "progress": 2, "segments": 6}),
        _event("roll.recorded", {"roll_id": "roll-1", "total": 15}),
        _event("note.added", {"note_id": "note-1", "text": "Storm clouds gather"}),
        _event(
            "rule.reference.added",
            {"reference_id": "ref-1", "label": "Cover", "source": "SRD"},
        ),
        _event(
            "ruling.added",
            {
                "ruling_id": "ruling-1",
                "question": "Can Marin jump?",
                "status": "open",
            },
        ),
    ]

    first = reduce_events(initial_snapshot(), events)
    second = reduce_events(initial_snapshot(), events)

    assert first == second  # nosec B101
    assert first.scene["summary"] == "Rainy docks"  # nosec B101
    assert first.actors["pc-1"]["name"] == "Marin"  # nosec B101
    assert first.npcs["npc-ada"]["name"] == "Ada"  # nosec B101
    assert first.quests["q1"]["title"] == "Find the map"  # nosec B101
    assert first.inventory["map"]["name"] == "Wet map"  # nosec B101
    assert first.locations["docks"]["name"] == "The docks"  # nosec B101
    assert first.factions["guild"]["name"] == "Harbor Guild"  # nosec B101
    assert first.clocks["storm"]["progress"] == 2  # nosec B101
    assert first.rolls[0]["roll_id"] == "roll-1"  # nosec B101
    assert first.notes[0]["note_id"] == "note-1"  # nosec B101
    assert first.rules_references[0]["reference_id"] == "ref-1"  # nosec B101
    assert (  # nosec B101
        first.unresolved_rulings["ruling-1"]["question"] == "Can Marin jump?"
    )


def test_reducer_rejects_unsupported_event_type():
    events = [_event("homebrew.mutates_state", {"id": "x"})]

    with pytest.raises(ValueError, match="Unsupported RPG event type"):
        reduce_events(initial_snapshot(), events)
