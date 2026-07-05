from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from tldw_Server_API.app.core.RPG.constants import (
    MAX_RPG_EVENT_PAYLOAD_BYTES,
    RPG_EVENT_SCHEMA_VERSION,
    RPG_SOURCE_TYPES,
)

_REQUIRED_EVENT_IDS = {
    "actor.upserted": "actor_id",
    "clock.updated": "clock_id",
    "faction.upserted": "faction_id",
    "inventory.item.upserted": "item_id",
    "location.upserted": "location_id",
    "note.added": "note_id",
    "npc.upserted": "npc_id",
    "quest.upserted": "quest_id",
    "roll.recorded": "roll_id",
    "rule.reference.added": "reference_id",
    "ruling.added": "ruling_id",
    "scene.updated": "scene_id",
}

SUPPORTED_EVENT_TYPES = frozenset(_REQUIRED_EVENT_IDS)


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must be JSON-serializable") from exc
    return encoded.encode("utf-8")


def canonical_request_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def validate_event_envelope(event: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise ValueError("event must be an object")

    event_type = str(event.get("event_type") or "").strip()
    if not event_type:
        raise ValueError("event_type is required")
    if event_type not in SUPPORTED_EVENT_TYPES:
        raise ValueError(f"Unsupported RPG event type: {event_type}")

    source_type = str(event.get("source_type") or "").strip()
    if source_type not in RPG_SOURCE_TYPES:
        raise ValueError("source_type is invalid")

    payload = event.get("event_payload")
    if not isinstance(payload, Mapping):
        raise ValueError("event_payload must be an object")

    payload_copy = deepcopy(dict(payload))
    payload_size = len(_canonical_json_bytes(payload_copy))
    if payload_size > MAX_RPG_EVENT_PAYLOAD_BYTES:
        raise ValueError("event_payload is too large")

    required_id = _REQUIRED_EVENT_IDS[event_type]
    stable_id = payload_copy.get(required_id)
    if not isinstance(stable_id, str) or not stable_id.strip():
        raise ValueError(f"{required_id} is required for {event_type}")

    normalized = dict(event)
    normalized["event_type"] = event_type
    normalized["source_type"] = source_type
    normalized["event_payload"] = payload_copy
    normalized.setdefault("event_schema_version", RPG_EVENT_SCHEMA_VERSION)
    return normalized
