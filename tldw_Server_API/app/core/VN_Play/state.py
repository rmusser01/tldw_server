"""Pure event replay for VN Play scene state."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from tldw_Server_API.app.core.VN_Play.constants import (
    EVENT_CHOICE_PRESENTED,
    EVENT_CHOICE_SELECTED,
    EVENT_MODEL_TURN_PARSE_FAILED,
    EVENT_SCENE_STATE_CHANGED,
    EVENT_SESSION_RESTORED,
    EVENT_SESSION_STARTED,
    EVENT_TURN_FAILED,
    EVENT_VISUAL_DIRECTIVE_APPLIED,
    EVENT_VISUAL_DIRECTIVE_REJECTED,
)
from tldw_Server_API.app.core.VN_Play.models import SceneState


def derive_scene_state(events: Iterable[Mapping[str, Any]]) -> SceneState:
    """Derive the current scene state by replaying ordered VN Play events."""
    state = SceneState()
    for event in events:
        event_type = str(event.get("event_type") or "")
        payload = _event_payload(event)

        if event_type == EVENT_SESSION_STARTED:
            _apply_scene_version(state, payload)
        elif event_type == EVENT_SCENE_STATE_CHANGED:
            _apply_scene_state_changed(state, payload)
        elif event_type == EVENT_CHOICE_PRESENTED:
            _apply_choice_presented(state, payload)
        elif event_type == EVENT_CHOICE_SELECTED:
            state.visible_choices = []
            _apply_int_field(state, "active_branch_node_id", payload, "branch_node_id")
            _apply_scene_version(state, payload)
        elif event_type == EVENT_SESSION_RESTORED:
            snapshot = payload.get("scene_state_snapshot") or payload.get("scene_state") or {}
            if isinstance(snapshot, Mapping):
                _apply_scene_snapshot(state, snapshot)
            _apply_scene_version(state, payload)
        elif event_type == EVENT_VISUAL_DIRECTIVE_APPLIED:
            _apply_visual_directive_applied(state, payload)
        elif event_type == EVENT_VISUAL_DIRECTIVE_REJECTED:
            state.warnings.append(_warning_from_payload("visual_directive_rejected", payload))
            _apply_scene_version(state, payload)
        elif event_type in {EVENT_TURN_FAILED, EVENT_MODEL_TURN_PARSE_FAILED}:
            state.warnings.append(_warning_from_payload(event_type, payload))
            _apply_scene_version(state, payload)

    return state


def _event_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event.get("event_payload")
    if payload is None:
        payload = event.get("event_payload_json")
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, str):
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return dict(loaded) if isinstance(loaded, Mapping) else {}
    return {}


def _apply_scene_state_changed(state: SceneState, payload: Mapping[str, Any]) -> None:
    _apply_int_field(
        state,
        "current_background_item_id",
        payload,
        "current_background_item_id",
        "background_item_id",
    )
    _apply_int_field(
        state,
        "current_depth_item_id",
        payload,
        "current_depth_item_id",
        "depth_item_id",
    )
    _apply_list_field(state, "active_sprite_items", payload, "active_sprite_items")
    _apply_text_field(state, "location_key", payload)
    _apply_text_field(state, "mood", payload)
    _apply_text_field(state, "time_of_day", payload)
    _apply_text_field(state, "weather", payload)
    _apply_int_field(state, "active_branch_node_id", payload, "active_branch_node_id")
    _apply_list_field(state, "visible_choices", payload, "visible_choices")
    _apply_int_field(state, "transcript_cursor", payload, "transcript_cursor")
    _append_payload_warnings(state, payload)
    _apply_scene_version(state, payload)


def _apply_choice_presented(state: SceneState, payload: Mapping[str, Any]) -> None:
    choices = payload.get("choices", payload.get("visible_choices", []))
    state.visible_choices = _list_of_dicts(choices)
    _apply_scene_version(state, payload)


def _apply_visual_directive_applied(state: SceneState, payload: Mapping[str, Any]) -> None:
    item = payload.get("item")
    if not isinstance(item, Mapping):
        _apply_scene_version(state, payload)
        return

    item_payload = dict(item)
    item_id = item_payload.get("item_id")
    asset_type = _asset_type_from_payload(payload, item_payload)
    if asset_type == "background" and isinstance(item_id, int):
        state.current_background_item_id = item_id
    elif asset_type == "depth_companion" and isinstance(item_id, int):
        state.current_depth_item_id = item_id
    elif asset_type == "sprite":
        state.active_sprite_items.append(item_payload)

    _apply_scene_version(state, payload)


def _apply_scene_snapshot(state: SceneState, snapshot: Mapping[str, Any]) -> None:
    _apply_int_field(
        state,
        "current_background_item_id",
        snapshot,
        "current_background_item_id",
        "background_item_id",
    )
    _apply_int_field(
        state,
        "current_depth_item_id",
        snapshot,
        "current_depth_item_id",
        "depth_item_id",
    )
    _apply_list_field(state, "active_sprite_items", snapshot, "active_sprite_items")
    _apply_text_field(state, "location_key", snapshot)
    _apply_text_field(state, "mood", snapshot)
    _apply_text_field(state, "time_of_day", snapshot)
    _apply_text_field(state, "weather", snapshot)
    _apply_int_field(state, "active_branch_node_id", snapshot, "active_branch_node_id")
    _apply_list_field(state, "visible_choices", snapshot, "visible_choices")
    _apply_int_field(state, "transcript_cursor", snapshot, "transcript_cursor")
    _append_payload_warnings(state, snapshot, replace=True)
    _apply_scene_version(state, snapshot)


def _asset_type_from_payload(
    payload: Mapping[str, Any],
    item_payload: Mapping[str, Any],
) -> str:
    raw_asset_type = payload.get("asset_type") or item_payload.get("asset_type")
    if not isinstance(raw_asset_type, str):
        return ""
    normalized = raw_asset_type.strip().lower()
    if normalized in {"background", "backgrounds"}:
        return "background"
    if normalized in {"depth", "depth_companion", "depth_companions"}:
        return "depth_companion"
    if normalized in {"sprite", "sprites"}:
        return "sprite"
    return normalized


def _apply_text_field(state: SceneState, field_name: str, payload: Mapping[str, Any]) -> None:
    if field_name not in payload:
        return
    value = payload[field_name]
    if value is None or isinstance(value, str):
        setattr(state, field_name, value)


def _apply_int_field(
    state: SceneState,
    field_name: str,
    payload: Mapping[str, Any],
    *payload_keys: str,
) -> None:
    for payload_key in payload_keys:
        if payload_key not in payload:
            continue
        value = payload[payload_key]
        if value is None or isinstance(value, int):
            setattr(state, field_name, value)
        return


def _apply_list_field(
    state: SceneState,
    field_name: str,
    payload: Mapping[str, Any],
    payload_key: str,
) -> None:
    if payload_key not in payload:
        return
    setattr(state, field_name, _list_of_dicts(payload[payload_key]))


def _apply_scene_version(state: SceneState, payload: Mapping[str, Any]) -> None:
    value = payload.get("scene_version")
    if isinstance(value, int):
        state.scene_version = value


def _append_payload_warnings(
    state: SceneState,
    payload: Mapping[str, Any],
    *,
    replace: bool = False,
) -> None:
    if "warnings" not in payload:
        return
    warnings = payload["warnings"]
    normalized: list[dict[str, Any]] = []
    if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
        for warning in warnings:
            if isinstance(warning, Mapping):
                normalized.append(dict(warning))
            else:
                normalized.append({"message": str(warning)})
    if replace:
        state.warnings = normalized
    else:
        state.warnings.extend(normalized)


def _warning_from_payload(event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    warning = dict(payload)
    warning.setdefault("event_type", event_type)
    return warning


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]
