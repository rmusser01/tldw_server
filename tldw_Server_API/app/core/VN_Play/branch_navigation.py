"""Pure read-model helpers for VN Play branch navigation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from tldw_Server_API.app.core.VN_Play.constants import (
    BRANCH_RESTORE_TARGET_CHOICE_POINT,
    BRANCH_RESTORE_TARGET_LATEST,
    ERROR_BRANCH_INTERVAL_REPLAY_LIMIT_EXCEEDED,
    ERROR_BRANCH_NOT_FOUND,
    EVENT_CHOICE_SELECTED,
    EVENT_SESSION_RESTORED,
    VN_PLAY_BRANCH_NAV_MAX_REPLAY_EVENTS,
)

_WARNING_PARENT_BRANCH_UNRESOLVED = "parent_branch_unresolved"
_WARNING_ACTIVE_BRANCH_UNRESOLVED = "active_branch_unresolved"
_WARNING_BRANCH_INTERVAL_REPLAY_AMBIGUOUS = "branch_interval_replay_ambiguous"


def build_branch_navigation(
    *,
    session: Mapping[str, Any],
    branches: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    scene_state: Mapping[str, Any] | None,
    replay_limit: int = VN_PLAY_BRANCH_NAV_MAX_REPLAY_EVENTS,
) -> dict[str, Any]:
    """Build the derived VN Play branch navigation read model."""
    branch_index = _build_branch_index(branches)
    event_branch_ids, warnings = _event_branch_ids(events, replay_limit=replay_limit)
    ranges = _event_ranges(
        branch_index=branch_index,
        events=events,
        event_branch_ids=event_branch_ids,
    )
    active_branch_id = _int_value((scene_state or {}).get("active_branch_node_id"))
    active_path_ids = _active_path_ids(active_branch_id, branch_index["parents"])

    nodes: list[dict[str, Any]] = []
    for branch in branch_index["branches"]:
        branch_id = branch["branch_id"]
        parent_branch_id = branch_index["parents"].get(branch_id)
        if parent_branch_id is None and branch["depth"] > 1:
            warnings.append(
                _warning(
                    _WARNING_PARENT_BRANCH_UNRESOLVED,
                    message="Parent branch could not be resolved from branch path prefix.",
                    branch_id=branch_id,
                )
            )

        direct_range = ranges["direct"].get(branch_id, _empty_range())
        subtree_range = ranges["subtree"].get(branch_id, _empty_range())
        node = {
            "branch_id": branch_id,
            "parent_branch_id": parent_branch_id,
            "parent_event_id": branch["parent_event_id"],
            "choice_selected_event_id": _choice_selected_event_id(events, branch_id),
            "branch_label": branch["branch_label"],
            "choice_id": branch["choice_id"],
            "choice_text": branch["choice_text"],
            "branch_path": branch["branch_path"],
            "depth": branch["depth"],
            "status": branch["status"],
            "is_active": branch_id == active_branch_id,
            "is_on_active_path": branch_id in active_path_ids,
            "event_range": direct_range,
            "subtree_event_range": subtree_range,
            "restore": _restore_payload(branch, direct_range),
            "warnings": [],
        }
        nodes.append(node)

    if active_branch_id is not None and active_branch_id not in branch_index["by_id"]:
        warnings.append(
            _warning(
                _WARNING_ACTIVE_BRANCH_UNRESOLVED,
                message="Active branch node is not present in branch rows.",
                branch_id=active_branch_id,
            )
        )

    node_by_id = {node["branch_id"]: node for node in nodes}
    active_path = [
        _active_path_step(node_by_id[branch_id])
        for branch_id in active_path_ids
        if branch_id in node_by_id
    ]

    return {
        "session_id": session.get("id"),
        "mode": session.get("mode"),
        "scene_version": (scene_state or {}).get("scene_version", session.get("scene_version")),
        "last_event_id": (scene_state or {}).get("last_event_id"),
        "active_branch_node_id": active_branch_id,
        "active_path": active_path,
        "branches": nodes,
        "warnings": warnings,
    }


def filter_branch_events(
    *,
    branch_id: int,
    branches: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    include_descendants: bool = False,
    after_sequence: int | None = None,
    limit: int = 100,
    replay_limit: int = VN_PLAY_BRANCH_NAV_MAX_REPLAY_EVENTS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return direct or subtree branch events plus frontend-safe warnings."""
    branch_index = _build_branch_index(branches)
    if branch_id not in branch_index["by_id"]:
        return [], [
            _warning(
                ERROR_BRANCH_NOT_FOUND,
                severity="error",
                recoverable=False,
                message="Branch was not found.",
                branch_id=branch_id,
            )
        ]

    event_branch_ids, warnings = _event_branch_ids(events, replay_limit=replay_limit)
    branch_ids = {branch_id}
    if include_descendants:
        branch_ids.update(_descendant_ids(branch_id, branch_index["parents"]))

    bounded_limit = max(0, limit)
    if bounded_limit == 0:
        return [], warnings

    filtered: list[dict[str, Any]] = []
    for event in _ordered_events(events):
        sequence_number = _int_value(event.get("sequence_number"))
        if (
            after_sequence is not None
            and sequence_number is not None
            and sequence_number <= after_sequence
        ):
            continue
        event_branch_id = event_branch_ids.get(_event_identity(event))
        if event_branch_id in branch_ids:
            filtered.append(dict(event))
        if len(filtered) >= bounded_limit:
            break
    return filtered, warnings


def _build_branch_index(branches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    path_to_branch_id: dict[tuple[tuple[Any, Any, Any], ...], int] = {}

    for branch in branches:
        branch_id = _int_value(branch.get("id"))
        if branch_id is None:
            continue
        branch_path = _normalized_branch_path(branch.get("branch_path"))
        path_identity = tuple(_path_step_identity(step) for step in branch_path)
        branch_data = {
            "branch_id": branch_id,
            "parent_event_id": _int_value(branch.get("parent_event_id")),
            "branch_label": _text_value(branch.get("branch_label")) or "",
            "branch_path": branch_path,
            "depth": len(branch_path),
            "status": _text_value(branch.get("status")) or "active",
            "choice_id": branch_path[-1].get("choice_id") if branch_path else None,
            "choice_text": _branch_choice_text(branch, branch_path),
        }
        normalized.append(branch_data)
        path_to_branch_id[path_identity] = branch_id

    parents: dict[int, int | None] = {}
    for branch in normalized:
        path_identity = tuple(_path_step_identity(step) for step in branch["branch_path"])
        if len(path_identity) <= 1:
            parents[branch["branch_id"]] = None
        else:
            parents[branch["branch_id"]] = path_to_branch_id.get(path_identity[:-1])

    return {
        "branches": sorted(normalized, key=lambda item: (item["depth"], item["branch_id"])),
        "by_id": {branch["branch_id"]: branch for branch in normalized},
        "parents": parents,
    }


def _normalized_branch_path(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)):
        return []

    normalized: list[dict[str, Any]] = []
    for raw_step in value:
        if not isinstance(raw_step, Mapping):
            continue
        step = {
            "type": _text_value(raw_step.get("type")) or "choice",
            "choice_id": raw_step.get("choice_id"),
            "choice_presented_event_id": _int_value(raw_step.get("choice_presented_event_id")),
            "scene_version": _int_value(raw_step.get("scene_version")),
        }
        choice_text = raw_step.get("choice_text", raw_step.get("choice_label", raw_step.get("text")))
        if choice_text is not None:
            step["choice_text"] = str(choice_text)
        normalized.append(step)
    return normalized


def _path_step_identity(step: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        step.get("choice_id"),
        step.get("choice_presented_event_id"),
        step.get("scene_version"),
    )


def _branch_choice_text(
    branch: Mapping[str, Any],
    branch_path: Sequence[Mapping[str, Any]],
) -> str | None:
    if branch_path:
        choice_text = branch_path[-1].get("choice_text")
        if choice_text is not None:
            return str(choice_text)
    branch_label = branch.get("branch_label")
    return str(branch_label) if branch_label is not None else None


def _event_branch_ids(
    events: Sequence[Mapping[str, Any]],
    *,
    replay_limit: int,
) -> tuple[dict[tuple[Any, Any], int], list[dict[str, Any]]]:
    ordered_events = _ordered_events(events)
    warnings: list[dict[str, Any]] = []
    event_branch_ids: dict[tuple[Any, Any], int] = {}

    for event in ordered_events:
        explicit_branch_id = _int_value(event.get("branch_node_id"))
        if explicit_branch_id is not None:
            event_branch_ids[_event_identity(event)] = explicit_branch_id

    missing_branch_tags = any(
        _int_value(event.get("branch_node_id")) is None for event in ordered_events
    )
    if not missing_branch_tags:
        return event_branch_ids, warnings

    bounded_replay_events = ordered_events[: max(0, replay_limit)]
    if len(ordered_events) > len(bounded_replay_events):
        warnings.append(
            _warning(
                ERROR_BRANCH_INTERVAL_REPLAY_LIMIT_EXCEEDED,
                message="Branch interval replay was capped before all untagged events could be derived.",
            )
        )

    active_branch_id: int | None = None
    branch_change_ambiguous = False
    for event in bounded_replay_events:
        event_type = _text_value(event.get("event_type")) or ""
        payload = _event_payload(event)
        if event_type == EVENT_CHOICE_SELECTED:
            active_branch_id = _int_value(event.get("branch_node_id"))
            if active_branch_id is None:
                active_branch_id = _int_value(payload.get("branch_node_id"))
            branch_change_ambiguous = active_branch_id is None
        elif event_type == EVENT_SESSION_RESTORED:
            restored_branch_id = _restored_active_branch_id(payload)
            active_branch_id = restored_branch_id
            branch_change_ambiguous = active_branch_id is None

        identity = _event_identity(event)
        if identity not in event_branch_ids and active_branch_id is not None:
            event_branch_ids[identity] = active_branch_id
        elif (
            identity not in event_branch_ids
            and branch_change_ambiguous
            and event_type not in {EVENT_CHOICE_SELECTED, EVENT_SESSION_RESTORED}
        ):
            warnings.append(
                _warning(
                    _WARNING_BRANCH_INTERVAL_REPLAY_AMBIGUOUS,
                    message="Branch interval replay could not attribute an untagged event.",
                    event_id=_int_value(event.get("id")),
                )
            )

    return event_branch_ids, warnings


def _event_ranges(
    *,
    branch_index: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    event_branch_ids: Mapping[tuple[Any, Any], int],
) -> dict[str, dict[int, dict[str, int | None]]]:
    direct_events: dict[int, list[Mapping[str, Any]]] = {}
    subtree_events: dict[int, list[Mapping[str, Any]]] = {}
    parents = branch_index["parents"]

    for event in _ordered_events(events):
        branch_id = event_branch_ids.get(_event_identity(event))
        if branch_id is None:
            continue
        direct_events.setdefault(branch_id, []).append(event)
        for ancestor_id in _ancestor_chain(branch_id, parents):
            subtree_events.setdefault(ancestor_id, []).append(event)

    return {
        "direct": {
            branch_id: _range_payload(branch_events)
            for branch_id, branch_events in direct_events.items()
        },
        "subtree": {
            branch_id: _range_payload(branch_events)
            for branch_id, branch_events in subtree_events.items()
        },
    }


def _range_payload(events: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    ordered_events = _ordered_events(events)
    start_event = _first_choice_selected(ordered_events) or ordered_events[0]
    latest_event = ordered_events[-1]
    return {
        "start_event_id": _int_value(start_event.get("id")),
        "start_sequence_number": _int_value(start_event.get("sequence_number")),
        "latest_event_id": _int_value(latest_event.get("id")),
        "latest_sequence_number": _int_value(latest_event.get("sequence_number")),
    }


def _empty_range() -> dict[str, int | None]:
    return {
        "start_event_id": None,
        "start_sequence_number": None,
        "latest_event_id": None,
        "latest_sequence_number": None,
    }


def _first_choice_selected(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for event in events:
        if event.get("event_type") == EVENT_CHOICE_SELECTED:
            return event
    return None


def _restore_payload(
    branch: Mapping[str, Any],
    direct_range: Mapping[str, Any] | None,
) -> dict[str, Any]:
    branch_latest = None
    if direct_range and direct_range.get("latest_event_id") is not None:
        branch_latest = {
            "event_id": direct_range.get("latest_event_id"),
            "sequence_number": direct_range.get("latest_sequence_number"),
        }
    choice_point = None
    if branch.get("parent_event_id") is not None:
        choice_point = {"event_id": branch.get("parent_event_id")}
    default_target = None
    if branch_latest is not None:
        default_target = BRANCH_RESTORE_TARGET_LATEST
    elif choice_point is not None:
        default_target = BRANCH_RESTORE_TARGET_CHOICE_POINT
    return {
        "supported": branch_latest is not None or choice_point is not None,
        "default_target": default_target,
        "target_names": [
            BRANCH_RESTORE_TARGET_LATEST,
            BRANCH_RESTORE_TARGET_CHOICE_POINT,
        ],
        "targets": {
            BRANCH_RESTORE_TARGET_LATEST: branch_latest,
            BRANCH_RESTORE_TARGET_CHOICE_POINT: choice_point,
        },
    }


def _choice_selected_event_id(events: Sequence[Mapping[str, Any]], branch_id: int) -> int | None:
    for event in _ordered_events(events):
        if event.get("event_type") != EVENT_CHOICE_SELECTED:
            continue
        payload = _event_payload(event)
        event_branch_id = _int_value(event.get("branch_node_id"))
        payload_branch_id = _int_value(payload.get("branch_node_id"))
        if event_branch_id == branch_id or payload_branch_id == branch_id:
            return _int_value(event.get("id"))
    return None


def _active_path_ids(active_branch_id: int | None, parents: Mapping[int, int | None]) -> list[int]:
    if active_branch_id is None:
        return []
    return list(reversed(_ancestor_chain(active_branch_id, parents)))


def _active_path_step(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "branch_id": node["branch_id"],
        "branch_label": node["branch_label"],
        "choice_id": node["choice_id"],
        "choice_text": node["choice_text"],
        "depth": node["depth"],
    }


def _ancestor_chain(branch_id: int, parents: Mapping[int, int | None]) -> list[int]:
    chain: list[int] = []
    seen: set[int] = set()
    current: int | None = branch_id
    while current is not None and current not in seen:
        seen.add(current)
        chain.append(current)
        current = parents.get(current)
    return chain


def _descendant_ids(branch_id: int, parents: Mapping[int, int | None]) -> set[int]:
    descendants: set[int] = set()
    for candidate_id in parents:
        if candidate_id == branch_id:
            continue
        if branch_id in _ancestor_chain(candidate_id, parents)[1:]:
            descendants.add(candidate_id)
    return descendants


def _ordered_events(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        events,
        key=lambda event: (
            _event_sort_value(event, "sequence_number"),
            _event_sort_value(event, "id"),
        ),
    )


def _event_sort_value(event: Mapping[str, Any], key: str) -> int:
    value = _int_value(event.get(key))
    return value if value is not None else -1


def _event_payload(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = event.get("event_payload")
    if payload is None:
        payload = event.get("event_payload_json")
    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _restored_active_branch_id(payload: Mapping[str, Any]) -> int | None:
    direct = _int_value(payload.get("active_branch_node_id", payload.get("branch_node_id")))
    if direct is not None:
        return direct
    snapshot = payload.get("scene_state_snapshot") or payload.get("scene_state")
    if isinstance(snapshot, Mapping):
        return _int_value(snapshot.get("active_branch_node_id"))
    return None


def _event_identity(event: Mapping[str, Any]) -> tuple[Any, Any]:
    return (event.get("id"), event.get("sequence_number"))


def _warning(
    code: str,
    *,
    severity: str = "warning",
    recoverable: bool = True,
    message: str | None = None,
    branch_id: int | None = None,
    event_id: int | None = None,
) -> dict[str, Any]:
    warning: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "recoverable": recoverable,
    }
    if message:
        warning["message"] = message
    if branch_id is not None:
        warning["branch_id"] = branch_id
    if event_id is not None:
        warning["event_id"] = event_id
    return warning


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)
