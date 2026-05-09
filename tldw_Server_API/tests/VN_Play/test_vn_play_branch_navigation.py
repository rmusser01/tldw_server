from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.VN_Play.branch_navigation import (
    build_branch_navigation,
    filter_branch_events,
)


def _branch(
    branch_id: int,
    *,
    parent_event_id: int,
    branch_label: str,
    branch_path: list[dict[str, Any]],
    status: str = "active",
) -> dict[str, Any]:
    return {
        "id": branch_id,
        "session_id": 1,
        "owner_user_id": 42,
        "parent_event_id": parent_event_id,
        "branch_label": branch_label,
        "branch_path": branch_path,
        "status": status,
    }


def _choice_step(
    choice_id: str,
    *,
    event_id: int,
    scene_version: int,
    text: str | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "type": "choice",
        "choice_id": choice_id,
        "choice_presented_event_id": event_id,
        "scene_version": scene_version,
    }
    if text is not None:
        step["choice_text"] = text
    return step


def _event(
    event_id: int,
    sequence: int,
    event_type: str,
    payload: dict[str, Any] | None = None,
    branch_node_id: int | None = None,
) -> dict[str, Any]:
    return {
        "id": event_id,
        "session_id": 1,
        "owner_user_id": 42,
        "sequence_number": sequence,
        "event_type": event_type,
        "event_payload": payload or {},
        "source": "runtime",
        "branch_node_id": branch_node_id,
    }


def _door_branches() -> list[dict[str, Any]]:
    open_step = _choice_step("open", event_id=2, scene_version=1, text="Open the door")
    inside_step = _choice_step(
        "inside",
        event_id=6,
        scene_version=2,
        text="Step inside",
    )
    return [
        _branch(
            10,
            parent_event_id=2,
            branch_label="Open the door",
            branch_path=[open_step],
        ),
        _branch(
            11,
            parent_event_id=6,
            branch_label="Step inside",
            branch_path=[open_step, inside_step],
        ),
    ]


def test_navigation_derives_active_path_and_parent_branch_ids() -> None:
    events = [
        _event(
            2,
            2,
            "choice_presented",
            {"choices": [{"id": "open"}], "scene_version": 1},
        ),
        _event(
            3,
            3,
            "choice_selected",
            {"choice_id": "open", "branch_node_id": 10, "scene_version": 1},
            10,
        ),
        _event(
            6,
            6,
            "choice_presented",
            {"choices": [{"id": "inside"}], "scene_version": 2},
            10,
        ),
        _event(
            7,
            7,
            "choice_selected",
            {"choice_id": "inside", "branch_node_id": 11, "scene_version": 2},
            11,
        ),
    ]

    navigation = build_branch_navigation(
        session={"id": 1, "mode": "story", "scene_version": 3},
        branches=_door_branches(),
        events=events,
        scene_state={"active_branch_node_id": 11, "last_event_id": 7, "scene_version": 3},
    )

    assert [step["branch_id"] for step in navigation["active_path"]] == [10, 11]
    parent = next(item for item in navigation["branches"] if item["branch_id"] == 10)
    child = next(item for item in navigation["branches"] if item["branch_id"] == 11)
    assert parent["parent_branch_id"] is None
    assert child["parent_branch_id"] == 10
    assert child["depth"] == 2
    assert child["is_active"] is True
    assert child["is_on_active_path"] is True
    assert child["choice_id"] == "inside"
    assert child["choice_text"] == "Step inside"


def test_navigation_separates_direct_and_subtree_event_ranges() -> None:
    events = [
        _event(3, 3, "choice_selected", {"branch_node_id": 10}, 10),
        _event(4, 4, "model_turn", {"text": "The door opens."}, 10),
        _event(7, 7, "choice_selected", {"branch_node_id": 11}, 11),
        _event(8, 8, "model_turn", {"text": "Inside."}, 11),
    ]

    navigation = build_branch_navigation(
        session={"id": 1, "mode": "story", "scene_version": 3},
        branches=_door_branches(),
        events=events,
        scene_state={"active_branch_node_id": 11, "last_event_id": 8, "scene_version": 3},
    )

    parent = next(item for item in navigation["branches"] if item["branch_id"] == 10)
    assert parent["event_range"] == {
        "start_event_id": 3,
        "start_sequence_number": 3,
        "latest_event_id": 4,
        "latest_sequence_number": 4,
    }
    assert parent["subtree_event_range"] == {
        "start_event_id": 3,
        "start_sequence_number": 3,
        "latest_event_id": 8,
        "latest_sequence_number": 8,
    }
    assert parent["restore"]["default_target"] == "branch_latest"
    assert parent["restore"]["targets"]["branch_latest"]["event_id"] == 4
    assert parent["restore"]["targets"]["branch_latest"]["sequence_number"] == 4


def test_filter_branch_events_returns_direct_branch_events_only() -> None:
    events = [
        _event(3, 3, "choice_selected", {"branch_node_id": 10}, 10),
        _event(4, 4, "model_turn", {}, 10),
        _event(7, 7, "choice_selected", {"branch_node_id": 11}, 11),
        _event(8, 8, "model_turn", {}, 11),
    ]

    filtered, warnings = filter_branch_events(
        branch_id=10,
        branches=_door_branches(),
        events=events,
        include_descendants=False,
    )

    assert [event["id"] for event in filtered] == [3, 4]
    assert warnings == []


def test_filter_branch_events_includes_descendant_branch_events() -> None:
    events = [
        _event(3, 3, "choice_selected", {"branch_node_id": 10}, 10),
        _event(4, 4, "model_turn", {}, 10),
        _event(7, 7, "choice_selected", {"branch_node_id": 11}, 11),
        _event(8, 8, "model_turn", {}, 11),
    ]

    filtered, warnings = filter_branch_events(
        branch_id=10,
        branches=_door_branches(),
        events=events,
        include_descendants=True,
    )

    assert [event["id"] for event in filtered] == [3, 4, 7, 8]
    assert warnings == []


def test_replay_fallback_assigns_untagged_events_to_active_branch_intervals() -> None:
    events = [
        _event(3, 3, "choice_selected", {"branch_node_id": 10}),
        _event(4, 4, "model_turn", {"text": "Untagged parent event."}),
        _event(7, 7, "choice_selected", {"branch_node_id": 11}),
        _event(8, 8, "model_turn", {"text": "Untagged child event."}),
        _event(
            9,
            9,
            "session_restored",
            {"scene_state_snapshot": {"active_branch_node_id": 10}},
        ),
        _event(10, 10, "model_turn", {"text": "Back on parent."}),
    ]

    direct, warnings = filter_branch_events(
        branch_id=10,
        branches=_door_branches(),
        events=events,
        include_descendants=False,
    )

    assert [event["id"] for event in direct] == [3, 4, 9, 10]
    assert warnings == []


def test_replay_cap_emits_stable_warning_payload() -> None:
    events = [
        _event(3, 3, "choice_selected", {"branch_node_id": 10}),
        _event(4, 4, "model_turn", {"text": "Untagged parent event."}),
    ]

    filtered, warnings = filter_branch_events(
        branch_id=10,
        branches=_door_branches(),
        events=events,
        replay_limit=1,
    )

    assert [event["id"] for event in filtered] == [3]
    assert warnings
    assert warnings[0]["code"] == "branch_interval_replay_limit_exceeded"
    assert warnings[0]["severity"] == "warning"
    assert warnings[0]["recoverable"] is True


def test_replay_cap_preserves_events_derived_before_the_cap() -> None:
    events = [
        _event(3, 3, "choice_selected", {"branch_node_id": 10}),
        _event(4, 4, "model_turn", {"text": "Derived before cap."}),
        _event(5, 5, "model_turn", {"text": "Beyond cap."}),
    ]

    filtered, warnings = filter_branch_events(
        branch_id=10,
        branches=_door_branches(),
        events=events,
        replay_limit=2,
    )

    assert [event["id"] for event in filtered] == [3, 4]
    assert warnings
    assert warnings[0]["code"] == "branch_interval_replay_limit_exceeded"
    assert warnings[0]["severity"] == "warning"
    assert warnings[0]["recoverable"] is True


def test_ambiguous_replay_fallback_emits_frontend_safe_warning() -> None:
    events = [
        _event(3, 3, "choice_selected", {"choice_id": "open"}),
        _event(4, 4, "model_turn", {"text": "Cannot attribute this event."}),
    ]

    filtered, warnings = filter_branch_events(
        branch_id=10,
        branches=_door_branches(),
        events=events,
    )

    assert filtered == []
    assert warnings
    warning = warnings[0]
    assert warning["code"] == "branch_interval_replay_ambiguous"
    assert warning["severity"] == "warning"
    assert warning["recoverable"] is True
    assert warning["event_id"] == 4
    assert "Traceback" not in str(warning)
    assert "Exception" not in str(warning)


def test_warning_payloads_are_frontend_safe() -> None:
    navigation = build_branch_navigation(
        session={"id": 1, "mode": "story", "scene_version": 3},
        branches=[
            _branch(
                10,
                parent_event_id=2,
                branch_label="Open the door",
                branch_path=[_choice_step("open", event_id=2, scene_version=1)],
            ),
            _branch(
                12,
                parent_event_id=9,
                branch_label="Missing parent",
                branch_path=[
                    _choice_step("open", event_id=2, scene_version=1),
                    _choice_step("missing", event_id=9, scene_version=9),
                ],
            ),
        ],
        events=[_event(3, 3, "choice_selected", {"branch_node_id": 10}, 10)],
        scene_state={"active_branch_node_id": 999, "last_event_id": 3, "scene_version": 3},
        replay_limit=0,
    )

    assert navigation["warnings"]
    for warning in navigation["warnings"]:
        assert set(warning) >= {"code", "severity", "recoverable"}
        warning_text = str(warning)
        assert "Traceback" not in warning_text
        assert "Exception" not in warning_text
