from tldw_Server_API.app.core.VN_Play.state import derive_scene_state


def test_replay_applies_scene_state_changed_event() -> None:
    events = [
        {"event_type": "session_started", "event_payload": {"schema_version": 1}},
        {
            "event_type": "scene_state_changed",
            "event_payload": {
                "background_item_id": 101,
                "active_sprite_items": [{"character_id": 1, "item_id": 201}],
                "location_key": "library",
                "scene_version": 1,
            },
        },
    ]

    state = derive_scene_state(events)

    assert state.current_background_item_id == 101
    assert state.location_key == "library"
    assert state.scene_version == 1


def test_replay_keeps_warning_for_rejected_visual_directive() -> None:
    state = derive_scene_state(
        [
            {
                "event_type": "visual_directive_rejected",
                "event_payload": {"reason": "asset_not_found", "slot_key": "sprite.happy"},
            }
        ]
    )

    assert state.warnings[0]["reason"] == "asset_not_found"


def test_replay_applies_visual_directive_assets() -> None:
    state = derive_scene_state(
        [
            {
                "event_type": "visual_directive_applied",
                "event_payload": {
                    "asset_type": "background",
                    "item": {"item_id": 101, "content_url": "/content/bg"},
                    "scene_version": 1,
                },
            },
            {
                "event_type": "visual_directive_applied",
                "event_payload": {
                    "asset_type": "sprite",
                    "item": {"item_id": 201, "content_url": "/content/sprite"},
                    "scene_version": 1,
                },
            },
        ]
    )

    assert state.current_background_item_id == 101
    assert state.active_sprite_items == [
        {"item_id": 201, "content_url": "/content/sprite"}
    ]
    assert state.scene_version == 1


def test_replay_replaces_visible_choices_after_selection() -> None:
    state = derive_scene_state(
        [
            {
                "event_type": "choice_presented",
                "event_payload": {
                    "choices": [
                        {"id": "a", "label": "Stay"},
                        {"id": "b", "label": "Leave"},
                    ],
                    "scene_version": 2,
                },
            },
            {
                "event_type": "choice_selected",
                "event_payload": {
                    "choice_id": "b",
                    "branch_node_id": 12,
                    "scene_version": 3,
                },
            },
        ]
    )

    assert state.visible_choices == []
    assert state.active_branch_node_id == 12
    assert state.scene_version == 3


def test_replay_restores_snapshot_state() -> None:
    state = derive_scene_state(
        [
            {
                "event_type": "scene_state_changed",
                "event_payload": {
                    "background_item_id": 101,
                    "location_key": "library",
                    "scene_version": 1,
                },
            },
            {
                "event_type": "session_restored",
                "event_payload": {
                    "scene_state_snapshot": {
                        "current_background_item_id": 202,
                        "current_depth_item_id": 203,
                        "active_sprite_items": [{"item_id": 301, "content_url": "/sprite"}],
                        "location_key": "garden",
                        "visible_choices": [{"id": "inspect", "label": "Inspect"}],
                        "scene_version": 7,
                    }
                },
            },
        ]
    )

    assert state.current_background_item_id == 202
    assert state.current_depth_item_id == 203
    assert state.active_sprite_items == [{"item_id": 301, "content_url": "/sprite"}]
    assert state.location_key == "garden"
    assert state.visible_choices == [{"id": "inspect", "label": "Inspect"}]
    assert state.scene_version == 7
