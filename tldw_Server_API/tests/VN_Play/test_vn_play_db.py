from collections.abc import Generator

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPlay_DB import (
    VNPlayRepository,
    ensure_vn_play_tables,
)

STORY_BRANCH_LABEL_MAX_LENGTH = 160


@pytest.fixture
def chacha_db() -> Generator[CharactersRAGDB, None, None]:
    database = CharactersRAGDB(":memory:", client_id="vn-play-test-client")
    yield database
    database.close_connection()


def test_initialized_creates_session_event_turn_and_state_tables(chacha_db: CharactersRAGDB) -> None:
    repo = VNPlayRepository.initialized(chacha_db)

    session = repo.create_session(
        owner_user_id=42,
        mode="freeform",
        title="Library night",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
        seed="seed-1",
        settings={},
    )
    event = repo.append_event(
        session_id=session["id"],
        owner_user_id=42,
        event_type="session_started",
        event_payload={"schema_version": 1},
        source="system",
    )
    turn = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="turn-1",
        request_payload_hash="hash-1",
        base_scene_version=0,
    )

    assert session["scene_version"] == 0
    assert event["sequence_number"] == 1
    assert event["event_payload"] == {"schema_version": 1}
    assert turn["status"] == "pending"

    cursor = chacha_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vn_play_%'"
    )
    table_names = {row[0] for row in cursor.fetchall()}
    assert {
        "vn_play_sessions",
        "vn_play_events",
        "vn_play_turn_requests",
        "vn_play_scene_state",
        "vn_play_branches",
        "vn_play_checkpoints",
    }.issubset(table_names)


def test_initialized_creates_session_actions_and_active_session_action_column(
    chacha_db: CharactersRAGDB,
) -> None:
    VNPlayRepository.initialized(chacha_db)

    table_cursor = chacha_db.execute_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = 'vn_play_session_actions'
        """
    )
    assert table_cursor.fetchone() is not None

    column_cursor = chacha_db.execute_query("PRAGMA table_info(vn_play_sessions)")
    column_names = {row["name"] for row in column_cursor.fetchall()}
    assert "active_session_action_id" in column_names


def test_turn_request_idempotency_key_is_unique_per_session(chacha_db: CharactersRAGDB) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="freeform",
        title="Library night",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
        seed="seed-1",
        settings={},
    )
    first = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="same",
        request_payload_hash="hash-a",
        base_scene_version=0,
    )
    replayed = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="same",
        request_payload_hash="hash-a",
        base_scene_version=0,
    )

    assert replayed["id"] == first["id"]

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_turn_request(
            session_id=session["id"],
            owner_user_id=42,
            idempotency_key="same",
            request_payload_hash="hash-b",
            base_scene_version=0,
        )


def test_session_action_idempotency_key_is_unique_per_session_and_decodes_json(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Library branch",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed-1",
    )
    first = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="restore-1",
        request_payload_hash="branch_restore:hash-a",
    )
    replayed = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="restore-1",
        request_payload_hash="branch_restore:hash-a",
    )

    assert replayed["id"] == first["id"]
    assert replayed["action_type"] == "branch_restore"
    assert replayed["response_payload"] is None
    assert replayed["error"] is None

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_session_action(
            session_id=session["id"],
            owner_user_id=42,
            action_type="branch_restore",
            idempotency_key="restore-1",
            request_payload_hash="branch_restore:hash-b",
        )

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_session_action(
            session_id=session["id"],
            owner_user_id=42,
            action_type="checkpoint_restore",
            idempotency_key="restore-1",
            request_payload_hash="checkpoint_restore:hash-a",
        )

    updated = repo.update_session_action(
        first["id"],
        {
            "status": "succeeded",
            "response_payload": {"scene_version": 3, "warnings": []},
            "error": {"code": "stale_scene_version"},
            "lease_owner": "worker-1",
        },
        owner_user_id=42,
    )

    assert updated["status"] == "succeeded"
    assert updated["response_payload"] == {"scene_version": 3, "warnings": []}
    assert updated["error"] == {"code": "stale_scene_version"}
    assert repo.get_session_action_by_key(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="restore-1",
    )["id"] == first["id"]


def test_turn_and_session_action_locks_share_session_mutation_gate(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Shared gate",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    turn = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="turn-1",
        request_payload_hash="turn-hash",
        base_scene_version=0,
    )
    action = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="restore-1",
        request_payload_hash="restore-hash",
    )
    other_action = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="checkpoint_restore",
        idempotency_key="restore-2",
        request_payload_hash="restore-hash-2",
    )

    assert repo.try_acquire_session_action_lock(
        session_id=session["id"],
        owner_user_id=42,
        action_id=action["id"],
        expected_scene_version=1,
    ) is False
    assert repo.try_acquire_session_action_lock(
        session_id=session["id"],
        owner_user_id=42,
        action_id=action["id"],
        expected_scene_version=0,
    ) is True
    assert (
        repo.get_session(session["id"], owner_user_id=42)["active_session_action_id"]
        == action["id"]
    )
    assert repo.latest_active_session_action(
        session_id=session["id"],
        owner_user_id=42,
    )["id"] == action["id"]
    assert repo.try_acquire_turn_lock(
        session_id=session["id"],
        owner_user_id=42,
        turn_request_id=turn["id"],
        expected_scene_version=0,
    ) is False

    repo.clear_session_action_lock(
        session_id=session["id"],
        owner_user_id=42,
        action_id=other_action["id"],
    )
    assert repo.get_session(session["id"], owner_user_id=42)["active_session_action_id"] == action["id"]

    repo.clear_session_action_lock(
        session_id=session["id"],
        owner_user_id=42,
        action_id=action["id"],
    )
    assert repo.get_session(session["id"], owner_user_id=42)["active_session_action_id"] is None

    assert repo.try_acquire_turn_lock(
        session_id=session["id"],
        owner_user_id=42,
        turn_request_id=turn["id"],
        expected_scene_version=0,
    ) is True
    assert repo.try_acquire_session_action_lock(
        session_id=session["id"],
        owner_user_id=42,
        action_id=action["id"],
        expected_scene_version=0,
    ) is False

    repo.update_session(session["id"], {"active_turn_request_id": None}, owner_user_id=42)
    assert repo.try_acquire_session_action_lock(
        session_id=session["id"],
        owner_user_id=42,
        action_id=other_action["id"],
        expected_scene_version=0,
    ) is True
    repo.clear_session_action_lock(session_id=session["id"], owner_user_id=42)
    assert repo.get_session(session["id"], owner_user_id=42)["active_session_action_id"] is None


def test_session_action_lock_rejects_invalid_or_unlockable_actions(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Lock target",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    other_session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Other session",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    other_owner_session = repo.create_session(
        owner_user_id=43,
        mode="story",
        title="Other owner",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    other_session_action = repo.create_session_action(
        session_id=other_session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="restore-other-session",
        request_payload_hash="restore-other-session-hash",
    )
    other_owner_action = repo.create_session_action(
        session_id=other_owner_session["id"],
        owner_user_id=43,
        action_type="branch_restore",
        idempotency_key="restore-other-owner",
        request_payload_hash="restore-other-owner-hash",
    )
    completed_action = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="restore-completed",
        request_payload_hash="restore-completed-hash",
        status="succeeded",
    )

    assert repo.try_acquire_session_action_lock(
        session_id=session["id"],
        owner_user_id=42,
        action_id=999_999,
        expected_scene_version=0,
    ) is False
    assert repo.try_acquire_session_action_lock(
        session_id=session["id"],
        owner_user_id=42,
        action_id=other_session_action["id"],
        expected_scene_version=0,
    ) is False
    assert repo.try_acquire_session_action_lock(
        session_id=session["id"],
        owner_user_id=42,
        action_id=other_owner_action["id"],
        expected_scene_version=0,
    ) is False
    assert repo.try_acquire_session_action_lock(
        session_id=session["id"],
        owner_user_id=42,
        action_id=completed_action["id"],
        expected_scene_version=0,
    ) is False
    assert repo.get_session(session["id"], owner_user_id=42)["active_session_action_id"] is None


def test_latest_active_session_action_ignores_mismatched_active_marker(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Bad marker target",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    other_session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Bad marker source",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    other_session_action = repo.create_session_action(
        session_id=other_session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="restore-other-session",
        request_payload_hash="restore-other-session-hash",
    )
    repo.update_session(
        session["id"],
        {"active_session_action_id": other_session_action["id"]},
        owner_user_id=42,
    )

    assert repo.latest_active_session_action(
        session_id=session["id"],
        owner_user_id=42,
    ) is None


def test_update_session_action_is_owner_scoped_and_keeps_identity_fields_immutable(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Immutable action",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    action = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="restore-immutable",
        request_payload_hash="restore-original-hash",
    )

    wrong_owner_update = repo.update_session_action(
        action["id"],
        {"status": "succeeded"},
        owner_user_id=43,
    )
    assert wrong_owner_update is None
    assert repo.get_session_action(action["id"])["status"] == "pending"

    updated = repo.update_session_action(
        action["id"],
        {
            "action_type": "checkpoint_restore",
            "request_payload_hash": "restore-mutated-hash",
            "status": "abandoned",
        },
        owner_user_id=42,
    )
    assert updated["status"] == "abandoned"
    assert updated["action_type"] == "branch_restore"
    assert updated["request_payload_hash"] == "restore-original-hash"


def test_scene_branches_and_checkpoints_round_trip_json(chacha_db: CharactersRAGDB) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Museum branch",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
        seed="seed-2",
        settings={"temperature": 0.7},
    )
    event = repo.append_event(
        session_id=session["id"],
        owner_user_id=42,
        event_type="scene_changed",
        event_payload={"location_key": "museum"},
    )

    state = repo.set_scene_state(
        session_id=session["id"],
        owner_user_id=42,
        last_event_id=event["id"],
        current_background_item_id=100,
        current_depth_item_id=101,
        active_sprite_items=[{"character_id": 1, "item_id": 200}],
        location_key="museum",
        mood="quiet",
        visible_choices=[{"id": "choice-1", "label": "Ask about the exhibit"}],
        scene_version=1,
        warnings=["low_confidence_asset"],
    )
    branch = repo.create_branch(
        session_id=session["id"],
        owner_user_id=42,
        parent_event_id=event["id"],
        branch_label="Ask about the exhibit",
        branch_path=["choice-1"],
    )
    checkpoint = repo.create_checkpoint(
        session_id=session["id"],
        owner_user_id=42,
        label="Before the question",
        event_id=event["id"],
        scene_version=1,
        scene_state_snapshot=state,
    )

    assert state["active_sprite_items"] == [{"character_id": 1, "item_id": 200}]
    assert state["visible_choices"] == [{"id": "choice-1", "label": "Ask about the exhibit"}]
    assert state["warnings"] == ["low_confidence_asset"]
    assert repo.get_scene_state(session["id"])["location_key"] == "museum"
    assert repo.list_branches(session["id"]) == [branch]
    assert repo.list_checkpoints(session["id"]) == [checkpoint]


def test_record_story_choice_selection_creates_branch_event_turn_and_state(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Door",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    choice_presented = repo.append_event(
        session_id=session["id"],
        owner_user_id=42,
        event_type="choice_presented",
        event_payload={
            "choices": [{"id": "open", "text": "Open the door"}],
            "scene_version": 1,
        },
    )
    repo.set_scene_state(
        session_id=session["id"],
        owner_user_id=42,
        last_event_id=choice_presented["id"],
        visible_choices=[{"id": "open", "text": "Open the door"}],
        current_background_item_id=100,
        current_depth_item_id=101,
        active_sprite_items=[{"character_id": 1, "item_id": 200}],
        location_key="hall",
        mood="curious",
        time_of_day="night",
        weather="rain",
        transcript_cursor=7,
        scene_version=1,
        warnings=["low_confidence_asset"],
    )
    repo.update_session(session["id"], {"scene_version": 1}, owner_user_id=42)
    turn = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="choice-1",
        request_payload_hash="hash-choice-1",
        base_scene_version=1,
    )

    result = repo.record_story_choice_selection(
        session_id=session["id"],
        owner_user_id=42,
        turn_request_id=turn["id"],
        client_scene_version=1,
        selected_choice={"id": "open", "text": "Open the door"},
        parent_event_id=choice_presented["id"],
        expected_scene_last_event_id=choice_presented["id"],
        branch_label="Open the door",
        branch_path=[
            {
                "schema_version": 1,
                "type": "choice",
                "choice_id": "open",
                "choice_text": "Open the door",
                "choice_presented_event_id": choice_presented["id"],
                "scene_version": 1,
            }
        ],
    )

    assert result["branch"]["branch_path"][0]["choice_id"] == "open"
    assert result["turn_started"]["event_type"] == "turn_started"
    assert result["choice_selected"]["event_type"] == "choice_selected"
    assert result["choice_selected"]["branch_node_id"] == result["branch"]["id"]

    state = repo.get_scene_state(session["id"], owner_user_id=42)
    assert state["last_event_id"] == result["choice_selected"]["id"]
    assert state["active_branch_node_id"] == result["branch"]["id"]
    assert state["visible_choices"] == []
    assert state["current_background_item_id"] == 100
    assert state["current_depth_item_id"] == 101
    assert state["active_sprite_items"] == [{"character_id": 1, "item_id": 200}]
    assert state["location_key"] == "hall"
    assert state["mood"] == "curious"
    assert state["time_of_day"] == "night"
    assert state["weather"] == "rain"
    assert state["transcript_cursor"] == 7
    assert state["scene_version"] == 1
    assert state["warnings"] == ["low_confidence_asset"]
    assert result["scene_state"] == state

    updated_turn = repo.get_turn_request(turn["id"])
    assert updated_turn["status"] == "model_calling"
    assert updated_turn["turn_started_event_id"] == result["turn_started"]["id"]
    assert updated_turn["input_event_id"] == result["choice_selected"]["id"]


def test_record_story_choice_selection_bounds_branch_metadata(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Door",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    long_choice_text = "Open " + ("the sealed archive door " * 20)
    choice_presented = repo.append_event(
        session_id=session["id"],
        owner_user_id=42,
        event_type="choice_presented",
        event_payload={
            "choices": [{"id": "open", "text": long_choice_text}],
            "scene_version": 1,
        },
    )
    repo.set_scene_state(
        session_id=session["id"],
        owner_user_id=42,
        last_event_id=choice_presented["id"],
        visible_choices=[{"id": "open", "text": long_choice_text}],
        scene_version=1,
    )
    repo.update_session(session["id"], {"scene_version": 1}, owner_user_id=42)
    turn = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="choice-long",
        request_payload_hash="hash-choice-long",
        base_scene_version=1,
    )

    result = repo.record_story_choice_selection(
        session_id=session["id"],
        owner_user_id=42,
        turn_request_id=turn["id"],
        client_scene_version=1,
        selected_choice={"id": "open", "text": long_choice_text},
        parent_event_id=choice_presented["id"],
        branch_label=long_choice_text,
        branch_path=[
            {
                "schema_version": 1,
                "type": "choice",
                "choice_id": "open",
                "choice_text": long_choice_text,
                "choice_presented_event_id": choice_presented["id"],
                "scene_version": 1,
            }
        ],
        expected_scene_last_event_id=choice_presented["id"],
    )

    expected_label = long_choice_text[:STORY_BRANCH_LABEL_MAX_LENGTH]
    assert result["branch"]["branch_label"] == expected_label
    assert result["branch"]["branch_path"][0]["choice_text"] == expected_label


def test_record_story_choice_selection_rejects_choice_not_visible_without_mutations(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Door",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    choice_presented = repo.append_event(
        session_id=session["id"],
        owner_user_id=42,
        event_type="choice_presented",
        event_payload={
            "choices": [{"id": "open", "text": "Open the door"}],
            "scene_version": 1,
        },
    )
    repo.set_scene_state(
        session_id=session["id"],
        owner_user_id=42,
        last_event_id=choice_presented["id"],
        visible_choices=[{"id": "wait", "text": "Wait outside"}],
        scene_version=1,
    )
    turn = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="choice-1",
        request_payload_hash="hash-choice-1",
        base_scene_version=1,
    )
    branches_before = repo.list_branches(session["id"], owner_user_id=42)
    events_before = repo.list_events(session["id"])
    state_before = repo.get_scene_state(session["id"], owner_user_id=42)

    with pytest.raises(RuntimeError, match="choice_not_visible"):
        repo.record_story_choice_selection(
            session_id=session["id"],
            owner_user_id=42,
            turn_request_id=turn["id"],
            client_scene_version=1,
            selected_choice={"id": "open", "text": "Open the door"},
            parent_event_id=choice_presented["id"],
            expected_scene_last_event_id=choice_presented["id"],
            branch_label="Open the door",
            branch_path=[{"choice_id": "open"}],
        )

    assert repo.list_branches(session["id"], owner_user_id=42) == branches_before
    assert repo.list_events(session["id"]) == events_before
    assert repo.get_scene_state(session["id"], owner_user_id=42) == state_before
    updated_turn = repo.get_turn_request(turn["id"])
    assert updated_turn["status"] == "pending"
    assert updated_turn["turn_started_event_id"] is None
    assert updated_turn["input_event_id"] is None


def test_record_story_choice_selection_rejects_moved_scene_state_without_mutations(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Door",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    choice_presented = repo.append_event(
        session_id=session["id"],
        owner_user_id=42,
        event_type="choice_presented",
        event_payload={
            "choices": [{"id": "open", "text": "Open the door"}],
            "scene_version": 1,
        },
    )
    scene_moved = repo.append_event(
        session_id=session["id"],
        owner_user_id=42,
        event_type="scene_state_changed",
        event_payload={"scene_version": 1},
    )
    repo.set_scene_state(
        session_id=session["id"],
        owner_user_id=42,
        last_event_id=scene_moved["id"],
        visible_choices=[{"id": "open", "text": "Open the door"}],
        scene_version=1,
    )
    turn = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="choice-1",
        request_payload_hash="hash-choice-1",
        base_scene_version=1,
    )
    branches_before = repo.list_branches(session["id"], owner_user_id=42)
    events_before = repo.list_events(session["id"])
    state_before = repo.get_scene_state(session["id"], owner_user_id=42)

    with pytest.raises(RuntimeError, match="scene_state_moved"):
        repo.record_story_choice_selection(
            session_id=session["id"],
            owner_user_id=42,
            turn_request_id=turn["id"],
            client_scene_version=1,
            selected_choice={"id": "open", "text": "Open the door"},
            parent_event_id=choice_presented["id"],
            expected_scene_last_event_id=choice_presented["id"],
            branch_label="Open the door",
            branch_path=[{"choice_id": "open"}],
        )

    assert repo.list_branches(session["id"], owner_user_id=42) == branches_before
    assert repo.list_events(session["id"]) == events_before
    assert repo.get_scene_state(session["id"], owner_user_id=42) == state_before
    updated_turn = repo.get_turn_request(turn["id"])
    assert updated_turn["status"] == "pending"
    assert updated_turn["turn_started_event_id"] is None
    assert updated_turn["input_event_id"] is None


def test_record_story_choice_selection_rejects_replayed_turn_without_mutations(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Door",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    choice_presented = repo.append_event(
        session_id=session["id"],
        owner_user_id=42,
        event_type="choice_presented",
        event_payload={
            "choices": [{"id": "open", "text": "Open the door"}],
            "scene_version": 1,
        },
    )
    repo.set_scene_state(
        session_id=session["id"],
        owner_user_id=42,
        last_event_id=choice_presented["id"],
        visible_choices=[{"id": "open", "text": "Open the door"}],
        scene_version=1,
    )
    turn = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="choice-1",
        request_payload_hash="hash-choice-1",
        base_scene_version=1,
    )
    first_result = repo.record_story_choice_selection(
        session_id=session["id"],
        owner_user_id=42,
        turn_request_id=turn["id"],
        client_scene_version=1,
        selected_choice={"id": "open", "text": "Open the door"},
        parent_event_id=choice_presented["id"],
        expected_scene_last_event_id=choice_presented["id"],
        branch_label="Open the door",
        branch_path=[{"choice_id": "open"}],
    )
    branches_before = repo.list_branches(session["id"], owner_user_id=42)
    events_before = repo.list_events(session["id"])
    state_before = repo.get_scene_state(session["id"], owner_user_id=42)

    with pytest.raises(RuntimeError, match="turn_request_not_pending"):
        repo.record_story_choice_selection(
            session_id=session["id"],
            owner_user_id=42,
            turn_request_id=turn["id"],
            client_scene_version=1,
            selected_choice={"id": "open", "text": "Open the door"},
            parent_event_id=choice_presented["id"],
            expected_scene_last_event_id=first_result["choice_selected"]["id"],
            branch_label="Open the door again",
            branch_path=[{"choice_id": "open", "replayed": True}],
        )

    assert repo.list_branches(session["id"], owner_user_id=42) == branches_before
    assert repo.list_events(session["id"]) == events_before
    assert repo.get_scene_state(session["id"], owner_user_id=42) == state_before
    updated_turn = repo.get_turn_request(turn["id"])
    assert updated_turn["turn_started_event_id"] == first_result["turn_started"]["id"]
    assert updated_turn["input_event_id"] == first_result["choice_selected"]["id"]


def test_record_story_choice_selection_rejects_scene_version_mismatch_without_mutations(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Door",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    choice_presented = repo.append_event(
        session_id=session["id"],
        owner_user_id=42,
        event_type="choice_presented",
        event_payload={
            "choices": [{"id": "open", "text": "Open the door"}],
            "scene_version": 1,
        },
    )
    repo.set_scene_state(
        session_id=session["id"],
        owner_user_id=42,
        last_event_id=choice_presented["id"],
        visible_choices=[{"id": "open", "text": "Open the door"}],
        scene_version=1,
    )
    turn = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="choice-1",
        request_payload_hash="hash-choice-1",
        base_scene_version=1,
    )
    branches_before = repo.list_branches(session["id"], owner_user_id=42)
    events_before = repo.list_events(session["id"])
    state_before = repo.get_scene_state(session["id"], owner_user_id=42)

    with pytest.raises(RuntimeError, match="turn_request_not_pending"):
        repo.record_story_choice_selection(
            session_id=session["id"],
            owner_user_id=42,
            turn_request_id=turn["id"],
            client_scene_version=2,
            selected_choice={"id": "open", "text": "Open the door"},
            parent_event_id=choice_presented["id"],
            expected_scene_last_event_id=choice_presented["id"],
            branch_label="Open the door",
            branch_path=[{"choice_id": "open", "scene_version": 2}],
        )

    assert repo.list_branches(session["id"], owner_user_id=42) == branches_before
    assert repo.list_events(session["id"]) == events_before
    assert repo.get_scene_state(session["id"], owner_user_id=42) == state_before
    updated_turn = repo.get_turn_request(turn["id"])
    assert updated_turn["status"] == "pending"
    assert updated_turn["turn_started_event_id"] is None
    assert updated_turn["input_event_id"] is None


def test_ensure_vn_play_tables_preserves_outer_transaction_rollback(
    chacha_db: CharactersRAGDB,
) -> None:
    title = "Rolled back VN session"

    with pytest.raises(RuntimeError, match="force rollback"):
        with chacha_db.transaction() as conn:
            ensure_vn_play_tables(chacha_db)
            conn.execute(
                """
                INSERT INTO vn_play_sessions (
                    owner_user_id,
                    mode,
                    title,
                    primary_character_id,
                    vn_asset_pack_id
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (42, "freeform", title, 1, 10),
            )
            raise RuntimeError("force rollback")

    ensure_vn_play_tables(chacha_db)
    cursor = chacha_db.execute_query(
        "SELECT id FROM vn_play_sessions WHERE title = ?",
        (title,),
    )
    assert cursor.fetchone() is None
