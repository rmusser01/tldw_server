from collections.abc import Generator
from typing import Any

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


def test_initialized_creates_generation_persistence_tables_on_fresh_and_upgraded_db(
    chacha_db: CharactersRAGDB,
) -> None:
    ensure_vn_play_tables(chacha_db)

    table_cursor = chacha_db.execute_query(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name LIKE 'vn_play_generation%'
        """
    )
    assert {row["name"] for row in table_cursor.fetchall()} == {
        "vn_play_generations",
        "vn_play_generation_requests",
        "vn_play_generation_actions",
        "vn_play_generation_revisions",
    }

    with chacha_db.transaction() as conn:
        conn.execute("DROP TABLE vn_play_generation_revisions")
        conn.execute("DROP TABLE vn_play_generation_actions")
        conn.execute("DROP TABLE vn_play_generation_requests")
        conn.execute("DROP TABLE vn_play_generations")
        conn.execute(
            """
            CREATE TABLE vn_play_generations (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL,
                owner_user_id INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE vn_play_generation_requests (
                id INTEGER PRIMARY KEY,
                generation_id INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                owner_user_id INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE vn_play_generation_actions (
                id INTEGER PRIMARY KEY,
                session_id INTEGER NOT NULL,
                owner_user_id INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE vn_play_generation_revisions (
                id INTEGER PRIMARY KEY,
                generation_id INTEGER NOT NULL,
                generation_request_id INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                owner_user_id INTEGER NOT NULL
            )
            """
        )
        conn.execute("INSERT INTO vn_play_generations (id, session_id, owner_user_id) VALUES (1, 1, 42)")
        conn.execute(
            """
            INSERT INTO vn_play_generation_requests (id, generation_id, session_id, owner_user_id)
            VALUES (1, 1, 1, 42)
            """
        )
        conn.execute(
            """
            INSERT INTO vn_play_generation_actions (id, session_id, owner_user_id)
            VALUES (1, 1, 42)
            """
        )
        conn.execute(
            """
            INSERT INTO vn_play_generation_revisions (
                id,
                generation_id,
                generation_request_id,
                session_id,
                owner_user_id
            )
            VALUES (1, 1, 1, 1, 42)
            """
        )

    ensure_vn_play_tables(chacha_db)

    column_cursor = chacha_db.execute_query("PRAGMA table_info(vn_play_generations)")
    column_names = {row["name"] for row in column_cursor.fetchall()}
    assert {
        "generation_point_key",
        "generation_profile_key",
        "generation_profile_snapshot_id",
        "active_revision_id",
        "latest_request_id",
        "status",
    }.issubset(column_names)
    action_column_cursor = chacha_db.execute_query("PRAGMA table_info(vn_play_generation_actions)")
    action_column_names = {row["name"] for row in action_column_cursor.fetchall()}
    assert {"created_at", "updated_at", "request_payload_hash"}.issubset(action_column_names)


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


def test_recover_expired_active_turn_lock_abandons_request_and_clears_lock(
    chacha_db: CharactersRAGDB,
) -> None:
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
    turn = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="stale-turn",
        request_payload_hash="hash-a",
        base_scene_version=0,
        status="model_calling",
    )
    repo.update_turn_request(
        turn["id"],
        {"locked_until": "2000-01-01 00:00:00", "lease_owner": "worker-1"},
        owner_user_id=42,
    )
    repo.update_session(
        session["id"],
        {"active_turn_request_id": int(turn["id"])},
        owner_user_id=42,
    )

    recovered = repo.recover_expired_active_turn_lock(
        session_id=session["id"],
        owner_user_id=42,
        error_code="turn_lock_abandoned",
    )

    assert recovered is not None
    assert recovered["id"] == turn["id"]
    assert recovered["status"] == "abandoned"
    assert recovered["error"] == {"code": "turn_lock_abandoned"}
    assert recovered["locked_until"] is None
    assert recovered["lease_owner"] is None
    assert repo.get_session(session["id"], owner_user_id=42)["active_turn_request_id"] is None


def test_recover_expired_active_turn_lock_preserves_fresh_and_terminal_turns(
    chacha_db: CharactersRAGDB,
) -> None:
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
    fresh = repo.create_turn_request(
        session_id=session["id"],
        owner_user_id=42,
        idempotency_key="fresh-turn",
        request_payload_hash="hash-a",
        base_scene_version=0,
        status="model_calling",
    )
    repo.update_turn_request(
        fresh["id"],
        {"locked_until": "2999-01-01 00:00:00", "lease_owner": "worker-1"},
        owner_user_id=42,
    )
    repo.update_session(
        session["id"],
        {"active_turn_request_id": int(fresh["id"])},
        owner_user_id=42,
    )

    assert repo.recover_expired_active_turn_lock(
        session_id=session["id"],
        owner_user_id=42,
        error_code="turn_lock_abandoned",
    ) is None
    assert repo.get_session(session["id"], owner_user_id=42)["active_turn_request_id"] == fresh["id"]
    assert repo.get_turn_request(fresh["id"])["status"] == "model_calling"

    completed = repo.update_turn_request(
        fresh["id"],
        {"status": "completed", "locked_until": "2000-01-01 00:00:00"},
        owner_user_id=42,
    )

    assert repo.recover_expired_active_turn_lock(
        session_id=session["id"],
        owner_user_id=42,
        error_code="turn_lock_abandoned",
    ) is None
    assert repo.get_session(session["id"], owner_user_id=42)["active_turn_request_id"] == fresh["id"]
    assert repo.get_turn_request(fresh["id"])["status"] == completed["status"]


def test_generation_point_is_unique_per_owner_session_and_key(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Generated branch",
        primary_character_id=1,
        vn_asset_pack_id=10,
        script_id=7,
        script_version_id=8,
        seed="seed",
    )
    other_session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Generated branch again",
        primary_character_id=1,
        vn_asset_pack_id=10,
        script_id=7,
        script_version_id=8,
        seed="seed",
    )
    first = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:3:map-clue",
        script_id=7,
        script_version_id=8,
        opcode_id="map-clue",
        opcode_label="archive",
        opcode_index=3,
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    replayed = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:3:map-clue",
        script_id=7,
        script_version_id=8,
        opcode_id="map-clue",
        opcode_label="archive",
        opcode_index=3,
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    other = repo.get_or_create_generation(
        session_id=other_session["id"],
        owner_user_id=42,
        generation_point_key="archive:3:map-clue",
        script_id=7,
        script_version_id=8,
        opcode_id="map-clue",
        opcode_label="archive",
        opcode_index=3,
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )

    assert replayed["id"] == first["id"]
    assert first["generation_point_key"] == "archive:3:map-clue"
    assert first["status"] == "not_started"
    assert other["id"] != first["id"]

    with pytest.raises(ValueError, match="generation_point_conflict"):
        repo.get_or_create_generation(
            session_id=session["id"],
            owner_user_id=42,
            generation_point_key="archive:3:map-clue",
            script_id=7,
            script_version_id=8,
            opcode_id="map-clue",
            opcode_label="archive",
            opcode_index=3,
            output_schema="scene_update",
            generation_profile_key="scene_writer",
            generation_profile_snapshot_id=45,
        )


def test_generation_action_idempotency_replays_and_rejects_changed_hash_or_kind(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Generated action",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    generation = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:3:map-clue",
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        request_kind="confirmation",
        client_scene_version=2,
        opcode_snapshot={"op": "generate"},
        prompt_fingerprint="prompt-a",
        checkpoint_id_before=11,
        status="pending_confirmation",
    )
    first = repo.create_generation_action(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        generation_request_id=request["id"],
        action_kind="execute",
        idempotency_key="generation-confirm-1",
        request_payload_hash="hash-a",
    )
    repo.update_generation_action(
        first["id"],
        {
            "status": "completed",
            "completed_action_response": {"generation_request_id": request["id"]},
        },
        owner_user_id=42,
    )

    replayed = repo.create_generation_action(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        generation_request_id=request["id"],
        action_kind="execute",
        idempotency_key="generation-confirm-1",
        request_payload_hash="hash-a",
    )

    assert replayed["id"] == first["id"]
    assert replayed["completed_action_response"] == {
        "generation_request_id": request["id"]
    }

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_generation_action(
            session_id=session["id"],
            owner_user_id=42,
            generation_id=generation["id"],
            generation_request_id=request["id"],
            action_kind="execute",
            idempotency_key="generation-confirm-1",
            request_payload_hash="hash-b",
        )

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_generation_action(
            session_id=session["id"],
            owner_user_id=42,
            generation_id=generation["id"],
            generation_request_id=request["id"],
            action_kind="cancel",
            idempotency_key="generation-confirm-1",
            request_payload_hash="hash-a",
        )

    second_request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        request_kind="regenerate",
        client_scene_version=3,
        status="pending",
    )
    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_generation_action(
            session_id=session["id"],
            owner_user_id=42,
            generation_id=generation["id"],
            generation_request_id=second_request["id"],
            action_kind="execute",
            idempotency_key="generation-confirm-1",
            request_payload_hash="hash-a",
        )


def test_generation_action_rejects_revision_from_different_generation(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Generated action mismatch",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    generation = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:3:map-clue",
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    other_generation = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:4:other",
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        request_kind="regenerate",
        client_scene_version=2,
        status="completed",
    )
    other_request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=other_generation["id"],
        request_kind="regenerate",
        client_scene_version=2,
        status="completed",
    )
    other_revision = repo.create_generation_revision(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=other_generation["id"],
        generation_request_id=other_request["id"],
        status="succeeded",
        output_schema="choice_set",
        public_output={"choices": []},
    )

    with pytest.raises(ValueError, match="generation_revision_mismatch"):
        repo.create_generation_action(
            session_id=session["id"],
            owner_user_id=42,
            generation_id=generation["id"],
            generation_request_id=request["id"],
            generation_revision_id=other_revision["id"],
            action_kind="activate",
            idempotency_key="activate-mismatch",
            request_payload_hash="hash-a",
        )


def test_generation_relation_updates_reject_cross_generation_links(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Generated relation guards",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    generation = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:3:map-clue",
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    other_generation = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:4:other",
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        request_kind="automatic",
        client_scene_version=2,
        status="in_progress",
    )
    other_request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=other_generation["id"],
        request_kind="automatic",
        client_scene_version=2,
        status="in_progress",
    )
    action = repo.create_generation_action(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=other_generation["id"],
        generation_request_id=other_request["id"],
        action_kind="execute",
        idempotency_key="other-execute",
        request_payload_hash="hash-a",
    )

    with pytest.raises(ValueError, match="generation_action_mismatch"):
        repo.update_generation_request(
            request["id"],
            {"execute_action_id": action["id"]},
            owner_user_id=42,
        )

    with pytest.raises(ValueError, match="generation_request_mismatch"):
        repo.update_generation_action(
            action["id"],
            {
                "generation_id": generation["id"],
                "generation_request_id": other_request["id"],
            },
            owner_user_id=42,
        )


def test_generation_request_status_update_syncs_parent_generation_status(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Generated status sync",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    generation = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:3:map-clue",
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        request_kind="automatic",
        client_scene_version=2,
        status="in_progress",
    )

    repo.update_generation_request(
        request["id"],
        {"status": "model_failed", "public_error_code": "provider_unavailable"},
        owner_user_id=42,
    )

    updated_generation = repo.get_generation(generation["id"], owner_user_id=42)
    assert updated_generation["status"] == "model_failed"


def test_generation_request_status_update_does_not_regress_from_stale_request(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Generated stale status",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    generation = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:3:map-clue",
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    stale_request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        request_kind="automatic",
        client_scene_version=2,
        status="in_progress",
    )
    latest_request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        request_kind="regenerate",
        client_scene_version=3,
        status="pending_confirmation",
    )

    repo.update_generation_request(
        stale_request["id"],
        {"status": "abandoned"},
        owner_user_id=42,
    )

    updated_generation = repo.get_generation(generation["id"], owner_user_id=42)
    assert updated_generation["latest_request_id"] == latest_request["id"]
    assert updated_generation["status"] == "pending_confirmation"


def test_generation_requests_and_revisions_store_public_error_debug_and_usage_json(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Generated request",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    generation = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:3:map-clue",
        output_schema="narrative_dialogue",
        generation_profile_key="default",
        generation_profile_snapshot_id=44,
    )
    request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        request_kind="automatic",
        client_scene_version=2,
        opcode_snapshot={"op": "generate", "profile_key": "default"},
        prompt_fingerprint="prompt-a",
        checkpoint_id_before=11,
        status="in_progress",
    )
    updated_request = repo.update_generation_request(
        request["id"],
        {
            "status": "model_failed",
            "public_error_code": "provider_unavailable",
            "provider_call_started_at": "2026-05-10T20:00:00Z",
            "provider_call_completed_at": "2026-05-10T20:00:01Z",
            "lease_expires_at": "2026-05-10T20:05:00Z",
        },
        owner_user_id=42,
    )
    revision = repo.create_generation_revision(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        generation_request_id=request["id"],
        status="model_failed",
        output_schema="narrative_dialogue",
        public_output={},
        public_error_code="provider_unavailable",
        raw_output_debug={"error": "timeout"},
        parser_diagnostics={"stage": "not_started"},
        moderation_diagnostics={"skipped": True},
        model_metadata={"provider": "mock"},
        usage_metadata={"prompt_tokens": 17},
        source="model",
    )

    assert updated_request["public_error_code"] == "provider_unavailable"
    assert updated_request["opcode_snapshot"] == {
        "op": "generate",
        "profile_key": "default",
    }
    assert revision["revision_number"] == 1
    assert revision["public_error_code"] == "provider_unavailable"
    assert revision["raw_output_debug"] == {"error": "timeout"}
    assert revision["parser_diagnostics"] == {"stage": "not_started"}
    assert revision["moderation_diagnostics"] == {"skipped": True}
    assert revision["model_metadata"] == {"provider": "mock"}
    assert revision["usage_metadata"] == {"prompt_tokens": 17}


def test_active_revision_requires_succeeded_revision_from_same_generation(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Generated activation",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    generation = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:3:map-clue",
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    foreign_generation = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:4:other",
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        request_kind="regenerate",
        client_scene_version=2,
        status="completed",
    )
    foreign_request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=foreign_generation["id"],
        request_kind="regenerate",
        client_scene_version=2,
        status="completed",
    )
    failed_revision = repo.create_generation_revision(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        generation_request_id=request["id"],
        status="parse_failed",
        output_schema="choice_set",
        public_error_code="invalid_model_output",
    )
    succeeded_revision = repo.create_generation_revision(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        generation_request_id=request["id"],
        status="succeeded",
        output_schema="choice_set",
        public_output={"choices": []},
    )
    foreign_revision = repo.create_generation_revision(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=foreign_generation["id"],
        generation_request_id=foreign_request["id"],
        status="succeeded",
        output_schema="choice_set",
        public_output={"choices": []},
    )

    with pytest.raises(ValueError, match="active_revision_not_succeeded"):
        repo.set_active_generation_revision(
            generation_id=generation["id"],
            owner_user_id=42,
            revision_id=failed_revision["id"],
        )
    with pytest.raises(ValueError, match="active_revision_generation_mismatch"):
        repo.set_active_generation_revision(
            generation_id=generation["id"],
            owner_user_id=42,
            revision_id=foreign_revision["id"],
        )

    updated = repo.set_active_generation_revision(
        generation_id=generation["id"],
        owner_user_id=42,
        revision_id=succeeded_revision["id"],
    )

    assert updated["active_revision_id"] == succeeded_revision["id"]
    assert updated["status"] == "completed"


def test_generation_revision_listing_uses_stable_offset_order(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Generated history",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    generation = repo.get_or_create_generation(
        session_id=session["id"],
        owner_user_id=42,
        generation_point_key="archive:3:map-clue",
        output_schema="choice_set",
        generation_profile_key="choice_writer",
        generation_profile_snapshot_id=44,
    )
    request = repo.create_generation_request(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        request_kind="regenerate",
        client_scene_version=2,
        status="completed",
    )
    first = repo.create_generation_revision(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        generation_request_id=request["id"],
        status="succeeded",
        output_schema="choice_set",
        public_output={"revision": 1},
        source="model",
    )
    second = repo.create_generation_revision(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        generation_request_id=request["id"],
        status="succeeded",
        output_schema="choice_set",
        public_output={"revision": 2},
        source="regenerated",
    )
    third = repo.create_generation_revision(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        generation_request_id=request["id"],
        status="succeeded",
        output_schema="choice_set",
        public_output={"revision": 3},
        source="regenerated",
    )

    page = repo.list_generation_revisions(
        session_id=session["id"],
        owner_user_id=42,
        generation_id=generation["id"],
        limit=2,
        offset=1,
    )

    assert [revision["id"] for revision in page] == [second["id"], first["id"]]
    assert [revision["revision_number"] for revision in [first, second, third]] == [
        1,
        2,
        3,
    ]


def test_session_action_duplicate_insert_replays_or_conflicts(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Race target",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    first = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="restore-race",
        request_payload_hash="branch_restore:hash-a",
    )
    original_get = repo.get_session_action_by_key
    hidden_reads = 0

    def hide_first_read(*args: Any, **kwargs: Any) -> dict[str, Any] | None:
        nonlocal hidden_reads
        if hidden_reads == 0:
            hidden_reads += 1
            return None
        return original_get(*args, **kwargs)

    monkeypatch.setattr(repo, "get_session_action_by_key", hide_first_read)

    replayed = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="restore-race",
        request_payload_hash="branch_restore:hash-a",
    )

    assert replayed["id"] == first["id"]

    conflict = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="restore-race-conflict",
        request_payload_hash="branch_restore:hash-a",
    )
    hidden_reads = 0

    with pytest.raises(ValueError, match="idempotency_key_conflict"):
        repo.create_session_action(
            session_id=session["id"],
            owner_user_id=42,
            action_type="branch_restore",
            idempotency_key="restore-race-conflict",
            request_payload_hash="branch_restore:hash-b",
        )
    assert conflict["id"] != first["id"]


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


def test_session_action_terminal_update_clears_only_matching_lock(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Terminal action",
        primary_character_id=1,
        vn_asset_pack_id=10,
        seed="seed",
    )
    active_action = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="active-action",
        request_payload_hash="active-hash",
    )
    other_action = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="branch_restore",
        idempotency_key="other-action",
        request_payload_hash="other-hash",
    )
    repo.update_session(
        session["id"],
        {"active_session_action_id": active_action["id"]},
        owner_user_id=42,
    )

    failed_other = repo.mark_session_action_terminal(
        session_id=session["id"],
        owner_user_id=42,
        action_id=other_action["id"],
        status="failed",
        error={"code": "branch_restore_target_unavailable"},
    )

    assert failed_other["status"] == "failed"
    assert failed_other["error"] == {"code": "branch_restore_target_unavailable"}
    assert (
        repo.get_session(session["id"], owner_user_id=42)["active_session_action_id"]
        == active_action["id"]
    )

    abandoned_active = repo.mark_session_action_terminal(
        session_id=session["id"],
        owner_user_id=42,
        action_id=active_action["id"],
        status="abandoned",
        error={"code": "stale_scene_version"},
    )

    assert abandoned_active["status"] == "abandoned"
    assert abandoned_active["error"] == {"code": "stale_scene_version"}
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


def test_restore_active_generation_revision_map_rejects_unknown_point_key(
    chacha_db: CharactersRAGDB,
) -> None:
    repo = VNPlayRepository.initialized(chacha_db)
    session = repo.create_session(
        owner_user_id=42,
        mode="story",
        title="Restore generated choice",
        primary_character_id=1,
        vn_asset_pack_id=10,
        content_rating="general",
    )
    action = repo.create_session_action(
        session_id=session["id"],
        owner_user_id=42,
        action_type="checkpoint_restore",
        idempotency_key="restore-unknown-generation",
        request_payload_hash="restore-unknown-generation-hash",
    )
    assert repo.try_acquire_session_action_lock(
        session_id=session["id"],
        owner_user_id=42,
        action_id=action["id"],
        expected_scene_version=0,
    ) is True

    with pytest.raises(ValueError, match="generation_point_not_found"):
        repo.commit_session_restore_action(
            session_id=session["id"],
            owner_user_id=42,
            action_id=action["id"],
            event_payload={"restore": "checkpoint"},
            scene_state={},
            scene_version=1,
            response_payload_factory=lambda payload: {"event_id": payload["event"]["id"]},
            active_generation_revisions={"missing:point": None},
        )


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
