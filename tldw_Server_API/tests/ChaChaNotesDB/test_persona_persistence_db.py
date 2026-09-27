from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.Characters.test_character_functionality_db import sample_card_data

pytestmark = pytest.mark.unit


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a temporary SQLite path for persona persistence database tests."""
    return tmp_path / "persona_persistence_test.sqlite"


@pytest.fixture
def db_instance(db_path: Path) -> Iterator[CharactersRAGDB]:
    """Provide a CharactersRAGDB instance and close it after each test."""
    db = CharactersRAGDB(db_path, "persona-persistence-test-client")
    yield db
    db.close_connection()


def _seed_historical_sqlite(db_path: Path, version: int, monkeypatch: pytest.MonkeyPatch) -> CharactersRAGDB:
    """Build the real historical schema without current compatibility repairs."""

    def initialize_historical(db: CharactersRAGDB) -> None:
        with db.transaction() as conn:
            db._apply_schema_v4(conn)
            steps = db._sqlite_linear_migration_steps()
            for prior in range(4, version):
                steps[prior](conn)
                assert db._get_db_version(conn) == prior + 1

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", version)
        patch.setattr(CharactersRAGDB, "_initialize_schema", initialize_historical)
        return CharactersRAGDB(db_path, "historical-seed-client")


def test_migration_v25_to_latest_creates_persona_tables(db_path: Path, monkeypatch: pytest.MonkeyPatch):
    seeded = _seed_historical_sqlite(db_path, 25, monkeypatch)
    try:
        with seeded.transaction() as conn:
            assert seeded._get_db_version(conn) == 25
            tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert {
                "persona_profiles",
                "persona_scope_rules",
                "persona_policy_rules",
                "persona_sessions",
                "persona_memory_entries",
                "note_attachments",
            }.isdisjoint(tables)
            conn.execute(
                "INSERT INTO character_cards (name, description, client_id) VALUES (?, ?, ?)",
                ("Historical v25 character", "Retained character content", "historical-seed-client"),
            )
            character_before = dict(conn.execute("SELECT * FROM character_cards").fetchone())
            # Prove the exact persona-table introduction before the current-head upgrade.
            seeded._migrate_from_v25_to_v26(conn)
            assert seeded._get_db_version(conn) == 26
            tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert {
                "persona_profiles",
                "persona_scope_rules",
                "persona_policy_rules",
                "persona_sessions",
                "persona_memory_entries",
            }.issubset(tables)
            conn.execute(
                "INSERT INTO persona_profiles (id, user_id, name, system_prompt) VALUES (?, ?, ?, ?)",
                ("historical-persona", "1", "Historical persona", "Retained persona prompt"),
            )
            conn.execute(
                "INSERT INTO persona_memory_entries (id, persona_id, user_id, memory_type, content) "
                "VALUES (?, ?, ?, ?, ?)",
                ("historical-memory", "historical-persona", "1", "fact", "Retained memory content"),
            )
            persona_before = dict(conn.execute("SELECT * FROM persona_profiles").fetchone())
            memory_before = dict(conn.execute("SELECT * FROM persona_memory_entries").fetchone())
    finally:
        seeded.close_all_connections()

    migrated = CharactersRAGDB(db_path, "migration-check-client")
    conn = migrated.get_connection()

    version = conn.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()["version"]
    assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION

    for table_name in (
        "persona_profiles",
        "persona_scope_rules",
        "persona_policy_rules",
        "persona_sessions",
        "persona_memory_entries",
    ):
        found = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table_name,),
        ).fetchone()
        assert found is not None

    profile_indexes = {row["name"] for row in conn.execute("PRAGMA index_list('persona_profiles')").fetchall()}
    assert "idx_persona_profiles_user_active" in profile_indexes
    profile_columns = {row["name"] for row in conn.execute("PRAGMA table_info('persona_profiles')").fetchall()}
    assert "use_persona_state_context_default" in profile_columns
    assert "origin_character_id" in profile_columns
    assert "origin_character_name" in profile_columns
    assert "origin_character_snapshot_at" in profile_columns
    memory_columns = {row["name"] for row in conn.execute("PRAGMA table_info('persona_memory_entries')").fetchall()}
    assert "scope_snapshot_id" in memory_columns
    assert "session_id" in memory_columns
    session_columns = {row["name"] for row in conn.execute("PRAGMA table_info('persona_sessions')").fetchall()}
    assert "activity_surface" in session_columns
    assert "preferences_json" in session_columns
    memory_indexes = {row["name"] for row in conn.execute("PRAGMA index_list('persona_memory_entries')").fetchall()}
    assert "idx_persona_memory_scope" in memory_indexes
    assert "idx_persona_memory_session" in memory_indexes

    character_after = dict(
        conn.execute("SELECT * FROM character_cards WHERE id = ?", (character_before["id"],)).fetchone()
    )
    persona_after = dict(
        conn.execute("SELECT * FROM persona_profiles WHERE id = ?", (persona_before["id"],)).fetchone()
    )
    memory_after = dict(
        conn.execute("SELECT * FROM persona_memory_entries WHERE id = ?", (memory_before["id"],)).fetchone()
    )
    assert all(character_after[key] == value for key, value in character_before.items())
    assert all(persona_after[key] == value for key, value in persona_before.items())
    assert all(memory_after[key] == value for key, value in memory_before.items())
    migrated.close_connection()


def test_migration_v36_to_latest_adds_voice_command_persona_columns(db_path: Path, monkeypatch: pytest.MonkeyPatch):
    seeded = _seed_historical_sqlite(db_path, 36, monkeypatch)
    try:
        with seeded.transaction() as conn:
            assert seeded._get_db_version(conn) == 36
            tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert "note_attachments" not in tables
            voice_columns = {row["name"] for row in conn.execute("PRAGMA table_info('voice_commands')")}
            assert {"persona_id", "connection_id"}.isdisjoint(voice_columns)
            conn.execute(
                "INSERT INTO persona_profiles (id, user_id, name, system_prompt) VALUES (?, ?, ?, ?)",
                ("historical-persona", "1", "Historical voice persona", "Retained voice persona prompt"),
            )
            conn.execute(
                "INSERT INTO voice_commands (id, user_id, name, phrases, action_type, action_config) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("historical-voice", 1, "Retained voice command", '["read my note"]', "llm_chat", '{"prompt":"Read"}'),
            )
            persona_before = dict(conn.execute("SELECT * FROM persona_profiles").fetchone())
            voice_before = dict(conn.execute("SELECT * FROM voice_commands").fetchone())
            seeded._migrate_from_v36_to_v37(conn)
            assert seeded._get_db_version(conn) == 37
            assert dict(conn.execute("SELECT * FROM voice_commands").fetchone()) == voice_before
    finally:
        seeded.close_all_connections()

    migrated = CharactersRAGDB(db_path, "voice-command-migration-check-client")
    conn = migrated.get_connection()

    version = conn.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()["version"]
    assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION

    voice_columns = {row["name"] for row in conn.execute("PRAGMA table_info('voice_commands')").fetchall()}
    assert "persona_id" in voice_columns
    assert "connection_id" in voice_columns

    voice_indexes = {row["name"] for row in conn.execute("PRAGMA index_list('voice_commands')").fetchall()}
    assert "idx_voice_commands_user_persona_enabled" in voice_indexes

    persona_after = dict(
        conn.execute("SELECT * FROM persona_profiles WHERE id = ?", (persona_before["id"],)).fetchone()
    )
    voice_after = dict(conn.execute("SELECT * FROM voice_commands WHERE id = ?", (voice_before["id"],)).fetchone())
    assert all(persona_after[key] == value for key, value in persona_before.items())
    assert all(voice_after[key] == value for key, value in voice_before.items())
    assert voice_after["persona_id"] is None
    assert voice_after["connection_id"] is None
    migrated.close_connection()


def test_persona_persistence_crud_and_user_scoping(db_instance: CharactersRAGDB):
    character_id = db_instance.add_character_card(sample_card_data(name="Research Persona Source"))
    assert character_id is not None

    persona_id = db_instance.create_persona_profile(
        {
            "user_id": "user-1",
            "name": "Research Persona",
            "mode": "session_scoped",
            "system_prompt": "You are a focused assistant.",
            "character_card_id": character_id,
            "is_active": True,
        }
    )
    assert isinstance(persona_id, str)

    profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert profile is not None
    assert profile["name"] == "Research Persona"
    assert profile["mode"] == "session_scoped"
    assert profile["character_card_id"] == character_id
    assert profile["is_active"] is True
    assert profile["use_persona_state_context_default"] is True
    assert profile["origin_character_id"] == character_id
    assert profile["origin_character_name"] == "Research Persona Source"
    assert profile["origin_character_snapshot_at"]
    expected_version = int(profile["version"])

    assert db_instance.update_persona_profile(
        persona_id=persona_id,
        user_id="user-1",
        update_data={
            "mode": "persistent_scoped",
            "is_active": True,
            "use_persona_state_context_default": False,
        },
        expected_version=expected_version,
    )
    updated_profile = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert updated_profile is not None
    assert updated_profile["mode"] == "persistent_scoped"
    assert updated_profile["use_persona_state_context_default"] is False

    inserted_scope_count = db_instance.replace_persona_scope_rules(
        persona_id=persona_id,
        user_id="user-1",
        rules=[
            {"rule_type": "conversation_id", "rule_value": "conv-a", "include": True},
            {"rule_type": "media_tag", "rule_value": "physics", "include": True},
        ],
    )
    assert inserted_scope_count == 2
    scope_rules = db_instance.list_persona_scope_rules(persona_id=persona_id, user_id="user-1")
    assert len(scope_rules) == 2
    assert {rule["rule_type"] for rule in scope_rules} == {"conversation_id", "media_tag"}

    inserted_policy_count = db_instance.replace_persona_policy_rules(
        persona_id=persona_id,
        user_id="user-1",
        rules=[
            {"rule_kind": "mcp_tool", "rule_name": "knowledge.search", "allowed": True},
            {
                "rule_kind": "skill",
                "rule_name": "digest",
                "allowed": True,
                "require_confirmation": False,
                "max_calls_per_turn": 2,
            },
        ],
    )
    assert inserted_policy_count == 2
    policy_rules = db_instance.list_persona_policy_rules(persona_id=persona_id, user_id="user-1")
    assert len(policy_rules) == 2
    assert {rule["rule_kind"] for rule in policy_rules} == {"mcp_tool", "skill"}

    session_id = db_instance.create_persona_session(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "mode": "persistent_scoped",
            "reuse_allowed": True,
            "status": "active",
            "scope_snapshot_json": {"conversations": ["conv-a"], "media_tags": ["physics"]},
            "preferences_json": {
                "use_memory_context": True,
                "use_companion_context": False,
                "memory_top_k": 4,
            },
        }
    )
    session = db_instance.get_persona_session(session_id, user_id="user-1")
    assert session is not None
    assert session["persona_id"] == persona_id
    assert session["reuse_allowed"] is True
    assert session["scope_snapshot"]["conversations"] == ["conv-a"]
    assert session["activity_surface"] == "api.persona"
    assert session["preferences"]["use_memory_context"] is True
    assert session["preferences"]["use_companion_context"] is False
    assert session["preferences"]["memory_top_k"] == 4
    session_version = int(session["version"])

    assert db_instance.update_persona_session(
        session_id=session_id,
        user_id="user-1",
        update_data={
            "status": "paused",
            "activity_surface": "companion.conversation",
            "preferences_json": {
                "use_memory_context": False,
                "use_companion_context": True,
                "memory_top_k": 7,
            },
        },
        expected_version=session_version,
    )
    paused_session = db_instance.get_persona_session(session_id, user_id="user-1")
    assert paused_session is not None
    assert paused_session["status"] == "paused"
    assert paused_session["activity_surface"] == "companion.conversation"
    assert paused_session["preferences"]["use_memory_context"] is False
    assert paused_session["preferences"]["use_companion_context"] is True
    assert paused_session["preferences"]["memory_top_k"] == 7

    memory_id = db_instance.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "memory_type": "summary",
            "content": "User prefers concise evidence-backed responses.",
            "scope_snapshot_id": "scope_1",
            "session_id": "sess_1",
            "salience": 0.9,
        }
    )
    memories = db_instance.list_persona_memory_entries(user_id="user-1", persona_id=persona_id)
    assert len(memories) == 1
    assert memories[0]["id"] == memory_id
    assert memories[0]["scope_snapshot_id"] == "scope_1"
    assert memories[0]["session_id"] == "sess_1"
    assert memories[0]["archived"] is False

    _ = db_instance.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "memory_type": "summary",
            "content": "Scope/session isolated memory.",
            "scope_snapshot_id": "scope_2",
            "session_id": "sess_2",
            "salience": 0.7,
        }
    )
    scoped_memories = db_instance.list_persona_memory_entries(
        user_id="user-1",
        persona_id=persona_id,
        scope_snapshot_id="scope_1",
    )
    assert len(scoped_memories) == 1
    assert scoped_memories[0]["id"] == memory_id
    session_memories = db_instance.list_persona_memory_entries(
        user_id="user-1",
        persona_id=persona_id,
        session_id="sess_2",
    )
    assert len(session_memories) == 1
    assert session_memories[0]["session_id"] == "sess_2"

    assert db_instance.set_persona_memory_archived(
        entry_id=memory_id,
        user_id="user-1",
        persona_id=persona_id,
        archived=True,
    )
    visible_memories = db_instance.list_persona_memory_entries(
        user_id="user-1",
        persona_id=persona_id,
        scope_snapshot_id="scope_1",
        include_archived=False,
    )
    assert visible_memories == []

    source_character = db_instance.get_character_card_by_id(character_id)
    assert source_character is not None
    assert db_instance.soft_delete_character_card(character_id, expected_version=int(source_character["version"])) is True
    assert db_instance.get_character_card_by_id(character_id) is None

    profile_after_source_delete = db_instance.get_persona_profile(persona_id, user_id="user-1")
    assert profile_after_source_delete is not None
    assert profile_after_source_delete["origin_character_id"] == character_id
    assert profile_after_source_delete["origin_character_name"] == "Research Persona Source"
    assert profile_after_source_delete["origin_character_snapshot_at"]

    assert db_instance.get_persona_profile(persona_id, user_id="user-2") is None
    assert db_instance.list_persona_profiles(user_id="user-2") == []
    assert db_instance.get_persona_session(session_id, user_id="user-2") is None
    assert db_instance.list_persona_sessions(user_id="user-2") == []
    assert not db_instance.set_persona_memory_archived(entry_id=memory_id, user_id="user-2", archived=False)


def test_get_persona_session_logs_warning_for_malformed_preferences_json(db_instance: CharactersRAGDB):
    persona_id = db_instance.create_persona_profile(
        {
            "id": "persona_invalid_prefs",
            "user_id": "user-1",
            "name": "Persona Invalid Prefs",
            "mode": "session_scoped",
            "system_prompt": "Test",
            "is_active": True,
        }
    )
    session_id = db_instance.create_persona_session(
        {
            "id": "sess_invalid_prefs",
            "persona_id": persona_id,
            "user_id": "user-1",
            "mode": "session_scoped",
            "reuse_allowed": True,
            "status": "active",
            "preferences_json": {"use_memory_context": True},
            "scope_snapshot_json": {"conversation_ids": ["conv-1"]},
        }
    )

    with db_instance.transaction():
        db_instance.execute_query(
            "UPDATE persona_sessions SET preferences_json = ? WHERE id = ?",
            ('{"use_memory_context": true', session_id),
        )

    messages: list[str] = []
    sink_id = logger.add(
        lambda msg: messages.append(str(msg.record.get("message") or "")),
        level="WARNING",
        format="{message}",
    )
    try:
        session = db_instance.get_persona_session(session_id, user_id="user-1")
    finally:
        logger.remove(sink_id)

    assert session is not None
    assert session["preferences"] == {}
    assert session["scope_snapshot"] == {"conversation_ids": ["conv-1"]}
    assert any(
        "preferences_json" in message and session_id in message
        for message in messages
    )


def test_list_persona_sessions_logs_warning_for_malformed_scope_snapshot_json(db_instance: CharactersRAGDB):
    persona_id = db_instance.create_persona_profile(
        {
            "id": "persona_invalid_scope",
            "user_id": "user-1",
            "name": "Persona Invalid Scope",
            "mode": "session_scoped",
            "system_prompt": "Test",
            "is_active": True,
        }
    )
    session_id = db_instance.create_persona_session(
        {
            "id": "sess_invalid_scope",
            "persona_id": persona_id,
            "user_id": "user-1",
            "mode": "session_scoped",
            "reuse_allowed": True,
            "status": "active",
            "preferences_json": {"use_memory_context": True},
            "scope_snapshot_json": {"workspace_ids": ["ws-1"]},
        }
    )

    with db_instance.transaction():
        db_instance.execute_query(
            "UPDATE persona_sessions SET scope_snapshot_json = ? WHERE id = ?",
            ('{"workspace_ids": ["ws-1"]', session_id),
        )

    messages: list[str] = []
    sink_id = logger.add(
        lambda msg: messages.append(str(msg.record.get("message") or "")),
        level="WARNING",
        format="{message}",
    )
    try:
        sessions = db_instance.list_persona_sessions(user_id="user-1")
    finally:
        logger.remove(sink_id)

    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id
    assert sessions[0]["scope_snapshot"] == {}
    assert sessions[0]["preferences"] == {"use_memory_context": True}
    assert any(
        "scope_snapshot_json" in message and session_id in message
        for message in messages
    )


def test_backfill_persona_memory_scope_namespace_updates_only_missing_scope(db_instance: CharactersRAGDB):
    persona_id = db_instance.create_persona_profile(
        {
            "id": "persona-backfill",
            "user_id": "user-1",
            "name": "Backfill Persona",
            "mode": "persistent_scoped",
            "system_prompt": "",
            "is_active": True,
        }
    )

    missing_scope_entry_id = db_instance.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "memory_type": "summary",
            "content": "legacy-unscoped",
            "scope_snapshot_id": None,
            "session_id": None,
            "salience": 0.4,
        }
    )
    existing_scope_entry_id = db_instance.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "memory_type": "summary",
            "content": "already-scoped",
            "scope_snapshot_id": "scope_already",
            "session_id": None,
            "salience": 0.4,
        }
    )
    session_scoped_entry_id = db_instance.add_persona_memory_entry(
        {
            "persona_id": persona_id,
            "user_id": "user-1",
            "memory_type": "summary",
            "content": "session-scoped",
            "scope_snapshot_id": None,
            "session_id": "sess_1",
            "salience": 0.4,
        }
    )

    updated_count = db_instance.backfill_persona_memory_scope_namespace(
        user_id="user-1",
        persona_id=persona_id,
        scope_snapshot_id="persistent_legacy_pid_example",
        require_missing_session_id=True,
        include_archived=False,
        include_deleted=False,
    )
    assert updated_count == 1

    rows = db_instance.list_persona_memory_entries(
        user_id="user-1",
        persona_id=persona_id,
        include_archived=True,
        include_deleted=True,
        limit=50,
        offset=0,
    )
    by_id = {str(row.get("id")): row for row in rows}
    assert by_id[missing_scope_entry_id]["scope_snapshot_id"] == "persistent_legacy_pid_example"
    assert by_id[existing_scope_entry_id]["scope_snapshot_id"] == "scope_already"
    assert by_id[session_scoped_entry_id]["scope_snapshot_id"] in (None, "")
