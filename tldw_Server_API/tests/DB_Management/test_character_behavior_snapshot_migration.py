"""Migration and store contract tests for character behavior snapshots."""

from __future__ import annotations

import json
import sqlite3
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import (
    BehaviorSnapshotV1,
    build_behavior_snapshot,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import (
    ConversationResumeStore,
)
from tldw_Server_API.app.core.DB_Management.chacha.message_store import MessageStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendConnectionWrapper,
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

pytestmark = pytest.mark.unit


def _snapshot() -> BehaviorSnapshotV1:
    return build_behavior_snapshot(
        {
            "schema_version": 1,
            "participants": [
                {
                    "source": {"kind": "character", "id": "1", "version": 1},
                    "identity": {"name": "Legacy Character", "aliases": []},
                    "prompt": {
                        "system_prompt": "Stay in character.",
                        "description": "A stored character.",
                        "personality": "Careful and precise.",
                        "scenario": "A migration test.",
                        "message_example": "Character: Hello",
                        "post_history_instructions": "Preserve history.",
                        "prompt_relevant_extensions": {},
                    },
                    "greeting": {
                        "content": "Hello.",
                        "source": "default",
                        "source_index": 0,
                    },
                    "generation_defaults": {},
                    "exemplars": [],
                    "world_books": [],
                    "default_memory": None,
                }
            ],
            "routing_defaults": {"turn_taking_mode": "single"},
        }
    )


def _create_v63_legacy_database(
    db_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    conversation_id: str = "legacy-character-conversation",
) -> str:
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 63)
        db = CharactersRAGDB(db_path=str(db_path), client_id="legacy-owner")
        created_id = db.add_conversation(
            {
                "id": conversation_id,
                "character_id": 1,
                "title": "Legacy character conversation",
            }
        )
        assert created_id == conversation_id
        assert db.upsert_conversation_settings(
            conversation_id,
            {"temperature": 0.25},
        )
        db.close_connection()

    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
        assert version == 63
        assert "history_version" not in {
            row[1] for row in conn.execute("PRAGMA table_info('conversations')")
        }
        assert "settings_version" not in {
            row[1] for row in conn.execute("PRAGMA table_info('conversation_settings')")
        }
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'conversation_behavior_snapshots'"
        ).fetchone() is None
    return conversation_id


def _table_columns(conn: sqlite3.Connection, table_name: str) -> dict[str, sqlite3.Row]:
    return {
        str(row["name"]): row
        for row in conn.execute(f"PRAGMA table_info('{table_name}')")  # nosec B608 - fixed test names
    }


def test_schema_head_allocates_exactly_v64_and_both_ladders_advance_once() -> None:
    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 64
    assert CharactersRAGDB._POSTGRES_SCHEMA_VERSION == 64

    db = CharactersRAGDB.__new__(CharactersRAGDB)
    assert db._sqlite_linear_migration_steps()[63].__name__ == "_migrate_from_v63_to_v64"
    assert hasattr(CharactersRAGDB, "_migrate_from_v63_to_v64_postgres")


def test_sqlite_v63_to_v64_migrates_catalog_and_legacy_reads_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "character-behavior-v63.sqlite"
    conversation_id = _create_v63_legacy_database(db_path, monkeypatch)

    migrated = CharactersRAGDB(db_path=str(db_path), client_id="legacy-owner")

    snapshot = migrated.get_conversation_behavior_snapshot(conversation_id)
    assert snapshot == {
        "status": "missing",
        "schema_version": None,
        "canonical_json": None,
        "digest": None,
        "size_bytes": None,
        "created_at": None,
        "payload": None,
    }
    state = migrated.get_roleplay_resume_state(conversation_id)
    assert state["behavior_snapshot"] == snapshot
    assert state["settings_version"] == 1
    assert state["history_version"] == 1
    migrated.close_connection()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()["version"]
        assert version == 64

        snapshot_columns = _table_columns(conn, "conversation_behavior_snapshots")
        assert set(snapshot_columns) == {
            "conversation_id",
            "status",
            "schema_version",
            "canonical_json",
            "digest",
            "size_bytes",
            "created_at",
        }
        assert snapshot_columns["conversation_id"]["pk"] == 1
        assert snapshot_columns["status"]["notnull"] == 1

        conversation_columns = _table_columns(conn, "conversations")
        assert conversation_columns["history_version"]["notnull"] == 1
        assert str(conversation_columns["history_version"]["dflt_value"]).strip("'") == "1"
        settings_columns = _table_columns(conn, "conversation_settings")
        assert settings_columns["settings_version"]["notnull"] == 1
        assert str(settings_columns["settings_version"]["dflt_value"]).strip("'") == "1"

        foreign_keys = conn.execute(
            "PRAGMA foreign_key_list('conversation_behavior_snapshots')"
        ).fetchall()
        assert len(foreign_keys) == 1
        assert foreign_keys[0]["table"] == "conversations"
        assert foreign_keys[0]["from"] == "conversation_id"
        assert foreign_keys[0]["to"] == "id"
        assert foreign_keys[0]["on_delete"].upper() == "CASCADE"

        indexes = {
            row["name"]: row
            for row in conn.execute("PRAGMA index_list('conversation_behavior_snapshots')")
        }
        assert "idx_conversation_behavior_snapshots_status" in indexes
        assert "idx_conversation_behavior_snapshots_digest" in indexes
        assert any(row["unique"] == 1 for row in indexes.values())

        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'conversation_behavior_snapshots'"
        ).fetchone()["sql"].lower()
        assert "status in ('valid','missing','invalid')" in table_sql
        assert "schema_version is not null" in table_sql
        assert "schema_version >= 1" in table_sql
        assert "size_bytes is not null" in table_sql
        assert "size_bytes >= 1" in table_sql
        assert "canonical_json is null" in table_sql
        assert "digest is null" in table_sql
        assert "size_bytes is null" in table_sql

        settings = conn.execute(
            "SELECT settings_json, settings_version FROM conversation_settings "
            "WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        assert settings["settings_json"] == '{"temperature": 0.25}'
        assert settings["settings_version"] == 1
        assert conn.execute(
            "SELECT history_version FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()["history_version"] == 1


def test_sqlite_v64_constraints_enforce_one_to_one_status_body_digest_and_versions(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "character-behavior-constraints.sqlite"
    db = CharactersRAGDB(db_path=str(db_path), client_id="constraint-owner")
    conversation_ids = []
    for suffix in range(20):
        conversation_id = db.add_conversation(
            {
                "id": f"constraint-conversation-{suffix}",
                "character_id": 1,
                "title": f"Constraint {suffix}",
            }
        )
        assert conversation_id is not None
        conversation_ids.append(conversation_id)
        assert db.upsert_conversation_settings(conversation_id, {})
    db.close_connection()

    snapshot = _snapshot()
    canonical_json = snapshot.canonical_bytes.decode("utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO conversation_behavior_snapshots "
            "(conversation_id, status, schema_version, canonical_json, digest, size_bytes) "
            "VALUES (?, 'valid', ?, ?, ?, ?)",
            (
                conversation_ids[0],
                snapshot.schema_version,
                canonical_json,
                snapshot.digest,
                snapshot.size_bytes,
            ),
        )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO conversation_behavior_snapshots(conversation_id, status) "
                "VALUES (?, 'missing')",
                (conversation_ids[0],),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO conversation_behavior_snapshots(conversation_id, status) "
                "VALUES ('missing-conversation', 'missing')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO conversation_behavior_snapshots(conversation_id, status) "
                "VALUES (?, 'unknown')",
                (conversation_ids[1],),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO conversation_behavior_snapshots "
                "(conversation_id, status, canonical_json) VALUES (?, 'missing', '{}')",
                (conversation_ids[2],),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO conversation_behavior_snapshots "
                "(conversation_id, status, schema_version, canonical_json, digest, size_bytes) "
                "VALUES (?, 'valid', 1, '{}', 'sha256:not-a-digest', 2)",
                (conversation_ids[3],),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO conversation_behavior_snapshots "
                "(conversation_id, status, schema_version, canonical_json, digest, size_bytes) "
                "VALUES (?, 'valid', ?, ?, ?, ?)",
                (
                    conversation_ids[4],
                    snapshot.schema_version,
                    canonical_json,
                    snapshot.digest,
                    snapshot.size_bytes + 1,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE conversations SET history_version = 0 WHERE id = ?",
                (conversation_ids[5],),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE conversation_settings SET settings_version = 0 WHERE conversation_id = ?",
                (conversation_ids[6],),
            )

        valid_values: dict[str, Any] = {
            "schema_version": snapshot.schema_version,
            "canonical_json": canonical_json,
            "digest": snapshot.digest,
            "size_bytes": snapshot.size_bytes,
        }
        for offset, field in enumerate(valid_values, start=7):
            values = dict(valid_values)
            values[field] = None
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO conversation_behavior_snapshots "
                    "(conversation_id, status, schema_version, canonical_json, digest, size_bytes) "
                    "VALUES (?, 'valid', ?, ?, ?, ?)",
                    (
                        conversation_ids[offset],
                        values["schema_version"],
                        values["canonical_json"],
                        values["digest"],
                        values["size_bytes"],
                    ),
                )

        nonvalid_values = {
            "schema_version": snapshot.schema_version,
            "canonical_json": canonical_json,
            "digest": snapshot.digest,
            "size_bytes": snapshot.size_bytes,
        }
        conversation_offset = 11
        for status in ("missing", "invalid"):
            for field, value in nonvalid_values.items():
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute(
                        "INSERT INTO conversation_behavior_snapshots "
                        f"(conversation_id, status, {field}) VALUES (?, ?, ?)",  # nosec B608
                        (conversation_ids[conversation_offset], status, value),
                    )
                conversation_offset += 1


def test_sqlite_v64_checkpoint_failure_rolls_back_all_schema_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "character-behavior-rollback.sqlite"
    _create_v63_legacy_database(db_path, monkeypatch)

    def _fail_checkpoint(_self: CharactersRAGDB, stage: str) -> None:
        assert stage == "schema-created"
        raise RuntimeError("injected v64 migration checkpoint failure")

    monkeypatch.setattr(CharactersRAGDB, "_migration_v64_checkpoint", _fail_checkpoint)
    with pytest.raises(CharactersRAGDBError, match="injected v64 migration checkpoint failure"):
        CharactersRAGDB(db_path=str(db_path), client_id="rollback-owner")

    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
        assert version == 63
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'conversation_behavior_snapshots'"
        ).fetchone() is None
        assert "history_version" not in {
            row[1] for row in conn.execute("PRAGMA table_info('conversations')")
        }
        assert "settings_version" not in {
            row[1] for row in conn.execute("PRAGMA table_info('conversation_settings')")
        }


@pytest.mark.parametrize("drift_object", ["table", "index", "column"])
def test_sqlite_v64_rejects_partial_preexisting_schema_and_preserves_v63(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_object: str,
) -> None:
    db_path = tmp_path / f"character-behavior-drift-{drift_object}.sqlite"
    _create_v63_legacy_database(db_path, monkeypatch)

    with sqlite3.connect(db_path) as conn:
        if drift_object == "table":
            conn.execute(
                """
                CREATE TABLE conversation_behavior_snapshots(
                  conversation_id TEXT PRIMARY KEY,
                  status TEXT NOT NULL,
                  schema_version INTEGER,
                  canonical_json TEXT,
                  digest TEXT,
                  size_bytes INTEGER,
                  created_at DATETIME
                )
                """
            )
        elif drift_object == "index":
            conn.execute(
                "CREATE INDEX idx_conversation_behavior_snapshots_status "
                "ON conversations(title)"
            )
        else:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN history_version INTEGER DEFAULT 7"
            )

    with pytest.raises(CharactersRAGDBError):
        migrated = CharactersRAGDB(db_path=str(db_path), client_id="drift-owner")
        migrated.close_connection()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 63
        if drift_object != "table":
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'conversation_behavior_snapshots'"
            ).fetchone() is None
        assert "settings_version" not in {
            row[1] for row in conn.execute("PRAGMA table_info('conversation_settings')")
        }


def test_snapshot_store_uses_canonical_authority_and_rejects_incoherent_values(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        db_path=str(tmp_path / "character-behavior-store.sqlite"),
        client_id="snapshot-owner",
    )
    conversation_id = db.add_conversation(
        {"character_id": 1, "title": "Canonical snapshot"}
    )
    assert conversation_id is not None
    snapshot = _snapshot()

    with db.transaction() as conn:
        db.put_behavior_snapshot(conversation_id, snapshot, conn=conn)

    stored = db.get_conversation_behavior_snapshot(conversation_id)
    assert stored["status"] == "valid"
    assert stored["schema_version"] == snapshot.schema_version
    assert stored["canonical_json"] == snapshot.canonical_bytes.decode("utf-8")
    assert stored["payload"] == snapshot.payload
    assert stored["digest"] == snapshot.digest
    assert stored["size_bytes"] == snapshot.size_bytes
    assert stored["created_at"] is not None

    incoherent = BehaviorSnapshotV1(
        schema_version=snapshot.schema_version,
        canonical_bytes=snapshot.canonical_bytes,
        digest="sha256:" + ("0" * 64),
        size_bytes=snapshot.size_bytes,
    )
    second_conversation = db.add_conversation(
        {"character_id": 1, "title": "Rejected snapshot"}
    )
    assert second_conversation is not None
    with db.transaction() as conn:
        with pytest.raises(InputError, match="digest"):
            db.put_behavior_snapshot(second_conversation, incoherent, conn=conn)

    assert db.get_conversation_behavior_snapshot(second_conversation)["status"] == "missing"
    db.close_connection()


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("canonical_json", "{}"),
        ("digest", "sha256:" + ("0" * 64)),
        ("size_bytes", 1),
        ("schema_version", 99),
    ],
)
def test_snapshot_store_fails_closed_on_persisted_authority_tamper(
    tmp_path: Path,
    field: str,
    tampered_value: Any,
) -> None:
    db_path = tmp_path / f"character-behavior-tamper-{field}.sqlite"
    db = CharactersRAGDB(db_path=str(db_path), client_id="tamper-owner")
    conversation_id = db.add_conversation(
        {"character_id": 1, "title": f"Tampered {field}"}
    )
    assert conversation_id is not None
    with db.transaction() as conn:
        db.put_behavior_snapshot(conversation_id, _snapshot(), conn=conn)
    db.close_connection()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            f"UPDATE conversation_behavior_snapshots SET {field} = ? WHERE conversation_id = ?",  # nosec B608
            (tampered_value, conversation_id),
        )

    reopened = CharactersRAGDB(db_path=str(db_path), client_id="tamper-owner")
    stored = reopened.get_conversation_behavior_snapshot(conversation_id)
    assert stored == {
        "status": "invalid",
        "schema_version": None,
        "canonical_json": None,
        "digest": None,
        "size_bytes": None,
        "created_at": None,
        "payload": None,
    }
    reopened.close_connection()


def test_caller_owned_creation_transaction_rolls_back_conversation_snapshot_and_message(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        db_path=str(tmp_path / "character-behavior-atomic.sqlite"),
        client_id="atomic-owner",
    )
    snapshot = _snapshot()
    conversation_id = "atomic-conversation"

    with pytest.raises(RuntimeError, match="rollback factory"):
        with db.transaction() as conn:
            assert db.add_conversation(
                {
                    "id": conversation_id,
                    "character_id": 1,
                    "title": "Atomic conversation",
                },
                conn=conn,
            ) == conversation_id
            db.put_behavior_snapshot(conversation_id, snapshot, conn=conn)
            assert db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": "Must roll back",
                },
                conn=conn,
            ) is not None
            raise RuntimeError("rollback factory")

    assert db.get_conversation_by_id(conversation_id, include_deleted=True) is None
    with sqlite3.connect(db.db_path_str) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM conversation_behavior_snapshots WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()[0] == 0
    db.close_connection()


def test_message_mutations_advance_history_once_and_rollback_with_caller_connection(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        db_path=str(tmp_path / "character-history-version.sqlite"),
        client_id="history-owner",
    )
    conversation_id = db.add_conversation(
        {"character_id": 1, "title": "Versioned history"}
    )
    assert conversation_id is not None
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 1

    root_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "Root"}
    )
    assert root_id is not None
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 2

    root = db.get_message_by_id(root_id)
    assert root is not None
    assert db.update_message(root_id, {"content": "Edited root"}, root["version"])
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 3

    branch_id = db.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": root_id,
            "sender": "assistant",
            "content": "Branch",
        }
    )
    assert branch_id is not None
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 4

    branch = db.get_message_by_id(branch_id)
    assert branch is not None
    assert db.update_message(
        branch_id,
        {"parent_message_id": None},
        branch["version"],
    )
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 5

    branch = db.get_message_by_id(branch_id)
    assert branch is not None
    assert db.soft_delete_message(branch_id, branch["version"])
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 6

    assert db.soft_delete_message(branch_id, branch["version"])
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 6

    with pytest.raises(RuntimeError, match="rollback message"):
        with db.transaction() as conn:
            assert db.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "user",
                    "content": "Rolled back",
                },
                conn=conn,
            ) is not None
            assert db.get_roleplay_resume_state(conversation_id, conn=conn)["history_version"] == 7
            raise RuntimeError("rollback message")

    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 6
    assert db.count_messages_for_conversation(conversation_id, include_deleted=True) == 2

    root = db.get_message_by_id(root_id)
    assert root is not None
    with pytest.raises(RuntimeError, match="rollback edit"):
        with db.transaction() as conn:
            assert db.update_message(
                root_id,
                {"content": "Rolled-back edit"},
                root["version"],
                conn=conn,
            )
            assert db.get_roleplay_resume_state(conversation_id, conn=conn)["history_version"] == 7
            raise RuntimeError("rollback edit")

    restored_root = db.get_message_by_id(root_id)
    assert restored_root is not None
    assert restored_root["content"] == "Edited root"
    assert restored_root["version"] == root["version"]
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 6

    with pytest.raises(RuntimeError, match="rollback delete"):
        with db.transaction() as conn:
            assert db.soft_delete_message(root_id, root["version"], conn=conn)
            assert db.get_roleplay_resume_state(conversation_id, conn=conn)["history_version"] == 7
            raise RuntimeError("rollback delete")

    restored_root = db.get_message_by_id(root_id)
    assert restored_root is not None
    assert restored_root["deleted"] == 0
    assert restored_root["version"] == root["version"]
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 6
    db.close_connection()


def test_prompt_visible_image_and_pin_mutations_advance_and_rollback_history_fence(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        db_path=str(tmp_path / "character-history-prompt-visible.sqlite"),
        client_id="prompt-visible-owner",
    )
    conversation_id = db.add_conversation(
        {"character_id": 1, "title": "Prompt-visible mutations"}
    )
    assert conversation_id is not None
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "See this"}
    )
    assert message_id is not None
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 2

    assert db.append_message_image(message_id, b"first-image", "image/png") == 0
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 3

    with pytest.raises(RuntimeError, match="rollback image"):
        with db.transaction() as conn:
            assert db.append_message_image(
                message_id,
                b"rolled-back-image",
                "image/png",
                conn=conn,
            ) == 1
            assert db.get_roleplay_resume_state(conversation_id, conn=conn)["history_version"] == 4
            raise RuntimeError("rollback image")

    assert len(db.get_message_images(message_id)) == 1
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 3

    assert db.set_message_metadata_extra(
        message_id,
        {"pinned": True},
        conn=None,
    )
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 4
    assert db.set_message_metadata_extra(message_id, {"pinned": True})
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 4

    with pytest.raises(RuntimeError, match="rollback pin"):
        with db.transaction() as conn:
            assert db.set_message_metadata_extra(
                message_id,
                {"pinned": False},
                conn=conn,
            )
            assert db.get_roleplay_resume_state(conversation_id, conn=conn)["history_version"] == 5
            raise RuntimeError("rollback pin")

    metadata = db.get_message_metadata(message_id)
    assert metadata is not None
    assert metadata["extra"]["pinned"] is True
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 4

    assert db.set_message_metadata_extra("missing-message", {"pinned": True}) is False
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 4
    db.close_connection()


@pytest.mark.parametrize("mutation", ["add_metadata", "set_extra"])
def test_caller_owned_metadata_fence_failure_propagates_and_rolls_back(
    tmp_path: Path,
    mutation: str,
) -> None:
    db = CharactersRAGDB(
        db_path=str(tmp_path / f"character-metadata-fence-{mutation}.sqlite"),
        client_id="metadata-fence-owner",
    )
    conversation_id = db.add_conversation(
        {"character_id": 1, "title": "Metadata fence failure"}
    )
    assert conversation_id is not None
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "Pin me"}
    )
    assert message_id is not None

    with db.transaction() as conn:
        conn.execute(
            "UPDATE conversations SET deleted = TRUE WHERE id = ?",
            (conversation_id,),
        )

    with sqlite3.connect(db.db_path_str) as conn:
        history_version = conn.execute(
            "SELECT history_version FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()[0]
    assert db.get_message_metadata(message_id) is None

    with pytest.raises(InputError, match="Cannot mutate message history"):
        with db.transaction() as conn:
            if mutation == "add_metadata":
                db.add_message_metadata(
                    message_id,
                    extra={"pinned": True},
                    conn=conn,
                )
            else:
                db.set_message_metadata_extra(
                    message_id,
                    {"pinned": True},
                    conn=conn,
                )

    assert db.get_message_metadata(message_id) is None
    with sqlite3.connect(db.db_path_str) as conn:
        assert conn.execute(
            "SELECT history_version FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()[0] == history_version
    db.close_connection()


def test_sync_append_advances_history_once_and_replay_does_not_advance(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        db_path=str(tmp_path / "character-sync-append-fence.sqlite"),
        client_id="sync-append-owner",
    )
    conversation_id = db.add_conversation(
        {"character_id": 1, "title": "Atomic Sync append"}
    )
    assert conversation_id is not None
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 1

    first = db.append_message_from_sync(
        stable_message_id="sync-atomic-append",
        conversation_id=conversation_id,
        sender="user",
        content="Synced once",
        timestamp="2026-08-28T08:00:00+00:00",
        sync_client_id="sync-device",
        object_revision=3,
        payload_hash="sha256:sync-atomic-append",
    )
    assert first["created"] is True
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 2
    assert db.get_message_by_id(first["message_id"])["version"] == 3
    assert db.get_conversation_behavior_snapshot(conversation_id)["status"] == "missing"

    replay = db.append_message_from_sync(
        stable_message_id="sync-atomic-append",
        conversation_id=conversation_id,
        sender="user",
        content="Synced once",
        timestamp="2026-08-28T08:00:00+00:00",
        sync_client_id="sync-device",
        object_revision=3,
        payload_hash="sha256:sync-atomic-append",
    )
    assert replay["idempotent"] is True
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 2
    db.close_connection()


def test_sync_append_metadata_failure_rolls_back_message_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(
        db_path=str(tmp_path / "character-sync-append-rollback.sqlite"),
        client_id="sync-append-rollback-owner",
    )
    conversation_id = db.add_conversation(
        {"character_id": 1, "title": "Rolled-back Sync append"}
    )
    assert conversation_id is not None
    original = db.message_store._set_sync_v2_message_metadata_or_raise

    def _fail_after_metadata(**kwargs: Any) -> None:
        original(**kwargs)
        raise CharactersRAGDBError("injected Sync append metadata failure")

    monkeypatch.setattr(
        db.message_store,
        "_set_sync_v2_message_metadata_or_raise",
        _fail_after_metadata,
    )
    with pytest.raises(CharactersRAGDBError, match="injected Sync append metadata failure"):
        db.append_message_from_sync(
            stable_message_id="sync-rollback-append",
            conversation_id=conversation_id,
            sender="user",
            content="Must roll back",
            timestamp="2026-08-28T08:01:00+00:00",
            sync_client_id="sync-device",
            object_revision=2,
            payload_hash="sha256:sync-rollback-append",
        )

    assert db.get_message_by_id("sync-rollback-append", include_deleted=True) is None
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 1
    assert db.get_conversation_behavior_snapshot(conversation_id)["status"] == "missing"
    db.close_connection()


def test_sync_tombstone_advances_history_once_and_replay_does_not_advance(
    tmp_path: Path,
) -> None:
    db = CharactersRAGDB(
        db_path=str(tmp_path / "character-sync-tombstone-fence.sqlite"),
        client_id="sync-tombstone-owner",
    )
    conversation_id = db.add_conversation(
        {"character_id": 1, "title": "Atomic Sync tombstone"}
    )
    assert conversation_id is not None
    db.append_message_from_sync(
        stable_message_id="sync-atomic-tombstone",
        conversation_id=conversation_id,
        sender="assistant",
        content="Delete through Sync",
        timestamp="2026-08-28T08:02:00+00:00",
        sync_client_id="sync-device",
        object_revision=1,
        payload_hash="sha256:sync-atomic-tombstone",
    )

    assert db.tombstone_message_from_sync(
        stable_message_id="sync-atomic-tombstone",
        sync_client_id="sync-device",
        object_revision=2,
        object_hash="sha256:sync-atomic-tombstone",
    )
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 3

    assert db.tombstone_message_from_sync(
        stable_message_id="sync-atomic-tombstone",
        sync_client_id="sync-device",
        object_revision=2,
        object_hash="sha256:sync-atomic-tombstone",
    )
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 3
    assert db.get_conversation_behavior_snapshot(conversation_id)["status"] == "missing"
    db.close_connection()


@pytest.mark.parametrize("failure_point", ["metadata", "fence"])
def test_sync_tombstone_failure_rolls_back_deletion_metadata_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    db = CharactersRAGDB(
        db_path=str(tmp_path / f"character-sync-tombstone-{failure_point}.sqlite"),
        client_id="sync-tombstone-rollback-owner",
    )
    conversation_id = db.add_conversation(
        {"character_id": 1, "title": "Rolled-back Sync tombstone"}
    )
    assert conversation_id is not None
    message_id = "sync-rollback-tombstone"
    db.append_message_from_sync(
        stable_message_id=message_id,
        conversation_id=conversation_id,
        sender="assistant",
        content="Keep after rollback",
        timestamp="2026-08-28T08:03:00+00:00",
        sync_client_id="sync-device",
        object_revision=1,
        payload_hash="sha256:sync-rollback-tombstone",
    )
    original_metadata = db.get_message_metadata(message_id)
    assert original_metadata is not None

    if failure_point == "metadata":
        original = db.message_store.set_message_metadata_extra

        def _fail_after_metadata(*args: Any, **kwargs: Any) -> bool:
            assert original(*args, **kwargs)
            raise CharactersRAGDBError("injected Sync tombstone metadata failure")

        monkeypatch.setattr(db.message_store, "set_message_metadata_extra", _fail_after_metadata)
    else:
        original_fence = db.message_store._advance_history_version

        def _fail_after_fence(conn: Any, target_conversation_id: str) -> None:
            original_fence(conn, target_conversation_id)
            raise CharactersRAGDBError("injected Sync tombstone fence failure")

        monkeypatch.setattr(db.message_store, "_advance_history_version", _fail_after_fence)

    with pytest.raises(CharactersRAGDBError, match=f"injected Sync tombstone {failure_point} failure"):
        db.tombstone_message_from_sync(
            stable_message_id=message_id,
            sync_client_id="sync-device",
            object_revision=2,
            object_hash="sha256:sync-rollback-tombstone",
        )

    message = db.get_message_by_id(message_id, include_deleted=True)
    assert message is not None
    assert bool(message["deleted"]) is False
    assert message["version"] == 1
    assert db.get_message_metadata(message_id) == original_metadata
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 2
    db.close_connection()


def test_commit_false_image_append_retries_without_splitting_history_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(
        db_path=str(tmp_path / "character-history-image-retry.sqlite"),
        client_id="image-retry-owner",
    )
    conversation_id = db.add_conversation(
        {"character_id": 1, "title": "Transactional image retry"}
    )
    assert conversation_id is not None
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "Retry"}
    )
    assert message_id is not None

    original_execute_query = db.execute_query
    conflicts = 0

    def _flaky_execute_query(query: str, *args: Any, **kwargs: Any) -> Any:
        nonlocal conflicts
        if "INSERT INTO message_images" in query and conflicts == 0:
            conflicts += 1
            raise sqlite3.IntegrityError(
                "UNIQUE constraint failed: message_images.message_id, message_images.position"
            )
        return original_execute_query(query, *args, **kwargs)

    monkeypatch.setattr(db, "execute_query", _flaky_execute_query)
    with db.transaction():
        assert db.append_message_image(
            message_id,
            b"retried-image",
            "image/png",
            commit=False,
        ) == 0

    assert conflicts == 1
    assert len(db.get_message_images(message_id)) == 1
    assert db.get_roleplay_resume_state(conversation_id)["history_version"] == 3
    db.close_connection()


class _PostgresTransaction:
    def __init__(self, backend: _PostgresRecordingBackend) -> None:
        self.backend = backend

    def __enter__(self) -> _PostgresRecordingBackend:
        self.backend._pending = []
        return self.backend

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.backend.committed_statements.extend(self.backend._pending)
        else:
            self.backend.rolled_back = True
        self.backend._pending = []
        return False


class _PostgresRecordingBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self, *, drift_object: str | None = None) -> None:
        self._pending: list[str] = []
        self.committed_statements: list[str] = []
        self.rolled_back = False
        self.schema_version = 63
        self.drift_object = drift_object

    def transaction(self) -> _PostgresTransaction:
        return _PostgresTransaction(self)

    def table_exists(self, _name: str, connection: object | None = None) -> bool:
        return True

    def execute(self, statement: str, *args: Any, **_kwargs: Any) -> SimpleNamespace:
        self._pending.append(str(statement))
        normalized = " ".join(str(statement).lower().split())
        params = args[0] if args else None
        drift_fragments = {
            "table": "create table conversation_behavior_snapshots",
            "index": "create index idx_conversation_behavior_snapshots_status",
            "column": "alter table conversations add column history_version",
        }
        drift_fragment = drift_fragments.get(self.drift_object or "")
        if drift_fragment and drift_fragment in normalized and "if not exists" not in normalized:
            raise BackendDatabaseError(f"pre-existing PostgreSQL {self.drift_object}")
        if normalized.startswith("update db_schema_version set version = %s"):
            if params == (64, CharactersRAGDB._SCHEMA_NAME, 63) and self.schema_version == 63:
                self.schema_version = 64
                return SimpleNamespace(rowcount=1)
            return SimpleNamespace(rowcount=0)
        return SimpleNamespace(rowcount=0)


def _postgres_db(backend: _PostgresRecordingBackend) -> CharactersRAGDB:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = backend
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()
    return db


class _PostgresStoreBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self, *, message_insert_error: str | None = None) -> None:
        self.conversation_id = "postgres-store-conversation"
        self.history_version = 1
        self.message_insert_error = message_insert_error
        self.messages: dict[str, dict[str, Any]] = {}
        self.images: list[tuple[str, int, bytes, str]] = []
        self.metadata: dict[str, dict[str, Any]] = {}
        self.snapshot_row: dict[str, Any] | None = None
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    @staticmethod
    def _result(rows: list[dict[str, Any]] | None = None, *, rowcount: int = 0) -> QueryResult:
        resolved_rows = rows or []
        description = [(key,) for key in resolved_rows[0]] if resolved_rows else None
        return QueryResult(rows=resolved_rows, rowcount=rowcount, description=description)

    def execute(
        self,
        statement: str,
        params: tuple[Any, ...] | None = None,
        **_kwargs: Any,
    ) -> QueryResult:
        normalized = " ".join(str(statement).lower().split())
        values = tuple(params or ())
        self.executed.append((normalized, values))

        if normalized.startswith("insert into conversation_behavior_snapshots"):
            self.snapshot_row = {
                "status": "valid",
                "schema_version": values[1],
                "canonical_json": values[2],
                "digest": values[3],
                "size_bytes": values[4],
                "created_at": "2026-08-28T00:00:00Z",
            }
            return self._result(rowcount=1)
        if "from conversation_behavior_snapshots" in normalized:
            return self._result(
                [dict(self.snapshot_row)] if self.snapshot_row is not None else [],
                rowcount=1 if self.snapshot_row is not None else 0,
            )
        if "select c.history_version, cs.settings_json, cs.settings_version" in normalized:
            return self._result(
                [
                    {
                        "history_version": self.history_version,
                        "settings_json": json.dumps({"temperature": 0.2}),
                        "settings_version": 1,
                    }
                ],
                rowcount=1,
            )
        if normalized.startswith("select 1 from conversations"):
            return self._result([{"present": 1}], rowcount=1)
        if normalized.startswith("insert into messages "):
            if self.message_insert_error is not None:
                raise BackendDatabaseError(self.message_insert_error)
            self.messages[str(values[0])] = {
                "conversation_id": str(values[1]),
                "content": values[4],
                "version": int(values[11]),
                "deleted": bool(values[12]),
            }
            return self._result(rowcount=1)
        if normalized.startswith("select coalesce(max(position), -1) + 1"):
            return self._result([{"next_position": len(self.images)}], rowcount=1)
        if normalized.startswith("insert into message_images"):
            self.images.append(
                (str(values[0]), int(values[1]), bytes(values[2]), str(values[3]))
            )
            return self._result(rowcount=1)
        if "left join message_metadata" in normalized:
            message_id = str(values[0])
            message = self.messages.get(message_id)
            if message is None:
                return self._result()
            metadata = self.metadata.get(message_id)
            return self._result(
                [
                    {
                        "conversation_id": message["conversation_id"],
                        "metadata_message_id": message_id if metadata is not None else None,
                        "tool_calls_json": metadata.get("tool_calls_json") if metadata else None,
                        "extra_json": metadata.get("extra_json") if metadata else None,
                    }
                ],
                rowcount=1,
            )
        if normalized.startswith("insert into message_metadata"):
            self.metadata[str(values[0])] = {
                "tool_calls_json": values[1],
                "extra_json": values[2],
                "last_modified": "2026-08-28T00:00:00Z",
            }
            return self._result(rowcount=1)
        if normalized.startswith("select tool_calls_json, extra_json, last_modified"):
            metadata = self.metadata.get(str(values[0]))
            return self._result([dict(metadata)] if metadata else [], rowcount=1 if metadata else 0)
        if normalized.startswith("select version, deleted from messages"):
            message = self.messages.get(str(values[0]))
            return self._result(
                [
                    {
                        "version": message["version"],
                        "deleted": message["deleted"],
                    }
                ]
                if message
                else [],
                rowcount=1 if message else 0,
            )
        if normalized.startswith("select conversation_id from messages"):
            message = self.messages.get(str(values[0]))
            return self._result(
                [{"conversation_id": message["conversation_id"]}] if message else [],
                rowcount=1 if message else 0,
            )
        if normalized.startswith("update messages set"):
            message_id = str(values[-2])
            message = self.messages[message_id]
            if "deleted = true" in normalized:
                message["deleted"] = True
                message["version"] = int(values[1])
            else:
                if "content = %s" in normalized:
                    message["content"] = values[0]
                message["version"] = int(values[-4])
            return self._result(rowcount=1)
        if normalized.startswith("update conversations set history_version"):
            self.history_version += 1
            return self._result(rowcount=1)

        raise AssertionError(f"Unexpected PostgreSQL store SQL: {normalized}")


def _postgres_store_db(
    backend: _PostgresStoreBackend,
) -> tuple[CharactersRAGDB, BackendConnectionWrapper]:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = backend
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()
    db.client_id = "postgres-store-owner"
    db.db_path_str = "postgresql://recording"
    db.message_store = MessageStore(db)
    db.conversation_resume_store = ConversationResumeStore(db)
    db._ensure_message_metadata_table = lambda: None
    wrapper = BackendConnectionWrapper(db, object(), backend)
    return db, wrapper


def test_postgres_backend_wrapper_roundtrips_resume_store_contract() -> None:
    backend = _PostgresStoreBackend()
    db, wrapper = _postgres_store_db(backend)
    snapshot = _snapshot()

    db.conversation_resume_store.put_behavior_snapshot(
        backend.conversation_id,
        snapshot,
        conn=wrapper,
    )
    state = db.conversation_resume_store.get_roleplay_resume_state(
        backend.conversation_id,
        conn=wrapper,
    )

    assert state["history_version"] == 1
    assert state["settings_version"] == 1
    assert state["behavior_snapshot"]["status"] == "valid"
    assert state["behavior_snapshot"]["canonical_json"] == snapshot.canonical_bytes.decode()
    assert state["behavior_snapshot"]["payload"] == snapshot.payload


def test_postgres_backend_wrapper_caller_owned_message_mutations_advance_once() -> None:
    backend = _PostgresStoreBackend()
    db, wrapper = _postgres_store_db(backend)
    store = db.message_store

    message_id = store.add_message(
        {
            "id": "postgres-message",
            "conversation_id": backend.conversation_id,
            "sender": "user",
            "content": "Original",
        },
        conn=wrapper,
    )
    assert message_id == "postgres-message"
    assert backend.history_version == 2

    assert store.update_message(
        message_id,
        {"content": "Edited"},
        1,
        conn=wrapper,
    )
    assert backend.history_version == 3

    assert store.append_message_image(
        message_id,
        b"postgres-image",
        "image/png",
        conn=wrapper,
    ) == 0
    assert backend.history_version == 4

    assert store.set_message_metadata_extra(
        message_id,
        {"pinned": True},
        conn=wrapper,
    )
    assert backend.history_version == 5
    assert store.set_message_metadata_extra(
        message_id,
        {"pinned": True},
        conn=wrapper,
    )
    assert backend.history_version == 5

    assert store.soft_delete_message(message_id, 2, conn=wrapper)
    assert backend.history_version == 6


@pytest.mark.parametrize("caller_owned", [False, True])
@pytest.mark.parametrize(
    ("backend_message", "expected_error"),
    [
        ("duplicate key value violates unique constraint messages_pkey", ConflictError),
        ("insert violates foreign key constraint messages_conversation_id_fkey", CharactersRAGDBError),
    ],
)
def test_postgres_backend_wrapper_add_message_normalizes_backend_errors(
    caller_owned: bool,
    backend_message: str,
    expected_error: type[Exception],
) -> None:
    backend = _PostgresStoreBackend(message_insert_error=backend_message)
    db, wrapper = _postgres_store_db(backend)
    if not caller_owned:
        db.transaction = lambda: nullcontext(wrapper)

    kwargs = {"conn": wrapper} if caller_owned else {}
    with pytest.raises(expected_error):
        db.message_store.add_message(
            {
                "id": "failed-postgres-message",
                "conversation_id": backend.conversation_id,
                "sender": "user",
                "content": "Failure",
            },
            **kwargs,
        )

    assert backend.messages == {}
    assert backend.history_version == 1


def test_postgres_v64_contract_has_matching_constraints_indexes_and_initializer_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ddl = " ".join(CharactersRAGDB._MIGRATION_SQL_V63_TO_V64_POSTGRES.lower().split())
    assert "conversation_id text primary key" in ddl
    assert "references conversations(id) on delete cascade" in ddl
    assert "status in ('valid','missing','invalid')" in ddl
    assert "schema_version is not null" in ddl
    assert "schema_version >= 1" in ddl
    assert "size_bytes is not null" in ddl
    assert "digest ~ '^sha256:[0-9a-f]{64}$'" in ddl
    assert "octet_length(canonical_json) = size_bytes" in ddl
    assert "canonical_json is null" in ddl
    assert "create table if not exists conversation_behavior_snapshots" not in ddl
    assert "create index if not exists idx_conversation_behavior_snapshots_status" not in ddl
    assert "alter table conversations add column if not exists history_version" not in ddl
    assert "alter table conversations add column history_version" in ddl
    assert "check (history_version >= 1)" in ddl
    assert "alter table conversation_settings add column if not exists settings_version" not in ddl
    assert "alter table conversation_settings add column settings_version" in ddl
    assert "check (settings_version >= 1)" in ddl
    assert "idx_conversation_behavior_snapshots_status" in ddl
    assert "idx_conversation_behavior_snapshots_digest" in ddl

    backend = _PostgresRecordingBackend()
    db = _postgres_db(backend)
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda conn: 63)
    monkeypatch.setattr(db, "_ensure_postgres_fts", lambda conn: None)
    applied: list[str] = []

    def _record_migration(conn: object) -> None:
        applied.append("63-to-64")

    monkeypatch.setattr(db, "_migrate_from_v63_to_v64_postgres", _record_migration)
    db._initialize_schema_postgres()

    assert applied == ["63-to-64"]


def test_postgres_v64_migration_executes_exact_version_advance() -> None:
    backend = _PostgresRecordingBackend()
    db = _postgres_db(backend)

    with backend.transaction() as conn:
        db._migrate_from_v63_to_v64_postgres(conn)

    assert backend.schema_version == 64
    assert any("conversation_behavior_snapshots" in sql for sql in backend.committed_statements)
    assert any(
        "where schema_name = %s and version = %s returning version" in " ".join(sql.lower().split())
        for sql in backend.committed_statements
    )


@pytest.mark.parametrize("drift_object", ["table", "index", "column"])
def test_postgres_v64_rejects_partial_preexisting_schema_and_preserves_v63(
    drift_object: str,
) -> None:
    backend = _PostgresRecordingBackend(drift_object=drift_object)
    db = _postgres_db(backend)

    with pytest.raises(BackendDatabaseError, match="pre-existing PostgreSQL"):
        with backend.transaction() as conn:
            db._migrate_from_v63_to_v64_postgres(conn)

    assert backend.rolled_back is True
    assert backend.schema_version == 63
    assert backend.committed_statements == []


def test_postgres_v64_checkpoint_failure_uses_outer_transaction_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _PostgresRecordingBackend()
    db = _postgres_db(backend)
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda conn: 63)

    def _fail_checkpoint(_self: CharactersRAGDB, stage: str) -> None:
        assert stage == "schema-created"
        raise RuntimeError("injected postgres v64 checkpoint failure")

    monkeypatch.setattr(CharactersRAGDB, "_migration_v64_checkpoint", _fail_checkpoint)
    with pytest.raises(RuntimeError, match="injected postgres v64 checkpoint failure"):
        db._initialize_schema_postgres()

    assert backend.rolled_back is True
    assert backend.schema_version == 63
    assert backend.committed_statements == []
    assert any("conversation_behavior_snapshots" in sql for sql in backend._pending) is False
