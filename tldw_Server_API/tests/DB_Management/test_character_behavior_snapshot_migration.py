"""Migration and store contract tests for character behavior snapshots."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import (
    BehaviorSnapshotV1,
    build_behavior_snapshot,
)
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
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
        assert "schema_version >= 1" in table_sql
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
    for suffix in range(7):
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

    def __init__(self) -> None:
        self._pending: list[str] = []
        self.committed_statements: list[str] = []
        self.rolled_back = False
        self.schema_version = 63

    def transaction(self) -> _PostgresTransaction:
        return _PostgresTransaction(self)

    def table_exists(self, _name: str, connection: object | None = None) -> bool:
        return True

    def execute(self, statement: str, *args: Any, **_kwargs: Any) -> SimpleNamespace:
        self._pending.append(str(statement))
        normalized = " ".join(str(statement).lower().split())
        params = args[0] if args else None
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


def test_postgres_v64_contract_has_matching_constraints_indexes_and_initializer_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ddl = " ".join(CharactersRAGDB._MIGRATION_SQL_V63_TO_V64_POSTGRES.lower().split())
    assert "conversation_id text primary key" in ddl
    assert "references conversations(id) on delete cascade" in ddl
    assert "status in ('valid','missing','invalid')" in ddl
    assert "schema_version >= 1" in ddl
    assert "digest ~ '^sha256:[0-9a-f]{64}$'" in ddl
    assert "octet_length(canonical_json) = size_bytes" in ddl
    assert "canonical_json is null" in ddl
    assert "alter table conversations add column if not exists history_version" in ddl
    assert "check (history_version >= 1)" in ddl
    assert "alter table conversation_settings add column if not exists settings_version" in ddl
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
