from __future__ import annotations

import inspect
import sqlite3
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.Chat_Macros.exceptions import MacroStorageError
from tldw_Server_API.app.core.Chat_Macros.repository import ChatMacroRepository
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


@pytest.fixture()
def raw_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "macros.db"), client_id="test_client")
    try:
        yield db
    finally:
        db.close_connection()


@pytest.fixture()
def repo(raw_db):
    return ChatMacroRepository(raw_db)


def _schema_version(db: CharactersRAGDB) -> int:
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()
    return int(row["version"])


def _table_names(db: CharactersRAGDB) -> set[str]:
    with db.transaction() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


def _index_names(db: CharactersRAGDB) -> set[str]:
    with db.transaction() as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    return {row["name"] for row in rows}


def _index_sql(db: CharactersRAGDB, index_name: str) -> str:
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (index_name,),
        ).fetchone()
    return row["sql"] if row is not None else ""


def test_create_run_final_post_and_branch_round_trip(repo):
    run = repo.create_run(
        user_id="1",
        macro_name="wrapup",
        macro_command="wrapup",
        normalized_args={},
    )
    assert repo.get_run(run.run_id).macro_name == "wrapup"

    repo.store_final_output(run.run_id, final_output="Done", final_output_format="markdown")
    repo.mark_final_posted(
        run.run_id,
        final_message_id="msg-1",
        post_idempotency_key="macro:run:post",
    )
    saved = repo.get_run(run.run_id)
    assert saved.final_message_id == "msg-1"
    assert saved.final_output == "Done"
    assert saved.final_output_format == "markdown"
    assert saved.normalized_args == {}

    repo.upsert_branch(
        run.run_id,
        step_id="summary",
        label="Summary",
        status="completed",
        output_text="S",
    )
    branches = repo.list_branches(run.run_id)
    assert branches[0].output_text == "S"
    assert branches[0].label == "Summary"


def test_mark_final_posted_allows_same_post_idempotency_key_on_different_runs(repo):
    first = repo.create_run(
        user_id="1",
        macro_name="wrapup",
        macro_command="wrapup",
        normalized_args={},
    )
    second = repo.create_run(
        user_id="1",
        macro_name="wrapup",
        macro_command="wrapup",
        normalized_args={},
    )

    repo.mark_final_posted(
        first.run_id,
        final_message_id="msg-1",
        post_idempotency_key="macro:shared",
    )
    repo.mark_final_posted(
        second.run_id,
        final_message_id="msg-2",
        post_idempotency_key="macro:shared",
    )

    assert repo.get_run(first.run_id).final_message_id == "msg-1"
    assert repo.get_run(second.run_id).final_message_id == "msg-2"


def test_mark_final_posted_is_idempotent_for_same_run_key_and_message(repo):
    run = repo.create_run(
        user_id="1",
        macro_name="wrapup",
        macro_command="wrapup",
        normalized_args={},
    )

    posted = repo.mark_final_posted(
        run.run_id,
        final_message_id="msg-1",
        post_idempotency_key="macro:shared",
    )
    retried = repo.mark_final_posted(
        run.run_id,
        final_message_id="msg-1",
        post_idempotency_key="macro:shared",
    )

    assert posted.final_message_id == "msg-1"
    assert retried.final_message_id == "msg-1"
    assert retried.post_idempotency_key == "macro:shared"


def test_mark_final_posted_rejects_same_key_with_different_message(repo):
    run = repo.create_run(
        user_id="1",
        macro_name="wrapup",
        macro_command="wrapup",
        normalized_args={},
    )

    repo.mark_final_posted(
        run.run_id,
        final_message_id="msg-1",
        post_idempotency_key="macro:shared",
    )

    with pytest.raises(MacroStorageError, match="idempotency"):
        repo.mark_final_posted(
            run.run_id,
            final_message_id="msg-2",
            post_idempotency_key="macro:shared",
        )
    assert repo.get_run(run.run_id).final_message_id == "msg-1"


def test_mark_final_posted_rejects_different_key_after_posted(repo):
    run = repo.create_run(
        user_id="1",
        macro_name="wrapup",
        macro_command="wrapup",
        normalized_args={},
    )

    repo.mark_final_posted(
        run.run_id,
        final_message_id="msg-1",
        post_idempotency_key="macro:first",
    )

    with pytest.raises(MacroStorageError, match="already posted"):
        repo.mark_final_posted(
            run.run_id,
            final_message_id="msg-2",
            post_idempotency_key="macro:second",
        )
    assert repo.get_run(run.run_id).post_idempotency_key == "macro:first"


def test_request_cancel_records_timestamp_and_status(repo):
    run = repo.create_run(
        user_id="1",
        macro_name="wrapup",
        macro_command="wrapup",
        normalized_args={},
    )

    repo.request_cancel(run.run_id)

    saved = repo.get_run(run.run_id)
    assert saved.status == "cancel_requested"
    assert saved.cancel_requested_at is not None


def test_cancel_request_is_noop_for_terminal_runs(repo):
    run = repo.create_run(
        user_id="1",
        macro_name="wrapup",
        macro_command="wrapup",
        normalized_args={},
    )
    repo.update_run_status(run.run_id, status="completed")

    cancelled = repo.request_cancel(run.run_id)

    assert cancelled.status == "completed"
    assert cancelled.cancel_requested_at is None


def test_cancel_requested_run_cannot_be_started_or_completed(repo):
    run = repo.create_run(
        user_id="1",
        macro_name="wrapup",
        macro_command="wrapup",
        normalized_args={},
    )
    repo.request_cancel(run.run_id)

    assert repo.update_run_status(run.run_id, status="running").status == "cancel_requested"
    assert repo.update_run_status(run.run_id, status="completed").status == "cancel_requested"
    assert repo.update_run_status(run.run_id, status="cancelled").status == "cancelled"


def test_repository_uses_guarded_updates_for_race_sensitive_fields():
    status_source = inspect.getsource(ChatMacroRepository.update_run_status)
    cancel_source = inspect.getsource(ChatMacroRepository.request_cancel)
    post_source = inspect.getsource(ChatMacroRepository.mark_final_posted)

    assert "status NOT IN ('completed', 'failed', 'cancelled')" in status_source
    assert "status != 'cancel_requested' OR ? IN ('cancelled', 'failed')" in status_source
    assert "status NOT IN ('completed', 'failed', 'cancelled')" in cancel_source
    assert "final_message_id IS NULL" in post_source
    assert "post_idempotency_key IS NULL" in post_source


def test_registry_settings_and_status_methods(repo):
    repo.ensure_ready()

    entry = repo.upsert_registry_entry(
        user_id="1",
        name="Wrap Up",
        command="wrapup",
        description="Summarize current work.",
        enabled=True,
        source="builtin",
        builtin_version=1,
        schema_version=1,
        digest="abc123",
        validation_status="valid",
        validation_error=None,
    )
    assert entry["command"] == "wrapup"
    assert repo.list_registry_entries("1")[0]["digest"] == "abc123"

    assert repo.get_settings("1") == {}
    repo.save_settings("1", {"output_profiles": [{"name": "brief"}]})
    assert repo.get_settings("1") == {"output_profiles": [{"name": "brief"}]}

    run = repo.create_run(
        user_id="1",
        macro_name="wrapup",
        macro_command="wrapup",
        normalized_args={"mode": "foreground"},
    )
    repo.update_run_status(run.run_id, status="running", error_code=None, error_message=None)
    saved = repo.get_run(run.run_id)
    assert saved.status == "running"
    assert saved.normalized_args == {"mode": "foreground"}


def test_fresh_db_creates_chat_macro_tables_and_indexes(raw_db):
    assert _schema_version(raw_db) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    assert {
        "chat_macro_registry",
        "chat_macro_settings",
        "chat_macro_runs",
        "chat_macro_run_branches",
    }.issubset(_table_names(raw_db))
    assert {
        "idx_chat_macro_registry_user_command",
        "idx_chat_macro_runs_user_status_created",
        "idx_chat_macro_runs_run_post_idempotency_key_unique",
        "idx_chat_macro_run_branches_run_step",
    }.issubset(_index_names(raw_db))
    index_sql = _index_sql(raw_db, "idx_chat_macro_runs_run_post_idempotency_key_unique")
    assert "run_id, post_idempotency_key" in index_sql


def test_current_schema_self_heals_missing_chat_macro_tables(tmp_path):
    db_path = tmp_path / "drifted.db"
    db = CharactersRAGDB(str(db_path), client_id="test_client")
    db.close_connection()

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(
            """
            DROP TABLE IF EXISTS chat_macro_run_branches;
            DROP TABLE IF EXISTS chat_macro_runs;
            DROP TABLE IF EXISTS chat_macro_settings;
            DROP TABLE IF EXISTS chat_macro_registry;
            """
        )

    healed = CharactersRAGDB(str(db_path), client_id="test_client")
    try:
        assert _schema_version(healed) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        assert "chat_macro_runs" in _table_names(healed)
        assert "idx_chat_macro_runs_run_post_idempotency_key_unique" in _index_names(healed)
    finally:
        healed.close_connection()


def test_postgres_chat_macro_schema_extension_contract_and_routing():
    script = getattr(CharactersRAGDB, "_CHAT_MACROS_SCHEMA_SQL", "")
    assert "CREATE TABLE IF NOT EXISTS chat_macro_registry" in script
    assert "CREATE TABLE IF NOT EXISTS chat_macro_settings" in script
    assert "CREATE TABLE IF NOT EXISTS chat_macro_runs" in script
    assert "CREATE TABLE IF NOT EXISTS chat_macro_run_branches" in script
    assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_macro_runs_run_post_idempotency_key_unique" in script
    assert "ON chat_macro_runs(run_id, post_idempotency_key)" in script
    assert "WHERE post_idempotency_key IS NOT NULL" in script
    assert "UPDATE db_schema_version" not in script

    db = CharactersRAGDB(":memory:", client_id="test_client")
    try:
        postgres_script = "\n".join(db._convert_sqlite_schema_to_postgres_statements(script))
    finally:
        db.close_connection()
    assert "TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP" in postgres_script
    assert "BOOLEAN NOT NULL DEFAULT FALSE" in postgres_script

    postgres_initializer = inspect.getsource(CharactersRAGDB._initialize_schema_postgres)
    assert "_ensure_chat_macros_schema_postgres" in postgres_initializer

    postgres_ensure = inspect.getsource(CharactersRAGDB._ensure_chat_macros_schema_postgres)
    assert "_CHAT_MACROS_SCHEMA_SQL" in postgres_ensure


def test_row_mapping_normalizes_datetime_timestamps():
    now = datetime(2026, 7, 3, 12, 30, tzinfo=timezone.utc)

    run = ChatMacroRepository._map_run(
        {
            "run_id": "run-1",
            "user_id": "1",
            "macro_name": "wrapup",
            "macro_command": "wrapup",
            "normalized_args": "{}",
            "created_at": now,
            "started_at": now,
            "completed_at": now,
            "cancel_requested_at": now,
        }
    )
    branch = ChatMacroRepository._map_branch(
        {
            "branch_id": "branch-1",
            "run_id": "run-1",
            "step_id": "summary",
            "citations": "[]",
            "usage": "{}",
            "created_at": now,
            "completed_at": now,
        }
    )

    assert run.created_at == now.isoformat()
    assert run.cancel_requested_at == now.isoformat()
    assert branch.created_at == now.isoformat()
    assert branch.completed_at == now.isoformat()


def test_malformed_run_json_raises_macro_storage_error(repo):
    run = repo.create_run(
        user_id="1",
        macro_name="wrapup",
        macro_command="wrapup",
        normalized_args={},
    )
    with repo.db.transaction() as conn:
        conn.execute(
            "UPDATE chat_macro_runs SET normalized_args = ? WHERE run_id = ?",
            ("{bad", run.run_id),
        )

    with pytest.raises(MacroStorageError, match="normalized_args"):
        repo.get_run(run.run_id)
