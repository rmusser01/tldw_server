# tests/unit/core/Prompts_Management/test_prompts_db_v2.py
# Description:
#
# Imports
import json
import os
import re
import sqlite3
import uuid
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from loguru import logger

#
# Local Imports
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    ConflictError,
    DatabaseError,
    InputError,
    PromptsDatabase,
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    # Standalone functions
    add_or_update_prompt as standalone_add_or_update_prompt,
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    export_prompt_keywords_to_csv as standalone_export_prompt_keywords_to_csv,
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    export_prompts_formatted as standalone_export_prompts_formatted,
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    load_prompt_details_for_ui as standalone_load_prompt_details_for_ui,
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    view_prompt_keywords_markdown as standalone_view_prompt_keywords_markdown,
)
from tldw_Server_API.app.core.Prompt_Management.service_prompts import (
    ServicePromptCorruptOverride,
    resolve_service_prompt,
)

#
########################################################################################################################
#
# Functions:

TEST_CLIENT_ID = "test_db_client"
SENSITIVE_SQLITE_ERROR = "PROMPT_BODY_MUST_NOT_APPEAR /private/DB_PATH_MUST_NOT_APPEAR.db"


@pytest.fixture
def memory_db():
    """Provides an in-memory PromptsDatabase instance for testing."""
    db = PromptsDatabase(db_path=":memory:", client_id=TEST_CLIENT_ID)
    yield db
    db.close_connection()


@pytest.fixture
def file_db(tmp_path):
    """Provides a file-based PromptsDatabase instance for testing."""
    db_file = tmp_path / "test_prompts.db"
    db = PromptsDatabase(db_path=db_file, client_id=TEST_CLIENT_ID)
    yield db
    db.close_connection()
    if os.path.exists(db_file):
        os.remove(db_file)


class _InsertRaceConnection:
    """Inject one competing insert immediately before the store insert."""

    def __init__(self, conn: sqlite3.Connection, parts_json: str, revision: str):
        self._conn = conn
        self._parts_json = parts_json
        self._revision = revision
        self._injected = False

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def execute(self, sql, parameters=()):
        if not self._injected and sql.lstrip().upper().startswith("INSERT INTO SERVICEPROMPTOVERRIDES"):
            self._injected = True
            self._conn.execute(
                """
                INSERT INTO ServicePromptOverrides (definition_id, parts_json, revision)
                VALUES (?, ?, ?)
                """,
                (parameters[0], self._parts_json, self._revision),
            )
        return self._conn.execute(sql, parameters)


class _CommitFailingConnection:
    """Fail commits after delegating all transaction body operations."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def commit(self):
        raise sqlite3.OperationalError("PROMPT_BODY_MUST_NOT_APPEAR")


class _CommitRollbackFailingConnection:
    """Fail one commit and rollback, then delegate to the real connection."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._fail_commit = True
        self._fail_rollback = True

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def commit(self):
        if self._fail_commit:
            self._fail_commit = False
            raise sqlite3.OperationalError(SENSITIVE_SQLITE_ERROR)
        return self._conn.commit()

    def rollback(self):
        if self._fail_rollback:
            self._fail_rollback = False
            raise sqlite3.OperationalError(SENSITIVE_SQLITE_ERROR)
        return self._conn.rollback()

    def close(self):
        return self._conn.close()


class _BeginFailingConnection:
    """Fail transaction entry before any transaction body operation."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def execute(self, sql, parameters=()):
        if sql.strip().upper() == "BEGIN IMMEDIATE":
            raise sqlite3.OperationalError(SENSITIVE_SQLITE_ERROR)
        return self._conn.execute(sql, parameters)


# --- Test PromptsDatabase Class ---


def test_database_initialization_memory(memory_db):

    assert memory_db is not None
    assert memory_db.client_id == TEST_CLIENT_ID
    assert memory_db.is_memory_db is True
    # Check if schema version table exists and has version 1
    conn = memory_db.get_connection()
    cursor = conn.execute("SELECT version FROM schema_version")
    assert cursor.fetchone()["version"] == PromptsDatabase._CURRENT_SCHEMA_VERSION


def test_database_initialization_file(file_db):

    assert file_db is not None
    assert file_db.client_id == TEST_CLIENT_ID
    assert file_db.is_memory_db is False
    assert os.path.exists(file_db.db_path)
    conn = file_db.get_connection()
    cursor = conn.execute("SELECT version FROM schema_version")
    assert cursor.fetchone()["version"] == PromptsDatabase._CURRENT_SCHEMA_VERSION


def test_schema_v1_migrates_to_v2_with_collections(tmp_path):
    db_file = tmp_path / "test_prompts_v1.db"

    with sqlite3.connect(db_file) as legacy_conn:
        legacy_conn.executescript(
            f"""
            {PromptsDatabase._TABLES_SQL_V1}
            {PromptsDatabase._INDICES_SQL_V1}
            {PromptsDatabase._TRIGGERS_SQL_V1}
            {PromptsDatabase._SCHEMA_UPDATE_VERSION_SQL_V1}
            """
        )
        legacy_conn.commit()
        legacy_version = legacy_conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert legacy_version == 1
        assert (
            legacy_conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='PromptCollections'"
            ).fetchone()
            is None
        )

    migrated_db = PromptsDatabase(db_path=db_file, client_id=TEST_CLIENT_ID)
    try:
        conn = migrated_db.get_connection()
        version_row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert version_row["version"] == PromptsDatabase._CURRENT_SCHEMA_VERSION
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='PromptCollections'"
        ).fetchone() is not None
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='PromptCollectionItems'"
        ).fetchone() is not None
    finally:
        migrated_db.close_connection()


def test_fresh_database_schema_v6_includes_service_prompt_overrides(memory_db):
    conn = memory_db.get_connection()

    version = conn.execute("SELECT version FROM schema_version").fetchone()["version"]
    table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ServicePromptOverrides'").fetchone()

    assert version == 6
    assert table is not None


def test_schema_v5_migrates_to_v6_preserving_prompt(tmp_path, monkeypatch):
    db_file = tmp_path / "test_prompts_v5.db"
    monkeypatch.setattr(PromptsDatabase, "_CURRENT_SCHEMA_VERSION", 5)
    legacy_db = PromptsDatabase(db_path=db_file, client_id=TEST_CLIENT_ID)
    try:
        legacy_conn = legacy_db.get_connection()
        assert legacy_conn.execute("SELECT version FROM schema_version").fetchone()["version"] == 5
        assert (
            legacy_conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ServicePromptOverrides'"
            ).fetchone()
            is None
        )
        prompt_id, prompt_uuid, _ = legacy_db.add_prompt(
            name="Prompt retained across v6 migration",
            author="Tester",
            details="Migration sentinel",
        )
    finally:
        legacy_db.close_connection()

    monkeypatch.setattr(PromptsDatabase, "_CURRENT_SCHEMA_VERSION", 6)
    migrated_db = PromptsDatabase(db_path=db_file, client_id=TEST_CLIENT_ID)
    try:
        conn = migrated_db.get_connection()
        version = conn.execute("SELECT version FROM schema_version").fetchone()["version"]
        table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ServicePromptOverrides'"
        ).fetchone()
        prompt = migrated_db.get_prompt_by_id(prompt_id)

        assert version == 6
        assert table is not None
        assert prompt is not None
        assert prompt["uuid"] == prompt_uuid
        assert prompt["name"] == "Prompt retained across v6 migration"
    finally:
        migrated_db.close_connection()


def test_service_prompt_override_raw_read_is_absent_then_preserves_stored_row(memory_db):
    definition_id = "chat.rag.answer"
    assert memory_db.get_service_prompt_override(definition_id) is None

    raw_parts_json = '{ "template" : "Raw stored prompt" }'
    revision = str(uuid.uuid4())
    conn = memory_db.get_connection()
    conn.execute(
        """
        INSERT INTO ServicePromptOverrides (definition_id, parts_json, revision)
        VALUES (?, ?, ?)
        """,
        (definition_id, raw_parts_json, revision),
    )
    conn.commit()

    row = memory_db.get_service_prompt_override(definition_id)

    assert row is not None
    assert row.definition_id == definition_id
    assert row.parts_json == raw_parts_json
    assert row.revision == revision


def test_service_prompt_override_first_insert_uses_deterministic_json_and_uuid(memory_db):
    parts = {"user_template": "Translate {text}", "system": "Be exact."}

    row = memory_db.save_service_prompt_override(
        "media.text.translation",
        parts,
        expected_revision=None,
    )

    assert row.definition_id == "media.text.translation"
    assert row.parts_json == json.dumps(parts, sort_keys=True, separators=(",", ":"))
    assert str(uuid.UUID(row.revision, version=4)) == row.revision
    assert memory_db.get_service_prompt_override("media.text.translation") == row
    with pytest.raises(FrozenInstanceError):
        row.revision = "changed"


def test_service_prompt_override_identical_save_is_no_op(memory_db):
    parts = {"template": "Answer from {context}: {question}"}
    first = memory_db.save_service_prompt_override("chat.rag.answer", parts, None)

    repeated = memory_db.save_service_prompt_override(
        "chat.rag.answer",
        dict(parts),
        first.revision,
    )

    assert repeated == first


def test_service_prompt_override_identical_stale_retry_returns_current_row(memory_db):
    first = memory_db.save_service_prompt_override(
        "chat.rag.answer",
        {"template": "First {context} {question}"},
        None,
    )
    current_parts = {"template": "Current {context} {question}"}
    current = memory_db.save_service_prompt_override(
        "chat.rag.answer",
        current_parts,
        first.revision,
    )

    retried = memory_db.save_service_prompt_override(
        "chat.rag.answer",
        current_parts,
        first.revision,
    )

    assert retried == current


def test_service_prompt_override_content_change_uses_cas_and_new_uuid(memory_db):
    first = memory_db.save_service_prompt_override(
        "chat.rag.answer",
        {"template": "First {context} {question}"},
        None,
    )

    updated = memory_db.save_service_prompt_override(
        "chat.rag.answer",
        {"template": "Updated {context} {question}"},
        first.revision,
    )

    assert updated.revision != first.revision
    assert str(uuid.UUID(updated.revision, version=4)) == updated.revision
    assert memory_db.get_service_prompt_override("chat.rag.answer") == updated


@pytest.mark.parametrize("expected_revision", [None, str(uuid.uuid4())])
def test_service_prompt_override_changed_save_conflicts_with_current_revision(
    memory_db,
    expected_revision,
):
    current = memory_db.save_service_prompt_override(
        "chat.rag.answer",
        {"template": "Current {context} {question}"},
        None,
    )

    with pytest.raises(ConflictError) as captured:
        memory_db.save_service_prompt_override(
            "chat.rag.answer",
            {"template": "Rejected {context} {question}"},
            expected_revision,
        )

    assert captured.type.__name__ == "ServicePromptRevisionConflict"
    assert captured.value.current_revision == current.revision
    assert str(captured.value) == "Service Prompt override changed concurrently."
    assert memory_db.get_service_prompt_override("chat.rag.answer") == current


@pytest.mark.parametrize(
    ("competing_parts", "expect_conflict"),
    [
        ({"template": "Requested {context} {question}"}, False),
        ({"template": "Competing {context} {question}"}, True),
    ],
)
def test_service_prompt_override_insert_race_refetches_and_classifies(
    memory_db,
    monkeypatch,
    competing_parts,
    expect_conflict,
):
    definition_id = "chat.rag.answer"
    requested_parts = {"template": "Requested {context} {question}"}
    assert memory_db.get_service_prompt_override(definition_id) is None
    race_revision = str(uuid.uuid4())
    race_conn = _InsertRaceConnection(
        memory_db.get_connection(),
        json.dumps(competing_parts, sort_keys=True, separators=(",", ":")),
        race_revision,
    )
    monkeypatch.setattr(memory_db, "get_connection", lambda: race_conn)

    if expect_conflict:
        with pytest.raises(ConflictError) as captured:
            memory_db.save_service_prompt_override(definition_id, requested_parts, None)
        assert captured.type.__name__ == "ServicePromptRevisionConflict"
        assert captured.value.current_revision == race_revision
    else:
        row = memory_db.save_service_prompt_override(definition_id, requested_parts, None)
        assert row.revision == race_revision
        assert json.loads(row.parts_json) == requested_parts


def test_service_prompt_override_absent_reset_without_revision_is_idempotent(memory_db):
    assert memory_db.reset_service_prompt_override("chat.rag.answer", None) is None


def test_service_prompt_override_matching_reset_deletes_row(memory_db):
    current = memory_db.save_service_prompt_override(
        "chat.rag.answer",
        {"template": "Saved {context} {question}"},
        None,
    )

    assert memory_db.reset_service_prompt_override("chat.rag.answer", current.revision) is None
    assert memory_db.get_service_prompt_override("chat.rag.answer") is None


def test_service_prompt_override_stale_and_already_reset_revision_conflict(memory_db):
    current = memory_db.save_service_prompt_override(
        "chat.rag.answer",
        {"template": "Saved {context} {question}"},
        None,
    )

    with pytest.raises(ConflictError) as stale:
        memory_db.reset_service_prompt_override("chat.rag.answer", str(uuid.uuid4()))
    assert stale.type.__name__ == "ServicePromptRevisionConflict"
    assert stale.value.current_revision == current.revision

    assert memory_db.reset_service_prompt_override("chat.rag.answer", current.revision) is None
    with pytest.raises(ConflictError) as already_reset:
        memory_db.reset_service_prompt_override("chat.rag.answer", current.revision)
    assert already_reset.type.__name__ == "ServicePromptRevisionConflict"
    assert already_reset.value.current_revision is None


def test_service_prompt_override_corrupt_json_is_readable_and_resettable(memory_db):
    definition_id = "chat.rag.answer"
    assert memory_db.get_service_prompt_override(definition_id) is None
    revision = str(uuid.uuid4())
    corrupt_parts_json = '{"template": PROMPT_BODY_MUST_NOT_APPEAR'
    conn = memory_db.get_connection()
    conn.execute(
        """
        INSERT INTO ServicePromptOverrides (definition_id, parts_json, revision)
        VALUES (?, ?, ?)
        """,
        (definition_id, corrupt_parts_json, revision),
    )
    conn.commit()

    row = memory_db.get_service_prompt_override(definition_id)

    assert row is not None
    assert row.revision == revision
    assert row.parts_json == corrupt_parts_json
    assert memory_db.reset_service_prompt_override(definition_id, revision) is None
    assert memory_db.get_service_prompt_override(definition_id) is None


def test_service_prompt_override_undecodable_text_can_be_reset_without_reading_content(memory_db):
    definition_id = "chat.rag.answer"
    revision = str(uuid.uuid4())
    conn = memory_db.get_connection()
    conn.execute(
        """
        INSERT INTO ServicePromptOverrides (definition_id, parts_json, revision)
        VALUES (?, CAST(X'80' AS TEXT), ?)
        """,
        (definition_id, revision),
    )
    conn.commit()

    assert memory_db.reset_service_prompt_override(definition_id, revision) is None
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM ServicePromptOverrides WHERE definition_id = ?",
            (definition_id,),
        ).fetchone()[0]
        == 0
    )


def test_service_prompt_override_undecodable_text_preserves_revision_for_resolver_and_reset(file_db):
    definition_id = "chat.rag.answer"
    revision = str(uuid.uuid4())
    conn = file_db.get_connection()
    conn.execute(
        """
        INSERT INTO ServicePromptOverrides (definition_id, parts_json, revision)
        VALUES (?, CAST(X'80' AS TEXT), ?)
        """,
        (definition_id, revision),
    )
    conn.commit()

    row = file_db.get_service_prompt_override(definition_id)

    assert row is not None
    assert row.definition_id == definition_id
    assert row.parts_json == b"\x80"
    assert row.revision == revision
    with pytest.raises(ServicePromptCorruptOverride) as captured:
        resolve_service_prompt(file_db, definition_id)
    assert captured.value.revision == revision

    assert file_db.reset_service_prompt_override(definition_id, row.revision) is None
    file_db.close_connection()
    assert file_db.get_service_prompt_override(definition_id) is None


def test_service_prompt_override_failed_save_rolls_back_without_leaking_content(memory_db):
    definition_id = "chat.rag.answer"
    original = memory_db.save_service_prompt_override(
        definition_id,
        {"template": "Original {context} {question}"},
        None,
    )
    conn = memory_db.get_connection()
    conn.execute(
        """
        CREATE TRIGGER fail_service_prompt_override_update
        AFTER UPDATE ON ServicePromptOverrides
        BEGIN
            SELECT RAISE(ABORT, 'PROMPT_BODY_MUST_NOT_APPEAR');
        END
        """
    )
    conn.commit()

    log_messages = []
    sink_id = logger.add(log_messages.append, format="{message}")
    try:
        with pytest.raises(DatabaseError) as captured:
            memory_db.save_service_prompt_override(
                definition_id,
                {"template": "Rejected PROMPT_BODY_MUST_NOT_APPEAR"},
                original.revision,
            )
    finally:
        logger.remove(sink_id)

    assert str(captured.value) == "Failed to save Service Prompt override."
    assert "PROMPT_BODY_MUST_NOT_APPEAR" not in str(captured.value)
    rendered_logs = "".join(str(message) for message in log_messages)
    assert "PROMPT_BODY_MUST_NOT_APPEAR" not in rendered_logs
    assert "Transaction failed, rolling back: IntegrityError" in rendered_logs
    assert "Rollback successful." in rendered_logs
    assert memory_db.get_service_prompt_override(definition_id) == original


@pytest.mark.parametrize(
    ("operation", "safe_message"),
    [
        ("save", "Failed to save Service Prompt override."),
        ("reset", "Failed to reset Service Prompt override."),
    ],
)
def test_service_prompt_override_begin_immediate_failure_is_wrapped_without_mutation_or_sensitive_logs(
    memory_db,
    monkeypatch,
    operation,
    safe_message,
):
    definition_id = "chat.rag.answer"
    original = memory_db.save_service_prompt_override(
        definition_id,
        {"template": "Original {context} {question}"},
        None,
    )
    failing_conn = _BeginFailingConnection(memory_db.get_connection())
    monkeypatch.setattr(memory_db, "get_connection", lambda: failing_conn)

    log_messages = []
    sink_id = logger.add(log_messages.append, format="{message}")
    try:
        with pytest.raises(DatabaseError) as captured:
            if operation == "save":
                memory_db.save_service_prompt_override(
                    definition_id,
                    {"template": "Rejected {context} {question}"},
                    original.revision,
                )
            else:
                memory_db.reset_service_prompt_override(definition_id, original.revision)
    finally:
        logger.remove(sink_id)

    assert type(captured.value) is DatabaseError
    assert str(captured.value) == safe_message
    rendered_logs = "".join(str(message) for message in log_messages)
    for sentinel in ("PROMPT_BODY_MUST_NOT_APPEAR", "DB_PATH_MUST_NOT_APPEAR"):
        assert sentinel not in str(captured.value)
        assert sentinel not in rendered_logs
    assert "Transaction failed, rolling back: OperationalError" in rendered_logs
    assert "Rollback successful." in rendered_logs
    assert memory_db.get_service_prompt_override(definition_id) == original


@pytest.mark.parametrize(
    ("operation", "safe_message"),
    [
        ("save", "Failed to save Service Prompt override."),
        ("reset", "Failed to reset Service Prompt override."),
    ],
)
def test_service_prompt_override_commit_failure_is_wrapped_and_rolled_back(
    memory_db,
    monkeypatch,
    operation,
    safe_message,
):
    definition_id = "chat.rag.answer"
    original = memory_db.save_service_prompt_override(
        definition_id,
        {"template": "Original {context} {question}"},
        None,
    )
    failing_conn = _CommitFailingConnection(memory_db.get_connection())
    monkeypatch.setattr(memory_db, "get_connection", lambda: failing_conn)

    log_messages = []
    sink_id = logger.add(log_messages.append, format="{message}")
    try:
        with pytest.raises(DatabaseError) as captured:
            if operation == "save":
                memory_db.save_service_prompt_override(
                    definition_id,
                    {"template": "Rejected {context} {question}"},
                    original.revision,
                )
            else:
                memory_db.reset_service_prompt_override(definition_id, original.revision)
    finally:
        logger.remove(sink_id)

    assert str(captured.value) == safe_message
    assert "PROMPT_BODY_MUST_NOT_APPEAR" not in str(captured.value)
    assert all("PROMPT_BODY_MUST_NOT_APPEAR" not in str(message) for message in log_messages)
    assert memory_db.get_service_prompt_override(definition_id) == original


@pytest.mark.parametrize(
    ("operation", "safe_message"),
    [
        ("save", "Failed to save Service Prompt override."),
        ("reset", "Failed to reset Service Prompt override."),
    ],
)
def test_service_prompt_override_rollback_failure_retires_connection_and_discards_transaction(
    file_db,
    operation,
    safe_message,
):
    definition_id = "chat.rag.answer"
    original = file_db.save_service_prompt_override(
        definition_id,
        {"template": "Original {context} {question}"},
        None,
    )
    raw_connection = file_db.get_connection()
    failing_connection = _CommitRollbackFailingConnection(raw_connection)
    file_db._local.conn = failing_connection

    log_messages = []
    sink_id = logger.add(log_messages.append, format="{message}")
    try:
        with pytest.raises(DatabaseError) as captured:
            if operation == "save":
                file_db.save_service_prompt_override(
                    definition_id,
                    {"template": "Rejected PROMPT_BODY_MUST_NOT_APPEAR"},
                    original.revision,
                )
            else:
                file_db.reset_service_prompt_override(definition_id, original.revision)
    finally:
        logger.remove(sink_id)

    later_connection = file_db.get_connection()
    later_connection.commit()
    with sqlite3.connect(file_db.db_path) as observer:
        stored = observer.execute(
            """
            SELECT parts_json, revision
            FROM ServicePromptOverrides
            WHERE definition_id = ?
            """,
            (definition_id,),
        ).fetchone()

    assert type(captured.value) is DatabaseError
    assert str(captured.value) == safe_message
    rendered_logs = "".join(str(message) for message in log_messages)
    for sentinel in ("PROMPT_BODY_MUST_NOT_APPEAR", "DB_PATH_MUST_NOT_APPEAR"):
        assert sentinel not in str(captured.value)
        assert sentinel not in rendered_logs
    assert "Rollback FAILED: OperationalError" in rendered_logs
    assert stored == (original.parts_json, original.revision)
    assert later_connection is not failing_connection
    assert later_connection is not raw_connection
    assert file_db._local.conn is later_connection
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        raw_connection.execute("SELECT 1")


def test_initialization_empty_client_id():

    with pytest.raises(ValueError, match="Client ID cannot be empty or None."):
        PromptsDatabase(db_path=":memory:", client_id="")


def test_add_keyword(memory_db: PromptsDatabase):
    kw_id, kw_uuid = memory_db.add_keyword("test_keyword")
    assert kw_id is not None
    assert isinstance(kw_id, int)
    assert kw_uuid is not None
    assert isinstance(uuid.UUID(kw_uuid, version=4), uuid.UUID)

    # Check if keyword exists
    res = memory_db.execute_query("SELECT * FROM PromptKeywordsTable WHERE id = ?", (kw_id,)).fetchone()
    assert res is not None
    assert res["keyword"] == "test_keyword"  # Normalized
    assert res["deleted"] == 0

    # Add same keyword again (should raise ConflictError now)
    kw_id_2, kw_uuid_2 = memory_db.add_keyword(" TeSt_KeYwOrD ")  # Normalized to "test_keyword"
    assert kw_id_2 == kw_id
    assert kw_uuid_2 == kw_uuid

    # Add empty keyword
    with pytest.raises(InputError):
        memory_db.add_keyword("  ")


def test_add_prompt(memory_db: PromptsDatabase):
    p_id, p_uuid, msg = memory_db.add_prompt(
        name="My Test Prompt",
        author="Tester",
        details="A prompt for testing.",
        system_prompt="System instructions.",
        user_prompt="User query.",
        keywords=["test", "example"],
    )
    assert p_id is not None
    assert isinstance(p_id, int)
    assert p_uuid is not None
    assert "added" in msg.lower()

    prompt_data = memory_db.fetch_prompt_details(p_id)
    assert prompt_data is not None
    assert prompt_data["name"] == "My Test Prompt"
    assert prompt_data["author"] == "Tester"
    assert "test" in prompt_data["keywords"]
    assert "example" in prompt_data["keywords"]

    # Try adding same prompt name without overwrite - should raise ConflictError
    with pytest.raises(ConflictError) as excinfo:
        memory_db.add_prompt(name="My Test Prompt", author="New Author", details=None)  # overwrite defaults to False
    assert "already exists" in str(excinfo.value).lower()

    # Add with overwrite
    p_id3, p_uuid3, msg3 = memory_db.add_prompt(
        name="My Test Prompt", author="Updated Author", details="Updated details.", overwrite=True
    )
    assert p_id3 == p_id
    assert "updated" in msg3.lower()
    updated_prompt = memory_db.fetch_prompt_details(p_id)
    assert updated_prompt["author"] == "Updated Author"


def test_create_and_get_prompt_collection(memory_db: PromptsDatabase):
    p1_id, _, _ = memory_db.add_prompt(name="Collection Prompt A", author="Tester", details="A")
    p2_id, _, _ = memory_db.add_prompt(name="Collection Prompt B", author="Tester", details="B")

    created = memory_db.create_prompt_collection(
        name="Research Bundle",
        description="Useful prompts",
        prompt_ids=[p1_id, p2_id],
    )
    assert created["collection_id"] > 0
    assert created["name"] == "Research Bundle"
    assert created["prompt_ids"] == [p1_id, p2_id]

    fetched = memory_db.get_prompt_collection_by_id(created["collection_id"])
    assert fetched is not None
    assert fetched["collection_id"] == created["collection_id"]
    assert fetched["name"] == "Research Bundle"
    assert fetched["description"] == "Useful prompts"
    assert fetched["prompt_ids"] == [p1_id, p2_id]


def test_create_prompt_collection_validates_prompt_ids(memory_db: PromptsDatabase):
    p1_id, _, _ = memory_db.add_prompt(name="Collection Prompt Existing", author="Tester", details="A")
    with pytest.raises(InputError, match=r"Prompt\(s\) not found or deleted"):
        memory_db.create_prompt_collection(
            name="Invalid Bundle",
            description=None,
            prompt_ids=[p1_id, 999999],
        )


def test_soft_delete_and_undelete_prompt(memory_db: PromptsDatabase):
    p_id, _, _ = memory_db.add_prompt(name="Deletable Prompt", author="Test", details="Details")
    assert p_id is not None

    # Soft delete
    deleted = memory_db.soft_delete_prompt(p_id)
    assert deleted is True
    assert memory_db.get_prompt_by_id(p_id) is None
    assert memory_db.get_prompt_by_id(p_id, include_deleted=True) is not None

    # Try deleting again (should return False)
    deleted_again = memory_db.soft_delete_prompt(p_id)
    assert deleted_again is False

    # Undelete (by adding with overwrite=True)
    memory_db.add_prompt(name="Deletable Prompt", author="Test", details="Restored", overwrite=True)
    restored_prompt = memory_db.get_prompt_by_id(p_id)
    assert restored_prompt is not None
    assert restored_prompt["deleted"] == 0
    assert restored_prompt["details"] == "Restored"


def test_soft_delete_keyword_and_links(memory_db: PromptsDatabase):
    memory_db.add_prompt(name="Prompt With Keyword", author="Test", details="...", keywords=["deletable_kw"])
    kw_info = memory_db.execute_query("SELECT id FROM PromptKeywordsTable WHERE keyword='deletable_kw'").fetchone()
    assert kw_info is not None
    kw_id = kw_info["id"]

    # Check link exists
    link = memory_db.execute_query("SELECT * FROM PromptKeywordLinks WHERE keyword_id=?", (kw_id,)).fetchone()
    assert link is not None

    # Soft delete keyword
    deleted = memory_db.soft_delete_keyword("deletable_kw")
    assert deleted is True

    # Verify keyword is deleted
    assert (
        memory_db.execute_query(
            "SELECT id FROM PromptKeywordsTable WHERE keyword='deletable_kw' AND deleted=0"
        ).fetchone()
        is None
    )
    # Verify link is gone (due to cascade or explicit delete in soft_delete_keyword)
    assert memory_db.execute_query("SELECT * FROM PromptKeywordLinks WHERE keyword_id=?", (kw_id,)).fetchone() is None


def test_update_keywords_for_prompt(memory_db: PromptsDatabase):
    p_id, _, _ = memory_db.add_prompt(
        name="Keyword Update Prompt", author="Test", details="...", keywords=["initial1", "initial2"]
    )
    assert p_id is not None

    memory_db.update_keywords_for_prompt(p_id, ["initial2", "new1", "new2"])
    updated_keywords = memory_db.fetch_keywords_for_prompt(p_id)
    assert sorted(updated_keywords) == sorted(["initial2", "new1", "new2"])

    memory_db.update_keywords_for_prompt(p_id, [])  # Remove all
    assert memory_db.fetch_keywords_for_prompt(p_id) == []


def test_search_prompts_fts(memory_db: PromptsDatabase):
    memory_db.add_prompt(
        name="Alpha Search", author="AuthorA", details="Unique detail alpha", keywords=["common", "alpha_k"]
    )
    memory_db.add_prompt(
        name="Beta Search", author="AuthorB", details="Common detail beta", keywords=["common", "beta_k"]
    )
    memory_db.add_prompt(name="Gamma NonMatch", author="AuthorC", details="Different info", keywords=["other"])
    # time.sleep(0.1) # For in-memory SQLite, FTS updates are typically synchronous.

    results, total = memory_db.search_prompts(search_query="alpha")
    assert total == 1
    assert len(results) == 1
    assert results[0]["name"] == "Alpha Search"

    results_k, total_k = memory_db.search_prompts(search_query="common", search_fields=["keywords"])
    assert total_k == 2
    assert len(results_k) == 2

    results_detail, total_detail = memory_db.search_prompts(search_query="detail", search_fields=["details"])
    assert total_detail == 2  # "Unique detail alpha", "Common detail beta"

    # Test FTS on system/user prompts
    # FIX: Add missing 'author' and 'details' arguments
    memory_db.add_prompt(
        name="SysUserPrompt",
        author="TestAuthorFTS",  # Added
        details="TestDetailsFTS",  # Added
        system_prompt="System test phrase",
        user_prompt="User specific content",
    )
    results_sys, total_sys_val = memory_db.search_prompts(search_query="phrase", search_fields=["system_prompt"])
    assert total_sys_val == 1  # Check the total count from search_prompts
    assert len(results_sys) == 1
    assert results_sys[0]["name"] == "SysUserPrompt"


def test_sync_log(memory_db: PromptsDatabase):
    p_id, p_uuid, _ = memory_db.add_prompt(name="Sync Log Test Prompt", author="Sync", details="...")
    kw_id, kw_uuid = memory_db.add_keyword("sync_keyword")
    memory_db.update_keywords_for_prompt(p_id, ["sync_keyword"])  # This will log a link

    log_entries = memory_db.get_sync_log_entries()
    assert len(log_entries) >= 3  # create prompt, create keyword, link them

    create_prompt_entry = next(e for e in log_entries if e["entity_uuid"] == p_uuid and e["operation"] == "create")
    assert create_prompt_entry is not None
    assert create_prompt_entry["payload"]["name"] == "Sync Log Test Prompt"

    create_kw_entry = next(e for e in log_entries if e["entity_uuid"] == kw_uuid and e["operation"] == "create")
    assert create_kw_entry is not None

    link_entry = next(e for e in log_entries if e["entity"] == "PromptKeywordLinks" and e["operation"] == "link")
    assert link_entry is not None
    assert link_entry["payload"]["prompt_uuid"] == p_uuid
    assert link_entry["payload"]["keyword_uuid"] == kw_uuid

    # Test deleting sync log entries
    change_ids_to_delete = [e["change_id"] for e in log_entries[:2]]
    deleted_count = memory_db.delete_sync_log_entries(change_ids_to_delete)
    assert deleted_count == len(change_ids_to_delete)
    remaining_entries = memory_db.get_sync_log_entries()
    assert len(remaining_entries) == len(log_entries) - deleted_count


def test_versioning_and_conflict(memory_db: PromptsDatabase):
    p_id, p_uuid, _ = memory_db.add_prompt(name="Version Test", author="V1", details="Initial")
    prompt_v1 = memory_db.get_prompt_by_id(p_id)
    assert prompt_v1["version"] == 1

    # Simulate an update with correct version increment (via add_prompt with overwrite)
    memory_db.add_prompt(name="Version Test", author="V2", details="Updated", overwrite=True)
    prompt_v2 = memory_db.get_prompt_by_id(p_id)
    assert prompt_v2["version"] == 2

    # Simulate a direct DB update with incorrect version (should be blocked by trigger)
    conn = memory_db.get_connection()

    # FIX: Escape the regex string for the match argument
    expected_error_message = "Sync Error (Prompts): Version must increment by exactly 1."
    with pytest.raises(sqlite3.IntegrityError, match=re.escape(expected_error_message)):
        with memory_db.transaction():  # Use transaction context
            conn.execute(
                "UPDATE Prompts SET details = ?, version = ?, client_id = ?, last_modified = ? WHERE id = ?",
                (
                    "Conflict attempt",
                    prompt_v2["version"] + 2,
                    TEST_CLIENT_ID,
                    memory_db._get_current_utc_timestamp_str(),
                    p_id,
                ),
            )
            # The transaction context will attempt to commit, which is when the trigger's RAISE(ABORT)
            # (or RAISE(FAIL)) takes effect.


# --- Test Standalone Functions ---


def test_standalone_add_or_update_prompt(memory_db: PromptsDatabase):
    p_id, p_uuid, msg = standalone_add_or_update_prompt(
        memory_db, name="Standalone Prompt", author="Standalone", details="Details"
    )
    assert p_id is not None
    assert "added" in msg or "updated" in msg  # First time it's added

    p_id2, _, msg2 = standalone_add_or_update_prompt(
        memory_db, name="Standalone Prompt", author="Standalone Updated", details="New Details"
    )
    assert p_id2 == p_id
    assert "updated" in msg2
    updated_prompt = memory_db.get_prompt_by_name("Standalone Prompt")
    assert updated_prompt["author"] == "Standalone Updated"


def test_standalone_load_prompt_details_for_ui(memory_db: PromptsDatabase):
    standalone_add_or_update_prompt(
        memory_db,
        name="UI Prompt",
        author="UI Author",
        details="UI Details",
        system_prompt="Sys UI",
        user_prompt="User UI",
        keywords=["ui_kw1", "ui_kw2"],
    )
    name, author, details, system, user, kws_str = standalone_load_prompt_details_for_ui(memory_db, "UI Prompt")
    assert name == "UI Prompt"
    assert author == "UI Author"
    assert details == "UI Details"
    assert system == "Sys UI"
    assert user == "User UI"
    assert "ui_kw1" in kws_str and "ui_kw2" in kws_str

    # Test non-existent prompt
    name_nf, _, _, _, _, _ = standalone_load_prompt_details_for_ui(memory_db, "NonExistentPrompt")
    assert name_nf == ""


def test_standalone_export_functions(memory_db: PromptsDatabase, tmp_path: Path):
    memory_db.add_prompt("Export Prompt 1", "Export Author", "Details1", keywords=["export_kw", "common_kw"])
    memory_db.add_prompt("Export Prompt 2", "Export Author", "Details2", keywords=["another_kw", "common_kw"])

    # Test export_prompts_formatted (CSV)
    status_csv, path_csv_str = standalone_export_prompts_formatted(memory_db, export_format="csv")
    path_csv = Path(path_csv_str)
    assert "Successfully exported" in status_csv
    assert path_csv.exists()
    assert path_csv.suffix == ".csv"
    with open(path_csv) as f:
        content = f.read()
        assert "Export Prompt 1" in content
        assert "common_kw" in content  # Assuming keywords are exported
    os.remove(path_csv)

    # Test export_prompts_formatted (Markdown)
    status_md, path_md_zip_str = standalone_export_prompts_formatted(memory_db, export_format="markdown")
    path_md_zip = Path(path_md_zip_str)
    assert "Successfully exported" in status_md
    assert path_md_zip.exists()
    assert path_md_zip.suffix == ".zip"  # It creates a zip of markdown files
    # Further inspection of zip contents could be done here.
    os.remove(path_md_zip)

    # Test export_prompt_keywords_to_csv
    status_kw_csv, path_kw_csv_str = standalone_export_prompt_keywords_to_csv(memory_db)
    path_kw_csv = Path(path_kw_csv_str)
    assert "Successfully exported" in status_kw_csv
    assert path_kw_csv.exists()
    assert path_kw_csv.suffix == ".csv"
    with open(path_kw_csv) as f:
        content = f.read()
        assert "export_kw" in content
        assert "common_kw" in content
    os.remove(path_kw_csv)

    # Test view_prompt_keywords_markdown
    md_output = standalone_view_prompt_keywords_markdown(memory_db)
    assert "Current Active Prompt Keywords" in md_output
    assert "export_kw" in md_output
    assert "common_kw (2 active prompts)" in md_output  # Check count


def test_get_next_version_logic(memory_db: PromptsDatabase):
    # This is an internal helper, but its logic is critical
    p_id, _, _ = memory_db.add_prompt(name="Version Helper Test", author="Test", details="...")
    prompt_data = memory_db.get_prompt_by_id(p_id)
    assert prompt_data["version"] == 1  # Initial version

    conn = memory_db.get_connection()
    version_info = memory_db._get_next_version(conn, "Prompts", "id", p_id)
    assert version_info is not None
    current_v, next_v = version_info
    assert current_v == 1
    assert next_v == 2

    # Simulate an update
    memory_db.add_prompt(name="Version Helper Test", author="Test", details="Updated", overwrite=True)
    version_info_after_update = memory_db._get_next_version(conn, "Prompts", "id", p_id)
    assert version_info_after_update is not None
    current_v_up, next_v_up = version_info_after_update
    assert current_v_up == 2
    assert next_v_up == 3

    # Test for non-existent record
    assert memory_db._get_next_version(conn, "Prompts", "id", 99999) is None
