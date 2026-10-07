"""The retired Sync v2 placeholder persona is cleared from stored chats (#3182).

Until #3182 the Sync conversation materializer stored a chat that named no
assistant as ``assistant_kind = persona``, ``assistant_id = sync-v2``. No such
persona exists, so every request that checks the chat's persona answered 404
``persona_not_found``. The materializer no longer writes it; this migration
(SQLite v75 -> v76, PostgreSQL v79 -> v80) repairs the rows that already hold it.

Only the exact placeholder is touched: a chat whose owner really has a persona
with that id keeps its binding, and so does every other identity.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat.assistant_startup import AssistantStartup, encode_assistant_startup
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.Persona.conversation_admission import PersonaAdmissionError, require_current_persona
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory

pytestmark = pytest.mark.integration

PREVIOUS_SQLITE_VERSION = 75
PREVIOUS_POSTGRES_VERSION = 79
OWNER = "user-1"
IDENTITY = ("assistant_kind", "assistant_id", "character_id", "persona_memory_mode", "assistant_startup_json")
PLAIN = (None, None, None, None, None)


def _previous_version(monkeypatch: pytest.MonkeyPatch, open_db: Callable[[], CharactersRAGDB]) -> CharactersRAGDB:
    """Open storage the way the release before this migration left it."""
    with monkeypatch.context() as old_code:
        old_code.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", PREVIOUS_SQLITE_VERSION)
        old_code.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", PREVIOUS_POSTGRES_VERSION)
        return open_db()


def _store_chat(db: CharactersRAGDB, chat_id: str, **columns: Any) -> None:
    """Insert a chat row exactly as given, the way an older server left it."""
    values = {"id": chat_id, "root_id": chat_id, "title": f"{chat_id} title", "client_id": str(db.client_id), **columns}
    names = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    with db.transaction() as conn:
        conn.execute(f"INSERT INTO conversations ({names}) VALUES ({marks})", tuple(values.values()))  # nosec B608


def _placeholder(db: CharactersRAGDB, chat_id: str, **columns: Any) -> None:
    _store_chat(db, chat_id, assistant_kind="persona", assistant_id="sync-v2", **columns)


def _row(db: CharactersRAGDB, chat_id: str) -> dict[str, Any]:
    row = db.get_conversation_by_id(chat_id, include_deleted=True)
    assert row is not None, chat_id
    return row


def _identity(db: CharactersRAGDB, chat_id: str) -> tuple[Any, ...]:
    row = _row(db, chat_id)
    return tuple(row[column] for column in IDENTITY)


def _schema_version(db: CharactersRAGDB) -> int:
    return db.execute_query(
        "SELECT version FROM db_schema_version WHERE schema_name = ?", (db._SCHEMA_NAME,), read_only=True
    ).fetchone()["version"]


def _expected_version(db: CharactersRAGDB) -> int:
    return CharactersRAGDB._POSTGRES_SCHEMA_VERSION if db.backend_type.value == "postgresql" else (
        CharactersRAGDB._CURRENT_SCHEMA_VERSION
    )


def test_this_migration_is_the_version_after_the_create_fingerprint() -> None:
    database = object.__new__(CharactersRAGDB)
    step = database._sqlite_linear_migration_steps()[PREVIOUS_SQLITE_VERSION]
    assert step.__func__ is CharactersRAGDB._migrate_from_v75_to_v76
    assert callable(CharactersRAGDB._migrate_from_v79_to_v80_postgres)
    assert CharactersRAGDB._CURRENT_SCHEMA_VERSION >= PREVIOUS_SQLITE_VERSION + 1
    assert CharactersRAGDB._POSTGRES_SCHEMA_VERSION >= PREVIOUS_POSTGRES_VERSION + 1


def test_upgrade_clears_the_placeholder_and_leaves_every_other_chat_alone(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _previous_version(monkeypatch, db_factory)
    character_id = old.add_character_card({"name": "Guide"})
    old.create_persona_profile({"id": "persona-real", "user_id": OWNER, "name": "Real"})
    # A persona that is only *named* like the placeholder has its own id.
    old.create_persona_profile({"id": "persona-lookalike", "user_id": OWNER, "name": "sync-v2"})
    startup = encode_assistant_startup(AssistantStartup(source="explicit"))

    _placeholder(old, "placeholder", version=4)
    _placeholder(old, "placeholder-in-trash", deleted=True, version=2)
    _placeholder(old, "placeholder-with-leftovers", persona_memory_mode="read_only", assistant_startup_json=startup)
    _store_chat(old, "plain")
    _store_chat(old, "real-persona", assistant_kind="persona", assistant_id="persona-real", persona_memory_mode="read_write",
                assistant_startup_json=startup)
    _store_chat(old, "lookalike-name", assistant_kind="persona", assistant_id="persona-lookalike")
    _store_chat(old, "longer-id", assistant_kind="persona", assistant_id="sync-v2-custom")
    _store_chat(old, "other-case", assistant_kind="persona", assistant_id="SYNC-V2")
    _store_chat(old, "character", assistant_kind="character", assistant_id=str(character_id), character_id=character_id)
    untouched = ("plain", "real-persona", "lookalike-name", "longer-id", "other-case", "character")
    before = {chat_id: _row(old, chat_id) for chat_id in ("placeholder", "placeholder-in-trash", *untouched)}
    assert _schema_version(old) == (
        PREVIOUS_POSTGRES_VERSION if old.backend_type.value == "postgresql" else PREVIOUS_SQLITE_VERSION
    )
    old.close_all_connections()

    upgraded = db_factory()

    assert _schema_version(upgraded) == _expected_version(upgraded)
    for chat_id in ("placeholder", "placeholder-in-trash", "placeholder-with-leftovers"):
        assert _identity(upgraded, chat_id) == PLAIN, chat_id
    # Nothing else about a repaired chat moves: it was not edited, only corrected.
    for chat_id in ("placeholder", "placeholder-in-trash"):
        after = _row(upgraded, chat_id)
        for column in ("title", "version", "last_modified", "created_at", "deleted", "client_id", "state", "root_id"):
            assert after[column] == before[chat_id][column], (chat_id, column)
    for chat_id in untouched:
        assert _row(upgraded, chat_id) == before[chat_id], chat_id
    # The repaired chat passes the persona check that answered 404 for it.
    assert require_current_persona(upgraded, owner_id=OWNER, conversation=_row(upgraded, "placeholder")) is None
    for chat_id in ("longer-id", "other-case"):
        with pytest.raises(PersonaAdmissionError):
            require_current_persona(upgraded, owner_id=OWNER, conversation=_row(upgraded, chat_id))


@pytest.mark.parametrize("profile_deleted", [False, True], ids=["live-persona", "deleted-persona"])
def test_upgrade_keeps_a_chat_bound_to_a_real_persona_with_the_placeholder_id(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch, profile_deleted: bool
) -> None:
    """If the owner has, or had, a persona with exactly this id, the binding is theirs and stays."""
    old = _previous_version(monkeypatch, db_factory)
    old.create_persona_profile({"id": "sync-v2", "user_id": OWNER, "name": "Really mine", "deleted": profile_deleted})
    _placeholder(old, "bound", persona_memory_mode="read_only")
    before = _row(old, "bound")
    old.close_all_connections()

    upgraded = db_factory()

    assert _schema_version(upgraded) == _expected_version(upgraded)
    assert _row(upgraded, "bound") == before


def test_upgrade_is_idempotent(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reopening changes nothing, and neither does running the migration again over repaired storage."""
    old = _previous_version(monkeypatch, db_factory)
    postgres = old.backend_type.value == "postgresql"
    _placeholder(old, "placeholder", version=3)
    _store_chat(old, "plain")
    old.close_all_connections()
    upgraded = db_factory()
    repaired = {chat_id: _row(upgraded, chat_id) for chat_id in ("placeholder", "plain")}
    assert _identity(upgraded, "placeholder") == PLAIN
    upgraded.close_all_connections()

    reopened = db_factory()
    assert _schema_version(reopened) == _expected_version(reopened)
    assert {chat_id: _row(reopened, chat_id) for chat_id in repaired} == repaired

    # Put the version back, as if the bump had been lost, so the migration itself runs a second time.
    with reopened.transaction() as conn:
        conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
            (PREVIOUS_POSTGRES_VERSION if postgres else PREVIOUS_SQLITE_VERSION, reopened._SCHEMA_NAME),
        )
    reopened.close_all_connections()

    rerun = db_factory()
    assert _schema_version(rerun) == _expected_version(rerun)
    assert {chat_id: _row(rerun, chat_id) for chat_id in repaired} == repaired


def test_interrupted_upgrade_rolls_back_the_repair_and_the_version(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _previous_version(monkeypatch, db_factory)
    _placeholder(old, "placeholder")
    before = _row(old, "placeholder")
    postgres = old.backend_type.value == "postgresql"
    old.close_all_connections()
    method = "_migrate_from_v79_to_v80_postgres" if postgres else "_migrate_from_v75_to_v76"
    assert hasattr(CharactersRAGDB, method), "the placeholder repair must be a registered migration"
    migrate = getattr(CharactersRAGDB, method)

    def fail_after_repair(self: CharactersRAGDB, conn: Any) -> None:
        migrate(self, conn)
        raise CharactersRAGDBError("injected placeholder migration failure")

    with monkeypatch.context() as failing:
        failing.setattr(CharactersRAGDB, method, fail_after_repair)
        with pytest.raises(CharactersRAGDBError, match="injected placeholder migration failure"):
            db_factory()

    reopened_old = _previous_version(monkeypatch, db_factory)
    assert _schema_version(reopened_old) == (PREVIOUS_POSTGRES_VERSION if postgres else PREVIOUS_SQLITE_VERSION)
    assert _row(reopened_old, "placeholder") == before


# ---------------------------------------------------------------------------
# PostgreSQL: one shared database, row-level security between owners
# ---------------------------------------------------------------------------


@pytest.fixture(params=["superuser", "restricted_owner"])
def shared_postgres(request: pytest.FixtureRequest) -> Iterator[Callable[[str], CharactersRAGDB]]:
    """Open handles for different owners on one PostgreSQL database.

    ``superuser`` is the documented deployment: the role bypasses row-level
    security. ``restricted_owner`` owns the tables without bypassing it, so the
    forced policies hide every other owner's rows from an ordinary statement.
    """
    opened: list[CharactersRAGDB] = []
    backends: list[Any] = []
    if request.param == "superuser":
        config: DatabaseConfig = request.getfixturevalue("pg_database_config")

        def backend_for_handle() -> Any:
            backend = DatabaseBackendFactory.create_backend(config)
            backends.append(backend)
            return backend
    else:
        restricted = request.getfixturevalue("pg_restricted_backend")

        def backend_for_handle() -> Any:
            return restricted

    def open_for(owner: str) -> CharactersRAGDB:
        database = CharactersRAGDB(":memory:", client_id=owner, backend=backend_for_handle())
        opened.append(database)
        return database

    try:
        yield open_for
    finally:
        # Handles may share one pool, so each only gives its connection back.
        for database in opened:
            database.close_connection()
        for backend in backends:
            backend.get_pool().close_all()


def test_postgres_upgrade_repairs_every_owner_and_respects_another_owners_persona(
    shared_postgres: Callable[[str], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One migration run repairs all owners' rows, and a persona is only its own owner's."""
    first = _previous_version(monkeypatch, lambda: shared_postgres("user-1"))
    second = _previous_version(monkeypatch, lambda: shared_postgres("user-2"))
    # user-2 really has a persona with the placeholder's id; user-1 does not.
    second.create_persona_profile({"id": "sync-v2", "user_id": "user-2", "name": "Really mine"})
    _placeholder(first, "first-placeholder")
    _placeholder(second, "second-real-binding", persona_memory_mode="read_only")
    _store_chat(second, "second-plain")
    second_before = _row(second, "second-real-binding")
    first.close_connection()
    second.close_connection()

    upgraded_first = shared_postgres("user-1")

    assert _schema_version(upgraded_first) == CharactersRAGDB._POSTGRES_SCHEMA_VERSION
    assert _identity(upgraded_first, "first-placeholder") == PLAIN
    upgraded_second = shared_postgres("user-2")
    assert _row(upgraded_second, "second-real-binding") == second_before


def test_postgres_upgrade_repairs_rows_hidden_by_row_level_security_and_restores_it(
    pg_restricted_backend: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run that upgrades the schema sees one owner; the other owner's placeholder must still be repaired."""
    opened: list[CharactersRAGDB] = []

    def open_for(owner: str) -> CharactersRAGDB:
        database = CharactersRAGDB(":memory:", client_id=owner, backend=pg_restricted_backend)
        opened.append(database)
        return database

    def forced() -> tuple[bool, bool]:
        with pg_restricted_backend.transaction() as conn:
            row = conn.execute(
                "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                "WHERE relname = 'conversations' AND relnamespace = current_schema()::regnamespace"
            ).fetchone()
        return bool(row["relrowsecurity"]), bool(row["relforcerowsecurity"])

    try:
        first = _previous_version(monkeypatch, lambda: open_for("user-1"))
        second = _previous_version(monkeypatch, lambda: open_for("user-2"))
        _placeholder(first, "first-placeholder")
        _placeholder(second, "second-placeholder")
        assert forced() == (True, True)
        # The policy is real: neither owner can read the other's chat.
        assert first.get_conversation_by_id("second-placeholder") is None
        first.close_connection()
        second.close_connection()

        upgraded_first = open_for("user-1")

        assert _schema_version(upgraded_first) == CharactersRAGDB._POSTGRES_SCHEMA_VERSION
        assert forced() == (True, True)
        assert _identity(upgraded_first, "first-placeholder") == PLAIN
        assert upgraded_first.get_conversation_by_id("second-placeholder") is None
        upgraded_second = open_for("user-2")
        assert _identity(upgraded_second, "second-placeholder") == PLAIN
    finally:
        # The fixture owns the shared pool; a handle only gives its connection back.
        for database in opened:
            database.close_connection()
