import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.context import ChaChaContext
from tldw_Server_API.app.core.DB_Management.chacha.migrations import sqlite_steps
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def _downgrade_schema_version_to_v38(db_path: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE db_schema_version
               SET version = 38
             WHERE schema_name = 'rag_char_chat_schema'
            """
        )
        conn.commit()


def _missing_sqlite_migration_steps() -> list[int]:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    steps = db._sqlite_linear_migration_steps()
    return [
        version
        for version in range(4, CharactersRAGDB._CURRENT_SCHEMA_VERSION)
        if version not in steps
    ]


def test_sqlite_linear_migration_registry_covers_current_schema() -> None:
    assert _missing_sqlite_migration_steps() == []  # nosec B101


def test_sqlite_linear_migration_registry_covers_v50_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _migrate_from_v49_to_v50(_ctx: object, _conn: sqlite3.Connection) -> None:
        return None

    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 50)
    monkeypatch.setattr(sqlite_steps, "_migrate_from_v49_to_v50", _migrate_from_v49_to_v50)

    assert _missing_sqlite_migration_steps() == []  # nosec B101
    steps = CharactersRAGDB.__new__(CharactersRAGDB)._sqlite_linear_migration_steps()
    assert steps[49].__wrapped__ is _migrate_from_v49_to_v50  # nosec B101  # the registry resolved the patched module attribute


def test_sqlite_legacy_reopen_uses_registry_for_v50(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha_v38_to_v50.db"

    db = CharactersRAGDB(db_path=str(db_path), client_id="migration-v50-bootstrap")
    db.close_connection()

    _downgrade_schema_version_to_v38(str(db_path))

    reached: list[str] = []

    def _migrate_from_v49_to_v50(ctx: ChaChaContext, conn: sqlite3.Connection) -> None:
        reached.append(ctx.schema_name)
        conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
            (50, ctx.schema_name),
        )

    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 50)
    monkeypatch.setattr(sqlite_steps, "_migrate_from_v49_to_v50", _migrate_from_v49_to_v50)

    migrated_db = CharactersRAGDB(db_path=str(db_path), client_id="migration-v50-reopen")
    migrated_db.close_connection()
    assert reached == ["rag_char_chat_schema"]  # nosec B101  # the reopen ran the patched registry step

    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            ("rag_char_chat_schema",),
        ).fetchone()[0]
        assert version == 50  # nosec B101


def test_sqlite_migration_v38_to_v39_historical_step_is_additive(tmp_path) -> None:
    db_path = tmp_path / "chacha_v38.db"

    db = CharactersRAGDB(db_path=str(db_path), client_id="migration-v39-bootstrap")
    db.close_connection()

    _downgrade_schema_version_to_v38(str(db_path))

    historical_step = CharactersRAGDB.__new__(CharactersRAGDB)
    historical_step.db_path_str = str(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sqlite_steps._migrate_from_v38_to_v39(historical_step.ctx, conn)
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            ("rag_char_chat_schema",),
        ).fetchone()[0]
        assert version == 39  # nosec B101

        workspace_cols = {
            row[1] for row in conn.execute("PRAGMA table_info('workspaces')").fetchall()
        }
        assert "study_materials_policy" in workspace_cols  # nosec B101

        quiz_cols = {
            row[1] for row in conn.execute("PRAGMA table_info('quizzes')").fetchall()
        }
        assert "workspace_id" in quiz_cols  # nosec B101

        deck_cols = {
            row[1] for row in conn.execute("PRAGMA table_info('decks')").fetchall()
        }
        assert "workspace_id" in deck_cols  # nosec B101
