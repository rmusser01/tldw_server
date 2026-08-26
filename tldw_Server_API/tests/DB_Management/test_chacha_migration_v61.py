"""SQLite schema-v61 contracts for recipient-owned shared chat state."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit

RECIPIENT = "recipient-a"
OWNER = "historical-owner"
CONVERSATION_ID = "conversation-a"


def _initialize(path: Path, *, schema_version: int | None = None) -> None:
    original = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    if schema_version is not None:
        CharactersRAGDB._CURRENT_SCHEMA_VERSION = schema_version
    try:
        db = CharactersRAGDB(str(path), client_id=RECIPIENT)
        db.close_all_connections()
    finally:
        CharactersRAGDB._CURRENT_SCHEMA_VERSION = original


def _insert_conversation(
    conn: sqlite3.Connection,
    conversation_id: str,
    *,
    client_id: str = RECIPIENT,
) -> None:
    conn.execute(
        "INSERT INTO conversations(id, root_id, client_id) VALUES (?, ?, ?)",
        (conversation_id, conversation_id, client_id),
    )


def _insert_thread(
    conn: sqlite3.Connection,
    *,
    recipient_user_id: str = RECIPIENT,
    share_id: int = 1,
    conversation_id: str = CONVERSATION_ID,
    owner_user_id: str = OWNER,
) -> None:
    conn.execute(
        """
        INSERT INTO shared_workspace_chat_threads(
            recipient_user_id, share_id, conversation_id, owner_user_id, workspace_id
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (recipient_user_id, share_id, conversation_id, owner_user_id, "workspace-a"),
    )


def _insert_request(
    conn: sqlite3.Connection,
    *,
    recipient_user_id: str = RECIPIENT,
    share_id: int = 1,
    request_id: str = "request-a",
    conversation_id: str = CONVERSATION_ID,
    status: str = "in_progress",
    lease_epoch: int = 1,
    source_mode: str | None = "all",
    user_message_id: str | None = None,
    assistant_message_id: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO shared_workspace_chat_requests(
            recipient_user_id, share_id, request_id, request_fingerprint,
            conversation_id, status, lease_epoch, source_mode,
            user_message_id, assistant_message_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            recipient_user_id,
            share_id,
            request_id,
            f"fingerprint-{request_id}",
            conversation_id,
            status,
            lease_epoch,
            source_mode,
            user_message_id,
            assistant_message_id,
        ),
    )


def _schema_snapshot(conn: sqlite3.Connection) -> tuple[tuple[str, str], ...]:
    rows = conn.execute(
        """
        SELECT name, sql
          FROM sqlite_master
         WHERE name IN (
             'shared_workspace_chat_threads',
             'shared_workspace_chat_requests',
             'idx_shared_workspace_chat_threads_conversation',
             'idx_shared_workspace_chat_requests_status_lease',
             'idx_shared_workspace_chat_requests_status_updated',
             'idx_shared_workspace_chat_requests_share_updated'
         )
         ORDER BY name
        """
    ).fetchall()
    return tuple((str(row[0]), " ".join(str(row[1]).split())) for row in rows)


def test_sqlite_v61_fresh_schema_has_composite_keys_foreign_keys_and_indexes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chacha-v61-fresh.sqlite"
    _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
        thread_columns = {
            str(row[1]): (str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in conn.execute("PRAGMA table_info(shared_workspace_chat_threads)")
        }
        request_columns = {
            str(row[1]): (str(row[2]).upper(), int(row[3]), int(row[5]))
            for row in conn.execute("PRAGMA table_info(shared_workspace_chat_requests)")
        }
        thread_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(shared_workspace_chat_threads)")
        }
        request_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(shared_workspace_chat_requests)")
        }
        request_fks = conn.execute(
            "PRAGMA foreign_key_list(shared_workspace_chat_requests)"
        ).fetchall()

    assert version == 63
    assert thread_columns["recipient_user_id"] == ("TEXT", 1, 1)
    assert thread_columns["share_id"] == ("INTEGER", 1, 2)
    assert thread_columns["owner_user_id"][:2] == ("TEXT", 1)
    assert request_columns["recipient_user_id"] == ("TEXT", 1, 1)
    assert request_columns["share_id"] == ("INTEGER", 1, 2)
    assert request_columns["request_id"] == ("TEXT", 1, 3)
    assert "idx_shared_workspace_chat_threads_conversation" in thread_indexes
    assert {
        "idx_shared_workspace_chat_requests_status_lease",
        "idx_shared_workspace_chat_requests_status_updated",
        "idx_shared_workspace_chat_requests_share_updated",
    } <= request_indexes
    composite_fk = [row for row in request_fks if row[2] == "shared_workspace_chat_threads"]
    assert [(row[3], row[4]) for row in sorted(composite_fk, key=lambda row: row[1])] == [
        ("recipient_user_id", "recipient_user_id"),
        ("share_id", "share_id"),
        ("conversation_id", "conversation_id"),
    ]
    assert {str(row[6]).upper() for row in composite_fk} == {"CASCADE"}
    assert {
        (str(row[2]), str(row[3]), str(row[6]).upper())
        for row in request_fks
        if row[2] == "messages"
    } == {
        ("messages", "assistant_message_id", "SET NULL"),
        ("messages", "user_message_id", "SET NULL"),
    }


@pytest.mark.parametrize(
    ("recipient_user_id", "share_id", "owner_user_id"),
    [
        ("", 1, OWNER),
        ("   ", 1, OWNER),
        (RECIPIENT, 0, OWNER),
        (RECIPIENT, -1, OWNER),
        (RECIPIENT, 1, ""),
        (RECIPIENT, 1, "   "),
    ],
)
def test_sqlite_v61_rejects_invalid_thread_tenant_and_share_keys(
    tmp_path: Path,
    recipient_user_id: str,
    share_id: int,
    owner_user_id: str,
) -> None:
    db_path = tmp_path / f"invalid-thread-{share_id}.sqlite"
    _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        _insert_conversation(conn, CONVERSATION_ID)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_thread(
                conn,
                recipient_user_id=recipient_user_id,
                share_id=share_id,
                owner_user_id=owner_user_id,
            )


@pytest.mark.parametrize(
    ("status", "lease_epoch", "source_mode"),
    [
        ("unknown", 1, "all"),
        ("in_progress", 0, "all"),
        ("in_progress", 1, "exclude"),
    ],
)
def test_sqlite_v61_rejects_invalid_request_state(
    tmp_path: Path,
    status: str,
    lease_epoch: int,
    source_mode: str,
) -> None:
    db_path = tmp_path / f"invalid-request-{status}-{lease_epoch}.sqlite"
    _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        _insert_conversation(conn, CONVERSATION_ID)
        _insert_thread(conn)
        with pytest.raises(sqlite3.IntegrityError):
            _insert_request(
                conn,
                status=status,
                lease_epoch=lease_epoch,
                source_mode=source_mode,
            )


def test_sqlite_v61_enforces_unique_conversation_composite_receipts_and_cascades(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chacha-v61-integrity.sqlite"
    _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        _insert_conversation(conn, CONVERSATION_ID)
        _insert_conversation(conn, "conversation-b", client_id="recipient-b")
        _insert_thread(conn)
        _insert_thread(
            conn,
            recipient_user_id="recipient-b",
            share_id=2,
            conversation_id="conversation-b",
        )

        with pytest.raises(sqlite3.IntegrityError):
            _insert_thread(
                conn,
                recipient_user_id="recipient-b",
                share_id=3,
                conversation_id=CONVERSATION_ID,
            )
        with pytest.raises(sqlite3.IntegrityError):
            _insert_request(
                conn,
                recipient_user_id=RECIPIENT,
                share_id=1,
                request_id="wrong-conversation",
                conversation_id="conversation-b",
            )

        conn.execute(
            """
            INSERT INTO messages(id, conversation_id, sender, content, client_id)
            VALUES ('user-message', ?, 'user', 'Question', ?),
                   ('assistant-message', ?, 'assistant', 'Answer', ?)
            """,
            (CONVERSATION_ID, RECIPIENT, CONVERSATION_ID, RECIPIENT),
        )
        _insert_request(
            conn,
            user_message_id="user-message",
            assistant_message_id="assistant-message",
        )
        conn.execute("DELETE FROM messages WHERE id = 'assistant-message'")
        receipt = conn.execute(
            "SELECT user_message_id, assistant_message_id FROM shared_workspace_chat_requests"
        ).fetchone()
        assert receipt == ("user-message", None)

        conn.execute("DELETE FROM conversations WHERE id = ?", (CONVERSATION_ID,))
        assert conn.execute(
            "SELECT COUNT(*) FROM shared_workspace_chat_threads WHERE conversation_id = ?",
            (CONVERSATION_ID,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM shared_workspace_chat_requests WHERE conversation_id = ?",
            (CONVERSATION_ID,),
        ).fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_sqlite_v60_upgrade_matches_fresh_schema_and_initializer_is_rerunnable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_path = tmp_path / "chacha-v61-fresh.sqlite"
    upgrade_path = tmp_path / "chacha-v60-upgrade.sqlite"
    _initialize(fresh_path)
    _initialize(upgrade_path, schema_version=60)
    _initialize(upgrade_path)
    _initialize(upgrade_path)

    with sqlite3.connect(fresh_path) as fresh, sqlite3.connect(upgrade_path) as upgraded:
        assert _schema_snapshot(fresh) == _schema_snapshot(upgraded)
        assert upgraded.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 63
        assert upgraded.execute("PRAGMA foreign_key_check").fetchall() == []
