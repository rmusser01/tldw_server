"""Tests for Workspace cross-resource membership persistence."""

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendDatabaseError,
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(
        db_path=str(tmp_path / "workspace_memberships.sqlite"),
        client_id="test-client",
    )
    database.upsert_workspace("ws-1", "Workspace One")
    database.upsert_workspace("ws-2", "Workspace Two")
    database.upsert_workspace("ws-3", "Workspace Three")
    return database


def _set_membership_updated_at(
    db: CharactersRAGDB,
    workspace_id: str,
    resource_type: str,
    resource_id: str,
    updated_at: str,
) -> None:
    db.execute_query(
        "UPDATE workspace_resource_memberships "
        "SET updated_at = ? "
        "WHERE workspace_id = ? AND resource_type = ? AND resource_id = ?",
        (updated_at, workspace_id, resource_type, resource_id),
        commit=True,
    )


def test_add_workspace_resource_membership_creates_normalized_row(db):
    row = db.add_workspace_resource_membership(
        "ws-1",
        {
            "resource_type": " media ",
            "resource_id": " 42 ",
            "role": " source ",
            "label": "Paper",
            "transfer_policy": " link ",
            "provenance": {"source_surface": "library"},
            "metadata": {"priority": "high"},
        },
        user_id="user-1",
    )

    assert row["workspace_id"] == "ws-1"
    assert row["resource_type"] == "media"
    assert row["resource_id"] == "42"
    assert row["role"] == "source"
    assert row["transfer_policy"] == "link"
    assert row["label"] == "Paper"
    assert row["provenance"] == {"source_surface": "library"}
    assert row["metadata"] == {"priority": "high"}
    assert row["created_by_user_id"] == "user-1"
    assert row["updated_by_user_id"] == "user-1"
    assert row["client_id"] == "test-client"
    assert row["version"] == 1
    assert row["deleted"] in (False, 0)


def test_duplicate_workspace_resource_membership_same_request_returns_existing(db):
    first = db.add_workspace_resource_membership(
        "ws-1",
        {
            "resource_type": "media",
            "resource_id": "42",
            "role": "source",
            "label": "Paper",
            "transfer_policy": "link",
            "provenance": {"source_surface": "library"},
            "metadata": {"priority": "high"},
        },
        user_id="user-1",
    )

    duplicate = db.add_workspace_resource_membership(
        "ws-1",
        {
            "resource_type": " media ",
            "resource_id": " 42 ",
            "role": " source ",
            "label": "Paper",
            "transfer_policy": " link ",
            "provenance": {"source_surface": "library"},
            "metadata": {"priority": "high"},
        },
        user_id="user-2",
    )

    assert duplicate == first


@pytest.mark.parametrize(
    "updates",
    [
        {"role": "member"},
        {"transfer_policy": "copy"},
        {"label": "Other label"},
        {"provenance": {"source_surface": "backfill"}},
        {"metadata": {"priority": "low"}},
    ],
)
def test_duplicate_workspace_resource_membership_conflict_raises(db, updates):
    db.add_workspace_resource_membership(
        "ws-1",
        {
            "resource_type": "media",
            "resource_id": "42",
            "role": "source",
            "label": "Paper",
            "transfer_policy": "link",
            "provenance": {"source_surface": "library"},
            "metadata": {"priority": "high"},
        },
    )

    with pytest.raises(ConflictError) as exc_info:
        db.add_workspace_resource_membership(
            "ws-1",
            {
                "resource_type": "media",
                "resource_id": "42",
                "role": "source",
                "label": "Paper",
                "transfer_policy": "link",
                "provenance": {"source_surface": "library"},
                "metadata": {"priority": "high"},
                **updates,
            },
        )

    assert exc_info.value.entity == "workspace_resource_memberships"


def test_delete_workspace_resource_membership_soft_deletes_and_default_reads_hide(db):
    db.add_workspace_resource_membership(
        "ws-1",
        {"resource_type": "media", "resource_id": "42", "role": "source"},
        user_id="user-1",
    )

    deleted = db.delete_workspace_resource_membership(
        "ws-1",
        "media",
        "42",
        user_id="user-2",
    )

    assert deleted is not None
    assert deleted["deleted"] in (True, 1)
    assert deleted["updated_by_user_id"] == "user-2"
    assert deleted["version"] == 2
    assert db.get_workspace_resource_membership("ws-1", "media", "42") is None
    assert db.list_workspace_resource_memberships("ws-1") == []

    include_deleted = db.get_workspace_resource_membership(
        "ws-1",
        "media",
        "42",
        include_deleted=True,
    )
    assert include_deleted is not None
    assert include_deleted["deleted"] in (True, 1)
    assert db.delete_workspace_resource_membership("ws-1", "media", "42") is None


def test_add_workspace_resource_membership_restores_deleted_row(db):
    created = db.add_workspace_resource_membership(
        "ws-1",
        {
            "resource_type": "media",
            "resource_id": "42",
            "role": "source",
            "label": "Original",
            "provenance": {"source_surface": "library"},
        },
        user_id="user-1",
    )
    deleted = db.delete_workspace_resource_membership(
        "ws-1",
        "media",
        "42",
        user_id="user-2",
    )
    assert deleted is not None

    restored = db.add_workspace_resource_membership(
        "ws-1",
        {
            "resource_type": "media",
            "resource_id": "42",
            "role": "member",
            "label": "Restored",
            "transfer_policy": "link",
            "provenance": {"source_surface": "restore"},
            "metadata": {"reason": "manual"},
            "restore_deleted": True,
        },
        user_id="user-3",
    )

    assert restored["deleted"] in (False, 0)
    assert restored["role"] == "member"
    assert restored["label"] == "Restored"
    assert restored["provenance"] == {"source_surface": "restore"}
    assert restored["metadata"] == {"reason": "manual"}
    assert restored["created_at"] == created["created_at"]
    assert restored["created_by_user_id"] == "user-1"
    assert restored["updated_by_user_id"] == "user-3"
    assert restored["version"] == deleted["version"] + 1


def test_list_workspace_resource_memberships_order_is_deterministic(db):
    db.add_workspace_resource_membership(
        "ws-1",
        {"resource_type": "workspace_source", "resource_id": "src-b"},
    )
    db.add_workspace_resource_membership(
        "ws-1",
        {"resource_type": "media", "resource_id": "99"},
    )
    db.add_workspace_resource_membership(
        "ws-1",
        {"resource_type": "media", "resource_id": "42"},
    )
    db.add_workspace_resource_membership(
        "ws-1",
        {"resource_type": "chat", "resource_id": "conv-1"},
    )
    _set_membership_updated_at(
        db,
        "ws-1",
        "workspace_source",
        "src-b",
        "2026-06-07T12:00:00.000Z",
    )
    _set_membership_updated_at(
        db,
        "ws-1",
        "media",
        "99",
        "2026-06-07T12:00:00.000Z",
    )
    _set_membership_updated_at(
        db,
        "ws-1",
        "media",
        "42",
        "2026-06-07T12:00:00.000Z",
    )
    _set_membership_updated_at(
        db,
        "ws-1",
        "chat",
        "conv-1",
        "2026-06-07T11:00:00.000Z",
    )

    rows = db.list_workspace_resource_memberships("ws-1")

    assert [(row["resource_type"], row["resource_id"]) for row in rows] == [
        ("media", "42"),
        ("media", "99"),
        ("workspace_source", "src-b"),
        ("chat", "conv-1"),
    ]


def test_list_workspace_resource_memberships_returns_limit_plus_one_for_pagination(db):
    for resource_type, resource_id in (
        ("media", "42"),
        ("media", "99"),
        ("workspace_source", "src-b"),
    ):
        db.add_workspace_resource_membership(
            "ws-1",
            {"resource_type": resource_type, "resource_id": resource_id},
        )
        _set_membership_updated_at(
            db,
            "ws-1",
            resource_type,
            resource_id,
            "2026-06-07T12:00:00.000Z",
        )

    rows = db.list_workspace_resource_memberships("ws-1", limit=2)

    assert [(row["resource_type"], row["resource_id"]) for row in rows] == [
        ("media", "42"),
        ("media", "99"),
        ("workspace_source", "src-b"),
    ]


def test_list_resource_workspace_memberships_returns_all_active_workspaces(db):
    db.add_workspace_resource_membership(
        "ws-2",
        {"resource_type": "media", "resource_id": "42", "role": "source"},
    )
    db.add_workspace_resource_membership(
        "ws-1",
        {"resource_type": "media", "resource_id": "42", "role": "source"},
    )
    db.add_workspace_resource_membership(
        "ws-3",
        {"resource_type": "media", "resource_id": "42", "role": "source"},
    )
    db.delete_workspace_resource_membership("ws-3", "media", "42")
    _set_membership_updated_at(db, "ws-1", "media", "42", "2026-06-07T12:00:00.000Z")
    _set_membership_updated_at(db, "ws-2", "media", "42", "2026-06-07T12:00:00.000Z")

    rows = db.list_resource_workspace_memberships("media", "42")

    assert [row["workspace_id"] for row in rows] == ["ws-1", "ws-2"]


def test_list_resource_workspace_memberships_returns_limit_plus_one_for_pagination(db):
    for workspace_id in ("ws-2", "ws-1", "ws-3"):
        db.add_workspace_resource_membership(
            workspace_id,
            {"resource_type": "media", "resource_id": "42", "role": "source"},
        )
        _set_membership_updated_at(
            db,
            workspace_id,
            "media",
            "42",
            "2026-06-07T12:00:00.000Z",
        )

    rows = db.list_resource_workspace_memberships("media", "42", limit=2)

    assert [row["workspace_id"] for row in rows] == ["ws-1", "ws-2", "ws-3"]


def test_delete_workspace_preserves_membership_history(db):
    created = db.add_workspace_resource_membership(
        "ws-1",
        {"resource_type": "media", "resource_id": "42", "role": "source"},
    )
    workspace = db.get_workspace("ws-1")
    assert workspace is not None

    assert db.delete_workspace("ws-1", expected_version=workspace["version"]) is True

    assert db.get_workspace("ws-1") is None
    preserved = db.get_workspace_resource_membership(
        "ws-1",
        "media",
        "42",
        include_deleted=True,
    )
    assert preserved is not None
    assert preserved["deleted"] in (False, 0)
    assert preserved["created_at"] == created["created_at"]


def test_hard_delete_workspace_removes_membership_rows(db):
    db.add_workspace_resource_membership(
        "ws-1",
        {"resource_type": "media", "resource_id": "42", "role": "source"},
    )

    db.hard_delete_workspace("ws-1")

    assert (
        db.get_workspace_resource_membership(
            "ws-1",
            "media",
            "42",
            include_deleted=True,
        )
        is None
    )


class _BackendErrorTransaction:
    def __init__(self, error: BackendDatabaseError):
        self.error = error

    def __enter__(self):
        raise self.error

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_workspace_membership_backend_error_wraps_unrelated_errors(db, monkeypatch):
    monkeypatch.setattr(
        db,
        "transaction",
        lambda: _BackendErrorTransaction(BackendDatabaseError("pg down")),
    )

    with pytest.raises(CharactersRAGDBError, match="pg down"):
        db.add_workspace_resource_membership(
            "ws-1",
            {"resource_type": "media", "resource_id": "42"},
        )


def test_workspace_membership_backend_constraint_error_maps_to_conflict(db, monkeypatch):
    monkeypatch.setattr(
        db,
        "transaction",
        lambda: _BackendErrorTransaction(
            BackendDatabaseError("duplicate key value violates unique constraint")
        ),
    )

    with pytest.raises(ConflictError) as exc_info:
        db.add_workspace_resource_membership(
            "ws-1",
            {"resource_type": "media", "resource_id": "42"},
        )

    assert exc_info.value.entity == "workspace_resource_memberships"


class _RowcountCursor:
    def __init__(self, rowcount: int):
        self.rowcount = rowcount


class _RowcountZeroConnection:
    def __init__(self):
        self.statements: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, statement: str, params: tuple[object, ...]):
        self.statements.append((statement, params))
        return _RowcountCursor(0)


class _RowcountZeroTransaction:
    def __init__(self, conn: _RowcountZeroConnection):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, exc_type, exc, traceback):
        return False


def _membership_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "workspace_id": "ws-1",
        "resource_type": "media",
        "resource_id": "42",
        "role": "member",
        "label": "Restored",
        "transfer_policy": "link",
        "provenance": {"source_surface": "restore"},
        "provenance_json": '{"source_surface": "restore"}',
        "metadata": {"reason": "manual"},
        "metadata_json": '{"reason": "manual"}',
        "created_by_user_id": "user-1",
        "updated_by_user_id": "user-2",
        "created_at": "2026-06-07T12:00:00.000Z",
        "updated_at": "2026-06-07T12:01:00.000Z",
        "deleted": 0,
        "client_id": "test-client",
        "version": 3,
    }
    row.update(updates)
    return row


def test_restore_deleted_membership_race_returns_already_restored_row(db, monkeypatch):
    fake_conn = _RowcountZeroConnection()
    calls: list[bool] = []
    deleted_row = _membership_row(deleted=1, version=2)
    active_row = _membership_row(deleted=0, version=3)

    def fake_getter(conn, workspace_id, resource_type, resource_id, *, include_deleted):
        calls.append(include_deleted)
        if len(calls) == 1:
            return deleted_row
        return active_row if include_deleted else None

    monkeypatch.setattr(db, "transaction", lambda: _RowcountZeroTransaction(fake_conn))
    monkeypatch.setattr(db, "_get_workspace_resource_membership_with_conn", fake_getter)

    restored = db.add_workspace_resource_membership(
        "ws-1",
        {
            "resource_type": "media",
            "resource_id": "42",
            "role": "member",
            "label": "Restored",
            "transfer_policy": "link",
            "provenance": {"source_surface": "restore"},
            "metadata": {"reason": "manual"},
            "restore_deleted": True,
        },
        user_id="user-2",
    )

    assert restored == active_row
    assert calls == [True, True]
    assert "AND deleted = ?" in fake_conn.statements[0][0]


def test_delete_membership_race_returns_none_for_already_deleted_row(db, monkeypatch):
    fake_conn = _RowcountZeroConnection()
    calls: list[bool] = []
    active_row = _membership_row(deleted=0, version=2)
    deleted_row = _membership_row(deleted=1, version=3)

    def fake_getter(conn, workspace_id, resource_type, resource_id, *, include_deleted):
        calls.append(include_deleted)
        return active_row if len(calls) == 1 else deleted_row

    monkeypatch.setattr(db, "transaction", lambda: _RowcountZeroTransaction(fake_conn))
    monkeypatch.setattr(db, "_get_workspace_resource_membership_with_conn", fake_getter)

    deleted = db.delete_workspace_resource_membership(
        "ws-1",
        "media",
        "42",
        user_id="user-2",
    )

    assert deleted is None
    assert calls == [True, True]
    assert "AND deleted = ?" in fake_conn.statements[0][0]


def test_workspace_subresource_public_getters_return_existing_scoped_rows(db):
    source = db.add_workspace_source(
        "ws-1",
        {
            "id": "src-1",
            "media_id": 42,
            "title": "Research Source",
            "source_type": "video",
        },
    )
    note = db.add_workspace_note(
        "ws-1",
        {
            "title": "Workspace Note",
            "content": "Evidence notes",
        },
    )

    assert db.get_workspace_source("ws-1", "src-1") == source
    assert db.get_workspace_note("ws-1", note["id"]) == note


def test_get_conversation_for_workspace_membership_returns_workspace_conversation(db):
    db.add_character_card({"name": "Workspace Character"})
    conversation_id = db.add_conversation(
        {
            "character_id": 1,
            "title": "Workspace Chat",
            "scope_type": "workspace",
            "workspace_id": "ws-1",
        }
    )

    row = db.get_conversation_for_workspace_membership(conversation_id)

    assert row is not None
    assert row["id"] == conversation_id
    assert row["title"] == "Workspace Chat"
    assert row["scope_type"] == "workspace"
    assert row["workspace_id"] == "ws-1"
    assert row["last_modified"] is not None
    assert row["version"] == 1
