from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


def _create_persona_conversation(
    db: CharactersRAGDB,
    conversation_id: str,
    *,
    title: str,
    topic_label: str | None = None,
) -> str:
    payload = {
        "id": conversation_id,
        "root_id": conversation_id,
        "assistant_kind": "persona",
        "assistant_id": f"persona-{conversation_id}",
        "persona_memory_mode": "read_only",
        "title": title,
        "topic_label": topic_label,
        "client_id": db.client_id,
    }
    return db.add_conversation(payload)


def _set_conversation_times(
    db: CharactersRAGDB,
    conversation_id: str,
    *,
    created_at: datetime,
    last_modified: datetime,
) -> None:
    db.execute_query(
        "UPDATE conversations SET created_at = ?, last_modified = ? WHERE id = ?",
        (
            created_at.isoformat(),
            last_modified.isoformat(),
            conversation_id,
        ),
    )


def test_search_conversations_page_keeps_global_bm25_norm(tmp_path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "conversation_search_page.sqlite"), client_id="user-1")

    _create_persona_conversation(db, "c1", title="alpha alpha alpha")
    _create_persona_conversation(db, "c2", title="alpha beta")

    page0, total0, max_bm250 = db.search_conversations_page(
        "alpha",
        client_id="user-1",
        order_by="bm25",
        limit=1,
        offset=0,
    )
    page1, total1, max_bm251 = db.search_conversations_page(
        "alpha",
        client_id="user-1",
        order_by="bm25",
        limit=1,
        offset=1,
    )

    assert total0 == 2
    assert total1 == 2
    assert max_bm250 > 0
    assert max_bm250 == pytest.approx(max_bm251, rel=1e-6)
    assert page0[0]["bm25_norm"] == pytest.approx(1.0, rel=1e-6)
    assert page1[0]["bm25_norm"] < 1.0


def test_search_conversations_page_bm25_without_query_falls_back_to_recency(tmp_path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "conversation_search_page.sqlite"), client_id="user-1")

    older = _create_persona_conversation(db, "older", title="Older conversation")
    newer = _create_persona_conversation(db, "newer", title="Newer conversation")

    now = datetime.now(timezone.utc)
    _set_conversation_times(
        db,
        older,
        created_at=now - timedelta(days=4),
        last_modified=now - timedelta(days=4),
    )
    _set_conversation_times(
        db,
        newer,
        created_at=now - timedelta(hours=1),
        last_modified=now - timedelta(hours=1),
    )

    rows, total, max_bm25 = db.search_conversations_page(
        None,
        client_id="user-1",
        order_by="bm25",
        limit=10,
        offset=0,
        as_of=now,
    )

    assert total == 2
    assert max_bm25 == 0.0
    assert [row["id"] for row in rows] == ["newer", "older"]


def test_search_conversations_page_topic_sorts_blank_labels_last(tmp_path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "conversation_search_page.sqlite"), client_id="user-1")

    alpha = _create_persona_conversation(db, "alpha-topic", title="Alpha title", topic_label="Alpha")
    beta = _create_persona_conversation(db, "beta-topic", title="Beta title", topic_label="Beta")
    blank = _create_persona_conversation(db, "blank-topic", title="Blank title", topic_label="   ")
    missing = _create_persona_conversation(db, "missing-topic", title="Missing title", topic_label=None)

    now = datetime.now(timezone.utc)
    for conversation_id in (alpha, beta, blank, missing):
        _set_conversation_times(
            db,
            conversation_id,
            created_at=now,
            last_modified=now,
        )

    rows, total, _ = db.search_conversations_page(
        None,
        client_id="user-1",
        order_by="topic",
        limit=10,
        offset=0,
        as_of=now,
    )

    assert total == 4
    assert [row["id"] for row in rows[:2]] == ["alpha-topic", "beta-topic"]
    assert {row["id"] for row in rows[2:]} == {"blank-topic", "missing-topic"}


def test_search_conversations_page_returns_total_separately_from_page_rows(tmp_path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "conversation_search_page.sqlite"), client_id="user-1")

    now = datetime.now(timezone.utc)
    for index in range(3):
        conversation_id = _create_persona_conversation(
            db,
            f"conv-{index}",
            title=f"Quota review {index}",
        )
        _set_conversation_times(
            db,
            conversation_id,
            created_at=now - timedelta(hours=index),
            last_modified=now - timedelta(hours=index),
        )

    rows, total, _ = db.search_conversations_page(
        "Quota",
        client_id="user-1",
        order_by="recency",
        limit=1,
        offset=1,
        as_of=now,
    )

    assert total == 3
    assert len(rows) == 1
    assert rows[0]["id"] == "conv-1"


def test_search_conversations_page_deleted_filters(tmp_path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "conversation_search_page.sqlite"), client_id="user-1")

    live_id = _create_persona_conversation(db, "live-conv", title="Quota cleanup live")
    deleted_id = _create_persona_conversation(db, "deleted-conv", title="Quota cleanup deleted")

    deleted_row = db.get_conversation_by_id(deleted_id)
    assert deleted_row is not None
    assert db.soft_delete_conversation(deleted_id, expected_version=deleted_row["version"]) is True

    live_rows, live_total, _ = db.search_conversations_page(
        "Quota",
        client_id="user-1",
        order_by="recency",
        limit=10,
        offset=0,
    )
    include_deleted_rows, include_deleted_total, _ = db.search_conversations_page(
        "Quota",
        client_id="user-1",
        order_by="recency",
        limit=10,
        offset=0,
        include_deleted=True,
    )
    deleted_only_rows, deleted_only_total, _ = db.search_conversations_page(
        "Quota",
        client_id="user-1",
        order_by="recency",
        limit=10,
        offset=0,
        deleted_only=True,
    )

    assert [row["id"] for row in live_rows] == [live_id]
    assert live_total == 1
    assert {row["id"] for row in include_deleted_rows} == {live_id, deleted_id}
    assert include_deleted_total == 2
    assert [row["id"] for row in deleted_only_rows] == [deleted_id]
    assert deleted_only_total == 1


def test_search_conversations_page_respects_workspace_scope(tmp_path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "conversation_search_scope.sqlite"), client_id="user-1")
    db.workspace_store.upsert_workspace("ws-a", "Workspace A")
    db.workspace_store.upsert_workspace("ws-b", "Workspace B")

    global_id = _create_persona_conversation(db, "global-conv", title="Quota global")
    ws_a_id = db.add_conversation(
        {
            "id": "workspace-a-conv",
            "root_id": "workspace-a-conv",
            "assistant_kind": "persona",
            "assistant_id": "persona-workspace-a-conv",
            "persona_memory_mode": "read_only",
            "title": "Quota workspace A",
            "client_id": db.client_id,
            "scope_type": "workspace",
            "workspace_id": "ws-a",
        }
    )
    _ = db.add_conversation(
        {
            "id": "workspace-b-conv",
            "root_id": "workspace-b-conv",
            "assistant_kind": "persona",
            "assistant_id": "persona-workspace-b-conv",
            "persona_memory_mode": "read_only",
            "title": "Quota workspace B",
            "client_id": db.client_id,
            "scope_type": "workspace",
            "workspace_id": "ws-b",
        }
    )

    global_rows, global_total, _ = db.search_conversations_page(
        "Quota",
        client_id="user-1",
        order_by="recency",
        scope_type="global",
        limit=10,
        offset=0,
    )
    workspace_rows, workspace_total, _ = db.search_conversations_page(
        "Quota",
        client_id="user-1",
        order_by="recency",
        scope_type="workspace",
        workspace_id="ws-a",
        limit=10,
        offset=0,
    )

    assert global_total == 1
    assert [row["id"] for row in global_rows] == [global_id]
    assert workspace_total == 1
    assert [row["id"] for row in workspace_rows] == [ws_a_id]
