import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
)
from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_contract import (
    notes_moodboard_note_object_hash,
    notes_moodboard_object_hash,
    parse_notes_moodboard_note_v1,
    parse_notes_moodboard_v1,
)


@pytest.fixture()
def moodboard_db(tmp_path):
    db_path = tmp_path / "moodboard_management.db"
    db = CharactersRAGDB(str(db_path), client_id="moodboard_unit_test")
    yield db
    try:
        db.close()
    except Exception:
        _ = None


def test_moodboard_schema_tables_exist_after_init(moodboard_db: CharactersRAGDB):
    conn = moodboard_db.get_connection()
    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('moodboards', 'moodboard_notes')"
        ).fetchall()
    }
    assert {"moodboards", "moodboard_notes"}.issubset(tables)

    indexes = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
    }
    assert "idx_moodboard_notes_scope_board_page" in indexes
    assert "idx_moodboard_notes_scope_note" in indexes


def test_moodboard_crud_roundtrip(moodboard_db: CharactersRAGDB):
    moodboard_id = moodboard_db.add_moodboard(
        name="Visual Research",
        description="Inspiration board",
        smart_rule={"query": "camera", "keyword_tokens": ["design"]},
    )
    assert moodboard_id is not None

    created = moodboard_db.get_moodboard_by_id(moodboard_id)
    assert created is not None
    assert created["name"] == "Visual Research"
    assert created["smart_rule"]["query"] == "camera"
    assert created["version"] == 1
    assert created["owner_user_id"] == moodboard_db.client_id
    assert created["dataset_id"] == "local-unbound"
    assert created["canonical_revision"] == 1
    assert created["canonical_hash"].startswith("sha256:")
    assert created["canvas_json"] == {"layout_mode": "masonry", "metadata": {}}

    listing = moodboard_db.list_moodboards(limit=20, offset=0)
    assert any(int(item["id"]) == int(moodboard_id) for item in listing)

    updated = moodboard_db.update_moodboard(
        moodboard_id=moodboard_id,
        update_data={"name": "Visual Research Updated", "description": "Updated"},
        expected_version=int(created["version"]),
    )
    assert updated is True

    after_update = moodboard_db.get_moodboard_by_id(moodboard_id)
    assert after_update is not None
    assert after_update["name"] == "Visual Research Updated"
    assert int(after_update["version"]) == 2
    assert int(after_update["canonical_revision"]) == 2
    assert after_update["canonical_hash"] != created["canonical_hash"]

    deleted = moodboard_db.delete_moodboard(moodboard_id=moodboard_id, expected_version=int(after_update["version"]))
    assert deleted is True
    assert moodboard_db.get_moodboard_by_id(moodboard_id) is None

    deleted_view = moodboard_db.list_moodboards(limit=20, offset=0, include_deleted=True)
    deleted_row = next((row for row in deleted_view if int(row["id"]) == int(moodboard_id)), None)
    assert deleted_row is not None
    assert bool(deleted_row["deleted"]) is True
    assert int(deleted_row["version"]) == 3
    assert int(deleted_row["canonical_revision"]) == 3
    assert deleted_row["canonical_hash"] != after_update["canonical_hash"]


def test_moodboard_update_rebuilds_complete_canonical_state_and_diagnostics(
    moodboard_db: CharactersRAGDB,
) -> None:
    moodboard_id = moodboard_db.add_moodboard(
        name="Initially blocked",
        smart_rule={"unknown": "legacy-only"},
    )
    assert moodboard_id is not None
    before = moodboard_db.get_moodboard_by_id(moodboard_id)
    assert before is not None
    assert before["source_diagnostic_code"] is not None

    assert moodboard_db.update_moodboard(
        moodboard_id=moodboard_id,
        update_data={
            "name": "Canonical board",
            "description": "Complete current state",
            "smart_rule": {
                "query": "Research",
                "keyword_tokens": ["Notes"],
                "collection_sync_ids": [],
                "sources": ["manual"],
                "updated": {"after": None, "before": None},
            },
            "canvas": {
                "layout_mode": "freeform",
                "metadata": {"theme": "paper"},
            },
        },
        expected_version=1,
    )
    updated = moodboard_db.get_moodboard_by_id(moodboard_id)
    assert updated is not None
    payload = parse_notes_moodboard_v1(
        {
            "moodboard_id": updated["sync_id"],
            "name": updated["name"],
            "description": updated["description"],
            "smart_rule": updated["smart_rule"],
            "canvas": updated["canvas_json"],
        }
    )
    assert updated["source_diagnostic_code"] is None
    assert updated["source_diagnostic_hash"] is None
    assert updated["canonical_hash"] == notes_moodboard_object_hash(
        payload,
        revision=updated["canonical_revision"],
        deleted=False,
    )


def test_moodboard_pin_unpin_is_idempotent(moodboard_db: CharactersRAGDB):
    note_id = moodboard_db.add_note(title="Pinned note", content="content")
    moodboard_id = moodboard_db.add_moodboard(name="Pins")
    assert note_id
    assert moodboard_id is not None

    first_pin = moodboard_db.link_note_to_moodboard(moodboard_id=moodboard_id, note_id=note_id)
    second_pin = moodboard_db.link_note_to_moodboard(moodboard_id=moodboard_id, note_id=note_id)
    assert first_pin is True
    assert second_pin is False

    placement = moodboard_db.execute_query(
        "SELECT owner_user_id,dataset_id,deleted,canonical_revision,canonical_hash "
        "FROM moodboard_notes WHERE moodboard_id=? AND note_id=?",
        (moodboard_id, note_id),
    ).fetchone()
    assert tuple(placement[:4]) == (
        moodboard_db.client_id,
        "local-unbound",
        0,
        1,
    )
    assert placement["canonical_hash"].startswith("sha256:")

    rows = moodboard_db.list_moodboard_notes(moodboard_id=moodboard_id, limit=20, offset=0)
    assert len(rows) == 1
    assert rows[0]["id"] == note_id
    assert rows[0]["membership_source"] == "manual"

    first_unpin = moodboard_db.unlink_note_from_moodboard(moodboard_id=moodboard_id, note_id=note_id)
    second_unpin = moodboard_db.unlink_note_from_moodboard(moodboard_id=moodboard_id, note_id=note_id)
    assert first_unpin is True
    assert second_unpin is False
    tombstone = moodboard_db.execute_query(
        "SELECT deleted FROM moodboard_notes WHERE moodboard_id=? AND note_id=?",
        (moodboard_id, note_id),
    ).fetchone()
    assert tombstone["deleted"] == 1


def test_moodboard_repin_hash_keeps_the_persisted_placement_layout(
    moodboard_db: CharactersRAGDB,
) -> None:
    first_note_id = moodboard_db.add_note(title="First", content="content")
    second_note_id = moodboard_db.add_note(title="Second", content="content")
    moodboard_id = moodboard_db.add_moodboard(name="Ordered pins")
    assert first_note_id and second_note_id and moodboard_id is not None
    assert moodboard_db.link_note_to_moodboard(moodboard_id, first_note_id) is True
    assert moodboard_db.link_note_to_moodboard(moodboard_id, second_note_id) is True
    assert moodboard_db.unlink_note_from_moodboard(moodboard_id, second_note_id) is True
    assert moodboard_db.link_note_to_moodboard(moodboard_id, second_note_id) is True

    board = moodboard_db.get_moodboard_by_id(moodboard_id)
    placement = moodboard_db.execute_query(
        "SELECT * FROM moodboard_notes WHERE moodboard_id=? AND note_id=?",
        (moodboard_id, second_note_id),
    ).fetchone()
    assert board is not None and placement is not None
    payload = parse_notes_moodboard_note_v1(
        {
            "moodboard_id": board["sync_id"],
            "note_id": second_note_id,
            "x": placement["x"],
            "y": placement["y"],
            "width": placement["width"],
            "height": placement["height"],
            "order_index": placement["order_index"],
            "display": {},
        }
    )
    assert placement["order_index"] == 1
    assert placement["canonical_hash"] == notes_moodboard_note_object_hash(
        payload,
        revision=placement["canonical_revision"],
        deleted=False,
    )


def test_moodboard_manual_and_smart_union_sources(moodboard_db: CharactersRAGDB):
    note_manual = moodboard_db.add_note(title="Manual only", content="A")
    note_smart = moodboard_db.add_note(title="Smart only", content="B")
    note_both = moodboard_db.add_note(title="Both membership", content="C")
    assert note_manual and note_smart and note_both

    keyword_id = moodboard_db.add_keyword("palette")
    assert keyword_id is not None

    assert moodboard_db.link_note_to_keyword(note_smart, keyword_id) is True
    assert moodboard_db.link_note_to_keyword(note_both, keyword_id) is True

    moodboard_id = moodboard_db.add_moodboard(
        name="Hybrid",
        smart_rule={"keyword_tokens": ["palette"]},
    )
    assert moodboard_id is not None

    assert moodboard_db.link_note_to_moodboard(moodboard_id, note_manual) is True
    assert moodboard_db.link_note_to_moodboard(moodboard_id, note_both) is True

    notes = moodboard_db.list_moodboard_notes(moodboard_id=moodboard_id, limit=50, offset=0)
    by_id = {row["id"]: row for row in notes}

    assert set(by_id.keys()) == {note_manual, note_smart, note_both}
    assert by_id[note_manual]["membership_source"] == "manual"
    assert by_id[note_smart]["membership_source"] == "smart"
    assert by_id[note_both]["membership_source"] == "both"


def test_moodboard_pagination_and_total_count(moodboard_db: CharactersRAGDB):
    moodboard_id = moodboard_db.add_moodboard(name="Paged board")
    assert moodboard_id is not None

    note_ids: list[str] = []
    for idx in range(5):
        note_id = moodboard_db.add_note(
            title=f"Paged note {idx + 1}",
            content=("x" * 400) + f"-{idx}",
        )
        assert note_id
        assert moodboard_db.link_note_to_moodboard(moodboard_id=moodboard_id, note_id=note_id) is True
        note_ids.append(note_id)

    total = moodboard_db.count_moodboard_notes(moodboard_id=moodboard_id)
    assert total == 5

    first_page = moodboard_db.list_moodboard_notes(moodboard_id=moodboard_id, limit=2, offset=0)
    second_page = moodboard_db.list_moodboard_notes(moodboard_id=moodboard_id, limit=2, offset=2)
    third_page = moodboard_db.list_moodboard_notes(moodboard_id=moodboard_id, limit=2, offset=4)

    assert len(first_page) == 2
    assert len(second_page) == 2
    assert len(third_page) == 1

    seen_ids = [row["id"] for row in first_page + second_page + third_page]
    assert len(seen_ids) == 5
    assert len(set(seen_ids)) == 5

    for row in first_page + second_page + third_page:
        assert row["membership_source"] == "manual"
        preview = row.get("content_preview")
        assert isinstance(preview, str) and len(preview) <= 280
