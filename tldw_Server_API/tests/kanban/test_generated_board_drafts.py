"""Generated board saves use the real Kanban DB and cannot leave partial state."""

import sqlite3
from copy import deepcopy

import pytest

from tldw_Server_API.app.core.DB_Management.Kanban_DB import ConflictError, InputError, KanbanDBError


def test_generated_draft_saves_complete_ordered_board_and_exports(kanban_db, draft):
    result = kanban_db.import_board(draft)
    board = kanban_db.get_board_with_lists_and_cards(result["board"]["id"])
    expected = [
        (row["name"], [(card["title"], card["description"]) for card in row["cards"]]) for row in draft["lists"]
    ]
    assert [
        (row["name"], [(card["title"], card["description"]) for card in row["cards"]]) for row in board["lists"]
    ] == expected  # nosec B101 - full ordered content round-trip.
    exported = kanban_db.export_board(board["id"])
    assert [
        (row["name"], [(card["title"], card["description"]) for card in row["cards"]]) for row in exported["lists"]
    ] == expected  # nosec B101 - existing exporter reads every saved item.
    assert (result["import_stats"]["lists_imported"], result["import_stats"]["cards_imported"]) == (2, 3)  # nosec B101 - truthful complete-save counts.


def test_matching_replay_returns_existing_board_without_rewriting(kanban_db, draft):
    first = kanban_db.import_board(draft)
    saved = kanban_db.get_board_with_lists_and_cards(first["board"]["id"])
    replay = kanban_db.import_board(deepcopy(draft))
    assert replay["board"]["id"] == first["board"]["id"] and kanban_db.list_boards()[1] == 1  # nosec B101 - lost acknowledgement cannot create another board.
    assert kanban_db.get_board_with_lists_and_cards(first["board"]["id"]) == saved  # nosec B101 - replay must not overwrite state.


def test_changed_replay_conflicts_without_overwriting(kanban_db, draft):
    saved = kanban_db.import_board(draft)["board"]
    before = kanban_db.get_board_with_lists_and_cards(saved["id"])
    changed = deepcopy(draft)
    changed["lists"][0]["cards"][0]["title"] = "Different intent"
    with pytest.raises(ConflictError):
        kanban_db.import_board(changed)
    assert kanban_db.get_board_with_lists_and_cards(saved["id"]) == before  # nosec B101 - changed intent cannot reuse a saved identity.


def test_same_identity_is_scoped_to_each_user(kanban_db, kanban_db_user2, draft):
    alice = kanban_db.import_board(draft)["board"]
    bob = kanban_db_user2.import_board(draft)["board"]
    assert alice["uuid"] != bob["uuid"] and alice["user_id"] != bob["user_id"]  # nosec B101 - duplicate client IDs cannot acquire another owner's board.


@pytest.mark.parametrize("failure", [sqlite3.OperationalError, RuntimeError, KeyboardInterrupt])
def test_late_card_or_activity_failure_rolls_back_every_new_item(kanban_db, draft, monkeypatch, failure):
    retained = kanban_db.create_board(name="Retained", client_id="retained-board")
    before = kanban_db.get_board_with_lists_and_cards(retained["id"])
    original = kanban_db._log_activity_internal
    calls = []

    def fail_late(conn, board_id, action, entity_type, *args, **kwargs):
        original(conn, board_id, action, entity_type, *args, **kwargs)
        if entity_type == "card":
            calls.append(True)
            if len(calls) == 2:
                raise failure("Controlled late activity failure")

    monkeypatch.setattr(kanban_db, "_log_activity_internal", fail_late)
    with pytest.raises(KanbanDBError if failure is sqlite3.OperationalError else failure):
        kanban_db.import_board(draft)
    assert len(calls) == 2  # nosec B101 - the causal failure occurs after real board/list/card inserts.
    assert kanban_db.list_boards()[1] == 1 and kanban_db.get_board_with_lists_and_cards(retained["id"]) == before  # nosec B101 - atomic rollback preserves pre-existing data.
    assert kanban_db.search_cards("Selene")[1] == 0  # nosec B101 - FTS cannot retain rolled-back card content.


def test_generated_draft_respects_existing_board_limit(kanban_db, draft, monkeypatch):
    kanban_db.create_board(name="Retained", client_id="at-limit")
    monkeypatch.setattr(kanban_db, "MAX_BOARDS_PER_USER", 1)
    with pytest.raises(InputError, match="Maximum boards"):
        kanban_db.import_board(draft)
    assert kanban_db.list_boards()[1] == 1  # nosec B101 - no partial save beyond the normal limit.


@pytest.mark.parametrize(
    "change", ["owner", "empty-name", "foreign-card-id", "empty-lists", "empty-cards", "oversized"]
)
def test_invalid_draft_is_rejected_before_writes(kanban_db, draft, change):
    if change == "owner":
        draft["board"]["user_id"] = "other-user"
    elif change == "empty-name":
        draft["board"]["name"] = "   "
    elif change == "foreign-card-id":
        draft["lists"][0]["cards"][0]["id"] = 17
    elif change == "empty-lists":
        draft["lists"] = []
    elif change == "empty-cards":
        for row in draft["lists"]:
            row["cards"] = []
    else:
        draft["lists"][0]["cards"][0]["description"] = "x" * 5001
    with pytest.raises(InputError, match="generated board draft"):
        kanban_db.import_board(draft)
    assert kanban_db.list_boards()[1] == 0  # nosec B101 - malformed model output cannot create a board.


def test_concurrent_matching_saves_create_one_complete_board(kanban_db, draft):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from tldw_Server_API.app.core.DB_Management.Kanban_DB import KanbanDB

    other = KanbanDB(db_path=kanban_db.db_path, user_id="1")
    ready = threading.Barrier(2)

    def save(db):
        ready.wait(timeout=2)
        return db.import_board(deepcopy(draft))["board"]["id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(save, db) for db in (kanban_db, other)]
        identities = [future.result(timeout=35) for future in futures]
    assert identities[0] == identities[1] and kanban_db.list_boards()[1] == 1  # nosec B101 - real SQLite writers serialize a single intent.
    board = kanban_db.get_board_with_lists_and_cards(identities[0])
    assert len(board["lists"]) == 2 and sum(len(row["cards"]) for row in board["lists"]) == 3  # nosec B101 - concurrent replay cannot duplicate children.


def test_replay_preserves_later_user_edits(kanban_db, draft):
    saved = kanban_db.import_board(draft)["board"]
    kanban_db.update_board(saved["id"], name="User's later title")
    before = kanban_db.get_board_with_lists_and_cards(saved["id"])
    assert kanban_db.import_board(draft)["board"]["id"] == saved["id"]  # nosec B101 - replay still resolves the original save.
    assert kanban_db.get_board_with_lists_and_cards(saved["id"]) == before  # nosec B101 - replay never overwrites a later edit.


def test_deleted_draft_identity_cannot_resurrect_a_board(kanban_db, draft):
    saved = kanban_db.import_board(draft)["board"]
    kanban_db.delete_board(saved["id"])
    with pytest.raises(ConflictError):
        kanban_db.import_board(draft)
    assert kanban_db.list_boards()[1] == 0  # nosec B101 - old save retries cannot reverse deletion.


@pytest.mark.parametrize("fail_late", [False, True])
def test_generated_cards_reach_vector_index_only_after_commit(kanban_db, draft, monkeypatch, fail_late):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.DB_Management.Kanban_DB import KanbanDB

    other = KanbanDB(db_path=kanban_db.db_path, user_id="1")
    indexed, visible = {}, []

    def index(card):
        visible.append(other.get_card(card["id"]))
        indexed[card["id"]] = card
        return True

    monkeypatch.setattr(kanban_db, "_get_vector_search", lambda: SimpleNamespace(index_card=index))
    original = kanban_db._log_activity_internal
    calls = []

    def fail(conn, board_id, action, entity_type, *args, **kwargs):
        original(conn, board_id, action, entity_type, *args, **kwargs)
        if entity_type == "card":
            calls.append(True)
            if fail_late and len(calls) == 2:
                raise sqlite3.OperationalError("Controlled late write")

    monkeypatch.setattr(kanban_db, "_log_activity_internal", fail)
    try:
        if fail_late:
            with pytest.raises(KanbanDBError):
                kanban_db.import_board(draft)
            assert len(calls) == 2 and not indexed and not visible  # nosec B101 - rolled-back cards must never enter external indexes.
        else:
            saved = kanban_db.import_board(draft)["board"]
            assert len(indexed) == 3 and all(card and card["board_id"] == saved["id"] for card in visible)  # nosec B101 - indexing sees all committed rows through another connection.
            assert {card["title"] for card in indexed.values()} == {
                card["title"] for row in draft["lists"] for card in row["cards"]
            }  # nosec B101 - every generated card reaches the existing index boundary.
            kanban_db.import_board(draft)
            assert len(visible) == 3  # nosec B101 - replay does not schedule duplicate embedding work.
    finally:
        other.close()
