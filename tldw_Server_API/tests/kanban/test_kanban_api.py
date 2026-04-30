# tldw_Server_API/tests/kanban/test_kanban_api.py
"""
Integration tests for Kanban API endpoints using a real Kanban DB.
No mocking of internal functions; only dependency override to inject a temp DB.
"""

import importlib
import sqlite3
from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    InputError,
    KanbanDB,
    KanbanDBError,
    NotFoundError,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


pytestmark = pytest.mark.integration


@pytest.fixture()
def client_with_kanban_db(tmp_path, monkeypatch):
    """Create a test client with a temporary Kanban database."""
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    auth_db_path = (tmp_path / "auth_users.db").resolve()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:////{auth_db_path}")
    db_path = DatabasePaths.get_kanban_db_path("integration_test_user")
    db = KanbanDB(str(db_path), user_id="integration_test_user")

    async def override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    # Inject per-user DB via dependency override
    from tldw_Server_API.app.api.v1.API_Deps.kanban_deps import get_kanban_db_for_user

    def override_db_dep():

        return db

    # Use full app profile so Kanban routes are included
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")

    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    fastapi_app = app_main.app

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_kanban_db_for_user] = override_db_dep

    with TestClient(fastapi_app) as client:
        yield client, db

    fastapi_app.dependency_overrides.clear()


# =============================================================================
# Board CRUD Tests
# =============================================================================

def test_board_crud(client_with_kanban_db):

    client, db = client_with_kanban_db

    # Create board
    create_resp = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Test Board", "client_id": "board-1", "description": "A test board"}
    )
    assert create_resp.status_code == 201, create_resp.text
    board = create_resp.json()
    board_id = board["id"]
    assert board["name"] == "Test Board"
    assert board["client_id"] == "board-1"

    # Get board
    get_resp = client.get(f"/api/v1/kanban/boards/{board_id}")
    assert get_resp.status_code == 200
    got = get_resp.json()
    assert got["name"] == "Test Board"

    # Update board
    upd_resp = client.patch(
        f"/api/v1/kanban/boards/{board_id}",
        json={"name": "Updated Board"}
    )
    assert upd_resp.status_code == 200
    assert upd_resp.json()["name"] == "Updated Board"

    # List boards
    list_resp = client.get("/api/v1/kanban/boards")
    assert list_resp.status_code == 200
    data = list_resp.json()
    assert "boards" in data
    assert any(b["id"] == board_id for b in data["boards"])

    # Archive board
    archive_resp = client.post(f"/api/v1/kanban/boards/{board_id}/archive")
    assert archive_resp.status_code == 200

    # Unarchive board
    unarchive_resp = client.post(f"/api/v1/kanban/boards/{board_id}/unarchive")
    assert unarchive_resp.status_code == 200

    # Delete board (soft) - returns 200 with detail message
    del_resp = client.delete(f"/api/v1/kanban/boards/{board_id}")
    assert del_resp.status_code == 200

    # Restore board
    restore_resp = client.post(f"/api/v1/kanban/boards/{board_id}/restore")
    assert restore_resp.status_code == 200


def test_board_list_counts(client_with_kanban_db):

    client, db = client_with_kanban_db

    board_resp = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Counts Board", "client_id": "board-counts-1"}
    )
    assert board_resp.status_code == 201
    board_id = board_resp.json()["id"]

    list_resp = client.post(
        f"/api/v1/kanban/boards/{board_id}/lists",
        json={"name": "Counts List", "client_id": "list-counts-1"}
    )
    assert list_resp.status_code == 201
    list_id = list_resp.json()["id"]

    card_resp = client.post(
        f"/api/v1/kanban/lists/{list_id}/cards",
        json={"title": "Counts Card", "client_id": "card-counts-1"}
    )
    assert card_resp.status_code == 201

    boards_resp = client.get("/api/v1/kanban/boards")
    assert boards_resp.status_code == 200
    boards = boards_resp.json()["boards"]
    board_entry = next(b for b in boards if b["id"] == board_id)
    assert board_entry["list_count"] == 1
    assert board_entry["card_count"] == 1


def test_board_list_pagination_legacy_page_params(client_with_kanban_db):
    """Legacy page/per_page should be translated to limit/offset."""
    client, _db = client_with_kanban_db

    for idx in range(5):
        resp = client.post(
            "/api/v1/kanban/boards",
            json={"name": f"Paged Board {idx}", "client_id": f"board-page-{idx}"},
        )
        assert resp.status_code == 201

    paged_resp = client.get("/api/v1/kanban/boards", params={"page": 2, "per_page": 2})
    assert paged_resp.status_code == 200
    payload = paged_resp.json()
    assert payload["pagination"]["limit"] == 2
    assert payload["pagination"]["offset"] == 2
    assert len(payload["boards"]) <= 2


def test_board_create_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while creating board")

    monkeypatch.setattr(db, "create_board", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Broken Board", "client_id": "board-db-create-1"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to create board"


def test_board_list_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while listing boards")

    monkeypatch.setattr(db, "list_boards", _raise_db_error)

    response = client.get("/api/v1/kanban/boards")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch boards"


def test_board_get_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while loading board")

    monkeypatch.setattr(db, "get_board_with_lists_and_cards", _raise_db_error)

    response = client.get("/api/v1/kanban/boards/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch board"


def test_board_update_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while updating board")

    monkeypatch.setattr(db, "update_board", _raise_db_error)

    response = client.patch(
        "/api/v1/kanban/boards/999",
        json={"name": "Broken Update"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to update board"


def test_board_archive_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while archiving board")

    monkeypatch.setattr(db, "archive_board", _raise_db_error)

    response = client.post("/api/v1/kanban/boards/999/archive")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to archive board"


def test_board_unarchive_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while unarchiving board")

    monkeypatch.setattr(db, "archive_board", _raise_db_error)

    response = client.post("/api/v1/kanban/boards/999/unarchive")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to unarchive board"


def test_board_delete_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while deleting board")

    monkeypatch.setattr(db, "delete_board", _raise_db_error)

    response = client.delete("/api/v1/kanban/boards/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete board"


def test_board_restore_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while restoring board")

    monkeypatch.setattr(db, "restore_board", _raise_db_error)

    response = client.post("/api/v1/kanban/boards/999/restore")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to restore board"


# =============================================================================
# List CRUD Tests
# =============================================================================

def test_list_crud(client_with_kanban_db):

    client, db = client_with_kanban_db

    # Create board first
    board_resp = client.post(
        "/api/v1/kanban/boards",
        json={"name": "List Test Board", "client_id": "board-list-1"}
    )
    board_id = board_resp.json()["id"]

    # Create list
    create_resp = client.post(
        f"/api/v1/kanban/boards/{board_id}/lists",
        json={"name": "Test List", "client_id": "list-1"}
    )
    assert create_resp.status_code == 201
    lst = create_resp.json()
    list_id = lst["id"]
    assert lst["name"] == "Test List"

    # Get lists for board
    get_resp = client.get(f"/api/v1/kanban/boards/{board_id}/lists")
    assert get_resp.status_code == 200
    assert any(l["id"] == list_id for l in get_resp.json()["lists"])

    # Update list
    upd_resp = client.patch(
        f"/api/v1/kanban/lists/{list_id}",
        json={"name": "Updated List"}
    )
    assert upd_resp.status_code == 200
    assert upd_resp.json()["name"] == "Updated List"

    # Reorder lists (schema uses 'ids' not 'list_ids')
    list2 = client.post(
        f"/api/v1/kanban/boards/{board_id}/lists",
        json={"name": "Second List", "client_id": "list-2"}
    ).json()
    reorder_resp = client.post(
        f"/api/v1/kanban/boards/{board_id}/lists/reorder",
        json={"ids": [list2["id"], list_id]}
    )
    assert reorder_resp.status_code == 200

    # Archive and unarchive
    archive_resp = client.post(f"/api/v1/kanban/lists/{list_id}/archive")
    assert archive_resp.status_code == 200

    unarchive_resp = client.post(f"/api/v1/kanban/lists/{list_id}/unarchive")
    assert unarchive_resp.status_code == 200

    # Delete and restore
    delete_resp = client.delete(f"/api/v1/kanban/lists/{list_id}")
    assert delete_resp.status_code == 200

    restore_resp = client.post(f"/api/v1/kanban/lists/{list_id}/restore")
    assert restore_resp.status_code == 200


def test_list_update_position_and_reorder_legacy_payload(client_with_kanban_db):
    """List position patch + list_positions reorder payload should both work."""
    client, _db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "List Compat Board", "client_id": "board-list-compat-1"},
    ).json()

    list_a = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "A", "client_id": "list-compat-a"},
    ).json()
    list_b = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "B", "client_id": "list-compat-b"},
    ).json()
    list_c = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "C", "client_id": "list-compat-c"},
    ).json()

    move_resp = client.patch(
        f"/api/v1/kanban/lists/{list_c['id']}",
        json={"position": 0},
    )
    assert move_resp.status_code == 200, move_resp.text

    ordered = client.get(f"/api/v1/kanban/boards/{board['id']}/lists").json()["lists"]
    assert ordered[0]["id"] == list_c["id"]

    reorder_resp = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists/reorder",
        json={
            "list_positions": [
                {"list_id": list_a["id"], "position": 2},
                {"list_id": list_b["id"], "position": 0},
                {"list_id": list_c["id"], "position": 1},
            ]
        },
    )
    assert reorder_resp.status_code == 200, reorder_resp.text

    reordered = client.get(f"/api/v1/kanban/boards/{board['id']}/lists").json()["lists"]
    assert [l["id"] for l in reordered] == [list_b["id"], list_c["id"], list_a["id"]]


def test_get_single_list(client_with_kanban_db):

    client, _db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Single List Board", "client_id": "board-single-list-1"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Single List", "client_id": "list-single-1"},
    ).json()

    response = client.get(f"/api/v1/kanban/lists/{lst['id']}")

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["id"] == lst["id"]
    assert data["name"] == "Single List"


def test_list_create_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "List Error Board", "client_id": "board-list-db-create-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while creating list")

    monkeypatch.setattr(db, "create_list", _raise_db_error)

    response = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Broken List", "client_id": "list-db-create-1"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to create list"


def test_list_list_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "List Error Board", "client_id": "board-list-db-list-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while listing lists")

    monkeypatch.setattr(db, "list_lists", _raise_db_error)

    response = client.get(f"/api/v1/kanban/boards/{board['id']}/lists")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch lists"


def test_get_single_list_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while loading list")

    monkeypatch.setattr(db, "get_list", _raise_db_error)

    response = client.get("/api/v1/kanban/lists/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch list"


def test_list_update_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while updating list")

    monkeypatch.setattr(db, "update_list", _raise_db_error)

    response = client.patch(
        "/api/v1/kanban/lists/999",
        json={"name": "Broken Update"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to update list"


def test_list_reorder_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "List Error Board", "client_id": "board-list-db-reorder-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while reordering lists")

    monkeypatch.setattr(db, "reorder_lists", _raise_db_error)

    response = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists/reorder",
        json={"ids": [1]},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to reorder lists"


def test_list_activities_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "List Error Board", "client_id": "board-list-db-activities-1"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Activity List", "client_id": "list-db-activities-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while loading list activities")

    monkeypatch.setattr(db, "get_list_activities", _raise_db_error)

    response = client.get(f"/api/v1/kanban/lists/{lst['id']}/activities")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch list activities"


def test_list_archive_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while archiving list")

    monkeypatch.setattr(db, "archive_list", _raise_db_error)

    response = client.post("/api/v1/kanban/lists/999/archive")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to archive list"


def test_list_unarchive_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while unarchiving list")

    monkeypatch.setattr(db, "archive_list", _raise_db_error)

    response = client.post("/api/v1/kanban/lists/999/unarchive")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to unarchive list"


def test_list_delete_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while deleting list")

    monkeypatch.setattr(db, "delete_list", _raise_db_error)

    response = client.delete("/api/v1/kanban/lists/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete list"


def test_list_restore_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while restoring list")

    monkeypatch.setattr(db, "restore_list", _raise_db_error)

    response = client.post("/api/v1/kanban/lists/999/restore")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to restore list"


# =============================================================================
# Card CRUD Tests
# =============================================================================

def test_card_crud(client_with_kanban_db):

    client, db = client_with_kanban_db

    # Setup board and list
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Card Test Board", "client_id": "board-card-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Card Test List", "client_id": "list-card-1"}
    ).json()

    # Create card
    create_resp = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={
            "title": "Test Card",
            "client_id": "card-1",
            "description": "A test card",
            "priority": "high"
        }
    )
    assert create_resp.status_code == 201
    card = create_resp.json()
    card_id = card["id"]
    assert card["title"] == "Test Card"
    assert card["priority"] == "high"

    # Get card
    get_resp = client.get(f"/api/v1/kanban/cards/{card_id}")
    assert get_resp.status_code == 200
    card_detail = get_resp.json()
    assert card_detail["title"] == "Test Card"
    assert isinstance(card_detail["labels"], list)
    assert isinstance(card_detail["checklists"], list)
    assert card_detail["comment_count"] == 0

    # Update card
    upd_resp = client.patch(
        f"/api/v1/kanban/cards/{card_id}",
        json={"title": "Updated Card", "priority": "low"}
    )
    assert upd_resp.status_code == 200
    assert upd_resp.json()["title"] == "Updated Card"
    assert upd_resp.json()["priority"] == "low"

    # List cards
    list_resp = client.get(f"/api/v1/kanban/lists/{lst['id']}/cards")
    assert list_resp.status_code == 200
    assert any(c["id"] == card_id for c in list_resp.json()["cards"])

    # Copy card
    copy_resp = client.post(
        f"/api/v1/kanban/cards/{card_id}/copy",
        json={"target_list_id": lst["id"], "new_client_id": "card-1-copy"}
    )
    assert copy_resp.status_code == 201

    # Move card (create another list first)
    lst2 = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Second List", "client_id": "list-card-2"}
    ).json()
    move_resp = client.post(
        f"/api/v1/kanban/cards/{card_id}/move",
        json={"target_list_id": lst2["id"]}
    )
    assert move_resp.status_code == 200

    # Archive and unarchive
    client.post(f"/api/v1/kanban/cards/{card_id}/archive")
    client.post(f"/api/v1/kanban/cards/{card_id}/unarchive")

    # Delete and restore
    client.delete(f"/api/v1/kanban/cards/{card_id}")
    client.post(f"/api/v1/kanban/cards/{card_id}/restore")


def test_card_create_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while creating card")

    monkeypatch.setattr(db, "create_card", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/lists/999/cards",
        json={"title": "Broken Card", "client_id": "card-db-create-1"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to create card"


def test_card_list_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while listing cards")

    monkeypatch.setattr(db, "list_cards", _raise_db_error)

    response = client.get("/api/v1/kanban/lists/999/cards")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch cards"


def test_card_get_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while fetching card")

    monkeypatch.setattr(db, "get_card_with_details", _raise_db_error)

    response = client.get("/api/v1/kanban/cards/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch card"


def test_card_update_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while updating card")

    monkeypatch.setattr(db, "update_card", _raise_db_error)

    response = client.patch(
        "/api/v1/kanban/cards/999",
        json={"title": "Still Broken"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to update card"


def test_card_move_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while moving card")

    monkeypatch.setattr(db, "move_card", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/999/move",
        json={"target_list_id": 1000},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to move card"


def test_card_copy_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while copying card")

    monkeypatch.setattr(db, "copy_card_with_checklists", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/999/copy",
        json={"target_list_id": 1000, "new_client_id": "card-copy-db-1"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to copy card"


def test_card_archive_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while archiving card")

    monkeypatch.setattr(db, "archive_card", _raise_db_error)

    response = client.post("/api/v1/kanban/cards/999/archive")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to archive card"


def test_card_unarchive_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while unarchiving card")

    monkeypatch.setattr(db, "archive_card", _raise_db_error)

    response = client.post("/api/v1/kanban/cards/999/unarchive")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to unarchive card"


def test_card_delete_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while deleting card")

    monkeypatch.setattr(db, "delete_card", _raise_db_error)

    response = client.delete("/api/v1/kanban/cards/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete card"


def test_card_restore_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while restoring card")

    monkeypatch.setattr(db, "restore_card", _raise_db_error)

    response = client.post("/api/v1/kanban/cards/999/restore")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to restore card"


def test_card_create_accepts_label_ids_and_reorder_legacy_payload(client_with_kanban_db):
    """Card create label_ids + card_positions reorder payload should work."""
    client, _db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Card Compat Board", "client_id": "board-card-compat-1"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Compat List", "client_id": "list-card-compat-1"},
    ).json()

    label_bug = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Bug", "color": "red"},
    ).json()
    label_feat = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Feature", "color": "green"},
    ).json()

    card1 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={
            "title": "Card one",
            "client_id": "card-compat-1",
            "label_ids": [label_bug["id"], label_feat["id"]],
        },
    ).json()
    card2 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Card two", "client_id": "card-compat-2"},
    ).json()

    labels_resp = client.get(f"/api/v1/kanban/cards/{card1['id']}/labels")
    assert labels_resp.status_code == 200
    label_ids = {l["id"] for l in labels_resp.json()["labels"]}
    assert label_bug["id"] in label_ids
    assert label_feat["id"] in label_ids

    reorder_resp = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards/reorder",
        json={
            "card_positions": [
                {"card_id": card1["id"], "position": 1},
                {"card_id": card2["id"], "position": 0},
            ]
        },
    )
    assert reorder_resp.status_code == 200, reorder_resp.text

    cards = client.get(f"/api/v1/kanban/lists/{lst['id']}/cards").json()["cards"]
    assert [c["id"] for c in cards] == [card2["id"], card1["id"]]


def test_reorder_cards_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while reordering cards")

    monkeypatch.setattr(db, "reorder_cards", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/lists/999/cards/reorder",
        json={"ids": [1, 2]},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to reorder cards"


def test_search_supports_legacy_page_params(client_with_kanban_db):
    """Search endpoint should accept page/per_page compatibility params."""
    client, _db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Search Compat Board", "client_id": "board-search-compat-1"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Search Compat List", "client_id": "list-search-compat-1"},
    ).json()

    for idx in range(5):
        client.post(
            f"/api/v1/kanban/lists/{lst['id']}/cards",
            json={"title": f"compat token {idx}", "client_id": f"search-compat-card-{idx}"},
        )

    resp = client.get(
        "/api/v1/kanban/search",
        params={"q": "compat", "page": 2, "per_page": 2},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["pagination"]["limit"] == 2
    assert payload["pagination"]["offset"] == 2
    assert len(payload["results"]) <= 2


# =============================================================================
# Activity Endpoints Tests
# =============================================================================

def test_activity_endpoints_filters(client_with_kanban_db):
    """Verify activity endpoint filtering behavior."""

    client, _db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Activity Board", "client_id": "board-activity-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Activity List", "client_id": "list-activity-1"}
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Activity Card", "client_id": "card-activity-1"}
    ).json()

    window_start = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    window_end = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()

    board_resp = client.get(
        f"/api/v1/kanban/boards/{board['id']}/activities",
        params={
            "created_after": window_start,
            "created_before": window_end,
            "action_type": "card_created",
            "entity_type": "card",
        },
    )
    assert board_resp.status_code == 200
    assert any(a["action_type"] == "card_created" for a in board_resp.json()["activities"])

    list_resp = client.get(
        f"/api/v1/kanban/lists/{lst['id']}/activities",
        params={
            "created_after": window_start,
            "created_before": window_end,
            "action_type": "list_created",
            "entity_type": "list",
        },
    )
    assert list_resp.status_code == 200
    assert any(a["action_type"] == "list_created" for a in list_resp.json()["activities"])

    card_resp = client.get(
        f"/api/v1/kanban/cards/{card['id']}/activities",
        params={
            "created_after": window_start,
            "created_before": window_end,
            "action_type": "card_created",
            "entity_type": "card",
        },
    )
    assert card_resp.status_code == 200
    assert any(a["action_type"] == "card_created" for a in card_resp.json()["activities"])


def test_card_activities_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while fetching card activities")

    monkeypatch.setattr(db, "get_card_activities", _raise_db_error)

    response = client.get("/api/v1/kanban/cards/999/activities")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch card activities"


# =============================================================================
# Label CRUD Tests
# =============================================================================

def test_label_crud(client_with_kanban_db):

    client, db = client_with_kanban_db

    # Create board
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Label Test Board", "client_id": "board-label-1"}
    ).json()

    # Create label (color must be one of: red, orange, yellow, green, blue, purple, pink, gray)
    create_resp = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Bug", "color": "red"}
    )
    assert create_resp.status_code == 201
    label = create_resp.json()
    label_id = label["id"]
    assert label["name"] == "Bug"
    assert label["color"] == "red"

    # List labels
    list_resp = client.get(f"/api/v1/kanban/boards/{board['id']}/labels")
    assert list_resp.status_code == 200
    assert any(l["id"] == label_id for l in list_resp.json()["labels"])

    # Update label
    upd_resp = client.patch(
        f"/api/v1/kanban/labels/{label_id}",
        json={"name": "Critical Bug", "color": "orange"}
    )
    assert upd_resp.status_code == 200
    assert upd_resp.json()["name"] == "Critical Bug"

    # Create list and card to test label assignment
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Label Test List", "client_id": "list-label-1"}
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Card for Labels", "client_id": "card-label-1"}
    ).json()

    # Add label to card
    add_resp = client.post(f"/api/v1/kanban/cards/{card['id']}/labels/{label_id}")
    assert add_resp.status_code == 200

    # Get card labels
    card_labels = client.get(f"/api/v1/kanban/cards/{card['id']}/labels")
    assert card_labels.status_code == 200
    assert any(l["id"] == label_id for l in card_labels.json()["labels"])

    # Remove label from card
    remove_resp = client.delete(f"/api/v1/kanban/cards/{card['id']}/labels/{label_id}")
    assert remove_resp.status_code == 204

    # Delete label
    del_resp = client.delete(f"/api/v1/kanban/labels/{label_id}")
    assert del_resp.status_code == 204


def test_label_create_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Label Error Board", "client_id": "board-label-db-create-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while creating label")

    monkeypatch.setattr(db, "create_label", _raise_db_error)

    response = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Bug", "color": "red"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to create label"


def test_label_list_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Label Error Board", "client_id": "board-label-db-list-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while listing labels")

    monkeypatch.setattr(db, "list_labels", _raise_db_error)

    response = client.get(f"/api/v1/kanban/boards/{board['id']}/labels")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch labels"


def test_label_get_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while loading label")

    monkeypatch.setattr(db, "get_label", _raise_db_error)

    response = client.get("/api/v1/kanban/labels/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch label"


def test_label_update_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while updating label")

    monkeypatch.setattr(db, "update_label", _raise_db_error)

    response = client.patch(
        "/api/v1/kanban/labels/999",
        json={"name": "Updated Label"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to update label"


def test_label_delete_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while deleting label")

    monkeypatch.setattr(db, "delete_label", _raise_db_error)

    response = client.delete("/api/v1/kanban/labels/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete label"


def test_label_assign_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Label Assign Error Board", "client_id": "board-label-assign-db-1"},
    ).json()
    label = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Bug", "color": "red"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Label Assign Error List", "client_id": "list-label-assign-db-1"},
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Label Assign Error Card", "client_id": "card-label-assign-db-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while assigning label to card")

    monkeypatch.setattr(db, "assign_label_to_card", _raise_db_error)

    response = client.post(f"/api/v1/kanban/cards/{card['id']}/labels/{label['id']}")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to assign label to card"


def test_card_label_list_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Card Labels Error Board", "client_id": "board-card-labels-db-1"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Card Labels Error List", "client_id": "list-card-labels-db-1"},
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Card Labels Error Card", "client_id": "card-card-labels-db-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while listing card labels")

    monkeypatch.setattr(db, "get_card_labels", _raise_db_error)

    response = client.get(f"/api/v1/kanban/cards/{card['id']}/labels")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch card labels"


def test_label_remove_from_card_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Label Remove Error Board", "client_id": "board-label-remove-db-1"},
    ).json()
    label = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Bug", "color": "red"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Label Remove Error List", "client_id": "list-label-remove-db-1"},
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Label Remove Error Card", "client_id": "card-label-remove-db-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while removing label from card")

    monkeypatch.setattr(db, "remove_label_from_card", _raise_db_error)

    response = client.delete(f"/api/v1/kanban/cards/{card['id']}/labels/{label['id']}")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to remove label from card"


# =============================================================================
# Checklist CRUD Tests
# =============================================================================

def test_checklist_crud(client_with_kanban_db):

    client, db = client_with_kanban_db

    # Setup board, list, card
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Checklist Test Board", "client_id": "board-cl-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Checklist Test List", "client_id": "list-cl-1"}
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Card for Checklists", "client_id": "card-cl-1"}
    ).json()

    # Create checklist
    create_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/checklists",
        json={"name": "Task List"}
    )
    assert create_resp.status_code == 201
    checklist = create_resp.json()
    checklist_id = checklist["id"]
    assert checklist["name"] == "Task List"

    # List checklists
    list_resp = client.get(f"/api/v1/kanban/cards/{card['id']}/checklists")
    assert list_resp.status_code == 200
    assert any(c["id"] == checklist_id for c in list_resp.json()["checklists"])

    # Update checklist
    upd_resp = client.patch(
        f"/api/v1/kanban/checklists/{checklist_id}",
        json={"name": "Updated Task List"}
    )
    assert upd_resp.status_code == 200
    assert upd_resp.json()["name"] == "Updated Task List"

    # Get checklist with items
    get_resp = client.get(f"/api/v1/kanban/checklists/{checklist_id}")
    assert get_resp.status_code == 200
    assert "items" in get_resp.json()

    # Delete checklist
    del_resp = client.delete(f"/api/v1/kanban/checklists/{checklist_id}")
    assert del_resp.status_code == 204


def test_checklist_create_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while creating checklist")

    monkeypatch.setattr(db, "create_checklist", _raise_db_error)

    response = client.post("/api/v1/kanban/cards/999/checklists", json={"name": "Checklist"})

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to create checklist"


def test_checklist_list_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while listing checklists")

    monkeypatch.setattr(db, "list_checklists", _raise_db_error)

    response = client.get("/api/v1/kanban/cards/999/checklists")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch checklists"


def test_checklist_get_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while loading checklist")

    monkeypatch.setattr(db, "get_checklist_with_items", _raise_db_error)

    response = client.get("/api/v1/kanban/checklists/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch checklist"


def test_checklist_update_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while updating checklist")

    monkeypatch.setattr(db, "update_checklist", _raise_db_error)

    response = client.patch("/api/v1/kanban/checklists/999", json={"name": "Updated Checklist"})

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to update checklist"


def test_checklist_delete_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while deleting checklist")

    monkeypatch.setattr(db, "delete_checklist", _raise_db_error)

    response = client.delete("/api/v1/kanban/checklists/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete checklist"


def test_checklist_reorder(client_with_kanban_db):

    client, _db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Checklist Reorder Board", "client_id": "board-checklist-reorder-1"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Checklist Reorder List", "client_id": "list-checklist-reorder-1"},
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Checklist Reorder Card", "client_id": "card-checklist-reorder-1"},
    ).json()

    checklist_a = client.post(
        f"/api/v1/kanban/cards/{card['id']}/checklists",
        json={"name": "Checklist A"},
    ).json()
    checklist_b = client.post(
        f"/api/v1/kanban/cards/{card['id']}/checklists",
        json={"name": "Checklist B"},
    ).json()

    reorder_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/checklists/reorder",
        json={"checklist_ids": [checklist_b["id"], checklist_a["id"]]},
    )

    assert reorder_resp.status_code == 200, reorder_resp.text
    assert [entry["id"] for entry in reorder_resp.json()["checklists"]] == [
        checklist_b["id"],
        checklist_a["id"],
    ]


def test_checklist_reorder_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while reordering checklists")

    monkeypatch.setattr(db, "reorder_checklists", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/999/checklists/reorder",
        json={"checklist_ids": [1]},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to reorder checklists"


def test_checklist_item_crud(client_with_kanban_db):


    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Item Test Board", "client_id": "board-item-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Item Test List", "client_id": "list-item-1"}
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Card for Items", "client_id": "card-item-1"}
    ).json()
    checklist = client.post(
        f"/api/v1/kanban/cards/{card['id']}/checklists",
        json={"name": "Items Checklist"}
    ).json()

    # Create item
    create_resp = client.post(
        f"/api/v1/kanban/checklists/{checklist['id']}/items",
        json={"name": "First Task"}
    )
    assert create_resp.status_code == 201
    item = create_resp.json()
    item_id = item["id"]
    assert item["name"] == "First Task"
    assert item["checked"] is False

    # List items
    list_resp = client.get(f"/api/v1/kanban/checklists/{checklist['id']}/items")
    assert list_resp.status_code == 200
    assert any(entry["id"] == item_id for entry in list_resp.json()["items"])

    # Get item
    get_resp = client.get(f"/api/v1/kanban/checklist-items/{item_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == item_id

    # Check item
    check_resp = client.post(f"/api/v1/kanban/checklist-items/{item_id}/check")
    assert check_resp.status_code == 200
    assert check_resp.json()["checked"] is True

    # Uncheck item
    uncheck_resp = client.post(f"/api/v1/kanban/checklist-items/{item_id}/uncheck")
    assert uncheck_resp.status_code == 200
    assert uncheck_resp.json()["checked"] is False

    # Update item
    upd_resp = client.patch(
        f"/api/v1/kanban/checklist-items/{item_id}",
        json={"name": "Updated Task", "checked": True}
    )
    assert upd_resp.status_code == 200
    assert upd_resp.json()["name"] == "Updated Task"

    # Delete item
    del_resp = client.delete(f"/api/v1/kanban/checklist-items/{item_id}")
    assert del_resp.status_code == 204


def test_checklist_item_reorder(client_with_kanban_db):

    client, _db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Checklist Item Reorder Board", "client_id": "board-checklist-item-reorder-1"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Checklist Item Reorder List", "client_id": "list-checklist-item-reorder-1"},
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Checklist Item Reorder Card", "client_id": "card-checklist-item-reorder-1"},
    ).json()
    checklist = client.post(
        f"/api/v1/kanban/cards/{card['id']}/checklists",
        json={"name": "Checklist Reorder"},
    ).json()

    item_a = client.post(
        f"/api/v1/kanban/checklists/{checklist['id']}/items",
        json={"name": "Item A"},
    ).json()
    item_b = client.post(
        f"/api/v1/kanban/checklists/{checklist['id']}/items",
        json={"name": "Item B"},
    ).json()

    reorder_resp = client.post(
        f"/api/v1/kanban/checklists/{checklist['id']}/items/reorder",
        json={"item_ids": [item_b["id"], item_a["id"]]},
    )

    assert reorder_resp.status_code == 200, reorder_resp.text
    assert [entry["id"] for entry in reorder_resp.json()["items"]] == [
        item_b["id"],
        item_a["id"],
    ]


def test_checklist_item_create_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while creating checklist item")

    monkeypatch.setattr(db, "create_checklist_item", _raise_db_error)

    response = client.post("/api/v1/kanban/checklists/999/items", json={"name": "Item"})

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to create checklist item"


def test_checklist_item_list_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while listing checklist items")

    monkeypatch.setattr(db, "list_checklist_items", _raise_db_error)

    response = client.get("/api/v1/kanban/checklists/999/items")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch checklist items"


def test_checklist_item_get_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while loading checklist item")

    monkeypatch.setattr(db, "get_checklist_item", _raise_db_error)

    response = client.get("/api/v1/kanban/checklist-items/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch checklist item"


def test_checklist_item_update_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while updating checklist item")

    monkeypatch.setattr(db, "update_checklist_item", _raise_db_error)

    response = client.patch("/api/v1/kanban/checklist-items/999", json={"name": "Updated Item"})

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to update checklist item"


def test_checklist_item_delete_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while deleting checklist item")

    monkeypatch.setattr(db, "delete_checklist_item", _raise_db_error)

    response = client.delete("/api/v1/kanban/checklist-items/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete checklist item"


def test_checklist_item_reorder_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while reordering checklist items")

    monkeypatch.setattr(db, "reorder_checklist_items", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/checklists/999/items/reorder",
        json={"item_ids": [1]},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to reorder checklist items"


def test_toggle_all_checklist_items(client_with_kanban_db):


    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Toggle Test Board", "client_id": "board-toggle-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Toggle Test List", "client_id": "list-toggle-1"}
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Card for Toggle", "client_id": "card-toggle-1"}
    ).json()
    checklist = client.post(
        f"/api/v1/kanban/cards/{card['id']}/checklists",
        json={"name": "Toggle Checklist"}
    ).json()

    # Add items
    client.post(f"/api/v1/kanban/checklists/{checklist['id']}/items", json={"name": "Task 1"})
    client.post(f"/api/v1/kanban/checklists/{checklist['id']}/items", json={"name": "Task 2"})
    client.post(f"/api/v1/kanban/checklists/{checklist['id']}/items", json={"name": "Task 3"})

    # Toggle all checked
    toggle_resp = client.post(
        f"/api/v1/kanban/checklists/{checklist['id']}/toggle-all",
        json={"checked": True}
    )
    assert toggle_resp.status_code == 200
    result = toggle_resp.json()
    assert all(item["checked"] for item in result["items"])

    # Toggle all unchecked
    toggle_resp2 = client.post(
        f"/api/v1/kanban/checklists/{checklist['id']}/toggle-all",
        json={"checked": False}
    )
    assert toggle_resp2.status_code == 200
    result2 = toggle_resp2.json()
    assert not any(item["checked"] for item in result2["items"])


def test_checklist_item_check_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while checking checklist item")

    monkeypatch.setattr(db, "update_checklist_item", _raise_db_error)

    response = client.post("/api/v1/kanban/checklist-items/999/check")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to check checklist item"


def test_checklist_item_uncheck_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while unchecking checklist item")

    monkeypatch.setattr(db, "update_checklist_item", _raise_db_error)

    response = client.post("/api/v1/kanban/checklist-items/999/uncheck")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to uncheck checklist item"


def test_toggle_all_checklist_items_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while toggling checklist items")

    monkeypatch.setattr(db, "toggle_all_checklist_items", _raise_db_error)

    response = client.post("/api/v1/kanban/checklists/999/toggle-all", json={"checked": True})

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to toggle checklist items"


# =============================================================================
# Comment CRUD Tests
# =============================================================================

def test_comment_crud(client_with_kanban_db):

    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Comment Test Board", "client_id": "board-comment-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Comment Test List", "client_id": "list-comment-1"}
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Card for Comments", "client_id": "card-comment-1"}
    ).json()

    # Create comment
    create_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/comments",
        json={"content": "This is a test comment", "author_name": "Tester"}
    )
    assert create_resp.status_code == 201
    comment = create_resp.json()
    comment_id = comment["id"]
    assert comment["content"] == "This is a test comment"

    # List comments
    list_resp = client.get(f"/api/v1/kanban/cards/{card['id']}/comments")
    assert list_resp.status_code == 200
    assert any(c["id"] == comment_id for c in list_resp.json()["comments"])

    # Update comment
    upd_resp = client.patch(
        f"/api/v1/kanban/comments/{comment_id}",
        json={"content": "Updated comment content"}
    )
    assert upd_resp.status_code == 200
    assert upd_resp.json()["content"] == "Updated comment content"

    # Delete comment (soft)
    del_resp = client.delete(f"/api/v1/kanban/comments/{comment_id}")
    assert del_resp.status_code == 204


def test_comment_create_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Comment Error Board", "client_id": "board-comment-db-create-1"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Comment Error List", "client_id": "list-comment-db-create-1"},
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Comment Error Card", "client_id": "card-comment-db-create-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while creating comment")

    monkeypatch.setattr(db, "create_comment", _raise_db_error)

    response = client.post(
        f"/api/v1/kanban/cards/{card['id']}/comments",
        json={"content": "This should fail", "author_name": "Tester"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to create comment"


def test_comment_list_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Comment Error Board", "client_id": "board-comment-db-list-1"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Comment Error List", "client_id": "list-comment-db-list-1"},
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Comment Error Card", "client_id": "card-comment-db-list-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while listing comments")

    monkeypatch.setattr(db, "list_comments", _raise_db_error)

    response = client.get(f"/api/v1/kanban/cards/{card['id']}/comments")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch comments"


def test_comment_get_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while loading comment")

    monkeypatch.setattr(db, "get_comment", _raise_db_error)

    response = client.get("/api/v1/kanban/comments/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch comment"


def test_comment_update_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while updating comment")

    monkeypatch.setattr(db, "update_comment", _raise_db_error)

    response = client.patch(
        "/api/v1/kanban/comments/999",
        json={"content": "This should fail"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to update comment"


def test_comment_delete_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while deleting comment")

    monkeypatch.setattr(db, "delete_comment", _raise_db_error)

    response = client.delete("/api/v1/kanban/comments/999")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete comment"


# =============================================================================
# Bulk Operations Tests
# =============================================================================

def test_bulk_move_cards(client_with_kanban_db):

    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Bulk Move Board", "client_id": "board-bulk-move-1"}
    ).json()
    lst1 = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Source List", "client_id": "list-bulk-src-1"}
    ).json()
    lst2 = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Target List", "client_id": "list-bulk-tgt-1"}
    ).json()

    # Create cards
    card1 = client.post(
        f"/api/v1/kanban/lists/{lst1['id']}/cards",
        json={"title": "Card 1", "client_id": "bulk-card-1"}
    ).json()
    card2 = client.post(
        f"/api/v1/kanban/lists/{lst1['id']}/cards",
        json={"title": "Card 2", "client_id": "bulk-card-2"}
    ).json()

    # Bulk move
    move_resp = client.post(
        "/api/v1/kanban/cards/bulk-move",
        json={"card_ids": [card1["id"], card2["id"]], "target_list_id": lst2["id"]}
    )
    assert move_resp.status_code == 200
    result = move_resp.json()
    assert result["success"] is True
    assert result["moved_count"] == 2

    # Verify cards are in target list
    cards_resp = client.get(f"/api/v1/kanban/lists/{lst2['id']}/cards")
    card_ids = [c["id"] for c in cards_resp.json()["cards"]]
    assert card1["id"] in card_ids
    assert card2["id"] in card_ids


def test_bulk_move_cards_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while bulk moving cards")

    monkeypatch.setattr(db, "bulk_move_cards", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/bulk-move",
        json={"card_ids": [1, 2], "target_list_id": 999},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to bulk move cards"


def test_bulk_archive_cards(client_with_kanban_db):


    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Bulk Archive Board", "client_id": "board-bulk-arch-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Archive Test List", "client_id": "list-bulk-arch-1"}
    ).json()

    card1 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Archive Card 1", "client_id": "arch-card-1"}
    ).json()
    card2 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Archive Card 2", "client_id": "arch-card-2"}
    ).json()

    # Bulk archive
    archive_resp = client.post(
        "/api/v1/kanban/cards/bulk-archive",
        json={"card_ids": [card1["id"], card2["id"]]}
    )
    assert archive_resp.status_code == 200
    assert archive_resp.json()["archived_count"] == 2

    # Bulk unarchive
    unarchive_resp = client.post(
        "/api/v1/kanban/cards/bulk-unarchive",
        json={"card_ids": [card1["id"], card2["id"]]}
    )
    assert unarchive_resp.status_code == 200
    assert unarchive_resp.json()["unarchived_count"] == 2


def test_bulk_archive_cards_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while bulk archiving cards")

    monkeypatch.setattr(db, "bulk_archive_cards", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/bulk-archive",
        json={"card_ids": [1, 2]},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to bulk archive cards"


def test_bulk_unarchive_cards_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while bulk unarchiving cards")

    monkeypatch.setattr(db, "bulk_archive_cards", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/bulk-unarchive",
        json={"card_ids": [1, 2]},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to bulk unarchive cards"


def test_bulk_delete_cards(client_with_kanban_db):


    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Bulk Delete Board", "client_id": "board-bulk-del-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Delete Test List", "client_id": "list-bulk-del-1"}
    ).json()

    card1 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Delete Card 1", "client_id": "del-card-1"}
    ).json()
    card2 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Delete Card 2", "client_id": "del-card-2"}
    ).json()

    # Bulk delete
    delete_resp = client.post(
        "/api/v1/kanban/cards/bulk-delete",
        json={"card_ids": [card1["id"], card2["id"]]}
    )
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted_count"] == 2


def test_bulk_delete_cards_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while bulk deleting cards")

    monkeypatch.setattr(db, "bulk_delete_cards", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/bulk-delete",
        json={"card_ids": [1, 2]},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to bulk delete cards"


def test_bulk_label_cards(client_with_kanban_db):


    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Bulk Label Board", "client_id": "board-bulk-lbl-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Label Test List", "client_id": "list-bulk-lbl-1"}
    ).json()

    card1 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Label Card 1", "client_id": "lbl-card-1"}
    ).json()
    card2 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Label Card 2", "client_id": "lbl-card-2"}
    ).json()

    label1 = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Bug", "color": "red"}
    ).json()
    label2 = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Feature", "color": "green"}
    ).json()

    # Bulk add labels
    add_resp = client.post(
        "/api/v1/kanban/cards/bulk-label",
        json={
            "card_ids": [card1["id"], card2["id"]],
            "add_label_ids": [label1["id"], label2["id"]]
        }
    )
    assert add_resp.status_code == 200
    assert add_resp.json()["updated_count"] == 2

    # Bulk remove labels
    remove_resp = client.post(
        "/api/v1/kanban/cards/bulk-label",
        json={
            "card_ids": [card1["id"], card2["id"]],
            "remove_label_ids": [label1["id"]]
        }
    )
    assert remove_resp.status_code == 200


def test_bulk_label_cards_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while bulk labeling cards")

    monkeypatch.setattr(db, "bulk_label_cards", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/bulk-label",
        json={"card_ids": [1, 2], "add_label_ids": [10]},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to bulk label cards"


# =============================================================================
# Card Filtering Tests
# =============================================================================

def test_filter_cards(client_with_kanban_db):

    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Filter Test Board", "client_id": "board-filter-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Filter Test List", "client_id": "list-filter-1"}
    ).json()

    # Create cards with different priorities
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "High Priority Card", "client_id": "filter-card-1", "priority": "high"}
    )
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Low Priority Card", "client_id": "filter-card-2", "priority": "low"}
    )
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "No Priority Card", "client_id": "filter-card-3"}
    )

    # Filter by priority
    filter_resp = client.get(
        f"/api/v1/kanban/boards/{board['id']}/cards",
        params={"priority": "high"}
    )
    assert filter_resp.status_code == 200
    result = filter_resp.json()
    assert len(result["cards"]) == 1
    assert result["cards"][0]["priority"] == "high"


def test_filter_cards_by_label(client_with_kanban_db):


    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Label Filter Board", "client_id": "board-lblfilter-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Label Filter List", "client_id": "list-lblfilter-1"}
    ).json()

    label = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Important", "color": "yellow"}
    ).json()

    card1 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Labeled Card", "client_id": "lblfilter-card-1"}
    ).json()
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Unlabeled Card", "client_id": "lblfilter-card-2"}
    )

    # Add label to card1
    client.post(f"/api/v1/kanban/cards/{card1['id']}/labels/{label['id']}")

    # Filter by label
    filter_resp = client.get(
        f"/api/v1/kanban/boards/{board['id']}/cards",
        params={"label_ids": str(label["id"])}
    )
    assert filter_resp.status_code == 200
    result = filter_resp.json()
    assert len(result["cards"]) == 1
    assert result["cards"][0]["id"] == card1["id"]


def test_filtered_cards_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while filtering cards")

    monkeypatch.setattr(db, "get_board_cards_filtered", _raise_db_error)

    response = client.get("/api/v1/kanban/boards/999/cards")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch filtered cards"


# =============================================================================
# Card Copy with Checklists Tests
# =============================================================================

def test_copy_card_with_checklists(client_with_kanban_db):

    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Copy Test Board", "client_id": "board-copy-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Copy Test List", "client_id": "list-copy-1"}
    ).json()

    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Card with Checklists", "client_id": "copy-card-1"}
    ).json()

    # Add checklist with items
    checklist = client.post(
        f"/api/v1/kanban/cards/{card['id']}/checklists",
        json={"name": "Tasks"}
    ).json()
    client.post(
        f"/api/v1/kanban/checklists/{checklist['id']}/items",
        json={"name": "Task 1", "checked": True}
    )
    client.post(
        f"/api/v1/kanban/checklists/{checklist['id']}/items",
        json={"name": "Task 2"}
    )

    # Copy with checklists
    copy_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/copy-with-checklists",
        json={
            "target_list_id": lst["id"],
            "new_client_id": "copy-card-with-cl",
            "copy_checklists": True,
            "new_title": "Copied Card"
        }
    )
    assert copy_resp.status_code == 201
    copied = copy_resp.json()
    assert copied["title"] == "Copied Card"

    # Verify checklist was copied
    copied_checklists = client.get(f"/api/v1/kanban/cards/{copied['id']}/checklists")
    assert len(copied_checklists.json()["checklists"]) == 1
    copied_checklist = copied_checklists.json()["checklists"][0]
    assert copied_checklist["name"] == "Tasks"


def test_copy_card_with_checklists_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while copying card with checklists")

    monkeypatch.setattr(db, "copy_card_with_checklists", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/999/copy-with-checklists",
        json={"target_list_id": 1000, "new_client_id": "copy-db-error-1"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to copy card"


# =============================================================================
# Export/Import Tests
# =============================================================================

def test_export_import_board(client_with_kanban_db):

    client, db = client_with_kanban_db

    # Create board with content
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Export Test Board", "client_id": "board-export-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Export Test List", "client_id": "list-export-1"}
    ).json()
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Export Test Card", "client_id": "card-export-1"}
    )

    # Export (POST endpoint with optional parameters)
    export_resp = client.post(f"/api/v1/kanban/boards/{board['id']}/export", json={})
    assert export_resp.status_code == 200
    exported = export_resp.json()
    assert exported["format"] == "tldw_kanban_v1"
    assert exported["board"]["name"] == "Export Test Board"
    assert len(exported["lists"]) == 1
    # Cards are nested inside lists
    assert len(exported["lists"][0].get("cards", [])) == 1

    # Import as new board (use board_name not name_override)
    import_resp = client.post(
        "/api/v1/kanban/boards/import",
        json={"data": exported, "board_name": "Imported Board"}
    )
    assert import_resp.status_code == 201
    imported = import_resp.json()
    assert imported["board"]["name"] == "Imported Board"
    assert imported["board"]["id"] != board["id"]


def test_export_board_get(client_with_kanban_db):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Export Get Board", "client_id": "board-export-get-1"}
    ).json()

    response = client.get(f"/api/v1/kanban/boards/{board['id']}/export")

    assert response.status_code == 200, response.text
    exported = response.json()
    assert exported["format"] == "tldw_kanban_v1"
    assert exported["board"]["name"] == "Export Get Board"


def test_export_board_get_not_found_is_preserved(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_not_found(*args, **kwargs):
        raise NotFoundError("Board 999 not found")

    monkeypatch.setattr(db, "export_board", _raise_not_found)

    response = client.get("/api/v1/kanban/boards/999/export")

    assert response.status_code == 404, response.text
    assert response.json()["detail"] == "Board 999 not found"


def test_export_board_get_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while exporting board")

    monkeypatch.setattr(db, "export_board", _raise_db_error)

    response = client.get("/api/v1/kanban/boards/999/export")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to export board"


def test_export_board_post_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while exporting board")

    monkeypatch.setattr(db, "export_board", _raise_db_error)

    response = client.post("/api/v1/kanban/boards/999/export", json={})

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to export board"


def test_import_board_input_error_is_preserved(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_input_error(*args, **kwargs):
        raise InputError("Import data is invalid")

    monkeypatch.setattr(db, "import_board", _raise_input_error)

    response = client.post("/api/v1/kanban/boards/import", json={"data": {"format": "tldw_kanban_v1"}})

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Import data is invalid"


def test_import_board_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while importing board")

    monkeypatch.setattr(db, "import_board", _raise_db_error)

    response = client.post("/api/v1/kanban/boards/import", json={"data": {"format": "tldw_kanban_v1"}})

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to import board"


# =============================================================================
# Activity Log Tests
# =============================================================================

def test_activity_log(client_with_kanban_db):

    client, db = client_with_kanban_db

    # Create board with some activity
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Activity Test Board", "client_id": "board-activity-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Activity Test List", "client_id": "list-activity-1"}
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Activity Test Card", "client_id": "card-activity-1"}
    ).json()

    # Get board activities
    activities_resp = client.get(f"/api/v1/kanban/boards/{board['id']}/activities")
    assert activities_resp.status_code == 200
    activities = activities_resp.json()
    assert "activities" in activities
    # Should have activities for board creation, list creation, card creation
    assert len(activities["activities"]) >= 3


def test_board_activities_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Board Activity Error", "client_id": "board-activity-db-1"},
    ).json()

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while loading board activities")

    monkeypatch.setattr(db, "get_board_activities", _raise_db_error)

    response = client.get(f"/api/v1/kanban/boards/{board['id']}/activities")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch board activities"


# =============================================================================
# Search API Tests (Phase 4)
# =============================================================================

def test_search_cards_get(client_with_kanban_db):

    """Test GET /api/v1/kanban/search endpoint."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Search Test Board", "client_id": "board-search-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Search Test List", "client_id": "list-search-1"}
    ).json()

    # Create cards with searchable content
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Implement authentication feature", "client_id": "search-card-1"}
    )
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Fix authentication bug", "client_id": "search-card-2"}
    )
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Add logging system", "client_id": "search-card-3"}
    )

    # Search for "authentication"
    search_resp = client.get("/api/v1/kanban/search", params={"q": "authentication"})
    assert search_resp.status_code == 200
    result = search_resp.json()

    assert result["query"] == "authentication"
    assert result["search_mode"] == "fts"
    assert len(result["results"]) == 2
    assert "pagination" in result
    assert result["pagination"]["total"] == 2


def test_search_cards_post(client_with_kanban_db):


    """Test POST /api/v1/kanban/search endpoint."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Search POST Board", "client_id": "board-search-post-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Search POST List", "client_id": "list-search-post-1"}
    ).json()

    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Database migration task", "client_id": "search-post-card-1"}
    )
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "API documentation", "client_id": "search-post-card-2"}
    )

    # Search via POST
    search_resp = client.post(
        "/api/v1/kanban/search",
        json={"query": "database", "limit": 10}
    )
    assert search_resp.status_code == 200
    result = search_resp.json()

    assert result["query"] == "database"
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Database migration task"


def test_search_cards_get_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while searching cards")

    monkeypatch.setattr(db, "search_cards", _raise_db_error)

    response = client.get("/api/v1/kanban/search", params={"q": "database"})

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to search cards"


def test_search_cards_post_db_error_is_sanitized(client_with_kanban_db, monkeypatch):

    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while searching cards")

    monkeypatch.setattr(db, "search_cards", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/search",
        json={"query": "database", "limit": 10},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to search cards"


def test_cards_search_post_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while searching cards")

    monkeypatch.setattr(db, "search_cards", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/search",
        json={"query": "database", "limit": 10},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to search cards"


def test_search_with_board_filter(client_with_kanban_db):


    """Test search filtering by board_id."""
    client, db = client_with_kanban_db

    # Create two boards
    board1_resp = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Board One", "client_id": "board-filter-1"}
    )
    assert board1_resp.status_code == 201, board1_resp.text
    board1 = board1_resp.json()

    board2_resp = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Board Two", "client_id": "board-filter-2"}
    )
    assert board2_resp.status_code == 201, board2_resp.text
    board2 = board2_resp.json()

    lst1_resp = client.post(
        f"/api/v1/kanban/boards/{board1['id']}/lists",
        json={"name": "List One", "client_id": "list-filter-1"}
    )
    assert lst1_resp.status_code == 201, lst1_resp.text
    lst1 = lst1_resp.json()

    lst2_resp = client.post(
        f"/api/v1/kanban/boards/{board2['id']}/lists",
        json={"name": "List Two", "client_id": "list-filter-2"}
    )
    assert lst2_resp.status_code == 201, lst2_resp.text
    lst2 = lst2_resp.json()

    # Create cards with same searchable term in both boards
    card1_resp = client.post(
        f"/api/v1/kanban/lists/{lst1['id']}/cards",
        json={"title": "Testing feature in board one", "client_id": "filter-card-1"}
    )
    assert card1_resp.status_code == 201, card1_resp.text

    card2_resp = client.post(
        f"/api/v1/kanban/lists/{lst2['id']}/cards",
        json={"title": "Testing feature in board two", "client_id": "filter-card-2"}
    )
    assert card2_resp.status_code == 201, card2_resp.text

    # Search with board filter
    search_resp = client.get(
        "/api/v1/kanban/search",
        params={"q": "testing", "board_id": board1["id"]}
    )
    assert search_resp.status_code == 200
    result = search_resp.json()

    assert result["pagination"]["total"] == 1
    assert result["results"][0]["board_id"] == board1["id"]


def test_search_with_priority_filter(client_with_kanban_db):


    """Test search filtering by priority."""
    client, db = client_with_kanban_db

    # Setup
    board_resp = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Priority Search Board", "client_id": "board-priority-search-1"}
    )
    assert board_resp.status_code == 201, board_resp.text
    board = board_resp.json()

    lst_resp = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Priority Search List", "client_id": "list-priority-search-1"}
    )
    assert lst_resp.status_code == 201, lst_resp.text
    lst = lst_resp.json()

    high_card_resp = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Critical bug fix", "client_id": "priority-card-1", "priority": "high"}
    )
    assert high_card_resp.status_code == 201, high_card_resp.text

    low_card_resp = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Minor bug fix", "client_id": "priority-card-2", "priority": "low"}
    )
    assert low_card_resp.status_code == 201, low_card_resp.text

    # Search with priority filter
    search_resp = client.get(
        "/api/v1/kanban/search",
        params={"q": "bug", "priority": "high"}
    )
    assert search_resp.status_code == 200
    result = search_resp.json()

    assert result["pagination"]["total"] == 1
    assert result["results"][0]["priority"] == "high"


def test_search_with_label_filter(client_with_kanban_db):


    """Test search filtering by label_ids."""
    client, db = client_with_kanban_db

    # Setup
    board_resp = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Label Search Board", "client_id": "board-label-search-1"}
    )
    assert board_resp.status_code == 201, board_resp.text
    board = board_resp.json()

    lst_resp = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Label Search List", "client_id": "list-label-search-1"}
    )
    assert lst_resp.status_code == 201, lst_resp.text
    lst = lst_resp.json()

    label_resp = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Bug", "color": "red"}
    )
    assert label_resp.status_code == 201, label_resp.text
    label = label_resp.json()

    card1_resp = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Feature request alpha", "client_id": "label-search-card-1"}
    )
    assert card1_resp.status_code == 201, card1_resp.text
    card1 = card1_resp.json()

    card2_resp = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Feature request beta", "client_id": "label-search-card-2"}
    )
    assert card2_resp.status_code == 201, card2_resp.text

    # Assign label to card1
    assign_resp = client.post(f"/api/v1/kanban/cards/{card1['id']}/labels/{label['id']}")
    assert assign_resp.status_code == 200, assign_resp.text

    # Search with label filter
    search_resp = client.get(
        "/api/v1/kanban/search",
        params={"q": "feature", "label_ids": str(label["id"])}
    )
    assert search_resp.status_code == 200
    result = search_resp.json()

    assert result["pagination"]["total"] == 1
    assert result["results"][0]["id"] == card1["id"]


def test_search_includes_enriched_data(client_with_kanban_db):


    """Test that search results include board_name, list_name, and labels."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Enriched Data Board", "client_id": "board-enriched-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Enriched Data List", "client_id": "list-enriched-1"}
    ).json()

    label = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Important", "color": "yellow"}
    ).json()

    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Unique enrichment card", "client_id": "enriched-card-1"}
    ).json()

    client.post(f"/api/v1/kanban/cards/{card['id']}/labels/{label['id']}")

    # Search and verify enriched data
    search_resp = client.get("/api/v1/kanban/search", params={"q": "enrichment"})
    assert search_resp.status_code == 200
    result = search_resp.json()

    assert len(result["results"]) == 1
    card_result = result["results"][0]
    assert card_result["board_name"] == "Enriched Data Board"
    assert card_result["list_name"] == "Enriched Data List"
    assert len(card_result["labels"]) == 1
    assert card_result["labels"][0]["name"] == "Important"


def test_search_pagination(client_with_kanban_db):


    """Test search pagination."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Pagination Board", "client_id": "board-pagination-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Pagination List", "client_id": "list-pagination-1"}
    ).json()

    # Create multiple cards
    for i in range(5):
        client.post(
            f"/api/v1/kanban/lists/{lst['id']}/cards",
            json={"title": f"Paginated item {i}", "client_id": f"pagination-card-{i}"}
        )

    # Search with pagination
    search_resp = client.get(
        "/api/v1/kanban/search",
        params={"q": "paginated", "limit": 2, "offset": 0}
    )
    assert search_resp.status_code == 200
    result = search_resp.json()

    assert len(result["results"]) == 2
    assert result["pagination"]["total"] == 5
    assert result["pagination"]["has_more"] is True

    # Get page 2
    search_resp2 = client.get(
        "/api/v1/kanban/search",
        params={"q": "paginated", "limit": 2, "offset": 2}
    )
    assert search_resp2.status_code == 200
    result2 = search_resp2.json()

    assert len(result2["results"]) == 2
    assert result2["pagination"]["offset"] == 2


def test_search_empty_query_rejected(client_with_kanban_db):


    """Test that empty search query is rejected."""
    client, db = client_with_kanban_db

    # Empty query should be rejected by FastAPI validation
    search_resp = client.get("/api/v1/kanban/search", params={"q": ""})
    assert search_resp.status_code == 422  # Validation error


def test_search_no_results(client_with_kanban_db):


    """Test search with no matching results."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Empty Search Board", "client_id": "board-empty-search-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Empty Search List", "client_id": "list-empty-search-1"}
    ).json()

    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Some card", "client_id": "empty-search-card-1"}
    )

    # Search for non-existent term
    search_resp = client.get("/api/v1/kanban/search", params={"q": "nonexistentterm12345"})
    assert search_resp.status_code == 200
    result = search_resp.json()

    assert len(result["results"]) == 0
    assert result["pagination"]["total"] == 0


def test_search_archived_cards(client_with_kanban_db):


    """Test search with include_archived parameter."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Archive Search Board", "client_id": "board-archive-search-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Archive Search List", "client_id": "list-archive-search-1"}
    ).json()

    card1 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Active searchable card", "client_id": "archive-search-card-1"}
    ).json()
    card2 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Archived searchable card", "client_id": "archive-search-card-2"}
    ).json()

    # Archive card2
    client.post(f"/api/v1/kanban/cards/{card2['id']}/archive")

    # Search without include_archived (default)
    search_resp = client.get("/api/v1/kanban/search", params={"q": "searchable"})
    assert search_resp.status_code == 200
    result = search_resp.json()
    assert result["pagination"]["total"] == 1

    # Search with include_archived
    search_resp2 = client.get(
        "/api/v1/kanban/search",
        params={"q": "searchable", "include_archived": "true"}
    )
    assert search_resp2.status_code == 200
    result2 = search_resp2.json()
    assert result2["pagination"]["total"] == 2


def test_search_status(client_with_kanban_db):


    """Test GET /api/v1/kanban/search/status endpoint."""
    client, db = client_with_kanban_db

    status_resp = client.get("/api/v1/kanban/search/status")
    assert status_resp.status_code == 200
    result = status_resp.json()

    # FTS is always available
    assert result["fts_available"] is True
    assert result["default_mode"] == "fts"
    assert "fts" in result["supported_modes"]
    assert "vector" in result["supported_modes"]
    assert "hybrid" in result["supported_modes"]

    # Check scoring weights are exposed
    assert "scoring_weights" in result
    weights = result["scoring_weights"]
    assert "fts_weight" in weights
    assert "vector_weight" in weights
    assert "vector_only_weight" in weights
    # Default weights
    assert weights["fts_weight"] == 0.6
    assert weights["vector_weight"] == 0.4
    assert weights["vector_only_weight"] == 0.3


def test_search_invalid_label_ids_format(client_with_kanban_db):


    """Test that invalid label_ids format returns 400 error."""
    client, db = client_with_kanban_db

    # Invalid format: non-numeric
    search_resp = client.get(
        "/api/v1/kanban/search",
        params={"q": "test", "label_ids": "abc,def"}
    )
    assert search_resp.status_code == 400
    assert "Invalid label_ids format" in search_resp.json()["detail"]


def test_search_with_search_mode_parameter(client_with_kanban_db):


    """Test search_mode parameter (fts, vector, hybrid)."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Search Mode Board", "client_id": "board-search-mode-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Search Mode List", "client_id": "list-search-mode-1"}
    ).json()

    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Test mode card", "client_id": "search-mode-card-1"}
    )

    # Test FTS mode (default)
    fts_resp = client.get(
        "/api/v1/kanban/search",
        params={"q": "mode", "search_mode": "fts"}
    )
    assert fts_resp.status_code == 200
    assert fts_resp.json()["search_mode"] == "fts"

    # Test vector mode (falls back to FTS when ChromaDB unavailable)
    vector_resp = client.get(
        "/api/v1/kanban/search",
        params={"q": "mode", "search_mode": "vector"}
    )
    assert vector_resp.status_code == 200
    # Vector mode falls back to fts when ChromaDB is unavailable
    assert vector_resp.json()["search_mode"] == "fts"

    # Test hybrid mode (falls back to FTS when ChromaDB unavailable)
    hybrid_resp = client.get(
        "/api/v1/kanban/search",
        params={"q": "mode", "search_mode": "hybrid"}
    )
    assert hybrid_resp.status_code == 200
    assert hybrid_resp.json()["search_mode"] == "fts"

    # Test invalid mode (should default to fts)
    invalid_resp = client.get(
        "/api/v1/kanban/search",
        params={"q": "mode", "search_mode": "invalid_mode"}
    )
    assert invalid_resp.status_code == 200
    assert invalid_resp.json()["search_mode"] == "fts"


def test_search_with_special_characters(client_with_kanban_db):


    """Test search handles special characters safely without SQL/FTS5 syntax errors."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Special Chars Board", "client_id": "board-special-chars-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Special Chars List", "client_id": "list-special-chars-1"}
    ).json()

    # Create cards with special characters in titles
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Card with 100% complete", "client_id": "special-card-1"}
    )
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Card with user_name field", "client_id": "special-card-2"}
    )
    client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Regular card", "client_id": "special-card-3"}
    )

    # Search for % - should not cause SQL/FTS5 syntax error
    # Note: FTS5 doesn't index special characters, so searching for just "%" returns no results
    percent_resp = client.get("/api/v1/kanban/search", params={"q": "%"})
    assert percent_resp.status_code == 200
    result = percent_resp.json()
    # Verify no error occurred (0 results is expected as % is not a word)
    assert "results" in result

    # Search for _ - should not cause SQL/FTS5 syntax error
    underscore_resp = client.get("/api/v1/kanban/search", params={"q": "_"})
    assert underscore_resp.status_code == 200

    # Search for FTS5 special characters (operators) - should not cause syntax errors
    for special_char in ["*", "(", ")", ":", "^", "+", "-", '"']:
        resp = client.get("/api/v1/kanban/search", params={"q": special_char})
        assert resp.status_code == 200, f"Search for '{special_char}' failed with status {resp.status_code}"

    # Search for a word containing special characters should still work
    complete_resp = client.get("/api/v1/kanban/search", params={"q": "complete"})
    assert complete_resp.status_code == 200
    result = complete_resp.json()
    assert result["pagination"]["total"] == 1
    assert "100%" in result["results"][0]["title"]

    # Search for user_name (which contains underscore) by searching for the whole term
    username_resp = client.get("/api/v1/kanban/search", params={"q": "user_name"})
    assert username_resp.status_code == 200
    result = username_resp.json()
    assert result["pagination"]["total"] == 1
    assert "user_name" in result["results"][0]["title"]


def test_search_post_with_all_parameters(client_with_kanban_db):


    """Test POST search with all available parameters."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Full POST Search Board", "client_id": "board-full-post-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Full POST Search List", "client_id": "list-full-post-1"}
    ).json()

    label = client.post(
        f"/api/v1/kanban/boards/{board['id']}/labels",
        json={"name": "Feature", "color": "blue"}
    ).json()

    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={
            "title": "Complete feature implementation",
            "client_id": "full-post-card-1",
            "priority": "high"
        }
    ).json()

    # Add label to card
    client.post(f"/api/v1/kanban/cards/{card['id']}/labels/{label['id']}")

    # Search with all parameters via POST
    search_resp = client.post(
        "/api/v1/kanban/search",
        json={
            "query": "feature",
            "board_id": board["id"],
            "label_ids": [label["id"]],
            "priority": "high",
            "include_archived": False,
            "search_mode": "fts",
            "limit": 10,
            "offset": 0
        }
    )
    assert search_resp.status_code == 200
    result = search_resp.json()

    assert result["pagination"]["total"] == 1
    assert result["results"][0]["title"] == "Complete feature implementation"
    assert result["results"][0]["priority"] == "high"


# =============================================================================
# Card Links API Tests (Phase 5: Content Integration)
# =============================================================================

def test_card_link_crud(client_with_kanban_db):

    """Test basic CRUD operations for card links."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Link Test Board", "client_id": "board-link-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Link Test List", "client_id": "list-link-1"}
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Card with Links", "client_id": "card-link-1"}
    ).json()

    # Create a link to a media item
    create_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links",
        json={"linked_type": "media", "linked_id": "media-123"}
    )
    assert create_resp.status_code == 201
    link = create_resp.json()
    assert link["card_id"] == card["id"]
    assert link["linked_type"] == "media"
    assert link["linked_id"] == "media-123"
    link_id = link["id"]

    # Create a link to a note
    note_link_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links",
        json={"linked_type": "note", "linked_id": "note-456"}
    )
    assert note_link_resp.status_code == 201

    # List links
    list_resp = client.get(f"/api/v1/kanban/cards/{card['id']}/links")
    assert list_resp.status_code == 200
    links = list_resp.json()["links"]
    assert len(links) == 2

    # Filter by type
    media_resp = client.get(
        f"/api/v1/kanban/cards/{card['id']}/links",
        params={"linked_type": "media"}
    )
    assert media_resp.status_code == 200
    assert len(media_resp.json()["links"]) == 1

    # Get link counts
    counts_resp = client.get(f"/api/v1/kanban/cards/{card['id']}/links/counts")
    assert counts_resp.status_code == 200
    counts = counts_resp.json()
    assert counts["media"] == 1
    assert counts["note"] == 1

    # Delete link by type/id
    delete_resp = client.delete(
        f"/api/v1/kanban/cards/{card['id']}/links/media/media-123"
    )
    assert delete_resp.status_code == 200

    # Delete link by card + link ID
    remaining_links = client.get(f"/api/v1/kanban/cards/{card['id']}/links").json()["links"]
    assert len(remaining_links) == 1
    delete_by_card_resp = client.delete(
        f"/api/v1/kanban/cards/{card['id']}/links/{remaining_links[0]['id']}"
    )
    assert delete_by_card_resp.status_code == 200

    # Recreate and delete link by ID
    recreate_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links",
        json={"linked_type": "note", "linked_id": "note-789"}
    )
    assert recreate_resp.status_code == 201
    delete_by_id_resp = client.delete(f"/api/v1/kanban/links/{recreate_resp.json()['id']}")
    assert delete_by_id_resp.status_code == 200

    # Verify all links removed
    final_links = client.get(f"/api/v1/kanban/cards/{card['id']}/links").json()["links"]
    assert len(final_links) == 0


def test_card_link_duplicate_rejected(client_with_kanban_db):


    """Test that duplicate links are rejected with 409 Conflict."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Duplicate Link Board", "client_id": "board-dup-link-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Duplicate Link List", "client_id": "list-dup-link-1"}
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Duplicate Test Card", "client_id": "card-dup-link-1"}
    ).json()

    # Create first link
    client.post(
        f"/api/v1/kanban/cards/{card['id']}/links",
        json={"linked_type": "media", "linked_id": "media-dup"}
    )

    # Try to create duplicate link
    dup_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links",
        json={"linked_type": "media", "linked_id": "media-dup"}
    )
    assert dup_resp.status_code == 409


def test_card_link_invalid_type_rejected(client_with_kanban_db):


    """Test that invalid linked_type is rejected with 422."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Invalid Link Type Board", "client_id": "board-inv-type-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Invalid Link Type List", "client_id": "list-inv-type-1"}
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Invalid Type Card", "client_id": "card-inv-type-1"}
    ).json()

    # Try to create link with invalid type
    invalid_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links",
        json={"linked_type": "invalid", "linked_id": "some-id"}
    )
    assert invalid_resp.status_code == 422


def test_card_link_create_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    """Create-link DB failures should not leak raw KanbanDBError details."""
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while creating link")

    monkeypatch.setattr(db, "add_card_link", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/999/links",
        json={"linked_type": "media", "linked_id": "media-db-error"},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to add card link"


def test_card_link_list_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    """List-link DB failures should return a stable endpoint detail."""
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while listing links")

    monkeypatch.setattr(db, "get_card_links", _raise_db_error)

    response = client.get("/api/v1/kanban/cards/999/links")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch card links"


def test_card_link_counts_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    """Count-link DB failures should return a stable endpoint detail."""
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while counting links")

    monkeypatch.setattr(db, "get_linked_content_counts", _raise_db_error)

    response = client.get("/api/v1/kanban/cards/999/links/counts")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch card link counts"


def test_remove_card_link_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    """Delete-by-type/id DB failures should not leak raw KanbanDBError details."""
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while removing card link")

    monkeypatch.setattr(db, "remove_card_link", _raise_db_error)

    response = client.delete("/api/v1/kanban/cards/999/links/media/media-db-error")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to remove card link"


def test_remove_card_link_by_id_for_card_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    """Card-scoped delete-by-id DB failures should not leak raw KanbanDBError details."""
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while removing card link by id for card")

    monkeypatch.setattr(db, "remove_card_link_by_id_for_card", _raise_db_error)

    response = client.delete("/api/v1/kanban/cards/999/links/123")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to remove card link"


def test_remove_card_link_by_id_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    """Global delete-by-id DB failures should not leak raw KanbanDBError details."""
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while removing card link by id")

    monkeypatch.setattr(db, "remove_card_link_by_id", _raise_db_error)

    response = client.delete("/api/v1/kanban/links/123")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to remove card link"


def test_bulk_card_links(client_with_kanban_db):


    """Test bulk add and remove card links."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Bulk Links Board", "client_id": "board-bulk-links-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Bulk Links List", "client_id": "list-bulk-links-1"}
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Bulk Links Card", "client_id": "card-bulk-links-1"}
    ).json()

    # Bulk add links
    bulk_add_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links/bulk-add",
        json={
            "links": [
                {"linked_type": "media", "linked_id": "media-1"},
                {"linked_type": "media", "linked_id": "media-2"},
                {"linked_type": "note", "linked_id": "note-1"}
            ]
        }
    )
    assert bulk_add_resp.status_code == 200
    result = bulk_add_resp.json()
    assert result["added_count"] == 3
    assert result["skipped_count"] == 0
    assert len(result["links"]) == 3

    # Bulk add with duplicates (should skip)
    bulk_add_dup_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links/bulk-add",
        json={
            "links": [
                {"linked_type": "media", "linked_id": "media-1"},  # duplicate
                {"linked_type": "media", "linked_id": "media-3"}   # new
            ]
        }
    )
    assert bulk_add_dup_resp.status_code == 200
    dup_result = bulk_add_dup_resp.json()
    assert dup_result["added_count"] == 1
    assert dup_result["skipped_count"] == 1

    # Verify total links
    links_resp = client.get(f"/api/v1/kanban/cards/{card['id']}/links")
    assert len(links_resp.json()["links"]) == 4

    # Bulk remove links
    bulk_remove_resp = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links/bulk-remove",
        json={
            "links": [
                {"linked_type": "media", "linked_id": "media-1"},
                {"linked_type": "media", "linked_id": "media-2"},
                {"linked_type": "note", "linked_id": "nonexistent"}  # doesn't exist
            ]
        }
    )
    assert bulk_remove_resp.status_code == 200
    remove_result = bulk_remove_resp.json()
    assert remove_result["removed_count"] == 2

    # Verify remaining links
    final_links = client.get(f"/api/v1/kanban/cards/{card['id']}/links")
    assert len(final_links.json()["links"]) == 2


def test_bulk_add_card_links_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    """Bulk-add DB failures should not leak raw KanbanDBError details."""
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while bulk adding links")

    monkeypatch.setattr(db, "bulk_add_card_links", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/999/links/bulk-add",
        json={"links": [{"linked_type": "media", "linked_id": "media-db-error"}]},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to bulk add card links"


def test_bulk_remove_card_links_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    """Bulk-remove DB failures should not leak raw KanbanDBError details."""
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while bulk removing links")

    monkeypatch.setattr(db, "bulk_remove_card_links", _raise_db_error)

    response = client.post(
        "/api/v1/kanban/cards/999/links/bulk-remove",
        json={"links": [{"linked_type": "media", "linked_id": "media-db-error"}]},
    )

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to bulk remove card links"


def test_bidirectional_lookup(client_with_kanban_db):


    """Test finding cards that link to a specific content item."""
    client, db = client_with_kanban_db

    # Setup - create two boards with cards linking to same content
    board1 = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Lookup Board 1", "client_id": "board-lookup-1"}
    ).json()
    board2 = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Lookup Board 2", "client_id": "board-lookup-2"}
    ).json()

    lst1 = client.post(
        f"/api/v1/kanban/boards/{board1['id']}/lists",
        json={"name": "Lookup List 1", "client_id": "list-lookup-1"}
    ).json()
    lst2 = client.post(
        f"/api/v1/kanban/boards/{board2['id']}/lists",
        json={"name": "Lookup List 2", "client_id": "list-lookup-2"}
    ).json()

    card1 = client.post(
        f"/api/v1/kanban/lists/{lst1['id']}/cards",
        json={"title": "Card linking to shared media", "client_id": "lookup-card-1"}
    ).json()
    card2 = client.post(
        f"/api/v1/kanban/lists/{lst2['id']}/cards",
        json={"title": "Another card linking to shared media", "client_id": "lookup-card-2"}
    ).json()

    # Both cards link to the same media item
    shared_media_id = "shared-media-123"
    client.post(
        f"/api/v1/kanban/cards/{card1['id']}/links",
        json={"linked_type": "media", "linked_id": shared_media_id}
    )
    client.post(
        f"/api/v1/kanban/cards/{card2['id']}/links",
        json={"linked_type": "media", "linked_id": shared_media_id}
    )

    # Bidirectional lookup - find all cards linking to this media
    lookup_resp = client.get(f"/api/v1/kanban/linked/media/{shared_media_id}/cards")
    assert lookup_resp.status_code == 200
    result = lookup_resp.json()

    assert result["linked_type"] == "media"
    assert result["linked_id"] == shared_media_id
    assert len(result["cards"]) == 2

    # Verify card details are included
    card_ids = [c["id"] for c in result["cards"]]
    assert card1["id"] in card_ids
    assert card2["id"] in card_ids

    # Verify board/list context is included
    for card_data in result["cards"]:
        assert "board_id" in card_data
        assert "board_name" in card_data
        assert "list_id" in card_data
        assert "list_name" in card_data
        assert "link_id" in card_data
        assert "linked_at" in card_data


def test_bidirectional_lookup_db_error_is_sanitized(client_with_kanban_db, monkeypatch):
    """Lookup DB failures should not expose raw storage errors."""
    client, db = client_with_kanban_db

    def _raise_db_error(*args, **kwargs):
        raise KanbanDBError("sqlite driver exploded while loading linked cards")

    monkeypatch.setattr(db, "get_cards_by_linked_content", _raise_db_error)

    response = client.get("/api/v1/kanban/linked/media/media-db-error/cards")

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch linked cards"


def test_bidirectional_lookup_respects_archived(client_with_kanban_db):


    """Test that bidirectional lookup respects include_archived flag."""
    client, db = client_with_kanban_db

    # Setup
    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Archived Lookup Board", "client_id": "board-archived-lookup-1"}
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Archived Lookup List", "client_id": "list-archived-lookup-1"}
    ).json()

    card1 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Active Card", "client_id": "archived-lookup-card-1"}
    ).json()
    card2 = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Archived Card", "client_id": "archived-lookup-card-2"}
    ).json()

    # Both cards link to same content
    content_id = "archived-test-content"
    client.post(
        f"/api/v1/kanban/cards/{card1['id']}/links",
        json={"linked_type": "note", "linked_id": content_id}
    )
    client.post(
        f"/api/v1/kanban/cards/{card2['id']}/links",
        json={"linked_type": "note", "linked_id": content_id}
    )

    # Archive card2
    client.post(f"/api/v1/kanban/cards/{card2['id']}/archive")

    # Lookup without include_archived (default)
    lookup_resp = client.get(f"/api/v1/kanban/linked/note/{content_id}/cards")
    assert lookup_resp.status_code == 200
    assert len(lookup_resp.json()["cards"]) == 1

    # Lookup with include_archived
    lookup_archived_resp = client.get(
        f"/api/v1/kanban/linked/note/{content_id}/cards",
        params={"include_archived": "true"}
    )
    assert lookup_archived_resp.status_code == 200
    assert len(lookup_archived_resp.json()["cards"]) == 2


def test_bidirectional_lookup_invalid_type(client_with_kanban_db):


    """Test that invalid linked_type in lookup returns 400."""
    client, db = client_with_kanban_db

    lookup_resp = client.get("/api/v1/kanban/linked/invalid/some-id/cards")
    assert lookup_resp.status_code == 400


def test_bidirectional_lookup_no_cards(client_with_kanban_db):


    """Test bidirectional lookup returns empty list when no cards link to content."""
    client, db = client_with_kanban_db

    lookup_resp = client.get("/api/v1/kanban/linked/media/nonexistent-media/cards")
    assert lookup_resp.status_code == 200
    result = lookup_resp.json()
    assert result["linked_type"] == "media"
    assert result["linked_id"] == "nonexistent-media"
    assert len(result["cards"]) == 0


def test_card_link_target_validation_when_content_db_available(client_with_kanban_db):
    """
    Link target validation should be strict only when Media/notes tables exist.

    This test provisions minimal Media and notes tables in the same user database
    directory and verifies valid links succeed while missing targets are rejected.
    """
    client, db = client_with_kanban_db

    board = client.post(
        "/api/v1/kanban/boards",
        json={"name": "Link Validation Board", "client_id": "board-link-validation-1"},
    ).json()
    lst = client.post(
        f"/api/v1/kanban/boards/{board['id']}/lists",
        json={"name": "Link Validation List", "client_id": "list-link-validation-1"},
    ).json()
    card = client.post(
        f"/api/v1/kanban/lists/{lst['id']}/cards",
        json={"title": "Link Validation Card", "client_id": "card-link-validation-1"},
    ).json()

    media_db_path = DatabasePaths.get_media_db_path(db.user_id)
    with sqlite3.connect(str(media_db_path)) as media_conn:
        media_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS Media (
                id INTEGER PRIMARY KEY,
                uuid TEXT UNIQUE,
                client_id TEXT,
                deleted INTEGER DEFAULT 0
            )
            """
        )
        media_conn.execute(
            "INSERT INTO Media (id, uuid, client_id, deleted) VALUES (?, ?, ?, 0)",
            (101, "media-uuid-101", "media-client-101"),
        )
        media_conn.commit()

    notes_db_path = DatabasePaths.get_chacha_db_path(db.user_id)
    with sqlite3.connect(str(notes_db_path)) as notes_conn:
        notes_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                client_id TEXT NOT NULL DEFAULT 'unknown',
                deleted INTEGER DEFAULT 0
            )
            """
        )
        notes_conn.execute(
            "INSERT INTO notes (id, client_id, deleted) VALUES (?, ?, 0)",
            ("note-abc", "note-client-abc"),
        )
        notes_conn.commit()

    ok_media = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links",
        json={"linked_type": "media", "linked_id": "101"},
    )
    assert ok_media.status_code == 201, ok_media.text

    missing_media = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links",
        json={"linked_type": "media", "linked_id": "999999"},
    )
    assert missing_media.status_code == 404

    ok_note = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links",
        json={"linked_type": "note", "linked_id": "note-abc"},
    )
    assert ok_note.status_code == 201, ok_note.text

    missing_note = client.post(
        f"/api/v1/kanban/cards/{card['id']}/links",
        json={"linked_type": "note", "linked_id": "note-missing"},
    )
    assert missing_note.status_code == 404
