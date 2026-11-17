"""Pytest fixtures for Kanban database tests."""
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Kanban_DB import KanbanDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


@pytest.fixture
def temp_db_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Create a temporary base directory for Kanban database tests."""
    base_dir = tmp_path / "user_dbs"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    return base_dir


@pytest.fixture
def temp_db_path(temp_db_dir: Path) -> Path:
    """Create a temporary database file path."""
    # Ensure USER_DB_BASE_DIR is configured via temp_db_dir fixture.
    _ = temp_db_dir
    return DatabasePaths.get_kanban_db_path("1")


@pytest.fixture
def kanban_db(temp_db_path: Path) -> KanbanDB:
    """Create a KanbanDB instance with a temporary database."""
    return KanbanDB(db_path=str(temp_db_path), user_id="1")


@pytest.fixture
def kanban_db_user2(temp_db_dir: Path) -> KanbanDB:
    """Create a second KanbanDB instance for testing user isolation."""
    # temp_db_dir fixture sets USER_DB_BASE_DIR for this test.
    _ = temp_db_dir
    db_path = DatabasePaths.get_kanban_db_path("2")
    return KanbanDB(db_path=str(db_path), user_id="2")


@pytest.fixture
def sample_board(kanban_db: KanbanDB) -> dict[str, Any]:
    """Create a sample board for testing."""
    return kanban_db.create_board(
        name="Test Board",
        client_id="board-client-1",
        description="A test board"
    )


@pytest.fixture
def sample_list(kanban_db: KanbanDB, sample_board: dict[str, Any]) -> dict[str, Any]:
    """Create a sample list for testing."""
    return kanban_db.create_list(
        board_id=sample_board["id"],
        name="Test List",
        client_id="list-client-1"
    )


@pytest.fixture
def sample_card(kanban_db: KanbanDB, sample_list: dict[str, Any]) -> dict[str, Any]:
    """Create a sample card for testing."""
    return kanban_db.create_card(
        list_id=sample_list["id"],
        title="Test Card",
        client_id="card-client-1",
        description="A test card"
    )


@pytest.fixture
def draft():
    return {
        "format": "tldw_kanban_draft_v1",
        "client_id": "s03-controlled-generated-draft",
        "board": {"name": "Rowan Tasks", "description": "Controlled source"},
        "lists": [
            {
                "name": "To Do",
                "cards": [
                    {"title": "Calibrate Selene", "description": "Telescope named Selene"},
                    {"title": "Prepare Friday tours", "description": "Tours begin at 18:00"},
                ],
            },
            {"name": "Done", "cards": [{"title": "Record 2019 opening", "description": "Rowan Observatory"}]},
        ],
    }
