# tldw_Server_API/tests/kanban/test_kanban_db.py
"""
Unit tests for Kanban_DB.py database operations.

Tests cover:
- Board CRUD operations
- List CRUD operations
- Card CRUD operations
- Position management and reordering
- Archive/delete/restore operations
- User isolation
- Search functionality
- Error handling
"""
from typing import Dict, Any
from datetime import datetime, timezone, timedelta
import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    KanbanDB,
    KanbanDBError,
    InputError,
    ConflictError,
    NotFoundError,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management import Kanban_DB as kanban_db_module


class TestDbPathValidation:
    """Tests for KanbanDB path validation."""

    def test_rejects_external_db_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path):
        monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
        external_path = tmp_path / "outside" / "Kanban.db"

        with pytest.raises(InputError):
            KanbanDB(db_path=str(external_path), user_id="1")

    def test_memory_db_initializes_schema(self):

        db = KanbanDB(db_path=":memory:", user_id="mem-user")
        board = db.create_board(name="Memory Board", client_id="mem-1")
        fetched = db.get_board(board["id"])

        assert fetched is not None
        assert fetched["id"] == board["id"]
        db.close()

    def test_configure_connection_raises_on_pragma_failure(self):

        class _StubConn:
            def __init__(self, fail_on: str):
                self.fail_on = fail_on
                self.row_factory = None

            def execute(self, statement: str):
                if self.fail_on in statement:
                    raise sqlite3.OperationalError("bad pragma")
                return None

        db = KanbanDB.__new__(KanbanDB)
        conn = _StubConn("PRAGMA foreign_keys=ON")

        with pytest.raises(KanbanDBError, match="foreign_keys=ON"):
            kanban_db_module.KanbanDB._configure_connection(db, conn, enable_wal=False)

    def test_get_vector_search_uses_active_db_root_for_user_db_base_dir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ):
        base_dir = tmp_path / "isolated_user_dbs"
        monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
        db_path = DatabasePaths.get_kanban_db_path("1")
        captured: dict[str, Any] = {}
        stale_base_dir = tmp_path / "stale_user_dbs"
        original_settings_get = kanban_db_module.settings.get

        def _fake_settings_get(key: str):
            if key == "USER_DB_BASE_DIR":
                return str(stale_base_dir)
            return original_settings_get(key)

        def _fake_create_vector_search(user_id: str, embedding_config: dict[str, Any]):
            captured["user_id"] = user_id
            captured["embedding_config"] = dict(embedding_config)
            return None

        monkeypatch.setattr(kanban_db_module.settings, "get", _fake_settings_get)
        db = KanbanDB(db_path=str(db_path), user_id="1", vector_search_factory=_fake_create_vector_search)
        try:
            db.get_vector_search()
        finally:
            db.close()

        expected_base_dir = str(db_path.resolve().parent.parent)
        assert captured["user_id"] == "test_user_1"
        assert captured["embedding_config"]["USER_DB_BASE_DIR"] == expected_base_dir


# =============================================================================
# Board Tests
# =============================================================================

class TestBoardOperations:
    """Tests for board CRUD operations."""

    def test_create_board(self, kanban_db: KanbanDB):
        """Test creating a new board."""
        board = kanban_db.create_board(
            name="My Board",
            client_id="board-1",
            description="Test description"
        )

        assert board["id"] is not None
        assert board["uuid"] is not None
        assert board["name"] == "My Board"
        assert board["description"] == "Test description"
        assert board["client_id"] == "board-1"
        assert board["user_id"] == "test_user_1"
        assert board["archived"] is False
        assert board["deleted"] is False
        assert board["version"] == 1

    def test_create_board_minimal(self, kanban_db: KanbanDB):
        """Test creating a board with only required fields."""
        board = kanban_db.create_board(
            name="Minimal Board",
            client_id="board-minimal"
        )

        assert board["name"] == "Minimal Board"
        assert board["description"] is None
        assert board["metadata"] is None

    def test_create_board_with_metadata(self, kanban_db: KanbanDB):
        """Test creating a board with metadata."""
        metadata = {"color": "blue", "icon": "star"}
        board = kanban_db.create_board(
            name="Board with Meta",
            client_id="board-meta",
            metadata=metadata
        )

        assert board["metadata"] == metadata

    def test_create_board_duplicate_client_id(self, kanban_db: KanbanDB):
        """Test that duplicate client_id raises ConflictError."""
        kanban_db.create_board(name="Board 1", client_id="dup-id")

        with pytest.raises(ConflictError) as exc_info:
            kanban_db.create_board(name="Board 2", client_id="dup-id")

        assert "already exists" in str(exc_info.value)

    def test_create_board_empty_name(self, kanban_db: KanbanDB):
        """Test that empty name raises InputError."""
        with pytest.raises(InputError):
            kanban_db.create_board(name="", client_id="board-empty")

        with pytest.raises(InputError):
            kanban_db.create_board(name="   ", client_id="board-whitespace")

    def test_get_board(self, kanban_db: KanbanDB, sample_board: dict):
        """Test getting a board by ID."""
        board = kanban_db.get_board(sample_board["id"])

        assert board is not None
        assert board["id"] == sample_board["id"]
        assert board["name"] == sample_board["name"]

    def test_get_board_not_found(self, kanban_db: KanbanDB):
        """Test getting a non-existent board returns None."""
        board = kanban_db.get_board(99999)
        assert board is None

    def test_list_boards(self, kanban_db: KanbanDB):
        """Test listing boards."""
        kanban_db.create_board(name="Board 1", client_id="b1")
        kanban_db.create_board(name="Board 2", client_id="b2")
        kanban_db.create_board(name="Board 3", client_id="b3")

        boards, total = kanban_db.list_boards()

        assert total == 3
        assert len(boards) == 3

    def test_list_boards_pagination(self, kanban_db: KanbanDB):
        """Test listing boards with pagination."""
        for i in range(10):
            kanban_db.create_board(name=f"Board {i}", client_id=f"b{i}")

        boards, total = kanban_db.list_boards(limit=3, offset=0)
        assert total == 10
        assert len(boards) == 3

        boards, total = kanban_db.list_boards(limit=3, offset=3)
        assert len(boards) == 3

    def test_update_board(self, kanban_db: KanbanDB, sample_board: dict):
        """Test updating a board."""
        updated = kanban_db.update_board(
            board_id=sample_board["id"],
            name="Updated Name",
            description="Updated description"
        )

        assert updated["name"] == "Updated Name"
        assert updated["description"] == "Updated description"
        assert updated["version"] == sample_board["version"] + 1

    def test_update_board_optimistic_locking(self, kanban_db: KanbanDB, sample_board: dict):
        """Test optimistic locking on update."""
        # Update with correct version
        updated = kanban_db.update_board(
            board_id=sample_board["id"],
            name="Updated",
            expected_version=sample_board["version"]
        )
        assert updated["version"] == sample_board["version"] + 1

        # Try with wrong version
        with pytest.raises(ConflictError) as exc_info:
            kanban_db.update_board(
                board_id=sample_board["id"],
                name="Should Fail",
                expected_version=sample_board["version"]  # Old version
            )
        assert "Version mismatch" in str(exc_info.value)

    def test_archive_board(self, kanban_db: KanbanDB, sample_board: dict):
        """Test archiving and unarchiving a board."""
        lst = kanban_db.create_list(
            board_id=sample_board["id"],
            name="Archive List",
            client_id="board-archive-list-1"
        )
        kanban_db.create_card(
            list_id=lst["id"],
            title="Archive Card",
            client_id="board-archive-card-1"
        )
        # Archive
        archived = kanban_db.archive_board(sample_board["id"], archive=True)
        assert archived["archived"] is True
        assert archived["archived_at"] is not None

        # Should not appear in default list
        boards, total = kanban_db.list_boards(include_archived=False)
        assert total == 0

        # Should appear with include_archived
        boards, total = kanban_db.list_boards(include_archived=True)
        assert total == 1

        lists = kanban_db.list_lists(sample_board["id"], include_archived=True)
        assert len(lists) == 1
        assert lists[0]["archived"] is True

        cards = kanban_db.list_cards(lst["id"], include_archived=True)
        assert len(cards) == 1
        assert cards[0]["archived"] is True

        # Unarchive
        unarchived = kanban_db.archive_board(sample_board["id"], archive=False)
        assert unarchived["archived"] is False
        assert unarchived["archived_at"] is None

    def test_delete_board(self, kanban_db: KanbanDB, sample_board: dict):
        """Test soft deleting a board."""
        success = kanban_db.delete_board(sample_board["id"])
        assert success is True

        # Should not appear in default list
        boards, total = kanban_db.list_boards()
        assert total == 0

        # Should appear with include_deleted
        boards, total = kanban_db.list_boards(include_deleted=True)
        assert total == 1
        assert boards[0]["deleted"] is True

    def test_restore_board(self, kanban_db: KanbanDB, sample_board: dict):
        """Test restoring a soft-deleted board."""
        kanban_db.delete_board(sample_board["id"])

        restored = kanban_db.restore_board(sample_board["id"])
        assert restored["deleted"] is False
        assert restored["deleted_at"] is None

        boards, total = kanban_db.list_boards()
        assert total == 1


# =============================================================================
# List Tests
# =============================================================================

class TestListOperations:
    """Tests for list CRUD operations."""

    def test_create_list(self, kanban_db: KanbanDB, sample_board: dict):
        """Test creating a new list."""
        lst = kanban_db.create_list(
            board_id=sample_board["id"],
            name="To Do",
            client_id="list-1"
        )

        assert lst["id"] is not None
        assert lst["name"] == "To Do"
        assert lst["board_id"] == sample_board["id"]
        assert lst["position"] == 0
        assert lst["version"] == 1

    def test_create_list_auto_position(self, kanban_db: KanbanDB, sample_board: dict):
        """Test that lists get auto-positioned at the end."""
        list1 = kanban_db.create_list(board_id=sample_board["id"], name="List 1", client_id="l1")
        list2 = kanban_db.create_list(board_id=sample_board["id"], name="List 2", client_id="l2")
        list3 = kanban_db.create_list(board_id=sample_board["id"], name="List 3", client_id="l3")

        assert list1["position"] == 0
        assert list2["position"] == 1
        assert list3["position"] == 2

    def test_create_list_board_not_found(self, kanban_db: KanbanDB):
        """Test creating list in non-existent board."""
        with pytest.raises(NotFoundError):
            kanban_db.create_list(board_id=99999, name="Test", client_id="l1")

    def test_list_lists(self, kanban_db: KanbanDB, sample_board: dict):
        """Test listing lists for a board."""
        kanban_db.create_list(board_id=sample_board["id"], name="List 1", client_id="l1")
        kanban_db.create_list(board_id=sample_board["id"], name="List 2", client_id="l2")

        lists = kanban_db.list_lists(sample_board["id"])

        assert len(lists) == 2
        # Should be ordered by position
        assert lists[0]["position"] < lists[1]["position"]

    def test_update_list(self, kanban_db: KanbanDB, sample_list: dict):
        """Test updating a list."""
        updated = kanban_db.update_list(
            list_id=sample_list["id"],
            name="Renamed List"
        )

        assert updated["name"] == "Renamed List"
        assert updated["version"] == sample_list["version"] + 1

    def test_reorder_lists(self, kanban_db: KanbanDB, sample_board: dict):
        """Test reordering lists in a board."""
        l1 = kanban_db.create_list(board_id=sample_board["id"], name="List 1", client_id="l1")
        l2 = kanban_db.create_list(board_id=sample_board["id"], name="List 2", client_id="l2")
        l3 = kanban_db.create_list(board_id=sample_board["id"], name="List 3", client_id="l3")

        # Reorder: 3, 1, 2
        reordered = kanban_db.reorder_lists(
            board_id=sample_board["id"],
            list_ids=[l3["id"], l1["id"], l2["id"]]
        )

        assert reordered[0]["id"] == l3["id"]
        assert reordered[0]["position"] == 0
        assert reordered[1]["id"] == l1["id"]
        assert reordered[1]["position"] == 1
        assert reordered[2]["id"] == l2["id"]
        assert reordered[2]["position"] == 2

    def test_archive_list(self, kanban_db: KanbanDB, sample_list: dict, sample_board: dict):
        """Test archiving and unarchiving a list."""
        kanban_db.create_card(
            list_id=sample_list["id"],
            title="Archived list card",
            client_id="card-archived-list-1"
        )
        archived = kanban_db.archive_list(sample_list["id"], archive=True)
        assert archived["archived"] is True

        cards = kanban_db.list_cards(sample_list["id"], include_archived=False)
        assert cards == []
        cards = kanban_db.list_cards(sample_list["id"], include_archived=True)
        assert len(cards) == 1
        assert cards[0]["archived"] is True

        # Should not appear in default list
        lists = kanban_db.list_lists(sample_board["id"], include_archived=False)
        assert len(lists) == 0

        # Should appear with include_archived
        lists = kanban_db.list_lists(sample_board["id"], include_archived=True)
        assert len(lists) == 1

    def test_delete_and_restore_list(self, kanban_db: KanbanDB, sample_list: dict, sample_board: dict):
        """Test soft delete and restore of a list."""
        kanban_db.delete_list(sample_list["id"])

        lists = kanban_db.list_lists(sample_board["id"])
        assert len(lists) == 0

        restored = kanban_db.restore_list(sample_list["id"])
        assert restored["deleted"] is False

        lists = kanban_db.list_lists(sample_board["id"])
        assert len(lists) == 1


# =============================================================================
# Card Tests
# =============================================================================

class TestCardOperations:
    """Tests for card CRUD operations."""

    def test_create_card(self, kanban_db: KanbanDB, sample_list: dict):
        """Test creating a new card."""
        card = kanban_db.create_card(
            list_id=sample_list["id"],
            title="My Task",
            client_id="card-1",
            description="Task description"
        )

        assert card["id"] is not None
        assert card["title"] == "My Task"
        assert card["description"] == "Task description"
        assert card["list_id"] == sample_list["id"]
        assert card["position"] == 0
        assert card["version"] == 1

    def test_create_card_with_all_fields(self, kanban_db: KanbanDB, sample_list: dict):
        """Test creating a card with all optional fields."""
        card = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Full Card",
            client_id="card-full",
            description="Description",
            due_date="2025-12-31T23:59:59",
            start_date="2025-12-01T00:00:00",
            priority="high",
            metadata={"tags": ["urgent", "review"]}
        )

        assert card["due_date"] is not None
        assert card["start_date"] is not None
        assert card["priority"] == "high"
        assert card["metadata"]["tags"] == ["urgent", "review"]

    def test_create_card_enforces_board_limit(self, kanban_db: KanbanDB, sample_list: dict):
        """Test board-level card limit enforcement."""
        kanban_db.MAX_CARDS_PER_BOARD = 2
        kanban_db.MAX_CARDS_PER_LIST = 10

        kanban_db.create_card(
            list_id=sample_list["id"],
            title="Card 1",
            client_id="card-client-1"
        )
        kanban_db.create_card(
            list_id=sample_list["id"],
            title="Card 2",
            client_id="card-client-2"
        )

        with pytest.raises(InputError) as exc_info:
            kanban_db.create_card(
                list_id=sample_list["id"],
                title="Card 3",
                client_id="card-client-3"
            )
        assert "per board" in str(exc_info.value).lower()

    def test_create_card_invalid_priority(self, kanban_db: KanbanDB, sample_list: dict):
        """Test that invalid priority raises InputError."""
        with pytest.raises(InputError):
            kanban_db.create_card(
                list_id=sample_list["id"],
                title="Bad Priority",
                client_id="card-bad",
                priority="invalid"
            )

    def test_list_cards(self, kanban_db: KanbanDB, sample_list: dict):
        """Test listing cards in a list."""
        kanban_db.create_card(list_id=sample_list["id"], title="Card 1", client_id="c1")
        kanban_db.create_card(list_id=sample_list["id"], title="Card 2", client_id="c2")

        cards = kanban_db.list_cards(sample_list["id"])

        assert len(cards) == 2
        assert cards[0]["position"] < cards[1]["position"]

    def test_update_card(self, kanban_db: KanbanDB, sample_card: dict):
        """Test updating a card."""
        updated = kanban_db.update_card(
            card_id=sample_card["id"],
            title="Updated Title",
            priority="urgent"
        )

        assert updated["title"] == "Updated Title"
        assert updated["priority"] == "urgent"
        assert updated["version"] == sample_card["version"] + 1

    def test_move_card(self, kanban_db: KanbanDB, sample_board: dict, sample_card: dict):
        """Test moving a card to another list."""
        # Create a second list
        list2 = kanban_db.create_list(
            board_id=sample_board["id"],
            name="Done",
            client_id="list-done"
        )

        moved = kanban_db.move_card(
            card_id=sample_card["id"],
            target_list_id=list2["id"],
            position=0
        )

        assert moved["list_id"] == list2["id"]
        assert moved["position"] == 0

    def test_copy_card(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict, sample_card: dict):
        """Test copying a card."""
        # Create a second list
        list2 = kanban_db.create_list(
            board_id=sample_board["id"],
            name="Backlog",
            client_id="list-backlog"
        )

        copy = kanban_db.copy_card(
            card_id=sample_card["id"],
            target_list_id=list2["id"],
            new_client_id="card-copy-1"
        )

        assert copy["id"] != sample_card["id"]
        assert copy["list_id"] == list2["id"]
        assert "Copy of" in copy["title"]
        assert copy["description"] == sample_card["description"]

    def test_copy_card_respects_board_limit(self, kanban_db: KanbanDB, sample_list: dict, sample_card: dict):
        """Test copy card fails when board limit is reached."""
        kanban_db.MAX_CARDS_PER_BOARD = 1
        kanban_db.MAX_CARDS_PER_LIST = 10

        with pytest.raises(InputError) as exc_info:
            kanban_db.copy_card(
                card_id=sample_card["id"],
                target_list_id=sample_list["id"],
                new_client_id="card-copy-limit"
            )
        assert "per board" in str(exc_info.value).lower()

    def test_copy_card_custom_title(self, kanban_db: KanbanDB, sample_list: dict, sample_card: dict):
        """Test copying a card with a custom title."""
        copy = kanban_db.copy_card(
            card_id=sample_card["id"],
            target_list_id=sample_list["id"],
            new_client_id="card-copy-custom",
            new_title="Custom Copy Title"
        )

        assert copy["title"] == "Custom Copy Title"

    def test_reorder_cards(self, kanban_db: KanbanDB, sample_list: dict):
        """Test reordering cards in a list."""
        c1 = kanban_db.create_card(list_id=sample_list["id"], title="Card 1", client_id="c1")
        c2 = kanban_db.create_card(list_id=sample_list["id"], title="Card 2", client_id="c2")
        c3 = kanban_db.create_card(list_id=sample_list["id"], title="Card 3", client_id="c3")

        # Reorder: 3, 1, 2
        reordered = kanban_db.reorder_cards(
            list_id=sample_list["id"],
            card_ids=[c3["id"], c1["id"], c2["id"]]
        )

        assert reordered[0]["id"] == c3["id"]
        assert reordered[0]["position"] == 0
        assert reordered[1]["id"] == c1["id"]
        assert reordered[2]["id"] == c2["id"]

    def test_archive_card(self, kanban_db: KanbanDB, sample_card: dict, sample_list: dict):
        """Test archiving and unarchiving a card."""
        archived = kanban_db.archive_card(sample_card["id"], archive=True)
        assert archived["archived"] is True

        cards = kanban_db.list_cards(sample_list["id"], include_archived=False)
        assert len(cards) == 0

        cards = kanban_db.list_cards(sample_list["id"], include_archived=True)
        assert len(cards) == 1

    def test_delete_and_restore_card(self, kanban_db: KanbanDB, sample_card: dict, sample_list: dict):
        """Test soft delete and restore of a card."""
        kanban_db.delete_card(sample_card["id"])

        cards = kanban_db.list_cards(sample_list["id"])
        assert len(cards) == 0

        restored = kanban_db.restore_card(sample_card["id"])
        assert restored["deleted"] is False

        cards = kanban_db.list_cards(sample_list["id"])
        assert len(cards) == 1


# =============================================================================
# Search Tests
# =============================================================================

class TestSearchOperations:
    """Tests for search functionality."""

    def test_search_cards(self, kanban_db: KanbanDB, sample_list: dict):
        """Test searching cards."""
        kanban_db.create_card(
            list_id=sample_list["id"],
            title="Fix login bug",
            client_id="c1",
            description="Authentication issue"
        )
        kanban_db.create_card(
            list_id=sample_list["id"],
            title="Add signup page",
            client_id="c2",
            description="New registration flow"
        )
        kanban_db.create_card(
            list_id=sample_list["id"],
            title="Update README",
            client_id="c3",
            description="Documentation update"
        )

        # Search for 'login'
        results, total = kanban_db.search_cards("login")
        assert total == 1
        assert "login" in results[0]["title"].lower()

        # Search for 'registration'
        results, total = kanban_db.search_cards("registration")
        assert total == 1

    def test_search_cards_empty_query(self, kanban_db: KanbanDB):
        """Test that empty search query raises InputError."""
        with pytest.raises(InputError):
            kanban_db.search_cards("")


# =============================================================================
# FTS Maintenance Tests
# =============================================================================

class TestFtsMaintenance:
    """Tests for FTS maintenance actions."""

    def test_optimize_and_rebuild_preserve_search(self, kanban_db: KanbanDB, sample_list: dict):
        """Optimize and rebuild should not drop searchable content."""
        card = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Optimize FTS index",
            client_id="fts-maint-card",
            description="FTS maintenance smoke"
        )

        results, total = kanban_db.search_cards("optimize")
        assert total >= 1
        assert any(r["id"] == card["id"] for r in results)

        kanban_db.optimize_fts()
        results, total = kanban_db.search_cards("optimize")
        assert total >= 1
        assert any(r["id"] == card["id"] for r in results)

        kanban_db.rebuild_fts()
        results, total = kanban_db.search_cards("optimize")
        assert total >= 1
        assert any(r["id"] == card["id"] for r in results)


# =============================================================================
# Activity Tests
# =============================================================================

class TestActivityOperations:
    """Tests for activity log functionality."""

    def test_log_activity(self, kanban_db: KanbanDB, sample_board: dict):
        """Test logging an activity."""
        activity = kanban_db.log_activity(
            board_id=sample_board["id"],
            action_type="create",
            entity_type="card",
            entity_id=1,
            details={"title": "New Card"}
        )

        assert activity["id"] is not None
        assert activity["action_type"] == "create"
        assert activity["entity_type"] == "card"
        assert activity["details"]["title"] == "New Card"

    def test_get_board_activities(self, kanban_db: KanbanDB, sample_board: dict):
        """Test getting activities for a board."""
        # Get initial count (includes auto-logged board_created activity)
        _, initial_count = kanban_db.get_board_activities(sample_board["id"])

        kanban_db.log_activity(
            board_id=sample_board["id"],
            action_type="create",
            entity_type="list"
        )
        kanban_db.log_activity(
            board_id=sample_board["id"],
            action_type="create",
            entity_type="card"
        )

        activities, total = kanban_db.get_board_activities(sample_board["id"])
        assert total == initial_count + 2  # 2 new activities added

    def test_cleanup_old_activities_respects_board_retention(self, kanban_db: KanbanDB, sample_board: dict):
        """Cleanup should honor per-board retention values."""
        short_board = kanban_db.create_board(
            name="Short Retention",
            client_id="board-retention-short",
            activity_retention_days=7,
        )
        act_long = kanban_db.log_activity(
            board_id=sample_board["id"],
            action_type="update",
            entity_type="card",
            entity_id=1,
        )
        act_short = kanban_db.log_activity(
            board_id=short_board["id"],
            action_type="update",
            entity_type="card",
            entity_id=2,
        )

        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
        conn = kanban_db._connect()
        try:
            conn.execute(
                "UPDATE kanban_activities SET created_at = ? WHERE id IN (?, ?)",
                (old_ts, act_long["id"], act_short["id"]),
            )
            conn.commit()
        finally:
            conn.close()

        deleted = kanban_db.cleanup_old_activities()
        assert deleted == 1

        conn = kanban_db._connect()
        try:
            cur = conn.execute("SELECT COUNT(*) as cnt FROM kanban_activities WHERE id = ?", (act_long["id"],))
            assert cur.fetchone()["cnt"] == 1
            cur = conn.execute("SELECT COUNT(*) as cnt FROM kanban_activities WHERE id = ?", (act_short["id"],))
            assert cur.fetchone()["cnt"] == 0
        finally:
            conn.close()


# =============================================================================
# User Isolation Tests
# =============================================================================

class TestUserIsolation:
    """Tests for user isolation."""

    def test_user_cannot_see_other_user_boards(self, kanban_db: KanbanDB, kanban_db_user2: KanbanDB):
        """Test that users can only see their own boards.

        Note: Since each user has their own separate database file, board IDs
        may overlap between users. The isolation is enforced by:
        1. Separate database files per user
        2. user_id filtering in all queries
        """
        # User 1 creates a board
        board1 = kanban_db.create_board(name="User 1 Board", client_id="b1")

        # User 2 creates a board
        board2 = kanban_db_user2.create_board(name="User 2 Board", client_id="b2")

        # User 1 should only see their board
        boards1, total1 = kanban_db.list_boards()
        assert total1 == 1
        assert boards1[0]["name"] == "User 1 Board"
        assert boards1[0]["user_id"] == "test_user_1"

        # User 2 should only see their board
        boards2, total2 = kanban_db_user2.list_boards()
        assert total2 == 1
        assert boards2[0]["name"] == "User 2 Board"
        assert boards2[0]["user_id"] == "test_user_2"

        # Both databases are separate (board1 and board2 both have id=1 in their respective DBs)
        assert board1["id"] == 1
        assert board2["id"] == 1

        # Verify user_id is correctly set
        assert board1["user_id"] == "test_user_1"
        assert board2["user_id"] == "test_user_2"


# =============================================================================
# Nested Response Tests
# =============================================================================

class TestNestedResponses:
    """Tests for nested response methods."""

    def test_get_board_with_lists_and_cards(self, kanban_db: KanbanDB, sample_board: dict):
        """Test getting a board with all nested data."""
        # Create lists and cards
        list1 = kanban_db.create_list(board_id=sample_board["id"], name="To Do", client_id="l1")
        list2 = kanban_db.create_list(board_id=sample_board["id"], name="Done", client_id="l2")

        card1 = kanban_db.create_card(list_id=list1["id"], title="Task 1", client_id="c1")
        kanban_db.create_card(list_id=list1["id"], title="Task 2", client_id="c2")
        kanban_db.create_card(list_id=list2["id"], title="Task 3", client_id="c3")

        label = kanban_db.create_label(board_id=sample_board["id"], name="Research", color="blue")
        kanban_db.assign_label_to_card(card1["id"], label["id"])

        checklist = kanban_db.create_checklist(card_id=card1["id"], name="Checklist 1")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 1", checked=True)
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 2", checked=False)
        kanban_db.create_comment(card_id=card1["id"], content="First comment")

        # Get nested response
        board = kanban_db.get_board_with_lists_and_cards(sample_board["id"])

        assert board is not None
        assert any(lbl["id"] == label["id"] for lbl in board["labels"])
        assert len(board["lists"]) == 2
        assert board["total_cards"] == 3

        # Check that lists have cards
        todo_list = next(l for l in board["lists"] if l["name"] == "To Do")
        assert len(todo_list["cards"]) == 2
        assert todo_list["card_count"] == 2
        todo_card = next(c for c in todo_list["cards"] if c["id"] == card1["id"])
        assert any(lbl["id"] == label["id"] for lbl in todo_card["labels"])
        assert todo_card["checklist_count"] == 1
        assert todo_card["checklist_total"] == 2
        assert todo_card["checklist_complete"] == 1
        assert todo_card["comment_count"] == 1

        done_list = next(l for l in board["lists"] if l["name"] == "Done")
        assert len(done_list["cards"]) == 1


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_restore_non_deleted_board(self, kanban_db: KanbanDB, sample_board: dict):
        """Test that restoring a non-deleted board raises InputError."""
        with pytest.raises(InputError) as exc_info:
            kanban_db.restore_board(sample_board["id"])
        assert "not deleted" in str(exc_info.value)

    def test_update_non_existent_board(self, kanban_db: KanbanDB):
        """Test updating a non-existent board raises NotFoundError."""
        with pytest.raises(NotFoundError):
            kanban_db.update_board(board_id=99999, name="New Name")

    def test_move_card_to_different_board(self, kanban_db: KanbanDB, sample_card: dict):
        """Test that moving a card to a list in a different board fails."""
        # Create a second board and list
        board2 = kanban_db.create_board(name="Board 2", client_id="b2")
        list2 = kanban_db.create_list(board_id=board2["id"], name="List 2", client_id="l2")

        with pytest.raises(InputError) as exc_info:
            kanban_db.move_card(
                card_id=sample_card["id"],
                target_list_id=list2["id"]
            )
        assert "different board" in str(exc_info.value)


# =============================================================================
# Phase 2: Label Tests
# =============================================================================

class TestLabelOperations:
    """Tests for label CRUD operations."""

    def test_create_label(self, kanban_db: KanbanDB, sample_board: dict):
        """Test creating a new label."""
        label = kanban_db.create_label(
            board_id=sample_board["id"],
            name="Bug",
            color="red"
        )

        assert label["id"] is not None
        assert label["uuid"] is not None
        assert label["name"] == "Bug"
        assert label["color"] == "red"
        assert label["board_id"] == sample_board["id"]

    def test_create_label_case_insensitive_color(self, kanban_db: KanbanDB, sample_board: dict):
        """Test that color is case-insensitive."""
        label = kanban_db.create_label(
            board_id=sample_board["id"],
            name="Feature",
            color="BLUE"
        )
        assert label["color"] == "blue"

    def test_create_label_invalid_color(self, kanban_db: KanbanDB, sample_board: dict):
        """Test that invalid color raises InputError."""
        with pytest.raises(InputError) as exc_info:
            kanban_db.create_label(
                board_id=sample_board["id"],
                name="Test",
                color="rainbow"
            )
        assert "Invalid color" in str(exc_info.value)

    def test_list_labels(self, kanban_db: KanbanDB, sample_board: dict):
        """Test listing labels for a board."""
        kanban_db.create_label(board_id=sample_board["id"], name="Bug", color="red")
        kanban_db.create_label(board_id=sample_board["id"], name="Feature", color="green")
        kanban_db.create_label(board_id=sample_board["id"], name="Documentation", color="blue")

        labels = kanban_db.list_labels(board_id=sample_board["id"])

        assert len(labels) == 3
        # Should be sorted alphabetically
        assert labels[0]["name"] == "Bug"
        assert labels[1]["name"] == "Documentation"
        assert labels[2]["name"] == "Feature"

    def test_update_label(self, kanban_db: KanbanDB, sample_board: dict):
        """Test updating a label."""
        label = kanban_db.create_label(
            board_id=sample_board["id"],
            name="Old Name",
            color="red"
        )

        updated = kanban_db.update_label(
            label_id=label["id"],
            name="New Name",
            color="green"
        )

        assert updated["name"] == "New Name"
        assert updated["color"] == "green"

    def test_delete_label(self, kanban_db: KanbanDB, sample_board: dict):
        """Test deleting a label."""
        label = kanban_db.create_label(
            board_id=sample_board["id"],
            name="ToDelete",
            color="gray"
        )

        result = kanban_db.delete_label(label_id=label["id"])
        assert result is True

        # Should be gone
        assert kanban_db.get_label(label["id"]) is None

    def test_assign_and_remove_label(self, kanban_db: KanbanDB, sample_card: dict, sample_board: dict):
        """Test assigning and removing a label from a card."""
        label = kanban_db.create_label(
            board_id=sample_board["id"],
            name="Priority",
            color="orange"
        )

        # Assign
        kanban_db.assign_label_to_card(card_id=sample_card["id"], label_id=label["id"])

        # Get card labels
        labels = kanban_db.get_card_labels(card_id=sample_card["id"])
        assert len(labels) == 1
        assert labels[0]["name"] == "Priority"

        # Remove
        kanban_db.remove_label_from_card(card_id=sample_card["id"], label_id=label["id"])

        labels = kanban_db.get_card_labels(card_id=sample_card["id"])
        assert len(labels) == 0

    def test_assign_label_from_different_board_fails(self, kanban_db: KanbanDB, sample_card: dict):
        """Test that assigning a label from a different board fails."""
        # Create another board with a label
        other_board = kanban_db.create_board(name="Other Board", client_id="other-b")
        other_label = kanban_db.create_label(
            board_id=other_board["id"],
            name="Other Label",
            color="purple"
        )

        with pytest.raises(InputError) as exc_info:
            kanban_db.assign_label_to_card(
                card_id=sample_card["id"],
                label_id=other_label["id"]
            )
        assert "card's board" in str(exc_info.value)


# =============================================================================
# Phase 2: Checklist Tests
# =============================================================================

class TestChecklistOperations:
    """Tests for checklist CRUD operations."""

    def test_create_checklist(self, kanban_db: KanbanDB, sample_card: dict):
        """Test creating a new checklist."""
        checklist = kanban_db.create_checklist(
            card_id=sample_card["id"],
            name="Todo List"
        )

        assert checklist["id"] is not None
        assert checklist["uuid"] is not None
        assert checklist["name"] == "Todo List"
        assert checklist["card_id"] == sample_card["id"]
        assert checklist["position"] == 0

    def test_create_checklist_auto_position(self, kanban_db: KanbanDB, sample_card: dict):
        """Test that checklists get auto-positioned."""
        cl1 = kanban_db.create_checklist(card_id=sample_card["id"], name="First")
        cl2 = kanban_db.create_checklist(card_id=sample_card["id"], name="Second")
        cl3 = kanban_db.create_checklist(card_id=sample_card["id"], name="Third")

        assert cl1["position"] == 0
        assert cl2["position"] == 1
        assert cl3["position"] == 2

    def test_list_checklists(self, kanban_db: KanbanDB, sample_card: dict):
        """Test listing checklists for a card."""
        kanban_db.create_checklist(card_id=sample_card["id"], name="First")
        kanban_db.create_checklist(card_id=sample_card["id"], name="Second")

        checklists = kanban_db.list_checklists(card_id=sample_card["id"])

        assert len(checklists) == 2
        assert checklists[0]["name"] == "First"
        assert checklists[1]["name"] == "Second"

    def test_update_checklist(self, kanban_db: KanbanDB, sample_card: dict):
        """Test updating a checklist."""
        checklist = kanban_db.create_checklist(
            card_id=sample_card["id"],
            name="Old Name"
        )

        updated = kanban_db.update_checklist(
            checklist_id=checklist["id"],
            name="New Name"
        )

        assert updated["name"] == "New Name"

    def test_reorder_checklists(self, kanban_db: KanbanDB, sample_card: dict):
        """Test reordering checklists."""
        cl1 = kanban_db.create_checklist(card_id=sample_card["id"], name="First")
        cl2 = kanban_db.create_checklist(card_id=sample_card["id"], name="Second")
        cl3 = kanban_db.create_checklist(card_id=sample_card["id"], name="Third")

        # Reverse order
        reordered = kanban_db.reorder_checklists(
            card_id=sample_card["id"],
            checklist_ids=[cl3["id"], cl2["id"], cl1["id"]]
        )

        assert reordered[0]["name"] == "Third"
        assert reordered[1]["name"] == "Second"
        assert reordered[2]["name"] == "First"

    def test_delete_checklist(self, kanban_db: KanbanDB, sample_card: dict):
        """Test deleting a checklist."""
        checklist = kanban_db.create_checklist(
            card_id=sample_card["id"],
            name="ToDelete"
        )

        result = kanban_db.delete_checklist(checklist_id=checklist["id"])
        assert result is True

        assert kanban_db.get_checklist(checklist["id"]) is None


# =============================================================================
# Phase 2: Checklist Item Tests
# =============================================================================

class TestChecklistItemOperations:
    """Tests for checklist item CRUD operations."""

    def test_create_checklist_item(self, kanban_db: KanbanDB, sample_card: dict):
        """Test creating a checklist item."""
        checklist = kanban_db.create_checklist(
            card_id=sample_card["id"],
            name="Tasks"
        )

        item = kanban_db.create_checklist_item(
            checklist_id=checklist["id"],
            name="Do the thing"
        )

        assert item["id"] is not None
        assert item["uuid"] is not None
        assert item["name"] == "Do the thing"
        assert item["checklist_id"] == checklist["id"]
        assert item["checked"] is False
        assert item["checked_at"] is None
        assert item["position"] == 0

    def test_create_item_checked(self, kanban_db: KanbanDB, sample_card: dict):
        """Test creating an item that starts checked."""
        checklist = kanban_db.create_checklist(
            card_id=sample_card["id"],
            name="Done Tasks"
        )

        item = kanban_db.create_checklist_item(
            checklist_id=checklist["id"],
            name="Already done",
            checked=True
        )

        assert item["checked"] is True
        assert item["checked_at"] is not None

    def test_check_and_uncheck_item(self, kanban_db: KanbanDB, sample_card: dict):
        """Test checking and unchecking an item."""
        checklist = kanban_db.create_checklist(
            card_id=sample_card["id"],
            name="Tasks"
        )
        item = kanban_db.create_checklist_item(
            checklist_id=checklist["id"],
            name="Task 1"
        )

        # Check
        updated = kanban_db.update_checklist_item(item_id=item["id"], checked=True)
        assert updated["checked"] is True
        assert updated["checked_at"] is not None

        # Uncheck
        updated = kanban_db.update_checklist_item(item_id=item["id"], checked=False)
        assert updated["checked"] is False
        assert updated["checked_at"] is None

    def test_list_checklist_items(self, kanban_db: KanbanDB, sample_card: dict):
        """Test listing items in a checklist."""
        checklist = kanban_db.create_checklist(
            card_id=sample_card["id"],
            name="Tasks"
        )

        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 1")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 2")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 3")

        items = kanban_db.list_checklist_items(checklist_id=checklist["id"])

        assert len(items) == 3
        assert items[0]["name"] == "Item 1"
        assert items[1]["name"] == "Item 2"
        assert items[2]["name"] == "Item 3"

    def test_reorder_checklist_items(self, kanban_db: KanbanDB, sample_card: dict):
        """Test reordering items in a checklist."""
        checklist = kanban_db.create_checklist(
            card_id=sample_card["id"],
            name="Tasks"
        )

        i1 = kanban_db.create_checklist_item(checklist_id=checklist["id"], name="First")
        i2 = kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Second")
        i3 = kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Third")

        # Reverse order
        reordered = kanban_db.reorder_checklist_items(
            checklist_id=checklist["id"],
            item_ids=[i3["id"], i2["id"], i1["id"]]
        )

        assert reordered[0]["name"] == "Third"
        assert reordered[1]["name"] == "Second"
        assert reordered[2]["name"] == "First"

    def test_delete_checklist_item(self, kanban_db: KanbanDB, sample_card: dict):
        """Test deleting a checklist item."""
        checklist = kanban_db.create_checklist(
            card_id=sample_card["id"],
            name="Tasks"
        )
        item = kanban_db.create_checklist_item(
            checklist_id=checklist["id"],
            name="ToDelete"
        )

        result = kanban_db.delete_checklist_item(item_id=item["id"])
        assert result is True

        assert kanban_db.get_checklist_item(item["id"]) is None

    def test_get_checklist_with_items(self, kanban_db: KanbanDB, sample_card: dict):
        """Test getting a checklist with its items and progress."""
        checklist = kanban_db.create_checklist(
            card_id=sample_card["id"],
            name="Tasks"
        )

        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 1")
        i2 = kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 2")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 3")

        # Check one item
        kanban_db.update_checklist_item(item_id=i2["id"], checked=True)

        result = kanban_db.get_checklist_with_items(checklist_id=checklist["id"])

        assert result["name"] == "Tasks"
        assert len(result["items"]) == 3
        assert result["total_items"] == 3
        assert result["checked_items"] == 1
        assert result["progress_percent"] == 33


# =============================================================================
# Phase 2: Comment Tests
# =============================================================================

class TestCommentOperations:
    """Tests for comment CRUD operations."""

    def test_create_comment(self, kanban_db: KanbanDB, sample_card: dict):
        """Test creating a comment."""
        comment = kanban_db.create_comment(
            card_id=sample_card["id"],
            content="This is a comment"
        )

        assert comment["id"] is not None
        assert comment["uuid"] is not None
        assert comment["content"] == "This is a comment"
        assert comment["card_id"] == sample_card["id"]
        assert comment["user_id"] == "test_user_1"
        assert comment["deleted"] is False

    def test_list_comments(self, kanban_db: KanbanDB, sample_card: dict):
        """Test listing comments for a card."""
        kanban_db.create_comment(card_id=sample_card["id"], content="Comment 1")
        kanban_db.create_comment(card_id=sample_card["id"], content="Comment 2")
        kanban_db.create_comment(card_id=sample_card["id"], content="Comment 3")

        comments, total = kanban_db.list_comments(card_id=sample_card["id"])

        assert total == 3
        assert len(comments) == 3
        # All comments should be present (ordering may vary if same timestamp)
        contents = {c["content"] for c in comments}
        assert contents == {"Comment 1", "Comment 2", "Comment 3"}

    def test_list_comments_pagination(self, kanban_db: KanbanDB, sample_card: dict):
        """Test pagination in list_comments."""
        for i in range(5):
            kanban_db.create_comment(card_id=sample_card["id"], content=f"Comment {i+1}")

        page1, total = kanban_db.list_comments(
            card_id=sample_card["id"],
            limit=2,
            offset=0
        )
        assert len(page1) == 2
        assert total == 5

        page2, _ = kanban_db.list_comments(
            card_id=sample_card["id"],
            limit=2,
            offset=2
        )
        assert len(page2) == 2

        page3, _ = kanban_db.list_comments(
            card_id=sample_card["id"],
            limit=2,
            offset=4
        )
        assert len(page3) == 1

    def test_update_comment(self, kanban_db: KanbanDB, sample_card: dict):
        """Test updating a comment."""
        comment = kanban_db.create_comment(
            card_id=sample_card["id"],
            content="Original content"
        )

        updated = kanban_db.update_comment(
            comment_id=comment["id"],
            content="Updated content"
        )

        assert updated["content"] == "Updated content"

    def test_soft_delete_comment(self, kanban_db: KanbanDB, sample_card: dict):
        """Test soft deleting a comment."""
        comment = kanban_db.create_comment(
            card_id=sample_card["id"],
            content="To delete"
        )

        result = kanban_db.delete_comment(comment_id=comment["id"], hard_delete=False)
        assert result is True

        # Should not be visible by default
        comments, total = kanban_db.list_comments(card_id=sample_card["id"])
        assert total == 0

        # But should be visible with include_deleted
        comments, total = kanban_db.list_comments(
            card_id=sample_card["id"],
            include_deleted=True
        )
        assert total == 1
        assert comments[0]["deleted"] is True

    def test_hard_delete_comment(self, kanban_db: KanbanDB, sample_card: dict):
        """Test hard deleting a comment."""
        comment = kanban_db.create_comment(
            card_id=sample_card["id"],
            content="To hard delete"
        )

        result = kanban_db.delete_comment(comment_id=comment["id"], hard_delete=True)
        assert result is True

        # Should be completely gone
        comments, total = kanban_db.list_comments(
            card_id=sample_card["id"],
            include_deleted=True
        )
        assert total == 0

    def test_comment_content_validation(self, kanban_db: KanbanDB, sample_card: dict):
        """Test comment content validation."""
        # Empty content should fail
        with pytest.raises(InputError) as exc_info:
            kanban_db.create_comment(
                card_id=sample_card["id"],
                content="   "
            )
        assert "required" in str(exc_info.value).lower()

        # Too long content should fail
        too_long = "x" * (kanban_db.MAX_COMMENT_SIZE + 1)
        with pytest.raises(InputError) as exc_info:
            kanban_db.create_comment(
                card_id=sample_card["id"],
                content=too_long
            )
        assert str(kanban_db.MAX_COMMENT_SIZE) in str(exc_info.value)


# =============================================================================
# Phase 3: Export/Import Tests
# =============================================================================

class TestExportOperations:
    """Tests for board export functionality."""

    def test_export_board_basic(self, kanban_db: KanbanDB, sample_board: dict):
        """Test exporting a board with basic data."""
        export = kanban_db.export_board(board_id=sample_board["id"])

        assert export["format"] == "tldw_kanban_v1"
        assert "exported_at" in export
        assert export["board"]["name"] == sample_board["name"]
        assert "labels" in export
        assert "lists" in export

    def test_export_board_with_lists_and_cards(self, kanban_db: KanbanDB, sample_board: dict):
        """Test exporting a board with nested lists and cards."""
        # Create some lists and cards
        list1 = kanban_db.create_list(board_id=sample_board["id"], name="Todo", client_id="list-todo")
        list2 = kanban_db.create_list(board_id=sample_board["id"], name="Done", client_id="list-done")

        kanban_db.create_card(list_id=list1["id"], title="Task 1", client_id="card-1")
        kanban_db.create_card(list_id=list1["id"], title="Task 2", client_id="card-2")
        kanban_db.create_card(list_id=list2["id"], title="Task 3", client_id="card-3")

        export = kanban_db.export_board(board_id=sample_board["id"])

        assert len(export["lists"]) == 2
        # Find the Todo list
        todo_list = next(l for l in export["lists"] if l["name"] == "Todo")
        assert len(todo_list["cards"]) == 2
        done_list = next(l for l in export["lists"] if l["name"] == "Done")
        assert len(done_list["cards"]) == 1

    def test_export_board_with_labels(self, kanban_db: KanbanDB, sample_board: dict, sample_card: dict):
        """Test exporting a board with labels."""
        label = kanban_db.create_label(
            board_id=sample_board["id"],
            name="Bug",
            color="red"
        )
        kanban_db.assign_label_to_card(card_id=sample_card["id"], label_id=label["id"])

        export = kanban_db.export_board(board_id=sample_board["id"])

        assert len(export["labels"]) == 1
        assert export["labels"][0]["name"] == "Bug"
        assert export["labels"][0]["color"] == "red"

    def test_export_board_with_checklists(self, kanban_db: KanbanDB, sample_card: dict, sample_board: dict):
        """Test exporting a board with checklists."""
        checklist = kanban_db.create_checklist(card_id=sample_card["id"], name="Tasks")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 1")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 2", checked=True)

        export = kanban_db.export_board(board_id=sample_board["id"])

        # Find the card with checklist (use uuid since export doesn't include id)
        card_found = False
        for lst in export["lists"]:
            for card in lst["cards"]:
                if card["uuid"] == sample_card["uuid"]:
                    assert len(card["checklists"]) == 1
                    assert card["checklists"][0]["name"] == "Tasks"
                    assert len(card["checklists"][0]["items"]) == 2
                    card_found = True
        assert card_found, "Sample card not found in export"

    def test_export_board_with_comments(self, kanban_db: KanbanDB, sample_card: dict, sample_board: dict):
        """Test exporting a board with comments."""
        kanban_db.create_comment(card_id=sample_card["id"], content="First comment")
        kanban_db.create_comment(card_id=sample_card["id"], content="Second comment")

        export = kanban_db.export_board(board_id=sample_board["id"])

        # Find the card with comments (use uuid since export doesn't include id)
        card_found = False
        for lst in export["lists"]:
            for card in lst["cards"]:
                if card["uuid"] == sample_card["uuid"]:
                    assert len(card["comments"]) == 2
                    card_found = True
        assert card_found, "Sample card not found in export"

    def test_export_board_excludes_archived_by_default(self, kanban_db: KanbanDB, sample_board: dict):
        """Test that archived items are excluded by default."""
        list1 = kanban_db.create_list(board_id=sample_board["id"], name="Active", client_id="list-active")
        list2 = kanban_db.create_list(board_id=sample_board["id"], name="Archived", client_id="list-archived")
        kanban_db.archive_list(list_id=list2["id"], archive=True)

        export = kanban_db.export_board(board_id=sample_board["id"])

        # Should only have the active list
        list_names = [l["name"] for l in export["lists"]]
        assert "Active" in list_names
        assert "Archived" not in list_names

    def test_export_board_includes_archived_when_requested(self, kanban_db: KanbanDB, sample_board: dict):
        """Test that archived items can be included."""
        list1 = kanban_db.create_list(board_id=sample_board["id"], name="Active", client_id="list-active")
        list2 = kanban_db.create_list(board_id=sample_board["id"], name="Archived", client_id="list-archived")
        kanban_db.archive_list(list_id=list2["id"], archive=True)

        export = kanban_db.export_board(board_id=sample_board["id"], include_archived=True)

        list_names = [l["name"] for l in export["lists"]]
        assert "Active" in list_names
        assert "Archived" in list_names

    def test_export_board_not_found(self, kanban_db: KanbanDB):
        """Test exporting a non-existent board raises NotFoundError."""
        with pytest.raises(NotFoundError):
            kanban_db.export_board(board_id=99999)


class TestImportOperations:
    """Tests for board import functionality."""

    def test_import_tldw_format_basic(self, kanban_db: KanbanDB):
        """Test importing a board in tldw format."""
        data = {
            "format": "tldw_kanban_v1",
            "exported_at": "2025-01-01T00:00:00",
            "board": {
                "name": "Imported Board",
                "description": "Test import"
            },
            "labels": [],
            "lists": []
        }

        result = kanban_db.import_board(data=data)

        assert result["board"]["name"] == "Imported Board"
        assert result["board"]["description"] == "Test import"
        assert result["import_stats"]["board_id"] == result["board"]["id"]
        assert result["import_stats"]["lists_imported"] == 0

    def test_import_tldw_format_with_override_name(self, kanban_db: KanbanDB):
        """Test importing with board name override."""
        data = {
            "format": "tldw_kanban_v1",
            "exported_at": "2025-01-01T00:00:00",
            "board": {
                "name": "Original Name"
            },
            "labels": [],
            "lists": []
        }

        result = kanban_db.import_board(data=data, board_name="Override Name")

        assert result["board"]["name"] == "Override Name"

    def test_import_tldw_format_with_lists_and_cards(self, kanban_db: KanbanDB):
        """Test importing a board with nested data."""
        data = {
            "format": "tldw_kanban_v1",
            "exported_at": "2025-01-01T00:00:00",
            "board": {"name": "Full Board"},
            "labels": [
                {"name": "Bug", "color": "red"},
                {"name": "Feature", "color": "green"}
            ],
            "lists": [
                {
                    "name": "Todo",
                    "position": 0,
                    "cards": [
                        {"title": "Card 1", "description": "First card", "position": 0},
                        {"title": "Card 2", "position": 1}
                    ]
                },
                {
                    "name": "Done",
                    "position": 1,
                    "cards": [
                        {"title": "Card 3", "position": 0}
                    ]
                }
            ]
        }

        result = kanban_db.import_board(data=data)

        assert result["import_stats"]["lists_imported"] == 2
        assert result["import_stats"]["cards_imported"] == 3
        assert result["import_stats"]["labels_imported"] == 2

        # Verify the board was created correctly
        board = kanban_db.get_board_with_lists_and_cards(result["board"]["id"])
        assert len(board["lists"]) == 2
        todo_list = next(l for l in board["lists"] if l["name"] == "Todo")
        assert len(todo_list["cards"]) == 2

    def test_import_tldw_format_with_checklists(self, kanban_db: KanbanDB):
        """Test importing a board with checklists."""
        data = {
            "format": "tldw_kanban_v1",
            "exported_at": "2025-01-01T00:00:00",
            "board": {"name": "Checklist Board"},
            "labels": [],
            "lists": [
                {
                    "name": "List 1",
                    "position": 0,
                    "cards": [
                        {
                            "title": "Card with Checklist",
                            "position": 0,
                            "checklists": [
                                {
                                    "name": "Tasks",
                                    "position": 0,
                                    "items": [
                                        {"name": "Task 1", "checked": False, "position": 0},
                                        {"name": "Task 2", "checked": True, "position": 1}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        result = kanban_db.import_board(data=data)

        assert result["import_stats"]["checklists_imported"] == 1
        assert result["import_stats"]["checklist_items_imported"] == 2

    def test_import_tldw_format_with_comments(self, kanban_db: KanbanDB):
        """Test importing a board with comments."""
        data = {
            "format": "tldw_kanban_v1",
            "exported_at": "2025-01-01T00:00:00",
            "board": {"name": "Comment Board"},
            "labels": [],
            "lists": [
                {
                    "name": "List 1",
                    "position": 0,
                    "cards": [
                        {
                            "title": "Card with Comments",
                            "position": 0,
                            "comments": [
                                {"content": "First comment"},
                                {"content": "Second comment"}
                            ]
                        }
                    ]
                }
            ]
        }

        result = kanban_db.import_board(data=data)

        assert result["import_stats"]["comments_imported"] == 2

    def test_import_trello_format_basic(self, kanban_db: KanbanDB):
        """Test importing a board in Trello format."""
        data = {
            "name": "Trello Board",
            "desc": "A board from Trello",
            "lists": [
                {"id": "list1", "name": "To Do", "pos": 1000, "closed": False},
                {"id": "list2", "name": "Doing", "pos": 2000, "closed": False}
            ],
            "cards": [
                {"id": "card1", "name": "Task 1", "desc": "", "idList": "list1", "pos": 100, "closed": False},
                {"id": "card2", "name": "Task 2", "desc": "", "idList": "list1", "pos": 200, "closed": False},
                {"id": "card3", "name": "Task 3", "desc": "", "idList": "list2", "pos": 100, "closed": False}
            ],
            "labels": [],
            "checklists": [],
            "actions": []
        }

        result = kanban_db.import_board(data=data)

        assert result["board"]["name"] == "Trello Board"
        assert result["import_stats"]["lists_imported"] == 2
        assert result["import_stats"]["cards_imported"] == 3

    def test_import_trello_format_with_labels(self, kanban_db: KanbanDB):
        """Test importing a Trello board with labels."""
        data = {
            "name": "Trello Board with Labels",
            "desc": "",
            "lists": [
                {"id": "list1", "name": "Todo", "pos": 1000, "closed": False}
            ],
            "cards": [
                {"id": "card1", "name": "Task", "desc": "", "idList": "list1", "pos": 100, "closed": False, "idLabels": ["label1"]}
            ],
            "labels": [
                {"id": "label1", "name": "Bug", "color": "red"}
            ],
            "checklists": [],
            "actions": []
        }

        result = kanban_db.import_board(data=data)

        assert result["import_stats"]["labels_imported"] == 1

    def test_import_trello_format_with_checklists(self, kanban_db: KanbanDB):
        """Test importing a Trello board with checklists."""
        data = {
            "name": "Trello Board with Checklists",
            "desc": "",
            "lists": [
                {"id": "list1", "name": "Todo", "pos": 1000, "closed": False}
            ],
            "cards": [
                {"id": "card1", "name": "Task", "desc": "", "idList": "list1", "pos": 100, "closed": False, "idChecklists": ["cl1"]}
            ],
            "labels": [],
            "checklists": [
                {
                    "id": "cl1",
                    "idCard": "card1",
                    "name": "Steps",
                    "pos": 1000,
                    "checkItems": [
                        {"name": "Step 1", "state": "incomplete", "pos": 100},
                        {"name": "Step 2", "state": "complete", "pos": 200}
                    ]
                }
            ],
            "actions": []
        }

        result = kanban_db.import_board(data=data)

        assert result["import_stats"]["checklists_imported"] == 1
        assert result["import_stats"]["checklist_items_imported"] == 2

    def test_import_trello_format_color_mapping(self, kanban_db: KanbanDB):
        """Test that Trello colors are mapped correctly."""
        data = {
            "name": "Color Test Board",
            "desc": "",
            "lists": [],
            "cards": [],
            "labels": [
                {"id": "l1", "name": "Sky Label", "color": "sky"},
                {"id": "l2", "name": "Lime Label", "color": "lime"},
                {"id": "l3", "name": "Black Label", "color": "black"}
            ],
            "checklists": [],
            "actions": []
        }

        result = kanban_db.import_board(data=data)

        # Verify the labels were created with mapped colors
        labels = kanban_db.list_labels(board_id=result["board"]["id"])
        color_map = {l["name"]: l["color"] for l in labels}

        # sky -> blue, lime -> green, black -> gray
        assert color_map["Sky Label"] == "blue"
        assert color_map["Lime Label"] == "green"
        assert color_map["Black Label"] == "gray"

    def test_import_invalid_format(self, kanban_db: KanbanDB):
        """Test importing with invalid/unrecognized format."""
        data = {"random": "data"}

        with pytest.raises(InputError) as exc_info:
            kanban_db.import_board(data=data)

        assert "format" in str(exc_info.value).lower() or "unrecognized" in str(exc_info.value).lower()

    def test_import_roundtrip(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict, sample_card: dict):
        """Test exporting and re-importing a board."""
        # Add some data
        label = kanban_db.create_label(board_id=sample_board["id"], name="Test Label", color="blue")
        kanban_db.assign_label_to_card(card_id=sample_card["id"], label_id=label["id"])

        checklist = kanban_db.create_checklist(card_id=sample_card["id"], name="Checklist")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 1")

        kanban_db.create_comment(card_id=sample_card["id"], content="Test comment")

        # Export
        export_data = kanban_db.export_board(board_id=sample_board["id"])

        # Import as new board
        result = kanban_db.import_board(data=export_data, board_name="Imported Copy")

        assert result["board"]["name"] == "Imported Copy"
        assert result["import_stats"]["lists_imported"] >= 1
        assert result["import_stats"]["cards_imported"] >= 1
        assert result["import_stats"]["labels_imported"] == 1
        assert result["import_stats"]["checklists_imported"] == 1
        assert result["import_stats"]["checklist_items_imported"] == 1
        assert result["import_stats"]["comments_imported"] == 1


# =============================================================================
# Phase 3: Bulk Operations Tests
# =============================================================================

class TestBulkOperations:
    """Tests for bulk card operations."""

    def test_bulk_move_cards(self, kanban_db: KanbanDB, sample_board: dict):
        """Test moving multiple cards to a list."""
        list1 = kanban_db.create_list(board_id=sample_board["id"], name="Source", client_id="list-source")
        list2 = kanban_db.create_list(board_id=sample_board["id"], name="Target", client_id="list-target")

        card1 = kanban_db.create_card(list_id=list1["id"], title="Card 1", client_id="card-1")
        card2 = kanban_db.create_card(list_id=list1["id"], title="Card 2", client_id="card-2")
        card3 = kanban_db.create_card(list_id=list1["id"], title="Card 3", client_id="card-3")

        result = kanban_db.bulk_move_cards(
            card_ids=[card1["id"], card2["id"], card3["id"]],
            target_list_id=list2["id"]
        )

        assert result["success"] is True
        assert result["moved_count"] == 3
        assert len(result["cards"]) == 3
        for card in result["cards"]:
            assert card["list_id"] == list2["id"]

    def test_bulk_move_cards_with_position(self, kanban_db: KanbanDB, sample_board: dict):
        """Test bulk move with specific starting position."""
        list1 = kanban_db.create_list(board_id=sample_board["id"], name="Source", client_id="list-source")
        list2 = kanban_db.create_list(board_id=sample_board["id"], name="Target", client_id="list-target")

        # Create existing card in target
        kanban_db.create_card(list_id=list2["id"], title="Existing", client_id="existing")

        card1 = kanban_db.create_card(list_id=list1["id"], title="Card 1", client_id="card-1")
        card2 = kanban_db.create_card(list_id=list1["id"], title="Card 2", client_id="card-2")

        result = kanban_db.bulk_move_cards(
            card_ids=[card1["id"], card2["id"]],
            target_list_id=list2["id"],
            start_position=5
        )

        assert result["success"] is True
        assert result["cards"][0]["position"] == 5
        assert result["cards"][1]["position"] == 6

    def test_bulk_move_cards_empty_list(self, kanban_db: KanbanDB, sample_list: dict):
        """Test bulk move with empty card list."""
        result = kanban_db.bulk_move_cards(card_ids=[], target_list_id=sample_list["id"])
        assert result["success"] is True
        assert result["moved_count"] == 0

    def test_bulk_archive_cards(self, kanban_db: KanbanDB, sample_board: dict):
        """Test archiving multiple cards."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        card1 = kanban_db.create_card(list_id=lst["id"], title="Card 1", client_id="card-1")
        card2 = kanban_db.create_card(list_id=lst["id"], title="Card 2", client_id="card-2")

        result = kanban_db.bulk_archive_cards(card_ids=[card1["id"], card2["id"]], archive=True)

        assert result["success"] is True
        assert result["archived_count"] == 2

        # Verify cards are archived
        cards = kanban_db.list_cards(list_id=lst["id"])
        assert len(cards) == 0  # Hidden by default

        cards_with_archived = kanban_db.list_cards(list_id=lst["id"], include_archived=True)
        assert len(cards_with_archived) == 2

    def test_bulk_unarchive_cards(self, kanban_db: KanbanDB, sample_board: dict):
        """Test unarchiving multiple cards."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        card1 = kanban_db.create_card(list_id=lst["id"], title="Card 1", client_id="card-1")
        card2 = kanban_db.create_card(list_id=lst["id"], title="Card 2", client_id="card-2")

        # Archive first
        kanban_db.bulk_archive_cards(card_ids=[card1["id"], card2["id"]], archive=True)

        # Unarchive
        result = kanban_db.bulk_archive_cards(card_ids=[card1["id"], card2["id"]], archive=False)

        assert result["success"] is True
        assert result["unarchived_count"] == 2

    def test_bulk_delete_cards(self, kanban_db: KanbanDB, sample_board: dict):
        """Test soft deleting multiple cards."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        card1 = kanban_db.create_card(list_id=lst["id"], title="Card 1", client_id="card-1")
        card2 = kanban_db.create_card(list_id=lst["id"], title="Card 2", client_id="card-2")

        result = kanban_db.bulk_delete_cards(card_ids=[card1["id"], card2["id"]])

        assert result["success"] is True
        assert result["deleted_count"] == 2

        # Verify cards are deleted
        cards = kanban_db.list_cards(list_id=lst["id"])
        assert len(cards) == 0

    def test_bulk_label_cards_add(self, kanban_db: KanbanDB, sample_board: dict):
        """Test adding labels to multiple cards."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        card1 = kanban_db.create_card(list_id=lst["id"], title="Card 1", client_id="card-1")
        card2 = kanban_db.create_card(list_id=lst["id"], title="Card 2", client_id="card-2")

        label = kanban_db.create_label(board_id=sample_board["id"], name="Bug", color="red")

        result = kanban_db.bulk_label_cards(
            card_ids=[card1["id"], card2["id"]],
            add_label_ids=[label["id"]]
        )

        assert result["success"] is True
        assert result["updated_count"] == 2

    def test_bulk_label_cards_remove(self, kanban_db: KanbanDB, sample_board: dict):
        """Test removing labels from multiple cards."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        card1 = kanban_db.create_card(list_id=lst["id"], title="Card 1", client_id="card-1")
        card2 = kanban_db.create_card(list_id=lst["id"], title="Card 2", client_id="card-2")

        label = kanban_db.create_label(board_id=sample_board["id"], name="Bug", color="red")
        kanban_db.assign_label_to_card(card_id=card1["id"], label_id=label["id"])
        kanban_db.assign_label_to_card(card_id=card2["id"], label_id=label["id"])

        result = kanban_db.bulk_label_cards(
            card_ids=[card1["id"], card2["id"]],
            remove_label_ids=[label["id"]]
        )

        assert result["success"] is True
        assert result["updated_count"] == 2

    def test_bulk_label_cards_add_and_remove(self, kanban_db: KanbanDB, sample_board: dict):
        """Test adding and removing labels in one operation."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        card1 = kanban_db.create_card(list_id=lst["id"], title="Card 1", client_id="card-1")

        label1 = kanban_db.create_label(board_id=sample_board["id"], name="Bug", color="red")
        label2 = kanban_db.create_label(board_id=sample_board["id"], name="Feature", color="green")

        kanban_db.assign_label_to_card(card_id=card1["id"], label_id=label1["id"])

        result = kanban_db.bulk_label_cards(
            card_ids=[card1["id"]],
            add_label_ids=[label2["id"]],
            remove_label_ids=[label1["id"]]
        )

        assert result["success"] is True
        assert result["updated_count"] == 1


# =============================================================================
# Phase 3: Card Filtering Tests
# =============================================================================

class TestCardFiltering:
    """Tests for card filtering."""

    def test_filter_by_priority(self, kanban_db: KanbanDB, sample_board: dict):
        """Test filtering cards by priority."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        kanban_db.create_card(list_id=lst["id"], title="High", client_id="card-1", priority="high")
        kanban_db.create_card(list_id=lst["id"], title="Low", client_id="card-2", priority="low")
        kanban_db.create_card(list_id=lst["id"], title="None", client_id="card-3")

        cards, total = kanban_db.get_board_cards_filtered(
            board_id=sample_board["id"],
            priority="high"
        )

        assert total == 1
        assert cards[0]["title"] == "High"

    def test_filter_by_label(self, kanban_db: KanbanDB, sample_board: dict):
        """Test filtering cards by label."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        card1 = kanban_db.create_card(list_id=lst["id"], title="Bug Card", client_id="card-1")
        kanban_db.create_card(list_id=lst["id"], title="No Label", client_id="card-2")

        label = kanban_db.create_label(board_id=sample_board["id"], name="Bug", color="red")
        kanban_db.assign_label_to_card(card_id=card1["id"], label_id=label["id"])

        cards, total = kanban_db.get_board_cards_filtered(
            board_id=sample_board["id"],
            label_ids=[label["id"]]
        )

        assert total == 1
        assert cards[0]["title"] == "Bug Card"

    def test_filter_has_due_date(self, kanban_db: KanbanDB, sample_board: dict):
        """Test filtering cards by due date presence."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        kanban_db.create_card(list_id=lst["id"], title="With Due", client_id="card-1", due_date="2025-12-31")
        kanban_db.create_card(list_id=lst["id"], title="No Due", client_id="card-2")

        cards, total = kanban_db.get_board_cards_filtered(
            board_id=sample_board["id"],
            has_due_date=True
        )

        assert total == 1
        assert cards[0]["title"] == "With Due"

        cards, total = kanban_db.get_board_cards_filtered(
            board_id=sample_board["id"],
            has_due_date=False
        )

        assert total == 1
        assert cards[0]["title"] == "No Due"

    def test_filter_has_checklist(self, kanban_db: KanbanDB, sample_board: dict):
        """Test filtering cards by checklist presence."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        card1 = kanban_db.create_card(list_id=lst["id"], title="With Checklist", client_id="card-1")
        kanban_db.create_card(list_id=lst["id"], title="No Checklist", client_id="card-2")

        kanban_db.create_checklist(card_id=card1["id"], name="Tasks")

        cards, total = kanban_db.get_board_cards_filtered(
            board_id=sample_board["id"],
            has_checklist=True
        )

        assert total == 1
        assert cards[0]["title"] == "With Checklist"

    def test_filter_is_complete(self, kanban_db: KanbanDB, sample_board: dict):
        """Test filtering cards by checklist completion."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        card1 = kanban_db.create_card(list_id=lst["id"], title="Complete", client_id="card-1")
        card2 = kanban_db.create_card(list_id=lst["id"], title="Incomplete", client_id="card-2")

        cl1 = kanban_db.create_checklist(card_id=card1["id"], name="Tasks")
        kanban_db.create_checklist_item(checklist_id=cl1["id"], name="Item 1", checked=True)

        cl2 = kanban_db.create_checklist(card_id=card2["id"], name="Tasks")
        kanban_db.create_checklist_item(checklist_id=cl2["id"], name="Item 1", checked=False)

        cards, total = kanban_db.get_board_cards_filtered(
            board_id=sample_board["id"],
            is_complete=True
        )

        assert total == 1
        assert cards[0]["title"] == "Complete"

    def test_filter_pagination(self, kanban_db: KanbanDB, sample_board: dict):
        """Test filtering with pagination."""
        lst = kanban_db.create_list(board_id=sample_board["id"], name="List", client_id="list-1")

        for i in range(10):
            kanban_db.create_card(list_id=lst["id"], title=f"Card {i}", client_id=f"card-{i}")

        cards, total = kanban_db.get_board_cards_filtered(
            board_id=sample_board["id"],
            limit=3,
            offset=0
        )

        assert total == 10
        assert len(cards) == 3

        cards, total = kanban_db.get_board_cards_filtered(
            board_id=sample_board["id"],
            limit=3,
            offset=9
        )

        assert total == 10
        assert len(cards) == 1  # Last page


# =============================================================================
# Phase 3: Toggle All Checklist Items Tests
# =============================================================================

class TestToggleAllChecklistItems:
    """Tests for toggle all checklist items."""

    def test_toggle_all_check(self, kanban_db: KanbanDB, sample_card: dict):
        """Test checking all items in a checklist."""
        checklist = kanban_db.create_checklist(card_id=sample_card["id"], name="Tasks")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 1")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 2")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 3")

        result = kanban_db.toggle_all_checklist_items(checklist_id=checklist["id"], checked=True)

        assert result["name"] == "Tasks"
        assert len(result["items"]) == 3
        for item in result["items"]:
            assert item["checked"] is True
            assert item["checked_at"] is not None

    def test_toggle_all_uncheck(self, kanban_db: KanbanDB, sample_card: dict):
        """Test unchecking all items in a checklist."""
        checklist = kanban_db.create_checklist(card_id=sample_card["id"], name="Tasks")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 1", checked=True)
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 2", checked=True)

        result = kanban_db.toggle_all_checklist_items(checklist_id=checklist["id"], checked=False)

        assert len(result["items"]) == 2
        for item in result["items"]:
            assert item["checked"] is False
            assert item["checked_at"] is None

    def test_toggle_all_partial(self, kanban_db: KanbanDB, sample_card: dict):
        """Test toggle all when some items are already in target state."""
        checklist = kanban_db.create_checklist(card_id=sample_card["id"], name="Tasks")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 1", checked=True)
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 2", checked=False)

        result = kanban_db.toggle_all_checklist_items(checklist_id=checklist["id"], checked=True)

        for item in result["items"]:
            assert item["checked"] is True


# =============================================================================
# Phase 3: Enhanced Card Copy Tests
# =============================================================================

class TestCopyCardWithChecklists:
    """Tests for copying cards with checklists."""

    def test_copy_with_checklists(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict, sample_card: dict):
        """Test copying a card with checklists."""
        checklist = kanban_db.create_checklist(card_id=sample_card["id"], name="Tasks")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 1")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 2", checked=True)

        copied = kanban_db.copy_card_with_checklists(
            card_id=sample_card["id"],
            target_list_id=sample_list["id"],
            new_client_id="copy-1",
            copy_checklists=True
        )

        assert copied["id"] != sample_card["id"]
        assert copied["title"].startswith("Copy of")

        # Verify checklists were copied
        checklists = kanban_db.list_checklists(card_id=copied["id"])
        assert len(checklists) == 1
        assert checklists[0]["name"] == "Tasks"

        items = kanban_db.list_checklist_items(checklist_id=checklists[0]["id"])
        assert len(items) == 2

    def test_copy_without_checklists(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict, sample_card: dict):
        """Test copying a card without checklists."""
        checklist = kanban_db.create_checklist(card_id=sample_card["id"], name="Tasks")
        kanban_db.create_checklist_item(checklist_id=checklist["id"], name="Item 1")

        copied = kanban_db.copy_card_with_checklists(
            card_id=sample_card["id"],
            target_list_id=sample_list["id"],
            new_client_id="copy-1",
            copy_checklists=False
        )

        # Verify checklists were NOT copied
        checklists = kanban_db.list_checklists(card_id=copied["id"])
        assert len(checklists) == 0

    def test_copy_with_labels(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict, sample_card: dict):
        """Test copying a card with labels."""
        label = kanban_db.create_label(board_id=sample_board["id"], name="Bug", color="red")
        kanban_db.assign_label_to_card(card_id=sample_card["id"], label_id=label["id"])

        copied = kanban_db.copy_card_with_checklists(
            card_id=sample_card["id"],
            target_list_id=sample_list["id"],
            new_client_id="copy-1",
            copy_labels=True
        )

        # Verify labels were copied
        labels = kanban_db.get_card_labels(card_id=copied["id"])
        assert len(labels) == 1
        assert labels[0]["name"] == "Bug"

    def test_copy_without_labels(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict, sample_card: dict):
        """Test copying a card without labels."""
        label = kanban_db.create_label(board_id=sample_board["id"], name="Bug", color="red")
        kanban_db.assign_label_to_card(card_id=sample_card["id"], label_id=label["id"])

        copied = kanban_db.copy_card_with_checklists(
            card_id=sample_card["id"],
            target_list_id=sample_list["id"],
            new_client_id="copy-1",
            copy_labels=False
        )

        # Verify labels were NOT copied
        labels = kanban_db.get_card_labels(card_id=copied["id"])
        assert len(labels) == 0

    def test_copy_with_custom_title(self, kanban_db: KanbanDB, sample_list: dict, sample_card: dict):
        """Test copying a card with custom title."""
        copied = kanban_db.copy_card_with_checklists(
            card_id=sample_card["id"],
            target_list_id=sample_list["id"],
            new_client_id="copy-1",
            new_title="My Custom Title"
        )

        assert copied["title"] == "My Custom Title"


# =============================================================================
# Search Operations Tests (Phase 4)
# =============================================================================

class TestSearchOperations:
    """Tests for FTS5 search functionality."""

    def test_search_cards_basic(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict):
        """Test basic card search."""
        # Create cards with searchable content
        card1 = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Machine Learning Tutorial",
            client_id="search-card-1",
            description="An introduction to neural networks and deep learning"
        )
        card2 = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Python Programming Guide",
            client_id="search-card-2",
            description="Learn Python basics and advanced concepts"
        )
        card3 = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Data Science with Machine Learning",
            client_id="search-card-3",
            description="Using ML for data analysis"
        )

        # Search for "machine learning"
        results, total = kanban_db.search_cards(query="machine learning")
        assert total == 2
        assert len(results) == 2
        titles = [r["title"] for r in results]
        assert "Machine Learning Tutorial" in titles
        assert "Data Science with Machine Learning" in titles

    def test_search_cards_with_board_filter(self, kanban_db: KanbanDB):
        """Test search with board filter."""
        # Create two boards
        board1 = kanban_db.create_board(name="Board 1", client_id="search-board-1")
        board2 = kanban_db.create_board(name="Board 2", client_id="search-board-2")

        list1 = kanban_db.create_list(board_id=board1["id"], name="List 1", client_id="search-list-1")
        list2 = kanban_db.create_list(board_id=board2["id"], name="List 2", client_id="search-list-2")

        kanban_db.create_card(
            list_id=list1["id"],
            title="Alpha Project",
            client_id="alpha-card",
            description="Alpha description"
        )
        kanban_db.create_card(
            list_id=list2["id"],
            title="Alpha Task",
            client_id="alpha-task",
            description="Another alpha item"
        )

        # Search without filter - should find both
        results, total = kanban_db.search_cards(query="alpha")
        assert total == 2

        # Search with board filter - should find only one
        results, total = kanban_db.search_cards(query="alpha", board_id=board1["id"])
        assert total == 1
        assert results[0]["title"] == "Alpha Project"

    def test_search_cards_with_priority_filter(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict):
        """Test search with priority filter."""
        kanban_db.create_card(
            list_id=sample_list["id"],
            title="Urgent Task Beta",
            client_id="beta-urgent",
            priority="high"
        )
        kanban_db.create_card(
            list_id=sample_list["id"],
            title="Regular Task Beta",
            client_id="beta-regular",
            priority="low"
        )

        # Search with priority filter
        results, total = kanban_db.search_cards(query="beta", priority="high")
        assert total == 1
        assert results[0]["title"] == "Urgent Task Beta"

    def test_search_cards_with_label_filter(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict):
        """Test search with label filter."""
        label1 = kanban_db.create_label(board_id=sample_board["id"], name="Bug", color="red")
        label2 = kanban_db.create_label(board_id=sample_board["id"], name="Feature", color="green")

        card1 = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Gamma Issue",
            client_id="gamma-1"
        )
        card2 = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Gamma Enhancement",
            client_id="gamma-2"
        )

        kanban_db.assign_label_to_card(card_id=card1["id"], label_id=label1["id"])
        kanban_db.assign_label_to_card(card_id=card2["id"], label_id=label2["id"])

        # Search with label filter
        results, total = kanban_db.search_cards(query="gamma", label_ids=[label1["id"]])
        assert total == 1
        assert results[0]["title"] == "Gamma Issue"

    def test_search_cards_excludes_archived(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict):
        """Test that archived cards are excluded by default."""
        card1 = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Active Delta Card",
            client_id="delta-active"
        )
        card2 = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Archived Delta Card",
            client_id="delta-archived"
        )
        kanban_db.archive_card(card_id=card2["id"])

        # Search without include_archived
        results, total = kanban_db.search_cards(query="delta")
        assert total == 1
        assert results[0]["title"] == "Active Delta Card"

        # Search with include_archived
        results, total = kanban_db.search_cards(query="delta", include_archived=True)
        assert total == 2

    def test_search_cards_excludes_deleted(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict):
        """Test that deleted cards are always excluded."""
        card1 = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Active Epsilon Card",
            client_id="epsilon-active"
        )
        card2 = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Deleted Epsilon Card",
            client_id="epsilon-deleted"
        )
        kanban_db.delete_card(card_id=card2["id"])

        # Search should only find active card
        results, total = kanban_db.search_cards(query="epsilon")
        assert total == 1
        assert results[0]["title"] == "Active Epsilon Card"

    def test_search_cards_returns_enriched_data(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict):
        """Test that search results include board_name, list_name, and labels."""
        label = kanban_db.create_label(board_id=sample_board["id"], name="Important", color="yellow")
        card = kanban_db.create_card(
            list_id=sample_list["id"],
            title="Enriched Zeta Card",
            client_id="zeta-card"
        )
        kanban_db.assign_label_to_card(card_id=card["id"], label_id=label["id"])

        results, total = kanban_db.search_cards(query="zeta")
        assert total == 1
        result = results[0]

        # Check enriched fields
        assert result["board_name"] == sample_board["name"]
        assert result["list_name"] == sample_list["name"]
        assert len(result["labels"]) == 1
        assert result["labels"][0]["name"] == "Important"

    def test_search_cards_pagination(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict):
        """Test search pagination."""
        # Create 5 cards
        for i in range(5):
            kanban_db.create_card(
                list_id=sample_list["id"],
                title=f"Pagination Test Eta {i}",
                client_id=f"eta-{i}"
            )

        # Get first page
        results, total = kanban_db.search_cards(query="eta", limit=2, offset=0)
        assert total == 5
        assert len(results) == 2

        # Get second page
        results, total = kanban_db.search_cards(query="eta", limit=2, offset=2)
        assert total == 5
        assert len(results) == 2

        # Get last page
        results, total = kanban_db.search_cards(query="eta", limit=2, offset=4)
        assert total == 5
        assert len(results) == 1

    def test_search_cards_empty_query_raises(self, kanban_db: KanbanDB):
        """Test that empty query raises InputError."""
        from tldw_Server_API.app.core.DB_Management.Kanban_DB import InputError
        import pytest

        with pytest.raises(InputError):
            kanban_db.search_cards(query="")

        with pytest.raises(InputError):
            kanban_db.search_cards(query="   ")

    def test_search_cards_no_results(self, kanban_db: KanbanDB, sample_board: dict, sample_list: dict):
        """Test search with no matching results."""
        kanban_db.create_card(
            list_id=sample_list["id"],
            title="Theta Card",
            client_id="theta-card"
        )

        results, total = kanban_db.search_cards(query="nonexistent xyz 12345")
        assert total == 0
        assert len(results) == 0
