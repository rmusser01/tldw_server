# tldw_Server_API/tests/kanban/test_kanban_performance.py
"""
Performance tests for Kanban module.

These tests measure the performance of:
- Board load times with varying amounts of data
- Search operations with different query patterns
- Bulk operations performance
- Database throughput under load

Tests use pytest-benchmark when available, otherwise fall back to timing assertions.
"""
import tempfile
import time
import uuid
from typing import Any, Callable, Generator
import statistics

import pytest

from tldw_Server_API.app.core.DB_Management.Kanban_DB import KanbanDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


# Performance thresholds (in seconds)
BOARD_LOAD_THRESHOLD_SMALL = 0.5  # Board with ~10 cards
BOARD_LOAD_THRESHOLD_MEDIUM = 1.0  # Board with ~100 cards
BOARD_LOAD_THRESHOLD_LARGE = 3.0  # Board with ~500 cards
SEARCH_THRESHOLD = 0.5  # FTS search should be fast
BULK_OPERATION_THRESHOLD = 2.0  # Bulk operations


class _TimingBenchmarkFallback:
    """Small benchmark adapter used when pytest-benchmark is not registered."""

    def __init__(self) -> None:
        self.stats: dict[str, float] = {}

    def __call__(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.stats = {
            "mean": elapsed,
            "min": elapsed,
            "max": elapsed,
        }
        return result

    def pedantic(
        self,
        func: Callable[..., Any],
        *,
        iterations: int = 1,
        rounds: int = 1,
        **_: Any,
    ) -> Any:
        result = None
        times: list[float] = []
        for _round in range(rounds):
            start = time.perf_counter()
            for _iteration in range(iterations):
                result = func()
            times.append((time.perf_counter() - start) / iterations)
        self.stats = {
            "mean": statistics.mean(times),
            "min": min(times),
            "max": max(times),
        }
        return result


@pytest.fixture
def benchmark_runner(request: pytest.FixtureRequest) -> Any:
    """Return pytest-benchmark's fixture when loaded, otherwise use a timing fallback."""
    try:
        return request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        return _TimingBenchmarkFallback()


@pytest.fixture
def perf_db(monkeypatch: pytest.MonkeyPatch) -> Generator[KanbanDB, None, None]:
    """Create a KanbanDB instance for performance testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("USER_DB_BASE_DIR", tmpdir)
        db_path = DatabasePaths.get_kanban_db_path("perf_test_user")
        db = KanbanDB(db_path=str(db_path), user_id="perf_test_user")
        yield db


@pytest.fixture
def populated_board_small(perf_db: KanbanDB) -> dict:
    """Create a board with ~10 cards for small load testing."""
    board = perf_db.create_board(
        name="Small Performance Board",
        client_id=f"perf-small-{uuid.uuid4().hex[:8]}",
        description="Board for small load performance testing"
    )

    # Create 3 lists with 3-4 cards each (total ~10)
    for list_idx in range(3):
        lst = perf_db.create_list(
            board_id=board["id"],
            name=f"List {list_idx}",
            client_id=f"list-{list_idx}-{uuid.uuid4().hex[:8]}"
        )

        for card_idx in range(4):
            perf_db.create_card(
                list_id=lst["id"],
                title=f"Task {list_idx}-{card_idx}: Implement feature for performance testing",
                client_id=f"card-{list_idx}-{card_idx}-{uuid.uuid4().hex[:8]}",
                description=f"This is card {card_idx} in list {list_idx}. It contains searchable text about testing.",
                priority="medium" if card_idx % 2 == 0 else "high"
            )

    return board


@pytest.fixture
def populated_board_medium(perf_db: KanbanDB) -> dict:
    """Create a board with ~100 cards for medium load testing."""
    board = perf_db.create_board(
        name="Medium Performance Board",
        client_id=f"perf-medium-{uuid.uuid4().hex[:8]}",
        description="Board for medium load performance testing"
    )

    # Create 5 lists with 20 cards each (total 100)
    for list_idx in range(5):
        lst = perf_db.create_list(
            board_id=board["id"],
            name=f"List {list_idx}",
            client_id=f"list-m-{list_idx}-{uuid.uuid4().hex[:8]}"
        )

        for card_idx in range(20):
            perf_db.create_card(
                list_id=lst["id"],
                title=f"Task {list_idx}-{card_idx}: Complex task with performance metrics",
                client_id=f"card-m-{list_idx}-{card_idx}-{uuid.uuid4().hex[:8]}",
                description=f"Description for card {card_idx} in list {list_idx}. "
                           f"Contains keywords: performance, testing, benchmark, kanban, "
                           f"{'priority' if card_idx % 3 == 0 else 'feature'}",
                priority=["low", "medium", "high", "urgent"][card_idx % 4]
            )

    return board


@pytest.fixture
def populated_board_large(perf_db: KanbanDB) -> dict:
    """Create a board with ~500 cards for large load testing."""
    board = perf_db.create_board(
        name="Large Performance Board",
        client_id=f"perf-large-{uuid.uuid4().hex[:8]}",
        description="Board for large load performance testing"
    )

    # Create 10 lists with 50 cards each (total 500)
    for list_idx in range(10):
        lst = perf_db.create_list(
            board_id=board["id"],
            name=f"List {list_idx}",
            client_id=f"list-l-{list_idx}-{uuid.uuid4().hex[:8]}"
        )

        for card_idx in range(50):
            perf_db.create_card(
                list_id=lst["id"],
                title=f"Task {list_idx}-{card_idx}: Enterprise feature implementation",
                client_id=f"card-l-{list_idx}-{card_idx}-{uuid.uuid4().hex[:8]}",
                description=f"Detailed description for card {card_idx} in list {list_idx}. "
                           f"This card tracks {'bug' if card_idx % 5 == 0 else 'feature'} work. "
                           f"Keywords: enterprise, scale, performance, testing",
                priority=["low", "medium", "high", "urgent"][card_idx % 4]
            )

    return board


class TestBoardLoadPerformance:
    """Test board load performance with varying data sizes."""

    def test_load_small_board_performance(self, perf_db: KanbanDB, populated_board_small: dict):
        """Loading a small board (~10 cards) should be fast."""
        board_id = populated_board_small["id"]

        # Warm up cache
        perf_db.get_board_with_lists_and_cards(board_id)

        # Measure load time over multiple iterations
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = perf_db.get_board_with_lists_and_cards(board_id)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        assert result is not None
        assert "lists" in result
        assert avg_time < BOARD_LOAD_THRESHOLD_SMALL, (
            f"Small board load took {avg_time:.3f}s, expected < {BOARD_LOAD_THRESHOLD_SMALL}s"
        )

    def test_load_medium_board_performance(self, perf_db: KanbanDB, populated_board_medium: dict):
        """Loading a medium board (~100 cards) should be reasonably fast."""
        board_id = populated_board_medium["id"]

        # Warm up
        perf_db.get_board_with_lists_and_cards(board_id)

        times = []
        for _ in range(3):
            start = time.perf_counter()
            result = perf_db.get_board_with_lists_and_cards(board_id)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        assert result is not None
        assert result["total_cards"] == 100
        assert avg_time < BOARD_LOAD_THRESHOLD_MEDIUM, (
            f"Medium board load took {avg_time:.3f}s, expected < {BOARD_LOAD_THRESHOLD_MEDIUM}s"
        )

    def test_load_large_board_performance(self, perf_db: KanbanDB, populated_board_large: dict):
        """Loading a large board (~500 cards) should complete within threshold."""
        board_id = populated_board_large["id"]

        # Warm up
        perf_db.get_board_with_lists_and_cards(board_id)

        times = []
        for _ in range(3):
            start = time.perf_counter()
            result = perf_db.get_board_with_lists_and_cards(board_id)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        assert result is not None
        assert result["total_cards"] == 500
        assert avg_time < BOARD_LOAD_THRESHOLD_LARGE, (
            f"Large board load took {avg_time:.3f}s, expected < {BOARD_LOAD_THRESHOLD_LARGE}s"
        )


class TestSearchPerformance:
    """Test FTS search performance."""

    def test_search_performance_small_dataset(self, perf_db: KanbanDB, populated_board_small: dict):
        """Search in small dataset should be very fast."""
        # Warm up
        perf_db.search_cards("testing")

        times = []
        for _ in range(5):
            start = time.perf_counter()
            results, total = perf_db.search_cards("testing", limit=50)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        assert total > 0, "Expected some search results"
        assert avg_time < SEARCH_THRESHOLD / 2, (
            f"Small dataset search took {avg_time:.3f}s, expected < {SEARCH_THRESHOLD / 2}s"
        )

    def test_search_performance_medium_dataset(self, perf_db: KanbanDB, populated_board_medium: dict):
        """Search in medium dataset should be fast."""
        # Warm up
        perf_db.search_cards("performance")

        times = []
        for _ in range(5):
            start = time.perf_counter()
            results, total = perf_db.search_cards("performance", limit=50)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        assert total > 0, "Expected some search results"
        assert avg_time < SEARCH_THRESHOLD, (
            f"Medium dataset search took {avg_time:.3f}s, expected < {SEARCH_THRESHOLD}s"
        )

    def test_search_performance_large_dataset(self, perf_db: KanbanDB, populated_board_large: dict):
        """Search in large dataset should complete within threshold."""
        # Warm up
        perf_db.search_cards("enterprise")

        times = []
        for _ in range(3):
            start = time.perf_counter()
            results, total = perf_db.search_cards("enterprise", limit=100)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        assert total > 0, "Expected some search results"
        assert avg_time < SEARCH_THRESHOLD * 2, (
            f"Large dataset search took {avg_time:.3f}s, expected < {SEARCH_THRESHOLD * 2}s"
        )

    def test_search_with_filters_performance(self, perf_db: KanbanDB, populated_board_large: dict):
        """Search with additional filters should still be fast."""
        board_id = populated_board_large["id"]

        # Warm up
        perf_db.search_cards("feature", board_id=board_id, priority="high")

        times = []
        for _ in range(3):
            start = time.perf_counter()
            results, total = perf_db.search_cards(
                "feature",
                board_id=board_id,
                priority="high",
                limit=50
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        assert avg_time < SEARCH_THRESHOLD * 1.5, (
            f"Filtered search took {avg_time:.3f}s, expected < {SEARCH_THRESHOLD * 1.5}s"
        )


class TestBulkOperationsPerformance:
    """Test bulk operation performance."""

    def test_bulk_move_cards_performance(self, perf_db: KanbanDB, populated_board_medium: dict):
        """Bulk moving cards should be efficient."""
        board_id = populated_board_medium["id"]

        # Get all lists and cards
        board = perf_db.get_board_with_lists_and_cards(board_id)
        source_list = board["lists"][0]
        target_list = board["lists"][1]
        card_ids = [card["id"] for card in source_list["cards"][:10]]

        start = time.perf_counter()
        result = perf_db.bulk_move_cards(
            card_ids=card_ids,
            target_list_id=target_list["id"]
        )
        elapsed = time.perf_counter() - start

        assert result["success"] is True
        assert result["moved_count"] == 10
        assert elapsed < BULK_OPERATION_THRESHOLD, (
            f"Bulk move took {elapsed:.3f}s, expected < {BULK_OPERATION_THRESHOLD}s"
        )

    def test_bulk_archive_cards_performance(self, perf_db: KanbanDB, populated_board_medium: dict):
        """Bulk archiving cards should be efficient."""
        board_id = populated_board_medium["id"]

        # Get cards to archive
        board = perf_db.get_board_with_lists_and_cards(board_id)
        card_ids = []
        for lst in board["lists"][:3]:
            card_ids.extend([card["id"] for card in lst["cards"][:5]])

        start = time.perf_counter()
        result = perf_db.bulk_archive_cards(card_ids=card_ids, archive=True)
        elapsed = time.perf_counter() - start

        assert result["success"] is True
        assert result.get("archived_count", 0) > 0
        assert elapsed < BULK_OPERATION_THRESHOLD, (
            f"Bulk archive took {elapsed:.3f}s, expected < {BULK_OPERATION_THRESHOLD}s"
        )

    def test_bulk_delete_cards_performance(self, perf_db: KanbanDB, populated_board_medium: dict):
        """Bulk deleting cards should be efficient."""
        board_id = populated_board_medium["id"]

        # Get cards to delete
        board = perf_db.get_board_with_lists_and_cards(board_id)
        card_ids = []
        for lst in board["lists"][3:]:  # Use different lists than archive test
            card_ids.extend([card["id"] for card in lst["cards"][:5]])

        start = time.perf_counter()
        result = perf_db.bulk_delete_cards(card_ids=card_ids, hard_delete=False)
        elapsed = time.perf_counter() - start

        assert result["success"] is True
        assert result["deleted_count"] > 0
        assert elapsed < BULK_OPERATION_THRESHOLD, (
            f"Bulk delete took {elapsed:.3f}s, expected < {BULK_OPERATION_THRESHOLD}s"
        )


class TestThroughput:
    """Test database throughput under simulated load."""

    def test_card_creation_throughput(self, perf_db: KanbanDB):
        """Measure card creation throughput."""
        board = perf_db.create_board(
            name="Throughput Test Board",
            client_id=f"throughput-{uuid.uuid4().hex[:8]}"
        )
        lst = perf_db.create_list(
            board_id=board["id"],
            name="Throughput List",
            client_id=f"throughput-list-{uuid.uuid4().hex[:8]}"
        )

        num_cards = 50
        start = time.perf_counter()

        for i in range(num_cards):
            perf_db.create_card(
                list_id=lst["id"],
                title=f"Throughput card {i}",
                client_id=f"throughput-card-{i}-{uuid.uuid4().hex[:8]}"
            )

        elapsed = time.perf_counter() - start
        throughput = num_cards / elapsed

        # Should be able to create at least 20 cards/second
        assert throughput > 20, (
            f"Card creation throughput: {throughput:.1f} cards/s, expected > 20 cards/s"
        )

    def test_read_throughput(self, perf_db: KanbanDB, populated_board_medium: dict):
        """Measure read throughput."""
        board_id = populated_board_medium["id"]

        num_reads = 20
        start = time.perf_counter()

        for _ in range(num_reads):
            perf_db.get_board_with_lists_and_cards(board_id)

        elapsed = time.perf_counter() - start
        throughput = num_reads / elapsed

        # Should be able to do at least 5 full board reads/second
        assert throughput > 5, (
            f"Read throughput: {throughput:.1f} reads/s, expected > 5 reads/s"
        )

    def test_update_throughput(self, perf_db: KanbanDB, populated_board_small: dict):
        """Measure update throughput."""
        board_id = populated_board_small["id"]
        board = perf_db.get_board_with_lists_and_cards(board_id)

        cards = []
        for lst in board["lists"]:
            cards.extend(lst["cards"])

        num_updates = min(20, len(cards))
        start = time.perf_counter()

        for i in range(num_updates):
            card = cards[i % len(cards)]
            perf_db.update_card(
                card_id=card["id"],
                title=f"Updated title {i}"
            )

        elapsed = time.perf_counter() - start
        throughput = num_updates / elapsed

        # Should be able to do at least 15 updates/second
        assert throughput > 15, (
            f"Update throughput: {throughput:.1f} updates/s, expected > 15 updates/s"
        )


class TestExportImportPerformance:
    """Test export/import performance with varying data sizes."""

    def test_export_medium_board_performance(self, perf_db: KanbanDB, populated_board_medium: dict):
        """Exporting a medium board should be fast."""
        board_id = populated_board_medium["id"]

        times = []
        for _ in range(3):
            start = time.perf_counter()
            export_data = perf_db.export_board(board_id)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        assert export_data is not None
        assert "board" in export_data
        assert avg_time < 1.0, (
            f"Export took {avg_time:.3f}s, expected < 1.0s"
        )

    def test_import_board_performance(self, perf_db: KanbanDB, populated_board_medium: dict):
        """Importing a board should complete within reasonable time."""
        board_id = populated_board_medium["id"]

        # Export first
        export_data = perf_db.export_board(board_id)

        # Measure import time
        start = time.perf_counter()
        result = perf_db.import_board(
            data=export_data,
            board_name="Imported Performance Board"
        )
        elapsed = time.perf_counter() - start

        assert result is not None
        assert "board" in result
        # Import is slower due to all the inserts
        assert elapsed < 5.0, (
            f"Import took {elapsed:.3f}s, expected < 5.0s"
        )


@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmarks using pytest-benchmark when loaded, with a timing fallback."""

    def test_benchmark_board_load(self, benchmark_runner, perf_db: KanbanDB, populated_board_medium: dict):
        """Benchmark board load operation."""
        board_id = populated_board_medium["id"]

        result = benchmark_runner(perf_db.get_board_with_lists_and_cards, board_id)
        assert result is not None

    def test_benchmark_search(self, benchmark_runner, perf_db: KanbanDB, populated_board_medium: dict):
        """Benchmark search operation."""
        def search_op():
            return perf_db.search_cards("performance", limit=50)

        result = benchmark_runner(search_op)
        assert result[0] is not None  # results

    def test_benchmark_card_create(self, benchmark_runner, perf_db: KanbanDB):
        """Benchmark card creation."""
        # Create a fresh board and list for each benchmark run to avoid hitting limits
        board = perf_db.create_board(
            name="Benchmark Board",
            client_id=f"bench-board-{uuid.uuid4().hex[:8]}"
        )
        lst = perf_db.create_list(
            board_id=board["id"],
            name="Benchmark List",
            client_id=f"bench-list-{uuid.uuid4().hex[:8]}"
        )
        list_id = lst["id"]

        counter = [0]  # Use list to allow modification in closure

        def create_card():

            counter[0] += 1
            return perf_db.create_card(
                list_id=list_id,
                title=f"Benchmark card {counter[0]}",
                client_id=f"bench-card-{counter[0]}-{uuid.uuid4().hex[:8]}"
            )

        # Use pedantic mode with fewer iterations to stay under the per-list card limit.
        result = benchmark_runner.pedantic(create_card, iterations=10, rounds=10)
        assert result is not None
