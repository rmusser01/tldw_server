"""
Tests for the Evaluations connection pool.

Tests connection pooling, health monitoring, and aiosqlite compatibility
preparation for the database layer.
"""

import pytest
import sqlite3
import tempfile
import time
import asyncio
from pathlib import Path
from threading import Thread
from unittest.mock import patch

from tldw_Server_API.app.core.Evaluations.connection_pool import (
    ConnectionPool,
    PooledConnection,
    EvaluationsConnectionManager,
    get_connection,
    get_connection_health,
    get_connection_stats
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    # Initialize test database. sqlite3's context manager commits/rolls back
    # but does not close the connection, which leaves the file locked on Windows.
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()

    yield db_path

    # Cleanup
    try:
        Path(db_path).unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def connection_pool(temp_db):
    """Create a connection pool for testing."""
    pool = ConnectionPool(
        db_path=temp_db,
        pool_size=3,
        max_overflow=2,
        pool_timeout=5.0,
        enable_monitoring=True
    )
    yield pool
    pool.shutdown()


class TestConnectionPool:
    """Test connection pool functionality."""

    def test_pool_initialization(self, connection_pool):

        """Test that connection pool initializes correctly."""
        stats = connection_pool.get_statistics()
        assert stats.total_connections >= 3  # Should pre-create core pool
        assert stats.idle_connections >= 3
        assert stats.active_connections == 0

    def test_connection_checkout_return(self, connection_pool):

        """Test basic connection checkout and return."""
        with connection_pool.get_connection() as conn:
            assert isinstance(conn, PooledConnection)
            assert conn.in_use

            # Test database operation
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

        # Connection should be returned to pool
        stats = connection_pool.get_statistics()
        assert stats.active_connections == 0
        assert stats.checkout_count == 1

    def test_multiple_connections(self, connection_pool):

        """Test multiple concurrent connections."""
        connections = []

        # Checkout multiple connections
        for _ in range(3):
            conn_context = connection_pool.get_connection()
            conn = conn_context.__enter__()
            connections.append((conn, conn_context))

        stats = connection_pool.get_statistics()
        assert stats.active_connections == 3

        # Return all connections
        for conn, context in connections:
            context.__exit__(None, None, None)

        stats = connection_pool.get_statistics()
        assert stats.active_connections == 0

    def test_overflow_connections(self, connection_pool):

        """Test overflow connection creation."""
        connections = []

        try:
            # Checkout more than pool size
            for _ in range(5):  # pool_size=3, max_overflow=2
                conn_context = connection_pool.get_connection()
                conn = conn_context.__enter__()
                connections.append((conn, conn_context))

            stats = connection_pool.get_statistics()
            assert stats.active_connections == 5
            assert stats.total_connections == 5

        finally:
            # Return all connections
            for conn, context in connections:
                context.__exit__(None, None, None)

    def test_pool_exhaustion(self, connection_pool):

        """Test pool exhaustion timeout."""
        connections = []

        try:
            # Exhaust the pool
            for _ in range(5):  # pool_size + max_overflow
                conn_context = connection_pool.get_connection()
                conn = conn_context.__enter__()
                connections.append((conn, conn_context))

            # This should timeout
            start_time = time.time()
            with pytest.raises(TimeoutError):
                with connection_pool.get_connection():
                    pass

            elapsed = time.time() - start_time
            assert elapsed >= 4.5  # Should timeout after ~5 seconds

        finally:
            # Return all connections
            for conn, context in connections:
                context.__exit__(None, None, None)

    def test_stale_connection_cleanup(self, connection_pool):

        """Test that stale connections are cleaned up."""
        # Mock time to make connections appear stale
        with patch('time.time') as mock_time:
            mock_time.return_value = 0

            # Create a connection (it will appear created at time 0)
            with connection_pool.get_connection() as conn:
                conn.execute("SELECT 1")

            # Advance time to make connection stale
            mock_time.return_value = 7200  # 2 hours later

            # Trigger maintenance
            connection_pool._perform_maintenance()

            # Should have recreated connections
            stats = connection_pool.get_statistics()
            assert stats.total_connections >= 3

    def test_database_operations(self, connection_pool):

        """Test various database operations through pool."""
        with connection_pool.get_connection() as conn:
            # Insert data
            conn.execute(
                "INSERT INTO test_table (name) VALUES (?)",
                ("test_item",)
            )
            conn.commit()

            # Query data
            cursor = conn.execute("SELECT name FROM test_table WHERE name = ?", ("test_item",))
            result = cursor.fetchone()
            assert result[0] == "test_item"

            # Multiple operations
            conn.executemany(
                "INSERT INTO test_table (name) VALUES (?)",
                [("item1",), ("item2",), ("item3",)]
            )
            conn.commit()

            cursor = conn.execute("SELECT COUNT(*) FROM test_table")
            count = cursor.fetchone()[0]
            assert count == 4

    @pytest.mark.asyncio
    async def test_async_connection_in_use_flag(self, connection_pool):
        """Ensure async checkout keeps the connection marked in use."""
        async with await connection_pool.get_connection_async() as conn:
            assert conn.in_use is True
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        assert conn.in_use is False

    def test_transaction_rollback(self, connection_pool):

        """Test transaction rollback functionality."""
        with connection_pool.get_connection() as conn:
            # Start transaction and insert data
            conn.execute("INSERT INTO test_table (name) VALUES (?)", ("rollback_test",))

            # Rollback transaction
            conn.rollback()

            # Verify data was not committed
            cursor = conn.execute("SELECT COUNT(*) FROM test_table WHERE name = ?", ("rollback_test",))
            count = cursor.fetchone()[0]
            assert count == 0

    def test_health_status(self, connection_pool):

        """Test health status reporting."""
        health = connection_pool.get_health_status()

        assert "health_score" in health
        assert "status" in health
        assert "statistics" in health
        assert "configuration" in health

        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert 0 <= health["health_score"] <= 100

        # Test with some load
        with connection_pool.get_connection() as conn:
            conn.execute("SELECT 1")

            health = connection_pool.get_health_status()
            assert health["statistics"]["active_connections"] == 1


class TestEvaluationsConnectionManager:
    """Test the connection manager integration."""

    def test_manager_initialization(self, temp_db):

        """Test connection manager initialization."""
        manager = EvaluationsConnectionManager(temp_db)

        try:
            health = manager.get_health_status()
            assert health["status"] in ["healthy", "degraded", "unhealthy"]

            stats = manager.get_statistics()
            assert stats.total_connections > 0

        finally:
            manager.shutdown()

    def test_manager_operations(self, temp_db):

        """Test database operations through manager."""
        manager = EvaluationsConnectionManager(temp_db)

        try:
            with manager.get_connection() as conn:
                # Test database operation
                cursor = conn.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1

        finally:
            manager.shutdown()

    @pytest.mark.asyncio
    async def test_async_connection(self, temp_db):
        """Test async connection interface."""
        manager = EvaluationsConnectionManager(temp_db)

        try:
            async with await manager.get_connection_async() as conn:
                # Test database operation
                cursor = conn.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1

        finally:
            manager.shutdown()


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_global_connection_access(self, monkeypatch):

        """Test global connection access functions."""
        # Mock the global connection manager
        mock_health = {"status": "healthy", "health_score": 100}
        mock_stats = type('MockStats', (), {'total_connections': 5, 'active_connections': 1})()

        with patch('tldw_Server_API.app.core.Evaluations.connection_pool.connection_manager') as mock_manager:
            mock_manager.get_health_status.return_value = mock_health
            mock_manager.get_statistics.return_value = mock_stats

            health = get_connection_health()
            stats = get_connection_stats()

            assert health == mock_health
            assert stats == mock_stats


@pytest.mark.integration
class TestConnectionPoolIntegration:
    """Integration tests for connection pool."""

    def test_concurrent_access(self, connection_pool):

        """Test concurrent access from multiple threads."""
        results = []
        errors = []

        def worker():

            try:
                with connection_pool.get_connection() as conn:
                    cursor = conn.execute("SELECT 1")
                    result = cursor.fetchone()[0]
                    results.append(result)
                    time.sleep(0.1)  # Simulate work
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for _ in range(10):
            t = Thread(target=worker)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10
        assert all(r == 1 for r in results)

    def test_connection_configuration(self, temp_db):

        """Test connection configuration options."""
        pool = ConnectionPool(
            db_path=temp_db,
            pool_size=2,
            max_overflow=1,
            pool_timeout=1.0,
            pool_recycle=10,
            enable_monitoring=False
        )

        try:
            assert pool.pool_size == 2
            assert pool.max_overflow == 1
            assert pool.pool_timeout == 1.0
            assert pool.pool_recycle == 10
            assert not pool.enable_monitoring

        finally:
            pool.shutdown()

    def test_pool_metrics_integration(self, connection_pool):

        """Test metrics integration if available."""
        # Perform some operations to generate metrics
        for _ in range(5):
            with connection_pool.get_connection() as conn:
                conn.execute("SELECT 1")

        stats = connection_pool.get_statistics()
        assert stats.checkout_count == 5
        assert stats.avg_checkout_time > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
