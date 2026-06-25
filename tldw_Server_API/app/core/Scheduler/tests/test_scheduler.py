"""
Comprehensive test suite for the Scheduler module.
"""

import asyncio
import contextlib
import os
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import uuid

from ..scheduler import Scheduler, create_scheduler
from ..base import Task, TaskStatus, TaskPriority
from ..base.registry import get_registry
from ..config import SchedulerConfig
from ..backends import create_backend
from ..backends.postgresql_backend import PostgreSQLBackend
from ..core.leader_election import LeaderElection
from ..services import LeaseService
from ..authorization import AuthContext, TaskPermission

DEFAULT_METADATA = {"user_id": "test-user"}


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePostgresConnection:
    def __init__(self):
        self.fetchrow_calls = []
        self.executemany_calls = []
        self.execute_calls = []

    async def fetchrow(self, query, *args):
        self.fetchrow_calls.append((query, args))
        return {"id": args[0]}

    async def executemany(self, query, values):
        self.executemany_calls.append((query, list(values)))

    async def execute(self, query, *args):
        self.execute_calls.append((query, args))
        return "SELECT 1"

    def transaction(self):
        return _AsyncContext(self)


class _FakePostgresPool:
    def __init__(self, connection):
        self.connection = connection

    def acquire(self):
        return _AsyncContext(self.connection)


@pytest.fixture
async def test_config():
    """Create test configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield SchedulerConfig(
            database_url=f"sqlite:///{tmpdir}/test.db",
            base_path=Path(tmpdir),
            min_workers=1,
            max_workers=5,
            write_buffer_size=10,
            write_buffer_flush_interval=0.1
        )


@pytest.fixture
async def scheduler(test_config):
    """Create and start a test scheduler."""
    scheduler = Scheduler(test_config)
    await scheduler.start(start_workers=False)  # No workers for unit tests
    try:
        yield scheduler
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_lifecycle(test_config):
    """Test scheduler start and stop."""
    scheduler = Scheduler(test_config)

    # Start scheduler
    await scheduler.start(start_workers=False)
    assert scheduler._started is True

    # Get status
    status = scheduler.get_status()
    assert status['started'] is True
    assert status['backend'] is not None

    # Stop scheduler
    await scheduler.stop()
    assert scheduler._started is False


@pytest.mark.asyncio
async def test_scheduler_background_loops_survive_start_if_leadership_setup_yields(test_config, monkeypatch):
    """Background loops should not observe the scheduler as stopped during startup."""
    original = LeaderElection.maintain_leadership

    async def yielding_maintain(self, resource, callback=None, ttl=None, renew_interval=None):
        await asyncio.sleep(0)
        task = await original(self, resource, callback=callback, ttl=ttl, renew_interval=renew_interval)
        return task

    monkeypatch.setattr(LeaderElection, "maintain_leadership", yielding_maintain)

    scheduler = Scheduler(test_config)
    await scheduler.start(start_workers=False)
    try:
        await asyncio.sleep(0)
        assert scheduler._cleanup_task is not None
        assert scheduler._monitor_task is not None
        assert scheduler._cleanup_task.done() is False
        assert scheduler._monitor_task.done() is False
    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_task_submission(scheduler):
    """Test submitting tasks to scheduler."""
    # Register a test handler
    registry = get_registry()

    @registry.task(name="test_handler")
    async def test_handler(payload):
        return {"result": payload.get("value", 0) * 2}

    # Submit task
    task_id = await scheduler.submit(
        handler="test_handler",
        payload={"value": 42},
        priority=TaskPriority.HIGH.value,
        metadata=DEFAULT_METADATA
    )

    assert task_id is not None

    # Force flush to database
    await scheduler.write_buffer.flush()

    # Retrieve task
    task = await scheduler.get_task(task_id)
    assert task is not None
    assert task.handler == "test_handler"
    assert task.payload == {"value": 42}
    assert task.priority == TaskPriority.HIGH.value
    assert task.metadata == DEFAULT_METADATA


@pytest.mark.asyncio
async def test_handler_name_allows_module_qualified_defaults(scheduler):
    """Module-qualified handler names should be accepted."""
    registry = get_registry()

    @registry.task()
    async def dotted_handler(payload):
        return payload

    task_id = await scheduler.submit(
        handler=dotted_handler._task_name,
        payload={"value": 1},
        metadata=DEFAULT_METADATA
    )

    await scheduler.write_buffer.flush()
    task = await scheduler.get_task(task_id)
    assert task is not None
    assert task.handler == dotted_handler._task_name


@pytest.mark.asyncio
async def test_sync_handler_executes_as_awaitable(scheduler):
    """Sync handlers should execute via executor without TypeError."""
    registry = get_registry()

    @registry.task(name="sync_handler_test")
    def sync_handler(payload):
        return {"ok": payload["value"]}

    result = await registry.execute_handler("sync_handler_test", {"value": 5})
    assert result == {"ok": 5}


@pytest.mark.asyncio
async def test_handler_defaults_applied_to_task(scheduler):
    """Handler defaults should populate queue/timeout/retries on tasks."""
    registry = get_registry()

    @registry.task(name="defaults_handler", queue="custom_queue", max_retries=5, timeout=123)
    async def defaults_handler(payload):
        return payload

    task_id = await scheduler.submit(
        handler="defaults_handler",
        payload={"value": 2},
        metadata=DEFAULT_METADATA
    )

    await scheduler.write_buffer.flush()
    task = await scheduler.get_task(task_id)
    assert task is not None
    assert task.queue_name == "custom_queue"
    assert task.max_retries == 5
    assert task.timeout == 123


@pytest.mark.asyncio
async def test_batch_submission(scheduler):
    """Test batch task submission."""
    # Register handler
    registry = get_registry()

    @registry.task(name="batch_handler")
    async def batch_handler(payload):
        return payload

    # Submit batch
    tasks = [
        {"handler": "batch_handler", "payload": {"id": i}, "metadata": DEFAULT_METADATA}
        for i in range(5)
    ]

    task_ids = await scheduler.submit_batch(tasks)
    assert len(task_ids) == 5

    # Force flush
    await scheduler.write_buffer.flush()

    # Verify all tasks created
    for i, task_id in enumerate(task_ids):
        task = await scheduler.get_task(task_id)
        assert task is not None
        assert task.payload == {"id": i}
        assert task.metadata == DEFAULT_METADATA


@pytest.mark.asyncio
async def test_batch_submission_idempotency_handling(scheduler):
    """Ensure duplicate idempotency keys in a batch map to the same task."""
    registry = get_registry()

    @registry.task(name="batch_idem_handler")
    async def handler(payload):
        return payload

    tasks = [
        {
            "handler": "batch_idem_handler",
            "payload": {"value": 1},
            "idempotency_key": "shared-key",
            "metadata": DEFAULT_METADATA
        },
        {
            "handler": "batch_idem_handler",
            "payload": {"value": 2},
            "idempotency_key": "shared-key",
            "metadata": DEFAULT_METADATA
        }
    ]

    task_ids = await scheduler.submit_batch(tasks)
    assert len(task_ids) == 2
    assert task_ids[0] == task_ids[1]

    stored_task = await scheduler.get_task(task_ids[0])
    assert stored_task is not None
    assert stored_task.metadata == DEFAULT_METADATA

    queue_status = await scheduler.get_queue_status("default")
    assert queue_status["size"] == 1


@pytest.mark.asyncio
async def test_batch_submission_requires_metadata(scheduler):
    """Batch submission should reject tasks without metadata."""
    registry = get_registry()

    @registry.task(name="batch_metadata_handler")
    async def handler(payload):
        return payload

    with pytest.raises(ValueError, match="metadata"):
        await scheduler.submit_batch([
            {
                "handler": "batch_metadata_handler",
                "payload": {"value": 1}
            }
        ])


@pytest.mark.asyncio
async def test_batch_submission_honours_authorization(scheduler):
    """Authorization checks are applied to batch submissions."""
    registry = get_registry()

    @registry.task(name="batch_protected")
    async def handler(payload):
        return payload

    scheduler.authorizer.register_handler_permissions(
        'batch_protected',
        [TaskPermission.SUBMIT],
        admin_only=True
    )

    user_context = AuthContext(user_id="regular", roles=["user"])

    with pytest.raises(PermissionError):
        await scheduler.submit_batch(
            [
                {
                    "handler": "batch_protected",
                    "payload": {"value": 1},
                    "metadata": {"user_id": "regular"}
                }
            ],
            auth_context=user_context
        )


@pytest.mark.asyncio
async def test_idempotency(scheduler):
    """Test idempotent task submission."""
    registry = get_registry()

    @registry.task(name="idempotent_handler")
    async def handler(payload):
        return payload

    # Submit task with idempotency key
    task_id1 = await scheduler.submit(
        handler="idempotent_handler",
        payload={"data": "test"},
        idempotency_key="unique-key-123",
        metadata=DEFAULT_METADATA
    )

    # Submit again with same key
    task_id2 = await scheduler.submit(
        handler="idempotent_handler",
        payload={"data": "different"},  # Different payload
        idempotency_key="unique-key-123"  # Same key
        ,
        metadata=DEFAULT_METADATA
    )

    # Should get same task ID
    assert task_id1 == task_id2


@pytest.mark.asyncio
async def test_task_dependencies(scheduler):
    """Test task dependency handling."""
    registry = get_registry()

    @registry.task(name="dep_handler")
    async def handler(payload):
        return payload

    # Create parent task
    parent_id = await scheduler.submit(
        handler="dep_handler",
        payload={"task": "parent"},
        metadata=DEFAULT_METADATA
    )

    # Create child task with dependency
    child_id = await scheduler.submit(
        handler="dep_handler",
        payload={"task": "child"},
        depends_on=[parent_id],
        metadata=DEFAULT_METADATA
    )

    # Force flush
    await scheduler.write_buffer.flush()

    # Check dependency service
    ready = await scheduler.dependency_service.check_dependencies(child_id)
    assert ready is False  # Parent not completed

    # Complete parent task
    await scheduler.backend.execute(
        "UPDATE tasks SET status = 'completed' WHERE id = ?",
        parent_id
    )

    # Now child should be ready
    ready = await scheduler.dependency_service.check_dependencies(child_id)
    assert ready is True


@pytest.mark.asyncio
async def test_worker_pool_integration(test_config):
    """Test scheduler with worker pool."""
    scheduler = Scheduler(test_config)
    await scheduler.start(start_workers=True)

    try:
        # Register handler
        registry = get_registry()

        @registry.task(name="worker_test")
        async def handler(payload):
            await asyncio.sleep(0.1)  # Simulate work
            return {"processed": payload}

        # Submit task
        task_id = await scheduler.submit(
            handler="worker_test",
            payload={"test": "data"},
            metadata=DEFAULT_METADATA
        )

        # Force flush
        await scheduler.write_buffer.flush()

        # Wait for task completion
        result = await scheduler.wait_for_task(task_id, timeout=5)
        assert result is not None
        assert result.status == TaskStatus.COMPLETED

        # Check worker pool status
        pool_status = scheduler.worker_pool.get_status()
        assert pool_status['total_tasks_processed'] > 0

    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_cancel_task_honors_metadata_owner(scheduler):
    """Ensure cancel_task enforces ownership based on persisted metadata."""
    registry = get_registry()

    @registry.task(name="cancel_test")
    async def handler(payload):
        return payload

    task_id = await scheduler.submit(
        handler="cancel_test",
        payload={"value": 1},
        metadata={"user_id": "owner-1"}
    )

    await scheduler.write_buffer.flush()

    with pytest.raises(PermissionError):
        await scheduler.cancel_task(task_id, auth_context=AuthContext(user_id="intruder"))

    cancelled = await scheduler.cancel_task(task_id, auth_context=AuthContext(user_id="owner-1"))
    assert cancelled is True

    task = await scheduler.get_task(task_id)
    assert task is not None
    assert task.status == TaskStatus.CANCELLED
    assert task.metadata.get("user_id") == "owner-1"


@pytest.mark.asyncio
async def test_sqlite_dependency_execution_runs_in_order(test_config):
    """Ensure SQLite backend releases dependent tasks once parents complete."""
    scheduler = Scheduler(test_config)
    await scheduler.start(start_workers=True)

    try:
        registry = get_registry()
        results = []

        @registry.task(name="dependency_parent_task")
        async def parent(payload):
            results.append(("parent", payload["value"]))
            return payload

        @registry.task(name="dependency_child_task")
        async def child(payload):
            results.append(("child", payload["value"]))
            return payload

        parent_id = await scheduler.submit(
            handler="dependency_parent_task",
            payload={"value": "parent"},
            metadata={"user_id": "dep-user"}
        )

        child_id = await scheduler.submit(
            handler="dependency_child_task",
            payload={"value": "child"},
            depends_on=[parent_id],
            metadata={"user_id": "dep-user"}
        )

        await scheduler.write_buffer.flush()

        parent_task = await scheduler.wait_for_task(parent_id, timeout=10)
        child_task = await scheduler.wait_for_task(child_id, timeout=10)

        assert parent_task is not None and parent_task.status == TaskStatus.COMPLETED
        assert child_task is not None and child_task.status == TaskStatus.COMPLETED
        assert [label for label, _ in results][:2] == ["parent", "child"]

    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_postgres_dependency_execution_runs_in_order(tmp_path):
    """Ensure Postgres backend handles dependent tasks (skips if unavailable)."""
    pytest.importorskip("asyncpg")
    dsn = os.getenv("SCHEDULER_TEST_POSTGRES_URL")
    if not dsn:
        pytest.skip("SCHEDULER_TEST_POSTGRES_URL not configured")

    config = SchedulerConfig(
        database_url=dsn,
        base_path=tmp_path / "scheduler_pg",
        min_workers=1,
        max_workers=1
    )

    scheduler = Scheduler(config)
    try:
        await scheduler.start(start_workers=True)
    except Exception as exc:
        await scheduler.stop()
        pytest.skip(f"Postgres backend unavailable: {exc}")

    try:
        registry = get_registry()
        results = []

        @registry.task(name="pg_parent_task")
        async def parent(payload):
            results.append(("parent", payload["value"]))
            return payload

        @registry.task(name="pg_child_task")
        async def child(payload):
            results.append(("child", payload["value"]))
            return payload

        parent_id = await scheduler.submit(
            handler="pg_parent_task",
            payload={"value": "parent"},
            metadata={"user_id": "dep-user"}
        )

        child_id = await scheduler.submit(
            handler="pg_child_task",
            payload={"value": "child"},
            depends_on=[parent_id],
            metadata={"user_id": "dep-user"}
        )

        await scheduler.write_buffer.flush()

        parent_task = await scheduler.wait_for_task(parent_id, timeout=20)
        child_task = await scheduler.wait_for_task(child_id, timeout=20)

        assert parent_task is not None and parent_task.status == TaskStatus.COMPLETED
        assert child_task is not None and child_task.status == TaskStatus.COMPLETED
        assert [label for label, _ in results][:2] == ["parent", "child"]

    finally:
        await scheduler.stop()


@pytest.mark.asyncio
async def test_sqlite_auto_renew_extends_lease_expiration():
    """Ensure SQLite backend renews leases using the aligned contract."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SchedulerConfig(
            database_url=f"sqlite:///{tmpdir}/lease.db",
            base_path=Path(tmpdir),
            lease_duration_seconds=10,
            lease_renewal_interval=3,
            min_workers=0,
            max_workers=0,
            write_buffer_size=1,
            write_buffer_flush_interval=0.01
        )

        backend = create_backend(config)
        await backend.connect()

        try:
            task = Task(handler="test.handler", payload={}, metadata={"user_id": "lease-test"})
            await backend.enqueue(task)

            # Simulate running task with a short timeout to control renewal horizon
            await backend.execute(
                "UPDATE tasks SET status = 'running', timeout = ? WHERE id = ?",
                20, task.id
            )

            lease_id = uuid.uuid4().hex
            original_expires = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=config.lease_duration_seconds)
            await backend.create_lease(lease_id, task.id, "worker-test", original_expires)

            lease_service = LeaseService(backend, config.lease_duration_seconds)
            renew_task = await lease_service.auto_renew(task.id, lease_id, renew_interval=0.2)

            try:
                await asyncio.sleep(0.5)  # allow renewal loop to run at least once
            finally:
                renew_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await renew_task

            row = await backend.fetchrow(
                "SELECT expires_at FROM task_leases WHERE lease_id = ?",
                lease_id
            )
            assert row is not None, "Lease record should exist after renewal"

            renewed_expires = datetime.fromisoformat(row['expires_at'])
            assert renewed_expires > original_expires, "Lease expiration should extend after renewal"
        finally:
            await backend.disconnect()


@pytest.mark.asyncio
async def test_sqlite_dequeues_iso_scheduled_task_on_current_day(tmp_path):
    """SQLite should compare ISO scheduled_at values as timestamps, not strings."""
    config = SchedulerConfig(
        database_url=f"sqlite:///{tmp_path}/scheduled.db",
        base_path=tmp_path / "scheduler",
        min_workers=0,
        max_workers=0,
    )
    backend = create_backend(config)
    await backend.connect()
    try:
        scheduled_at = datetime.now(timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=None,
        )
        task = Task(
            handler="test.handler",
            payload={},
            scheduled_at=scheduled_at,
            metadata=DEFAULT_METADATA,
        )
        await backend.enqueue(task)

        dequeued = await backend.dequeue_atomic("default", "worker-iso-time")

        assert dequeued is not None
        assert dequeued.id == task.id
    finally:
        await backend.disconnect()


@pytest.mark.asyncio
async def test_sqlite_reclaims_iso_expired_lease(tmp_path):
    """SQLite should compare ISO lease expiry values as timestamps, not strings."""
    config = SchedulerConfig(
        database_url=f"sqlite:///{tmp_path}/lease-expiry.db",
        base_path=tmp_path / "scheduler",
        lease_duration_seconds=30,
        lease_renewal_interval=5,
        min_workers=0,
        max_workers=0,
    )
    backend = create_backend(config)
    await backend.connect()
    try:
        task = Task(handler="test.handler", payload={}, metadata=DEFAULT_METADATA)
        await backend.enqueue(task)
        dequeued = await backend.dequeue_atomic("default", "worker-expired")
        assert dequeued is not None

        expired_at = datetime.now(timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=None,
        )
        await backend.execute(
            "UPDATE task_leases SET expires_at = ? WHERE lease_id = ?",
            expired_at.isoformat(),
            dequeued.lease_id,
        )

        assert await backend.reclaim_expired_leases() == 1
        reclaimed = await backend.get_task(task.id)
        assert reclaimed is not None
        assert reclaimed.status == TaskStatus.QUEUED
    finally:
        await backend.disconnect()


@pytest.mark.asyncio
async def test_sqlite_ack_rejects_stale_lease_owner(tmp_path):
    """A reclaimed task must not be completed by a stale worker lease."""
    config = SchedulerConfig(
        database_url=f"sqlite:///{tmp_path}/lease-owner.db",
        base_path=tmp_path / "scheduler",
        lease_duration_seconds=30,
        lease_renewal_interval=5,
        min_workers=0,
        max_workers=0,
    )
    backend = create_backend(config)
    await backend.connect()
    try:
        task = Task(handler="test.handler", payload={}, metadata=DEFAULT_METADATA)
        await backend.enqueue(task)

        first_claim = await backend.dequeue_atomic("default", "worker-1")
        assert first_claim is not None
        stale_lease_id = first_claim.lease_id

        expired_at = datetime.now(timezone.utc).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=None,
        )
        await backend.execute(
            "UPDATE task_leases SET expires_at = ? WHERE lease_id = ?",
            expired_at.isoformat(),
            stale_lease_id,
        )
        assert await backend.reclaim_expired_leases() == 1

        second_claim = await backend.dequeue_atomic("default", "worker-2")
        assert second_claim is not None
        assert second_claim.lease_id != stale_lease_id

        stale_ack = await backend.ack(
            task.id,
            {"stale": True},
            lease_id=stale_lease_id,
            worker_id="worker-1",
        )
        assert stale_ack is False

        still_running = await backend.get_task(task.id)
        assert still_running is not None
        assert still_running.status == TaskStatus.RUNNING
        assert still_running.worker_id == "worker-2"

        current_ack = await backend.ack(
            task.id,
            {"ok": True},
            lease_id=second_claim.lease_id,
            worker_id="worker-2",
        )
        assert current_ack is True
    finally:
        await backend.disconnect()


@pytest.mark.asyncio
async def test_postgresql_enqueue_normalizes_pending_tasks_to_queued():
    """PostgreSQL inserts must persist newly submitted tasks as queued."""
    backend = PostgreSQLBackend.__new__(PostgreSQLBackend)
    backend.config = SimpleNamespace(
        payload_threshold_bytes=10_000_000,
        payload_compression=False,
        default_queue_name="default",
    )
    connection = _FakePostgresConnection()
    backend.pool = _FakePostgresPool(connection)

    task = Task(handler="test.handler", payload={"value": 1}, metadata=DEFAULT_METADATA)
    assert task.status == TaskStatus.PENDING

    await backend.enqueue(task)

    insert_args = connection.fetchrow_calls[0][1]
    assert insert_args[4] == TaskStatus.QUEUED.value

    await backend.bulk_enqueue([
        Task(handler="test.handler", payload={"value": 2}, metadata=DEFAULT_METADATA),
        Task(handler="test.handler", payload={"value": 3}, metadata=DEFAULT_METADATA),
    ])

    bulk_values = connection.executemany_calls[0][1]
    assert [value[4] for value in bulk_values] == [
        TaskStatus.QUEUED.value,
        TaskStatus.QUEUED.value,
    ]


@pytest.mark.asyncio
async def test_queue_management(scheduler):
    """Test queue operations."""
    registry = get_registry()

    @registry.task(name="queue_test")
    async def handler(payload):
        return payload

    # Submit to different queues
    task1 = await scheduler.submit(
        handler="queue_test",
        payload={"id": 1},
        queue_name="high_priority",
        metadata=DEFAULT_METADATA
    )

    task2 = await scheduler.submit(
        handler="queue_test",
        payload={"id": 2},
        queue_name="low_priority",
        metadata=DEFAULT_METADATA
    )

    # Force flush
    await scheduler.write_buffer.flush()

    # Check queue sizes
    high_status = await scheduler.get_queue_status("high_priority")
    assert high_status['size'] == 1

    low_status = await scheduler.get_queue_status("low_priority")
    assert low_status['size'] == 1


@pytest.mark.asyncio
async def test_scheduler_context_manager(test_config):
    """Test scheduler as context manager."""
    async with Scheduler(test_config) as scheduler:
        registry = get_registry()

        @registry.task(name="context_test")
        async def handler(payload):
            return payload

        task_id = await scheduler.submit(
            handler="context_test",
            payload={"test": True},
            metadata=DEFAULT_METADATA
        )

        await scheduler.write_buffer.flush()

        task = await scheduler.get_task(task_id)
        assert task is not None


@pytest.mark.asyncio
async def test_leader_election(test_config):
    """Test leader election with multiple schedulers."""
    # Create two scheduler instances
    scheduler1 = Scheduler(test_config)
    scheduler2 = Scheduler(test_config)

    await scheduler1.start(start_workers=False)
    await scheduler2.start(start_workers=False)

    try:
        # Try to acquire leadership on both
        leader1 = await scheduler1.leader_election.acquire_leadership("test_resource")
        leader2 = await scheduler2.leader_election.acquire_leadership("test_resource")

        # Only one should be leader
        assert leader1 != leader2
        assert leader1 or leader2  # At least one should succeed

    finally:
        await scheduler1.stop()
        await scheduler2.stop()


@pytest.mark.asyncio
async def test_payload_service(scheduler):
    """Test large payload handling."""
    registry = get_registry()

    @registry.task(name="payload_test")
    async def handler(payload):
        return len(payload.get("data", ""))

    # Create large payload
    large_data = "x" * 100000  # 100KB of data

    task_id = await scheduler.submit(
        handler="payload_test",
        payload={"data": large_data},
        metadata=DEFAULT_METADATA
    )

    await scheduler.write_buffer.flush()

    # Check if payload was externalized
    should_external = scheduler.payload_service.should_externalize({"data": large_data})
    assert should_external is True

    # Get stats
    stats = await scheduler.payload_service.get_stats()
    assert stats['storage_path'] is not None


@pytest.mark.asyncio
async def test_error_handling(scheduler):
    """Test error handling in scheduler."""
    # Try to submit with non-existent handler
    with pytest.raises(ValueError, match="not registered"):
        await scheduler.submit(
            handler="non_existent",
            payload={},
            metadata=DEFAULT_METADATA
        )

    # Test scheduler not started
    new_scheduler = Scheduler(scheduler.config)
    with pytest.raises(Exception):
        await new_scheduler.submit(
            handler="test",
            payload={},
            metadata=DEFAULT_METADATA
        )


if __name__ == "__main__":
    # Run tests
    with tempfile.TemporaryDirectory() as tmpdir:
        asyncio.run(test_scheduler_lifecycle(SchedulerConfig(
            database_url="sqlite://:memory:",
            base_path=Path(tmpdir)
        )))
    print("✓ Scheduler lifecycle test passed")

    print("\n✅ All scheduler tests passed!")
