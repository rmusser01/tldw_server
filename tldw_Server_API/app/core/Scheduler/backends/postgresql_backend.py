"""
PostgreSQL backend with SKIP LOCKED for high-performance queuing.
Leverages PostgreSQL-specific features for optimal performance.
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

try:
    import asyncpg
    from asyncpg import Connection, Pool
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    Pool = Any
    Connection = Any

from loguru import logger

from ..base import Task, TaskStatus
from ..base.exceptions import BackendError
from ..base.queue_backend import QueueBackend
from ..config import SchedulerConfig


class PostgreSQLBackend(QueueBackend):
    """
    PostgreSQL backend implementation with advanced features.

    Leverages:
    - SKIP LOCKED for lock-free atomic dequeue
    - NOTIFY/LISTEN for real-time updates
    - Advisory locks for leader election
    - Efficient bulk operations with unnest()
    - Partial indexes for performance
    """

    def __init__(self, config: SchedulerConfig):
        """
        Initialize PostgreSQL backend.

        Args:
            config: Scheduler configuration
        """
        if not HAS_ASYNCPG:
            raise ImportError(
                "asyncpg is required for PostgreSQL backend. "
                "Install with: pip install asyncpg"
            )

        self.config = config
        self.pool: Optional[Pool] = None
        self._listener_task: Optional[asyncio.Task] = None
        self._notifications: dict[str, asyncio.Queue] = {}

        # Parse connection string
        self.dsn = config.database_url
        if self.dsn.startswith('postgresql://'):
            self.dsn = self.dsn.replace('postgresql://', '')
        elif self.dsn.startswith('postgres://'):
            self.dsn = self.dsn.replace('postgres://', '')

    async def connect(self) -> None:
        """
        Establish connection pool and set up database.
        """
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.config.db_pool_min_size,
                max_size=self.config.db_pool_max_size,
                max_queries=50000,
                max_cached_statement_lifetime=300,
                max_inactive_connection_lifetime=60.0,
                command_timeout=60.0
            )

            # Initialize schema
            await self._initialize_schema()

            # Start listener for NOTIFY/LISTEN
            self._listener_task = asyncio.create_task(self._listen_for_notifications())

            logger.info(
                f"PostgreSQL backend connected: pool={self.config.db_pool_min_size}-"
                f"{self.config.db_pool_max_size}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise BackendError(f"Connection failed: {e}") from e

    async def disconnect(self) -> None:
        """
        Close connection pool and cleanup.
        """
        if self._listener_task:
            self._listener_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._listener_task

        if self.pool:
            await self.pool.close()
            self.pool = None

        logger.info("PostgreSQL backend disconnected")

    async def _initialize_schema(self) -> None:
        """
        Create tables and indexes if they don't exist.
        """
        async with self.pool.acquire() as conn:
            await conn.execute("""
                -- Tasks table with optimized indexes
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    handler TEXT NOT NULL,
                    payload JSONB,
                    payload_ref TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    priority INTEGER DEFAULT 0,
                    queue_name TEXT NOT NULL DEFAULT 'default',
                    scheduled_at TIMESTAMPTZ,
                    depends_on TEXT[],
                    idempotency_key TEXT UNIQUE,
                    max_retries INTEGER DEFAULT 3,
                    retry_count INTEGER DEFAULT 0,
                    retry_delay INTEGER DEFAULT 60,
                    timeout INTEGER DEFAULT 300,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    worker_id TEXT,
                    lease_id TEXT,
                    lease_expires_at TIMESTAMPTZ,
                    error TEXT,
                    result JSONB,
                    metadata JSONB
                );

                -- Partial indexes for efficient queries
                CREATE INDEX IF NOT EXISTS idx_tasks_dequeue
                ON tasks (queue_name, priority DESC, created_at)
                WHERE status = 'queued' AND (scheduled_at IS NULL OR scheduled_at <= NOW());

                CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks (status) WHERE status IN ('queued', 'running');

                CREATE INDEX IF NOT EXISTS idx_tasks_scheduled
                ON tasks (scheduled_at) WHERE scheduled_at IS NOT NULL AND status = 'queued';

                CREATE INDEX IF NOT EXISTS idx_tasks_dependencies
                ON tasks USING GIN (depends_on) WHERE depends_on IS NOT NULL;

                CREATE INDEX IF NOT EXISTS idx_tasks_lease
                ON tasks (lease_expires_at) WHERE status = 'running';

                CREATE INDEX IF NOT EXISTS idx_tasks_worker
                ON tasks (worker_id) WHERE worker_id IS NOT NULL;

                -- Leader election table
                CREATE TABLE IF NOT EXISTS leader_election (
                    resource TEXT PRIMARY KEY,
                    leader_id TEXT NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    metadata JSONB
                );

                -- External payload storage
                CREATE TABLE IF NOT EXISTS payloads (
                    id TEXT PRIMARY KEY,
                    task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
                    data BYTEA NOT NULL,
                    compressed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                -- Dead letter queue
                CREATE TABLE IF NOT EXISTS dead_letter_queue (
                    id TEXT PRIMARY KEY,
                    original_task_id TEXT NOT NULL,
                    queue_name TEXT NOT NULL,
                    handler TEXT NOT NULL,
                    payload JSONB,
                    error_count INTEGER DEFAULT 1,
                    last_error TEXT,
                    moved_at TIMESTAMPTZ DEFAULT NOW(),
                    original_created_at TIMESTAMPTZ,
                    metadata JSONB
                );

                -- Schema version tracking
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT NOW()
                );

                INSERT INTO schema_version (version)
                VALUES (1)
                ON CONFLICT (version) DO NOTHING;

                -- Trigger for updated_at
                CREATE OR REPLACE FUNCTION update_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;

                CREATE TRIGGER update_tasks_updated_at
                BEFORE UPDATE ON tasks
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at();
            """)

    async def create_schema(self) -> None:
        """Create database schema (idempotent)."""
        await self._initialize_schema()

    async def get_schema_version(self) -> int:
        """Get current schema version."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT MAX(version) FROM schema_version") or 0

    async def migrate_schema(self, target_version: int) -> None:
        """Placeholder schema migration hook."""
        current = await self.get_schema_version()
        if current < target_version:
            # TODO: implement migrations
            pass

    async def enqueue(self, task: Task) -> str:
        """
        Add task to queue with idempotency support.
        """
        async with self.pool.acquire() as conn:
            try:
                # Handle large payload
                payload_ref = None
                payload_data = task.payload

                if payload_data and len(json.dumps(payload_data)) > self.config.payload_threshold_bytes:
                    payload_ref = await self._store_external_payload(conn, task.id, payload_data)
                    payload_data = None

                # Insert with ON CONFLICT for idempotency and return canonical ID
                row = await conn.fetchrow("""
                    INSERT INTO tasks (
                        id, handler, payload, payload_ref, status, priority,
                        queue_name, scheduled_at, depends_on, idempotency_key,
                        max_retries, retry_delay, timeout, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (idempotency_key)
                    DO UPDATE SET idempotency_key = EXCLUDED.idempotency_key
                    RETURNING id
                """,
                    task.id, task.handler, json.dumps(payload_data) if payload_data else None,
                    payload_ref, TaskStatus.QUEUED.value, task.priority,
                    task.queue_name or self.config.default_queue_name,
                    task.scheduled_at, task.depends_on, task.idempotency_key,
                    task.max_retries, task.retry_delay, task.timeout,
                    json.dumps(task.metadata) if task.metadata else None
                )

                # Notify listeners
                await self._notify_queue(conn, task.queue_name or self.config.default_queue_name)

                # Return canonical ID (existing or newly inserted)
                return row['id'] if row and 'id' in row else task.id

            except Exception as e:
                logger.error(f"Failed to enqueue task: {e}")
                raise BackendError(f"Enqueue failed: {e}") from e

    async def bulk_enqueue(self, tasks: list[Task]) -> list[str]:
        """
        Efficiently enqueue multiple tasks in a single transaction.
        """
        if not tasks:
            return []

        async with self.pool.acquire() as conn, conn.transaction():
            task_ids = []

            # Prepare data for bulk insert
            values = []
            for task in tasks:
                # Handle large payloads
                payload_ref = None
                payload_data = task.payload

                if payload_data and len(json.dumps(payload_data)) > self.config.payload_threshold_bytes:
                    payload_ref = await self._store_external_payload(conn, task.id, payload_data)
                    payload_data = None

                values.append((
                    task.id, task.handler,
                    json.dumps(payload_data) if payload_data else None,
                    payload_ref, TaskStatus.QUEUED.value, task.priority,
                    task.queue_name or self.config.default_queue_name,
                    task.scheduled_at, task.depends_on, task.idempotency_key,
                    task.max_retries, task.retry_delay, task.timeout,
                    json.dumps(task.metadata) if task.metadata else None
                ))
                task_ids.append(task.id)

            # Bulk insert with ON CONFLICT
            await conn.executemany("""
                    INSERT INTO tasks (
                        id, handler, payload, payload_ref, status, priority,
                        queue_name, scheduled_at, depends_on, idempotency_key,
                        max_retries, retry_delay, timeout, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                    ON CONFLICT (idempotency_key) DO NOTHING
                """, values)

            # Notify all affected queues
            queues = {t.queue_name or self.config.default_queue_name for t in tasks}
            for queue in queues:
                await self._notify_queue(conn, queue)

            return task_ids

    async def dequeue_atomic(self, queue_name: str, worker_id: str) -> Optional[Task]:
        """
        Atomically dequeue next available task using SKIP LOCKED.

        This is the crown jewel of PostgreSQL queuing - completely lock-free
        atomic dequeue that scales to thousands of workers.
        """
        lease_id = str(uuid.uuid4())
        lease_expires = datetime.now(timezone.utc) + timedelta(seconds=self.config.lease_duration_seconds)

        async with self.pool.acquire() as conn:
            # Single atomic query with SKIP LOCKED
            row = await conn.fetchrow("""
                UPDATE tasks
                SET status = 'running',
                    worker_id = $1,
                    lease_id = $2,
                    lease_expires_at = $3,
                    started_at = NOW(),
                    retry_count = retry_count + 1
                WHERE id = (
                    SELECT id FROM tasks
                    WHERE queue_name = $4
                      AND status = 'queued'
                      AND (scheduled_at IS NULL OR scheduled_at <= NOW())
                      AND (depends_on IS NULL OR NOT EXISTS (
                          SELECT 1 FROM unnest(depends_on) AS dep_id
                          WHERE EXISTS (
                              SELECT 1 FROM tasks t2
                              WHERE t2.id = dep_id
                              AND t2.status != 'completed'
                          )
                      ))
                    ORDER BY priority ASC, created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING *
            """, worker_id, lease_id, lease_expires, queue_name)

            if not row:
                return None

            # Convert to Task object
            return await self._row_to_task(conn, row)

    async def ack(
        self,
        task_id: str,
        result: Optional[Any] = None,
        *,
        lease_id: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """
        Acknowledge task completion.
        """
        async with self.pool.acquire() as conn:
            affected = await conn.execute("""
                UPDATE tasks
                SET status = 'completed',
                    completed_at = NOW(),
                    result = $1,
                    worker_id = NULL,
                    lease_id = NULL,
                    lease_expires_at = NULL
                WHERE id = $2 AND status = 'running'
                  AND ($3::text IS NULL OR lease_id = $3)
                  AND ($4::text IS NULL OR worker_id = $4)
            """, json.dumps(result) if result is not None else None, task_id, lease_id, worker_id)

            return affected != "UPDATE 0"

    async def nack(
        self,
        task_id: str,
        error: Optional[str] = None,
        retry: bool = True,
        *,
        lease_id: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """
        Negative acknowledge - task failed but may be retried.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                UPDATE tasks
                SET status = CASE
                        WHEN $3 = FALSE THEN 'failed'
                        WHEN retry_count >= max_retries THEN 'failed'
                        ELSE 'queued'
                    END,
                    error = $1,
                    lease_id = NULL,
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    scheduled_at = CASE
                        WHEN $3 = TRUE AND retry_count < max_retries
                        THEN NOW() + INTERVAL '1 second' * retry_delay
                        ELSE NULL
                    END
                WHERE id = $2 AND status = 'running'
                  AND ($4::text IS NULL OR lease_id = $4)
                  AND ($5::text IS NULL OR worker_id = $5)
                RETURNING status, queue_name
            """, error, task_id, retry, lease_id, worker_id)

            if row and row['status'] == 'queued':
                # Notify queue for retry
                await self._notify_queue(conn, row['queue_name'])

            return row is not None

    async def renew_lease(self, task_id: str, lease_id: str) -> bool:
        """
        Renew task lease to prevent timeout.
        """
        new_expires = datetime.now(timezone.utc) + timedelta(seconds=self.config.lease_duration_seconds)

        async with self.pool.acquire() as conn:
            affected = await conn.execute("""
                UPDATE tasks
                SET lease_expires_at = $1
                WHERE id = $2 AND lease_id = $3 AND status = 'running'
            """, new_expires, task_id, lease_id)

            return affected != "UPDATE 0"

    async def create_lease(self, lease_id: str, task_id: str,
                           worker_id: str, expires_at: datetime) -> bool:
        """Create lease for task."""
        async with self.pool.acquire() as conn:
            affected = await conn.execute("""
                UPDATE tasks
                SET lease_id = $1,
                    worker_id = $2,
                    lease_expires_at = $3
                WHERE id = $4
            """, lease_id, worker_id, expires_at, task_id)
            return affected != "UPDATE 0"

    async def delete_lease(self, lease_id: str) -> bool:
        """Delete lease by lease_id."""
        async with self.pool.acquire() as conn:
            affected = await conn.execute("""
                UPDATE tasks
                SET lease_id = NULL,
                    lease_expires_at = NULL
                WHERE lease_id = $1
            """, lease_id)
            return affected != "UPDATE 0"

    async def get_expired_leases(self) -> list[dict[str, Any]]:
        """Get expired leases."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT lease_id, id AS task_id, worker_id, lease_expires_at AS expires_at
                FROM tasks
                WHERE lease_id IS NOT NULL
                  AND lease_expires_at < NOW()
            """)
            return [dict(row) for row in rows]

    async def reclaim_expired_leases(self) -> int:
        """
        Reclaim tasks with expired leases.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                UPDATE tasks
                SET status = 'queued',
                    lease_id = NULL,
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    error = 'Lease expired'
                WHERE status = 'running'
                  AND lease_expires_at < NOW()
                RETURNING queue_name
            """)

            # Notify affected queues
            queues = {row['queue_name'] for row in rows}
            for queue in queues:
                await self._notify_queue(conn, queue)

            return len(rows)

    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM tasks WHERE id = $1", task_id)
            if not row:
                return None
            return await self._row_to_task(conn, row)

    async def get_task_by_idempotency_key(self, idempotency_key: str) -> Optional[str]:
        """Get task ID by idempotency key."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT id FROM tasks WHERE idempotency_key = $1",
                idempotency_key
            )

    async def update_task(self, task: Task) -> bool:
        """Update task state."""
        async with self.pool.acquire() as conn:
            affected = await conn.execute("""
                UPDATE tasks
                SET status = $1,
                    error = $2,
                    result = $3,
                    worker_id = $4,
                    lease_id = $5,
                    lease_expires_at = $6,
                    completed_at = COALESCE($7, completed_at)
                WHERE id = $8
            """,
                task.status.value,
                task.error,
                json.dumps(task.result) if task.result is not None else None,
                task.worker_id,
                task.lease_id,
                task.lease_expires_at,
                task.completed_at,
                task.id
            )
            return affected != "UPDATE 0"

    async def get_queue_size(self, queue_name: str) -> int:
        """
        Get number of queued tasks.
        """
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COUNT(*) FROM tasks
                WHERE queue_name = $1 AND status = 'queued'
            """, queue_name)
            return result or 0

    async def clear_queue(self, queue_name: str) -> int:
        """Clear queued tasks for a queue."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                DELETE FROM tasks
                WHERE queue_name = $1 AND status = 'queued'
                RETURNING id
            """, queue_name)
            return len(rows)

    async def get_dead_letter_queue(self) -> list[Task]:
        """Get DLQ tasks."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM dead_letter_queue")
        tasks: list[Task] = []
        for row in rows:
            row_dict = dict(row)
            payload = row_dict.get('payload')
            if isinstance(payload, str):
                with suppress(json.JSONDecodeError):
                    payload = json.loads(payload)
            metadata = row_dict.get('metadata')
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = None
            if payload is None:
                payload = {}
            if metadata is None:
                metadata = {}
            tasks.append(Task(
                id=row_dict['original_task_id'],
                queue_name=row_dict['queue_name'],
                handler=row_dict['handler'],
                payload=payload,
                status=TaskStatus.DEAD,
                error=row_dict.get('last_error'),
                metadata=metadata,
                created_at=row_dict.get('original_created_at') or datetime.now(timezone.utc)
            ))
        return tasks

    async def move_to_dlq(self, task_id: str, reason: str) -> bool:
        """Move task to DLQ."""
        task = await self.get_task(task_id)
        if not task:
            return False

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO dead_letter_queue (
                    id, original_task_id, queue_name, handler, payload,
                    last_error, original_created_at, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                str(uuid.uuid4()),
                task.id,
                task.queue_name,
                task.handler,
                json.dumps(task.payload) if task.payload is not None else None,
                reason,
                task.created_at,
                json.dumps(task.metadata) if task.metadata else None
            )

            await conn.execute(
                "UPDATE tasks SET status = 'dead' WHERE id = $1",
                task_id
            )

        return True

    async def get_ready_tasks(self, queue_name: Optional[str] = None) -> list[str]:
        """
        Get IDs of tasks ready to run (dependencies satisfied).
        Uses efficient CTE for dependency checking.
        """
        async with self.pool.acquire() as conn:
            base_query = """
                WITH dependency_status AS (
                    SELECT
                        t1.id,
                        t1.depends_on,
                        CASE
                            WHEN t1.depends_on IS NULL THEN TRUE
                            WHEN NOT EXISTS (
                                SELECT 1 FROM unnest(t1.depends_on) AS dep_id
                                WHERE EXISTS (
                                    SELECT 1 FROM tasks t2
                                    WHERE t2.id = dep_id
                                    AND t2.status != 'completed'
                                )
                            ) THEN TRUE
                            ELSE FALSE
                        END AS ready
                    FROM tasks t1
                    WHERE t1.status = 'queued'
                      AND (t1.scheduled_at IS NULL OR t1.scheduled_at <= NOW())
            """
            params = []
            if queue_name:
                base_query += " AND t1.queue_name = $1"
                params.append(queue_name)

            base_query += """
                )
                SELECT id FROM dependency_status WHERE ready = TRUE
            """

            rows = await conn.fetch(base_query, *params)

            return [row['id'] for row in rows]

    async def acquire_leader(self, resource: str, leader_id: str, ttl: int) -> bool:
        """
        Try to acquire leadership using advisory locks.
        """
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

        async with self.pool.acquire() as conn:
            # Use advisory lock for atomic operation
            lock_id = hash(resource) % 2147483647  # Convert to valid int4

            # Try to acquire advisory lock
            acquired = await conn.fetchval("SELECT pg_try_advisory_lock($1)", lock_id)

            if not acquired:
                return False

            try:
                # Check and update leader
                result = await conn.fetchval("""
                    INSERT INTO leader_election (resource, leader_id, expires_at)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (resource) DO UPDATE
                    SET leader_id = $2, expires_at = $3
                    WHERE leader_election.expires_at < NOW()
                       OR leader_election.leader_id = $2
                    RETURNING leader_id
                """, resource, leader_id, expires_at)

                return result == leader_id

            finally:
                # Release advisory lock
                await conn.execute("SELECT pg_advisory_unlock($1)", lock_id)

    async def renew_leader(self, resource: str, leader_id: str, ttl: int) -> bool:
        """
        Renew leadership lease.
        """
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

        async with self.pool.acquire() as conn:
            affected = await conn.execute("""
                UPDATE leader_election
                SET expires_at = $1
                WHERE resource = $2 AND leader_id = $3
            """, expires_at, resource, leader_id)

            return affected != "UPDATE 0"

    async def release_leader(self, resource: str, leader_id: str) -> bool:
        """
        Release leadership.
        """
        async with self.pool.acquire() as conn:
            affected = await conn.execute("""
                DELETE FROM leader_election
                WHERE resource = $1 AND leader_id = $2
            """, resource, leader_id)

            return affected != "DELETE 0"

    async def execute(self, query: str, *args) -> Any:
        """
        Execute raw query (for testing/debugging).
        """
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[dict[str, Any]]:
        """Execute query and fetch rows as dictionaries."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetchval(self, query: str, *args) -> Any:
        """Execute query and fetch a single value."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[dict[str, Any]]:
        """Execute query and fetch a single row as dictionary."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager (yields a live connection)."""
        async with self.pool.acquire() as conn, conn.transaction():
            yield conn

    async def _store_external_payload(self, conn: Connection, task_id: str, payload: dict) -> str:
        """
        Store large payload externally.
        """
        payload_id = str(uuid.uuid4())
        data = json.dumps(payload).encode('utf-8')

        # Optionally compress
        compressed = False
        if self.config.payload_compression and len(data) > 1024:
            import gzip
            data = gzip.compress(data)
            compressed = True

        await conn.execute("""
            INSERT INTO payloads (id, task_id, data, compressed)
            VALUES ($1, $2, $3, $4)
        """, payload_id, task_id, data, compressed)

        return payload_id

    async def _load_external_payload(self, conn: Connection, payload_ref: str) -> Optional[dict]:
        """
        Load externally stored payload.
        """
        row = await conn.fetchrow("""
            SELECT data, compressed FROM payloads WHERE id = $1
        """, payload_ref)

        if not row:
            return None

        data = row['data']

        if row['compressed']:
            import gzip
            data = gzip.decompress(data)

        return json.loads(data.decode('utf-8'))

    async def _row_to_task(self, conn: Connection, row: asyncpg.Record) -> Task:
        """
        Convert database row to Task object.
        """
        def _maybe_load_json(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, (dict, list)):
                return value
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value

        # Load payload (from column or external storage)
        payload = None
        if row['payload']:
            payload = _maybe_load_json(row['payload'])
        elif row['payload_ref']:
            payload = await self._load_external_payload(conn, row['payload_ref'])

        # Parse metadata
        metadata = {}
        if row['metadata']:
            metadata = _maybe_load_json(row['metadata']) or {}

        # Parse result
        result = None
        if row['result']:
            result = _maybe_load_json(row['result'])

        return Task(
            id=row['id'],
            handler=row['handler'],
            payload=payload,
            status=TaskStatus(row['status']),
            priority=row['priority'],
            queue_name=row['queue_name'],
            scheduled_at=row['scheduled_at'],
            depends_on=row['depends_on'] or [],
            idempotency_key=row['idempotency_key'],
            max_retries=row['max_retries'],
            retry_count=row['retry_count'],
            retry_delay=row['retry_delay'],
            timeout=row['timeout'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            worker_id=row['worker_id'],
            lease_id=row['lease_id'],
            lease_expires_at=row['lease_expires_at'],
            error=row['error'],
            result=result,
            metadata=metadata
        )

    async def _notify_queue(self, conn: Connection, queue_name: str) -> None:
        """
        Send NOTIFY for queue updates.
        """
        await conn.execute("SELECT pg_notify('queue_update', $1)", queue_name)

    async def _listen_for_notifications(self) -> None:
        """
        Background task to listen for NOTIFY events.
        """
        try:
            conn = await asyncpg.connect(self.dsn)
            await conn.add_listener('queue_update', self._handle_notification)

            # Keep connection alive
            while True:
                await asyncio.sleep(30)
                await conn.fetchval("SELECT 1")  # Heartbeat

        except asyncio.CancelledError:
            if conn:
                await conn.remove_listener('queue_update', self._handle_notification)
                await conn.close()
            raise
        except Exception as e:
            logger.error(f"Notification listener error: {e}")

    def _handle_notification(self, connection, pid, channel, payload):
        """
        Handle NOTIFY events.
        """
        # Notify any waiting consumers
        if payload in self._notifications:
            queue = self._notifications[payload]
            queue.put_nowait(True)

    async def wait_for_queue(self, queue_name: str, timeout: float = None) -> bool:
        """
        Wait for queue activity (used by workers).
        """
        if queue_name not in self._notifications:
            self._notifications[queue_name] = asyncio.Queue()

        queue = self._notifications[queue_name]

        try:
            await asyncio.wait_for(queue.get(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def get_status(self) -> dict[str, Any]:
        """
        Get backend status for monitoring.
        """
        if not self.pool:
            return {"status": "disconnected"}

        return {
            "status": "connected",
            "pool_size": self.pool.get_size(),
            "pool_free": self.pool.get_idle_size(),
            "max_size": self.config.db_pool_max_size
        }
