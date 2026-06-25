"""
SQLite backend implementation for the scheduler.
Optimized for development and single-user deployments.
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite
from loguru import logger

from ...DB_Management.sqlite_policy import configure_sqlite_connection_async
from ..base import ConnectionError, QueueBackend, Task, TaskStatus, TransactionError
from ..config import SchedulerConfig


class SQLiteBackend(QueueBackend):
    """
    SQLite backend implementation with optimizations:
    - INSERT OR IGNORE for idempotency
    - 2-query dependency resolution (no N+1)
    - Write-ahead logging for concurrency
    - JSON text for compatibility
    """

    def __init__(self, config: SchedulerConfig):
        """
        Initialize SQLite backend with connection pooling.

        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.db_path = self._extract_db_path(config.database_url)
        self._connection: Optional[aiosqlite.Connection] = None

        # Connection pool for read operations
        self._read_pool: list[aiosqlite.Connection] = []
        self._read_pool_size = getattr(config, 'sqlite_pool_size', 5)
        self._pool_lock = asyncio.Lock()

        # Separate connection for write operations
        self._write_conn: Optional[aiosqlite.Connection] = None
        self._write_lock = asyncio.Lock()

        # Legacy lock for backward compatibility
        self._lock = self._write_lock

        logger.info(f"Initializing SQLite backend with database: {self.db_path}, pool size: {self._read_pool_size}")

    def _extract_db_path(self, url: str) -> str:
        """Extract database path from URL"""
        if url.startswith('sqlite:///'):
            return url[10:]  # Remove 'sqlite:///'
        elif url == ':memory:':
            return ':memory:'
        else:
            return url

    @asynccontextmanager
    async def _get_read_connection(self):
        """Get a connection from the read pool"""
        async with self._pool_lock:
            if not self._read_pool:
                # Fallback to main connection if pool is empty
                yield self._connection
            else:
                conn = self._read_pool.pop()
                try:
                    yield conn
                finally:
                    # Return connection to pool
                    self._read_pool.append(conn)

    async def _get_write_connection(self):
        """Get the write connection (no context manager, uses lock directly)"""
        return self._write_conn or self._connection

    async def connect(self) -> None:
        """Initialize database connections and create schema"""
        if self._connection:
            return

        # Ensure directory exists for file-based database
        if self.db_path != ':memory:':
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            # Create main connection for backward compatibility
            self._connection = await aiosqlite.connect(
                self.db_path,
                isolation_level=None  # Autocommit mode
            )

            # Create write connection
            self._write_conn = await aiosqlite.connect(
                self.db_path,
                isolation_level=None
            )

            # Create read pool
            for _ in range(self._read_pool_size):
                conn = await aiosqlite.connect(
                    self.db_path,
                    isolation_level=None
                )
                self._read_pool.append(conn)

            # Apply optimizations to all connections
            all_conns = [self._connection, self._write_conn] + self._read_pool
            for conn in all_conns:
                await configure_sqlite_connection_async(
                    conn,
                    busy_timeout_ms=5000,
                    cache_size=10000,
                    temp_store="MEMORY",
                )

            # Create schema using write connection
            await self.create_schema()

            logger.info(f"SQLite backend connected with {self._read_pool_size} read connections")

        except Exception as e:
            logger.error(f"Failed to connect to SQLite: {e}")
            raise ConnectionError(f"SQLite connection failed: {e}") from e

    async def disconnect(self) -> None:
        """Close all database connections"""
        # Close read pool
        for conn in self._read_pool:
            await conn.close()
        self._read_pool.clear()

        # Close write connection
        if self._write_conn:
            await self._write_conn.close()
            self._write_conn = None

        # Close main connection
        if self._connection:
            await self._connection.close()
            self._connection = None

        logger.info("SQLite backend disconnected")

    async def create_schema(self) -> None:
        """Create database schema (idempotent)"""
        schema = """
        -- Main tasks table
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            queue_name TEXT NOT NULL,
            handler TEXT NOT NULL,
            payload TEXT,  -- JSON text
            priority INTEGER DEFAULT 2,
            status TEXT NOT NULL DEFAULT 'queued',

            -- Scheduling
            scheduled_at TEXT,  -- ISO format timestamp
            expires_at TEXT,

            -- Execution control
            max_retries INTEGER DEFAULT 3,
            retry_count INTEGER DEFAULT 0,
            retry_delay INTEGER DEFAULT 60,
            timeout INTEGER DEFAULT 300,

            -- Dependencies as JSON array
            depends_on TEXT,
            idempotency_key TEXT,

            -- Timestamps
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            queued_at TEXT,
            started_at TEXT,
            completed_at TEXT,

            -- Tracking
            worker_id TEXT,
            lease_id TEXT,
            execution_time REAL,  -- Seconds
            error TEXT,
            result TEXT,  -- JSON

            -- References for large payloads
            payload_ref TEXT,
            result_ref TEXT,

            -- Task metadata (stored as JSON)
            metadata TEXT NOT NULL DEFAULT '{}',

            CHECK (status IN ('pending', 'queued', 'running', 'completed', 'failed', 'cancelled', 'dead'))
        );

        -- Critical indexes for performance
        CREATE INDEX IF NOT EXISTS idx_dequeue
            ON tasks(queue_name, status, priority DESC, created_at)
            WHERE status = 'queued';

        CREATE INDEX IF NOT EXISTS idx_scheduled
            ON tasks(scheduled_at)
            WHERE scheduled_at IS NOT NULL AND status = 'queued';

        -- CRITICAL: Unique index for idempotency with INSERT OR IGNORE
        CREATE UNIQUE INDEX IF NOT EXISTS idx_idempotency
            ON tasks(idempotency_key)
            WHERE idempotency_key IS NOT NULL;

        -- For efficient keyset pagination during migration
        CREATE INDEX IF NOT EXISTS idx_migration
            ON tasks(id)
            WHERE status IN ('queued', 'scheduled');

        -- Task leases with unique constraint
        CREATE TABLE IF NOT EXISTS task_leases (
            lease_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            worker_id TEXT NOT NULL,
            acquired_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT NOT NULL,
            renewal_count INTEGER DEFAULT 0,

            -- Ensures only one lease per task
            UNIQUE(task_id),
            FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_lease_expiry
            ON task_leases(expires_at);

        -- Service leaders with TTL support
        CREATE TABLE IF NOT EXISTS service_leaders (
            service_name TEXT PRIMARY KEY,
            leader_id TEXT NOT NULL,
            acquired_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Index for efficient stale lock detection
        CREATE INDEX IF NOT EXISTS idx_leader_stale
            ON service_leaders(acquired_at);

        -- Dead letter queue
        CREATE TABLE IF NOT EXISTS dead_letter_queue (
            id TEXT PRIMARY KEY,
            original_task_id TEXT NOT NULL,
            queue_name TEXT NOT NULL,
            handler TEXT NOT NULL,
            payload TEXT,
            error_count INTEGER DEFAULT 1,
            last_error TEXT,
            moved_at TEXT DEFAULT CURRENT_TIMESTAMP,
            original_created_at TEXT,
            metadata TEXT
        );

        -- Schema version for migrations
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Insert initial version if not exists
        INSERT OR IGNORE INTO schema_version (version) VALUES (1);
        """

        async with self._lock:
            await self._connection.executescript(schema)
            await self._connection.commit()

            # Backfill metadata column for existing deployments if missing
            cursor = await self._connection.execute("PRAGMA table_info(tasks)")
            columns = await cursor.fetchall()
            column_names = {col[1] for col in columns}
            if 'metadata' not in column_names:
                await self._connection.execute("ALTER TABLE tasks ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
                await self._connection.commit()

    async def enqueue(self, task: Task) -> str:
        """Add a task to the queue"""
        # Check idempotency
        if task.idempotency_key:
            existing = await self.fetchval(
                "SELECT id FROM tasks WHERE idempotency_key = ?",
                task.idempotency_key
            )
            if existing:
                logger.debug(f"Task with idempotency_key {task.idempotency_key} already exists")
                return existing

        # Insert task
        await self.execute("""
            INSERT INTO tasks (
                id, queue_name, handler, payload, priority, status,
                scheduled_at, expires_at, max_retries, retry_count,
                retry_delay, timeout, depends_on, idempotency_key,
                created_at, queued_at, payload_ref, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            task.id,
            task.queue_name,
            task.handler,
            json.dumps(task.payload) if task.payload else None,
            task.priority,
            TaskStatus.QUEUED.value,
            task.scheduled_at.isoformat() if task.scheduled_at else None,
            task.expires_at.isoformat() if task.expires_at else None,
            task.max_retries,
            task.retry_count,
            task.retry_delay,
            task.timeout,
            json.dumps(task.depends_on) if task.depends_on else None,
            task.idempotency_key,
            datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            task.payload_ref,
            json.dumps(task.metadata) if task.metadata else '{}'
        )

        return task.id

    async def bulk_enqueue(self, tasks: list[Task]) -> list[str]:
        """
        Efficiently enqueue multiple tasks using INSERT OR IGNORE.
        This avoids N+1 queries for idempotency checks.
        """
        if not tasks:
            return []

        # Prepare values for bulk insert
        values = []
        for task in tasks:
            values.append((
                task.id,
                task.queue_name,
                task.handler,
                json.dumps(task.payload) if task.payload else None,
                task.priority,
                TaskStatus.QUEUED.value,
                task.scheduled_at.isoformat() if task.scheduled_at else None,
                task.expires_at.isoformat() if task.expires_at else None,
                task.max_retries,
                task.retry_count,
                task.retry_delay,
                task.timeout,
                json.dumps(task.depends_on) if task.depends_on else None,
                task.idempotency_key,
                datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
                task.payload_ref,
                json.dumps(task.metadata) if task.metadata else '{}'
            ))

        # Single atomic bulk insert with conflict handling
        async with self._lock:
            cursor = await self._connection.executemany("""
                INSERT OR IGNORE INTO tasks (
                    id, queue_name, handler, payload, priority, status,
                    scheduled_at, expires_at, max_retries, retry_count,
                    retry_delay, timeout, depends_on, idempotency_key,
                    created_at, queued_at, payload_ref, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, values)

            await self._connection.commit()

            inserted_count = cursor.rowcount
            logger.info(f"Bulk inserted {inserted_count} tasks (duplicates ignored)")

            # Return IDs of tasks (SQLite doesn't tell us which were ignored)
            return [task.id for task in tasks]

    async def dequeue_atomic(self, queue_name: str, worker_id: str) -> Optional[Task]:
        """
        Atomically dequeue next available task.
        SQLite limitation: Single writer, so this is a bottleneck.
        """
        async with self._lock:
            # Begin immediate transaction for write lock
            await self._connection.execute("BEGIN IMMEDIATE")

            try:
                # Find next available task
                cursor = await self._connection.execute("""
                    SELECT * FROM tasks
                    WHERE queue_name = ?
                      AND status = 'queued'
                      AND (scheduled_at IS NULL OR datetime(replace(scheduled_at, 'T', ' ')) <= datetime('now'))
                      AND (expires_at IS NULL OR datetime(replace(expires_at, 'T', ' ')) > datetime('now'))
                      AND NOT EXISTS (
                          SELECT 1
                          FROM json_each(COALESCE(depends_on, '[]')) AS deps
                          LEFT JOIN tasks dep ON dep.id = deps.value
                          WHERE dep.id IS NULL OR dep.status != 'completed'
                      )
                    ORDER BY priority ASC, created_at ASC
                    LIMIT 1
                """, (queue_name,))
                row = await cursor.fetchone()

                if not row:
                    await self._connection.rollback()
                    return None

                # Convert row to dict using cursor description
                columns = [desc[0] for desc in cursor.description]
                task = await self._row_to_task(dict(zip(columns, row)))

                # Create lease
                lease_id = str(uuid.uuid4())
                lease_duration = task.timeout or self.config.lease_duration_seconds
                expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=lease_duration)

                await self._connection.execute("""
                    INSERT INTO task_leases (lease_id, task_id, worker_id, expires_at)
                    VALUES (?, ?, ?, ?)
                """, (lease_id, task.id, worker_id, expires_at.isoformat()))

                # Update task status
                await self._connection.execute("""
                    UPDATE tasks
                    SET status = 'running',
                        worker_id = ?,
                        lease_id = ?,
                        started_at = datetime('now')
                    WHERE id = ?
                """, (worker_id, lease_id, task.id))

                await self._connection.commit()

                task.status = TaskStatus.RUNNING
                task.worker_id = worker_id
                task.lease_id = lease_id
                task.started_at = datetime.now(timezone.utc).replace(tzinfo=None)

                return task

            except Exception as e:
                await self._connection.rollback()
                raise TransactionError(f"Dequeue failed: {e}") from e

    async def get_ready_tasks(self, queue_name: Optional[str] = None) -> list[str]:
        """
        Efficient dependency resolution with only 2 queries.
        Avoids N+1 query pattern.
        """
        # Step 1: Get all queued tasks in a single query
        query = (
            "SELECT id, depends_on FROM tasks "
            "WHERE status = 'queued' "
            "AND (scheduled_at IS NULL OR datetime(replace(scheduled_at, 'T', ' ')) <= datetime('now')) "
            "AND (expires_at IS NULL OR datetime(replace(expires_at, 'T', ' ')) > datetime('now'))"
        )
        params = []
        if queue_name:
            query += " AND queue_name = ?"
            params.append(queue_name)

        rows = await self.fetch(query, *params)

        ready = []
        tasks_with_deps = {}
        all_dependency_ids = set()

        # Process tasks and collect all dependency IDs
        for row in rows:
            task_id = row['id']
            deps = json.loads(row['depends_on']) if row['depends_on'] else []

            if not deps:
                # No dependencies, task is ready
                ready.append(task_id)
            else:
                # Has dependencies, need to check them
                tasks_with_deps[task_id] = set(deps)
                all_dependency_ids.update(deps)

        if not all_dependency_ids:
            # No dependencies to check
            return ready

        # Step 2: Get status of ALL dependencies in a SINGLE query
        placeholders = ','.join(['?' for _ in all_dependency_ids])
        dependency_ids_clause = f"({placeholders})"
        completed_deps_query_template = """
            SELECT id FROM tasks
            WHERE status = 'completed'
            AND id IN {dependency_ids_clause}
        """
        completed_deps_query = completed_deps_query_template.format_map(locals())  # nosec B608
        completed_deps_rows = await self.fetch(
            completed_deps_query,
            *list(all_dependency_ids)
        )
        completed_deps_set = {row['id'] for row in completed_deps_rows}

        # Step 3: Check dependencies in memory (fast!)
        for task_id, deps in tasks_with_deps.items():
            if deps.issubset(completed_deps_set):
                ready.append(task_id)

        return ready

    async def ack(
        self,
        task_id: str,
        result: Optional[Any] = None,
        *,
        lease_id: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """Acknowledge task completion"""
        result_json = json.dumps(result) if result is not None else None

        affected = await self.execute(
            """
            UPDATE tasks
            SET status = 'completed',
                completed_at = datetime('now'),
                result = ?,
                execution_time = (julianday('now') - julianday(started_at)) * 86400,
                worker_id = NULL,
                lease_id = NULL
            WHERE id = ? AND status = 'running'
              AND (? IS NULL OR lease_id = ?)
              AND (? IS NULL OR worker_id = ?)
            """,
            result_json,
            task_id,
            lease_id,
            lease_id,
            worker_id,
            worker_id,
        )

        if affected:
            # Delete lease
            await self.execute("DELETE FROM task_leases WHERE task_id = ?", task_id)

        return affected > 0

    async def nack(
        self,
        task_id: str,
        error: str,
        retry: bool = True,
        *,
        lease_id: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """Handle task failure"""
        task = await self.get_task(task_id)
        if not task:
            return False

        if retry and task.retry_count < task.max_retries:
            # Schedule retry
            retry_delay = task.calculate_retry_delay()
            scheduled_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=retry_delay)

            affected = await self.execute(
                """
                UPDATE tasks
                SET status = 'queued',
                    retry_count = retry_count + 1,
                    scheduled_at = ?,
                    error = ?,
                    worker_id = NULL,
                    lease_id = NULL
                WHERE id = ? AND status = 'running'
                  AND (? IS NULL OR lease_id = ?)
                  AND (? IS NULL OR worker_id = ?)
                """,
                scheduled_at.isoformat(),
                error,
                task_id,
                lease_id,
                lease_id,
                worker_id,
                worker_id,
            )
        else:
            # Move to failed or DLQ
            affected = await self.execute(
                """
                UPDATE tasks
                SET status = 'failed',
                    completed_at = datetime('now'),
                    error = ?,
                    worker_id = NULL,
                    lease_id = NULL
                WHERE id = ? AND status = 'running'
                  AND (? IS NULL OR lease_id = ?)
                  AND (? IS NULL OR worker_id = ?)
                """,
                error,
                task_id,
                lease_id,
                lease_id,
                worker_id,
                worker_id,
            )

        if affected <= 0:
            return False

        # Delete lease
        await self.execute("DELETE FROM task_leases WHERE task_id = ?", task_id)

        return True

    async def _row_to_task(self, row: dict[str, Any]) -> Task:
        """Convert database row dict to Task object"""
        task_dict = dict(row)

        # Parse JSON fields
        for key in ['payload', 'depends_on', 'result', 'metadata']:
            if isinstance(task_dict.get(key), str):
                try:
                    task_dict[key] = json.loads(task_dict[key])
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON for key {key} in task {task_dict.get('id')}")

        # Convert status string to enum
        if task_dict.get('status'):
            task_dict['status'] = TaskStatus(task_dict['status'])

        return Task.from_dict(task_dict)

    # Implement remaining abstract methods...

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        row = await self.fetchrow("SELECT * FROM tasks WHERE id = ?", task_id)
        if row:
            return await self._row_to_task(row)
        return None

    async def get_task_by_idempotency_key(self, idempotency_key: str) -> Optional[str]:
        """Get task ID by idempotency key"""
        return await self.fetchval(
            "SELECT id FROM tasks WHERE idempotency_key = ?",
            idempotency_key
        )

    async def update_task(self, task: Task) -> bool:
        """Update task"""
        affected = await self.execute("""
            UPDATE tasks
            SET status = ?, error = ?, result = ?, worker_id = ?, lease_id = ?
            WHERE id = ?
        """,
            task.status.value,
            task.error,
            json.dumps(task.result) if task.result else None,
            task.worker_id,
            task.lease_id,
            task.id
        )
        return affected > 0

    async def get_queue_size(self, queue_name: str) -> int:
        """Get queue size"""
        return await self.fetchval(
            "SELECT COUNT(*) FROM tasks WHERE queue_name = ? AND status = 'queued'",
            queue_name
        ) or 0

    async def clear_queue(self, queue_name: str) -> int:
        """Clear queue"""
        affected = await self.execute(
            "DELETE FROM tasks WHERE queue_name = ? AND status = 'queued'",
            queue_name
        )
        return affected

    async def get_dead_letter_queue(self) -> list[Task]:
        """Get DLQ tasks"""
        await self.fetch("SELECT * FROM dead_letter_queue")
        # Convert to tasks (simplified)
        return []  # TODO: Implement if needed

    async def move_to_dlq(self, task_id: str, reason: str) -> bool:
        """Move task to DLQ"""
        task = await self.get_task(task_id)
        if not task:
            return False

        await self.execute("""
            INSERT INTO dead_letter_queue (
                id, original_task_id, queue_name, handler, payload, last_error, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            str(uuid.uuid4()),
            task.id,
            task.queue_name,
            task.handler,
            json.dumps(task.payload),
            reason,
            json.dumps(task.metadata) if task.metadata else '{}'
        )

        await self.execute("UPDATE tasks SET status = 'dead' WHERE id = ?", task_id)
        return True

    # Lease management

    async def create_lease(self, lease_id: str, task_id: str,
                          worker_id: str, expires_at: datetime) -> bool:
        """Create lease"""
        try:
            await self.execute("""
                INSERT INTO task_leases (lease_id, task_id, worker_id, expires_at)
                VALUES (?, ?, ?, ?)
            """, lease_id, task_id, worker_id, expires_at.isoformat())
            return True
        except Exception as e:
            logger.warning(f"Failed to create task lease: task_id={task_id}, lease_id={lease_id}, error={e}")
            return False

    async def renew_lease(self, task_id: str, lease_id: str) -> bool:
        """Renew lease by extending expiration based on task timeout or config."""
        duration_seconds = self.config.lease_duration_seconds
        try:
            task_timeout = await self.fetchval(
                "SELECT timeout FROM tasks WHERE id = ?",
                task_id
            )
            if task_timeout:
                # Ensure positive integer duration
                task_timeout = int(task_timeout)
                if task_timeout > 0:
                    duration_seconds = task_timeout
        except Exception as timeout_lookup_error:
            # Fall back to config duration if lookup fails or value invalid
            logger.debug("Scheduler sqlite backend failed to resolve task timeout override", exc_info=timeout_lookup_error)

        new_expires = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=duration_seconds)

        affected = await self.execute("""
            UPDATE task_leases
            SET expires_at = ?, renewal_count = renewal_count + 1
            WHERE lease_id = ? AND task_id = ?
        """, new_expires.isoformat(), lease_id, task_id)
        return affected > 0

    async def delete_lease(self, lease_id: str) -> bool:
        """Delete lease"""
        affected = await self.execute(
            "DELETE FROM task_leases WHERE lease_id = ?",
            lease_id
        )
        return affected > 0

    async def get_expired_leases(self) -> list[dict[str, Any]]:
        """Get expired leases"""
        return await self.fetch(
            "SELECT * FROM task_leases WHERE datetime(replace(expires_at, 'T', ' ')) < datetime('now')"
        )

    # Transaction support

    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager"""
        async with self._lock:
            await self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise

    # Utility methods

    async def execute(self, query: str, *args) -> int:
        """Execute query and return affected rows"""
        async with self._lock:
            cursor = await self._connection.execute(query, args)
            await self._connection.commit()
            return cursor.rowcount

    async def fetch(self, query: str, *args) -> list[dict[str, Any]]:
        """Fetch multiple rows using read pool for better concurrency"""
        async with self._get_read_connection() as conn:
            cursor = await conn.execute(query, args)
            rows = await cursor.fetchall()
            if rows:
                # Convert to dictionaries
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
            return []

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch single value using read pool"""
        async with self._get_read_connection() as conn:
            cursor = await conn.execute(query, args)
            row = await cursor.fetchone()
            return row[0] if row else None

    async def fetchrow(self, query: str, *args) -> Optional[dict[str, Any]]:
        """Fetch single row using read pool and return as dict"""
        async with self._get_read_connection() as conn:
            cursor = await conn.execute(query, args)
            row = await cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None

    async def reclaim_expired_leases(self) -> int:
        """
        Reclaim tasks with expired leases back to queued status.

        Returns:
            Number of tasks reclaimed
        """
        async with self._lock:
            # Begin transaction
            await self._connection.execute("BEGIN IMMEDIATE")

            try:
                # Find all expired leases
                cursor = await self._connection.execute("""
                    SELECT task_id FROM task_leases
                    WHERE datetime(replace(expires_at, 'T', ' ')) < datetime('now')
                """)
                expired_rows = await cursor.fetchall()

                if not expired_rows:
                    await self._connection.rollback()
                    return 0

                task_ids = [row[0] for row in expired_rows]

                # Reset tasks to queued status
                placeholders = ','.join(['?' for _ in task_ids])
                task_ids_clause = f"({placeholders})"
                reclaim_tasks_sql_template = """
                    UPDATE tasks
                    SET status = 'queued',
                        worker_id = NULL,
                        lease_id = NULL,
                        started_at = NULL,
                        retry_count = retry_count + 1
                    WHERE id IN {task_ids_clause}
                    AND status = 'running'
                """
                reclaim_tasks_sql = reclaim_tasks_sql_template.format_map(locals())  # nosec B608
                await self._connection.execute(reclaim_tasks_sql, task_ids)

                # Delete expired leases
                delete_leases_sql_template = """
                    DELETE FROM task_leases
                    WHERE task_id IN {task_ids_clause}
                """
                delete_leases_sql = delete_leases_sql_template.format_map(locals())  # nosec B608
                await self._connection.execute(delete_leases_sql, task_ids)

                await self._connection.commit()

                reclaimed_count = len(task_ids)
                if reclaimed_count > 0:
                    logger.info(f"Reclaimed {reclaimed_count} tasks with expired leases")

                return reclaimed_count

            except Exception as e:
                await self._connection.rollback()
                logger.error(f"Failed to reclaim expired leases: {e}")
                raise TransactionError(f"Lease reclamation failed: {e}") from e

    # Leader election (TTL-based) using service_leaders table

    async def acquire_leader(self, resource: str, leader_id: str, ttl: int) -> bool:
        """Try to acquire leadership for a resource using TTL semantics."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self._lock:
            # Begin immediate transaction to avoid races
            await self._connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await self._connection.execute(
                    "SELECT leader_id, acquired_at FROM service_leaders WHERE service_name = ?",
                    (resource,)
                )
                row = await cursor.fetchone()
                if not row:
                    # No leader, acquire
                    await self._connection.execute(
                        "INSERT INTO service_leaders (service_name, leader_id, acquired_at) VALUES (?, ?, ?)",
                        (resource, leader_id, now.isoformat())
                    )
                    await self._connection.commit()
                    return True
                # Existing leader; check TTL expiry or same leader
                current_leader, acquired_at = row[0], row[1]
                expired = False
                try:
                    acquired_dt = datetime.fromisoformat(acquired_at) if isinstance(acquired_at, str) else now
                    expired = (now - acquired_dt).total_seconds() >= ttl
                except Exception:
                    expired = True
                if expired or current_leader == leader_id:
                    await self._connection.execute(
                        "UPDATE service_leaders SET leader_id = ?, acquired_at = ? WHERE service_name = ?",
                        (leader_id, now.isoformat(), resource)
                    )
                    await self._connection.commit()
                    return True
                # Another valid leader holds it
                await self._connection.rollback()
                return False
            except Exception:
                await self._connection.rollback()
                return False

    async def renew_leader(self, resource: str, leader_id: str, ttl: int) -> bool:
        """Renew leadership by updating acquired_at if leader matches."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self._lock:
            cursor = await self._connection.execute(
                "SELECT leader_id FROM service_leaders WHERE service_name = ?",
                (resource,)
            )
            row = await cursor.fetchone()
            if not row or row[0] != leader_id:
                return False
            await self._connection.execute(
                "UPDATE service_leaders SET acquired_at = ? WHERE service_name = ?",
                (now.isoformat(), resource)
            )
            await self._connection.commit()
            return True

    async def release_leader(self, resource: str, leader_id: str) -> bool:
        """Release leadership if currently held by leader_id."""
        async with self._lock:
            affected = await self._connection.execute(
                "DELETE FROM service_leaders WHERE service_name = ? AND leader_id = ?",
                (resource, leader_id)
            )
            await self._connection.commit()
            return affected.rowcount > 0

    # Schema management

    async def get_schema_version(self) -> int:
        """Get schema version"""
        return await self.fetchval("SELECT MAX(version) FROM schema_version") or 0

    async def migrate_schema(self, target_version: int) -> None:
        """Migrate schema (placeholder)"""
        current = await self.get_schema_version()
        if current < target_version:
            # TODO: Implement migrations
            pass

    def get_status(self) -> dict[str, Any]:
        """Get backend status for monitoring"""
        return {
            'type': 'sqlite',
            'database': self.db_path,
            'connected': self._connection is not None,
            'read_pool_size': len(self._read_pool),
            'write_connection': self._write_conn is not None
        }
