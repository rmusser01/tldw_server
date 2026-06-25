"""
In-memory backend for testing and development.
Not suitable for production use.
"""

import asyncio
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from loguru import logger

from ..base import Task, TaskStatus
from ..base.queue_backend import QueueBackend
from ..config import SchedulerConfig


class MemoryBackend(QueueBackend):
    """
    Simple in-memory backend for testing.

    WARNING: This backend is NOT persistent and NOT thread-safe.
    All data is lost when the process terminates.
    Use only for testing and development.
    """

    def __init__(self, config: SchedulerConfig):
        """
        Initialize in-memory backend.

        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.tasks: dict[str, Task] = {}
        self.queues: dict[str, list[str]] = defaultdict(list)
        self.idempotency_keys: dict[str, str] = {}
        self.leaders: dict[str, dict[str, Any]] = {}
        self.dead_letter_queue: list[Task] = []
        self._schema_version = 1
        self._lock = asyncio.Lock()

        logger.warning(
            "MemoryBackend initialized. This backend is NOT persistent "
            "and should only be used for testing!"
        )

    async def connect(self) -> None:
        """No-op for memory backend."""
        logger.info("Memory backend connected")

    async def disconnect(self) -> None:
        """Clear all data."""
        async with self._lock:
            self.tasks.clear()
            self.queues.clear()
            self.idempotency_keys.clear()
            self.leaders.clear()
        logger.info("Memory backend disconnected")

    async def enqueue(self, task: Task) -> str:
        """Add task to queue."""
        async with self._lock:
            # Check idempotency
            if task.idempotency_key:
                if task.idempotency_key in self.idempotency_keys:
                    # Return existing task ID
                    return self.idempotency_keys[task.idempotency_key]
                self.idempotency_keys[task.idempotency_key] = task.id

            # Store task
            self.tasks[task.id] = task

            # Add to queue if ready
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.QUEUED

            if task.status == TaskStatus.QUEUED:
                queue_name = task.queue_name or self.config.default_queue_name
                self.queues[queue_name].append(task.id)

            return task.id

    async def bulk_enqueue(self, tasks: list[Task]) -> list[str]:
        """Enqueue multiple tasks."""
        task_ids = []
        for task in tasks:
            task_id = await self.enqueue(task)
            task_ids.append(task_id)
        return task_ids

    async def dequeue_atomic(self, queue_name: str, worker_id: str) -> Optional[Task]:
        """Atomically dequeue next task."""
        async with self._lock:
            if queue_name not in self.queues or not self.queues[queue_name]:
                return None

            # Find next available task
            for i, task_id in enumerate(self.queues[queue_name]):
                task = self.tasks.get(task_id)
                if not task:
                    continue

                # Check if scheduled
                if task.scheduled_at and task.scheduled_at > datetime.now(timezone.utc).replace(tzinfo=None):
                    continue

                # Check dependencies
                if task.depends_on:
                    deps_complete = all(
                        self.tasks.get(dep_id, Task()).status == TaskStatus.COMPLETED
                        for dep_id in task.depends_on
                    )
                    if not deps_complete:
                        continue

                # Found eligible task
                self.queues[queue_name].pop(i)

                # Update task
                task.status = TaskStatus.RUNNING
                task.worker_id = worker_id
                task.lease_id = str(uuid.uuid4())
                task.lease_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
                    seconds=self.config.lease_duration_seconds
                )
                task.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
                task.retry_count += 1

                return task

            return None

    async def ack(
        self,
        task_id: str,
        result: Optional[Any] = None,
        *,
        lease_id: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """Acknowledge task completion."""
        async with self._lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]
            if task.status != TaskStatus.RUNNING:
                return False
            if lease_id is not None and task.lease_id != lease_id:
                return False
            if worker_id is not None and task.worker_id != worker_id:
                return False

            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
            task.result = result
            task.lease_id = None
            task.lease_expires_at = None
            task.worker_id = None

            return True

    async def nack(
        self,
        task_id: str,
        error: Optional[str] = None,
        retry: bool = True,
        *,
        lease_id: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """Negative acknowledge."""
        async with self._lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]
            if task.status != TaskStatus.RUNNING:
                return False
            if lease_id is not None and task.lease_id != lease_id:
                return False
            if worker_id is not None and task.worker_id != worker_id:
                return False

            task.error = error

            if not retry or task.retry_count >= task.max_retries:
                task.status = TaskStatus.FAILED
            else:
                task.status = TaskStatus.QUEUED
                task.scheduled_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=task.retry_delay)
                queue_name = task.queue_name or self.config.default_queue_name
                self.queues[queue_name].append(task_id)

            task.lease_id = None
            task.lease_expires_at = None
            task.worker_id = None

            return True

    async def renew_lease(self, task_id: str, lease_id: str) -> bool:
        """Renew task lease."""
        async with self._lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]
            if task.lease_id != lease_id or task.status != TaskStatus.RUNNING:
                return False

            task.lease_expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
                seconds=self.config.lease_duration_seconds
            )
            return True

    async def create_lease(self, lease_id: str, task_id: str,
                           worker_id: str, expires_at: datetime) -> bool:
        """Create a lease for a task."""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            task.lease_id = lease_id
            task.worker_id = worker_id
            task.lease_expires_at = expires_at
            return True

    async def delete_lease(self, lease_id: str) -> bool:
        """Delete a lease by lease ID."""
        async with self._lock:
            for task in self.tasks.values():
                if task.lease_id == lease_id:
                    task.lease_id = None
                    task.lease_expires_at = None
                    return True
            return False

    async def get_expired_leases(self) -> list[dict[str, Any]]:
        """Get expired leases."""
        expired = []
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self._lock:
            for task in self.tasks.values():
                if task.lease_id and task.lease_expires_at and task.lease_expires_at < now:
                    expired.append({
                        'lease_id': task.lease_id,
                        'task_id': task.id,
                        'worker_id': task.worker_id,
                        'expires_at': task.lease_expires_at
                    })
        return expired

    async def reclaim_expired_leases(self) -> int:
        """Reclaim tasks with expired leases."""
        reclaimed = 0
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        async with self._lock:
            for task in self.tasks.values():
                if (task.status == TaskStatus.RUNNING and
                    task.lease_expires_at and
                    task.lease_expires_at < now):

                    task.status = TaskStatus.QUEUED
                    task.lease_id = None
                    task.lease_expires_at = None
                    task.worker_id = None
                    task.error = "Lease expired"

                    queue_name = task.queue_name or self.config.default_queue_name
                    self.queues[queue_name].append(task.id)
                    reclaimed += 1

        return reclaimed

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)

    async def get_task_by_idempotency_key(self, idempotency_key: str) -> Optional[str]:
        """Lookup task ID for a given idempotency key."""
        return self.idempotency_keys.get(idempotency_key)

    async def update_task(self, task: Task) -> bool:
        """Update task state."""
        async with self._lock:
            if task.id not in self.tasks:
                return False
            self.tasks[task.id] = task
            return True

    async def get_queue_size(self, queue_name: str) -> int:
        """Get queue size."""
        return len(self.queues.get(queue_name, []))

    async def clear_queue(self, queue_name: str) -> int:
        """Clear queued tasks for a queue."""
        async with self._lock:
            queue_ids = list(self.queues.get(queue_name, []))
            removed = 0
            for task_id in queue_ids:
                task = self.tasks.get(task_id)
                if task and task.status == TaskStatus.QUEUED:
                    if task.idempotency_key:
                        self.idempotency_keys.pop(task.idempotency_key, None)
                    self.tasks.pop(task_id, None)
                    removed += 1
            self.queues[queue_name] = []
            return removed

    async def get_dead_letter_queue(self) -> list[Task]:
        """Get tasks in the dead letter queue."""
        return list(self.dead_letter_queue)

    async def move_to_dlq(self, task_id: str, reason: str) -> bool:
        """Move task to dead letter queue."""
        async with self._lock:
            task = self.tasks.get(task_id)
            if not task:
                return False
            task.status = TaskStatus.DEAD
            task.error = reason
            if task.queue_name in self.queues and task_id in self.queues[task.queue_name]:
                self.queues[task.queue_name].remove(task_id)
            self.dead_letter_queue.append(task)
            return True

    async def get_ready_tasks(self, queue_name: Optional[str] = None) -> list[str]:
        """Get tasks ready to run."""
        ready = []
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.QUEUED:
                continue
            if queue_name and task.queue_name != queue_name:
                continue

            # Check scheduled time
            if task.scheduled_at and task.scheduled_at > now:
                continue

            # Check dependencies
            if task.depends_on:
                deps_complete = all(
                    self.tasks.get(dep_id, Task()).status == TaskStatus.COMPLETED
                    for dep_id in task.depends_on
                )
                if not deps_complete:
                    continue

            ready.append(task_id)

        return ready

    async def acquire_leader(self, resource: str, leader_id: str, ttl: int) -> bool:
        """Try to acquire leadership."""
        async with self._lock:
            now = datetime.now(timezone.utc).replace(tzinfo=None)

            if resource in self.leaders:
                leader = self.leaders[resource]
                # Check if expired or same leader
                if leader['expires_at'] > now and leader['leader_id'] != leader_id:
                    return False

            self.leaders[resource] = {
                'leader_id': leader_id,
                'expires_at': now + timedelta(seconds=ttl)
            }
            return True

    async def renew_leader(self, resource: str, leader_id: str, ttl: int) -> bool:
        """Renew leadership."""
        async with self._lock:
            if resource not in self.leaders:
                return False

            leader = self.leaders[resource]
            if leader['leader_id'] != leader_id:
                return False

            leader['expires_at'] = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=ttl)
            return True

    async def release_leader(self, resource: str, leader_id: str) -> bool:
        """Release leadership."""
        async with self._lock:
            if resource not in self.leaders:
                return False

            leader = self.leaders[resource]
            if leader['leader_id'] != leader_id:
                return False

            del self.leaders[resource]
            return True

    async def create_schema(self) -> None:
        """No-op schema creation for memory backend."""
        return None

    async def get_schema_version(self) -> int:
        """Return in-memory schema version."""
        return self._schema_version

    async def migrate_schema(self, target_version: int) -> None:
        """No-op migration for memory backend."""
        self._schema_version = max(self._schema_version, target_version)

    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager using the in-memory lock."""
        async with self._lock:
            yield self

    async def fetch(self, query: str, *args) -> list[dict[str, Any]]:
        """Not supported for memory backend."""
        raise NotImplementedError("Memory backend does not support raw queries")

    async def fetchval(self, query: str, *args) -> Any:
        """Not supported for memory backend."""
        raise NotImplementedError("Memory backend does not support raw queries")

    async def fetchrow(self, query: str, *args) -> Optional[dict[str, Any]]:
        """Not supported for memory backend."""
        raise NotImplementedError("Memory backend does not support raw queries")

    async def execute(self, query: str, *args) -> Any:
        """Not supported for memory backend."""
        raise NotImplementedError("Memory backend does not support raw queries")

    def get_status(self) -> dict[str, Any]:
        """Get backend status."""
        return {
            "type": "memory",
            "tasks": len(self.tasks),
            "queues": {name: len(ids) for name, ids in self.queues.items()},
            "leaders": list(self.leaders.keys())
        }
