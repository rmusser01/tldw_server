"""
Abstract base class for queue backends.
Defines the interface that all backends must implement.
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from .task import Task


class QueueBackend(ABC):
    """
    Abstract base class for queue backends.

    All backend implementations (PostgreSQL, SQLite, Memory) must
    implement these methods to ensure compatibility with the scheduler.
    """

    @abstractmethod
    async def connect(self) -> None:
        """
        Initialize backend connection and create schema if needed.
        This method should be idempotent.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close backend connection and cleanup resources.
        """
        pass

    @abstractmethod
    async def enqueue(self, task: Task) -> str:
        """
        Add a task to the queue.

        Args:
            task: Task to enqueue

        Returns:
            Task ID

        Raises:
            DuplicateTaskError: If idempotency_key already exists
        """
        pass

    @abstractmethod
    async def bulk_enqueue(self, tasks: list[Task]) -> list[str]:
        """
        Add multiple tasks to the queue in a single operation.

        Args:
            tasks: List of tasks to enqueue

        Returns:
            List of task IDs that were successfully enqueued
        """
        pass

    @abstractmethod
    async def dequeue_atomic(self, queue_name: str, worker_id: str) -> Optional[Task]:
        """
        Atomically dequeue the next available task.

        This operation must be atomic - the task should be marked as
        running by the worker in a single operation to prevent races.

        Args:
            queue_name: Name of queue to dequeue from
            worker_id: ID of worker claiming the task

        Returns:
            Next available task or None if queue is empty
        """
        pass

    @abstractmethod
    async def ack(
        self,
        task_id: str,
        result: Optional[Any] = None,
        *,
        lease_id: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """
        Acknowledge successful task completion.

        Args:
            task_id: ID of completed task
            result: Task execution result
            lease_id: Optional lease identifier that must still own the task
            worker_id: Optional worker identifier that must still own the task

        Returns:
            True if task was acknowledged
        """
        pass

    @abstractmethod
    async def nack(
        self,
        task_id: str,
        error: str,
        retry: bool = True,
        *,
        lease_id: Optional[str] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """
        Negative acknowledgment - task failed.

        Args:
            task_id: ID of failed task
            error: Error message
            retry: Whether to retry the task
            lease_id: Optional lease identifier that must still own the task
            worker_id: Optional worker identifier that must still own the task

        Returns:
            True if task was nacked
        """
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task or None if not found
        """
        pass

    @abstractmethod
    async def get_task_by_idempotency_key(self, idempotency_key: str) -> Optional[str]:
        """
        Get a task ID by idempotency key, if it exists.

        Args:
            idempotency_key: Idempotency key to look up

        Returns:
            Task ID if found, otherwise None
        """
        pass

    @abstractmethod
    async def update_task(self, task: Task) -> bool:
        """
        Update an existing task.

        Args:
            task: Updated task

        Returns:
            True if task was updated
        """
        pass

    @abstractmethod
    async def get_queue_size(self, queue_name: str) -> int:
        """
        Get number of pending tasks in queue.

        Args:
            queue_name: Queue name

        Returns:
            Number of pending tasks
        """
        pass

    @abstractmethod
    async def get_ready_tasks(self, queue_name: Optional[str] = None) -> list[str]:
        """
        Get IDs of tasks that are ready to run (dependencies satisfied).

        Args:
            queue_name: Optional queue filter

        Returns:
            List of task IDs
        """
        pass

    @abstractmethod
    async def clear_queue(self, queue_name: str) -> int:
        """
        Remove all tasks from a queue.

        Args:
            queue_name: Queue to clear

        Returns:
            Number of tasks removed
        """
        pass

    @abstractmethod
    async def get_dead_letter_queue(self) -> list[Task]:
        """
        Get tasks in the dead letter queue.

        Returns:
            List of dead tasks
        """
        pass

    @abstractmethod
    async def move_to_dlq(self, task_id: str, reason: str) -> bool:
        """
        Move a task to the dead letter queue.

        Args:
            task_id: Task to move
            reason: Reason for moving to DLQ

        Returns:
            True if task was moved
        """
        pass

    # Lease management

    @abstractmethod
    async def create_lease(self, lease_id: str, task_id: str,
                          worker_id: str, expires_at: datetime) -> bool:
        """
        Create a lease for a task.

        Args:
            lease_id: Unique lease ID
            task_id: Task being leased
            worker_id: Worker holding the lease
            expires_at: Lease expiration time

        Returns:
            True if lease was created
        """
        pass

    @abstractmethod
    async def renew_lease(self, task_id: str, lease_id: str) -> bool:
        """
        Renew an existing lease.

        Args:
            task_id: Task whose lease should be renewed
            lease_id: Lease identifier

        Returns:
            True if lease was renewed
        """
        pass

    @abstractmethod
    async def delete_lease(self, lease_id: str) -> bool:
        """
        Delete a lease.

        Args:
            lease_id: Lease to delete

        Returns:
            True if lease was deleted
        """
        pass

    @abstractmethod
    async def get_expired_leases(self) -> list[dict[str, Any]]:
        """
        Get all expired leases.

        Returns:
            List of expired lease records
        """
        pass

    # Transaction support

    @abstractmethod
    @asynccontextmanager
    async def transaction(self) -> AbstractAsyncContextManager:
        """
        Create a database transaction context.

        Usage:
            async with backend.transaction():
                await backend.enqueue(task1)
                await backend.enqueue(task2)
                # Both tasks committed atomically
        """
        pass

    # Utility methods

    @abstractmethod
    async def execute(self, query: str, *args) -> Any:
        """
        Execute a raw query (for backend-specific operations).

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Query result
        """
        pass

    @abstractmethod
    async def fetch(self, query: str, *args) -> list[dict[str, Any]]:
        """
        Execute a query and fetch results.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            List of result rows as dictionaries
        """
        pass

    @abstractmethod
    async def fetchval(self, query: str, *args) -> Any:
        """
        Execute a query and fetch a single value.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single value from first row
        """
        pass

    @abstractmethod
    async def fetchrow(self, query: str, *args) -> Optional[dict[str, Any]]:
        """
        Execute a query and fetch a single row.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            First row as dictionary or None
        """
        pass

    # Schema management

    @abstractmethod
    async def create_schema(self) -> None:
        """
        Create database schema for the queue system.
        This should be idempotent.
        """
        pass

    @abstractmethod
    async def get_schema_version(self) -> int:
        """
        Get current schema version.

        Returns:
            Schema version number
        """
        pass

    @abstractmethod
    async def migrate_schema(self, target_version: int) -> None:
        """
        Migrate schema to target version.

        Args:
            target_version: Target schema version
        """
        pass
