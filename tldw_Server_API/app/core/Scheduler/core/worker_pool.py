"""
Worker pool management for task execution.
Handles dynamic scaling, health checks, and graceful shutdown.
"""

import asyncio
import contextlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from loguru import logger

from ..base import Task, TaskStatus
from ..base.queue_backend import QueueBackend
from ..base.registry import TaskRegistry
from ..config import SchedulerConfig
from ..services.lease_service import LeaseService


class WorkerState(Enum):
    """Worker states."""
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class Worker:
    """
    Individual worker that processes tasks.
    """

    def __init__(self,
                 worker_id: str,
                 backend: QueueBackend,
                 registry: TaskRegistry,
                 config: SchedulerConfig,
                 queue_name: str = "default"):
        """
        Initialize worker.

        Args:
            worker_id: Unique worker identifier
            backend: Queue backend
            registry: Task handler registry
            config: Scheduler configuration
            queue_name: Queue to process
        """
        self.worker_id = worker_id
        self.backend = backend
        self.registry = registry
        self.config = config
        self.queue_name = queue_name

        self.state = WorkerState.IDLE
        self.current_task: Optional[Task] = None
        self.tasks_processed = 0
        self.tasks_failed = 0
        self.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.last_task_at: Optional[datetime] = None

        self._task: Optional[asyncio.Task] = None
        self._lease_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start the worker."""
        if self._task:
            logger.warning(f"Worker {self.worker_id} already running")
            return

        self._task = asyncio.create_task(self._run())
        logger.info(f"Worker {self.worker_id} started for queue {self.queue_name}")

    async def stop(self, timeout: int = 30) -> None:
        """
        Stop the worker gracefully.

        Args:
            timeout: Maximum time to wait for current task
        """
        logger.info(f"Stopping worker {self.worker_id}")
        self.state = WorkerState.STOPPING
        self._stop_event.set()

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Worker {self.worker_id} stop timeout, cancelling")
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

        # Cancel lease renewal if active
        if self._lease_task:
            self._lease_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._lease_task

        self.state = WorkerState.STOPPED
        logger.info(f"Worker {self.worker_id} stopped")

    async def _run(self) -> None:
        """Main worker loop."""
        logger.debug(f"Worker {self.worker_id} entering main loop")

        while not self._stop_event.is_set():
            try:
                # Check if we should recycle
                if (self.config.worker_recycle_after_tasks > 0 and
                    self.tasks_processed >= self.config.worker_recycle_after_tasks):
                    logger.info(
                        f"Worker {self.worker_id} recycling after "
                        f"{self.tasks_processed} tasks"
                    )
                    break

                # Try to get a task
                self.state = WorkerState.IDLE
                task = await self._get_next_task()

                if not task:
                    # No task available, wait a bit
                    await asyncio.sleep(1)
                    continue

                # Check if task was cancelled before processing
                fresh_task = await self.backend.get_task(task.id)
                if fresh_task and fresh_task.status == TaskStatus.CANCELLED:
                    logger.info(f"Skipping cancelled task {task.id}")
                    continue

                # Process the task
                self.state = WorkerState.BUSY
                self.current_task = task
                self.last_task_at = datetime.now(timezone.utc).replace(tzinfo=None)

                success = await self._process_task(task)

                if success:
                    self.tasks_processed += 1
                else:
                    self.tasks_failed += 1

                self.current_task = None

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                self.state = WorkerState.ERROR
                await asyncio.sleep(5)  # Back off on error

    async def _get_next_task(self) -> Optional[Task]:
        """
        Get next task from queue.

        Returns:
            Task if available, None otherwise
        """
        try:
            return await self.backend.dequeue_atomic(
                self.queue_name, self.worker_id
            )
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None

    async def _process_task(self, task: Task) -> bool:
        """
        Process a single task.

        Args:
            task: Task to process

        Returns:
            True if successful
        """
        logger.info(f"Worker {self.worker_id} processing task {task.id}")

        # Initialize lease task to None for proper cleanup
        lease_task = None

        try:
            # Start lease renewal
            lease_service = LeaseService(self.backend, self.config.lease_duration_seconds)
            lease_task = await lease_service.auto_renew(
                task.id, task.lease_id, self.config.lease_renewal_interval
            )
            self._lease_task = lease_task

            # Get handler
            self.registry.get_handler(task.handler)

            # Set timeout
            timeout = task.timeout or self.config.default_task_timeout

            # Execute with timeout
            result = await asyncio.wait_for(
                self.registry.execute_handler(task.handler, task.payload),
                timeout=timeout
            )

            # Acknowledge success
            await self.backend.ack(
                task.id,
                result,
                lease_id=task.lease_id,
                worker_id=self.worker_id,
            )
            logger.info(f"Task {task.id} completed successfully")
            return True

        except asyncio.TimeoutError:
            error = f"Task timeout after {timeout} seconds"
            logger.error(f"Task {task.id} failed: {error}")
            await self.backend.nack(
                task.id,
                error,
                lease_id=task.lease_id,
                worker_id=self.worker_id,
            )
            return False

        except Exception as e:
            error = str(e)
            logger.error(f"Task {task.id} failed: {error}")
            await self.backend.nack(
                task.id,
                error,
                lease_id=task.lease_id,
                worker_id=self.worker_id,
            )
            return False

        finally:
            # CRITICAL FIX: Always stop lease renewal, even if lease_task creation failed
            if lease_task:
                lease_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await lease_task
            self._lease_task = None

    def get_status(self) -> dict[str, Any]:
        """Get worker status."""
        uptime = (datetime.now(timezone.utc).replace(tzinfo=None) - self.started_at).total_seconds()

        return {
            'worker_id': self.worker_id,
            'state': self.state.value,
            'queue': self.queue_name,
            'current_task': self.current_task.id if self.current_task else None,
            'tasks_processed': self.tasks_processed,
            'tasks_failed': self.tasks_failed,
            'uptime_seconds': uptime,
            'last_task_at': self.last_task_at.isoformat() if self.last_task_at else None
        }


class WorkerPool:
    """
    Manages a pool of workers with dynamic scaling.
    """

    def __init__(self,
                 backend: QueueBackend,
                 registry: TaskRegistry,
                 config: SchedulerConfig):
        """
        Initialize worker pool.

        Args:
            backend: Queue backend
            registry: Task handler registry
            config: Scheduler configuration
        """
        self.backend = backend
        self.registry = registry
        self.config = config

        self.workers: dict[str, Worker] = {}
        self.queue_workers: dict[str, list[str]] = {}

        self._scaling_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start the worker pool."""
        logger.info(
            f"Starting worker pool: min={self.config.min_workers}, "
            f"max={self.config.max_workers}"
        )

        # Start minimum workers
        await self._ensure_min_workers()

        # Start monitoring and scaling tasks
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._scaling_task = asyncio.create_task(self._scaling_loop())

    async def stop(self, timeout: int = 30) -> None:
        """
        Stop all workers gracefully.

        Args:
            timeout: Maximum time to wait
        """
        logger.info("Stopping worker pool")
        self._stop_event.set()

        # Cancel background tasks
        for task in [self._monitor_task, self._scaling_task]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Stop all workers
        stop_tasks = []
        for worker in self.workers.values():
            stop_tasks.append(worker.stop(timeout))

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        self.workers.clear()
        self.queue_workers.clear()
        logger.info("Worker pool stopped")

    async def scale_to(self, target: int, queue_name: str = "default") -> int:
        """
        Scale workers for a queue to target count.

        Args:
            target: Target worker count
            queue_name: Queue to scale

        Returns:
            Actual worker count after scaling
        """
        current = len(self.queue_workers.get(queue_name, []))

        if target > current:
            # Scale up
            to_add = min(target - current, self.config.max_workers - len(self.workers))
            for _ in range(to_add):
                await self._add_worker(queue_name)

        elif target < current:
            # Scale down
            to_remove = current - max(target, self.config.min_workers)
            queue_worker_ids = self.queue_workers.get(queue_name, [])[:to_remove]

            for worker_id in queue_worker_ids:
                await self._remove_worker(worker_id)

        return len(self.queue_workers.get(queue_name, []))

    async def _add_worker(self, queue_name: str = "default") -> Worker:
        """Add a new worker."""
        worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        worker = Worker(
            worker_id=worker_id,
            backend=self.backend,
            registry=self.registry,
            config=self.config,
            queue_name=queue_name
        )

        await worker.start()
        self.workers[worker_id] = worker

        if queue_name not in self.queue_workers:
            self.queue_workers[queue_name] = []
        self.queue_workers[queue_name].append(worker_id)

        logger.info(f"Added worker {worker_id} for queue {queue_name}")
        return worker

    async def _remove_worker(self, worker_id: str) -> None:
        """Remove a worker."""
        if worker_id not in self.workers:
            return

        worker = self.workers[worker_id]
        await worker.stop()

        # Remove from queue mapping
        for queue_workers in self.queue_workers.values():
            if worker_id in queue_workers:
                queue_workers.remove(worker_id)

        del self.workers[worker_id]
        logger.info(f"Removed worker {worker_id}")

    async def _ensure_min_workers(self) -> None:
        """Ensure minimum workers are running."""
        current = len(self.workers)
        if current < self.config.min_workers:
            for _ in range(self.config.min_workers - current):
                await self._add_worker()

    async def _monitor_loop(self) -> None:
        """Monitor worker health and recycle as needed."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Check for workers that need recycling
                for worker_id, worker in list(self.workers.items()):
                    if worker.state == WorkerState.ERROR:
                        logger.warning(f"Recycling errored worker {worker_id}")
                        queue_name = worker.queue_name
                        await self._remove_worker(worker_id)
                        await self._add_worker(queue_name)

                    elif (self.config.worker_recycle_after_tasks > 0 and
                          worker.tasks_processed >= self.config.worker_recycle_after_tasks):
                        logger.info(f"Recycling worker {worker_id} after {worker.tasks_processed} tasks")
                        queue_name = worker.queue_name
                        await self._remove_worker(worker_id)
                        await self._add_worker(queue_name)

                # Ensure minimum workers
                await self._ensure_min_workers()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

    async def _scaling_loop(self) -> None:
        """Auto-scale workers based on queue size."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                # Get queue sizes
                for queue_name in list(self.queue_workers.keys()):
                    queue_size = await self.backend.get_queue_size(queue_name)
                    current_workers = len(self.queue_workers.get(queue_name, []))

                    # Simple scaling logic
                    if queue_size > current_workers * 10:  # Scale up threshold
                        if len(self.workers) < self.config.max_workers:
                            await self._add_worker(queue_name)
                            logger.info(f"Scaled up queue {queue_name} due to high load")

                    elif queue_size == 0 and current_workers > 1:  # Scale down
                        if len(self.workers) > self.config.min_workers:
                            # Remove idle worker
                            for worker_id in self.queue_workers[queue_name]:
                                worker = self.workers[worker_id]
                                if worker.state == WorkerState.IDLE:
                                    await self._remove_worker(worker_id)
                                    logger.info(f"Scaled down queue {queue_name} due to low load")
                                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling loop error: {e}")

    def get_status(self) -> dict[str, Any]:
        """Get pool status."""
        return {
            'total_workers': len(self.workers),
            'min_workers': self.config.min_workers,
            'max_workers': self.config.max_workers,
            'workers_by_state': self._get_workers_by_state(),
            'workers_by_queue': {q: len(w) for q, w in self.queue_workers.items()},
            'total_tasks_processed': sum(w.tasks_processed for w in self.workers.values()),
            'total_tasks_failed': sum(w.tasks_failed for w in self.workers.values())
        }

    def _get_workers_by_state(self) -> dict[str, int]:
        """Get worker count by state."""
        state_counts = {}
        for worker in self.workers.values():
            state = worker.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        return state_counts
