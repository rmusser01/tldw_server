"""
Improved worker pool with better resource management and cleanup.

Key improvements:
- Proper cleanup of all resources (tasks, connections, leases)
- Graceful shutdown with timeout handling
- Better error recovery and worker recycling
- Resource leak prevention
- Improved monitoring and health checks
"""

import asyncio
import contextlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from loguru import logger

from ..base import Task, TaskStatus
from ..base.exceptions import WorkerError
from ..base.queue_backend import QueueBackend
from ..base.registry import TaskRegistry
from ..config import SchedulerConfig
from ..services.lease_service import LeaseService


class WorkerState(Enum):
    """Worker states"""
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    RECYCLING = "recycling"


@dataclass
class WorkerMetrics:
    """Metrics for worker performance"""
    tasks_processed: int = 0
    tasks_failed: int = 0
    tasks_timeout: int = 0
    total_processing_time: float = 0.0
    max_processing_time: float = 0.0
    errors: list[str] = field(default_factory=list)
    last_error_at: Optional[datetime] = None


class ResourceTracker:
    """
    Tracks all resources created by workers for proper cleanup.
    """

    def __init__(self):
        self.tasks: set[asyncio.Task] = set()
        self.leases: dict[str, asyncio.Task] = {}
        self.connections: list[Any] = []
        self._lock = asyncio.Lock()

    async def register_task(self, task: asyncio.Task) -> None:
        """Register an asyncio task for tracking"""
        async with self._lock:
            self.tasks.add(task)
            # Use weak reference to auto-remove completed tasks
            task.add_done_callback(lambda t: self.tasks.discard(t))

    async def register_lease(self, task_id: str, lease_task: asyncio.Task) -> None:
        """Register a lease renewal task"""
        async with self._lock:
            self.leases[task_id] = lease_task

    async def unregister_lease(self, task_id: str) -> None:
        """Unregister and cancel a lease renewal task"""
        async with self._lock:
            if task_id in self.leases:
                lease_task = self.leases.pop(task_id)
                if not lease_task.done():
                    lease_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await lease_task

    async def cleanup_all(self, timeout: float = 5.0) -> None:
        """Clean up all tracked resources"""
        async with self._lock:
            # Cancel all lease tasks
            for _task_id, lease_task in list(self.leases.items()):
                if not lease_task.done():
                    lease_task.cancel()

            # Wait for lease tasks to complete
            if self.leases:
                lease_tasks = list(self.leases.values())
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*lease_tasks, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for {len(lease_tasks)} lease tasks")

            self.leases.clear()

            # Cancel all other tasks
            for task in list(self.tasks):
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self.tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.tasks, return_exceptions=True),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for {len(self.tasks)} tasks")

            self.tasks.clear()

            # Close any connections
            for conn in self.connections:
                try:
                    if hasattr(conn, 'close'):
                        await conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")

            self.connections.clear()


class ImprovedWorker:
    """
    Worker with improved resource management and error handling.
    """

    def __init__(self,
                 worker_id: str,
                 backend: QueueBackend,
                 registry: TaskRegistry,
                 config: SchedulerConfig,
                 queue_name: str = "default"):
        """Initialize improved worker"""
        self.worker_id = worker_id
        self.backend = backend
        self.registry = registry
        self.config = config
        self.queue_name = queue_name

        self.state = WorkerState.IDLE
        self.current_task: Optional[Task] = None
        self.metrics = WorkerMetrics()
        self.resource_tracker = ResourceTracker()

        self.started_at = datetime.now(timezone.utc).replace(tzinfo=None)
        self.last_task_at: Optional[datetime] = None

        self._main_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._stopped_event = asyncio.Event()

        # Health check
        self._last_heartbeat = datetime.now(timezone.utc).replace(tzinfo=None)
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5

    async def start(self) -> None:
        """Start the worker"""
        if self._main_task and not self._main_task.done():
            logger.warning(f"Worker {self.worker_id} already running")
            return

        self._stop_event.clear()
        self._stopped_event.clear()
        self._main_task = asyncio.create_task(self._run())
        await self.resource_tracker.register_task(self._main_task)

        logger.info(f"Worker {self.worker_id} started for queue {self.queue_name}")

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the worker gracefully with proper cleanup.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if self.state == WorkerState.STOPPED:
            return

        logger.info(f"Stopping worker {self.worker_id}")
        self.state = WorkerState.STOPPING
        self._stop_event.set()

        try:
            # Wait for main task to complete
            if self._main_task and not self._main_task.done():
                await asyncio.wait_for(self._stopped_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Worker {self.worker_id} graceful stop timeout")

            # Force stop current task if running
            if self.current_task:
                await self._force_stop_current_task()

            # Cancel main task
            if self._main_task and not self._main_task.done():
                self._main_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._main_task

        # Clean up all resources
        await self.resource_tracker.cleanup_all()

        self.state = WorkerState.STOPPED
        self._stopped_event.set()

        logger.info(
            f"Worker {self.worker_id} stopped. "
            f"Processed: {self.metrics.tasks_processed}, "
            f"Failed: {self.metrics.tasks_failed}"
        )

    async def _run(self) -> None:
        """Main worker loop with improved error handling"""
        logger.debug(f"Worker {self.worker_id} entering main loop")

        try:
            while not self._stop_event.is_set():
                try:
                    # Update heartbeat
                    self._last_heartbeat = datetime.now(timezone.utc).replace(tzinfo=None)

                    # Check if we should recycle
                    if self._should_recycle():
                        logger.info(f"Worker {self.worker_id} requesting recycle")
                        self.state = WorkerState.RECYCLING
                        break

                    # Check error threshold
                    if self._consecutive_errors >= self._max_consecutive_errors:
                        logger.error(
                            f"Worker {self.worker_id} exceeded error threshold "
                            f"({self._consecutive_errors} errors)"
                        )
                        self.state = WorkerState.ERROR
                        break

                    # Try to get a task
                    self.state = WorkerState.IDLE
                    task = await self._get_next_task()

                    if not task:
                        # No task available, wait with exponential backoff
                        wait_time = min(2 ** self._consecutive_errors, 30)
                        await asyncio.sleep(wait_time)
                        continue

                    # Process the task
                    self.state = WorkerState.BUSY
                    self.current_task = task
                    self.last_task_at = datetime.now(timezone.utc).replace(tzinfo=None)

                    success = await self._process_task_safely(task)

                    if success:
                        self.metrics.tasks_processed += 1
                        self._consecutive_errors = 0  # Reset error counter
                    else:
                        self.metrics.tasks_failed += 1
                        self._consecutive_errors += 1

                    self.current_task = None

                except asyncio.CancelledError:
                    logger.debug(f"Worker {self.worker_id} cancelled")
                    break

                except Exception as e:
                    logger.error(f"Worker {self.worker_id} loop error: {e}")
                    self._consecutive_errors += 1
                    self.metrics.errors.append(str(e))
                    self.metrics.last_error_at = datetime.now(timezone.utc).replace(tzinfo=None)

                    # Back off on error
                    await asyncio.sleep(min(2 ** self._consecutive_errors, 30))

        finally:
            self.state = WorkerState.STOPPED
            self._stopped_event.set()
            logger.debug(f"Worker {self.worker_id} exited main loop")

    async def _get_next_task(self) -> Optional[Task]:
        """Get next task with error handling"""
        try:
            task = await asyncio.wait_for(
                self.backend.dequeue_atomic(self.queue_name, self.worker_id),
                timeout=5.0  # Don't wait forever
            )

            if task:
                # Verify task is still valid
                fresh_task = await self.backend.get_task(task.id)
                if fresh_task and fresh_task.status == TaskStatus.CANCELLED:
                    logger.info(f"Skipping cancelled task {task.id}")
                    return None
                return task

        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None

    async def _process_task_safely(self, task: Task) -> bool:
        """
        Process task with comprehensive error handling and resource cleanup.

        Args:
            task: Task to process

        Returns:
            True if successful
        """
        logger.info(f"Worker {self.worker_id} processing task {task.id}")

        lease_task = None
        start_time = asyncio.get_event_loop().time()

        try:
            # Start lease renewal
            lease_service = LeaseService(self.backend, self.config.lease_duration_seconds)
            lease_task = await lease_service.auto_renew(
                task.id, task.lease_id, self.config.lease_renewal_interval
            )
            await self.resource_tracker.register_lease(task.id, lease_task)

            # Get handler
            handler = self.registry.get_handler(task.handler)
            if not handler:
                raise WorkerError(f"Handler '{task.handler}' not found")

            # Set timeout
            timeout = task.timeout or self.config.default_task_timeout

            # Execute with timeout and cancellation support
            try:
                result = await asyncio.wait_for(
                    self._execute_handler(task, handler),
                    timeout=timeout
                )

                # Check if we're stopping
                if self._stop_event.is_set():
                    logger.info(f"Worker stopping, not acknowledging task {task.id}")
                    return False

                # Acknowledge success
                await self.backend.ack(
                    task.id,
                    result,
                    lease_id=task.lease_id,
                    worker_id=self.worker_id,
                )

                # Update metrics
                processing_time = asyncio.get_event_loop().time() - start_time
                self.metrics.total_processing_time += processing_time
                self.metrics.max_processing_time = max(
                    self.metrics.max_processing_time,
                    processing_time
                )

                logger.info(
                    f"Task {task.id} completed successfully in {processing_time:.2f}s"
                )
                return True

            except asyncio.TimeoutError:
                self.metrics.tasks_timeout += 1
                error = f"Task timeout after {timeout} seconds"
                logger.error(f"Task {task.id} failed: {error}")

                # Don't acknowledge if we're stopping
                if not self._stop_event.is_set():
                    await self.backend.nack(
                        task.id,
                        error,
                        lease_id=task.lease_id,
                        worker_id=self.worker_id,
                    )
                return False

        except asyncio.CancelledError:
            # Worker is being stopped, don't acknowledge
            logger.info(f"Task {task.id} processing cancelled")
            raise

        except Exception as e:
            error = str(e)
            logger.error(f"Task {task.id} failed: {error}", exc_info=True)

            # Don't acknowledge if we're stopping
            if not self._stop_event.is_set():
                try:
                    await self.backend.nack(
                        task.id,
                        error,
                        lease_id=task.lease_id,
                        worker_id=self.worker_id,
                    )
                except Exception as nack_error:
                    logger.error(f"Failed to nack task {task.id}: {nack_error}")

            return False

        finally:
            # Always clean up lease
            await self.resource_tracker.unregister_lease(task.id)

    async def _execute_handler(self, task: Task, handler: Any) -> Any:
        """Execute handler with proper error handling"""
        try:
            return await self.registry.execute_handler(task.handler, task.payload)
        except Exception as e:
            # Log but re-raise
            logger.error(f"Handler execution failed for task {task.id}: {e}")
            raise

    async def _force_stop_current_task(self) -> None:
        """Force stop the current task"""
        if not self.current_task:
            return

        logger.warning(f"Force stopping task {self.current_task.id}")

        try:
            await self.backend.nack(
                self.current_task.id,
                "Worker stopped before task completion",
                retry=True,
                lease_id=self.current_task.lease_id,
                worker_id=self.worker_id,
            )
        except Exception as e:
            logger.error(f"Failed to release task {self.current_task.id}: {e}")

        # Clean up lease if exists
        await self.resource_tracker.unregister_lease(self.current_task.id)

        self.current_task = None

    def _should_recycle(self) -> bool:
        """Check if worker should be recycled"""
        # Recycle after certain number of tasks
        if (self.config.worker_recycle_after_tasks > 0 and
            self.metrics.tasks_processed >= self.config.worker_recycle_after_tasks):
            return True

        # Recycle after certain uptime
        uptime = (datetime.now(timezone.utc).replace(tzinfo=None) - self.started_at).total_seconds()
        max_uptime = getattr(self.config, 'worker_max_uptime_seconds', 3600)
        if uptime > max_uptime:
            return True

        # Recycle if too many errors
        error_rate = self.metrics.tasks_failed / max(self.metrics.tasks_processed, 1)
        return bool(error_rate > 0.5 and self.metrics.tasks_processed > 10)

    def is_healthy(self) -> bool:
        """Check if worker is healthy"""
        # Check heartbeat
        heartbeat_age = (datetime.now(timezone.utc).replace(tzinfo=None) - self._last_heartbeat).total_seconds()
        if heartbeat_age > 60:
            return False

        # Check error state
        if self.state == WorkerState.ERROR:
            return False

        # Check consecutive errors
        return not self._consecutive_errors >= self._max_consecutive_errors

    def get_status(self) -> dict[str, Any]:
        """Get worker status"""
        uptime = (datetime.now(timezone.utc).replace(tzinfo=None) - self.started_at).total_seconds()

        return {
            'worker_id': self.worker_id,
            'state': self.state.value,
            'queue': self.queue_name,
            'current_task': self.current_task.id if self.current_task else None,
            'healthy': self.is_healthy(),
            'metrics': {
                'tasks_processed': self.metrics.tasks_processed,
                'tasks_failed': self.metrics.tasks_failed,
                'tasks_timeout': self.metrics.tasks_timeout,
                'avg_processing_time': (
                    self.metrics.total_processing_time / max(self.metrics.tasks_processed, 1)
                ),
                'max_processing_time': self.metrics.max_processing_time,
                'error_count': len(self.metrics.errors),
                'last_error_at': (
                    self.metrics.last_error_at.isoformat()
                    if self.metrics.last_error_at else None
                )
            },
            'uptime_seconds': uptime,
            'last_task_at': self.last_task_at.isoformat() if self.last_task_at else None,
            'consecutive_errors': self._consecutive_errors
        }


class ImprovedWorkerPool:
    """
    Worker pool with improved resource management and monitoring.
    """

    def __init__(self,
                 backend: QueueBackend,
                 registry: TaskRegistry,
                 config: SchedulerConfig):
        """Initialize improved worker pool"""
        self.backend = backend
        self.registry = registry
        self.config = config

        self.workers: dict[str, ImprovedWorker] = {}
        self.queue_workers: dict[str, set[str]] = {}
        self.resource_tracker = ResourceTracker()

        # Consolidated ops task to minimize background tasks
        self._ops_task: Optional[asyncio.Task] = None

        self._stop_event = asyncio.Event()
        self._stopped_event = asyncio.Event()

        # Worker recycling queue
        self._recycle_queue: asyncio.Queue = asyncio.Queue()
        self._recycle_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the worker pool"""
        logger.info(
            f"Starting improved worker pool: "
            f"min={self.config.min_workers}, max={self.config.max_workers}"
        )

        # Start minimum workers
        await self._ensure_min_workers()

        # Start consolidated ops task
        self._ops_task = asyncio.create_task(self._ops_loop())
        await self.resource_tracker.register_task(self._ops_task)

        logger.info("Improved worker pool started")

    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the worker pool with proper cleanup.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        logger.info("Stopping improved worker pool")
        self._stop_event.set()

        # Stop accepting new work
        start_time = asyncio.get_event_loop().time()

        # Cancel and await background tasks to ensure cleanup
        bg_tasks = [self._ops_task]
        for task in bg_tasks:
            if task and not task.done():
                task.cancel()
        for task in bg_tasks:
            if task:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._ops_task = None

        # Stop all workers gracefully
        stop_tasks = []
        for worker in self.workers.values():
            remaining_time = max(
                timeout - (asyncio.get_event_loop().time() - start_time),
                1.0
            )
            stop_tasks.append(worker.stop(timeout=remaining_time))

        if stop_tasks:
            results = await asyncio.gather(*stop_tasks, return_exceptions=True)
            for _i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error stopping worker: {result}")

        # Clean up pool resources
        await self.resource_tracker.cleanup_all()

        # Clear worker mappings
        self.workers.clear()
        self.queue_workers.clear()

        self._stopped_event.set()
        logger.info("Improved worker pool stopped")

    async def _ensure_min_workers(self) -> None:
        """Ensure minimum workers are running"""
        current = len(self.workers)
        if current < self.config.min_workers:
            for _ in range(self.config.min_workers - current):
                await self._add_worker()

    async def _add_worker(self, queue_name: str = "default") -> ImprovedWorker:
        """Add a new worker with resource tracking"""
        worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        worker = ImprovedWorker(
            worker_id=worker_id,
            backend=self.backend,
            registry=self.registry,
            config=self.config,
            queue_name=queue_name
        )

        await worker.start()

        self.workers[worker_id] = worker

        if queue_name not in self.queue_workers:
            self.queue_workers[queue_name] = set()
        self.queue_workers[queue_name].add(worker_id)

        logger.info(f"Added worker {worker_id} for queue {queue_name}")
        return worker

    async def _remove_worker(self, worker_id: str, timeout: float = 10.0) -> None:
        """Remove a worker with proper cleanup"""
        if worker_id not in self.workers:
            return

        worker = self.workers[worker_id]
        queue_name = worker.queue_name

        # Stop the worker
        try:
            await asyncio.wait_for(worker.stop(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout removing worker {worker_id}")

        # Remove from mappings
        del self.workers[worker_id]

        if queue_name in self.queue_workers:
            self.queue_workers[queue_name].discard(worker_id)
            if not self.queue_workers[queue_name]:
                del self.queue_workers[queue_name]

        logger.info(f"Removed worker {worker_id}")

    async def _recycle_worker(self, worker_id: str) -> None:
        """Recycle a worker by replacing it with a new one"""
        if worker_id not in self.workers:
            return

        worker = self.workers[worker_id]
        queue_name = worker.queue_name

        logger.info(f"Recycling worker {worker_id}")

        # Remove old worker
        await self._remove_worker(worker_id)

        # Add new worker if not stopping
        if not self._stop_event.is_set():
            await self._add_worker(queue_name)

    async def _monitor_loop(self) -> None:
        """Monitor worker health and performance"""
        # Deprecated: logic moved into _ops_loop
        return

    async def _health_check_loop(self) -> None:
        """Perform health checks on workers"""
        # Deprecated: logic moved into _ops_loop
        return

    async def _recycle_loop(self) -> None:
        """Process worker recycle requests"""
        # Deprecated: logic moved into _ops_loop
        return

    async def _scaling_loop(self) -> None:
        """Auto-scale workers based on load"""
        # Single pass scaling for backward-compatible tests
        for queue_name in list(self.queue_workers.keys()):
            queue_size = await self.backend.get_queue_size(queue_name)
            current_workers = len(self.queue_workers.get(queue_name, set()))
            if queue_size > current_workers * 10:
                if len(self.workers) < self.config.max_workers:
                    await self._add_worker(queue_name)
                    logger.info(
                        f"Scaled up queue {queue_name}: {queue_size} tasks, {current_workers + 1} workers"
                    )
            elif queue_size == 0 and current_workers > 1 and len(self.workers) > self.config.min_workers:
                for worker_id in list(self.queue_workers[queue_name]):
                    worker = self.workers.get(worker_id)
                    if worker and worker.state == WorkerState.IDLE:
                        await self._remove_worker(worker_id)
                        logger.info(
                            f"Scaled down queue {queue_name}: 0 tasks, {current_workers - 1} workers"
                        )
                        break

    async def _ops_loop(self) -> None:
        """Consolidated operations loop handling monitoring, health, scaling, recycling"""
        logger.debug("Ops loop started")
        last_monitor = 0.0
        last_health = 0.0
        last_scale = 0.0
        while not self._stop_event.is_set():
            try:
                now = asyncio.get_event_loop().time()
                # Monitor every 30s
                if now - last_monitor >= 30:
                    # Check each worker
                    for worker_id, worker in list(self.workers.items()):
                        if worker.state == WorkerState.RECYCLING:
                            await self._recycle_queue.put(worker_id)
                        elif worker.state == WorkerState.ERROR:
                            logger.warning(f"Worker {worker_id} in error state, recycling")
                            await self._recycle_queue.put(worker_id)
                    await self._ensure_min_workers()
                    last_monitor = now
                # Health check every 15s
                if now - last_health >= 15:
                    unhealthy = [wid for wid, w in list(self.workers.items()) if not w.is_healthy()]
                    for wid in unhealthy:
                        logger.warning(f"Worker {wid} unhealthy, recycling")
                        await self._recycle_queue.put(wid)
                    last_health = now
                # Scaling every 10s
                if now - last_scale >= 10:
                    for queue_name in list(self.queue_workers.keys()):
                        queue_size = await self.backend.get_queue_size(queue_name)
                        current_workers = len(self.queue_workers.get(queue_name, set()))
                        if queue_size > current_workers * 10:
                            if len(self.workers) < self.config.max_workers:
                                await self._add_worker(queue_name)
                                logger.info(
                                    f"Scaled up queue {queue_name}: {queue_size} tasks, {current_workers + 1} workers"
                                )
                        elif queue_size == 0 and current_workers > 1:
                            if len(self.workers) > self.config.min_workers:
                                # Find idle worker to remove
                                for worker_id in list(self.queue_workers[queue_name]):
                                    worker = self.workers.get(worker_id)
                                    if worker and worker.state == WorkerState.IDLE:
                                        await self._remove_worker(worker_id)
                                        logger.info(
                                            f"Scaled down queue {queue_name}: 0 tasks, {current_workers - 1} workers"
                                        )
                                        break
                    last_scale = now
                # Process recycle queue quickly without blocking
                try:
                    while True:
                        worker_id = self._recycle_queue.get_nowait()
                        await self._recycle_worker(worker_id)
                except asyncio.QueueEmpty:
                    pass
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ops loop error: {e}")
                await asyncio.sleep(0.5)
        logger.debug("Ops loop stopped")

    def get_status(self) -> dict[str, Any]:
        """Get detailed pool status"""
        total_processed = sum(w.metrics.tasks_processed for w in self.workers.values())
        total_failed = sum(w.metrics.tasks_failed for w in self.workers.values())
        total_timeout = sum(w.metrics.tasks_timeout for w in self.workers.values())

        return {
            'total_workers': len(self.workers),
            'min_workers': self.config.min_workers,
            'max_workers': self.config.max_workers,
            'workers_by_state': self._get_workers_by_state(),
            'workers_by_queue': {q: len(w) for q, w in self.queue_workers.items()},
            'healthy_workers': sum(1 for w in self.workers.values() if w.is_healthy()),
            'metrics': {
                'total_tasks_processed': total_processed,
                'total_tasks_failed': total_failed,
                'total_tasks_timeout': total_timeout,
                'success_rate': (
                    total_processed / max(total_processed + total_failed, 1)
                )
            },
            'workers': [w.get_status() for w in self.workers.values()]
        }

    def _get_workers_by_state(self) -> dict[str, int]:
        """Get worker count by state"""
        state_counts = {}
        for worker in self.workers.values():
            state = worker.state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        return state_counts
