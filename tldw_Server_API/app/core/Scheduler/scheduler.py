"""
Main Scheduler class that orchestrates all components.
"""

import asyncio
import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from .authorization import AuthContext, get_authorizer
from .backends import BackendManager
from .base import Task, TaskPriority, TaskStatus
from .base.exceptions import SchedulerError
from .base.registry import get_registry
from .config import SchedulerConfig, get_config
from .core.leader_election import LeaderElection
from .core.worker_pool import WorkerPool
from .core.write_buffer import SafeWriteBuffer
from .services import DependencyService, LeaseService, PayloadService


class Scheduler:
    """
    Main scheduler that orchestrates task queuing and execution.

    This is the primary interface for the scheduler system, providing:
    - Task submission and management
    - Worker pool management
    - Leader election for distributed deployments
    - Monitoring and health checks
    """

    def __init__(self, config: Optional[SchedulerConfig] = None):
        """
        Initialize scheduler.

        Args:
            config: Scheduler configuration (uses global if not provided)
        """
        self.config = config or get_config()
        self.registry = get_registry()
        self.authorizer = get_authorizer()

        # Core components (initialized in start())
        self.backend_manager = BackendManager(self.config)
        self.backend = None
        self.write_buffer: Optional[SafeWriteBuffer] = None
        self.worker_pool: Optional[WorkerPool] = None
        self.leader_election: Optional[LeaderElection] = None

        # Services
        self.lease_service: Optional[LeaseService] = None
        self.dependency_service: Optional[DependencyService] = None
        self.payload_service: Optional[PayloadService] = None

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None

        # State
        self._started = False
        self._stopping = False

        logger.info(f"Scheduler initialized with backend: {self.config.database_url}")

    async def start(self, start_workers: bool = True) -> None:
        """
        Start the scheduler.

        Args:
            start_workers: Whether to start worker pool
        """
        if self._started:
            logger.warning("Scheduler already started")
            return

        logger.info("Starting scheduler...")

        try:
            # Connect to backend
            self.backend = await self.backend_manager.connect()

            # Initialize write buffer
            self.write_buffer = SafeWriteBuffer(self.backend, self.config)

            # Check for and recover from any emergency backups
            await self._recover_from_backups()

            # Initialize services
            self.lease_service = LeaseService(self.backend, self.config.lease_duration_seconds)
            self.dependency_service = DependencyService(self.backend)
            self.payload_service = PayloadService(self.backend, self.config)

            # Start lease reaper
            await self.lease_service.start_reaper(self.config.lease_reaper_interval)

            # Initialize leader election
            self.leader_election = LeaderElection(self.backend, self.config)

            # Start worker pool if requested
            if start_workers:
                self.worker_pool = WorkerPool(self.backend, self.registry, self.config)
                await self.worker_pool.start()

            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._monitor_task = asyncio.create_task(self._monitor_loop())

            # Try to become leader for cleanup tasks
            await self.leader_election.maintain_leadership(
                "scheduler:cleanup",
                callback=self._on_become_cleanup_leader
            )

            self._started = True
            logger.info("Scheduler started successfully")

        except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as e:
            logger.error(f"Failed to start scheduler: {e}")
            await self.stop()
            raise SchedulerError(f"Scheduler start failed: {e}") from e

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if self._stopping:
            return

        self._stopping = True
        logger.info("Stopping scheduler...")

        # Stop background tasks
        for task in [self._cleanup_task, self._monitor_task]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        # Stop worker pool
        if self.worker_pool:
            await self.worker_pool.stop()

        # Stop leader election
        if self.leader_election:
            await self.leader_election.stop_all()

        # Stop lease reaper
        if self.lease_service:
            await self.lease_service.stop_reaper()

        # Flush write buffer
        if self.write_buffer:
            await self.write_buffer.close()

        # Disconnect backend
        await self.backend_manager.disconnect()

        self._started = False
        self._stopping = False
        logger.info("Scheduler stopped")

    async def submit(self,
                     handler: str,
                     payload: Optional[Any] = None,
                     priority: int = TaskPriority.NORMAL.value,
                     queue_name: Optional[str] = None,
                     depends_on: Optional[list[str]] = None,
                     idempotency_key: Optional[str] = None,
                     metadata: Optional[dict[str, Any]] = None,
                     max_retries: Optional[int] = None,
                     auth_context: Optional[AuthContext] = None) -> str:
        """
        Submit a task to the scheduler.

        Args:
            handler: Task handler name (must be registered)
            payload: Task payload
            priority: Task priority
            queue_name: Target queue
            depends_on: Task dependencies
            idempotency_key: Idempotency key for deduplication
            metadata: Additional metadata
            max_retries: Optional retry override for this submission
            auth_context: Authorization context for the request

        Returns:
            Task ID
        """
        if not self._started:
            raise SchedulerError("Scheduler not started")

        task, existing_task_id = await self._prepare_task_for_submission(
            handler=handler,
            payload=payload,
            priority=priority,
            queue_name=queue_name,
            depends_on=depends_on,
            idempotency_key=idempotency_key,
            metadata=metadata,
            max_retries=max_retries,
            auth_context=auth_context
        )

        if existing_task_id:
            return existing_task_id

        # Dependency validation before buffering to avoid enqueuing invalid tasks
        if depends_on and self.dependency_service:
            # Validate that all dependencies exist in persistent store or are pending in buffer
            missing_deps = await self.dependency_service.validate_dependencies(task)
            if missing_deps and self.write_buffer and hasattr(self.write_buffer, 'buffer'):
                # Filter out dependencies that are pending in the write buffer
                pending_ids = {t.id for t in list(self.write_buffer.buffer)}
                missing_deps = [d for d in missing_deps if d not in pending_ids]
            if missing_deps:
                logger.error(f"Missing dependencies for task {task.id}: {missing_deps}")
                raise ValueError(f"Task depends on non-existent tasks: {missing_deps}")

            # Now check for circular dependencies (best-effort when possible)
            try:
                if await self.dependency_service.detect_circular_dependencies(task.id):
                    logger.error(f"Circular dependency detected for task {task.id}")
                    raise ValueError(f"Circular dependency detected for task {task.id} with dependencies {depends_on}")
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # If detection can't be completed (e.g., due to buffered tasks), skip here
                pass

        # Add to write buffer
        add_result = self.write_buffer.add(task)
        try:
            import inspect
            if inspect.isawaitable(add_result):
                task_id = await add_result
            else:
                task_id = add_result
        except TypeError:
            # Fallback: not awaitable
            task_id = add_result

        logger.debug(f"Task {task_id} submitted to queue {task.queue_name}")
        return task_id

    async def submit_batch(self, tasks: list[dict[str, Any]], auth_context: Optional[AuthContext] = None) -> list[str]:
        """
        Submit multiple tasks atomically.

        All tasks are validated first, then submitted as a batch.
        If any validation fails, no tasks are submitted.

        Args:
            tasks: List of task specifications
            auth_context: Optional authorization context applied to every task in the batch

        Returns:
            List of task IDs

        Raises:
            ValueError: If any task validation fails
            SchedulerError: If batch submission fails
        """
        if not self._started:
            raise SchedulerError("Scheduler not started")

        if not tasks:
            return []

        pending_idempotency: dict[str, str] = {}
        prepared_tasks: list[tuple[int, Task]] = []
        result_ids: list[str] = []

        for idx, task_spec in enumerate(tasks):
            handler = task_spec.get('handler')
            if not handler:
                raise ValueError(f"Task {idx}: Missing handler")

            payload = task_spec.get('payload')
            priority = task_spec.get('priority', TaskPriority.NORMAL.value)
            queue_name = task_spec.get('queue_name')
            depends_on = task_spec.get('depends_on')
            idempotency_key = task_spec.get('idempotency_key')
            metadata = task_spec.get('metadata')
            max_retries = task_spec.get('max_retries')

            try:
                task, existing_id = await self._prepare_task_for_submission(
                    handler=handler,
                    payload=payload,
                    priority=priority,
                    queue_name=queue_name,
                    depends_on=depends_on,
                    idempotency_key=idempotency_key,
                    metadata=metadata,
                    max_retries=max_retries,
                    auth_context=auth_context,
                    pending_idempotency=pending_idempotency
                )
            except PermissionError as exc:
                logger.error(f"Batch validation failed at task {idx}: {exc}")
                raise PermissionError(str(exc)) from exc
            except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.error(f"Batch validation failed at task {idx}: {exc}")
                raise ValueError(f"Batch submission failed - task {idx}: {exc}") from exc

            if existing_id:
                result_ids.append(existing_id)
                if idempotency_key:
                    pending_idempotency[idempotency_key] = existing_id
                continue

            if task is None:
                raise SchedulerError("Internal error preparing task for submission")

            prepared_tasks.append((idx, task))
            result_ids.append(task.id)
            if task.idempotency_key:
                pending_idempotency[task.idempotency_key] = task.id

        if self.dependency_service and prepared_tasks:
            pending_ids = {task.id for _, task in prepared_tasks}
            existing_ids = set(result_ids) - pending_ids

            for idx, task in prepared_tasks:
                if task.depends_on:
                    missing_deps = await self.dependency_service.validate_dependencies(task)
                    if missing_deps:
                        missing_deps = [dep for dep in missing_deps if dep not in pending_ids and dep not in existing_ids]
                    if missing_deps:
                        raise ValueError(
                            f"Batch submission failed - task {idx}: depends on non-existent tasks {missing_deps}"
                        )

                    try:
                        if await self.dependency_service.detect_circular_dependencies(task.id):
                            raise ValueError(
                                f"Batch submission failed - task {idx}: circular dependency detected for {task.id}"
                            )
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        # Best-effort detection; ignore if the graph cannot yet be built
                        pass

        tasks_to_enqueue = [task for _, task in prepared_tasks]

        if tasks_to_enqueue:
            try:
                await self.backend.bulk_enqueue(tasks_to_enqueue)
                logger.info(
                    f"Successfully submitted batch of {len(tasks_to_enqueue)} tasks "
                    f"({len(result_ids) - len(tasks_to_enqueue)} idempotent hits)"
                )
            except (AttributeError, LookupError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.error(f"Batch submission failed: {exc}")
                raise SchedulerError(f"Failed to submit batch: {exc}") from exc
        else:
            logger.info("Batch submission contained only idempotent tasks; nothing enqueued")

        return result_ids

    async def get_task(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task if found
        """
        if not self._started:
            raise SchedulerError("Scheduler not started")

        return await self.backend.get_task(task_id)

    async def cancel_task(self, task_id: str, reason: str = "User requested cancellation", auth_context: Optional[AuthContext] = None) -> bool:
        """
        Cancel a pending or running task.

        Args:
            task_id: Task ID
            reason: Cancellation reason
            auth_context: Authorization context for the request

        Returns:
            True if cancelled
        """
        if not self._started:
            raise SchedulerError("Scheduler not started")

        task = await self.backend.get_task(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found for cancellation")
            return False

        # Authorization check
        if auth_context:
            # Get task owner from metadata if available
            task_owner = task.metadata.get('user_id') if hasattr(task, 'metadata') and task.metadata else None
            can_cancel, reason_denied = self.authorizer.can_cancel_task(task_owner, auth_context)
            if not can_cancel:
                logger.warning(f"Task cancellation denied for task {task_id}: {reason_denied}")
                raise PermissionError(f"Not authorized to cancel task: {reason_denied}")

        # Can only cancel tasks that haven't completed
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.DEAD]:
            logger.info(f"Task {task_id} already in terminal state: {task.status}")
            return False

        # Update task status to cancelled
        task.status = TaskStatus.CANCELLED
        task.error = reason
        task.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)

        # Update in backend
        success = await self.backend.update_task(task)

        if success:
            # If task was running, release its lease
            if task.lease_id:
                await self.backend.delete_lease(task.lease_id)

            logger.info(f"Task {task_id} cancelled successfully: {reason}")
        else:
            logger.error(f"Failed to cancel task {task_id}")

        return success

    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Task]:
        """
        Wait for a task to complete.

        Args:
            task_id: Task ID
            timeout: Maximum wait time in seconds

        Returns:
            Completed task or None if timeout
        """
        start = datetime.now(timezone.utc).replace(tzinfo=None)

        while True:
            task = await self.backend.get_task(task_id)

            if not task:
                return None

            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.DEAD]:
                return task

            # Check timeout
            if timeout:
                elapsed = (datetime.now(timezone.utc).replace(tzinfo=None) - start).total_seconds()
                if elapsed >= timeout:
                    return None

            # Wait before checking again
            await asyncio.sleep(1)

    async def get_queue_status(self, queue_name: Optional[str] = None) -> dict[str, Any]:
        """
        Get queue status.

        Args:
            queue_name: Queue name (all queues if None)

        Returns:
            Status dictionary
        """
        if not self._started:
            raise SchedulerError("Scheduler not started")

        if queue_name:
            size = await self.backend.get_queue_size(queue_name)
            return {
                'queue': queue_name,
                'size': size
            }

        # Get all queue statuses
        # This would need backend support for listing queues
        return {
            'default': await self.backend.get_queue_size('default')
        }

    async def scale_workers(self, target: int, queue_name: str = "default") -> int:
        """
        Scale workers for a queue.

        Args:
            target: Target worker count
            queue_name: Queue name

        Returns:
            Actual worker count
        """
        if not self._started or not self.worker_pool:
            raise SchedulerError("Worker pool not available")

        return await self.worker_pool.scale_to(target, queue_name)

    def get_status(self) -> dict[str, Any]:
        """
        Get scheduler status.

        Returns:
            Status dictionary
        """
        status = {
            'started': self._started,
            'backend': self.backend.get_status() if self.backend else None,
            'write_buffer': self.write_buffer.get_status() if self.write_buffer else None,
            'worker_pool': self.worker_pool.get_status() if self.worker_pool else None,
            'leader_election': self.leader_election.get_status() if self.leader_election else None,
            'registry': {
                'handlers': len(self.registry),
                'handler_names': self.registry.list_handlers()
            }
        }

        return status

    async def _prepare_task_for_submission(
        self,
        handler: str,
        payload: Optional[Any],
        priority: int,
        queue_name: Optional[str],
        depends_on: Optional[list[str]],
        idempotency_key: Optional[str],
        metadata: Optional[dict[str, Any]],
        max_retries: Optional[int],
        auth_context: Optional[AuthContext],
        pending_idempotency: Optional[dict[str, str]] = None
    ) -> tuple[Optional[Task], Optional[str]]:
        """Validate inputs and build a Task instance matching single-submit semantics."""

        # Handler validation
        import re
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", handler or ""):
            logger.error(f"Invalid handler name format: {handler}")
            raise ValueError(f"Handler name contains invalid characters: {handler}")

        if handler not in self.registry:
            logger.error(f"Attempted to submit task with unregistered handler: {handler}")
            raise ValueError(f"Handler '{handler}' not registered. Available handlers: {self.registry.list_handlers()}")

        handler_meta = self.registry.get_metadata(handler)
        queue = queue_name or handler_meta.get('queue') or self.config.default_queue_name

        # Authorization guard
        if auth_context:
            can_submit, reason = self.authorizer.can_submit_task(handler, queue, auth_context)
            if not can_submit:
                logger.warning(f"Task submission denied for handler '{handler}': {reason}")
                raise PermissionError(f"Not authorized to submit task: {reason}")

            valid, error = self.authorizer.validate_payload_for_handler(handler, payload, auth_context)
            if not valid:
                logger.warning(f"Payload validation failed for handler '{handler}': {error}")
                raise ValueError(f"Payload validation failed: {error}")

        # Queue validation
        if queue and not queue.replace('_', '').replace('-', '').isalnum():
            logger.error(f"Invalid queue name format: {queue}")
            raise ValueError(f"Queue name contains invalid characters: {queue}")

        # Metadata is required
        metadata = self._validate_metadata(metadata)

        sanitized_payload = payload
        if payload is not None:
            import json

            if self._has_code_injection(payload):
                logger.warning("Code injection patterns detected in payload")
                raise ValueError("Payload contains potentially malicious content")

            sanitized_payload = self._sanitize_payload(payload)

            try:
                payload_json = json.dumps(sanitized_payload)
                payload_size = len(payload_json)
            except (TypeError, ValueError) as exc:
                logger.error(f"Payload serialization failed: {exc}")
                raise ValueError(f"Payload cannot be serialized to JSON: {exc}") from exc

            max_payload_size = getattr(self.config, 'max_payload_size', 1048576)
            if payload_size > max_payload_size:
                logger.error(f"Payload size {payload_size} exceeds maximum {max_payload_size}")
                raise ValueError(f"Payload size {payload_size} bytes exceeds maximum allowed size of {max_payload_size} bytes")

            if self._contains_suspicious_content(payload_json):
                logger.warning(f"Suspicious content still present after sanitization for handler {handler}")
                raise ValueError("Payload contains potentially malicious content")

        if idempotency_key:
            if len(idempotency_key) > 255:
                raise ValueError(f"Idempotency key too long: {len(idempotency_key)} > 255")

            existing_task_id = await self.backend.get_task_by_idempotency_key(idempotency_key)
            if existing_task_id:
                logger.info(
                    f"Idempotent task submission: key '{idempotency_key}' already exists, returning task {existing_task_id}"
                )
                return None, existing_task_id

            if pending_idempotency and idempotency_key in pending_idempotency:
                return None, pending_idempotency[idempotency_key]

            if self.write_buffer and hasattr(self.write_buffer, 'buffer'):
                for pending in list(self.write_buffer.buffer):
                    try:
                        if getattr(pending, 'idempotency_key', None) == idempotency_key:
                            logger.info(
                                f"Idempotent (buffer) submission: key '{idempotency_key}' maps to task {pending.id}"
                            )
                            return None, pending.id
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        continue

        if self.payload_service:
            sanitized_payload = self.payload_service.prepare_payload(sanitized_payload)

        retry_limit = handler_meta.get('max_retries', self.config.default_max_retries)
        if max_retries is not None:
            if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
                raise ValueError("max_retries must be a non-negative integer")
            retry_limit = max_retries

        task = Task(
            handler=handler,
            payload=sanitized_payload,
            priority=priority,
            queue_name=queue,
            depends_on=depends_on,
            idempotency_key=idempotency_key,
            metadata=metadata,
            max_retries=retry_limit,
            timeout=handler_meta.get('timeout', self.config.default_task_timeout)
        )

        return task, None

    def _validate_metadata(self, metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
        """
        Validate required task metadata and return a defensive copy.

        Args:
            metadata: Metadata dictionary supplied with the task

        Returns:
            Sanitized metadata dictionary

        Raises:
            ValueError: If metadata is missing or invalid
        """
        if metadata is None:
            raise ValueError("Task metadata is required")

        if not isinstance(metadata, dict):
            raise ValueError("Task metadata must be a dictionary")

        user_id = metadata.get('user_id')
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("Task metadata must include a non-empty 'user_id'")

        # Create a shallow copy to avoid caller mutation after submission
        sanitized = dict(metadata)
        sanitized['user_id'] = user_id.strip()

        return sanitized

    async def _cleanup_loop(self) -> None:
        """Background task for cleanup operations."""
        while self._started and not self._stopping:
            try:
                await asyncio.sleep(self.config.cleanup_interval_seconds)

                # Only run if we're the leader
                if self.leader_election and await self.leader_election.is_leader("scheduler:cleanup"):
                    # Reclaim expired leases
                    reclaimed = await self.backend.reclaim_expired_leases()
                    if reclaimed > 0:
                        logger.info(f"Reclaimed {reclaimed} expired leases")

                    # Clean up old payloads
                    if self.payload_service:
                        deleted = await self.payload_service.cleanup_old_payloads()
                        if deleted > 0:
                            logger.info(f"Cleaned up {deleted} old payloads")

            except asyncio.CancelledError:
                break
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError, asyncio.TimeoutError) as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _monitor_loop(self) -> None:
        """Background task for monitoring."""
        while self._started and not self._stopping:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Log status periodically
                status = self.get_status()
                logger.debug(f"Scheduler status: {status}")

            except asyncio.CancelledError:
                break
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError, asyncio.TimeoutError) as e:
                logger.error(f"Monitor loop error: {e}")

    async def _on_become_cleanup_leader(self) -> None:
        """Callback when becoming cleanup leader."""
        logger.info("This instance is now the cleanup leader")

    async def _recover_from_backups(self) -> None:
        """
        Check for and recover from any emergency backup files.
        This is called on startup to ensure no tasks are lost.
        """
        try:
            backup_dir = self.config.emergency_backup_path.parent if hasattr(self.config, 'emergency_backup_path') else Path('./backups')

            if not backup_dir.exists():
                return

            # Look for backup files
            backup_files = list(backup_dir.glob('buffer_backup_*.json'))

            if not backup_files:
                return

            logger.warning(f"Found {len(backup_files)} emergency backup files")

            total_recovered = 0
            for backup_file in backup_files:
                try:
                    recovered = await self.write_buffer.recover_from_backup(backup_file)
                    total_recovered += recovered
                    logger.info(f"Recovered {recovered} tasks from {backup_file}")
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError, asyncio.TimeoutError) as e:
                    logger.error(f"Failed to recover from backup {backup_file}: {e}")

            if total_recovered > 0:
                logger.warning(f"RECOVERY COMPLETE: Restored {total_recovered} tasks from emergency backups")

        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, asyncio.TimeoutError) as e:
            logger.error(f"Error during backup recovery: {e}")
            # Don't fail startup if recovery fails

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    def _sanitize_payload(self, payload: Any) -> Any:
        """
        Sanitize payload to remove potentially dangerous content.

        Args:
            payload: Input payload

        Returns:
            Sanitized payload
        """
        if payload is None:
            return None

        if isinstance(payload, dict):
            # Recursively sanitize dictionary
            sanitized = {}
            for key, value in payload.items():
                # Skip keys that look like code injection attempts
                if isinstance(key, str):
                    if any(dangerous in key.lower() for dangerous in ['__', 'eval', 'exec', 'import', 'os.', 'sys.']):
                        logger.warning(f"Skipping potentially dangerous key: {key}")
                        continue
                    # Limit key length
                    if len(key) > 256:
                        logger.warning(f"Skipping overly long key: {key[:50]}...")
                        continue
                sanitized[key] = self._sanitize_payload(value)
            return sanitized

        elif isinstance(payload, list):
            # Recursively sanitize list
            return [self._sanitize_payload(item) for item in payload[:1000]]  # Limit list size

        elif isinstance(payload, str):
            # Sanitize string content
            # Remove null bytes
            payload = payload.replace('\x00', '')
            # Limit string length
            if len(payload) > 65536:  # 64KB max per string
                payload = payload[:65536]
            # Remove potential script tags or SQL commands
            dangerous_patterns = [
                '<script', '</script>', 'javascript:', 'onerror=',
                'DROP TABLE', 'DELETE FROM', 'INSERT INTO', 'UPDATE SET'
            ]
            for pattern in dangerous_patterns:
                if pattern.lower() in payload.lower():
                    logger.warning(f"Removing dangerous pattern from payload: {pattern}")
                    payload = payload.replace(pattern, '')
                    payload = payload.replace(pattern.lower(), '')
                    payload = payload.replace(pattern.upper(), '')
            return payload

        elif isinstance(payload, (int, float, bool)):
            # Numeric and boolean types are safe
            return payload

        else:
            # Convert other types to string and sanitize
            return self._sanitize_payload(str(payload))

    def _contains_suspicious_content(self, content: str) -> bool:
        """
        Check if content contains suspicious patterns that might indicate an attack.

        Args:
            content: JSON string to check

        Returns:
            True if suspicious content detected
        """
        suspicious_patterns = [
            # Code execution attempts
            '__import__', 'eval(', 'exec(', 'compile(', 'globals(', 'locals(',
            # File system access
            'open(', 'file(', 'input(', 'raw_input(',
            # Network attempts
            'urllib', 'requests.', 'socket.',
            # Command injection
            '; rm ', '&& rm ', '| rm ', '`rm ',
            # Path traversal
            '../', '..\\',
            # SQL injection (already checked but double-check)
            '; DROP ', 'UNION SELECT', 'OR 1=1'
        ]

        content_lower = content.lower()
        return any(pattern.lower() in content_lower for pattern in suspicious_patterns)

    def _has_code_injection(self, obj: Any) -> bool:
        """
        Detect code-execution patterns in nested payload structures.
        This is stricter than general suspicious content detection and is used
        to hard-reject clearly malicious submissions.
        """
        patterns = [
            '__import__(', 'eval(', 'exec(', 'compile(', 'os.system(', 'subprocess.Popen(', 'subprocess.call(', 'sh -c', 'bash -c'
        ]
        try:
            if isinstance(obj, dict):
                # Reject certain critical keys outright
                for k, v in obj.items():
                    lk = str(k).lower()
                    if 'eval' in lk or 'exec' in lk:
                        return True
                    if self._has_code_injection(v):
                        return True
                return False
            elif isinstance(obj, list):
                return any(self._has_code_injection(x) for x in obj)
            elif isinstance(obj, str):
                s = obj.lower()
                return any(p in s for p in (p.lower() for p in patterns))
            else:
                return False
        except (AttributeError, RecursionError, RuntimeError, TypeError, ValueError):
            return True


# Convenience functions
async def create_scheduler(config: Optional[SchedulerConfig] = None,
                          start_workers: bool = True) -> Scheduler:
    """
    Create and start a scheduler.

    Args:
        config: Scheduler configuration
        start_workers: Whether to start workers

    Returns:
        Started scheduler instance
    """
    scheduler = Scheduler(config)
    await scheduler.start(start_workers)
    return scheduler


# Global scheduler singleton helper
_GLOBAL_SCHEDULER: Optional[Scheduler] = None
_GLOBAL_SCHEDULER_LOCK = asyncio.Lock()


async def get_global_scheduler(config: Optional[SchedulerConfig] = None,
                               start_workers: bool = True) -> Scheduler:
    """Return a process-global Scheduler instance, starting it if needed.

    Ensures only one Scheduler instance is created per process and that
    callers reuse the same worker pool and backend connections.
    """
    global _GLOBAL_SCHEDULER
    async with _GLOBAL_SCHEDULER_LOCK:
        if _GLOBAL_SCHEDULER is None:
            _GLOBAL_SCHEDULER = Scheduler(config or get_config())
            await _GLOBAL_SCHEDULER.start(start_workers)
        elif not getattr(_GLOBAL_SCHEDULER, "_started", False):
            await _GLOBAL_SCHEDULER.start(start_workers)
        return _GLOBAL_SCHEDULER


async def get_existing_global_scheduler() -> Optional[Scheduler]:
    """Return the started global Scheduler without creating or starting one."""
    async with _GLOBAL_SCHEDULER_LOCK:
        if _GLOBAL_SCHEDULER is None:
            return None
        if not getattr(_GLOBAL_SCHEDULER, "_started", False):
            return None
        return _GLOBAL_SCHEDULER


async def stop_global_scheduler() -> None:
    """Stop and clear the process-global Scheduler instance (primarily for tests)."""
    global _GLOBAL_SCHEDULER
    async with _GLOBAL_SCHEDULER_LOCK:
        if _GLOBAL_SCHEDULER is not None:
            try:
                await _GLOBAL_SCHEDULER.stop()
            finally:
                _GLOBAL_SCHEDULER = None
