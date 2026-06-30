"""
Centralized Task Scheduler Module for tldw_server.

A production-ready, atomic task scheduler with support for both
PostgreSQL (production) and SQLite (development) backends.

Features:
- Atomic task operations with no data loss guarantees
- Dual backend support (PostgreSQL with SKIP LOCKED, SQLite with optimizations)
- Stateless architecture - all state in database
- Leader election for distributed deployments
- Dynamic worker pool management
- Task dependencies and scheduling
- Large payload handling with compression
- Automatic lease management and renewal
- Idempotency support
- Comprehensive monitoring

Quick Start:
    from tldw_Server_API.app.core.Scheduler import Scheduler, task

    # Register a task handler
    @task(max_retries=3, timeout=300)
    async def process_video(video_id: int):
        # Process video
        return {"status": "completed"}

    # Create and start scheduler
    scheduler = Scheduler()
    await scheduler.start()

    # Submit a task
    task_id = await scheduler.submit("process_video", {"video_id": 123})

    # Wait for completion
    result = await scheduler.wait_for_task(task_id)
"""

from .backends import BackendManager, create_backend
from .base import Task, TaskPriority, TaskStatus
from .base.exceptions import (
    BackendError,
    BufferClosedError,
    BufferError,
    BufferFlushError,
    DependencyError,
    LeaseError,
    PayloadError,
    SchedulerError,
    TaskNotFoundError,
    WorkerError,
)
from .base.registry import get_registry, task
from .config import SchedulerConfig, get_config, reset_config, set_config
from .scheduler import Scheduler, create_scheduler, get_existing_global_scheduler, get_global_scheduler, stop_global_scheduler

__version__ = "1.0.0"

__all__ = [
    # Main interfaces
    'Scheduler',
    'create_scheduler',
    'get_existing_global_scheduler',
    'get_global_scheduler',
    'stop_global_scheduler',
    'task',
    'get_registry',

    # Core types
    'Task',
    'TaskStatus',
    'TaskPriority',

    # Configuration
    'SchedulerConfig',
    'get_config',
    'set_config',
    'reset_config',

    # Backend
    'create_backend',
    'BackendManager',

    # Exceptions
    'SchedulerError',
    'BackendError',
    'TaskNotFoundError',
    'DependencyError',
    'LeaseError',
    'PayloadError',
    'BufferError',
    'BufferClosedError',
    'BufferFlushError',
    'WorkerError'
]
