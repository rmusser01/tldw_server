# event_broadcaster.py
# Event broadcasting system for Prompt Studio real-time updates

import asyncio
import contextlib
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    DatabaseError,
    PromptStudioDatabase,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.monitoring import (
    prompt_studio_metrics,
)

########################################################################################################################
# Event Types

class EventType(str, Enum):
    """Types of events that can be broadcast."""

    # Job events
    JOB_CREATED = "job_created"
    JOB_STARTED = "job_started"
    JOB_PROGRESS = "job_progress"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"
    JOB_RETRYING = "job_retrying"

    # Evaluation events
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_PROGRESS = "evaluation_progress"
    EVALUATION_COMPLETED = "evaluation_completed"

    # Optimization events
    OPTIMIZATION_STARTED = "optimization_started"
    OPTIMIZATION_ITERATION = "optimization_iteration"
    OPTIMIZATION_COMPLETED = "optimization_completed"

    # Test case events
    TEST_CASE_CREATED = "test_case_created"
    TEST_CASE_UPDATED = "test_case_updated"
    TEST_CASE_DELETED = "test_case_deleted"
    TEST_CASES_GENERATED = "test_cases_generated"

    # Project events
    PROJECT_UPDATED = "project_updated"
    PROJECT_DELETED = "project_deleted"

    # System events
    SYSTEM_MESSAGE = "system_message"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"

########################################################################################################################
# Event Broadcaster


def prompt_studio_connection_scope(
    user_id: str,
    project_id: Optional[int] = None,
    *,
    client_id: Optional[str] = None,
) -> str:
    """Build a collision-free connection scope for one Prompt Studio user."""
    owner = str(user_id).strip()
    if not owner:
        raise ValueError("Prompt Studio connection scope requires a user ID")
    if client_id is not None:
        channel = ["client", str(client_id)]
    elif project_id is not None:
        channel = ["project", int(project_id)]
    else:
        channel = ["global"]
    return json.dumps(["prompt-studio", owner, *channel], separators=(",", ":"))

class EventBroadcaster:
    """Broadcasts events to connected WebSocket clients."""

    def __init__(self, connection_manager, db: PromptStudioDatabase):
        """
        Initialize EventBroadcaster.

        Args:
            connection_manager: WebSocket connection manager
            db: Database instance
        """
        self.connection_manager = connection_manager
        self.db = db
        self.client_id = db.client_id
        self.user_id = str(getattr(db, "user_id", "") or "").strip() or None

        # Track subscriptions
        self.subscriptions: dict[str, set[str]] = {}

        # Event queue for async processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
        self._process_task = None

    ####################################################################################################################
    # Broadcasting Methods

    async def broadcast_event(self, event_type: EventType, data: dict[str, Any],
                             client_ids: Optional[list[str]] = None,
                             project_id: Optional[int] = None):
        """
        Broadcast an event to clients.

        Args:
            event_type: Type of event
            data: Event data
            client_ids: Specific clients to broadcast to (None = all)
            project_id: Associated project ID
        """
        if project_id is None:
            project_id = self._resolve_project_id(data)

        event = {
            "type": event_type.value,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "project_id": project_id
        }

        # Log event to database
        await self._log_event(event_type, data, project_id)

        # Convert to JSON
        message = json.dumps(event)

        if self.user_id is None:
            logger.warning("Prompt Studio event dropped because its database has no user scope")
            return

        # Every target includes the authenticated owner. Project events also
        # reach the owner's global stream, but never another user's stream.
        if client_ids:
            for client_id in client_ids:
                scope = prompt_studio_connection_scope(
                    self.user_id,
                    client_id=client_id,
                )
                await self.connection_manager.broadcast_to_client(scope, message)
        else:
            scopes = [prompt_studio_connection_scope(self.user_id)]
            if project_id is not None:
                scopes.append(prompt_studio_connection_scope(self.user_id, project_id))
            for scope in scopes:
                await self.connection_manager.broadcast_to_client(scope, message)

        logger.debug(f"Broadcast {event_type.value} to {client_ids or 'all'}")
        # Optional metrics per WS message
        with contextlib.suppress(Exception):
            prompt_studio_metrics.record_websocket_message(event_type.value)

    # Backward-compat: some tests patch EventBroadcaster.broadcast; provide a thin wrapper
    async def broadcast(self, *args, **kwargs):  # noqa: D401 - compatibility wrapper
        """Compatibility wrapper that delegates to broadcast_event when possible."""
        try:
            # Support calling with explicit kwargs: event_type, data, client_ids, project_id
            if "event_type" in kwargs and "data" in kwargs:
                et = kwargs.get("event_type")
                if isinstance(et, str):
                    try:
                        et = EventType(et)
                    except Exception:
                        et = EventType.SYSTEM_MESSAGE
                await self.broadcast_event(
                    event_type=et if isinstance(et, EventType) else EventType.SYSTEM_MESSAGE,
                    data=kwargs.get("data", {}),
                    client_ids=kwargs.get("client_ids"),
                    project_id=kwargs.get("project_id")
                )
                return
            # Support a single positional dict message
            if args and isinstance(args[0], dict):
                message = args[0]
                et = message.get("type", "system_message")
                try:
                    et_enum = EventType(et)
                except Exception:
                    et_enum = EventType.SYSTEM_MESSAGE
                await self.broadcast_event(et_enum, message.get("data", {}), None, message.get("project_id"))
        except Exception as e:
            logger.debug(f"Compatibility broadcast ignored error: {e}")

    async def broadcast_job_event(self, job_id: int, event_type: EventType,
                                 additional_data: Optional[dict] = None):
        """
        Broadcast a job-related event.

        Args:
            job_id: Job ID
            event_type: Event type
            additional_data: Additional event data
        """
        # Get job details from core Jobs via adapter
        from .jobs_adapter import PromptStudioJobsAdapter
        adapter = PromptStudioJobsAdapter()
        job = adapter.get_job(
            str(job_id),
            db=self.db,
            user_id=None,
        )

        if not job:
            logger.warning(f"Job {job_id} not found for event broadcast")
            return

        # Prepare event data
        data = {
            "job_id": job_id,
            "job_type": job["job_type"],
            "entity_id": job["entity_id"],
            "status": job["status"],
            "progress": job.get("progress", 0)
        }

        if additional_data:
            data.update(additional_data)

        # Get project ID from entity
        project_id = await self._get_project_for_job(job)

        # Broadcast event
        await self.broadcast_event(
            event_type=event_type,
            data=data,
            project_id=project_id
        )

    async def broadcast_progress(self, job_id: int, progress: float,
                                message: Optional[str] = None):
        """
        Broadcast job progress update.

        Args:
            job_id: Job ID
            progress: Progress percentage (0-100)
            message: Optional progress message
        """
        data = {
            "progress": progress,
            "message": message
        }

        await self.broadcast_job_event(
            job_id=job_id,
            event_type=EventType.JOB_PROGRESS,
            additional_data=data
        )

    async def broadcast_evaluation_progress(self, evaluation_id: int,
                                          tests_completed: int, total_tests: int,
                                          current_test: Optional[str] = None):
        """
        Broadcast evaluation progress.

        Args:
            evaluation_id: Evaluation ID
            tests_completed: Number of tests completed
            total_tests: Total number of tests
            current_test: Currently running test name
        """
        progress = (tests_completed / total_tests * 100) if total_tests > 0 else 0

        data = {
            "evaluation_id": evaluation_id,
            "tests_completed": tests_completed,
            "total_tests": total_tests,
            "progress": progress,
            "current_test": current_test
        }

        await self.broadcast_event(
            event_type=EventType.EVALUATION_PROGRESS,
            data=data
        )

    async def broadcast_optimization_iteration(self, optimization_id: int,
                                              iteration: int, max_iterations: int,
                                              current_metric: float,
                                              best_metric: float,
                                              extra_data: Optional[dict[str, Any]] = None):
        """
        Broadcast optimization iteration update.

        Args:
            optimization_id: Optimization ID
            iteration: Current iteration number
            max_iterations: Maximum iterations
            current_metric: Current iteration metric
            best_metric: Best metric so far
            extra_data: Additional payload fields to include
        """
        data = {
            "optimization_id": optimization_id,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "current_metric": current_metric,
            "best_metric": best_metric,
            "progress": (iteration / max_iterations * 100) if max_iterations > 0 else 0
        }
        if extra_data:
            data.update(extra_data)

        await self.broadcast_event(
            event_type=EventType.OPTIMIZATION_ITERATION,
            data=data
        )

    ####################################################################################################################
    # Subscription Management

    def subscribe(self, client_id: str, entity_type: str, entity_id: int):
        """
        Subscribe a client to entity updates.

        Args:
            client_id: Client ID
            entity_type: Type of entity (job, evaluation, etc.)
            entity_id: Entity ID
        """
        key = f"{entity_type}:{entity_id}"

        if key not in self.subscriptions:
            self.subscriptions[key] = set()

        self.subscriptions[key].add(client_id)
        logger.debug(f"Client {client_id} subscribed to {key}")

    def unsubscribe(self, client_id: str, entity_type: str, entity_id: int):
        """
        Unsubscribe a client from entity updates.

        Args:
            client_id: Client ID
            entity_type: Type of entity
            entity_id: Entity ID
        """
        key = f"{entity_type}:{entity_id}"

        if key in self.subscriptions:
            self.subscriptions[key].discard(client_id)

            # Clean up empty subscriptions
            if not self.subscriptions[key]:
                del self.subscriptions[key]

        logger.debug(f"Client {client_id} unsubscribed from {key}")

    def get_subscribers(self, entity_type: str, entity_id: int) -> set[str]:
        """
        Get subscribers for an entity.

        Args:
            entity_type: Type of entity
            entity_id: Entity ID

        Returns:
            Set of client IDs
        """
        key = f"{entity_type}:{entity_id}"
        return self.subscriptions.get(key, set())

    ####################################################################################################################
    # Event Processing

    async def start_processing(self):
        """Start processing events from the queue."""
        if self._processing:
            logger.warning("Event processing already running")
            return

        self._processing = True
        self._process_task = asyncio.create_task(self._process_events())
        logger.info("Started event processor")

    async def stop_processing(self):
        """Stop processing events."""
        self._processing = False

        if self._process_task:
            self._process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_task

        logger.info("Stopped event processor")

    async def _process_events(self):
        """Process events from the queue."""
        while self._processing:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )

                # Process event
                await self._handle_queued_event(event)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    async def _handle_queued_event(self, event: dict[str, Any]):
        """
        Handle a queued event.

        Args:
            event: Event data
        """
        event_type = event.get("type")
        data = event.get("data")
        client_ids = event.get("client_ids")
        project_id = event.get("project_id")

        await self.broadcast_event(
            event_type=EventType(event_type),
            data=data,
            client_ids=client_ids,
            project_id=project_id
        )

    async def queue_event(self, event_type: EventType, data: dict[str, Any],
                         client_ids: Optional[list[str]] = None,
                         project_id: Optional[int] = None):
        """
        Queue an event for async processing.

        Args:
            event_type: Type of event
            data: Event data
            client_ids: Target client IDs
            project_id: Associated project ID
        """
        event = {
            "type": event_type.value,
            "data": data,
            "client_ids": client_ids,
            "project_id": project_id
        }

        await self.event_queue.put(event)

    ####################################################################################################################
    # Helper Methods

    def _resolve_project_id(self, data: dict[str, Any]) -> Optional[int]:
        """Resolve the owning project for event payloads that only carry an entity ID."""
        raw_project_id = data.get("project_id")
        if raw_project_id is not None:
            try:
                return int(raw_project_id)
            except (TypeError, ValueError):
                return None

        lookups = (
            ("evaluation_id", "get_evaluation"),
            ("optimization_id", "get_optimization"),
        )
        for id_key, getter_name in lookups:
            entity_id = data.get(id_key)
            getter = getattr(self.db, getter_name, None)
            if entity_id is None or not callable(getter):
                continue
            try:
                entity = getter(int(entity_id))
                if entity and entity.get("project_id") is not None:
                    return int(entity["project_id"])
            except (DatabaseError, KeyError, TypeError, ValueError):
                logger.warning(
                    "Prompt Studio event project lookup failed for {}",
                    id_key,
                )
        return None

    async def _get_project_for_job(self, job: dict[str, Any]) -> Optional[int]:
        """
        Get project ID for a job.

        Args:
            job: Job data

        Returns:
            Project ID or None
        """
        job_type = job["job_type"]
        entity_id = job["entity_id"]

        conn = self.db.get_connection()
        conn.cursor()

        try:
            if job_type == "evaluation":
                evaluation = self.db.get_evaluation(entity_id)
                return evaluation.get("project_id") if evaluation else None
            if job_type == "optimization":
                optimization = self.db.get_optimization(entity_id)
                return optimization.get("project_id") if optimization else None
            return None

        except DatabaseError as exc:
            logger.error(f"Database error resolving project for job {entity_id}: {exc}")
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Error getting project for job: {exc}")
            return None

    async def _log_event(self, event_type: EventType, data: dict[str, Any],
                        project_id: Optional[int] = None):
        """
        Log event to database for history/audit.

        Args:
            event_type: Event type
            data: Event data
            project_id: Associated project ID
        """
        try:
            # Route through DB helper to keep schema consistent across backends
            payload = dict(data or {})
            if project_id is not None:
                payload.setdefault("project_id", project_id)
            event_uuid = str(uuid.uuid4())
            # _log_sync_event(entity, entity_uuid, operation, payload)
            # The helper handles missing sync_log gracefully.
            self.db._log_sync_event(
                entity="prompt_studio_event",
                entity_uuid=event_uuid,
                operation=event_type.value,
                payload=payload,
            )
        except Exception as e:
            logger.debug(f"Failed to log event to sync_log (non-fatal): {e}")

########################################################################################################################
# Event Hooks for Integration

class EventHooks:
    """Hooks for integrating event broadcasting into operations."""

    def __init__(self, broadcaster: EventBroadcaster):
        """
        Initialize EventHooks.

        Args:
            broadcaster: EventBroadcaster instance
        """
        self.broadcaster = broadcaster

    async def on_job_created(self, job_id: int):
        """Hook for job creation."""
        await self.broadcaster.broadcast_job_event(
            job_id=job_id,
            event_type=EventType.JOB_CREATED
        )

    async def on_job_started(self, job_id: int):
        """Hook for job start."""
        await self.broadcaster.broadcast_job_event(
            job_id=job_id,
            event_type=EventType.JOB_STARTED
        )

    async def on_job_completed(self, job_id: int, result: dict[str, Any]):
        """Hook for job completion."""
        await self.broadcaster.broadcast_job_event(
            job_id=job_id,
            event_type=EventType.JOB_COMPLETED,
            additional_data={"result": result}
        )

    async def on_job_failed(self, job_id: int, error: str):
        """Hook for job failure."""
        await self.broadcaster.broadcast_job_event(
            job_id=job_id,
            event_type=EventType.JOB_FAILED,
            additional_data={"error": error}
        )

    async def on_test_cases_generated(self, project_id: int, count: int):
        """Hook for test case generation."""
        await self.broadcaster.broadcast_event(
            event_type=EventType.TEST_CASES_GENERATED,
            data={"count": count},
            project_id=project_id
        )
