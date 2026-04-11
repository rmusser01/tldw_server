"""Agent health monitoring with auto-disable and recovery.

Periodically checks agent availability via the registry and tracks
health status with consecutive failure counting.
"""
from __future__ import annotations

import asyncio
import copy
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger


@dataclass
class AgentHealthStatus:
    """Health status for a single agent."""
    agent_type: str
    health: str  # "healthy" | "degraded" | "unavailable" | "unknown"
    consecutive_failures: int = 0
    last_check: str | None = None
    last_healthy: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class AgentHealthMonitor:
    """Monitors agent health by periodically calling check_availability().

    Auto-disables agents after consecutive failures hit threshold,
    auto-recovers when they come back online.
    """

    def __init__(
        self,
        registry: Any = None,
        db: Any = None,
        check_interval: float = 60.0,
        failure_threshold: int = 3,
        stall_threshold_seconds: float = 300.0,
    ) -> None:
        self._registry = registry
        self._db = db
        self._check_interval = check_interval
        self._failure_threshold = failure_threshold
        self._stall_threshold_seconds = stall_threshold_seconds
        self._statuses: dict[str, AgentHealthStatus] = {}
        self._lock = threading.Lock()
        self._task: asyncio.Task | None = None
        self._running = False

    def check_all(self) -> dict[str, AgentHealthStatus]:
        """Check health of all registered agents synchronously."""
        if self._registry is None:
            return {}

        now = _utcnow_iso()
        with self._lock:
            for entry in self._registry.entries:
                avail = entry.check_availability()
                status = self._statuses.get(entry.type)
                if status is None:
                    status = AgentHealthStatus(agent_type=entry.type, health="unknown")
                    self._statuses[entry.type] = status

                status.last_check = now
                status.details = avail

                is_available = avail.get("status") == "available"
                if is_available:
                    if status.consecutive_failures > 0:
                        logger.info(
                            "Agent '{}' recovered after {} failures",
                            entry.type,
                            status.consecutive_failures,
                        )
                    status.health = "healthy"
                    status.consecutive_failures = 0
                    status.last_healthy = now
                else:
                    status.consecutive_failures += 1
                    if status.consecutive_failures >= self._failure_threshold:
                        status.health = "unavailable"
                    else:
                        status.health = "degraded"

                # Persist to DB if available
                if self._db is not None:
                    try:
                        self._db.record_health_check(
                            agent_type=entry.type,
                            health=status.health,
                            consecutive_failures=status.consecutive_failures,
                            details=json.dumps(avail),
                        )
                    except Exception as exc:
                        logger.warning("Failed to persist health check for '{}': {}",
                                       entry.type, exc)

        return self._statuses

    def get_status(self, agent_type: str) -> AgentHealthStatus | None:
        """Get health status for a specific agent (returns a copy)."""
        with self._lock:
            status = self._statuses.get(agent_type)
            return copy.deepcopy(status) if status else None

    def get_all_statuses(self) -> list[AgentHealthStatus]:
        """Get health statuses for all monitored agents (returns copies)."""
        with self._lock:
            return [copy.deepcopy(s) for s in self._statuses.values()]

    def _check_stalled_sessions(self) -> int:
        """Mark sessions as stalled when last_activity_at exceeds threshold.

        A session is considered stalled when its phase is ``running`` but
        ``last_activity_at`` is older than :attr:`_stall_threshold_seconds`.
        The previous activity value is saved in ``stalled_from_activity``
        so the platform can restore it if the session resumes.

        Returns the count of sessions marked stalled.
        """
        if self._db is None:
            return 0

        now = datetime.now(timezone.utc)
        threshold = now - timedelta(seconds=self._stall_threshold_seconds)
        threshold_str = threshold.isoformat()

        try:
            stalled_rows = self._db.list_stalled_sessions(threshold_str)
        except Exception as exc:
            logger.warning("Failed to query stalled sessions: {}", exc)
            return 0

        stalled_count = 0
        for row in stalled_rows:
            session_id = row["session_id"]
            previous_activity = row.get("activity")
            try:
                self._db.mark_session_stalled(
                    session_id,
                    stalled_from_activity=previous_activity,
                )
                logger.info(
                    "Session '{}' marked stalled (was activity='{}')",
                    session_id,
                    previous_activity,
                )
                stalled_count += 1
            except Exception as exc:
                logger.warning(
                    "Failed to mark session '{}' as stalled: {}",
                    session_id,
                    exc,
                )

        return stalled_count

    async def start(self) -> None:
        """Start the background health check loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info(
            "Agent health monitor started (interval={}s, threshold={})",
            self._check_interval,
            self._failure_threshold,
        )

    async def stop(self) -> None:
        """Stop the background health check loop."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Agent health monitor stopped")

    async def _check_loop(self) -> None:
        """Background loop that periodically checks all agents."""
        loop = asyncio.get_running_loop()
        while self._running:
            try:
                await loop.run_in_executor(None, self.check_all)
                # Stalled session detection
                stalled = await loop.run_in_executor(
                    None, self._check_stalled_sessions
                )
                if stalled:
                    logger.info("Marked {} sessions as stalled", stalled)
            except Exception as exc:
                logger.error("Health check failed: {}", exc)
            await asyncio.sleep(self._check_interval)


def _utcnow_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# Module-level singleton
_monitor: AgentHealthMonitor | None = None


def get_health_monitor() -> AgentHealthMonitor:
    """Return the module-level health monitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = AgentHealthMonitor()
    return _monitor


def configure_health_monitor(registry: Any = None, db: Any = None) -> AgentHealthMonitor:
    """Configure the singleton health monitor with registry and DB.

    Call this once at application startup after dependencies are available.
    """
    global _monitor
    _monitor = AgentHealthMonitor(registry=registry, db=db)
    return _monitor
