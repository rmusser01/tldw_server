"""
Health check module for RAG service components.

Provides health status monitoring for:
- Vector store connectivity
- Database connectivity
- Embedding service availability
- Search index status
"""

import asyncio
import contextlib
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional

from loguru import logger

from .vector_stores.base import VectorStoreAdapter


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus
    message: str
    response_time: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None
    last_check: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


class RAGHealthChecker:
    """Health checker for RAG service components."""

    def __init__(
        self,
        vector_store: Optional[VectorStoreAdapter] = None,
        check_interval: int = 60  # seconds
    ):
        """
        Initialize health checker.

        Args:
            vector_store: Vector store adapter to check
            check_interval: Interval between health checks in seconds
        """
        self.vector_store = vector_store
        self.check_interval = check_interval
        self._health_cache: dict[str, ComponentHealth] = {}
        self._running = False
        self._check_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start periodic health checks."""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(self._periodic_check())
        logger.info("RAG health checker started")

    async def stop(self):
        """Stop periodic health checks."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._check_task
        logger.info("RAG health checker stopped")

    async def _periodic_check(self):
        """Perform periodic health checks."""
        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.error("Error in periodic health check")
                await asyncio.sleep(self.check_interval)

    async def check_all(self) -> dict[str, ComponentHealth]:
        """
        Check health of all components.

        Returns:
            Dictionary of component health statuses
        """
        results = {}

        # Check vector store
        if self.vector_store:
            results["vector_store"] = await self.check_vector_store()

        # Check database (using SQLite)
        results["database"] = await self.check_database()

        # Check embedding service
        results["embeddings"] = await self.check_embedding_service()

        # Check search index
        results["search_index"] = await self.check_search_index()

        # Update cache
        self._health_cache.update(results)

        return results

    async def check_vector_store(self) -> ComponentHealth:
        """Check vector store connectivity."""
        start = time.time()

        try:
            if not self.vector_store:
                return ComponentHealth(
                    name="vector_store",
                    status=HealthStatus.UNKNOWN,
                    message="Vector store not configured",
                    last_check=time.time()
                )

            # Try to perform a simple operation
            # This assumes the vector store has a health check method
            collections = await self.vector_store.list_collections()
            response_time = time.time() - start

            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.HEALTHY,
                message=f"Connected, {len(collections)} collections available",
                response_time=response_time,
                metadata={"collections": collections},
                last_check=time.time()
            )

        except Exception:
            response_time = time.time() - start
            logger.error("Vector store health check failed")

            return ComponentHealth(
                name="vector_store",
                status=HealthStatus.UNHEALTHY,
                message="Vector store connection failed",
                response_time=response_time,
                last_check=time.time()
            )

    async def check_database(self) -> ComponentHealth:
        """Check database connectivity."""
        start = time.time()

        try:
            import sqlite3

            # Quick SQLite connectivity test
            db_path = self._resolve_sqlite_db_path()
            conn = sqlite3.connect(db_path, timeout=1.0)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()

            response_time = time.time() - start

            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                response_time=response_time,
                last_check=time.time()
            )

        except Exception:
            response_time = time.time() - start
            logger.error("Database health check failed")

            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message="Database connection failed",
                response_time=response_time,
                last_check=time.time()
            )

    async def check_embedding_service(self) -> ComponentHealth:
        """Check embedding service availability."""
        start = time.time()

        try:
            # Check if embeddings jobs worker module is available
            from ...Embeddings.services import jobs_worker  # noqa: F401

            # This is a simple availability check; production should probe provider health.
            response_time = time.time() - start

            return ComponentHealth(
                name="embeddings",
                status=HealthStatus.HEALTHY,
                message="Embeddings jobs worker available",
                response_time=response_time,
                last_check=time.time()
            )

        except ImportError:
            response_time = time.time() - start

            return ComponentHealth(
                name="embeddings",
                status=HealthStatus.DEGRADED,
                message="Embedding service not configured",
                response_time=response_time,
                last_check=time.time()
            )

        except Exception:
            response_time = time.time() - start
            logger.error("Embedding service health check failed")

            return ComponentHealth(
                name="embeddings",
                status=HealthStatus.UNHEALTHY,
                message="Embedding service check failed",
                response_time=response_time,
                last_check=time.time()
            )

    async def check_search_index(self) -> ComponentHealth:
        """Check search index status."""
        start = time.time()

        try:
            import sqlite3

            # Check FTS5 index
            db_path = self._resolve_sqlite_db_path()
            conn = sqlite3.connect(db_path, timeout=1.0)
            cursor = conn.cursor()

            # Check if FTS table exists and is accessible
            cursor.execute("""
                SELECT COUNT(*) FROM sqlite_master
                WHERE type='table' AND name LIKE '%fts%'
            """)
            fts_tables = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            response_time = time.time() - start

            if fts_tables > 0:
                return ComponentHealth(
                    name="search_index",
                    status=HealthStatus.HEALTHY,
                    message=f"{fts_tables} FTS indexes available",
                    response_time=response_time,
                    metadata={"fts_table_count": fts_tables},
                    last_check=time.time()
                )
            else:
                return ComponentHealth(
                    name="search_index",
                    status=HealthStatus.DEGRADED,
                    message="No FTS indexes found",
                    response_time=response_time,
                    last_check=time.time()
                )

        except Exception:
            response_time = time.time() - start
            logger.error("Search index health check failed")

            return ComponentHealth(
                name="search_index",
                status=HealthStatus.UNHEALTHY,
                message="Search index check failed",
                response_time=response_time,
                last_check=time.time()
            )

    def _resolve_sqlite_db_path(self) -> str:
        try:
            from ....core.config import load_and_log_configs

            cfg = load_and_log_configs() or {}
            db_url = cfg.get("DATABASE_URL")
            if isinstance(db_url, str) and db_url.startswith("sqlite:///"):
                return db_url.replace("sqlite:///", "")
        except Exception:
            logger.debug("Health check db path fallback")

        return ":memory:"

    def get_overall_health(self) -> HealthStatus:
        """
        Get overall system health based on component statuses.

        Returns:
            Overall health status
        """
        if not self._health_cache:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in self._health_cache.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    def get_health_summary(self) -> dict[str, Any]:
        """
        Get health summary for all components.

        Returns:
            Health summary dictionary
        """
        return {
            "overall_status": self.get_overall_health().value,
            "components": {
                name: health.to_dict()
                for name, health in self._health_cache.items()
            },
            "last_check": max(
                (h.last_check for h in self._health_cache.values()),
                default=0
            )
        }
