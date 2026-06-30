# batch_processing.py
"""
Batch processing capabilities for the RAG service.

This module provides efficient batch query processing, parallel execution,
and resource management for handling multiple queries simultaneously.
"""

import asyncio
import contextlib
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from loguru import logger


class BatchStatus(Enum):
    """Status of a batch job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"  # Some queries succeeded, some failed


class PriorityLevel(Enum):
    """Priority levels for batch processing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BatchQuery:
    """A single query in a batch."""
    id: str
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: PriorityLevel = PriorityLevel.NORMAL
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    retry_count: int = 0


@dataclass
class BatchJob:
    """A batch processing job."""
    id: str
    queries: list[BatchQuery]
    status: BatchStatus = BatchStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0

    @property
    def total_queries(self) -> int:
        """Total number of queries in batch."""
        return len(self.queries)

    @property
    def completed_queries(self) -> int:
        """Number of completed queries."""
        return sum(
            1 for q in self.queries
            if q.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]
        )

    @property
    def success_rate(self) -> float:
        """Success rate of completed queries."""
        completed = self.completed_queries
        if completed == 0:
            return 0.0

        successful = sum(1 for q in self.queries if q.status == BatchStatus.COMPLETED)
        return successful / completed

    def update_progress(self):
        """Update job progress."""
        self.progress = (self.completed_queries / self.total_queries) * 100 if self.total_queries > 0 else 0


class BatchProcessor:
    """Processes batches of queries efficiently."""

    def __init__(
        self,
        max_concurrent: int = 10,
        max_retries: int = 3,
        batch_timeout: float = 300.0
    ):
        """
        Initialize batch processor.

        Args:
            max_concurrent: Maximum concurrent queries
            max_retries: Maximum retries per query
            batch_timeout: Timeout for batch processing
        """
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.batch_timeout = batch_timeout

        # Job tracking
        self.jobs: dict[str, BatchJob] = {}
        self.active_jobs: set = set()

        # Resource management
        self.semaphore = asyncio.Semaphore(max_concurrent)

        # Statistics
        self.stats = BatchProcessingStats()

    async def process_batch(
        self,
        queries: list[str],
        process_func: Callable,
        config: Optional[dict[str, Any]] = None,
        priority: PriorityLevel = PriorityLevel.NORMAL
    ) -> BatchJob:
        """
        Process a batch of queries.

        Args:
            queries: List of query strings
            process_func: Function to process each query
            config: Configuration for processing
            priority: Priority level for batch

        Returns:
            BatchJob with results
        """
        # Create batch job
        job_id = str(uuid.uuid4())[:8]
        batch_queries = [
            BatchQuery(
                id=f"{job_id}_{i}",
                query=query,
                priority=priority
            )
            for i, query in enumerate(queries)
        ]

        job = BatchJob(
            id=job_id,
            queries=batch_queries,
            config=config or {}
        )

        self.jobs[job_id] = job
        self.active_jobs.add(job_id)

        # Start processing
        job.status = BatchStatus.RUNNING
        job.started_at = time.time()

        logger.info(f"Starting batch job {job_id} with {len(queries)} queries")

        try:
            # Process queries with timeout
            await asyncio.wait_for(
                self._process_job(job, process_func),
                timeout=self.batch_timeout
            )

            # Update job status
            if all(q.status == BatchStatus.COMPLETED for q in job.queries):
                job.status = BatchStatus.COMPLETED
            elif any(q.status == BatchStatus.COMPLETED for q in job.queries):
                job.status = BatchStatus.PARTIAL
            else:
                job.status = BatchStatus.FAILED

        except asyncio.TimeoutError:
            logger.error(f"Batch job {job_id} timed out")
            job.status = BatchStatus.FAILED
            job.metadata["error"] = "Batch processing timeout"

        except Exception:
            logger.error("Batch job failed")
            job.status = BatchStatus.FAILED
            job.metadata["error"] = "Batch processing failed"

        finally:
            job.completed_at = time.time()
            self.active_jobs.discard(job_id)

            # Update statistics
            self.stats.record_job(job)

        logger.info(
            f"Batch job {job_id} completed with status {job.status.value}. "
            f"Success rate: {job.success_rate:.1%}"
        )

        return job

    async def _process_job(
        self,
        job: BatchJob,
        process_func: Callable
    ) -> None:
        """Process all queries in a job."""
        # Sort queries by priority
        sorted_queries = sorted(
            job.queries,
            key=lambda q: q.priority.value,
            reverse=True
        )

        # Process queries concurrently with semaphore
        tasks = [
            self._process_query_with_retry(query, process_func, job.config)
            for query in sorted_queries
        ]

        # Process with progress updates
        for completed in asyncio.as_completed(tasks):
            await completed
            job.update_progress()

            # Log progress periodically
            if job.completed_queries % max(1, job.total_queries // 10) == 0:
                logger.debug(
                    f"Batch job {job.id} progress: {job.progress:.1f}% "
                    f"({job.completed_queries}/{job.total_queries})"
                )

    async def _process_query_with_retry(
        self,
        query: BatchQuery,
        process_func: Callable,
        config: dict[str, Any]
    ) -> None:
        """Process a single query with retry logic."""
        async with self.semaphore:
            start_time = time.time()

            for attempt in range(self.max_retries):
                try:
                    # Process query
                    if asyncio.iscoroutinefunction(process_func):
                        result = await process_func(query.query, config)
                    else:
                        result = await asyncio.to_thread(process_func, query.query, config)

                    # Success
                    query.status = BatchStatus.COMPLETED
                    query.result = result
                    query.processing_time = time.time() - start_time

                    self.stats.record_query_success(query.processing_time)
                    return

                except Exception:
                    query.retry_count = attempt + 1
                    query.error = "Query processing failed"

                    if attempt < self.max_retries - 1:
                        # Exponential backoff
                        await asyncio.sleep(2 ** attempt)
                    else:
                        # Final failure
                        query.status = BatchStatus.FAILED
                        query.processing_time = time.time() - start_time

                        self.stats.record_query_failure()
                        logger.error("Query failed after retry attempts")

    async def process_stream(
        self,
        query_stream: AsyncIterator[str],
        process_func: Callable,
        batch_size: int = 10,
        config: Optional[dict[str, Any]] = None
    ) -> AsyncIterator[BatchJob]:
        """
        Process a stream of queries in batches.

        Args:
            query_stream: Async iterator of queries
            process_func: Function to process queries
            batch_size: Size of each batch
            config: Processing configuration

        Yields:
            Completed batch jobs
        """
        batch = []

        async for query in query_stream:
            batch.append(query)

            if len(batch) >= batch_size:
                # Process batch
                job = await self.process_batch(batch, process_func, config)
                yield job
                batch = []

        # Process remaining queries
        if batch:
            job = await self.process_batch(batch, process_func, config)
            yield job

    def get_job_status(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get status of a batch job."""
        job = self.jobs.get(job_id)

        if not job:
            return None

        return {
            "id": job.id,
            "status": job.status.value,
            "progress": job.progress,
            "total_queries": job.total_queries,
            "completed_queries": job.completed_queries,
            "success_rate": job.success_rate,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "metadata": job.metadata
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job."""
        job = self.jobs.get(job_id)

        if not job or job.status != BatchStatus.RUNNING:
            return False

        job.status = BatchStatus.CANCELLED
        job.completed_at = time.time()
        self.active_jobs.discard(job_id)

        # Cancel pending queries
        for query in job.queries:
            if query.status == BatchStatus.PENDING:
                query.status = BatchStatus.CANCELLED

        logger.info(f"Cancelled batch job {job_id}")
        return True

    def get_statistics(self) -> dict[str, Any]:
        """Get batch processing statistics."""
        return self.stats.get_summary()


class BatchQueue:
    """Priority queue for batch processing."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize batch queue.

        Args:
            max_size: Maximum queue size
        """
        self.max_size = max_size
        self.queues: dict[PriorityLevel, deque[dict[str, Any]]] = {
            priority: deque()
            for priority in PriorityLevel
        }
        self.total_size = 0
        self.lock = asyncio.Lock()

    async def add(
        self,
        queries: list[str],
        priority: PriorityLevel = PriorityLevel.NORMAL,
        metadata: Optional[dict[str, Any]] = None
    ) -> str:
        """
        Add queries to queue.

        Args:
            queries: List of queries
            priority: Priority level
            metadata: Additional metadata

        Returns:
            Batch ID
        """
        async with self.lock:
            if self.total_size + len(queries) > self.max_size:
                raise ValueError("Queue is full")

            batch_id = str(uuid.uuid4())[:8]

            batch = {
                "id": batch_id,
                "queries": queries,
                "priority": priority,
                "metadata": metadata or {},
                "added_at": time.time()
            }

            self.queues[priority].append(batch)
            self.total_size += len(queries)

            logger.debug(f"Added batch {batch_id} with {len(queries)} queries to queue")

            return batch_id

    async def get_next(self) -> Optional[dict[str, Any]]:
        """Get next batch from queue (highest priority first)."""
        async with self.lock:
            # Check queues in priority order
            for priority in sorted(PriorityLevel, key=lambda p: p.value, reverse=True):
                if self.queues[priority]:
                    batch = self.queues[priority].popleft()
                    self.total_size -= len(batch["queries"])
                    return batch

            return None

    async def size(self) -> int:
        """Get total number of queries in queue."""
        async with self.lock:
            return self.total_size

    async def clear(self) -> None:
        """Clear all queues."""
        async with self.lock:
            for queue in self.queues.values():
                queue.clear()
            self.total_size = 0


@dataclass
class BatchProcessingStats:
    """Statistics for batch processing."""
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_processing_time: float = 0.0
    query_times: list[float] = field(default_factory=list)

    def record_job(self, job: BatchJob):
        """Record job statistics."""
        self.total_jobs += 1

        if job.status == BatchStatus.COMPLETED:
            self.successful_jobs += 1
        elif job.status in [BatchStatus.FAILED, BatchStatus.CANCELLED]:
            self.failed_jobs += 1

        self.total_queries += job.total_queries

        for query in job.queries:
            if query.status == BatchStatus.COMPLETED:
                self.successful_queries += 1
            elif query.status == BatchStatus.FAILED:
                self.failed_queries += 1

    def record_query_success(self, processing_time: float):
        """Record successful query."""
        self.query_times.append(processing_time)
        self.total_processing_time += processing_time

        # Keep only recent times for statistics
        if len(self.query_times) > 1000:
            self.query_times = self.query_times[-1000:]

    def record_query_failure(self):
        """Record failed query."""
        pass  # Counted in record_job

    def get_summary(self) -> dict[str, Any]:
        """Get statistics summary."""
        avg_query_time = (
            sum(self.query_times) / len(self.query_times)
            if self.query_times else 0
        )

        return {
            "total_jobs": self.total_jobs,
            "successful_jobs": self.successful_jobs,
            "failed_jobs": self.failed_jobs,
            "job_success_rate": (
                self.successful_jobs / self.total_jobs
                if self.total_jobs > 0 else 0
            ),
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "query_success_rate": (
                self.successful_queries / self.total_queries
                if self.total_queries > 0 else 0
            ),
            "avg_query_time": avg_query_time,
            "total_processing_time": self.total_processing_time
        }


class BatchScheduler:
    """Schedules and manages batch processing."""

    def __init__(
        self,
        processor: BatchProcessor,
        queue: BatchQueue,
        process_func: Callable
    ):
        """
        Initialize batch scheduler.

        Args:
            processor: Batch processor
            queue: Batch queue
            process_func: Function to process queries
        """
        self.processor = processor
        self.queue = queue
        self.process_func = process_func
        self.running = False
        self.scheduler_task: Optional[asyncio.Task[None]] = None

    async def start(self):
        """Start batch scheduler."""
        if not self.running:
            self.running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info("Batch scheduler started")

    async def stop(self):
        """Stop batch scheduler."""
        if self.running:
            self.running = False
            if self.scheduler_task:
                self.scheduler_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.scheduler_task
            logger.info("Batch scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                # Get next batch from queue
                batch = await self.queue.get_next()

                if batch:
                    # Process batch
                    asyncio.create_task(
                        self._process_batch(batch)
                    )
                else:
                    # No batches, wait
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(1)

    async def _process_batch(self, batch: dict[str, Any]):
        """Process a batch from queue."""
        try:
            await self.processor.process_batch(
                queries=batch["queries"],
                process_func=self.process_func,
                config=batch.get("metadata", {}).get("config"),
                priority=batch["priority"]
            )
        except Exception as e:
            logger.error(f"Failed to process batch {batch['id']}: {e}")


# Pipeline integration functions

async def batch_process_queries(
    queries: list[str],
    pipeline_func: Callable,
    config: Optional[dict[str, Any]] = None,
    max_concurrent: int = 10
) -> list[Any]:
    """
    Process multiple queries through pipeline in batch.

    Args:
        queries: List of queries to process
        pipeline_func: Pipeline function to use
        config: Configuration for pipeline
        max_concurrent: Maximum concurrent processing

    Returns:
        List of results
    """
    processor = BatchProcessor(max_concurrent=max_concurrent)

    # Create wrapper for pipeline function that avoids deprecated context
    async def process_query(query: str, config: dict[str, Any]) -> Any:
        cfg = config or {}
        result = await pipeline_func(query, cfg)
        return result.documents if hasattr(result, 'documents') else result

    # Process batch
    job = await processor.process_batch(
        queries=queries,
        process_func=process_query,
        config=config
    )

    # Extract results
    results = []
    for query in job.queries:
        if query.status == BatchStatus.COMPLETED:
            results.append(query.result)
        else:
            results.append(None)

    logger.info(
        f"Batch processing completed: {job.success_rate:.1%} success rate"
    )

    return results


async def stream_process_queries(
    query_stream: AsyncIterator[str],
    pipeline_func: Callable,
    batch_size: int = 10,
    config: Optional[dict[str, Any]] = None
) -> AsyncIterator[list[Any]]:
    """
    Process stream of queries in batches.

    Args:
        query_stream: Async iterator of queries
        pipeline_func: Pipeline function to use
        batch_size: Size of each batch
        config: Configuration

    Yields:
        Batch results
    """
    processor = BatchProcessor()

    # Create wrapper that avoids deprecated context
    async def process_query(query: str, config: dict[str, Any]) -> Any:
        cfg = config or {}
        result = await pipeline_func(query, cfg)
        return result

    # Process stream
    async for job in processor.process_stream(
        query_stream=query_stream,
        process_func=process_query,
        batch_size=batch_size,
        config=config
    ):
        # Yield batch results
        results = [
            q.result if q.status == BatchStatus.COMPLETED else None
            for q in job.queries
        ]
        yield results
