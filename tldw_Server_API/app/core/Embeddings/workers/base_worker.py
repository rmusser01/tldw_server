# base_worker.py
# Base worker class for all embedding pipeline workers

import asyncio
import json
import signal
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar

import redis.asyncio as redis
from loguru import logger
from pydantic import BaseModel

from ..queue_schemas import EmbeddingJobMessage, JobInfo, JobStatus, WorkerMetrics


T = TypeVar('T', bound=EmbeddingJobMessage)


class WorkerConfig(BaseModel):
    """Base configuration for workers"""
    worker_id: str
    worker_type: str
    redis_url: str = "redis://localhost:6379"
    queue_name: str
    consumer_group: str
    batch_size: int = 1
    poll_interval_ms: int = 100
    max_retries: int = 3
    heartbeat_interval: int = 30
    shutdown_timeout: int = 30
    metrics_interval: int = 60


class BaseWorker(ABC):
    """Abstract base class for all pipeline workers"""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.running = False
        self.jobs_processed = 0
        self.jobs_failed = 0
        self.processing_times: List[float] = []
        self._tasks: List[asyncio.Task] = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Initialized {self.config.worker_type} worker: {self.config.worker_id}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    @asynccontextmanager
    async def _redis_connection(self):
        """Context manager for Redis connection"""
        try:
            self.redis_client = await redis.from_url(
                self.config.redis_url,
                decode_responses=True
            )
            yield self.redis_client
        finally:
            if self.redis_client:
                await self.redis_client.close()
    
    async def start(self):
        """Start the worker"""
        async with self._redis_connection():
            self.running = True
            
            # Start background tasks
            self._tasks = [
                asyncio.create_task(self._process_messages()),
                asyncio.create_task(self._heartbeat_loop()),
                asyncio.create_task(self._metrics_loop()),
            ]
            
            logger.info(f"Worker {self.config.worker_id} started")
            
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Worker tasks cancelled")
            finally:
                await self._cleanup()
    
    async def _process_messages(self):
        """Main message processing loop"""
        while self.running:
            try:
                # Read messages from stream
                messages = await self.redis_client.xreadgroup(
                    self.config.consumer_group,
                    self.config.worker_id,
                    {self.config.queue_name: '>'},
                    count=self.config.batch_size,
                    block=self.config.poll_interval_ms
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        await self._process_batch(stream_messages)
                        
            except Exception as e:
                logger.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_batch(self, messages: List[tuple]):
        """Process a batch of messages"""
        for message_id, data in messages:
            start_time = time.time()
            
            try:
                # Parse message
                message = self._parse_message(data)
                
                # Update job status
                await self._update_job_status(message.job_id, JobStatus.CHUNKING)
                
                # Process the message
                result = await self.process_message(message)
                
                # Send to next stage
                if result:
                    await self._send_to_next_stage(result)
                
                # Acknowledge message
                await self.redis_client.xack(
                    self.config.queue_name,
                    self.config.consumer_group,
                    message_id
                )
                
                # Update metrics
                self.jobs_processed += 1
                self.processing_times.append(time.time() - start_time)
                
            except Exception as e:
                logger.error(f"Error processing message {message_id}: {e}")
                await self._handle_failed_message(message_id, data, e)
                self.jobs_failed += 1
    
    @abstractmethod
    async def process_message(self, message: T) -> Optional[BaseModel]:
        """Process a single message. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _parse_message(self, data: Dict[str, Any]) -> T:
        """Parse raw message data into typed message. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _send_to_next_stage(self, result: BaseModel):
        """Send processed result to next stage. Must be implemented by subclasses."""
        pass
    
    async def _handle_failed_message(self, message_id: str, data: Dict[str, Any], error: Exception):
        """Handle failed message processing"""
        try:
            message = self._parse_message(data)
            
            if message.retry_count < message.max_retries:
                # Increment retry count and requeue
                message.retry_count += 1
                message.updated_at = datetime.utcnow()
                
                # Add back to queue with exponential backoff
                backoff_ms = (2 ** message.retry_count) * 1000
                await asyncio.sleep(backoff_ms / 1000)
                
                await self.redis_client.xadd(
                    self.config.queue_name,
                    message.dict()
                )
                
                logger.warning(f"Requeued message {message.job_id} (retry {message.retry_count})")
            else:
                # Max retries exceeded, mark as failed
                await self._update_job_status(
                    message.job_id,
                    JobStatus.FAILED,
                    error_message=str(error)
                )
                logger.error(f"Message {message.job_id} failed after {message.max_retries} retries")
                
            # Always acknowledge to prevent reprocessing
            await self.redis_client.xack(
                self.config.queue_name,
                self.config.consumer_group,
                message_id
            )
            
        except Exception as e:
            logger.error(f"Error handling failed message: {e}")
    
    async def _update_job_status(self, job_id: str, status: JobStatus, error_message: Optional[str] = None):
        """Update job status in Redis"""
        job_key = f"job:{job_id}"
        
        updates = {
            "status": status.value,
            "updated_at": datetime.utcnow().isoformat(),
            "current_stage": self.config.worker_type
        }
        
        if error_message:
            updates["error_message"] = error_message
        
        if status == JobStatus.COMPLETED:
            updates["completed_at"] = datetime.utcnow().isoformat()
        
        await self.redis_client.hset(job_key, mapping=updates)
        
        # Set TTL for completed/failed jobs
        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            await self.redis_client.expire(job_key, 86400)  # 24 hours
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
    
    async def _send_heartbeat(self):
        """Send worker heartbeat to Redis"""
        heartbeat_key = f"worker:heartbeat:{self.config.worker_id}"
        await self.redis_client.setex(
            heartbeat_key,
            self.config.heartbeat_interval * 2,  # TTL = 2x heartbeat interval
            datetime.utcnow().isoformat()
        )
    
    async def _metrics_loop(self):
        """Report metrics periodically"""
        while self.running:
            try:
                await self._report_metrics()
                await asyncio.sleep(self.config.metrics_interval)
            except Exception as e:
                logger.error(f"Error reporting metrics: {e}")
    
    async def _report_metrics(self):
        """Report worker metrics"""
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0
        )
        
        metrics = WorkerMetrics(
            worker_id=self.config.worker_id,
            worker_type=self.config.worker_type,
            jobs_processed=self.jobs_processed,
            jobs_failed=self.jobs_failed,
            average_processing_time_ms=avg_processing_time * 1000,
            current_load=await self._calculate_load(),
            last_heartbeat=datetime.utcnow()
        )
        
        metrics_key = f"worker:metrics:{self.config.worker_id}"
        await self.redis_client.setex(
            metrics_key,
            self.config.metrics_interval * 2,
            metrics.json()
        )
        
        # Reset processing times to prevent unbounded growth
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-100:]
    
    async def _calculate_load(self) -> float:
        """Calculate current worker load (0-1)"""
        # This is a simple implementation - can be overridden by subclasses
        queue_length = await self.redis_client.xlen(self.config.queue_name)
        return min(1.0, queue_length / 100)  # Normalize to 0-1
    
    async def _cleanup(self):
        """Cleanup resources before shutdown"""
        logger.info(f"Cleaning up worker {self.config.worker_id}")
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info(f"Worker {self.config.worker_id} shutdown complete")