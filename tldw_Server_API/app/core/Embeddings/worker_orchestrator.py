# worker_orchestrator.py
# Orchestrates and manages worker pools for the embedding pipeline

import asyncio
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
from prometheus_client import start_http_server, Gauge, Counter

from .job_manager import EmbeddingJobManager, JobManagerConfig
from .worker_config import (
    OrchestrationConfig,
    ChunkingWorkerPoolConfig,
    EmbeddingWorkerPoolConfig,
    StorageWorkerPoolConfig,
)
from .workers import (
    ChunkingWorker,
    EmbeddingWorker,
    EmbeddingWorkerConfig,
    StorageWorker,
    WorkerConfig,
)


# Prometheus metrics
WORKER_COUNT = Gauge("embedding_worker_count", "Number of active workers", ["worker_type"])
QUEUE_DEPTH = Gauge("embedding_queue_depth", "Current queue depth", ["queue_name"])
JOBS_TOTAL = Counter("embedding_jobs_total", "Total jobs processed", ["status"])


class WorkerPool:
    """Manages a pool of workers of the same type"""
    
    def __init__(self, pool_config: WorkerPoolConfig):
        self.config = pool_config
        self.workers: List[asyncio.Task] = []
        self.running = False
    
    async def start(self, redis_url: str):
        """Start all workers in the pool"""
        self.running = True
        
        for i in range(self.config.num_workers):
            worker_id = f"{self.config.worker_type}-{i}"
            worker = await self._create_worker(worker_id, redis_url)
            
            # Start worker in background task
            task = asyncio.create_task(worker.start())
            self.workers.append(task)
            
            logger.info(f"Started worker {worker_id}")
        
        # Update metrics
        WORKER_COUNT.labels(worker_type=self.config.worker_type).set(len(self.workers))
    
    async def stop(self):
        """Stop all workers in the pool"""
        self.running = False
        
        # Cancel all worker tasks
        for task in self.workers:
            task.cancel()
        
        # Wait for all to complete
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        WORKER_COUNT.labels(worker_type=self.config.worker_type).set(0)
        
        logger.info(f"Stopped all {self.config.worker_type} workers")
    
    async def scale(self, new_count: int, redis_url: str):
        """Scale the worker pool to new count"""
        current_count = len(self.workers)
        
        if new_count > current_count:
            # Scale up
            for i in range(current_count, new_count):
                worker_id = f"{self.config.worker_type}-{i}"
                worker = await self._create_worker(worker_id, redis_url)
                
                task = asyncio.create_task(worker.start())
                self.workers.append(task)
                
                logger.info(f"Scaled up: started worker {worker_id}")
        
        elif new_count < current_count:
            # Scale down
            workers_to_stop = self.workers[new_count:]
            self.workers = self.workers[:new_count]
            
            for task in workers_to_stop:
                task.cancel()
            
            await asyncio.gather(*workers_to_stop, return_exceptions=True)
            
            logger.info(f"Scaled down: stopped {len(workers_to_stop)} workers")
        
        WORKER_COUNT.labels(worker_type=self.config.worker_type).set(len(self.workers))
    
    async def _create_worker(self, worker_id: str, redis_url: str):
        """Create a worker instance based on type"""
        base_config = WorkerConfig(
            worker_id=worker_id,
            worker_type=self.config.worker_type,
            redis_url=redis_url,
            queue_name=self.config.queue_name,
            consumer_group=self.config.consumer_group,
            batch_size=self.config.batch_size,
            poll_interval_ms=self.config.poll_interval_ms,
            max_retries=self.config.max_retries,
            heartbeat_interval=self.config.heartbeat_interval,
            shutdown_timeout=self.config.shutdown_timeout,
            metrics_interval=self.config.metrics_interval,
        )
        
        if isinstance(self.config, ChunkingWorkerPoolConfig):
            return ChunkingWorker(base_config)
        
        elif isinstance(self.config, EmbeddingWorkerPoolConfig):
            # Assign GPU in round-robin fashion
            gpu_id = None
            if self.config.gpu_allocation:
                worker_index = int(worker_id.split('-')[-1])
                gpu_id = self.config.gpu_allocation[worker_index % len(self.config.gpu_allocation)]
            
            embedding_config = EmbeddingWorkerConfig(
                **base_config.dict(),
                default_model_provider=self.config.default_model_provider,
                default_model_name=self.config.default_model_name,
                max_batch_size=self.config.batch_size,
                gpu_id=gpu_id
            )
            return EmbeddingWorker(embedding_config)
        
        elif isinstance(self.config, StorageWorkerPoolConfig):
            return StorageWorker(base_config)
        
        else:
            raise ValueError(f"Unknown worker type: {self.config.worker_type}")


class WorkerOrchestrator:
    """Orchestrates all worker pools and manages the embedding pipeline"""
    
    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.pools: Dict[str, WorkerPool] = {}
        self.job_manager: Optional[EmbeddingJobManager] = None
        self.running = False
        self._tasks: List[asyncio.Task] = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
    
    async def start(self):
        """Start the orchestrator and all worker pools"""
        logger.info("Starting Worker Orchestrator")
        
        # Configure logging
        logger.remove()
        logger.add(sys.stderr, level=self.config.log_level)
        
        # Start monitoring if enabled
        if self.config.enable_monitoring:
            start_http_server(self.config.monitoring_port)
            logger.info(f"Monitoring dashboard available at http://localhost:{self.config.monitoring_port}")
        
        # Initialize job manager
        job_manager_config = JobManagerConfig(redis_url=self.config.redis.url)
        self.job_manager = EmbeddingJobManager(job_manager_config)
        await self.job_manager.initialize()
        
        # Start worker pools
        for pool_name, pool_config in self.config.worker_pools.items():
            pool = WorkerPool(pool_config)
            self.pools[pool_name] = pool
            await pool.start(self.config.redis.url)
        
        self.running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._monitor_queues()),
        ]
        
        if self.config.enable_autoscaling:
            self._tasks.append(asyncio.create_task(self._autoscale_loop()))
        
        logger.info("Worker Orchestrator started successfully")
        
        try:
            # Wait for shutdown signal
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            logger.info("Orchestrator tasks cancelled")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop all worker pools and cleanup"""
        logger.info("Stopping Worker Orchestrator")
        
        # Cancel background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Stop all worker pools
        stop_tasks = []
        for pool in self.pools.values():
            stop_tasks.append(pool.stop())
        
        await asyncio.gather(*stop_tasks)
        
        # Close job manager
        if self.job_manager:
            await self.job_manager.close()
        
        logger.info("Worker Orchestrator stopped")
    
    async def _monitor_queues(self):
        """Monitor queue depths and update metrics"""
        while self.running:
            try:
                if self.job_manager:
                    stats = await self.job_manager.get_queue_stats()
                    
                    for queue_name, depth in stats.items():
                        QUEUE_DEPTH.labels(queue_name=queue_name).set(depth)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring queues: {e}")
                await asyncio.sleep(30)
    
    async def _autoscale_loop(self):
        """Auto-scale worker pools based on queue depth"""
        while self.running:
            try:
                await self._check_scaling()
                await asyncio.sleep(self.config.scale_check_interval)
                
            except Exception as e:
                logger.error(f"Error in auto-scaling: {e}")
                await asyncio.sleep(self.config.scale_check_interval)
    
    async def _check_scaling(self):
        """Check if scaling is needed based on queue depths"""
        if not self.job_manager:
            return
        
        stats = await self.job_manager.get_queue_stats()
        
        for pool_name, pool in self.pools.items():
            queue_depth = stats.get(pool.config.queue_name, 0)
            current_workers = len(pool.workers)
            
            # Calculate load factor
            load_factor = queue_depth / (current_workers * 100) if current_workers > 0 else 1.0
            
            if load_factor > self.config.scale_up_threshold:
                # Scale up
                new_count = min(
                    current_workers + 1,
                    pool.config.num_workers * 2,  # Max 2x configured workers
                    self.config.max_total_workers - self._total_workers() + current_workers
                )
                
                if new_count > current_workers:
                    logger.info(f"Scaling up {pool_name} from {current_workers} to {new_count} workers")
                    await pool.scale(new_count, self.config.redis.url)
            
            elif load_factor < self.config.scale_down_threshold and current_workers > pool.config.num_workers:
                # Scale down
                new_count = max(pool.config.num_workers, current_workers - 1)
                
                if new_count < current_workers:
                    logger.info(f"Scaling down {pool_name} from {current_workers} to {new_count} workers")
                    await pool.scale(new_count, self.config.redis.url)
    
    def _total_workers(self) -> int:
        """Get total number of workers across all pools"""
        return sum(len(pool.workers) for pool in self.pools.values())


async def main():
    """Main entry point for the orchestrator"""
    # Load configuration
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding Pipeline Worker Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file",
        default=None
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of workers per pool (overrides config)",
        default=None
    )
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = OrchestrationConfig.from_yaml(args.config)
    else:
        config = OrchestrationConfig.default_config()
    
    # Override worker counts if specified
    if args.workers:
        for pool_config in config.worker_pools.values():
            pool_config.num_workers = args.workers
    
    # Create and start orchestrator
    orchestrator = WorkerOrchestrator(config)
    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(main())