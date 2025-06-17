# job_manager.py
# Job manager for coordinating embedding pipeline jobs

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import redis.asyncio as redis
from loguru import logger
from pydantic import BaseModel

from .queue_schemas import (
    ChunkingConfig,
    ChunkingMessage,
    JobInfo,
    JobPriority,
    JobStatus,
    UserTier,
)


class JobManagerConfig(BaseModel):
    """Configuration for job manager"""
    redis_url: str = "redis://localhost:6379"
    chunking_queue: str = "embeddings:chunking"
    embedding_queue: str = "embeddings:embedding"
    storage_queue: str = "embeddings:storage"
    job_ttl_seconds: int = 86400  # 24 hours
    max_concurrent_jobs_per_user: Dict[str, int] = {
        UserTier.FREE: 2,
        UserTier.PREMIUM: 5,
        UserTier.ENTERPRISE: 20
    }
    daily_quota_per_user: Dict[str, int] = {
        UserTier.FREE: 1000,      # chunks per day
        UserTier.PREMIUM: 10000,
        UserTier.ENTERPRISE: 100000
    }


class UserQuotaManager:
    """Manages user quotas and usage tracking"""
    
    def __init__(self, redis_client: redis.Redis, config: JobManagerConfig):
        self.redis = redis_client
        self.config = config
    
    async def check_quota(self, user_id: str, user_tier: UserTier, chunks_requested: int) -> bool:
        """Check if user has enough quota for the requested chunks"""
        quota_key = f"user:quota:{user_id}:{datetime.utcnow().date()}"
        
        # Get current usage
        current_usage = await self.redis.get(quota_key)
        current_usage = int(current_usage) if current_usage else 0
        
        # Get daily limit
        daily_limit = self.config.daily_quota_per_user.get(user_tier, 1000)
        
        # Check if request would exceed quota
        if current_usage + chunks_requested > daily_limit:
            return False
        
        return True
    
    async def consume_quota(self, user_id: str, chunks_used: int):
        """Consume user quota"""
        quota_key = f"user:quota:{user_id}:{datetime.utcnow().date()}"
        
        # Increment usage
        await self.redis.incrby(quota_key, chunks_used)
        
        # Set expiry to end of day
        await self.redis.expire(quota_key, 86400)
    
    async def get_remaining_quota(self, user_id: str, user_tier: UserTier) -> int:
        """Get remaining quota for user"""
        quota_key = f"user:quota:{user_id}:{datetime.utcnow().date()}"
        
        current_usage = await self.redis.get(quota_key)
        current_usage = int(current_usage) if current_usage else 0
        
        daily_limit = self.config.daily_quota_per_user.get(user_tier, 1000)
        
        return max(0, daily_limit - current_usage)


class PriorityCalculator:
    """Calculates effective job priority based on various factors"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def calculate_priority(
        self,
        user_id: str,
        user_tier: UserTier,
        base_priority: int,
        created_at: datetime
    ) -> int:
        """Calculate effective priority for job scheduling"""
        
        # Base tier multipliers
        tier_multipliers = {
            UserTier.FREE: 1.0,
            UserTier.PREMIUM: 2.0,
            UserTier.ENTERPRISE: 3.0
        }
        
        tier_multiplier = tier_multipliers.get(user_tier, 1.0)
        
        # Recent usage penalty (prevent monopolization)
        usage_penalty = await self._get_usage_penalty(user_id)
        
        # Age bonus (prevent starvation)
        age_minutes = (datetime.utcnow() - created_at).total_seconds() / 60
        age_bonus = min(20, age_minutes * 2)  # Max 20 point bonus
        
        # Calculate final priority
        effective_priority = (base_priority * tier_multiplier) - usage_penalty + age_bonus
        
        return max(0, min(100, int(effective_priority)))
    
    async def _get_usage_penalty(self, user_id: str) -> float:
        """Get usage penalty based on recent job submissions"""
        usage_key = f"user:recent_jobs:{user_id}"
        
        # Count jobs in last hour
        recent_jobs = await self.redis.zcount(
            usage_key,
            (datetime.utcnow() - timedelta(hours=1)).timestamp(),
            datetime.utcnow().timestamp()
        )
        
        # Apply penalty: 2 points per recent job
        return min(30, recent_jobs * 2)  # Max 30 point penalty


class EmbeddingJobManager:
    """Manages the lifecycle of embedding jobs"""
    
    def __init__(self, config: JobManagerConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.quota_manager: Optional[UserQuotaManager] = None
        self.priority_calculator: Optional[PriorityCalculator] = None
    
    async def initialize(self):
        """Initialize Redis connection and sub-managers"""
        self.redis_client = await redis.from_url(
            self.config.redis_url,
            decode_responses=True
        )
        self.quota_manager = UserQuotaManager(self.redis_client, self.config)
        self.priority_calculator = PriorityCalculator(self.redis_client)
        
        # Create consumer groups if they don't exist
        await self._ensure_consumer_groups()
        
        logger.info("EmbeddingJobManager initialized")
    
    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
    
    async def create_job(
        self,
        media_id: int,
        user_id: str,
        user_tier: UserTier,
        content: str,
        content_type: str = "text",
        chunking_config: Optional[ChunkingConfig] = None,
        priority: int = JobPriority.NORMAL,
        metadata: Optional[Dict] = None
    ) -> str:
        """Create a new embedding job"""
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Check concurrent jobs limit
        active_jobs = await self._get_user_active_jobs(user_id)
        max_concurrent = self.config.max_concurrent_jobs_per_user.get(user_tier, 2)
        
        if len(active_jobs) >= max_concurrent:
            raise ValueError(f"User {user_id} has reached concurrent job limit ({max_concurrent})")
        
        # Estimate chunks (rough estimate)
        chunking_config = chunking_config or ChunkingConfig()
        estimated_chunks = len(content) // chunking_config.chunk_size + 1
        
        # Check quota
        if not await self.quota_manager.check_quota(user_id, user_tier, estimated_chunks):
            remaining = await self.quota_manager.get_remaining_quota(user_id, user_tier)
            raise ValueError(f"Insufficient quota. Remaining: {remaining} chunks")
        
        # Calculate effective priority
        created_at = datetime.utcnow()
        effective_priority = await self.priority_calculator.calculate_priority(
            user_id, user_tier, priority, created_at
        )
        
        # Create job info
        job_info = JobInfo(
            job_id=job_id,
            user_id=user_id,
            media_id=media_id,
            status=JobStatus.PENDING,
            priority=effective_priority,
            created_at=created_at,
            updated_at=created_at
        )
        
        # Store job info
        job_key = f"job:{job_id}"
        await self.redis_client.hset(
            job_key,
            mapping=job_info.dict()
        )
        await self.redis_client.expire(job_key, self.config.job_ttl_seconds)
        
        # Track user's active jobs
        await self._track_user_job(user_id, job_id)
        
        # Create chunking message
        chunking_message = ChunkingMessage(
            job_id=job_id,
            user_id=user_id,
            media_id=media_id,
            priority=effective_priority,
            user_tier=user_tier,
            content=content,
            content_type=content_type,
            chunking_config=chunking_config,
            source_metadata=metadata or {}
        )
        
        # Add to chunking queue
        await self.redis_client.xadd(
            self.config.chunking_queue,
            chunking_message.dict()
        )
        
        logger.info(f"Created job {job_id} for user {user_id} with priority {effective_priority}")
        
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[JobInfo]:
        """Get current job status"""
        job_key = f"job:{job_id}"
        job_data = await self.redis_client.hgetall(job_key)
        
        if not job_data:
            return None
        
        return JobInfo(**job_data)
    
    async def cancel_job(self, job_id: str, user_id: str) -> bool:
        """Cancel a job"""
        job_info = await self.get_job_status(job_id)
        
        if not job_info:
            return False
        
        # Verify ownership
        if job_info.user_id != user_id:
            raise ValueError("User does not own this job")
        
        # Check if job can be cancelled
        if job_info.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            return False
        
        # Update job status
        job_key = f"job:{job_id}"
        await self.redis_client.hset(
            job_key,
            mapping={
                "status": JobStatus.CANCELLED,
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
        # Remove from user's active jobs
        await self._remove_user_job(user_id, job_id)
        
        logger.info(f"Cancelled job {job_id}")
        
        return True
    
    async def get_user_jobs(
        self,
        user_id: str,
        include_completed: bool = False,
        limit: int = 100
    ) -> List[JobInfo]:
        """Get all jobs for a user"""
        if include_completed:
            # Get all jobs from recent history
            pattern = f"job:*"
            cursor = 0
            jobs = []
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor, match=pattern, count=100
                )
                
                for key in keys:
                    job_data = await self.redis_client.hgetall(key)
                    if job_data and job_data.get("user_id") == user_id:
                        jobs.append(JobInfo(**job_data))
                
                if cursor == 0:
                    break
            
            # Sort by created_at descending
            jobs.sort(key=lambda x: x.created_at, reverse=True)
            
            return jobs[:limit]
        
        else:
            # Get only active jobs
            active_job_ids = await self._get_user_active_jobs(user_id)
            jobs = []
            
            for job_id in active_job_ids:
                job_info = await self.get_job_status(job_id)
                if job_info:
                    jobs.append(job_info)
            
            return jobs
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get current queue statistics"""
        stats = {}
        
        for queue_name in [self.config.chunking_queue, self.config.embedding_queue, self.config.storage_queue]:
            length = await self.redis_client.xlen(queue_name)
            stats[queue_name] = length
        
        return stats
    
    async def _ensure_consumer_groups(self):
        """Ensure consumer groups exist for all queues"""
        queues = [
            (self.config.chunking_queue, "chunking-workers"),
            (self.config.embedding_queue, "embedding-workers"),
            (self.config.storage_queue, "storage-workers")
        ]
        
        for queue_name, group_name in queues:
            try:
                await self.redis_client.xgroup_create(
                    queue_name, group_name, id='0'
                )
                logger.info(f"Created consumer group {group_name} for {queue_name}")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    # Group already exists
                    pass
                else:
                    raise
    
    async def _track_user_job(self, user_id: str, job_id: str):
        """Track user's active jobs"""
        # Add to active jobs set
        active_key = f"user:active_jobs:{user_id}"
        await self.redis_client.sadd(active_key, job_id)
        
        # Add to recent jobs sorted set for usage tracking
        recent_key = f"user:recent_jobs:{user_id}"
        await self.redis_client.zadd(
            recent_key,
            {job_id: datetime.utcnow().timestamp()}
        )
        
        # Clean up old entries (older than 1 hour)
        cutoff = (datetime.utcnow() - timedelta(hours=1)).timestamp()
        await self.redis_client.zremrangebyscore(recent_key, 0, cutoff)
    
    async def _remove_user_job(self, user_id: str, job_id: str):
        """Remove job from user's active jobs"""
        active_key = f"user:active_jobs:{user_id}"
        await self.redis_client.srem(active_key, job_id)
    
    async def _get_user_active_jobs(self, user_id: str) -> List[str]:
        """Get user's active job IDs"""
        active_key = f"user:active_jobs:{user_id}"
        return await self.redis_client.smembers(active_key)