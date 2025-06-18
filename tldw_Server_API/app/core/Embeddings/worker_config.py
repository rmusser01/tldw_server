# worker_config.py
# Configuration schemas for worker pools and orchestration

import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RedisConfig(BaseModel):
    """Redis connection configuration"""
    url: str = Field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"), 
        description="Redis connection URL"
    )
    max_connections: int = Field(default=50, ge=1, description="Maximum connections in pool")
    decode_responses: bool = Field(default=True, description="Decode responses to strings")


class WorkerPoolConfig(BaseModel):
    """Configuration for a worker pool"""
    worker_type: str = Field(..., description="Type of worker (chunking, embedding, storage)")
    num_workers: int = Field(default=1, ge=1, description="Number of worker instances")
    queue_name: str = Field(..., description="Redis stream queue name")
    consumer_group: str = Field(..., description="Redis consumer group name")
    batch_size: int = Field(default=1, ge=1, description="Messages to process per batch")
    poll_interval_ms: int = Field(default=100, ge=10, description="Queue poll interval")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    heartbeat_interval: int = Field(default=30, ge=10, description="Heartbeat interval in seconds")
    shutdown_timeout: int = Field(default=30, ge=10, description="Graceful shutdown timeout")
    metrics_interval: int = Field(default=60, ge=30, description="Metrics reporting interval")


class ChunkingWorkerPoolConfig(WorkerPoolConfig):
    """Specific configuration for chunking workers"""
    worker_type: str = "chunking"
    queue_name: str = "embeddings:chunking"
    consumer_group: str = "chunking-workers"
    default_chunk_size: int = Field(default=1000, ge=100, description="Default chunk size")
    default_overlap: int = Field(default=200, ge=0, description="Default chunk overlap")


class EmbeddingWorkerPoolConfig(WorkerPoolConfig):
    """Specific configuration for embedding workers"""
    worker_type: str = "embedding"
    queue_name: str = "embeddings:embedding"
    consumer_group: str = "embedding-workers"
    batch_size: int = Field(default=16, ge=1, description="Embeddings per batch")
    gpu_allocation: Optional[List[int]] = Field(default=None, description="GPU IDs to use")
    default_model_provider: str = Field(default="huggingface", description="Default embedding provider")
    default_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    max_model_cache_size: int = Field(default=5, ge=1, description="Maximum models to cache")
    model_unload_timeout: int = Field(default=300, ge=60, description="Model unload timeout")


class StorageWorkerPoolConfig(WorkerPoolConfig):
    """Specific configuration for storage workers"""
    worker_type: str = "storage"
    queue_name: str = "embeddings:storage"
    consumer_group: str = "storage-workers"
    batch_size: int = Field(default=100, ge=1, description="Storage operations per batch")
    chroma_batch_size: int = Field(default=100, ge=1, description="ChromaDB batch size")
    transaction_timeout: int = Field(default=30, ge=10, description="Database transaction timeout")


class OrchestrationConfig(BaseModel):
    """Main orchestration configuration"""
    redis: RedisConfig = Field(default_factory=RedisConfig)
    worker_pools: Dict[str, WorkerPoolConfig] = Field(default_factory=dict)
    enable_monitoring: bool = Field(default=True, description="Enable monitoring dashboard")
    monitoring_port: int = Field(default=9090, description="Monitoring dashboard port")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Auto-scaling configuration
    enable_autoscaling: bool = Field(default=False, description="Enable auto-scaling")
    scale_up_threshold: float = Field(default=0.8, ge=0.1, le=1.0, description="Queue depth threshold for scaling up")
    scale_down_threshold: float = Field(default=0.2, ge=0.1, le=1.0, description="Queue depth threshold for scaling down")
    scale_check_interval: int = Field(default=60, ge=30, description="Auto-scaling check interval")
    
    # Resource limits
    max_total_workers: int = Field(default=50, ge=1, description="Maximum total workers across all pools")
    max_memory_gb: float = Field(default=32.0, ge=1.0, description="Maximum total memory usage")
    
    @classmethod
    def default_config(cls) -> "OrchestrationConfig":
        """Create default configuration with all worker pools"""
        return cls(
            worker_pools={
                "chunking": ChunkingWorkerPoolConfig(num_workers=2),
                "embedding": EmbeddingWorkerPoolConfig(
                    num_workers=2,
                    gpu_allocation=[0, 1]  # Use first 2 GPUs if available
                ),
                "storage": StorageWorkerPoolConfig(num_workers=1)
            }
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> "OrchestrationConfig":
        """Load configuration from YAML file"""
        import yaml
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        import yaml
        
        with open(path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)