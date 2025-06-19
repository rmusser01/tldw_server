# queue_schemas.py
# Queue message schemas for the embeddings scale-out architecture

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    LOW = 0
    NORMAL = 50
    HIGH = 75
    CRITICAL = 100


class UserTier(str, Enum):
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ChunkingConfig(BaseModel):
    """Configuration for text chunking"""
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    overlap: int = Field(default=200, ge=0, le=500)
    separator: str = "\n"
    preserve_metadata: bool = True
    contextualize: bool = False  # Whether to add context via LLM


class ChunkData(BaseModel):
    """Individual chunk data"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    start_index: int
    end_index: int
    sequence_number: int


class EmbeddingData(BaseModel):
    """Embedding result data"""
    chunk_id: str
    embedding: List[float]
    model_used: str
    dimensions: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Base job message that all queue messages inherit from
class EmbeddingJobMessage(BaseModel):
    """Base message for all embedding pipeline jobs"""
    job_id: str = Field(..., description="Unique job identifier")
    user_id: str = Field(..., description="User who initiated the job")
    media_id: int = Field(..., description="Media item being processed")
    priority: int = Field(default=JobPriority.NORMAL, ge=0, le=100)
    user_tier: UserTier = Field(default=UserTier.FREE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = Field(default=0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    trace_id: Optional[str] = Field(None, description="For distributed tracing")
    
    model_config = ConfigDict(use_enum_values=True)


# Chunking stage message
class ChunkingMessage(EmbeddingJobMessage):
    """Message for chunking queue"""
    content: str = Field(..., description="Raw content to be chunked")
    content_type: str = Field(..., description="Type of content (text, document, etc)")
    chunking_config: ChunkingConfig = Field(default_factory=ChunkingConfig)
    source_metadata: Dict[str, Any] = Field(default_factory=dict)


# Embedding stage message  
class EmbeddingMessage(EmbeddingJobMessage):
    """Message for embedding queue"""
    chunks: List[ChunkData] = Field(..., description="Chunks to be embedded")
    model_config: Dict[str, Any] = Field(..., description="Embedding model configuration")
    model_provider: str = Field(..., description="Provider type (huggingface, openai, etc)")
    batch_size: Optional[int] = Field(None, description="Override default batch size")


# Storage stage message
class StorageMessage(EmbeddingJobMessage):
    """Message for storage queue"""
    embeddings: List[EmbeddingData] = Field(..., description="Generated embeddings")
    collection_name: str = Field(..., description="ChromaDB collection name")
    total_chunks: int = Field(..., description="Total number of chunks processed")
    processing_time_ms: int = Field(..., description="Time taken for embedding generation")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Job status tracking
class JobInfo(BaseModel):
    """Complete job information for status tracking"""
    job_id: str
    user_id: str
    media_id: int
    status: JobStatus
    priority: int
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    chunks_processed: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=0, ge=0)
    current_stage: Optional[str] = None
    
    model_config = ConfigDict(use_enum_values=True)


# Worker health/metrics
class WorkerMetrics(BaseModel):
    """Metrics reported by workers"""
    worker_id: str
    worker_type: str  # chunking, embedding, storage
    jobs_processed: int
    jobs_failed: int
    average_processing_time_ms: float
    current_load: float = Field(ge=0.0, le=1.0)  # 0-1 representing load
    available_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)


# Queue configuration
class QueueConfig(BaseModel):
    """Configuration for a specific queue"""
    queue_name: str
    max_length: int = Field(default=10000, ge=100)
    ttl_seconds: int = Field(default=3600, ge=60)  # Message TTL
    consumer_group: str
    batch_size: int = Field(default=1, ge=1)
    poll_interval_ms: int = Field(default=100, ge=10)