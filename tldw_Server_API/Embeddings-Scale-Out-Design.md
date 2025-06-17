# Embeddings Scale-Out Architecture Design

## Overview

This document outlines the design for scaling the embeddings creation system from a synchronous, single-threaded architecture to a distributed, queue-based system capable of handling multiple concurrent users with different service tiers.

## Current System Analysis

### Limitations
1. **Synchronous Processing**: Embeddings are created inline with API requests
2. **Resource Contention**: Multiple users compete for the same GPU/model resources
3. **No Horizontal Scaling**: Cannot distribute load across multiple machines
4. **Memory Pressure**: Large models loaded/unloaded frequently
5. **Limited Fault Tolerance**: Failures block entire processing pipeline

### Strengths to Preserve
1. **Model Flexibility**: Support for multiple embedding providers (HF, ONNX, OpenAI, Local API)
2. **User Isolation**: Per-user ChromaDB collections
3. **Batch Processing**: Existing batch embedding capabilities
4. **Resource Management**: Model caching and automatic unloading

## Proposed Architecture

### High-Level Design

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   API Layer │────▶│ Job Manager  │────▶│  Message Broker │
└─────────────┘     └──────────────┘     │  (Redis/RMQ)    │
                                          └────────┬────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    │                              │                              │
             ┌──────▼────────┐            ┌───────▼────────┐            ┌────────▼───────┐
             │ Chunking Queue│            │Embedding Queue │            │ Storage Queue  │
             └──────┬────────┘            └───────┬────────┘            └────────┬───────┘
                    │                              │                              │
             ┌──────▼────────┐            ┌───────▼────────┐            ┌────────▼───────┐
             │Chunking Workers│           │Embedding Workers│           │Storage Workers │
             └───────────────┘            └────────────────┘            └────────────────┘
```

### Component Design

#### 1. Job Manager
- **Responsibilities**:
  - Accept embedding requests from API layer
  - Create job records with unique IDs
  - Route jobs to appropriate queues based on user tier
  - Track job status and provide updates
  - Handle job cancellation and cleanup

- **Implementation**:
  ```python
  class EmbeddingJobManager:
      def create_job(self, media_id: int, user_id: str, priority: str) -> str
      def get_job_status(self, job_id: str) -> JobStatus
      def cancel_job(self, job_id: str) -> bool
      def get_user_jobs(self, user_id: str) -> List[JobInfo]
  ```

#### 2. Message Broker Layer

**Option 1: Redis Streams** (Recommended)
- Already in use for rate limiting
- Supports consumer groups for scaling
- Built-in persistence and replication
- Lower operational overhead

**Option 2: RabbitMQ**
- More advanced routing capabilities
- Better for complex workflows
- Higher operational complexity

#### 3. Worker Types

##### Chunking Workers
- **Purpose**: Split content into embedding-ready chunks
- **Scaling**: CPU-bound, easily scalable
- **Features**:
  - Configurable chunk sizes and overlap
  - Context window awareness
  - Metadata preservation

##### Embedding Workers
- **Purpose**: Generate embeddings from chunks
- **Scaling**: GPU/CPU hybrid based on model
- **Features**:
  - Model pooling with LRU eviction
  - Batch processing for efficiency
  - Provider-specific optimizations
  - Circuit breaker for API calls

##### Storage Workers
- **Purpose**: Store embeddings in ChromaDB and update SQL
- **Scaling**: I/O bound, moderate scaling needs
- **Features**:
  - Transactional updates
  - Retry logic for failures
  - Batch insertion support

### Queue Schema Design

```python
# Base message schema
class EmbeddingJobMessage(BaseModel):
    job_id: str
    user_id: str
    media_id: int
    priority: int  # 0-100, higher = more priority
    created_at: float
    retry_count: int = 0
    max_retries: int = 3

# Chunking queue message
class ChunkingMessage(EmbeddingJobMessage):
    content: str
    content_type: str  # "text", "document", etc.
    chunking_config: ChunkingConfig

# Embedding queue message
class EmbeddingMessage(EmbeddingJobMessage):
    chunks: List[ChunkData]
    model_config: Union[HFModelCfg, ONNXModelCfg, OpenAIModelCfg, LocalAPICfg]
    
# Storage queue message
class StorageMessage(EmbeddingJobMessage):
    embeddings: List[EmbeddingData]
    collection_name: str
    metadata: Dict[str, Any]
```

### Priority Queue System

```python
class UserPriorityManager:
    """Manages user priorities and fair scheduling"""
    
    def calculate_priority(self, user_id: str, base_priority: int) -> int:
        """Calculate effective priority based on:
        - User tier (free, premium, enterprise)
        - Recent usage (prevent monopolization)
        - Job age (prevent starvation)
        """
        tier_multiplier = self.get_tier_multiplier(user_id)
        usage_penalty = self.get_usage_penalty(user_id)
        age_bonus = self.get_age_bonus(job_created_at)
        
        return base_priority * tier_multiplier - usage_penalty + age_bonus
```

### Resource Management

#### Model Pool Design
```python
class EmbeddingModelPool:
    """Manages a pool of embedding models with smart eviction"""
    
    def __init__(self, max_models: int = 5, max_memory_gb: float = 24.0):
        self.models: Dict[str, ModelWrapper] = {}
        self.usage_stats: Dict[str, ModelUsageStats] = {}
        self.lock = threading.Lock()
        
    def get_model(self, config: BaseModelCfg) -> ModelWrapper:
        """Get or load a model, evicting if necessary"""
        
    def release_model(self, model_id: str):
        """Mark model as available, start unload timer"""
```

#### GPU Allocation Strategy
```python
class GPUAllocator:
    """Intelligent GPU allocation for embedding workers"""
    
    def allocate_gpu(self, model_size: float, priority: int) -> Optional[int]:
        """Allocate GPU based on availability and priority"""
        
    def get_optimal_batch_size(self, model_id: str, gpu_id: int) -> int:
        """Calculate optimal batch size for GPU memory"""
```

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Create base worker classes and interfaces
- [ ] Implement job manager with in-memory tracking
- [ ] Set up Redis Streams for message passing
- [ ] Create basic monitoring and logging

### Phase 2: Core Workers (Week 3-4)
- [ ] Implement chunking workers with existing logic
- [ ] Create embedding workers with model pooling
- [ ] Build storage workers with transaction support
- [ ] Add comprehensive error handling

### Phase 3: User Awareness (Week 5-6)
- [ ] Implement priority queue system
- [ ] Add per-user resource quotas
- [ ] Create fair scheduling algorithm
- [ ] Build user dashboard for job tracking

### Phase 4: Advanced Features (Week 7-8)
- [ ] Implement streaming for large results
- [ ] Add intelligent batching for GPU efficiency
- [ ] Create auto-scaling based on queue depth
- [ ] Build comprehensive monitoring dashboard

## Migration Strategy

### Parallel Running Approach
1. Deploy new system alongside existing
2. Route percentage of traffic to new system
3. Monitor performance and errors
4. Gradually increase traffic percentage
5. Deprecate old system after validation

### Rollback Plan
- Feature flags for instant rollback
- Queue draining procedures
- Data consistency verification
- Performance baseline comparison

## Monitoring and Alerting

### Key Metrics
- **Queue Depth**: Jobs waiting per queue
- **Processing Time**: P50, P95, P99 per stage
- **Model Utilization**: GPU/CPU usage per model
- **Error Rates**: Failures per worker type
- **User Metrics**: Jobs per user, wait times by tier

### Alerts
- Queue depth > threshold
- Worker failures > threshold
- GPU memory > 90%
- Processing time > SLA
- Model loading failures

## Security Considerations

1. **Job Isolation**: Ensure jobs cannot access other users' data
2. **Resource Limits**: Prevent DoS through quota enforcement
3. **Model Security**: Validate model sources and checksums
4. **API Security**: Secure internal worker APIs

## Performance Targets

- **Throughput**: 1000 embedding jobs/minute
- **Latency**: < 5s for small texts, < 30s for documents
- **Availability**: 99.9% uptime
- **Scalability**: Linear scaling up to 10 workers per type

## Cost Optimization

1. **Spot Instances**: Use for non-critical workloads
2. **Model Sharing**: Maximize model reuse across users
3. **Batch Processing**: Group similar requests
4. **Tiered Storage**: Archive old embeddings to cheaper storage

## Future Enhancements

1. **Multi-Region**: Deploy workers across regions
2. **Edge Processing**: Client-side embeddings for privacy
3. **Model A/B Testing**: Test new models on subset of traffic
4. **Adaptive Scaling**: ML-based prediction of load patterns