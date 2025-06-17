# Embeddings Scale-Out Architecture

This directory contains the implementation of a scalable, distributed embeddings processing pipeline for the tldw_server.

## Overview

The architecture follows a fan-out pattern with specialized worker pools for each stage of the embeddings pipeline:

1. **Chunking Workers** - Split content into embedding-ready chunks
2. **Embedding Workers** - Generate embeddings from chunks using various models
3. **Storage Workers** - Store embeddings in ChromaDB and update SQL database

## Key Components

### Job Manager (`job_manager.py`)
- Manages job lifecycle and tracking
- Implements user quotas and priority scheduling
- Provides job status and monitoring APIs

### Worker Classes (`workers/`)
- `base_worker.py` - Abstract base class for all workers
- `chunking_worker.py` - Text chunking implementation
- `embedding_worker.py` - Embedding generation with model pooling
- `storage_worker.py` - ChromaDB and SQL storage

### Queue Schemas (`queue_schemas.py`)
- Pydantic models for all queue messages
- Type-safe message passing between stages
- Job tracking and status information

### Orchestration (`worker_orchestrator.py`)
- Manages worker pools and scaling
- Provides monitoring and metrics
- Handles graceful shutdown and resource management

## Getting Started

### Prerequisites
- Redis server running locally or accessible
- Python dependencies installed (`redis`, `pydantic`, `loguru`, etc.)
- GPU drivers and CUDA (for GPU-accelerated embeddings)

### Basic Usage

1. **Start the orchestrator with default configuration:**
```bash
python -m app.core.Embeddings.worker_orchestrator
```

2. **Use a custom configuration file:**
```bash
python -m app.core.Embeddings.worker_orchestrator --config embeddings_config.yaml
```

3. **Override worker counts:**
```bash
python -m app.core.Embeddings.worker_orchestrator --workers 5
```

### Integration with API

```python
from app.core.Embeddings.job_manager import EmbeddingJobManager, JobManagerConfig
from app.core.Embeddings.queue_schemas import UserTier, ChunkingConfig

# Initialize job manager
config = JobManagerConfig()
job_manager = EmbeddingJobManager(config)
await job_manager.initialize()

# Create a job
job_id = await job_manager.create_job(
    media_id=123,
    user_id="user_456",
    user_tier=UserTier.PREMIUM,
    content="Large text content to embed...",
    content_type="text",
    chunking_config=ChunkingConfig(chunk_size=1000, overlap=200),
    priority=50
)

# Check job status
job_info = await job_manager.get_job_status(job_id)
print(f"Job {job_id} status: {job_info.status}")
```

## Configuration

See `embeddings_config.yaml` for a complete example. Key configuration options:

- **Worker Pools**: Configure number of workers, batch sizes, and resources
- **Redis**: Connection settings and pooling
- **Auto-scaling**: Thresholds and limits for dynamic scaling
- **Monitoring**: Prometheus metrics and logging

## Monitoring

The system exposes Prometheus metrics on port 9090 (configurable):

- `embedding_worker_count` - Active workers by type
- `embedding_queue_depth` - Current queue depths
- `embedding_jobs_total` - Total jobs processed

## Architecture Benefits

1. **Horizontal Scalability** - Add workers dynamically based on load
2. **Fault Tolerance** - Workers can fail without affecting others
3. **Resource Optimization** - GPU pooling and batch processing
4. **Multi-tenant Support** - User quotas and fair scheduling
5. **Observability** - Built-in metrics and monitoring

## Migration from Synchronous System

The new architecture can run alongside the existing synchronous system:

1. Deploy workers and Redis
2. Route a percentage of traffic to the new system
3. Monitor performance and errors
4. Gradually increase traffic percentage
5. Deprecate old system once validated

## Future Enhancements

- Multi-region deployment
- Advanced model routing based on content
- Real-time streaming of results
- Edge processing capabilities
- A/B testing framework for models