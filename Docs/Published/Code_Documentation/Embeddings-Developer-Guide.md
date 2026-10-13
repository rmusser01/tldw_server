# Embeddings System Developer Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Working with the Codebase](#working-with-the-codebase)
4. [Adding New Providers](#adding-new-providers)
5. [Testing Guidelines](#testing-guidelines)
6. [Debugging & Troubleshooting](#debugging-troubleshooting)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring & Observability](#monitoring-observability)

## Introduction

This guide is for developers working on the tldw_server embeddings system codebase. It covers implementation details, best practices, and common development tasks.

## Architecture Overview

### System Components

```mermaid
flowchart LR
    Client[API client] -->|POST /api/v1/embeddings| SyncEndpoint[embeddings_v5_production_enhanced.py]
    Client -->|POST /api/v1/media/{id}/embeddings| JobEndpoint[media_embeddings.py]
    SyncEndpoint --> Pipeline[Embeddings pipeline (chunking + Embeddings_Create + storage)]
    JobEndpoint --> JobsAdapter[EmbeddingsJobsAdapter]
    JobsAdapter --> JobManager[Core Jobs JobManager (root status/billing)]
    JobsAdapter --> RedisStreams[Redis Streams queues]
    RedisStreams --> RedisWorker[Embeddings redis_worker]
    RedisWorker --> Pipeline
    Pipeline --> Stores[ChromaDB + media DB updates]
```

### File Structure

```
tldw_Server_API/
├── app/
│   ├── api/v1/endpoints/
│   │   ├── embeddings_v5_production_enhanced.py  # Synchronous API with circuit breaker
│   │   ├── media_embeddings.py                   # Media chunking + embeddings storage
│   │   └── vector_stores_openai.py              # Vector store ops that use embeddings
│   │
│   └── core/
│       └── Embeddings/
│           ├── jobs_adapter.py                  # Core Jobs adapter for embeddings jobs
│           ├── services/redis_worker.py         # Redis Streams worker for media + content stages
│           ├── services/jobs_worker.py          # Legacy Jobs worker (compat only)
│           └── Embeddings_Server/
│               └── Embeddings_Create.py         # Core embedding logic (OpenAI/HF/ONNX/local)
│
├── tests/Embeddings/                            # If present (naming may vary)
│   └── ...
│
└── Config_Files/
    └── embeddings_production_config.yaml        # Example configuration (if used)
```

## Working with the Codebase

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/tldw_server.git
cd tldw_server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (with dev extras)
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Configuration Management

```python
# Load configuration in your code
from tldw_Server_API.app.core.config import settings

# Access configuration values
api_key = settings.get("OPENAI_API_KEY")
cache_config = settings.get("CACHE_CONFIG")

# Override for testing
test_settings = {
    "CACHE_CONFIG": {
        "max_size": 100,
        "ttl_seconds": 60
    }
}
```

### Using the Synchronous API

```python
from tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced import (
    create_embeddings_batch_async,
    TTLCache,
    ConnectionPoolManager
)

# Initialize components
cache = TTLCache(max_size=1000, ttl_seconds=3600)
connection_manager = ConnectionPoolManager()

# Create embeddings
async def generate_embeddings(texts: List[str]):
    embeddings = await create_embeddings_batch_async(
        texts=texts,
        provider="openai",
        model_id="text-embedding-3-small",
        dimensions=1536  # optional; applies to specific providers only
    )
    return embeddings

"""
Notes:
- Token arrays are accepted by the REST API (`POST /api/v1/embeddings`); the service decodes tokens with the model tokenizer or `cl100k_base` fallback.
- Provider fallback is configurable via `EMBEDDINGS_FALLBACK_CHAIN` or defaults (e.g., openai→huggingface→onnx→local_api).
- Policy enforcement can be enabled via `EMBEDDINGS_ENFORCE_POLICY`; admin bypass can be disabled with `EMBEDDINGS_ENFORCE_POLICY_STRICT=true`.
- For non-OpenAI providers, `dimensions` is applied as post-processing based on `EMBEDDINGS_DIMENSION_POLICY` (reduce|pad|ignore).
"""
```

## Choosing Chunking for RAG

- Default (general text): use `words` with `max_size≈400`, `overlap≈200` via `Chunker.process_text`. These defaults balance semantic continuity and redundancy.
- Long/structured docs: prefer `hierarchical=True` or `structure_aware` to preserve headings, lists, tables, and code fences. Embed the flattened rows for better retrieval and re‑ranking.
- Sentences: for QA-style matching or when sentence boundaries matter, use `method="sentences"` with smaller windows (e.g., 8–10) and small overlap (e.g., 2).
- Tokens: when you must align to model token windows, use `method="tokens"` and pass `tokenizer_name_or_path`. Offsets come from tokenizer `offset_mapping` when available; a robust fallback is used otherwise.
- Propositions: for claim/fact retrieval, use `method="propositions"`. Defaults for engine and tuning live in `tldw_Server_API/app/core/Chunking/__init__.py` (`DEFAULT_CHUNK_OPTIONS`). LLM‑based extraction requires wiring `llm_call_func`/`llm_config`.
- Code: use `method="code"` with `code_mode="ast"` for Python (auto‑routes to AST if language hints start with `py`). This yields structure‑aligned chunks.
- JSON/XML: enable `preserve_metadata` and `single_metadata_reference` to avoid repeating large metadata objects; adjust `metadata_reference_key` as needed.
- Streaming vs. generator: for very large files on disk use `chunk_file_stream`; for whole in‑memory text without metadata, prefer `chunk_text_generator`. For normalized rows with metadata, use `process_text`.
- Discoverability: `GET /api/v1/chunking/capabilities` returns the runtime set of methods and defaults. See also `Docs/Code_Documentation/Guides/Chunking_Code_Guide.md`.

### Using the Job-Based System

```python
import asyncio
from typing import List

import httpx

async def process_large_batch(texts: List[str]) -> dict:
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Create job
        resp = await client.post(
            "/api/v1/media/123/embeddings",
            json={"embedding_model": "text-embedding-3-small"},
            headers={"X-API-KEY": "your-api-key"}
        )
        resp.raise_for_status()
        job_id = resp.json()["job_id"]

        # Poll status
        while True:
            status_resp = await client.get(
                f"/api/v1/media/embeddings/jobs/{job_id}",
                headers={"X-API-KEY": "your-api-key"}
            )
            status_resp.raise_for_status()
            payload = status_resp.json()
            status = payload.get("status")
            if status in ["completed", "failed"]:
                return payload
            await asyncio.sleep(1)
```

Via SDK (if direct integration is needed):

```python
from tldw_Server_API.app.core.Embeddings.jobs_adapter import EmbeddingsJobsAdapter

adapter = EmbeddingsJobsAdapter()
job = adapter.create_job(
    user_id="user123",
    media_id=123,
    embedding_model="text-embedding-3-small"
)
job_status = adapter.get_job(job["id"], "user123")
```

See `tldw_Server_API/tests/Embeddings/test_media_embedding_jobs.py` for a complete working example.

## Adding New Providers

### Step 1: Define Provider Configuration

```python
# In embeddings_v5_production_enhanced.py

class EmbeddingProvider(str, Enum):
    # ... existing providers ...
    NEWPROVIDER = "newprovider"

PROVIDER_MODELS = {
    # ... existing models ...
    EmbeddingProvider.NEWPROVIDER: [
        "newprovider-model-1",
        "newprovider-model-2"
    ]
}
```

### Step 2: Implement Provider Configuration Builder (endpoint)

```python
def build_provider_config(
    provider: EmbeddingProvider,
    model: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    dimensions: Optional[int] = None
) -> Dict[str, Any]:
    # ... existing providers ...

    elif provider == EmbeddingProvider.NEWPROVIDER:
        return {
            "provider": "newprovider",
            "model_name_or_path": model,
            "api_key": api_key or settings.get("NEWPROVIDER_API_KEY"),
            "api_url": api_url or "https://api.newprovider.com/v1",
            "dimensions": dimensions
        }
```

### Step 3: Implement Embedding Creation (engine)

```python
# In Embeddings_Create.py

class NewProviderCfg(BaseModelCfg):
    provider: str = "newprovider"
    api_url: str
    api_key: Optional[str] = None

class NewProviderEmbedder(BaseEmbedder):
    def __init__(self, config: NewProviderCfg):
        self.config = config
        self.client = self._init_client()

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        # Implement provider-specific logic
        response = self.client.embeddings.create(
            input=texts,
            model=self.config.model_name_or_path
        )
        return np.array([e.embedding for e in response.data])
```

Also add a branch handling `provider == "newprovider"` in `create_embeddings_batch(...)`.

### Step 4: Add Tests

```python
# In test_embeddings_v5_unit.py

@pytest.mark.unit
def test_newprovider_configuration():
    config = build_provider_config(
        EmbeddingProvider.NEWPROVIDER,
        "newprovider-model-1",
        api_key="test-key"
    )
    assert config["provider"] == "newprovider"
    assert config["api_key"] == "test-key"

@pytest.mark.unit
async def test_newprovider_embeddings():
    with patch('newprovider.client') as mock_client:
        mock_client.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1, 0.2, 0.3])]
        )

        embeddings = await create_embeddings_batch_async(
            texts=["test"],
            provider="newprovider",
            model_id="newprovider-model-1"
        )

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 3
```

## Testing Guidelines

### Test Structure

```mermaid
graph TD
    subgraph "Test Pyramid"
        UNIT[Unit Tests<br/>~70%]
        INTEGRATION[Integration Tests<br/>~20%]
        E2E[End-to-End Tests<br/>~10%]
    end

    subgraph "Test Types"
        FUNC[Functional Tests]
        PROP[Property Tests]
        PERF[Performance Tests]
        SEC[Security Tests]
    end

    UNIT --> FUNC
    UNIT --> PROP
    INTEGRATION --> FUNC
    INTEGRATION --> PERF
    E2E --> FUNC
    E2E --> SEC
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/Embeddings/test_embeddings_v5_unit.py

# Run with coverage
pytest --cov=tldw_Server_API --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests (requires services)
RUN_INTEGRATION_TESTS=true pytest -m integration

# Run property tests
pytest tests/Embeddings/test_embeddings_v5_property.py

# Run with verbose output
pytest -v

# Run specific test
pytest -k test_cache_ttl_expiration
```

### Writing Tests

#### Unit Test Example

```python
@pytest.mark.unit
class TestCacheOperations:
    async def test_cache_set_and_get(self):
        cache = TTLCache(max_size=10, ttl_seconds=60)

        # Test set
        await cache.set("key1", [1.0, 2.0])

        # Test get
        value = await cache.get("key1")
        assert value == [1.0, 2.0]

        # Test non-existent key
        value = await cache.get("nonexistent")
        assert value is None
```

#### Integration Test Example

```python
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Integration tests require RUN_INTEGRATION_TESTS=true"
)
async def test_real_openai_embeddings(client):
    response = await client.post(
        "/api/v1/embeddings",
        json={
            "input": "Real test with OpenAI",
            "model": "text-embedding-3-small"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["data"][0]["embedding"]) == 1536
```

#### Property Test Example

```python
from hypothesis import given, strategies as st

@given(
    max_size=st.integers(min_value=1, max_value=100),
    num_items=st.integers(min_value=0, max_value=200)
)
async def test_cache_never_exceeds_max_size(max_size, num_items):
    cache = TTLCache(max_size=max_size)

    for i in range(num_items):
        await cache.set(f"key_{i}", [float(i)])

    stats = cache.stats()
    assert stats['size'] <= max_size
```

## Debugging & Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors

```python
# Problem: ImportError: cannot import name 'create_embeddings_batch'

# Solution: Check Python path
import sys
sys.path.append('/path/to/tldw_server')

# Or set PYTHONPATH
export PYTHONPATH=/path/to/tldw_server:$PYTHONPATH
```

#### 2. Provider Connection Issues

```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use loguru
from loguru import logger
logger.add("debug.log", level="DEBUG")

# Test provider connection
async def test_provider_connection():
    try:
        embeddings = await create_embeddings_batch_async(
            texts=["test"],
            provider="openai",
            model_id="text-embedding-3-small"
        )
        logger.info("Provider connection successful")
    except Exception as e:
        logger.error(f"Provider connection failed: {e}")
```

#### 3. Cache Issues

```python
# Debug cache operations
cache = TTLCache(max_size=100, ttl_seconds=60)

# Enable cache statistics
async def debug_cache():
    stats = cache.stats()
    logger.info(f"Cache stats: {stats}")

    # Check specific key
    key = get_cache_key("text", "provider", "model")
    value = await cache.get(key)
    logger.info(f"Cache value for {key}: {value}")

    # Clear cache if needed
    await cache.clear()
```

#### 4. Memory Issues

```python
# Monitor memory usage
import psutil
import gc

def check_memory():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

    # Force garbage collection
    gc.collect()

    # Check for memory leaks
    import tracemalloc
    tracemalloc.start()
    # ... run operations ...
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    for stat in top_stats[:10]:
        logger.info(stat)
```

### Debugging Tools

#### Using pdb

```python
import pdb

async def debug_function():
    # Set breakpoint
    pdb.set_trace()

    # Or use breakpoint() in Python 3.7+
    breakpoint()

    # Inspect variables
    result = await some_operation()
    return result
```

#### Using IPython

```python
# Install IPython
pip install ipython

# Use IPython for debugging
from IPython import embed

async def debug_with_ipython():
    result = await some_operation()
    embed()  # Drops into IPython shell
    return result
```

#### Performance Profiling

```python
import cProfile
import pstats
from io import StringIO

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()

    # Run code to profile
    result = expensive_operation()

    profiler.disable()

    # Print statistics
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

    logger.info(stream.getvalue())
    return result
```

## Performance Optimization

### Caching Strategies

```python
# 1. Multi-level caching
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = {}  # In-memory
        self.l2_cache = Redis()  # Redis

    async def get(self, key):
        # Check L1
        if key in self.l1_cache:
            return self.l1_cache[key]

        # Check L2
        value = await self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
        return value

# 2. Preemptive cache warming
async def warm_cache(popular_texts: List[str]):
    for text in popular_texts:
        key = get_cache_key(text, "openai", "text-embedding-3-small")
        if not await cache.get(key):
            embedding = await create_embedding(text)
            await cache.set(key, embedding)
```

### Batching Optimizations

```python
# Optimal batch sizes per provider
OPTIMAL_BATCH_SIZES = {
    "openai": 100,
    "huggingface": 32,
    "cohere": 96
}

async def optimize_batching(texts: List[str], provider: str):
    batch_size = OPTIMAL_BATCH_SIZES.get(provider, 50)

    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = await create_embeddings_batch_async(
            texts=batch,
            provider=provider
        )
        results.extend(embeddings)

    return results
```

### Connection Pool Tuning

```python
# Tune connection pool per provider
CONNECTION_CONFIGS = {
    "openai": {
        "limit": 100,
        "limit_per_host": 30,
        "ttl_dns_cache": 300
    },
    "huggingface": {
        "limit": 50,
        "limit_per_host": 10,
        "keepalive_timeout": 30
    }
}

async def create_optimized_session(provider: str):
    config = CONNECTION_CONFIGS.get(provider, {})
    connector = aiohttp.TCPConnector(**config)
    timeout = aiohttp.ClientTimeout(total=30)

    return aiohttp.ClientSession(
        connector=connector,
        timeout=timeout
    )
```

### Database Optimizations

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_timeout=30,
    pool_recycle=3600
)

# Batch inserts
async def batch_insert_embeddings(embeddings: List[Dict]):
    async with engine.begin() as conn:
        await conn.execute(
            embeddings_table.insert(),
            embeddings
        )
```

## Monitoring & Observability

### Metrics Implementation

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
embedding_requests = Counter(
    'embedding_requests_total',
    'Total embedding requests',
    ['provider', 'model', 'status']
)

embedding_latency = Histogram(
    'embedding_latency_seconds',
    'Embedding request latency',
    ['provider', 'model']
)

cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Cache hit rate percentage'
)

# Use metrics
async def track_request(provider: str, model: str):
    with embedding_latency.labels(provider, model).time():
        try:
            result = await create_embeddings()
            embedding_requests.labels(provider, model, 'success').inc()
            return result
        except Exception as e:
            embedding_requests.labels(provider, model, 'error').inc()
            raise
```

### Logging Best Practices

```python
from loguru import logger
import contextvars

# Request ID for tracing
request_id = contextvars.ContextVar('request_id', default=None)

# Configure structured logging
logger.add(
    "logs/embeddings.log",
    format="{time} {level} {message} {extra}",
    rotation="100 MB",
    retention="30 days",
    compression="zip"
)

# Log with context
async def process_request(request):
    req_id = str(uuid.uuid4())
    request_id.set(req_id)

    logger.info(
        "Processing embedding request",
        extra={
            "request_id": req_id,
            "provider": request.provider,
            "model": request.model,
            "input_count": len(request.input)
        }
    )

    try:
        result = await create_embeddings(request)
        logger.info(
            "Request completed successfully",
            extra={"request_id": req_id, "duration": elapsed}
        )
        return result
    except Exception as e:
        logger.error(
            f"Request failed: {e}",
            extra={"request_id": req_id, "error": str(e)}
        )
        raise
```

### Health Checks

```python
async def comprehensive_health_check():
    health = {
        "status": "healthy",
        "checks": {},
        "timestamp": datetime.utcnow().isoformat()
    }

    # Check cache
    try:
        await cache.get("health_check_key")
        health["checks"]["cache"] = "healthy"
    except Exception as e:
        health["checks"]["cache"] = f"unhealthy: {e}"
        health["status"] = "degraded"

    # Check providers
    for provider in ["openai", "huggingface"]:
        try:
            await test_provider(provider)
            health["checks"][provider] = "healthy"
        except Exception as e:
            health["checks"][provider] = f"unhealthy: {e}"
            health["status"] = "degraded"

    # Check database
    try:
        await db.execute("SELECT 1")
        health["checks"]["database"] = "healthy"
    except Exception as e:
        health["checks"]["database"] = f"unhealthy: {e}"
        health["status"] = "unhealthy"

    return health
```

### Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Setup Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Use tracing
async def traced_operation():
    with tracer.start_as_current_span("create_embeddings") as span:
        span.set_attribute("provider", "openai")
        span.set_attribute("model", "text-embedding-3-small")

        try:
            result = await create_embeddings()
            span.set_attribute("success", True)
            return result
        except Exception as e:
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            raise
```

## Best Practices

### Code Style

- Follow PEP 8
- Use type hints
- Write comprehensive docstrings
- Keep functions focused and small
- Use async/await consistently

### Error Handling

```python
class EmbeddingError(Exception):
    """Base exception for embedding errors"""
    pass

class ProviderError(EmbeddingError):
    """Provider-specific errors"""
    pass

class QuotaExceededError(EmbeddingError):
    """User quota exceeded"""
    pass

async def handle_errors():
    try:
        result = await create_embeddings()
    except ProviderError as e:
        logger.error(f"Provider error: {e}")
        # Try fallback provider
        result = await create_embeddings(provider="fallback")
    except QuotaExceededError as e:
        logger.warning(f"Quota exceeded: {e}")
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

    return result
```

### Security

- Never log sensitive data (API keys, user data)
- Validate all inputs
- Use parameterized queries
- Implement rate limiting
- Keep dependencies updated

### Documentation

- Document all public APIs
- Include examples in docstrings
- Keep README updated
- Document configuration options
- Maintain changelog

---

For API usage documentation, see [Embeddings API Guide](../API-related/Embeddings-API-Guide.md)
