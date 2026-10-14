# Embeddings Module Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [API Reference](#api-reference)
4. [Configuration](#configuration)
5. [Security Features](#security-features)
6. [Provider Integration](#provider-integration)
7. [Performance & Scaling](#performance--scaling)
8. [Monitoring & Observability](#monitoring--observability)
9. [Troubleshooting](#troubleshooting)
10. [Development Guide](#development-guide)

---

## Overview

The Embeddings Module provides a unified interface for generating text embeddings across multiple providers with built-in caching, optional rate limiting, resource management, and observability. Today the production path supports OpenAI and HuggingFace; the core engine also supports ONNX (optimum + onnxruntime) and a Local API mode. Additional providers (Cohere, Google, Mistral, Voyage) are defined in configuration and provider resolution but are not yet fully integrated end-to-end in the embedding engine.

### Key Features
- OpenAI-compatible synchronous API with circuit breaker and resilient connection handling
- Token-array inputs supported (single `List[int]` or batch `List[List[int]]`)
- TTL-based caching with background cleanup and Prometheus metrics
- Provider fallback chain with model mapping and response headers (`X-Embeddings-Provider`, `X-Embeddings-Fallback-From`)
- Dimension policy for non-native sizes (`reduce`, `pad`, `ignore`) with `X-Embeddings-Dimensions-Policy`
- Resource management and model caching (LRU eviction) in the core engine
- Security hardening with input validation and audit logging
- Optional rate limiting (disabled by default; enable with `EMBEDDINGS_RATE_LIMIT=on`)

### Current Version
- Production System: `embeddings_v5_production_enhanced.py` (circuit breaker, caching, metrics)
- Future System: Worker-based scale-out architecture (implemented under `/app/core/Embeddings/` and exposed via `/api/v1/media` job endpoints like `POST /media/{media_id}/embeddings`, `POST /media/embeddings/batch`, `GET /media/embeddings/jobs`, `GET /media/embeddings/jobs/{job_id}`, `GET /media/{media_id}/embeddings/status`)

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                     API Layer                            │
│  /api/v1/embeddings (OpenAI-compatible REST API)        │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Service Layer                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │Rate Limiter │  │Circuit Break │  │  Auth Check   │ │
│  └─────────────┘  └──────────────┘  └───────────────┘ │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│           Embedding Engine                               │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐   │
│  │  Cache   │  │ Provider │  │ Resource Manager   │   │
│  │  (TTL)   │  │  Router  │  │  (LRU Eviction)    │   │
│  └──────────┘  └──────────┘  └────────────────────┘   │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Provider Adapters                           │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌─────────┐ │
│  │ OpenAI   │ │HuggingFace │ │  Cohere  │ │  Local  │ │
│  └──────────┘ └────────────┘ └──────────┘ └─────────┘ │
└──────────────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              Storage Layer                               │
│  ┌──────────────┐           ┌──────────────────────────┐│
│  │  ChromaDB    │           │ Unified Audit Service    ││
│  │(Vector Store)│           │ (Security Events; DI)    ││
│  └──────────────┘           └──────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

### File Structure

```
app/core/Embeddings/
├── __init__.py
├── README.md                          # Architecture overview
├── ChromaDB_Library.py                # Vector storage management
├── circuit_breaker.py                 # Fault tolerance
├── (uses unified audit service)       # Security audit logging via DI
├── rate_limiter.py                    # Per-user rate limiting
├── embeddings_config.yaml             # Configuration file
├── jobs_adapter.py                    # Core Jobs adapter for embeddings jobs
├── worker_config.py                   # Legacy config (unused; retained for reference)
├── services/
│   └── redis_worker.py                # Redis Streams worker for embeddings stages
└── Embeddings_Server/
    ├── __init__.py
    └── Embeddings_Create.py          # Core embedding logic
```

---

## API Reference

### Create Embeddings

**Endpoint:** `POST /api/v1/embeddings`

**Request:**
```json
{
  "input": "string or array of strings",
  "model": "text-embedding-3-small",
  "encoding_format": "float",
  "user": "user_123"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    { "object": "embedding", "index": 0, "embedding": [0.123, -0.456, ...] }
  ],
  "model": "text-embedding-3-small",        // or "huggingface:sentence-transformers/all-MiniLM-L6-v2"
  "usage": { "prompt_tokens": 8, "total_tokens": 8 }
}
```

Notes:
- Inputs may be a string, list of strings, or token arrays (`List[int]` or `List[List[int]]`). Token arrays are decoded to text using the model’s tokenizer when available or `cl100k_base` fallback; usage accounting uses the supplied token counts.
- Up to 2048 inputs per request; per-model token limits are enforced with a dedicated error payload (`{"error":"input_too_long", ...}`).
- Dimensions: For OpenAI `text-embedding-3-*`, the `dimensions` parameter is honored by the upstream API. For HuggingFace/ONNX/Local backends, `dimensions` is applied as a post-processing step (policy: `reduce` slices to first-N, `pad` zero-pads, `ignore` leaves native size). Configure via `EMBEDDINGS_DIMENSION_POLICY`.
- Encoding: Set `encoding_format` to `"base64"` to receive base64-encoded vectors; otherwise vectors are returned as normalized float arrays.
- Provider selection: set header `x-provider: openai|huggingface|onnx|local_api`, or prefix the model `provider:model` (e.g., `huggingface:sentence-transformers/all-MiniLM-L6-v2`). If omitted, the server auto-detects from the model name or defaults to OpenAI.
- Authentication: Multi-user mode uses `Authorization: Bearer <JWT>`. In single-user mode the `X-API-KEY: <key>` header is required (the `Authorization` header alone is not sufficient).

**Error Responses:**
- `400 Bad Request`: Invalid input or parameters
- `401 Unauthorized`: Missing or invalid API key
- `403 Forbidden`: Rate limit exceeded
- `500 Internal Server Error`: Provider failure

### Batch Embeddings (strings only)

**Endpoint:** `POST /api/v1/embeddings/batch`

**Request:**
```json
{
  "texts": ["First", "Second"],
  "model": "text-embedding-3-small",
  "provider": "openai",
  "dimensions": 512
}
```

**Response:**
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.05, -0.12, ...]],
  "model": "text-embedding-3-small",
  "provider": "openai",
  "count": 2
}
```

Notes:
- This endpoint accepts strings only (`texts: List[str]`). Token arrays are not supported here.

### Health Check

**Endpoint:** `GET /api/v1/embeddings/health`

**Response:**
```json
{
  "status": "healthy",
  "service": "embeddings_v5_production_enhanced",
  "timestamp": "2024-01-01T12:00:00Z",
  "cache_stats": { "size": 123, "max_size": 5000, "ttl_seconds": 3600 },
  "active_requests": 2,
  "circuit_breakers": { "openai": { "state": "closed", "failure_count": 0 } }
}
```

Notes:
- When the embeddings implementation is unavailable (e.g., optional dependencies not installed), the endpoint responds with HTTP 503 and `status: "degraded"`.

### Configuration and Admin Endpoints

#### Providers/Models Configuration
**Endpoint:** `GET /api/v1/embeddings/providers-config`

Returns default provider/model and enabled providers with their models (from simplified embeddings configuration).

#### Clear Cache (Admin Only)
**Endpoint:** `DELETE /api/v1/embeddings/cache`

#### Get Metrics (Admin Only)
**Endpoint:** `GET /api/v1/embeddings/metrics`

#### List Models
**Endpoint:** `GET /api/v1/embeddings/models`

Returns known models with allowlist/default flags.

#### Model Metadata
**Endpoint:** `GET /api/v1/embeddings/models/{model_id}`

Returns provider autodetection, dimension, and max tokens for a model.

#### Reset Circuit Breaker (Admin Only)
**Endpoint:** `POST /api/v1/embeddings/circuit-breakers/{provider}/reset`

#### Model Warmup/Download (Admin Only)
- `POST /api/v1/embeddings/models/warmup`
- `POST /api/v1/embeddings/models/download`

Preload or prepare a model.

### ChromaDB Collection Management

- `POST /api/v1/embeddings/collections` - Create a collection (auto-detects embedding dimension)
- `GET /api/v1/embeddings/collections` - List collections
- `DELETE /api/v1/embeddings/collections/{collection_name}` - Delete a collection
- `GET /api/v1/embeddings/collections/{collection_name}/stats` - Collection size and embedding dimension

---

## Configuration

### Main Configuration File

Edit `Config_Files/config.txt`:

```ini
[Embeddings]
# Provider settings
embedding_provider = openai
embedding_model = text-embedding-3-small
embedding_api_url = http://localhost:8080/v1/embeddings    ; For provider "local_api" (overridable via LOCAL_API_URL)
embedding_api_key = your_api_key_here                      ; For provider "local_api"

# Chunking settings
chunk_size = 400
overlap = 200

# Contextual chunking (optional)
enable_contextual_chunking = false
contextual_llm_model = gpt-4o
contextual_chunk_method = situate_context

# Resource Management Settings
max_models_in_memory = 3
max_model_memory_gb = 8
model_lru_ttl_seconds = 3600
```

### Provider Configuration

#### OpenAI
```python
{
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key": "sk-...",
    "dimensions": 1536,
    "max_batch_size": 2048
}
```

#### HuggingFace
```python
{
    "provider": "huggingface",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "cache_dir": "./models/huggingface_cache",
    "device": "cuda",  # or "cpu"
    "max_length": 512
}
```

#### Local API
```python
{
    "provider": "local_api",
    "api_url": "http://localhost:8080/v1/embeddings",
    "api_key": "optional_key",
    "timeout": 30
}
```

### Environment Variables

```bash
# Optional overrides
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."
export EMBEDDING_CACHE_SIZE="10000"
export EMBEDDING_CACHE_TTL="3600"

# Policy and fallback
export EMBEDDINGS_ENFORCE_POLICY=true            # Enforce provider/model allowlists in production
export EMBEDDINGS_ENFORCE_POLICY_STRICT=false    # If true, admin bypass disabled
export EMBEDDINGS_DIMENSION_POLICY=reduce        # reduce|pad|ignore for non-native dimension requests
export EMBEDDINGS_FALLBACK_CHAIN='{"openai":["huggingface","onnx","local_api"]}'
export LOCAL_API_URL="http://localhost:8080/v1/embeddings"  # Overrides local_api provider URL
```

---

## Security Features

### Input Validation

All user inputs are validated to prevent security vulnerabilities:

```python
# User ID validation
- Alphanumeric + underscore/hyphen only
- Maximum 255 characters
- No path traversal patterns

# Model ID validation
- Prevents injection attacks
- Validates against known models
- Sanitizes special characters
```

### Path Traversal Protection

```python
# Secure path construction
user_path = validate_user_id(user_id)
base_path = Path(base_dir).resolve()
final_path = (base_path / user_path).resolve()

# Verify path is within base directory
final_path.relative_to(base_path)  # Raises if outside
```

### Audit Logging

Security events are logged to `logs/embeddings_audit.jsonl`:

```json
{
  "timestamp": "2024-01-20T10:30:00Z",
  "event_type": "path_traversal_attempt",
  "user_id": "user_123",
  "severity": "WARNING",
  "details": {
    "attempted_value": "../../../etc/passwd"
  }
}
```

### Rate Limiting

Rate limiting is disabled by default. When enabled via RG policies, the create-embeddings endpoint applies limits defined in `resource_governor_policies.yaml`.

---

## Provider Integration

### Adding a New Provider (engine path)

The current engine implements OpenAI, HuggingFace, ONNX, and Local API. Additional providers (Cohere, Google, Mistral, Voyage) are scaffolded in configuration but not yet wired end-to-end in the engine.

To add a new provider end-to-end:

1. Extend the model config types in `Embeddings_Create.py` (add a new `BaseModelCfg` subclass if needed).
2. Add a provider branch in `create_embeddings_batch(...)` to call the new backend and return `List[List[float]]`.
3. Update `build_provider_config(...)` in `embeddings_v5_production_enhanced.py` to construct the provider-specific config.
4. Optionally update provider detection in `guess_provider_for_model(...)` and model mapping in `map_model_for_provider(...)`.
5. Wire any keys via `Config_Files/config.txt` (merged under `EMBEDDING_CONFIG`).

Example config:
```ini
[Embeddings]
embedding_provider = new_provider
new_provider_api_key = xxx
new_provider_model = model-name
```

### Provider Fallback & Headers

Configure automatic fallback when primary provider fails via `EMBEDDINGS_FALLBACK_CHAIN` (settings/env). Defaults:

- `openai` → `huggingface` → `onnx` → `local_api`
- `huggingface` → `onnx` → `local_api`
- `onnx` → `huggingface` → `local_api`
- `local_api` → `huggingface`

Response headers for observability:
- `X-Embeddings-Provider`: the provider that actually served the request
- `X-Embeddings-Fallback-From`: original provider when a fallback occurred
- `X-Embeddings-Dimensions-Policy`: `reduce|pad|ignore` when dimension post-processing is applied

---

## Performance & Scaling

### Caching Strategy

**TTL-based LRU Cache (responses):**
- Default TTL: 1 hour
- Max size: 5,000 entries
- Cache key: SHA256 of `text|provider|model[|dimensions]`

```python
# Cache hit example
Input: "Hello world" + "text-embedding-3-small"
Key: "a2f3b8c9..."
Result: [0.123, -0.456, ...] (from cache)
```

### Resource Management (models)

**Model Memory Limits:**
- Maximum models in memory: 3 (configurable)
- Maximum total memory: 8GB (configurable)
- LRU eviction when limits exceeded
- Automatic cleanup after TTL (1 hour)

**Model Loading Strategy (engine path):**
```python
1. Check if model in memory → Use it
2. Check if at capacity → Evict LRU model
3. Check memory limit → Evict until under limit
4. Load new model
5. Track usage for future eviction
```

### Batch Processing

Optimize for throughput with batching:

```python
# Automatic batching (endpoint)
texts = ["text1", "text2", ..., "text100"]
# Processed in batches of 100
```

### Connection Pooling

Reuse connections for better performance:

```python
# HTTP connection pooling
max_connections = 20
```

---

## Monitoring & Observability

### Prometheus Metrics

Available at `/metrics` (Prometheus scrape endpoint) and JSON summary at `/api/v1/metrics`:

```
# Request rate
embedding_requests_total{provider="openai",model="text-embedding-3-small",status="success"} 12345

# Cache performance
embedding_cache_hits_total{provider="openai",model="text-embedding-3-small"} 6789
embedding_cache_size 5678

# In-flight
active_embedding_requests 2

# Latency (histogram)
embedding_request_duration_seconds_bucket{provider="openai",model="text-embedding-3-small",le="0.1"} 100

# Fallback and policy metrics
embedding_provider_failures_total{provider="openai",model="text-embedding-3-small",reason="http_503"} 2
embedding_fallbacks_total{from_provider="openai",to_provider="huggingface"} 1
embedding_policy_denied_total{provider="huggingface",model="some-model",policy_type="model"} 3
embedding_dimension_adjustments_total{provider="huggingface",model="sentence-transformers/all-MiniLM-L6-v2",method="reduce"} 5
embedding_token_inputs_total{mode="batch"} 42
```

### Logging

Structured logging with Loguru:

```python
# Log levels
DEBUG: Detailed diagnostic information
INFO: General operational messages
WARNING: Warning messages (degraded performance)
ERROR: Error conditions (recoverable)
CRITICAL: Critical failures (non-recoverable)

# Log format
2024-01-20 10:30:00.123 | INFO | embeddings.create | Created embedding for user_123
```

### Health Monitoring

Use `GET /api/v1/embeddings/health` for service status (includes cache stats and circuit breaker states).

---

## Troubleshooting

### Common Issues

#### 1. Model Loading Failures
```
ERROR: Failed to load model "text-embedding-3-small"
SOLUTION:
- Check API keys in config.txt
- Verify network connectivity
- Check disk space for model downloads
```

#### 2. Rate Limit Exceeded
```
ERROR: 429 Rate limit exceeded for user_123
SOLUTION:
- Implement request batching
- Upgrade user tier
- Add retry with exponential backoff
```

#### 3. Memory Issues
```
ERROR: Cannot load model - memory limit exceeded
SOLUTION:
- Reduce max_models_in_memory
- Increase max_model_memory_gb
- Use smaller models
```

#### 4. Cache Misses
```
WARNING: Cache hit rate below 50%
SOLUTION:
- Increase cache size
- Extend TTL for stable content
- Review cache key generation
```

### Debug Mode

Enable debug logging:

```python
# In config.txt
[Logging]
log_level = DEBUG

# Or via environment
export LOG_LEVEL=DEBUG
```

### Performance Profiling

```python
# Enable profiling
python -m cProfile -o embeddings.prof app.py

# Analyze results
python -m pstats embeddings.prof
```

---

## Development Guide

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/tldw_server
cd tldw_server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/Embeddings/ -v
```

### Running Tests

```bash
# Unit tests
pytest tests/Embeddings/test_embeddings_v5_unit.py

# Integration tests
pytest tests/Embeddings/test_embeddings_v5_integration.py

# Performance tests
pytest tests/Embeddings/test_embeddings_v5_production.py::TestLoadTesting

# With coverage
pytest --cov=app.core.Embeddings --cov-report=html
```

### Code Style

Follow PEP 8 with these additions:
- Type hints for all functions
- Docstrings for all public methods
- Maximum line length: 120 characters

```python
def create_embedding(
    text: str,
    model: str = "text-embedding-3-small",
    user_id: Optional[str] = None
) -> List[float]:
    """
    Create an embedding for the given text.

    Args:
        text: Input text to embed
        model: Model identifier
        user_id: Optional user ID for tracking

    Returns:
        List of embedding values

    Raises:
        ValueError: If text is empty or invalid
        RateLimitError: If user exceeds rate limit
    """
    pass
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests and linting
6. Submit a pull request

### API Client Examples

#### Python
```python
import json
from urllib.request import Request, urlopen

payload = {
    "input": "Hello world",
    "model": "text-embedding-3-small",
}
req = Request(
    "http://localhost:8000/api/v1/embeddings",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY",
    },
    method="POST",
)
with urlopen(req) as resp:
    embedding = json.loads(resp.read().decode("utf-8"))["data"][0]["embedding"]
```

#### JavaScript
```javascript
const response = await fetch('http://localhost:8000/api/v1/embeddings', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({
        input: 'Hello world',
        model: 'text-embedding-3-small'
    })
});

const data = await response.json();
const embedding = data.data[0].embedding;
```

#### cURL
```bash
curl -X POST http://localhost:8000/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "input": "Hello world",
    "model": "text-embedding-3-small"
  }'
```

---

## Appendix

### Models and Status

| Provider | Model | Dimensions | Max Tokens | Status |
|----------|-------|------------|------------|--------|
| OpenAI | text-embedding-3-small | 1536 | 8192 | Integrated |
| OpenAI | text-embedding-3-large | 3072 | 8192 | Integrated |
| HuggingFace | sentence-transformers/all-MiniLM-L6-v2 | 384 | 512 | Integrated |
| HuggingFace | sentence-transformers/all-mpnet-base-v2 | 768 | 512 | Integrated |
| ONNX | mirrors of HF models | varies | 512 | Engine integrated |
| Local API | custom | varies | varies | Engine integrated |
| Cohere | embed-english-v3.0 | 1024 | 512 | Planned (scaffolded) |

### Performance Notes

- Throughput and latency depend on the chosen provider and model, batching, and hardware. For best performance, use batching, enable caching, and select lighter models when possible.

### Security Checklist

- [ ] API keys stored securely (environment variables or secrets manager)
- [ ] Input validation enabled
- [ ] Rate limiting configured
- [ ] Audit logging enabled
- [ ] HTTPS/TLS for API endpoints
- [ ] Regular security updates
- [ ] Backup and recovery procedures
- [ ] Incident response plan

### Support

- **GitHub Issues**: https://github.com/cpacker/tldw_server/issues
- **Documentation**: See this MkDocs site (Edit link in page footer)

---

*Last Updated: October 2025*
*Version: 1.0.0*
