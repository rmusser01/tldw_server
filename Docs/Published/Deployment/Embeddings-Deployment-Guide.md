# Embeddings Service Deployment Guide

Version: 1.0
Last Updated: 2025-10-08
Service Version: v5 Enhanced (with Circuit Breaker)

## Table of Contents
1. Architecture Overview
2. Deployment Decision Tree
3. Single-User Deployment
4. Enterprise Deployment
5. Configuration Reference
6. Monitoring & Observability
7. Troubleshooting
8. Support

---

## Architecture Overview

The embeddings service is implemented as part of the FastAPI application and supports two deployment topologies:

### Single-User Architecture (<5 concurrent users)
- Implementation: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Processing: Inline HTTP handling; no external queue required
- Infra: Minimal (no Redis/queues required)
- Best for: Personal use, small teams, development

### Enterprise Architecture (≥5 concurrent users)
- Implementation: Redis Streams worker (`core/Embeddings/services/redis_worker.py`)
- Processing: Queue-based Redis pipeline (chunking → embedding → storage + content) with Jobs as the root status/billing record
- Infra: Redis + Jobs DB (SQLite/Postgres) and worker processes
- Best for: Production, multi-tenant, high volume

Note: The “mode” depends on which processes you run (API only vs API + Redis worker). There is no runtime flag that switches the API behavior; `EMBEDDINGS_MODE` is a deployment convention, not an API setting.

---

## Deployment Decision Tree

```
┌─────────────────────────┐
│ How many concurrent     │
│ users will you have?    │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    │               │
   <5             ≥5
    │               │
    v               v
┌──────────┐  ┌──────────┐
│ Single-  │  │Enterprise│
│  User    │  │  Mode    │
└──────────┘  └──────────┘
```

---

## Single-User Deployment

### Prerequisites
```bash
# Required packages (via pyproject)
pip install -e .

# Optional (choose what your environment supports)
pip install torch              # GPU acceleration for HuggingFace models
pip install onnxruntime-gpu    # GPU acceleration for ONNX models
```

### Environment Variables
```bash
# Auth mode (pick one)
export AUTH_MODE=single_user
export SINGLE_USER_API_KEY="your-single-user-api-key"   # Used via X-API-KEY header
# OR
# export AUTH_MODE=multi_user
# export JWT_SECRET_KEY="your-32+char-jwt-secret"       # Multi-user JWT mode

# Provider API keys (as needed by your configured providers)
export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."
export VOYAGE_API_KEY="..."

# Optional: user DB base directory (default: Databases/user_databases)
export USER_DB_BASE_DIR="$(pwd)/Databases/user_databases"  # USER_DB_BASE is deprecated alias

# Optional: enable endpoint rate limiting guard on embeddings
export EMBEDDINGS_RATE_LIMIT=on   # Uses built-in limit in the API endpoint
```

`USER_DB_BASE_DIR` is defined in `tldw_Server_API.app.core.config` (defaults to `Databases/user_databases/` under the project root). Override via environment variable or `Config_Files/config.txt` as needed.

### Dockerfile (API)
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install from pyproject (editable)
COPY pyproject.toml ./pyproject.toml
RUN pip install --no-cache-dir -e .

# Copy application
COPY tldw_Server_API/ ./tldw_Server_API/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/v1/embeddings/health || exit 1

# Run API
ENV PYTHONPATH=/app
CMD ["uvicorn", "tldw_Server_API.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose (API only)
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AUTH_MODE=single_user
      - SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./Databases/user_databases:/app/Databases/user_databases
      - ./models/embedding_models_data:/app/models/embedding_models_data
    restart: unless-stopped
```

### systemd Service (API)
```ini
[Unit]
Description=TLDW Embeddings API (Single-User)
After=network.target

[Service]
Type=simple
User=tldw
Group=tldw
WorkingDirectory=/opt/tldw_server
Environment="AUTH_MODE=single_user"
Environment="PYTHONPATH=/opt/tldw_server"
ExecStart=/usr/bin/python3 -m uvicorn tldw_Server_API.app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## Enterprise Deployment

### Prerequisites
```bash
# Required packages (same as single-user)
pip install -e .

# Optional: GPU support for HF models
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Environment and Redis Worker
```bash
export JOBS_DB_URL="sqlite:///Databases/jobs.db"  # optional; set Postgres URL for shared Jobs DB
export REDIS_URL="redis://localhost:6379"
```

Start the Redis Streams worker (stage workers controlled via env vars):

```bash
export EMBEDDINGS_REDIS_WORKERS_CHUNKING=2
export EMBEDDINGS_REDIS_WORKERS_EMBEDDING=2
export EMBEDDINGS_REDIS_WORKERS_STORAGE=1
export EMBEDDINGS_REDIS_WORKERS_CONTENT=1
python -m tldw_Server_API.app.core.Embeddings.services.redis_worker --stage all
```

### Docker Compose (API + Redis Worker)
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - AUTH_MODE=single_user
      - SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./Databases/user_databases:/app/Databases/user_databases
      - ./models/embedding_models_data:/app/models/embedding_models_data

  redis-worker:
    build: .
    command: python -m tldw_Server_API.app.core.Embeddings.services.redis_worker --stage all
    environment:
      - REDIS_URL=redis://redis:6379
      - JOBS_DB_URL=sqlite:///Databases/jobs.db
      - EMBEDDINGS_REDIS_WORKERS_CHUNKING=2
      - EMBEDDINGS_REDIS_WORKERS_EMBEDDING=2
      - EMBEDDINGS_REDIS_WORKERS_STORAGE=1
      - EMBEDDINGS_REDIS_WORKERS_CONTENT=1
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./Databases:/app/Databases
      - ./models/embedding_models_data:/app/models/embedding_models_data
```

### Kubernetes (example)
Deploy API and Redis worker as separate Deployments. Ensure Redis and the Jobs DB (SQLite volume or Postgres service) are reachable. Health probes should use `GET /api/v1/embeddings/health` on the API.

---

## Configuration Reference

### Core Settings
Most runtime behavior is configured via code defaults or YAML, not environment variables. Key toggles:
- `AUTH_MODE` (`single_user` | `multi_user`)
- `SINGLE_USER_API_KEY` (single-user mode)
- `JWT_SECRET_KEY` (multi-user mode)
- Provider keys: `OPENAI_API_KEY`, `COHERE_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `VOYAGE_API_KEY`
- `USER_DB_BASE_DIR` (optional; defined in `tldw_Server_API.app.core.config`, default `Databases/user_databases/` under the project root; override via environment variable or `Config_Files/config.txt`)
- `EMBEDDINGS_RATE_LIMIT=on` enables the built-in rate limiter for the embeddings endpoint
- `ALLOW_ZERO_EMBEDDINGS_MEDIA_TYPES` allows media-embeddings jobs to succeed with zero vectors for types like `audio,video` (also configurable via `Config_Files/config.txt`).
- Embeddings compactor: `COMPACTOR_USER_ID` (required in multi-user mode; defaults to `SINGLE_USER_FIXED_ID` in single-user), `EMBEDDINGS_COMPACTOR_INTERVAL_SECONDS` (default: 1800), optional `MEDIA_DB_PATH` override.

Advanced constants like batch size, cache TTL, and connection pool are code-level defaults in `embeddings_v5_production_enhanced.py`.

### Provider Configuration (model definitions)

For local models and provider defaults used by the embeddings implementation, define an Embeddings config compatible with `EmbeddingConfigSchema` (used by `Embeddings_Create`):

```yaml
default_model_id: hf_mpnet
model_storage_base_dir: ./models/embedding_models_data/
models:
  openai_small:
    provider: openai
    model_name_or_path: text-embedding-3-small
  hf_mpnet:
    provider: huggingface
    model_name_or_path: sentence-transformers/all-mpnet-base-v2
    trust_remote_code: false
  onnx_minilm:
    provider: onnx
    model_name_or_path: sentence-transformers/all-MiniLM-L6-v2
    onnx_providers: ["CPUExecutionProvider"]
  local_api_default:
    provider: local_api
    model_name_or_path: nomic-embed-text
    api_url: http://localhost:11434/api/embeddings
```

---

## Monitoring & Observability

### Prometheus Metrics
Prometheus metrics are exposed via the unified text endpoint:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'tldw'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

Key embeddings metrics (subset):
- `embedding_requests_total` - Request counter
- `embedding_request_duration_seconds` - Latency histogram
- `embedding_cache_hits_total` - Cache hit rate
- `active_embedding_requests` - Current load
- Circuit breaker status is available via admin endpoints

### Health Checks and Admin Metrics

```bash
# Basic health check
curl http://localhost:8000/api/v1/embeddings/health

# Detailed embeddings metrics (requires admin)
curl -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  http://localhost:8000/api/v1/embeddings/metrics

# Circuit breaker status (requires admin)
curl -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  http://localhost:8000/api/v1/embeddings/circuit-breakers
```

### Logging
The service uses Loguru and Prometheus instrumentation. Tune `LOG_LEVEL`, and scrape `/metrics` for Prometheus-formatted metrics.

---

## Troubleshooting

### 1) Circuit Breaker Open
Symptom: 503 Service Unavailable.
Check and reset (admin-only in single-user):
```bash
curl -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  http://localhost:8000/api/v1/embeddings/circuit-breakers

curl -X POST -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  http://localhost:8000/api/v1/embeddings/circuit-breakers/openai/reset
```

### 2) High Memory Usage
Symptom: OOM or slow responses.
Actions:
- Prefer smaller models (e.g., `text-embedding-3-small`, `all-MiniLM-L6-v2`)
- Reduce concurrent load; scale out using additional Redis workers
- For HF models, unload inactive models sooner (see `model_unload_timeout` in YAML)

### 3) Slow Embeddings
Symptom: High latency.
Actions:
- Verify GPU availability: `nvidia-smi`
- Warm up models (admin-only):
```bash
curl -X POST -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  http://localhost:8000/api/v1/embeddings/models/warmup \
  -H 'Content-Type: application/json' \
  -d '{"provider":"huggingface","model_id":"sentence-transformers/all-mpnet-base-v2"}'
```

### 4) Rate Limiting
Symptom: 429 Too Many Requests.
To enable the built-in limiter for the embeddings endpoint:
```bash
export EMBEDDINGS_RATE_LIMIT=on
```

### Debug Mode
```bash
export LOG_LEVEL=DEBUG
```

### Performance Tuning
GPU:
```bash
export CUDA_ALLOW_TF32=1   # Mixed precision where applicable
```
CPU:
```bash
export OMP_NUM_THREADS=8
```

---

## Support

- GitHub Issues: https://github.com/rmusser01/tldw_server/issues
- Documentation:
  - API-related: Embeddings API Documentation
  - Code: Embeddings Documentation
- Metrics: `/metrics` (Prometheus) or `/api/v1/metrics/text`
