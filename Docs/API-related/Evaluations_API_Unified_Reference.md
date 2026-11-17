# tldw_server Unified Evaluations API Reference

## Overview

The tldw_server Evaluations API provides a comprehensive, OpenAI-compatible evaluation framework for assessing AI-generated content quality. This unified API combines industry-standard evaluation patterns with tldw-specific features for advanced content assessment.

**Version**: 1.0.0 (Unified)
**Base URL**: `/api/v1/evaluations`
**Standards**: OpenAI Evals compatible with extensions

Authentication
- Single-user: `X-API-KEY: <key>`
- Multi-user: `Authorization: Bearer <JWT>`
Rate limiting is enforced on core run endpoints (geval, rag, response-quality, propositions, batch).

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│  /api/v1/evaluations/* (Unified OpenAI-compatible + tldw)    │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Evaluation   │  │   Dataset    │  │     Run      │      │
│  │  Manager     │  │   Manager    │  │   Manager    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Metrics    │  │   Webhooks   │  │Rate Limiting │      │
│  │   Service    │  │   Manager    │  │   Service    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Engines                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   G-Eval     │  │     RAG      │  │   Response   │      │
│  │   Engine     │  │  Evaluator   │  │   Quality    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Model-Graded  │  │Exact Match   │  │   Custom     │      │
│  │  Evaluator   │  │  Evaluator   │  │  Evaluator   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │         Unified Evaluations Database               │     │
│  │  - evaluations table (definitions)                 │     │
│  │  - runs table (execution records)                  │     │
│  │  - datasets table (test data)                      │     │
│  │  - results table (evaluation outputs)              │     │
│  │  - metrics table (performance data)                │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Authentication

The API supports multiple authentication modes:

### Single-User Mode (Default)
```bash
# Using API Key header
curl -H "X-API-KEY: your-api-key" https://api.example.com/api/v1/evaluations

# Using Bearer token
curl -H "Authorization: Bearer your-api-key" https://api.example.com/api/v1/evaluations
```

### Multi-User Mode (JWT)
```bash
# Using JWT token
curl -H "Authorization: Bearer eyJhbGc..." https://api.example.com/api/v1/evaluations
```

### OpenAI Compatibility
```bash
# Using OpenAI-style key
curl -H "Authorization: Bearer sk-..." https://api.example.com/api/v1/evaluations
```

## Rate Limiting

### Global Limits
- **Standard Operations**: 60 requests/minute
- **Evaluation Runs**: 10 requests/minute
- **Batch Operations**: 5 requests/minute
- **Burst Protection**: 10 requests/second max

### User-Based Limits (Multi-User Mode)
| Tier       | Requests/Min | Tokens/Min | Batch Size |
|------------|-------------|------------|------------|
| Free       | 10          | 10,000     | 10         |
| Basic      | 30          | 50,000     | 50         |
| Premium    | 100         | 200,000    | 200        |
| Enterprise | Unlimited   | Unlimited  | 1000       |

### Rate Limit Response Headers
Unified evaluation endpoints include standard response headers that surface current limits and remaining allowances:

- X-RateLimit-Tier
- X-RateLimit-PerMinute-Limit
- X-RateLimit-PerMinute-Remaining
- X-RateLimit-Daily-Limit
- X-RateLimit-Daily-Remaining
- X-RateLimit-Tokens-Remaining

In addition, draft standard headers are provided for compatibility with proxies and SDKs:

- RateLimit-Limit
- RateLimit-Remaining
- RateLimit-Reset (seconds until the current minute window resets)

Note: Remaining values are based on the latest limiter decision; actual token usage may adjust after the request completes when providers return precise usage metadata.

## Core API Endpoints

### Evaluation Management

#### Create Evaluation
`POST /api/v1/evaluations`

Creates a new evaluation definition that can be run multiple times.

**Request:**
```json
{
  "name": "summary_quality_eval",
  "description": "Evaluate summary quality using multiple metrics",
  "eval_type": "model_graded",
  "eval_spec": {
    "metrics": ["fluency", "consistency", "relevance", "coherence"],
    "thresholds": {
      "pass": 0.7,
      "excellent": 0.9
    },
    "model": "gpt-4",
    "temperature": 0.3
  },
  "dataset_id": "dataset_123",
  "metadata": {
    "project": "content_quality",
    "version": "1.0"
  }
}
```

**Response (201 Created):**
```json
{
  "id": "eval_abc123",
  "name": "summary_quality_eval",
  "description": "Evaluate summary quality using multiple metrics",
  "eval_type": "model_graded",
  "eval_spec": {...},
  "dataset_id": "dataset_123",
  "created_at": 1234567890,
  "created_by": "user_123",
  "metadata": {...}
}
```

#### List Evaluations
`GET /api/v1/evaluations`

**Query Parameters:**
- `limit`: 1-100 (default: 20)
- `after`: Cursor for pagination
- `eval_type`: Filter by type
- `project`: Filter by project metadata

**Response:**
```json
{
  "object": "list",
  "data": [...],
  "has_more": true,
  "first_id": "eval_abc123",
  "last_id": "eval_xyz789"
}
```

#### Get Evaluation
`GET /api/v1/evaluations/{eval_id}`

#### Update Evaluation
`PATCH /api/v1/evaluations/{eval_id}`

#### Delete Evaluation
`DELETE /api/v1/evaluations/{eval_id}`

### Evaluation Runs

#### Create Run
`POST /api/v1/evaluations/{eval_id}/runs`

Starts an asynchronous evaluation run.

**Request:**
```json
{
  "target_model": "gpt-4o",
  "dataset_override": null,
  "config": {
    "temperature": 0.7,
    "max_workers": 4,
    "timeout_seconds": 300
  },
  "webhook_url": "https://example.com/webhook"
}
```

**Response (202 Accepted):**
```json
{
  "id": "run_def456",
  "eval_id": "eval_abc123",
  "status": "pending",
  "target_model": "gpt-4o",
  "created_at": 1234567890,
  "progress": {
    "completed_samples": 0,
    "total_samples": 100
  }
}
```

#### Get Run Status
`GET /api/v1/evaluations/runs/{run_id}`

Results are included in the run object when `status` becomes `completed`. There is no separate results endpoint.

#### Stream Run Progress (SSE)
Not available on the unified router. Poll `GET /api/v1/evaluations/runs/{run_id}` for status.

#### Cancel Run
`POST /api/v1/evaluations/runs/{run_id}/cancel`

### Dataset Management

#### Create Dataset
`POST /api/v1/evaluations/datasets`

**Request:**
```json
{
  "name": "qa_test_set",
  "description": "Question-answer pairs for testing",
  "samples": [
    {
      "input": "What is the capital of France?",
      "expected": "Paris",
      "metadata": {"difficulty": "easy"}
    }
  ]
}
```

#### List Datasets
`GET /api/v1/evaluations/datasets`

#### Get Dataset
`GET /api/v1/evaluations/datasets/{dataset_id}`

#### Delete Dataset
`DELETE /api/v1/evaluations/datasets/{dataset_id}`

## Specialized Evaluation Endpoints

### RAG Pipeline Presets
`POST /api/v1/evaluations/rag/pipeline/presets`
Create a named RAG pipeline preset.

Request:
```json
{ "name": "baseline_hybrid", "config": { "chunking": {"method": "sentences", "size": 8, "overlap": 1}, "retriever": {"mode": "hybrid", "k": 8, "alpha": 0.5}, "reranker": {"provider": "cohere", "model": "rerank-3"}, "rag": {"max_context_tokens": 2000} } }
```

Responses:
- `201` PipelinePresetResponse `{ "name": "...", "config": {..}, "created_at": 123, "updated_at": 123 }`

`GET /api/v1/evaluations/rag/pipeline/presets`
List presets. Response `{ "items": [ {"name": "...", "config": {...}} ], "total": 1 }`

`GET /api/v1/evaluations/rag/pipeline/presets/{name}`
Get a preset by name.

`DELETE /api/v1/evaluations/rag/pipeline/presets/{name}`
Delete a preset. Response `204 No Content`.

`POST /api/v1/evaluations/rag/pipeline/cleanup`
Cleanup ephemeral vector store collections. Response `{ "expired_count": 0, "deleted_count": 0, "errors": [] }`

### Embeddings A/B Tests
`POST /api/v1/evaluations/embeddings/abtest`
Create an embeddings A/B test. Request `{ "name": "string", "config": { "arms": [...], "media_ids": [], "chunking": {...}, "retrieval": {...}, "queries": [...] }, "run_immediately": false }`. Response `{ "test_id": "...", "status": "created" }`.

`POST /api/v1/evaluations/embeddings/abtest/{test_id}/run`
Start execution. Response `{ "test_id": "...", "status": "running", "progress": { } }`.

`GET /api/v1/evaluations/embeddings/abtest/{test_id}`
Summary: `{ "test_id": "...", "status": "...", "arms": [ {"arm_id":"...","provider":"...","model":"...","metrics": {"ndcg": 0.72}, "latency_ms": {"p50": 30.3} } ] }`.

`GET /api/v1/evaluations/embeddings/abtest/{test_id}/results`
Paginated results. Response `{ "summary": {...}, "results": [ { "result_id": "...", "test_id": "...", "arm_id": "...", "query_id": "...", "ranked_ids": ["..."], "scores": [0.9], "metrics": {"ndcg": 0.72}, "latency_ms": 12.3, "ranked_distances": [0.1], "ranked_metadatas": [{"source": "..."}], "ranked_documents": ["..."], "rerank_scores": [0.7], "created_at": "2026-01-12T00:00:00Z" } ], "page": 1, "page_size": 50, "total": 120 }`.

`GET /api/v1/evaluations/embeddings/abtest/{test_id}/significance?metric=ndcg`
Statistical significance for chosen metric.

`GET /api/v1/evaluations/embeddings/abtest/{test_id}/events`
SSE event stream of progress/updates.

`GET /api/v1/evaluations/embeddings/abtest/{test_id}/export?format=json|csv`
Export results (admin-only).

`DELETE /api/v1/evaluations/embeddings/abtest/{test_id}`
Delete a test.

### G-Eval Summarization
`POST /api/v1/evaluations/geval`

Evaluates summary quality using Google's G-Eval framework.

**Request:**
```json
{
  "source_text": "Original long document...",
  "summary": "Concise summary...",
  "metrics": ["fluency", "consistency", "relevance", "coherence"],
  "api_name": "openai",
  "save_results": true
}
```

**Response:**
```json
{
  "metrics": {
    "fluency": {
      "score": 0.85,
      "raw_score": 2.55,
      "explanation": "Well-structured with minor grammatical issues"
    },
    "consistency": {
      "score": 0.92,
      "raw_score": 4.6,
      "explanation": "Highly consistent with source material"
    }
  },
  "average_score": 0.88,
  "summary_assessment": "High-quality summary with excellent factual accuracy",
  "evaluation_time": 2.34,
  "metadata": {
    "evaluation_id": "eval_result_789"
  }
}
```

### RAG Evaluation
`POST /api/v1/evaluations/rag`

Evaluates retrieval-augmented generation quality.

**Request:**
```json
{
  "query": "What are the benefits of exercise?",
  "retrieved_contexts": [
    "Exercise improves cardiovascular health...",
    "Regular physical activity boosts mood..."
  ],
  "generated_response": "Exercise provides numerous benefits including...",
  "ground_truth": "Expected answer for comparison",
  "metrics": ["relevance", "faithfulness", "answer_similarity", "context_precision", "claim_faithfulness"]
}
```

**Response:**
```json
{
  "metrics": {
    "relevance": {"score": 0.89, "explanation": "Highly relevant to query"},
    "faithfulness": {"score": 0.95, "explanation": "Well-grounded in contexts"},
    "answer_similarity": {"score": 0.82, "explanation": "Close to ground truth"},
    "context_precision": {"score": 0.78, "explanation": "Good context selection"},
    "claim_faithfulness": {"score": 0.90, "explanation": "Most extracted claims are supported by contexts"}
  },
  "overall_score": 0.86,
  "retrieval_quality": 0.78,
  "generation_quality": 0.89,
  "suggestions": [
    "Consider adding more diverse contexts",
    "Response could be more concise"
  ]
}
```

### Response Quality
`POST /api/v1/evaluations/response-quality`

Evaluates general response quality and format compliance.

**Request:**
```json
{
  "prompt": "Write a professional email...",
  "response": "Dear colleague...",
  "expected_format": "email",
  "evaluation_criteria": {
    "professionalism": "Appropriate tone and language",
    "completeness": "All required elements present",
    "clarity": "Clear and unambiguous"
  }
}
```

### Batch Evaluation
`POST /api/v1/evaluations/batch`

Process multiple evaluations in parallel. Supported `evaluation_type` values: `geval`, `rag`, `response_quality`, `ocr`, `propositions`.

**Request:**
```json
{
  "evaluation_type": "geval",
  "items": [...],
  "parallel_workers": 4,
  "continue_on_error": true
}
```

Example (curl):
```bash
curl -X POST "http://localhost:8000/api/v1/evaluations/batch" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
        "evaluation_type": "geval",
        "items": [
          {
            "source_text": "The mitochondrion is the powerhouse of the cell.",
            "summary": "Mitochondria produce energy in cells.",
            "metrics": ["coherence", "consistency"]
          },
          {
            "source_text": "Deep learning uses neural networks to model complex patterns.",
            "summary": "Neural networks model complex patterns in deep learning.",
            "metrics": ["coherence"]
          }
        ],
        "parallel_workers": 2,
        "continue_on_error": true
      }'
```

Example response:
```json
{
  "total_items": 2,
  "successful": 2,
  "failed": 0,
  "results": [
    {
      "evaluation_id": "eval_01HXXXX",
      "status": "completed",
      "results": {
        "metrics": {
          "coherence": {"score": 0.94, "explanation": "Strong logical flow"},
          "consistency": {"score": 0.91, "explanation": "Consistent details"}
        },
        "average_score": 0.925
      }
    },
    {
      "evaluation_id": "eval_01HYYYY",
      "status": "completed",
      "results": {
        "metrics": {
          "coherence": {"score": 0.89, "explanation": "Minor clarity issues"}
        },
        "average_score": 0.89
      }
    }
  ],
  "aggregate_metrics": {"coherence": 0.915},
  "processing_time": 1.82
}
```

Additional examples:

Propositions (Jaccard):
```bash
curl -X POST "http://localhost:8000/api/v1/evaluations/batch" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
        "evaluation_type": "propositions",
        "items": [
          {
            "extracted": ["Alice founded Acme in 2020", "Bob joined in 2021"],
            "reference": ["Alice founded Acme in 2020"],
            "method": "jaccard",
            "threshold": 0.5
          }
        ],
        "parallel_workers": 1
      }'
```

OCR (text-based items):
```bash
curl -X POST "http://localhost:8000/api/v1/evaluations/batch" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
        "evaluation_type": "ocr",
        "items": [
          {
            "items": [
              {"id": "d1", "extracted_text": "hello world", "ground_truth_text": "hello world"}
            ],
            "metrics": ["cer", "wer"]
          }
        ],
        "parallel_workers": 1
      }'
```

### Propositions Evaluation
`POST /api/v1/evaluations/propositions`

Evaluates proposition extraction quality with precision/recall/F1, density, and length metrics.

Request (example):
```json
{
  "extracted": ["Claim A", "Claim B"],
  "reference": ["Claim A", "Claim C"],
  "method": "semantic",
  "threshold": 0.7
}
```

Example (curl):
```bash
curl -X POST "http://localhost:8000/api/v1/evaluations/propositions" \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
        "extracted": ["Mitochondria produce ATP", "Cells contain nuclei"],
        "reference": ["Mitochondria produce energy", "Cells contain nuclei"],
        "method": "semantic",
        "threshold": 0.7
      }'
```

Example response:
```json
{
  "precision": 0.50,
  "recall": 1.00,
  "f1": 0.67,
  "matched": 1,
  "total_extracted": 2,
  "total_reference": 2,
  "claim_density_per_100_tokens": 2.3,
  "avg_prop_len_tokens": 7.8,
  "dedup_rate": 0.0,
  "details": {
    "matches": [
      {"extracted": "Cells contain nuclei", "reference": "Cells contain nuclei", "score": 1.0}
    ],
    "misses": [
      {"extracted": "Mitochondria produce ATP", "closest": "Mitochondria produce energy", "score": 0.65}
    ]
  },
  "metadata": {"evaluation_id": "eval_01HZZZZ", "evaluation_time": 0.21}
}
```

### OCR Evaluation
`POST /api/v1/evaluations/ocr` - Evaluate OCR text quality on provided content

`POST /api/v1/evaluations/ocr-pdf` - Evaluate OCR text quality on uploaded PDF

## Webhook Management

### Register Webhook
`POST /api/v1/evaluations/webhooks`

```json
{
  "url": "https://example.com/webhook",
  "events": ["evaluation.completed", "evaluation.failed"],
  "secret": "webhook_secret_key"
}
```

### List Webhooks
`GET /api/v1/evaluations/webhooks`

Returns all registered webhooks for the current user.

### Unregister Webhook
`DELETE /api/v1/evaluations/webhooks?url=...`

Removes the specified webhook URL.

### Test Webhook
`POST /api/v1/evaluations/webhooks/test`

Sends a test event to the provided URL and returns delivery stats.

### Webhook Events

Events are sent as POST requests with HMAC-SHA256 signatures.

**Headers:**
- `X-Webhook-Signature`: HMAC-SHA256 signature
- `X-Webhook-Timestamp`: Unix timestamp
- `X-Webhook-Event`: Event type

**Payload Example:**
```json
{
  "event": "evaluation.completed",
  "timestamp": 1234567890,
  "data": {
    "evaluation_id": "eval_123",
    "run_id": "run_456",
    "results": {...}
  }
}
```

## Metrics & Monitoring

### Health Check
`GET /api/v1/evaluations/health`

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "database": "connected",
  "rate_limit": {
    "requests_remaining": 50,
    "reset_at": 1234567890
  }
}
```

### Prometheus Metrics
`GET /api/v1/evaluations/metrics`

Exports metrics in Prometheus format:
- `evaluation_requests_total`
- `evaluation_duration_seconds`
- `evaluation_errors_total`
- `evaluation_queue_depth`
- `evaluation_cost_dollars`

### Rate Limit Status
`GET /api/v1/evaluations/rate-limits`

Returns current tier, limits, usage, remaining allowance, and reset time.

Responses from evaluation endpoints also include standard `X-RateLimit-*` headers:
- `X-RateLimit-Tier`
- `X-RateLimit-PerMinute-Limit`
- `X-RateLimit-Daily-Limit`
- `X-RateLimit-Daily-Remaining`
- `X-RateLimit-Tokens-Remaining`

### Idempotency

For create endpoints, supply `Idempotency-Key` to make requests safe to retry. When a prior successful request with the same key exists (scoped per user and entity type), the API returns the original resource instead of creating a duplicate.

- `POST /api/v1/evaluations` - create evaluation definition
- `POST /api/v1/evaluations/datasets` - create dataset
- `POST /api/v1/evaluations/{eval_id}/runs` - create run
- `POST /api/v1/evaluations/embeddings/abtest` - create embeddings A/B test (scaffold)
- `POST /api/v1/evaluations/embeddings/abtest/{test_id}/run` - start A/B test (admin-gated)

Example:
```
Idempotency-Key: 9c20c0b8-5e5b-42d1-ae6a-6b1ae1a4f3de
```

Keys are stored server-side and are unique per `{user_id, entity_type, key}`.

### Admin Gating

Some heavy operations (e.g., embeddings A/B test run and export) are admin-gated by default. Control this behavior with the environment variable:

- `EVALS_HEAVY_ADMIN_ONLY=true|false` (default: `true`)

When enabled, non-admin users receive `403` for gated endpoints.

## Error Handling

All errors follow a consistent format:

```json
{
  "error": {
    "message": "Detailed error description",
    "type": "error_category",
    "param": "field_name",
    "code": "ERROR_CODE"
  }
}
```

### Error Types
- `authentication_error`: Auth failures
- `invalid_request_error`: Validation errors
- `rate_limit_error`: Rate limit exceeded
- `not_found_error`: Resource not found
- `server_error`: Internal errors

### HTTP Status Codes
- `200`: Success
- `201`: Created
- `202`: Accepted (async operation)
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error

## Migration Guide

### From Legacy Endpoints

#### Old tldw endpoints → New unified endpoints
- `/api/v1/evaluations/geval` → `/api/v1/evaluations/geval` (unchanged)
- `/api/v1/evaluations/rag` → `/api/v1/evaluations/rag` (unchanged)
- `/api/v1/evaluations/batch` → `/api/v1/evaluations/batch` (unchanged)

#### Old OpenAI endpoints → New unified endpoints
- `/api/v1/evals` → `/api/v1/evaluations`
- `/api/v1/evals/{id}/runs` → `/api/v1/evaluations/{id}/runs`
- `/api/v1/runs/{id}` → `/api/v1/evaluations/runs/{id}`

Note: Current server builds expose the unified routes. Legacy aliases may exist in older deployments only.

### Breaking Changes
1. Webhook event names standardized (see Webhook Events section)
2. Rate limit headers now use `X-RateLimit-*` prefix
3. Dataset samples format standardized to OpenAI format

### Deprecation Timeline
- **Current**: Unified endpoints are the supported API surface.
- **Legacy deployments**: May still expose alias routes (`/evals`, `/runs`) behind compatibility shims.
- **Future**: Alias routes, where present, are subject to removal without feature updates.

## Examples

### Python (requests)
```python
import json
from urllib.request import Request, urlopen

API_KEY = "your-key"
BASE = "http://localhost:8000/api/v1/evaluations"
headers = {"X-API-KEY": API_KEY}

def request_json(method, url, payload=None, headers=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = Request(url, data=data, headers=hdrs, method=method)
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

# Create evaluation
payload = {
  "name": "my_eval",
  "eval_type": "model_graded",
  "eval_spec": {"metrics": ["fluency"], "model": "gpt-4"},
  "dataset": [{"input": {"text": "hi"}, "expected": {"text": "hi"}}]
}
e = request_json("POST", BASE, payload=payload, headers=headers)

# Start a run
r = request_json(
    "POST",
    f"{BASE}/{e['id']}/runs",
    payload={"target_model": "gpt-4"},
    headers=headers,
)

# Poll run status
status = request_json("GET", f"{BASE}/runs/{r['id']}", headers=headers)
```

### Pipeline Presets & Cleanup

Create or update a pipeline preset:
```bash
curl -X POST "$BASE/rag/pipeline/presets" \
  -H "X-API-KEY: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"standard","config": {"retrieval": {"k": 8}}}'
```

List presets:
```bash
curl "$BASE/rag/pipeline/presets" -H "X-API-KEY: $API_KEY"
```

Cleanup expired ephemeral collections:
```bash
curl -X POST "$BASE/rag/pipeline/cleanup" -H "X-API-KEY: $API_KEY"
```

### Embeddings A/B Test (SSE)

Stream events for a running A/B test:
```bash
curl "$BASE/embeddings/abtest/abtest_123/events" -H "X-API-KEY: $API_KEY"
```

## Best Practices

### Evaluation Design
1. **Choose appropriate metrics** for your use case
2. **Use consistent datasets** for comparable results
3. **Set reasonable thresholds** based on baseline testing
4. **Version your evaluations** for reproducibility

### Performance Optimization
1. **Batch similar evaluations** to reduce overhead
2. **Use parallel workers** for large datasets
3. **Set appropriate timeouts** to prevent hanging
4. **Cache evaluation results** when possible

### Cost Management
1. **Monitor token usage** via metrics endpoint
2. **Use smaller models** for initial testing
3. **Implement sampling** for large datasets
4. **Set spending limits** with `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd` (see `Docs/Operations/Usage_Quotas.md`)

### Security
1. **Rotate API keys** regularly
2. **Use webhook secrets** for verification
3. **Implement IP allowlisting** for production
4. **Audit evaluation access** via logs

## Support

### Resources
- GitHub Issues: https://github.com/tldw/tldw_server/issues
- Documentation: https://docs.tldw.ai/evaluations
- Discord Community: https://discord.gg/tldw

### Feature Requests
Submit feature requests via GitHub issues with the `enhancement` label.

### Contributing
See CONTRIBUTING.md for guidelines on contributing to the evaluation module.

---

*Last Updated: 2024*
*Version: 1.0.0*
*Status: Unified Implementation*
## History

### Evaluation History
`POST /api/v1/evaluations/history`

Retrieve evaluation history for a user with optional filters.

Request:
```json
{
  "user_id": "optional-user-id",
  "evaluation_type": "rag|geval|response_quality|...",
  "start_date": "2025-01-01T00:00:00Z",
  "end_date": "2025-01-31T23:59:59Z",
  "limit": 20,
  "offset": 0
}
```
