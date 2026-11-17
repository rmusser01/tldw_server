# Evaluations API Reference

## Overview

The Evaluations API provides comprehensive capabilities for assessing the quality of AI-generated content. The system supports multiple evaluation types including summarization quality (G-Eval), RAG system evaluation, and various text matching strategies.

**Base URL**: `http://localhost:8000`
**API Version**: `v1`
**API Prefix**: `/api/v1`
**Authentication**: See Authentication section

## Authentication

All API requests require authentication. In single-user mode use `X-API-KEY` or a Bearer token; in multi-user use JWT Bearer tokens.

### Single-User Mode (Development)
```http
X-API-KEY: YOUR_SINGLE_USER_API_KEY
```

or

```http
Authorization: Bearer YOUR_SINGLE_USER_API_KEY
```

### Multi-User Mode (Production)
```http
Authorization: Bearer YOUR_PERSONAL_API_KEY
```

**Configuration**: Authentication mode is set in `tldw_Server_API/Config_Files/config.txt` or via environment variables:
- `AUTH_MODE`: "single_user" or "multi_user"
- `SINGLE_USER_API_KEY`: API key for single-user mode
- `JWT_SECRET_KEY`: Secret for JWT tokens in multi-user mode

## Endpoints

### Evaluations

#### Create Evaluation
`POST /api/v1/evaluations`

Creates a new evaluation definition.

**Request Body:**
```json
{
  "name": "string",           // Required, unique name for the evaluation
  "description": "string",    // Optional description
  "eval_type": "string",      // Required: "model_graded" | "exact_match" | "fuzzy_match" | "includes"
  "eval_spec": {              // Required, evaluation configuration
    // For model_graded evaluations:
    "sub_type": "string",     // Required for model_graded: "summarization" | "rag" | "response_quality"
    "evaluator_model": "string", // A provider ("openai"), a model ("gpt-4") or "provider:model"; a model runs on api_name/provider, else OpenAI
    "metrics": ["string"],    // Metrics to evaluate (type-specific)
    "threshold": 0.7,         // Pass/fail threshold (0-1, default: 0.7)
    "scoring_prompt": "string", // Optional custom scoring prompt
    "custom_criteria": {}     // Optional custom evaluation criteria
  },
  // Must provide EITHER dataset OR dataset_id, not both:
  "dataset": [                // Option 1: Inline dataset
    {
      "input": {},            // Input data for evaluation
      "expected": {}          // Expected output (format varies by eval_type)
    }
  ],
  "dataset_id": "string",     // Option 2: Reference to existing dataset
  "metadata": {               // Optional metadata
    "author": "string",
    "tags": ["string"],
    "version": "string",
    "custom_fields": {}
  }
}
```

**Response (201 Created):**
```json
{
  "id": "eval_xxxxxxxxxxxx",  // Unique evaluation ID
  "object": "evaluation",
  "created": 1234567890,       // Unix timestamp
  "name": "string",
  "description": "string",
  "eval_type": "string",
  "eval_spec": {},
  "dataset_id": "string",
  "metadata": {}
}
```

**Error Responses:**
- `400 Bad Request` - Invalid request body or missing required fields
- `401 Unauthorized` - Missing or invalid API key
- `409 Conflict` - Evaluation name already exists
- `422 Unprocessable Entity` - Validation error (e.g., invalid eval_type)

**Important Notes:**
- You MUST provide either `dataset` (inline data) or `dataset_id` (reference), but not both
- The `sub_type` field is REQUIRED when `eval_type` is "model_graded"
- All scores and thresholds use a 0-1 scale where 1 is best

---

#### List Evaluations
`GET /api/v1/evaluations`

Lists evaluations with pagination support.

**Query Parameters:**
- `limit` (integer, 1-100, default: 20) - Number of items per page
- `after` (string) - Cursor for pagination (use `last_id` from previous response)
- `order` (string, "asc" | "desc", default: "desc") - Sort order by creation time
- `eval_type` (string) - Filter by evaluation type

**Response (200 OK):**
```json
{
  "object": "list",
  "data": [
    {
      "id": "eval_xxxxxxxxxxxx",
      "object": "evaluation",
      "created": 1234567890,
      "name": "string",
      "description": "string",
      "eval_type": "string",
      "eval_spec": {},
      "dataset_id": "string",
      "metadata": {}
    }
  ],
  "has_more": false,           // More results available
  "first_id": "eval_xxxxxxxxxxxx",
  "last_id": "eval_yyyyyyyyyyyy"
}
```

**Error Responses:**
- `401 Unauthorized` - Invalid API key

---

#### Get Evaluation
`GET /api/v1/evaluations/{eval_id}`

Retrieves a specific evaluation by ID.

**Path Parameters:**
- `eval_id` (string) - Evaluation ID (format: `eval_xxxxxxxxxxxx`)

**Response (200 OK):**
```json
{
  "id": "eval_xxxxxxxxxxxx",
  "object": "evaluation",
  "created": 1234567890,
  "name": "string",
  "description": "string",
  "eval_type": "string",
  "eval_spec": {},
  "dataset_id": "string",
  "metadata": {}
}
```

**Error Responses:**
- `401 Unauthorized` - Invalid API key
- `404 Not Found` - Evaluation not found

---

#### Update Evaluation
`PATCH /api/v1/evaluations/{eval_id}`

Updates an existing evaluation. Only provided fields are updated.

**Path Parameters:**
- `eval_id` (string) - Evaluation ID

**Request Body:**
```json
{
  "name": "string",           // Optional
  "description": "string",    // Optional
  "eval_spec": {},           // Optional, partial updates supported
  "dataset_id": "string",    // Optional
  "metadata": {}             // Optional, replaces entire metadata
}
```

**Response (200 OK):** Updated evaluation object

**Error Responses:**
- `400 Bad Request` - Invalid update data
- `401 Unauthorized` - Invalid API key
- `404 Not Found` - Evaluation not found

---

#### Delete Evaluation
`DELETE /api/v1/evaluations/{eval_id}`

Soft deletes an evaluation (may be recoverable).

**Path Parameters:**
- `eval_id` (string) - Evaluation ID

**Response (204 No Content):** Empty response on success

**Error Responses:**
- `401 Unauthorized` - Invalid API key
- `404 Not Found` - Evaluation not found

---

### Evaluation Runs

#### Create Run
`POST /api/v1/evaluations/{eval_id}/runs`

Starts an asynchronous evaluation run.

**Path Parameters:**
- `eval_id` (string) - Evaluation ID

**Request Body:**
```json
{
  "target_model": "string",    // Optional, model being evaluated
  "dataset_override": {        // Optional, override evaluation's dataset
    "samples": [
      {
        "input": {},
        "expected": {}
      }
    ]
  },
  "config": {                  // Run configuration
    "temperature": 0.0,        // LLM temperature (0-2, default: 0)
    "max_workers": 4,          // Parallel workers (1-16, default: 4)
    "timeout_seconds": 300,    // Timeout per sample (default: 300)
    "batch_size": 10          // Batch size (1-100, default: 10)
  },
  "webhook_url": "string"      // Optional webhook for completion notification
}
```

**Response (202 Accepted):**
```json
{
  "id": "run_xxxxxxxxxxxx",
  "object": "evaluation.run",
  "created": 1234567890,
  "eval_id": "eval_xxxxxxxxxxxx",
  "status": "pending",         // "pending" | "running" | "completed" | "failed" | "cancelled"
  "target_model": "string",
  "progress": {
    "total_samples": 0,
    "completed_samples": 0,
    "failed_samples": 0,
    "current_batch": 0,
    "percent_complete": 0.0
  },
  "estimated_completion": null,
  "error_message": null,
  "metadata": {}
}
```

**Error Responses:**
- `401 Unauthorized` - Invalid API key
- `404 Not Found` - Evaluation not found
- `429 Too Many Requests` - Rate limit exceeded (50/minute default)

---

#### List Runs
`GET /api/v1/evaluations/{eval_id}/runs`

Lists runs for a specific evaluation.

**Path Parameters:**
- `eval_id` (string) - Evaluation ID

**Query Parameters:**
- `limit` (integer, 1-100, default: 20) - Results per page
- `after` (string) - Cursor for pagination
- `status` (string) - Filter by status: "pending" | "running" | "completed" | "failed" | "cancelled"

**Response (200 OK):** List of run objects

---

#### Get Run Status
`GET /api/v1/evaluations/runs/{run_id}`

Gets current status and progress of a run.

**Path Parameters:**
- `run_id` (string) - Run ID (format: `run_xxxxxxxxxxxx`)

**Response (200 OK):**
```json
{
  "id": "run_xxxxxxxxxxxx",
  "object": "evaluation.run",
  "created": 1234567890,
  "eval_id": "eval_xxxxxxxxxxxx",
  "status": "running",
  "target_model": "string",
  "progress": {
    "total_samples": 100,
    "completed_samples": 45,
    "failed_samples": 2,
    "current_batch": 5,
    "percent_complete": 45.0
  },
  "estimated_completion": 1234567950,  // Unix timestamp
  "error_message": null,
  "metadata": {}
}
```

**Error Responses:**
- `401 Unauthorized` - Invalid API key
- `404 Not Found` - Run not found

---

#### Get Run Results
`GET /api/v1/evaluations/runs/{run_id}/results`

Gets complete results for a finished run.

**Path Parameters:**
- `run_id` (string) - Run ID

**Response (200 OK):**
```json
{
  "id": "run_xxxxxxxxxxxx",
  "object": "evaluation.run.result",
  "eval_id": "eval_xxxxxxxxxxxx",
  "status": "completed",
  "started_at": 1234567890,
  "completed_at": 1234567920,
  "results": {
    "aggregate": {
      "mean_score": 0.85,      // Average score (0-1)
      "std_dev": 0.12,         // Standard deviation
      "min_score": 0.60,       // Minimum score
      "max_score": 0.98,       // Maximum score
      "pass_rate": 0.75,       // Percentage passing threshold (0-1)
      "total_samples": 100,
      "failed_samples": 25   // Unscored (no score; a failed judge call): in no mean_score or pass_rate
    },
    "by_metric": {             // Breakdown by metric (if applicable)
      "fluency": {
        "mean": 0.88,
        "std": 0.10,
        "min": 0.65,
        "max": 0.98,
        "median": 0.89
      },
      "relevance": {
        "mean": 0.82,
        "std": 0.15,
        "min": 0.55,
        "max": 0.96,
        "median": 0.84
      }
    },
    "sample_results": [        // Individual sample results
      {
        "sample_id": "sample_0001",
        "scores": {
          "fluency": 0.85,
          "relevance": 0.90
        },
        "passed": true,        // Met threshold
        "error": null
      }
    ],
    "failed_samples": []       // Samples that failed to evaluate
  },
  "usage": {                   // Token usage (for LLM evaluations)
    "total_tokens": 15000,
    "prompt_tokens": 10000,
    "completion_tokens": 5000,
    "cost_estimate": 0.45      // Estimated cost in USD
  },
  "duration_seconds": 30
}
```

**Error Responses:**
- `400 Bad Request` - Run not completed yet
- `401 Unauthorized` - Invalid API key
- `404 Not Found` - Run not found

---

#### Cancel Run
`POST /api/v1/evaluations/runs/{run_id}/cancel`

Cancels a running evaluation.

**Path Parameters:**
- `run_id` (string) - Run ID

**Response (200 OK):**
```json
{
  "id": "run_xxxxxxxxxxxx",
  "status": "cancelled"        // or "cancelling" if still processing
}
```

**Error Responses:**
- `400 Bad Request` - Run already completed or failed
- `401 Unauthorized` - Invalid API key
- `404 Not Found` - Run not found

---

#### Stream Run Progress
Not currently available on the unified router. Poll `GET /api/v1/evaluations/runs/{run_id}` for status updates.

---

### Datasets

#### Create Dataset
`POST /api/v1/evaluations/datasets`

Creates a reusable dataset for evaluations.

**Request Body:**
```json
{
  "name": "string",            // Required, unique name
  "description": "string",     // Optional description
  "samples": [                 // Required, dataset samples
    {
      "input": {},            // Input data (format depends on eval_type)
      "expected": {}          // Expected output
    }
  ],
  "metadata": {}              // Optional metadata
}
```

**Response (201 Created):**
```json
{
  "id": "dataset_xxxxxxxxxxxx",
  "object": "dataset",
  "created": 1234567890,
  "name": "string",
  "description": "string",
  "sample_count": 100,
  "samples": [...],           // Full sample data
  "metadata": {}
}
```

**Error Responses:**
- `400 Bad Request` - Invalid dataset format
- `401 Unauthorized` - Invalid API key
- `409 Conflict` - Dataset name already exists

---

#### List Datasets
`GET /api/v1/evaluations/datasets`

Lists available datasets.

**Query Parameters:**
- `limit` (integer, 1-100, default: 20)
- `after` (string) - Cursor for pagination

**Response (200 OK):**
```json
{
  "object": "list",
  "data": [
    {
      "id": "dataset_xxxxxxxxxxxx",
      "object": "dataset",
      "created": 1234567890,
      "name": "string",
      "description": "string",
      "sample_count": 100,
      "samples": null,        // Samples omitted in list view
      "metadata": {}
    }
  ],
  "has_more": false,
  "first_id": "dataset_xxxxxxxxxxxx",
  "last_id": "dataset_yyyyyyyyyyyy"
}
```

---

#### Get Dataset
`GET /api/v1/evaluations/datasets/{dataset_id}`

Gets a specific dataset including all samples.

**Path Parameters:**
- `dataset_id` (string) - Dataset ID

**Response (200 OK):** Full dataset object with samples

**Error Responses:**
- `401 Unauthorized` - Invalid API key
- `404 Not Found` - Dataset not found

---

#### Delete Dataset
`DELETE /api/v1/evaluations/datasets/{dataset_id}`

Permanently deletes a dataset.

**Path Parameters:**
- `dataset_id` (string) - Dataset ID

**Response (204 No Content):** Empty response on success

**Error Responses:**
- `401 Unauthorized` - Invalid API key
- `404 Not Found` - Dataset not found

---

## Evaluation Types & Formats

### model_graded
Uses an LLM to evaluate outputs. Requires `sub_type` specification.

#### summarization (G-Eval)
Evaluates text summarization quality.

**eval_spec:**
```json
{
  "sub_type": "summarization",
  "evaluator_model": "gpt-4",  // or "anthropic", "groq", etc.
  "metrics": ["fluency", "consistency", "relevance", "coherence"],
  "threshold": 0.7
}
```

**Sample Format:**
```json
{
  "input": {
    "source_text": "Original document text to be summarized",
    "summary": "The summary to evaluate"
  }
}
```

**Metrics:**
- `fluency`: Grammar and readability (0-1)
- `consistency`: Factual alignment with source (0-1)
- `relevance`: Information selection quality (0-1)
- `coherence`: Logical flow and structure (0-1)

#### rag
Evaluates Retrieval-Augmented Generation systems.

**eval_spec:**
```json
{
  "sub_type": "rag",
  "evaluator_model": "gpt-4",
  "metrics": ["relevance", "faithfulness", "answer_similarity", "context_precision", "context_recall", "claim_faithfulness"],
  "threshold": 0.75
}
```

**Sample Format:**
```json
{
  "input": {
    "query": "User's question",
    "contexts": ["Retrieved context 1", "Retrieved context 2"],
    "response": "Generated response"
  },
  "expected": {
    "answer": "Ground truth answer (optional)"
  }
}
```

**Metrics:**
- `relevance`: Response relevance to query (0-1)
- `faithfulness`: Grounding in provided contexts (0-1)
- `answer_similarity`: Similarity to ground truth (0-1)
- `context_precision`: Precision of retrieved contexts (0-1)
- `context_recall`: Recall of relevant information (0-1)
 - `claim_faithfulness`: Fraction of extracted claims supported by contexts (0-1)

#### response_quality
General response quality evaluation with custom criteria.

**eval_spec:**
```json
{
  "sub_type": "response_quality",
  "evaluator_model": "gpt-4",
  "custom_criteria": {
    "helpfulness": "How helpful is the response?",
    "accuracy": "Is the information accurate?",
    "completeness": "Does it fully address the prompt?"
  },
  "threshold": 0.8
}
```

**Sample Format:**
```json
{
  "input": {
    "prompt": "User's prompt",
    "response": "Generated response",
    "expected_format": "Expected response format (optional)"
  }
}
```

### exact_match
Checks for exact string match (case-insensitive).

**eval_spec:**
```json
{
  "threshold": 1.0  // Usually 1.0 for exact match
}
```

**Sample Format:**
```json
{
  "input": {"output": "Generated text"},
  "expected": {"output": "Expected exact text"}
}
```

**Scoring:** 1.0 if exact match (case-insensitive), 0.0 otherwise

### fuzzy_match
Uses string similarity algorithms (e.g., Levenshtein distance).

**eval_spec:**
```json
{
  "threshold": 0.85  // Similarity threshold (0-1)
}
```

**Sample Format:**
```json
{
  "input": {"output": "Generated text"},
  "expected": {"output": "Similar expected text"}
}
```

**Scoring:** Similarity score from 0 (completely different) to 1 (identical)

### includes
Checks if output contains all expected items.

**eval_spec:**
```json
{
  "threshold": 0.8  // Percentage of items that must be present (0-1)
}
```

**Sample Format:**
```json
{
  "input": {"output": "The generated text to check"},
  "expected": {"includes": ["keyword1", "keyword2", "phrase to find"]}
}
```

**Scoring:** Percentage of expected items found in output (0-1)

## Error Response Format

All error responses follow a consistent format:

```json
{
  "error": {
    "message": "Human-readable error description",
    "type": "error_type",     // See error types below
    "param": "parameter_name", // Optional, which parameter caused the error
    "code": "error_code"       // Optional, specific error code
  }
}
```

**Note:** Some error responses may wrap the error object in a `detail` field:
```json
{
  "detail": {
    "error": {
      "message": "...",
      "type": "...",
      "code": "..."
    }
  }
}
```

**Error Types:**
- `invalid_request_error` - Invalid request parameters or body
- `authentication_error` - Invalid or missing API key
- `permission_error` - Insufficient permissions
- `not_found_error` - Resource not found
- `rate_limit_error` - Rate limit exceeded
- `server_error` - Internal server error

**HTTP Status Codes:**
- `200 OK` - Successful GET/PATCH request
- `201 Created` - Successful POST creating new resource
- `202 Accepted` - Request accepted for async processing
- `204 No Content` - Successful DELETE
- `400 Bad Request` - Invalid request format or parameters
- `401 Unauthorized` - Authentication required or failed
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource already exists
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

## Rate Limits

Rate limits (route-specific for evaluations):
- Standard evaluation requests: 60 requests/minute
- Run operations: 10 requests/minute
- Batch operations: 5 requests/minute
- Burst protection: 10 requests/second

## Webhooks

When `webhook_url` is provided in a run request, the following payload is sent upon completion:

```json
{
  "event": "run.completed",    // or "run.failed" | "run.cancelled"
  "run_id": "run_xxxxxxxxxxxx",
  "eval_id": "eval_xxxxxxxxxxxx",
  "status": "completed",
  "completed_at": 1234567890,
  "results_url": "/api/v1/evaluations/runs/run_xxxxxxxxxxxx/results",
  "summary": {
    "mean_score": 0.85,
    "pass_rate": 0.75,
    "total_samples": 100,
    "duration_seconds": 30
  },
  "error": null                // Error message if failed
}
```

**Webhook Requirements:**
- Must accept POST requests
- Should respond with 2xx status code
- Timeout: 10 seconds
- Retries: 3 attempts with exponential backoff

## Code Examples

> ⚠️ Generate a strong API key (new format) and set it via `SINGLE_USER_API_KEY` before running these examples:
> ```bash
> python -m tldw_Server_API.app.core.AuthNZ.initialize
> # Copy SINGLE_USER_API_KEY into your environment
> ```
> Replace `YOUR_API_KEY` below with that value (or load it from your environment at runtime).

### Python
```python
import json
from urllib.request import Request, urlopen

# Configuration
API_KEY = "YOUR_API_KEY"
BASE_URL = "http://localhost:8000"
headers = {"Authorization": f"Bearer {API_KEY}"}

def request_json(method, url, payload=None, headers=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = Request(url, data=data, headers=hdrs, method=method)
    with urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

# Create evaluation
eval_request = {
    "name": "my_evaluation",
    "eval_type": "exact_match",
    "eval_spec": {"threshold": 1.0},
    "dataset": [
        {"input": {"output": "test"}, "expected": {"output": "test"}}
    ]
}

eval_response = request_json(
    "POST",
    f"{BASE_URL}/api/v1/evaluations",
    payload=eval_request,
    headers=headers,
)
eval_id = eval_response["id"]

# Run evaluation
run_request = {"config": {"temperature": 0}}
run_response = request_json(
    "POST",
    f"{BASE_URL}/api/v1/evaluations/{eval_id}/runs",
    payload=run_request,
    headers=headers,
)
run_id = run_response["id"]

# Get results
results = request_json(
    "GET",
    f"{BASE_URL}/api/v1/evaluations/runs/{run_id}/results",
    headers=headers,
)
print(results)
```

### cURL
```bash
# Set API key
export API_KEY="YOUR_API_KEY"

# Create evaluation
curl -X POST http://localhost:8000/api/v1/evaluations \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test_eval",
    "eval_type": "exact_match",
    "eval_spec": {"threshold": 1.0},
    "dataset": [
      {"input": {"output": "test"}, "expected": {"output": "test"}}
    ]
  }'

# Run evaluation
curl -X POST http://localhost:8000/api/v1/evaluations/eval_xxx/runs \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"config": {"temperature": 0}}'

# Get results
curl http://localhost:8000/api/v1/evaluations/runs/run_xxx/results \
  -H "Authorization: Bearer $API_KEY"
```

### JavaScript/TypeScript
```javascript
const API_KEY = 'YOUR_API_KEY';
const BASE_URL = 'http://localhost:8000';

// Create evaluation
const evalResponse = await fetch(`${BASE_URL}/api/v1/evaluations`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    name: 'js_eval',
    eval_type: 'exact_match',
    eval_spec: { threshold: 1.0 },
    dataset: [
      { input: { output: 'test' }, expected: { output: 'test' } }
    ]
  })
});
const { id: evalId } = await evalResponse.json();

// Run evaluation
const runResponse = await fetch(`${BASE_URL}/api/v1/evaluations/${evalId}/runs`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${API_KEY}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ config: { temperature: 0 } })
});
const { id: runId } = await runResponse.json();

// Get results (after waiting)
const resultsResponse = await fetch(`${BASE_URL}/api/v1/evaluations/runs/${runId}/results`, {
  headers: { 'Authorization': `Bearer ${API_KEY}` }
});
const results = await resultsResponse.json();
console.log(results);
```

## API Specification

The complete OpenAPI 3.0 specification is available at:
- **JSON**: `http://localhost:8000/openapi.json`
- **Interactive Docs (Swagger UI)**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Configuration

### LLM Provider Configuration

Configure LLM providers in `tldw_Server_API/Config_Files/config.txt`:

```ini
[API]
# OpenAI
openai_api_key = sk-...
openai_model = gpt-4

# Anthropic
anthropic_api_key = sk-ant-...
anthropic_model = claude-opus-4-20250514

# Google
google_api_key = ...
google_model = gemini-pro

# Groq
groq_api_key = gsk_...
groq_model = mixtral-8x7b-32768

# Other providers...
```

### Supported LLM Providers
- OpenAI (GPT-4o family, etc.)
- Anthropic (Claude models)
- Google (Gemini models)
- Groq (Mixtral, Llama, etc.)
- Cohere
- Mistral
- DeepSeek
- HuggingFace
- OpenRouter
- Local models (via Ollama, llama.cpp, etc.)

## Important Notes

1. **Dataset Requirement**: Every evaluation MUST have either an inline `dataset` or reference a `dataset_id`. The API will return a 422 error if neither is provided.

2. **Score Interpretation**: All scores use a 0-1 scale where 1 is best. The `threshold` determines pass/fail.

3. **Async Processing**: Evaluation runs are processed asynchronously. Use polling or SSE streaming to monitor progress.

4. **Rate Limiting**: Default rate limits apply unless `TESTING=true` is set. Plan accordingly for large evaluations.

5. **Error Handling**: Always check for both `error` and `detail.error` formats when handling error responses.

6. **API Path**: All endpoints use the `/api/v1/` prefix. Using `/v1/` alone will result in 404 errors.

## Support

- **Documentation**: See Quick Start Guide and User Guide
- **Issues**: Report at https://github.com/rmusser01/tldw_server/issues
- **API Status**: Check `/health` endpoint
