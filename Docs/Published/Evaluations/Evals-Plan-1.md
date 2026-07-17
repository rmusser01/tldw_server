# OpenAI-Compatible Evals API Implementation Plan

## Overview
Create a fully OpenAI-compatible evaluation API that wraps existing backend evaluation logic while providing proper async processing, data persistence, and RESTful design.

## 1. API Endpoint Structure (RESTful)

```
# Evaluation Management
POST   /v1/evals                           # Create evaluation definition
GET    /v1/evals                           # List evaluations (paginated)
GET    /v1/evals/{eval_id}                 # Get evaluation details
PATCH  /v1/evals/{eval_id}                 # Update evaluation
DELETE /v1/evals/{eval_id}                 # Delete evaluation

# Run Management
POST   /v1/evals/{eval_id}/runs            # Create & start a run
GET    /v1/evals/{eval_id}/runs            # List runs for an eval
GET    /v1/runs/{run_id}                   # Get run details/status
GET    /v1/runs/{run_id}/results           # Get run results (when complete)
POST   /v1/runs/{run_id}/cancel            # Cancel a running evaluation

# Dataset Management
POST   /v1/datasets                        # Upload dataset
GET    /v1/datasets                        # List datasets
GET    /v1/datasets/{dataset_id}           # Get dataset details
DELETE /v1/datasets/{dataset_id}           # Delete dataset
```

## 2. Data Models (OpenAI-Compatible)

### Evaluation Definition
```python
{
  "id": "eval_abc123",
  "object": "evaluation",
  "created": 1234567890,
  "name": "summarization_quality_v1",
  "description": "Evaluates summary quality using G-Eval metrics",
  "eval_type": "model_graded",  # or "exact_match", "includes", "rag", "custom"
  "eval_spec": {
    "evaluator_model": "gpt-4",  # Model doing the evaluation
    "metrics": ["fluency", "consistency", "relevance", "coherence"],
    "scoring_prompt": "...",     # Optional custom prompt
    "threshold": 0.7              # Pass/fail threshold
  },
  "dataset_id": "dataset_xyz789",  # Reference to dataset
  "metadata": {
    "author": "user@example.com",
    "tags": ["summarization", "quality"],
    "version": "1.0.0"
  }
}
```

### Run Request
```python
{
  "target_model": "gpt-3.5-turbo",  # Model being evaluated
  "dataset_override": {              # Optional: override eval's dataset
    "samples": [
      {
        "input": {"source_text": "...", "summary": "..."},
        "expected": {"score": 0.85}
      }
    ]
  },
  "config": {
    "temperature": 0.0,
    "max_workers": 4,              # Parallel processing
    "timeout_seconds": 300
  },
  "webhook_url": "https://..."      # Optional: for completion notification
}
```

### Run Response (Async)
```python
{
  "id": "run_def456",
  "object": "evaluation.run",
  "created": 1234567890,
  "eval_id": "eval_abc123",
  "status": "pending",  # pending -> running -> completed/failed/cancelled
  "target_model": "gpt-3.5-turbo",
  "progress": {
    "total_samples": 100,
    "completed_samples": 0,
    "failed_samples": 0,
    "current_batch": 0
  },
  "estimated_completion": 1234567990,
  "metadata": {}
}
```

### Run Results (When Complete)
```python
{
  "id": "run_def456",
  "object": "evaluation.run.result",
  "eval_id": "eval_abc123",
  "status": "completed",
  "started_at": 1234567890,
  "completed_at": 1234567990,
  "results": {
    "aggregate": {
      "mean_score": 0.875,
      "std_dev": 0.043,
      "min_score": 0.72,
      "max_score": 0.96,
      "pass_rate": 0.92  # Based on threshold
    },
    "by_metric": {
      "fluency": {"mean": 0.88, "std": 0.05},
      "consistency": {"mean": 0.91, "std": 0.03},
      "relevance": {"mean": 0.84, "std": 0.06},
      "coherence": {"mean": 0.87, "std": 0.04}
    },
    "sample_results": [  # Individual sample scores
      {
        "sample_id": "sample_001",
        "scores": {"fluency": 0.9, "consistency": 0.92, ...},
        "passed": true
      }
    ],
    "failed_samples": []
  },
  "usage": {
    "total_tokens": 45000,
    "prompt_tokens": 30000,
    "completion_tokens": 15000
  },
  "duration_seconds": 98.5
}
```

### List Response (Paginated)
```python
{
  "object": "list",
  "data": [...],  # Array of evaluations/runs
  "has_more": true,
  "first_id": "eval_abc123",
  "last_id": "eval_xyz789"
}
```

### Error Response
```python
{
  "error": {
    "message": "Evaluation not found",
    "type": "invalid_request_error",
    "param": "eval_id",
    "code": "resource_not_found"
  }
}
```

## 3. Database Schema

```sql
-- Evaluations table
CREATE TABLE evaluations (
    id VARCHAR(50) PRIMARY KEY,  -- eval_abc123 format
    name VARCHAR(255) NOT NULL,
    description TEXT,
    eval_type VARCHAR(50) NOT NULL,
    eval_spec JSONB NOT NULL,
    dataset_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255),
    metadata JSONB,
    deleted_at TIMESTAMP NULL  -- Soft delete
);

-- Runs table
CREATE TABLE evaluation_runs (
    id VARCHAR(50) PRIMARY KEY,  -- run_def456 format
    eval_id VARCHAR(50) REFERENCES evaluations(id),
    status VARCHAR(20) NOT NULL,  -- pending/running/completed/failed/cancelled
    target_model VARCHAR(100),
    config JSONB,
    progress JSONB,
    results JSONB,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    webhook_url TEXT,
    usage JSONB
);

-- Datasets table
CREATE TABLE datasets (
    id VARCHAR(50) PRIMARY KEY,  -- dataset_xyz789 format
    name VARCHAR(255) NOT NULL,
    description TEXT,
    samples JSONB NOT NULL,  -- Or reference to file storage
    sample_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255),
    metadata JSONB
);

-- Indexes
CREATE INDEX idx_evals_created ON evaluations(created_at DESC);
CREATE INDEX idx_runs_eval ON evaluation_runs(eval_id);
CREATE INDEX idx_runs_status ON evaluation_runs(status);
```

## 4. Implementation Details

### Backend Integration Mapping
```python
EVAL_TYPE_MAPPING = {
    "model_graded": {
        "summarization": lambda req: ms_g_eval.run_geval(**req),
        "rag": lambda req: rag_evaluator.evaluate(**req),
        "response_quality": lambda req: quality_evaluator.evaluate(**req)
    },
    "exact_match": lambda req: exact_match_eval(**req),
    "includes": lambda req: includes_eval(**req),
    "custom": lambda req: custom_eval_runner(**req)
}
```

### Async Processing with Background Tasks
```python
from fastapi import BackgroundTasks
import asyncio
from typing import Dict, Any

async def run_evaluation_async(run_id: str, eval_config: Dict[str, Any]):
    """Background task for running evaluations"""
    try:
        # Update status to running
        await update_run_status(run_id, "running")

        # Get evaluation function
        eval_fn = EVAL_TYPE_MAPPING[eval_config["eval_type"]][eval_config["sub_type"]]

        # Process samples in batches
        results = []
        for batch in create_batches(eval_config["samples"], batch_size=10):
            batch_results = await asyncio.gather(*[
                eval_fn(sample) for sample in batch
            ])
            results.extend(batch_results)

            # Update progress
            await update_run_progress(run_id, len(results))

        # Calculate aggregate metrics
        aggregate = calculate_aggregate_metrics(results)

        # Store results
        await store_run_results(run_id, results, aggregate)
        await update_run_status(run_id, "completed")

        # Send webhook if configured
        if eval_config.get("webhook_url"):
            await send_webhook_notification(eval_config["webhook_url"], run_id)

    except Exception as e:
        await update_run_status(run_id, "failed", error=str(e))
```

### Status Polling Implementation
```python
@router.get("/v1/runs/{run_id}")
async def get_run_status(run_id: str):
    """Get current status of evaluation run"""
    run = await get_run_from_db(run_id)

    if run["status"] == "completed":
        # Include results in response
        run["results"] = await get_run_results(run_id)
    elif run["status"] == "running":
        # Include progress information
        run["progress"] = await get_run_progress(run_id)
        run["estimated_completion"] = estimate_completion_time(run)

    return run
```

### Files to Create/Modify

1. Unified endpoints live at `/app/api/v1/endpoints/evaluations_unified.py` (legacy `evals_openai.py` removed)
   - All OpenAI-compatible endpoints
   - Async processing with BackgroundTasks
   - Proper error handling

2. **New: `/app/api/v1/schemas/openai_eval_schemas.py`**
   - All request/response models
   - Validation rules
   - OpenAI-compatible field names

3. **New: `/app/core/Evaluations/eval_runner.py`**
   - Async evaluation orchestration
   - Batch processing
   - Progress tracking
   - Result aggregation

4. **New: `/app/core/DB_Management/Evaluations_DB.py`**
   - Database operations for evals/runs/datasets
   - Transaction management
   - Query optimization

5. **Update: `/app/main.py`**
   - Register new router at `/v1/evals`
   - Remove old `/evaluations` router

## 5. Authentication & Rate Limiting

```python
from fastapi import Security, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify OpenAI-style Bearer token"""
    token = credentials.credentials
    if not token.startswith("sk-"):
        raise HTTPException(status_code=401, detail="Invalid API key format")
    # Validate against database/config
    return await validate_api_key(token)

# Apply to all endpoints
@router.post("/v1/evals", dependencies=[Depends(verify_api_key)])
```

## 6. Testing Strategy

1. **Unit Tests**: Test each evaluation type mapping
2. **Integration Tests**: Test full evaluation flow
3. **Load Tests**: Verify async processing under load
4. **OpenAI SDK Compatibility**: Test with OpenAI Python client patterns

## 7. Migration Steps

1. Create database tables ✅
2. Implement new schemas
3. Create dataset management endpoints
4. Implement evaluation CRUD endpoints
5. Add async run processing
6. Integrate with existing evaluation backend
7. Add authentication/rate limiting
8. Test thoroughly
9. Update documentation
10. Remove old endpoints

---

## Architecture Decision Records (ADRs)

### ADR-001: Database Choice for Evaluations
**Date**: 2024-01-09
**Status**: Implemented
**Decision**: Use SQLite for evaluation data storage
**Context**: Need persistent storage for evaluations, runs, and datasets
**Consequences**:
- Simple deployment (single file database)
- Good enough performance for evaluation workloads
- May need to migrate to PostgreSQL for production scale
**Implementation**: Created `Evaluations_DB.py` with SQLite backend

### ADR-002: ID Format Convention
**Date**: 2024-01-09
**Status**: Implemented
**Decision**: Use prefixed UUIDs (e.g., `eval_abc123`, `run_def456`, `dataset_xyz789`)
**Context**: Need unique, readable identifiers that match OpenAI's style
**Consequences**:
- Easy to identify resource types from IDs
- Compatible with OpenAI client expectations
- 12-character hex suffix provides sufficient uniqueness
**Implementation**: Applied in `Evaluations_DB.py` create methods

### ADR-003: Soft Delete vs Hard Delete
**Date**: 2024-01-09
**Status**: Implemented
**Decision**: Use soft deletes for evaluations, hard deletes for datasets
**Context**: Need to preserve evaluation history while allowing dataset cleanup
**Consequences**:
- Evaluations can be recovered if needed
- Dataset storage can be fully reclaimed
- Need to filter soft-deleted items in queries
**Implementation**: Added `deleted_at` column to evaluations table

### ADR-004: JSON Storage in SQLite
**Date**: 2024-01-09
**Status**: Implemented
**Decision**: Store complex objects as JSON TEXT in SQLite
**Context**: SQLite doesn't have native JSONB like PostgreSQL
**Consequences**:
- Need to serialize/deserialize in application code
- Can't query JSON fields directly in SQL
- Simple implementation for MVP
**Implementation**: Using `json.dumps/loads` in database methods

### ADR-005: Schema Design Pattern
**Date**: 2024-01-09
**Status**: Implemented
**Decision**: Use separate request/response models following OpenAI conventions
**Context**: Need to match OpenAI's API format exactly for compatibility
**Consequences**:
- Consistent field naming (snake_case)
- Standard `object` field for resource type identification
- Unix timestamps instead of ISO dates
- Generic list response wrapper
**Implementation**: Created `openai_eval_schemas.py` with OpenAI-compatible models

### ADR-006: Async Evaluation Processing
**Date**: 2024-01-09
**Status**: Implemented
**Decision**: Use asyncio with background tasks for evaluation runs
**Context**: Evaluations can be long-running and need progress tracking
**Consequences**:
- Non-blocking API responses for evaluation starts
- Progress tracking capability
- Webhook support for completion notifications
- Ability to cancel running evaluations
**Implementation**: Created `eval_runner.py` with async task management

### ADR-007: Backend Integration Strategy
**Date**: 2024-01-09
**Status**: Implemented
**Decision**: Wrap existing evaluation modules rather than rewrite
**Context**: Existing evaluation logic (G-Eval, RAG, etc.) works well
**Consequences**:
- Preserves tested evaluation logic
- Uses asyncio.to_thread for sync-to-async adaptation
- Maintains compatibility with existing code
**Implementation**: Integrated ms_g_eval, rag_evaluator, and response_quality_evaluator in eval_runner

---

## Latest Updates (2025-01-09)

### Authentication Improvements
- **Enhanced API Key Support**: Updated `verify_api_key` function to support:
  - Single-user deployments must set `SINGLE_USER_API_KEY` (generate via `python -m tldw_Server_API.app.core.AuthNZ.initialize`)
  - Environment variable configuration via `SINGLE_USER_API_KEY` or legacy `API_BEARER`
  - OpenAI-style `sk-` prefixed keys for compatibility
  - Proper error responses matching OpenAI format

### Real-time Progress Tracking (SSE)
- **New Endpoint**: `GET /v1/runs/{run_id}/stream`
- **Event Types**:
  - `progress`: Sample completion updates
  - `completed`: Final results
  - `failed`: Error details
  - `cancelled`: Cancellation confirmation
  - `heartbeat`: Keep-alive signals
- **Features**:
  - No-cache headers for real-time updates
  - Nginx buffering disabled for smooth streaming
  - Automatic connection management

### Comprehensive Test Suite
- Tests target unified endpoints under `tests/Evaluations/`
- **Test Coverage**:
  - Authentication (default key, sk- keys, invalid keys)
  - CRUD operations for evaluations, runs, datasets
  - Async evaluation processing
  - Error handling and edge cases
  - Pagination and filtering
  - Concurrent operations
- **27 test cases** covering all major functionality

### Rate Limiting Implementation
- **Using Resource Governor (RG)** for ingress rate limiting
- **Configured Limits**:
  - Create operations: 10/minute
  - Read operations: 100/minute
  - Run operations: 5/minute (resource intensive)
- **Applied to all endpoints** with appropriate limits

### Enhanced API Documentation
- **Comprehensive docstrings** for all endpoints
- **Includes**:
  - Rate limit information
  - Request/response details
  - Usage examples
  - Parameter descriptions
  - Return value specifications
- **OpenAPI/Swagger compatible** documentation

---

## Implementation Progress

### Completed
- [x] Database schema design
- [x] Database implementation (`Evaluations_DB.py`)
- [x] ID generation strategy
- [x] Basic CRUD operations for evaluations, runs, datasets
- [x] OpenAI-compatible schemas (`openai_eval_schemas.py`)
- [x] Async evaluation runner (`eval_runner.py`)
- [x] Integration with existing eval backends (G-Eval, RAG, Response Quality)
- [x] API endpoints implementation (`evaluations_unified.py`)
- [x] Authentication system (Bearer token support)
- [x] Error handling (OpenAI-style error responses)
- [x] Router registration in main.py
- [x] Basic testing and verification

### Future Enhancements
- [ ] Full webhook notification implementation
- [x] Real-time progress tracking via SSE (COMPLETED - 2025-01-09)
- [x] Comprehensive test suite (COMPLETED - 2025-01-09)
- [x] API documentation (COMPLETED - 2025-01-09)
- [x] Rate limiting (COMPLETED - 2025-01-09)
- [ ] Result caching
