# Benchmark Creation and Runs (API + WebUI/Extension)

This guide covers how to run benchmarks using:

- The API (`/api/v1/evaluations/benchmarks...`)
- The WebUI/extension Evaluations workflow (`benchmark-run`)

It reflects current shipped behavior in this repository as of March 2, 2026.

## Who This Guide Is For

- Operators who want copy/paste benchmark runs
- Team members validating model quality with existing benchmark definitions
- Contributors who need a map of the benchmark run surface

## Prerequisites

- Server running (for example `http://127.0.0.1:8000`)
- Auth configured:
  - Single-user: API key (`X-API-KEY` or Bearer token)
  - Multi-user: JWT bearer token
- Permission scope that can run evaluations/benchmarks

## Current State (Shipped)

### API Routes

- `GET /api/v1/evaluations/benchmarks`
- `GET /api/v1/evaluations/benchmarks/{benchmark_name}`
- `POST /api/v1/evaluations/benchmarks/{benchmark_name}/run`

### WebUI/Extension Flow

- Open Evaluations
- Go to Runs
- Use Ad-hoc evaluator mode
- Select endpoint `benchmark-run`
- Pick a benchmark from the selector
- Submit JSON config

## WebUI/Extension Quickstart

1. Open `/evaluations`.
2. Open the `Runs` tab.
3. In `Ad-hoc evaluator`, set endpoint to `benchmark-run`.
4. Choose a benchmark.
5. Use a payload like:

```json
{
  "limit": 25,
  "api_name": "openai",
  "parallel": 4,
  "save_results": true
}
```

6. Click `Start run`.
7. Review returned summary in the result panel (totals, average score, min/max, category breakdown when available).

### Optional Filters

If a benchmark dataset includes taxonomy labels, you can filter categories:

```json
{
  "limit": 100,
  "api_name": "openai",
  "parallel": 4,
  "save_results": true,
  "filter_categories": ["science", "history"]
}
```

## API Quickstart

Examples below use single-user API key mode. Replace auth header for JWT mode as needed.

### 1) List Benchmarks

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/evaluations/benchmarks" \
  -H "X-API-KEY: YOUR_API_KEY"
```

### 2) Get One Benchmark Definition

```bash
curl -X GET "http://127.0.0.1:8000/api/v1/evaluations/benchmarks/bullshit_benchmark" \
  -H "X-API-KEY: YOUR_API_KEY"
```

### 3) Run Benchmark

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/evaluations/benchmarks/bullshit_benchmark/run" \
  -H "X-API-KEY: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "limit": 25,
    "api_name": "openai",
    "parallel": 4,
    "save_results": true
  }'
```

Typical response shape:

```json
{
  "benchmark": "bullshit_benchmark",
  "total_samples": 25,
  "results_summary": {
    "total_evaluated": 25,
    "successful": 25,
    "failed": 0,
    "average_score": 0.84,
    "min_score": 0.5,
    "max_score": 1.0
  },
  "evaluation_id": "eval_..."
}
```

## Creating Custom Benchmarks (Current Supported Path)

Current behavior distinguishes between:

- Running registered benchmark definitions (`/benchmarks/...`)
- Creating custom evaluation definitions/datasets (`/api/v1/evaluations` and `/api/v1/evaluations/datasets`)

Important: there is no dedicated end-user API in this route group for creating brand-new benchmark types dynamically. In current shipped behavior, benchmark names are loaded from server-side benchmark registry/configuration.

If your goal is a custom quality test today without backend benchmark registration:

1. Create a dataset in `/api/v1/evaluations/datasets`
2. Create an evaluation definition in `/api/v1/evaluations`
3. Run it through `/api/v1/evaluations/{eval_id}/runs`

## Troubleshooting

### 401/403

- Verify auth header type for your mode.
- Confirm user has permission to read/run evaluations benchmarks.

### 402 LLM quota

- The user is over an LLM limit: `detail.error` is `limit_exceeded`, and `Retry-After` says when to try again. The run is refused before its dataset loads, so nothing is evaluated.
- A user who reaches the limit during a run has the remaining items refused; they count under `failed` in the summary.

### 404 benchmark not found

- Run benchmark list first and use returned `name` exactly.

### 404 dataset load failure during run

- Benchmark exists but backing dataset failed to load. Check server logs and benchmark data configuration.

### Run failures in summary

- `failed > 0` means per-item evaluator errors occurred.
- Reduce `parallel` and retry.
- Confirm provider credentials (`api_name`) are valid server-side.

### Rate limiting

- If requests are throttled, retry after the server rate-limit window resets.

## Roadmap (Not Yet Shipped)

Potential future improvements (not currently guaranteed in this build):

- Dedicated benchmark creation UI flow in Evaluations.
- First-class benchmark create/register API for operator workflows.
- Richer run telemetry and per-item drill-down in UI benchmark mode.

Treat this section as directional only. Use the current-state sections above for operational steps.

## Contributor Appendix (Code Map)

- Benchmark API routes:
  - `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py`
- Unified router mounting:
  - `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`
- WebUI runs tab benchmark mode:
  - `apps/packages/ui/src/components/Option/Evaluations/tabs/RunsTab.tsx`
- WebUI hooks/service calls:
  - `apps/packages/ui/src/components/Option/Evaluations/hooks/useRuns.ts`
  - `apps/packages/ui/src/services/evaluations.ts`
