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
    "unscored": 0,
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

- The user is over an LLM limit: `detail.error` is `limit_exceeded`, and `Retry-After` says when to try again. The run is refused before its dataset loads, so nothing is evaluated. A run whose `api_name` runs on a key the user stored that bills them (see `Docs/User_Guides/Server/BYOK_User_Guide.md`) is not refused: every item's model response and judge call run on that key and do not count. This route takes no `api_key`; only the standalone `POST /api/v1/benchmarks/{name}/run` does, and a run there that sends one is refused, because that key counts.
- An admitted run runs to completion, even if it takes the user past the limit; no item is refused mid-run. The next run on the server's key is refused.

### 404 benchmark not found

- Run benchmark list first and use returned `name` exactly.

### 404 dataset load failure during run

- Benchmark exists but backing dataset failed to load. Check server logs and benchmark data configuration.

### Run failures in summary

- `failed > 0` counts items whose evaluation raised outside the model and judge calls, such as an item whose data could not be built into a prompt. A failed model or judge call is never `failed`: it is `unscored`.
- `unscored > 0` counts items with no score: the model's own answer call failed or raised (provider error, timeout, rate limit, or a local server that is down), the judge call failed or raised (the same, or an empty or truncated reply), or the judge's reply held no readable score. They are neither `successful` nor `failed`, and they are left out of `average_score`, `min_score`, `max_score`, `by_category` and the stored scores. The server logs `Model response generation failed`, `Judge call failed for metric`, `Custom metric evaluation failed` or `Could not parse valid score`.
- If every item is unscored or failed, `successful` is 0 and the three scores read 0.0; they are not a measured zero.
- Reduce `parallel` and retry.
- Confirm provider credentials (`api_name`) are valid server-side.

### Every item scored 0.5

- Before the November 2026 fix (TASK-15485), the judge's call never reached the provider, so every item got the 0.5 default. The `average_score`, `min_score`, `max_score` and `by_category` of those runs, and the results they stored (`save_results`), carry no information.
- Re-run `simple_bench` or `bullshit_benchmark`: their scores are now the judge's. Do not re-run the others expecting meaningful scores. `mmlu_pro` rows are not yet mapped to the benchmark's fields, so its judge is told the correct answer is "None", and its scores look real but carry no information. `simpleqa_verified` has no loader entry and `gpqa_diamond` most likely fails the same way (404), `mmlu_pro` and `gpqa_diamond` also need the `datasets` package (the `data-tools` extra), and six more benchmarks answer 501.
- Each item's score is the judge's reply mapped to 0-1: a 1-10 score (10 is 1.0, 1 is 0.0). A reply with no score, a failed judge call or a failed model answer leaves the item unscored (see "Run failures in summary"). The run's response does not show per-item explanations, so look in the server log; the CLI's saved results keep a `null` score and the explanation (the `Error:` text for a failed call). Re-running such an item may succeed.

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
