# Evaluations Code Guide (Developers)

This guide helps project developers get up to speed on the Evaluations module: what’s in it, how it works end‑to‑end, and how to work with or extend it.

See also:
- Overview: `Docs/Code_Documentation/Evaluations/index.md:1`
- Deep dev guide: `Docs/Code_Documentation/Evaluations_Developer_Guide.md:1`
- Unified API reference: `Docs/API-related/Evaluations_API_Unified_Reference.md:1`
- Module README: `tldw_Server_API/app/core/Evaluations/README.md:1`
- Config (rates/tiers): `tldw_Server_API/Config_Files/evaluations_config.yaml:1`

Navigation:
- Endpoints (router): `tldw_Server_API/app/api/v1/endpoints/evaluations_unified.py:1`
- Schemas (unified): `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py:1`
- Service: `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py:1`
- Runner: `tldw_Server_API/app/core/Evaluations/eval_runner.py:1`
- DB manager: `tldw_Server_API/app/core/DB_Management/Evaluations_DB.py:1`

**Scope & Goals**
- Unified, OpenAI‑compatible evaluation surface under `/api/v1/evaluations`.
- Covers model‑graded (G‑Eval), RAG, response quality, OCR, propositions, label/NLI scoring, plus batch runs.
- First‑class datasets, runs, history, idempotency, webhooks, rate limits, and metrics.
- RAG Pipeline sweeps (chunking/retriever/reranker/generation) with leaderboards and ephemeral collections.
- Embeddings A/B testing (providers/models/arms) with export and significance.

**Quick Map (Where Things Live)**
- API layer
  - Router: `tldw_Server_API/app/api/v1/endpoints/evaluations_unified.py:1`
  - Split routers: CRUD `.../evaluations_crud.py:1`, datasets `.../evaluations_datasets.py:1`, webhooks `.../evaluations_webhooks.py:1`, rag pipeline `.../evaluations_rag_pipeline.py:1`, embeddings A/B `.../evaluations_embeddings_abtest.py:1`
  - Auth/rate helpers: `tldw_Server_API/app/api/v1/endpoints/evaluations_auth.py:1`
  - Schemas: `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py:1`, embeddings A/B `tldw_Server_API/app/api/v1/schemas/embeddings_abtest_schemas.py:1`
- Core services
  - Unified service: `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py:1`
  - Async runner: `tldw_Server_API/app/core/Evaluations/eval_runner.py:1`
  - Evaluators: G‑Eval `.../ms_g_eval.py:1`, RAG `.../rag_evaluator.py:1`, Response quality `.../response_quality_evaluator.py:1`, OCR `.../ocr_evaluator.py:1`
  - Webhooks: `tldw_Server_API/app/core/Evaluations/webhook_manager.py:1`
  - Rate limits (per‑user): `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py:1`
  - Metrics/circuit breaker: `.../metrics_advanced.py:1`, `.../metrics.py:1`, `.../circuit_breaker.py:1`
  - Embeddings A/B: `.../embeddings_abtest_service.py:1`, `.../embeddings_abtest_repository.py:1`
- Database
  - Manager + unified schema: `tldw_Server_API/app/core/DB_Management/Evaluations_DB.py:1`
  - Paths and backends: `tldw_Server_API/app/core/DB_Management/db_path_utils.py:1`, `.../content_backend.py:1`, `.../backends/*`
- CLI
  - Entry: `tldw_Server_API/cli/evals_cli.py:1`
  - Commands: `tldw_Server_API/cli/commands/evaluation.py:1`, `.../webhooks.py:1`, `.../testing.py:1`
  - Domain‑specific (module): `tldw_Server_API/app/core/Evaluations/cli/evals_cli.py:1`, `.../evals_cli_enhanced.py:1`, `.../benchmark_cli.py:1`
- Tests (good references)
  - Integration: `tldw_Server_API/tests/Evaluations/integration/test_api_endpoints.py:1`, `.../test_rate_limits_endpoint.py:1`
  - Runner/pipeline: `tldw_Server_API/tests/Evaluations/test_rag_pipeline_runner.py:1`
  - Embeddings A/B: `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_idempotency.py:1`, `.../test_embeddings_abtest_run_api.py:1`
  - OCR: `tldw_Server_API/tests/Evaluations/integration/test_ocr_pdf_dots_backend_integration.py:1`

**Key Endpoints**
- Base: `/api/v1/evaluations`
- CRUD: `POST /` create, `GET /` list, `GET /{eval_id}`, `PATCH /{eval_id}`, `DELETE /{eval_id}`
- Runs: `POST /{eval_id}/runs`, `GET /{eval_id}/runs`, `GET /runs/{run_id}`, `POST /runs/{run_id}/cancel`
- Datasets: `POST /datasets`, `GET /datasets`, `GET /datasets/{dataset_id}`, `DELETE /datasets/{dataset_id}`
- tldw evals: `POST /geval`, `POST /rag`, `POST /response-quality`, `POST /propositions`, `POST /batch`
- OCR: `POST /ocr`, `POST /ocr-pdf`
- RAG pipeline presets/cleanup: `POST /rag/pipeline/presets`, `GET /rag/pipeline/presets`, `GET /rag/pipeline/presets/{name}`, `DELETE /rag/pipeline/presets/{name}`, `POST /rag/pipeline/cleanup`
- Embeddings A/B (selected): `POST /embeddings/abtest`, `POST /embeddings/abtest/{test_id}/run`, `GET /embeddings/abtest/{test_id}`, `GET /embeddings/abtest/{test_id}/events`, `GET /embeddings/abtest/{test_id}/export`
  - More A/B endpoints: `GET /embeddings/abtest/{test_id}/results`, `GET /embeddings/abtest/{test_id}/significance`, `DELETE /embeddings/abtest/{test_id}`
- Webhooks: `POST /webhooks`, `GET /webhooks`, `DELETE /webhooks`, `POST /webhooks/test`
- Admin: `POST /admin/idempotency/cleanup`
- Health/Metrics/Rate: `GET /health`, `GET /metrics`, `GET /rate-limits`
 - History: `POST /history`

Auth
- Single‑user: send `X-API-KEY`; Multi‑user: `Authorization: Bearer <JWT>`.
- Heavy endpoints can require admin when `EVALS_HEAVY_ADMIN_ONLY=true`.
 - Scopes: selected endpoints (rag pipeline, embeddings A/B) require `workflows` scope via `require_token_scope`.

**Architecture & Data Flow**
- Request comes into `evaluations_unified.py` → validates via Pydantic schemas and `evaluations_auth.py`.
- For CRUD/datasets/runs, router calls `UnifiedEvaluationService` which orchestrates DB + `EvaluationRunner` and optional webhooks.
- tldw eval endpoints (`/geval`, `/rag`, `/response-quality`, `/ocr*`) call service helpers directly and store results into the unified DB (internal_evaluations / evaluations_unified), then optionally emit webhooks.
- Idempotency: create endpoints accept `Idempotency-Key` header; mapping recorded in `idempotency_keys` to safely replay.
- Rate limiting: coarse path limiter (`check_evaluation_rate_limit`) + per‑user limiter (`UserRateLimiter`) with tiered quotas; headers `X‑RateLimit-*` and basic `RateLimit-*` are set.
- Webhooks: events include started/completed/failed; HMAC‑SHA256 signatures with shared secrets; retries and security validation are handled by `webhook_manager.py`.

**Data Model (DB)**
- Core tables: `evaluations`, `evaluation_runs`, `datasets`
- Unified/aux tables: `internal_evaluations`, `pipeline_presets`, `ephemeral_collections`, `webhook_registrations`, `idempotency_keys`
- Embeddings A/B: `embedding_abtests`, `embedding_abtest_arms`, `embedding_abtest_queries`, `embedding_abtest_results`
 - Webhooks delivery log: `webhook_deliveries`
- Manager: `tldw_Server_API/app/core/DB_Management/Evaluations_DB.py:1` (SQLite or Postgres via content backend)

**Working Programmatically**
- Unified service usage
```python
from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService

svc = UnifiedEvaluationService(db_path="Databases/evaluations.db")
await svc.initialize()

# Create evaluation and run
evaluation = await svc.create_evaluation(
    name="my_eval", eval_type="model_graded",
    eval_spec={"sub_type": "summarization", "metrics": ["fluency","relevance"]},
    created_by="dev"
)
run = await svc.create_run(eval_id=evaluation["id"], target_model="gpt-4o", config={"batch_size": 5}, created_by="dev")
status = await svc.get_run(run["id"])
```

- Direct evaluators
```python
from tldw_Server_API.app.core.Evaluations.rag_evaluator import RAGEvaluator
rag_eval = RAGEvaluator()
results = await rag_eval.evaluate(
    query="What is Paris?",
    contexts=["Paris is the capital of France."],
    response="Paris is the capital of France.",
    ground_truth="Paris",
    metrics=["relevance","faithfulness","answer_similarity"],
)
```

**RAG Pipeline (sweeps & leaderboard)**
- Define `eval_spec.sub_type = "rag_pipeline"` with blocks for `chunking`, `retrievers`, `rerankers`, `rag` (each may hold lists → grid/random search).
- Runner builds a config grid, executes `unified_rag_pipeline` per sample/config, aggregates per‑config metrics, and produces a leaderboard with `config_score`.
- Ephemeral collections are registered in DB (TTL) and can be cleaned via `POST /rag/pipeline/cleanup`.
- Presets can be saved/loaded via `/rag/pipeline/presets` endpoints for repeatability.

**Embeddings A/B Testing**
- Create tests with two or more arms (provider/model); run to collect retrieval metrics/latency; export results; significance via `compute_significance`.
- Persistence uses either SQLAlchemy repo or DB manager tables based on `EVALS_ABTEST_PERSISTENCE`.
- Endpoints and SSE are mounted from `evaluations_unified.py`.

**OCR Evaluations**
- Text‑to‑text scoring: `POST /ocr` with items of `extracted_text` + `ground_truth_text`; metrics include CER/WER/coverage (page coverage optional).
- PDF path: `POST /ocr-pdf` (multipart) runs OCR (backend selectable) then computes metrics against provided ground truths; tune via form fields (dpi, lang, mode, min chars).

**Auth, Limits, and Error Handling**
- Auth modes handled in `evaluations_auth.verify_api_key`; admin gating is claim-first (`get_auth_principal` + `RequireRole("admin")` / heavy-admin claim checks).
- Path limiter (`check_evaluation_rate_limit`) + per‑user `UserRateLimiter` estimate tokens and enforce daily/cost quotas; usage headers are applied by helpers.
- Errors are normalized via `create_error_response` and `sanitize_error_message` for consistent API error shapes.

**API Usage Examples (curl)**
- Setup
```bash
API="http://127.0.0.1:8000/api/v1"; KEY="<API_KEY_OR_BEARER>"
```
- G‑Eval
```bash
curl -sS -X POST "$API/evaluations/geval" \
  -H "Content-Type: application/json" -H "X-API-KEY: $KEY" \
  -d '{"source_text":"...","summary":"...","metrics":["coherence","relevance"],"api_name":"openai"}'
```
- RAG
```bash
curl -sS -X POST "$API/evaluations/rag" \
  -H "Content-Type: application/json" -H "X-API-KEY: $KEY" \
  -d '{"question":"?","contexts":["..."],"answer":"...","ground_truth":"..."}'
```
- Create evaluation + run
```bash
curl -sS -X POST "$API/evaluations" -H "X-API-KEY: $KEY" -H "Content-Type: application/json" \
  -d '{"name":"demo","eval_type":"model_graded","eval_spec":{"sub_type":"rag","metrics":["relevance","faithfulness"]}}'
curl -sS -X POST "$API/evaluations/<eval_id>/runs" -H "X-API-KEY: $KEY" -H "Content-Type: application/json" \
  -d '{"target_model":"gpt-4o","config":{"batch_size":5}}'
```
- Datasets
```bash
curl -sS -X POST "$API/evaluations/datasets" -H "X-API-KEY: $KEY" -H "Content-Type: application/json" \
  -d '{"name":"ds1","samples":[{"input":{"text":"hello"},"expected":{"label":"greeting"}}]}'
```

**Common Gotchas & Tips**
- Idempotency: supply `Idempotency-Key` for create endpoints (evaluations/datasets/runs) to avoid duplicates on retries.
- Single‑user webhook IDs: service normalizes `user_id` for single‑user deployments so webhook registrations align.
- Embeddings optional: RAG evaluator auto‑falls back to LLM heuristics if embeddings backend isn’t available; in tests, `TEST_MODE` can synthesize.
- Circuit breaker: evaluator calls to LLMs are wrapped by a breaker to prevent cascading failures; handle ValueError from metric calls.
- Update merges: `update_evaluation` merges metadata with existing rather than replacing blindly.
- Rate limits: heavy/batch endpoints trigger stricter per‑minute checks; respect returned `Retry-After`.
- Testing DB: set `EVALUATIONS_TEST_DB_PATH` to isolate tests and exercise idempotency/webhooks predictably; `TEST_MODE` awaits webhook sends.

**Where To Extend**
- New evaluator: add `*_evaluator.py` in core, expose `evaluate(...)`; wire into `UnifiedEvaluationService` or `EvaluationRunner._get_evaluation_function` and add a schema knob if needed.
- New metrics: extend evaluator return shapes (normalized 0‑1 scores with optional raw_score/explanation); update aggregation where applicable.
- RAG pipeline: add new sweep knobs/blocks in `evaluation_schemas_unified.py` and teach runner to consume them; update leaderboard aggregation weights.
- DB features: extend `Evaluations_DB.py` with new tables/indices and wire into service/routers; include migrations for Postgres.
- Webhooks: add events/enrich payloads in `webhook_manager.py` and expose status via `evaluations_webhooks.py`.

**Testing Pointers**
- Endpoints: `tldw_Server_API/tests/Evaluations/integration/test_api_endpoints.py:1`
- Pipeline runner: `tldw_Server_API/tests/Evaluations/test_rag_pipeline_runner.py:1`
- Abtests: `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_idempotency.py:1`, `.../test_embeddings_abtest_run_api.py:1`
- OCR: `tldw_Server_API/tests/Evaluations/integration/test_ocr_pdf_dots_backend_integration.py:1`
- Error/limits/security: `tldw_Server_API/tests/Evaluations/test_error_scenarios.py:1`, `.../test_rate_limits_endpoint.py:1`, `.../test_security.py:1`

**Configuration Notes**
- Primary config file: `tldw_Server_API/Config_Files/evaluations_config.yaml:1` (tiers, quotas, TTLs, delivery)
- Env toggles (selected): `EVALS_HEAVY_ADMIN_ONLY`, `TEST_MODE`, `EVALUATIONS_TEST_DB_PATH`, `EVALS_ABTEST_PERSISTENCE`, provider API keys.
- DB path resolution is per‑user via `db_path_utils`; single‑user defaults to `Databases/evaluations.db`.
