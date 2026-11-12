# Evaluations

This document describes the server's evaluation model: the unified service and runner, the evaluation-type catalog, webhook delivery with HMAC signing and SSRF-safe targets, recipe runs (the run-group analog), the persona judge contract (the chatbook "character probe" counterpart), and the core-only WordBench runner.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Unified service | `app/core/Evaluations/unified_evaluation_service.py` | `EvaluationType` (model_graded, exact_match, includes, geval, rag, response_quality, proposition_extraction, qa3, ocr, label_choice, nli_factcheck, custom); `UnifiedEvaluationService` — create/run/cancel, datasets, history; circuit-breaker config |
| Runner | `app/core/Evaluations/eval_runner.py` | `EvaluationRunner` — background task registry, per-run progress, config-grid cartesian runs, evaluator dispatch, per-run webhook delivery |
| Webhooks | `app/core/Evaluations/webhook_manager.py` | `WebhookEvent` set (evaluation started/progress/completed/failed/cancelled, batch events); SSRF-safe delivery, HMAC `X-Webhook-Signature`, retries (default 3, delays 1/5/15 s), per-webhook timeout clamped ≤5 s, delivery records + stats |
| Routes | `app/api/v1/endpoints/evaluations/` | `/api/v1/evaluations`: CRUD + runs (+cancel), unified evaluator endpoints (geval/rag/response-quality/propositions/batch/ocr/ocr-pdf/history/rate-limits/health/metrics + admin idempotency cleanup), webhook registration/test, recipes/datasets/benchmarks/synthetic/abtest subrouters |
| DB | `app/core/DB_Management/Evaluations_DB.py` | `evaluations`, `evaluation_runs` (status/model/config/progress/results/webhook/usage), `datasets`, **`evaluation_recipe_runs` + children**, `webhook_registrations`, `idempotency_keys`, embedding-abtest and synthetic tables |
| Recipe runs | `app/core/Evaluations/recipe_runs_service.py` + `recipes/` | reuse-hash dedupe, reports, sensitive-config redaction; recipe registry (persona dialogue-tree robustness, RAG answer quality/retrieval tuning, summarization, embeddings) |
| Persona judge | `persona_chat_judge*.py` + `cli/persona_chat_judge_cli.py` | binary Pass/Fail per dimension; inputs carry `assistant_kind`/`assistant_id`/`persona_memory_mode`/labels/evidence; **CLI-exposed, not REST** |
| WordBench | `wordbench_runner.py` + `configs/wordbench.yaml` | next-token/logprob capture (`top_k=10`); **core-only — no route, no CLI command** |
| Infra | `user_rate_limiter.py`, `circuit_breaker.py`, `identity.py`, `benchmark_registry.py` | rate limits, breaker, canonical `EvaluationIdentity` (never use raw API-key returns as ownership subjects) |

## Run lifecycle with webhook (dataflow)

1. `POST /evaluations/{id}/runs` (optional `Idempotency-Key` — replays flagged via header; optional `webhook_url`) → mandatory audit first, then the durable run row.
2. If a webhook is configured, `EVALUATION_STARTED` fires **only after the row exists**.
3. A background task (registered for cancellation) runs: status running, samples fetched, progress streamed into the DB, per-type evaluator dispatch, results + usage stored.
4. Completion/failure/cancellation deliver the matching webhook event with results; cancellation honors status-transition rules.
5. Delivery: DNS/SSRF validation (skipped only under real pytest, never deployment TEST_MODE), HMAC signature, retry loop with configured delays, delivery rows and stats.
6. Separately registered webhooks (persisted per user via the canonical identity) receive the same event set.

Recipe runs reuse Evaluations run records plus a Jobs worker — no parallel status system — with parent `evaluation_recipe_runs` + children composing grids, and configs redacted before persistence.

## The chatbook-mirror note (verified)

- **No `run_group` concept exists server-side** — the grouping analog is recipe runs + children; the chatbook client's run groups are client-side.
- **No "character probes" server-side by that name** — the mirrored shapes are the persona judge contract (dimensions with success/failure definitions, evidence fields, binary verdicts; judges can score/soft-prune, deterministic policy checks are the only hard blockers) and the dialogue-tree recipe's targets/candidates shape. Both are offline-only.
- The chatbook claim "local webhooks unsupported → the server does them" matches: delivery, retries, HMAC, and records are all here.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Circuit breaker open | Call short-circuits; run fails with an error message |
| Cancel | Registered task cancelled; status moves only on allowed transitions |
| Unsafe webhook URL | Delivery skipped; stats record the failure |
| Webhook non-2xx / network error | Retries then a failed delivery row |
| Batch strict fail-fast | `continue_on_error=false` cancels remaining work |
| Rate limit | Dependency check + visible `/rate-limits` |

## Config keys (exact)

Webhook delivery under config `webhooks.delivery`: `max_retries` (3), `retry_delays` ([1,5,15]), `timeout_seconds` (30, clamped ≤5 at delivery), `batch_size` (10). Env: `EVALUATIONS_TEST_DB_PATH` (tests). Health is public; `/metrics` needs `EVALS_READ`; dataset mutations need `EVALS_MANAGE`.

## Verified gotchas

1. `WordBenchRunner` is referenced by nothing — core-only; do not document a `/evaluations/wordbench` route.
2. Webhook timeouts are effectively ≤5 s regardless of registration values.
3. `Docs/Code_Documentation/Evaluations_Developer_Guide.md` is stale (February; missing recipes/wordbench/persona-judge) — the fresh references are the in-module `EVALS_DEVELOPER_GUIDE.md` / `README.md` and `Docs/Code_Documentation/Guides/Evaluations_Code_Guide.md`.
4. DNS/SSRF validation is skipped only under real pytest — never under a deployment `TEST_MODE` flag.

## Governing decisions

ADR-012 (`Docs/ADR/012-evaluations-resource-id-prefixes.md`), ADR-013 (deletion lifecycle), ADR-014 (OpenAI-compatible schemas), ADR-015 (existing evaluator integration).

## Related docs

- [llm-providers.md](./llm-providers.md) — the provider calls evaluators make
- [jobs-background.md](./jobs-background.md) — the recipe/abtest workers
