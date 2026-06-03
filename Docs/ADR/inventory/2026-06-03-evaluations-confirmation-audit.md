# Evaluations ADR Confirmation Audit - 2026-06-03

**Related task:** TASK-517
**Follow-up backfill task:** TASK-518
**Scope:** `Docs/Evals/Evals-Plan-1.md` embedded ADRs mapped to inventory rows INV-009 through INV-015.

## Purpose

Confirm which embedded Evaluations ADRs still describe current governing behavior before promoting any of them into canonical ADRs.

This audit does not create accepted ADRs. It separates confirmed current decisions from stale, superseded, or partial historical decisions so the follow-up backfill can stay one-decision-per-ADR.

## Evidence Reviewed

| Area | Evidence |
| --- | --- |
| Embedded ADR source | `Docs/Evals/Evals-Plan-1.md`, Architecture Decision Records section |
| Evaluations storage and CRUD | `tldw_Server_API/app/core/DB_Management/Evaluations_DB.py` |
| Current module documentation | `tldw_Server_API/app/core/Evaluations/README.md` |
| API schemas | `tldw_Server_API/app/api/v1/schemas/openai_eval_schemas.py`, `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py` |
| Run orchestration | `tldw_Server_API/app/core/Evaluations/eval_runner.py`, `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py` |
| Jobs boundary | `tldw_Server_API/app/services/startup_sidecar_owned_jobs_pollers.py`, `tldw_Server_API/app/core/Evaluations/recipe_runs_jobs_worker.py` |
| Tests sampled | `tldw_Server_API/tests/Evaluations/test_evaluations_unified.py`, `tldw_Server_API/tests/Evaluations/test_evaluations_postgres_crud.py`, `tldw_Server_API/tests/Evaluations/test_recipe_runs_jobs_worker.py`, `tldw_Server_API/tests/DB_Management/test_evaluations_unified_and_crud.py` |

## Dispositions

| Inventory ID | Embedded decision | Disposition | Evidence summary | Next action |
| --- | --- | --- | --- | --- |
| INV-009 | Use SQLite for evaluation data storage. | Superseded | `EvaluationsDatabase` is now backend-aware and documents SQLite or PostgreSQL support. It resolves the shared content backend, keeps SQLite initialization, and has a PostgreSQL bootstrap path with JSONB columns. The Evaluations README also describes optional PostgreSQL and RLS support plus per-user DB paths. | Do not backfill the old SQLite-only decision as accepted. A future persistence ADR should be backend-aware if the owner wants one. |
| INV-010 | Use prefixed UUIDs for evaluations, runs, and datasets. | Current governing | `create_evaluation`, `create_run`, and `create_dataset` still generate `eval_`, `run_`, and `dataset_` IDs. `UnifiedEvaluationService.create_run` pre-generates a `run_` ID before persistence. API tests assert `eval_` IDs. | Include in TASK-518 as a resource ID convention ADR. |
| INV-011 | Use soft deletes for evaluations and hard deletes for datasets. | Current governing | The evaluations table has `deleted_at`; get/list/update paths filter `deleted_at IS NULL`; `delete_evaluation` updates `deleted_at`; `delete_dataset` executes `DELETE FROM datasets`. Unified tests cover delete behavior. | Include in TASK-518 as a deletion lifecycle ADR. |
| INV-012 | Store complex objects as JSON TEXT in SQLite. | Needs owner review | SQLite DDL stores complex fields as `TEXT` and CRUD methods serialize with `json.dumps` and parse with `_json_maybe`. PostgreSQL DDL stores matching fields as `JSONB`, and `_json_maybe` accepts already parsed JSON-like values. The old SQLite-only text is true for SQLite but incomplete for the current backend-aware design. | Do not backfill the old text as accepted. Fold into a backend-aware persistence representation ADR only after owner review. |
| INV-013 | Use separate request/response schemas following OpenAI conventions. | Current governing | `openai_eval_schemas.py` explicitly defines OpenAI-style request and response models, `object` fields, Unix timestamps, and list wrappers. `evaluation_schemas_unified.py` keeps separate create/update/response/run/dataset models with compatible `object` and `created` fields. Tests assert list/object response shape. | Include in TASK-518 as an API schema convention ADR. |
| INV-014 | Use asyncio/background tasks for runs, progress, webhooks, and cancellation. | Needs owner review | Core evaluation runs still use `asyncio.create_task`, tracked `running_tasks`, progress updates, webhook dispatch, and cancellation. However, current module docs say user-visible persona dialogue-tree recipe runs must use Jobs, and startup code starts an Evaluation recipe-run Jobs worker. The old broad decision is therefore only partially current. | Do not backfill the broad embedded ADR as accepted. Split core eval-run async behavior from recipe-run Jobs ownership if the owner wants ADR coverage. |
| INV-015 | Wrap existing evaluation modules rather than rewrite. | Current governing | `eval_runner.py` imports and delegates to existing `ms_g_eval`, `RAGEvaluator`, `ResponseQualityEvaluator`, proposition evaluation, and the unified RAG pipeline. `unified_evaluation_service.py` maps GEval, RAG, response quality, OCR, and other types to dedicated evaluator services. | Include in TASK-518 as an evaluator integration strategy ADR. |

## Follow-Up Scope

TASK-518 should backfill only these confirmed current decisions:

- INV-010: Evaluations resource ID prefixes.
- INV-011: Evaluation and dataset deletion lifecycle.
- INV-013: OpenAI-compatible request/response schema shape.
- INV-015: Reuse/wrap existing evaluator modules.

TASK-518 should exclude these rows from direct accepted backfill:

- INV-009: superseded by backend-aware SQLite/PostgreSQL storage.
- INV-012: partially current for SQLite but incomplete without the PostgreSQL JSONB representation.
- INV-014: partially current for core runs but incomplete without the recipe-run Jobs boundary.

## Verification Notes

This is a documentation-only audit. No Python code is changed by TASK-517, so Bandit is not applicable beyond recording the docs-only skip.
