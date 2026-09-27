# Stage 1 — Inventory, churn ranking, and the evaluator behaviour matrix

## Scope

Establish the reading list for this module from **size x churn**, not from a seed list (no shared
briefing cluster maps to Evaluations). Locate the tests by import-grep. Enumerate the evaluator
family and build a behaviour matrix so later stages can say *which* implementation is best before
proposing anything.

No findings are raised in this stage; it exists so stages 2-3 can cite a ranked reading list and a
matrix rather than re-deriving them. The one thing this stage does assert is ADR compliance for the
two mechanically-checkable ADRs that turned out clean (012, 013).

## Code Paths Reviewed

Module shape: 70 `.py` files, 35,805 LOC, 4 markdown guides
(`README.md`, `EVALS_DEVELOPER_GUIDE.md`, `EVALS_USER_GUIDE.md`, `SECURITY.md` — the module is
**not** missing documentation).

Top of the size x churn ranking (LOC from `2026-09-21-stage1-source-inventory.txt`,
commits/12mo from `2026-09-21-stage1-churn-baseline.txt`):

| File | LOC | Commits | Read in this audit |
| --- | ---: | ---: | --- |
| `eval_runner.py` | 2,423 | 26 | full, twice |
| `unified_evaluation_service.py` | 1,751 | 42 | orchestration + helpers |
| `user_rate_limiter.py` | 1,353 | 32 | outline + persistence |
| `recipe_runs_jobs_worker.py` | 1,290 | 11 | execution loops |
| `benchmark_utils.py` | 1,197 | 10 | loader/aggregation surface |
| `recipes/rag_answer_quality_execution.py` | 1,188 | — | scoring helpers |
| `webhook_manager.py` | 1,117 | 17 | schema + delivery + lookup |
| `rag_evaluator.py` | 1,092 | 21 | full |
| `embeddings_abtest_service.py` | 1,027 | 21 | search/score loop |
| `evaluation_manager.py` | 883 | 15 | init + persistence |
| `ms_g_eval.py` | 736 | 16 | full |
| `connection_pool.py` | 705 | 14 | full |
| `metrics.py` / `metrics_advanced.py` | 661 / 538 | — | outline only |
| `response_quality_evaluator.py` | 599 | 8 | full |

Supporting persistence layer (outside the module but inseparable from it):
`tldw_Server_API/app/core/DB_Management/Evaluations_DB.py` (2,968 LOC).

### ADR checks that came back clean

- **ADR-012 (resource ID prefixes)** — `Evaluations_DB.create_evaluation` emits `eval_`
  (`Evaluations_DB.py:1017`), `create_run` emits `run_` (`:1879`), `create_dataset` emits `dataset_`
  (`:2051`), and `UnifiedEvaluationService.create_run` pre-generates the matching `run_` id
  (`unified_evaluation_service.py:484`). Adjacent families use non-overlapping prefixes:
  `recipe_run_` (`Evaluations_DB.py:2188`), `abtest_`/`q_`/`res_`
  (`embeddings_abtest_repository.py:525,569,602`), `synth_action_`/`synth_promo_`/`synth_gen_`
  (`synthetic_eval_repository.py:352,483`, `synthetic_eval_service.py:87`). **Compliant.**
  One observation, not a finding: `unified_evaluation_service._store_evaluation_result:1521` mints
  `eval_{evaluation_type}_{hex}` — still `eval_`-prefixed and therefore ADR-conformant, but it is a
  second id *shape* for the same family. `evaluation_manager.store_evaluation:325` uses a bare
  `str(uuid.uuid4())`, which is outside ADR-012's scope because the `internal_evaluations` table it
  writes is not a public API resource family.
- **ADR-013 (deletion lifecycle)** — `delete_evaluation` is a soft delete
  (`Evaluations_DB.py:1851-1856`, `UPDATE evaluations SET deleted_at = CURRENT_TIMESTAMP`), and the
  read/list/update/count paths all filter `deleted_at IS NULL` (`:1044, :1065, :1103, :1149, :1841`).
  `delete_dataset` (`:2158`) is a hard delete. **Compliant in both directions.**
- **ADR-014 (OpenAI-compatible schemas)** — request/response models are separate and the response
  models carry `object` and Unix `created`
  (`api/v1/schemas/evaluation_schemas_unified.py:294, 300, 330, 334, 379, 384, 434`). The *shape* is
  compliant. The **value** written into `created` is not, on the SQLite backend — see EVAL-001 in
  stage 2.
- **ADR-015 (wrap, don't rewrite evaluators)** — the runner does delegate:
  `eval_runner._eval_summarization:1399` -> `ms_g_eval.run_geval`, `_eval_rag:1478` ->
  `RAGEvaluator.evaluate`, `_eval_response_quality:1533` -> `ResponseQualityEvaluator.evaluate`.
  **Compliant in the runner.** The drift is elsewhere — see EVAL-004 and EVAL-006 in stages 2-3.

### The evaluator family — behaviour matrix

Every implementation that turns a model output into a number, and how it does it:

| Implementation | Judge call | Raw scale | Normalization | On judge failure | Aggregate |
| --- | --- | --- | --- | --- | --- |
| `ms_g_eval.geval_summarization (448-551)` + `parse_output (425-446)` | `_call_adapter_text:85` (BYOK-validated) | 1-5, 1-3 for fluency | caller divides; `parse_output` enforces `<= max` and rejects multi-number replies | raises | `aggregate:137` / `aggregate_llm_scores:572` |
| `RAGEvaluator._evaluate_relevance (511-585)` | `llm_circuit_breaker.call_with_breaker` | 1-5 | `raw / 5.0` (`:556`) | **raises** (`raise_detached_error`) | `_calculate_overall_score:1042` |
| `RAGEvaluator._evaluate_faithfulness (587-656)` | same | 1-5 | `raw / 5.0` (`:635`) | raises | same |
| `RAGEvaluator._evaluate_answer_similarity (658-830)` | `_run_bounded_rag_analyze` | 1-5 | `raw / 5.0` (`:815`) | raises | same |
| `RAGEvaluator._evaluate_context_precision (832-884)` | per-context, sequential | 1-5 | `raw / 5.0` (`:862`) | appends `0.0` | unweighted mean of contexts |
| `RAGEvaluator._evaluate_context_relevance (886-945)` | per-context, sequential | 1-5 | `raw / 5.0` (`:920`) | appends `0.0` (explicit `ValueError` guard at `:922-925`) | unweighted mean |
| `RAGEvaluator._evaluate_context_recall (947-999)` | single call | 1-5 | `raw / 5.0` (`:985`) | raises | — |
| `RAGEvaluator._normalize_score (1027-1039)` | n/a | 1-5 | `(clamp(s,1,5) - 1) / 4` — **the mathematically correct min-max** | n/a | **zero production callers** |
| `ResponseQualityEvaluator._evaluate_{relevance,completeness,clarity,accuracy} (180,245,310,372)` + `_evaluate_custom_criterion (505)` | `llm_circuit_breaker.call_with_breaker` | 1-5 | `raw / 5.0` (`:226, :291, :353, :418, :550`) | **returns `score: 0.0`** | `evaluate:49-179` |
| `eval_runner._eval_summarization (1380-1450)` | `run_geval` in executor | 1-5 / 1-3 | `raw / max_score if raw >= 1.0 else raw` (`:1424`), and the legacy-string branch at `:1435-1439` divides unconditionally | returns `{"error": ...}` | `statistics.mean(scores.values())` |
| `eval_runner._eval_label_choice (1743-1938)` / `_eval_nli_factcheck (1939-2133)` | `eval_runner._call_adapter_text:99` (**no BYOK validation**) | binary | `1.0 if correct else 0.0` | `avg_score: 0.0` | mean |
| `recipes/rag_answer_quality_execution._coerce_unit_score (1102-1112)` | n/a | mixed | `<=1.0` passthrough, else `clamp(raw/5.0)` | n/a | — |
| `recipes/summarization_quality._normalize_score (289)` | n/a | caller-supplied `max_score` (1.0, 5.0, 3.0 at `:268-279`) | divide by the passed max | n/a | — |
| `evaluation_manager` scoring (`:613`, `:632`) | n/a | 1-10 | `raw / 10.0` (`:613`); `raw/10.0 if raw > 1 else raw` (`:632`) | n/a | — |
| `Prompt_Management/prompt_studio/evaluation_manager._calculate_score (420)` | own `_call_adapter_text (45)` | similarity | bespoke | `score: 0.0` (`:273`) | `avg_score` (`:280`), `passed = score >= 0.5` (`:235`) |

**Best implementation, stated up front:** `RAGEvaluator._normalize_score` is the correct
normalization (min-max over the stated 1-5 domain) and `ms_g_eval.parse_output` is the correct
parse (tolerant extraction plus range and ambiguity checks). Both are already in-module, already
tested, and both are bypassed by every LLM-judge in the family. Stage 2 builds on that.

## Tests Reviewed

Located by import-grep, never by path:

```
grep -rl "core\.Evaluations" tldw_Server_API/tests
```

109 files (full list in `2026-09-21-stage1-test-inventory.txt`), distributed:
`tests/Evaluations` 62, `tests/Evaluations/unit` 21, `tests/Evaluations/integration` 9,
`tests/Evaluations/property` 1, plus 16 files in `tests/LLM_Calls`, `tests/DB_Management`,
`tests/Workflows`, `tests/Web_Scraping`, `tests/Services`, `tests/Resource_Governance`,
`tests/RAG_NEW/unit`, `tests/MediaIngestion_NEW/unit`, `tests/Infrastructure`, `tests/http_client`,
`tests/Config`, `tests/AuthNZ/unit`, `tests/AuthNZ/integration`.

**This module is well covered. It must not be reported as untested.** What follows is per-area
reachability, not measured coverage.

| Area | Tests reaching it | Does it downgrade the risk? |
| --- | --- | --- |
| `rag_evaluator.py` | `unit/test_rag_evaluator.py`, `test_rag_evaluator_embeddings.py`, `property/test_evaluation_invariants.py`, `test_error_scenarios.py`, `test_evaluation_integration.py`, `unit/test_specialized_provider_auth_failures.py` | Partly. `unit/test_rag_evaluator.py:363-375` and `property/test_evaluation_invariants.py:102-136` pin `_normalize_score`'s contract — but nothing production calls it, so the assurance is misdirected (EVAL-002). |
| `response_quality_evaluator.py` | `unit/test_response_quality_provider_boundary.py`, `unit/test_specialized_provider_auth_failures.py`, `property/test_evaluation_invariants.py` | Provider boundary is covered; the fail-soft-to-0.0 semantic is not contrasted against RAGEvaluator's fail-loud (EVAL-005). |
| `eval_runner.py` | `unit/test_eval_runner.py`, `test_evaluations_core_hardening.py`, `test_evaluations_stage2_user_isolation_and_usage_accounting.py`, `test_evaluations_stage3_batch_failfast_and_metrics_none.py`, `test_evaluations_unified.py`, `test_rag_pipeline_runner.py`, `unit/test_unified_evaluation_service_mapping.py` | Batch fail-fast and metrics-none paths are explicitly covered. The rag_pipeline aggregation (EVAL-003) has a test file but no assertion on `chunk_cohesion` provenance. |
| `webhook_manager.py` | 19 files incl. `unit/test_webhook_manager_backend_schema.py`, `integration/test_webhook_multi_user_api.py`, `test_evaluations_webhooks_endpoint_sanitization.py` | Backend schema is covered by a dedicated unit test — which is why the *second* owner of the same DDL (EVAL-007) is surprising. |
| `Evaluations_DB.py` timestamps | `tests/DB_Management` (2 files), plus every endpoint test asserting `created` | **No.** Every CI host runs UTC, where the defect in EVAL-001 evaluates to a zero offset. |
| `connection_pool.py` | `tests/Services/test_startup_evaluations_warmup.py`, `tests/Services/test_shutdown_evaluations_resources.py`, `tests/AuthNZ/unit/test_test_mode_runtime_guard.py` | The tests assert the pool *starts and stops*. None asserts anything is served from it — consistent with EVAL-009. |

## Validation Commands

```
$ find tldw_Server_API/app/core/Evaluations -name '*.py' | xargs wc -l | sort -rn | head -1
   35805 total

$ find tldw_Server_API/app/core/Evaluations -name '*.py' | wc -l
      70

$ grep -rl "core\.Evaluations" tldw_Server_API/tests | wc -l
     109

$ git log --since='12 months ago' --name-only --pretty=format: -- tldw_Server_API/app/core/Evaluations \
    | grep '\.py$' | sort | uniq -c | sort -rn | head -3
  42 tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py
  32 tldw_Server_API/app/core/Evaluations/user_rate_limiter.py
  26 tldw_Server_API/app/core/Evaluations/eval_runner.py

$ python -m pytest tldw_Server_API/tests/Evaluations --collect-only -q --no-header
749 tests collected, 8 errors in 5.79s

$ python -m pytest tldw_Server_API/tests/Evaluations --collect-only -q --no-header 2>&1 \
    | grep -E "ModuleNotFoundError" | sort | uniq -c
   2 E   ModuleNotFoundError: No module named 'hypothesis'
   6 E   ModuleNotFoundError: No module named 'sklearn'

$ grep -n "scikit-learn\|hypothesis" pyproject.toml | head -2
61:  "hypothesis",
87:  "scikit-learn>=1.3.0",
```

All 8 collection errors are the local venv missing two **declared core dependencies**. This is an
environment gap, not a code defect, and it bounds what stages 2-3 could execute locally.

```
$ python -m pytest tldw_Server_API/tests/Evaluations/unit/test_rag_evaluator.py -q --no-header
6 failed, 34 passed, 7 warnings in 2.34s

$ python -m pytest "tldw_Server_API/tests/Evaluations/unit/test_rag_evaluator.py::TestAnswerSimilarity::test_answer_similarity_identical_texts" -q --no-header 2>&1 | grep '^E '
E   ModuleNotFoundError: No module named 'sklearn'
```

All 6 failures share that single cause. No test failure observed in this audit is attributable to
a code defect.

## Findings

None in this stage by design. ADR-012 and ADR-013 were checked and are **compliant**; recording a
clean result matters as much as recording a defect, because it removes two whole hypothesis classes
from stages 2-3.

## Suggested Refactor/Actions

None. This stage is inventory. Actions are proposed in stages 2-4 against specific findings.
