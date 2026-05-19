# Evals Module Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved Evals module review and produce one cumulative, evidence-backed review document covering the unified API, orchestration, persistence, feature surfaces, and cross-slice contract gaps in `tldw_server`.

**Architecture:** This is a read-first, docs-only review plan. Execution starts by freezing the current Evals working-tree baseline inside a single cumulative review document, then moves through the approved slices in order, updating the same review file after each pass. Each slice must record findings before improvements, use the smallest relevant test set to validate backend-sensitive claims, and clearly label any findings that are specific to the current dirty working tree rather than the last committed state.

**Tech Stack:** Python 3, FastAPI, SQLite/PostgreSQL adapters, pytest, ripgrep, git, Markdown

---

## Review File Map

**Create during execution:**
- `Docs/superpowers/reviews/evals-module/README.md`

**Primary source files to inspect during the review:**
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py`
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py`
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py`
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_recipes.py`
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py`
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py`
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_synthetic.py`
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_embeddings_abtest.py`
- `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_webhooks.py`
- `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py`
- `tldw_Server_API/app/api/v1/schemas/evaluation_schema.py`
- `tldw_Server_API/app/api/v1/schemas/evaluation_recipe_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/embeddings_abtest_schemas.py`
- `tldw_Server_API/app/api/v1/schemas/synthetic_eval_schemas.py`
- `tldw_Server_API/app/core/Evaluations/README.md`
- `tldw_Server_API/app/core/Evaluations/EVALS_DEVELOPER_GUIDE.md`
- `tldw_Server_API/app/core/Evaluations/SECURITY.md`
- `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- `tldw_Server_API/app/core/Evaluations/eval_runner.py`
- `tldw_Server_API/app/core/Evaluations/evaluation_manager.py`
- `tldw_Server_API/app/core/Evaluations/db_adapter.py`
- `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py`
- `tldw_Server_API/app/core/Evaluations/ms_g_eval.py`
- `tldw_Server_API/app/core/Evaluations/rag_evaluator.py`
- `tldw_Server_API/app/core/Evaluations/response_quality_evaluator.py`
- `tldw_Server_API/app/core/Evaluations/recipe_runs_service.py`
- `tldw_Server_API/app/core/Evaluations/recipe_runs_jobs.py`
- `tldw_Server_API/app/core/Evaluations/recipes/base.py`
- `tldw_Server_API/app/core/Evaluations/recipes/registry.py`
- `tldw_Server_API/app/core/Evaluations/recipes/rag_answer_quality.py`
- `tldw_Server_API/app/core/Evaluations/recipes/rag_answer_quality_execution.py`
- `tldw_Server_API/app/core/Evaluations/recipes/rag_retrieval_tuning.py`
- `tldw_Server_API/app/core/Evaluations/recipes/rag_retrieval_tuning_execution.py`
- `tldw_Server_API/app/core/Evaluations/recipes/embeddings_retrieval.py`
- `tldw_Server_API/app/core/Evaluations/benchmark_registry.py`
- `tldw_Server_API/app/core/Evaluations/benchmark_loaders.py`
- `tldw_Server_API/app/core/Evaluations/benchmark_utils.py`
- `tldw_Server_API/app/core/Evaluations/synthetic_eval_service.py`
- `tldw_Server_API/app/core/Evaluations/synthetic_eval_repository.py`
- `tldw_Server_API/app/core/Evaluations/synthetic_eval_generation.py`
- `tldw_Server_API/app/core/Evaluations/embeddings_abtest_repository.py`
- `tldw_Server_API/app/core/Evaluations/embeddings_abtest_runner.py`
- `tldw_Server_API/app/core/Evaluations/embeddings_abtest_service.py`
- `tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs.py`
- `tldw_Server_API/app/core/Evaluations/webhook_identity.py`
- `tldw_Server_API/app/core/Evaluations/webhook_manager.py`
- `tldw_Server_API/app/core/Evaluations/webhook_security.py`
- `tldw_Server_API/app/core/Evaluations/audit_adapter.py`
- `tldw_Server_API/Config_Files/evaluations_config.yaml`

**High-value existing tests to reuse during the review:**
- `tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`
- `tldw_Server_API/tests/Evaluations/test_evaluations_unified.py`
- `tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_evaluations_permissions_claims.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_evaluations_invariants.py`
- `tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py`
- `tldw_Server_API/tests/Evaluations/unit/test_unified_evaluation_service_mapping.py`
- `tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py`
- `tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py`
- `tldw_Server_API/tests/Evaluations/unit/test_evaluations_db_filters.py`
- `tldw_Server_API/tests/Evaluations/test_evaluations_backend_dual.py`
- `tldw_Server_API/tests/Evaluations/test_evaluations_postgres_crud.py`
- `tldw_Server_API/tests/DB_Management/test_evaluations_unified_and_crud.py`
- `tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py`
- `tldw_Server_API/tests/Evaluations/test_evaluation_integration.py`
- `tldw_Server_API/tests/Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py`
- `tldw_Server_API/tests/Evaluations/test_rag_evaluator_embeddings.py`
- `tldw_Server_API/tests/Evaluations/unit/test_rag_evaluator.py`
- `tldw_Server_API/tests/Evaluations/test_recipe_embeddings_retrieval.py`
- `tldw_Server_API/tests/Evaluations/test_recipe_rag_retrieval_tuning.py`
- `tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py`
- `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`
- `tldw_Server_API/tests/Evaluations/integration/test_synthetic_eval_api.py`
- `tldw_Server_API/tests/Evaluations/test_synthetic_eval_service.py`
- `tldw_Server_API/tests/Evaluations/unit/test_evals_cli_benchmark_commands.py`
- `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_retrieval.py`
- `tldw_Server_API/tests/Evaluations/unit/test_evaluations_abtest_store_init.py`
- `tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py`
- `tldw_Server_API/tests/Evaluations/unit/test_webhook_manager_backend_schema.py`
- `tldw_Server_API/tests/Evaluations/property/test_evaluation_invariants.py`
- `tldw_Server_API/tests/e2e/test_evaluations_workflow.py`
- `tldw_Server_API/tests/server_e2e_tests/test_evaluations_workflow.py`

## Stage Overview

## Stage 1: Review Artifact Setup and Baseline Snapshot
**Goal:** Create the cumulative review document, capture the active `HEAD` and dirty-file baseline, and freeze the review structure before deep reading starts.
**Success Criteria:** `Docs/superpowers/reviews/evals-module/README.md` exists, lists the approved slice order, records the working-tree baseline, and defines the exact finding schema that later slices must follow.
**Tests:** None
**Status:** Not Started

## Stage 2: Unified API and Auth Surface
**Goal:** Review the public Evals entrypoint, auth gating, rate-limit behavior, and provider-credential handling at the highest-blast-radius boundary.
**Success Criteria:** Request routing, auth behavior, provider-key checks, test-mode behavior, and rate-limit or principal assumptions are documented with evidence and tied to focused tests where needed.
**Tests:** Stage 1 route, unified API, and AuthNZ evaluation guard tests.
**Status:** Not Started

## Stage 3: Core Orchestration and Execution
**Goal:** Review run dispatch, batch semantics, evaluator selection, and orchestration behavior in the unified service and runner layer.
**Success Criteria:** Control-flow assumptions, timeout behavior, evaluator mapping, batch error handling, and hidden coupling are recorded with evidence and validated against runner and service tests.
**Tests:** Runner, service mapping, and batch-failfast tests.
**Status:** Not Started

## Stage 4: Persistence and State Management
**Goal:** Review database initialization, adapter routing, migration behavior, filtering, and storage invariants that affect Evals state.
**Success Criteria:** Persistence flows, migration and backend assumptions, filter behavior, and cross-backend parity risks are captured with evidence and checked against focused DB tests.
**Tests:** Evaluation manager, DB filter, backend dual-mode, Postgres CRUD, migration, and DB-management evaluation tests.
**Status:** Not Started

## Stage 5: CRUD and Run Lifecycle Endpoints
**Goal:** Review creation, read, history, export, and run lifecycle endpoints after the persistence layer has already been traced.
**Success Criteria:** Endpoint-to-storage contracts, run-state assumptions, tenancy handling, and audit-side effects are documented with evidence and validated against focused CRUD and integration tests.
**Tests:** CRUD create-run, unified-and-CRUD DB, integration, and user-isolation tests.
**Status:** Not Started

## Stage 6: Retrieval and Recipe-Driven Evaluation Surfaces
**Goal:** Review RAG evaluation, response-quality, and recipe execution paths, including where they depend on shared orchestration or schema assumptions.
**Success Criteria:** Retrieval and recipe flows are documented with evidence, duplication or coupling between endpoint and recipe layers is identified, and the most important gaps are tied to targeted tests.
**Tests:** RAG evaluator and recipe tests.
**Status:** Not Started

## Stage 7: Benchmark, Dataset, and Synthetic Evaluation Surfaces
**Goal:** Review benchmark APIs, dataset generation or permission paths, and synthetic evaluation service flows.
**Success Criteria:** Benchmark-loading assumptions, dataset permissions, synthetic generation paths, and related test coverage gaps are recorded with evidence and prioritized by blast radius.
**Tests:** Benchmarks, dataset-permissions, synthetic API, synthetic service, and benchmark CLI tests.
**Status:** Not Started

## Stage 8: Embeddings A/B and Webhook Surfaces
**Goal:** Review embeddings A/B evaluation state, webhook registration and delivery, and their security-sensitive integration seams.
**Success Criteria:** Job or repository assumptions, multi-user webhook behavior, backend schema expectations, and delivery-security risks are documented with evidence and validated against focused tests.
**Tests:** Embeddings A/B and webhook tests.
**Status:** Not Started

## Stage 9: Cross-Slice Contract Synthesis and Final Review
**Goal:** Reconcile shared schema and config assumptions across slices, remove duplicate findings, and finalize the ranked cumulative review.
**Success Criteria:** Shared schema or config drift is synthesized, working-tree-specific findings are labeled correctly, residual coverage gaps are prioritized, and the final README is coherent enough to serve as the canonical review record.
**Tests:** Property and end-to-end evaluation sanity pack only where needed to settle disputed cross-slice claims.
**Status:** Not Started

### Task 1: Create the Cumulative Review Artifact and Freeze the Baseline

**Files:**
- Create: `Docs/superpowers/reviews/evals-module/README.md`
- Inspect: `Docs/superpowers/plans/2026-04-07-evals-module-review-execution-plan.md`
- Test: none

- [ ] **Step 1: Create the review output directory**

Run:
```bash
mkdir -p Docs/superpowers/reviews/evals-module
```

Expected: the `Docs/superpowers/reviews/evals-module` directory exists and no source files change.

- [ ] **Step 2: Verify the execution environment is safe for a docs-only sequential review**

Run:
```bash
git rev-parse --show-toplevel
git rev-parse --short HEAD
git status --short tldw_Server_API/app/api/v1/endpoints/evaluations tldw_Server_API/app/core/Evaluations tldw_Server_API/tests/Evaluations tldw_Server_API/tests/AuthNZ tldw_Server_API/tests/DB_Management
```

Expected: the repo root resolves cleanly, `HEAD` is captured for later recording, and the current dirty Evals-related file set is visible before review notes begin.

- [ ] **Step 3: Create `Docs/superpowers/reviews/evals-module/README.md` with the fixed review template**

Write this structure:
```markdown
# Evals Module Review

## Baseline Snapshot
- Head commit:
- Dirty Evals-related files at review start:

## Scope and Slice Order
1. Unified API and auth surface
2. Core orchestration and execution
3. Persistence and state management
4. CRUD and run lifecycle endpoints
5. Retrieval and recipe-driven evaluation surfaces
6. Benchmark, dataset, and synthetic evaluation surfaces
7. Embeddings A/B and webhook surfaces
8. Cross-slice contract synthesis

## Review Method
- findings before improvements
- uncertain items labeled `needs verification`
- working-tree-specific findings labeled explicitly

## Severity and Priority Model
- Critical / High / Medium / Low
- Immediate / Near-term / Later

## Slice 1: Unified API and Auth Surface
### Files Reviewed
### Baseline Notes
### Control and Data Flow Notes
### Findings
### Open Questions
### Slice Status

## Slice 2: Core Orchestration and Execution
### Files Reviewed
### Baseline Notes
### Control and Data Flow Notes
### Findings
### Open Questions
### Slice Status

## Slice 3: Persistence and State Management
### Files Reviewed
### Baseline Notes
### Control and Data Flow Notes
### Findings
### Open Questions
### Slice Status

## Slice 4: CRUD and Run Lifecycle Endpoints
### Files Reviewed
### Baseline Notes
### Control and Data Flow Notes
### Findings
### Open Questions
### Slice Status

## Slice 5: Retrieval and Recipe-Driven Evaluation Surfaces
### Files Reviewed
### Baseline Notes
### Control and Data Flow Notes
### Findings
### Open Questions
### Slice Status

## Slice 6: Benchmark, Dataset, and Synthetic Evaluation Surfaces
### Files Reviewed
### Baseline Notes
### Control and Data Flow Notes
### Findings
### Open Questions
### Slice Status

## Slice 7: Embeddings A/B and Webhook Surfaces
### Files Reviewed
### Baseline Notes
### Control and Data Flow Notes
### Findings
### Open Questions
### Slice Status

## Slice 8: Cross-Slice Contract Synthesis
### Shared Schemas and Config
### Cross-Slice Systemic Issues
### Priority Summary
### Recommended Remediation Order
### Coverage Gaps and Verification Items
### Slice Status
```

- [ ] **Step 4: Record the baseline snapshot in the new README**

Run:
```bash
git rev-parse --short HEAD
git status --short tldw_Server_API/app/api/v1/endpoints/evaluations tldw_Server_API/app/core/Evaluations tldw_Server_API/tests/Evaluations tldw_Server_API/tests/AuthNZ tldw_Server_API/tests/DB_Management
```

Expected: the README contains the exact short `HEAD` commit and the Evals-related dirty-file inventory captured before slice 1 begins.

- [ ] **Step 5: Freeze the final finding schema before deep reading**

Use this per-finding structure inside `## Findings`:
```markdown
1. Severity: High
   Confidence: High
   Priority: Immediate
   Why it matters: ...
   File references: `path/to/file.py:line`
   Recommended fix: ...
   Recommended tests: ...
   Verification note: ...
```

Expected: every later finding can be added without inventing a new format.

- [ ] **Step 6: Stage only the review artifact and plan file, then verify the staged set**

Run:
```bash
git add Docs/superpowers/reviews/evals-module/README.md
git diff --cached --name-only
```

Expected: the cached file list shows only `Docs/superpowers/reviews/evals-module/README.md`.

- [ ] **Step 7: Commit the scaffold and baseline snapshot**

Run:
```bash
git commit -m "docs: scaffold evals module review artifacts"
```

Expected: one docs-only commit captures the review scaffold and baseline snapshot before substantive findings are added.

### Task 2: Execute Slice 1 Unified API and Auth Surface

**Files:**
- Modify: `Docs/superpowers/reviews/evals-module/README.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/user_rate_limiter.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_unified.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_evaluations_permissions_claims.py`
- Test: `tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_evaluations_invariants.py`

- [ ] **Step 1: Read the unified API and auth files in order**

Trace:
- router registration and route prefixes
- current-user resolution
- permission and role checks
- heavy-eval admin gating
- provider credential validation
- rate-limit header behavior
- test-mode branches

Expected: the control-flow notes for Slice 1 name the exact request guards and the code paths that can bypass, weaken, or duplicate them.

- [ ] **Step 2: Search the Slice 1 files for broad exception handling and test-mode branches**

Run:
```bash
source .venv/bin/activate
rg -n "except Exception|_is_test_mode|pytest|TEST|fallback|record_byok_missing_credentials|HTTPException|require_eval_permissions|check_evaluation_rate_limit" \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_unified.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_auth.py \
  tldw_Server_API/app/core/Evaluations/user_rate_limiter.py
```

Expected: a shortlist of guardrail-heavy branches that deserve manual inspection before any findings are written.

- [ ] **Step 3: Run the focused Slice 1 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage1_route_and_error_regressions.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_unified.py \
  tldw_Server_API/tests/AuthNZ/unit/test_evaluations_auth_runtime_guards.py \
  tldw_Server_API/tests/AuthNZ/integration/test_evaluations_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ/integration/test_auth_principal_evaluations_invariants.py
```

Expected: PASS for the existing suite, or a stable failing signal that sharpens a suspected Slice 1 finding.

- [ ] **Step 4: Update Slice 1 in the cumulative review README**

Record:
- exact files reviewed
- baseline-sensitive notes if the current dirty working tree affects interpretation
- control and data flow summary
- findings ordered by severity
- open questions only if evidence is incomplete
- `Slice Status: reviewed`

- [ ] **Step 5: Commit the Slice 1 review update**

Run:
```bash
git add Docs/superpowers/reviews/evals-module/README.md
git diff --cached --name-only
git commit -m "docs: record evals slice 1 review"
```

Expected: one docs-only commit contains the Slice 1 findings and only the cumulative review README.

### Task 3: Execute Slice 2 Core Orchestration and Execution

**Files:**
- Modify: `Docs/superpowers/reviews/evals-module/README.md`
- Inspect: `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/eval_runner.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/ms_g_eval.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/rag_evaluator.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/response_quality_evaluator.py`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_unified_evaluation_service_mapping.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py`
- Test: `tldw_Server_API/tests/Evaluations/test_eval_test_mode_truthiness.py`

- [ ] **Step 1: Read the orchestration files in request-to-evaluator order**

Trace:
- service entrypoints into the runner
- evaluator selection and mapping
- background versus foreground execution
- batch semantics and timeout handling
- cross-evaluator shared helpers

Expected: Slice 2 notes identify where orchestration state, batching, and evaluator routing can diverge from endpoint assumptions.

- [ ] **Step 2: Search for async, semaphore, timeout, and task-tracking hotspots**

Run:
```bash
source .venv/bin/activate
rg -n "asyncio|Semaphore|timeout|running_tasks|background|create_task|gather|CancelledError|eval_timeout" \
  tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py \
  tldw_Server_API/app/core/Evaluations/eval_runner.py
```

Expected: a focused list of orchestration hotspots for race, leak, and batch-failure analysis.

- [ ] **Step 3: Run the focused Slice 2 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py \
  tldw_Server_API/tests/Evaluations/unit/test_unified_evaluation_service_mapping.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py \
  tldw_Server_API/tests/Evaluations/test_eval_test_mode_truthiness.py
```

Expected: PASS for the current suite, or a stable failure that validates a concrete orchestration risk.

- [ ] **Step 4: Update Slice 2 in the cumulative review README and commit it**

Run:
```bash
git add Docs/superpowers/reviews/evals-module/README.md
git diff --cached --name-only
git commit -m "docs: record evals slice 2 review"
```

Expected: one docs-only commit contains the Slice 2 findings only.

### Task 4: Execute Slice 3 Persistence and State Management

**Files:**
- Modify: `Docs/superpowers/reviews/evals-module/README.md`
- Inspect: `tldw_Server_API/app/core/Evaluations/evaluation_manager.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/db_adapter.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/connection_pool.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/audit_adapter.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/DB_Manager.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/db_path_utils.py`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_evaluations_db_filters.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_backend_dual.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_postgres_crud.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_migration_cli.py`
- Test: `tldw_Server_API/tests/DB_Management/test_evaluations_unified_and_crud.py`

- [ ] **Step 1: Read the persistence files from path resolution through storage write paths**

Trace:
- DB path selection and containment
- backend selection and adapter routing
- migration invocation and fallback behavior
- filtering and listing behavior
- storage writes for evaluation runs and audit hooks

Expected: Slice 3 notes can explain exactly how Evals chooses a backend and where persistence invariants could break.

- [ ] **Step 2: Search for path, migration, and fallback hotspots**

Run:
```bash
source .venv/bin/activate
rg -n "resolve|fallback|migrate|sqlite|postgres|db_path|relative_to|CREATE TABLE|OperationalError|RuntimeError" \
  tldw_Server_API/app/core/Evaluations/evaluation_manager.py \
  tldw_Server_API/app/core/Evaluations/db_adapter.py \
  tldw_Server_API/app/core/Evaluations/connection_pool.py
```

Expected: a focused list of persistence branches where backend parity or containment assumptions need validation.

- [ ] **Step 3: Run the focused Slice 3 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py \
  tldw_Server_API/tests/Evaluations/unit/test_evaluations_db_filters.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_backend_dual.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_postgres_crud.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_migration_cli.py \
  tldw_Server_API/tests/DB_Management/test_evaluations_unified_and_crud.py
```

Expected: PASS for the targeted persistence suite, or a reproducible failure tied to a concrete Slice 3 risk.

- [ ] **Step 4: Update Slice 3 in the cumulative review README and commit it**

Run:
```bash
git add Docs/superpowers/reviews/evals-module/README.md
git diff --cached --name-only
git commit -m "docs: record evals slice 3 review"
```

Expected: one docs-only commit contains the Slice 3 persistence findings only.

### Task 5: Execute Slice 4 CRUD and Run Lifecycle Endpoints

**Files:**
- Modify: `Docs/superpowers/reviews/evals-module/README.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/evaluation_schema.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/audit_adapter.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluation_integration.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py`
- Test: `tldw_Server_API/tests/DB_Management/test_evaluations_unified_and_crud.py`

- [ ] **Step 1: Read the CRUD endpoint and run-lifecycle paths**

Trace:
- evaluation creation and read flows
- run creation and status retrieval
- history and pagination behavior
- export and audit touchpoints
- user isolation and usage-accounting hooks

Expected: Slice 4 notes identify where endpoint contracts rely on persistence or service assumptions that may already have findings from Slices 2 and 3.

- [ ] **Step 2: Search for status, pagination, and ownership hotspots**

Run:
```bash
source .venv/bin/activate
rg -n "status|history|page|limit|offset|user_id|owner|audit|export|run_id|evaluation_id" \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py \
  tldw_Server_API/app/core/Evaluations/audit_adapter.py
```

Expected: a shortlist of run-lifecycle and isolation branches that deserve evidence-backed writeups.

- [ ] **Step 3: Run the focused Slice 4 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py \
  tldw_Server_API/tests/Evaluations/test_evaluation_integration.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py \
  tldw_Server_API/tests/DB_Management/test_evaluations_unified_and_crud.py
```

Expected: PASS for the focused CRUD suite, or a stable failing path that confirms a run-lifecycle defect.

- [ ] **Step 4: Update Slice 4 in the cumulative review README and commit it**

Run:
```bash
git add Docs/superpowers/reviews/evals-module/README.md
git diff --cached --name-only
git commit -m "docs: record evals slice 4 review"
```

Expected: one docs-only commit contains the Slice 4 findings only.

### Task 6: Execute Slice 5 Retrieval and Recipe-Driven Evaluation Surfaces

**Files:**
- Modify: `Docs/superpowers/reviews/evals-module/README.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_recipes.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/evaluation_recipe_schemas.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/rag_evaluator.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/response_quality_evaluator.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/recipe_runs_service.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/recipe_runs_jobs.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/recipes/base.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/recipes/registry.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/recipes/rag_answer_quality.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/recipes/rag_answer_quality_execution.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/recipes/rag_retrieval_tuning.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/recipes/rag_retrieval_tuning_execution.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/recipes/embeddings_retrieval.py`
- Test: `tldw_Server_API/tests/Evaluations/test_rag_evaluator_embeddings.py`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_rag_evaluator.py`
- Test: `tldw_Server_API/tests/Evaluations/test_recipe_embeddings_retrieval.py`
- Test: `tldw_Server_API/tests/Evaluations/test_recipe_rag_retrieval_tuning.py`

- [ ] **Step 1: Read the retrieval and recipe paths from endpoint to execution**

Trace:
- endpoint request normalization
- recipe lookup and registration
- recipe execution and job handoff
- shared evaluator use
- result shaping back to the API layer

Expected: Slice 5 notes can distinguish endpoint bugs from recipe-framework or evaluator bugs without duplicating earlier findings.

- [ ] **Step 2: Search for registry, candidate, and execution hotspots**

Run:
```bash
source .venv/bin/activate
rg -n "registry|execute|candidate|metric|retrieval|quality|job|background|dataset|snapshot" \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_rag_pipeline.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_recipes.py \
  tldw_Server_API/app/core/Evaluations/recipe_runs_service.py \
  tldw_Server_API/app/core/Evaluations/recipes
```

Expected: a focused list of recipe and retrieval branches where state, schema, or control-flow drift is most likely.

- [ ] **Step 3: Run the focused Slice 5 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Evaluations/test_rag_evaluator_embeddings.py \
  tldw_Server_API/tests/Evaluations/unit/test_rag_evaluator.py \
  tldw_Server_API/tests/Evaluations/test_recipe_embeddings_retrieval.py \
  tldw_Server_API/tests/Evaluations/test_recipe_rag_retrieval_tuning.py
```

Expected: PASS for the focused recipe and retrieval suite, or a stable failure that confirms a concrete Slice 5 issue.

- [ ] **Step 4: Update Slice 5 in the cumulative review README and commit it**

Run:
```bash
git add Docs/superpowers/reviews/evals-module/README.md
git diff --cached --name-only
git commit -m "docs: record evals slice 5 review"
```

Expected: one docs-only commit contains the Slice 5 findings only.

### Task 7: Execute Slice 6 Benchmark, Dataset, and Synthetic Evaluation Surfaces

**Files:**
- Modify: `Docs/superpowers/reviews/evals-module/README.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_synthetic.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/synthetic_eval_schemas.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/benchmark_registry.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/benchmark_loaders.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/benchmark_utils.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/synthetic_eval_service.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/synthetic_eval_repository.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/synthetic_eval_generation.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`
- Test: `tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py`
- Test: `tldw_Server_API/tests/Evaluations/integration/test_synthetic_eval_api.py`
- Test: `tldw_Server_API/tests/Evaluations/test_synthetic_eval_service.py`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_evals_cli_benchmark_commands.py`

- [ ] **Step 1: Read the benchmark, dataset, and synthetic flows in order**

Trace:
- benchmark registry and loader resolution
- dataset endpoint ownership and permission checks
- synthetic generation request flow
- repository persistence and retrieval
- API response shaping

Expected: Slice 6 notes capture where dataset, benchmark, and synthetic behaviors share assumptions or silently diverge.

- [ ] **Step 2: Search for loader, registry, permission, and generation hotspots**

Run:
```bash
source .venv/bin/activate
rg -n "registry|loader|dataset|permission|synthetic|generate|review|job|yaml|config" \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_datasets.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_synthetic.py \
  tldw_Server_API/app/core/Evaluations/benchmark_registry.py \
  tldw_Server_API/app/core/Evaluations/synthetic_eval_service.py
```

Expected: a focused list of stage-specific branches that deserve close inspection and test correlation.

- [ ] **Step 3: Run the focused Slice 6 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py \
  tldw_Server_API/tests/Evaluations/test_evaluations_stage4_auth_policy_and_dataset_permissions.py \
  tldw_Server_API/tests/Evaluations/integration/test_synthetic_eval_api.py \
  tldw_Server_API/tests/Evaluations/test_synthetic_eval_service.py \
  tldw_Server_API/tests/Evaluations/unit/test_evals_cli_benchmark_commands.py
```

Expected: PASS for the focused benchmark and synthetic suite, or a stable failure that sharpens a Slice 6 finding.

- [ ] **Step 4: Update Slice 6 in the cumulative review README and commit it**

Run:
```bash
git add Docs/superpowers/reviews/evals-module/README.md
git diff --cached --name-only
git commit -m "docs: record evals slice 6 review"
```

Expected: one docs-only commit contains the Slice 6 findings only.

### Task 8: Execute Slice 7 Embeddings A/B and Webhook Surfaces

**Files:**
- Modify: `Docs/superpowers/reviews/evals-module/README.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_embeddings_abtest.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_webhooks.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/embeddings_abtest_schemas.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/embeddings_abtest_repository.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/embeddings_abtest_runner.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/embeddings_abtest_service.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/embeddings_abtest_jobs.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/webhook_identity.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/webhook_manager.py`
- Inspect: `tldw_Server_API/app/core/Evaluations/webhook_security.py`
- Test: `tldw_Server_API/tests/Evaluations/test_embeddings_abtest_retrieval.py`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_evaluations_abtest_store_init.py`
- Test: `tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_webhook_manager_backend_schema.py`

- [ ] **Step 1: Read the embeddings A/B and webhook files from endpoint through storage and delivery**

Trace:
- endpoint normalization
- repository and job handoff
- delivery signing and identity rules
- multi-user webhook registration and callback flow
- backend schema assumptions

Expected: Slice 7 notes identify where asynchronous evaluation jobs and webhook delivery introduce state, tenancy, or schema risks beyond earlier slices.

- [ ] **Step 2: Search for signing, retry, backend, and job hotspots**

Run:
```bash
source .venv/bin/activate
rg -n "signature|secret|retry|backend|schema|deliver|job|queue|callback|tenant|user_id" \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_embeddings_abtest.py \
  tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_webhooks.py \
  tldw_Server_API/app/core/Evaluations/embeddings_abtest_repository.py \
  tldw_Server_API/app/core/Evaluations/embeddings_abtest_service.py \
  tldw_Server_API/app/core/Evaluations/webhook_manager.py \
  tldw_Server_API/app/core/Evaluations/webhook_security.py
```

Expected: a short list of stateful branches where correctness or security findings should be evidence-backed before recording them.

- [ ] **Step 3: Run the focused Slice 7 tests**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Evaluations/test_embeddings_abtest_retrieval.py \
  tldw_Server_API/tests/Evaluations/unit/test_evaluations_abtest_store_init.py \
  tldw_Server_API/tests/Evaluations/integration/test_webhook_multi_user_api.py \
  tldw_Server_API/tests/Evaluations/unit/test_webhook_manager_backend_schema.py
```

Expected: PASS for the focused embeddings and webhook suite, or a reproducible failure tied to a concrete Slice 7 issue.

- [ ] **Step 4: Update Slice 7 in the cumulative review README and commit it**

Run:
```bash
git add Docs/superpowers/reviews/evals-module/README.md
git diff --cached --name-only
git commit -m "docs: record evals slice 7 review"
```

Expected: one docs-only commit contains the Slice 7 findings only.

### Task 9: Execute Slice 8 Cross-Slice Contract Synthesis and Finalize the Review

**Files:**
- Modify: `Docs/superpowers/reviews/evals-module/README.md`
- Inspect: `tldw_Server_API/app/api/v1/schemas/evaluation_schemas_unified.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/evaluation_schema.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/evaluation_recipe_schemas.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/embeddings_abtest_schemas.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/synthetic_eval_schemas.py`
- Inspect: `tldw_Server_API/Config_Files/evaluations_config.yaml`
- Inspect: `tldw_Server_API/app/core/Evaluations/README.md`
- Inspect: `tldw_Server_API/app/core/Evaluations/EVALS_DEVELOPER_GUIDE.md`
- Inspect: `tldw_Server_API/app/core/Evaluations/SECURITY.md`
- Test: `tldw_Server_API/tests/Evaluations/property/test_evaluation_invariants.py`
- Test: `tldw_Server_API/tests/e2e/test_evaluations_workflow.py`
- Test: `tldw_Server_API/tests/server_e2e_tests/test_evaluations_workflow.py`

- [ ] **Step 1: Re-read the shared schemas and config after the slice notes exist**

Trace:
- request and response field drift across slices
- shared defaults and enum values
- config assumptions that contradict endpoint or service behavior
- documentation mismatches that affect operator or maintainer expectations

Expected: Slice 8 distinguishes true cross-slice problems from slice-local findings already captured earlier.

- [ ] **Step 2: Reconcile duplicates and label working-tree-specific findings**

Update the cumulative README so that:
- duplicate findings are merged
- each merged finding points to the strongest evidence
- any finding materially influenced by local uncommitted Evals changes is labeled as working-tree-specific
- unresolved items remain in `Coverage Gaps and Verification Items` instead of staying embedded as pseudo-findings

- [ ] **Step 3: Run the minimal cross-slice sanity pack only if needed**

Run:
```bash
source .venv/bin/activate
python -m pytest -v \
  tldw_Server_API/tests/Evaluations/property/test_evaluation_invariants.py \
  tldw_Server_API/tests/e2e/test_evaluations_workflow.py \
  tldw_Server_API/tests/server_e2e_tests/test_evaluations_workflow.py
```

Expected: use this pack only to settle disputed cross-slice claims or to confirm the absence of a larger contract break; otherwise record why it was not needed.

- [ ] **Step 4: Finalize the priority summary and remediation order**

Write:
- the ranked issue summary
- the recommended remediation order
- the highest-value coverage gaps
- any open verification items that remain after the earlier slices

Expected: `Docs/superpowers/reviews/evals-module/README.md` is now the canonical final review record and can be consumed without reading the plan.

- [ ] **Step 5: Commit the final cumulative review**

Run:
```bash
git add Docs/superpowers/reviews/evals-module/README.md
git diff --cached --name-only
git commit -m "docs: finalize evals module review"
```

Expected: one docs-only commit contains the final ranked review.
