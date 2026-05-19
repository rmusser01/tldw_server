# Workflows Backend Review Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved Workflows backend review and deliver one findings-first, evidence-backed report covering correctness, security, maintainability, performance, and operational risks across the Workflows API, engine, persistence, scheduler, and selected high-risk adapter boundaries.

**Architecture:** This is a read-first, contract-driven review plan. Execution starts by locking the worktree baseline and final report contract, then traces API and authorization assumptions into engine and persistence behavior, then inspects scheduler and operations handoff paths, then selectively reviews the adapters and helpers most likely to create safety or control-flow defects, and finally runs only the narrowest targeted test slices needed to confirm or weaken candidate findings. No repository source changes are part of execution; the deliverable is the final in-session review output.

**Tech Stack:** Python 3, pytest, git, rg, find, sed, Markdown

---

## Scope Lock

Keep these decisions fixed during execution:

- review the current working tree by default, not only `HEAD`
- label findings that depend on uncommitted local changes
- keep code scope inside backend Workflows surfaces only
- exclude frontend workflow editor, chat-workflows UI, browser extension workflow surfaces, and unrelated scheduler domains
- use `tldw_Server_API/tests/Workflows` as the primary evidence set and pull from AuthNZ, Resource_Governance, DB_Management, and CI tests only when they directly validate an in-scope Workflows contract
- separate `Confirmed finding`, `Probable risk`, and `Improvement`
- keep adapter review selective and risk-driven rather than reading every adapter file exhaustively
- do not modify repository source files during the review itself
- do not run broad blanket suites; use the smallest targeted verification needed to answer a concrete question
- keep blind spots explicit instead of implying unreviewed surfaces are safe

## Review File Map

**No repository source files should be modified during execution.**

**Spec and plan inputs:**
- `Docs/superpowers/specs/2026-04-07-workflows-backend-review-design.md`
- `Docs/superpowers/plans/2026-04-07-workflows-backend-review-execution-plan.md`

**Primary docs and contract references:**
- `Docs/Code_Documentation/Workflows_Module.md`
- `Docs/Operations/Workflows_Runbook.md`
- `Docs/Operations/Workflows_Debugging.md`
- `Docs/Operations/Workflows_Performance.md`
- `tldw_Server_API/app/core/Workflows/README.md`

**Primary API and schema files to inspect first:**
- `tldw_Server_API/app/api/v1/endpoints/workflows.py`
- `tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py`
- `tldw_Server_API/app/api/v1/schemas/workflows.py`

**Primary engine and persistence files to inspect first:**
- `tldw_Server_API/app/core/Workflows/engine.py`
- `tldw_Server_API/app/core/Workflows/registry.py`
- `tldw_Server_API/app/core/Workflows/capabilities.py`
- `tldw_Server_API/app/core/Workflows/failures.py`
- `tldw_Server_API/app/core/Workflows/investigation.py`
- `tldw_Server_API/app/core/DB_Management/Workflows_DB.py`
- `tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py`

**Operational and handoff files to inspect when the active trace requires them:**
- `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`
- `tldw_Server_API/app/core/Workflows/daily_ledger.py`
- `tldw_Server_API/app/core/Workflows/research_wait_bridge.py`
- `tldw_Server_API/app/core/Workflows/subprocess_utils.py`
- `tldw_Server_API/app/core/Workflows/metrics.py`
- `tldw_Server_API/app/services/workflows_scheduler.py`
- `tldw_Server_API/app/services/workflows_artifact_gc_service.py`
- `tldw_Server_API/app/services/workflows_db_maintenance.py`
- `tldw_Server_API/app/services/workflows_webhook_dlq_service.py`

**High-risk adapter and helper files to inspect selectively:**
- `tldw_Server_API/app/core/Workflows/adapters/_registry.py`
- `tldw_Server_API/app/core/Workflows/adapters/_common.py`
- `tldw_Server_API/app/core/Workflows/adapters/control/flow.py`
- `tldw_Server_API/app/core/Workflows/adapters/control/orchestration.py`
- `tldw_Server_API/app/core/Workflows/adapters/integration/webhook.py`
- `tldw_Server_API/app/core/Workflows/adapters/integration/mcp.py`
- `tldw_Server_API/app/core/Workflows/adapters/integration/acp.py`
- `tldw_Server_API/app/core/Workflows/adapters/research/launch.py`
- `tldw_Server_API/app/core/Workflows/adapters/research/wait.py`
- `tldw_Server_API/app/core/Workflows/adapters/research/load_bundle.py`
- `tldw_Server_API/app/core/Workflows/adapters/media/ingest.py`

**High-value tests to inspect and selectively run:**
- `tldw_Server_API/tests/Workflows/test_workflows_api.py`
- `tldw_Server_API/tests/Workflows/test_workflow_preflight_api.py`
- `tldw_Server_API/tests/Workflows/test_workflow_templates_api.py`
- `tldw_Server_API/tests/Workflows/test_workflow_investigation_api.py`
- `tldw_Server_API/tests/Workflows/test_workflow_approval_permissions.py`
- `tldw_Server_API/tests/Workflows/test_runs_cursor_pagination.py`
- `tldw_Server_API/tests/Workflows/test_events_cursor_pagination.py`
- `tldw_Server_API/tests/Workflows/test_runs_listing_combinations.py`
- `tldw_Server_API/tests/Workflows/test_engine_state_contracts.py`
- `tldw_Server_API/tests/Workflows/test_engine_idempotent_lifecycle.py`
- `tldw_Server_API/tests/Workflows/test_engine_retry_cancel_delay.py`
- `tldw_Server_API/tests/Workflows/test_engine_retry_classifier.py`
- `tldw_Server_API/tests/Workflows/test_engine_scheduler.py`
- `tldw_Server_API/tests/Workflows/test_engine_step_types.py`
- `tldw_Server_API/tests/Workflows/test_engine_template_resolution.py`
- `tldw_Server_API/tests/Workflows/test_workflow_attempt_failures.py`
- `tldw_Server_API/tests/Workflows/test_workflows_db.py`
- `tldw_Server_API/tests/Workflows/test_versions_idempotency.py`
- `tldw_Server_API/tests/Workflows/test_workflows_idempotency_ttl.py`
- `tldw_Server_API/tests/Workflows/test_workflows_pg_event_seq_concurrency.py`
- `tldw_Server_API/tests/Workflows/test_dual_backend_engine.py`
- `tldw_Server_API/tests/Workflows/test_dual_backend_workflows.py`
- `tldw_Server_API/tests/Workflows/test_workflows_scheduler.py`
- `tldw_Server_API/tests/Workflows/test_adapter_path_security.py`
- `tldw_Server_API/tests/Workflows/test_egress_policy.py`
- `tldw_Server_API/tests/Workflows/test_mcp_tool_policy.py`
- `tldw_Server_API/tests/Workflows/test_mcp_tool_allowlist_integration.py`
- `tldw_Server_API/tests/Workflows/test_webhook_adapter_smoke.py`
- `tldw_Server_API/tests/Workflows/test_webhook_admin_endpoints.py`
- `tldw_Server_API/tests/Workflows/test_webhook_deliveries_history.py`
- `tldw_Server_API/tests/Workflows/test_webhook_dlq_worker.py`
- `tldw_Server_API/tests/Workflows/test_webhook_step_controls_unit.py`
- `tldw_Server_API/tests/Workflows/test_webhook_step_controls_integration.py`
- `tldw_Server_API/tests/Workflows/test_workflow_step_capabilities.py`
- `tldw_Server_API/tests/Workflows/test_step_registry_runtime_coverage.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_runs_permissions_claims.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_control_permissions_claims.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_events_permissions_claims.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_artifacts_permissions_claims.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_scheduler_workflows_permissions_claims.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_webhook_dlq_permissions_claims.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_virtual_key_permissions_claims.py`
- `tldw_Server_API/tests/Resource_Governance/test_rg_cutover_workflows_quota.py`
- `tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py`
- `tldw_Server_API/tests/CI/test_required_workflow_contracts.py`

**Scratch artifacts allowed during execution:**
- `/tmp/workflows_review_notes.md`
- `/tmp/workflows_api_pytest.log`
- `/tmp/workflows_engine_pytest.log`
- `/tmp/workflows_scheduler_pytest.log`
- `/tmp/workflows_boundary_pytest.log`

## Stage Overview

## Stage 1: Baseline and Report Contract
**Goal:** Lock the dirty-worktree baseline, confirm the exact backend Workflows review surface, and fix the final output template before deep reading starts.
**Success Criteria:** The baseline, scope boundary, hotspot order, and final report structure are fixed before any candidate finding is treated as actionable.
**Tests:** No pytest execution in this stage.
**Status:** Not Started

## Stage 2: API, Schema, and Authorization Pass
**Goal:** Trace public and operator-facing contracts from endpoint and schema files into ownership, RBAC, artifact access, and scheduler-route expectations.
**Success Criteria:** API-visible invariants and permission assumptions are mapped with exact file references, and any mismatch is labeled as confirmed defect, probable risk, or ambiguity.
**Tests:** Read the relevant API and AuthNZ permission tests first, then run only the endpoint and permission slices needed to validate suspect paths.
**Status:** Not Started

## Stage 3: Engine Lifecycle and Persistence Pass
**Goal:** Trace run creation, step execution, state transitions, retries, idempotency, event sequencing, artifact persistence, and backend divergence.
**Success Criteria:** Candidate findings are tied to exact engine and DB paths with enough evidence to explain failure mode, impact, and whether tests already cover it.
**Tests:** Read and run the lifecycle, DB, idempotency, and cursor-pagination tests named in this plan.
**Status:** Not Started

## Stage 4: Scheduler and Operations Pass
**Goal:** Inspect recurring scheduling, scheduler-to-engine handoff, daily-ledger side effects, orphan recovery, webhook retry plumbing, and support services that can duplicate, lose, or misclassify work.
**Success Criteria:** Scheduling and operations assumptions are traced far enough to support evidence-backed claims about duplicate work, skipped work, fail-open behavior, or operator blind spots.
**Tests:** Read and run only the scheduler, quota, DB-path, and orphan-requeue slices needed to validate suspect behavior.
**Status:** Not Started

## Stage 5: Selective Adapter Boundary Pass
**Goal:** Inspect only the adapters and helpers that materially affect safety, network egress, artifact scope, subprocess control, or external-service handoff.
**Success Criteria:** Boundary defects and policy gaps are backed by implementation traces and the smallest useful adapter or boundary test slice.
**Tests:** Read and run only the path-security, egress, MCP, webhook, and registry or capability slices needed to confirm or weaken candidate findings.
**Status:** Not Started

## Stage 6: Targeted Verification and Final Synthesis
**Goal:** Reconcile code reading with test evidence, run any final narrow verification needed to settle disputes, and produce the final findings-first report.
**Success Criteria:** Every major claim in the final review is backed by code inspection, test inspection, executed verification, or an explicit open-question label.
**Tests:** Only the additional narrow slices needed to settle unresolved findings.
**Status:** Not Started

### Task 1: Lock the Baseline and Final Output Contract

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/superpowers/specs/2026-04-07-workflows-backend-review-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-07-workflows-backend-review-execution-plan.md`
- Inspect: `tldw_Server_API/app/core/Workflows`
- Inspect: `tldw_Server_API/tests/Workflows`
- Inspect: `tldw_Server_API/tests/AuthNZ_Unit`
- Inspect: `tldw_Server_API/tests/Resource_Governance`
- Inspect: `tldw_Server_API/tests/DB_Management`
- Test: none

- [ ] **Step 1: Capture the dirty-worktree baseline**

Run:
```bash
git status --short
```

Expected: a list of uncommitted files, including whether any Workflows-related files already differ from committed history.

- [ ] **Step 2: Record the commit baseline used for the review**

Run:
```bash
git rev-parse --short HEAD
```

Expected: one short commit hash to cite when a finding depends on committed behavior rather than only local edits.

- [ ] **Step 3: Enumerate the exact backend Workflows code surface**

Run:
```bash
rg --files tldw_Server_API/app/core/Workflows
rg --files tldw_Server_API/app/api/v1 | rg 'endpoints/(workflows|scheduler_workflows)\.py|schemas/workflows\.py$'
rg --files tldw_Server_API/app/core/DB_Management | rg 'Workflows_DB\.py$|Workflows_Scheduler_DB\.py$'
rg --files tldw_Server_API/app/core/Scheduler/handlers | rg 'workflows\.py$'
rg --files tldw_Server_API/app/services | rg 'workflows_'
```

Expected: the concrete file inventory that anchors the review and prevents drift into frontend or chat-workflows surfaces.

- [ ] **Step 4: Enumerate the primary and adjacent test surface**

Run:
```bash
rg --files tldw_Server_API/tests | rg 'Workflows/|AuthNZ_Unit/test_.*workflows.*claims|AuthNZ_Unit/test_scheduler_workflows_permissions_claims\.py$|Resource_Governance/test_rg_cutover_workflows_quota\.py$|DB_Management/test_workflows_scheduler_db_paths\.py$|CI/test_required_workflow_contracts\.py$'
```

Expected: a test inventory that makes permission, quota, scheduler-path, and contract coverage visible before verification begins.

- [ ] **Step 5: Fix the final response contract before deep reading**

Use this exact final structure:
```markdown
## Findings
- severity-ordered findings
- each item states issue class, confidence, exact file references, impact, and fix direction when clear

## Open Questions / Assumptions
- only unresolved items that materially affect confidence

## Improvements
- lower-priority maintainability, performance, or operational suggestions that are not immediate bugs

## Verification
- tests run, important files inspected, and what remains unverified
```

### Task 2: Execute the API, Schema, and Authorization Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/Code_Documentation/Workflows_Module.md`
- Inspect: `Docs/Operations/Workflows_Runbook.md`
- Inspect: `tldw_Server_API/app/core/Workflows/README.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/workflows.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/workflows.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflows_api.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflow_preflight_api.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflow_templates_api.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflow_investigation_api.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflow_approval_permissions.py`
- Inspect: `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_runs_permissions_claims.py`
- Inspect: `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_control_permissions_claims.py`
- Inspect: `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_events_permissions_claims.py`
- Inspect: `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_artifacts_permissions_claims.py`
- Inspect: `tldw_Server_API/tests/AuthNZ_Unit/test_scheduler_workflows_permissions_claims.py`
- Test: `tldw_Server_API/tests/Workflows/test_workflows_api.py`
- Test: `tldw_Server_API/tests/Workflows/test_workflow_preflight_api.py`
- Test: `tldw_Server_API/tests/Workflows/test_workflow_templates_api.py`
- Test: `tldw_Server_API/tests/Workflows/test_workflow_investigation_api.py`
- Test: `tldw_Server_API/tests/Workflows/test_workflow_approval_permissions.py`

- [ ] **Step 1: Read the public and operator-facing Workflow docs first**

Run:
```bash
sed -n '1,220p' Docs/Code_Documentation/Workflows_Module.md
sed -n '1,240p' Docs/Operations/Workflows_Runbook.md
sed -n '1,220p' tldw_Server_API/app/core/Workflows/README.md
```

Expected: the intended contract for definitions, runs, events, artifacts, controls, scheduler behavior, and webhook operations.

- [ ] **Step 2: Locate API and authorization landmarks before full reads**

Run:
```bash
rg -n '@router|require_permissions|WORKFLOWS_|_get_authorized_run_or_404|idempotency|validation_mode|preflight|investigation|artifacts|events|cursor|run_now|schedule' \
  tldw_Server_API/app/api/v1/endpoints/workflows.py \
  tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py \
  tldw_Server_API/app/api/v1/schemas/workflows.py
```

Expected: a stable reading map for ownership, authorization, contract normalization, and run-control behavior.

- [ ] **Step 3: Read the endpoint and schema files in contract order**

Capture during reading:
- accepted request shapes and normalization rules
- defaulting and validation behavior
- ownership, tenant, and admin bypass assumptions
- artifact and event access rules
- where scheduler APIs diverge from core run APIs
- any mismatch between docs, schemas, and route behavior

Expected: a candidate finding list for API-visible correctness, auth, and contract drift.

- [ ] **Step 4: Read the API and permission tests before running them**

Capture for each test file:
- which contract invariant it protects
- whether it covers only the happy path or also bad-input and cross-user paths
- which adjacent behaviors still appear untested

Expected: a test-backed map of what the API surface claims to guarantee.

- [ ] **Step 5: Run the focused API and permission slices**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Workflows/test_workflows_api.py \
  tldw_Server_API/tests/Workflows/test_workflow_preflight_api.py \
  tldw_Server_API/tests/Workflows/test_workflow_templates_api.py \
  tldw_Server_API/tests/Workflows/test_workflow_investigation_api.py \
  tldw_Server_API/tests/Workflows/test_workflow_approval_permissions.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_workflows_runs_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_workflows_control_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_workflows_events_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_workflows_artifacts_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_scheduler_workflows_permissions_claims.py
```

Expected: passing tests or concrete failures that validate or weaken suspected API and authorization defects.

### Task 3: Execute the Engine Lifecycle and Persistence Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/Workflows/engine.py`
- Inspect: `tldw_Server_API/app/core/Workflows/failures.py`
- Inspect: `tldw_Server_API/app/core/Workflows/registry.py`
- Inspect: `tldw_Server_API/app/core/Workflows/capabilities.py`
- Inspect: `tldw_Server_API/app/core/Workflows/investigation.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/Workflows_DB.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_engine_state_contracts.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_engine_idempotent_lifecycle.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_engine_retry_cancel_delay.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_engine_retry_classifier.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_engine_scheduler.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_engine_step_types.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_engine_template_resolution.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflow_attempt_failures.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflows_db.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_versions_idempotency.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflows_idempotency_ttl.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_runs_cursor_pagination.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_events_cursor_pagination.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_runs_listing_combinations.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflows_pg_event_seq_concurrency.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_dual_backend_engine.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_dual_backend_workflows.py`
- Test: `tldw_Server_API/tests/Workflows/test_engine_state_contracts.py`
- Test: `tldw_Server_API/tests/Workflows/test_engine_idempotent_lifecycle.py`
- Test: `tldw_Server_API/tests/Workflows/test_engine_retry_cancel_delay.py`
- Test: `tldw_Server_API/tests/Workflows/test_engine_retry_classifier.py`
- Test: `tldw_Server_API/tests/Workflows/test_engine_scheduler.py`
- Test: `tldw_Server_API/tests/Workflows/test_engine_step_types.py`
- Test: `tldw_Server_API/tests/Workflows/test_engine_template_resolution.py`
- Test: `tldw_Server_API/tests/Workflows/test_workflows_db.py`
- Test: `tldw_Server_API/tests/Workflows/test_versions_idempotency.py`
- Test: `tldw_Server_API/tests/Workflows/test_workflows_idempotency_ttl.py`

- [ ] **Step 1: Locate engine lifecycle landmarks before the deep read**

Run:
```bash
rg -n '_is_allowed_transition|submit\(|pause|resume|cancel|retry|waiting_human|waiting_approval|idempot|event|artifact|step_run|template|retryable|reason_code|secrets' \
  tldw_Server_API/app/core/Workflows/engine.py \
  tldw_Server_API/app/core/Workflows/failures.py \
  tldw_Server_API/app/core/Workflows/investigation.py
```

Expected: a reading map for lifecycle, retry, failure-classification, and evidence-emission code paths.

- [ ] **Step 2: Locate DB and pagination landmarks before the deep read**

Run:
```bash
rg -n 'create_run|update_run|append_event|event_seq|cursor|idempot|artifact|list_runs|get_run|step_run|aggregate_run_token_usage|workflow_event_counters|postgres|sqlite' \
  tldw_Server_API/app/core/DB_Management/Workflows_DB.py
```

Expected: a reading map for persistence invariants, concurrency-sensitive sections, and backend divergence points.

- [ ] **Step 3: Read the implementation files in state-flow order**

Trace and capture:
- legal and illegal status transitions
- duplicate submission and retry behavior
- event and artifact ordering assumptions
- token and cost aggregation behavior where it affects run summaries
- whether persistence updates can drift from emitted events or API-visible state
- whether SQLite and Postgres assumptions diverge in correctness-sensitive paths

Expected: a candidate finding list for lifecycle, persistence, and cross-backend correctness risks.

- [ ] **Step 4: Read the lifecycle and persistence tests before running them**

Capture for each selected test file:
- the invariant it actually protects
- whether it exercises concurrency, partial failure, or only happy paths
- what important adjacent behavior still appears uncovered

Expected: a clear map of where tests already defend the state machine and where they do not.

- [ ] **Step 5: Run the focused engine, idempotency, and DB slices**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Workflows/test_engine_state_contracts.py \
  tldw_Server_API/tests/Workflows/test_engine_idempotent_lifecycle.py \
  tldw_Server_API/tests/Workflows/test_engine_retry_cancel_delay.py \
  tldw_Server_API/tests/Workflows/test_engine_retry_classifier.py \
  tldw_Server_API/tests/Workflows/test_engine_scheduler.py \
  tldw_Server_API/tests/Workflows/test_engine_step_types.py \
  tldw_Server_API/tests/Workflows/test_engine_template_resolution.py \
  tldw_Server_API/tests/Workflows/test_workflow_attempt_failures.py \
  tldw_Server_API/tests/Workflows/test_workflows_db.py \
  tldw_Server_API/tests/Workflows/test_versions_idempotency.py \
  tldw_Server_API/tests/Workflows/test_workflows_idempotency_ttl.py \
  tldw_Server_API/tests/Workflows/test_runs_cursor_pagination.py \
  tldw_Server_API/tests/Workflows/test_events_cursor_pagination.py \
  tldw_Server_API/tests/Workflows/test_runs_listing_combinations.py
```

Expected: passing tests or concrete failures that confirm or weaken lifecycle, persistence, and pagination concerns.

### Task 4: Execute the Scheduler and Operations Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/Operations/Workflows_Debugging.md`
- Inspect: `Docs/Operations/Workflows_Performance.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py`
- Inspect: `tldw_Server_API/app/services/workflows_scheduler.py`
- Inspect: `tldw_Server_API/app/core/Scheduler/handlers/workflows.py`
- Inspect: `tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py`
- Inspect: `tldw_Server_API/app/core/Workflows/daily_ledger.py`
- Inspect: `tldw_Server_API/app/core/Workflows/research_wait_bridge.py`
- Inspect: `tldw_Server_API/app/services/workflows_artifact_gc_service.py`
- Inspect: `tldw_Server_API/app/services/workflows_db_maintenance.py`
- Inspect: `tldw_Server_API/app/services/workflows_webhook_dlq_service.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflows_scheduler.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_orphan_requeue_unit.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_orphan_requeue_integration.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_dual_backend_workflows.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflows_postgres_migrations.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflows_postgres_indexes.py`
- Inspect: `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_webhook_dlq_permissions_claims.py`
- Inspect: `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_virtual_key_permissions_claims.py`
- Inspect: `tldw_Server_API/tests/Resource_Governance/test_rg_cutover_workflows_quota.py`
- Inspect: `tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py`
- Test: `tldw_Server_API/tests/Workflows/test_workflows_scheduler.py`
- Test: `tldw_Server_API/tests/Workflows/test_orphan_requeue_unit.py`
- Test: `tldw_Server_API/tests/Workflows/test_orphan_requeue_integration.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_webhook_dlq_permissions_claims.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_workflows_virtual_key_permissions_claims.py`
- Test: `tldw_Server_API/tests/Resource_Governance/test_rg_cutover_workflows_quota.py`
- Test: `tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py`

- [ ] **Step 1: Read the operations docs before reading the service code**

Run:
```bash
sed -n '1,220p' Docs/Operations/Workflows_Debugging.md
sed -n '1,220p' Docs/Operations/Workflows_Performance.md
```

Expected: the intended operational model for recurring schedules, backlog handling, retries, and operator diagnostics.

- [ ] **Step 2: Locate scheduler and operations landmarks before the deep read**

Run:
```bash
rg -n 'cron|coalesce|misfire|jitter|queue|skip|rescan|remove_job|add_job|run_now|virtual|ledger|orphan|retry|dlq|batch|tenant|user_id' \
  tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py \
  tldw_Server_API/app/services/workflows_scheduler.py \
  tldw_Server_API/app/core/Scheduler/handlers/workflows.py \
  tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py \
  tldw_Server_API/app/core/Workflows/daily_ledger.py \
  tldw_Server_API/app/services/workflows_webhook_dlq_service.py
```

Expected: a reading map for duplicate-work prevention, recurring scheduling semantics, quota side effects, and DLQ or orphan handling.

- [ ] **Step 3: Read the implementation files in handoff order**

Trace and capture:
- how schedules become engine submissions
- where duplicate or skipped work can occur
- whether user, tenant, and virtual-key attribution remain consistent across handoff boundaries
- whether daily-ledger accounting can drift from actual run creation
- whether maintenance and recovery services can mutate state without matching evidence or operator visibility

Expected: a candidate finding list for scheduler correctness, operational safety, and quota or recovery behavior.

- [ ] **Step 4: Read the scheduler, quota, and DB-path tests before running them**

Capture for each test:
- which invariant it protects
- whether it covers only nominal scheduler behavior or also race, path, and recovery edges
- which scheduler or ops assumptions still appear untested

Expected: a test-backed view of scheduler and operations guarantees.

- [ ] **Step 5: Run the focused scheduler and ops slices**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Workflows/test_workflows_scheduler.py \
  tldw_Server_API/tests/Workflows/test_orphan_requeue_unit.py \
  tldw_Server_API/tests/Workflows/test_orphan_requeue_integration.py \
  tldw_Server_API/tests/Workflows/test_dual_backend_workflows.py \
  tldw_Server_API/tests/Workflows/test_workflows_postgres_migrations.py \
  tldw_Server_API/tests/Workflows/test_workflows_postgres_indexes.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_workflows_webhook_dlq_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_workflows_virtual_key_permissions_claims.py \
  tldw_Server_API/tests/Resource_Governance/test_rg_cutover_workflows_quota.py \
  tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py
```

Expected: passing tests or concrete failures that validate or weaken scheduler, quota, and maintenance concerns.

### Task 5: Execute the Selective Adapter Boundary Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/_registry.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/_common.py`
- Inspect: `tldw_Server_API/app/core/Workflows/subprocess_utils.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/control/flow.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/control/orchestration.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/integration/webhook.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/integration/mcp.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/integration/acp.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/research/launch.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/research/wait.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/research/load_bundle.py`
- Inspect: `tldw_Server_API/app/core/Workflows/adapters/media/ingest.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_adapter_path_security.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_egress_policy.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_mcp_tool_policy.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_mcp_tool_allowlist_integration.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_webhook_adapter_smoke.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_webhook_admin_endpoints.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_webhook_deliveries_history.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_webhook_dlq_worker.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_webhook_step_controls_unit.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_webhook_step_controls_integration.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_workflow_step_capabilities.py`
- Inspect: `tldw_Server_API/tests/Workflows/test_step_registry_runtime_coverage.py`
- Test: `tldw_Server_API/tests/Workflows/test_adapter_path_security.py`
- Test: `tldw_Server_API/tests/Workflows/test_egress_policy.py`
- Test: `tldw_Server_API/tests/Workflows/test_mcp_tool_policy.py`
- Test: `tldw_Server_API/tests/Workflows/test_mcp_tool_allowlist_integration.py`
- Test: `tldw_Server_API/tests/Workflows/test_webhook_adapter_smoke.py`
- Test: `tldw_Server_API/tests/Workflows/test_webhook_admin_endpoints.py`
- Test: `tldw_Server_API/tests/Workflows/test_webhook_deliveries_history.py`
- Test: `tldw_Server_API/tests/Workflows/test_webhook_dlq_worker.py`
- Test: `tldw_Server_API/tests/Workflows/test_webhook_step_controls_unit.py`
- Test: `tldw_Server_API/tests/Workflows/test_webhook_step_controls_integration.py`
- Test: `tldw_Server_API/tests/Workflows/test_workflow_step_capabilities.py`
- Test: `tldw_Server_API/tests/Workflows/test_step_registry_runtime_coverage.py`

- [ ] **Step 1: Locate registry, boundary, and policy landmarks before the deep read**

Run:
```bash
rg -n 'register|parallel|allowlist|denylist|policy|artifact|subpath|subprocess|terminate|webhook|mcp|acp|wait|launch|bundle|ingest' \
  tldw_Server_API/app/core/Workflows/adapters/_registry.py \
  tldw_Server_API/app/core/Workflows/adapters/_common.py \
  tldw_Server_API/app/core/Workflows/subprocess_utils.py \
  tldw_Server_API/app/core/Workflows/adapters/control/flow.py \
  tldw_Server_API/app/core/Workflows/adapters/control/orchestration.py \
  tldw_Server_API/app/core/Workflows/adapters/integration/webhook.py \
  tldw_Server_API/app/core/Workflows/adapters/integration/mcp.py \
  tldw_Server_API/app/core/Workflows/adapters/integration/acp.py \
  tldw_Server_API/app/core/Workflows/adapters/research/launch.py \
  tldw_Server_API/app/core/Workflows/adapters/research/wait.py \
  tldw_Server_API/app/core/Workflows/adapters/research/load_bundle.py \
  tldw_Server_API/app/core/Workflows/adapters/media/ingest.py
```

Expected: a reading map for safety-critical adapter boundaries and control hooks.

- [ ] **Step 2: Read only the adapters and helpers implicated by live traces or obvious policy risk**

Prioritize during reading:
- webhook delivery, replay, DLQ, and allowlist behavior
- MCP tool policy and allowlist enforcement
- ACP or research wait handoff behavior
- subprocess launch or termination logic
- artifact path and scope helpers
- registry and capability declarations that can drift from runtime behavior

Expected: a candidate finding list for boundary safety and external-service control paths without exhaustive adapter sprawl.

- [ ] **Step 3: Read the boundary and registry tests before running them**

Capture for each test:
- the exact policy or boundary invariant it protects
- whether it covers denial paths and path traversal edges, not only happy paths
- which adjacent external-handoff assumptions still appear untested

Expected: a test-backed map for boundary-security and adapter-runtime claims.

- [ ] **Step 4: Run the focused boundary and adapter slices**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Workflows/test_adapter_path_security.py \
  tldw_Server_API/tests/Workflows/test_egress_policy.py \
  tldw_Server_API/tests/Workflows/test_mcp_tool_policy.py \
  tldw_Server_API/tests/Workflows/test_mcp_tool_allowlist_integration.py \
  tldw_Server_API/tests/Workflows/test_webhook_adapter_smoke.py \
  tldw_Server_API/tests/Workflows/test_webhook_admin_endpoints.py \
  tldw_Server_API/tests/Workflows/test_webhook_deliveries_history.py \
  tldw_Server_API/tests/Workflows/test_webhook_dlq_worker.py \
  tldw_Server_API/tests/Workflows/test_webhook_step_controls_unit.py \
  tldw_Server_API/tests/Workflows/test_webhook_step_controls_integration.py \
  tldw_Server_API/tests/Workflows/test_workflow_step_capabilities.py \
  tldw_Server_API/tests/Workflows/test_step_registry_runtime_coverage.py
```

Expected: passing tests or concrete failures that validate or weaken security and boundary concerns.

### Task 6: Reconcile Evidence and Produce the Final Review

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/superpowers/specs/2026-04-07-workflows-backend-review-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-07-workflows-backend-review-execution-plan.md`
- Inspect: `/tmp/workflows_review_notes.md`
- Test: only any extra narrow slice needed to settle a disputed claim

- [ ] **Step 1: Reconcile candidate findings against evidence quality**

For each candidate issue, label it as one of:
- `Confirmed finding`: supported by direct code evidence and not contradicted by tests
- `Probable risk`: credible concern with incomplete direct proof
- `Improvement`: worthwhile change that does not require claiming a current defect

Expected: no major claim remains unlabeled or evidence-free.

- [ ] **Step 2: Run only the final narrow verification needed to settle disputes**

Run only when necessary:
```bash
source .venv/bin/activate && python -m pytest -v <single test file or ::single_test_name>
```

Expected: either the dispute is settled or the remaining uncertainty is explicit in the final report.

- [ ] **Step 3: Draft the final review using the locked structure**

Use this exact structure:
```markdown
## Findings
- severity-ordered findings with issue class, confidence, exact file references, impact, and fix direction when clear

## Open Questions / Assumptions
- only unresolved items that materially affect confidence

## Improvements
- lower-priority maintainability, performance, or operational suggestions that are not immediate bugs

## Verification
- tests run, important files inspected, and what remains unverified
```

Expected: a concise, findings-first review that matches the approved spec and clearly separates defects, risks, and improvements.
