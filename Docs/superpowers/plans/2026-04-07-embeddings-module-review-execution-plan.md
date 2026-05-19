# Embeddings Module Review Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved Embeddings module audit and deliver one consolidated, evidence-backed review covering correctness and data-integrity risks across the embeddings API, orchestration layer, adapter boundary, and persistence paths.

**Architecture:** This is a read-first, correctness-first review plan. Execution starts by locking the current worktree baseline, then traces contracts from API entrypoints into orchestration and provider adapters, then inspects storage, batching, fallback, and persistence semantics, and only after that runs focused test slices to validate or weaken candidate findings. No repository source changes are part of execution; the deliverable is the final in-session review output.

**Tech Stack:** Python 3, pytest, git, find, grep, sed, Markdown

---

## Scope Lock

Keep these decisions fixed during execution:

- review the current working tree by default, not just `HEAD`
- label findings that depend on uncommitted local changes
- exclude Embeddings ABTest/Evaluations unless a shared helper is required to understand an in-scope contract
- prioritize correctness and data integrity over style, broad refactors, or throughput tuning
- treat adapters as in-scope because contract mismatches there can silently corrupt results
- separate confirmed defects from likely risks and non-bug improvement suggestions
- do not modify repository source files during the review itself
- do not run unrelated blanket test suites; use the targeted test-selection rule from the spec

## Review File Map

**No repository source files should be modified during execution.**

**Spec and plan inputs:**
- `Docs/superpowers/specs/2026-04-07-embeddings-audit-design.md`
- `Docs/superpowers/plans/2026-04-07-embeddings-module-review-execution-plan.md`

**Primary API and schema files to inspect:**
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- `tldw_Server_API/app/api/v1/endpoints/media_embeddings.py`
- `tldw_Server_API/app/api/v1/schemas/embeddings_models.py`

**Primary orchestration and adapter files to inspect:**
- `tldw_Server_API/app/core/Embeddings/async_embeddings.py`
- `tldw_Server_API/app/core/Embeddings/Embeddings_Server/Embeddings_Create.py`
- `tldw_Server_API/app/core/LLM_Calls/embeddings_adapter_registry.py`
- `tldw_Server_API/app/core/LLM_Calls/providers/openai_embeddings_adapter.py`
- `tldw_Server_API/app/core/LLM_Calls/providers/google_embeddings_adapter.py`
- `tldw_Server_API/app/core/LLM_Calls/providers/huggingface_embeddings_adapter.py`

**Support files to inspect when the active trace requires them:**
- `tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py`
- `tldw_Server_API/app/core/Embeddings/request_batching.py`
- `tldw_Server_API/app/core/Embeddings/multi_tier_cache.py`
- `tldw_Server_API/app/core/Embeddings/jobs_adapter.py`
- `tldw_Server_API/app/core/Embeddings/vector_store_batches_db.py`
- `tldw_Server_API/app/core/Embeddings/vector_store_meta_db.py`
- `tldw_Server_API/app/core/Embeddings/services/jobs_worker.py`
- `tldw_Server_API/app/core/Embeddings/services/redis_worker.py`
- `tldw_Server_API/app/core/Embeddings/README.md`
- `Docs/Code_Documentation/Embeddings-Developer-Guide.md`
- `Docs/API-related/Embeddings_Module_Documentation.md`

**High-value tests to inspect and selectively run:**
- `tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py`
- `tldw_Server_API/tests/Embeddings/test_embeddings_v5_integration.py`
- `tldw_Server_API/tests/Embeddings/test_async_embeddings_normalization.py`
- `tldw_Server_API/tests/Embeddings/test_async_embeddings_provider_url_override.py`
- `tldw_Server_API/tests/Embeddings/test_embeddings_batch_dimensions.py`
- `tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py`
- `tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py`
- `tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py`
- `tldw_Server_API/tests/Embeddings/test_request_batching.py`
- `tldw_Server_API/tests/Embeddings/test_media_embeddings_submission_semantics.py`
- `tldw_Server_API/tests/Embeddings/test_media_embeddings_storage_scope.py`
- `tldw_Server_API/tests/Embeddings/test_embedding_storage_paths.py`
- `tldw_Server_API/tests/Embeddings/test_storage_idempotency_property.py`
- `tldw_Server_API/tests/Embeddings/test_pgvector_upsert_idempotency.py`
- `tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_adapter_endpoint.py`
- `tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_adapter_endpoint_multi.py`
- `tldw_Server_API/tests/LLM_Adapters/unit/test_openai_embeddings_adapter_batch_single.py`
- `tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_google_native_http.py`
- `tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_huggingface_native_http.py`

**Scratch artifacts allowed during execution:**
- `/tmp/embeddings_review_notes.md`
- `/tmp/embeddings_review_pytest.log`
- `/tmp/embeddings_adapter_pytest.log`

## Stage Overview

## Stage 1: Baseline and Report Lock
**Goal:** Fix the exact review surface, capture the dirty-worktree baseline, and lock the final output template before deep reading starts.
**Success Criteria:** The scope boundary, representative tests, and final review structure are fixed before any candidate finding is recorded.
**Tests:** No pytest execution in this stage.
**Status:** Not Started

## Stage 2: API and Schema Contract Pass
**Goal:** Trace embeddings request and response contracts from API entrypoints into the first core layer and identify mismatches in input shape, model selection, dimensions, ordering, and nullability.
**Success Criteria:** Endpoint and schema assumptions are mapped, and any contract ambiguity is recorded with exact file references.
**Tests:** Read endpoint-focused tests after the static pass if contract behavior remains unclear.
**Status:** Not Started

## Stage 3: Core Orchestration and Adapter Pass
**Goal:** Inspect async orchestration and provider adapter boundaries for silent coercion, normalization drift, batching mismatches, and incorrect result ordering.
**Success Criteria:** The end-to-end embedding generation path is traced with candidate findings labeled as confirmed issue, likely risk, or improvement.
**Tests:** Read and run focused unit slices for normalization, dimensions, fallback, and adapter behavior.
**Status:** Not Started

## Stage 4: Persistence and Failure-Path Pass
**Goal:** Inspect storage, cache, batch persistence, worker handoff, and failure-handling paths for idempotency breaks, stale metadata coupling, and partial-write hazards.
**Success Criteria:** Storage semantics are traced far enough to support evidence-backed claims about persistence integrity and retry safety.
**Tests:** Read and run focused storage and media embeddings tests after the static pass.
**Status:** Not Started

## Stage 5: Targeted Test Execution and Evidence Reconciliation
**Goal:** Run the focused test slices needed to validate or weaken candidate findings and reconcile any conflict between code reading and test expectations.
**Success Criteria:** Each major claim in the final review is tied to code inspection, test execution, or an explicitly labeled open question.
**Tests:** Only the targeted slices named in this plan plus any directly adjacent test needed to settle a disputed invariant.
**Status:** Not Started

## Stage 6: Final Synthesis
**Goal:** Produce the final review with findings first, severity ordered, and backed by file references and evidence notes.
**Success Criteria:** The final output matches the approved spec and clearly separates confirmed defects, likely risks, open questions, and improvement suggestions.
**Tests:** No new tests unless a disputed claim still needs confirmation.
**Status:** Not Started

### Task 1: Lock the Review Baseline and Output Structure

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/superpowers/specs/2026-04-07-embeddings-audit-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-07-embeddings-module-review-execution-plan.md`
- Inspect: `tldw_Server_API/app/core/Embeddings`
- Inspect: `tldw_Server_API/tests/Embeddings`
- Test: none

- [ ] **Step 1: Capture the dirty-worktree baseline**

Run:
```bash
git status --short
```

Expected: a list of uncommitted files, including whether any embeddings-related files are locally modified.

- [ ] **Step 2: Record the commit baseline used for the review**

Run:
```bash
git rev-parse --short HEAD
```

Expected: one short commit hash to cite when a finding depends on committed behavior rather than only local edits.

- [ ] **Step 3: Enumerate the exact embeddings review surface**

Run:
```bash
find tldw_Server_API/app/core/Embeddings -maxdepth 2 -type f | sort
find tldw_Server_API/app/core/LLM_Calls -maxdepth 2 -type f | grep 'embeddings' | sort
find tldw_Server_API/app/api/v1/endpoints -maxdepth 1 -type f | grep 'embeddings\\|media_embeddings' | sort
find tldw_Server_API/app/api/v1/schemas -maxdepth 1 -type f | grep 'embeddings' | sort
```

Expected: the concrete file inventory that anchors the review and prevents accidental scope creep into ABTest/Evaluations.

- [ ] **Step 4: Enumerate the dedicated embeddings and adapter tests**

Run:
```bash
find tldw_Server_API/tests/Embeddings -maxdepth 1 -type f | sort
find tldw_Server_API/tests/LLM_Adapters/unit -maxdepth 1 -type f | grep 'embeddings' | sort
```

Expected: the test inventory that will anchor the later verification pass.

- [ ] **Step 5: Fix the final response template before reading deeply**

Use this structure for the final review:
```markdown
## Findings
- severity-ordered findings with exact file references, trigger conditions, and evidence labels

## Open Questions / Assumptions
- only unresolved items that materially affect confidence

## Improvements
- non-bug suggestions that reduce correctness or integrity risk

## Verification
- tests run, important files inspected, and what remains unverified
```

### Task 2: Execute the API and Schema Contract Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `Docs/API-related/Embeddings_Module_Documentation.md`
- Inspect: `Docs/Code_Documentation/Embeddings-Developer-Guide.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/media_embeddings.py`
- Inspect: `tldw_Server_API/app/api/v1/schemas/embeddings_models.py`
- Inspect: `tldw_Server_API/app/core/Embeddings/README.md`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_v5_integration.py`
- Test: `tldw_Server_API/tests/Embeddings/test_media_embeddings_submission_semantics.py`
- Test: `tldw_Server_API/tests/Embeddings/test_media_embeddings_storage_scope.py`

- [ ] **Step 1: Read the operator-facing docs first**

Run:
```bash
sed -n '1,240p' Docs/API-related/Embeddings_Module_Documentation.md
sed -n '1,240p' Docs/Code_Documentation/Embeddings-Developer-Guide.md
sed -n '1,220p' tldw_Server_API/app/core/Embeddings/README.md
```

Expected: the intended public contracts for request shapes, storage behavior, batching, and media embeddings workflows.

- [ ] **Step 2: Locate endpoint and schema landmarks before full reads**

Run:
```bash
grep -n "router\\|@router\\|class .*Embedding\\|def .*embedding\\|async def .*embedding\\|dimensions\\|encoding_format\\|input" tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
grep -n "router\\|@router\\|async def .*embedding\\|job\\|storage\\|scope\\|tenant\\|user" tldw_Server_API/app/api/v1/endpoints/media_embeddings.py
grep -n "class .*Embedding\\|dimensions\\|encoding_format\\|input\\|model" tldw_Server_API/app/api/v1/schemas/embeddings_models.py
```

Expected: a stable reading map for request/response schemas, dimensions handling, and media submission semantics.

- [ ] **Step 3: Read the endpoint and schema files in contract order**

Trace and capture:
- accepted input shapes and coercions
- defaulting rules for model and dimensions
- ordering guarantees from request to response
- error response behavior for unsupported or malformed requests
- any mismatch between docs, schema, and endpoint behavior

Expected: a candidate finding list for API-visible correctness and contract drift.

- [ ] **Step 4: Read the endpoint-focused tests before running them**

Capture for each test file:
- which contract invariant it protects
- whether it covers the happy path only or also bad-input cases
- what adjacent contract behavior still appears untested

- [ ] **Step 5: Run the focused endpoint and API contract tests**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_integration.py \
  tldw_Server_API/tests/Embeddings/test_media_embeddings_submission_semantics.py \
  tldw_Server_API/tests/Embeddings/test_media_embeddings_storage_scope.py
```

Expected: passing tests or concrete failures that either validate a suspected contract bug or narrow the concern to an untested path.

### Task 3: Execute the Core Orchestration and Adapter Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/Embeddings/async_embeddings.py`
- Inspect: `tldw_Server_API/app/core/Embeddings/Embeddings_Server/Embeddings_Create.py`
- Inspect: `tldw_Server_API/app/core/Embeddings/request_batching.py`
- Inspect: `tldw_Server_API/app/core/LLM_Calls/embeddings_adapter_registry.py`
- Inspect: `tldw_Server_API/app/core/LLM_Calls/providers/openai_embeddings_adapter.py`
- Inspect: `tldw_Server_API/app/core/LLM_Calls/providers/google_embeddings_adapter.py`
- Inspect: `tldw_Server_API/app/core/LLM_Calls/providers/huggingface_embeddings_adapter.py`
- Test: `tldw_Server_API/tests/Embeddings/test_async_embeddings_normalization.py`
- Test: `tldw_Server_API/tests/Embeddings/test_async_embeddings_provider_url_override.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_batch_dimensions.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py`
- Test: `tldw_Server_API/tests/Embeddings/test_request_batching.py`
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_adapter_endpoint.py`
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_adapter_endpoint_multi.py`
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_openai_embeddings_adapter_batch_single.py`
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_google_native_http.py`
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_huggingface_native_http.py`

- [ ] **Step 1: Map the orchestration landmarks before full reads**

Run:
```bash
grep -n "async def .*embed\\|def .*embed\\|normalize\\|dimensions\\|fallback\\|batch\\|cache\\|provider" tldw_Server_API/app/core/Embeddings/async_embeddings.py
grep -n "def .*create\\|async def .*create\\|batch\\|dimensions\\|provider\\|normalize\\|fallback" tldw_Server_API/app/core/Embeddings/Embeddings_Server/Embeddings_Create.py
grep -n "register\\|adapter\\|provider\\|embed" tldw_Server_API/app/core/LLM_Calls/embeddings_adapter_registry.py
```

Expected: a stable reading map for the highest-risk orchestration and adapter entrypoints.

- [ ] **Step 2: Read the core orchestration path end to end**

Trace and capture:
- request-to-batch flow
- batch result ordering and reshaping
- dimension policy enforcement
- normalization policy and when it is applied
- fallback model selection and error propagation

Expected: candidate findings for silent coercion, ordering drift, dimension mismatch, or fallback misbehavior.

- [ ] **Step 3: Read the provider adapters with a contract checklist**

For each provider adapter, check:
- input payload shape and batch handling
- output parsing and indexing assumptions
- dimensions support or rejection semantics
- normalization assumptions
- provider-specific fallback or endpoint override behavior

Expected: a provider-by-provider note of contract guarantees and suspicious divergences.

- [ ] **Step 4: Read the orchestration and adapter tests before running them**

Capture for each test file:
- which invariant is explicitly protected
- whether the test would catch wrong ordering or wrong dimensions, not just exceptions
- what failure mode still looks weakly covered

- [ ] **Step 5: Run the focused orchestration and adapter tests**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Embeddings/test_async_embeddings_normalization.py \
  tldw_Server_API/tests/Embeddings/test_async_embeddings_provider_url_override.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_batch_dimensions.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py \
  tldw_Server_API/tests/Embeddings/test_request_batching.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_adapter_endpoint.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_adapter_endpoint_multi.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_openai_embeddings_adapter_batch_single.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_google_native_http.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_embeddings_huggingface_native_http.py
```

Expected: evidence about adapter contract conformance and whether suspected orchestration issues are already covered by tests.

### Task 4: Execute the Persistence and Failure-Path Pass

**Files:**
- Create: none
- Modify: none
- Inspect: `tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py`
- Inspect: `tldw_Server_API/app/core/Embeddings/vector_store_batches_db.py`
- Inspect: `tldw_Server_API/app/core/Embeddings/vector_store_meta_db.py`
- Inspect: `tldw_Server_API/app/core/Embeddings/jobs_adapter.py`
- Inspect: `tldw_Server_API/app/core/Embeddings/multi_tier_cache.py`
- Inspect: `tldw_Server_API/app/core/Embeddings/services/jobs_worker.py`
- Inspect: `tldw_Server_API/app/core/Embeddings/services/redis_worker.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embedding_storage_paths.py`
- Test: `tldw_Server_API/tests/Embeddings/test_storage_idempotency_property.py`
- Test: `tldw_Server_API/tests/Embeddings/test_pgvector_upsert_idempotency.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_jobs_adapter.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_jobs_worker.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_redis_worker.py`

- [ ] **Step 1: Locate persistence and handoff landmarks before full reads**

Run:
```bash
grep -n "upsert\\|insert\\|delete\\|batch\\|vector\\|metadata\\|tenant\\|user" tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py
grep -n "create table\\|insert\\|upsert\\|conflict\\|idempot\\|status" tldw_Server_API/app/core/Embeddings/vector_store_batches_db.py
grep -n "create table\\|insert\\|upsert\\|metadata\\|tenant\\|user" tldw_Server_API/app/core/Embeddings/vector_store_meta_db.py
grep -n "enqueue\\|job\\|retry\\|fail\\|chunk\\|storage" tldw_Server_API/app/core/Embeddings/jobs_adapter.py
```

Expected: a reading map for storage semantics, metadata coupling, and job handoff paths.

- [ ] **Step 2: Read the persistence path in integrity order**

Trace and capture:
- how vector data and metadata are coupled
- how tenant or user scoping is enforced
- whether retries can duplicate or diverge writes
- whether delete and upsert paths preserve invariants
- how cache layers can return stale or mismatched data

Expected: candidate findings for persistence drift, stale reads, or idempotency hazards.

- [ ] **Step 3: Read worker and fallback paths for partial-failure handling**

Check:
- what happens when storage succeeds after orchestration failure or the reverse
- whether worker retries can reapply partially committed batches
- whether batch status tables can drift from vector-store truth
- whether failure handling hides bad state behind success codes

Expected: candidate findings for partial-write hazards and false-success conditions.

- [ ] **Step 4: Read the persistence-focused tests before running them**

Capture for each test file:
- which storage invariant it protects
- whether it checks correctness of persisted content, not just job completion
- which retry or partial-failure path still seems weakly covered

- [ ] **Step 5: Run the focused persistence and worker tests**

Run:
```bash
source .venv/bin/activate && python -m pytest -v \
  tldw_Server_API/tests/Embeddings/test_embedding_storage_paths.py \
  tldw_Server_API/tests/Embeddings/test_storage_idempotency_property.py \
  tldw_Server_API/tests/Embeddings/test_pgvector_upsert_idempotency.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_jobs_adapter.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_jobs_worker.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_redis_worker.py
```

Expected: evidence about storage and retry correctness, or concrete failures that narrow the defect surface.

### Task 5: Reconcile Evidence and Build the Final Findings Set

**Files:**
- Create: optional scratch notes in `/tmp`
- Modify: none
- Inspect: all in-scope files and test results already gathered
- Test: only an adjacent test if a disputed finding still needs confirmation

- [ ] **Step 1: Sort each candidate item into one bucket**

Use these buckets:
- `Confirmed issue`: code path and evidence support a concrete correctness or integrity defect
- `Likely risk`: the code path is suspicious, but proof is incomplete or depends on an unresolved assumption
- `Improvement suggestion`: non-bug hardening that reduces integrity risk or review cost

Expected: no mixed labels and no severity assignment without evidence.

- [ ] **Step 2: Apply the severity rubric from the spec**

For each confirmed or likely issue, assign:
- `Critical` for silent corruption or incorrect persisted/returned embeddings without reliable detection
- `High` for high-probability wrong results or durable inconsistency
- `Medium` for narrower but real correctness risk
- `Low` for defensive hardening gaps with limited immediate impact

Expected: every finding has explicit severity rationale tied to the spec rubric.

- [ ] **Step 3: Verify file and line references before drafting**

Run:
```bash
nl -ba tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py | sed -n '1,260p'
nl -ba tldw_Server_API/app/api/v1/endpoints/media_embeddings.py | sed -n '1,260p'
nl -ba tldw_Server_API/app/core/Embeddings/async_embeddings.py | sed -n '1,260p'
nl -ba tldw_Server_API/app/core/Embeddings/Embeddings_Server/Embeddings_Create.py | sed -n '1,260p'
```

Expected: precise line references for the highest-priority findings before the final review is written.

- [ ] **Step 4: Draft the findings-first final review**

Use this exact skeleton:
```markdown
## Findings
- [Severity] file:line - issue, trigger, impact, expected vs actual, evidence note

## Open Questions / Assumptions
- unresolved assumptions that materially affect confidence

## Improvements
- optional hardening suggestions that are not presented as confirmed bugs

## Verification
- tests run
- key files inspected
- anything that could not be verified
```

Expected: a concise final review that matches both the approved spec and the repository's review conventions.
