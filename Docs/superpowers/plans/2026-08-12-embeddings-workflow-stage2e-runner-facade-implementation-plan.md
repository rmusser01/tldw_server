# Embeddings Workflow Stage 2E Runner and Facade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the inline workflow runner sequence the concrete preparation and execution components, reduce `EmbeddingRequestOrchestrator` to a compatibility facade, and move canonical outcome response mapping to the endpoint without changing approved endpoint behavior.

**Architecture:** `EmbeddingInlineWorkflowRunner` receives an `EmbeddingPreparationPipeline`-compatible component and an `EmbeddingExecutionCoordinator`-compatible component directly. `EmbeddingRequestOrchestrator` remains the dependency-wiring and legacy compatibility facade, exposing its two concrete components read-only while its public `prepare()` and `execute()` methods only delegate and map the canonical outcome. The workflow endpoint consumes `EmbeddingExecutionOutcome`, maps HTTP headers at the endpoint boundary, and retains resource governance, active-request accounting, credential touches, metrics, and response formatting.

**Tech Stack:** Python 3.14, FastAPI, dataclasses, structural `Protocol` typing, pytest, pytest-asyncio, Hypothesis, Ruff, Bandit.

## Global Constraints

- Implement only approved Stage 2E behavior from `Docs/superpowers/specs/2026-07-18-embeddings-workflow-stage2-concrete-api-steps-design.md`; add no Stage 3 durability, persistence, pause/resume, retry, lease, or cancellation-state behavior.
- Preserve the workflow feature flag as the operational rollback path and leave the legacy endpoint path unchanged.
- Keep resource-governor reservation commit, failure charging, cancellation cleanup, active-request accounting, metrics, usage logging, response formatting, and final credential touching endpoint-owned.
- Keep provider batch, adapter, and accepted-late-result credential touches executor-owned.
- Keep `EmbeddingExecutionResult` and compatibility HTTP header synthesis only in `map_outcome_to_legacy_execution_result()` for non-workflow callers until Stage 6.
- Emit no provider/model identity, raw input, token arrays, cache keys, credentials, response bodies, caller-controlled headers, or observability tags in workflow traces.
- Use test-driven development for each behavior change and do not bypass existing tests or commit hooks.

---

### Task 1: Direct Runner Component Contracts and Exact Lifecycle

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/workflow_runner.py`
- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py`

**Interfaces:**
- Consumes: `EmbeddingPreparationPipeline.prepare(raw_input, context, phase_sink=None) -> PreparedEmbeddingRequest` and `EmbeddingExecutionCoordinator.execute(prepared) -> Awaitable[EmbeddingExecutionOutcome]`.
- Produces: `EmbeddingInlineWorkflowRunner(preparation_pipeline, execution_coordinator, *, trace_collector=None, pre_execute=None)` and `run(raw_input, context) -> EmbeddingExecutionOutcome`.

- [x] **Step 1: Replace the orchestrator fake with independent preparation and execution fakes**

Create a synchronous fake preparation component that accepts and invokes the phase sink in this exact order:

```python
for phase in ("resolving_intent", "normalizing", "resolving_policy", "planning"):
    phase_sink(phase)
return prepared
```

Create an asynchronous fake coordinator returning `EmbeddingExecutionOutcome` with explicit `attempt_count` and `fallback_attempt_count` values.

- [x] **Step 2: Write the failing exact-success-sequence test**

Assert this event sequence and phase sequence:

```python
event_types = [
    "workflow_started",
    "phase_changed",
    "phase_changed",
    "phase_changed",
    "phase_changed",
    "prepare_completed",
    "phase_changed",
    "execute_completed",
    "phase_changed",
    "workflow_completed",
]
phases = [
    "created",
    "resolving_intent",
    "normalizing",
    "resolving_policy",
    "planning",
    "planning",
    "executing",
    "executing",
    "finalizing",
    "finalizing",
]
```

Assert `execute_completed.metadata` equals the fixed aggregate shape:

```python
{
    "attempt_count": 3,
    "fallback_attempt_count": 1,
    "vector_count": 2,
    "cache_hits": 1,
    "cache_misses": 1,
    "adapter_used": False,
}
```

Assert `response_header_count` is absent and the returned object is the canonical outcome.

- [x] **Step 3: Run the new runner success test and confirm the old constructor/result contract fails**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py::test_runner_returns_canonical_outcome_and_records_exact_safe_success_events -q
```

Expected: FAIL because the runner still accepts one orchestrator and returns `EmbeddingExecutionResult`.

- [x] **Step 4: Implement direct component injection and phase-sink tracing**

Replace `PrepareExecuteOrchestrator` with two narrow protocols:

```python
class PreparationPipeline(Protocol):
    def prepare(
        self,
        raw_input: Any,
        context: EmbeddingRequestContext,
        phase_sink: Callable[[EmbeddingWorkflowPhase], None] | None = None,
    ) -> PreparedEmbeddingRequest: ...

class ExecutionCoordinator(Protocol):
    async def execute(self, prepared: PreparedEmbeddingRequest) -> EmbeddingExecutionOutcome: ...
```

Have `run()` pass a phase sink that updates the current phase before recording `phase_changed`. After preparation, record `prepare_completed`, await `pre_execute` while the current phase remains `planning`, record `executing`, call the coordinator, record one aggregate `execute_completed`, record `finalizing`, then record `workflow_completed`.

- [x] **Step 5: Preserve failure and cancellation semantics**

Adapt existing tests to assert preparation failures report the last phase entered, reservation failures report `planning`, execution failures report `executing`, failure collector errors do not replace the original exception, and `asyncio.CancelledError` propagates without a failed/completed terminal event.

- [x] **Step 6: Add a property-based cardinality and safety test**

Use Hypothesis to vary input size from 1 through 25 and `fallback_attempt_count` from 0 through 20 while returning a matching aggregate outcome. Assert every successful run emits exactly ten events, exactly one `execute_completed`, no `item_state_changed`, and no event metadata key named `response_header_count`.

- [x] **Step 7: Run all runner tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py -q
```

Expected: PASS.

- [x] **Step 8: Commit the runner lifecycle change**

```bash
git add tldw_Server_API/app/core/Embeddings/workflow_runner.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py
git commit -m "refactor(embeddings): sequence concrete workflow components"
```

---

### Task 2: Compatibility Facade Reduction

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/orchestrator.py`
- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`

**Interfaces:**
- Consumes: `EmbeddingPreparationPipeline`, `EmbeddingExecutionCoordinator`, `EmbeddingExecutionOutcome`, and `map_outcome_to_legacy_execution_result(outcome)`.
- Produces: read-only `preparation_pipeline` and `execution_coordinator` properties plus compatibility methods `prepare(...) -> PreparedEmbeddingRequest` and `execute(...) -> EmbeddingExecutionResult`.

- [x] **Step 1: Write failing facade-boundary tests**

Assert the facade exposes the exact pipeline and coordinator instances it wires, `prepare()` delegates once with no phase sink, and `execute()` delegates once then returns the compatibility mapper output. Add a source-boundary assertion that the obsolete `_execute_misses`, `_execute_coherent_fallback`, `_execute_adapter`, `_response_headers`, and `_coerce_executor_output` methods are absent from `EmbeddingRequestOrchestrator`.

- [x] **Step 2: Run focused facade tests and confirm obsolete methods still fail the boundary**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py -q
```

Expected: FAIL on missing component accessors and retained obsolete execution branches.

- [x] **Step 3: Remove superseded private execution code**

Delete `_ProviderExecution`, `_NON_FALLBACKABLE_ERROR_CODES`, `_execute_misses`, `_execute_coherent_fallback`, `_execute_adapter`, `_cache_key`, `_response_headers`, `_coerce_executor_output`, `_is_fallback_eligible`, and `_select_exhausted_error` from `orchestrator.py`, along with imports used only by those branches.

- [x] **Step 4: Expose concrete components read-only and retain compatibility delegation**

Add:

```python
@property
def preparation_pipeline(self) -> EmbeddingPreparationPipeline:
    return self._preparation_pipeline

@property
def execution_coordinator(self) -> EmbeddingExecutionCoordinator:
    return self._execution_coordinator
```

Keep `prepare()` as a direct call to `self._preparation_pipeline.prepare(raw_input, context)` and `execute()` as coordinator delegation followed only by `map_outcome_to_legacy_execution_result(outcome)`.

- [x] **Step 5: Search all imports before removing compatibility re-exports**

Run:

```bash
rg "from tldw_Server_API\.app\.core\.Embeddings\.orchestrator import" tldw_Server_API
```

Preserve any currently imported request DTO names in `orchestrator.__all__` unless every caller is migrated in this task; this task does not widen into unrelated import cleanup.

- [x] **Step 6: Run facade, preparation, coordinator, provider-attempt, and result-mapping tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_preparation_pipeline.py tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py tldw_Server_API/tests/Embeddings_isolated/test_result_mapping.py -q
```

Expected: PASS.

- [x] **Step 7: Commit the compatibility-facade reduction**

```bash
git add tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py
git commit -m "refactor(embeddings): reduce orchestrator to compatibility facade"
```

---

### Task 3: Endpoint Canonical Outcome Boundary

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Modify: `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`

**Interfaces:**
- Consumes: facade component properties, `EmbeddingInlineWorkflowRunner.run(...) -> EmbeddingExecutionOutcome`, and `map_embedding_response_headers(outcome) -> dict[str, str]`.
- Produces: unchanged HTTP response semantics and endpoint-owned resource/credential/metrics behavior for the workflow-enabled path.

- [ ] **Step 1: Write failing canonical-outcome endpoint tests**

Update workflow-path fakes to return `EmbeddingExecutionOutcome` without `response_headers`. Patch `map_embedding_response_headers` and assert its returned headers are applied to the FastAPI `Response`. Assert the runner factory receives the facade's preparation pipeline and execution coordinator rather than the facade itself.

- [ ] **Step 2: Add exact successful resource-actual fallback tests**

Parameterize successful outcomes and expected committed actual units:

```python
[
    ({"total_tokens": 11, "prompt_tokens": 7}, 11),
    ({"total_tokens": 0, "prompt_tokens": 7}, 7),
    ({"total_tokens": 0, "prompt_tokens": 0}, reserved_units),
]
```

Keep existing failure-after-reservation and cancellation tests asserting reserved units are committed from the endpoint `finally` block and the active-request gauge is decremented.

- [ ] **Step 3: Run focused endpoint tests and confirm they fail against legacy result/header handling**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py -k "workflow or orchestrator or resource or cancellation or header or credential" -q
```

Expected: FAIL where the endpoint still reads compatibility `response_headers` and the runner factory still receives the facade as one combined object.

- [ ] **Step 4: Construct the runner from concrete facade components**

Change `_build_embedding_inline_workflow_runner()` to pass `orchestrator.preparation_pipeline` and `orchestrator.execution_coordinator` as separate constructor arguments while preserving `pre_execute` and the default disabled collector.

- [ ] **Step 5: Consume the canonical outcome and map HTTP headers at the endpoint boundary**

Import `EmbeddingExecutionOutcome` from `request_types` and `map_embedding_response_headers` from `result_mapping`. Treat the workflow result as the canonical outcome for final provider/model credential touch, cache metrics, response formatting, usage logging, and duration metrics. Replace compatibility-header iteration with:

```python
for header_name, header_value in map_embedding_response_headers(outcome).items():
    response.headers[header_name] = header_value
```

- [ ] **Step 6: Preserve endpoint-owned accounting and cleanup**

Calculate successful actual units explicitly in this order:

```python
rg_actual_units = int(outcome.total_tokens or outcome.prompt_tokens or rg_reserved_units)
```

Leave reservation acquisition after preparation through `pre_execute`, commit in `finally`, failure/cancellation reserved-unit charging, and active gauge decrement in their existing endpoint scope. Do not add resource-governor or gauge ownership to the runner.

- [ ] **Step 7: Verify credential, fallback, cache, and rollback parity**

Run the complete endpoint parity module and assert the feature flag still selects the unchanged legacy path. Confirm workflow success performs the endpoint final touch for the actual outcome provider/model while existing executor tests retain validated-batch, adapter, and accepted-late-result touches.

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py -q
```

Expected: PASS with only the already approved Stage 2C fallback-write-identity and Stage 2D source-routing divergences.

- [ ] **Step 8: Commit the endpoint boundary migration**

```bash
git add tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
git commit -m "refactor(embeddings): consume canonical workflow outcomes"
```

---

### Task 4: Cross-Scope Verification and Task Finalization

**Files:**
- Modify: `backlog/tasks/task-12973.5 - Integrate-concrete-Embeddings-steps-with-the-inline-runner.md` through the Backlog.md MCP or CLI workflow.
- Modify: `Docs/superpowers/plans/2026-08-12-embeddings-workflow-stage2e-runner-facade-implementation-plan.md`

**Interfaces:**
- Consumes: completed runner, facade, and endpoint changes from Tasks 1-3.
- Produces: verified Stage 2E branch with current task evidence and no unrecorded security or test failures.

- [ ] **Step 1: Run the complete Stage 2E regression set**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py \
  tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py \
  tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings_isolated/test_preparation_pipeline.py \
  tldw_Server_API/tests/Embeddings_isolated/test_request_types.py \
  tldw_Server_API/tests/Embeddings_isolated/test_result_mapping.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run static checks on every touched Python file**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Embeddings/workflow_runner.py \
  tldw_Server_API/app/core/Embeddings/orchestrator.py \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q \
  tldw_Server_API/app/core/Embeddings/workflow_runner.py \
  tldw_Server_API/app/core/Embeddings/orchestrator.py \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 3: Run Bandit on the touched production scope**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/Embeddings/workflow_runner.py \
  tldw_Server_API/app/core/Embeddings/orchestrator.py \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
  -f json -o /tmp/bandit_embeddings_stage2e.json
```

Expected: no new findings in changed code.

- [ ] **Step 4: Review the final diff against every acceptance criterion**

Confirm direct concrete sequencing, facade-only compatibility mapping, fixed-cardinality safe traces, exact finalizing events, cancellation behavior, endpoint-owned governance/accounting, endpoint header mapping, credential-touch parity, feature-flag rollback, and absence of Stage 3 behavior. Remove only regressions introduced by this branch.

- [ ] **Step 5: Record verification and complete TASK-12973.5**

Through the Backlog.md workflow, check all twelve acceptance criteria and six definition-of-done items, add the exact test/Ruff/compile/Bandit results, record any known pre-existing warnings or skips, and add the final summary and branch reference.

- [ ] **Step 6: Mark this plan complete and commit tracking updates**

Change each completed plan checkbox to `[x]`, then run:

```bash
git add Docs/superpowers/plans/2026-08-12-embeddings-workflow-stage2e-runner-facade-implementation-plan.md "backlog/tasks/task-12973.5 - Integrate-concrete-Embeddings-steps-with-the-inline-runner.md"
git commit -m "docs(embeddings): record stage 2e verification"
```
