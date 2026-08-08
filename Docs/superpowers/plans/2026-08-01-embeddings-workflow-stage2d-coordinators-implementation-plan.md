# Embeddings Workflow Stage 2D Coordinators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkboxes for tracking.

**Goal:** Replace the Embeddings orchestrator's production execution routing with explicit adapter, fallback, and execution coordinators while preserving all existing behavior except the approved source-aware fallback correction.

**Architecture:** Add one focused `execution_coordinator.py` module alongside the Stage 2C provider-attempt boundary. The adapter component owns only preferred-adapter invocation and validation, the fallback component owns ordered candidate traversal and provider-failure policy, and the execution coordinator owns sequencing plus canonical outcome assembly. `EmbeddingRequestOrchestrator.execute()` delegates to the coordinator and the existing legacy result mapper; its superseded private methods remain temporarily for Stage 2E removal so this PR stays reviewable and rollback-friendly.

**Tech Stack:** Python 3.14, frozen dataclasses, structural protocols, async/await, pytest with pytest-asyncio, Ruff, compileall, Bandit.

**Status:** Complete 2026-08-08. Tasks 1-5 and final verification are complete. The approved temporary compatibility-facade duplication remains until Stage 2E, which owns runner integration and cleanup.

## Global Constraints

- Preserve adapter execution before primary readiness, cache access, and provider execution.
- Primary readiness failures propagate unchanged and never enter fallback.
- Only an `EmbeddingDomainError` raised by `ProviderAttemptExecutor.create()` becomes `ProviderCallFailure`.
- Identity resolution, cache-key derivation, cache access, vector validation, postprocessing, and writeback failures propagate unchanged and never activate another provider.
- Fallback candidates use the complete normalized request; primary cache hits and partial primary state never enter a fallback result.
- Exclude the already-attempted primary provider while preserving the remaining policy-approved fallback order.
- Skip `missing_provider_credentials` from fallback readiness or provider calls; apply existing eligibility and exhausted-error precedence to other domain failures.
- Preserve the exact selected error object, including rate-limit retry metadata.
- Preserve current adapter compatibility accounting: zero cache hits and one cache miss per input item on adapter success.
- Do not add workflow persistence, Jobs integration, durable retries, endpoint header migration, trace-summary changes, or Stage 2E runner wiring.
- Do not remove the old private execution branches in this stage; Stage 2E owns compatibility-facade cleanup.
- Raw input, cache keys, backend identities, credentials, and provider bodies must not enter DTO representations, logs, or trace metadata.

---

### Task 1: Adapter Attempt Boundary

**Files:**
- Create: `tldw_Server_API/app/core/Embeddings/execution_coordinator.py`
- Create: `tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py`

**Interfaces:**
- Consumes: `PreparedEmbeddingRequest`, `EmbeddingExecutorOutput`, and `EmbeddingVectorProcessor`.
- Produces: `AdapterAttemptResult(attempted: bool, success: ProviderAttemptSuccess | None)` and `EmbeddingAdapterAttempt.execute(prepared) -> AdapterAttemptResult`.

- [x] **Step 1: Add explicit coordinator test scaffolding**

Build prepared requests directly from the frozen Stage 2B contracts so tests do not depend on the compatibility orchestrator:

```python
def _prepared(
    *,
    texts: list[str] | None = None,
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    fallback_chain: list[str] | None = None,
    fallback_allowed: bool = True,
    execution_path: Literal["legacy", "adapter"] = "legacy",
) -> PreparedEmbeddingRequest:
    ordered_texts = texts or ["alpha"]
    chain = list(fallback_chain or [provider])
    return PreparedEmbeddingRequest(
        normalized_input=NormalizedEmbeddingInput(
            texts=ordered_texts,
            token_counts=[1 for _ in ordered_texts],
            total_tokens=len(ordered_texts),
        ),
        provider_intent=ProviderModelIntent(
            provider=provider,
            model=model,
            requested_provider=provider,
            requested_model=model,
            provider_was_explicit=True,
            model_was_provider_qualified=False,
        ),
        policy_decision=EmbeddingPolicyDecision(
            provider=provider,
            model=model,
            dimensions=None,
            fallback_chain=chain,
            fallback_allowed=fallback_allowed,
            enforce_policy=True,
        ),
        execution_plan=EmbeddingExecutionPlan(
            provider=provider,
            model=model,
            dimensions=None,
            backend_identity="stale-plan-identity",
            fallback_chain=chain,
            execution_path=execution_path,
        ),
        effective_dimension_policy="reduce",
        prompt_tokens=len(ordered_texts),
        total_tokens=len(ordered_texts),
    )
```

Use recording fakes with explicit async methods and stable call records:

```python
@dataclass(frozen=True, slots=True)
class AttemptCall:
    prepared: PreparedEmbeddingRequest
    provider: str
    model: str


class RecordingExecutor:
    def __init__(
        self,
        *,
        adapter_output: EmbeddingExecutorOutput | None = None,
        adapter_error: Exception | None = None,
    ) -> None:
        self.adapter_output = adapter_output
        self.adapter_error = adapter_error
        self.adapter_calls: list[tuple[list[str], str, str, int | None]] = []

    async def create_adapter(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
    ) -> EmbeddingExecutorOutput | None:
        self.adapter_calls.append((texts, provider, model, dimensions))
        if self.adapter_error is not None:
            raise self.adapter_error
        return self.adapter_output
```

Later tasks extend this file with `RecordingReadiness`, `RecordingProviderAttempt`, and `RecordingFallbackCoordinator`; each exposes a `calls` list and returns or raises the exact configured result object.

- [x] **Step 2: Write adapter ordering and result tests**

Add focused tests that instantiate `EmbeddingAdapterAttempt` directly:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_attempt_skips_non_adapter_execution_plan():
    executor = RecordingExecutor()
    attempt = EmbeddingAdapterAttempt(executor=executor)

    result = await attempt.execute(_prepared(execution_path="legacy"))

    assert result == AdapterAttemptResult(attempted=False)
    assert executor.adapter_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_attempt_decline_is_counted_without_provider_result():
    executor = RecordingExecutor(adapter_output=None)
    attempt = EmbeddingAdapterAttempt(executor=executor)

    result = await attempt.execute(_prepared(execution_path="adapter"))

    assert result == AdapterAttemptResult(attempted=True)
    assert len(executor.adapter_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_attempt_validates_and_processes_success():
    executor = RecordingExecutor(
        adapter_output=EmbeddingExecutorOutput(
            vectors=[[3.0, 4.0]],
            embeddings_from_adapter=True,
        )
    )
    attempt = EmbeddingAdapterAttempt(executor=executor)

    result = await attempt.execute(_prepared(execution_path="adapter"))

    assert result.success == ProviderAttemptSuccess(
        vectors=[[3.0, 4.0]],
        provider="openai",
        model="text-embedding-3-small",
        cache_hits=0,
        cache_misses=1,
        embeddings_from_adapter=True,
    )
```

Also assert that an adapter exception is the exact exception re-raised and that `EmbeddingExecutorOutput(..., embeddings_from_adapter=False)` is an attempted decline.

- [x] **Step 3: Run adapter tests and verify RED**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py -k adapter -q
```

Expected: collection fails because `execution_coordinator.py` and its adapter contracts do not exist.

- [x] **Step 4: Implement the minimal adapter boundary**

Accept the existing provider executor object and discover its optional adapter capability dynamically, matching the compatibility orchestrator. Add the frozen result DTO:

```python
@dataclass(frozen=True, slots=True)
class AdapterAttemptResult:
    attempted: bool
    success: ProviderAttemptSuccess | None = None

    def __post_init__(self) -> None:
        if self.success is not None and not self.attempted:
            raise ValueError("adapter success requires an attempted adapter path")
```

Implement `EmbeddingAdapterAttempt.execute()` so it:

1. Stores the injected `ProviderAttemptExecutor` and returns `attempted=False` when the plan is not `adapter` or `getattr(executor, "create_adapter", None)` is not callable.
2. Invokes `create_adapter` with the complete ordered input when available.
3. Returns `attempted=True, success=None` for `None` or non-adapter output.
4. Validates vector count and applies request-specific postprocessing.
5. Returns a `ProviderAttemptSuccess` with compatibility cache counts on success.
6. Does not catch exceptions.

- [x] **Step 5: Run adapter tests and verify GREEN**

Run the Step 3 command. Expected: all adapter-selected tests pass.

- [x] **Step 6: Commit the adapter boundary**

```bash
git add tldw_Server_API/app/core/Embeddings/execution_coordinator.py tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py
git commit -m "refactor(embeddings): extract adapter attempt boundary"
```

### Task 2: Source-Aware Fallback Coordinator

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/execution_coordinator.py`
- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py`

**Interfaces:**
- Consumes: `EmbeddingProviderReadinessCheck.check(provider, model)`, `EmbeddingProviderAttempt.execute(prepared, provider=..., model=...)`, and the primary `ProviderCallFailure`.
- Produces: `FallbackExecutionSuccess(success: ProviderAttemptSuccess, attempt_count: int)` and `EmbeddingFallbackCoordinator.execute(prepared, primary_failure) -> FallbackExecutionSuccess`.

- [x] **Step 1: Write failing fallback traversal tests**

Add the provider-result helpers and recording fakes used by fallback tests:

```python
def _retryable(provider: str) -> EmbeddingProviderError:
    return EmbeddingProviderError(
        "provider_unavailable",
        "provider unavailable",
        provider=provider,
        model={
            "openai": "text-embedding-3-small",
            "cohere": "embed-english-v3.0",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        }[provider],
        retryable=True,
    )


def _success(provider: str) -> ProviderAttemptSuccess:
    return ProviderAttemptSuccess(
        vectors=[[0.25, 0.75]],
        provider=provider,
        model={
            "cohere": "embed-english-v3.0",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        }.get(provider, "text-embedding-3-small"),
        cache_hits=0,
        cache_misses=1,
    )


def _model_map() -> dict[str, object]:
    return {
        "openai:text-embedding-3-small": {
            "cohere": "embed-english-v3.0",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        }
    }


class RecordingReadiness:
    def __init__(
        self,
        errors: dict[str, EmbeddingDomainError] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.errors = errors or {}
        self.events = events
        self.calls: list[tuple[str, str]] = []

    async def check(self, provider: str, model: str) -> None:
        self.calls.append((provider, model))
        if self.events is not None:
            self.events.append(f"readiness:{provider}")
        error = self.errors.get(provider)
        if error is not None:
            raise error


class RecordingProviderAttempt:
    def __init__(
        self,
        outcomes: dict[str, ProviderAttemptSuccess | ProviderCallFailure],
        raised: dict[str, EmbeddingDomainError] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.raised = raised or {}
        self.events = events
        self.calls: list[AttemptCall] = []

    async def execute(
        self,
        prepared: PreparedEmbeddingRequest,
        *,
        provider: str,
        model: str,
    ) -> ProviderAttemptSuccess | ProviderCallFailure:
        self.calls.append(AttemptCall(prepared, provider, model))
        if self.events is not None:
            self.events.append(f"attempt:{provider}")
        error = self.raised.get(provider)
        if error is not None:
            raise error
        return self.outcomes[provider]
```

Cover these behaviors with those fakes:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_excludes_primary_and_preserves_candidate_order():
    readiness = RecordingReadiness()
    attempt = RecordingProviderAttempt(
        outcomes={
            "cohere": ProviderCallFailure(_retryable("cohere")),
            "huggingface": _success("huggingface"),
        }
    )
    coordinator = EmbeddingFallbackCoordinator(
        readiness=readiness,
        provider_attempt=attempt,
        settings_fallback_model_map=_model_map(),
    )

    result = await coordinator.execute(
        _prepared(fallback_chain=["openai", "cohere", "openai", "huggingface"]),
        ProviderCallFailure(_retryable("openai")),
    )

    assert readiness.calls == [
        ("cohere", "embed-english-v3.0"),
        ("huggingface", "sentence-transformers/all-MiniLM-L6-v2"),
    ]
    assert [call.provider for call in attempt.calls] == ["cohere", "huggingface"]
    assert result.attempt_count == 2
```

Add separate tests proving:

- readiness and provider-call missing credentials skip the candidate;
- eligible readiness and `ProviderCallFailure` results continue;
- ineligible failures raise the exact original object immediately;
- a rate-limit error is selected on exhaustion with its original `retry_after` value;
- a fallback success always receives the complete prepared request;
- backend identity, key, cache, validation, postprocessing, and writeback errors escaping `EmbeddingProviderAttempt` are not caught and do not advance to another candidate.

- [x] **Step 2: Run fallback tests and verify RED**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py -k fallback -q
```

Expected: failures because the fallback coordinator contracts are absent.

- [x] **Step 3: Implement policy helpers and fallback traversal**

Copy the existing eligibility constants and behavior into the new module without changing values. Retain the orchestrator's private copies until Stage 2E removes its superseded private execution branches:

```python
_NON_FALLBACKABLE_ERROR_CODES = frozenset(
    {
        "empty_input",
        "invalid_input_type",
        "too_many_inputs",
        "input_too_long",
        "invalid_token_array",
        "unknown_provider",
        "provider_model_mismatch",
        "invalid_dimensions",
        "provider_denied",
        "model_denied",
        "model_required",
        "provider_unsupported",
        "missing_provider_credentials",
        "provider_malformed_response",
    }
)


def _is_fallback_eligible(error: EmbeddingDomainError) -> bool:
    return bool(error.retryable) and error.code not in _NON_FALLBACKABLE_ERROR_CODES


def _select_exhausted_error(
    errors: Sequence[EmbeddingDomainError],
) -> EmbeddingDomainError | None:
    for error in errors:
        if isinstance(error, EmbeddingProviderError) and not error.retryable:
            return error
    for error in errors:
        if error.code == "provider_rate_limited":
            return error
    return errors[-1] if errors else None
```

Implement candidate traversal with this exception boundary:

```python
try:
    await self._readiness.check(provider, model)
except EmbeddingDomainError as error:
    if error.code == "missing_provider_credentials":
        continue
    errors.append(error)
    if not _is_fallback_eligible(error):
        raise
    continue

attempt_result = await self._provider_attempt.execute(
    prepared,
    provider=provider,
    model=model,
)
if isinstance(attempt_result, ProviderAttemptSuccess):
    return FallbackExecutionSuccess(attempt_result, attempt_count)

error = attempt_result.error
if error.code == "missing_provider_credentials":
    continue
errors.append(error)
if not _is_fallback_eligible(error):
    raise error
```

Do not wrap the provider-attempt call in a broad `EmbeddingDomainError` catch. This is the approved Stage 2D source-routing correction.

- [x] **Step 4: Run fallback tests and verify GREEN**

Run the Step 2 command. Expected: all fallback-selected tests pass.

- [x] **Step 5: Commit the fallback coordinator**

```bash
git add tldw_Server_API/app/core/Embeddings/execution_coordinator.py tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py
git commit -m "refactor(embeddings): extract fallback coordinator"
```

### Task 3: Primary Execution Coordinator and Canonical Outcome

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/execution_coordinator.py`
- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py`

**Interfaces:**
- Consumes: `EmbeddingAdapterAttempt`, `EmbeddingProviderReadinessCheck`, `EmbeddingProviderAttempt`, `EmbeddingFallbackCoordinator`, and `assemble_embedding_execution_outcome`.
- Produces: `EmbeddingExecutionCoordinator.execute(prepared) -> EmbeddingExecutionOutcome`.

- [x] **Step 1: Write failing execution-order and counter tests**

Add execution-level adapter and fallback fakes plus a single constructor helper:

```python
class RecordingAdapterAttempt:
    def __init__(self, result: AdapterAttemptResult, events: list[str]) -> None:
        self.result = result
        self.events = events

    async def execute(self, prepared: PreparedEmbeddingRequest) -> AdapterAttemptResult:
        del prepared
        if self.result.attempted:
            self.events.append("adapter")
        return self.result


class RecordingFallbackCoordinator:
    def __init__(
        self,
        result: FallbackExecutionSuccess | None = None,
    ) -> None:
        self.result = result or FallbackExecutionSuccess(_success("huggingface"), 1)
        self.calls: list[tuple[PreparedEmbeddingRequest, ProviderCallFailure]] = []

    async def execute(
        self,
        prepared: PreparedEmbeddingRequest,
        primary_failure: ProviderCallFailure,
    ) -> FallbackExecutionSuccess:
        self.calls.append((prepared, primary_failure))
        return self.result


def _coordinator(
    *,
    events: list[str] | None = None,
    adapter_declines: bool = False,
    readiness_error: EmbeddingDomainError | None = None,
    primary_result: ProviderAttemptSuccess | ProviderCallFailure | None = None,
    fallback_result: FallbackExecutionSuccess | None = None,
    fallback: RecordingFallbackCoordinator | None = None,
) -> EmbeddingExecutionCoordinator:
    ordered_events = events if events is not None else []
    adapter = RecordingAdapterAttempt(
        AdapterAttemptResult(attempted=adapter_declines),
        ordered_events,
    )
    readiness = RecordingReadiness(
        errors={"openai": readiness_error} if readiness_error is not None else None,
        events=ordered_events,
    )
    attempt = RecordingProviderAttempt(
        outcomes={"openai": primary_result or _success("openai")},
        events=ordered_events,
    )
    fallback_coordinator = fallback or RecordingFallbackCoordinator(fallback_result)
    return EmbeddingExecutionCoordinator(
        adapter_attempt=adapter,
        readiness=readiness,
        provider_attempt=attempt,
        fallback_coordinator=fallback_coordinator,
    )
```

Add tests for the exact sequencing and aggregate counters:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_execution_orders_adapter_before_primary_readiness_and_attempt():
    events: list[str] = []
    coordinator = _coordinator(events=events, adapter_declines=True)

    outcome = await coordinator.execute(_prepared(execution_path="adapter"))

    assert events == ["adapter", "readiness:openai", "attempt:openai"]
    assert outcome.attempt_count == 2
    assert outcome.fallback_attempt_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_primary_readiness_failure_does_not_enter_fallback():
    error = _retryable("openai")
    fallback = RecordingFallbackCoordinator()
    coordinator = _coordinator(readiness_error=error, fallback=fallback)

    with pytest.raises(EmbeddingDomainError) as raised:
        await coordinator.execute(_prepared())

    assert raised.value is error
    assert fallback.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_eligible_primary_provider_failure_uses_complete_request_fallback():
    coordinator = _coordinator(
        primary_result=ProviderCallFailure(_retryable("openai")),
        fallback_result=FallbackExecutionSuccess(
            success=_success("huggingface"),
            attempt_count=2,
        ),
    )

    outcome = await coordinator.execute(_prepared())

    assert outcome.provider == "huggingface"
    assert outcome.fallback_from == "openai"
    assert outcome.attempt_count == 3
    assert outcome.fallback_attempt_count == 2
```

Also cover adapter success, full primary cache hit, non-retryable primary failure, fallback-disabled primary failure, and successful primary execution. Assert exact `cache_hits`, `cache_misses`, provider/model, adapter flag, and vector order in every resulting outcome.

- [x] **Step 2: Run execution tests and verify RED**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py -k execution -q
```

Expected: failures because `EmbeddingExecutionCoordinator` does not exist.

- [x] **Step 3: Implement execution sequencing**

Implement the coordinator in this exact order:

```python
adapter_result = await self._adapter_attempt.execute(prepared)
base_attempt_count = int(adapter_result.attempted)
if adapter_result.success is not None:
    return self._assemble(
        prepared,
        adapter_result.success,
        attempt_count=base_attempt_count,
        fallback_attempt_count=0,
        fallback_from=None,
    )

plan = prepared.execution_plan
await self._readiness.check(plan.provider, plan.model)
base_attempt_count += 1
primary = await self._provider_attempt.execute(
    prepared,
    provider=plan.provider,
    model=plan.model,
)
if isinstance(primary, ProviderAttemptSuccess):
    return self._assemble(
        prepared,
        primary,
        attempt_count=base_attempt_count,
        fallback_attempt_count=0,
        fallback_from=None,
    )

error = primary.error
if not prepared.policy_decision.fallback_allowed or not _is_fallback_eligible(error):
    raise error

fallback = await self._fallback.execute(prepared, primary)
return self._assemble(
    prepared,
    fallback.success,
    attempt_count=base_attempt_count + fallback.attempt_count,
    fallback_attempt_count=fallback.attempt_count,
    fallback_from=plan.provider,
)
```

Use `assemble_embedding_execution_outcome()` for all three success paths. Do not create HTTP headers in this module.

- [x] **Step 4: Run the full coordinator test file and verify GREEN**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py -q
```

Expected: all coordinator tests pass.

- [x] **Step 5: Commit the execution coordinator**

```bash
git add tldw_Server_API/app/core/Embeddings/execution_coordinator.py tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py
git commit -m "refactor(embeddings): extract execution coordinator"
```

### Task 4: Production Orchestrator Delegation and Source-Routing Correction

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/orchestrator.py`
- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`
- Modify: `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`

**Interfaces:**
- Consumes: `EmbeddingExecutionCoordinator.execute(prepared) -> EmbeddingExecutionOutcome` and `map_outcome_to_legacy_execution_result(outcome) -> EmbeddingExecutionResult`.
- Produces: unchanged `EmbeddingRequestOrchestrator.execute(prepared) -> EmbeddingExecutionResult` public behavior, with approved source-aware fallback routing.

- [x] **Step 1: Rewrite the Stage 2D guardrail as the new expected behavior**

Change `test_current_fallback_wide_domain_catch_advances_after_infrastructure_failure` into a source-routing regression that expects the exact fallback infrastructure error to propagate and prevents later fallback candidates:

```python
@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    ["backend_identity", "cache_key", "cache_get", "cache_set"],
)
async def test_fallback_non_provider_failure_propagates_without_advancing(boundary):
    # Existing fixture setup remains, including a successful later provider.
    with pytest.raises(EmbeddingExecutionError) as raised:
        await orchestrator.execute(prepared)

    assert raised.value is original
    assert "huggingface" not in [call["provider"] for call in executor.calls]
```

Add an endpoint parity correction case that asserts the workflow-enabled path now propagates fallback cache/identity failures rather than silently activating another provider. Do not modify unchanged parity scenarios.

- [x] **Step 2: Run production-routing tests and verify RED**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py -k "fallback_non_provider_failure" tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py -q
```

Expected: the orchestrator regression fails because the old broad fallback catch still advances; the parity suite otherwise remains green.

- [x] **Step 3: Construct and delegate to the coordinator**

In `EmbeddingRequestOrchestrator.__init__`, build one shared vector processor, readiness check, provider attempt, adapter attempt, fallback coordinator, and execution coordinator from the existing injected dependencies. Keep constructor arguments unchanged.

Replace the public execution body with:

```python
async def execute(self, prepared: PreparedEmbeddingRequest) -> EmbeddingExecutionResult:
    """Execute a prepared request through the Stage 2 coordinator boundary."""
    outcome = await self._execution_coordinator.execute(prepared)
    return map_outcome_to_legacy_execution_result(outcome)
```

Leave `_execute_misses`, `_execute_coherent_fallback`, `_execute_adapter`, `_cache_key`, `_response_headers`, and their compatibility helpers in place for Stage 2E removal. They must be unreachable from the public `execute()` path after this step.

- [x] **Step 4: Run focused orchestrator and parity tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_result_mapping.py tldw_Server_API/tests/Embeddings_isolated/test_request_types.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py -q
```

Expected: all selected tests pass. The only approved assertion change is source-aware fallback failure routing.

- [x] **Step 5: Commit production delegation**

```bash
git add tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
git commit -m "refactor(embeddings): route execution through coordinators"
```

### Task 5: Documentation, Regression, Security, and Backlog Finalization

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-18-embeddings-workflow-stage2-concrete-api-steps-design.md`
- Modify: `Docs/superpowers/plans/2026-08-01-embeddings-workflow-stage2d-coordinators-implementation-plan.md`
- Modify: `backlog/tasks/task-12973.4 - Extract-Embeddings-fallback-and-execution-coordinators.md`
- Verify: `tldw_Server_API/app/core/Embeddings/execution_coordinator.py`
- Verify: `tldw_Server_API/app/core/Embeddings/orchestrator.py`
- Verify: `tldw_Server_API/app/core/Embeddings/provider_attempt.py`

**Interfaces:**
- Consumes: completed Stage 2D production and test changes.
- Produces: verified, reviewable Stage 2D branch with authoritative Backlog evidence.

- [x] **Step 1: Run the focused Stage 2D regression suite**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_request_types.py tldw_Server_API/tests/Embeddings_isolated/test_result_mapping.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py -q
```

Expected: all selected tests pass.

- [x] **Step 2: Run static and security checks**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check tldw_Server_API/app/core/Embeddings/execution_coordinator.py tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_execution_coordinator.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q tldw_Server_API/app/core/Embeddings/execution_coordinator.py tldw_Server_API/app/core/Embeddings/orchestrator.py
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Embeddings/execution_coordinator.py tldw_Server_API/app/core/Embeddings/orchestrator.py -f json -o /tmp/bandit_embeddings_stage2d.json
```

Expected: Ruff and compileall exit zero; Bandit reports zero findings in touched production code.

- [x] **Step 3: Run diff and scope review**

Run:

```bash
git diff --check
git diff --stat origin/dev
git status --short
```

Confirm the diff contains only the Stage 2D coordinator, focused tests, orchestrator delegation, approved source-routing assertion change, spec status correction, implementation plan, and Backlog updates.

- [x] **Step 4: Finalize Backlog.md through the official Backlog workflow**

Use `backlog task edit 12973.4` to:

1. Check acceptance criteria 1 through 9.
2. Check Definition of Done items 1 through 6.
3. Record exact test totals, Ruff/compileall/Bandit results, and any known skips.
4. Set modified files and the plan reference if supported by the installed CLI; otherwise record them in implementation notes.
5. Add a final summary that explains both the source-aware routing correction and why Stage 2E retains endpoint/runner integration ownership.
6. Set status to `Done` only after all verification succeeds.

- [x] **Step 5: Commit final documentation and tracking**

```bash
git add Docs/superpowers/specs/2026-07-18-embeddings-workflow-stage2-concrete-api-steps-design.md Docs/superpowers/plans/2026-08-01-embeddings-workflow-stage2d-coordinators-implementation-plan.md "backlog/tasks/task-12973.4 - Extract-Embeddings-fallback-and-execution-coordinators.md"
git commit -m "docs(embeddings): finalize stage 2d coordinators"
```

## Self-Review Checklist

- Spec coverage: adapter ordering, primary readiness policy, full-request fallback, missing-credential skipping, source-aware provider failures, primary exclusion, model mapping, eligibility, error identity, rate-limit precedence, canonical attempt counters, and privacy constraints all map to explicit tasks.
- Scope check: no inline runner, endpoint header migration, trace-summary event, resource-governor, credential-touch, Jobs, persistence, or Stage 3 behavior is introduced.
- Placeholder scan: no `TBD`, `TODO`, generic error-handling step, or undefined implementation dependency remains.
- Type consistency: adapter and fallback result DTOs carry `ProviderAttemptSuccess`; the execution coordinator returns `EmbeddingExecutionOutcome`; the orchestrator maps that outcome to `EmbeddingExecutionResult`.
- Source-routing check: the fallback coordinator catches readiness domain errors and inspects `ProviderCallFailure`, but never broadly catches domain errors escaping provider attempts.
- Compatibility check: the orchestrator public constructor and return type remain unchanged, and the legacy result mapper remains the temporary header-compatibility boundary.
- Verification check: focused regression, Ruff, compileall, Bandit, diff review, and Backlog finalization commands are explicit.
