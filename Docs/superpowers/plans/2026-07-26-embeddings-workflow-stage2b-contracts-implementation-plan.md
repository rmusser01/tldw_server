# Embeddings Workflow Stage 2B Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the Embeddings preparation pipeline, deterministic vector processor, canonical HTTP-independent outcome, endpoint header mapper, and temporary legacy result adapter without changing endpoint or provider execution behavior.

**Architecture:** Move immutable request/outcome DTOs into `request_types.py`; put stateful preparation and vector dependencies behind focused components; keep result assembly and response mapping as pure functions. `EmbeddingRequestOrchestrator.prepare` and vector handling delegate to the extracted components, while its existing execute flow and the production endpoint continue returning `EmbeddingExecutionResult` until later Stage 2 work. The canonical outcome/header mapper are introduced with focused parity tests but are not wired into the endpoint or inline runner in this stage.

**Tech Stack:** Python 3.14, frozen dataclasses, typing protocols/literals, pytest, pytest-asyncio, Ruff, Bandit.

**Tracking:** `TASK-12973.2`

**Approved design:** `Docs/superpowers/specs/2026-07-18-embeddings-workflow-stage2-concrete-api-steps-design.md`

---

## Scope Guardrails

- Do not change FastAPI endpoint wiring, public response formatting, resource-governor accounting, metrics, credential touching, cache semantics, provider traversal, or fallback routing.
- Do not implement the Stage 2C post-call identity correction or the Stage 2D source-aware fallback correction.
- Keep `response_header_count` accepted and emitted by the current legacy inline runner until Stage 2E replaces its result contract.
- Define `attempt_count` and `fallback_attempt_count` on the canonical outcome, but require callers to supply them; do not infer incomplete counts from the current orchestrator.
- Preserve `EmbeddingExecutionResult` and re-export `PreparedEmbeddingRequest` from `orchestrator.py` for existing imports.
- Keep the endpoint header mapper pure and unwired until Stage 2E.
- Document the legacy result adapter as the sole Stage 6 compatibility exception to endpoint-owned headers.

## File Map

- Create `tldw_Server_API/app/core/Embeddings/preparation.py`: ordered preparation pipeline and one-argument phase sink.
- Create `tldw_Server_API/app/core/Embeddings/vector_processing.py`: vector validation, canonicalization, and dimension processing.
- Create `tldw_Server_API/app/core/Embeddings/result_mapping.py`: canonical outcome assembly, endpoint header mapping, and Stage 6 legacy adapter.
- Modify `tldw_Server_API/app/core/Embeddings/request_types.py`: move `PreparedEmbeddingRequest` here and add `EmbeddingExecutionOutcome`.
- Modify `tldw_Server_API/app/core/Embeddings/workflow_types.py`: add `resolving_intent`, aggregate attempt counters, and completed-event validation.
- Modify `tldw_Server_API/app/core/Embeddings/orchestrator.py`: delegate preparation and vector processing; preserve execute behavior and legacy exports.
- Modify `tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py`: phase/status/finalization contract tests.
- Create `tldw_Server_API/tests/Embeddings_isolated/test_preparation_pipeline.py`: order, phase-sink, and error-precedence tests.
- Create `tldw_Server_API/tests/Embeddings_isolated/test_vector_processing.py`: deterministic vector behavior tests.
- Create `tldw_Server_API/tests/Embeddings_isolated/test_result_mapping.py`: outcome immutability, header parity, adapter compatibility, and adapter cache-count tests.
- Modify `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`: point ordering probes at the extracted preparation module and retain behavioral parity.
- Modify `backlog/tasks/task-12973.2 - Extract-Embeddings-preparation-and-result-contracts.md`: record progress, evidence, touched files, and final status through Backlog.md tooling only.

### Task 1: Extend Workflow Contract Vocabulary

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/workflow_types.py`
- Test: `tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py`

- [x] **Step 1: Write failing phase and finalization tests**

Add tests that inspect the literal contracts and validate terminal event construction:

```python
from typing import get_args


def test_stage2_phase_contract_adds_resolving_intent_without_completed_phase():
    assert "resolving_intent" in get_args(workflow_types.EmbeddingWorkflowPhase)
    assert "completed" not in get_args(workflow_types.EmbeddingWorkflowPhase)
    assert "completed" in get_args(workflow_types.EmbeddingWorkflowStatus)
    assert "resolving_intent" in workflow_types.SAFE_METADATA_ENUM_VALUES["phase"]


def test_stage2_trace_metadata_accepts_aggregate_attempt_counts():
    metadata = safe_workflow_metadata({"attempt_count": 3, "fallback_attempt_count": 2})
    assert metadata == {"attempt_count": 3, "fallback_attempt_count": 2}


def test_workflow_completed_requires_finalizing_completed_contract():
    event = EmbeddingWorkflowEvent(
        event_type="workflow_completed",
        workflow_id=WORKFLOW_ID,
        phase="finalizing",
        status="completed",
    )
    assert event.phase == "finalizing"
    assert event.status == "completed"

    with pytest.raises(EmbeddingWorkflowTraceError):
        EmbeddingWorkflowEvent(
            event_type="workflow_completed",
            workflow_id=WORKFLOW_ID,
            phase="executing",
            status="completed",
        )

    with pytest.raises(EmbeddingWorkflowTraceError):
        EmbeddingWorkflowEvent(
            event_type="workflow_completed",
            workflow_id=WORKFLOW_ID,
            phase="finalizing",
            status="running",
        )
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py -q
```

Expected: failures because `resolving_intent` and attempt counters are absent and `workflow_completed` does not validate its finalizing contract.

- [x] **Step 3: Implement the minimal workflow contract changes**

In `workflow_types.py`:

```python
EmbeddingWorkflowPhase = Literal[
    "created",
    "resolving_intent",
    "normalizing",
    "resolving_policy",
    "planning",
    "serving_cache",
    "executing",
    "postprocessing",
    "persisting_outputs",
    "finalizing",
]
```

Add `resolving_intent` to the safe `phase` enum and add `attempt_count` plus `fallback_attempt_count` to `SAFE_METADATA_NONNEGATIVE_INTEGER_FIELDS`. Preserve `response_header_count` for the still-legacy Stage 2B runner.

In `EmbeddingWorkflowEvent.__post_init__`, reject a `workflow_completed` event unless `phase == "finalizing"` and `status == "completed"`, then perform the existing workflow-ID and metadata validation. Update the existing bounded-collector overflow test so its third `workflow_completed` event supplies `phase="finalizing"` and `status="completed"`; the event must be valid and the assertion must continue proving the collector itself fails closed at its configured cap.

- [x] **Step 4: Run the workflow contract tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

- [x] **Step 5: Commit the workflow contract slice**

```bash
git add tldw_Server_API/app/core/Embeddings/workflow_types.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py
git commit -m "refactor(embeddings): extend stage 2 workflow contracts"
```

### Task 2: Add Canonical Outcome and Mapping Boundaries

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/request_types.py`
- Create: `tldw_Server_API/app/core/Embeddings/result_mapping.py`
- Create: `tldw_Server_API/tests/Embeddings_isolated/test_result_mapping.py`
- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_request_types.py`

- [x] **Step 1: Write failing outcome and mapper tests**

Cover these behaviors:

```python
def _prepared_request() -> PreparedEmbeddingRequest:
    return PreparedEmbeddingRequest(
        normalized_input=NormalizedEmbeddingInput(
            texts=["one", "two"],
            token_counts=[1, 1],
            total_tokens=2,
        ),
        provider_intent=ProviderModelIntent(
            provider="openai",
            model="text-embedding-3-small",
            requested_provider="openai",
            requested_model="text-embedding-3-small",
            provider_was_explicit=True,
            model_was_provider_qualified=False,
        ),
        policy_decision=EmbeddingPolicyDecision(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
            fallback_chain=["openai", "huggingface"],
            fallback_allowed=True,
            enforce_policy=True,
        ),
        execution_plan=EmbeddingExecutionPlan(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=None,
            backend_identity="openai:test",
            fallback_chain=["openai", "huggingface"],
        ),
        effective_dimension_policy="reduce",
        prompt_tokens=2,
        total_tokens=2,
    )


def _outcome(
    *,
    fallback_from: str | None = None,
    requested_dimensions: int | None = None,
) -> EmbeddingExecutionOutcome:
    return EmbeddingExecutionOutcome(
        vectors=((0.1, 0.2),),
        provider="huggingface" if fallback_from else "openai",
        model="all-MiniLM-L6-v2" if fallback_from else "text-embedding-3-small",
        prompt_tokens=2,
        total_tokens=2,
        cache_hits=0,
        cache_misses=1,
        requested_dimensions=requested_dimensions,
        effective_dimension_policy="reduce",
        attempt_count=2 if fallback_from else 1,
        fallback_attempt_count=1 if fallback_from else 0,
        fallback_from=fallback_from,
    )


def test_canonical_outcome_has_no_http_header_contract():
    hints = get_type_hints(EmbeddingExecutionOutcome)
    assert "response_headers" not in hints
    assert hints["vectors"] == tuple[tuple[float, ...], ...]
    assert hints["attempt_count"] is int
    assert hints["fallback_attempt_count"] is int


def test_result_assembly_freezes_vectors_and_preserves_adapter_cache_counts():
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    outcome = assemble_embedding_execution_outcome(
        _prepared_request(),
        vectors=vectors,
        provider="huggingface",
        model="all-MiniLM-L6-v2",
        cache_hits=0,
        cache_misses=2,
        fallback_from=None,
        embeddings_from_adapter=True,
        attempt_count=1,
        fallback_attempt_count=0,
    )
    vectors[0][0] = 9.0
    assert outcome.vectors == ((0.1, 0.2), (0.3, 0.4))
    assert (outcome.cache_hits, outcome.cache_misses) == (0, 2)


@pytest.mark.parametrize(
    ("fallback_from", "dimensions", "expected"),
    [
        (None, None, {"X-Embeddings-Provider": "openai"}),
        (
            None,
            2,
            {
                "X-Embeddings-Provider": "openai",
                "X-Embeddings-Dimensions-Policy": "reduce",
            },
        ),
        ("huggingface", None, {"X-Embeddings-Provider": "huggingface"}),
        (
            "openai",
            2,
            {
                "X-Embeddings-Provider": "huggingface",
                "X-Embeddings-Fallback-From": "openai",
                "X-Embeddings-Dimensions-Policy": "reduce",
            },
        ),
    ],
)
def test_endpoint_header_mapper_matches_legacy_contract(fallback_from, dimensions, expected):
    outcome = _outcome(fallback_from=fallback_from, requested_dimensions=dimensions)
    assert map_embedding_response_headers(outcome) == expected


def test_legacy_adapter_preserves_result_shape_without_sharing_mutable_vectors():
    outcome = _outcome()
    legacy = map_outcome_to_legacy_execution_result(outcome)
    assert legacy == EmbeddingExecutionResult(
        vectors=[[0.1, 0.2]],
        provider=outcome.provider,
        model=outcome.model,
        prompt_tokens=outcome.prompt_tokens,
        total_tokens=outcome.total_tokens,
        cache_hits=outcome.cache_hits,
        cache_misses=outcome.cache_misses,
        fallback_from=outcome.fallback_from,
        response_headers=map_embedding_response_headers(outcome),
        embeddings_from_adapter=outcome.embeddings_from_adapter,
    )
    legacy.vectors[0][0] = 9.0
    assert outcome.vectors[0][0] == 0.1
```

- [x] **Step 2: Run the focused tests and verify RED**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_request_types.py tldw_Server_API/tests/Embeddings_isolated/test_result_mapping.py -q
```

Expected: import failures because the outcome and mapping module do not exist.

- [x] **Step 3: Add immutable outcome contracts**

Move `PreparedEmbeddingRequest` from `orchestrator.py` into `request_types.py` without changing its fields. Add:

```python
@dataclass(frozen=True, slots=True)
class EmbeddingExecutionOutcome:
    vectors: tuple[tuple[float, ...], ...]
    provider: str
    model: str
    prompt_tokens: int
    total_tokens: int
    cache_hits: int
    cache_misses: int
    requested_dimensions: int | None
    effective_dimension_policy: str
    attempt_count: int
    fallback_attempt_count: int
    fallback_from: str | None = None
    embeddings_from_adapter: bool = False
```

Do not add HTTP headers. Export both DTOs from `request_types.py`.

- [x] **Step 4: Implement pure result assembly and mapping**

Create `result_mapping.py` with:

```python
def assemble_embedding_execution_outcome(
    prepared: PreparedEmbeddingRequest,
    *,
    vectors: Sequence[Sequence[float]],
    provider: str,
    model: str,
    cache_hits: int,
    cache_misses: int,
    fallback_from: str | None,
    embeddings_from_adapter: bool,
    attempt_count: int,
    fallback_attempt_count: int,
) -> EmbeddingExecutionOutcome:
    return EmbeddingExecutionOutcome(
        vectors=tuple(tuple(vector) for vector in vectors),
        provider=provider,
        model=model,
        prompt_tokens=prepared.prompt_tokens,
        total_tokens=prepared.total_tokens,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        requested_dimensions=prepared.execution_plan.dimensions,
        effective_dimension_policy=prepared.effective_dimension_policy,
        fallback_from=fallback_from,
        embeddings_from_adapter=embeddings_from_adapter,
        attempt_count=attempt_count,
        fallback_attempt_count=fallback_attempt_count,
    )


def map_embedding_response_headers(outcome: EmbeddingExecutionOutcome) -> dict[str, str]:
    headers = {"X-Embeddings-Provider": outcome.provider}
    if outcome.fallback_from and outcome.fallback_from != outcome.provider:
        headers["X-Embeddings-Fallback-From"] = outcome.fallback_from
    if outcome.requested_dimensions is not None:
        headers["X-Embeddings-Dimensions-Policy"] = outcome.effective_dimension_policy
    return headers


def map_outcome_to_legacy_execution_result(
    outcome: EmbeddingExecutionOutcome,
) -> EmbeddingExecutionResult:
    """Stage 6 compatibility exception for callers that still require headers."""
    return EmbeddingExecutionResult(
        vectors=[list(vector) for vector in outcome.vectors],
        provider=outcome.provider,
        model=outcome.model,
        prompt_tokens=outcome.prompt_tokens,
        total_tokens=outcome.total_tokens,
        cache_hits=outcome.cache_hits,
        cache_misses=outcome.cache_misses,
        fallback_from=outcome.fallback_from,
        response_headers=map_embedding_response_headers(outcome),
        embeddings_from_adapter=outcome.embeddings_from_adapter,
    )
```

Document explicitly that the header mapper is endpoint-owned but remains unwired until Stage 2E, and the legacy mapper is scheduled for Stage 6 removal.

- [x] **Step 5: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all tests pass.

- [x] **Step 6: Commit the result-contract slice**

```bash
git add tldw_Server_API/app/core/Embeddings/request_types.py tldw_Server_API/app/core/Embeddings/result_mapping.py tldw_Server_API/tests/Embeddings_isolated/test_request_types.py tldw_Server_API/tests/Embeddings_isolated/test_result_mapping.py
git commit -m "refactor(embeddings): add canonical result contracts"
```

### Task 3: Extract the Ordered Preparation Pipeline

**Files:**
- Create: `tldw_Server_API/app/core/Embeddings/preparation.py`
- Modify: `tldw_Server_API/app/core/Embeddings/orchestrator.py`
- Create: `tldw_Server_API/tests/Embeddings_isolated/test_preparation_pipeline.py`
- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`

- [x] **Step 1: Write failing pipeline order and phase-safety tests**

Build a pipeline through a local test factory with the same lightweight token/config fixtures as `test_embedding_orchestrator.py`. Patch the four module boundaries and assert:

```python
def _context() -> EmbeddingRequestContext:
    return EmbeddingRequestContext(
        user_id="u1",
        model_field="sentence-transformers/all-MiniLM-L6-v2",
        provider_header="huggingface",
        dimensions=None,
        encoding_format="float",
        request_id="req-1",
    )


def _tokens_to_texts(tokens_input, model):
    del model
    if tokens_input and isinstance(tokens_input[0], int):
        return ["decoded"], len(tokens_input), [len(tokens_input)]
    return [f"decoded-{index}" for index, _ in enumerate(tokens_input)], 0, [
        len(item) for item in tokens_input
    ]


def _pipeline(**overrides) -> EmbeddingPreparationPipeline:
    kwargs = {
        "count_tokens": lambda text, model: len(text.split()),
        "tokens_to_texts": _tokens_to_texts,
        "settings_config": {},
        "max_tokens": 128,
        "implemented_providers": {"huggingface", "openai"},
        "allowed_providers": None,
        "allowed_models": None,
        "enforce_policy": True,
        "allow_fallback_with_header": True,
        "settings_fallback_chain": None,
        "settings_fallback_model_map": None,
        "dimension_policy": "reduce",
        "require_model": True,
        "guess_provider": None,
        "backend_identity_resolver": lambda provider, model: f"{provider}:{model}:backend",
        "cache_namespace": None,
        "batch_size": None,
        "execution_path": "legacy",
    }
    kwargs.update(overrides)
    return EmbeddingPreparationPipeline(**kwargs)


def test_pipeline_reports_only_phase_identifiers_in_execution_order(monkeypatch):
    calls = []
    phases = []
    real_resolve = preparation_module.resolve_provider_model
    real_normalize = preparation_module.normalize_embedding_input
    real_policy = preparation_module.enforce_embedding_policy

    def resolve_probe(*args, **kwargs):
        calls.append("resolve_intent")
        return real_resolve(*args, **kwargs)

    def normalize_probe(*args, **kwargs):
        calls.append("normalize")
        return real_normalize(*args, **kwargs)

    def policy_probe(*args, **kwargs):
        calls.append("resolve_policy")
        return real_policy(*args, **kwargs)

    def identity_probe(provider, model):
        calls.append("plan_identity")
        return f"{provider}:{model}:backend"

    monkeypatch.setattr(preparation_module, "resolve_provider_model", resolve_probe)
    monkeypatch.setattr(preparation_module, "normalize_embedding_input", normalize_probe)
    monkeypatch.setattr(preparation_module, "enforce_embedding_policy", policy_probe)
    prepared = _pipeline(backend_identity_resolver=identity_probe).prepare(
        "ordered preparation",
        _context(),
        phase_sink=lambda phase: phases.append(phase),
    )
    assert calls == ["resolve_intent", "normalize", "resolve_policy", "plan_identity"]
    assert phases == ["resolving_intent", "normalizing", "resolving_policy", "planning"]
    assert all(isinstance(phase, str) for phase in phases)
    assert "ordered preparation" not in repr(phases)
    assert prepared.execution_plan.observability_tags not in phases
```

Add a parameterized test that raises one sentinel exception from each preparation boundary, asserts object identity is preserved, and asserts no later boundary or phase is entered.

- [x] **Step 2: Run the pipeline tests and verify RED**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_preparation_pipeline.py -q
```

Expected: import failure because `preparation.py` does not exist.

- [x] **Step 3: Implement `EmbeddingPreparationPipeline`**

Create a class with meaningful injected configuration: token counter/decoder, provider settings and policy allowlists, fallback settings, default dimension policy, provider guesser, backend identity resolver, cache namespace, batch size, and execution path. Its synchronous `prepare` method must:

1. Report `resolving_intent`, then call `resolve_provider_model`.
2. Report `normalizing`, then call `normalize_embedding_input` with the resolved model.
3. Report `resolving_policy`, then call `enforce_embedding_policy`.
4. Report `planning`, compute effective dimension policy, resolve plan identity, build `EmbeddingExecutionPlan`, and return `PreparedEmbeddingRequest`.

Define `PhaseSink = Callable[[EmbeddingWorkflowPhase], None]` and accept `phase_sink: PhaseSink | None = None`; never pass raw input, intent, context, prepared data, or observability tags to it. Export a pure `effective_dimension_policy(encoding_format, dimensions, configured_policy)` helper so base64 plus explicit dimensions always returns `reduce`.

- [x] **Step 4: Delegate orchestrator preparation**

Construct one `EmbeddingPreparationPipeline` in `EmbeddingRequestOrchestrator.__init__` and replace the inline `prepare` algorithm with:

```python
def prepare(self, raw_input: Any, context: EmbeddingRequestContext) -> PreparedEmbeddingRequest:
    return self._preparation_pipeline.prepare(raw_input, context)
```

Keep the compatibility facade call free of a phase sink. Import and re-export `PreparedEmbeddingRequest` from `orchestrator.py` so endpoint and test imports remain valid. Update the existing order probe to patch `preparation` module functions rather than deleted orchestrator globals.

- [x] **Step 5: Run focused preparation and orchestrator tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_preparation_pipeline.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py -q
```

Expected: all tests pass with unchanged preparation outputs and error identity.

- [x] **Step 6: Commit the preparation slice**

```bash
git add tldw_Server_API/app/core/Embeddings/preparation.py tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_preparation_pipeline.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py
git commit -m "refactor(embeddings): extract preparation pipeline"
```

### Task 4: Extract Deterministic Vector Processing

**Files:**
- Create: `tldw_Server_API/app/core/Embeddings/vector_processing.py`
- Modify: `tldw_Server_API/app/core/Embeddings/orchestrator.py`
- Create: `tldw_Server_API/tests/Embeddings_isolated/test_vector_processing.py`
- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`

- [x] **Step 1: Write failing vector processor tests**

Cover count errors, malformed/non-finite values, float canonicalization, dimension reduction/padding, recorder calls, and cached-vector processing:

```python
def test_validate_vector_count_canonicalizes_finite_numeric_vectors():
    processor = EmbeddingVectorProcessor()
    assert processor.validate_vector_count(
        [[1, 2], [3.5, 4]], expected=2, provider="p", model="m"
    ) == [[1.0, 2.0], [3.5, 4.0]]


def test_validate_vector_count_raises_existing_domain_error_for_count_mismatch():
    with pytest.raises(EmbeddingProviderError) as exc_info:
        EmbeddingVectorProcessor().validate_vector_count(
            [[1.0]], expected=2, provider="p", model="m"
        )
    assert exc_info.value.code == "provider_malformed_response"


@pytest.mark.parametrize("vectors", [[[float("nan"), 1.0]], [[float("inf"), 1.0]], [[True, 1.0]]])
def test_validate_vector_count_rejects_malformed_numeric_values(vectors):
    with pytest.raises(EmbeddingProviderError) as exc_info:
        EmbeddingVectorProcessor().validate_vector_count(
            vectors, expected=1, provider="p", model="m"
        )
    assert exc_info.value.code == "provider_malformed_response"


@pytest.mark.parametrize(
    "vector",
    [None, [], ["not-a-number"], [float("nan"), 0.0], [float("inf"), 0.0], [True, 0.0]],
)
def test_validate_cached_vector_returns_none_for_malformed_cache_values(vector):
    assert EmbeddingVectorProcessor().validate_cached_vector(vector) is None


def test_validate_cached_vector_returns_canonical_finite_vector():
    assert EmbeddingVectorProcessor().validate_cached_vector([1, 2.5]) == [1.0, 2.5]


def test_process_vectors_applies_dimensions_and_records_adjustment():
    calls = []
    processor = EmbeddingVectorProcessor(record_dimension_adjustment=lambda *args: calls.append(args))
    assert processor.process_vectors(
        [[1, 2, 3]], provider="p", model="m", dimensions=2, dimension_policy="reduce"
    ) == [[1.0, 2.0]]
    assert calls == [("p", "m", "reduce")]
```

- [x] **Step 2: Run the vector tests and verify RED**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_vector_processing.py -q
```

Expected: import failure because `vector_processing.py` does not exist.

- [x] **Step 3: Implement `EmbeddingVectorProcessor`**

The class owns only the optional dimension-adjustment recorder. Implement:

- `validate_vector_count(vectors, expected, provider, model)` using `validated_embedding_vectors` and the existing exact `provider_malformed_response` messages.
- `validate_cached_vector(vector) -> list[float] | None` using `validated_embedding_vectors([vector], expected=1)` so malformed, non-finite, boolean, empty, and missing cache values remain misses rather than becoming request errors.
- `process_vectors(vectors, provider, model, dimensions, dimension_policy)` using float canonicalization before and after `adjust_dimensions`.
- `process_cached_vector(...)` by applying `process_vectors` only after `validate_cached_vector` has returned a canonical vector.

Keep cache, executor, endpoint response, and workflow collector dependencies out of this module.

- [x] **Step 4: Delegate all orchestrator vector operations**

Construct `EmbeddingVectorProcessor(record_dimension_adjustment=record_dimension_adjustment)` in the orchestrator. Replace provider-response validation/postprocessing/canonicalization call sites with processor methods or the module’s canonicalization function. Replace both primary and fallback cache-read validation with `validate_cached_vector`; a `None` result must continue through the existing miss path. Remove the superseded private methods and direct `adjust_dimensions`/`validated_embedding_vectors` imports.

Add a fallback-path parity test beside the existing primary `test_malformed_cached_vector_becomes_miss_and_is_replaced`: force an eligible primary provider failure, seed a malformed fallback cache value, return a valid fallback provider vector, and assert the malformed value is treated as a miss, replaced under the existing fallback identity, and returned successfully. Do not change fallback identity timing in this stage.

- [x] **Step 5: Run vector, orchestrator, and endpoint parity tests**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_vector_processing.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py -q
```

Expected: all tests pass; error codes/messages, cache write values, headers, and endpoint bodies remain unchanged.

- [x] **Step 6: Commit the vector slice**

```bash
git add tldw_Server_API/app/core/Embeddings/vector_processing.py tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_vector_processing.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py
git commit -m "refactor(embeddings): extract vector processing"
```

### Task 5: Verify Scope, Update Tracking, and Prepare Review

**Files:**
- Modify: `Docs/superpowers/plans/2026-07-26-embeddings-workflow-stage2b-contracts-implementation-plan.md`
- Modify through Backlog.md tooling: `backlog/tasks/task-12973.2 - Extract-Embeddings-preparation-and-result-contracts.md`

- [x] **Step 1: Run the full focused Stage 2B contract suite**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Embeddings_isolated/test_request_types.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py \
  tldw_Server_API/tests/Embeddings_isolated/test_preparation_pipeline.py \
  tldw_Server_API/tests/Embeddings_isolated/test_vector_processing.py \
  tldw_Server_API/tests/Embeddings_isolated/test_result_mapping.py \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py -q
```

- [x] **Step 2: Run the full isolated Embeddings suite**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Embeddings_isolated -q
```

- [x] **Step 3: Run static and security checks**

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m ruff check \
  tldw_Server_API/app/core/Embeddings/request_types.py \
  tldw_Server_API/app/core/Embeddings/workflow_types.py \
  tldw_Server_API/app/core/Embeddings/preparation.py \
  tldw_Server_API/app/core/Embeddings/vector_processing.py \
  tldw_Server_API/app/core/Embeddings/result_mapping.py \
  tldw_Server_API/app/core/Embeddings/orchestrator.py \
  tldw_Server_API/tests/Embeddings_isolated
python -m ruff format --check \
  tldw_Server_API/app/core/Embeddings/request_types.py \
  tldw_Server_API/app/core/Embeddings/workflow_types.py \
  tldw_Server_API/app/core/Embeddings/preparation.py \
  tldw_Server_API/app/core/Embeddings/vector_processing.py \
  tldw_Server_API/app/core/Embeddings/result_mapping.py \
  tldw_Server_API/app/core/Embeddings/orchestrator.py \
  tldw_Server_API/tests/Embeddings_isolated
python -m bandit -r \
  tldw_Server_API/app/core/Embeddings/request_types.py \
  tldw_Server_API/app/core/Embeddings/workflow_types.py \
  tldw_Server_API/app/core/Embeddings/preparation.py \
  tldw_Server_API/app/core/Embeddings/vector_processing.py \
  tldw_Server_API/app/core/Embeddings/result_mapping.py \
  tldw_Server_API/app/core/Embeddings/orchestrator.py \
  -f json -o /tmp/bandit_embeddings_stage2b.json
git diff --check origin/dev
```

- [x] **Step 4: Audit scope and compatibility**

Confirm:

```bash
git diff --name-only origin/dev
git diff origin/dev -- tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
rg -n "Stage 6 compatibility|Stage 2E" tldw_Server_API/app/core/Embeddings/result_mapping.py
```

Expected: no endpoint diff; the mapper deferral and removal stage are explicit; no Stage 2C/2D behavior appears in the diff.

- [x] **Step 5: Request independent code review and address validated findings**

Review the complete `origin/dev...HEAD` diff for behavior drift, DTO mutability, import cycles, phase/error precedence, accidental endpoint wiring, and missing parity tests. Apply only findings reproduced against current code, then rerun Steps 1–4.

- [x] **Step 6: Finalize task evidence through Backlog.md tooling**

Check all satisfied acceptance criteria and definition-of-done items on `TASK-12973.2`. Record exact test counts, Ruff/Bandit results, known warnings/skips, changed files, commits, and the final summary. Keep parent `TASK-12973` in progress because Stage 2C–2E remain.

- [ ] **Step 7: Commit final plan/tracking updates and prepare the PR**

```bash
git add Docs/superpowers/plans/2026-07-26-embeddings-workflow-stage2b-contracts-implementation-plan.md backlog/tasks/task-12973.2\ -\ Extract-Embeddings-preparation-and-result-contracts.md
git commit -m "docs(embeddings): finalize stage 2b evidence"
```

Before creating the PR, rebase on current `origin/dev`, rerun the focused suite and required static/security checks, and require a requester-authored Change summary that explains both what changed and why these boundaries were chosen.
