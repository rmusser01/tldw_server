# Embeddings Workflow Stage 2A Characterization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete `TASK-12973.1` by adding behavior-first tests for the remaining Embeddings orchestration boundaries without changing production execution behavior.

**Architecture:** Keep `EmbeddingRequestOrchestrator`, `EmbeddingInlineWorkflowRunner`, and the endpoint implementation unchanged. Extend their existing focused test modules with source-aware failure probes, exact lifecycle assertions, and resource-accounting checks; reuse the credential-isolation and malformed-cache coverage already present on current `dev`.

**Tech Stack:** Python 3.11, FastAPI, asyncio, pytest, pytest-asyncio, unittest.mock, Ruff, Bandit.

---

## Scope And Existing Coverage

The branch was rebased onto `origin/dev` at `1b9518c68c` before this plan was written. Stage 2A is test-only.

**Files modified during implementation:**

- `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`
  - Preparation order, primary/fallback readiness, source-shaped failures, adapter counters, malformed cache values, and write-before-validation protection.
- `tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py`
  - Exact Stage 1 trace status sequence and cancellation behavior.
- `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`
  - Endpoint cache-adapter exception identity, reservation cleanup, zero-token accounting, active-request accounting, and final credential-touch identity.
- `backlog/tasks/task-12973.1 - Characterize-Embeddings-workflow-execution-contracts.md`
  - Verification notes, acceptance criteria, and final status.

**Production files modified:** none.

Current `dev` already supplies the following required evidence. Do not duplicate these tests:

- Private credentials bypass shared reads/writes:
  - `test_orchestrator_resolved_credentials_bypass_shared_provider_cache`
  - `test_orchestrator_adapter_miss_with_resolved_key_bypasses_shared_cache`
  - `test_resolved_keys_never_share_cached_vectors_sequentially_or_concurrently`
- Credential usage timing:
  - `test_executor_uses_adapter_registry_before_provider_execution`
  - `test_executor_multibatch_marks_first_valid_batch_before_second_error`
  - `test_public_multibatch_marks_first_valid_batch_before_second_cancel`
- Public malformed-cache replacement:
  - `test_public_malformed_cached_vector_is_replaced_from_provider_boundary`
- Primary execution cardinality:
  - `test_fallback_execution_maps_model_and_returns_fallback_headers`
  - `test_endpoint_real_orchestrator_applies_fallback_response_headers`
- Complete primary result validation before writeback:
  - `test_execute_rejects_mixed_width_partial_cache_and_provider_result_without_writeback`

### Task 1: Characterize Preparation And Readiness Order

**Files:**

- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`

- [x] **Step 1: Allow the isolated factory to inject runtime identity and cache-key probes**

Extend `_orchestrator` with narrow optional overrides while preserving every existing default:

```python
def _orchestrator(
    *,
    cache: RecordingCache | None = None,
    executor: RecordingExecutor | None = None,
    settings_fallback_chain: dict[str, object] | None = None,
    settings_fallback_model_map: dict[str, object] | None = None,
    dimension_policy: str = "reduce",
    provider_preflight=None,
    execution_path: str = "legacy",
    cache_key_fn=_cache_key,
    backend_identity_resolver=None,
) -> EmbeddingRequestOrchestrator:
    identity_resolver = backend_identity_resolver or (
        lambda provider, model: f"{provider}:{model}:backend"
    )
    return EmbeddingRequestOrchestrator(
        count_tokens=_count_tokens,
        tokens_to_texts=_tokens_to_texts,
        cache_key_fn=cache_key_fn,
        cache=cache or RecordingCache(),
        executor=executor or RecordingExecutor(),
        settings_config={},
        max_tokens=100,
        implemented_providers={"openai", "cohere", "huggingface", "onnx", "local_api"},
        allowed_providers=None,
        allowed_models=None,
        enforce_policy=True,
        allow_fallback_with_header=True,
        settings_fallback_chain=settings_fallback_chain,
        settings_fallback_model_map=settings_fallback_model_map,
        dimension_policy=dimension_policy,
        backend_identity_resolver=identity_resolver,
        provider_preflight=provider_preflight,
        execution_path=execution_path,  # type: ignore[arg-type]
    )
```

- [x] **Step 2: Add a preparation-order characterization test**

Import the orchestrator module for monkeypatching:

```python
from tldw_Server_API.app.core.Embeddings import orchestrator as orchestrator_module
```

Add a test that wraps the real preparation functions and records only their order:

```python
@pytest.mark.unit
def test_prepare_orders_intent_normalization_policy_and_plan_identity(monkeypatch):
    order: list[str] = []
    original_resolve = orchestrator_module.resolve_provider_model
    original_normalize = orchestrator_module.normalize_embedding_input
    original_policy = orchestrator_module.enforce_embedding_policy

    def resolve_probe(*args, **kwargs):
        order.append("resolve_intent")
        return original_resolve(*args, **kwargs)

    def normalize_probe(*args, **kwargs):
        order.append("normalize")
        return original_normalize(*args, **kwargs)

    def policy_probe(*args, **kwargs):
        order.append("resolve_policy")
        return original_policy(*args, **kwargs)

    def identity_probe(provider: str, model: str) -> str:
        order.append("plan_identity")
        return f"{provider}:{model}:backend"

    monkeypatch.setattr(orchestrator_module, "resolve_provider_model", resolve_probe)
    monkeypatch.setattr(orchestrator_module, "normalize_embedding_input", normalize_probe)
    monkeypatch.setattr(orchestrator_module, "enforce_embedding_policy", policy_probe)
    orchestrator = _orchestrator(backend_identity_resolver=identity_probe)

    orchestrator.prepare("ordered preparation", _context())

    assert order == [
        "resolve_intent",
        "normalize",
        "resolve_policy",
        "plan_identity",
    ]
```

- [x] **Step 3: Add a primary-preflight failure test**

The test must prove readiness runs before cache access and that even a retryable primary readiness failure does not enter fallback:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_primary_preflight_failure_propagates_without_cache_or_fallback():
    original = EmbeddingExecutionError(
        "provider_unavailable",
        "primary preflight failed",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    cache = RecordingCache()
    executor = RecordingExecutor(
        provider_vectors={"huggingface": [[0.5, 0.25]]}
    )

    async def provider_preflight(provider: str, model: str) -> None:
        assert (provider, model) == ("openai", "text-embedding-3-small")
        raise original

    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        provider_preflight=provider_preflight,
        settings_fallback_chain={"openai": ["huggingface"]},
    )
    prepared = orchestrator.prepare(
        "preflight failure",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    with pytest.raises(EmbeddingExecutionError) as exc_info:
        await orchestrator.execute(prepared)

    assert exc_info.value is original
    assert cache.get_keys == []
    assert cache.set_calls == []
    assert executor.calls == []
```

- [x] **Step 4: Add retryable fallback-preflight traversal coverage**

Keep the existing missing-credential skip test. Add a separate retryable readiness case that proves the failed candidate is skipped and the next candidate is attempted:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_retryable_fallback_preflight_failure_continues_to_next_candidate():
    primary_error = EmbeddingProviderError(
        "provider_unavailable",
        "primary unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    fallback_preflight_error = EmbeddingExecutionError(
        "circuit_breaker_open",
        "cohere circuit open",
        provider="cohere",
        model="embed-english-v3.0",
        retryable=True,
    )
    executor = RecordingExecutor(
        failures={"openai": primary_error},
        provider_vectors={"huggingface": [[0.5, 0.25]]},
    )
    preflight_calls: list[tuple[str, str]] = []

    async def provider_preflight(provider: str, model: str) -> None:
        preflight_calls.append((provider, model))
        if provider == "cohere":
            raise fallback_preflight_error

    orchestrator = _orchestrator(
        executor=executor,
        provider_preflight=provider_preflight,
        settings_fallback_chain={"openai": ["cohere", "huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "cohere": "embed-english-v3.0",
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            }
        },
    )
    prepared = orchestrator.prepare(
        "retry fallback readiness",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.provider == "huggingface"
    assert preflight_calls == [
        ("openai", "text-embedding-3-small"),
        ("cohere", "embed-english-v3.0"),
        ("huggingface", "sentence-transformers/all-MiniLM-L6-v2"),
    ]
    assert [call["provider"] for call in executor.calls] == [
        "openai",
        "huggingface",
    ]
```

- [x] **Step 5: Run the new preparation/readiness tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py::test_prepare_orders_intent_normalization_policy_and_plan_identity \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py::test_primary_preflight_failure_propagates_without_cache_or_fallback \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py::test_retryable_fallback_preflight_failure_continues_to_next_candidate \
  -q
```

Expected: 3 passed. These are characterization tests; a failure means the test assumption must be reconciled with current behavior, not that production code should be changed.

- [x] **Step 6: Commit the preparation/readiness tests**

```bash
git add tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py
git commit -m "test(embeddings): characterize preparation and readiness order"
```

### Task 2: Pin Infrastructure Failure Sources

**Files:**

- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`
- Modify: `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`

- [x] **Step 1: Add a primary infrastructure-failure parameterization**

Add a small raising cache inside the test and cover read-time identity, key derivation, and cache access. Each error is a generic infrastructure exception and must propagate unchanged without provider fallback:

```python
@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    ["backend_identity", "cache_key", "cache_get"],
)
async def test_primary_infrastructure_failure_propagates_without_fallback(boundary):
    original = RuntimeError(f"{boundary} failed")
    identity_calls = 0

    def identity_resolver(provider: str, model: str) -> str:
        nonlocal identity_calls
        identity_calls += 1
        if boundary == "backend_identity" and identity_calls > 1:
            raise original
        return f"{provider}:{model}:backend"

    def cache_key_fn(text, provider, model, dimensions, backend_identity):
        if boundary == "cache_key":
            raise original
        return _cache_key(text, provider, model, dimensions, backend_identity)

    class FailingGetCache(RecordingCache):
        async def get(self, key: str) -> list[float] | None:
            if boundary == "cache_get":
                raise original
            return await super().get(key)

    cache = FailingGetCache()
    executor = RecordingExecutor(
        provider_vectors={"huggingface": [[0.5, 0.25]]}
    )
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        cache_key_fn=cache_key_fn,
        backend_identity_resolver=identity_resolver,
        settings_fallback_chain={"openai": ["huggingface"]},
    )
    prepared = orchestrator.prepare(
        "primary infrastructure failure",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await orchestrator.execute(prepared)

    assert exc_info.value is original
    assert executor.calls == []
    assert cache.set_calls == []
```

- [x] **Step 2: Characterize the current broad fallback-domain catch**

This test intentionally records behavior that Stage 2D will correct. A retryable domain-shaped error from fallback identity, key, cache read, or cache write currently advances to the next provider:

```python
@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    ["backend_identity", "cache_key", "cache_get", "cache_set"],
)
async def test_fallback_domain_shaped_infrastructure_failure_currently_advances_candidate(
    boundary,
):
    original = EmbeddingExecutionError(
        "internal_execution_failure",
        f"{boundary} failed",
        provider="cohere",
        model="embed-english-v3.0",
        retryable=True,
    )
    primary_error = EmbeddingProviderError(
        "provider_unavailable",
        "primary unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )

    def identity_resolver(provider: str, model: str) -> str:
        if boundary == "backend_identity" and provider == "cohere":
            raise original
        return f"{provider}:{model}:backend"

    def cache_key_fn(text, provider, model, dimensions, backend_identity):
        if boundary == "cache_key" and provider == "cohere":
            raise original
        return _cache_key(text, provider, model, dimensions, backend_identity)

    class BoundaryCache(RecordingCache):
        async def get(self, key: str) -> list[float] | None:
            self.get_keys.append(key)
            if boundary == "cache_get" and "|cohere|" in key:
                raise original
            return self.values.get(key)

        async def set(self, key: str, value: list[float]) -> object:
            if boundary == "cache_set" and "|cohere|" in key:
                raise original
            return await super().set(key, value)

    executor = RecordingExecutor(
        failures={"openai": primary_error},
        provider_vectors={
            "cohere": [[0.4, 0.6]],
            "huggingface": [[0.5, 0.25]],
        },
    )
    orchestrator = _orchestrator(
        cache=BoundaryCache(),
        executor=executor,
        cache_key_fn=cache_key_fn,
        backend_identity_resolver=identity_resolver,
        settings_fallback_chain={"openai": ["cohere", "huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "cohere": "embed-english-v3.0",
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            }
        },
    )
    prepared = orchestrator.prepare(
        "fallback infrastructure failure",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.provider == "huggingface"
    providers = [call["provider"] for call in executor.calls]
    assert providers[0] == "openai"
    assert providers[-1] == "huggingface"
    assert providers.count("openai") == 1
```

- [x] **Step 3: Characterize endpoint cache-adapter exception identity**

In `test_embeddings_orchestrator_endpoint_parity.py`, add:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["get", "set"])
@pytest.mark.parametrize(
    "error_factory",
    [
        pytest.param(
            lambda: RuntimeError("cache dependency failed"),
            id="unexpected",
        ),
        pytest.param(
            lambda: EmbeddingExecutionError(
                "internal_execution_failure",
                "domain-shaped cache failure",
                retryable=True,
            ),
            id="domain",
        ),
    ],
)
async def test_endpoint_cache_adapter_propagates_dependency_error_unchanged(
    monkeypatch,
    method_name,
    error_factory,
):
    from tldw_Server_API.app.api.v1.endpoints import (
        embeddings_v5_production_enhanced as mod,
    )

    original = error_factory()
    dependency = AsyncMock(side_effect=original)
    monkeypatch.setattr(mod.embedding_cache, method_name, dependency)
    cache = mod._EndpointEmbeddingCache()

    with pytest.raises(type(original)) as exc_info:
        if method_name == "get":
            await cache.get("cache-key")
        else:
            await cache.set("cache-key", [0.1, 0.2])

    assert exc_info.value is original
    dependency.assert_awaited_once()
```

- [x] **Step 4: Run the infrastructure-source tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py \
  -k "infrastructure_failure or endpoint_cache_adapter_propagates" \
  -q
```

Expected: 11 passed: 3 primary cases, 4 current fallback cases, and 4 endpoint adapter cases.

- [x] **Step 5: Commit the failure-source tests**

```bash
git add \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
git commit -m "test(embeddings): pin orchestration failure sources"
```

### Task 3: Characterize Adapter And Cache-Validation Contracts

**Files:**

- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`

- [x] **Step 1: Add successful adapter accounting and bypass coverage**

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_successful_adapter_reports_compatibility_cache_counts_and_bypasses_provider():
    cache = RecordingCache()
    preflight_calls: list[tuple[str, str]] = []

    async def provider_preflight(provider: str, model: str) -> None:
        preflight_calls.append((provider, model))

    executor = AdapterAwareExecutor(
        vectors=[[9.0, 9.0]],
        adapter_output=EmbeddingExecutorOutput(
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            embeddings_from_adapter=True,
        ),
    )
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        execution_path="adapter",
        provider_preflight=provider_preflight,
    )
    prepared = orchestrator.prepare(
        ["one", "two"],
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert result.embeddings_from_adapter is True
    assert result.cache_hits == 0
    assert result.cache_misses == 2
    assert preflight_calls == []
    assert cache.get_keys == []
    assert cache.set_calls == []
    assert executor.calls == []
```

- [x] **Step 2: Add isolated malformed/non-finite cache replacement cases**

```python
@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cached",
    [
        pytest.param([], id="empty"),
        pytest.param(["not-a-number"], id="nonnumeric"),
        pytest.param([float("nan"), 0.0], id="nan"),
        pytest.param([float("inf"), 0.0], id="infinite"),
    ],
)
async def test_malformed_cached_vector_becomes_miss_and_is_replaced(cached):
    cache_key = (
        "replace|huggingface|sentence-transformers/all-MiniLM-L6-v2|"
        "huggingface:sentence-transformers/all-MiniLM-L6-v2:backend"
    )
    cache = RecordingCache({cache_key: cached})  # type: ignore[dict-item]
    executor = RecordingExecutor(vectors=[[0.25, 0.75]])
    orchestrator = _orchestrator(cache=cache, executor=executor)
    prepared = orchestrator.prepare("replace", _context())

    result = await orchestrator.execute(prepared)

    assert result.vectors == [[0.25, 0.75]]
    assert result.cache_hits == 0
    assert result.cache_misses == 1
    assert len(executor.calls) == 1
    assert cache.set_calls == [(cache_key, [0.25, 0.75])]
```

- [x] **Step 3: Add fallback complete-result validation before writeback**

Keep the existing primary mixed-width test unchanged. Add the equivalent full fallback path:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_complete_result_is_validated_before_first_cache_write():
    primary_error = EmbeddingProviderError(
        "provider_unavailable",
        "primary unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    cache = RecordingCache()
    executor = RecordingExecutor(
        failures={"openai": primary_error},
        provider_vectors={
            "huggingface": [[0.1, 0.2], [0.3, 0.4, 0.5]],
        },
    )
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        settings_fallback_chain={"openai": ["huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            }
        },
    )
    prepared = orchestrator.prepare(
        ["one", "two"],
        _context(model="text-embedding-3-small", provider="openai"),
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await orchestrator.execute(prepared)

    assert exc_info.value.code == "provider_malformed_response"
    assert cache.set_calls == []
```

- [x] **Step 4: Reassert primary execution cardinality**

In `test_fallback_execution_maps_model_and_returns_fallback_headers`, add:

```python
    assert prepared.execution_plan.fallback_chain == ["openai", "huggingface"]
    assert [call["provider"] for call in executor.calls].count("openai") == 1
```

- [x] **Step 5: Run adapter/cache characterization**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  -k "successful_adapter or malformed_cached_vector or validated_before_first_cache_write or fallback_execution_maps_model" \
  -q
```

Expected: 7 passed.

- [x] **Step 6: Commit adapter/cache characterization**

```bash
git add tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py
git commit -m "test(embeddings): characterize adapter and cache validation"
```

### Task 4: Pin Stage 1 Trace And Cancellation Behavior

**Files:**

- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py`

- [x] **Step 1: Assert the exact current success statuses**

Extend `test_runner_returns_orchestrator_result_and_records_safe_success_events`:

```python
    assert [event.status for event in collector.events] == [
        "running",
        "running",
        "running",
        "running",
        "running",
        "running",
        "completed",
    ]
```

This intentionally pins the Stage 1 sequence. Do not add Stage 2 phases in Stage 2A.

- [x] **Step 2: Add cancellation propagation and trace truncation**

Import `asyncio`, then add:

```python
@pytest.mark.asyncio
async def test_execute_cancellation_propagates_without_terminal_trace_event():
    cancellation = asyncio.CancelledError()
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    runner = EmbeddingInlineWorkflowRunner(
        FakeOrchestrator(execute_error=cancellation),
        trace_collector=collector,
    )

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await runner.run(["one"], _request_context())

    assert exc_info.value is cancellation
    assert [event.event_type for event in collector.events] == [
        "workflow_started",
        "phase_changed",
        "phase_changed",
        "prepare_completed",
        "phase_changed",
    ]
    assert collector.events[-1].phase == "executing"
    assert all(
        event.event_type not in {"workflow_failed", "workflow_completed"}
        for event in collector.events
    )
```

- [x] **Step 3: Run the runner characterization tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py \
  -q
```

Expected: all runner tests pass, including the new cancellation case.

- [x] **Step 4: Commit runner characterization**

```bash
git add tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py
git commit -m "test(embeddings): pin inline runner cancellation"
```

### Task 5: Characterize Endpoint Lifecycle And Accounting

**Files:**

- Modify: `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`

- [x] **Step 1: Add a recording active-request gauge**

Place this beside the existing metric fakes:

```python
class _RecordingGauge(_NoopMetric):
    def __init__(self) -> None:
        self.inc_count = 0
        self.dec_count = 0

    def inc(self, *_args, **_kwargs):
        self.inc_count += 1

    def dec(self, *_args, **_kwargs):
        self.dec_count += 1
```

- [x] **Step 2: Make the endpoint fake support zero-token preparation**

Change only the test fake:

```python
class FakeOrchestrator:
    def __init__(
        self,
        *,
        result=None,
        prepare_error=None,
        execute_error=None,
        prepared_total_tokens: int = 3,
    ) -> None:
        self.result = result
        self.prepare_error = prepare_error
        self.execute_error = execute_error
        self.prepared_total_tokens = prepared_total_tokens
        self.prepare_calls = []
        self.execute_calls = []

    def prepare(self, raw_input, context):
        self.prepare_calls.append((raw_input, context))
        if self.prepare_error is not None:
            raise self.prepare_error
        return FakePrepared(total_tokens=self.prepared_total_tokens)
```

- [x] **Step 3: Strengthen existing success and post-reservation failure tests**

In both `test_orchestrator_path_uses_inline_workflow_runner_and_preserves_rg_reservation` and `test_orchestrator_path_commits_reserved_units_after_execute_failure`, install a fresh `_RecordingGauge`:

```python
    active_requests = _RecordingGauge()
    monkeypatch.setattr(mod, "active_embedding_requests", active_requests)
```

At the end of each test, assert:

```python
    assert active_requests.inc_count == 1
    assert active_requests.dec_count == 1
```

In the success test, replace `_EndpointEmbeddingExecutor` with a test double and assert the endpoint's final touch uses the actual outcome identity:

```python
    endpoint_executor = SimpleNamespace(touch_resolved_credentials=AsyncMock())
    monkeypatch.setattr(
        mod,
        "_EndpointEmbeddingExecutor",
        lambda **_kwargs: endpoint_executor,
    )
```

After the response:

```python
    endpoint_executor.touch_resolved_credentials.assert_awaited_once_with(
        "huggingface",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
```

- [x] **Step 4: Add successful zero-token reserved-unit fallback**

```python
def test_orchestrator_zero_token_success_commits_reserved_unit(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import (
        embeddings_v5_production_enhanced as mod,
    )

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        prepared_total_tokens=0,
        result=EmbeddingExecutionResult(
            vectors=[[0.25, 0.75]],
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            prompt_tokens=0,
            total_tokens=0,
            cache_hits=0,
            cache_misses=1,
        ),
    )
    governor = SimpleNamespace(commit=AsyncMock())
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
    )

    async def reserve(*, request, current_user, token_total):
        del request, current_user
        assert token_total == 1
        return governor, "zero-handle", "zero-op", 1

    monkeypatch.setattr(mod, "_reserve_embedding_rg_tokens", reserve)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "input": "zero accounting",
        },
    )

    assert response.status_code == status.HTTP_200_OK
    governor.commit.assert_awaited_once_with(
        "zero-handle",
        actuals={"tokens": 1},
        op_id="zero-op",
    )
```

- [x] **Step 5: Add direct endpoint cancellation cleanup**

Call the endpoint coroutine directly so `CancelledError` is observable rather than translated by `TestClient`:

```python
@pytest.mark.asyncio
async def test_orchestrator_cancellation_commits_reserved_units_and_decrements_active(
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints import (
        embeddings_v5_production_enhanced as mod,
    )

    cancellation = asyncio.CancelledError()
    active_requests = _RecordingGauge()
    governor = SimpleNamespace(commit=AsyncMock())
    fake_orchestrator = FakeOrchestrator(prepared_total_tokens=3)

    async def no_backpressure(*_args, **_kwargs):
        return None

    async def reserve(*, request, current_user, token_total):
        del request, current_user
        return governor, "cancel-handle", "cancel-op", token_total

    class CancellingRunner:
        def __init__(self, orchestrator, *, trace_collector=None, pre_execute=None):
            del trace_collector
            self.orchestrator = orchestrator
            self.pre_execute = pre_execute

        async def run(self, raw_input, context):
            prepared = self.orchestrator.prepare(raw_input, context)
            assert self.pre_execute is not None
            await self.pre_execute(prepared)
            raise cancellation

    request = SimpleNamespace(
        state=SimpleNamespace(),
        headers={},
        method="POST",
        url=SimpleNamespace(path="/api/v1/embeddings"),
    )
    monkeypatch.setattr(mod, "EMBEDDINGS_AVAILABLE", True)
    monkeypatch.setattr(mod, "active_embedding_requests", active_requests)
    monkeypatch.setattr(mod, "_check_backpressure_and_quotas", no_backpressure)
    monkeypatch.setattr(mod, "_reserve_embedding_rg_tokens", reserve)
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
    )
    monkeypatch.setattr(mod, "EmbeddingInlineWorkflowRunner", CancellingRunner)

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await mod._create_embedding_with_orchestrator(
            request=request,
            embedding_request=mod.CreateEmbeddingRequest(
                input="cancel endpoint",
                model="sentence-transformers/all-MiniLM-L6-v2",
            ),
            current_user=_user(),
            background_tasks=SimpleNamespace(),
            x_provider="huggingface",
            response=SimpleNamespace(headers={}),
        )

    assert exc_info.value is cancellation
    governor.commit.assert_awaited_once_with(
        "cancel-handle",
        actuals={"tokens": 3},
        op_id="cancel-op",
    )
    assert active_requests.inc_count == 1
    assert active_requests.dec_count == 1
```

- [x] **Step 6: Run endpoint lifecycle characterization**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py \
  -k "preserves_rg_reservation or commits_reserved_units_after_execute_failure or zero_token_success or cancellation_commits_reserved_units" \
  -q
```

Expected: 4 passed.

- [x] **Step 7: Commit endpoint lifecycle characterization**

```bash
git add tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
git commit -m "test(embeddings): characterize endpoint workflow accounting"
```

### Task 6: Verify Existing Credential Contracts And Close Stage 2A

**Files:**

- Modify: `backlog/tasks/task-12973.1 - Characterize-Embeddings-workflow-execution-contracts.md`

- [x] **Step 1: Run all touched test modules**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py \
  -q
```

Expected: all tests pass with no unexpected skips or warnings introduced by Stage 2A.

- [x] **Step 2: Run the broader isolated Embeddings suite**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Embeddings_isolated -q
```

Expected: all isolated Embeddings tests pass.

- [x] **Step 3: Explicitly rerun the latest-dev credential/cache evidence**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py \
  -k "resolved_credentials_bypass_shared_provider_cache or adapter_miss_with_resolved_key_bypasses_shared_cache or resolved_keys_never_share_cached_vectors or executor_uses_adapter_registry_before_provider_execution or executor_multibatch_marks_first_valid_batch_before_second_error or public_multibatch_marks_first_valid_batch_before_second_cancel or public_malformed_cached_vector_is_replaced_from_provider_boundary or endpoint_real_orchestrator_applies_fallback_response_headers" \
  -q
```

Expected: all selected credential, cache-isolation, malformed-cache, and primary-cardinality cases pass.

- [x] **Step 4: Run Ruff on touched Python files**

```bash
source .venv/bin/activate
python -m ruff check \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
```

Expected: no lint findings.

- [x] **Step 5: Run Bandit on the touched test scope**

Assertions are the purpose of these test files, so exclude Bandit's test-only `B101` rule:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py \
  -s B101 \
  -f json \
  -o /tmp/bandit_embeddings_stage2a.json
```

Expected: zero new security findings.

- [x] **Step 6: Verify the diff is test/tracking-only**

```bash
git diff --check
git diff --name-only origin/dev...HEAD
```

Expected implementation paths are the three test files plus the approved design, plan, and Backlog task files. No path under `tldw_Server_API/app/` may be changed by Stage 2A.

- [x] **Step 7: Finalize `TASK-12973.1` through Backlog**

Using the Backlog MCP or CLI:

1. Check all acceptance criteria and definition-of-done items.
2. Record exact test counts, Ruff output, Bandit output path, and any skips.
3. Add the final summary stating that Stage 2A changed tests/tracking only.
4. Mark `TASK-12973.1` Done.

- [x] **Step 8: Commit Stage 2A closeout metadata**

```bash
git add \
  "backlog/tasks/task-12973.1 - Characterize-Embeddings-workflow-execution-contracts.md"
git commit -m "chore(backlog): close embeddings workflow stage 2a"
```
