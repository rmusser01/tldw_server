# Embeddings Workflow Stage 2C Provider Attempts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract provider readiness and single-provider execution attempts into workflow-compatible components while preserving production routing, except for the approved fallback writeback backend-identity correction.

**Architecture:** Add a side-by-side `EmbeddingProviderReadinessCheck` and `EmbeddingProviderAttempt` that depend on narrow cache, executor, cache-key, identity, and vector-processing contracts. The new attempt returns either a complete ordered provider result or an exact provider-call failure DTO, and it catches domain errors only around the executor call. Existing `EmbeddingRequestOrchestrator.execute` remains the production coordinator until Stage 2D; Stage 2C only patches `_execute_coherent_fallback` to re-resolve backend identity before fallback cache writeback.

**Tech Stack:** Python 3.14, frozen dataclasses, typing protocols, pytest, pytest-asyncio, focused isolated tests, Bandit.

**Tracking:** `TASK-12973.3`

**Approved design:** `Docs/superpowers/specs/2026-07-18-embeddings-workflow-stage2-concrete-api-steps-design.md`

**Base:** stacked on `origin/codex/embeddings-workflow-stage2b-contracts`; do not recreate or edit the Stage 2B PR branch.

---

## Scope Guardrails

- Do not wire the new provider-attempt component into the inline runner or production orchestrator execution path until Stage 2D.
- Do not fix the Stage 2D source-aware fallback catch in this task. The existing `test_current_fallback_wide_domain_catch_advances_after_infrastructure_failure` must keep pinning the current legacy behavior.
- Do not change endpoint response formatting, resource-governor accounting, metrics, credential touch policy, cache-key inputs, fallback candidate ordering, or the workflow feature flag.
- Do not make cache writes transactional. Validate the complete ordered response before writeback, but preserve existing partial write behavior if a later cache write fails.
- Keep cache eligibility delegated to the injected cache. The attempt should call `get` and `set`; request-private cache bypass remains an adapter/cache concern.
- Use read-time backend identity only for cache reads and write-time backend identity only for cache writes. Never use `EmbeddingExecutionPlan.backend_identity` as authoritative runtime identity.
- The only production behavior correction in Stage 2C is fallback writeback identity re-resolution in `_execute_coherent_fallback`.

## File Map

- Create `tldw_Server_API/app/core/Embeddings/provider_attempt.py`: readiness wrapper, provider-attempt result DTOs, protocols, executor-output coercion, ordered cache/miss execution, and writeback.
- Modify `tldw_Server_API/app/core/Embeddings/request_types.py`: move `EmbeddingExecutorOutput` here if needed so both the orchestrator and provider-attempt component can share the executor-output contract without a future circular import.
- Modify `tldw_Server_API/app/core/Embeddings/orchestrator.py`: re-export `EmbeddingExecutorOutput` if moved, and apply only the fallback writeback identity correction.
- Create `tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py`: direct tests for readiness and single-provider attempts.
- Modify `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`: add the fallback write-identity correction regression test.
- Modify `backlog/tasks/task-12973.3 - Extract-single-provider-Embeddings-execution-attempts.md`: record plan path, progress, verification, touched files, and final summary through Backlog.md tooling.

### Task 1: Shared Executor Output Contract and Readiness Wrapper

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/request_types.py`
- Modify: `tldw_Server_API/app/core/Embeddings/orchestrator.py`
- Create: `tldw_Server_API/app/core/Embeddings/provider_attempt.py`
- Create: `tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py`
- Test: `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`

- [x] **Step 1: Write the failing readiness and executor-output import tests**

Create `tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py` with the initial helper fixtures and tests:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Embeddings.provider_attempt import (
    EmbeddingProviderReadinessCheck,
)
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionError,
    EmbeddingExecutorOutput,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_readiness_check_delegates_to_preflight():
    calls: list[tuple[str, str]] = []

    async def preflight(provider: str, model: str) -> None:
        calls.append((provider, model))

    readiness = EmbeddingProviderReadinessCheck(preflight)

    await readiness.check("openai", "text-embedding-3-small")

    assert calls == [("openai", "text-embedding-3-small")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_readiness_check_propagates_exact_error():
    error = EmbeddingExecutionError(
        "circuit_breaker_open",
        "provider unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )

    async def preflight(provider: str, model: str) -> None:
        del provider, model
        raise error

    readiness = EmbeddingProviderReadinessCheck(preflight)

    with pytest.raises(EmbeddingExecutionError) as exc_info:
        await readiness.check("openai", "text-embedding-3-small")

    assert exc_info.value is error


@pytest.mark.unit
def test_executor_output_contract_is_shared_from_request_types():
    output = EmbeddingExecutorOutput(
        vectors=[[0.1, 0.2]],
        embeddings_from_adapter=True,
    )

    assert output.vectors == [[0.1, 0.2]]
    assert output.embeddings_from_adapter is True
```

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py::test_successful_adapter_reports_compatibility_cache_counts_and_bypasses_provider -q
```

Expected: import failures because `provider_attempt.py` does not exist and `EmbeddingExecutorOutput` is not exported from `request_types.py`.

- [x] **Step 3: Implement the minimal shared contract and readiness wrapper**

Move the existing frozen `EmbeddingExecutorOutput` dataclass from `orchestrator.py` to `request_types.py`:

```python
@dataclass(frozen=True, slots=True)
class EmbeddingExecutorOutput:
    vectors: list[list[float]]
    embeddings_from_adapter: bool = False
```

Add `"EmbeddingExecutorOutput"` to `request_types.__all__`.

In `orchestrator.py`, import `EmbeddingExecutorOutput` from `request_types.py` and leave it in `orchestrator.__all__` so existing callers continue to import it from the old location.

Create `provider_attempt.py` with:

```python
"""Provider readiness and single-provider execution attempts for embeddings."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutorOutput,
)


ProviderPreflight = Callable[[str, str], Awaitable[None]]


async def _no_provider_preflight(provider: str, model: str) -> None:
    del provider, model


class EmbeddingProviderReadinessCheck:
    """Run provider readiness checks without cache or execution side effects."""

    def __init__(self, provider_preflight: ProviderPreflight | None = None) -> None:
        self._provider_preflight = provider_preflight or _no_provider_preflight

    async def check(self, provider: str, model: str) -> None:
        await self._provider_preflight(provider, model)


__all__ = [
    "EmbeddingProviderReadinessCheck",
    "EmbeddingExecutorOutput",
]
```

- [x] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [x] **Step 5: Commit the shared contract slice**

```bash
git add tldw_Server_API/app/core/Embeddings/request_types.py tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/app/core/Embeddings/provider_attempt.py tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py
git commit -m "refactor(embeddings): add provider readiness boundary"
```

### Task 2: Ordered Single-Provider Cache and Miss Execution

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/provider_attempt.py`
- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py`

- [x] **Step 1: Write failing ordered cache and miss tests**

Extend `test_provider_attempt.py` with the helper fakes and tests:

```python
from tldw_Server_API.app.core.Embeddings.provider_attempt import (
    EmbeddingProviderAttempt,
    ProviderAttemptSuccess,
)
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionPlan,
    EmbeddingPolicyDecision,
    NormalizedEmbeddingInput,
    PreparedEmbeddingRequest,
    ProviderModelIntent,
)
from tldw_Server_API.app.core.Embeddings.vector_processing import (
    EmbeddingVectorProcessor,
)


class RecordingCache:
    def __init__(self, values: dict[str, list[float]] | None = None) -> None:
        self.values = values or {}
        self.get_keys: list[str] = []
        self.set_calls: list[tuple[str, list[float]]] = []

    async def get(self, key: str) -> list[float] | None:
        self.get_keys.append(key)
        return self.values.get(key)

    async def set(self, key: str, value: list[float]) -> object:
        self.set_calls.append((key, value))
        self.values[key] = value
        return None


class RecordingExecutor:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors or []
        self.calls: list[dict[str, object]] = []

    async def create(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "texts": texts,
                "provider": provider,
                "model": model,
                "dimensions": dimensions,
            }
        )
        return self.vectors


def _cache_key(
    text: str,
    provider: str,
    model: str,
    dimensions: int | None,
    backend_identity: str | None,
) -> str:
    parts = [text, provider, model]
    if dimensions is not None:
        parts.append(str(dimensions))
    if backend_identity is not None:
        parts.append(backend_identity)
    return "|".join(parts)


def _prepared(
    texts: list[str],
    *,
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    dimensions: int | None = None,
    backend_identity: str | None = "stale-plan-identity",
    cache_namespace: str | None = "ignored-namespace",
) -> PreparedEmbeddingRequest:
    return PreparedEmbeddingRequest(
        normalized_input=NormalizedEmbeddingInput(
            texts=texts,
            token_counts=[1 for _ in texts],
            total_tokens=len(texts),
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
            dimensions=dimensions,
            fallback_chain=[provider],
            fallback_allowed=True,
            enforce_policy=True,
        ),
        execution_plan=EmbeddingExecutionPlan(
            provider=provider,
            model=model,
            dimensions=dimensions,
            backend_identity=backend_identity,
            fallback_chain=[provider],
            cache_namespace=cache_namespace,
        ),
        effective_dimension_policy="reduce",
        prompt_tokens=len(texts),
        total_tokens=len(texts),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_preserves_order_and_executes_only_cache_misses():
    cache = RecordingCache(
        {
            "hit|openai|text-embedding-3-small|read-openai": [1.0, 0.0],
            "hit2|openai|text-embedding-3-small|read-openai": [0.0, 1.0],
        }
    )
    executor = RecordingExecutor(vectors=[[0.25, 0.75]])
    identity_calls: list[tuple[str, str]] = []

    def backend_identity(provider: str, model: str) -> str:
        identity_calls.append((provider, model))
        return "read-openai" if len(identity_calls) == 1 else "write-openai"

    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=cache,
        executor=executor,
        backend_identity_resolver=backend_identity,
        vector_processor=EmbeddingVectorProcessor(),
    )

    result = await attempt.execute(
        _prepared(["hit", "miss", "hit2"]),
        provider="openai",
        model="text-embedding-3-small",
    )

    assert isinstance(result, ProviderAttemptSuccess)
    assert result.vectors == [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]]
    assert result.cache_hits == 2
    assert result.cache_misses == 1
    assert executor.calls == [
        {
            "texts": ["miss"],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": None,
        }
    ]
    assert cache.get_keys == [
        "hit|openai|text-embedding-3-small|read-openai",
        "miss|openai|text-embedding-3-small|read-openai",
        "hit2|openai|text-embedding-3-small|read-openai",
    ]
    assert cache.set_calls == [
        ("miss|openai|text-embedding-3-small|write-openai", [0.25, 0.75])
    ]
    assert identity_calls == [
        ("openai", "text-embedding-3-small"),
        ("openai", "text-embedding-3-small"),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_cache_keys_exclude_plan_identity_and_namespace():
    cache = RecordingCache()
    executor = RecordingExecutor(vectors=[[0.1, 0.2]])
    cache_key_calls: list[tuple[str, str, str, int | None, str | None]] = []

    def cache_key_probe(
        text: str,
        provider: str,
        model: str,
        dimensions: int | None,
        backend_identity: str | None,
    ) -> str:
        cache_key_calls.append((text, provider, model, dimensions, backend_identity))
        return _cache_key(text, provider, model, dimensions, backend_identity)

    attempt = EmbeddingProviderAttempt(
        cache_key_fn=cache_key_probe,
        cache=cache,
        executor=executor,
        backend_identity_resolver=lambda provider, model: f"runtime:{provider}:{model}",
        vector_processor=EmbeddingVectorProcessor(),
    )

    await attempt.execute(
        _prepared(["one"], dimensions=2, backend_identity="stale-plan", cache_namespace="ns"),
        provider="openai",
        model="text-embedding-3-small",
    )

    assert cache_key_calls == [
        ("one", "openai", "text-embedding-3-small", 2, "runtime:openai:text-embedding-3-small"),
        ("one", "openai", "text-embedding-3-small", 2, "runtime:openai:text-embedding-3-small"),
    ]
    assert all("stale-plan" not in key for key in cache.get_keys)
    assert all("ns" not in key for key in cache.get_keys)
```

- [x] **Step 2: Run the new tests and verify RED**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py -q
```

Expected: failures because `EmbeddingProviderAttempt` and `ProviderAttemptSuccess` do not exist.

- [x] **Step 3: Implement provider-attempt DTOs and ordered cache/miss execution**

In `provider_attempt.py`, add:

```python
from dataclasses import dataclass

from tldw_Server_API.app.core.Embeddings.preparation import BackendIdentityResolver
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    PreparedEmbeddingRequest,
)
from tldw_Server_API.app.core.Embeddings.vector_processing import (
    EmbeddingVectorProcessor,
)


CacheKeyFn = Callable[[str, str, str, int | None, str | None], str]


class ProviderAttemptCache(Protocol):
    async def get(self, key: str) -> list[float] | None:
        raise NotImplementedError

    async def set(self, key: str, value: list[float]) -> object:
        raise NotImplementedError


class ProviderAttemptExecutor(Protocol):
    async def create(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]] | EmbeddingExecutorOutput:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ProviderAttemptSuccess:
    vectors: list[list[float]]
    provider: str
    model: str
    cache_hits: int
    cache_misses: int
    embeddings_from_adapter: bool = False


@dataclass(frozen=True, slots=True)
class ProviderCallFailure:
    error: EmbeddingDomainError


class EmbeddingProviderAttempt:
    def __init__(
        self,
        *,
        cache_key_fn: CacheKeyFn,
        cache: ProviderAttemptCache,
        executor: ProviderAttemptExecutor,
        backend_identity_resolver: BackendIdentityResolver,
        vector_processor: EmbeddingVectorProcessor | None = None,
    ) -> None:
        self._cache_key_fn = cache_key_fn
        self._cache = cache
        self._executor = executor
        self._backend_identity_resolver = backend_identity_resolver
        self._vector_processor = vector_processor or EmbeddingVectorProcessor()

    async def execute(
        self,
        prepared: PreparedEmbeddingRequest,
        *,
        provider: str,
        model: str,
    ) -> ProviderAttemptSuccess | ProviderCallFailure:
        plan = prepared.execution_plan
        read_identity = self._backend_identity_resolver(provider, model)
        results: list[list[float] | None] = []
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for index, text in enumerate(prepared.normalized_input.texts):
            key = self._cache_key_fn(text, provider, model, plan.dimensions, read_identity)
            cached = await self._cache.get(key)
            cached_vector = self._vector_processor.validate_cached_vector(cached)
            if cached_vector is None:
                results.append(None)
                miss_indices.append(index)
                miss_texts.append(text)
                continue
            results.append(
                self._vector_processor.process_cached_vector(
                    cached_vector,
                    provider=provider,
                    model=model,
                    dimensions=plan.dimensions,
                    dimension_policy=prepared.effective_dimension_policy,
                )
            )

        embeddings_from_adapter = False
        pending_cache_writes: list[tuple[str, list[float]]] = []
        if miss_texts:
            try:
                output = await self._executor.create(
                    miss_texts,
                    provider=provider,
                    model=model,
                    dimensions=plan.dimensions,
                )
            except EmbeddingDomainError as exc:
                return ProviderCallFailure(exc)

            miss_vectors, embeddings_from_adapter = _coerce_executor_output(output)
            cache_vectors = self._vector_processor.validate_vector_count(
                miss_vectors,
                expected=len(miss_texts),
                provider=provider,
                model=model,
            )
            processed_misses = self._vector_processor.process_vectors(
                cache_vectors,
                provider=provider,
                model=model,
                dimensions=plan.dimensions,
                dimension_policy=prepared.effective_dimension_policy,
            )
            write_identity = self._backend_identity_resolver(provider, model)
            for index, text, vector, cache_vector in zip(
                miss_indices,
                miss_texts,
                processed_misses,
                cache_vectors,
            ):
                results[index] = vector
                if not embeddings_from_adapter:
                    key = self._cache_key_fn(text, provider, model, plan.dimensions, write_identity)
                    pending_cache_writes.append((key, cache_vector))

        vectors = self._vector_processor.validate_vector_count(
            results,
            expected=len(prepared.normalized_input.texts),
            provider=provider,
            model=model,
        )
        for key, vector in pending_cache_writes:
            await self._cache.set(key, vector)
        return ProviderAttemptSuccess(
            vectors=vectors,
            provider=provider,
            model=model,
            cache_hits=len(results) - len(miss_indices),
            cache_misses=len(miss_indices),
            embeddings_from_adapter=embeddings_from_adapter,
        )
```

Also add `_coerce_executor_output` locally:

```python
def _coerce_executor_output(
    output: list[list[float]] | EmbeddingExecutorOutput,
) -> tuple[list[list[float]], bool]:
    if isinstance(output, EmbeddingExecutorOutput):
        return output.vectors, output.embeddings_from_adapter
    return output, False
```

Export the new protocols and DTOs in `provider_attempt.__all__`.

- [x] **Step 4: Run provider-attempt tests and verify GREEN**

Run the command from Step 2. Expected: all provider-attempt tests pass.

- [x] **Step 5: Commit the ordered attempt slice**

```bash
git add tldw_Server_API/app/core/Embeddings/provider_attempt.py tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py
git commit -m "refactor(embeddings): extract provider attempt cache execution"
```

### Task 3: Validation, Writeback, Adapter-Origin, and Failure Classification

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/provider_attempt.py`
- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py`

- [x] **Step 1: Write failing validation and adapter-origin tests**

Add these tests:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_writes_provider_native_vectors_after_full_response_validation():
    cache = RecordingCache({"hit|openai|text-embedding-3-small|read": [1.0, 0.0, 0.0]})
    executor = RecordingExecutor(vectors=[[0.25, 0.75, 0.5]])

    identities = iter(["read", "write"])
    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=cache,
        executor=executor,
        backend_identity_resolver=lambda provider, model: next(identities),
        vector_processor=EmbeddingVectorProcessor(),
    )

    result = await attempt.execute(
        _prepared(["hit", "miss"], dimensions=2),
        provider="openai",
        model="text-embedding-3-small",
    )

    assert result.vectors == [[1.0, 0.0], [0.25, 0.75]]
    assert cache.set_calls == [
        ("miss|openai|text-embedding-3-small|2|write", [0.25, 0.75, 0.5])
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_rejects_provider_malformed_response_before_writeback():
    cache = RecordingCache({"hit|openai|text-embedding-3-small|read": [1.0, 0.0]})
    executor = RecordingExecutor(vectors=[[0.25, 0.75, 0.5]])

    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=cache,
        executor=executor,
        backend_identity_resolver=lambda provider, model: "read",
        vector_processor=EmbeddingVectorProcessor(),
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await attempt.execute(
            _prepared(["hit", "miss"]),
            provider="openai",
            model="text-embedding-3-small",
        )

    assert exc_info.value.code == "provider_malformed_response"
    assert cache.set_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_treats_malformed_cached_vector_as_miss():
    cache = RecordingCache({"bad|openai|text-embedding-3-small|read": [float("nan")]})
    executor = RecordingExecutor(vectors=[[0.1, 0.2]])

    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=cache,
        executor=executor,
        backend_identity_resolver=lambda provider, model: "read",
        vector_processor=EmbeddingVectorProcessor(),
    )

    result = await attempt.execute(
        _prepared(["bad"]),
        provider="openai",
        model="text-embedding-3-small",
    )

    assert result.vectors == [[0.1, 0.2]]
    assert result.cache_hits == 0
    assert result.cache_misses == 1
    assert executor.calls[0]["texts"] == ["bad"]
    assert cache.set_calls == [("bad|openai|text-embedding-3-small|read", [0.1, 0.2])]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_skips_cache_write_for_adapter_originated_executor_output():
    cache = RecordingCache()

    class AdapterOriginExecutor(RecordingExecutor):
        async def create(self, texts, *, provider, model, dimensions):
            self.calls.append(
                {
                    "texts": texts,
                    "provider": provider,
                    "model": model,
                    "dimensions": dimensions,
                }
            )
            return EmbeddingExecutorOutput(
                vectors=[[0.1, 0.2]],
                embeddings_from_adapter=True,
            )

    executor = AdapterOriginExecutor()
    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=cache,
        executor=executor,
        backend_identity_resolver=lambda provider, model: "identity",
        vector_processor=EmbeddingVectorProcessor(),
    )

    result = await attempt.execute(
        _prepared(["adapter-origin"]),
        provider="openai",
        model="text-embedding-3-small",
    )

    assert result.vectors == [[0.1, 0.2]]
    assert result.embeddings_from_adapter is True
    assert cache.set_calls == []
```

- [x] **Step 2: Run the tests and verify RED where behavior is incomplete**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py -q
```

Expected: any missing validation/writeback behavior fails. If all tests pass because Task 2 implementation already covered them, record that the RED requirement was satisfied by adding the tests before any further production changes in this task.

- [x] **Step 3: Write failing provider-call and non-provider failure tests**

Add:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_returns_exact_provider_call_failure_from_executor():
    error = EmbeddingProviderError(
        "provider_unavailable",
        "provider down",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )

    class FailingExecutor(RecordingExecutor):
        async def create(self, texts, *, provider, model, dimensions):
            self.calls.append(
                {
                    "texts": texts,
                    "provider": provider,
                    "model": model,
                    "dimensions": dimensions,
                }
            )
            raise error

    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=RecordingCache(),
        executor=FailingExecutor(),
        backend_identity_resolver=lambda provider, model: "identity",
        vector_processor=EmbeddingVectorProcessor(),
    )

    result = await attempt.execute(
        _prepared(["one"]),
        provider="openai",
        model="text-embedding-3-small",
    )

    assert isinstance(result, ProviderCallFailure)
    assert result.error is error


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["identity", "cache_key", "cache_get", "cache_set"])
async def test_provider_attempt_non_provider_failures_propagate_without_call_failure(boundary):
    original = RuntimeError(f"{boundary} failed")

    def identity(provider: str, model: str) -> str:
        if boundary == "identity":
            raise original
        return "identity"

    def cache_key(text: str, provider: str, model: str, dimensions: int | None, backend_identity: str | None) -> str:
        if boundary == "cache_key":
            raise original
        return _cache_key(text, provider, model, dimensions, backend_identity)

    class BoundaryCache(RecordingCache):
        async def get(self, key: str) -> list[float] | None:
            if boundary == "cache_get":
                raise original
            return await super().get(key)

        async def set(self, key: str, value: list[float]) -> object:
            if boundary == "cache_set":
                raise original
            return await super().set(key, value)

    attempt = EmbeddingProviderAttempt(
        cache_key_fn=cache_key,
        cache=BoundaryCache(),
        executor=RecordingExecutor(vectors=[[0.1, 0.2]]),
        backend_identity_resolver=identity,
        vector_processor=EmbeddingVectorProcessor(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await attempt.execute(
            _prepared(["one"]),
            provider="openai",
            model="text-embedding-3-small",
        )

    assert exc_info.value is original
```

- [x] **Step 4: Implement the minimal missing behavior**

If the tests from Steps 1 or 3 fail, adjust only `provider_attempt.py` so:

- `EmbeddingDomainError` is caught only around `executor.create`.
- `ProviderCallFailure.error` is the exact original exception object.
- Identity resolution, cache-key derivation, cache access, vector validation, vector postprocessing, and cache writeback exceptions propagate.
- The complete ordered result is validated before any `cache.set`.
- `EmbeddingExecutorOutput(..., embeddings_from_adapter=True)` skips writeback but still returns processed vectors and miss counts.

- [x] **Step 5: Run provider-attempt tests and verify GREEN**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py -q
```

Expected: all provider-attempt tests pass.

- [x] **Step 6: Commit validation and failure classification**

```bash
git add tldw_Server_API/app/core/Embeddings/provider_attempt.py tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py
git commit -m "test(embeddings): cover provider attempt failure routing"
```

### Task 4: Legacy Fallback Write Identity Correction

**Files:**
- Modify: `tldw_Server_API/app/core/Embeddings/orchestrator.py`
- Modify: `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`

- [ ] **Step 1: Write the failing fallback write-identity correction test**

Add this test near the fallback identity coverage in `test_embedding_orchestrator.py`:

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_writeback_re_resolves_backend_identity_after_provider_execution():
    fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
    primary_error = EmbeddingProviderError(
        "provider_unavailable",
        "openai unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    cache = RecordingCache()
    executor = RecordingExecutor(
        failures={"openai": primary_error},
        provider_vectors={"huggingface": [[0.25, 0.75]]},
    )
    identity_calls: list[tuple[str, str]] = []

    def backend_identity_resolver(provider: str, model: str) -> str:
        identity_calls.append((provider, model))
        if provider == "huggingface":
            count = sum(1 for call in identity_calls if call == (provider, model))
            return f"{provider}:{model}:{'read' if count == 1 else 'write'}"
        return f"{provider}:{model}:identity"

    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        backend_identity_resolver=backend_identity_resolver,
        settings_fallback_chain={"openai": ["huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "huggingface": fallback_model,
            }
        },
    )
    prepared = orchestrator.prepare(
        "identity correction",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.provider == "huggingface"
    assert cache.get_keys == [
        "identity correction|openai|text-embedding-3-small|"
        "openai:text-embedding-3-small:identity",
        f"identity correction|huggingface|{fallback_model}|"
        f"huggingface:{fallback_model}:read",
    ]
    assert cache.set_calls == [
        (
            f"identity correction|huggingface|{fallback_model}|"
            f"huggingface:{fallback_model}:write",
            [0.25, 0.75],
        )
    ]
    assert identity_calls == [
        ("openai", "text-embedding-3-small"),
        ("openai", "text-embedding-3-small"),
        ("huggingface", fallback_model),
        ("huggingface", fallback_model),
    ]
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py::test_fallback_writeback_re_resolves_backend_identity_after_provider_execution -q
```

Expected: failure because fallback cache writeback uses the read-time identity.

- [ ] **Step 3: Apply the minimal fallback correction**

In `_execute_coherent_fallback`, keep the existing read identity before cache lookup. After executor success, vector-count validation, and request-specific postprocessing, resolve a second identity before building cache write keys:

```python
            write_backend_identity = self._backend_identity_resolver(provider, model)
            for index, text, vector, cache_vector in zip(
                miss_indices,
                miss_texts,
                canonical_vectors,
                cache_vectors,
            ):
                results[index] = vector
                if not embeddings_from_adapter:
                    key = self._cache_key(
                        text,
                        provider,
                        model,
                        plan.dimensions,
                        write_backend_identity,
                    )
                    pending_cache_writes.append((key, cache_vector))
```

Do not change primary execution or fallback exception classification in this task.

- [ ] **Step 4: Run the correction and guardrail tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py::test_fallback_writeback_re_resolves_backend_identity_after_provider_execution tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py::test_current_fallback_wide_domain_catch_advances_after_infrastructure_failure -q
```

Expected: the new correction test passes and the Stage 2D guardrail test still passes.

- [ ] **Step 5: Commit the correction**

```bash
git add tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py
git commit -m "fix(embeddings): re-resolve fallback write identity"
```

### Task 5: Focused Regression, Bandit, and Tracking Finalization

**Files:**
- Modify: `backlog/tasks/task-12973.3 - Extract-single-provider-Embeddings-execution-attempts.md`
- Verify: `tldw_Server_API/app/core/Embeddings/provider_attempt.py`
- Verify: `tldw_Server_API/app/core/Embeddings/orchestrator.py`
- Verify: `tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py`
- Verify: `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`

- [ ] **Step 1: Run the focused Stage 2C regression suite**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py -q
```

Expected: all selected tests pass. Investigate and fix failures before continuing.

- [ ] **Step 2: Run Bandit on touched production code**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Embeddings/provider_attempt.py tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/app/core/Embeddings/request_types.py -f json -o /tmp/bandit_embeddings_stage2c.json
```

Expected: no new security findings in touched production code.

- [ ] **Step 3: Run a git self-review**

Run:

```bash
git diff --check
git diff origin/codex/embeddings-workflow-stage2b-contracts -- tldw_Server_API/app/core/Embeddings/provider_attempt.py tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/app/core/Embeddings/request_types.py tldw_Server_API/tests/Embeddings_isolated/test_provider_attempt.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py
```

Expected: no whitespace errors; diff shows only Stage 2C provider-attempt extraction, shared executor-output relocation, fallback write identity correction, tests, plan, and backlog updates.

- [ ] **Step 4: Update Backlog acceptance criteria and notes**

Use Backlog.md tooling, not manual edits, to check completed acceptance criteria, append verification output, and add touched files. Example:

```bash
/opt/homebrew/bin/backlog task edit 12973.3 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --check-ac 7 --check-ac 8 --check-ac 9 --check-ac 10 --check-ac 11 --check-ac 12 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --append-notes "Verification: provider-attempt, orchestrator, workflow-types, endpoint parity focused tests passed; Bandit touched scope passed."
```

- [ ] **Step 5: Commit final tracking updates**

If the plan has not yet been committed, include it with the final tracking commit:

```bash
git add Docs/superpowers/plans/2026-07-27-embeddings-workflow-stage2c-provider-attempts-implementation-plan.md "backlog/tasks/task-12973.3 - Extract-single-provider-Embeddings-execution-attempts.md"
git commit -m "docs(embeddings): plan stage 2c provider attempts"
```

If the plan was already committed before implementation, commit only the final backlog updates:

```bash
git add "backlog/tasks/task-12973.3 - Extract-single-provider-Embeddings-execution-attempts.md"
git commit -m "docs(embeddings): record stage 2c verification"
```

## Self-Review Checklist

- Spec coverage: maps to Stage 2C readiness, cache order, provider-native writeback, identity timing, provider-call failure DTO, cache failure propagation, and side-by-side production isolation.
- Scope check: leaves Stage 2D fallback source-aware routing and Stage 2E runner wiring untouched.
- Type consistency: shared executor-output contract is imported from `request_types.py` and re-exported by `orchestrator.py`.
- Privacy check: no workflow events, trace metadata, raw input logging, cache keys in traces, provider response bodies, or credential material are added.
- Verification check: focused tests and Bandit commands are explicit and runnable from the Stage 2C worktree with the repository venv.
