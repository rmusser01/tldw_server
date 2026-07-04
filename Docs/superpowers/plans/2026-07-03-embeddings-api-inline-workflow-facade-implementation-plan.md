# Embeddings API Inline Workflow Facade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage 1 of the canonical Embeddings workflow architecture by wrapping the feature-flagged `/api/v1/embeddings` orchestrator path in a typed inline workflow runner with optional redacted test traces.

**Architecture:** Add dependency-light workflow contracts and an inline runner under `tldw_Server_API/app/core/Embeddings/`. The runner calls the existing `EmbeddingRequestOrchestrator.prepare()` and `.execute()` methods, preserves endpoint-owned ResourceGovernor reservation through a pre-execute hook, and returns the existing `EmbeddingExecutionResult`. Public endpoint behavior, feature flags, response headers, metrics, schemas, logs, and legacy shims stay unchanged.

**Tech Stack:** Python 3.14-compatible syntax already used in tests, dataclasses, typing protocols, FastAPI endpoint integration, pytest, pytest-asyncio, existing Embeddings orchestrator tests, Bandit.

---

## Source References

- Approved spec: `Docs/superpowers/specs/2026-07-03-embeddings-workflow-architecture-design.md`
- Backlog planning task: `TASK-12141`
- Existing orchestrator: `tldw_Server_API/app/core/Embeddings/orchestrator.py`
- Existing request contracts: `tldw_Server_API/app/core/Embeddings/request_types.py`
- Existing endpoint path: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Existing endpoint parity tests: `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`
- Existing isolated orchestrator tests: `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`

## File Structure

- Create `tldw_Server_API/app/core/Embeddings/workflow_types.py`
  - Owns workflow phase/status/item-state literals, safe metadata validation, events, no-op collector, and bounded in-memory collector.
  - Must not import FastAPI, DB clients, Redis clients, provider clients, endpoint schemas, or raw provider execution helpers.

- Create `tldw_Server_API/app/core/Embeddings/workflow_runner.py`
  - Owns `EmbeddingInlineWorkflowRunner`, runner-level sequencing, optional async pre-execute hook, and redacted trace emission.
  - Calls the existing `EmbeddingRequestOrchestrator` public methods and returns `EmbeddingExecutionResult`.

- Modify `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
  - Import `EmbeddingInlineWorkflowRunner`.
  - Add a small `_build_embedding_inline_workflow_runner(...)` helper for endpoint integration tests.
  - Replace direct `prepare()` / RG reserve / `execute()` calls in `_create_embedding_with_orchestrator()` with the runner plus a pre-execute hook that preserves current RG ordering.

- Create `tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py`
  - Unit tests for safe metadata, forbidden fields, bounded collection, no-op behavior, and event ordering.

- Create `tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py`
  - Unit tests with fake orchestrators for success, pre-execute hook ordering, domain error redaction, unexpected error redaction, and no-op collector behavior.

- Modify `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`
  - Add one narrow endpoint integration test proving the feature-flagged path uses the inline runner and preserves current response/RG ordering.

## Data Contracts

Use these names consistently across tasks:

```python
EmbeddingWorkflowPhase = Literal[
    "created",
    "normalizing",
    "resolving_policy",
    "planning",
    "serving_cache",
    "executing",
    "postprocessing",
    "persisting_outputs",
    "finalizing",
]

EmbeddingWorkflowStatus = Literal[
    "running",
    "completed",
    "failed",
    "paused",
    "cancelled",
    "retry_scheduled",
]

EmbeddingWorkflowItemState = Literal[
    "pending",
    "normalized",
    "cache_hit",
    "cache_miss",
    "provider_pending",
    "provider_succeeded",
    "postprocessed",
    "output_recorded",
    "failed",
]

EmbeddingWorkflowEventType = Literal[
    "workflow_started",
    "phase_changed",
    "prepare_completed",
    "execute_completed",
    "workflow_completed",
    "workflow_failed",
    "item_state_changed",
]
```

Safe metadata scalar values are `str | int | float | bool | None`. Safe metadata dictionaries may contain scalar values or short lists of scalar values. Metadata field names must reject raw input/secrets by field name.

Forbidden field names:

```python
FORBIDDEN_METADATA_FIELDS = frozenset(
    {
        "raw_input",
        "input",
        "texts",
        "token_arrays",
        "api_key",
        "authorization",
        "cookie",
        "nonce",
        "provider_response",
        "provider_body",
    }
)
FORBIDDEN_FIELD_SUBSTRINGS = ("secret", "password")
SAFE_TOKEN_COUNT_FIELDS = frozenset({"token_count", "token_counts", "total_tokens", "prompt_tokens"})
```

## Implementation Tasks

### Task 1: Add Workflow Type Contracts And Safe Trace Collectors

**Files:**
- Create: `tldw_Server_API/app/core/Embeddings/workflow_types.py`
- Create: `tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py`

- [ ] **Step 1: Write failing workflow type tests**

Add `tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py`:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Embeddings.workflow_types import (
    EmbeddingInMemoryWorkflowTraceCollector,
    EmbeddingNoopWorkflowTraceCollector,
    EmbeddingWorkflowContext,
    EmbeddingWorkflowEvent,
    EmbeddingWorkflowTraceError,
    safe_workflow_metadata,
)


pytestmark = pytest.mark.unit


def test_workflow_context_uses_request_id_as_safe_workflow_id():
    context = EmbeddingWorkflowContext.from_request(
        request_id="req-123",
        user_id=42,
        endpoint_path="/api/v1/embeddings",
        runner_mode="inline",
    )

    assert context.workflow_id == "req-123"
    assert context.request_id == "req-123"
    assert context.user_id == "42"
    assert context.runner_mode == "inline"
    assert not hasattr(context, "raw_input")
    assert not hasattr(context, "texts")
    assert not hasattr(context, "api_key")


def test_safe_workflow_metadata_rejects_raw_input_and_secret_fields():
    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"raw_input": "do not store"})

    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"api_secret": "do not store"})

    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"authorization": "Bearer token"})


def test_safe_workflow_metadata_allows_token_count_fields():
    metadata = safe_workflow_metadata(
        {
            "token_count": 3,
            "token_counts": [1, 2],
            "total_tokens": 3,
            "prompt_tokens": 3,
            "provider": "huggingface",
        }
    )

    assert metadata == {
        "token_count": 3,
        "token_counts": [1, 2],
        "total_tokens": 3,
        "prompt_tokens": 3,
        "provider": "huggingface",
    }


def test_event_metadata_is_sanitized_on_construction():
    event = EmbeddingWorkflowEvent(
        event_type="phase_changed",
        workflow_id="wf-1",
        phase="normalizing",
        metadata={"provider": "huggingface", "fallback_chain_length": 1},
    )

    assert event.metadata == {"provider": "huggingface", "fallback_chain_length": 1}


def test_event_rejects_unsafe_metadata_on_construction():
    with pytest.raises(EmbeddingWorkflowTraceError):
        EmbeddingWorkflowEvent(
            event_type="workflow_failed",
            workflow_id="wf-1",
            phase="executing",
            metadata={"provider_body": "raw provider body"},
        )


def test_in_memory_collector_preserves_event_order_and_fails_closed_at_bound():
    collector = EmbeddingInMemoryWorkflowTraceCollector(max_events=2)

    collector.record(
        EmbeddingWorkflowEvent(
            event_type="workflow_started",
            workflow_id="wf-1",
            status="running",
        )
    )
    collector.record(
        EmbeddingWorkflowEvent(
            event_type="phase_changed",
            workflow_id="wf-1",
            phase="normalizing",
        )
    )

    assert [event.event_type for event in collector.events] == [
        "workflow_started",
        "phase_changed",
    ]

    with pytest.raises(EmbeddingWorkflowTraceError):
        collector.record(
            EmbeddingWorkflowEvent(
                event_type="workflow_completed",
                workflow_id="wf-1",
                status="completed",
            )
        )


def test_noop_collector_is_disabled_and_retains_no_events():
    collector = EmbeddingNoopWorkflowTraceCollector()
    event = EmbeddingWorkflowEvent(
        event_type="workflow_started",
        workflow_id="wf-1",
        status="running",
    )

    collector.record(event)

    assert collector.enabled is False
```

- [ ] **Step 2: Run the workflow type tests and verify they fail**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py
```

Expected result: collection fails with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Embeddings.workflow_types'`.

- [ ] **Step 3: Implement `workflow_types.py`**

Create `tldw_Server_API/app/core/Embeddings/workflow_types.py` with these public objects:

```python
"""Workflow state and trace contracts for Embeddings execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias
from uuid import uuid4

EmbeddingWorkflowPhase = Literal[
    "created",
    "normalizing",
    "resolving_policy",
    "planning",
    "serving_cache",
    "executing",
    "postprocessing",
    "persisting_outputs",
    "finalizing",
]
EmbeddingWorkflowStatus = Literal[
    "running",
    "completed",
    "failed",
    "paused",
    "cancelled",
    "retry_scheduled",
]
EmbeddingWorkflowItemState = Literal[
    "pending",
    "normalized",
    "cache_hit",
    "cache_miss",
    "provider_pending",
    "provider_succeeded",
    "postprocessed",
    "output_recorded",
    "failed",
]
EmbeddingWorkflowEventType = Literal[
    "workflow_started",
    "phase_changed",
    "prepare_completed",
    "execute_completed",
    "workflow_completed",
    "workflow_failed",
    "item_state_changed",
]
EmbeddingWorkflowRunnerMode = Literal["inline", "durable"]

SafeWorkflowScalar: TypeAlias = str | int | float | bool | None
SafeWorkflowMetadataValue: TypeAlias = SafeWorkflowScalar | list[SafeWorkflowScalar]

FORBIDDEN_METADATA_FIELDS = frozenset(
    {
        "raw_input",
        "input",
        "texts",
        "token_arrays",
        "api_key",
        "authorization",
        "cookie",
        "nonce",
        "provider_response",
        "provider_body",
    }
)
FORBIDDEN_FIELD_SUBSTRINGS = ("secret", "password")
SAFE_TOKEN_COUNT_FIELDS = frozenset({"token_count", "token_counts", "total_tokens", "prompt_tokens"})


class EmbeddingWorkflowTraceError(ValueError):
    """Raised when workflow trace metadata is unsafe or exceeds collector bounds."""


def _validate_metadata_name(name: str) -> None:
    normalized = name.strip().lower()
    if normalized in FORBIDDEN_METADATA_FIELDS:
        raise EmbeddingWorkflowTraceError(f"Unsafe workflow metadata field: {name}")
    if "token" in normalized and normalized not in SAFE_TOKEN_COUNT_FIELDS:
        raise EmbeddingWorkflowTraceError(f"Unsafe workflow metadata field: {name}")
    if any(part in normalized for part in FORBIDDEN_FIELD_SUBSTRINGS):
        raise EmbeddingWorkflowTraceError(f"Unsafe workflow metadata field: {name}")


def _safe_metadata_value(value: object) -> SafeWorkflowMetadataValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list) and all(item is None or isinstance(item, (str, int, float, bool)) for item in value):
        return value
    raise EmbeddingWorkflowTraceError("Workflow metadata values must be safe scalars or lists of safe scalars")


def safe_workflow_metadata(metadata: dict[str, object] | None = None) -> dict[str, SafeWorkflowMetadataValue]:
    if not metadata:
        return {}
    safe: dict[str, SafeWorkflowMetadataValue] = {}
    for key, value in metadata.items():
        _validate_metadata_name(str(key))
        safe[str(key)] = _safe_metadata_value(value)
    return safe


@dataclass(frozen=True, slots=True)
class EmbeddingWorkflowContext:
    workflow_id: str
    runner_mode: EmbeddingWorkflowRunnerMode
    request_id: str | None = None
    user_id: str | None = None
    endpoint_path: str = "/api/v1/embeddings"

    @classmethod
    def from_request(
        cls,
        *,
        request_id: str | None,
        user_id: str | int | None,
        endpoint_path: str,
        runner_mode: EmbeddingWorkflowRunnerMode,
    ) -> "EmbeddingWorkflowContext":
        workflow_id = request_id or f"emb-wf-{uuid4().hex}"
        return cls(
            workflow_id=workflow_id,
            request_id=request_id,
            user_id=str(user_id) if user_id is not None else None,
            endpoint_path=endpoint_path,
            runner_mode=runner_mode,
        )


@dataclass(frozen=True, slots=True)
class EmbeddingWorkflowEvent:
    event_type: EmbeddingWorkflowEventType
    workflow_id: str
    phase: EmbeddingWorkflowPhase | None = None
    status: EmbeddingWorkflowStatus | None = None
    item_index: int | None = None
    item_state: EmbeddingWorkflowItemState | None = None
    metadata: dict[str, SafeWorkflowMetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", safe_workflow_metadata(dict(self.metadata)))


class EmbeddingWorkflowTraceCollector(Protocol):
    enabled: bool

    def record(self, event: EmbeddingWorkflowEvent) -> None:
        raise NotImplementedError


class EmbeddingNoopWorkflowTraceCollector:
    enabled = False

    def record(self, event: EmbeddingWorkflowEvent) -> None:
        del event


@dataclass(slots=True)
class EmbeddingInMemoryWorkflowTraceCollector:
    max_events: int = 256
    enabled: bool = True
    events: list[EmbeddingWorkflowEvent] = field(default_factory=list)

    def record(self, event: EmbeddingWorkflowEvent) -> None:
        if len(self.events) >= self.max_events:
            raise EmbeddingWorkflowTraceError("Workflow trace event limit exceeded")
        self.events.append(event)


__all__ = [
    "EmbeddingInMemoryWorkflowTraceCollector",
    "EmbeddingNoopWorkflowTraceCollector",
    "EmbeddingWorkflowContext",
    "EmbeddingWorkflowEvent",
    "EmbeddingWorkflowEventType",
    "EmbeddingWorkflowItemState",
    "EmbeddingWorkflowPhase",
    "EmbeddingWorkflowRunnerMode",
    "EmbeddingWorkflowStatus",
    "EmbeddingWorkflowTraceCollector",
    "EmbeddingWorkflowTraceError",
    "SafeWorkflowMetadataValue",
    "SafeWorkflowScalar",
    "safe_workflow_metadata",
]
```

- [ ] **Step 4: Run the workflow type tests and verify they pass**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py
```

Expected result: all tests in `test_workflow_types.py` pass.

- [ ] **Step 5: Compile and run Bandit on the new contract module**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q tldw_Server_API/app/core/Embeddings/workflow_types.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Embeddings/workflow_types.py -f json -o /tmp/bandit_embeddings_workflow_types.json
```

Expected result: compile exits `0`; Bandit JSON has zero findings.

- [ ] **Step 6: Commit Task 1**

```bash
git add tldw_Server_API/app/core/Embeddings/workflow_types.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py
git commit -m "feat(embeddings): add workflow trace contracts"
```

### Task 2: Add Inline Workflow Runner

**Files:**
- Create: `tldw_Server_API/app/core/Embeddings/workflow_runner.py`
- Create: `tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py`

- [ ] **Step 1: Write failing inline runner tests**

Add `tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py`:

```python
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionResult,
    EmbeddingProviderError,
    EmbeddingRequestContext,
)
from tldw_Server_API.app.core.Embeddings.workflow_runner import EmbeddingInlineWorkflowRunner
from tldw_Server_API.app.core.Embeddings.workflow_types import (
    EmbeddingInMemoryWorkflowTraceCollector,
    EmbeddingNoopWorkflowTraceCollector,
)


pytestmark = pytest.mark.unit


class FakePrepared:
    def __init__(self) -> None:
        self.normalized_input = SimpleNamespace(texts=["one", "two"], total_tokens=3)
        self.provider_intent = SimpleNamespace(provider="huggingface", model="sentence-transformers/all-MiniLM-L6-v2")
        self.policy_decision = SimpleNamespace(fallback_allowed=True, fallback_chain=["huggingface"])
        self.execution_plan = SimpleNamespace(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            dimensions=None,
            execution_path="legacy",
            cache_namespace="endpoint",
        )
        self.prompt_tokens = 3
        self.total_tokens = 3


class FakeOrchestrator:
    def __init__(self, *, prepare_error=None, execute_error=None) -> None:
        self.prepare_error = prepare_error
        self.execute_error = execute_error
        self.prepare_calls = []
        self.execute_calls = []
        self.prepared = FakePrepared()

    def prepare(self, raw_input, context):
        self.prepare_calls.append((raw_input, context))
        if self.prepare_error is not None:
            raise self.prepare_error
        return self.prepared

    async def execute(self, prepared):
        self.execute_calls.append(prepared)
        if self.execute_error is not None:
            raise self.execute_error
        return EmbeddingExecutionResult(
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            prompt_tokens=3,
            total_tokens=3,
            cache_hits=1,
            cache_misses=1,
            response_headers={"X-Embeddings-Provider": "huggingface"},
        )


def _context() -> EmbeddingRequestContext:
    return EmbeddingRequestContext(
        user_id="u1",
        model_field="sentence-transformers/all-MiniLM-L6-v2",
        provider_header="huggingface",
        dimensions=None,
        encoding_format="float",
        request_id="req-1",
    )


@pytest.mark.asyncio
async def test_inline_runner_returns_existing_execution_result_and_records_safe_events():
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    orchestrator = FakeOrchestrator()
    runner = EmbeddingInlineWorkflowRunner(orchestrator, trace_collector=collector)

    result = await runner.run(["one", "two"], _context())

    assert result.provider == "huggingface"
    assert result.cache_hits == 1
    assert orchestrator.prepare_calls[0][0] == ["one", "two"]
    assert orchestrator.execute_calls == [orchestrator.prepared]
    assert [event.event_type for event in collector.events] == [
        "workflow_started",
        "phase_changed",
        "prepare_completed",
        "phase_changed",
        "execute_completed",
        "workflow_completed",
    ]
    assert collector.events[2].metadata["item_count"] == 2
    assert collector.events[2].metadata["total_tokens"] == 3
    assert collector.events[4].metadata["cache_hits"] == 1
    assert collector.events[4].metadata["response_header_names"] == ["X-Embeddings-Provider"]


@pytest.mark.asyncio
async def test_inline_runner_awaits_pre_execute_hook_between_prepare_and_execute():
    call_order: list[str] = []
    orchestrator = FakeOrchestrator()

    async def pre_execute(prepared):
        assert prepared is orchestrator.prepared
        call_order.append("pre_execute")

    class OrderedOrchestrator(FakeOrchestrator):
        def prepare(self, raw_input, context):
            call_order.append("prepare")
            return super().prepare(raw_input, context)

        async def execute(self, prepared):
            call_order.append("execute")
            return await super().execute(prepared)

    ordered = OrderedOrchestrator()
    runner = EmbeddingInlineWorkflowRunner(ordered, pre_execute=pre_execute)

    await runner.run("input", _context())

    assert call_order == ["prepare", "pre_execute", "execute"]


@pytest.mark.asyncio
async def test_inline_runner_traces_domain_errors_with_redacted_metadata_and_reraises():
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    error = EmbeddingProviderError(
        "provider_unavailable",
        "provider failed with sensitive raw body that must not be copied",
        retryable=True,
        provider="openai",
        model="text-embedding-3-small",
        cause_class="TimeoutError",
    )
    runner = EmbeddingInlineWorkflowRunner(
        FakeOrchestrator(execute_error=error),
        trace_collector=collector,
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await runner.run("input", _context())

    assert exc_info.value is error
    failed = collector.events[-1]
    assert failed.event_type == "workflow_failed"
    assert failed.metadata["error_code"] == "provider_unavailable"
    assert failed.metadata["provider"] == "openai"
    assert failed.metadata["retryable"] is True
    assert "sensitive raw body" not in str(failed.metadata)


@pytest.mark.asyncio
async def test_inline_runner_traces_unexpected_errors_by_class_only_and_reraises():
    collector = EmbeddingInMemoryWorkflowTraceCollector()
    error = RuntimeError("raw provider body should not enter metadata")
    runner = EmbeddingInlineWorkflowRunner(
        FakeOrchestrator(execute_error=error),
        trace_collector=collector,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await runner.run("input", _context())

    assert exc_info.value is error
    failed = collector.events[-1]
    assert failed.metadata == {"cause_class": "RuntimeError", "phase": "executing"}


@pytest.mark.asyncio
async def test_noop_collector_does_not_retain_events():
    collector = EmbeddingNoopWorkflowTraceCollector()
    runner = EmbeddingInlineWorkflowRunner(FakeOrchestrator(), trace_collector=collector)

    result = await runner.run("input", _context())

    assert result.provider == "huggingface"
    assert collector.enabled is False
```

- [ ] **Step 2: Run the inline runner tests and verify they fail**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py
```

Expected result: collection fails with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Embeddings.workflow_runner'`.

- [ ] **Step 3: Implement `workflow_runner.py`**

Create `tldw_Server_API/app/core/Embeddings/workflow_runner.py`:

```python
"""Inline workflow runner for the feature-flagged Embeddings create path."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from tldw_Server_API.app.core.Embeddings.orchestrator import (
    EmbeddingRequestOrchestrator,
    PreparedEmbeddingRequest,
)
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingExecutionResult,
    EmbeddingRequestContext,
)
from tldw_Server_API.app.core.Embeddings.workflow_types import (
    EmbeddingNoopWorkflowTraceCollector,
    EmbeddingWorkflowContext,
    EmbeddingWorkflowEvent,
    EmbeddingWorkflowPhase,
    EmbeddingWorkflowTraceCollector,
)

PreExecuteHook = Callable[[PreparedEmbeddingRequest], Awaitable[None]]


class PrepareExecuteOrchestrator(Protocol):
    def prepare(self, raw_input: Any, context: EmbeddingRequestContext) -> PreparedEmbeddingRequest:
        raise NotImplementedError

    async def execute(self, prepared: PreparedEmbeddingRequest) -> EmbeddingExecutionResult:
        raise NotImplementedError


class EmbeddingInlineWorkflowRunner:
    """Run one embedding workflow inline while preserving current orchestrator behavior."""

    def __init__(
        self,
        orchestrator: PrepareExecuteOrchestrator | EmbeddingRequestOrchestrator,
        *,
        trace_collector: EmbeddingWorkflowTraceCollector | None = None,
        pre_execute: PreExecuteHook | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._trace_collector = trace_collector or EmbeddingNoopWorkflowTraceCollector()
        self._pre_execute = pre_execute

    async def run(self, raw_input: Any, context: EmbeddingRequestContext) -> EmbeddingExecutionResult:
        workflow_context = EmbeddingWorkflowContext.from_request(
            request_id=context.request_id,
            user_id=context.user_id,
            endpoint_path=context.endpoint_path,
            runner_mode="inline",
        )
        phase: EmbeddingWorkflowPhase = "created"
        self._emit("workflow_started", workflow_context, status="running")
        try:
            phase = "normalizing"
            self._emit("phase_changed", workflow_context, phase=phase)
            prepared = self._orchestrator.prepare(raw_input, context)
            self._emit_prepare_completed(workflow_context, prepared)

            if self._pre_execute is not None:
                await self._pre_execute(prepared)

            phase = "executing"
            self._emit("phase_changed", workflow_context, phase=phase)
            result = await self._orchestrator.execute(prepared)
            self._emit_execute_completed(workflow_context, result)
            self._emit("workflow_completed", workflow_context, status="completed")
            return result
        except EmbeddingDomainError as exc:
            self._emit_domain_failure(workflow_context, exc, phase)
            raise
        except Exception as exc:
            self._emit_unexpected_failure(workflow_context, exc, phase)
            raise

    def _emit(
        self,
        event_type: str,
        workflow_context: EmbeddingWorkflowContext,
        *,
        phase: EmbeddingWorkflowPhase | None = None,
        status: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        if not self._trace_collector.enabled:
            return
        self._trace_collector.record(
            EmbeddingWorkflowEvent(
                event_type=event_type,  # type: ignore[arg-type]
                workflow_id=workflow_context.workflow_id,
                phase=phase,
                status=status,  # type: ignore[arg-type]
                metadata=metadata or {},
            )
        )

    def _emit_prepare_completed(
        self,
        workflow_context: EmbeddingWorkflowContext,
        prepared: PreparedEmbeddingRequest,
    ) -> None:
        metadata = {
            "item_count": len(prepared.normalized_input.texts),
            "total_tokens": int(prepared.total_tokens),
            "prompt_tokens": int(prepared.prompt_tokens),
            "provider": prepared.execution_plan.provider,
            "model": prepared.execution_plan.model,
            "fallback_allowed": prepared.policy_decision.fallback_allowed,
            "fallback_chain_length": len(prepared.execution_plan.fallback_chain),
            "execution_path": prepared.execution_plan.execution_path,
        }
        if prepared.execution_plan.dimensions is not None:
            metadata["dimensions"] = prepared.execution_plan.dimensions
        if prepared.execution_plan.cache_namespace is not None:
            metadata["cache_namespace"] = prepared.execution_plan.cache_namespace
        self._emit("prepare_completed", workflow_context, phase="planning", metadata=metadata)

    def _emit_execute_completed(
        self,
        workflow_context: EmbeddingWorkflowContext,
        result: EmbeddingExecutionResult,
    ) -> None:
        self._emit(
            "execute_completed",
            workflow_context,
            phase="executing",
            metadata={
                "vector_count": len(result.vectors),
                "cache_hits": int(result.cache_hits),
                "cache_misses": int(result.cache_misses),
                "provider": result.provider,
                "model": result.model,
                "fallback_from": result.fallback_from,
                "embeddings_from_adapter": result.embeddings_from_adapter,
                "response_header_names": sorted(str(name) for name in result.response_headers),
            },
        )

    def _emit_domain_failure(
        self,
        workflow_context: EmbeddingWorkflowContext,
        exc: EmbeddingDomainError,
        phase: EmbeddingWorkflowPhase,
    ) -> None:
        self._emit(
            "workflow_failed",
            workflow_context,
            phase=phase,
            status="failed",
            metadata={
                "error_code": exc.code,
                "provider": exc.provider,
                "model": exc.model,
                "retryable": exc.retryable,
                "cause_class": exc.cause_class,
                "phase": phase,
            },
        )

    def _emit_unexpected_failure(
        self,
        workflow_context: EmbeddingWorkflowContext,
        exc: Exception,
        phase: EmbeddingWorkflowPhase,
    ) -> None:
        self._emit(
            "workflow_failed",
            workflow_context,
            phase=phase,
            status="failed",
            metadata={"cause_class": exc.__class__.__name__, "phase": phase},
        )


__all__ = ["EmbeddingInlineWorkflowRunner", "PreExecuteHook", "PrepareExecuteOrchestrator"]
```

- [ ] **Step 4: Run the inline runner tests and verify they pass**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py
```

Expected result: both workflow isolated test files pass.

- [ ] **Step 5: Compile and run Bandit on workflow production modules**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q tldw_Server_API/app/core/Embeddings/workflow_types.py tldw_Server_API/app/core/Embeddings/workflow_runner.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Embeddings/workflow_types.py tldw_Server_API/app/core/Embeddings/workflow_runner.py -f json -o /tmp/bandit_embeddings_workflow_runner.json
```

Expected result: compile exits `0`; Bandit JSON has zero findings.

- [ ] **Step 6: Commit Task 2**

```bash
git add tldw_Server_API/app/core/Embeddings/workflow_runner.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py
git commit -m "feat(embeddings): add inline workflow runner"
```

### Task 3: Wire The Feature-Flagged Endpoint Through The Inline Runner

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Modify: `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`

- [ ] **Step 1: Write a failing endpoint integration test for runner usage and RG ordering**

Append this test to `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`:

```python
def test_orchestrator_path_uses_inline_workflow_runner_and_preserves_rg_reservation(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        result=EmbeddingExecutionResult(
            vectors=[[0.25, 0.75]],
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            prompt_tokens=3,
            total_tokens=3,
            cache_hits=0,
            cache_misses=1,
        )
    )
    runner_calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    async def fake_reserve_embedding_rg_tokens(*, request, current_user, token_total):
        del request, current_user
        runner_calls.append(("reserved", token_total))
        return None, None, None, token_total

    monkeypatch.setattr(
        mod,
        "_reserve_embedding_rg_tokens",
        fake_reserve_embedding_rg_tokens,
        raising=False,
    )

    class RunnerProbe:
        def __init__(self, orchestrator, *, trace_collector=None, pre_execute=None):
            assert trace_collector is None
            self.orchestrator = orchestrator
            self.pre_execute = pre_execute

        async def run(self, raw_input, context):
            runner_calls.append(("runner_started", raw_input))
            prepared = self.orchestrator.prepare(raw_input, context)
            runner_calls.append(("prepared", prepared.normalized_input.total_tokens))
            assert self.pre_execute is not None
            await self.pre_execute(prepared)
            result = await self.orchestrator.execute(prepared)
            runner_calls.append(("executed", result.provider))
            return result

    monkeypatch.setattr(mod, "EmbeddingInlineWorkflowRunner", RunnerProbe, raising=False)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "workflow facade"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["data"][0]["index"] == 0
    assert runner_calls == [
        ("runner_started", "workflow facade"),
        ("prepared", 3),
        ("reserved", 3),
        ("executed", "huggingface"),
    ]
```

- [ ] **Step 2: Run the new endpoint test and verify it fails before endpoint wiring**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q --tb=short tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py::test_orchestrator_path_uses_inline_workflow_runner_and_preserves_rg_reservation
```

Expected result: test fails because `embeddings_v5_production_enhanced` has no `EmbeddingInlineWorkflowRunner` import or because direct `prepare()` / `execute()` calls bypass the runner probe.

- [ ] **Step 3: Import the runner and add an endpoint-local builder seam**

In `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`, update imports near the existing orchestrator import:

```python
from tldw_Server_API.app.core.Embeddings.workflow_runner import (
    EmbeddingInlineWorkflowRunner,
    PreExecuteHook,
)
```

Add this helper near `_build_embedding_request_orchestrator(...)`:

```python
def _build_embedding_inline_workflow_runner(
    orchestrator: EmbeddingRequestOrchestrator,
    *,
    pre_execute: PreExecuteHook | None = None,
) -> EmbeddingInlineWorkflowRunner:
    """Build the inline workflow runner for the feature-flagged embeddings path."""
    return EmbeddingInlineWorkflowRunner(orchestrator, pre_execute=pre_execute)
```

- [ ] **Step 4: Replace direct prepare/execute calls with the runner plus pre-execute hook**

In `_create_embedding_with_orchestrator(...)`, replace the current inner block:

```python
prepared = orchestrator.prepare(embedding_request.input, context)
rg_reserved_units = max(1, _prepared_total_tokens(prepared))
rg_governor, rg_handle_id, rg_commit_op_id, rg_reserved_units = await _reserve_embedding_rg_tokens(
    request=request,
    current_user=current_user,
    token_total=rg_reserved_units,
)
result = await orchestrator.execute(prepared)
```

with:

```python
async def _reserve_before_execute(prepared):
    nonlocal rg_governor, rg_handle_id, rg_commit_op_id, rg_reserved_units
    rg_reserved_units = max(1, _prepared_total_tokens(prepared))
    rg_governor, rg_handle_id, rg_commit_op_id, rg_reserved_units = await _reserve_embedding_rg_tokens(
        request=request,
        current_user=current_user,
        token_total=rg_reserved_units,
    )

workflow_runner = _build_embedding_inline_workflow_runner(
    orchestrator,
    pre_execute=_reserve_before_execute,
)
result = await workflow_runner.run(embedding_request.input, context)
```

Do not add trace collection in the endpoint. The default runner path must use the no-op collector.

- [ ] **Step 5: Run the new endpoint integration test and verify it passes**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py::test_orchestrator_path_uses_inline_workflow_runner_and_preserves_rg_reservation
```

Expected result: the new test passes and `runner_calls` is `[("runner_started", "workflow facade"), ("prepared", 3), ("reserved", 3), ("executed", "huggingface")]`.

- [ ] **Step 6: Run existing orchestrator endpoint parity coverage**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
```

Expected result: endpoint parity tests pass with no public response-shape changes.

- [ ] **Step 7: Compile and run Bandit on touched production files**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q tldw_Server_API/app/core/Embeddings/workflow_types.py tldw_Server_API/app/core/Embeddings/workflow_runner.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Embeddings/workflow_types.py tldw_Server_API/app/core/Embeddings/workflow_runner.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py -f json -o /tmp/bandit_embeddings_inline_workflow_endpoint.json
```

Expected result: compile exits `0`; Bandit JSON has zero new findings.

- [ ] **Step 8: Commit Task 3**

```bash
git add tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
git commit -m "feat(embeddings): route orchestrator path through workflow runner"
```

### Task 4: Focused Verification And Task Finalization

**Files:**
- Modify: `backlog/tasks/task-12142 - Implement-Embeddings-API-inline-workflow-facade.md`
- Modify: `Docs/superpowers/plans/2026-07-03-embeddings-api-inline-workflow-facade-implementation-plan.md` if execution notes are recorded in the plan.

- [ ] **Step 1: Run the focused Stage 1 workflow suite**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
```

Expected result: all selected tests pass.

- [ ] **Step 2: Run broader focused Embeddings regression coverage**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Embeddings_isolated/test_request_types.py \
  tldw_Server_API/tests/Embeddings_isolated/test_input_normalizer.py \
  tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_policy.py \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py \
  tldw_Server_API/tests/Embeddings/test_batch_rate_headers.py
```

Expected result: all selected tests pass. Existing warnings are acceptable if no new failures appear.

- [ ] **Step 3: Run compile, Bandit, and diff checks**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q \
  tldw_Server_API/app/core/Embeddings/workflow_types.py \
  tldw_Server_API/app/core/Embeddings/workflow_runner.py \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/Embeddings/workflow_types.py \
  tldw_Server_API/app/core/Embeddings/workflow_runner.py \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
  -f json -o /tmp/bandit_embeddings_inline_workflow_final.json
git diff --check
```

Expected result: compile exits `0`; Bandit JSON has zero new findings; `git diff --check` has no output.

- [ ] **Step 4: Update Backlog implementation task**

Update `TASK-12142` and record:

```text
Implemented Stage 1 API inline workflow facade. Added workflow type contracts, bounded/no-op trace collectors, inline runner with pre-execute RG boundary hook, and feature-flagged endpoint integration. Verification: workflow isolated tests, orchestrator isolated tests, endpoint parity tests, broader focused Embeddings regression suite, compileall, Bandit, and git diff --check.
```

Add the final verification commands and results to the task notes.

- [ ] **Step 5: Commit final task/plan updates**

```bash
git add "backlog/tasks/task-12142 - Implement-Embeddings-API-inline-workflow-facade.md" \
  Docs/superpowers/plans/2026-07-03-embeddings-api-inline-workflow-facade-implementation-plan.md
git commit -m "docs(embeddings): record workflow facade verification"
```

Create this commit only when the implementation task file or plan has changed after verification.

## Plan Self-Review Checklist

- Spec coverage:
  - Canonical workflow contracts: Task 1.
  - No-op/in-memory trace collection: Task 1.
  - Inline runner around existing orchestrator: Task 2.
  - Pre-execute RG boundary hook: Tasks 2 and 3.
  - Feature-flagged endpoint integration without public trace exposure: Task 3.
  - Existing behavior and parity verification: Tasks 3 and 4.

- Scope control:
  - No durable Jobs runner.
  - No Embeddings workflow database tables.
  - No media embeddings, vector-store batch, Redis worker, DLQ, compactor, or re-embed migration.
  - No new production metrics, logs, headers, schemas, or debug endpoints.
  - No default promotion of `EMBEDDINGS_ORCHESTRATOR_ENABLED`.

- Type consistency:
  - `EmbeddingInlineWorkflowRunner.run(raw_input, context)` returns `EmbeddingExecutionResult`.
  - `PreExecuteHook` receives `PreparedEmbeddingRequest` and returns `Awaitable[None]`.
  - `EmbeddingWorkflowTraceCollector.record(event)` is synchronous.
  - Default collector is `EmbeddingNoopWorkflowTraceCollector`.

## Final Verification Commands

Run before PR or completion:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q \
  tldw_Server_API/tests/Embeddings_isolated/test_request_types.py \
  tldw_Server_API/tests/Embeddings_isolated/test_input_normalizer.py \
  tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_policy.py \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py \
  tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py \
  tldw_Server_API/tests/Embeddings/test_batch_rate_headers.py
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q \
  tldw_Server_API/app/core/Embeddings/workflow_types.py \
  tldw_Server_API/app/core/Embeddings/workflow_runner.py \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r \
  tldw_Server_API/app/core/Embeddings/workflow_types.py \
  tldw_Server_API/app/core/Embeddings/workflow_runner.py \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
  -f json -o /tmp/bandit_embeddings_inline_workflow_final.json
git diff --check
```
