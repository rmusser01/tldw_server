from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Embeddings.request_types import EmbeddingExecutionResult
from tldw_Server_API.app.core.Notes_Graph import semantic_embeddings
from tldw_Server_API.app.core.Notes_Graph.semantic_content import build_semantic_chunks
from tldw_Server_API.app.core.Notes_Graph.semantic_embeddings import (
    DIMENSION_PROBE_TEXT,
    NotesEmbeddingExecutionIdentity,
    NotesEmbeddingRuntime,
    NotesSemanticEmbedder,
    PendingSemanticConfig,
    ResolvedDimension,
    ResolvedSemanticConfig,
    SemanticEmbeddingSystemError,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings

pytestmark = pytest.mark.unit

_SUPPORTS_TASK_CANCELLATION_COUNTS = all(
    hasattr(asyncio.Task, attribute) for attribute in ("cancelling", "uncancel")
)


def test_embedding_execution_identity_repr_hides_runtime_endpoint_details() -> None:
    endpoint = (
        "https://creduser:credsecret@example.test/private-segment/credential-path"
        "?api_key=query-secret"
    )
    identity = NotesEmbeddingExecutionIdentity(endpoint_base_url=endpoint)

    serialized = repr(identity)

    assert identity.endpoint_base_url == endpoint
    for forbidden in (
        "creduser",
        "credsecret",
        "private-segment",
        "credential-path",
        "api_key",
        "query-secret",
    ):
        assert forbidden not in serialized


@dataclass
class RecordingOrchestrator:
    vectors: list[list[float]]
    result_provider: str = "openai"
    result_model: str = "text-embedding-3-small"

    def __post_init__(self) -> None:
        self.contexts: list[object] = []
        self.inputs: list[list[str]] = []
        self.vector_offset = 0

    def prepare(self, raw_input: list[str], context: Any) -> object:
        self.inputs.append(raw_input)
        self.contexts.append(context)
        return SimpleNamespace(
            execution_plan=SimpleNamespace(
                provider=context.provider_header,
                model=context.model_field,
                dimensions=context.dimensions,
                fallback_chain=[context.provider_header],
            ),
            policy_decision=SimpleNamespace(fallback_allowed=False),
        )

    async def execute(self, prepared: object) -> EmbeddingExecutionResult:
        del prepared
        vector_count = len(self.inputs[-1])
        vectors = self.vectors[self.vector_offset : self.vector_offset + vector_count]
        self.vector_offset += vector_count
        return EmbeddingExecutionResult(
            vectors=vectors,
            provider=self.result_provider,
            model=self.result_model,
            prompt_tokens=7,
            total_tokens=7,
            cache_hits=0,
            cache_misses=len(vectors),
        )


class SequencedOrchestrator(RecordingOrchestrator):
    def __init__(
        self,
        outcomes: list[EmbeddingExecutionResult | Exception],
        revisions: list[str | None],
    ) -> None:
        super().__init__([])
        self.outcomes = outcomes
        self.revisions = revisions
        self.identity = SimpleNamespace(
            model_revision=None,
            endpoint_origin="https://api.openai.com",
            credential_source="server_default",
            provider_attempt_sequence=0,
            provider_input_count=0,
            provider_prompt_tokens=0,
            provider_request_count=0,
            provider_status=None,
        )

    async def execute(self, prepared: object) -> EmbeddingExecutionResult:
        del prepared
        index = len(self.inputs) - 1
        outcome = self.outcomes[index]
        self.identity.model_revision = self.revisions[index]
        self.identity.provider_attempt_sequence += 1
        self.identity.provider_input_count = len(self.inputs[-1])
        self.identity.provider_prompt_tokens = 5
        self.identity.provider_request_count = 1
        if isinstance(outcome, Exception):
            self.identity.provider_status = "failed"
            raise outcome
        if outcome.cache_misses == 0:
            self.identity.provider_attempt_sequence -= 1
            self.identity.provider_input_count = 0
            self.identity.provider_prompt_tokens = 0
            self.identity.provider_request_count = 0
            self.identity.provider_status = None
        else:
            self.identity.provider_input_count = outcome.cache_misses
            self.identity.provider_status = "success"
        return outcome


def _settings(**overrides: int) -> SemanticIndexSettings:
    values = {
        "max_stored_note_bytes": 10_000,
        "max_canonical_field_code_points": 10_000,
        "max_chunk_code_points": 20,
        "max_chunks_per_note": 20,
        "max_chunks_per_run": 20,
        "max_provider_input_bytes": 1_000,
        "max_provider_batch_inputs": 10,
        "max_provider_batch_bytes": 10_000,
        "max_provider_bytes_per_run": 20_000,
        "max_provider_requests_per_run": 10,
    }
    values.update(overrides)
    return SemanticIndexSettings(**values)


def _pending(**overrides: object) -> PendingSemanticConfig:
    values = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "model_revision": None,
        "endpoint_origin": "https://api.openai.com",
        "credential_source": "server_default",
        "consented": True,
        "dimensions": None,
    }
    values.update(overrides)
    return PendingSemanticConfig(**values)


def _resolved(**overrides: object) -> ResolvedSemanticConfig:
    values = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "model_revision": "revision-1",
        "endpoint_origin": "https://api.openai.com",
        "credential_source": "server_default",
        "dimensions": 2,
    }
    values.update(overrides)
    return ResolvedSemanticConfig(**values)


def _runtime(orchestrator: RecordingOrchestrator, revision: str | None = None) -> NotesEmbeddingRuntime:
    return NotesEmbeddingRuntime(
        orchestrator=orchestrator,
        execution_identity=lambda: SimpleNamespace(
            model_revision=revision,
            endpoint_origin="https://api.openai.com",
            credential_source="server_default",
            provider_attempt_sequence=len(orchestrator.inputs),
            provider_input_count=len(orchestrator.inputs[-1]) if orchestrator.inputs else 0,
            provider_prompt_tokens=0,
            provider_request_count=1 if orchestrator.inputs else 0,
            provider_status="success" if orchestrator.inputs else None,
        ),
    )


def test_shared_batch_planner_applies_byte_and_input_boundaries_exactly() -> None:
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title=None,
        content="aa bb cc dd",
        content_version=1,
        settings=_settings(max_chunk_code_points=2),
    )
    settings = _settings(
        max_chunk_code_points=2,
        max_provider_batch_inputs=10,
        max_provider_batch_bytes=2,
        max_provider_input_bytes=2,
        max_provider_requests_per_run=20,
    )

    plan = semantic_embeddings.plan_semantic_embedding_batches(chunks, settings)

    assert plan.input_count == len(chunks)
    assert plan.request_count == len(chunks)
    assert all(len(batch) == 1 for batch in plan.batches)
    assert all(size <= settings.max_provider_batch_bytes for size in plan.batch_bytes)


@pytest.mark.asyncio
async def test_embedding_batch_proves_runtime_endpoint_scope_and_request_count() -> None:
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title=None,
        content="alpha beta gamma delta",
        content_version=1,
        settings=_settings(max_chunk_code_points=6),
    )
    orchestrator = RecordingOrchestrator([[1.0, 0.0] for _ in chunks])
    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: _runtime(orchestrator, "revision-1"),
        usage_logger=lambda **kwargs: asyncio.sleep(0),
        settings=_settings(max_chunk_code_points=6, max_provider_batch_inputs=1),
    )

    batch = await embedder.embed_chunks(chunks, _resolved(), user_id="7")

    assert batch.endpoint_origin == "https://api.openai.com"
    assert batch.credential_source == "server_default"
    assert batch.provider_request_count == len(chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("identity_change", "code"),
    [
        ({"endpoint_origin": "https://wrong.example"}, "endpoint_origin_mismatch"),
        ({"credential_source": "user"}, "credential_scope_mismatch"),
    ],
)
async def test_embedding_rejects_provider_runtime_identity_drift(identity_change, code) -> None:
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="Title",
        content="Body",
        content_version=1,
    )
    orchestrator = RecordingOrchestrator([[1.0, 0.0] for _ in chunks])
    identity = {
        "model_revision": "revision-1",
        "endpoint_origin": "https://api.openai.com",
        "credential_source": "server_default",
        "provider_attempt_sequence": 1,
        "provider_input_count": len(chunks),
        "provider_prompt_tokens": 1,
        "provider_request_count": 1,
        "provider_status": "success",
    }
    identity.update(identity_change)
    runtime = NotesEmbeddingRuntime(
        orchestrator=orchestrator,
        execution_identity=lambda: SimpleNamespace(**identity),
    )
    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: runtime,
        usage_logger=lambda **kwargs: asyncio.sleep(0),
    )

    with pytest.raises(SemanticEmbeddingSystemError, match=code):
        await embedder.embed_chunks(chunks, _resolved(), user_id="7")


@pytest.mark.asyncio
async def test_known_dimension_requires_no_probe() -> None:
    orchestrator = RecordingOrchestrator([[1.0, 0.0]])
    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: _runtime(orchestrator),
    )

    resolved = await embedder.resolve_dimensions(
        _pending(dimensions=2),
        user_id="7",
    )

    assert resolved.dimensions == 2
    assert orchestrator.inputs == []


@pytest.mark.asyncio
async def test_unknown_dimension_uses_fixed_probe_after_consent_and_captures_revision() -> None:
    orchestrator = RecordingOrchestrator([[3.0, 4.0, 5.0]])
    cas_calls: list[ResolvedDimension] = []
    usage_calls: list[dict[str, object]] = []

    async def publish_dimension(
        config: PendingSemanticConfig,
        resolved_dimension: ResolvedDimension,
    ) -> bool:
        del config
        cas_calls.append(resolved_dimension)
        return True

    async def record_usage(**kwargs: object) -> None:
        usage_calls.append(kwargs)

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: _runtime(orchestrator, "digest-7"),
        dimension_cas=publish_dimension,
        usage_logger=record_usage,
    )

    resolved = await embedder.resolve_dimensions(_pending(), user_id="7")

    assert orchestrator.inputs == [[DIMENSION_PROBE_TEXT]]
    assert resolved.dimensions == 3
    assert resolved.model == "text-embedding-3-small"
    assert resolved.model_revision == "digest-7"
    assert cas_calls == [resolved]
    assert cas_calls[0] is resolved
    assert len(usage_calls) == 1
    assert usage_calls[0]["operation"] == "notes_semantic_dimension_probe"
    assert usage_calls[0]["status"] == 200
    assert usage_calls[0]["usage_metadata"] == {
        "attempt_status": "success",
        "cache_hit_count": 0,
        "cache_miss_count": 1,
        "provider_input_count": 1,
        "provider_request_count": 1,
    }
    assert "Public semantic" not in repr(usage_calls)


@pytest.mark.asyncio
async def test_probe_fails_before_any_note_input_when_consent_missing_or_cas_lost() -> None:
    orchestrator = RecordingOrchestrator([[1.0, 0.0]])
    cas_calls: list[ResolvedDimension] = []

    def lose_cas(
        config: PendingSemanticConfig,
        resolved_dimension: ResolvedDimension,
    ) -> bool:
        del config
        cas_calls.append(resolved_dimension)
        return False

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: _runtime(
            orchestrator,
            "digest-on-cas-loss",
        ),
        dimension_cas=lose_cas,
    )

    with pytest.raises(SemanticEmbeddingSystemError, match="consent_required"):
        await embedder.resolve_dimensions(_pending(consented=False), user_id="7")
    assert orchestrator.inputs == []

    with pytest.raises(SemanticEmbeddingSystemError, match="dimension_cas_lost"):
        await embedder.resolve_dimensions(_pending(), user_id="7")
    assert orchestrator.inputs == [[DIMENSION_PROBE_TEXT]]
    assert cas_calls == [
        ResolvedDimension(
            dimensions=2,
            provider="openai",
            model="text-embedding-3-small",
            model_revision="digest-on-cas-loss",
            endpoint_origin="https://api.openai.com",
            credential_source="server_default",
        )
    ]


@pytest.mark.asyncio
async def test_unknown_dimension_sync_cas_receives_complete_discovered_identity() -> None:
    orchestrator = RecordingOrchestrator([[3.0, 4.0]])
    cas_calls: list[ResolvedDimension] = []

    def publish_dimension(
        config: PendingSemanticConfig,
        resolved_dimension: ResolvedDimension,
    ) -> bool:
        del config
        cas_calls.append(resolved_dimension)
        return True

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: _runtime(
            orchestrator,
            "digest-sync",
        ),
        dimension_cas=publish_dimension,
    )

    resolved = await embedder.resolve_dimensions(_pending(), user_id="7")

    assert cas_calls == [resolved]
    assert cas_calls[0] is resolved
    assert resolved == ResolvedDimension(
        dimensions=2,
        provider="openai",
        model="text-embedding-3-small",
        model_revision="digest-sync",
        endpoint_origin="https://api.openai.com",
        credential_source="server_default",
    )


@pytest.mark.asyncio
async def test_cancelled_dimension_probe_records_one_failed_provider_attempt() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    usage_calls: list[dict[str, object]] = []

    class BlockingOrchestrator(RecordingOrchestrator):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.identity = SimpleNamespace(
                model_revision=None,
                endpoint_origin="https://api.openai.com",
                credential_source="server_default",
                provider_attempt_sequence=0,
                provider_input_count=0,
                provider_prompt_tokens=0,
                provider_request_count=0,
                provider_status=None,
            )

        async def execute(self, prepared: object) -> EmbeddingExecutionResult:
            del prepared
            self.identity.provider_attempt_sequence = 1
            self.identity.provider_input_count = 1
            self.identity.provider_prompt_tokens = 5
            self.identity.provider_request_count = 1
            self.identity.provider_status = "started"
            started.set()
            await release.wait()
            raise AssertionError("cancelled provider unexpectedly resumed")

    orchestrator = BlockingOrchestrator([])

    async def record_usage(**kwargs: object) -> None:
        usage_calls.append(kwargs)

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: NotesEmbeddingRuntime(
            orchestrator=orchestrator,
            execution_identity=lambda: orchestrator.identity,
        ),
        dimension_cas=lambda config, resolution: True,
        usage_logger=record_usage,
    )

    task = asyncio.create_task(embedder.resolve_dimensions(_pending(), user_id="7"))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(usage_calls) == 1
    assert usage_calls[0]["status"] == 502
    assert usage_calls[0]["usage_metadata"] == {
        "attempt_status": "failed",
        "cache_hit_count": 0,
        "cache_miss_count": 1,
        "provider_input_count": 1,
        "provider_request_count": 1,
    }


@pytest.mark.asyncio
async def test_cancellation_during_success_usage_drains_one_success_row() -> None:
    """Drain the append-only success row instead of retrying or reclassifying billing."""

    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="",
        content="body",
        content_version=1,
        settings=_settings(),
    )
    usage_started = asyncio.Event()
    release_usage = asyncio.Event()
    usage_calls: list[dict[str, object]] = []

    async def record_usage(**kwargs: object) -> None:
        usage_started.set()
        await release_usage.wait()
        usage_calls.append(kwargs)

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: _runtime(
            RecordingOrchestrator([[1.0, 0.0]])
        ),
        usage_logger=record_usage,
        settings=_settings(),
    )
    baseline_tasks = set(asyncio.all_tasks())
    operation = asyncio.create_task(embedder.embed_chunks(chunks, _resolved(), user_id="7"))
    await usage_started.wait()

    assert operation.cancel("accounting-cancel") is True
    await asyncio.sleep(0)
    completed_while_logger_blocked = operation.done()
    release_usage.set()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await operation
    await asyncio.sleep(0)

    pending_children = [
        task
        for task in asyncio.all_tasks() - baseline_tasks
        if task is not asyncio.current_task() and not task.done()
    ]
    assert completed_while_logger_blocked is False
    if _SUPPORTS_TASK_CANCELLATION_COUNTS:
        assert exc_info.value.args == ("accounting-cancel",)
        assert operation.cancelling() == 0
    assert len(usage_calls) == 1
    assert usage_calls[0]["status"] == 200
    assert usage_calls[0]["usage_metadata"] == {
        "attempt_status": "success",
        "cache_hit_count": 0,
        "cache_miss_count": 1,
        "provider_input_count": 1,
        "provider_request_count": 1,
    }
    assert pending_children == []


@pytest.mark.asyncio
async def test_repeated_cancellation_drains_one_failed_usage_row() -> None:
    provider_started = asyncio.Event()
    provider_release = asyncio.Event()
    usage_started = asyncio.Event()
    release_usage = asyncio.Event()
    usage_calls: list[dict[str, object]] = []

    class BlockingOrchestrator(RecordingOrchestrator):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.identity = SimpleNamespace(
                model_revision=None,
                endpoint_origin="https://api.openai.com",
                credential_source="server_default",
                provider_attempt_sequence=0,
                provider_input_count=0,
                provider_prompt_tokens=0,
                provider_request_count=0,
                provider_status=None,
            )

        async def execute(self, prepared: object) -> EmbeddingExecutionResult:
            del prepared
            self.identity.provider_attempt_sequence = 1
            self.identity.provider_input_count = 1
            self.identity.provider_prompt_tokens = 5
            self.identity.provider_request_count = 1
            self.identity.provider_status = "started"
            provider_started.set()
            await provider_release.wait()
            raise AssertionError("cancelled provider unexpectedly resumed")

    orchestrator = BlockingOrchestrator([])

    async def record_usage(**kwargs: object) -> None:
        usage_started.set()
        await release_usage.wait()
        usage_calls.append(kwargs)

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: NotesEmbeddingRuntime(
            orchestrator=orchestrator,
            execution_identity=lambda: orchestrator.identity,
        ),
        dimension_cas=lambda config, resolution: True,
        usage_logger=record_usage,
    )
    baseline_tasks = set(asyncio.all_tasks())
    operation = asyncio.create_task(embedder.resolve_dimensions(_pending(), user_id="7"))
    await provider_started.wait()

    assert operation.cancel("provider-cancel") is True
    await usage_started.wait()
    assert operation.cancel("repeat-one") is True
    await asyncio.sleep(0)
    second_repeat_was_accepted = operation.cancel("repeat-two")
    await asyncio.sleep(0)
    completed_while_logger_blocked = operation.done()
    release_usage.set()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await operation
    await asyncio.sleep(0)

    pending_children = [
        task
        for task in asyncio.all_tasks() - baseline_tasks
        if task is not asyncio.current_task() and not task.done()
    ]
    assert second_repeat_was_accepted is True
    assert completed_while_logger_blocked is False
    if _SUPPORTS_TASK_CANCELLATION_COUNTS:
        assert exc_info.value.args == ("provider-cancel",)
        assert operation.cancelling() == 0
    assert len(usage_calls) == 1
    assert usage_calls[0]["status"] == 502
    assert usage_calls[0]["usage_metadata"] == {
        "attempt_status": "failed",
        "cache_hit_count": 0,
        "cache_miss_count": 1,
        "provider_input_count": 1,
        "provider_request_count": 1,
    }
    assert pending_children == []


@pytest.mark.asyncio
async def test_logger_exception_after_provider_cancellation_preserves_cancelled_error() -> None:
    provider_started = asyncio.Event()
    usage_calls: list[dict[str, object]] = []

    class BlockingOrchestrator(RecordingOrchestrator):
        def __post_init__(self) -> None:
            super().__post_init__()
            self.identity = SimpleNamespace(
                model_revision=None,
                endpoint_origin="https://api.openai.com",
                credential_source="server_default",
                provider_attempt_sequence=0,
                provider_input_count=0,
                provider_prompt_tokens=0,
                provider_request_count=0,
                provider_status=None,
            )

        async def execute(self, prepared: object) -> EmbeddingExecutionResult:
            del prepared
            self.identity.provider_attempt_sequence = 1
            self.identity.provider_input_count = 1
            self.identity.provider_prompt_tokens = 5
            self.identity.provider_request_count = 1
            self.identity.provider_status = "started"
            provider_started.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled provider unexpectedly resumed")

    orchestrator = BlockingOrchestrator([])

    async def record_usage(**kwargs: object) -> None:
        usage_calls.append(kwargs)
        raise RuntimeError("usage logger failed")

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: NotesEmbeddingRuntime(
            orchestrator=orchestrator,
            execution_identity=lambda: orchestrator.identity,
        ),
        dimension_cas=lambda config, resolution: True,
        usage_logger=record_usage,
    )
    baseline_tasks = set(asyncio.all_tasks())
    operation = asyncio.create_task(embedder.resolve_dimensions(_pending(), user_id="7"))
    await provider_started.wait()

    operation.cancel("provider-cancel")
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await operation
    await asyncio.sleep(0)

    pending_children = [
        task
        for task in asyncio.all_tasks() - baseline_tasks
        if task is not asyncio.current_task() and not task.done()
    ]
    if _SUPPORTS_TASK_CANCELLATION_COUNTS:
        assert exc_info.value.args == ("provider-cancel",)
        assert operation.cancelling() == 0
    assert len(usage_calls) == 1
    assert usage_calls[0]["status"] == 502
    assert usage_calls[0]["usage_metadata"] == {
        "attempt_status": "failed",
        "cache_hit_count": 0,
        "cache_miss_count": 1,
        "provider_input_count": 1,
        "provider_request_count": 1,
    }
    assert pending_children == []


@pytest.mark.asyncio
async def test_logger_exception_after_accounting_cancellation_preserves_cancelled_error() -> None:
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="",
        content="body",
        content_version=1,
        settings=_settings(),
    )
    usage_started = asyncio.Event()
    release_usage = asyncio.Event()
    usage_calls: list[dict[str, object]] = []

    async def record_usage(**kwargs: object) -> None:
        usage_calls.append(kwargs)
        usage_started.set()
        await release_usage.wait()
        raise RuntimeError("usage logger failed")

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: _runtime(
            RecordingOrchestrator([[1.0, 0.0]])
        ),
        usage_logger=record_usage,
        settings=_settings(),
    )
    baseline_tasks = set(asyncio.all_tasks())
    operation = asyncio.create_task(embedder.embed_chunks(chunks, _resolved(), user_id="7"))
    await usage_started.wait()

    operation.cancel("accounting-cancel")
    await asyncio.sleep(0)
    release_usage.set()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await operation
    await asyncio.sleep(0)

    pending_children = [
        task
        for task in asyncio.all_tasks() - baseline_tasks
        if task is not asyncio.current_task() and not task.done()
    ]
    if _SUPPORTS_TASK_CANCELLATION_COUNTS:
        assert exc_info.value.args == ("accounting-cancel",)
        assert operation.cancelling() == 0
    assert len(usage_calls) == 1
    assert usage_calls[0]["status"] == 200
    assert usage_calls[0]["usage_metadata"] == {
        "attempt_status": "success",
        "cache_hit_count": 0,
        "cache_miss_count": 1,
        "provider_input_count": 1,
        "provider_request_count": 1,
    }
    assert pending_children == []


@pytest.mark.asyncio
async def test_non_cancelled_logger_exception_remains_visible() -> None:
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="",
        content="body",
        content_version=1,
        settings=_settings(),
    )
    usage_calls: list[dict[str, object]] = []

    async def record_usage(**kwargs: object) -> None:
        usage_calls.append(kwargs)
        raise RuntimeError("usage logger failed")

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: _runtime(
            RecordingOrchestrator([[1.0, 0.0]])
        ),
        usage_logger=record_usage,
        settings=_settings(),
    )

    with pytest.raises(RuntimeError, match="usage logger failed"):
        await embedder.embed_chunks(chunks, _resolved(), user_id="7")

    assert len(usage_calls) == 1
    assert usage_calls[0]["status"] == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("vectors", "code"),
    [
        ([[0.0, 0.0]], "zero_norm_vector"),
        ([[float("nan"), 1.0]], "invalid_vectors"),
        ([[1.0, 2.0, 3.0]], "dimension_mismatch"),
    ],
)
async def test_embedding_rejects_invalid_provider_vectors(
    vectors: list[list[float]],
    code: str,
) -> None:
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="Title",
        content="Body",
        content_version=1,
        settings=_settings(),
    )
    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: _runtime(RecordingOrchestrator(vectors)),
        settings=_settings(),
    )

    with pytest.raises(SemanticEmbeddingSystemError, match=code):
        await embedder.embed_chunks(chunks, _resolved(), user_id="7")


@pytest.mark.asyncio
async def test_embedding_batches_by_input_count_and_bytes_and_records_content_free_usage() -> None:
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="",
        content="abcdefghijkl",
        content_version=1,
        settings=_settings(max_chunk_code_points=4),
    )
    orchestrator = RecordingOrchestrator([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    usage_calls: list[dict[str, object]] = []

    async def record_usage(**kwargs: object) -> None:
        usage_calls.append(kwargs)

    def factory(config: object, user_id: str) -> NotesEmbeddingRuntime:
        del config, user_id
        return _runtime(orchestrator, "revision-1")

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=factory,
        usage_logger=record_usage,
        settings=_settings(max_provider_batch_inputs=2),
    )

    batch = await embedder.embed_chunks(chunks, _resolved(), user_id="7")

    assert len(batch.vectors) == 3
    assert batch.provider == "openai"
    assert batch.model == "text-embedding-3-small"
    assert batch.model_revision == "revision-1"
    assert len(usage_calls) == 2
    assert all(
        not {"texts", "chunks", "content", "title"}.intersection(call)
        and "abcd" not in repr(call)
        for call in usage_calls
    )


@pytest.mark.asyncio
async def test_embedding_rejects_provider_or_model_drift_and_run_caps_before_execution() -> None:
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="",
        content="abcdefgh",
        content_version=1,
        settings=_settings(max_chunk_code_points=4),
    )
    drift = RecordingOrchestrator(
        [[1.0, 0.0], [0.0, 1.0]],
        result_provider="openai",
        result_model="text-embedding-3-large",
    )
    drift_embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: _runtime(drift),
        settings=_settings(),
    )

    with pytest.raises(SemanticEmbeddingSystemError, match="provider_model_drift"):
        await drift_embedder.embed_chunks(chunks, _resolved(), user_id="7")

    cap_orchestrator = RecordingOrchestrator([[1.0, 0.0]])
    capped = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: _runtime(cap_orchestrator),
        settings=_settings(
            max_chunks_per_note=1,
            max_chunks_per_run=1,
            max_provider_batch_inputs=1,
            max_query_vectors_per_call=1,
        ),
    )
    with pytest.raises(SemanticEmbeddingSystemError, match="run_chunk_cap_exceeded"):
        await capped.embed_chunks(chunks, _resolved(), user_id="7")
    assert cap_orchestrator.inputs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("second_revision", ["revision-2", None])
async def test_first_discovered_revision_is_pinned_across_batches(
    second_revision: str | None,
) -> None:
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="",
        content="abcdefgh",
        content_version=1,
        settings=_settings(max_chunk_code_points=4),
    )
    outcomes = [
        EmbeddingExecutionResult(
            vectors=[[1.0, 0.0]],
            provider="openai",
            model="text-embedding-3-small",
            prompt_tokens=5,
            total_tokens=5,
            cache_hits=0,
            cache_misses=1,
        ),
        EmbeddingExecutionResult(
            vectors=[[0.0, 1.0]],
            provider="openai",
            model="text-embedding-3-small",
            prompt_tokens=5,
            total_tokens=5,
            cache_hits=0,
            cache_misses=1,
        ),
    ]
    orchestrator = SequencedOrchestrator(
        outcomes,
        ["revision-1", second_revision],
    )
    usage_calls: list[dict[str, object]] = []

    async def record_usage(**kwargs: object) -> None:
        usage_calls.append(kwargs)

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: NotesEmbeddingRuntime(
            orchestrator=orchestrator,
            execution_identity=lambda: orchestrator.identity,
        ),
        usage_logger=record_usage,
        settings=_settings(max_provider_batch_inputs=1),
    )

    with pytest.raises(SemanticEmbeddingSystemError, match="model_revision_drift"):
        await embedder.embed_chunks(
            chunks,
            _resolved(model_revision=None),
            user_id="7",
        )

    assert [call["status"] for call in usage_calls] == [200, 502]
    assert usage_calls[1]["usage_metadata"] == {
        "attempt_status": "failed",
        "cache_hit_count": 0,
        "cache_miss_count": 1,
        "provider_input_count": 1,
        "provider_request_count": 1,
    }


@pytest.mark.asyncio
async def test_failed_provider_attempt_is_recorded_without_content() -> None:
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="",
        content="secret note body",
        content_version=1,
        settings=_settings(),
    )
    orchestrator = SequencedOrchestrator(
        [SemanticEmbeddingSystemError("provider_execution_failed")],
        [None],
    )
    usage_calls: list[dict[str, object]] = []

    async def record_usage(**kwargs: object) -> None:
        usage_calls.append(kwargs)

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: NotesEmbeddingRuntime(
            orchestrator=orchestrator,
            execution_identity=lambda: orchestrator.identity,
        ),
        usage_logger=record_usage,
        settings=_settings(),
    )

    with pytest.raises(SemanticEmbeddingSystemError, match="provider_execution_failed"):
        await embedder.embed_chunks(chunks, _resolved(), user_id="7")

    assert len(usage_calls) == 1
    assert usage_calls[0]["status"] == 502
    assert usage_calls[0]["prompt_tokens"] == 5
    assert usage_calls[0]["usage_metadata"] == {
        "attempt_status": "failed",
        "cache_hit_count": 0,
        "cache_miss_count": 1,
        "provider_input_count": 1,
        "provider_request_count": 1,
    }
    assert "secret note body" not in repr(usage_calls)


@pytest.mark.asyncio
async def test_full_cache_hit_batch_records_no_provider_work_or_tokens() -> None:
    chunks = build_semantic_chunks(
        generation_id="generation-1",
        note_id="note-1",
        title="",
        content="repeat",
        content_version=1,
        settings=_settings(),
    )
    outcomes = [
        EmbeddingExecutionResult(
            vectors=[[1.0, 0.0]],
            provider="openai",
            model="text-embedding-3-small",
            prompt_tokens=5,
            total_tokens=5,
            cache_hits=0,
            cache_misses=1,
        ),
        EmbeddingExecutionResult(
            vectors=[[1.0, 0.0]],
            provider="openai",
            model="text-embedding-3-small",
            prompt_tokens=5,
            total_tokens=5,
            cache_hits=1,
            cache_misses=0,
        ),
    ]
    orchestrator = SequencedOrchestrator(outcomes, ["revision-1", "revision-1"])
    usage_calls: list[dict[str, object]] = []

    async def record_usage(**kwargs: object) -> None:
        usage_calls.append(kwargs)

    embedder = NotesSemanticEmbedder(
        orchestrator_factory=lambda config, user_id: NotesEmbeddingRuntime(
            orchestrator=orchestrator,
            execution_identity=lambda: orchestrator.identity,
        ),
        usage_logger=record_usage,
        settings=_settings(max_provider_batch_inputs=1),
    )

    batch = await embedder.embed_chunks(
        [chunks[0], chunks[0]],
        _resolved(),
        user_id="7",
    )

    assert len(batch.vectors) == 2
    assert batch.prompt_tokens == 5
    assert batch.total_tokens == 5
    assert len(usage_calls) == 1
    assert usage_calls[0]["usage_metadata"] == {
        "attempt_status": "success",
        "cache_hit_count": 0,
        "cache_miss_count": 1,
        "provider_input_count": 1,
        "provider_request_count": 1,
    }
