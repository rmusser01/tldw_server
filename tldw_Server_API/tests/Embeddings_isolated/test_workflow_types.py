from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Embeddings import workflow_types
from tldw_Server_API.app.core.Embeddings.workflow_types import (
    EmbeddingInMemoryWorkflowTraceCollector,
    EmbeddingNoopWorkflowTraceCollector,
    EmbeddingWorkflowContext,
    EmbeddingWorkflowEvent,
    EmbeddingWorkflowTraceError,
    safe_workflow_metadata,
)


pytestmark = pytest.mark.unit


def test_workflow_types_public_exports_include_safety_constants():
    assert "FORBIDDEN_METADATA_FIELDS" in workflow_types.__all__
    assert "FORBIDDEN_FIELD_SUBSTRINGS" in workflow_types.__all__
    assert "SAFE_TOKEN_COUNT_FIELDS" in workflow_types.__all__


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
    assert not hasattr(collector, "events")
