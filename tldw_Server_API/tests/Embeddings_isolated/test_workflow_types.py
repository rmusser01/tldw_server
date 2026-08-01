from __future__ import annotations

from typing import get_args

import pytest

from tldw_Server_API.app.core import exceptions as core_exceptions
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
WORKFLOW_ID = "emb-wf-00000000000000000000000000000000"


def test_workflow_types_public_exports_include_safety_constants():
    assert "FORBIDDEN_METADATA_FIELDS" in workflow_types.__all__
    assert "FORBIDDEN_FIELD_SUBSTRINGS" in workflow_types.__all__
    assert "SAFE_TOKEN_COUNT_FIELDS" in workflow_types.__all__
    assert "MAX_METADATA_LIST_ITEMS" in workflow_types.__all__
    assert "MAX_METADATA_STRING_LENGTH" in workflow_types.__all__
    assert "SAFE_METADATA_ENUM_VALUES" in workflow_types.__all__
    assert "WORKFLOW_ID_PATTERN" in workflow_types.__all__


def test_workflow_trace_error_uses_central_exception_contract():
    assert EmbeddingWorkflowTraceError is getattr(
        core_exceptions,
        "EmbeddingWorkflowTraceError",
        None,
    )


def test_workflow_phase_and_status_literals_keep_completion_distinct():
    phases = get_args(workflow_types.EmbeddingWorkflowPhase)
    statuses = set(get_args(workflow_types.EmbeddingWorkflowStatus))

    assert phases[:5] == (
        "created",
        "resolving_intent",
        "normalizing",
        "resolving_policy",
        "planning",
    )
    assert "completed" not in phases
    assert "completed" in statuses
    assert "resolving_intent" in workflow_types.SAFE_METADATA_ENUM_VALUES["phase"]


def test_workflow_context_generates_id_without_retaining_request_or_user_identifiers():
    context = EmbeddingWorkflowContext.create(
        endpoint_path="/api/v1/embeddings",
        runner_mode="inline",
    )

    assert context.workflow_id.startswith("emb-wf-")
    assert context.workflow_id != "req-123"
    assert context.runner_mode == "inline"
    assert not hasattr(context, "request_id")
    assert not hasattr(context, "user_id")
    assert not hasattr(context, "raw_input")
    assert not hasattr(context, "texts")
    assert not hasattr(context, "api_key")


def test_workflow_contracts_reject_caller_controlled_workflow_ids():
    with pytest.raises(EmbeddingWorkflowTraceError):
        EmbeddingWorkflowContext(
            workflow_id="sk-proj-0123456789abcdef",
            runner_mode="inline",
        )

    with pytest.raises(EmbeddingWorkflowTraceError):
        EmbeddingWorkflowEvent(
            event_type="workflow_started",
            workflow_id="AKIAIOSFODNN7EXAMPLE",
            status="running",
        )


def test_safe_workflow_metadata_rejects_raw_input_and_secret_fields():
    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"raw_input": "do not store"})

    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"api_secret": "do not store"})

    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"authorization": "Bearer token"})


def test_safe_workflow_metadata_rejects_sensitive_field_name_fragments():
    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"input_texts": "do not store"})

    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"normalized_texts": 2})

    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"openai_api_key": "do not store"})

    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"request_authorization": "Bearer token"})

    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"provider_body_sample": "raw provider body"})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("provider", "Bearer provider-secret"),
        ("model", "sk-secret-model"),
        ("provider", "sk-proj-0123456789abcdef"),
        ("model", "hf_0123456789abcdef"),
        ("fallback_source", "github_pat_0123456789abcdef"),
        ("model", "AIzaSyA123456789012345678901234567890123"),
        ("provider", "AKIAIOSFODNN7EXAMPLE"),
        ("model", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature"),
        ("fallback_source", "raw input must not be traced"),
        ("response_header_names", ["x-request-id", "Bearer provider-secret"]),
        ("response_header_names", ["x-request-id", "sk-proj-0123456789abcdef"]),
    ],
)
def test_safe_workflow_metadata_rejects_sensitive_string_values(field_name, value):
    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({field_name: value})


def test_safe_workflow_metadata_allows_token_count_fields():
    metadata = safe_workflow_metadata(
        {
            "token_count": 3,
            "token_counts": [1, 2],
            "total_tokens": 3,
            "prompt_tokens": 3,
            "runner_mode": "inline",
            "execution_path": "legacy",
        }
    )

    assert metadata == {
        "token_count": 3,
        "token_counts": (1, 2),
        "total_tokens": 3,
        "prompt_tokens": 3,
        "runner_mode": "inline",
        "execution_path": "legacy",
    }


def test_safe_workflow_metadata_allows_attempt_counters_and_legacy_header_count():
    metadata = safe_workflow_metadata(
        {
            "attempt_count": 2,
            "fallback_attempt_count": 1,
            "response_header_count": 3,
        }
    )

    assert metadata == {
        "attempt_count": 2,
        "fallback_attempt_count": 1,
        "response_header_count": 3,
    }


def test_safe_workflow_metadata_rejects_unknown_fields_and_unapproved_enum_values():
    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"provider": "huggingface"})

    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"runner_mode": "sk-proj-0123456789abcdef"})


def test_safe_workflow_metadata_rejects_oversized_lists():
    max_items = getattr(workflow_types, "MAX_METADATA_LIST_ITEMS", 256)

    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"token_counts": [1] * (max_items + 1)})


def test_safe_workflow_metadata_rejects_oversized_strings():
    max_length = getattr(workflow_types, "MAX_METADATA_STRING_LENGTH", 4096)

    with pytest.raises(EmbeddingWorkflowTraceError):
        safe_workflow_metadata({"endpoint_path": "/" + "a" * max_length})


def test_safe_workflow_metadata_copies_accepted_lists():
    token_counts = [1]
    metadata = safe_workflow_metadata({"token_counts": token_counts})

    token_counts.append(2)

    assert metadata["token_counts"] == (1,)


def test_event_metadata_list_is_not_affected_by_later_caller_mutation():
    token_counts = [1]
    event = EmbeddingWorkflowEvent(
        event_type="workflow_started",
        workflow_id=WORKFLOW_ID,
        status="running",
        metadata={"token_counts": token_counts},
    )

    token_counts.append(2)

    assert event.metadata["token_counts"] == (1,)


def test_event_metadata_is_immutable_after_validation():
    event = EmbeddingWorkflowEvent(
        event_type="workflow_started",
        workflow_id=WORKFLOW_ID,
        status="running",
        metadata={
            "runner_mode": "inline",
            "token_counts": [1],
        },
    )

    with pytest.raises(TypeError):
        event.metadata["runner_mode"] = "durable"

    assert event.metadata["token_counts"] == (1,)
    assert not hasattr(event.metadata["token_counts"], "append")


def test_event_metadata_is_sanitized_on_construction():
    event = EmbeddingWorkflowEvent(
        event_type="phase_changed",
        workflow_id=WORKFLOW_ID,
        phase="normalizing",
        metadata={"execution_path": "legacy", "fallback_chain_length": 1},
    )

    assert event.metadata == {"execution_path": "legacy", "fallback_chain_length": 1}


def test_event_accepts_prevalidated_metadata_snapshot():
    metadata = safe_workflow_metadata(
        {
            "runner_mode": "inline",
            "token_counts": [1],
        }
    )

    event = EmbeddingWorkflowEvent(
        event_type="workflow_started",
        workflow_id=WORKFLOW_ID,
        status="running",
        metadata=metadata,
    )

    assert event.metadata == metadata
    assert event.metadata["token_counts"] == (1,)


def test_event_rejects_unsafe_metadata_on_construction():
    with pytest.raises(EmbeddingWorkflowTraceError):
        EmbeddingWorkflowEvent(
            event_type="workflow_failed",
            workflow_id=WORKFLOW_ID,
            phase="executing",
            metadata={"provider_body": "raw provider body"},
        )


def test_workflow_completed_event_accepts_finalizing_completed_contract():
    event = EmbeddingWorkflowEvent(
        event_type="workflow_completed",
        workflow_id=WORKFLOW_ID,
        phase="finalizing",
        status="completed",
    )

    assert event.phase == "finalizing"
    assert event.status == "completed"


def test_workflow_completed_event_rejects_wrong_phase():
    with pytest.raises(EmbeddingWorkflowTraceError):
        EmbeddingWorkflowEvent(
            event_type="workflow_completed",
            workflow_id=WORKFLOW_ID,
            phase="executing",
            status="completed",
        )


def test_workflow_completed_event_rejects_wrong_status():
    with pytest.raises(EmbeddingWorkflowTraceError):
        EmbeddingWorkflowEvent(
            event_type="workflow_completed",
            workflow_id=WORKFLOW_ID,
            phase="finalizing",
            status="running",
        )


def test_in_memory_collector_preserves_event_order_and_fails_closed_at_bound():
    collector = EmbeddingInMemoryWorkflowTraceCollector(max_events=2)

    collector.record(
        EmbeddingWorkflowEvent(
            event_type="workflow_started",
            workflow_id=WORKFLOW_ID,
            status="running",
        )
    )
    collector.record(
        EmbeddingWorkflowEvent(
            event_type="phase_changed",
            workflow_id=WORKFLOW_ID,
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
                workflow_id=WORKFLOW_ID,
                phase="finalizing",
                status="completed",
            )
        )


def test_noop_collector_is_disabled_and_retains_no_events():
    collector = EmbeddingNoopWorkflowTraceCollector()
    event = EmbeddingWorkflowEvent(
        event_type="workflow_started",
        workflow_id=WORKFLOW_ID,
        status="running",
    )

    collector.record(event)

    assert collector.enabled is False
    assert not hasattr(collector, "events")
