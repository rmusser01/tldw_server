import pytest

import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import (
    UnifiedBatchRequest,
    UnifiedRAGRequest,
)


pytestmark = pytest.mark.integration


def test_standard_and_batch_shared_fields_resolve_to_the_same_effective_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_ep.DatabasePaths,
        "get_single_user_id",
        staticmethod(lambda: 42),
    )

    single_request = UnifiedRAGRequest(
        query="compare shared contract semantics",
        corpus="tenant-alpha",
        search_mode="vector",
        top_k=7,
        min_score=0.15,
        enable_generation=True,
        generation_prompt="concise",
        max_generation_tokens=640,
        user_id="single_user",
        feedback_user_id="single_user",
    )
    batch_request = UnifiedBatchRequest(
        queries=["compare shared contract semantics"],
        corpus="tenant-alpha",
        search_mode="vector",
        top_k=7,
        min_score=0.15,
        enable_generation=True,
        generation_prompt="concise",
        max_generation_tokens=640,
        user_id="single_user",
        feedback_user_id="single_user",
    )

    db_paths = {
        "media_db_path": None,
        "notes_db_path": None,
        "character_db_path": None,
        "kanban_db_path": None,
    }

    standard_bundle = rag_ep._build_standard_request_bundle(
        single_request,
        current_user=None,
        db_paths=db_paths,
        media_db=None,  # type: ignore[arg-type]
        chacha_db=None,  # type: ignore[arg-type]
    )
    batch_bundle = rag_ep._build_batch_request_bundle(
        request=batch_request,
        db_paths=db_paths,
        current_user=None,
    )
    resume_request = rag_ep._build_resume_batch_request(
        checkpoint_config=batch_request.model_dump(),
        remaining_queries=["compare shared contract semantics"],
        max_concurrent=3,
    )
    resume_bundle = rag_ep._build_batch_request_bundle(
        request=resume_request,
        db_paths=db_paths,
        current_user=None,
    )

    single_kwargs = standard_bundle.pipeline_kwargs
    batch_kwargs = batch_bundle.pipeline_kwargs
    resume_kwargs = resume_bundle.pipeline_kwargs

    assert standard_bundle.retrieval_plan.index_namespace == "tenant-alpha"
    assert batch_bundle.retrieval_plan.index_namespace == "tenant-alpha"
    assert resume_bundle.retrieval_plan.index_namespace == "tenant-alpha"
    assert standard_bundle.retrieval_plan.search_mode == "vector"
    assert batch_bundle.retrieval_plan.search_mode == "vector"
    assert resume_bundle.retrieval_plan.search_mode == "vector"
    assert standard_bundle.retrieval_plan.top_k == 7
    assert batch_bundle.retrieval_plan.top_k == 7
    assert resume_bundle.retrieval_plan.top_k == 7
    assert standard_bundle.retrieval_plan.min_score == 0.15
    assert batch_bundle.retrieval_plan.min_score == 0.15
    assert resume_bundle.retrieval_plan.min_score == 0.15

    assert standard_bundle.resolved_request.index_namespace == "tenant-alpha"
    assert batch_bundle.resolved_request.index_namespace == "tenant-alpha"
    assert resume_bundle.resolved_request.index_namespace == "tenant-alpha"
    assert standard_bundle.resolved_request.user_id == "42"
    assert batch_bundle.resolved_request.user_id == "42"
    assert resume_bundle.resolved_request.user_id == "42"
    assert standard_bundle.resolved_request.feedback_user_id == "42"
    assert batch_bundle.resolved_request.feedback_user_id == "42"
    assert resume_bundle.resolved_request.feedback_user_id == "42"

    assert single_kwargs["resolved_request"] is standard_bundle.resolved_request
    assert batch_kwargs["resolved_request"] is batch_bundle.resolved_request
    assert resume_kwargs["resolved_request"] is resume_bundle.resolved_request
    assert single_kwargs["retrieval_plan"] is standard_bundle.retrieval_plan
    assert batch_kwargs["retrieval_plan"] is batch_bundle.retrieval_plan
    assert resume_kwargs["retrieval_plan"] is resume_bundle.retrieval_plan

    assert single_kwargs["index_namespace"] == "tenant-alpha"
    assert batch_kwargs["index_namespace"] == "tenant-alpha"
    assert resume_kwargs["index_namespace"] == "tenant-alpha"
    assert single_kwargs["user_id"] == "42"
    assert batch_kwargs["user_id"] == "42"
    assert resume_kwargs["user_id"] == "42"
    assert single_kwargs["feedback_user_id"] == "42"
    assert batch_kwargs["feedback_user_id"] == "42"
    assert resume_kwargs["feedback_user_id"] == "42"
    assert "query" not in batch_kwargs
    assert "query" not in resume_kwargs

    standard_only = rag_ep._build_unified_pipeline_kwargs(
        request=single_request,
        db_paths={
            "media_db_path": None,
            "notes_db_path": None,
            "character_db_path": None,
            "kanban_db_path": None,
        },
        media_db=None,  # type: ignore[arg-type]
        chacha_db=None,  # type: ignore[arg-type]
        current_user=None,
        resolved_request=standard_bundle.resolved_request,
        retrieval_plan=standard_bundle.retrieval_plan,
    )
    assert standard_only["index_namespace"] == single_kwargs["index_namespace"]
    assert standard_only["user_id"] == single_kwargs["user_id"]
    assert standard_only["feedback_user_id"] == single_kwargs["feedback_user_id"]

    shared_fields = (
        "index_namespace",
        "search_mode",
        "top_k",
        "min_score",
        "enable_generation",
        "generation_prompt",
        "max_generation_tokens",
        "user_id",
        "feedback_user_id",
    )
    for field_name in shared_fields:
        assert single_kwargs[field_name] == batch_kwargs[field_name]
        assert single_kwargs[field_name] == resume_kwargs[field_name]
