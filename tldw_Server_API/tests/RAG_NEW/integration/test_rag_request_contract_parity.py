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

    single_kwargs = rag_ep._build_unified_pipeline_kwargs(
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
    )
    batch_kwargs = rag_ep._build_batch_pipeline_kwargs(
        request=batch_request,
        db_paths={
            "media_db_path": None,
            "notes_db_path": None,
            "character_db_path": None,
            "kanban_db_path": None,
        },
        current_user=None,
    )

    assert single_kwargs["index_namespace"] == "tenant-alpha"
    assert batch_kwargs["index_namespace"] == "tenant-alpha"
    assert single_kwargs["user_id"] == "42"
    assert batch_kwargs["user_id"] == "42"
    assert single_kwargs["feedback_user_id"] == "42"
    assert batch_kwargs["feedback_user_id"] == "42"
    assert "query" not in batch_kwargs

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
