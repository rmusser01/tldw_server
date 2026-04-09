from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.RAG.rag_service.request_resolution import (
    ResolvedRAGRequest,
    resolve_rag_request,
)


pytestmark = pytest.mark.unit


def test_resolve_rag_request_applies_profile_aliases_and_user_fallbacks() -> None:
    request_payload = {
        "query": "profile alias resolution",
        "rag_profile": "quality",
        "user_id": "single_user",
        "corpus": "tenant-a",
    }

    resolved = resolve_rag_request(
        request_payload,
        current_user=None,
        get_profile_kwargs_fn=lambda profile: {"max_generation_tokens": 2200, "top_k": 16},
        single_user_id_resolver=lambda: 41,
    )

    assert isinstance(resolved, ResolvedRAGRequest)
    assert resolved.query == "profile alias resolution"
    assert resolved.strategy == "standard"
    assert resolved.index_namespace == "tenant-a"
    assert resolved.rag_profile == "accuracy"
    assert resolved.user_id == "41"
    assert resolved.feedback_user_id == "41"
    assert resolved.payload["rag_profile"] == "accuracy"
    assert resolved.payload["index_namespace"] == "tenant-a"
    assert resolved.payload["top_k"] == 16
    assert resolved.payload["max_generation_tokens"] == 2200


def test_resolve_rag_request_explicit_fields_override_profile_defaults() -> None:
    request = SimpleNamespace(
        query="explicit wins",
        rag_profile="speed",
        max_generation_tokens=700,
        model_fields_set={"query", "rag_profile", "max_generation_tokens"},
    )

    resolved = resolve_rag_request(
        request,
        get_profile_kwargs_fn=lambda profile: {"max_generation_tokens": 2200, "top_k": 6},
    )

    assert resolved.rag_profile == "fast"
    assert resolved.query == "explicit wins"
    assert resolved.strategy == "standard"
    assert resolved.payload["max_generation_tokens"] == 700
    assert resolved.payload["top_k"] == 6
