from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.RAG.rag_service.request_resolution import (
    ResolvedRAGRequest,
    resolve_legacy_standard_pipeline_request,
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


def test_resolve_rag_request_applies_search_agent_defaults_when_fields_omitted() -> None:
    request = SimpleNamespace(
        query="apply core search agent defaults",
        model_fields_set={"query"},
    )

    resolved = resolve_rag_request(
        request,
        get_profile_kwargs_fn=lambda profile: {},
        search_agent_setting_fn=lambda env_key, cfg_key: {
            ("SEARCH_QUERY_CLASSIFICATION", "search_query_classification"): "true",
            ("SEARCH_DEFAULT_MODE", "search_default_mode"): "quality",
            ("SEARCH_DISCUSSION_PLATFORMS", "search_discussion_platforms"): "reddit,stackoverflow",
            ("SEARCH_CLASSIFIER_PROVIDER", "search_classifier_provider"): "openai",
            ("SEARCH_CLASSIFIER_MODEL", "search_classifier_model"): "gpt-4o-mini",
            ("SEARCH_MAX_ITERATIONS_BALANCED", "search_max_iterations_balanced"): "7",
        }.get((env_key, cfg_key)),
    )

    assert resolved.payload["enable_query_classification"] is True
    assert resolved.payload["search_depth_mode"] == "quality"
    assert resolved.payload["discussion_platforms"] == ["reddit", "stackoverflow"]
    assert resolved.payload["classifier_provider"] == "openai"
    assert resolved.payload["classifier_model"] == "gpt-4o-mini"
    assert resolved.payload["research_max_iterations_balanced"] == 7


def test_resolve_rag_request_does_not_override_explicit_search_agent_fields() -> None:
    request = SimpleNamespace(
        query="explicit search agent values win",
        enable_query_classification=False,
        search_depth_mode="speed",
        discussion_platforms=["quora"],
        model_fields_set={
            "query",
            "enable_query_classification",
            "search_depth_mode",
            "discussion_platforms",
        },
    )

    resolved = resolve_rag_request(
        request,
        get_profile_kwargs_fn=lambda profile: {},
        search_agent_setting_fn=lambda env_key, cfg_key: {
            ("SEARCH_QUERY_CLASSIFICATION", "search_query_classification"): "true",
            ("SEARCH_DEFAULT_MODE", "search_default_mode"): "quality",
            ("SEARCH_DISCUSSION_PLATFORMS", "search_discussion_platforms"): "reddit,stackoverflow",
        }.get((env_key, cfg_key)),
    )

    assert resolved.payload["enable_query_classification"] is False
    assert resolved.payload["search_depth_mode"] == "speed"
    assert resolved.payload["discussion_platforms"] == ["quora"]


def test_resolve_rag_request_preserves_explicit_user_id_before_current_user() -> None:
    resolved = resolve_rag_request(
        {
            "query": None,
            "queries": ["fallback query"],
            "strategy": None,
            "user_id": "explicit-user",
        },
        current_user=SimpleNamespace(id=77),
        get_profile_kwargs_fn=lambda profile: {},
    )

    assert resolved.user_id == "explicit-user"
    assert resolved.feedback_user_id == "explicit-user"
    assert resolved.payload["user_id"] == "explicit-user"
    assert resolved.payload["feedback_user_id"] == "explicit-user"
    assert resolved.query == "fallback query"
    assert resolved.strategy == "standard"
    assert resolved.payload["query"] == "fallback query"
    assert resolved.payload["strategy"] == "standard"


def test_legacy_standard_request_payload_matches_identity_contract() -> None:
    resolved = resolve_legacy_standard_pipeline_request(
        query="legacy identity",
        search_mode="hybrid",
        top_k=3,
        sources=["media_db"],
        index_namespace="tenant-a",
        rag_profile="fast",
        user_id="user-1",
        feedback_user_id=None,
    )

    assert resolved.payload["user_id"] == "user-1"
    assert resolved.payload["feedback_user_id"] == "user-1"
    assert resolved.payload["index_namespace"] == "tenant-a"
    assert resolved.payload["rag_profile"] == "fast"
