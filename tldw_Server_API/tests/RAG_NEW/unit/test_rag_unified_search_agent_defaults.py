import pytest
from types import SimpleNamespace

import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedBatchRequest, UnifiedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import resolve_rag_request


pytestmark = pytest.mark.unit


_EMPTY_DB_PATHS = {
    "media_db_path": None,
    "notes_db_path": None,
    "character_db_path": None,
    "kanban_db_path": None,
}


def _config_value_factory(mapping: dict[str, str]):
    def _fake_get_config_value(section: str, key: str, default=None, reload: bool = False):  # noqa: ANN001, ARG001
        if section != "Search-Agent":
            return default
        return mapping.get(key, default)

    return _fake_get_config_value

def test_search_agent_defaults_apply_when_request_fields_omitted(monkeypatch):
    for env_key in (
        "SEARCH_QUERY_CLASSIFICATION",
        "SEARCH_DEFAULT_MODE",
        "SEARCH_QUERY_REFORMULATION",
        "SEARCH_RESEARCH_LOOP",
        "SEARCH_DISCUSSIONS_ENABLED",
        "SEARCH_DISCUSSION_PLATFORMS",
        "SEARCH_PROGRESS_STREAMING",
        "SEARCH_URL_SCRAPING",
        "SEARCH_SUGGESTIONS",
        "SEARCH_STRUCTURED_RESPONSE",
        "SEARCH_IMAGE_SEARCH",
        "SEARCH_VIDEO_SEARCH",
        "SEARCH_CLASSIFIER_PROVIDER",
        "SEARCH_CLASSIFIER_MODEL",
        "SEARCH_MAX_ITERATIONS_SPEED",
        "SEARCH_MAX_ITERATIONS_BALANCED",
        "SEARCH_MAX_ITERATIONS_QUALITY",
    ):
        monkeypatch.delenv(env_key, raising=False)

    monkeypatch.setattr(
        rag_ep,
        "get_config_value",
        _config_value_factory(
            {
                "search_query_classification": "true",
                "search_default_mode": "quality",
                "search_query_reformulation": "false",
                "search_research_loop": "true",
                "search_discussions_enabled": "true",
                "search_discussion_platforms": "reddit,stackoverflow",
                "search_progress_streaming": "true",
                "search_url_scraping": "false",
                "search_suggestions": "true",
                "search_structured_response": "true",
                "search_image_search": "true",
                "search_video_search": "true",
                "search_classifier_provider": "openai",
                "search_classifier_model": "gpt-4o-mini",
                "search_max_iterations_speed": "3",
                "search_max_iterations_balanced": "7",
                "search_max_iterations_quality": "15",
            }
        ),
    )

    request = UnifiedRAGRequest(query="default behavior check")
    kwargs = rag_ep._build_unified_pipeline_kwargs(
        request=request,
        db_paths=_EMPTY_DB_PATHS,
        media_db=None,  # type: ignore[arg-type]
        chacha_db=None,  # type: ignore[arg-type]
        current_user=None,
    )

    assert kwargs["enable_query_classification"] is True
    assert kwargs["search_depth_mode"] == "quality"
    assert kwargs["enable_query_reformulation"] is False
    assert kwargs["enable_research_loop"] is True
    assert kwargs["enable_discussion_search"] is True
    assert kwargs["discussion_platforms"] == ["reddit", "stackoverflow"]
    assert kwargs["enable_research_progress"] is True
    assert kwargs["search_url_scraping"] is False
    assert kwargs["enable_suggestions"] is True
    assert kwargs["enable_structured_response"] is True
    assert kwargs["enable_image_search"] is True
    assert kwargs["enable_video_search"] is True
    assert kwargs["classifier_provider"] == "openai"
    assert kwargs["classifier_model"] == "gpt-4o-mini"
    assert kwargs["research_max_iterations_speed"] == 3
    assert kwargs["research_max_iterations_balanced"] == 7
    assert kwargs["research_max_iterations_quality"] == 15

def test_search_agent_defaults_do_not_override_explicit_request_values(monkeypatch):
    monkeypatch.setattr(
        rag_ep,
        "get_config_value",
        _config_value_factory(
            {
                "search_query_classification": "true",
                "search_default_mode": "quality",
                "search_query_reformulation": "true",
                "search_research_loop": "true",
                "search_discussions_enabled": "true",
                "search_discussion_platforms": "reddit,stackoverflow",
                "search_url_scraping": "true",
                "search_suggestions": "true",
                "search_structured_response": "true",
                "search_image_search": "true",
                "search_video_search": "true",
                "search_classifier_provider": "openai",
                "search_classifier_model": "gpt-4.1",
                "search_max_iterations_speed": "9",
                "search_max_iterations_balanced": "9",
                "search_max_iterations_quality": "9",
            }
        ),
    )

    request = UnifiedRAGRequest(
        query="explicit should win",
        enable_query_classification=False,
        search_depth_mode="speed",
        enable_query_reformulation=False,
        enable_research_loop=False,
        enable_discussion_search=False,
        discussion_platforms=["quora"],
        search_url_scraping=False,
        enable_suggestions=False,
        enable_structured_response=False,
        enable_image_search=False,
        enable_video_search=False,
        classifier_provider="anthropic",
        classifier_model="claude-3-5-sonnet",
        research_max_iterations_speed=1,
        research_max_iterations_balanced=2,
        research_max_iterations_quality=3,
    )
    kwargs = rag_ep._build_unified_pipeline_kwargs(
        request=request,
        db_paths=_EMPTY_DB_PATHS,
        media_db=None,  # type: ignore[arg-type]
        chacha_db=None,  # type: ignore[arg-type]
        current_user=None,
    )

    assert kwargs["enable_query_classification"] is False
    assert kwargs["search_depth_mode"] == "speed"
    assert kwargs["enable_query_reformulation"] is False
    assert kwargs["enable_research_loop"] is False
    assert kwargs["enable_discussion_search"] is False
    assert kwargs["discussion_platforms"] == ["quora"]
    assert kwargs["search_url_scraping"] is False
    assert kwargs["enable_suggestions"] is False
    assert kwargs["enable_structured_response"] is False
    assert kwargs["enable_image_search"] is False
    assert kwargs["enable_video_search"] is False
    assert kwargs["classifier_provider"] == "anthropic"
    assert kwargs["classifier_model"] == "claude-3-5-sonnet"
    assert kwargs["research_max_iterations_speed"] == 1
    assert kwargs["research_max_iterations_balanced"] == 2
    assert kwargs["research_max_iterations_quality"] == 3

def test_search_agent_env_overrides_config_defaults(monkeypatch):
    monkeypatch.setattr(
        rag_ep,
        "get_config_value",
        _config_value_factory(
            {
                "search_query_classification": "false",
                "search_default_mode": "speed",
                "search_discussion_platforms": "reddit",
                "search_suggestions": "false",
            }
        ),
    )

    monkeypatch.setenv("SEARCH_QUERY_CLASSIFICATION", "true")
    monkeypatch.setenv("SEARCH_DEFAULT_MODE", "balanced")
    monkeypatch.setenv("SEARCH_DISCUSSION_PLATFORMS", "stackoverflow,hackernews")
    monkeypatch.setenv("SEARCH_SUGGESTIONS", "true")

    request = UnifiedRAGRequest(query="env should override config")
    kwargs = rag_ep._build_unified_pipeline_kwargs(
        request=request,
        db_paths=_EMPTY_DB_PATHS,
        media_db=None,  # type: ignore[arg-type]
        chacha_db=None,  # type: ignore[arg-type]
        current_user=None,
    )

    assert kwargs["enable_query_classification"] is True
    assert kwargs["search_depth_mode"] == "balanced"
    assert kwargs["discussion_platforms"] == ["stackoverflow", "hackernews"]
    assert kwargs["enable_suggestions"] is True

def test_build_kwargs_uses_authenticated_numeric_user_id_for_storage_paths():
    request = UnifiedRAGRequest(query="normalize authenticated user id")
    current_user = SimpleNamespace(id=1, id_int=1, username="single_user")

    kwargs = rag_ep._build_unified_pipeline_kwargs(
        request=request,
        db_paths=_EMPTY_DB_PATHS,
        media_db=None,  # type: ignore[arg-type]
        chacha_db=None,  # type: ignore[arg-type]
        current_user=current_user,  # type: ignore[arg-type]
    )

    assert kwargs["user_id"] == "1"
    assert kwargs["feedback_user_id"] == "1"

def test_resolve_implicit_feedback_user_id_maps_single_user_alias_without_user(monkeypatch):
    monkeypatch.setattr(
        rag_ep.DatabasePaths,
        "get_single_user_id",
        staticmethod(lambda: 7),
    )

    resolved = rag_ep._resolve_implicit_feedback_user_id("single_user", None)

    assert resolved == "7"


def test_core_request_resolution_matches_endpoint_single_user_alias(monkeypatch):
    monkeypatch.setattr(
        rag_ep.DatabasePaths,
        "get_single_user_id",
        staticmethod(lambda: 7),
    )

    endpoint_value = rag_ep._resolve_implicit_feedback_user_id("single_user", None)
    core_resolved = resolve_rag_request(
        {"query": "q", "user_id": "single_user"},
        current_user=None,
        get_profile_kwargs_fn=lambda _: {},
        single_user_id_resolver=rag_ep.DatabasePaths.get_single_user_id,
    )

    assert endpoint_value == core_resolved.user_id

def test_batch_round2_defaults_apply_when_fields_omitted(monkeypatch):
    for env_key in (
        "SEARCH_SUGGESTIONS",
        "SEARCH_STRUCTURED_RESPONSE",
        "SEARCH_IMAGE_SEARCH",
        "SEARCH_VIDEO_SEARCH",
    ):
        monkeypatch.delenv(env_key, raising=False)

    monkeypatch.setattr(
        rag_ep,
        "get_config_value",
        _config_value_factory(
            {
                "search_suggestions": "true",
                "search_structured_response": "true",
                "search_image_search": "false",
                "search_video_search": "true",
            }
        ),
    )

    request = UnifiedBatchRequest(queries=["q1"])
    kwargs = rag_ep._build_batch_pipeline_kwargs(
        request,
        db_paths=_EMPTY_DB_PATHS,
        current_user=None,
    )

    assert kwargs["enable_suggestions"] is True
    assert kwargs["enable_structured_response"] is True
    assert kwargs["enable_image_search"] is False
    assert kwargs["enable_video_search"] is True


def test_build_batch_kwargs_uses_core_user_normalization_when_single_user_resolution_fails(
    monkeypatch,
):
    def _raise_single_user_id():
        raise RuntimeError("single-user id unavailable")

    monkeypatch.setattr(
        rag_ep.DatabasePaths,
        "get_single_user_id",
        staticmethod(_raise_single_user_id),
    )

    kwargs = rag_ep._build_batch_pipeline_kwargs(
        request=UnifiedBatchRequest(
            queries=["q1"],
            user_id="single_user",
            feedback_user_id="single_user",
        ),
        db_paths=_EMPTY_DB_PATHS,
        current_user=None,
    )

    assert kwargs["user_id"] is None
    assert kwargs["feedback_user_id"] is None
    assert "query" not in kwargs

def test_rag_profile_applies_defaults_when_fields_omitted():
    request = UnifiedRAGRequest(query="profile apply", rag_profile="fast")
    kwargs = rag_ep._build_unified_pipeline_kwargs(
        request=request,
        db_paths=_EMPTY_DB_PATHS,
        media_db=None,  # type: ignore[arg-type]
        chacha_db=None,  # type: ignore[arg-type]
        current_user=None,
    )

    assert kwargs["generation_prompt"] == "instruction_tuned"
    assert kwargs["max_generation_tokens"] == 440


def test_explicit_request_fields_override_profile_defaults():
    request = UnifiedRAGRequest(
        query="profile override",
        rag_profile="accuracy",
        max_generation_tokens=700,
        generation_prompt="concise",
    )
    kwargs = rag_ep._build_unified_pipeline_kwargs(
        request=request,
        db_paths=_EMPTY_DB_PATHS,
        media_db=None,  # type: ignore[arg-type]
        chacha_db=None,  # type: ignore[arg-type]
        current_user=None,
    )

    assert kwargs["max_generation_tokens"] == 700
    assert kwargs["generation_prompt"] == "concise"
    assert kwargs["reranking_strategy"] == "two_tier"


def test_profile_defaults_override_search_agent_defaults_when_request_omits_field(monkeypatch):
    monkeypatch.setenv("SEARCH_STRUCTURED_RESPONSE", "false")

    request = UnifiedRAGRequest(query="precedence", rag_profile="balanced")
    kwargs = rag_ep._build_unified_pipeline_kwargs(
        request=request,
        db_paths=_EMPTY_DB_PATHS,
        media_db=None,  # type: ignore[arg-type]
        chacha_db=None,  # type: ignore[arg-type]
        current_user=None,
    )

    assert kwargs["enable_structured_response"] is True
