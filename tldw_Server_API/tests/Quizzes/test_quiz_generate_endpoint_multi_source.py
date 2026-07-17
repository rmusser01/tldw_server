from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import quizzes as quizzes_endpoint
from tldw_Server_API.app.api.v1.schemas.quizzes import (
    QuizGenerateRequest,
    QuizGenerationProfileDefinition,
)
from tldw_Server_API.app.services.quiz_generator import QuizProvenanceValidationError

pytestmark = pytest.mark.unit


def test_generation_profiles_endpoint_is_rate_limited():
    route = next(
        route
        for route in quizzes_endpoint.router.routes
        if getattr(route, "path", None) == "/quizzes/generation-profiles"
    )

    assert "quizzes.read" in [
        getattr(dependency.call, "_tldw_rate_limit_resource", None)
        for dependency in route.dependant.dependencies
    ]


def test_generation_profiles_endpoint_has_explicit_return_type():
    assert quizzes_endpoint.list_quiz_generation_profiles.__annotations__["return"] == list[
        QuizGenerationProfileDefinition
    ]


def test_generation_profiles_endpoint_lists_available_best_of_five_profile():
    profiles = quizzes_endpoint.list_quiz_generation_profiles()

    assert any(
        profile.id == "best_of_five"
        and profile.status == "available"
        and profile.default_question_types == ["multiple_choice"]
        for profile in profiles
    )


def test_generation_profiles_endpoint_lists_available_assertion_reasoning_profile():
    profiles = quizzes_endpoint.list_quiz_generation_profiles()

    assert any(
        profile.id == "assertion_reasoning"
        and profile.status == "available"
        and profile.default_question_types == ["multiple_choice"]
        for profile in profiles
    )


@pytest.mark.asyncio
async def test_generate_quiz_legacy_media_id_maps_to_single_media_source(monkeypatch):
    captured: dict = {}

    async def fake_generate_quiz_from_sources(**kwargs):
        captured.update(kwargs)
        return {"quiz": {"id": 1}, "questions": []}

    monkeypatch.setattr(quizzes_endpoint, "generate_quiz_from_sources", fake_generate_quiz_from_sources)

    request = QuizGenerateRequest.model_validate({"media_id": 42, "num_questions": 5, "workspace_id": "ws-1"})
    await quizzes_endpoint.generate_quiz(request=request, db=Mock(), media_db=Mock())

    assert captured["sources"] == [{"source_type": "media", "source_id": "42"}]
    assert captured["workspace_id"] == "ws-1"


@pytest.mark.asyncio
async def test_generate_quiz_forwards_sources_array(monkeypatch):
    captured: dict = {}

    async def fake_generate_quiz_from_sources(**kwargs):
        captured.update(kwargs)
        return {"quiz": {"id": 1}, "questions": []}

    monkeypatch.setattr(quizzes_endpoint, "generate_quiz_from_sources", fake_generate_quiz_from_sources)

    request = QuizGenerateRequest.model_validate(
        {
            "num_questions": 6,
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "workspace_id": "ws-2",
        }
    )
    await quizzes_endpoint.generate_quiz(request=request, db=Mock(), media_db=Mock())

    assert captured["sources"] == [{"source_type": "note", "source_id": "note-1"}]
    assert captured["workspace_id"] == "ws-2"


@pytest.mark.asyncio
async def test_generate_quiz_forwards_generation_profile(monkeypatch):
    captured: dict = {}

    async def fake_generate_quiz_from_sources(**kwargs):
        captured.update(kwargs)
        return {"quiz": {"id": 1}, "questions": []}

    monkeypatch.setattr(quizzes_endpoint, "generate_quiz_from_sources", fake_generate_quiz_from_sources)

    request = QuizGenerateRequest.model_validate(
        {
            "num_questions": 5,
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "generation_profile": "best_of_five",
        }
    )
    await quizzes_endpoint.generate_quiz(request=request, db=Mock(), media_db=Mock())

    assert captured["generation_profile"] == "best_of_five"


@pytest.mark.asyncio
async def test_generate_quiz_forwards_model_and_api_provider(monkeypatch):
    captured: dict = {}

    async def fake_generate_quiz_from_sources(**kwargs):
        captured.update(kwargs)
        return {"quiz": {"id": 1}, "questions": []}

    monkeypatch.setattr(quizzes_endpoint, "generate_quiz_from_sources", fake_generate_quiz_from_sources)

    request = QuizGenerateRequest.model_validate(
        {
            "media_id": 42,
            "num_questions": 3,
            "model": "gpt-4o-mini",
            "api_provider": "openai",
        }
    )
    await quizzes_endpoint.generate_quiz(request=request, db=Mock(), media_db=Mock())

    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_provider"] == "openai"


@pytest.mark.asyncio
async def test_generate_quiz_forwards_claims_verification_provider_override(monkeypatch):
    captured: dict = {}

    async def fake_generate_quiz_from_sources(**kwargs):
        captured.update(kwargs)
        return {"quiz": {"id": 1}, "questions": [], "claim_verification": {"verdict": "grounded"}}

    monkeypatch.setattr(quizzes_endpoint, "generate_quiz_from_sources", fake_generate_quiz_from_sources)

    request = QuizGenerateRequest.model_validate(
        {
            "media_id": 42,
            "num_questions": 3,
            "model": "generation-model",
            "api_provider": "llamacpp",
            "claims_verification_provider": "openrouter",
            "claims_verification_model": "claims-model",
        }
    )
    await quizzes_endpoint.generate_quiz(request=request, db=Mock(), media_db=Mock())

    assert captured["api_provider"] == "llamacpp"
    assert captured["model"] == "generation-model"
    assert captured["claims_verification_provider"] == "openrouter"
    assert captured["claims_verification_model"] == "claims-model"


@pytest.mark.asyncio
async def test_generate_quiz_maps_provenance_validation_error_to_422(monkeypatch):
    async def fake_generate_quiz_from_sources(**kwargs):
        raise QuizProvenanceValidationError("invalid source citations")

    monkeypatch.setattr(quizzes_endpoint, "generate_quiz_from_sources", fake_generate_quiz_from_sources)

    request = QuizGenerateRequest.model_validate(
        {
            "num_questions": 6,
            "sources": [{"source_type": "note", "source_id": "note-1"}],
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await quizzes_endpoint.generate_quiz(request=request, db=Mock(), media_db=Mock())

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_generate_quiz_rejects_unknown_workspace_before_generation(monkeypatch):
    async def fake_generate_quiz_from_sources(**kwargs):
        raise AssertionError("generation should not run when workspace is invalid")

    monkeypatch.setattr(quizzes_endpoint, "generate_quiz_from_sources", fake_generate_quiz_from_sources)

    db = Mock()
    db.get_workspace.return_value = None
    request = QuizGenerateRequest.model_validate(
        {
            "num_questions": 4,
            "sources": [{"source_type": "note", "source_id": "note-1"}],
            "workspace_id": "missing-ws",
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        await quizzes_endpoint.generate_quiz(request=request, db=db, media_db=Mock())

    assert exc_info.value.status_code == 404
