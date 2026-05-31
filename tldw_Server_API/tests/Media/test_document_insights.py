# Tests for Document Insights Endpoint
#
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.media import document_insights as insights_mod
from tldw_Server_API.app.api.v1.schemas.document_insights import GenerateInsightsRequest
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user


async def _allow_non_authz_dep() -> None:
    return None


app = FastAPI()
app.include_router(insights_mod.router, prefix="/api/v1/media")


def _install_route_dependency_overrides() -> None:
    for route in app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        for dep in getattr(dependant, "dependencies", []):
            call = getattr(dep, "call", None)
            if getattr(call, "_tldw_rate_limit_resource", None) is not None:
                app.dependency_overrides[call] = _allow_non_authz_dep


@pytest.fixture(autouse=True)
def reset_app_dependency_overrides():
    app.dependency_overrides.clear()
    _install_route_dependency_overrides()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_user():
    user = MagicMock()
    user.id = 1
    user.username = "testuser"
    return user


@pytest.fixture
def mock_db(tmp_path):
    db = MagicMock()
    db.get_media_by_id = MagicMock(return_value={"id": 1, "type": "pdf", "content": "Sample document content."})
    db.db_path_str = str(tmp_path / "test_media.db")
    return db


class _StubAdapter:
    """Stub adapter that always returns a preset payload for chat calls."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        """Initialize with a preset payload, defaulting to `{\"ok\": True}` when omitted."""
        self._payload: dict[str, Any] = payload if payload is not None else {"ok": True}

    def chat(self, _payload: dict[str, Any]) -> dict[str, Any]:
        """Accept a chat payload and return the preset payload configured on this stub."""
        return self._payload


class _ExplodingAdapter:
    def chat(self, _payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("insights llm exploded at /private/insights.key")


class _LoggerStub:
    def __init__(self) -> None:
        self.error_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.warning_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def error(self, *args: Any, **kwargs: Any) -> None:
        self.error_calls.append((args, kwargs))

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.warning_calls.append((args, kwargs))

    def info(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def debug(self, *_args: Any, **_kwargs: Any) -> None:
        return


@pytest.mark.asyncio
async def test_generate_document_insights_success(mock_user, mock_db):
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    insights_payload = {
        "insights": [
            {
                "category": "summary",
                "title": "Summary",
                "content": "Short summary of the document.",
            }
        ]
    }

    with (
        patch.object(insights_mod, "_get_adapter", return_value=_StubAdapter()),
        patch.object(insights_mod, "resolve_provider_api_key", return_value=("key", None)),
        patch.object(insights_mod, "provider_requires_api_key", return_value=False),
        patch.object(insights_mod, "_resolve_model", return_value="test-model"),
        patch.object(insights_mod, "extract_response_content", return_value=insights_payload),
        patch.object(insights_mod, "get_cached_response", return_value=None),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/media/1/insights")

    assert response.status_code == 200
    data = response.json()
    assert data["media_id"] == 1
    assert data["insights"][0]["category"] == "summary"
    assert data["model_used"] == "test-model"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_generate_document_insights_cached(mock_user, mock_db):
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    cached_payload = {
        "media_id": 1,
        "insights": [
            {
                "category": "summary",
                "title": "Cached summary",
                "content": "Cached content.",
            }
        ],
        "model_used": "cached-model",
        "cached": False,
    }

    with (
        patch.object(insights_mod, "get_cached_response", return_value=("etag", cached_payload)),
        patch.object(insights_mod, "_get_adapter", side_effect=AssertionError("LLM should not be called")),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/media/1/insights")

    assert response.status_code == 200
    data = response.json()
    assert data["cached"] is True
    assert data["insights"][0]["title"] == "Cached summary"

    app.dependency_overrides.clear()


def test_build_insights_cache_key_includes_scope_and_length(mock_db):
    request = GenerateInsightsRequest(max_content_length=1234)
    key = insights_mod._build_insights_cache_key(
        7,
        request,
        user_id="42",
        db_scope=mock_db.db_path_str,
        max_content_length=1234,
    )
    assert "user:42" in key
    assert f"db:{mock_db.db_path_str}" in key
    assert "maxlen:1234" in key


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_document_insights_parses_fenced_json_with_think(mock_user, mock_db):
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db

    fenced_payload = (
        "<think>analysis</think>\n"
        "```json\n"
        '{"insights":[{"category":"summary","title":"T","content":"C"}]}\n'
        "```"
    )

    with (
        patch.object(insights_mod, "_get_adapter", return_value=_StubAdapter()),
        patch.object(insights_mod, "resolve_provider_api_key", return_value=("key", None)),
        patch.object(insights_mod, "provider_requires_api_key", return_value=False),
        patch.object(insights_mod, "_resolve_model", return_value="test-model"),
        patch.object(insights_mod, "extract_response_content", return_value=fenced_payload),
        patch.object(insights_mod, "get_cached_response", return_value=None),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/media/1/insights")

    assert response.status_code == 200
    payload = response.json()
    assert payload["insights"][0]["category"] == "summary"
    assert payload["insights"][0]["title"] == "T"

    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_document_insights_sanitizes_missing_insights_list_warning(
    mock_user,
    mock_db,
    monkeypatch,
):
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    logger_stub = _LoggerStub()
    monkeypatch.setattr(insights_mod, "logger", logger_stub, raising=True)

    with (
        patch.object(insights_mod, "_get_adapter", return_value=_StubAdapter()),
        patch.object(insights_mod, "resolve_provider_api_key", return_value=("key", None)),
        patch.object(insights_mod, "provider_requires_api_key", return_value=False),
        patch.object(insights_mod, "_resolve_model", return_value="test-model"),
        patch.object(insights_mod, "extract_response_content", return_value={"unexpected": "shape"}),
        patch.object(insights_mod, "get_cached_response", return_value=None),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/media/1/insights")

    assert response.status_code == 200
    assert response.json()["insights"] == []
    assert [args[0] for args, _kwargs in logger_stub.warning_calls if args] == [
        "LLM response did not include an insights list"
    ]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.warning_calls)
    rendered_calls = repr(logger_stub.warning_calls)
    assert "media_id" not in rendered_calls
    assert "NoneType" not in rendered_calls

    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_document_insights_sanitizes_llm_service_error_log(
    mock_user,
    mock_db,
    monkeypatch,
):
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    logger_stub = _LoggerStub()
    monkeypatch.setattr(insights_mod, "logger", logger_stub, raising=True)

    with (
        patch.object(insights_mod, "_get_adapter", return_value=_ExplodingAdapter()),
        patch.object(insights_mod, "resolve_provider_api_key", return_value=("key", None)),
        patch.object(insights_mod, "provider_requires_api_key", return_value=False),
        patch.object(insights_mod, "_resolve_model", return_value="test-model"),
        patch.object(insights_mod, "get_cached_response", return_value=None),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/media/1/insights")

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to generate insights. LLM service error."
    assert [args[0] for args, _kwargs in logger_stub.error_calls if args] == ["LLM call failed for document insights"]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)
    rendered_calls = repr(logger_stub.error_calls)
    assert "insights llm exploded" not in rendered_calls
    assert "/private/insights.key" not in rendered_calls

    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_document_insights_sanitizes_db_fetch_error_log(
    mock_user,
    mock_db,
    monkeypatch,
):
    mock_db.get_media_by_id = MagicMock(side_effect=RuntimeError("insights db failed at /private/insights.db"))
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    logger_stub = _LoggerStub()
    monkeypatch.setattr(insights_mod, "logger", logger_stub, raising=True)

    with patch.object(insights_mod, "get_cached_response", return_value=None):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/media/1/insights")

    assert response.status_code == 500
    assert response.json()["detail"] == "Database error while fetching media item"
    assert [args[0] for args, _kwargs in logger_stub.error_calls if args] == ["Database error fetching media item"]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)
    rendered_calls = repr(logger_stub.error_calls)
    assert "insights db failed" not in rendered_calls
    assert "/private/insights.db" not in rendered_calls

    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_document_insights_invalid_json_returns_500(mock_user, mock_db, monkeypatch):
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    logger_stub = _LoggerStub()
    monkeypatch.setattr(insights_mod, "logger", logger_stub, raising=True)

    with (
        patch.object(insights_mod, "_get_adapter", return_value=_StubAdapter()),
        patch.object(insights_mod, "resolve_provider_api_key", return_value=("key", None)),
        patch.object(insights_mod, "provider_requires_api_key", return_value=False),
        patch.object(insights_mod, "_resolve_model", return_value="test-model"),
        patch.object(insights_mod, "extract_response_content", return_value="this is not json"),
        patch.object(insights_mod, "get_cached_response", return_value=None),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/media/1/insights")

    assert response.status_code == 500
    assert "Failed to parse insights" in response.text
    assert [args[0] for args, _kwargs in logger_stub.error_calls if args] == [
        "Failed to parse LLM response for document insights"
    ]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)
    rendered_calls = repr(logger_stub.error_calls)
    assert "this is not json" not in rendered_calls

    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_document_insights_sanitizes_configuration_errors(mock_user, mock_db, monkeypatch):
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    logger_stub = _LoggerStub()
    monkeypatch.setattr(insights_mod, "logger", logger_stub, raising=True)

    with (
        patch.object(insights_mod, "_get_adapter", return_value=_StubAdapter()),
        patch.object(insights_mod, "resolve_provider_api_key", return_value=("key", None)),
        patch.object(insights_mod, "provider_requires_api_key", return_value=False),
        patch.object(insights_mod, "_resolve_model", return_value=None),
        patch.object(insights_mod, "get_cached_response", return_value=None),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/media/1/insights")

    assert response.status_code == 503
    assert response.json()["detail"] == "LLM provider configuration error"
    assert [args[0] for args, _kwargs in logger_stub.error_calls if args] == [
        "LLM configuration error for insights"
    ]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)
    rendered_calls = repr(logger_stub.error_calls)
    assert "Model is required" not in rendered_calls

    app.dependency_overrides.clear()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_document_insights_sanitizes_missing_api_key(mock_user, mock_db, monkeypatch):
    app.dependency_overrides[get_request_user] = lambda: mock_user
    app.dependency_overrides[get_media_db_for_user] = lambda: mock_db
    logger_stub = _LoggerStub()
    monkeypatch.setattr(insights_mod, "logger", logger_stub, raising=True)

    with (
        patch.object(insights_mod, "resolve_provider_api_key", return_value=("", None)),
        patch.object(insights_mod, "provider_requires_api_key", return_value=True),
        patch.object(insights_mod, "get_cached_response", return_value=None),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/media/1/insights")

    assert response.status_code == 503
    assert response.json()["detail"] == "LLM provider configuration error"
    assert [args[0] for args, _kwargs in logger_stub.error_calls if args] == [
        "No API key available for configured provider"
    ]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)
    rendered_calls = repr(logger_stub.error_calls)
    assert "openai" not in rendered_calls

    app.dependency_overrides.clear()
