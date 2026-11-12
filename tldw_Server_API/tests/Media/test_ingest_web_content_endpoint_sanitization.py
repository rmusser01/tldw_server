from typing import Any

import pytest
from fastapi import BackgroundTasks, HTTPException, Request

from tldw_Server_API.app.api.v1.endpoints.media import ingest_web_content as ingest_endpoint
from tldw_Server_API.app.api.v1.schemas.media_request_models import IngestWebContentRequest, ScrapeMethod

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.error_calls = []
        self.info_calls = []
        self.exception_calls = []

    def error(self, *args, **kwargs):
        self.error_calls.append((args, kwargs))

    def info(self, *args, **kwargs):
        self.info_calls.append((args, kwargs))

    def exception(self, *args, **kwargs):
        self.exception_calls.append((args, kwargs))


_SENSITIVE_MARKERS = (
    "ingest backend leaked",
    "/private/tmp/ingest-web-content.db",
)


def _assert_sanitized_error_log(logger_stub: _LoggerStub) -> None:
    assert logger_stub.exception_calls == []
    assert logger_stub.error_calls
    assert [args[0] for args, _kwargs in logger_stub.error_calls if args] == ["Web content ingestion failed"]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.error_calls)

    rendered_calls = repr(logger_stub.error_calls)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered_calls


async def test_ingest_web_content_sanitizes_orchestration_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    async def _raise_orchestration_failure(**_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("ingest backend leaked /private/tmp/ingest-web-content.db")

    monkeypatch.setattr(ingest_endpoint, "logger", logger_stub, raising=True)
    monkeypatch.setattr(
        ingest_endpoint,
        "ingest_web_content_orchestrate",
        _raise_orchestration_failure,
        raising=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        await ingest_endpoint.ingest_web_content(
            request=IngestWebContentRequest(urls=["https://example.com/"], perform_analysis=False),
            http_request=Request({"type": "http"}),
            background_tasks=BackgroundTasks(),
            token=object(),
            db=object(),
            usage_log=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to ingest web content"
    _assert_sanitized_error_log(logger_stub)


@pytest.mark.parametrize(
    "options",
    [
        {"urls": []},
        {"urls": ["https://example.com", "https://example.org"]},
        {"urls": ["ftp://example.com/a"]},
        {"urls": ["https://user:secret@example.com/a"]},
        {"scrape_method": ScrapeMethod.RECURSIVE},
        {"perform_analysis": True},
        {"perform_chunking": True},
        {"perform_translation": True},
        {"use_cookies": True},
        {"cookies": "[]"},
        {"auto_chunking_use_llm": True},
        {"perform_rolling_summarization": True},
        {"perform_confabulation_check_of_analysis": True},
        {"api_key": "secret"},
        {"overwrite_existing": True},
    ],
)
def test_credential_free_profile_rejects_conflicting_options(options):
    from pydantic import ValidationError

    values = {
        "urls": ["https://example.com/a"],
        "credential_free": True,
        "perform_analysis": False,
        "perform_chunking": False,
    }
    values.update(options)
    with pytest.raises(ValidationError):
        IngestWebContentRequest(**values)


@pytest.mark.parametrize(
    "article, expected_error",
    [
        ({"extraction_successful": False, "error": "policy_error"}, "policy_error"),
        ({"extraction_successful": False, "error": "secret" * 500}, "extraction_error"),
        ({"extraction_successful": True, "content": " \n "}, "empty_content"),
        ({"extraction_successful": True, "content": "a" * 1_000_001}, "content_too_large"),
    ],
)
async def test_credential_free_preview_preserves_bounded_failure(monkeypatch, article, expected_error):
    from types import SimpleNamespace

    from tldw_Server_API.app.services import web_scraping_service as service

    async def scrape(url, **kwargs):
        assert kwargs == {"custom_cookies": None, "allow_llm_extraction": False, "credential_free": True}
        return article

    monkeypatch.setattr(service, "scrape_article", scrape)
    result = await ingest_endpoint.ingest_web_content(
        request=IngestWebContentRequest(
            urls=["https://example.com/a"], credential_free=True, perform_analysis=False, perform_chunking=False
        ),
        http_request=Request({"type": "http"}),
        background_tasks=BackgroundTasks(),
        db=SimpleNamespace(),
        usage_log=SimpleNamespace(),
    )
    assert result["status"] == "error"
    assert result["results"] == [
        {"url": "https://example.com/a", "content": "", "extraction_successful": False, "error": expected_error}
    ]


async def test_credential_free_preview_trim_title_utc_usage_without_monitoring(monkeypatch):
    from datetime import datetime, timedelta
    from types import SimpleNamespace
    from unittest.mock import Mock

    from tldw_Server_API.app.core.Monitoring import topic_monitoring_service as monitoring
    from tldw_Server_API.app.services import web_scraping_service as service

    async def scrape(url, **kwargs):
        return {"url": url, "title": "Actual title", "content": "  readable text  ", "extraction_successful": True}

    usage = Mock()
    monitor = Mock(side_effect=AssertionError("preview triggered monitoring"))
    monkeypatch.setattr(service, "scrape_article", scrape)
    monkeypatch.setattr(monitoring, "get_topic_monitoring_service", monitor)
    result = await ingest_endpoint.ingest_web_content(
        request=IngestWebContentRequest(
            urls=["https://example.com/a"],
            credential_free=True,
            perform_analysis=False,
            perform_chunking=False,
            timestamp_option=False,
        ),
        http_request=Request({"type": "http"}),
        background_tasks=BackgroundTasks(),
        db=SimpleNamespace(),
        usage_log=usage,
    )
    assert result["status"] == "success"
    assert result["results"][0]["title"] == "Actual title"
    assert result["results"][0]["content"] == "readable text"
    assert datetime.fromisoformat(result["results"][0]["ingested_at"]).utcoffset() == timedelta(0)
    assert "analysis" not in result["results"][0]
    usage.log_event.assert_called_once()
    monitor.assert_not_called()


@pytest.mark.parametrize(
    "permissions, expected_owner, expected_status",
    [
        ([], "42", 403),
        (["media.create"], "99", 412),
        (["media.create"], "42", 200),
    ],
)
async def test_public_extraction_route_authorization_and_optional_legacy_token(
    monkeypatch, permissions, expected_owner, expected_status
):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from tldw_Server_API.app.api.v1.API_Deps import auth_deps
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

    app = FastAPI()
    app.include_router(ingest_endpoint.router)
    principal = AuthPrincipal(kind="user", user_id=42, permissions=permissions)
    app.dependency_overrides[auth_deps.get_auth_principal] = lambda: principal
    app.dependency_overrides[ingest_endpoint.get_request_user] = lambda: SimpleNamespace(id=42)
    app.dependency_overrides[ingest_endpoint.get_media_db_for_user] = lambda: object()
    app.dependency_overrides[ingest_endpoint.get_usage_event_logger] = lambda: object()
    app.dependency_overrides[ingest_endpoint.guard_backpressure_and_quota] = lambda: None
    # Leave real permission and expected-user checks; isolate infrastructure/quota stores.
    for dependency in ingest_endpoint.router.routes[0].dependant.dependencies:
        call = dependency.call
        if getattr(call, "_tldw_rate_limit_resource", None) or getattr(call, "_tldw_token_scope", False):
            app.dependency_overrides[call] = lambda: None
    calls = []

    async def orchestrate(**kwargs):
        calls.append(kwargs)
        return [{"url": "https://example.com/a", "content": "article", "extraction_successful": True}]

    monkeypatch.setattr(ingest_endpoint, "ingest_web_content_orchestrate", orchestrate)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/ingest-web-content",
            headers={"X-TLDW-Expected-User-ID": expected_owner},
            json={
                "urls": ["https://example.com/a"],
                "credential_free": True,
                "perform_analysis": False,
                "perform_chunking": False,
            },
        )
    assert response.status_code == expected_status, response.text
    assert len(calls) == (1 if expected_status == 200 else 0)
