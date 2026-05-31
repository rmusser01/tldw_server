from __future__ import annotations

import pytest
from fastapi import HTTPException

import tldw_Server_API.app.api.v1.endpoints.web_scraping as web_scraping_endpoints


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message, *args, **kwargs) -> None:
        self.errors.append(message.format(*args) if args else message)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_scraping_job_status_preserves_http_exception(monkeypatch):
    class _Service:
        async def get_job_status(self, job_id, current_user):
            raise HTTPException(status_code=404, detail="missing")

    monkeypatch.setattr(
        web_scraping_endpoints,
        "get_web_scraping_service",
        lambda: _Service(),
        raising=True,
    )

    with pytest.raises(HTTPException) as excinfo:
        await web_scraping_endpoints.get_scraping_job_status(
            "job-1",
            current_user=object(),
        )

    assert excinfo.value.status_code == 404
    assert excinfo.value.detail == "missing"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancel_scraping_job_preserves_http_exception(monkeypatch):
    class _Service:
        async def cancel_job(self, job_id, current_user):
            raise HTTPException(status_code=403, detail="forbidden")

    monkeypatch.setattr(
        web_scraping_endpoints,
        "get_web_scraping_service",
        lambda: _Service(),
        raising=True,
    )

    with pytest.raises(HTTPException) as excinfo:
        await web_scraping_endpoints.cancel_scraping_job(
            "job-2",
            current_user=object(),
        )

    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "forbidden"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("handler_factory", "expected_detail", "expected_log"),
    [
        (
            lambda user: web_scraping_endpoints.get_scraping_service_status(current_user=user),
            "Failed to get scraping service status",
            "Failed to get scraping service status",
        ),
        (
            lambda user: web_scraping_endpoints.get_scraping_job_status("job-3", current_user=user),
            "Failed to get scraping job status",
            "Failed to get scraping job status",
        ),
        (
            lambda user: web_scraping_endpoints.cancel_scraping_job("job-3", current_user=user),
            "Failed to cancel scraping job",
            "Failed to cancel scraping job",
        ),
        (
            lambda user: web_scraping_endpoints.initialize_scraping_service(current_user=user),
            "Failed to initialize scraping service",
            "Failed to initialize scraping service",
        ),
        (
            lambda user: web_scraping_endpoints.shutdown_scraping_service(current_user=user),
            "Failed to shutdown scraping service",
            "Failed to shutdown scraping service",
        ),
        (
            lambda user: web_scraping_endpoints.get_scraping_progress("task-1", current_user=user),
            "Failed to get scraping progress",
            "Failed to get scraping progress",
        ),
        (
            lambda user: web_scraping_endpoints.get_cookies_for_domain("example.com", current_user=user),
            "Failed to get cookies for domain",
            "Failed to get cookies for domain",
        ),
        (
            lambda user: web_scraping_endpoints.set_cookies_for_domain(
                "example.com",
                [{"name": "sid", "value": "123"}],
                current_user=user,
            ),
            "Failed to set cookies for domain",
            "Failed to set cookies for domain",
        ),
        (
            lambda user: web_scraping_endpoints.check_url_duplicate(
                "https://example.com/article",
                current_user=user,
            ),
            "Failed to check URL duplicate",
            "Failed to check URL duplicate",
        ),
    ],
    ids=[
        "status",
        "job_status",
        "cancel_job",
        "initialize",
        "shutdown",
        "progress",
        "get_cookies",
        "set_cookies",
        "duplicate",
    ],
)
@pytest.mark.asyncio
async def test_web_scraping_management_handlers_sanitize_non_http_exceptions(
    monkeypatch,
    handler_factory,
    expected_detail,
    expected_log,
):
    def _raise_service_error():
        raise RuntimeError("web scraping backend exploded at /private/web-scraping.db")

    logger = _LoggerStub()
    monkeypatch.setattr(web_scraping_endpoints, "logger", logger, raising=True)

    monkeypatch.setattr(
        web_scraping_endpoints,
        "get_web_scraping_service",
        _raise_service_error,
        raising=True,
    )

    with pytest.raises(HTTPException) as excinfo:
        await handler_factory(object())

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == expected_detail
    assert logger.errors == [expected_log]
    error_text = "\n".join(logger.errors)
    assert "web scraping backend exploded" not in error_text
    assert "/private/web-scraping.db" not in error_text
