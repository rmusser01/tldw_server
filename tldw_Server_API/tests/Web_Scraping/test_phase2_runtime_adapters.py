from __future__ import annotations

from types import SimpleNamespace

import pytest

import tldw_Server_API.app.core.Web_Scraping.runtime.fetch as runtime_fetch
from tldw_Server_API.app.core.Web_Scraping.runtime import FetchRequest
from tldw_Server_API.app.core.Web_Scraping.runtime.fetch import DefaultFetchClient


@pytest.mark.unit
def test_default_fetch_client_uses_simplified_get_path_without_method(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_fetch(url, **kwargs):
        calls["url"] = url
        calls["kwargs"] = kwargs
        return {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "text": "<html>ok</html>",
            "url": url,
            "backend": "curl",
        }

    monkeypatch.setattr(runtime_fetch, "http_fetch", fake_fetch)

    response = DefaultFetchClient().fetch(
        FetchRequest(
            url="https://example.com/article",
            headers={"User-Agent": "UA"},
            cookies={"session": "abc"},
            timeout=15.0,
            backend="curl",
            allow_redirects=True,
            impersonate="chrome120",
            proxies={"https": "http://proxy.example:8080"},
        )
    )

    assert calls["url"] == "https://example.com/article"
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert "method" not in kwargs
    assert kwargs["backend"] == "curl"
    assert kwargs["follow_redirects"] is True
    assert kwargs["impersonate"] == "chrome120"
    assert kwargs["timeout"] == 15.0
    assert kwargs["headers"] == {"User-Agent": "UA"}
    assert kwargs["cookies"] == {"session": "abc"}
    assert kwargs["proxies"] == {"https": "http://proxy.example:8080"}
    assert response.status == 200
    assert response.backend == "curl"


@pytest.mark.unit
def test_default_fetch_client_normalizes_object_like_response(monkeypatch) -> None:
    def fake_fetch(url, **kwargs):
        return SimpleNamespace(
            status_code=201,
            headers={"X-Test": "true"},
            text="<html>created</html>",
            url="https://example.com/final",
        )

    monkeypatch.setattr(runtime_fetch, "http_fetch", fake_fetch)

    response = DefaultFetchClient().fetch(
        FetchRequest(url="https://example.com/article", backend="httpx", timeout=15.0)
    )

    assert response.status == 201
    assert response.headers["X-Test"] == "true"
    assert response.text == "<html>created</html>"
    assert response.url == "https://example.com/final"
    assert response.backend == "httpx"


@pytest.mark.unit
def test_default_fetch_client_measures_elapsed_with_monotonic_clock(monkeypatch) -> None:
    times = iter([10.0, 12.5])

    def fake_fetch(url, **kwargs):
        return {
            "status": 200,
            "headers": {},
            "text": "<html>ok</html>",
            "url": url,
        }

    monkeypatch.setattr(runtime_fetch, "http_fetch", fake_fetch)
    monkeypatch.setattr(runtime_fetch.time, "monotonic", lambda: next(times))

    response = DefaultFetchClient().fetch(
        FetchRequest(url="https://example.com/article", backend="httpx", timeout=15.0)
    )

    assert response.elapsed_seconds == 2.5


@pytest.mark.unit
def test_default_fetch_client_rejects_non_get_method() -> None:
    with pytest.raises(ValueError, match="only supports GET"):
        DefaultFetchClient().fetch(FetchRequest(url="https://example.com/article", method="POST"))
