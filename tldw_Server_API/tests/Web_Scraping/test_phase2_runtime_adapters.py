from __future__ import annotations

from types import SimpleNamespace

import pytest

import tldw_Server_API.app.core.Web_Scraping.runtime.fetch as runtime_fetch
from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.Web_Scraping.runtime import FetchRequest
from tldw_Server_API.app.core.Web_Scraping.runtime.fetch import DefaultFetchClient


@pytest.mark.unit
def test_default_fetch_client_uses_simplified_get_path_without_method_for_curl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_fetch(url: str, **kwargs: object) -> dict[str, object]:
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
def test_default_fetch_client_uses_response_mode_for_httpx_security(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_fetch(*args: object, **kwargs: object) -> SimpleNamespace:
        calls["args"] = args
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            status_code=201,
            headers={"X-Test": "true"},
            text="<html>created</html>",
            url="https://example.com/final",
        )

    monkeypatch.setattr(runtime_fetch, "http_fetch", fake_fetch)

    response = DefaultFetchClient().fetch(
        FetchRequest(
            url="https://example.com/article",
            headers={"User-Agent": "UA"},
            cookies={"session": "abc"},
            backend="httpx",
            timeout=15.0,
            allow_redirects=True,
            proxies={"https": "http://proxy.example:8080"},
        )
    )

    assert calls["args"] == ()
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["method"] == "GET"
    assert kwargs["url"] == "https://example.com/article"
    assert kwargs["allow_redirects"] is True
    assert kwargs["headers"] == {"User-Agent": "UA"}
    assert kwargs["cookies"] == {"session": "abc"}
    assert kwargs["proxies"] == {"https": "http://proxy.example:8080"}
    assert "backend" not in kwargs
    assert "follow_redirects" not in kwargs
    assert response.status == 201
    assert response.headers["X-Test"] == "true"
    assert response.text == "<html>created</html>"
    assert response.url == "https://example.com/final"
    assert response.backend == "httpx"


@pytest.mark.unit
def test_default_fetch_client_measures_elapsed_with_monotonic_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([10.0, 12.5])

    def fake_fetch(url: str, **kwargs: object) -> dict[str, object]:
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
    with pytest.raises(BadRequestError, match="only supports GET"):
        DefaultFetchClient().fetch(
            FetchRequest(url="https://example.com/article", method="POST")
        )


@pytest.mark.unit
async def test_default_policy_checker_delegates_to_existing_outbound_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.Web_Scraping.policy.adapters as policy_adapters
    from tldw_Server_API.app.core.Web_Scraping.policy import DefaultWebOutboundPolicyChecker
    from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext

    calls: dict[str, object] = {}

    async def fake_decide_web_outbound_policy(url: str, **kwargs: object) -> SimpleNamespace:
        calls["url"] = url
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            allowed=False,
            mode="strict",
            reason="robots_disallowed",
            stage="pre_fetch",
            source="article_extract",
            details={"policy": "test"},
        )

    monkeypatch.setattr(
        policy_adapters,
        "decide_web_outbound_policy",
        fake_decide_web_outbound_policy,
    )

    decision = await DefaultWebOutboundPolicyChecker().decide(
        "https://example.com/article",
        respect_robots=True,
        user_agent="UA",
        context=RuntimeRequestContext(source="article_extract", stage="pre_fetch"),
        config={"web_scraper": {"web_outbound_policy_mode": "strict"}},
    )

    assert calls["url"] == "https://example.com/article"
    assert calls["kwargs"] == {
        "respect_robots": True,
        "user_agent": "UA",
        "source": "article_extract",
        "stage": "pre_fetch",
        "config": {"web_scraper": {"web_outbound_policy_mode": "strict"}},
    }
    assert decision.allowed is False
    assert decision.mode == "strict"
    assert decision.reason == "robots_disallowed"
    assert decision.stage == "pre_fetch"
    assert decision.source == "article_extract"
    assert decision.details == {"policy": "test"}


@pytest.mark.unit
async def test_default_policy_checker_defaults_context_source_and_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.Web_Scraping.policy.adapters as policy_adapters
    from tldw_Server_API.app.core.Web_Scraping.policy import DefaultWebOutboundPolicyChecker
    from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext

    async def fake_decide_web_outbound_policy(url: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            allowed=True,
            mode="compat",
            reason="allowed",
            stage=kwargs["stage"],
            source=kwargs["source"],
            details=None,
        )

    monkeypatch.setattr(
        policy_adapters,
        "decide_web_outbound_policy",
        fake_decide_web_outbound_policy,
    )

    decision = await DefaultWebOutboundPolicyChecker().decide(
        "https://example.com/article",
        respect_robots=False,
        user_agent=None,
        context=RuntimeRequestContext(),
        config=None,
    )

    assert decision.allowed is True
    assert decision.stage == "runtime"
    assert decision.source == "web_scraping"
