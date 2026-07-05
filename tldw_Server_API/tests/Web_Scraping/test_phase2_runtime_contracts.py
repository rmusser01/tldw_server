from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from tldw_Server_API.app.core.Web_Scraping.runtime import (
    FetchRequest,
    FetchResponse,
    PolicyDecision,
    RuntimeRequestContext,
)


@pytest.mark.unit
def test_runtime_request_context_freezes_metadata() -> None:
    metadata = {"trace": {"id": "abc"}}
    context = RuntimeRequestContext(
        source="article_extract",
        stage="pre_fetch",
        user_id=123,
        request_id="req-1",
        metadata=metadata,
    )

    metadata["trace"]["id"] = "mutated"

    assert isinstance(context.metadata, MappingProxyType)
    assert context.metadata["trace"]["id"] == "abc"
    assert context.source == "article_extract"
    assert context.stage == "pre_fetch"
    assert context.user_id == "123"
    assert context.request_id == "req-1"


@pytest.mark.unit
def test_fetch_request_normalizes_fields_and_proxy_maps() -> None:
    headers = {"User-Agent": "UA"}
    cookies = {"session": "redacted"}
    proxies = {"https": "http://proxy.example:8080"}
    request = FetchRequest(
        url=" https://example.com/article ",
        method="get",
        headers=headers,
        cookies=cookies,
        timeout=15,
        backend="curl",
        allow_redirects=True,
        impersonate="chrome120",
        proxies=proxies,
    )

    headers["User-Agent"] = "mutated"
    cookies["session"] = "mutated"
    proxies["https"] = "mutated"

    assert request.url == "https://example.com/article"
    assert request.method == "GET"
    assert request.headers["User-Agent"] == "UA"
    assert request.cookies["session"] == "redacted"
    assert request.timeout == 15.0
    assert request.backend == "curl"
    assert request.allow_redirects is True
    assert request.impersonate == "chrome120"
    assert request.proxies["https"] == "http://proxy.example:8080"


@pytest.mark.unit
def test_fetch_request_rejects_missing_url() -> None:
    with pytest.raises(ValueError, match="url is required"):
        FetchRequest(url=" ")


@pytest.mark.unit
def test_fetch_response_normalizes_mapping_response() -> None:
    response = FetchResponse.from_raw(
        {
            "status": 200,
            "headers": {"Content-Type": "text/html"},
            "text": "<html>ok</html>",
            "url": "https://example.com/final",
            "backend": "curl",
        },
        fallback_url="https://example.com/article",
        fallback_backend="httpx",
        elapsed_seconds=0.25,
    )

    assert response.status == 200
    assert response.headers["Content-Type"] == "text/html"
    assert response.text == "<html>ok</html>"
    assert response.url == "https://example.com/final"
    assert response.backend == "curl"
    assert response.elapsed_seconds == 0.25


@pytest.mark.unit
def test_fetch_response_normalizes_object_response_status_code() -> None:
    raw = SimpleNamespace(
        status_code=204,
        headers={"X-Test": "true"},
        text="",
        url="https://example.com/no-content",
    )

    response = FetchResponse.from_raw(
        raw,
        fallback_url="https://example.com/article",
        fallback_backend="httpx",
    )

    assert response.status == 204
    assert response.headers["X-Test"] == "true"
    assert response.text == ""
    assert response.url == "https://example.com/no-content"
    assert response.backend == "httpx"


@pytest.mark.unit
def test_policy_decision_matches_legacy_policy_fields() -> None:
    decision = PolicyDecision(
        allowed=False,
        mode="strict",
        reason="robots_disallowed",
        stage="pre_fetch",
        source="article_extract",
        details={"sanitized": True},
    )

    assert decision.allowed is False
    assert decision.mode == "strict"
    assert decision.reason == "robots_disallowed"
    assert decision.stage == "pre_fetch"
    assert decision.source == "article_extract"
    assert decision.details["sanitized"] is True


@pytest.mark.unit
def test_runtime_package_does_not_import_legacy_wrappers_or_policy_modules() -> None:
    forbidden_roots = {
        "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
        "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
        "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs",
        "tldw_Server_API.app.core.Web_Scraping.outbound_policy",
        "tldw_Server_API.app.core.Security.egress",
    }
    runtime_dir = Path("tldw_Server_API/app/core/Web_Scraping/runtime")

    for path in runtime_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = {alias.name for alias in node.names}
                assert imports.isdisjoint(forbidden_roots), (path, imports & forbidden_roots)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in forbidden_roots, (path, node.module)
