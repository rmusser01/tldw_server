from __future__ import annotations

import ast
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

import tldw_Server_API.app.core.Web_Scraping.runtime as runtime_pkg
from tldw_Server_API.app.core.Web_Scraping.runtime import (
    FetchRequest,
    FetchResponse,
    PolicyDecision,
    RuntimeRequestContext,
)


@pytest.mark.unit
def test_runtime_request_context_freezes_metadata() -> None:
    metadata = {"trace": {"id": "abc"}, "items": ["one", {"nested": ["two"]}]}
    context = RuntimeRequestContext(
        source="article_extract",
        stage="pre_fetch",
        user_id=123,
        request_id="req-1",
        metadata=metadata,
    )

    metadata["trace"]["id"] = "mutated"

    assert isinstance(context.metadata, MappingProxyType)
    assert isinstance(context.metadata["trace"], MappingProxyType)
    assert context.metadata["trace"]["id"] == "abc"
    with pytest.raises(TypeError):
        context.metadata["trace"]["id"] = "blocked"
    assert context.metadata["items"] == ("one", {"nested": ("two",)})
    assert isinstance(context.metadata["items"][1], MappingProxyType)
    with pytest.raises(TypeError):
        context.metadata["items"][1]["nested"] = "blocked"
    assert context.source == "article_extract"
    assert context.stage == "pre_fetch"
    assert context.user_id == "123"
    assert context.request_id == "req-1"
    with pytest.raises(TypeError):
        context.metadata["new"] = "blocked"


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
@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_fetch_request_normalizes_false_like_allow_redirects(value: str) -> None:
    request = FetchRequest(
        url="https://example.com/article",
        allow_redirects=value,
    )

    assert request.allow_redirects is False


@pytest.mark.unit
@pytest.mark.parametrize("value", ["sometimes", "2", "", object()])
def test_fetch_request_rejects_ambiguous_allow_redirects(value: object) -> None:
    with pytest.raises(ValueError, match="allow_redirects"):
        FetchRequest(
            url="https://example.com/article",
            allow_redirects=value,
        )


@pytest.mark.unit
def test_fetch_request_rejects_missing_url() -> None:
    with pytest.raises(ValueError, match="url is required"):
        FetchRequest(url=" ")


@pytest.mark.unit
def test_fetch_request_rejects_negative_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be non-negative"):
        FetchRequest(url="https://example.com/article", timeout=-1)


@pytest.mark.unit
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_fetch_request_rejects_non_finite_timeout(value: float) -> None:
    with pytest.raises(ValueError, match="timeout must be finite"):
        FetchRequest(url="https://example.com/article", timeout=value)


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, False])
def test_fetch_request_rejects_boolean_timeout(value: bool) -> None:
    with pytest.raises(ValueError, match="timeout must be a float or int, not a boolean"):
        FetchRequest(url="https://example.com/article", timeout=value)


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
        details={"sanitized": True, "checks": ["robots", {"stages": ["pre_fetch"]}]},
    )

    assert decision.allowed is False
    assert decision.mode == "strict"
    assert decision.reason == "robots_disallowed"
    assert decision.stage == "pre_fetch"
    assert decision.source == "article_extract"
    assert decision.details["sanitized"] is True
    assert decision.details["checks"] == ("robots", {"stages": ("pre_fetch",)})
    assert isinstance(decision.details["checks"][1], MappingProxyType)
    with pytest.raises(TypeError):
        decision.details["checks"][1]["stages"] = "blocked"
    with pytest.raises(TypeError):
        decision.details["new"] = "blocked"


@pytest.mark.unit
@pytest.mark.parametrize("value", ["false", "0", "no", "off"])
def test_policy_decision_normalizes_false_like_allowed(value: str) -> None:
    decision = PolicyDecision(
        allowed=value,
        mode="strict",
        reason="robots_disallowed",
        stage="pre_fetch",
        source="article_extract",
    )

    assert decision.allowed is False


@pytest.mark.unit
@pytest.mark.parametrize("value", ["sometimes", "2", "", object()])
def test_policy_decision_rejects_ambiguous_allowed(value: object) -> None:
    with pytest.raises(ValueError, match="allowed"):
        PolicyDecision(
            allowed=value,
            mode="strict",
            reason="robots_disallowed",
            stage="pre_fetch",
            source="article_extract",
        )


@pytest.mark.unit
def test_runtime_package_does_not_import_legacy_wrappers_or_policy_modules() -> None:
    forbidden_roots = {
        "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
        "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
        "tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs",
        "tldw_Server_API.app.core.Web_Scraping.outbound_policy",
        "tldw_Server_API.app.core.Security.egress",
        "playwright",
    }
    forbidden_package_aliases = {
        "tldw_Server_API.app.core.Web_Scraping": {
            "Article_Extractor_Lib",
            "enhanced_web_scraping",
            "WebSearch_APIs",
            "outbound_policy",
        },
        "tldw_Server_API.app.core.Security": {"egress"},
    }
    runtime_dir = Path(runtime_pkg.__file__).parent

    def is_forbidden_module(module: str) -> bool:
        return any(module == root or module.startswith(f"{root}.") for root in forbidden_roots)

    for path in runtime_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = {alias.name for alias in node.names}
                forbidden_imports = {module for module in imports if is_forbidden_module(module)}
                assert not forbidden_imports, (path, forbidden_imports)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not is_forbidden_module(node.module), (path, node.module)
                forbidden_aliases = forbidden_package_aliases.get(node.module)
                if forbidden_aliases:
                    imported_names = {alias.name for alias in node.names}
                    assert imported_names.isdisjoint(forbidden_aliases), (
                        path,
                        node.module,
                        imported_names & forbidden_aliases,
                    )


from tldw_Server_API.app.core.Web_Scraping.runtime import (
    BrowserLaunchOptions,
    RuntimeCookie,
    RuntimeSessionState,
    RuntimeTimeouts,
    is_cancellation,
)


@pytest.mark.unit
def test_runtime_session_state_freezes_cookies_and_headers() -> None:
    cookies = [RuntimeCookie(name="session", value="abc", domain="example.com")]
    headers = {"User-Agent": "UA"}
    state = RuntimeSessionState(cookies=cookies, headers=headers)

    headers["User-Agent"] = "mutated"

    assert state.cookies[0].name == "session"
    assert state.cookies[0].value == "abc"
    assert state.cookies[0].domain == "example.com"
    assert state.headers["User-Agent"] == "UA"


@pytest.mark.unit
def test_runtime_session_state_recursively_freezes_header_values() -> None:
    headers = {"X-Metadata": {"trace": ["one", {"nested": ["two"]}]}}
    state = RuntimeSessionState(headers=headers)

    headers["X-Metadata"]["trace"][1]["nested"].append("mutated")
    headers["X-Metadata"]["trace"].append("mutated")
    headers["X-Metadata"]["new"] = "mutated"

    assert isinstance(state.headers["X-Metadata"], MappingProxyType)
    assert state.headers["X-Metadata"]["trace"] == ("one", {"nested": ("two",)})
    assert isinstance(state.headers["X-Metadata"]["trace"][1], MappingProxyType)
    with pytest.raises(TypeError):
        state.headers["X-Metadata"]["trace"][1]["nested"] = "blocked"
    with pytest.raises(TypeError):
        state.headers["X-Metadata"]["new"] = "blocked"


@pytest.mark.unit
def test_runtime_timeout_contract_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="fetch_timeout_s must be non-negative"):
        RuntimeTimeouts(fetch_timeout_s=-1)


@pytest.mark.unit
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_runtime_timeout_contract_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="fetch_timeout_s must be finite"):
        RuntimeTimeouts(fetch_timeout_s=value)


@pytest.mark.unit
@pytest.mark.parametrize("field_name", ["fetch_timeout_s", "browser_timeout_s", "preflight_timeout_s"])
@pytest.mark.parametrize("value", [True, False])
def test_runtime_timeout_contract_rejects_boolean_values(field_name: str, value: bool) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be a float or int, not a boolean"):
        RuntimeTimeouts(**{field_name: value})


@pytest.mark.unit
def test_browser_launch_options_normalize_viewport() -> None:
    options = BrowserLaunchOptions(headless=True, viewport_width=1280, viewport_height=720)

    assert options.headless is True
    assert options.viewport == {"width": 1280, "height": 720}


@pytest.mark.unit
def test_browser_launch_options_coerce_viewport_dimensions() -> None:
    options = BrowserLaunchOptions(viewport_width="1280", viewport_height=720.0)

    assert options.viewport_width == 1280
    assert options.viewport_height == 720
    assert options.viewport == {"width": 1280, "height": 720}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "value", "expected"),
    [
        ("viewport_width", 1280, 1280),
        ("viewport_width", "1280", 1280),
        ("viewport_width", 1280.0, 1280),
        ("viewport_height", 720, 720),
        ("viewport_height", "720", 720),
        ("viewport_height", 720.0, 720),
    ],
)
def test_browser_launch_options_accept_integral_viewport_dimensions(
    field_name: str,
    value: int | float | str,
    expected: int,
) -> None:
    options = BrowserLaunchOptions(**{field_name: value})

    assert getattr(options, field_name) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("viewport_width", 0, "viewport_width must be >= 1"),
        ("viewport_width", -1, "viewport_width must be >= 1"),
        ("viewport_height", 0, "viewport_height must be >= 1"),
        ("viewport_height", -1, "viewport_height must be >= 1"),
    ],
)
def test_browser_launch_options_reject_invalid_viewport_dimensions(
    field_name: str,
    value: int,
    message: str,
) -> None:
    kwargs = {field_name: value}

    with pytest.raises(ValueError, match=message):
        BrowserLaunchOptions(**kwargs)


@pytest.mark.unit
@pytest.mark.parametrize("field_name", ["viewport_width", "viewport_height"])
@pytest.mark.parametrize("value", [720.5, True, float("inf"), float("nan"), "wide"])
def test_browser_launch_options_reject_non_integral_viewport_dimensions(
    field_name: str,
    value: bool | float | str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        BrowserLaunchOptions(**{field_name: value})


@pytest.mark.unit
def test_cancellation_helper_preserves_asyncio_cancelled_error() -> None:
    import asyncio

    assert is_cancellation(asyncio.CancelledError()) is True
    assert is_cancellation(RuntimeError("not cancelled")) is False
