"""Behavior contracts for governed browser preflight analyzers."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping.preflight.context import (
    PreflightDeadlineExceeded,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import (
    BrowserProbeOptions,
    ProbeError,
    ProbeHttpRequest,
    ProbeHttpResponse,
    ProbeTimeout,
    ProbeUnavailable,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.utils.impersonate_target import (
    get_impersonate_target,
)

pytestmark = pytest.mark.unit

_URL = "https://example.com/path?token=secret"
_IDENTITY = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua-platform": '"Windows"',
}


class FakeHttpProbe:
    def __init__(
        self,
        response: ProbeHttpResponse | BaseException,
    ) -> None:
        self.response = response
        self.requests: list[ProbeHttpRequest] = []

    async def get(self, request: ProbeHttpRequest) -> ProbeHttpResponse:
        self.requests.append(request)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class FakeBrowserPage:
    """Protocol-level browser page fake with deterministic snapshots."""

    def __init__(
        self,
        *,
        contents: list[str] | None = None,
        evaluations: list[Any] | None = None,
        visible_links: tuple[bool, ...] = (),
        captured_urls: tuple[str, ...] = (),
        reload_request_batches: tuple[tuple[str, ...], ...] = (),
        errors: Mapping[str, BaseException] | None = None,
    ) -> None:
        self.contents = list(contents or ["<html></html>"])
        self.evaluations = list(evaluations or [])
        self.visible_links = visible_links
        self.request_urls = list(captured_urls)
        self.reload_request_batches = reload_request_batches
        self.errors = dict(errors or {})
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.content_calls = 0
        self.evaluate_calls = 0
        self.reload_calls = 0
        self.clear_calls = 0

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, dict(kwargs)))
        error = self.errors.get(name)
        if error is not None:
            raise error

    async def goto(self, url: str, *, wait_until: str, timeout_ms: float) -> None:
        self._record("goto", url, wait_until=wait_until, timeout_ms=timeout_ms)

    async def reload(self, *, wait_until: str, timeout_ms: float) -> None:
        self._record("reload", wait_until=wait_until, timeout_ms=timeout_ms)
        if self.reload_calls < len(self.reload_request_batches):
            self.request_urls.extend(self.reload_request_batches[self.reload_calls])
        self.reload_calls += 1

    async def wait_for_load_state(self, state: str, *, timeout_ms: float) -> None:
        self._record("wait_for_load_state", state, timeout_ms=timeout_ms)

    async def wait_for_timeout(self, timeout_ms: float) -> None:
        self._record("wait_for_timeout", timeout_ms)

    async def content(self) -> str:
        self._record("content")
        index = min(self.content_calls, len(self.contents) - 1)
        self.content_calls += 1
        return self.contents[index]

    async def evaluate(self, expression: str, argument: Any = None) -> Any:
        self._record("evaluate", expression, argument)
        if self.evaluate_calls >= len(self.evaluations):
            raise AssertionError("unexpected browser evaluation")
        result = self.evaluations[self.evaluate_calls]
        self.evaluate_calls += 1
        if isinstance(result, BaseException):
            raise result
        return result

    async def link_count(self) -> int:
        self._record("link_count")
        return len(self.visible_links)

    async def link_is_visible(self, index: int) -> bool:
        self._record("link_is_visible", index)
        return self.visible_links[index]

    def captured_request_urls(self) -> tuple[str, ...]:
        self._record("captured_request_urls")
        return tuple(self.request_urls)

    def clear_captured_request_urls(self) -> None:
        self._record("clear_captured_request_urls")
        self.request_urls.clear()
        self.clear_calls += 1


class FakeBrowserProbe:
    def __init__(self, pages: list[FakeBrowserPage | BaseException]) -> None:
        self.pages = list(pages)
        self.options: list[BrowserProbeOptions] = []
        self.exits = 0

    def open_page(self, options: BrowserProbeOptions) -> Any:
        index = len(self.options)
        self.options.append(options)

        @asynccontextmanager
        async def manager() -> Any:
            if index >= len(self.pages):
                raise AssertionError("unexpected browser page")
            page = self.pages[index]
            if isinstance(page, BaseException):
                raise page
            try:
                yield page
            finally:
                self.exits += 1

        return manager()


class FakeAnalyzerContext:
    def __init__(
        self,
        *,
        http: FakeHttpProbe,
        browser: FakeBrowserProbe,
        identity: Mapping[str, str],
    ) -> None:
        self.http = http
        self.browser = browser
        self.identity = dict(identity)
        self.identity_calls = 0

    def browser_identity(self) -> dict[str, str]:
        self.identity_calls += 1
        return dict(self.identity)


def fake_context(
    *,
    http_text: str = "<body>no js</body>",
    http_error: BaseException | None = None,
    browser_pages: list[FakeBrowserPage | BaseException] | None = None,
    identity: Mapping[str, str] = _IDENTITY,
) -> FakeAnalyzerContext:
    response: ProbeHttpResponse | BaseException
    if http_error is not None:
        response = http_error
    else:
        response = ProbeHttpResponse(
            url=_URL,
            status=200,
            headers={"content-type": "text/html"},
            text=http_text,
        )
    return FakeAnalyzerContext(
        http=FakeHttpProbe(response),
        browser=FakeBrowserProbe(list(browser_pages or [])),
        identity=identity,
    )


def _canonical_analyzers() -> tuple[Any, Any, Any, Any, Any]:
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.behavioral_detector import (
        _detect_honeypots,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.captcha_detector import (
        _detect_captcha,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.fingerprint_analyzer import (
        _analyze_fingerprinting,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.integrity_analyzer import (
        _analyze_function_integrity,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.js_detector import (
        _analyze_js_rendering,
    )

    return (
        _analyze_js_rendering,
        _detect_honeypots,
        _detect_captcha,
        _analyze_fingerprinting,
        _analyze_function_integrity,
    )


def test_private_analyzer_signatures_are_exact_async_contracts() -> None:
    js, honeypot, captcha, fingerprint, integrity = _canonical_analyzers()

    assert all(inspect.iscoroutinefunction(call) for call in (js, honeypot, captcha, fingerprint, integrity))
    assert str(inspect.signature(js)) == ("(url: 'str', context: 'PreflightExecutionContext') -> 'dict[str, Any]'")
    assert str(inspect.signature(honeypot)) == (
        "(url: 'str', context: 'PreflightExecutionContext', scan_depth: 'ScanDepth' = " "'default') -> 'dict[str, Any]'"
    )
    assert str(inspect.signature(captcha)) == ("(url: 'str', context: 'PreflightExecutionContext') -> 'dict[str, Any]'")
    assert str(inspect.signature(fingerprint)) == (
        "(url: 'str', context: 'PreflightExecutionContext') -> 'dict[str, Any]'"
    )
    assert str(inspect.signature(integrity)) == (
        "(url: 'str', context: 'PreflightExecutionContext') -> 'dict[str, Any]'"
    )


def test_visible_text_parser_preserves_historical_normalization() -> None:
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.js_detector import (
        _extract_visible_text,
    )

    assert _extract_visible_text("") == ""
    assert (
        _extract_visible_text(
            "<html><style>hidden</style><body>  first phrase\n second </body>" "<script>also hidden</script></html>"
        )
        == "first\nphrase\nsecond"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("without_js", "expected_difference", "js_required", "is_spa"),
    [
        (100, 0.0, False, False),
        (75, 25.0, False, False),
        (74, 26.0, True, False),
        (25, 75.0, True, False),
        (24, 76.0, True, True),
    ],
)
async def test_js_thresholds_preserve_existing_semantics(
    without_js: int,
    expected_difference: float,
    js_required: bool,
    is_spa: bool,
) -> None:
    js, _, _, _, _ = _canonical_analyzers()
    page = FakeBrowserPage(contents=[f"<body>{'x' * 100}</body>"])
    context = fake_context(
        http_text=f"<body>{'x' * without_js}</body>",
        browser_pages=[page],
    )

    assert await js(_URL, context) == {
        "status": "success",
        "js_required": js_required,
        "is_spa": is_spa,
        "content_difference_%": expected_difference,
    }


@pytest.mark.asyncio
async def test_js_zero_rendered_text_preserves_error_shape() -> None:
    js, _, _, _, _ = _canonical_analyzers()
    context = fake_context(browser_pages=[FakeBrowserPage(contents=["<script>x</script>"])])

    assert await js(_URL, context) == {
        "status": "error",
        "message": "Could not extract content from the page with JS enabled.",
    }


@pytest.mark.asyncio
async def test_js_reuses_identity_for_http_and_browser_and_preserves_options() -> None:
    js, _, _, _, _ = _canonical_analyzers()
    page = FakeBrowserPage(contents=["<body>rendered</body>"])
    context = fake_context(
        http_text="<body>rendered</body>",
        browser_pages=[page],
    )

    await js(_URL, context)

    assert context.identity_calls == 1
    request = context.http.requests[0]
    assert request.url == _URL
    assert dict(request.headers) == _IDENTITY
    assert request.timeout_s == 30
    assert request.impersonate == get_impersonate_target(_IDENTITY["User-Agent"])
    assert request.allow_redirects is True
    options = context.browser.options[0]
    assert dict(options.extra_headers) == _IDENTITY
    assert options.user_agent is None
    assert options.capture_requests is False
    assert options.block_resource_types == ()
    assert page.calls[:3] == [
        ("goto", (_URL,), {"wait_until": "load", "timeout_ms": 30_000}),
        ("wait_for_load_state", ("networkidle",), {"timeout_ms": 5_000}),
        ("wait_for_timeout", (2_000,), {}),
    ]


@pytest.mark.asyncio
async def test_js_navigation_is_best_effort() -> None:
    js, _, _, _, _ = _canonical_analyzers()
    page = FakeBrowserPage(
        contents=["<body>rendered</body>"],
        errors={"goto": RuntimeError("navigation secret token=abc")},
    )
    context = fake_context(
        http_text="<body>rendered</body>",
        browser_pages=[page],
    )

    assert await js(_URL, context) == {
        "status": "success",
        "js_required": False,
        "is_spa": False,
        "content_difference_%": 0.0,
    }
    assert [call[0] for call in page.calls] == [
        "goto",
        "wait_for_timeout",
        "content",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("analyzer_name", ["js", "captcha"])
async def test_governed_navigation_probe_error_is_propagated(
    analyzer_name: str,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.captcha_detector import (
        _detect_captcha_impl,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.js_detector import (
        _analyze_js_rendering_impl,
    )

    failure = ProbeError("policy_denied", "Probe destination was denied.")
    page = FakeBrowserPage(
        contents=["<body>rendered</body>"],
        errors={"goto": failure},
    )
    context = fake_context(
        http_text="<body>rendered</body>",
        browser_pages=[page],
    )
    analyzer = {
        "js": _analyze_js_rendering_impl,
        "captcha": _detect_captcha_impl,
    }[analyzer_name]

    with pytest.raises(ProbeError) as raised:
        await analyzer(_URL, context)

    assert raised.value is failure
    assert [call[0] for call in page.calls] == ["goto"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("total_links", "scan_depth", "expected_checked"),
    [
        (0, "default", 0),
        (1_000, "default", 250),
        (100, "thorough", 66),
        (7, "deep", 7),
    ],
)
async def test_honeypot_scan_depths_preserve_sampling(
    total_links: int,
    scan_depth: str,
    expected_checked: int,
) -> None:
    _, honeypot, _, _, _ = _canonical_analyzers()
    page = FakeBrowserPage(visible_links=(True,) * total_links)
    context = fake_context(browser_pages=[page])

    assert await honeypot(_URL, context, scan_depth=scan_depth) == {
        "status": "success",
        "total_links": total_links,
        "invisible_links": 0,
        "honeypot_detected": False,
        "links_checked": expected_checked,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("invisible", "detected"), [(3, False), (4, True)])
async def test_honeypot_threshold_is_strictly_greater_than_three(
    invisible: int,
    detected: bool,
) -> None:
    _, honeypot, _, _, _ = _canonical_analyzers()
    visible_links = (False,) * invisible + (True,) * (10 - invisible)
    page = FakeBrowserPage(visible_links=visible_links)
    context = fake_context(browser_pages=[page])

    assert await honeypot(_URL, context, scan_depth="deep") == {
        "status": "success",
        "total_links": 10,
        "invisible_links": invisible,
        "honeypot_detected": detected,
        "links_checked": 10,
    }


@pytest.mark.asyncio
async def test_honeypot_uses_cached_identity_and_historical_navigation() -> None:
    _, honeypot, _, _, _ = _canonical_analyzers()
    page = FakeBrowserPage(visible_links=())
    context = fake_context(browser_pages=[page])

    await honeypot(_URL, context, scan_depth=None)

    assert context.identity_calls == 1
    assert dict(context.browser.options[0].extra_headers) == _IDENTITY
    assert page.calls[0] == (
        "goto",
        (_URL,),
        {"wait_until": "domcontentloaded", "timeout_ms": 30_000},
    )


def test_captcha_parser_preserves_provider_order_and_case_insensitivity() -> None:
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.captcha_detector import (
        _scan_for_captcha_fingerprints,
    )

    assert (
        _scan_for_captcha_fingerprints(
            '<div class="H-CAPTCHA G-RECAPTCHA"></div>',
            ("HTTPS://CHALLENGES.CLOUDFLARE.COM/TURNSTILE/V0/API.JS",),
        )
        == "reCAPTCHA"
    )
    assert _scan_for_captcha_fingerprints("<body>none</body>", ()) is None


@pytest.mark.asyncio
async def test_captcha_detects_on_load_without_reloads() -> None:
    _, _, captcha, _, _ = _canonical_analyzers()
    page = FakeBrowserPage(
        contents=["<div class='g-recaptcha'></div>"],
        captured_urls=("HTTPS://UNRELATED.EXAMPLE/ASSET.JS",),
    )
    context = fake_context(browser_pages=[page])

    assert await captcha(_URL, context) == {
        "status": "success",
        "captcha_detected": True,
        "captcha_type": "reCAPTCHA",
        "trigger_condition": "on page load",
    }
    assert page.reload_calls == 0
    assert context.browser.options[0].capture_requests is True


@pytest.mark.asyncio
async def test_captcha_detects_after_ten_reloads_on_same_governed_page() -> None:
    _, _, captcha, _, _ = _canonical_analyzers()
    reload_batches = ((),) * 9 + (("HTTPS://HCAPTCHA.COM/1/API.JS?TOKEN=SECRET",),)
    page = FakeBrowserPage(
        contents=["<body>none</body>", "<body>still none</body>"],
        captured_urls=("HTTPS://INITIAL.EXAMPLE/ASSET.JS",),
        reload_request_batches=reload_batches,
    )
    context = fake_context(browser_pages=[page])

    assert await captcha(_URL, context) == {
        "status": "success",
        "captcha_detected": True,
        "captcha_type": "hCaptcha",
        "trigger_condition": "after burst of requests",
    }
    assert page.reload_calls == 10
    assert page.clear_calls == 1
    assert len(context.browser.options) == 1
    assert context.browser.exits == 1


@pytest.mark.asyncio
async def test_captcha_none_preserves_compact_success_shape() -> None:
    _, _, captcha, _, _ = _canonical_analyzers()
    page = FakeBrowserPage(contents=["<body>none</body>", "<body>none</body>"])
    context = fake_context(browser_pages=[page])

    assert await captcha(_URL, context) == {
        "status": "success",
        "captcha_detected": False,
    }
    assert page.reload_calls == 10


@pytest.mark.asyncio
async def test_captcha_navigation_is_best_effort() -> None:
    _, _, captcha, _, _ = _canonical_analyzers()
    page = FakeBrowserPage(
        contents=["<body>none</body>", "<body>none</body>"],
        errors={"goto": RuntimeError("navigation secret token=abc")},
    )
    context = fake_context(browser_pages=[page])

    assert await captcha(_URL, context) == {
        "status": "success",
        "captcha_detected": False,
    }
    assert page.reload_calls == 10
    assert page.clear_calls == 1


@pytest.mark.asyncio
async def test_fingerprinting_preserves_all_signals_and_page_options() -> None:
    _, _, _, fingerprint, _ = _canonical_analyzers()
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.fingerprint_analyzer import (
        JS_PROBE_SCRIPT,
    )

    page = FakeBrowserPage(
        evaluations=[
            {
                "canvas_patched": True,
                "found_globals": ["DataDome", "Kasada", "DataDome"],
            },
            ["scroll", "mousemove", "scroll"],
        ],
        captured_urls=(
            "https://client.perimeterx.net/main.js",
            "https://js.datadome.co/agent.js",
            "https://cdn.example/app.js",
        ),
    )
    context = fake_context(browser_pages=[page])

    assert await fingerprint(_URL, context) == {
        "status": "success",
        "message": "Analysis complete.",
        "detected_services": ["PerimeterX (HUMAN)", "DataDome", "Kasada"],
        "canvas_fingerprinting_signal": True,
        "behavioral_listeners_detected": ["mousemove", "scroll"],
    }
    options = context.browser.options[0]
    assert context.identity_calls == 1
    assert dict(options.extra_headers) == _IDENTITY
    assert options.block_resource_types == ("image", "font", "media")
    assert options.init_scripts == (JS_PROBE_SCRIPT,)
    assert options.capture_requests is True
    assert page.calls[0] == (
        "goto",
        (_URL,),
        {"wait_until": "load", "timeout_ms": 30_000},
    )
    assert page.calls[1] == ("wait_for_timeout", (3_000,), {})


@pytest.mark.asyncio
async def test_integrity_uses_separate_clean_and_target_pages_and_exact_messages() -> None:
    _, _, _, _, integrity = _canonical_analyzers()
    clean = FakeBrowserPage(
        evaluations=[
            {
                "window.fetch": "function fetch() { [native code] }",
                "Date.now": "function now() { [native code] }",
                "console.log": "function log() { [native code] }",
            }
        ]
    )
    target = FakeBrowserPage(
        evaluations=[
            {
                "window.fetch": "function wrappedFetch() {}",
                "Date.now": "function wrappedNow() {}",
                "console.log": "function log() { [native code] }",
            }
        ]
    )
    context = fake_context(browser_pages=[clean, target])

    assert await integrity(_URL, context) == {
        "status": "success",
        "message": "Analysis complete.",
        "modified_functions": {
            "window.fetch": "Indicator of network traffic monitoring.",
            "Date.now": "Indicator of timing/behavioral analysis.",
        },
    }
    assert len(context.browser.options) == 2
    assert context.identity_calls == 0
    assert all(options.block_resource_types == ("image", "font", "media") for options in context.browser.options)
    assert clean.calls[0] == (
        "goto",
        ("about:blank",),
        {"wait_until": "load", "timeout_ms": 30_000},
    )
    assert target.calls[0] == (
        "goto",
        (_URL,),
        {"wait_until": "load", "timeout_ms": 30_000},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (
            0,
            {"status": "error", "message": "Page load timed out.", "error_code": "timeout"},
        ),
        (
            1,
            {"status": "error", "message": "Page load timed out.", "error_code": "timeout"},
        ),
        (
            2,
            {"status": "error", "message": "Page load timed out.", "error_code": "timeout"},
        ),
        (
            3,
            {
                "status": "error",
                "message": "Page load timed out.",
                "detected_services": [],
                "canvas_fingerprinting_signal": False,
                "behavioral_listeners_detected": [],
            },
        ),
        (
            4,
            {
                "status": "error",
                "message": "Page load timed out.",
                "modified_functions": {},
            },
        ),
    ],
)
async def test_browser_analyzer_timeout_shapes_are_deliberately_preserved(
    index: int,
    expected: dict[str, Any],
) -> None:
    analyzers = _canonical_analyzers()
    context = fake_context(browser_pages=[ProbeTimeout()])

    call = analyzers[index]
    result = await call(_URL, context)

    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_code",
    [
        "unavailable",
        "missing_dependency",
        "browser_transport_disabled",
        "browser_transport_unattested",
        "browser_transport_config_invalid",
    ],
)
async def test_probe_unavailable_maps_to_stable_capability_result(error_code: str) -> None:
    analyzers = _canonical_analyzers()
    if error_code == "missing_dependency":
        expected_message = "Probe dependency is unavailable."
    elif error_code == "unavailable":
        expected_message = "Probe capability is unavailable."
    else:
        expected_message = "Safe browser transport is unavailable."

    for call in analyzers:
        context = fake_context(
            browser_pages=[ProbeUnavailable(error_code=error_code)],
        )
        assert await call(_URL, context) == {
            "status": "error",
            "message": expected_message,
            "error_code": error_code,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (0, {"status": "error", "message": "JavaScript rendering analysis failed."}),
        (1, {"status": "error", "message": "Honeypot detection failed."}),
        (2, {"status": "error", "message": "Captcha detection failed."}),
        (
            3,
            {
                "status": "error",
                "message": "Fingerprint analysis failed.",
                "error_code": "analyzer_error",
                "detected_services": [],
                "canvas_fingerprinting_signal": False,
                "behavioral_listeners_detected": [],
            },
        ),
        (
            4,
            {
                "status": "error",
                "message": "Function integrity analysis failed.",
                "error_code": "analyzer_error",
                "modified_functions": {},
            },
        ),
    ],
)
async def test_unknown_failures_are_sanitized_without_losing_legacy_shapes(
    index: int,
    expected: dict[str, Any],
) -> None:
    analyzers = _canonical_analyzers()
    secret = "https://user:password@example.com/path?api_key=abc123"
    context = fake_context(browser_pages=[RuntimeError(secret)])

    result = await analyzers[index](_URL, context)

    assert result == expected
    assert secret not in str(result)
    assert "api_key" not in str(result).lower()
    assert "password" not in str(result).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_factory",
    [asyncio.CancelledError, PreflightDeadlineExceeded],
)
async def test_cancellation_and_deadline_propagate_unchanged(
    failure_factory: type[BaseException],
) -> None:
    for call in _canonical_analyzers():
        failure = failure_factory()
        context = fake_context(browser_pages=[failure])
        with pytest.raises(failure_factory) as caught:
            await call(_URL, context)
        assert caught.value is failure
