"""Behavior contracts for governed nonbrowser preflight analyzers."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from typing import Any

import pytest

from tldw_Server_API.app.core.Web_Scraping.preflight.context import (
    PreflightDeadlineExceeded,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.probes import (
    ExternalToolResult,
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

_URL = "https://example.com/path"
_IDENTITY = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua-platform": '"Windows"',
}


class RecordingHttpProbe:
    """Queue-backed HTTP probe that also records burst concurrency."""

    def __init__(self, responses: list[ProbeHttpResponse | BaseException]) -> None:
        self.responses = list(responses)
        self.requests: list[ProbeHttpRequest] = []
        self.active_requests = 0
        self.max_active_requests = 0

    async def get(self, request: ProbeHttpRequest) -> ProbeHttpResponse:
        index = len(self.requests)
        self.requests.append(request)
        self.active_requests += 1
        self.max_active_requests = max(self.max_active_requests, self.active_requests)
        try:
            await asyncio.sleep(0)
            if index >= len(self.responses):
                raise AssertionError("unexpected HTTP probe")
            response = self.responses[index]
            if isinstance(response, BaseException):
                raise response
            return response
        finally:
            self.active_requests -= 1


class RecordingExternalToolProbe:
    def __init__(self, results: list[ExternalToolResult | BaseException]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def run_waf(
        self,
        url: str,
        *,
        find_all: bool,
        enabled: bool | None,
    ) -> ExternalToolResult:
        index = len(self.calls)
        self.calls.append({"url": url, "find_all": find_all, "enabled": enabled})
        await asyncio.sleep(0)
        if index >= len(self.results):
            raise AssertionError("unexpected external-tool probe")
        result = self.results[index]
        if isinstance(result, BaseException):
            raise result
        return result


class RecordingControls:
    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.sleep_delays: list[float] = []

    async def sleep(self, delay_s: float) -> None:
        self.sleep_delays.append(delay_s)
        if self.error is not None:
            raise self.error
        await asyncio.sleep(0)


class FakeAnalyzerContext:
    def __init__(
        self,
        *,
        http: RecordingHttpProbe,
        external_tools: RecordingExternalToolProbe,
        controls: RecordingControls,
        identity: Mapping[str, str],
    ) -> None:
        self.http = http
        self.external_tools = external_tools
        self.controls = controls
        self.identity = dict(identity)
        self.identity_calls = 0

    def browser_identity(self) -> dict[str, str]:
        self.identity_calls += 1
        return dict(self.identity)


def _response(status: int, *, text: str = "", headers: Mapping[str, str] | None = None) -> ProbeHttpResponse:
    return ProbeHttpResponse(
        url=_URL,
        status=status,
        headers=headers or {},
        text=text,
    )


def fake_context(
    *,
    http_responses: list[ProbeHttpResponse | BaseException] | None = None,
    http_statuses: list[int] | None = None,
    external_results: list[ExternalToolResult | BaseException] | None = None,
    controls_error: BaseException | None = None,
    identity: Mapping[str, str] = _IDENTITY,
) -> FakeAnalyzerContext:
    responses = list(http_responses or [])
    responses.extend(_response(status) for status in (http_statuses or []))
    return FakeAnalyzerContext(
        http=RecordingHttpProbe(responses),
        external_tools=RecordingExternalToolProbe(list(external_results or [])),
        controls=RecordingControls(controls_error),
        identity=identity,
    )


def _canonical_analyzers() -> tuple[Any, Any, Any, Any]:
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.rate_limit_profiler import (
        _profile_rate_limits,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.robots_checker import (
        _check_robots_txt,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.tls_analyzer import (
        _analyze_tls_fingerprint,
    )
    from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.waf_detector import (
        _detect_waf,
    )

    return (
        _check_robots_txt,
        _analyze_tls_fingerprint,
        _profile_rate_limits,
        _detect_waf,
    )


@pytest.mark.asyncio
async def test_private_analyzer_signatures_are_exact_async_contracts() -> None:
    robots, tls, rate, waf = _canonical_analyzers()

    assert all(inspect.iscoroutinefunction(call) for call in (robots, tls, rate, waf))
    assert str(inspect.signature(robots)) == ("(url: 'str', context: 'PreflightExecutionContext') -> 'dict[str, Any]'")
    assert str(inspect.signature(tls)) == ("(url: 'str', context: 'PreflightExecutionContext') -> 'dict[str, Any]'")
    assert str(inspect.signature(rate)) == (
        "(url: 'str', context: 'PreflightExecutionContext', crawl_delay: 'float | None', "
        "impersonate: 'bool' = False) -> 'dict[str, Any]'"
    )
    assert str(inspect.signature(waf)) == (
        "(url: 'str', context: 'PreflightExecutionContext', find_all: 'bool' = False, "
        "external_tools_enabled: 'bool | None' = None) -> 'dict[str, Any]'"
    )


@pytest.mark.asyncio
async def test_robots_parser_preserves_success_shape_and_generic_agent_scope() -> None:
    robots, _, _, _ = _canonical_analyzers()
    context = fake_context(
        http_responses=[
            ProbeHttpResponse(
                url="https://example.com/robots.txt",
                status=200,
                headers={"Content-Type": "text/plain"},
                text=(
                    "# ignored\n"
                    "User-agent: named-bot\n"
                    "Disallow: /\n"
                    "User-agent: *\n"
                    "Crawl-delay: invalid\n"
                    "Crawl-delay: 2.5\n"
                    "Disallow: /\n"
                ),
            )
        ]
    )

    assert await robots(_URL, context) == {
        "status": "success",
        "crawl_delay": 2.5,
        "scraping_disallowed": True,
    }


@pytest.mark.asyncio
async def test_robots_success_without_generic_directives_is_unrestricted() -> None:
    robots, _, _, _ = _canonical_analyzers()
    context = fake_context(
        http_responses=[
            ProbeHttpResponse(
                url="https://example.com/robots.txt",
                status=200,
                headers={"Content-Type": "text/plain"},
                text="User-agent: named-bot\nDisallow: /\n",
            )
        ]
    )

    assert await robots(_URL, context) == {
        "status": "success",
        "crawl_delay": None,
        "scraping_disallowed": False,
    }


@pytest.mark.asyncio
async def test_robots_request_uses_origin_robots_path_and_legacy_headers() -> None:
    robots, _, _, _ = _canonical_analyzers()
    context = fake_context(http_statuses=[404])

    assert await robots("https://example.com:8443/a/b?query=1#fragment", context) == {"status": "not_found"}
    request = context.http.requests[0]
    assert request.url == "https://example.com:8443/robots.txt"
    assert dict(request.headers) == {"User-Agent": "Mozilla/5.0 (compatible; caniscrape-bot/1.0)"}
    assert request.timeout_s == 10
    assert request.impersonate is None
    assert request.allow_redirects is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        ProbeHttpResponse(
            url="https://example.com/robots.txt",
            status=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html></html>",
        ),
        ProbeHttpResponse(
            url="https://example.com/robots.txt",
            status=200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html></html>",
        ),
        _response(400),
        _response(499),
    ],
)
async def test_robots_not_found_branches_are_preserved(response: ProbeHttpResponse) -> None:
    robots, _, _, _ = _canonical_analyzers()
    context = fake_context(http_responses=[response])

    assert await robots(_URL, context) == {"status": "not_found"}


@pytest.mark.asyncio
async def test_robots_non_client_status_preserves_deliberate_error_shape() -> None:
    robots, _, _, _ = _canonical_analyzers()
    context = fake_context(http_statuses=[503])

    assert await robots(_URL, context) == {"status": "error", "message": "503"}


@pytest.mark.asyncio
async def test_tls_active_shape_uses_standard_and_impersonated_probe() -> None:
    _, tls, _, _ = _canonical_analyzers()
    context = fake_context(http_statuses=[403, 200])

    assert await tls(_URL, context) == {
        "status": "active",
        "details": "Site blocks standard Python clients but allows browser-like clients.",
    }
    assert len(context.http.requests) == 2
    standard, impersonated = context.http.requests
    assert dict(standard.headers) == _IDENTITY
    assert dict(impersonated.headers) == _IDENTITY
    assert standard.headers is not impersonated.headers
    assert standard.impersonate is None
    assert impersonated.impersonate == get_impersonate_target(_IDENTITY["User-Agent"])
    assert impersonated.impersonate == "chrome131"
    assert standard.timeout_s == impersonated.timeout_s == 20
    assert context.identity_calls == 1


@pytest.mark.asyncio
async def test_tls_inactive_shape_is_preserved() -> None:
    _, tls, _, _ = _canonical_analyzers()
    context = fake_context(http_statuses=[200, 200])

    assert await tls(_URL, context) == {
        "status": "inactive",
        "details": "Site does not appear to block based on TLS fingerprint.",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses", [[403, 503], [200, 403]])
async def test_tls_inconclusive_shapes_are_preserved(statuses: list[int]) -> None:
    _, tls, _, _ = _canonical_analyzers()
    context = fake_context(http_statuses=statuses)

    assert await tls(_URL, context) == {
        "status": "inconclusive",
        "details": "Could not determine fingerprinting status; site may be blocking all requests.",
    }


@pytest.mark.asyncio
async def test_tls_missing_impersonation_dependency_preserves_public_result() -> None:
    _, tls, _, _ = _canonical_analyzers()
    context = fake_context(
        http_responses=[
            _response(200),
            ProbeUnavailable(error_code="missing_dependency"),
        ]
    )

    assert await tls(_URL, context) == {
        "status": "error",
        "message": "curl-cffi is required for TLS impersonation.",
        "error_code": "missing_dependency",
    }


@pytest.mark.asyncio
async def test_rate_limit_default_delay_runs_four_gentle_then_eight_concurrent() -> None:
    _, _, rate, _ = _canonical_analyzers()
    context = fake_context(http_statuses=[200] * 12)

    assert await rate(_URL, context, None) == {
        "status": "success",
        "results": {
            "requests_sent": 12,
            "blocking_code": None,
            "details": "No blocking detected after 12 requests.",
        },
    }
    assert context.controls.sleep_delays == [3.0, 3.0, 3.0]
    assert len(context.http.requests) == 12
    assert context.http.max_active_requests == 8
    assert all(request.impersonate is None for request in context.http.requests)
    assert all(dict(request.headers) == _IDENTITY for request in context.http.requests)
    assert all(request.timeout_s == 15 for request in context.http.requests)
    assert all(request.allow_redirects is True for request in context.http.requests)
    assert context.identity_calls == 1


@pytest.mark.asyncio
async def test_rate_limit_crawl_delay_and_optional_impersonation_apply_to_all_requests() -> None:
    _, _, rate, _ = _canonical_analyzers()
    context = fake_context(http_statuses=[200] * 12)

    result = await rate(_URL, context, 1.25, impersonate=True)

    assert result["status"] == "success"
    assert result["results"]["requests_sent"] == 12
    assert context.controls.sleep_delays == [1.25, 1.25, 1.25]
    assert {request.impersonate for request in context.http.requests} == {"chrome131"}
    assert context.identity_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("blocking_code", [401, 403, 429, 503])
async def test_rate_limit_gentle_probe_stops_at_each_blocking_code(
    blocking_code: int,
) -> None:
    _, _, rate, _ = _canonical_analyzers()
    context = fake_context(http_statuses=[200, 200, blocking_code])

    assert await rate(_URL, context, 2.0) == {
        "status": "success",
        "results": {
            "requests_sent": 3,
            "blocking_code": blocking_code,
            "details": "Blocked after 3 requests with a 2.0s delay.",
        },
    }
    assert len(context.http.requests) == 3
    assert context.controls.sleep_delays == [2.0, 2.0]
    assert context.http.max_active_requests == 1


@pytest.mark.asyncio
async def test_rate_limit_burst_reports_first_blocking_status_in_response_order() -> None:
    _, _, rate, _ = _canonical_analyzers()
    context = fake_context(http_statuses=[200] * 5 + [503] + [200] * 6)

    assert await rate(_URL, context, 0.0) == {
        "status": "success",
        "results": {
            "requests_sent": 12,
            "blocking_code": 503,
            "details": "Blocked during a concurrent burst of 8 requests.",
        },
    }
    assert context.controls.sleep_delays == [0.0, 0.0, 0.0]
    assert context.http.max_active_requests == 8


@pytest.mark.asyncio
async def test_rate_limit_failed_burst_retires_siblings_before_returning() -> None:
    _, _, rate, _ = _canonical_analyzers()
    failure = ProbeError("probe_error", "HTTP probe failed.")

    class BlockingBurstHttpProbe:
        def __init__(self) -> None:
            self.requests: list[ProbeHttpRequest] = []
            self.all_started = asyncio.Event()
            self.release_siblings = asyncio.Event()
            self.all_finished = asyncio.Event()
            self.finished: set[int] = set()
            self.cancelled: set[int] = set()

        async def get(self, request: ProbeHttpRequest) -> ProbeHttpResponse:
            request_index = len(self.requests)
            self.requests.append(request)
            if request_index < 4:
                return _response(200)

            burst_index = request_index - 4
            if len(self.requests) == 12:
                self.all_started.set()
            await self.all_started.wait()

            try:
                if burst_index == 0:
                    raise failure
                await self.release_siblings.wait()
                return _response(200)
            except asyncio.CancelledError:
                self.cancelled.add(burst_index)
                raise
            finally:
                self.finished.add(burst_index)
                if len(self.finished) == 8:
                    self.all_finished.set()

    http = BlockingBurstHttpProbe()
    context = FakeAnalyzerContext(
        http=http,  # type: ignore[arg-type]
        external_tools=RecordingExternalToolProbe([]),
        controls=RecordingControls(),
        identity=_IDENTITY,
    )
    profile_task = asyncio.create_task(rate(_URL, context, 0.0))

    try:
        await asyncio.wait_for(http.all_started.wait(), timeout=1.0)
        result = await asyncio.wait_for(profile_task, timeout=1.0)
        all_finished_before_return = http.all_finished.is_set()
    finally:
        http.release_siblings.set()
        if not profile_task.done():
            profile_task.cancel()
            await asyncio.gather(profile_task, return_exceptions=True)
        await asyncio.wait_for(http.all_finished.wait(), timeout=1.0)

    assert all_finished_before_return
    assert http.cancelled == set(range(1, 8))
    assert result == {
        "status": "error",
        "message": "HTTP probe failed.",
        "error_code": "probe_error",
    }
    assert context.identity_calls == 1
    assert all(dict(request.headers) == _IDENTITY for request in http.requests)


@pytest.mark.asyncio
async def test_rate_limit_missing_impersonation_dependency_preserves_public_result() -> None:
    _, _, rate, _ = _canonical_analyzers()
    context = fake_context(http_responses=[ProbeUnavailable(error_code="missing_dependency")])

    assert await rate(_URL, context, None, impersonate=True) == {
        "status": "error",
        "message": "curl-cffi is not installed; install the 'scrape-analyzers[browser]' extra.",
        "error_code": "missing_dependency",
    }


@pytest.mark.asyncio
async def test_waf_parses_tuples_and_forwards_tool_options() -> None:
    _, _, _, waf = _canonical_analyzers()
    context = fake_context(
        external_results=[
            ExternalToolResult(
                returncode=0,
                stdout="The site is behind Cloudflare WAF (Cloudflare Inc)",
                stderr="",
            )
        ]
    )

    assert await waf(
        _URL,
        context,
        find_all=True,
        external_tools_enabled=False,
    ) == {
        "status": "success",
        "wafs": [("behind Cloudflare WAF", "Cloudflare Inc")],
    }
    assert context.external_tools.calls == [{"url": _URL, "find_all": True, "enabled": False}]


@pytest.mark.asyncio
async def test_waf_empty_success_is_preserved() -> None:
    _, _, _, waf = _canonical_analyzers()
    context = fake_context(external_results=[ExternalToolResult(returncode=0, stdout="", stderr="")])

    assert await waf(_URL, context) == {"status": "success", "wafs": []}
    assert context.external_tools.calls == [{"url": _URL, "find_all": False, "enabled": None}]


@pytest.mark.asyncio
async def test_waf_parsed_detection_wins_over_nonzero_exit() -> None:
    _, _, _, waf = _canonical_analyzers()
    context = fake_context(
        external_results=[
            ExternalToolResult(
                returncode=2,
                stdout="The site is behind Cloudflare WAF (Cloudflare Inc)",
                stderr="private diagnostic",
            )
        ]
    )

    assert await waf(_URL, context) == {
        "status": "success",
        "wafs": [("behind Cloudflare WAF", "Cloudflare Inc")],
    }


@pytest.mark.asyncio
async def test_waf_nonzero_without_detection_returns_fixed_safe_error() -> None:
    _, _, _, waf = _canonical_analyzers()
    context = fake_context(
        external_results=[
            ExternalToolResult(
                returncode=2,
                stdout="",
                stderr="secret stderr at https://user:token@example.com/private",
            )
        ]
    )

    result = await waf(_URL, context)

    assert result == {
        "status": "error",
        "message": "Analyzer failed.",
        "error_code": "analyzer_error",
    }
    assert "secret" not in str(result)
    assert "token" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (
            ProbeUnavailable(error_code="missing_dependency"),
            {
                "status": "error",
                "message": "wafw00f missing",
                "error_code": "missing_dependency",
            },
        ),
        (
            ProbeTimeout(),
            {"status": "error", "message": "timeout", "error_code": "timeout"},
        ),
        (
            ProbeError(
                "external_tool_disabled",
                "External tool probing is disabled.",
            ),
            {
                "status": "error",
                "message": "External tool probing is disabled.",
                "error_code": "external_tool_disabled",
            },
        ),
    ],
)
async def test_waf_preserves_safe_dependency_and_timeout_results(
    failure: BaseException,
    expected: dict[str, Any],
) -> None:
    _, _, _, waf = _canonical_analyzers()
    context = fake_context(external_results=[failure])

    assert await waf(_URL, context) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("analyzer_name", ["robots", "tls", "rate", "waf"])
async def test_probe_failures_keep_only_approved_public_fields(
    analyzer_name: str,
) -> None:
    robots, tls, rate, waf = _canonical_analyzers()
    if analyzer_name == "waf":
        context = fake_context(external_results=[ProbeError("probe_error", "Probe failed.")])
        result = await waf(_URL, context)
        expected_message = "Probe failed."
    else:
        context = fake_context(http_responses=[ProbeError("probe_error", "HTTP probe failed.")])
        selected = {"robots": robots, "tls": tls, "rate": rate}[analyzer_name]
        result = await selected(_URL, context, None) if analyzer_name == "rate" else await selected(_URL, context)
        expected_message = "HTTP probe failed."

    assert result == {
        "status": "error",
        "message": expected_message,
        "error_code": "probe_error",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("analyzer_name", "safe_message"),
    [
        ("robots", "Robots.txt check failed."),
        ("tls", "Analyzer failed."),
        ("rate", "Rate limit profiling failed."),
        ("waf", "Analyzer failed."),
    ],
)
async def test_unexpected_failures_are_sanitized_without_exception_detail(
    analyzer_name: str,
    safe_message: str,
) -> None:
    robots, tls, rate, waf = _canonical_analyzers()
    failure = RuntimeError("backend exploded at https://user:secret@example.com/private?api_key=token")
    if analyzer_name == "waf":
        context = fake_context(external_results=[failure])
        result = await waf(_URL, context)
    else:
        context = fake_context(http_responses=[failure])
        selected = {"robots": robots, "tls": tls, "rate": rate}[analyzer_name]
        result = await selected(_URL, context, None) if analyzer_name == "rate" else await selected(_URL, context)

    assert result == {
        "status": "error",
        "message": safe_message,
        "error_code": "analyzer_error",
    }
    assert "secret" not in str(result)
    assert "api_key" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("analyzer_name", ["robots", "tls", "rate", "waf"])
@pytest.mark.parametrize(
    "failure_factory",
    [asyncio.CancelledError, PreflightDeadlineExceeded],
)
async def test_cancellation_and_overall_deadline_always_propagate(
    analyzer_name: str,
    failure_factory: type[BaseException],
) -> None:
    robots, tls, rate, waf = _canonical_analyzers()
    failure = failure_factory()
    if analyzer_name == "waf":
        context = fake_context(external_results=[failure])
        call = waf(_URL, context)
    else:
        context = fake_context(http_responses=[failure])
        selected = {"robots": robots, "tls": tls, "rate": rate}[analyzer_name]
        call = selected(_URL, context, None) if analyzer_name == "rate" else selected(_URL, context)

    with pytest.raises(failure_factory) as caught:
        await call
    assert caught.value is failure
