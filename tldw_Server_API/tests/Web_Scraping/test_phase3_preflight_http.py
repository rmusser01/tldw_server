from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.Web_Scraping.preflight import (
    PreflightDeadlineExceeded,
    PreflightLimits,
    PreflightRuntimeControls,
    ProbeBudgetExhausted,
    ProbeError,
    ProbeHttpRequest,
    ProbeTimeout,
    ProbeUnavailable,
)
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext
from tldw_Server_API.tests.Web_Scraping.preflight_fakes import (
    FakeClock,
    FakeHttpTransport,
    FakeProbeEgressGuard,
    FakeRawResponse,
)

pytestmark = pytest.mark.unit

_ADAPTER_MODULE = "tldw_Server_API.app.core.Web_Scraping.preflight.adapters.http"


def _adapter_module() -> Any | None:
    try:
        return importlib.import_module(_ADAPTER_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in {_ADAPTER_MODULE, _ADAPTER_MODULE.rpartition(".")[0]}:
            return None
        raise


def _required(name: str) -> Any:
    module = _adapter_module()
    assert module is not None, "Task 4 governed HTTP adapter module is missing"
    assert hasattr(module, name), f"Task 4 governed HTTP adapter {name} is missing"
    return getattr(module, name)


def _controls(
    *,
    requests: int | None = None,
    deadline: float | None = None,
    clock: FakeClock | None = None,
) -> PreflightRuntimeControls:
    return PreflightRuntimeControls(
        RuntimeRequestContext(
            source="preflight",
            stage="preflight",
            user_id="7",
            request_id="request-1",
            metadata={"scope": "task-4"},
        ),
        limits=PreflightLimits(requests=requests),
        deadline=deadline,
        clock=clock or FakeClock(),
    )


def _probe(
    *,
    controls: PreflightRuntimeControls,
    guard: FakeProbeEgressGuard,
    transport: Any,
    curl_transport: Any | None = None,
) -> Any:
    return _required("GuardedHttpProbe")(
        controls=controls,
        egress_guard=guard,
        transport=transport,
        curl_transport=curl_transport,
    )


class _FakeSession:
    def __init__(
        self,
        responses: list[Any],
        *,
        events: list[str] | None = None,
        block_get: bool = False,
        close_error: BaseException | None = None,
    ) -> None:
        self.responses = list(responses)
        self.events = events
        self.block_get = block_get
        self.close_error = close_error
        self.get_calls: list[tuple[str, dict[str, Any]]] = []
        self.get_started = asyncio.Event()
        self._release_get = asyncio.Event()
        self.close_calls = 0
        self.closed = False

    async def get(self, url: str, **kwargs: Any) -> Any:
        self.get_calls.append((url, dict(kwargs)))
        self.get_started.set()
        if self.events is not None:
            self.events.append(f"session:get:{url}")
        if self.block_get:
            await self._release_get.wait()
        if not self.responses:
            raise AssertionError("unexpected curl session request")
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def close(self) -> None:
        self.close_calls += 1
        self.closed = True
        if self.events is not None:
            self.events.append("session:close")
        if self.close_error is not None:
            raise self.close_error

    def release_get(self) -> None:
        self._release_get.set()


class _SessionFactory:
    def __init__(
        self,
        sessions: list[_FakeSession],
        *,
        events: list[str] | None = None,
    ) -> None:
        self.sessions = list(sessions)
        self.events = events
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _FakeSession:
        self.calls.append(dict(kwargs))
        if self.events is not None:
            self.events.append("session:create")
        if not self.sessions:
            raise AssertionError("unexpected curl session creation")
        return self.sessions.pop(0)


@pytest.mark.asyncio
async def test_http_probe_checks_every_dispatch_and_closes_responses() -> None:
    events: list[str] = []
    guard = FakeProbeEgressGuard([True, True], events=events)
    responses = [
        FakeRawResponse(302, headers={"Location": "/next"}, events=events),
        FakeRawResponse(200, text="ok", events=events),
    ]
    transport = FakeHttpTransport(responses, events=events)
    controls = _controls()
    probe = _probe(controls=controls, guard=guard, transport=transport)

    response = await probe.get(ProbeHttpRequest(url="https://example.com/start", timeout_s=20.0))

    assert response.url == "https://example.com/next"
    assert response.status == 200
    assert response.text == "ok"
    assert [call.url for call in transport.calls] == [
        "https://example.com/start",
        "https://example.com/next",
    ]
    assert guard.urls == [
        "https://example.com/start",
        "https://example.com/next",
    ]
    assert controls.consumed.requests == 2
    assert all(raw.closed for raw in responses)
    assert events == [
        "guard:https://example.com/start",
        "transport:https://example.com/start",
        "response:close",
        "guard:https://example.com/next",
        "transport:https://example.com/next",
        "response:close",
    ]
    assert all(context.stage == "preflight_subrequest" for context in guard.contexts)


@pytest.mark.asyncio
async def test_http_probe_resolves_absolute_redirects() -> None:
    guard = FakeProbeEgressGuard([True, True])
    transport = FakeHttpTransport(
        [
            FakeRawResponse(
                301,
                headers={"location": "https://other.example/final"},
            ),
            FakeRawResponse(204),
        ]
    )
    probe = _probe(controls=_controls(), guard=guard, transport=transport)

    response = await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert response.url == "https://other.example/final"
    assert [call.url for call in transport.calls] == [
        "https://example.com/start",
        "https://other.example/final",
    ]


@pytest.mark.asyncio
async def test_http_probe_disables_redirects_at_transport_when_following_is_off() -> None:
    raw = FakeRawResponse(302, headers={"Location": "/not-followed"})
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([raw])
    probe = _probe(controls=_controls(), guard=guard, transport=transport)

    response = await probe.get(
        ProbeHttpRequest(
            url="https://example.com/start",
            allow_redirects=False,
        )
    )

    assert response.status == 302
    assert len(transport.calls) == 1
    assert transport.calls[0].allow_redirects is False
    assert raw.closed is True


@pytest.mark.asyncio
async def test_http_probe_returns_redirect_response_without_location() -> None:
    raw = FakeRawResponse(302)
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([raw])
    probe = _probe(controls=_controls(), guard=guard, transport=transport)

    response = await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert response.status == 302
    assert len(transport.calls) == 1
    assert raw.closed is True


@pytest.mark.asyncio
async def test_redirect_loop_fails_before_reservation_guard_or_dispatch() -> None:
    controls = _controls()
    raw = FakeRawResponse(302, headers={"Location": "/start"})
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([raw])
    probe = _probe(controls=controls, guard=guard, transport=transport)

    with pytest.raises(ProbeError) as raised:
        await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert raised.value.error_code == "redirect_loop"
    assert raised.value.public_message == "Redirect loop detected."
    assert controls.consumed.requests == 1
    assert len(guard.urls) == 1
    assert len(transport.calls) == 1
    assert raw.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "location",
    [" ", "https://", "mailto:secret@example.com", "https://[bad", "\n/next"],
)
async def test_invalid_redirect_fails_closed_without_target_dispatch(
    location: str,
) -> None:
    controls = _controls()
    raw = FakeRawResponse(302, headers={"Location": location})
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([raw])
    probe = _probe(controls=controls, guard=guard, transport=transport)

    with pytest.raises(ProbeError) as raised:
        await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert raised.value.error_code == "invalid_redirect"
    assert raised.value.public_message == "Redirect target is invalid."
    assert controls.consumed.requests == 1
    assert len(guard.urls) == 1
    assert len(transport.calls) == 1
    assert raw.closed is True


@pytest.mark.asyncio
async def test_https_redirect_downgrade_fails_before_target_dispatch() -> None:
    raw = FakeRawResponse(
        302,
        headers={"Location": "http://example.com/insecure"},
    )
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([raw])
    probe = _probe(controls=_controls(), guard=guard, transport=transport)

    with pytest.raises(ProbeError) as raised:
        await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert raised.value.error_code == "invalid_redirect"
    assert len(guard.urls) == 1
    assert len(transport.calls) == 1
    assert raw.closed is True


@pytest.mark.asyncio
async def test_redirect_limit_closes_every_response_and_stops() -> None:
    dispatches = http_client.DEFAULT_MAX_REDIRECTS + 1
    responses = [FakeRawResponse(302, headers={"Location": f"/hop-{index + 1}"}) for index in range(dispatches)]
    guard = FakeProbeEgressGuard([True] * dispatches)
    transport = FakeHttpTransport(responses)
    controls = _controls(requests=dispatches + 1)
    probe = _probe(controls=controls, guard=guard, transport=transport)

    with pytest.raises(ProbeError) as raised:
        await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert raised.value.error_code == "too_many_redirects"
    assert raised.value.public_message == "Redirect limit exceeded."
    assert len(transport.calls) == dispatches
    assert len(guard.urls) == dispatches
    assert controls.consumed.requests == dispatches
    assert all(raw.closed for raw in responses)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "expected_code"),
    [("address_forbidden", "policy_denied"), ("policy_error", "policy_error")],
)
async def test_denied_initial_url_never_reaches_transport(
    reason: str,
    expected_code: str,
) -> None:
    controls = _controls()
    guard = FakeProbeEgressGuard([reason])
    transport = FakeHttpTransport([])
    probe = _probe(controls=controls, guard=guard, transport=transport)

    with pytest.raises(ProbeError) as raised:
        await probe.get(ProbeHttpRequest(url="https://private.example/secret"))

    assert raised.value.error_code == expected_code
    assert raised.value.public_message == "Probe destination was denied."
    assert controls.consumed.requests == 1
    assert transport.calls == []


@pytest.mark.asyncio
async def test_denied_redirect_never_reaches_transport() -> None:
    first = FakeRawResponse(
        302,
        headers={"Location": "https://private.example/secret"},
    )
    controls = _controls()
    guard = FakeProbeEgressGuard([True, False])
    transport = FakeHttpTransport([first])
    probe = _probe(controls=controls, guard=guard, transport=transport)

    with pytest.raises(ProbeError) as raised:
        await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert raised.value.error_code == "policy_denied"
    assert guard.urls == [
        "https://example.com/start",
        "https://private.example/secret",
    ]
    assert controls.consumed.requests == 2
    assert [call.url for call in transport.calls] == ["https://example.com/start"]
    assert first.closed is True


@pytest.mark.asyncio
async def test_request_budget_exhaustion_prevents_initial_transport() -> None:
    controls = _controls(requests=0)
    guard = FakeProbeEgressGuard([])
    transport = FakeHttpTransport([])
    probe = _probe(controls=controls, guard=guard, transport=transport)

    with pytest.raises(ProbeBudgetExhausted):
        await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert controls.consumed.requests == 0
    assert guard.urls == []
    assert transport.calls == []


@pytest.mark.asyncio
async def test_redirect_budget_exhaustion_prevents_target_guard_and_transport() -> None:
    first = FakeRawResponse(302, headers={"Location": "/next"})
    controls = _controls(requests=1)
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([first])
    probe = _probe(controls=controls, guard=guard, transport=transport)

    with pytest.raises(ProbeBudgetExhausted):
        await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert controls.consumed.requests == 1
    assert guard.urls == ["https://example.com/start"]
    assert len(transport.calls) == 1
    assert first.closed is True


@pytest.mark.asyncio
async def test_expired_overall_deadline_prevents_transport() -> None:
    clock = FakeClock(5.0)
    controls = _controls(deadline=5.0, clock=clock)
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([])
    probe = _probe(controls=controls, guard=guard, transport=transport)

    with pytest.raises(PreflightDeadlineExceeded):
        await probe.get(ProbeHttpRequest(url="https://example.com/start", timeout_s=20.0))

    assert controls.consumed.requests == 1
    assert guard.urls == ["https://example.com/start"]
    assert transport.calls == []


@pytest.mark.asyncio
async def test_per_dispatch_timeout_is_capped_by_overall_deadline() -> None:
    clock = FakeClock(4.0)
    controls = _controls(deadline=10.0, clock=clock)
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([FakeRawResponse(200)])
    probe = _probe(controls=controls, guard=guard, transport=transport)

    await probe.get(ProbeHttpRequest(url="https://example.com/start", timeout_s=20.0))

    assert transport.calls[0].timeout_s == 6.0


@pytest.mark.asyncio
async def test_transport_timeout_is_analyzer_local_when_deadline_remains() -> None:
    controls = _controls()
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([TimeoutError("secret timeout details")])
    probe = _probe(controls=controls, guard=guard, transport=transport)

    with pytest.raises(ProbeTimeout) as raised:
        await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert raised.value.error_code == "timeout"
    assert "secret" not in str(raised.value)
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_transport_timeout_is_overall_deadline_when_deadline_expires() -> None:
    clock = FakeClock(0.0)

    def expire() -> TimeoutError:
        clock.advance(2.0)
        return TimeoutError("secret timeout details")

    controls = _controls(deadline=1.0, clock=clock)
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([expire])
    probe = _probe(controls=controls, guard=guard, transport=transport)

    with pytest.raises(PreflightDeadlineExceeded):
        await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_caller_cancellation_propagates_from_transport() -> None:
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([FakeRawResponse(200)], block_send=True)
    probe = _probe(controls=_controls(), guard=guard, transport=transport)
    task = asyncio.create_task(probe.get(ProbeHttpRequest(url="https://example.com/start")))
    await transport.send_started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_acquired_response_finishes_closing_before_cancellation_propagates() -> None:
    raw = FakeRawResponse(200, block_close=True)
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([raw])
    probe = _probe(controls=_controls(), guard=guard, transport=transport)
    task = asyncio.create_task(probe.get(ProbeHttpRequest(url="https://example.com/start")))
    await raw.close_started.wait()

    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    raw.release_close()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert raw.closed is True
    assert raw.close_calls == 1


@pytest.mark.asyncio
async def test_response_close_failure_does_not_replace_success() -> None:
    raw = FakeRawResponse(
        200,
        text="ok",
        close_error=RuntimeError("secret cleanup detail"),
    )
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([raw])
    probe = _probe(controls=_controls(), guard=guard, transport=transport)

    response = await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert response.text == "ok"
    assert raw.closed is True
    assert raw.close_calls == 1


@pytest.mark.asyncio
async def test_transport_failure_is_sanitized_and_not_retried() -> None:
    guard = FakeProbeEgressGuard([True])
    transport = FakeHttpTransport([RuntimeError("Authorization: Bearer secret at ?token=secret")])
    controls = _controls()
    probe = _probe(controls=controls, guard=guard, transport=transport)

    with pytest.raises(ProbeError) as raised:
        await probe.get(ProbeHttpRequest(url="https://example.com/start"))

    assert raised.value.error_code == "probe_error"
    assert raised.value.public_message == "HTTP probe failed."
    assert "secret" not in str(raised.value)
    assert controls.consumed.requests == 1
    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_cross_origin_redirect_strips_all_sensitive_credentials() -> None:
    guard = FakeProbeEgressGuard([True, True])
    transport = FakeHttpTransport(
        [
            FakeRawResponse(
                302,
                headers={"Location": "https://b.example/final"},
            ),
            FakeRawResponse(200),
        ]
    )
    probe = _probe(controls=_controls(), guard=guard, transport=transport)
    request = ProbeHttpRequest(
        url="https://a.example/start",
        headers={
            "Authorization": "Bearer secret",
            "Proxy-Authorization": "Basic secret",
            "Cookie": "header-secret",
            "X-API-Key": "secret",
            "API-Key": "secret",
            "X-Auth-Token": "secret",
            "Accept": "text/html",
        },
        cookies={"session": "secret"},
    )

    await probe.get(request)

    second = transport.calls[1]
    assert dict(second.headers) == {"Accept": "text/html"}
    assert dict(second.cookies) == {}


@pytest.mark.asyncio
async def test_normalized_same_origin_redirect_preserves_credentials() -> None:
    headers = {"authorization": "Bearer secret", "Accept": "text/html"}
    cookies = {"session": "secret"}
    guard = FakeProbeEgressGuard([True, True])
    transport = FakeHttpTransport(
        [
            FakeRawResponse(
                302,
                headers={
                    "Location": "https://xn--bcher-kva.example:443/final",
                },
            ),
            FakeRawResponse(200),
        ]
    )
    probe = _probe(controls=_controls(), guard=guard, transport=transport)

    await probe.get(
        ProbeHttpRequest(
            url="https://b\u00fccher.example/start",
            headers=headers,
            cookies=cookies,
        )
    )

    second = transport.calls[1]
    assert dict(second.headers) == headers
    assert dict(second.cookies) == cookies


def test_malformed_origin_comparison_fails_closed() -> None:
    credentials_for_hop = _required("_credentials_for_hop")

    headers, cookies = credentials_for_hop(
        {"Authorization": "Bearer secret", "Accept": "text/html"},
        {"session": "secret"},
        original_url="https://[bad",
        target_url="https://example.com/next",
    )

    assert headers == {"Accept": "text/html"}
    assert cookies == {}


@pytest.mark.asyncio
async def test_httpx_transport_uses_central_single_attempt_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    raw = FakeRawResponse(200)

    async def fake_afetch(**kwargs: Any) -> FakeRawResponse:
        calls.append(dict(kwargs))
        return raw

    module = _adapter_module()
    assert module is not None, "Task 4 governed HTTP adapter module is missing"
    monkeypatch.setattr(module.http_client, "afetch", fake_afetch)
    transport = _required("HttpxProbeTransport")()
    request = ProbeHttpRequest(
        url="https://example.com/start",
        headers={"Accept": "text/html"},
        cookies={"session": "value"},
        timeout_s=3.0,
        proxies={"https": "http://proxy.example:8080"},
    )

    result = await transport.send(request)

    assert result is raw
    assert len(calls) == 1
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == request.url
    assert calls[0]["headers"] == dict(request.headers)
    assert calls[0]["cookies"] == dict(request.cookies)
    assert calls[0]["timeout"] == 3.0
    assert calls[0]["allow_redirects"] is False
    assert calls[0]["proxies"] == dict(request.proxies)
    assert isinstance(calls[0]["retry"], http_client.RetryPolicy)
    assert calls[0]["retry"].attempts == 1


@pytest.mark.asyncio
async def test_impersonated_request_selects_curl_transport() -> None:
    plain = FakeHttpTransport([])
    curl = FakeHttpTransport([FakeRawResponse(200)])
    guard = FakeProbeEgressGuard([True])
    probe = _probe(
        controls=_controls(),
        guard=guard,
        transport=plain,
        curl_transport=curl,
    )

    await probe.get(
        ProbeHttpRequest(
            url="https://example.com/start",
            impersonate="chrome120",
        )
    )

    assert plain.calls == []
    assert len(curl.calls) == 1


@pytest.mark.asyncio
async def test_curl_transport_rechecks_egress_immediately_before_get_and_closes() -> None:
    events: list[str] = []
    raw = FakeRawResponse(200, events=events)
    session = _FakeSession([raw], events=events)
    factory = _SessionFactory([session], events=events)
    guard = FakeProbeEgressGuard([True, True], events=events)
    controls = _controls()
    curl = _required("CurlCffiProbeTransport")(
        egress_guard=guard,
        request_context=controls.request_context,
        session_factory=factory,
    )
    probe = _probe(
        controls=controls,
        guard=guard,
        transport=FakeHttpTransport([]),
        curl_transport=curl,
    )

    await probe.get(
        ProbeHttpRequest(
            url="https://example.com/start",
            headers={"Accept": "text/html"},
            cookies={"session": "value"},
            timeout_s=4.0,
            impersonate="chrome120",
            proxies={"https": "http://proxy.example:8080"},
        )
    )

    assert guard.urls == ["https://example.com/start"] * 2
    assert factory.calls == [{"impersonate": "chrome120"}]
    assert session.get_calls == [
        (
            "https://example.com/start",
            {
                "headers": {"Accept": "text/html"},
                "cookies": {"session": "value"},
                "timeout": 4.0,
                "allow_redirects": False,
                "proxies": {"https": "http://proxy.example:8080"},
            },
        )
    ]
    assert events == [
        "guard:https://example.com/start",
        "session:create",
        "guard:https://example.com/start",
        "session:get:https://example.com/start",
        "response:close",
        "session:close",
    ]
    assert raw.closed is True
    assert session.closed is True


@pytest.mark.asyncio
async def test_curl_second_egress_denial_prevents_get_and_closes_session() -> None:
    session = _FakeSession([])
    factory = _SessionFactory([session])
    guard = FakeProbeEgressGuard([False])
    context = _controls().request_context
    transport = _required("CurlCffiProbeTransport")(
        egress_guard=guard,
        request_context=context,
        session_factory=factory,
    )

    with pytest.raises(ProbeError) as raised:
        await transport.send(
            ProbeHttpRequest(
                url="https://private.example/start",
                impersonate="chrome120",
            )
        )

    assert raised.value.error_code == "policy_denied"
    assert session.get_calls == []
    assert session.closed is True


@pytest.mark.asyncio
async def test_missing_curl_dependency_is_safe_and_does_no_transport_work() -> None:
    guard = FakeProbeEgressGuard([])
    transport = _required("CurlCffiProbeTransport")(
        egress_guard=guard,
        request_context=_controls().request_context,
        session_factory=None,
    )

    with pytest.raises(ProbeUnavailable) as raised:
        await transport.send(
            ProbeHttpRequest(
                url="https://example.com/start",
                impersonate="chrome120",
            )
        )

    assert raised.value.error_code == "missing_dependency"
    assert raised.value.public_message == "Probe dependency is unavailable."
    assert guard.urls == []


@pytest.mark.asyncio
async def test_curl_transport_closes_session_on_timeout() -> None:
    session = _FakeSession([TimeoutError("secret timeout")])
    guard = FakeProbeEgressGuard([True])
    transport = _required("CurlCffiProbeTransport")(
        egress_guard=guard,
        request_context=_controls().request_context,
        session_factory=_SessionFactory([session]),
    )

    with pytest.raises(TimeoutError):
        await transport.send(
            ProbeHttpRequest(
                url="https://example.com/start",
                impersonate="chrome120",
            )
        )

    assert session.closed is True
    assert session.close_calls == 1


@pytest.mark.asyncio
async def test_curl_transport_closes_session_before_propagating_cancellation() -> None:
    session = _FakeSession([], block_get=True)
    guard = FakeProbeEgressGuard([True])
    transport = _required("CurlCffiProbeTransport")(
        egress_guard=guard,
        request_context=_controls().request_context,
        session_factory=_SessionFactory([session]),
    )
    task = asyncio.create_task(
        transport.send(
            ProbeHttpRequest(
                url="https://example.com/start",
                impersonate="chrome120",
            )
        )
    )
    await session.get_started.wait()

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.closed is True
    assert session.close_calls == 1


@pytest.mark.asyncio
async def test_curl_response_close_failure_still_closes_session() -> None:
    raw = FakeRawResponse(
        200,
        close_error=RuntimeError("secret response cleanup"),
    )
    session = _FakeSession([raw])
    guard = FakeProbeEgressGuard([True, True])
    controls = _controls()
    curl = _required("CurlCffiProbeTransport")(
        egress_guard=guard,
        request_context=controls.request_context,
        session_factory=_SessionFactory([session]),
    )
    probe = _probe(
        controls=controls,
        guard=guard,
        transport=FakeHttpTransport([]),
        curl_transport=curl,
    )

    response = await probe.get(
        ProbeHttpRequest(
            url="https://example.com/start",
            impersonate="chrome120",
        )
    )

    assert response.status == 200
    assert raw.closed is True
    assert session.closed is True


@pytest.mark.asyncio
async def test_curl_response_close_cancellation_is_secondary_and_closes_session() -> None:
    raw = FakeRawResponse(200, close_error=asyncio.CancelledError())
    session = _FakeSession([raw])
    guard = FakeProbeEgressGuard([True, True])
    controls = _controls()
    curl = _required("CurlCffiProbeTransport")(
        egress_guard=guard,
        request_context=controls.request_context,
        session_factory=_SessionFactory([session]),
    )
    probe = _probe(
        controls=controls,
        guard=guard,
        transport=FakeHttpTransport([]),
        curl_transport=curl,
    )

    response = await probe.get(
        ProbeHttpRequest(
            url="https://example.com/start",
            impersonate="chrome120",
        )
    )

    assert response.status == 200
    assert raw.closed is True
    assert session.closed is True


def test_adapter_package_exports_only_task_4_http_adapters() -> None:
    package = importlib.import_module("tldw_Server_API.app.core.Web_Scraping.preflight.adapters")

    assert package.__all__ == [
        "CurlCffiProbeTransport",
        "GuardedHttpProbe",
        "HttpxProbeTransport",
    ]
    assert package.CurlCffiProbeTransport is _required("CurlCffiProbeTransport")
    assert package.GuardedHttpProbe is _required("GuardedHttpProbe")
    assert package.HttpxProbeTransport is _required("HttpxProbeTransport")
