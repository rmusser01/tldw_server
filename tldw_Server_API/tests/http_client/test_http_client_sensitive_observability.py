from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from threading import Barrier
from typing import Any

import httpx
import pytest
from loguru import logger
from opentelemetry import context as otel_context

from tldw_Server_API.app.core import http_client

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _allow_synthetic_public_observability_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the public test host reachable under CI's hostile egress allowlist."""
    configured = os.getenv("WORKFLOWS_EGRESS_ALLOWLIST", "")
    if not configured.strip():
        return
    entries = [entry.strip() for entry in configured.split(",") if entry.strip()]
    if "8.8.8.8" not in entries:
        entries.append("8.8.8.8")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", ",".join(entries))


class _MetricRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def increment(
        self,
        name: str,
        _value: float = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.calls.append((name, dict(labels or {})))

    def observe(
        self,
        name: str,
        _value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.calls.append((name, dict(labels or {})))


class _TraceRecorder:
    def __init__(self) -> None:
        self.span_attributes: list[dict[str, Any]] = []
        self.updated_attributes: list[dict[str, Any]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.recorded_exceptions: list[str] = []

    @contextmanager
    def span(self, _name: str, *, attributes: dict[str, Any]):
        self.span_attributes.append(dict(attributes))
        try:
            yield None
        except BaseException as exc:
            self.recorded_exceptions.append(str(exc))
            raise

    @asynccontextmanager
    async def async_span(self, _name: str, *, attributes: dict[str, Any]):
        self.span_attributes.append(dict(attributes))
        try:
            yield None
        except BaseException as exc:
            self.recorded_exceptions.append(str(exc))
            raise

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.updated_attributes.append(dict(attributes))

    def add_event(self, name: str, attributes: dict[str, Any]) -> None:
        self.events.append((name, dict(attributes)))

    def get_baggage(self, _key: str) -> None:
        return None


def test_sensitive_sync_request_uses_real_url_without_observability_disclosure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34:8443/runtime-secret-path/embeddings?tenant=private"
    observed_requests: list[str] = []
    suppression_states: list[object] = []
    metrics = _MetricRecorder()
    traces = _TraceRecorder()
    log_records: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(str(request.url))
        suppression_states.append(
            otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY)
        )
        status = 500 if len(observed_requests) == 1 else 200
        return httpx.Response(status, request=request, json={"ok": status == 200})

    monkeypatch.setattr(http_client, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)
    monkeypatch.setattr(http_client.time, "sleep", lambda _delay: None)
    sink_id = logger.add(lambda message: log_records.append(dict(message.record)), level="DEBUG")
    client = http_client.create_client(transport=httpx.MockTransport(handler))

    try:
        response = http_client.fetch(
            method="GET",
            url=endpoint,
            client=client,
            retry=http_client.RetryPolicy(attempts=2, backoff_base_ms=1),
            sensitive_observability=True,
        )
    finally:
        client.close()
        logger.remove(sink_id)

    assert response.status_code == 200
    assert observed_requests == [endpoint, endpoint]
    assert suppression_states == [True, True]
    assert otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY) is None
    assert traces.span_attributes[0]["url.full"] == http_client._SENSITIVE_OBSERVABILITY_URL
    observability = repr(
        {
            "logs": log_records,
            "metrics": metrics.calls,
            "spans": traces.span_attributes,
            "updated": traces.updated_attributes,
            "events": traces.events,
        }
    )
    for sensitive_fragment in (
        "93.184.216.34",
        "runtime-secret-path",
        "tenant=private",
        endpoint,
    ):
        assert sensitive_fragment not in observability


def test_sensitive_context_promotes_sync_request_observability_without_callsite_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34/context-secret?credential=private"
    observed_requests: list[str] = []
    traces = _TraceRecorder()

    def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(str(request.url))
        return httpx.Response(200, request=request, json={"ok": True})

    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)
    client = http_client.create_client(transport=httpx.MockTransport(handler))
    try:
        with http_client.sensitive_http_observability_context():
            response = http_client.fetch(
                method="GET",
                url=endpoint,
                client=client,
            )
    finally:
        client.close()

    assert response.status_code == 200
    assert observed_requests == [endpoint]
    assert traces.span_attributes[0]["url.full"] == (
        http_client._SENSITIVE_OBSERVABILITY_URL
    )
    assert endpoint not in repr(traces.span_attributes)


def test_sensitive_context_span_redaction_isolated_from_concurrent_public_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_endpoint = "http://93.184.216.34/context-secret"
    public_endpoint = "http://8.8.8.8/public-health"
    request_barrier = Barrier(2)
    traces = _TraceRecorder()

    def handler(request: httpx.Request) -> httpx.Response:
        request_barrier.wait(timeout=2)
        return httpx.Response(200, request=request, json={"ok": True})

    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)
    client = http_client.create_client(transport=httpx.MockTransport(handler))

    def send(url: str, *, sensitive: bool) -> int:
        if sensitive:
            with http_client.sensitive_http_observability_context():
                return http_client.fetch(
                    method="GET",
                    url=url,
                    client=client,
                ).status_code
        return http_client.fetch(
            method="GET",
            url=url,
            client=client,
        ).status_code

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            sensitive_future = executor.submit(send, sensitive_endpoint, sensitive=True)
            public_future = executor.submit(send, public_endpoint, sensitive=False)
            assert sensitive_future.result(timeout=3) == 200
            assert public_future.result(timeout=3) == 200
    finally:
        client.close()

    span_urls = {attributes["url.full"] for attributes in traces.span_attributes}
    assert span_urls == {
        http_client._SENSITIVE_OBSERVABILITY_URL,
        public_endpoint,
    }
    assert sensitive_endpoint not in repr(traces.span_attributes)


def test_sensitive_context_redacts_cross_origin_redirect_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_url = "http://93.184.216.34/redirect-context-secret"
    final_url = "http://8.8.8.8/redirect-target-secret"
    observed_requests: list[str] = []
    log_records: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(str(request.url))
        if str(request.url) == start_url:
            return httpx.Response(
                302,
                request=request,
                headers={"location": final_url},
            )
        return httpx.Response(200, request=request, json={"ok": True})

    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", _TraceRecorder)
    sink_id = logger.add(
        lambda message: log_records.append(dict(message.record)),
        level="DEBUG",
    )
    client = http_client.create_client(transport=httpx.MockTransport(handler))
    try:
        with http_client.sensitive_http_observability_context():
            response = http_client.fetch(
                method="GET",
                url=start_url,
                client=client,
                headers={"Authorization": "Bearer redirect-secret"},
            )
    finally:
        client.close()
        logger.remove(sink_id)

    assert response.status_code == 200
    assert observed_requests == [start_url, final_url]
    redirect_records = [
        record
        for record in log_records
        if "Stripped sensitive headers" in str(record["message"])
    ]
    assert redirect_records
    assert redirect_records[0]["extra"]["target_host"] == (
        "sensitive-endpoint.invalid"
    )
    observability = repr(log_records)
    for sensitive_fragment in (
        "93.184.216.34",
        "8.8.8.8",
        "redirect-context-secret",
        "redirect-target-secret",
        "redirect-secret",
    ):
        assert sensitive_fragment not in observability


def test_sensitive_context_redacts_download_retry_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    endpoint = "http://93.184.216.34/download-context-secret?credential=private"
    observed_requests: list[str] = []
    log_records: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(str(request.url))
        status = 500 if len(observed_requests) == 1 else 200
        return httpx.Response(status, request=request, content=b"ok")

    monkeypatch.setattr(http_client.time, "sleep", lambda _delay: None)
    sink_id = logger.add(
        lambda message: log_records.append(dict(message.record)),
        level="DEBUG",
    )
    client = http_client.create_client(transport=httpx.MockTransport(handler))
    destination = tmp_path / "download.bin"
    try:
        with http_client.sensitive_http_observability_context():
            result = http_client.download(
                url=endpoint,
                dest=destination,
                client=client,
                retry=http_client.RetryPolicy(attempts=2, backoff_base_ms=1),
            )
    finally:
        client.close()
        logger.remove(sink_id)

    assert result == destination
    assert destination.read_bytes() == b"ok"
    assert observed_requests == [endpoint, endpoint]
    observability = repr(log_records)
    for sensitive_fragment in (
        "93.184.216.34",
        "download-context-secret",
        "credential=private",
        endpoint,
    ):
        assert sensitive_fragment not in observability


def test_public_sync_httpx_redirect_observes_final_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_url = "http://93.184.216.34/redirect-start"
    final_url = "http://8.8.8.8/redirect-final"
    metrics = _MetricRecorder()
    outbound_records: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == start_url:
            return httpx.Response(302, request=request, headers={"location": final_url})
        return httpx.Response(200, request=request, json={"ok": True})

    monkeypatch.setattr(http_client, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(http_client, "get_tracing_manager", _TraceRecorder)
    monkeypatch.setattr(
        http_client,
        "_log_outbound_request",
        lambda **kwargs: outbound_records.append(kwargs),
    )
    client = http_client.create_client(transport=httpx.MockTransport(handler))
    try:
        response = http_client.fetch(
            method="GET",
            url=start_url,
            client=client,
            allow_redirects=True,
        )
    finally:
        client.close()

    metric_hosts = {
        labels.get("host")
        for name, labels in metrics.calls
        if name in {"http_client_requests_total", "http_client_request_duration_seconds"}
    }
    assert response.status_code == 200
    assert metric_hosts == {"8.8.8.8"}
    assert outbound_records[-1]["url"] == final_url


def test_sensitive_egress_dns_failure_redacts_full_loguru_record_and_resets_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Security import egress

    endpoint = "https://credential-derived.private.example/secret/path?tenant=private"
    log_records: list[dict[str, Any]] = []

    def fail_dns(*_args: object, **_kwargs: object) -> list[object]:
        raise OSError("deterministic resolver failure")

    monkeypatch.setattr(http_client, "is_explicit_pytest_runtime", lambda: False)
    monkeypatch.setattr(http_client, "env_flag_enabled", lambda _name: False)
    monkeypatch.setattr(http_client, "is_test_mode", lambda: False)
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    monkeypatch.setattr(egress.socket, "getaddrinfo", fail_dns)
    sink_id = logger.add(
        lambda message: log_records.append(dict(message.record)),
        level="DEBUG",
        backtrace=True,
        diagnose=True,
    )

    try:
        with pytest.raises(http_client.EgressPolicyError) as exc_info:
            http_client.fetch(
                method="GET",
                url=endpoint,
                retry=http_client.RetryPolicy(attempts=1),
                sensitive_observability=True,
            )
    finally:
        logger.remove(sink_id)

    assert str(exc_info.value) == "Sensitive endpoint denied by egress policy"
    assert http_client._SENSITIVE_HTTP_LOG_CONTEXT.get() is False
    rendered_records = repr(log_records)
    for sensitive_fragment in (
        "credential-derived.private.example",
        "secret/path",
        "tenant=private",
        endpoint,
    ):
        assert sensitive_fragment not in rendered_records


def test_sensitive_context_promotes_simple_fetch_egress_observability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Security import egress

    endpoint = "https://credential-derived.private.example/secret/path"
    calls: list[tuple[str, dict[str, Any]]] = []

    class Denied:
        allowed = False

    def evaluate_url_policy(url: str, **kwargs: Any) -> Denied:
        calls.append((url, kwargs))
        return Denied()

    monkeypatch.setattr(egress, "evaluate_url_policy", evaluate_url_policy)

    with http_client.sensitive_http_observability_context():
        with pytest.raises(ValueError, match="Egress denied for URL"):
            http_client.fetch(endpoint)

    assert calls == [(endpoint, {"sensitive_observability": True})]
    assert http_client._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


def test_sensitive_log_filter_does_not_hide_concurrent_public_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_endpoint = "http://93.184.216.34/runtime-secret-path/embeddings"
    public_endpoint = "http://8.8.8.8/public-health?scope=visible"
    request_barrier = Barrier(2)
    log_records: list[dict[str, Any]] = []
    auto_instrumented_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY):
            auto_instrumented_urls.append(str(request.url))
        request_barrier.wait(timeout=2)
        return httpx.Response(200, request=request, json={"ok": True})

    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", _TraceRecorder)
    sink_id = logger.add(lambda message: log_records.append(dict(message.record)), level="DEBUG")
    client = http_client.create_client(transport=httpx.MockTransport(handler))

    def send(url: str, *, sensitive: bool) -> int:
        return http_client.fetch(
            method="GET",
            url=url,
            client=client,
            sensitive_observability=sensitive,
        ).status_code

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            sensitive_future = executor.submit(send, sensitive_endpoint, sensitive=True)
            public_future = executor.submit(send, public_endpoint, sensitive=False)
            assert sensitive_future.result(timeout=3) == 200
            assert public_future.result(timeout=3) == 200
    finally:
        client.close()
        logger.remove(sink_id)

    rendered_logs = "\n".join(str(record["message"]) for record in log_records)
    httpx_logs = "\n".join(
        str(record["message"])
        for record in log_records
        if str(record["name"]).startswith(("httpx", "httpcore"))
    )
    assert "8.8.8.8" in rendered_logs
    assert "public-health" in httpx_logs
    assert "93.184.216.34" not in rendered_logs
    assert "runtime-secret-path" not in rendered_logs
    assert auto_instrumented_urls == [public_endpoint]
    assert otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY) is None


@pytest.mark.asyncio
async def test_sensitive_async_httpx_request_uses_real_url_without_observability_disclosure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "https://93.184.216.34:8443/runtime-secret-path/embeddings?tenant=private"
    observed_requests: list[str] = []
    suppression_states: list[object] = []
    pin_sensitivity: list[bool] = []
    metrics = _MetricRecorder()
    traces = _TraceRecorder()
    log_records: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(str(request.url))
        suppression_states.append(
            otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY)
        )
        status = 500 if len(observed_requests) == 1 else 200
        return httpx.Response(status, request=request, json={"ok": status == 200})

    monkeypatch.setattr(http_client, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)
    monkeypatch.setattr(
        http_client,
        "_check_cert_pinning",
        lambda *_args, sensitive_observability=False, **_kwargs: pin_sensitivity.append(
            sensitive_observability
        ),
    )

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(http_client.asyncio, "sleep", no_sleep)
    sink_id = logger.add(lambda message: log_records.append(dict(message.record)), level="DEBUG")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._tldw_cert_pinning = {"93.184.216.34": {"test-pin"}}

    try:
        response = await http_client.afetch(
            method="GET",
            url=endpoint,
            client=client,
            retry=http_client.RetryPolicy(attempts=2, backoff_base_ms=1),
            sensitive_observability=True,
        )
    finally:
        await client.aclose()
        logger.remove(sink_id)

    assert response.status_code == 200
    assert observed_requests == [endpoint, endpoint]
    assert suppression_states == [True, True]
    assert pin_sensitivity == [True, True]
    assert otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY) is None
    assert traces.span_attributes[0]["url.full"] == http_client._SENSITIVE_OBSERVABILITY_URL
    observability = repr(
        {
            "logs": log_records,
            "metrics": metrics.calls,
            "spans": traces.span_attributes,
            "updated": traces.updated_attributes,
            "events": traces.events,
        }
    )
    for sensitive_fragment in (
        "93.184.216.34",
        "runtime-secret-path",
        "tenant=private",
        endpoint,
    ):
        assert sensitive_fragment not in observability


@pytest.mark.asyncio
async def test_sensitive_context_promotes_async_httpx_observability_without_callsite_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34/async-context-secret"
    observed_requests: list[str] = []
    traces = _TraceRecorder()

    async def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(str(request.url))
        return httpx.Response(200, request=request, json={"ok": True})

    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with http_client.sensitive_http_observability_context():
            response = await http_client.afetch(
                method="GET",
                url=endpoint,
                client=client,
            )
    finally:
        await client.aclose()

    assert response.status_code == 200
    assert observed_requests == [endpoint]
    assert traces.span_attributes[0]["url.full"] == (
        http_client._SENSITIVE_OBSERVABILITY_URL
    )
    assert endpoint not in repr(traces.span_attributes)


@pytest.mark.asyncio
async def test_sensitive_async_aiohttp_path_redacts_retry_metrics_traces_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34:8443/aiohttp-secret/embeddings?tenant=private"
    observed_requests: list[str] = []
    suppression_states: list[object] = []
    metrics = _MetricRecorder()
    traces = _TraceRecorder()
    log_records: list[dict[str, Any]] = []

    class RawResponse:
        def __init__(self, status: int, url: str) -> None:
            self.status = status
            self.url = url
            self.headers = {"content-type": "application/json"}
            self.charset = "utf-8"

        def release(self) -> None:
            return None

    async def fake_io(**kwargs: Any) -> Any:
        observed_requests.append(kwargs["url"])
        suppression_states.append(
            otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY)
        )
        status = 500 if len(observed_requests) == 1 else 200
        return http_client._AiohttpResponse(
            RawResponse(status, kwargs["url"]),
            b'{"ok": true}',
        )

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(http_client, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(http_client, "_aiohttp_request_io", fake_io)
    monkeypatch.setattr(http_client, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)
    monkeypatch.setattr(http_client.asyncio, "sleep", no_sleep)
    sink_id = logger.add(lambda message: log_records.append(dict(message.record)), level="DEBUG")

    try:
        response = await http_client.afetch(
            method="GET",
            url=endpoint,
            client=object(),
            retry=http_client.RetryPolicy(attempts=2, backoff_base_ms=1),
            sensitive_observability=True,
        )
    finally:
        logger.remove(sink_id)

    assert response.status_code == 200
    assert observed_requests == [endpoint, endpoint]
    assert suppression_states == [True, True]
    assert otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY) is None
    assert traces.span_attributes[0]["url.full"] == http_client._SENSITIVE_OBSERVABILITY_URL
    observability = repr(
        {
            "logs": log_records,
            "metrics": metrics.calls,
            "spans": traces.span_attributes,
            "updated": traces.updated_attributes,
            "events": traces.events,
        }
    )
    for sensitive_fragment in (
        "93.184.216.34",
        "aiohttp-secret",
        "tenant=private",
        endpoint,
    ):
        assert sensitive_fragment not in observability


@pytest.mark.asyncio
async def test_sensitive_context_promotes_async_aiohttp_observability_without_callsite_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34/aiohttp-context-secret"
    observed_requests: list[str] = []
    traces = _TraceRecorder()

    class RawResponse:
        status = 200
        headers = {"content-type": "application/json"}
        charset = "utf-8"

        def __init__(self, url: str) -> None:
            self.url = url

        def release(self) -> None:
            return None

    async def fake_io(**kwargs: Any) -> Any:
        observed_requests.append(kwargs["url"])
        return http_client._AiohttpResponse(
            RawResponse(kwargs["url"]),
            b'{"ok": true}',
        )

    monkeypatch.setattr(http_client, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(http_client, "_aiohttp_request_io", fake_io)
    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)

    with http_client.sensitive_http_observability_context():
        response = await http_client.afetch(
            method="GET",
            url=endpoint,
            client=object(),
        )

    assert response.status_code == 200
    assert observed_requests == [endpoint]
    assert traces.span_attributes[0]["url.full"] == (
        http_client._SENSITIVE_OBSERVABILITY_URL
    )
    assert endpoint not in repr(traces.span_attributes)


@pytest.mark.asyncio
@pytest.mark.parametrize("sensitive_observability", [False, True])
async def test_byte_stream_closes_adapter_delegate_once_on_early_close(
    monkeypatch: pytest.MonkeyPatch,
    sensitive_observability: bool,
) -> None:
    finalized: list[str] = []

    class Adapter:
        async def stream_bytes(self, **_kwargs: Any):
            try:
                yield b"one"
                yield b"two"
            finally:
                finalized.append("closed")

    monkeypatch.setattr(
        http_client,
        "_get_transport_adapter",
        lambda _name: Adapter(),
    )
    traces = _TraceRecorder()
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)

    stream = http_client.astream_bytes(
        method="GET",
        url="http://93.184.216.34/early-close-secret",
        client=object(),
        sensitive_observability=sensitive_observability,
    )
    assert await stream.__anext__() == b"one"
    await stream.aclose()

    assert finalized == ["closed"]
    assert traces.span_attributes == []
    assert http_client._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


@pytest.mark.asyncio
async def test_public_byte_stream_does_not_hold_trace_context_across_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span_entries = 0

    class TaskBoundTrace:
        @asynccontextmanager
        async def async_span(self, _name: str, *, attributes: dict[str, Any]):
            nonlocal span_entries
            span_entries += 1
            entered_task = asyncio.current_task()
            try:
                yield None
            finally:
                if asyncio.current_task() is not entered_task:
                    raise RuntimeError("trace context crossed task boundary")

    class Adapter:
        async def stream_bytes(self, **_kwargs: Any):
            yield b"one"
            yield b"two"

    monkeypatch.setattr(http_client, "get_tracing_manager", TaskBoundTrace)
    monkeypatch.setattr(
        http_client,
        "_get_transport_adapter",
        lambda _name: Adapter(),
    )

    stream = http_client.astream_bytes(
        method="GET",
        url="http://8.8.8.8/public-stream",
        client=object(),
    )
    assert await stream.__anext__() == b"one"
    await asyncio.create_task(stream.aclose())

    assert span_entries == 0


@pytest.mark.asyncio
async def test_sensitive_byte_stream_installs_log_filter_once_per_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_calls = 0

    def install_filter() -> None:
        nonlocal install_calls
        install_calls += 1

    async def fake_stream_bytes_httpx(**_kwargs: Any):
        for index in range(5):
            yield str(index).encode()

    monkeypatch.setattr(
        http_client,
        "_install_sensitive_http_log_filter",
        install_filter,
    )
    monkeypatch.setattr(
        http_client,
        "_astream_bytes_httpx",
        fake_stream_bytes_httpx,
    )

    chunks = [
        chunk
        async for chunk in http_client.HttpxAdapter().stream_bytes(
            method="GET",
            url="http://93.184.216.34/filter-secret",
            client=object(),
            sensitive_observability=True,
        )
    ]

    assert chunks == [b"0", b"1", b"2", b"3", b"4"]
    assert install_calls == 1


@pytest.mark.asyncio
async def test_sensitive_async_egress_failure_redacts_and_resets_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Security import egress

    endpoint = "https://credential-derived.private.example/secret/path?tenant=private"
    log_records: list[dict[str, Any]] = []

    def fail_dns(*_args: object, **_kwargs: object) -> list[object]:
        raise OSError("deterministic resolver failure")

    monkeypatch.setattr(http_client, "is_explicit_pytest_runtime", lambda: False)
    monkeypatch.setattr(http_client, "env_flag_enabled", lambda _name: False)
    monkeypatch.setattr(http_client, "is_test_mode", lambda: False)
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    monkeypatch.setattr(egress.socket, "getaddrinfo", fail_dns)
    sink_id = logger.add(
        lambda message: log_records.append(dict(message.record)),
        level="DEBUG",
        backtrace=True,
        diagnose=True,
    )

    try:
        with pytest.raises(http_client.EgressPolicyError) as exc_info:
            await http_client.afetch(
                method="GET",
                url=endpoint,
                retry=http_client.RetryPolicy(attempts=1),
                sensitive_observability=True,
            )
    finally:
        logger.remove(sink_id)

    assert str(exc_info.value) == "Sensitive endpoint denied by egress policy"
    assert http_client._SENSITIVE_HTTP_LOG_CONTEXT.get() is False
    rendered_records = repr(log_records)
    for sensitive_fragment in (
        "credential-derived.private.example",
        "secret/path",
        "tenant=private",
        endpoint,
    ):
        assert sensitive_fragment not in rendered_records


@pytest.mark.asyncio
async def test_sensitive_async_log_filter_preserves_concurrent_public_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_endpoint = "http://93.184.216.34/runtime-secret-path/embeddings"
    public_endpoint = "http://8.8.8.8/public-health?scope=visible"
    stdlib_messages: list[str] = []
    auto_instrumented_urls: list[str] = []
    transport_logger = logging.getLogger("httpx.test-sensitive-observability")

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            stdlib_messages.append(record.getMessage())

    handler_sink = CaptureHandler()
    transport_logger.addHandler(handler_sink)
    transport_logger.setLevel(logging.DEBUG)
    transport_logger.propagate = False

    async def handler(request: httpx.Request) -> httpx.Response:
        if not otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY):
            auto_instrumented_urls.append(str(request.url))
        transport_logger.debug("transport request %s", request.url)
        await asyncio.sleep(0)
        return httpx.Response(200, request=request, json={"ok": True})

    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", _TraceRecorder)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def send(url: str, *, sensitive: bool) -> int:
        response = await http_client.afetch(
            method="GET",
            url=url,
            client=client,
            sensitive_observability=sensitive,
        )
        return response.status_code

    try:
        statuses = await asyncio.gather(
            send(sensitive_endpoint, sensitive=True),
            send(public_endpoint, sensitive=False),
        )
    finally:
        await client.aclose()
        transport_logger.removeHandler(handler_sink)

    rendered = "\n".join(stdlib_messages)
    assert statuses == [200, 200]
    assert "8.8.8.8" in rendered
    assert "public-health" in rendered
    assert "93.184.216.34" not in rendered
    assert "runtime-secret-path" not in rendered
    assert auto_instrumented_urls == [public_endpoint]
    assert http_client._SENSITIVE_HTTP_LOG_CONTEXT.get() is False
    assert otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY) is None


@pytest.mark.asyncio
async def test_sensitive_async_aiohttp_suppresses_auto_instrumentation_and_logs_only_for_sensitive_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_endpoint = "http://93.184.216.34/aiohttp-secret/embeddings"
    public_endpoint = "http://8.8.8.8/aiohttp-public?scope=visible"
    auto_instrumented_urls: list[str] = []
    stdlib_messages: list[str] = []
    transport_logger = logging.getLogger("aiohttp.client")

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            stdlib_messages.append(record.getMessage())

    class RawResponse:
        def __init__(self, url: str) -> None:
            self.status = 200
            self.url = url
            self.headers = {"content-type": "application/json"}
            self.charset = "utf-8"

        def release(self) -> None:
            return None

    handler_sink = CaptureHandler()
    transport_logger.addHandler(handler_sink)
    transport_logger.setLevel(logging.DEBUG)
    transport_logger.propagate = False

    async def fake_io(**kwargs: Any) -> Any:
        url = kwargs["url"]
        if not otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY):
            auto_instrumented_urls.append(url)
        transport_logger.debug("aiohttp request %s", url)
        await asyncio.sleep(0)
        return http_client._AiohttpResponse(RawResponse(url), b'{"ok": true}')

    monkeypatch.setattr(http_client, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(http_client, "_aiohttp_request_io", fake_io)
    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", _TraceRecorder)

    async def send(url: str, *, sensitive: bool) -> int:
        response = await http_client.afetch(
            method="GET",
            url=url,
            client=object(),
            sensitive_observability=sensitive,
        )
        return response.status_code

    try:
        statuses = await asyncio.gather(
            send(sensitive_endpoint, sensitive=True),
            send(public_endpoint, sensitive=False),
        )
    finally:
        transport_logger.removeHandler(handler_sink)

    rendered = "\n".join(stdlib_messages)
    assert statuses == [200, 200]
    assert auto_instrumented_urls == [public_endpoint]
    assert "8.8.8.8" in rendered
    assert "aiohttp-public" in rendered
    assert "93.184.216.34" not in rendered
    assert "aiohttp-secret" not in rendered
    assert otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY) is None


@pytest.mark.asyncio
async def test_sensitive_async_otel_suppression_resets_after_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34/runtime-secret-path/embeddings"
    failure_states: list[object] = []

    async def failing_handler(request: httpx.Request) -> httpx.Response:
        failure_states.append(
            otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY)
        )
        raise httpx.ConnectError("deterministic failure", request=request)

    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", _TraceRecorder)
    failing_client = httpx.AsyncClient(transport=httpx.MockTransport(failing_handler))
    try:
        with pytest.raises(http_client.NetworkError):
            await http_client.afetch(
                method="GET",
                url=endpoint,
                client=failing_client,
                retry=http_client.RetryPolicy(attempts=1),
                sensitive_observability=True,
            )
    finally:
        await failing_client.aclose()

    assert failure_states == [True]
    assert otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY) is None


@pytest.mark.asyncio
async def test_sensitive_apost_suppresses_transport_url_logs_and_resets_context() -> None:
    endpoint = "http://93.184.216.34:8443/tts-secret/audio?credential=private"
    observed_requests: list[str] = []
    suppression_states: list[object] = []
    stdlib_messages: list[str] = []
    transport_logger = logging.getLogger("httpx")

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            stdlib_messages.append(record.getMessage())

    async def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(str(request.url))
        suppression_states.append(
            otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY)
        )
        transport_logger.debug("httpx request %s", request.url)
        return httpx.Response(200, request=request, content=b"audio")

    capture_handler = CaptureHandler()
    transport_logger.addHandler(capture_handler)
    transport_logger.setLevel(logging.DEBUG)
    transport_logger.propagate = False
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await http_client.apost(
            url=endpoint,
            client=client,
            sensitive_observability=True,
        )
    finally:
        await client.aclose()
        transport_logger.removeHandler(capture_handler)

    assert response.status_code == 200
    assert observed_requests == [endpoint]
    assert suppression_states == [True]
    assert endpoint not in "\n".join(stdlib_messages)
    assert http_client._SENSITIVE_HTTP_LOG_CONTEXT.get() is False
    assert otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY) is None


@pytest.mark.asyncio
async def test_sensitive_async_byte_stream_redacts_retry_metrics_traces_logs_and_auto_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34:8443/tts-secret/audio?credential=private"
    observed_requests: list[str] = []
    suppression_states: list[object] = []
    metrics = _MetricRecorder()
    traces = _TraceRecorder()
    log_records: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(str(request.url))
        suppression_states.append(
            otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY)
        )
        status = 500 if len(observed_requests) == 1 else 200
        return httpx.Response(status, request=request, content=b"audio" if status == 200 else b"")

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(http_client, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)
    monkeypatch.setattr(http_client.asyncio, "sleep", no_sleep)
    sink_id = logger.add(lambda message: log_records.append(dict(message.record)), level="DEBUG")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    chunks: list[bytes] = []
    try:
        async for chunk in http_client.astream_bytes(
            method="POST",
            url=endpoint,
            client=client,
            retry=http_client.RetryPolicy(
                attempts=2,
                backoff_base_ms=1,
                retry_on_unsafe=True,
            ),
            sensitive_observability=True,
        ):
            chunks.append(chunk)
    finally:
        await client.aclose()
        logger.remove(sink_id)

    assert chunks == [b"audio"]
    assert observed_requests == [endpoint, endpoint]
    assert suppression_states == [True, True]
    assert traces.span_attributes == []
    assert http_client._SENSITIVE_HTTP_LOG_CONTEXT.get() is False
    assert otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY) is None
    observability = repr(
        {
            "logs": log_records,
            "metrics": metrics.calls,
            "spans": traces.span_attributes,
            "updated": traces.updated_attributes,
            "events": traces.events,
        }
    )
    for sensitive_fragment in (
        "93.184.216.34",
        "tts-secret",
        "credential=private",
        endpoint,
    ):
        assert sensitive_fragment not in observability


@pytest.mark.asyncio
async def test_sensitive_context_promotes_byte_stream_observability_without_callsite_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34/stream-context-secret"
    observed_requests: list[str] = []
    traces = _TraceRecorder()

    async def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(str(request.url))
        return httpx.Response(200, request=request, content=b"audio")

    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chunks: list[bytes] = []
    try:
        with http_client.sensitive_http_observability_context():
            async for chunk in http_client.astream_bytes(
                method="GET",
                url=endpoint,
                client=client,
            ):
                chunks.append(chunk)
    finally:
        await client.aclose()

    assert chunks == [b"audio"]
    assert observed_requests == [endpoint]
    assert traces.span_attributes == []


@pytest.mark.asyncio
async def test_sensitive_context_promotes_aiohttp_byte_stream_observability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34/aiohttp-stream-context-secret"
    observed_requests: list[str] = []
    traces = _TraceRecorder()

    class RawResponse:
        status = 200
        headers: dict[str, str] = {}

        def __init__(self, url: str) -> None:
            self.url = url

    @asynccontextmanager
    async def fake_stream_io(**kwargs: Any):
        observed_requests.append(kwargs["url"])

        async def chunks():
            yield b"audio"

        yield RawResponse(kwargs["url"]), chunks()

    monkeypatch.setattr(http_client, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(http_client, "_aiohttp_stream_io", fake_stream_io)
    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)
    chunks: list[bytes] = []
    with http_client.sensitive_http_observability_context():
        async for chunk in http_client.astream_bytes(
            method="GET",
            url=endpoint,
            client=object(),
        ):
            chunks.append(chunk)

    assert chunks == [b"audio"]
    assert observed_requests == [endpoint]
    assert traces.span_attributes == []


@pytest.mark.asyncio
async def test_sensitive_stream_scope_exits_before_yield_and_cross_task_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope_active = False
    scope_tasks: list[asyncio.Task[Any] | None] = []

    @asynccontextmanager
    async def tracking_scope(_enabled: bool):
        nonlocal scope_active
        entered_task = asyncio.current_task()
        scope_tasks.append(entered_task)
        scope_active = True
        try:
            yield
        finally:
            scope_active = False
            if asyncio.current_task() is not entered_task:
                raise RuntimeError("sensitive scope crossed task boundary")

    async def fake_stream_bytes_httpx(**_kwargs: Any):
        yield b"one"
        yield b"two"

    monkeypatch.setattr(
        http_client,
        "_sensitive_http_stream_context",
        tracking_scope,
    )
    monkeypatch.setattr(
        http_client,
        "_astream_bytes_httpx",
        fake_stream_bytes_httpx,
    )
    monkeypatch.setattr(http_client, "get_tracing_manager", _TraceRecorder)

    stream = http_client.astream_bytes(
        method="GET",
        url="http://93.184.216.34/scoped-stream-secret",
        client=object(),
        sensitive_observability=True,
    )
    assert await stream.__anext__() == b"one"
    assert scope_active is False
    await asyncio.create_task(stream.aclose())
    assert scope_active is False
    assert len(scope_tasks) >= 2


@pytest.mark.asyncio
async def test_sensitive_httpx_sse_redacts_transport_observability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34/sse-secret?credential=private"
    observed_requests: list[str] = []
    suppression_states: list[object] = []
    log_records: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(str(request.url))
        suppression_states.append(
            otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY)
        )
        return httpx.Response(
            200,
            request=request,
            content=b"data: hello\n\n",
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    sink_id = logger.add(
        lambda message: log_records.append(dict(message.record)),
        level="DEBUG",
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        events = [
            event
            async for event in http_client.astream_sse(
                url=endpoint,
                client=client,
                sensitive_observability=True,
            )
        ]
    finally:
        await client.aclose()
        logger.remove(sink_id)

    assert [event.data for event in events] == ["hello"]
    assert observed_requests == [endpoint]
    assert suppression_states == [True]
    assert endpoint not in repr(log_records)
    assert "sse-secret" not in repr(log_records)
    assert http_client._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


@pytest.mark.asyncio
async def test_sensitive_stream_error_is_not_recorded_with_real_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34/stream-error-context-secret?token=private"
    traces = _TraceRecorder()

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request, content=b"denied")

    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", lambda: traces)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(http_client.NetworkError, match=r"^HTTP 403$") as exc_info:
            with http_client.sensitive_http_observability_context():
                async for _chunk in http_client.astream_bytes(
                    method="GET",
                    url=endpoint,
                    client=client,
                ):
                    pass
    finally:
        await client.aclose()

    assert endpoint not in str(exc_info.value)
    assert "private" not in str(exc_info.value)
    assert traces.recorded_exceptions == []
    assert endpoint not in repr(traces.span_attributes)


@pytest.mark.asyncio
async def test_sensitive_async_otel_suppression_resets_after_transport_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = "http://93.184.216.34/runtime-secret-path/embeddings"
    transport_started = asyncio.Event()
    cancellation_states: list[object] = []
    reset_states: list[object] = []

    monkeypatch.setattr(http_client, "get_metrics_registry", _MetricRecorder)
    monkeypatch.setattr(http_client, "get_tracing_manager", _TraceRecorder)

    async def blocking_handler(request: httpx.Request) -> httpx.Response:
        cancellation_states.append(
            otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY)
        )
        transport_started.set()
        await asyncio.Event().wait()
        return httpx.Response(200, request=request)

    blocking_client = httpx.AsyncClient(transport=httpx.MockTransport(blocking_handler))

    async def send_until_cancelled() -> None:
        try:
            await http_client.afetch(
                method="GET",
                url=endpoint,
                client=blocking_client,
                sensitive_observability=True,
            )
        finally:
            reset_states.append(
                otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY)
            )

    request_task = asyncio.create_task(send_until_cancelled())
    try:
        await transport_started.wait()
        request_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request_task
    finally:
        await blocking_client.aclose()

    assert cancellation_states == [True]
    assert reset_states == [None]
    assert otel_context.get_value(otel_context._SUPPRESS_HTTP_INSTRUMENTATION_KEY) is None


@pytest.mark.asyncio
async def test_public_async_httpx_redirect_observes_final_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_url = "http://93.184.216.34/redirect-start"
    final_url = "http://8.8.8.8/redirect-final"
    metrics = _MetricRecorder()
    outbound_records: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == start_url:
            return httpx.Response(302, request=request, headers={"location": final_url})
        return httpx.Response(200, request=request, json={"ok": True})

    monkeypatch.setattr(http_client, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(http_client, "get_tracing_manager", _TraceRecorder)
    monkeypatch.setattr(
        http_client,
        "_log_outbound_request",
        lambda **kwargs: outbound_records.append(kwargs),
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        response = await http_client.afetch(
            method="GET",
            url=start_url,
            client=client,
            allow_redirects=True,
        )
    finally:
        await client.aclose()

    metric_hosts = {
        labels.get("host")
        for name, labels in metrics.calls
        if name in {"http_client_requests_total", "http_client_request_duration_seconds"}
    }
    assert response.status_code == 200
    assert metric_hosts == {"8.8.8.8"}
    assert outbound_records[-1]["url"] == final_url


@pytest.mark.asyncio
async def test_public_async_aiohttp_redirect_observes_final_destination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start_url = "http://93.184.216.34/redirect-start"
    final_url = "http://8.8.8.8/redirect-final"
    metrics = _MetricRecorder()
    outbound_records: list[dict[str, Any]] = []

    class RawResponse:
        def __init__(self, status: int, url: str, headers: dict[str, str]) -> None:
            self.status = status
            self.url = url
            self.headers = headers
            self.charset = "utf-8"

        def release(self) -> None:
            return None

    async def fake_io(**kwargs: Any) -> Any:
        url = kwargs["url"]
        if url == start_url:
            raw = RawResponse(302, url, {"location": final_url})
        else:
            raw = RawResponse(200, url, {"content-type": "application/json"})
        return http_client._AiohttpResponse(raw, b'{"ok": true}')

    monkeypatch.setattr(http_client, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(http_client, "_aiohttp_request_io", fake_io)
    monkeypatch.setattr(http_client, "get_metrics_registry", lambda: metrics)
    monkeypatch.setattr(http_client, "get_tracing_manager", _TraceRecorder)
    monkeypatch.setattr(
        http_client,
        "_log_outbound_request",
        lambda **kwargs: outbound_records.append(kwargs),
    )
    response = await http_client.afetch(
        method="GET",
        url=start_url,
        client=object(),
        allow_redirects=True,
    )

    metric_hosts = {
        labels.get("host")
        for name, labels in metrics.calls
        if name in {"http_client_requests_total", "http_client_request_duration_seconds"}
    }
    assert response.status_code == 200
    assert metric_hosts == {"8.8.8.8"}
    assert outbound_records[-1]["url"] == final_url
