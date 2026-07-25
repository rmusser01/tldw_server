"""Concurrency and privacy tests for planned STT HTTP observability."""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import httpx
import pytest
from loguru import logger

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.Security import egress


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class _TracingRecorder:
    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []

    @asynccontextmanager
    async def async_span(
        self,
        _name: str,
        *,
        attributes: dict[str, Any],
    ):
        self.spans.append(dict(attributes))
        yield self

    def set_attributes(self, _attributes: dict[str, Any]) -> None:
        return None

    def add_event(
        self,
        _name: str,
        _attributes: dict[str, Any],
    ) -> None:
        return None


class _MetricsRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    def increment(
        self,
        _name: str,
        _value: int,
        *,
        labels: dict[str, str],
    ) -> None:
        self.calls.append(("increment", dict(labels)))

    def observe(
        self,
        _name: str,
        _value: float,
        *,
        labels: dict[str, str],
    ) -> None:
        self.calls.append(("observe", dict(labels)))


class _ExceptionRecordingTracing:
    def __init__(self) -> None:
        self.recorded_exceptions: list[str] = []

    @contextmanager
    def span(
        self,
        _name: str,
        *,
        attributes: dict[str, Any],
    ):
        _ = attributes
        try:
            yield self
        except Exception as exc:
            self.recorded_exceptions.append(str(exc))
            raise


@pytest.mark.unit
def test_opaque_stt_exception_stays_outside_application_span() -> None:
    endpoint_id = "sha256:" + "e" * 64
    secret = "https://private.example/secret-path?token=private"
    tracing = _ExceptionRecordingTracing()

    with pytest.raises(RuntimeError, match="private.example"):
        with http_client.opaque_stt_http_observability(endpoint_id):
            with http_client._http_client_observability_span(
                tracing,
                "http.client",
                attributes={"stt.endpoint_id": endpoint_id},
                sensitive_observability=False,
            ):
                raise RuntimeError(secret)

    assert tracing.recorded_exceptions == []


@pytest.mark.unit
def test_planned_observability_suppresses_optional_http_instrumentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint_id = "sha256:" + "b" * 64
    events: list[str] = []

    @contextmanager
    def fake_suppression():
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(
        http_client,
        "_suppress_http_instrumentation",
        fake_suppression,
        raising=False,
    )

    with http_client.opaque_stt_http_observability(endpoint_id):
        events.append("body")
        assert http_client._opaque_stt_endpoint_id() == endpoint_id

    assert events == ["enter", "body", "exit"]


@pytest.mark.unit
def test_opaque_stt_log_factory_redacts_httpx_child_logger() -> None:
    endpoint_id = "sha256:" + "f" * 64
    secret_url = "https://private.example/audio?token=private"
    stdlib_handler = _ListHandler()
    transport_logger = logging.getLogger("httpx.transport")
    previous_level = transport_logger.level
    transport_logger.setLevel(logging.DEBUG)
    transport_logger.addHandler(stdlib_handler)

    try:
        with http_client.opaque_stt_http_observability(endpoint_id):
            transport_logger.debug("request URL=%s", secret_url)
    finally:
        transport_logger.removeHandler(stdlib_handler)
        transport_logger.setLevel(previous_level)

    assert stdlib_handler.messages == [
        f"planned STT HTTP endpoint_id={endpoint_id}"
    ]


@pytest.mark.unit
def test_planned_egress_dns_logs_are_opaque_across_worker_context_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint_id = "sha256:" + "c" * 64
    secret_host = "private-resolver-host.example"
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
        format="{message}|{extra}",
    )

    class ReleaseFailureSlots:
        def acquire(
            self,
            blocking: bool = True,
            timeout: float | None = None,
        ) -> bool:
            return True

        def release(self) -> None:
            raise ValueError("forced double release")

    monkeypatch.setattr(
        egress,
        "_DNS_RESOLVER_SLOTS",
        ReleaseFailureSlots(),
    )
    monkeypatch.setattr(
        egress.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [],
    )

    try:
        with http_client.opaque_stt_http_observability(endpoint_id):
            assert (
                egress._getaddrinfo_with_timeout(
                    secret_host,
                    timeout_s=0.5,
                )
                == []
            )
    finally:
        logger.remove(sink_id)

    combined = "\n".join(messages)
    assert "dns_resolver_slot_release_failed" in combined
    assert endpoint_id in combined
    assert secret_host not in combined


@pytest.mark.unit
@pytest.mark.parametrize(
    "failure",
    ("resolver", "saturation"),
)
def test_planned_egress_failure_logs_never_expose_host(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    endpoint_id = "sha256:" + "d" * 64
    secret_host = "private-egress-host.example"
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
        format="{message}|{extra}",
    )

    if failure == "resolver":
        monkeypatch.setattr(
            egress,
            "_DNS_RESOLVER_SLOTS",
            threading.BoundedSemaphore(1),
        )

        def fail_resolver(
            *_args: object,
            **_kwargs: object,
        ) -> list[object]:
            raise OSError("forced resolver failure")

        monkeypatch.setattr(
            egress.socket,
            "getaddrinfo",
            fail_resolver,
        )
    else:
        class SaturatedSlots:
            def acquire(
                self,
                blocking: bool = True,
                timeout: float | None = None,
            ) -> bool:
                return False

        monkeypatch.setattr(
            egress,
            "_DNS_RESOLVER_SLOTS",
            SaturatedSlots(),
        )

    try:
        with http_client.opaque_stt_http_observability(endpoint_id):
            assert (
                egress._getaddrinfo_with_timeout(
                    secret_host,
                    timeout_s=0.5,
                )
                == []
            )
    finally:
        logger.remove(sink_id)

    combined = "\n".join(messages)
    assert endpoint_id in combined
    assert secret_host not in combined


@pytest.mark.unit
@pytest.mark.asyncio
async def test_afetch_explicit_transport_selector_is_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected: list[str] = []

    class Adapter:
        async def arequest(self, **_kwargs: Any) -> object:
            return object()

    def select(name: str) -> Adapter:
        selected.append(name)
        return Adapter()

    monkeypatch.setattr(
        http_client,
        "_get_transport_adapter",
        select,
    )

    await http_client.afetch(
        method="POST",
        url="http://127.0.0.1/transcriptions",
        transport="httpx",
    )

    assert selected == ["httpx"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_planned_observability_is_opaque_and_concurrency_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint_id = "sha256:" + "a" * 64
    secret_url = (
        "http://93.184.216.34/private/auth/request-model"
        "?api_key=private-token"
    )
    legacy_url = "http://93.184.216.35/legacy-visible"
    arrived = 0
    both_arrived = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both_arrived.set()
        await asyncio.wait_for(both_arrived.wait(), timeout=2)
        return httpx.Response(
            200,
            json={"text": "ok"},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    planned_client = httpx.AsyncClient(transport=transport)
    legacy_client = httpx.AsyncClient(transport=transport)
    tracing = _TracingRecorder()
    metrics = _MetricsRecorder()
    monkeypatch.setattr(
        http_client,
        "get_tracing_manager",
        lambda: tracing,
    )
    monkeypatch.setattr(
        http_client,
        "get_metrics_registry",
        lambda: metrics,
    )

    stdlib_handler = _ListHandler()
    httpx_logger = logging.getLogger("httpx")
    previous_level = httpx_logger.level
    httpx_logger.setLevel(logging.DEBUG)
    httpx_logger.addHandler(stdlib_handler)
    aiohttp_logger = logging.getLogger("aiohttp.client")
    previous_aiohttp_level = aiohttp_logger.level
    aiohttp_logger.setLevel(logging.DEBUG)
    aiohttp_logger.addHandler(stdlib_handler)
    loguru_messages: list[str] = []
    sink_id = logger.add(
        lambda message: loguru_messages.append(str(message)),
        level="DEBUG",
        format="{message}|{extra}",
    )

    async def planned_request() -> None:
        with http_client.opaque_stt_http_observability(endpoint_id):
            response = await http_client.afetch(
                method="POST",
                url=secret_url,
                client=planned_client,
                allow_redirects=False,
                retry=http_client.RetryPolicy(attempts=1),
            )
            await response.aclose()

    async def legacy_request() -> None:
        response = await http_client.afetch(
            method="GET",
            url=legacy_url,
            client=legacy_client,
            retry=http_client.RetryPolicy(attempts=1),
        )
        await response.aclose()

    try:
        await asyncio.gather(planned_request(), legacy_request())
        with http_client.opaque_stt_http_observability(endpoint_id):
            aiohttp_logger.debug(
                "request URL=%s",
                "https://aiohttp-secret.example/private"
                "?token=aiohttp-private-token",
            )
        aiohttp_logger.debug(
            "request URL=%s",
            "https://legacy.example/aiohttp-legacy-visible",
        )
    finally:
        logger.remove(sink_id)
        httpx_logger.removeHandler(stdlib_handler)
        httpx_logger.setLevel(previous_level)
        aiohttp_logger.removeHandler(stdlib_handler)
        aiohttp_logger.setLevel(previous_aiohttp_level)
        await planned_client.aclose()
        await legacy_client.aclose()

    combined = "\n".join(
        [*stdlib_handler.messages, *loguru_messages]
    )
    assert endpoint_id in combined
    assert "legacy-visible" in combined
    assert "aiohttp-legacy-visible" in combined
    for secret in (
        "93.184.216.34",
        "private/auth",
        "request-model",
        "private-token",
        "api_key",
        "aiohttp-secret.example",
        "aiohttp-private-token",
    ):
        assert secret not in combined

    assert {"stt.endpoint_id": endpoint_id} in tracing.spans
    assert any(
        span.get("url.full", "").endswith("/legacy-visible")
        for span in tracing.spans
    )
    assert all(
        "net.host.name" not in span
        and "url.full" not in span
        for span in tracing.spans
        if span.get("stt.endpoint_id") == endpoint_id
    )
    planned_metric_labels = [
        labels
        for _operation, labels in metrics.calls
        if labels.get("endpoint_id") == endpoint_id
    ]
    legacy_metric_labels = [
        labels
        for _operation, labels in metrics.calls
        if labels.get("method") == "GET"
    ]
    assert planned_metric_labels
    assert all(
        "host" not in labels and "method" not in labels
        for labels in planned_metric_labels
    )
    assert any(
        labels.get("host") == "93.184.216.35"
        for labels in legacy_metric_labels
    )
