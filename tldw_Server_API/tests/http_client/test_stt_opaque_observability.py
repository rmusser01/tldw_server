"""Concurrency and privacy tests for planned STT HTTP observability."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from loguru import logger

from tldw_Server_API.app.core import http_client


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
