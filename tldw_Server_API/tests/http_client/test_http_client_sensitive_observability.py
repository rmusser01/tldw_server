from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Barrier
from typing import Any

import httpx
import pytest
from loguru import logger

from tldw_Server_API.app.core import http_client

pytestmark = pytest.mark.unit


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

    @contextmanager
    def span(self, _name: str, *, attributes: dict[str, Any]):
        self.span_attributes.append(dict(attributes))
        yield None

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
    metrics = _MetricRecorder()
    traces = _TraceRecorder()
    log_records: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(str(request.url))
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


def test_sensitive_log_filter_does_not_hide_concurrent_public_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_endpoint = "http://93.184.216.34/runtime-secret-path/embeddings"
    public_endpoint = "http://8.8.8.8/public-health?scope=visible"
    request_barrier = Barrier(2)
    log_records: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
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
