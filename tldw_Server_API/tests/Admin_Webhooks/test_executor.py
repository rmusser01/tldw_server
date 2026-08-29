from __future__ import annotations

import asyncio
import dataclasses
import importlib
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from types import ModuleType
from typing import Any

import pytest

from tldw_Server_API.app.core import exceptions as core_exceptions
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    DeliveryKind,
    ValidatedWebhookTarget,
)
from tldw_Server_API.app.core.Security import http_hop
from tldw_Server_API.app.core.Security.egress import URLPolicyResult

SECRET = "whsec_1111111111111111111111111111111111111111111111111111111111111111"
BODY = (
    b'{"api_version":"2026-07-01","created_at":"2026-08-23T00:00:00Z",'
    b'"data":{"synthetic":true},"id":"00000000-0000-4000-8000-000000000001",'
    b'"type":"user.created"}'
)
EVENT_ID = "00000000-0000-4000-8000-000000000001"
DELIVERY_ID = "00000000-0000-4000-8000-000000000002"


@pytest.fixture
def executor_module() -> ModuleType:
    return importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.executor"
    )


class FakeClock:
    def __init__(
        self,
        *,
        utc_values: Sequence[datetime] = (
            datetime(2026, 8, 23, tzinfo=timezone.utc),
        ),
        monotonic_values: Sequence[float] = (100.0, 100.125),
    ) -> None:
        self.utc_values = list(utc_values)
        self.monotonic_values = list(monotonic_values)
        self.utc_calls = 0
        self.monotonic_calls = 0

    def utc_now(self) -> datetime:
        self.utc_calls += 1
        return self.utc_values.pop(0)

    def monotonic(self) -> float:
        self.monotonic_calls += 1
        return self.monotonic_values.pop(0)


class EgressRecorder:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[http_hop.NormalizedHTTPHopRequest] = []

    async def __call__(
        self,
        request: http_hop.NormalizedHTTPHopRequest,
    ) -> http_hop.StatusOnlyHTTPHopResponse:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]


def _command(module: ModuleType, **overrides: object) -> Any:
    values: dict[str, object] = {
        "target": ValidatedWebhookTarget(
            url="https://receiver.example/hooks/admin",
            hostname="receiver.example",
            target_display="https://receiver.example",
        ),
        "body": BODY,
        "signing_secret": SECRET,
        "timeout_seconds": 10,
        "event_type": "user.created",
        "event_id": EVENT_ID,
        "delivery_id": DELIVERY_ID,
        "attempt_number": 1,
        "secret_version": 7,
        "kind": DeliveryKind.AUTOMATIC,
    }
    values.update(overrides)
    return module.AttemptExecutionRequest(**values)


def _allow_policy(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str] | None = None,
) -> None:
    def allow(url: str) -> URLPolicyResult:
        if calls is not None:
            calls.append(url)
        return URLPolicyResult(True, resolved_ips=("8.8.8.8",))

    monkeypatch.setattr(module, "evaluate_platform_webhook_url_policy", allow)


def _executor(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    egress: EgressRecorder,
    *,
    clock: FakeClock | None = None,
    allow_http_dev: bool = False,
) -> Any:
    _allow_policy(module, monkeypatch)
    return module.DeliveryAttemptExecutor(
        egress=egress,
        clock=clock or FakeClock(),
        allow_http_dev=allow_http_dev,
    )


def _status(
    status_code: int,
    *,
    latency_ms: int = 999,
    retry_after_seconds: int | None = None,
) -> http_hop.StatusOnlyHTTPHopResponse:
    return http_hop.StatusOnlyHTTPHopResponse(
        status_code=status_code,
        latency_ms=latency_ms,
        retry_after_seconds=retry_after_seconds,
    )


def _header(request: http_hop.NormalizedHTTPHopRequest, name: str) -> str:
    return dict(request.headers)[name]


@pytest.mark.unit
def test_attempt_contracts_are_frozen_bounded_and_hide_sensitive_inputs(
    executor_module: ModuleType,
) -> None:
    request = _command(executor_module)
    result = executor_module.AttemptExecutionResult(
        outcome=executor_module.AttemptOutcome.SUCCESS,
        status_code=200,
        latency_ms=10,
        reason_code=None,
        retry_delay_seconds=None,
    )

    fields = {field.name: field for field in dataclasses.fields(request)}
    assert fields["target"].repr is False
    assert fields["body"].repr is False
    assert fields["signing_secret"].repr is False
    assert SECRET not in repr(request)
    assert "hooks/admin" not in repr(request)
    assert BODY.decode("ascii") not in repr(request)
    assert tuple(field.name for field in dataclasses.fields(result)) == (
        "outcome",
        "status_code",
        "latency_ms",
        "reason_code",
        "retry_delay_seconds",
    )
    assert not hasattr(result, "headers")
    assert not hasattr(result, "body")
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.attempt_number = 2


@pytest.mark.unit
async def test_published_signature_vector_and_exact_request_headers(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    egress = EgressRecorder(_status(200))
    executor = _executor(
        executor_module,
        monkeypatch,
        egress,
        clock=clock,
    )

    result = await executor.execute(_command(executor_module))

    assert len(egress.requests) == 1
    request = egress.requests[0]
    assert request.scheme == "https"
    assert request.host == "receiver.example"
    assert request.port == 443
    assert request.method == "POST"
    assert request.target == "/hooks/admin"
    assert request.body is BODY
    assert request.headers == (
        ("content-type", "application/json"),
        ("x-tldw-webhook-event", "user.created"),
        ("x-tldw-webhook-event-id", EVENT_ID),
        ("x-tldw-webhook-delivery-id", DELIVERY_ID),
        ("x-tldw-webhook-timestamp", "1787443200"),
        ("x-tldw-webhook-secret-version", "7"),
        (
            "x-tldw-webhook-signature",
            "v1=294bc280642cfd89fd011f606fbbe39633a77372db8ae9efd4281b2a3e509811",
        ),
    )
    assert result.outcome is executor_module.AttemptOutcome.SUCCESS
    assert result.status_code == 200
    assert result.latency_ms == 125
    assert result.reason_code is None
    assert result.retry_delay_seconds is None


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (DeliveryKind.AUTOMATIC, False),
        (DeliveryKind.MANUAL, False),
        (DeliveryKind.TEST, True),
    ],
)
@pytest.mark.unit
async def test_test_header_is_present_only_for_test_deliveries(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    kind: DeliveryKind,
    expected: bool,
) -> None:
    egress = EgressRecorder(_status(200))
    executor = _executor(executor_module, monkeypatch, egress)

    await executor.execute(_command(executor_module, kind=kind))

    headers = dict(egress.requests[0].headers)
    assert (headers.get("x-tldw-webhook-test") == "true") is expected


@pytest.mark.unit
async def test_retries_regenerate_only_timestamp_and_signature(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock(
        utc_values=(
            datetime(2026, 8, 23, tzinfo=timezone.utc),
            datetime(2026, 8, 23, tzinfo=timezone.utc) + timedelta(seconds=60),
        ),
        monotonic_values=(10.0, 10.010, 20.0, 20.020),
    )
    egress = EgressRecorder(_status(500), _status(500))
    executor = _executor(
        executor_module,
        monkeypatch,
        egress,
        clock=clock,
    )
    first = _command(executor_module, attempt_number=1)
    second = dataclasses.replace(first, attempt_number=2)

    await executor.execute(first)
    await executor.execute(second)

    first_request, second_request = egress.requests
    assert first_request.body is BODY
    assert second_request.body is BODY
    for name in (
        "x-tldw-webhook-event",
        "x-tldw-webhook-event-id",
        "x-tldw-webhook-delivery-id",
        "x-tldw-webhook-secret-version",
    ):
        assert _header(first_request, name) == _header(second_request, name)
    assert _header(first_request, "x-tldw-webhook-timestamp") == "1787443200"
    assert _header(second_request, "x-tldw-webhook-timestamp") == "1787443260"
    assert _header(first_request, "x-tldw-webhook-signature") != _header(
        second_request,
        "x-tldw-webhook-signature",
    )
    assert first.body is BODY
    assert second.body is BODY


@pytest.mark.parametrize(
    ("status_code", "outcome", "reason", "retry_delay"),
    [
        (200, "success", None, None),
        (299, "success", None, None),
        (300, "failed", "http_redirect", None),
        (399, "failed", "http_redirect", None),
        (400, "failed", "http_client_error", None),
        (407, "failed", "http_client_error", None),
        (408, "retryable", "http_request_timeout", 60),
        (429, "retryable", "http_rate_limited", 60),
        (500, "retryable", "http_server_error", 60),
        (503, "retryable", "http_server_error", 60),
        (599, "retryable", "http_server_error", 60),
        (199, "failed", "http_status_invalid", None),
    ],
)
@pytest.mark.unit
async def test_http_status_classification_is_closed(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    outcome: str,
    reason: str | None,
    retry_delay: int | None,
) -> None:
    egress = EgressRecorder(_status(status_code))
    executor = _executor(executor_module, monkeypatch, egress)

    result = await executor.execute(_command(executor_module))

    assert result.outcome.value == outcome
    assert result.reason_code is None if reason is None else result.reason_code.value == reason
    assert result.retry_delay_seconds == retry_delay


@pytest.mark.parametrize(
    ("attempt_number", "expected_delay"),
    [(1, 60), (2, 300), (3, 1_800)],
)
@pytest.mark.unit
async def test_retry_schedule_uses_the_network_attempt_number(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    attempt_number: int,
    expected_delay: int,
) -> None:
    egress = EgressRecorder(_status(500))
    executor = _executor(executor_module, monkeypatch, egress)

    result = await executor.execute(
        _command(executor_module, attempt_number=attempt_number)
    )

    assert result.outcome.value == "retryable"
    assert result.retry_delay_seconds == expected_delay


@pytest.mark.parametrize(
    ("status_code", "attempt_number", "receiver_delay", "expected_delay"),
    [
        (429, 1, 600, 600),
        (503, 2, 1_200, 1_200),
        (429, 3, 10, 1_800),
    ],
)
@pytest.mark.unit
async def test_only_429_and_503_receiver_evidence_can_raise_retry_delay(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    attempt_number: int,
    receiver_delay: int,
    expected_delay: int,
) -> None:
    egress = EgressRecorder(
        _status(status_code, retry_after_seconds=receiver_delay)
    )
    executor = _executor(executor_module, monkeypatch, egress)

    result = await executor.execute(
        _command(executor_module, attempt_number=attempt_number)
    )

    assert result.retry_delay_seconds == expected_delay


@pytest.mark.unit
async def test_fourth_retryable_status_is_terminal_attempt_budget_exhausted(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    egress = EgressRecorder(_status(503, retry_after_seconds=1_800))
    executor = _executor(executor_module, monkeypatch, egress)

    result = await executor.execute(
        _command(executor_module, attempt_number=4)
    )

    assert len(egress.requests) == 1
    assert result.outcome.value == "failed"
    assert result.reason_code.value == "attempt_budget_exhausted"
    assert result.retry_delay_seconds is None


_RETRYABLE_HTTP_HOP_CODES = frozenset(
    {
        "dns_resolution_failed",
        "dns_timeout",
        "connect_timeout",
        "read_timeout",
        "write_timeout",
        "total_timeout",
        "tls_error",
        "transport_error",
    }
)
_ALL_HTTP_HOP_CODES = (
    "invalid_request",
    "dns_resolution_failed",
    "dns_timeout",
    "dns_address_denied",
    "connect_timeout",
    "read_timeout",
    "write_timeout",
    "total_timeout",
    "peer_verification_failed",
    "tls_error",
    "protocol_error",
    "response_headers_too_large",
    "response_too_large",
    "decompressed_response_too_large",
    "parser_input_too_large",
    "unsupported_content_encoding",
    "invalid_content_encoding",
    "transport_error",
)


@pytest.mark.parametrize("code", _ALL_HTTP_HOP_CODES)
@pytest.mark.unit
async def test_every_http_hop_error_maps_to_a_closed_local_reason(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
) -> None:
    error = core_exceptions.HTTPHopError(
        code,  # type: ignore[arg-type]
        retryable=code in _RETRYABLE_HTTP_HOP_CODES,
    )
    egress = EgressRecorder(error)
    executor = _executor(executor_module, monkeypatch, egress)

    result = await executor.execute(_command(executor_module))

    expected_outcome = "retryable" if code in _RETRYABLE_HTTP_HOP_CODES else "failed"
    assert result.outcome.value == expected_outcome
    assert result.reason_code.value == f"http_hop_{code}"
    assert result.status_code is None
    assert len(egress.requests) == 1


@pytest.mark.unit
async def test_unknown_egress_exception_is_sanitized_without_exception_text(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canary = "receiver-exception-sensitive-detail"
    egress = EgressRecorder(RuntimeError(canary))
    executor = _executor(executor_module, monkeypatch, egress)

    result = await executor.execute(_command(executor_module))

    assert result.outcome.value == "retryable"
    assert result.reason_code.value == "transport_error"
    assert canary not in repr(result)


@pytest.mark.unit
async def test_fourth_retryable_transport_error_is_terminal_budget_exhausted(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    egress = EgressRecorder(core_exceptions.HTTPHopError("read_timeout", retryable=True))
    executor = _executor(executor_module, monkeypatch, egress)

    result = await executor.execute(
        _command(executor_module, attempt_number=4)
    )

    assert len(egress.requests) == 1
    assert result.outcome.value == "failed"
    assert result.reason_code.value == "attempt_budget_exhausted"


@pytest.mark.unit
async def test_egress_cancellation_propagates_after_exactly_one_call(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    egress = EgressRecorder(asyncio.CancelledError())
    executor = _executor(executor_module, monkeypatch, egress)

    with pytest.raises(asyncio.CancelledError):
        await executor.execute(_command(executor_module))

    assert len(egress.requests) == 1


@pytest.mark.parametrize("attempt_number", [0, 5, -1, True, 1.0])
@pytest.mark.unit
async def test_invalid_attempt_number_fails_before_policy_clock_or_io(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    attempt_number: object,
) -> None:
    policy_calls: list[str] = []
    _allow_policy(executor_module, monkeypatch, policy_calls)
    clock = FakeClock()
    egress = EgressRecorder(_status(200))
    executor = executor_module.DeliveryAttemptExecutor(
        egress=egress,
        clock=clock,
        allow_http_dev=False,
    )

    with pytest.raises(ValueError, match="attempt number"):
        await executor.execute(
            _command(executor_module, attempt_number=attempt_number)
        )

    assert policy_calls == []
    assert clock.utc_calls == 0
    assert clock.monotonic_calls == 0
    assert egress.requests == []


@pytest.mark.unit
async def test_url_is_idna_and_percent_normalized_before_policy_and_egress(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_calls: list[str] = []
    _allow_policy(executor_module, monkeypatch, policy_calls)
    egress = EgressRecorder(_status(200))
    executor = executor_module.DeliveryAttemptExecutor(
        egress=egress,
        clock=FakeClock(),
        allow_http_dev=False,
    )
    target = ValidatedWebhookTarget(
        url="https://B\u00dcCHER.example/caf\u00e9?q=hello world&next=%2f",
        hostname="xn--bcher-kva.example",
        target_display="https://xn--bcher-kva.example",
    )

    await executor.execute(_command(executor_module, target=target))

    expected_url = (
        "https://xn--bcher-kva.example/caf%C3%A9?q=hello%20world&next=%2F"
    )
    assert policy_calls == [expected_url]
    assert egress.requests[0].host == "xn--bcher-kva.example"
    assert egress.requests[0].target == "/caf%C3%A9?q=hello%20world&next=%2F"


@pytest.mark.parametrize(
    "url",
    [
        "https://receiver.example/hook%",
        "https://receiver.example/hook%0g",
        "https://user@receiver.example/hook",
        "https://receiver.example/hook#fragment",
        "https://receiver.example/hook\\path",
        "https://receiver.example/hook\r\nnext",
        "https://receiver.example/hook%0Dnext",
        "https://receiver.example/hook%5cnext",
    ],
)
@pytest.mark.unit
async def test_malformed_or_ambiguous_target_fails_before_policy_and_io(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    policy_calls: list[str] = []
    _allow_policy(executor_module, monkeypatch, policy_calls)
    egress = EgressRecorder(_status(200))
    executor = executor_module.DeliveryAttemptExecutor(
        egress=egress,
        clock=FakeClock(),
        allow_http_dev=False,
    )
    target = ValidatedWebhookTarget(
        url=url,
        hostname="receiver.example",
        target_display="https://receiver.example",
    )

    result = await executor.execute(_command(executor_module, target=target))

    assert result.outcome.value == "failed"
    assert result.reason_code.value == "target_invalid"
    assert result.status_code is None
    assert policy_calls == []
    assert egress.requests == []


@pytest.mark.unit
async def test_policy_denial_returns_closed_terminal_result_without_io(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        executor_module,
        "evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(False, "sensitive policy detail", reason_code="denied"),
    )
    egress = EgressRecorder(_status(200))
    executor = executor_module.DeliveryAttemptExecutor(
        egress=egress,
        clock=FakeClock(),
        allow_http_dev=False,
    )

    result = await executor.execute(_command(executor_module))

    assert result.outcome.value == "failed"
    assert result.reason_code.value == "target_rejected"
    assert "sensitive policy detail" not in repr(result)
    assert egress.requests == []


@pytest.mark.unit
async def test_http_requires_the_explicit_development_override(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = ValidatedWebhookTarget(
        url="http://receiver.example/hooks/admin",
        hostname="receiver.example",
        target_display="http://receiver.example",
    )
    denied_egress = EgressRecorder(_status(200))
    denied = _executor(executor_module, monkeypatch, denied_egress)

    denied_result = await denied.execute(
        _command(executor_module, target=target)
    )

    assert denied_result.reason_code.value == "target_invalid"
    assert denied_egress.requests == []

    allowed_egress = EgressRecorder(_status(200))
    allowed = _executor(
        executor_module,
        monkeypatch,
        allowed_egress,
        allow_http_dev=True,
    )
    allowed_result = await allowed.execute(
        _command(executor_module, target=target)
    )

    assert allowed_result.outcome.value == "success"
    assert allowed_egress.requests[0].scheme == "http"
    assert allowed_egress.requests[0].port == 80


@pytest.mark.parametrize("timeout_seconds", [0, 31, True, 10.0])
@pytest.mark.unit
async def test_registration_timeout_is_rejected_outside_exact_integer_bounds(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    timeout_seconds: object,
) -> None:
    egress = EgressRecorder(_status(200))
    executor = _executor(executor_module, monkeypatch, egress)

    with pytest.raises(ValueError, match="timeout"):
        await executor.execute(
            _command(executor_module, timeout_seconds=timeout_seconds)
        )

    assert egress.requests == []


@pytest.mark.unit
async def test_all_transport_timeouts_are_bounded_by_registration_timeout(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    egress = EgressRecorder(_status(200))
    executor = _executor(executor_module, monkeypatch, egress)

    await executor.execute(
        _command(executor_module, timeout_seconds=30)
    )

    limits = egress.requests[0].limits
    assert limits.total_timeout_seconds == 30
    assert max(
        limits.dns_timeout_seconds,
        limits.connect_timeout_seconds,
        limits.read_timeout_seconds,
        limits.write_timeout_seconds,
        limits.total_timeout_seconds,
    ) <= 30


@pytest.mark.parametrize(
    "overrides",
    [
        {"body": bytearray(BODY)},
        {"body": b"x" * (64 * 1024 + 1)},
        {"signing_secret": "whsec_" + "g" * 64},
        {"event_id": "not-a-uuid"},
        {"delivery_id": "00000000-0000-4000-8000-000000000001".upper()},
        {"secret_version": 0},
        {"kind": "test"},
    ],
)
@pytest.mark.unit
async def test_invalid_sensitive_or_identity_input_fails_before_io(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
) -> None:
    egress = EgressRecorder(_status(200))
    executor = _executor(executor_module, monkeypatch, egress)

    with pytest.raises(ValueError):
        await executor.execute(_command(executor_module, **overrides))

    assert egress.requests == []


@pytest.mark.unit
async def test_executor_uses_its_monotonic_clock_not_egress_latency(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    egress = EgressRecorder(_status(200, latency_ms=29_999))
    executor = _executor(
        executor_module,
        monkeypatch,
        egress,
        clock=FakeClock(monotonic_values=(5.0, 5.250)),
    )

    result = await executor.execute(_command(executor_module))

    assert result.latency_ms == 250


@pytest.mark.parametrize(
    "clock",
    [
        FakeClock(utc_values=(datetime(2026, 8, 23),)),
        FakeClock(monotonic_values=(float("nan"),)),
    ],
    ids=("naive-utc", "nonfinite-monotonic"),
)
@pytest.mark.unit
async def test_invalid_clock_fails_closed_before_io(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    clock: FakeClock,
) -> None:
    egress = EgressRecorder(_status(200))
    executor = _executor(
        executor_module,
        monkeypatch,
        egress,
        clock=clock,
    )

    result = await executor.execute(_command(executor_module))

    assert result.outcome.value == "retryable"
    assert result.reason_code.value == "clock_error"
    assert result.latency_ms is None
    assert egress.requests == []


@pytest.mark.unit
async def test_backward_monotonic_clock_fails_closed_after_one_io(
    executor_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    egress = EgressRecorder(_status(200))
    executor = _executor(
        executor_module,
        monkeypatch,
        egress,
        clock=FakeClock(monotonic_values=(5.0, 4.0)),
    )

    result = await executor.execute(_command(executor_module))

    assert len(egress.requests) == 1
    assert result.outcome.value == "retryable"
    assert result.reason_code.value == "clock_error"
    assert result.latency_ms is None
