from __future__ import annotations

import asyncio
import dataclasses
import inspect
import threading
from collections.abc import Sequence

import pytest

from tldw_Server_API.app.core import exceptions as core_exceptions
from tldw_Server_API.app.core.Security import http_hop

pytestmark = pytest.mark.unit


def _request(**overrides: object) -> http_hop.NormalizedHTTPHopRequest:
    values: dict[str, object] = {
        "scheme": "https",
        "host": "api.example.com",
        "port": 443,
        "method": "GET",
        "target": "/works?q=bounded",
        "headers": (("accept", "application/json"),),
    }
    values.update(overrides)
    return http_hop.NormalizedHTTPHopRequest(**values)  # type: ignore[arg-type]


def test_request_contract_is_immutable_and_uses_bounded_defaults() -> None:
    request = _request()

    assert request.limits.max_response_header_bytes == 64 * 1024
    assert request.headers == (("accept", "application/json"),)
    with pytest.raises(dataclasses.FrozenInstanceError):
        request.port = 8443  # type: ignore[misc]


def test_request_repr_hides_request_material_without_changing_semantics() -> None:
    target = "/opaque-path-sentinel?pageToken=opaque-query-sentinel"
    headers = (
        ("content-type", "application/json"),
        ("x-tldw-webhook-signature", "v1=opaque-signature-sentinel"),
    )
    body = b'{"event":"opaque-body-sentinel"}'
    request = _request(method="POST", target=target, headers=headers, body=body)
    equal_request = _request(
        method="POST",
        target=target,
        headers=headers,
        body=body,
    )

    assert request.target == target
    assert request.headers == headers
    assert request.body == body
    assert request == equal_request
    assert hash(request) == hash(equal_request)
    rendered = repr(request)
    for canary in (
        "opaque-path-sentinel",
        "opaque-query-sentinel",
        "opaque-signature-sentinel",
        "opaque-body-sentinel",
    ):
        assert canary not in rendered


def test_status_only_response_is_frozen_slotted_and_exposes_exactly_three_fields() -> None:
    response = http_hop.StatusOnlyHTTPHopResponse(
        status_code=503,
        latency_ms=125,
        retry_after_seconds=300,
    )

    assert tuple(field.name for field in dataclasses.fields(response)) == (
        "status_code",
        "latency_ms",
        "retry_after_seconds",
    )
    assert not hasattr(response, "__dict__")
    for forbidden in (
        "headers",
        "body",
        "stream",
        "target",
        "resolved_ips",
        "connected_ip",
    ):
        assert not hasattr(response, forbidden)
    with pytest.raises(dataclasses.FrozenInstanceError):
        response.status_code = 200  # type: ignore[misc]


def test_public_status_only_function_accepts_only_the_request() -> None:
    signature = inspect.signature(http_hop.request_http_hop_status)

    assert tuple(signature.parameters) == ("request",)
    assert signature.parameters["request"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert "request_http_hop_status" in http_hop.__all__
    assert "StatusOnlyHTTPHopResponse" in http_hop.__all__


@pytest.mark.parametrize(
    "values",
    [
        {"status_code": 99, "latency_ms": 0, "retry_after_seconds": None},
        {"status_code": 600, "latency_ms": 0, "retry_after_seconds": None},
        {"status_code": 200, "latency_ms": -1, "retry_after_seconds": None},
        {"status_code": 200, "latency_ms": 0, "retry_after_seconds": 1},
        {"status_code": 429, "latency_ms": 0, "retry_after_seconds": 0},
        {"status_code": 503, "latency_ms": 0, "retry_after_seconds": 1_801},
    ],
)
def test_status_only_response_rejects_unbounded_or_inconsistent_evidence(
    values: dict[str, int | None],
) -> None:
    with pytest.raises(ValueError):
        http_hop.StatusOnlyHTTPHopResponse(**values)  # type: ignore[arg-type]


def test_http_hop_error_is_the_centralized_core_exception() -> None:
    assert http_hop.HTTPHopError is core_exceptions.HTTPHopError
    assert http_hop.HTTPHopErrorCode is core_exceptions.HTTPHopErrorCode
    assert isinstance(http_hop.HTTPHopError("invalid_request"), core_exceptions.HTTPHopError)


@pytest.mark.parametrize(
    "host",
    [
        "API.example.com",
        "api.example.com.",
        "https://api.example.com",
        "user@api.example.com",
        "api..example.com",
        "api_example.com",
        "xn--a.example",
        "[2001:4860:4860::8888]",
        "fe80::1%en0",
        "127.1",
        "2130706433",
        "0177.0.0.1",
        "0x7f000001",
        "0x7f.0.0.1",
    ],
)
def test_request_rejects_noncanonical_or_ambiguous_hosts(host: str) -> None:
    with pytest.raises(http_hop.HTTPHopError) as exc:
        _request(host=host)

    assert exc.value.code == "invalid_request"


@pytest.mark.parametrize(
    "host",
    [
        "api.example.com",
        "xn--bcher-kva.example",
        "8.8.8.8",
        "2001:4860:4860::8888",
    ],
)
def test_request_accepts_canonical_ascii_hosts(host: str) -> None:
    assert _request(host=host).host == host


def test_invalid_idna_error_has_no_exception_context() -> None:
    with pytest.raises(http_hop.HTTPHopError) as exc:
        _request(host="xn--a.example")

    assert exc.value.code == "invalid_request"
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scheme", "HTTPS"),
        ("scheme", "ftp"),
        ("method", "get"),
        ("method", "DELETE"),
        ("port", True),
        ("port", 0),
        ("port", 65536),
        ("target", "works?q=x"),
        ("target", "//other.example/path"),
        ("target", "/works#fragment"),
        ("target", "/works query=x"),
        ("target", "/works\\query=x"),
        ("target", "/works%2"),
        ("target", "/works%zz"),
        ("target", "/works\r\nInjected: yes"),
        ("body", bytearray(b"not immutable")),
    ],
)
def test_request_rejects_noncanonical_components(field: str, value: object) -> None:
    with pytest.raises(http_hop.HTTPHopError) as exc:
        _request(**{field: value})

    assert exc.value.code == "invalid_request"


def test_unexpected_request_type_does_not_leak_validation_context() -> None:
    secret = "secret-invalid-scheme"

    with pytest.raises(http_hop.HTTPHopError) as exc:
        _request(scheme=[secret])

    assert exc.value.code == "invalid_request"
    assert secret not in str(exc.value)
    assert secret not in repr(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


@pytest.mark.parametrize(
    "name",
    [
        "host",
        "content-length",
        "transfer-encoding",
        "proxy-authorization",
        "connection",
        "proxy-connection",
        "te",
        "trailer",
        "upgrade",
        "keep-alive",
        "expect",
        "accept-encoding",
    ],
)
def test_request_rejects_transport_controlled_headers(name: str) -> None:
    with pytest.raises(http_hop.HTTPHopError) as exc:
        _request(headers=((name, "unsafe"),))

    assert exc.value.code == "invalid_request"


@pytest.mark.parametrize(
    "headers",
    [
        (("Accept", "application/json"),),
        (("x-test", "ok"), ("x-test", "duplicate")),
        (("x-test", "bad\r\nvalue"),),
        (("bad header", "value"),),
    ],
)
def test_request_rejects_ambiguous_headers(headers: tuple[tuple[str, str], ...]) -> None:
    with pytest.raises(http_hop.HTTPHopError) as exc:
        _request(headers=headers)

    assert exc.value.code == "invalid_request"


def test_plaintext_http_rejects_explicit_authorization() -> None:
    with pytest.raises(http_hop.HTTPHopError) as exc:
        _request(
            scheme="http",
            port=80,
            headers=(("authorization", "Bearer explicit-route-secret"),),
        )

    assert exc.value.code == "invalid_request"


def test_https_accepts_explicit_authorization() -> None:
    request = _request(headers=(("authorization", "Bearer explicit-route-secret"),))

    assert request.scheme == "https"
    assert request.headers == (("authorization", "Bearer explicit-route-secret"),)


@pytest.mark.parametrize(
    ("limit_name", "limit_value"),
    [
        ("dns_timeout_seconds", True),
        ("dns_timeout_seconds", 0.0),
        ("read_timeout_seconds", float("nan")),
        ("total_timeout_seconds", float("inf")),
        ("max_request_target_bytes", True),
        ("max_request_headers", 0),
        ("max_wire_bytes", -1),
    ],
)
def test_limits_reject_nonfinite_boolean_or_nonpositive_values(
    limit_name: str,
    limit_value: object,
) -> None:
    values = dataclasses.asdict(http_hop.HTTPHopLimits())
    values[limit_name] = limit_value

    with pytest.raises(ValueError, match=limit_name):
        http_hop.HTTPHopLimits(**values)


def test_limits_reject_parser_ceiling_above_decompressed_ceiling() -> None:
    with pytest.raises(ValueError, match="max_parser_input_bytes"):
        http_hop.HTTPHopLimits(
            max_decompressed_bytes=32,
            max_parser_input_bytes=33,
        )


def test_request_enforces_target_header_count_header_bytes_and_body_limits() -> None:
    tight_limits = http_hop.HTTPHopLimits(
        max_request_target_bytes=4,
        max_request_header_bytes=8,
        max_request_headers=1,
        max_request_body_bytes=2,
    )

    for overrides in (
        {"target": "/tool", "headers": (), "limits": tight_limits},
        {"headers": (("x-a", "1"), ("x-b", "2")), "limits": tight_limits},
        {
            "headers": (("x-long", "value"),),
            "limits": dataclasses.replace(tight_limits, max_request_headers=16),
        },
        {"method": "POST", "headers": (), "body": b"abc", "limits": tight_limits},
    ):
        with pytest.raises(http_hop.HTTPHopError) as exc:
            _request(**overrides)
        assert exc.value.code == "invalid_request"


def test_complete_transport_header_limits_accept_only_exact_or_larger_boundaries() -> None:
    expected_headers = [
        (b"Host", b"api.example.com"),
        (b"Connection", b"close"),
        (b"Accept-Encoding", b"gzip, deflate"),
        (b"Content-Length", b"0"),
    ]
    exact_bytes = sum(len(name) + 2 + len(value) + 2 for name, value in expected_headers)
    exact_limits = http_hop.HTTPHopLimits(
        max_request_headers=len(expected_headers),
        max_request_header_bytes=exact_bytes,
    )

    request = _request(method="POST", headers=(), body=b"", limits=exact_limits)
    assert http_hop._transport_headers(request) == expected_headers

    for limits in (
        dataclasses.replace(exact_limits, max_request_headers=len(expected_headers) - 1),
        dataclasses.replace(exact_limits, max_request_header_bytes=exact_bytes - 1),
    ):
        with pytest.raises(http_hop.HTTPHopError) as exc:
            _request(method="POST", headers=(), body=b"", limits=limits)
        assert exc.value.code == "invalid_request"


def test_get_and_head_reject_request_bodies() -> None:
    for method in ("GET", "HEAD"):
        with pytest.raises(http_hop.HTTPHopError) as exc:
            _request(method=method, body=b"unexpected")
        assert exc.value.code == "invalid_request"


def test_error_text_is_stable_bounded_and_has_no_cause_detail() -> None:
    error = http_hop.HTTPHopError("dns_resolution_failed", retryable=True)

    assert error.code == "dns_resolution_failed"
    assert error.retryable is True
    assert str(error) == "The destination could not be resolved."
    assert len(str(error)) <= 96
    assert "secret" not in repr(error).lower()


@pytest.mark.parametrize(
    "resolved_ips",
    [
        (),
        ("127.0.0.1",),
        ("8.8.8.8", "10.0.0.1"),
        ("not-an-ip",),
        ("::ffff:8.8.8.8",),
        ("64:ff9b::808:808",),
        ("2002:0808:0808::1",),
        ("2001:0000:4136:e378:8000:63bf:3fff:fdd2",),
        ("fec0::1",),
        ("2001:4860:4860::8888%1",),
        ("224.0.0.1",),
        ("2001:4860:4860:0:0:0:0:8888",),
    ],
)
def test_complete_dns_set_rejects_empty_mixed_unsafe_or_noncanonical_answers(
    resolved_ips: Sequence[str],
) -> None:
    with pytest.raises(http_hop.HTTPHopError) as exc:
        http_hop._validate_resolved_ips(resolved_ips)

    assert exc.value.code in {"dns_resolution_failed", "dns_address_denied"}


def test_malformed_resolved_ip_detail_and_context_are_not_exposed() -> None:
    secret = "malformed-secret-resolver-value"

    with pytest.raises(http_hop.HTTPHopError) as exc:
        http_hop._validate_resolved_ips((secret,))

    assert exc.value.code == "dns_address_denied"
    assert secret not in str(exc.value)
    assert secret not in repr(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


def test_complete_dns_set_preserves_order_and_deduplicates_public_answers() -> None:
    assert http_hop._validate_resolved_ips(("8.8.8.8", "2001:4860:4860::8888", "8.8.8.8")) == (
        "8.8.8.8",
        "2001:4860:4860::8888",
    )


async def test_resolver_is_called_once_for_a_hostname() -> None:
    calls: list[tuple[str, int, float]] = []

    async def resolver(host: str, port: int, timeout_seconds: float) -> Sequence[str]:
        calls.append((host, port, timeout_seconds))
        return ("8.8.8.8", "1.1.1.1")

    resolved = await http_hop._resolve_validated_ips(_request(), resolver=resolver)

    assert resolved == ("8.8.8.8", "1.1.1.1")
    assert calls == [("api.example.com", 443, 2.0)]


async def test_canonical_ip_literal_does_not_invoke_dns() -> None:
    async def resolver(_host: str, _port: int, _timeout_seconds: float) -> Sequence[str]:
        raise AssertionError("IP literals must not be re-resolved")

    assert await http_hop._resolve_validated_ips(
        _request(host="8.8.8.8"),
        resolver=resolver,
    ) == ("8.8.8.8",)


async def test_resolver_timeout_returns_sanitized_typed_error() -> None:
    blocker = asyncio.Event()

    async def resolver(_host: str, _port: int, _timeout_seconds: float) -> Sequence[str]:
        await blocker.wait()
        return ("8.8.8.8",)

    request = _request(limits=http_hop.HTTPHopLimits(dns_timeout_seconds=0.01))
    with pytest.raises(http_hop.HTTPHopError) as exc:
        await http_hop._resolve_validated_ips(request, resolver=resolver)

    assert exc.value.code == "dns_timeout"
    assert str(exc.value) == "Destination resolution timed out."


async def test_resolver_exception_detail_and_context_are_not_exposed() -> None:
    secret = "resolver leaked secret-route-token"

    async def resolver(_host: str, _port: int, _timeout_seconds: float) -> Sequence[str]:
        raise RuntimeError(secret)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await http_hop._resolve_validated_ips(_request(), resolver=resolver)

    assert exc.value.code == "dns_resolution_failed"
    assert secret not in str(exc.value)
    assert secret not in repr(exc.value)
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


async def test_default_resolver_offloads_blocking_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_resolver(_host: str, *, timeout_s: float) -> tuple[str, ...]:
        assert timeout_s == 2.0
        started.set()
        release.wait(timeout=1.0)
        return ("8.8.8.8",)

    monkeypatch.setattr(http_hop, "resolve_host_ips", blocking_resolver)
    task = asyncio.create_task(http_hop._default_resolver("api.example.com", 443, 2.0))

    async def wait_until_started() -> None:
        while not started.is_set():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_until_started(), timeout=1.0)

    heartbeat_ran = False

    async def heartbeat() -> None:
        nonlocal heartbeat_ran
        await asyncio.sleep(0)
        heartbeat_ran = True

    try:
        await heartbeat()
        assert heartbeat_ran is True
    finally:
        release.set()

    assert await task == ("8.8.8.8",)
