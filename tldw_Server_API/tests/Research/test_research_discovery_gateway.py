from __future__ import annotations

import asyncio
from dataclasses import asdict, replace

import pytest

from tldw_Server_API.app.core.Research.discovery.contracts import (
    AccessRoute,
    CredentialRequirement,
    DispatchIntent,
    ExactOrigin,
    OperationKind,
    QueryMode,
    QueryPair,
    RouteKind,
    RouteLimits,
    RoutePolicy,
    SourceConstraint,
)
from tldw_Server_API.app.core.Research.discovery.gateway import (
    DiscoveryGatewayError,
    DiscoveryGatewayResponse,
    dispatch_once,
)
from tldw_Server_API.app.core.Security.http_hop import (
    HTTPHopError,
    HTTPHopLimits,
    HTTPHopResponse,
    NormalizedHTTPHopRequest,
)


def _route_and_intent() -> tuple[AccessRoute, DispatchIntent]:
    limits = RouteLimits(
        max_pages=1,
        max_redirects=1,
        max_retries=0,
        timeout_ms=250,
        max_response_bytes=64,
        max_results=5,
    )
    policy = RoutePolicy(
        policy_version="discovery-route-policy-v1",
        origin=ExactOrigin(scheme="https", host="api.example.test", port=443),
        methods=("GET",),
        paths=("/search",),
        allowed_query_keys=("q", "page"),
        limits=limits,
    )
    route = AccessRoute(
        route_id="example.search",
        backend_id="example",
        adapter_id="example.search",
        route_kind=RouteKind.DIRECT,
        query_modes=(QueryMode.GENERAL_FREE_TEXT,),
        source_constraint=SourceConstraint.NATIVE_CORPUS,
        attribution_basis="native corpus",
        credential_requirement=CredentialRequirement.NONE,
        fallback_order=0,
        max_physical_dispatches=2,
        adapter_version="v1",
        policy=policy,
    )
    intent = DispatchIntent(
        route_id=route.route_id,
        policy_digest=policy.policy_digest,
        operation_kind=OperationKind.SEARCH,
        method="GET",
        path="/search",
        query_pairs=(
            QueryPair(name="q", value="quantum mechanics"),
            QueryPair(name="page", value="1"),
        ),
        limits=limits,
    )
    return route, intent


def _hop_response(
    *,
    status_code: int = 200,
    headers: tuple[tuple[str, str], ...] = (("Content-Type", "application/json"),),
    body: bytes = b'{"ok":true}',
) -> HTTPHopResponse:
    return HTTPHopResponse(
        status_code=status_code,
        headers=headers,
        body=body,
        resolved_ips=("93.184.216.34",),
        connected_ip="93.184.216.34",
        response_header_bytes=123,
        wire_bytes=len(body),
    )


@pytest.mark.asyncio
async def test_dispatch_once_emits_one_credential_free_normalized_hop() -> None:
    route, intent = _route_and_intent()
    calls: list[NormalizedHTTPHopRequest] = []

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        calls.append(request)
        return _hop_response()

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda route_id, digest: (route_id == route.route_id and digest == route.policy.policy_digest),
        one_hop=one_hop,
    )

    assert calls == [
        NormalizedHTTPHopRequest(
            scheme="https",
            host="api.example.test",
            port=443,
            method="GET",
            target="/search?q=quantum%20mechanics&page=1",
            headers=(),
            body=b"",
            limits=HTTPHopLimits(
                dns_timeout_seconds=0.25,
                connect_timeout_seconds=0.25,
                read_timeout_seconds=0.25,
                write_timeout_seconds=0.25,
                total_timeout_seconds=0.25,
                max_wire_bytes=64,
                max_decompressed_bytes=64,
                max_parser_input_bytes=64,
            ),
        )
    ]
    assert isinstance(response, DiscoveryGatewayResponse)
    assert response.status_code == 200
    assert response.body == b'{"ok":true}'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mismatch",
    ("route_id", "policy_digest", "method", "path", "query_key", "limits", "inactive"),
)
async def test_policy_mismatches_reject_before_the_hop(mismatch: str) -> None:
    route, intent = _route_and_intent()
    if mismatch == "route_id":
        intent = replace(intent, route_id="other.search")
    elif mismatch == "policy_digest":
        object.__setattr__(route.policy, "policy_digest", "0" * 64)
        intent = replace(intent, policy_digest="0" * 64)
    elif mismatch == "method":
        intent = replace(intent, method="POST")
    elif mismatch == "path":
        intent = replace(intent, path="/other")
    elif mismatch == "query_key":
        intent = replace(intent, query_pairs=(QueryPair(name="token", value="secret"),))
    elif mismatch == "limits":
        intent = replace(intent, limits=replace(intent.limits, max_results=6))

    calls = 0

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response()

    with pytest.raises(DiscoveryGatewayError):
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: mismatch != "inactive",
            one_hop=one_hop,
        )

    assert calls == 0


@pytest.mark.asyncio
async def test_redirect_is_returned_as_typed_data_and_not_followed() -> None:
    route, intent = _route_and_intent()
    calls = 0

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        nonlocal calls
        calls += 1
        return _hop_response(
            status_code=302,
            headers=(("Location", "https://api.example.test/search?page=2"),),
            body=b"",
        )

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _digest: True,
        one_hop=one_hop,
    )

    assert isinstance(response, DiscoveryGatewayResponse)
    assert response.status_code == 302
    assert response.headers == ()
    assert calls == 1


@pytest.mark.asyncio
async def test_http_hop_error_maps_to_stable_bounded_safe_error() -> None:
    route, intent = _route_and_intent()
    unsafe = HTTPHopError("total_timeout", retryable=True)
    unsafe.args = ("secret?query=quantum mechanics body=/private/file",)

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        raise unsafe

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    error = caught.value
    assert error.code == "hop_failed"
    assert error.retryable is True
    assert str(error) == "Discovery gateway hop failed"
    assert len(str(error)) <= 64
    assert error.__cause__ is None
    assert error.__context__ is None


@pytest.mark.asyncio
async def test_cancellation_propagates() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )


@pytest.mark.asyncio
async def test_response_headers_and_trace_contain_only_safe_bounded_evidence() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(
            headers=(
                ("Content-Type", "application/json"),
                ("Set-Cookie", "session=secret"),
                ("Authorization", "Bearer secret"),
                ("X-Unsafe", "quantum mechanics /private/file"),
            ),
            body=b'{"secret":"body-only"}',
        )

    response = await dispatch_once(
        route,
        intent,
        is_policy_active=lambda _route_id, _digest: True,
        one_hop=one_hop,
    )

    assert response.headers == (("content-type", "application/json"),)
    assert response.trace.route_id == route.route_id
    assert response.trace.policy_digest == route.policy.policy_digest
    assert response.trace.requested_host == "api.example.test"
    assert response.trace.tls_server_name == "api.example.test"
    assert response.trace.query_keys == ("q", "page")
    assert response.trace.resolved_ips == ("93.184.216.34",)
    assert response.trace.connected_ip == "93.184.216.34"
    assert response.trace.response_header_bytes == 123
    assert response.trace.wire_bytes == len(response.body)
    assert response.trace.decoded_bytes == len(response.body)
    assert response.trace.max_response_bytes == 64
    assert response.trace.http_limits.total_timeout_seconds == 0.25
    assert response.trace.http_limits.max_parser_input_bytes == 64
    trace_text = repr(asdict(response.trace))
    for unsafe_text in ("secret", "quantum mechanics", "body-only", "/private/file"):
        assert unsafe_text not in trace_text


@pytest.mark.asyncio
async def test_injected_hop_cannot_return_body_over_route_limit() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(body=b"x" * 65)

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "invalid_hop_response"
    assert str(caught.value) == "Discovery gateway hop response rejected"


@pytest.mark.asyncio
async def test_malformed_hop_headers_map_to_safe_invalid_response() -> None:
    route, intent = _route_and_intent()

    async def one_hop(request: NormalizedHTTPHopRequest) -> HTTPHopResponse:
        return _hop_response(headers=(("Content-Type",),))

    with pytest.raises(DiscoveryGatewayError) as caught:
        await dispatch_once(
            route,
            intent,
            is_policy_active=lambda _route_id, _digest: True,
            one_hop=one_hop,
        )

    assert caught.value.code == "invalid_hop_response"
    assert str(caught.value) == "Discovery gateway hop response rejected"
